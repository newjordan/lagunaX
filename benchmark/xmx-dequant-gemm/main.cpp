// xmx-dequant-gemm — standalone SYCL microbenchmark (no llama.cpp).
//
// Question: can a joint_matrix (XMX) kernel that dequantizes q4_K weights on the
// fly beat oneMKL fp16 GEMM (dequant-then-gemm, the llama.cpp status quo) at the
// MoE expert shapes on an Intel Arc B70 (BMG)?
//
// Shapes: M=out-features=512, K=in-features=2048, N=tokens in {16,32,64,128}.
//   Weight  W: [M x K] q4_K, row-major (K contiguous), K/256 blocks per row.
//   Acts    X: [K x N] fp16, column-major (each token's K values contiguous),
//              i.e. x[n*K + k] — matches llama's src1 fp16 staging.
//   Output  C: [M x N] fp32, column-major (each token's M outputs contiguous),
//              i.e. out[n*M + m] — matches llama's dst_dd layout (ldc = M).
//
// Contenders:
//   BASELINE-MKL : dequant q4_K -> fp16 (llama's dequantize_block_q4_K kernel,
//                  32 threads/block), then oneapi::mkl::blas::column_major::gemm
//                  fp16 A/B, fp16 C (trans A, like ggml_sycl_op_mul_mat_sycl),
//                  then fp16->fp32 convert kernel.
//   XMX-FUSED    : one kernel: WG dequantizes a q4_K weight tile into SLM in
//                  VNNI/ext_intel_packed fp16 layout, then sub_group joint_matrix
//                  (16x16x16 fp16, fp32 acc — the winning BMG combo) against
//                  fp16 activation tiles loaded straight from global. Computes
//                  C^T tiles: A = activations (N x K row-major), B = weights
//                  (K x M packed in SLM). Split-K across workgroups + reduce
//                  kernel; optional oversubscribed dequant sub-groups and
//                  n-inner token-group sweep (see template params).
//   SIMT-REF     : scalar one-thread-per-output fused dequant dot (sanity floor).
//
// The q4_K dequant math (get_scale_min_k4, dequantize_q4_K_common) is copied
// verbatim from llama.cpp ggml/src/ggml-sycl/dequantize.hpp; vec_aligned_load
// and get_pointer from common.hpp.

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <oneapi/mkl.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <random>
#include <string>
#include <vector>

namespace sem = sycl::ext::oneapi::experimental::matrix;
using sycl::half;

// ---------------------------------------------------------------- q4_K format
#define QK_K 256
#define K_SCALE_SIZE 12

typedef sycl::half  ggml_half;
typedef sycl::half2 ggml_half2;

struct block_q4_K {
    ggml_half2 dm;                    // d = dm[0], dmin = dm[1]
    uint8_t    scales[K_SCALE_SIZE];  // 12 bytes packed 6-bit scales/mins
    uint8_t    qs[QK_K / 2];          // 128 bytes of 4-bit quants
};
static_assert(sizeof(block_q4_K) == 2 * sizeof(ggml_half) + K_SCALE_SIZE + QK_K / 2,
              "wrong q4_K block size/padding");

// ---- copied verbatim from llama.cpp ggml-sycl/common.hpp -------------------
template <typename Tp, int n>
inline sycl::vec<Tp, n> vec_aligned_load(const Tp* aligned_ptr) {
    return *reinterpret_cast<const sycl::vec<Tp, n>*>(aligned_ptr);
}

template <typename Tp, int dim>
static inline Tp* get_pointer(sycl::local_accessor<Tp, dim> acc) {
    return acc.template get_multi_ptr<sycl::access::decorated::no>().get();
}

// ---- copied verbatim from llama.cpp ggml-sycl/dequantize.hpp ---------------
static inline void get_scale_min_k4(int j, const uint8_t * q, uint8_t & d, uint8_t & m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}

template <typename dst_t>
inline void dequantize_q4_K_common(dst_t * __restrict__ y, const uint8_t * __restrict__ qs_ptr, const float dall,
                                   const float dmin, uint8_t * __restrict__ scales_local, int il, int ir) {
    const int is = 2 * il;
    constexpr int n  = 4;

    uint8_t sc, m;
    get_scale_min_k4(is + 0, scales_local, sc, m);
    const float d1 = dall * sc;
    const float m1 = dmin * m;

    get_scale_min_k4(is + 1, scales_local, sc, m);
    const float d2 = dall * sc;
    const float m2 = dmin * m;

    sycl::vec<uint8_t, n> q_vec = vec_aligned_load<uint8_t, n>(qs_ptr + 32 * il + n * ir);
    for (int l = 0; l < n; ++l) {
        y[l + 0]  = d1 * (q_vec[l] & 0xF) - m1;
        y[l + 32] = d2 * (q_vec[l] >> 4) - m2;
    }
}

// Same math as dequantize_q4_K_common, but the stores go through a functor
// store2(j, v0, v1) with j (even) the in-block index of the value pair — needed
// to write the VNNI/ext_intel_packed SLM layout of the fused kernel with
// vectorized half2 stores. Value math identical.
template <typename Store2Fn>
inline void dequantize_q4_K_math(Store2Fn && store2, const uint8_t * __restrict__ qs_ptr, const float dall,
                                 const float dmin, const uint8_t * __restrict__ scales_local, int il, int ir) {
    const int is = 2 * il;
    constexpr int n  = 4;

    uint8_t sc, m;
    get_scale_min_k4(is + 0, scales_local, sc, m);
    const float d1 = dall * sc;
    const float m1 = dmin * m;

    get_scale_min_k4(is + 1, scales_local, sc, m);
    const float d2 = dall * sc;
    const float m2 = dmin * m;

    sycl::vec<uint8_t, n> q_vec = vec_aligned_load<uint8_t, n>(qs_ptr + 32 * il + n * ir);
    const int j0 = 64 * il + n * ir;
    store2(j0 + 0,  d1 * (q_vec[0] & 0xF) - m1, d1 * (q_vec[1] & 0xF) - m1);
    store2(j0 + 2,  d1 * (q_vec[2] & 0xF) - m1, d1 * (q_vec[3] & 0xF) - m1);
    store2(j0 + 32, d2 * (q_vec[0] >>  4) - m2, d2 * (q_vec[1] >>  4) - m2);
    store2(j0 + 34, d2 * (q_vec[2] >>  4) - m2, d2 * (q_vec[3] >>  4) - m2);
}

// ---- copied verbatim (QK_K==256 branch) from llama.cpp dequantize.hpp ------
template<typename dst_t>
static void dequantize_block_q4_K(const void * __restrict__ vx, dst_t * __restrict__ yy,
                                  uint8_t* scales_local, const sycl::nd_item<3> &item_ct1) {
    const block_q4_K * x = (const block_q4_K *) vx;

    const int64_t i = item_ct1.get_group(2);

    const int64_t tid = item_ct1.get_local_id(2);
    const int64_t il  = tid / 8;
    const int64_t ir  = tid % 8;

    dst_t * y = yy + i * QK_K + 64 * il + 4 * ir;

    const sycl::half2 dm = x[i].dm;
    const float dall = dm[0];
    const float dmin = dm[1];

    if (tid < 12) {
        scales_local[tid] = x[i].scales[tid];
    }

    item_ct1.barrier(sycl::access::fence_space::local_space);
    dequantize_q4_K_common(y, x[i].qs, dall, dmin, scales_local, il, ir);
}

// ------------------------------------------------------------------ helpers
template <typename T>
static inline auto gptr(T * p) {
    return sycl::address_space_cast<sycl::access::address_space::global_space,
                                    sycl::access::decorated::no>(p);
}

// XMX tile shape (Arc/BMG fp16 combination; verified via matrix_combinations query)
#ifndef XMX_TM
#define XMX_TM 8
#endif
#ifndef XMX_TN
#define XMX_TN 16
#endif
#ifndef XMX_TK
#define XMX_TK 16
#endif
#ifndef XMX_SG_SIZE
#define XMX_SG_SIZE 16
#endif
constexpr int TM = XMX_TM;
constexpr int TN = XMX_TN;
constexpr int TK = XMX_TK;
constexpr int SG_SIZE = XMX_SG_SIZE;

// ------------------------------------------------------------ XMX-FUSED kernel
//
// Computes C^T = X * W^T tile-wise:
//   A (use::a)  8x16 fp16  = activation tile, row-major straight from global
//                            (x is [N x K] row-major: x[n*K + k]).
//   B (use::b) 16x16 fp16  = dequantized weight tile, ext_intel_packed (VNNI)
//                            in SLM: element (k,m) at (k/2)*(2*WG_M) + m*2 + k%2.
//   C (acc)     8x16 fp32  -> stored row-major with stride M: out[n*M + m].
//
// Grid: (M/WG_M) x SPLIT_K x ceil(N/NB_WG) workgroups (linearized).
// Each WG owns WG_M output columns (of M), NB_WG token rows, and K/SPLIT_K of
// the reduction. K loop steps KB values (KB=64/128 splits a q4_K block into
// il-quarters for more split-K parallelism): cooperative dequant of the weight
// tile into SLM (VNNI layout, half2 stores), barrier, then joint_matrix MADs.
// SPLIT_K > 1 writes fp32 partials; a tiny reduce kernel sums them.
// TMv x TNv x TKv: joint_matrix shape. BMG supports fp16 8x16x16 (max_msize 8),
// 16x16x16, 1x64x16, 32x64x16, 1x64x32, 32x64x32 (fp32 accumulate).
// NSG_TOT: total sub-groups per WG. The first WG_M/SG_COLS SGs do the
// joint_matrix work; ALL SGs cooperate on the dequant phase (the dequant phase
// is latency-bound, so oversubscribing it with extra threads is free occupancy).
// N_INNER=1 (requires chunks_per_split == 1, i.e. SPLIT_K == K/KB): the WG
// loops over ALL token groups internally, so each weight tile is dequantized
// exactly once per iteration (no per-token-group redundancy). Accumulators are
// scoped per token group, so register pressure stays at NB_WG/TMv tiles.
template <int TMv, int TNv, int TKv, int WG_M, int SG_COLS, int NB_WG, int SPLIT_K, int KB, int NSG_TOT,
          int N_INNER = 0>
static sycl::event xmx_fused_submit(sycl::queue &q, const block_q4_K *wq_, const half *x_,
                                    float *dst, int M, int N, int K) {
    constexpr int NSG        = WG_M / SG_COLS;   // MAD sub-groups per WG
    constexpr int WG_THREADS = NSG_TOT * SG_SIZE;
    constexpr int RM         = SG_COLS / TNv;    // m-direction tiles per SG
    constexpr int RN_MAX     = NB_WG / TMv;      // n-direction tiles per SG (max)
    constexpr int Q_PER_ROW  = KB / 64;          // q4_K il-quarters per row per chunk
    constexpr int TASKS      = WG_M * Q_PER_ROW; // 8-thread dequant tasks per chunk
    constexpr int TASKS_PER_PASS = WG_THREADS / 8;
    static_assert(KB == 64 || KB == 128 || KB == 256, "KB must be 64/128/256");
    static_assert(KB % TKv == 0 && SG_COLS % TNv == 0, "tile shape must divide the WG tile");
    static_assert(NB_WG % TMv == 0, "NB_WG must be a multiple of TMv");
    static_assert(NSG_TOT >= NSG, "need at least the MAD sub-groups");
    static_assert(WG_THREADS % 8 == 0 && TASKS % TASKS_PER_PASS == 0, "dequant task split");

    const block_q4_K *wq = wq_;
    half *x = const_cast<half *>(x_);

    const int nbk              = K / QK_K;
    const int m_groups         = M / WG_M;
    const int n_groups         = N_INNER ? 1 : (N + NB_WG - 1) / NB_WG;
    const int chunks_per_split = (K / KB) / SPLIT_K;   // == 1 when N_INNER
    const size_t MN            = (size_t)M * N;
    const int total_wgs        = m_groups * SPLIT_K * n_groups;

    return q.submit([&](sycl::handler &h) {
        sycl::local_accessor<half, 1> wtile(sycl::range<1>((size_t)KB * WG_M), h);
        h.parallel_for(
            sycl::nd_range<1>((size_t)total_wgs * WG_THREADS, WG_THREADS),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                sycl::sub_group sg = it.get_sub_group();
                const int wg    = (int)it.get_group(0);
                const int mg    = wg % m_groups;
                const int split = (wg / m_groups) % SPLIT_K;
                const int ng    = N_INNER ? 0 : wg / (m_groups * SPLIT_K);
                const int m0    = mg * WG_M;
                const int lid   = (int)it.get_local_id(0);
                const int sgid  = (int)sg.get_group_id()[0];
                const int sg_m0 = sgid * SG_COLS;
                const bool is_mad_sg = sgid < NSG;

                half *wt = wtile.template get_multi_ptr<sycl::access::decorated::no>().get();
                auto  wt_mp = wtile.template get_multi_ptr<sycl::access::decorated::no>();

                const int tq = lid / 8;   // dequant task slot within pass
                const int ir = lid % 8;   // llama's ir: 4-value column within quarter

                // cooperative dequant of chunk kb: weight tile -> packed (VNNI) SLM
                auto dequant_chunk = [&](int kb) {
                    const int blk     = (kb * KB) / QK_K;        // q4_K block column
                    const int il_base = (kb * KB % QK_K) / 64;   // first quarter in chunk
                    for (int pass = 0; pass < TASKS / TASKS_PER_PASS; ++pass) {
                        const int t  = pass * TASKS_PER_PASS + tq;
                        const int ml = t / Q_PER_ROW;            // weight row in tile
                        const int il = il_base + t % Q_PER_ROW;  // quarter in block
                        const block_q4_K *b = wq + (size_t)(m0 + ml) * nbk + blk;
                        const sycl::half2 dm = b->dm;
                        const float dall = dm[0];
                        const float dmin = dm[1];
                        dequantize_q4_K_math(
                            [&](int j, float v0, float v1) {
                                const int kl = j - 64 * il_base;  // local k in chunk, even
                                *reinterpret_cast<sycl::vec<half, 2> *>(
                                    &wt[(kl >> 1) * (2 * WG_M) + ml * 2]) =
                                    sycl::vec<half, 2>{(half)v0, (half)v1};
                            },
                            b->qs, dall, dmin, b->scales, il, ir);
                    }
                };

                using CArr = sem::joint_matrix<sycl::sub_group, float, sem::use::accumulator, TMv, TNv>[RN_MAX][RM];

                auto fill_c = [&](CArr &C, int RN) {
                    for (int rn = 0; rn < RN; ++rn)
                        for (int rm = 0; rm < RM; ++rm)
                            sem::joint_matrix_fill(sg, C[rn][rm], 0.0f);
                };
                auto mad_chunk = [&](CArr &C, int kb, int n0, int RN) {
                    for (int kt = 0; kt < KB / TKv; ++kt) {
                        sem::joint_matrix<sycl::sub_group, half, sem::use::a, TMv, TKv, sem::layout::row_major> A[RN_MAX];
                        for (int rn = 0; rn < RN; ++rn) {
                            sem::joint_matrix_load(sg, A[rn],
                                gptr(x) + ((size_t)(n0 + rn * TMv) * K + (size_t)kb * KB + kt * TKv),
                                (size_t)K);
                        }
                        for (int rm = 0; rm < RM; ++rm) {
                            sem::joint_matrix<sycl::sub_group, half, sem::use::b, TKv, TNv, sem::layout::ext_intel_packed> B;
                            sem::joint_matrix_load(sg, B,
                                wt_mp + (size_t)(kt * TKv / 2) * (2 * WG_M) + (sg_m0 + rm * TNv) * 2,
                                (size_t)(2 * WG_M));
                            for (int rn = 0; rn < RN; ++rn) {
                                sem::joint_matrix_mad(sg, C[rn][rm], A[rn], B, C[rn][rm]);
                            }
                        }
                    }
                };
                auto store_c = [&](CArr &C, float *outp, int n0, int RN) {
                    for (int rn = 0; rn < RN; ++rn) {
                        for (int rm = 0; rm < RM; ++rm) {
                            sem::joint_matrix_store(sg, C[rn][rm],
                                gptr(outp) + ((size_t)(n0 + rn * TMv) * M + m0 + sg_m0 + rm * TNv),
                                (size_t)M, sem::layout::row_major);
                        }
                    }
                };

                float *outp = dst + (SPLIT_K > 1 ? (size_t)split * MN : 0);

                if constexpr (N_INNER) {
                    // dequant once, then sweep all token groups against the tile
                    const int kb = split;
                    dequant_chunk(kb);
                    sycl::group_barrier(it.get_group());
                    if (is_mad_sg) {
                        const int n_blocks = (N + NB_WG - 1) / NB_WG;
                        for (int ngi = 0; ngi < n_blocks; ++ngi) {
                            const int n0 = ngi * NB_WG;
                            const int RN = std::min(NB_WG, N - n0) / TMv;
                            CArr C;
                            fill_c(C, RN);
                            mad_chunk(C, kb, n0, RN);
                            store_c(C, outp, n0, RN);
                        }
                    }
                } else {
                    const int n0 = ng * NB_WG;
                    const int RN = std::min(NB_WG, N - n0) / TMv;
                    CArr C;
                    fill_c(C, RN);
                    for (int c = 0; c < chunks_per_split; ++c) {
                        const int kb = split * chunks_per_split + c;
                        dequant_chunk(kb);
                        sycl::group_barrier(it.get_group());
                        if (is_mad_sg) {
                            mad_chunk(C, kb, n0, RN);
                        }
                        sycl::group_barrier(it.get_group());
                    }
                    if (is_mad_sg) {
                        store_c(C, outp, n0, RN);
                    }
                }
            });
    });
}

using EvVec = std::vector<std::pair<std::string, sycl::event>>;

template <int TMv, int TNv, int TKv, int WG_M, int SG_COLS, int NB_WG, int SPLIT_K, int KB, int NSG_TOT,
          int N_INNER = 0>
static void xmx_run_cfg(sycl::queue &q, const block_q4_K *wq, const half *x, float *out,
                        float *part, int M, int N, int K, EvVec &evs) {
    const size_t MN = (size_t)M * N;
    float *dst = SPLIT_K > 1 ? part : out;
    sycl::event ev = xmx_fused_submit<TMv, TNv, TKv, WG_M, SG_COLS, NB_WG, SPLIT_K, KB, NSG_TOT, N_INNER>(q, wq, x, dst, M, N, K);
    evs.emplace_back("xmx-fused", ev);
    if constexpr (SPLIT_K > 1) {
        sycl::event ev2 = q.parallel_for(sycl::range<1>(MN), [=](sycl::id<1> i) {
            float s = 0.f;
#pragma unroll
            for (int sp = 0; sp < SPLIT_K; ++sp) s += part[(size_t)sp * MN + i[0]];
            out[i[0]] = s;
        });
        evs.emplace_back("xmx-reduce", ev2);
    }
}

struct XmxCfg {
    const char *name;
    int tm, tn, tk, wg_m, sg_cols, nb_wg, split_k, kb, nsg_tot, n_inner;
    void (*run)(sycl::queue &, const block_q4_K *, const half *, float *, float *, int, int, int, EvVec &);
};

#define CFG(t, u, v, a, b, c, d, e, f) \
    XmxCfg{ "t" #t "x" #u "x" #v "-wgm" #a "-sgc" #b "-nb" #c "-sk" #d "-kb" #e "-sg" #f, \
            t, u, v, a, b, c, d, e, f, 0, &xmx_run_cfg<t, u, v, a, b, c, d, e, f, 0> }
#define CFGNI(t, u, v, a, b, c, d, e, f) \
    XmxCfg{ "t" #t "x" #u "x" #v "-wgm" #a "-sgc" #b "-nb" #c "-sk" #d "-kb" #e "-sg" #f "-ni", \
            t, u, v, a, b, c, d, e, f, 1, &xmx_run_cfg<t, u, v, a, b, c, d, e, f, 1> }

static const std::vector<XmxCfg> g_cfgs = {
    // 16x16x16 tiles, oversubscribed dequant SGs (the winners of the tuning
    // sweeps; earlier iterations also tried TM=8, KB=256, SG_COLS=32, NB=64,
    // NSG_TOT==NSG, and the reported-but-emulated 32x64x16/32 combos — all lost)
    CFG(16, 16, 16, 64, 16, 16, 32, 64, 4),
    CFG(16, 16, 16, 64, 16, 16, 16, 64, 16),
    CFG(16, 16, 16, 64, 16, 32, 16, 64, 16),
    CFG(16, 16, 16, 64, 16, 32, 16, 128, 16),
    CFG(16, 16, 16, 128, 16, 32, 8, 64, 16),
    CFG(16, 16, 16, 128, 16, 32, 16, 64, 16),
    // n-inner: dequant each tile once, sweep token groups inside the WG
    // (needs SPLIT_K == K/KB); aimed at large N where redundancy dominates
    CFGNI(16, 16, 16, 64, 16, 32, 8, 256, 8),
    CFGNI(16, 16, 16, 64, 16, 32, 8, 256, 16),
    CFGNI(16, 16, 16, 128, 16, 32, 8, 256, 16),
    CFGNI(16, 16, 16, 64, 16, 32, 16, 128, 8),
    CFGNI(16, 16, 16, 64, 16, 32, 16, 128, 16),
    CFGNI(16, 16, 16, 64, 16, 32, 32, 64, 4),
    CFGNI(16, 16, 16, 64, 16, 32, 32, 64, 8),
};

// ------------------------------------------------------------- host reference
static void host_get_scale_min_k4(int j, const uint8_t *q, uint8_t &d, uint8_t &m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

// dequantize one block to double, same math/order as dequantize_q4_K_common
static void host_dequant_block(const block_q4_K &b, double *y) {
    const float dall = (float)b.dm[0];
    const float dmin = (float)b.dm[1];
    for (int il = 0; il < 4; ++il) {
        uint8_t sc, mn;
        host_get_scale_min_k4(2 * il + 0, b.scales, sc, mn);
        const float d1 = dall * sc, m1 = dmin * mn;
        host_get_scale_min_k4(2 * il + 1, b.scales, sc, mn);
        const float d2 = dall * sc, m2 = dmin * mn;
        for (int r = 0; r < 32; ++r) {
            const uint8_t qb = b.qs[32 * il + r];
            y[64 * il + r]      = (double)d1 * (qb & 0xF) - m1;
            y[64 * il + 32 + r] = (double)d2 * (qb >> 4) - m2;
        }
    }
}

// ------------------------------------------------------------------ SIMT-REF
static sycl::event simt_submit(sycl::queue &q, const block_q4_K *wq_, const half *x_,
                               float *out, int M, int N, int K) {
    const block_q4_K *wq = wq_;
    const half *x = x_;
    const int nbk = K / QK_K;
    return q.parallel_for(sycl::range<1>((size_t)M * N), [=](sycl::id<1> idx) {
        const int n = (int)(idx[0] / M);
        const int m = (int)(idx[0] % M);
        const block_q4_K *row = wq + (size_t)m * nbk;
        const half *xv = x + (size_t)n * K;
        float acc = 0.f;
        for (int kb = 0; kb < nbk; ++kb) {
            const block_q4_K *b = row + kb;
            const sycl::half2 dm = b->dm;
            const float dall = dm[0];
            const float dmin = dm[1];
            const half *xk = xv + kb * QK_K;
            for (int il = 0; il < 4; ++il) {
                uint8_t sc, mn;
                get_scale_min_k4(2 * il + 0, b->scales, sc, mn);
                const float d1 = dall * sc, m1 = dmin * mn;
                get_scale_min_k4(2 * il + 1, b->scales, sc, mn);
                const float d2 = dall * sc, m2 = dmin * mn;
                for (int r = 0; r < 32; ++r) {
                    const uint8_t qb = b->qs[32 * il + r];
                    acc += (d1 * (qb & 0xF) - m1) * (float)xk[64 * il + r];
                    acc += (d2 * (qb >> 4) - m2) * (float)xk[64 * il + 32 + r];
                }
            }
        }
        out[(size_t)n * M + m] = acc;
    });
}

// --------------------------------------------------------------- timing utils
struct TimedResult {
    std::map<std::string, double> label_us;  // mean per-iter, per label
    double total_us = 0.0;
};

// Burn ~1 ms of fp32 FMAs so the GPU is at steady clocks before a timed
// section (otherwise DVFS ramp-up makes the first contender look slow).
static void heat_gpu(sycl::queue &q, float *sink) {
    q.parallel_for(sycl::range<1>(1 << 16), [=](sycl::id<1> i) {
         float a = (float)i[0] * 1e-6f + 1.0f;
         float b = 1.0000001f, c = 0.9999999f, d = 1.0f;
         for (int t = 0; t < 40000; ++t) {
             a = sycl::fma(a, b, 1e-7f);
             d = sycl::fma(d, c, 1e-7f);
         }
         sink[i[0] & 1023] = a + d;
     }).wait();
}

// Heat until the heater kernel's wall time stops improving (clock ramp done).
static void heat_until_stable(sycl::queue &q, float *sink) {
    double prev_ms = 1e30;
    for (int i = 0; i < 40; ++i) {
        auto t0 = std::chrono::steady_clock::now();
        heat_gpu(q, sink);
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (i >= 2 && ms > prev_ms * 0.97) break;
        prev_ms = std::min(prev_ms, ms);
    }
}

static TimedResult time_contender(sycl::queue &q, int warmup, int iters, float *heat_sink,
                                  const std::function<void(EvVec &)> &once) {
    heat_until_stable(q, heat_sink);
    for (int i = 0; i < warmup; ++i) {
        EvVec e;
        once(e);
    }
    q.wait();
    std::vector<EvVec> all((size_t)iters);
    for (auto &e : all) once(e);
    q.wait();
    TimedResult r;
    for (auto &e : all) {
        for (auto &[lbl, ev] : e) {
            const uint64_t t0 = ev.get_profiling_info<sycl::info::event_profiling::command_start>();
            const uint64_t t1 = ev.get_profiling_info<sycl::info::event_profiling::command_end>();
            r.label_us[lbl] += (double)(t1 - t0) / 1e3;
        }
    }
    for (auto &[k, v] : r.label_us) {
        v /= iters;
        r.total_us += v;
    }
    return r;
}

struct ErrStat {
    double max_abs = 0.0, rel_to_scale = 0.0, max_rel = 0.0;
};

static ErrStat compare(const float *out, const double *ref, size_t n) {
    ErrStat e;
    double scale = 0.0;
    for (size_t i = 0; i < n; ++i) scale = std::max(scale, std::fabs(ref[i]));
    for (size_t i = 0; i < n; ++i) {
        const double d = std::fabs((double)out[i] - ref[i]);
        e.max_abs = std::max(e.max_abs, d);
        e.max_rel = std::max(e.max_rel, d / std::max(1.0, std::fabs(ref[i])));
    }
    e.rel_to_scale = scale > 0 ? e.max_abs / scale : 0.0;
    return e;
}

static const char *matrix_type_name(sem::matrix_type t) {
    switch (t) {
        case sem::matrix_type::bf16:  return "bf16";
        case sem::matrix_type::fp16:  return "fp16";
        case sem::matrix_type::tf32:  return "tf32";
        case sem::matrix_type::fp32:  return "fp32";
        case sem::matrix_type::fp64:  return "fp64";
        case sem::matrix_type::sint8: return "sint8";
        case sem::matrix_type::sint16:return "sint16";
        case sem::matrix_type::sint32:return "sint32";
        case sem::matrix_type::sint64:return "sint64";
        case sem::matrix_type::uint8: return "uint8";
        case sem::matrix_type::uint16:return "uint16";
        case sem::matrix_type::uint32:return "uint32";
        case sem::matrix_type::uint64:return "uint64";
        default: return "?";
    }
}

// -------------------------------------------------------------------- main
int main(int argc, char **argv) {
    bool query_only = false;
    bool no_simt = false;
    int warmup = 10, iters = 50, reps = 2;
    std::vector<int> Ns = {16, 32, 64, 128};
    std::string cfg_filter;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--query") query_only = true;
        else if (a == "--no-simt") no_simt = true;
        else if (a == "--iters" && i + 1 < argc) iters = atoi(argv[++i]);
        else if (a == "--warmup" && i + 1 < argc) warmup = atoi(argv[++i]);
        else if (a == "--reps" && i + 1 < argc) reps = atoi(argv[++i]);
        else if (a == "--cfg" && i + 1 < argc) cfg_filter = argv[++i];
        else if (a == "--N" && i + 1 < argc) {
            Ns.clear();
            char *tok = strtok(argv[++i], ",");
            while (tok) { Ns.push_back(atoi(tok)); tok = strtok(nullptr, ","); }
        } else {
            fprintf(stderr, "usage: %s [--query] [--no-simt] [--iters I] [--warmup W] [--N 16,32] [--cfg substr]\n", argv[0]);
            return 1;
        }
    }

    sycl::queue q{sycl::gpu_selector_v,
                  sycl::property_list{sycl::property::queue::in_order{},
                                      sycl::property::queue::enable_profiling{}}};
    auto dev = q.get_device();
    printf("device        : %s\n", dev.get_info<sycl::info::device::name>().c_str());
    printf("compute units : %u\n", dev.get_info<sycl::info::device::max_compute_units>());
    printf("max clock     : %u MHz\n", dev.get_info<sycl::info::device::max_clock_frequency>());
    printf("local mem     : %zu bytes\n", (size_t)dev.get_info<sycl::info::device::local_mem_size>());
    {
        auto sgs = dev.get_info<sycl::info::device::sub_group_sizes>();
        printf("sub-group szs :");
        for (auto s : sgs) printf(" %zu", s);
        printf("\n");
    }
    printf("joint_matrix supported combinations (msize x nsize x ksize, a/b/c/d types):\n");
    bool have_8x16x16_fp16 = false;
    try {
        namespace sex = sycl::ext::oneapi::experimental;
        auto combos = dev.get_info<sex::info::device::matrix_combinations>();
        for (auto &c : combos) {
            printf("  %2zu x %2zu x %2zu (max %zu x %zu x %zu)  a=%s b=%s c=%s d=%s\n",
                   (size_t)c.msize, (size_t)c.nsize, (size_t)c.ksize,
                   (size_t)c.max_msize, (size_t)c.max_nsize, (size_t)c.max_ksize,
                   matrix_type_name(c.atype), matrix_type_name(c.btype),
                   matrix_type_name(c.ctype), matrix_type_name(c.dtype));
            const size_t msz = c.msize ? c.msize : c.max_msize;
            if (c.atype == sem::matrix_type::fp16 && c.btype == sem::matrix_type::fp16 &&
                c.ctype == sem::matrix_type::fp32 && msz >= (size_t)TM &&
                (size_t)c.nsize == (size_t)TN && (size_t)c.ksize == (size_t)TK) {
                have_8x16x16_fp16 = true;
            }
        }
    } catch (const std::exception &e) {
        printf("  (matrix_combinations query failed: %s)\n", e.what());
    }
    printf("using tile    : TM=%d TN=%d TK=%d fp16*fp16+fp32, sub-group %d%s\n",
           TM, TN, TK, SG_SIZE, have_8x16x16_fp16 ? "" : "  [WARNING: not in reported combos]");
    if (query_only) return 0;

    const int M = 512, K = 2048;
    const int nbk = K / QK_K;
    const size_t nblocks = (size_t)M * nbk;
    const int NMAX = *std::max_element(Ns.begin(), Ns.end());
    const size_t slm_bytes = (size_t)dev.get_info<sycl::info::device::local_mem_size>();

    // ---------------- host data
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist_d(0.001f, 0.1f);
    std::uniform_int_distribution<int> dist_byte(0, 255);
    std::normal_distribution<float> dist_n(0.f, 1.f);

    std::vector<block_q4_K> h_w(nblocks);
    for (auto &b : h_w) {
        b.dm = ggml_half2{(half)dist_d(rng), (half)dist_d(rng)};
        for (auto &s : b.scales) s = (uint8_t)dist_byte(rng);
        for (auto &v : b.qs) v = (uint8_t)dist_byte(rng);
    }
    std::vector<half> h_x((size_t)NMAX * K);
    for (auto &v : h_x) v = (half)dist_n(rng);

    // double-dequantized weights for CPU reference
    std::vector<double> wd((size_t)M * K);
    for (int m = 0; m < M; ++m)
        for (int kb = 0; kb < nbk; ++kb)
            host_dequant_block(h_w[(size_t)m * nbk + kb], &wd[(size_t)m * K + kb * QK_K]);

    // ---------------- device buffers
    block_q4_K *d_w  = sycl::malloc_device<block_q4_K>(nblocks, q);
    half  *d_x       = sycl::malloc_device<half>((size_t)NMAX * K, q);
    half  *d_wf16    = sycl::malloc_device<half>((size_t)M * K, q);
    half  *d_cf16    = sycl::malloc_device<half>((size_t)M * NMAX, q);
    float *d_out     = sycl::malloc_device<float>((size_t)M * NMAX, q);
    float *d_part    = sycl::malloc_device<float>((size_t)32 * M * NMAX, q);
    q.memcpy(d_w, h_w.data(), nblocks * sizeof(block_q4_K));
    q.memcpy(d_x, h_x.data(), h_x.size() * sizeof(half));
    q.wait();

    std::vector<float> h_out((size_t)M * NMAX);

    auto run_baseline = [&](int N, EvVec &evs) {
        // 1) dequant q4_K -> fp16 (llama's kernel: 32 threads per 256-value block)
        sycl::event e1 = q.submit([&](sycl::handler &h) {
            sycl::local_accessor<uint8_t, 1> scale_local_acc(sycl::range<1>(12), h);
            const void *vx = d_w;
            half *y = d_wf16;
            h.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nblocks * 32), sycl::range<3>(1, 1, 32)),
                           [=](sycl::nd_item<3> it) {
                               dequantize_block_q4_K<half>(vx, y, get_pointer(scale_local_acc), it);
                           });
        });
        evs.emplace_back("mkl-dequant", e1);
        // 2) fp16 GEMM, fp16 out (ggml_sycl_op_mul_mat_sycl's dpct::gemm equivalent)
        sycl::event e2 = oneapi::mkl::blas::column_major::gemm(
            q, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans,
            (std::int64_t)M, (std::int64_t)N, (std::int64_t)K,
            half(1.0f), d_wf16, (std::int64_t)K, d_x, (std::int64_t)K,
            half(0.0f), d_cf16, (std::int64_t)M);
        evs.emplace_back("mkl-gemm", e2);
        // 3) fp16 -> fp32 convert (llama does this after the fp16 gemm)
        half  *cf = d_cf16;
        float *of = d_out;
        sycl::event e3 = q.parallel_for(sycl::range<1>((size_t)M * N),
                                        [=](sycl::id<1> i) { of[i[0]] = (float)cf[i[0]]; });
        evs.emplace_back("mkl-cvt", e3);
    };

    struct ShapeResult {
        int N;
        TimedResult mkl;
        ErrStat mkl_err;
        TimedResult simt;
        ErrStat simt_err;
        std::string best_cfg;
        TimedResult best_xmx;
        ErrStat best_err;
        std::vector<std::tuple<std::string, TimedResult, ErrStat>> all_xmx;
    };
    std::vector<ShapeResult> results;

    const double flop_scale = 1e-6;  // us -> TFLOPS: flops/(us*1e6)/1e12 = flops/us * 1e-6 ... computed below

    // Two full passes by default: pass 1 pays JIT + DVFS ramp; the summary
    // reports the final (warm) pass so MKL and XMX see identical conditions.
    for (int rep = 0; rep < reps; ++rep) {
    if (reps > 1) printf("\n######## PASS %d/%d %s ########\n", rep + 1, reps,
                         rep + 1 == reps ? "(scored)" : "(warm-up pass)");
    results.clear();
    for (int N : Ns) {
        printf("\n================ shape M=%d K=%d N=%d ================\n", M, K, N);
        const size_t MN = (size_t)M * N;
        const double flops = 2.0 * M * N * (double)K;
        auto tflops = [&](double us) { return us > 0 ? flops / (us * 1e-6) / 1e12 : 0.0; };

        // CPU double reference
        std::vector<double> ref(MN);
        for (int n = 0; n < N; ++n) {
            for (int m = 0; m < M; ++m) {
                double acc = 0.0;
                const double *wrow = &wd[(size_t)m * K];
                const half *xcol = &h_x[(size_t)n * K];
                for (int k = 0; k < K; ++k) acc += wrow[k] * (double)xcol[k];
                ref[(size_t)n * M + m] = acc;
            }
        }

        ShapeResult R;
        R.N = N;

        // ---- BASELINE-MKL
        {
            q.memset(d_out, 0, MN * sizeof(float)).wait();
            EvVec e;
            run_baseline(N, e);
            q.wait();
            q.memcpy(h_out.data(), d_out, MN * sizeof(float)).wait();
            R.mkl_err = compare(h_out.data(), ref.data(), MN);
            R.mkl = time_contender(q, warmup, iters, d_part, [&](EvVec &evs) { run_baseline(N, evs); });
            printf("BASELINE-MKL : dequant %8.2f us | gemm %8.2f us | cvt %6.2f us | total %8.2f us | %6.3f TFLOPS(total) | maxabs %.3e relscale %.3e\n",
                   R.mkl.label_us["mkl-dequant"], R.mkl.label_us["mkl-gemm"], R.mkl.label_us["mkl-cvt"],
                   R.mkl.total_us, tflops(R.mkl.total_us), R.mkl_err.max_abs, R.mkl_err.rel_to_scale);
        }

        // ---- SIMT-REF
        if (!no_simt) {
            q.memset(d_out, 0, MN * sizeof(float)).wait();
            simt_submit(q, d_w, d_x, d_out, M, N, K).wait();
            q.memcpy(h_out.data(), d_out, MN * sizeof(float)).wait();
            R.simt_err = compare(h_out.data(), ref.data(), MN);
            R.simt = time_contender(q, warmup, iters, d_part, [&](EvVec &evs) {
                evs.emplace_back("simt", simt_submit(q, d_w, d_x, d_out, M, N, K));
            });
            printf("SIMT-REF     : %38s total %8.2f us | %6.3f TFLOPS        | maxabs %.3e relscale %.3e\n",
                   "", R.simt.total_us, tflops(R.simt.total_us), R.simt_err.max_abs, R.simt_err.rel_to_scale);
        }

        // ---- XMX-FUSED configs
        double best_us = 1e30;
        for (const auto &cfg : g_cfgs) {
            if (!cfg_filter.empty() && std::string(cfg.name).find(cfg_filter) == std::string::npos) continue;
            const size_t need_slm = (size_t)cfg.kb * cfg.wg_m * sizeof(half);
            if (need_slm > slm_bytes) {
                printf("XMX %-28s: skipped (needs %zu B SLM > %zu)\n", cfg.name, need_slm, slm_bytes);
                continue;
            }
            if (M % cfg.wg_m != 0 || (K / cfg.kb) % cfg.split_k != 0) continue;
            if (N % cfg.tm != 0) continue;  // big tiles need N to fill the token rows
            if (cfg.n_inner && (K / cfg.kb) != cfg.split_k) continue;  // n-inner needs cps == 1
            try {
                q.memset(d_out, 0, MN * sizeof(float)).wait();
                EvVec e;
                cfg.run(q, d_w, d_x, d_out, d_part, M, N, K, e);
                q.wait();
                q.memcpy(h_out.data(), d_out, MN * sizeof(float)).wait();
                ErrStat err = compare(h_out.data(), ref.data(), MN);
                TimedResult t = time_contender(q, warmup, iters, d_part, [&](EvVec &evs) {
                    cfg.run(q, d_w, d_x, d_out, d_part, M, N, K, evs);
                });
                printf("XMX %-28s: fused %8.2f us | reduce %6.2f us | total %8.2f us | %6.3f TFLOPS | maxabs %.3e relscale %.3e\n",
                       cfg.name, t.label_us["xmx-fused"], t.label_us.count("xmx-reduce") ? t.label_us["xmx-reduce"] : 0.0,
                       t.total_us, tflops(t.total_us), err.max_abs, err.rel_to_scale);
                R.all_xmx.emplace_back(cfg.name, t, err);
                const bool ok = err.rel_to_scale < 5e-3;
                if (ok && t.total_us < best_us) {
                    best_us = t.total_us;
                    R.best_cfg = cfg.name;
                    R.best_xmx = t;
                    R.best_err = err;
                }
            } catch (const std::exception &ex) {
                printf("XMX %-28s: FAILED (%s)\n", cfg.name, ex.what());
            }
        }
        // Re-measure MKL after the sweep (GPU guaranteed warm) and keep the
        // faster sample — guards the comparison against residual DVFS drift.
        {
            TimedResult mkl2 = time_contender(q, warmup, iters, d_part, [&](EvVec &evs) { run_baseline(N, evs); });
            printf("BASELINE-MKL (re-run): dequant %8.2f us | gemm %8.2f us | cvt %6.2f us | total %8.2f us\n",
                   mkl2.label_us["mkl-dequant"], mkl2.label_us["mkl-gemm"], mkl2.label_us["mkl-cvt"], mkl2.total_us);
            if (mkl2.total_us < R.mkl.total_us) R.mkl = mkl2;
        }
        if (!R.best_cfg.empty()) {
            printf("--> best XMX %s: %.2f us vs MKL total %.2f us => XMX is %.2fx %s\n",
                   R.best_cfg.c_str(), R.best_xmx.total_us, R.mkl.total_us,
                   R.mkl.total_us > R.best_xmx.total_us ? R.mkl.total_us / R.best_xmx.total_us
                                                        : R.best_xmx.total_us / R.mkl.total_us,
                   R.mkl.total_us > R.best_xmx.total_us ? "FASTER" : "slower");
        }
        results.push_back(std::move(R));
    }
    }
    (void)flop_scale;

    // ---------------- summary
    printf("\n==================== SUMMARY (M=512 K=2048, %d iters) ====================\n", iters);
    printf("%4s | %10s %10s %8s %10s | %10s %8s | %8s | %-20s | %10s %10s\n",
           "N", "mkl-deq us", "mkl-gemm", "mkl-cvt", "mkl-total", "xmx-total", "xmx-TF", "speedup", "best-cfg", "mkl-maxabs", "xmx-maxabs");
    for (auto &R : results) {
        const double flops = 2.0 * M * R.N * (double)K;
        printf("%4d | %10.2f %10.2f %8.2f %10.2f | %10.2f %8.3f | %7.2fx | %-20s | %10.3e %10.3e\n",
               R.N, R.mkl.label_us["mkl-dequant"], R.mkl.label_us["mkl-gemm"], R.mkl.label_us["mkl-cvt"],
               R.mkl.total_us,
               R.best_xmx.total_us, R.best_xmx.total_us > 0 ? flops / (R.best_xmx.total_us * 1e-6) / 1e12 : 0.0,
               R.best_xmx.total_us > 0 ? R.mkl.total_us / R.best_xmx.total_us : 0.0,
               R.best_cfg.c_str(), R.mkl_err.max_abs, R.best_err.max_abs);
    }

    sycl::free(d_w, q);
    sycl::free(d_x, q);
    sycl::free(d_wf16, q);
    sycl::free(d_cf16, q);
    sycl::free(d_out, q);
    sycl::free(d_part, q);
    return 0;
}
