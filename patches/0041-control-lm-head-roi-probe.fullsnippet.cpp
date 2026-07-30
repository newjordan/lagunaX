// Research snippet — lm_head ROI probe (default OFF). Not a scored tip.
// Apply conceptually to ggml-sycl/mmvq.cpp:
//  1) add helpers after includes
//  2) replace reorder_mul_mat_vec_q6_k_q8_1_sycl body
// Envs: GGML_SYCL_PROFILE_LM_HEAD=1, GGML_SYCL_LM_HEAD_ROW_LIMIT=N
// See notes/SHIP_20260730_lm_head_roi_ceiling.md

// Research: large-vocab Q6_K lm_head (Laguna output.weight 2048×100352).
// Default OFF — tip path unchanged when unset.
//   GGML_SYCL_PROFILE_LM_HEAD=1     → host-timed large-nrows Q6_K GEMV (stderr)
//   GGML_SYCL_LM_HEAD_ROW_LIMIT=N   → compute only first N rows; rest = -inf (ceiling / prune ROI; not golden)
static int ggml_sycl_lm_head_row_limit() {
    static int lim = []() {
        const char * e = getenv("GGML_SYCL_LM_HEAD_ROW_LIMIT");
        return (e && e[0]) ? atoi(e) : 0;
    }();
    return lim;
}

static bool ggml_sycl_profile_lm_head() {
    static int on = []() {
        const char * e = getenv("GGML_SYCL_PROFILE_LM_HEAD");
        return (e && e[0] != '\0' && e[0] != '0') ? 1 : 0;
    }();
    return on != 0;
}


static void reorder_mul_mat_vec_q6_k_q8_1_sycl(const void * vx, const void * vy, float * dst, const int ncols,
                                               const int nrows, dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    // Round up to a whole number of subgroup-sized workgroups; out-of-range rows are skipped inside the kernel.
    constexpr size_t num_subgroups = WARP_SIZE;

    // Research ceiling: only GEMV first N rows of large vocab (lm_head). Mask tail to -inf.
    // Active only when GGML_SYCL_LM_HEAD_ROW_LIMIT>0 and nrows is lm_head-scale.
    int nrows_compute = nrows;
    const int row_lim = ggml_sycl_lm_head_row_limit();
    const bool is_lm_head_scale = (nrows >= 65536 && ncols == 2048);
    if (row_lim > 0 && is_lm_head_scale && row_lim < nrows) {
        nrows_compute = row_lim;
        static std::atomic<int> once{0};
        if (once.fetch_add(1) == 0) {
            fprintf(stderr,
                    "[lx-lm-head] ROW_LIMIT=%d of nrows=%d ncols=%d (research ceiling; not golden-safe)\n",
                    row_lim, nrows, ncols);
        }
    }

    const int block_num_y = ceil_div(nrows_compute, GGML_SYCL_MMV_Y * (int) num_subgroups);
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, num_subgroups * WARP_SIZE);
    const float * row_addend = g_mmvq_row_addend;
    const bool do_profile = ggml_sycl_profile_lm_head() && is_lm_head_scale;

    if (do_profile) {
        stream->wait();
    }
    const auto t0 = std::chrono::steady_clock::now();

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q6_K>>(
                                 vx, vy, dst, ncols, nrows_compute, nd_item, row_addend);
                         });
    });

    // Mask unused vocab rows so argmax cannot pick garbage (still wrong logits for golden).
    if (nrows_compute < nrows) {
        const int n_tail = nrows - nrows_compute;
        float * tail = dst + nrows_compute;
        const float neg_inf = -std::numeric_limits<float>::infinity();
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(sycl::range<1>(n_tail), [=](sycl::id<1> i) {
                tail[i] = neg_inf;
            });
        });
    }

    if (do_profile) {
        stream->wait();
        const auto t1 = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        static std::atomic<long long> n_calls{0};
        static std::atomic<long long> sum_us{0};
        const long long us = (long long) (ms * 1000.0);
        const long long c  = n_calls.fetch_add(1) + 1;
        const long long s  = sum_us.fetch_add(us) + us;
        if (c <= 3 || (c % 32) == 0) {
            fprintf(stderr,
                    "[lx-lm-head] Q6_K GEMV nrows_compute=%d/%d %.3f ms  avg=%.3f ms over %lld calls\n",
                    nrows_compute, nrows, ms, (s / 1000.0) / (double) c, c);
        }
    }
}

