// Ship: mul_mat+add+add decode (ne11<=32). Default ON with mm-add fuse.
// notes/SHIP_20260730_mul_mat_add_add_decode.md
// Kill: GGML_SYCL_DISABLE_MUL_MAT_ADD_FUSE=1
// Also needs mmvq.hpp/cpp row_addend2 support.

// Graph fuse: MUL_MAT → ADD residual (Laguna o_proj + inpSA; shexp down + moe).
// Reorder MMVQ writes gemv+addend into ADD buffer (one launch). Default ON.
// Kill: GGML_SYCL_DISABLE_MUL_MAT_ADD_FUSE=1
int ggml_sycl_fuse_mul_mat_add(
    ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int i) {
    static const bool enabled = []() {
        const char * dis = getenv("GGML_SYCL_DISABLE_MUL_MAT_ADD_FUSE");
        return !(dis != nullptr && std::atoi(dis) != 0);
    }();
    if (!enabled || i + 1 >= cgraph->n_nodes) {
        return 0;
    }
    ggml_tensor * mm  = cgraph->nodes[i];
    ggml_tensor * add = cgraph->nodes[i + 1];
    if (mm->op != GGML_OP_MUL_MAT) {
        return 0;
    }
    // Skip non-ADD neighbors but allow VIEW/NONE between mul_mat and add (rare).
    int j_add = i + 1;
    while (j_add < cgraph->n_nodes && j_add <= i + 3) {
        ggml_tensor * n = cgraph->nodes[j_add];
        if (n->op == GGML_OP_ADD) {
            break;
        }
        if (n->op == GGML_OP_VIEW || n->op == GGML_OP_RESHAPE || n->op == GGML_OP_PERMUTE ||
            n->op == GGML_OP_TRANSPOSE || n->op == GGML_OP_NONE || ggml_is_empty(n)) {
            j_add++;
            continue;
        }
        break;
    }
    if (j_add >= cgraph->n_nodes || cgraph->nodes[j_add]->op != GGML_OP_ADD) {
        return 0;
    }
    add = cgraph->nodes[j_add];
    if (add->src[0] != mm && add->src[1] != mm) {
        return 0;
    }
    const ggml_tensor * residual = add->src[0] == mm ? add->src[1] : add->src[0];
    if (!residual || !mm->src[0] || !mm->src[1] || !residual->data || !add->data) {
        return 0;
    }
    // F32 contiguous same-shape residual; decode-ish activation batch (ne11==1)
    // or small prefill (ne11<=8) — MMVQ path. Prefer decode for score.
    if (mm->type != GGML_TYPE_F32 || add->type != GGML_TYPE_F32 ||
        residual->type != GGML_TYPE_F32 ||
        !ggml_are_same_shape(mm, add) || !ggml_are_same_shape(mm, residual) ||
        !ggml_is_contiguous(add) || !ggml_is_contiguous(residual) ||
        !ggml_is_contiguous(mm->src[1])) {
        return 0;
    }
    const ggml_tensor * w = mm->src[0];
    const ggml_tensor * x = mm->src[1];
    if (!ggml_is_quantized(w->type) || x->type != GGML_TYPE_F32) {
        return 0;
    }
    // QUALITY-SAFE (2026-07-31): decode only (ne11==1). Any-batch prefill fuse
    // wrecks wikitext PPL (1e5–1e6) — GEMM/large-N paths drop residual addends while
    // still eliding the graph ADD. Decode reorder-MMVQ epilogue stays correct.
    // Opt-in any-batch research: GGML_SYCL_ENABLE_MUL_MAT_ADD_ANY_BATCH=1 (not quality-safe).
    if (x->ne[1] < 1) {
        return 0;
    }
    {
        static const bool any_batch = []() {
            const char * e = getenv("GGML_SYCL_ENABLE_MUL_MAT_ADD_ANY_BATCH");
            return e != nullptr && std::atoi(e) != 0;
        }();
        if (!any_batch && x->ne[1] != 1) {
            return 0;
        }
    }
    // Only types with reorder MMVQ addend epilogue wired (Q4_K/Q5_K/Q6_K).
    if (w->type != GGML_TYPE_Q4_K && w->type != GGML_TYPE_Q5_K && w->type != GGML_TYPE_Q6_K) {
        return 0;
    }
    // Optional second ADD: (mul + r0) + r1  e.g. Laguna shexp+moe then +ffn_inp.
    // Decode / small-batch only (ne11<=32): MMVQ epilogue carries addends. Large prefill
    // may take GEMM or per-col MMVQ×N and regressed formal pp (~−1250 t/s).
    ggml_tensor *        add2       = nullptr;
    const ggml_tensor *  residual2  = nullptr;
    int                  j_add2     = j_add;
    if (x->ne[1] <= 32) {
        int j = j_add + 1;
        while (j < cgraph->n_nodes && j <= j_add + 3) {
            ggml_tensor * n = cgraph->nodes[j];
            if (n->op == GGML_OP_ADD) {
                break;
            }
            if (n->op == GGML_OP_VIEW || n->op == GGML_OP_RESHAPE || n->op == GGML_OP_PERMUTE ||
                n->op == GGML_OP_TRANSPOSE || n->op == GGML_OP_NONE || ggml_is_empty(n)) {
                j++;
                continue;
            }
            break;
        }
        if (j < cgraph->n_nodes && cgraph->nodes[j]->op == GGML_OP_ADD) {
            ggml_tensor * cand = cgraph->nodes[j];
            if ((cand->src[0] == add || cand->src[1] == add) && cand->type == GGML_TYPE_F32 &&
                ggml_are_same_shape(add, cand) && ggml_is_contiguous(cand) && cand->data) {
                const ggml_tensor * r2 = cand->src[0] == add ? cand->src[1] : cand->src[0];
                if (r2 && r2->type == GGML_TYPE_F32 && r2->data && ggml_are_same_shape(add, r2) &&
                    ggml_is_contiguous(r2)) {
                    add2      = cand;
                    residual2 = r2;
                    j_add2    = j;
                }
            }
        }
    }

    // Fuse span may include skipped views between mul_mat and add(s).
    const int n_span = j_add2 - i + 1;
    // Both lookahead windows are bounded to three intervening nodes, so the
    // complete double-ADD span is at most seven nodes. Keep this decode-hot
    // control path on the stack instead of allocating a vector every graph run.
    ggml_op ops[7] = {};
    for (int k = 0; k < n_span; ++k) {
        ops[k] = cgraph->nodes[i + k]->op;
    }
    const int output = j_add2;
    if (!ggml_can_fuse_subgraph(cgraph, i, n_span, ops, &output, 1)) {
        // Fall back to single-ADD span if double-ADD can_fuse fails.
        if (add2) {
            add2      = nullptr;
            residual2 = nullptr;
            j_add2    = j_add;
            const int n_span1 = j_add - i + 1;
            ggml_op ops1[4] = {};
            for (int k = 0; k < n_span1; ++k) {
                ops1[k] = cgraph->nodes[i + k]->op;
            }
            const int out1 = j_add;
            if (!ggml_can_fuse_subgraph(cgraph, i, n_span1, ops1, &out1, 1)) {
                return 0;
            }
        } else {
            return 0;
        }
    }
    // residual may alias add (in-place ADD on residual, e.g. Laguna shexp:
    // add(moe_out, ffn_shexp) often reuses moe_out). MMVQ epilogue is
    // dst[row] = sum + addend[row] [+ addend2] — row-parallel, safe when addend==dst.

    // Write gemv+residual(+residual2) into final ADD output; elide intermediates.
    ggml_tensor * dst_add = add2 ? add2 : add;
    void * saved_mm_data = mm->data;
    mm->data = dst_add->data;
    ggml_sycl_mmvq_set_row_addend(static_cast<const float *>(residual->data));
    if (residual2) {
        ggml_sycl_mmvq_set_row_addend2(static_cast<const float *>(residual2->data));
    }
    ggml_sycl_mul_mat(ctx, mm->src[0], mm->src[1], mm);
    ggml_sycl_mmvq_set_row_addend(nullptr);
    ggml_sycl_mmvq_set_row_addend2(nullptr);
    mm->data = saved_mm_data;

    // Decode executes this path once per eligible graph evaluation. Keep diagnostics
    // opt-in so normal inference does not perform an atomic load for every fused op.
    static const bool trace_fuse = []() {
        const char * e = getenv("GGML_SYCL_TRACE_MUL_MAT_ADD_FUSE");
        return e != nullptr && std::atoi(e) != 0;
    }();
    if (trace_fuse) {
        static std::atomic<int> once_single{0};
        static std::atomic<int> once_double{0};
        std::atomic<int> * once = residual2 ? &once_double : &once_single;
        if (once->load(std::memory_order_relaxed) == 0 &&
              once->exchange(1, std::memory_order_relaxed) == 0) {
            fprintf(stderr,
                    "[lx-control-mm-add] fuse hit (mul_mat+add%s) ne0=%" PRId64 " ne1=%" PRId64
                    " wtype=%s alias_res=%d mm='%s' add='%s' add2='%s'\n",
                    residual2 ? "+add" : "",
                    dst_add->ne[0], dst_add->ne[1], ggml_type_name(w->type),
                    (int) (residual->data == add->data),
                    mm->name ? mm->name : "?", add->name ? add->name : "?",
                    (residual2 && add2 && add2->name) ? add2->name : "-");
        }
    }
    return j_add2 - i; // skip through final ADD (and any elided views)
}

