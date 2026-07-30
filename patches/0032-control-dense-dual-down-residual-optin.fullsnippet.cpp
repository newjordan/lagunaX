        (int) ncols, (int) nrows, (int) ncols_dst, bytes_per_qrow, dst_col_stride,
        stream);
}

// Graph fuse: MUL_MAT + MUL_MAT + GLU(swiglu) for dense shared expert.
// Prefer dual+down+residual: ... + MUL_MAT(down) + ADD when present (shexp → moe residual).
// Kill dual: DISABLE_DENSE_DUAL_SWIGLU=1
// Kill dual+down only: DISABLE_DENSE_DUAL_DOWN=1
int ggml_sycl_fuse_dense_dual_swiglu(
    ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int i) {
    if (i + 2 >= cgraph->n_nodes) {
        return 0;
    }
    ggml_tensor * node = cgraph->nodes[i];
    if (node->op != GGML_OP_MUL_MAT) {
        return 0;
    }
    ggml_tensor * mm1 = cgraph->nodes[i + 1];
    ggml_tensor * glu = cgraph->nodes[i + 2];
    if (mm1->op != GGML_OP_MUL_MAT || glu->op != GGML_OP_GLU) {
        return 0;
    }
    if (ggml_get_glu_op(glu) != GGML_GLU_OP_SWIGLU) {
        return 0;
    }
    const bool edges_ok =
        ((glu->src[0] == node && glu->src[1] == mm1) ||
         (glu->src[0] == mm1 && glu->src[1] == node)) &&
        node->src[1] == mm1->src[1];
    if (!edges_ok) {
        return 0;
    }
    ggml_tensor * gate = glu->src[0];
    ggml_tensor * up   = glu->src[1];

    // dual + down + residual ADD (Laguna shexp / dense residual).
    // Decode-only (ncols<=32): residual addend is MMVQ-only.
    // OPT-IN: GGML_SYCL_ENABLE_DENSE_DUAL_DOWN=1 (formal 20260730T103913Z +50.06%
    // under tip +50.93%; keep off until ≥ tip). Kill: DISABLE_DENSE_DUAL_DOWN=1.
    static const bool enable_dual_down = []() {
        const char * en  = getenv("GGML_SYCL_ENABLE_DENSE_DUAL_DOWN");
        const char * dis = getenv("GGML_SYCL_DISABLE_DENSE_DUAL_DOWN");
        if (dis != nullptr && std::atoi(dis) != 0) {
            return false;
        }
        return en != nullptr && std::atoi(en) != 0;
    }();
    if (enable_dual_down && i + 4 < cgraph->n_nodes) {
        ggml_tensor * down = cgraph->nodes[i + 3];
        ggml_tensor * add  = cgraph->nodes[i + 4];
        if (down->op == GGML_OP_MUL_MAT && down->src[1] == glu &&
            add->op == GGML_OP_ADD &&
            (add->src[0] == down || add->src[1] == down) &&
            down->src[0] &&
            (down->src[0]->type == GGML_TYPE_Q4_K || down->src[0]->type == GGML_TYPE_Q5_K ||
             down->src[0]->type == GGML_TYPE_Q6_K) &&
            down->type == GGML_TYPE_F32 && add->type == GGML_TYPE_F32 &&
            ggml_are_same_shape(down, add) &&
            ggml_is_contiguous(add) && ggml_is_contiguous(glu)) {
            const ggml_tensor * residual =
                add->src[0] == down ? add->src[1] : add->src[0];
            const int64_t ncols = gate->ne[1];
            // residual may alias add (in-place ADD); MMVQ dst[row]=sum+addend[row] is safe.
            if (residual && residual->type == GGML_TYPE_F32 && residual->data &&
                ggml_are_same_shape(residual, add) && ggml_is_contiguous(residual) &&
                down->src[0]->ne[0] == glu->ne[0] &&
                down->src[0]->ne[1] == down->ne[0] &&
                ncols <= 32 &&
                ggml_can_fuse_subgraph(
                    cgraph, i,
                    { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT, GGML_OP_GLU, GGML_OP_MUL_MAT,
                      GGML_OP_ADD },
                    { i + 4 }) &&
                ggml_sycl_mul_mat_dense_dual_swiglu_fused(ctx, gate, up, glu)) {
                void * saved = down->data;
                if (down->data != add->data) {
                    down->data = add->data;
                }
                ggml_sycl_mmvq_set_row_addend(static_cast<const float *>(residual->data));
                ggml_sycl_mul_mat(ctx, down->src[0], down->src[1], down);
                ggml_sycl_mmvq_set_row_addend(nullptr);
                down->data = saved;
                {
                    static std::atomic<int> once{0};
                    if (once.fetch_add(1) == 0) {
                        fprintf(stderr,
                                "[lx-control-dense-dual] fuse hit (dual+down+residual) "
                                "ncols_dst=%" PRId64 " embd=%" PRId64 "\n",
                                gate->ne[1], add->ne[0]);
                    }
                }
                return 4; // skip up, glu, down, add
            }
        }
    }

    const int output = i + 2;
    if (!ggml_can_fuse_subgraph(
            cgraph, i,
            { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT, GGML_OP_GLU },
            { output })) {
        return 0;
    }
    if (ggml_sycl_mul_mat_dense_dual_swiglu_fused(ctx, gate, up, glu)) {
        static std::atomic<int> once{0};
        if (once.fetch_add(1) == 0) {
            fprintf(stderr,
                    "[lx-control-dense-dual] fuse hit (shared gate+up+swiglu) ncols_dst=%" PRId64 "\n",
                    gate->ne[1]);
        }
        return 2;
    }
    return 0;
}

