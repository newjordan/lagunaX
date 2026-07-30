// Ship: mul_mat+add residual alias (Q6 shexp). Default ON.
// Replace ggml_sycl_fuse_mul_mat_add in ggml-sycl.cpp
// Kill: GGML_SYCL_DISABLE_MUL_MAT_ADD_FUSE=1
// notes/SHIP_20260730_mul_mat_add_shexp_alias.md

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
    // Any batch size: multi-col 2..8 uses switch_ncols addend; larger uses per-col loop offset.
    if (x->ne[1] < 1) {
        return 0;
    }
    // Only types with reorder MMVQ addend epilogue wired (Q4_K/Q5_K/Q6_K).
    if (w->type != GGML_TYPE_Q4_K && w->type != GGML_TYPE_Q5_K && w->type != GGML_TYPE_Q6_K) {
        return 0;
    }
    // Fuse span may include skipped views between mul_mat and add.
    const int n_span = j_add - i + 1;
    std::vector<ggml_op> ops(n_span, GGML_OP_NONE);
    ops[0] = GGML_OP_MUL_MAT;
    for (int k = 1; k < n_span - 1; ++k) {
        ops[k] = cgraph->nodes[i + k]->op;
    }
    ops[n_span - 1] = GGML_OP_ADD;
    const int output = j_add;
    if (!ggml_can_fuse_subgraph(cgraph, i, n_span, ops.data(), &output, 1)) {
        return 0;
    }
    // residual may alias add (in-place ADD on residual, e.g. Laguna shexp:
    // add(moe_out, ffn_shexp) often reuses moe_out). MMVQ epilogue is
    // dst[row] = sum + addend[row] — row-parallel, safe when addend==dst.
    // (Prior reject of residual==add blocked Q6_K shexp down+add fuse.)

    // Write gemv+residual into ADD output; elide MUL_MAT intermediate.
    void * saved_mm_data = mm->data;
    mm->data = add->data;
    ggml_sycl_mmvq_set_row_addend(static_cast<const float *>(residual->data));
    ggml_sycl_mul_mat(ctx, mm->src[0], mm->src[1], mm);
    ggml_sycl_mmvq_set_row_addend(nullptr);
    mm->data = saved_mm_data;

    {
        // First hit per weight type (q4 o_proj; q6 shexp down + dense residual).
        static std::atomic<int> once_q4{0};
        static std::atomic<int> once_q6{0};
        std::atomic<int> * once = (w->type == GGML_TYPE_Q6_K) ? &once_q6 : &once_q4;
        if (w->type != GGML_TYPE_Q5_K && once->fetch_add(1) == 0) {
            fprintf(stderr,
                    "[lx-control-mm-add] fuse hit (mul_mat+add) ne0=%" PRId64 " ne1=%" PRId64
                    " wtype=%s alias_res=%d mm='%s' add='%s'\n",
                    add->ne[0], add->ne[1], ggml_type_name(w->type),
                    (int) (residual->data == add->data),
                    mm->name ? mm->name : "?", add->name ? add->name : "?");
        }
    }
    return j_add - i; // skip through ADD (and any elided views)
}

