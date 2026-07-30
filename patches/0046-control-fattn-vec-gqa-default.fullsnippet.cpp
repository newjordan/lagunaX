// Ship: FA VEC default for GQA decode. Kill: GGML_SYCL_FATTN_FORCE_TILE=1
// notes/SHIP_20260730_fattn_vec_gqa_default.md
// Golden re-captured under this default.

    // Vector kernel for small Q batch (serial decode Q.ne[1]==1).
    // Laguna GQA previously fell through to TILE when gqa_opt_applies; VEC is
    // ~+3.6 tg64 on B70. FA numerics differ (greedy tokens diverge after ~16
    // gens) — intentional for serial tip; kill TILE restore:
    //   GGML_SYCL_FATTN_FORCE_TILE=1
    if (can_use_vector_kernel) {
        static const bool force_tile = []() {
            const char * e = getenv("GGML_SYCL_FATTN_FORCE_TILE");
            return e != nullptr && e[0] != '\0' && e[0] != '0';
        }();
        if (!ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {
            if (Q->ne[1] == 1 && !force_tile) {
                return BEST_FATTN_KERNEL_VEC;
            }
            // force_tile or Q batch > 1: continue
            if (Q->ne[1] == 1 && force_tile && !gqa_opt_applies) {
                return BEST_FATTN_KERNEL_VEC; // stock non-GQA still VEC
            }
        } else {
            if (Q->ne[1] <= 2) {
                return BEST_FATTN_KERNEL_VEC;
            }
        }
    }
    return BEST_FATTN_KERNEL_TILE;
}
