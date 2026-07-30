// Research snippet — expert-loop pack before counts-wait (default ON, flat score).
// In ggml_sycl_mul_mat_id_dual_down_multitoken_expert_loop device-sort path.
// See notes/SHIP_20260730_expert_loop_pack_overlap.md

// Keep counts/next alive across pack so we can D2H-wait AFTER submitting pack
    // (pack only needs device mapping — host offsets can build while pack runs).
    ggml_sycl_pool_alloc<int> dev_counts;
    ggml_sycl_pool_alloc<int> dev_next;

    const bool use_device_sort =
        ggml_sycl_mmid_device_sort_enabled() && ids->nb[0] == sizeof(int32_t);

    if (use_device_sort) {
        const int n_tokens_i = (int) ne12;
        const int n_ids_i    = (int) n_ids;
        const int n_as_i     = (int) n_as;
        const int ids_stride = (int) (ids->nb[1] / sizeof(int32_t));
        const int32_t * ids_d = (const int32_t *) ids->data;
        dev_counts.alloc(ctx.pool(), n_as);
        dev_next.alloc(ctx.pool(), n_as);
        SYCL_CHECK(CHECK_TRY_ERROR(stream->memset(dev_counts.get(), 0, sizeof(int) * n_as)));
        constexpr int BS = 256;
        const int total   = n_ids_i * n_tokens_i;
        const int nblocks = (total + BS - 1) / BS;
        int * counts_d = dev_counts.get();
        int * next_d   = dev_next.get();
        stream->parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t) nblocks * BS), sycl::range<1>(BS)),
            [=](sycl::nd_item<1> item) {
                mmid_device_count_experts(ids_d, counts_d, n_ids_i, n_tokens_i, ids_stride, n_as_i, item);
            });
        stream->parallel_for(
            sycl::nd_range<1>(sycl::range<1>(1), sycl::range<1>(1)),
            [=](sycl::nd_item<1> item) {
                mmid_device_exclusive_scan(counts_d, next_d, n_as_i, item);
            });
        mmid_row_mapping * map_d = dev_row_mapping.get();
        stream->parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t) nblocks * BS), sycl::range<1>(BS)),
            [=](sycl::nd_item<1> item) {
                mmid_device_fill_mapping(ids_d, next_d, map_d, n_ids_i, n_tokens_i, ids_stride, n_as_i, item);
            });
        host_counts_device.resize(n_as);
        // Async D2H only — do NOT host-wait yet. Pack needs mapping only.
        counts_ready_ev = stream->memcpy(host_counts_device.data(), counts_d, sizeof(int) * n_as);
        device_sort_active = true;
    } else {
        std::vector<char> ids_host(ggml_nbytes(ids));
        SYCL_CHECK(CHECK_TRY_ERROR(
            stream->memcpy(ids_host.data(), ids->data, ggml_nbytes(ids))));
        SYCL_CHECK(CHECK_TRY_ERROR(stream->wait()));
        std::vector<mmid_row_mapping> & routed_row_src = ctx.mmid_row_mapping_host;
        mmid_counting_sort_rows(ids, ids_host.data(), n_ids, n_as, n_routed_rows,
                                expert_row_counts, expert_row_offsets, routed_row_src);
        SYCL_CHECK(CHECK_TRY_ERROR(
            stream->memcpy(dev_row_mapping.get(), routed_row_src.data(),
                           n_routed_rows * sizeof(mmid_row_mapping))));
    }

    // Pack activations once (device mapping only — no host counts needed).
    // Submitted before counts D2H wait so host offset build can overlap pack on GPU.
    {
        const unsigned int max_wg = ggml_sycl_info().max_work_group_sizes[ctx.device];
        sycl::range<3> block_dims(1, 1, std::min((unsigned int) ne10, max_wg));
        sycl::range<3> grid_dims(1, 1, n_routed_rows);
        char * src1_contig = src1_contiguous.get();
        mmid_row_mapping * map_d = dev_row_mapping.get();
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1) {
                    k_copy_src1_to_contiguous(
                        src1_original, src1_contig, map_d,
                        ne11, ne10, nb11, nb12, item_ct1);
                });
        });
    }

    // Host offsets after pack is queued: D2H wait returns when counts ready;
    // pack is already in-flight (or finished) on the in-order stream after D2H.
    // Note: on pure in-order, pack starts after D2H device-side — host wait then
    // overlaps host exclusive-scan build with pack execution.
    if (device_sort_active) {
        counts_ready_ev.wait();
        expert_row_counts.assign(n_as, 0);
        expert_row_offsets.assign(n_as + 1, 0);
        for (int64_t e = 0; e < n_as; ++e) {
            expert_row_counts[e] = host_counts_device[e];
            expert_row_offsets[e + 1] = expert_row_offsets[e] + expert_row_counts[e];
        }
        GGML_ASSERT(expert_row_offsets[n_as] == n_routed_rows);
    }
    // dev_counts/dev_next free at function end (after all stream uses of mapping).

    