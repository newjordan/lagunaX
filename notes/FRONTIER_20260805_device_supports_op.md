# FRONTIER: SYCL scheduler op-support query overhead (device_supports_op)

## Direction
The ggml scheduler's per-node backend-capability query `ggml_backend_sycl_device_supports_op()`
— the dispatch-decision layer that runs BEFORE any kernel submit, fusion, or graph
capture — is the single most-invoked SYCL host function in the decode trace.

No prior direction (1–24) examined this layer. They all analyzed what happens AFTER
a node is dispatched (kernel launch count, fusion, graph capture, precision, memory
pools). The scheduler's decision overhead is upstream of all of them.

## Evidence

From the ggml-sycl call trace (decode-ggml), sorted by call count:

| SYCL function                     | calls    |
|-----------------------------------|----------|
| device_supports_op                | 677,040  |
| buffer_init_tensor                |  76,973  |
| synchronize                       |   6,056  |
| buffer_set_tensor                 |   5,573  |
| get_tensor_async                  |     544  |
| buffer_clear                      |      48  |
| host_buffer_type                  |      48  |
| init                              |      24  |
| buffer_reset                      |      24  |

Source: `grep -oP '\[SYCL\] call \K\w+' results/ktrace-tip-20260730/decode-ggml/trace.log | sort | uniq -c | sort -rn`

## Key ratios
- 677,040 device_supports_op / 6,056 synchronize = **111.8×** more capability queries than device syncs
- 677,040 device_supports_op / 76,973 buffer_init_tensor = **8.8×** more capability queries than tensor inits
- 677,040 device_supports_op / 128 (tg128 decode steps) = **~5,290 queries per decode step**

## Why this is new
- Direction 1 (dispatch coalescing): looks at kernel *launch* count, not the *pre-launch* query
- Direction 11 (serialization barrier): looks at oneDNN cache_hit interleaving, not the scheduler query
- Direction 18 (ggml-backend dispatch): looks at the native-quantized *path selection*, not the per-op *support check*
- Direction 21 (graph-capture block chain): looks at runtime graph capture, not scheduler graph allocation
- None of 1–24 measured `device_supports_op` call frequency

## Open questions
- Is device_supports_op called once per node per graph build, or multiple times?
  If the MoE graph is rebuilt every decode step (dynamic expert routing), every
  node in all 40 layers is re-queried → 5,290 queries/step is consistent with
  ~130 nodes/layer × 40 layers ≈ 5,200
- Is the query itself cheap (a switch on op type), or does it do device-feature
  probing that costs host cycles?
- Could the scheduler cache support results per (op_type, src_type, ...),
  eliminating repeated queries for the same op signature across steps?
