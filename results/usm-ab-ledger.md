# USM axis A/B ledger — Level Zero driver/allocator policy (vs same-window CTRL)
# Drift bound: ±0.68% tg (between-run card-state, see knob-ab-ledger.md).
| stamp | arm | tg_tok_s | pp_tok_s | note |
| 20260806T111255Z | CTRL | tg=137.860783 | pp=1159.421102 | same-window control (no extra env) — bounds ambient drift |
| 20260806T111333Z | USM_DEVICE_RESIDENT | tg=137.771867 | pp=1162.071903 | pin device allocations device-resident (USM_RESIDENT_DEVICE=1) |
| 20260806T111412Z | USM_ALLOCATOR | tg=137.890587 | pp=1158.908869 | plugin-managed USM allocator (USE_USM_ALLOCATOR=1) |
| 20260806T111603Z | POLL_0 | tg=137.908336 | pp=1152.237488 | host submission busy-poll (LX_POLL=0 vs default 50) |
