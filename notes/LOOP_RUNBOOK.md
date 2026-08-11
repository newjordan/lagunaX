# Runbook — the endless lx improvement loop (2026-08-06)

Operator-facing. The agent's rules live in `AGENTS.md` (loaded every iteration).

## Why the last run stopped

It didn't crash and it wasn't waiting on the GPU. `loop_ctl.rs` pauses a loop
when the model claims done and **no acceptance command is bound**:

> "model reported done, but no acceptance command is bound; /loop resume to keep
> going or /goal cmd \<check\> to verify"

State was `status: paused`, `accept_cmd: null`, iteration 33. Binding an
acceptance command is what makes a "done" claim get *verified* instead of
parking the run. That is step 3 below.

## Launch sequence

**1. Restart the cockpit.** Required — the running process holds its loop state
and its stale steer notes in memory, and the harness fix only lands in a fresh
binary. It is idle, so nothing is lost.

```
/quit          # then, from the shell:
angelX /home/frosty40/turbo/lx
```

**2. Start the loop.**

```
/loop improve Laguna XS 2.1 serial decode and prefill on the B70 by changing kernel source in benchmark/kernel/. One change at a time, each gated per AGENTS.md. No regressions.
```

Use `/loop`, **not** `/loop podrace`. Podrace is the harness's endless mode, but
it is also one of the phrases that arms competition posture ("mutate → validate
→ SUBMIT → score", plus a tight recon budget). Endless without that posture is
what step 3 buys.

**3. Make it endless, and bind the gate.**

```
/loop endless                            # max_iters = 0
/loop stall 0                            # stall_stop = 0 disables the stall-stop entirely
/loop pivot 2                            # still change direction after 2 stale iterations
/goal cmd /bin/bash scripts/loop-accept.sh    # a done claim gets verified, not parked
```

`stall_stop = 0` is the key one — `stall_limit_reached()` is
`st.stall_stop > 0 && st.stale_count >= st.stall_stop`, so zero means it never
fires, so the loop never escalates to the SOTA tier (which would block on an
approval prompt nobody is there to answer) and never stops itself.

`loop-accept.sh` passes only when the board reaches `LX_TARGET_SCORE`
(default 1.40, currently 1.218) **with a passing KLD receipt and a real
promotion**. Until then it keeps working. Raise the target if you want it to run
longer; it cannot be satisfied by a broken build.

**4. Do not re-add the old steer notes.** These rode every iteration of the last
run and are half of why it ground:

> "dont let anythign stop you from porgress, and your to run and improve the
> model for a couple days on end. so... just keep at it."

`AGENTS.md` now carries the standing direction, and unlike steer notes it can be
edited without restarting anything.

## Model

Already routed (`~/.angel/brain-route.json`): `deepseek-v4-flash`, effort `high`,
agent `sota`. Verified live — the key answers, the model id resolves, reasoning
tokens come back.

Cost is the thing to watch, since this runs unattended on a metered key:

```
curl -s https://api.deepseek.com/user/balance -H "Authorization: Bearer $DEEPSEEK_API_KEY" | jq .
```

Balance was **$44.97** at launch; the previous 33-iteration run spent ~320k
tokens. If you want a hard stop instead of watching, `/loop budget <tokens>`
territory is `ANGEL_LOOP_TOKEN_BUDGET` (harness default 2,000,000; the last run
had it set to 0 = unlimited).

## What it should be working on

`benchmark/kernel/` builds now (oneAPI/MKL link fixes applied 2026-08-06). The
knob axes are swept out; the remaining ground is kernel source. First lead in
`AGENTS.md`: the honest source champion benches prefill ~1.02x — find where the
batched path really differs, carefully.
