# Spark development, November 1

SSH and sudo access work through `tools/spark_ssh.py`. It loads the existing
`SSH_NAME` and `SSH_PASS` variables locally, sends passwords only to authentication
prompts, suppresses transport diagnostic buffers, and redacts connection details.
Do not print the dotenv file or put credential values in commands, source, logs,
notes, or uploads. Code transfers used an explicit whitelist, excluding every
.env file, the SSH helper, and the BF16 project directory.

The remote development workspace is `/var/tmp/codex-muse-spark/src`.
Docker CDI (`nvidia.com/gpu=all`) exposes the GB10; per-command sudo is used.
No Docker group changes, daemon configuration changes, or host Python installs
were needed. The cached public image is `nvcr.io/nvidia/pytorch:25.11-py3`, digest
`sha256:417cbf33f87b5378849df37983552cd1f8bc8b62fe1ceabe004de816a55dff21`.
It provides PyTorch `2.10.0a0+b558c986e8.nv25.11` and CUDA/nvcc13.0.
Hardware:48SMs, CC12.1,121.7GiB memory,101376-byte opt-in shared memory/block.
All development containers finished and were removed; sources/caches/results stay.

## Validated native subset

`tools/spark_tiny_port.py` is a **developer-only**, non-submittable1..16-token
port: native BF16 warp MMA and TMA, one ring slot, bounded shared routing arrays,
no Hopper WGMMA. It passed24functional cases and192mixed graph replays: normal,
skewed, all-local, and empty routing forM1,2,4,8,9,16. Nonempty cases use the supplied
independent FP32 reference; empty cases require exact zeros. GB10 timings in
`validation_result.json` are **not H200 scores or predictions**.

## H200 offline compiler diagnostics

`tools/hopper_crosscompile.py` targets `sm_90a`, without executing on H200.
Exact device CUDA and compile flags are used; Torch host launchers are replaced
by explicit kernel template instantiations. Register counts agree with earlier
H200 observations. All11existing configurations compiled:

| Variant | Registers | Spill stores / loads reported by ptxas |
|---|---:|---:|
| Published tiny / tiny2 |149|0/0bytes|
| Published mid |152|0/0bytes|
| Published main |96|0/0bytes|
| Capped tiny NS2 |96|112/48bytes|
| Distributed M1 NS2 / NS3 |96|0/0bytes|
| Four-warp NS2 |168|72/72bytes|
| Narrow-N GS4 main |96|144/308bytes|
| Batched-release / compact main |96|0/0bytes|

These are static compiler diagnostics, not dynamic traffic counts or proof of a
runtime speedup. Full reports are in `hopper_result.json`; CUBIN/SASS/source
artifacts remain under the remote workspace's `hopper_build/`.

## New two-warpgroup candidate

`cuda_persistent_two_wg.py` is **H200-UNVERIFIED** and has not been submitted.
Two physical warpgroups each handle two logical64-row tiles, preserving all four
logical tiles, packed weights, fixup addresses and BF16/FP32 math boundaries.
Ready publications count2logical tiles after both stores; split-unit counters
coalesce per physical group. Gather and finalization each cover two chunks per
thread. The intent is fewer group-control operations and a higher register budget.

The refactored4-group control compiles at96registers with no spills. The2-group
candidate compiles at168registers with8bytes of spill stores/loads; disassembly
places that pointer spill near setup/final grid-barrier code, outside theKloop.
KeepingGS2atwideN instead adds20/16bytes of spills, so the current candidate uses
GS1forNTT>=5. See `two_wg_hopper_result.json`.

`tools/spark_wgmma_emulate.py` replaces WGMMA with synchronous warp-level MMA,
shrinks rings to one slot, and retires eachxslot synchronously at stage end.
The published4-group operator, refactored4-group control, and new2-group operator
**each passed44functional cases and352graph replays** onGB10: M1,4,8,16,17,32,33,
128,256,257,385; normal, skewed, all-local, empty routing. Nonempty cases use an
independent FP32 reference. This validates tile mapping, math and the emulated
protocol, **not Hopper asynchronous WGMMA retirement or H200 performance**.
Initial emulator testing caught an incorrect contiguousBfragment load in the
emulator's control; it was fixed to pairs atK2t andK8+2t before these passing runs.

After the Nov2 UTC reset, the CLI limits changed to10private checks and5scored
submissions per day. First real H200 two-group check `a8b06a9748b54177` is QUEUED;
one private check used, nine remain. All8user-authorized scored submissions are
spent, with none used this UTC day. The published2.382249325× source stays intact.

Additional striped2WG and1WG variants each passed44reference cases and352graph
replays under the synchronous GB10 emulator. Both compile forsm_90a:2WG uses168
registers with8/8-byte spills;1WG uses255 with64/56-byte spills. Reports are in
`striped_hopper_result.json` and `striped_emulation_result.json`. Neither has run
onH200. A paired cold-graph comparison against the exact published MAIN source
is prepared in `cuda_persistent_two_wg_paired.py`, not dispatched. Inspect the
first H200 result and prepare-time budget before choosing another check.
No scored submission is authorized beyond8. No automation was added.
