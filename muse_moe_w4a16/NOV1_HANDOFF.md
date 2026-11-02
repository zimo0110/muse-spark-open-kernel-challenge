# MoE progress — November 1, 2026

Published candidate: `submission.py`, byte-identical to `cuda_persistent_aligned_safe.py`.
Scored job **c913470f59834b10**: **2.382249325×**, CI **2.346515585–2.417983065**.
Account **zimo0110**, live rank **185** at the last read; leader **dolanzhao 2.7642×**.
The previous published score was 1.833725418×. Rank one has not been achieved.

The main architecture is the credited Apache-2.0 vjeux/alygg77 persistent CUDA kernel,
not an original architectural invention here. Local work reproduced its score range,
removed unreachable template builds, aligned BF16 tensor boundaries to the pinned
SGLang implementation, added the empty-job scheduler guard, and validated replay
behavior. Preserve the attribution, LICENSE and NOTICE in research/alygg77_2_4015/.
Readable CUDA review copies are in research/verified_local_cuda/.

Independent private checks: aligned-safe 2.370773507×; identical GPU algorithm with
24,000 queued replays 2.378511185×. All official tests and scored cases passed.
The queued soak used 12 shapes, 2,000 replays each, 10 fresh input sets and a host
sync every 20 mixed-shape cycles. Maximum self-comparison relative L2 drift was
6.57512e-5; 251 checkpoints differed bitwise, so do not claim bitwise determinism.
Official FP32-reference correctness was checked separately by the judge.

No confirmed gain from compact BF16 scratch, batched readiness releases, larger
narrow-N WGMMA groups, lower grid counts, two-block occupancy, or distributed M1
activation storage. Several runs passed using their baseline fallback. Do not
attribute their overall score to the rejected experimental paths.

A real resource issue was found: tiny variants reserved the unused fat path's
shared memory (214,784 bytes). Pruning it and bounding shared routing arrays made
NS2 use 95,360 bytes and allow two blocks per SM at 96 registers, but the paired
measurements still lost. Four-warps use 168 registers and 86,656 shared bytes;
its first check stopped at our 0.1% control-comparison bound (observed 0.1104%),
before official reference validation. Final check19 rejected the uncapped M1 distributed trial on timing (baseline ran;2.35798×).
Final check20 forced NW4 through all official tests and passed at2.28405×, slower
than the published implementation; its224mixedreplays also passed. Neither was promoted.

Exactly **8/8 user-authorized scored attempts** have been spent. No further scored
submission is authorized. Exactly **20/20 free private checks** were dispatched
on UTC November 1; next reset is **00:00 UTC November 2 = 4 p.m. PST November 1**.
Do not use another problem or account to evade the quota.

All BF16 challenge work and root Claude summary remain untouched. The old MoE
source is preserved as register_ready.py. The original two-session automation
remains paused; no new wake-up was created. Spark access now works securely via
tools/spark_ssh.py and the existing locally loaded SSH_NAME/SSH_PASS variables.
Never print the dotenv file or credential values. CUDA13.0/PyTorch2.10 run in
NVIDIA's25.11container through existing DockerCDI. See research/spark/README.md.

The new cuda_persistent_two_wg.py is ready for a FUTURE H200 private check:
Hoppercross-compilation passes;44functional cases+352graphreplays pass through
synchronous GB10 warp-MMA emulation. Actual Hopperasync behavior and speed are
UNVERIFIED; do not replace or publish over the verified2.382249xsource. AllSpark
containers have completed; no newokcjobs or scored attempts were dispatched.
