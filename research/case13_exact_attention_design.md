# O84 Case-13 exact shared-CTA design

This is the pre-submit proof and investment packet for the exact CUDA BF16
Case-13 path.  The implementation is based on the unified exact-Case-6 winner
commit `226bc2e`, tree `d2c8d609da2141053a506404aa066be5f0b40d8b`.

## Measured gate

Profile job `job-1788211390844-806632a9c3872e48` measured a `16.057184219 ms`
candidate call.  QK (`3.212958 ms`), native BF16-output softmax
(`5.706468 ms`), and PV (`4.326032 ms`) consume `13.245458 ms`, or `81.829%`.
The later unified winner measured Case 13 at `16.013311 ms`, so the final 2%
experiment gate is a `0.320266 ms` saving against that control.

The four prefixes contain `671,088,640` score elements over four layers.  The
current exact boundaries write/read FP16 scores and write/read BF16
probabilities, moving `5,368,709,120 B = 5 GiB` per forward.  The fused path
keeps those two tensors in shared memory.  It also replaces, per prefix, the
current QK + native-softmax + PV launch grids (roughly 18,432 through 21,504
CTAs) with 4,096 fused CTAs.  Even a 6.5% removal from the profiled attention
triplet clears 2%; removing the global score/probability boundaries is a
credible larger structural effect.

## Geometry and exact arithmetic

- One 256-thread CTA owns one `(batch, head, 16-query-row)` tile.  Grid order
  keeps batch/head as the fast dimension and has 16 row blocks for each of the
  four fixed prefixes 256/512/768/1024.
- Eight QK warps partition all `M16xN8` score fragments.  The prefix variants
  hold 4/8/12/16 fragments per warp.  Every fragment uses exactly two dependent
  K16 MMAs for head dimension 32.
- The corrected official row-major A order is
  `[row0 K0-1, row8 K0-1, row0 K8-9, row8 K8-9]`.  Official column-major B is
  `[(2t,g),(2t+1,g)]`, then `[(2t+8,g),(2t+9,g)]`; C/D is
  `(g,2t),(g,2t+1),(g+8,2t),(g+8,2t+1)`.  The executable oracle checks every
  lane, full coverage, and no duplicate fragment owner.  O74's only fragment
  defect was exchanging A registers 1 and 2.
- QK preserves FP32 MMA -> RNE BF16 dot -> FP32 scale -> RNE BF16 score.  It
  then applies the established RNE BF16-to-FP16 score conversion (including
  ordinary overflow and infinity behavior) before the native-softmax load.
- Eight warps process the 16 rows in two waves.  Each warp uses ATen's
  `lane + iteration*32` element ownership, serial local max/sum order,
  descending XOR max/sum tree, `std::exp`, division/zero-sum behavior, then RNE
  BF16 probability output.  Prefix 768 exactly retains ATen's 1024-element
  next-power-of-two schedule: 24 valid plus 8 `-inf` iterations per lane.
- Four warps own the four M16xN8 PV fragments.  Each destination tuple is fed
  back through monotonically increasing K16 starts 0..prefix-16, with no split
  partial sums.  BF16 probabilities and values feed FP32 accumulators; context
  crosses one final RNE BF16 boundary.
- Two CTA barriers separate QK stores, in-place softmax overwrite, and PV loads.
  Every score/probability coordinate has one QK writer, one softmax writer, and
  read-only PV consumers between barriers.

## Resource and reload model

The shared allocation is 9/17/25/33 KiB: an 8/16/24/32-KiB 16-row
score/probability buffer plus one 1-KiB M16xK32 query tile.  Staging Q once per
CTA prevents the N8 output decomposition from repeatedly issuing global loads
for the same A tile.  K and V need no intra-CTA staging because each loaded
fragment contributes to one output tile.  Each QK output fragment finishes its
two dependent K16 MMAs and stores before the warp starts another independent
fragment.  This changes no output reduction but avoids keeping all fragments
live simultaneously.

The actual sm89 build uses 40/40/64/64 registers for prefixes
256/512/768/1024, with zero stack, local memory, or spill.  SM89's 65,536
registers and 100 KiB shared memory therefore still permit 6/5/4/3 CTAs, or
48/40/32/24 resident warps (100%/83%/67%/50% of its 48-warp limit).  PTX has
24/48/72/96 MMAs: QK contributes 8/16/24/32 and PV contributes
16/32/48/64.  All 240 instructions self-chain identical destination and
accumulator tuples.  SASS has three barriers per specialization (Q staging,
QK-to-softmax, and softmax-to-PV), ten descending butterfly shuffles per
softmax wave, the expected 176 exponential operations (including the 768
prefix's ATen padding), and no local
loads/stores.

Relative to historical M128 QK and M64 PV ownership, M16 CTAs issue eight times
as many K loads and four times as many V loads.  Across four layers those are
about `1.17 GB` and `2.01 GB` of additional load requests.  Query staging
instead reduces Q requests from about `0.34 GB` to `0.07 GB`.  Each prefix's
complete K or V working set over all 256 batch/head pairs is at most 16 MiB,
below the RTX 4070's 36 MiB L2, and the grid makes batch/head the fast dimension
so successive row-block waves reuse that resident prefix.  In contrast,
score/probability prefixes are hundreds of MiB and cannot remain in L2.  This
is why the 5-GiB eliminated global store/read boundary traffic is not canceled
one-for-one by the roughly 2.92-GB net increase in cacheable load requests.

Rollback conditions: any oracle failure, changed K16 self-chain, local-memory
spill, inability to reproduce native softmax instructions, strict correctness
failure, or focused Case-13 gain below 2% closes this route.  Cases 6 and 12 are
unchanged negative controls.

The A/B/C fragment mapping is independently encoded here, independently
reviewed by O82 commit `f24f385`, and inherited from immutable O81 oracle
`bf87500f` (tree `177d0c7`).  O81's minimal corrected implementation
`74e3d28` then passed Case-6 focused correctness bitwise exactly, providing an
end-to-end check of the same single-MMA lane mapping without substituting for
this design's separate Case-13 M16 ownership proof.
