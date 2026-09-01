# O81 Case-6 exact MMA fragment oracle

This is a static proof artifact. It does not execute CUDA and does not alter the
candidate implementation.

## Fixed evidence

- Source winner: `0aa3258114e1283ba35ba98ece9264db961e414a`, tree
  `39d3fd7aaad6774b724750a51134fad1195dbe00`.
- Rejected O74 source/result commit: `7f5c86f`, tree
  `c3889ad46f560d2e637b748d9fec4cf031071be2`.
- Historical PTX: immutable profile job
  `job-1788197940210-334e5529e71454fb`.
- ISA authority: NVIDIA PTX ISA, “Matrix Fragments for
  `mma.m16n8k16` with floating point type”.

## Root cause

For lane `l`, let `g=l>>2`, `t=l&3`, and `k=2*t`. The PTX ISA requires the
four packed A registers, each listed low BF16 then high BF16, in this order:

1. `[(g,k), (g,k+1)]`
2. `[(g+8,k), (g+8,k+1)]`
3. `[(g,k+8), (g,k+9)]`
4. `[(g+8,k+8), (g+8,k+9)]`

O74 instead passed registers 2 and 3 in the opposite order. Hardware therefore
interpreted the upper-row/lower-K pair as lower-row/upper-K, and vice versa, in
every lane and every QK and PV MMA. Exactly half of every A tile was assigned
the wrong logical row/K coordinate. This is sufficient to explain the broad
Case-6 failure; it is not softmax drift.

The B loaders are already correct. For each lane their two registers map to
`[(2t,g),(2t+1,g)]` and `[(2t+8,g),(2t+9,g)]`, for both logical `K^T` in QK
and logical `V` in PV. The four FP32 accumulator/store coordinates are also
correct: `(g,2t)`, `(g,2t+1)`, `(g+8,2t)`, `(g+8,2t+1)`.

Swapping only `a[1]` and `a[2]` makes the direct Q/probability loaders match
the ISA in all 32 lanes. The executable oracle proves exact A/B/C coverage,
no overlap, and all four Case-6 prefix schedules.

## K order and boundaries

- QK always issues K16 starts `0,16`, feeding each destination tuple back as
  the next accumulator.
- PV issues K16 starts `0..padded_key_count-16`; K96 uses the historical
  padded-K128 chain with zero probability and value operands for K96..127.
- The historical PTX digests contain QK/PV MMA counts `4/4`, `8/8`, `16/16`,
  and `16/16` for key counts `32/64/96/128`. Every destination tuple equals
  its accumulator tuple, and repeated tuples appear in increasing K16 order.
- Numerical boundaries remain: FP32 QK MMA -> BF16 -> FP32 scale -> BF16;
  ATen-order FP32 softmax -> BF16 probability; continuous FP32 PV accumulation
  -> BF16 context.

Run the proof plus immutable PTX audit with:

```bash
python research/case6_fragment_oracle.py \
  --historical-cache-root \
  /home/w/Project/GitHub/Personal/TikTokTechJam2026Track3/.benchmarkctl/jobs/job-1788197940210-334e5529e71454fb/home/.triton/cache
```

## Scope and remaining risk

The oracle closes the logical fragment defect for the existing O74 Case-6
design. It does not by itself prove a Case-13 fused CTA throughput model. The
same single-MMA lane formulas apply to Case 13, but its M128 QK and M64 PV
ownership must be decomposed into M16 owners, its FP16 score/BF16 probability
in-place shared representation must be retained, and duplicated K/V loads and
idle PV warps still require a separate performance design.
