# Ninety-Second Demo Script

Record the narration first, then match a silent screen recording to it. Exact
numbers stay visible in the tables; spoken numbers are written as English words.

## 0:00–0:18 — Result

**Visual:** Project title and the results table in `README.md`.

**Narration:**

> Hi, I am Jinye Wu. For TikTok TechJam Track Three, I used AI coding agents to
> optimize the supplied PyTorch Transformer benchmark on an NVIDIA R T X forty
> seventy. My final candidate is three point nine times faster by geometric mean
> across Cases One through Thirteen, and reduces aggregate latency from five
> hundred fifty-five point four three to one hundred three point four eight
> milliseconds. All accuracy trials passed bitwise exactly.

## 0:18–0:43 — AI Optimization Loop

**Visual:** Show the workflow diagram, followed by `benchmarkctl list` and the
final completed job.

**Narration:**

> OpenAI Codex agents investigated different bottlenecks in isolated Git
> worktrees. Every GPU experiment went through our Rust benchmark control tool.
> It captured an immutable source snapshot, serialized access to the shared GPU,
> and returned structured correctness and timing results. Project-specific agent
> skills required correctness to pass before any optimization could be promoted.

## 0:43–1:02 — Kernel Optimization Methods

**Visual:** Show the optimization map and briefly scroll through two custom
kernel files.

**Narration:**

> The final source is one unified implementation, not a collage of best numbers.
> We use CUDA Graph replay, exact feed-forward and projection fusion, residual
> epilogues, packed query, key, and value layouts, direct-write attention context
> kernels, causal work elimination, native-order B F sixteen softmax, and
> shape-specialized attention and LayerNorm. Every route checks its exact shape,
> data type, mask, layout, and hardware contract, with a compatible fallback
> outside that scope.

## 1:02–1:22 — Evidence and MFU

**Visual:** Show zero failed elements, the latency summary, and the MFU table.

**Narration:**

> Across more than nine hundred thirty-eight million evaluated outputs, there
> were zero failed elements. Aggregate latency fell from about five hundred
> fifty-five milliseconds to about one hundred three milliseconds. As
> supplemental analysis, dense-equivalent M F U increased from about six
> percent to more than thirty-three percent.

## 1:22–1:35 — Case Fourteen and Close

**Visual:** Show the structured Case Fourteen OOM result, then the repository.

**Narration:**

> Case Fourteen was attempted without changing its shape, but its input and
> required output already exceed the twelve-gigabyte GPU capacity. We report
> that limitation rather than hiding the case. This project shows how AI code
> generation becomes credible GPU engineering when paired with immutable
> evaluation and exact correctness. Thank you.

## Recording Checklist

- [ ] Record the five narration sections separately.
- [ ] Keep the finished video below three minutes.
- [ ] Hide credentials, private paths, notifications, and unrelated windows.
- [x] Upload to YouTube with public visibility.
- [x] Add the YouTube URL to the Devpost draft and `README.md`.
- [ ] Change the GitHub repository to public before submission.
