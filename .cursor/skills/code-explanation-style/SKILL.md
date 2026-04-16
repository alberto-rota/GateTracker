---
name: code-explanation-style
description: Explains code in structured markdown with mathematical notation and tensor-shape annotations. Use when the user asks how code works, wants a walkthrough, architecture explanation, or documentation-style commentary on implementations.
---

# Code explanation style (this project)

## When this skill applies

Use for any request to **explain**, **walk through**, **document**, or **clarify** code behavior—not for refactors or fixes unless explanation is explicitly requested.

## Output format (required)

1. **Markdown only** — use headings (`##`, `###`), bullet lists, and fenced code citations when pointing at the repo (Cursor code-reference format when available).

2. **Formulas where they add precision** — use standard LaTeX-in-markdown conventions the renderer supports, for example:
   - Inline: \(f(x) = \ldots\)
   - Block: \[ \mathbf{y} = W\mathbf{x} + \mathbf{b} \]
   Use formulas for losses, geometry (projection, epipolar), attention, or any numeric relationship; skip decorative math.

3. **Tensor shapes** — when discussing PyTorch tensors, annotate shapes explicitly, e.g. “`x`: `[B, C, H, W]`”. Prefer a small convention table for the snippet if several symbols appear.

4. **Audience** — assume strong Python, PyTorch, geometry, and CV background; do not restate textbook definitions unless the logic is non-obvious or project-specific.

5. **Structure** — default outline:
   - **Purpose** — what the module/function is for (one short paragraph).
   - **Inputs / outputs** — dtypes and shapes where relevant.
   - **Control / data flow** — ordered steps or a brief diagram (mermaid only if the flow is non-linear).
   - **Edge cases & invariants** — masks, empty batches, numerical stability, device/dtype assumptions.
   - **Connections** — which callers or downstream modules depend on this behavior (if clear from context).

## Customization (edit below)

_Add your personal constraints here so the agent follows them every time._

- Preferred depth: _e.g. “always show one minimal numerical example” / “never more than 8 bullets”_
- Forbidden: _e.g. “no analogies to unrelated domains”_
- Extra: _e.g. “always relate back to the training objective”_

## Anti-patterns

- Do not replace explanation with “here is different code” unless the user asked for a change.
- Do not dump the entire file; cite the minimal regions that carry the argument.
