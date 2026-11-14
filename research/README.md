# Extension type research

This directory holds research notes about extension types and casts in Vortex. The notes are based
on `develop` at `5fc5cb8` (2026-10-17). They live on the `ct/ext-types-research` branch and are not
meant to be merged.

On 2026-10-29, the extension type notes were checked again against `develop` at `1d09649bbd`.
A design pass later on 2026-10-29 settled the extension type mechanism (claims, function
declarations and result mappings), and part 5 of the extension notes has a new phase plan.
On 2026-10-30, a second design pass settled extension nullability and the storage declaration
(X25 to X34 in [`decisions.md`](decisions.md)). Its new findings are for `develop` at
`493967ce33`.

The Decimal notes, the type roadmap, and the `Widen` plan moved to
[`ct/decimal-research`](https://github.com/vortex-data/vortex/tree/ct/decimal-research/research/decimal).

| Topic | Summary for people | Detail for agents |
| --- | --- | --- |
| Extension types | [`extension-types/`](extension-types/README.md) | [`extension-types/agents/`](extension-types/agents/) |

Each summary document links to the agent documents. The agent documents hold file and line
references, reproductions, fix sketches and file lists.

[`decisions.md`](decisions.md) records the decisions that the maintainer made while reviewing these
notes, and the questions that stay open. The other documents follow it.
