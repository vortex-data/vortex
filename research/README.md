# Type system research

This directory holds research notes about the Vortex type system. The notes are based on `develop`
at `5fc5cb8` (2026-10-17). They live on the `ct/ext-types-research` branch and are not meant to be
merged.

Reviews on 2026-10-22, 2026-10-23 and 2026-10-26 updated the Decimal notes and
[`decisions.md`](decisions.md). On 2026-10-29, the extension type notes were checked again against
`develop` at `1d09649bbd`, and the Decimal plan was updated for the PRs that merged or opened
since 2026-10-26. A design pass later on 2026-10-29 settled the extension type mechanism (claims,
function declarations and result mappings), and part 5 of the extension notes has a new phase plan.
On 2026-10-30, a second design pass settled extension nullability and the storage declaration
(X25 to X34 in [`decisions.md`](decisions.md)). Its new findings are for `develop` at
`493967ce33`. Part 4 of the Decimal notes holds the implementation plan for `Widen` and the
canonical decimal fix. The tracking issues [#10701](https://github.com/vortex-data/vortex/issues/10701) and
[#10733](https://github.com/vortex-data/vortex/issues/10733) hold the state of each PR. The
decimal, DecimalByteParts, serialization and versioning code that the notes cite is the same on
`develop` at `dbaf5f8` (2026-10-22).

| Topic | Summary for people | Detail for agents |
| --- | --- | --- |
| Extension types | [`extension-types/`](extension-types/README.md) | [`extension-types/agents/`](extension-types/agents/) |
| Decimal and the type roadmap | [`decimal/`](decimal/README.md) | [`decimal/agents/`](decimal/agents/) |

Each summary document links to the agent documents. The agent documents hold file and line
references, reproductions, fix sketches and file lists.

[`decisions.md`](decisions.md) records the decisions that the maintainer made while reviewing these
notes, and the questions that stay open. The other documents follow it.
