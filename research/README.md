# Type system research

This directory holds research notes about the Vortex type system. The notes are based on `develop`
at `5fc5cb8` (2026-10-17). They live on the `ct/ext-types-research` branch and are not meant to be
merged.

Reviews on 2026-10-22, 2026-10-23 and 2026-10-26 updated the Decimal notes and
[`decisions.md`](decisions.md). Part 4 of the Decimal notes holds the implementation plan for
`Narrow` and the canonical decimal fix, and the state of each PR.
The decimal, DecimalByteParts, serialization and versioning code that the notes cite is the same on
`develop` at `dbaf5f8` (2026-10-22).

| Topic | Summary for people | Detail for agents |
| --- | --- | --- |
| Extension types | [`extension-types/`](extension-types/README.md) | [`extension-types/agents/`](extension-types/agents/) |
| Decimal and the type roadmap | [`decimal/`](decimal/README.md) | [`decimal/agents/`](decimal/agents/) |

Each summary document links to the agent documents. The agent documents hold file and line
references, reproductions, fix sketches and file lists.

[`decisions.md`](decisions.md) records the decisions that the maintainer made while reviewing these
notes, and the questions that stay open. The other documents follow it.
