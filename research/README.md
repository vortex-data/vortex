# Decimal research

This directory holds research notes about Decimal and the Vortex type roadmap. The notes are based
on `develop` at `5fc5cb8` (2026-10-17). They live on the `ct/decimal-research` branch and are not
meant to be merged.

Reviews on 2026-10-22, 2026-10-23 and 2026-10-26 updated the Decimal notes and
[`decisions.md`](decisions.md). On 2026-10-29, the Decimal plan was updated for the PRs that merged
or opened since 2026-10-26. Part 4 of the Decimal notes holds the implementation plan for `Widen`
and the canonical decimal fix. The tracking issues
[#10701](https://github.com/vortex-data/vortex/issues/10701) and
[#10733](https://github.com/vortex-data/vortex/issues/10733) hold the state of each PR. The
decimal, DecimalByteParts, serialization and versioning code that the notes cite is the same on
`develop` at `dbaf5f8` (2026-10-22).

The extension type notes stay on
[`ct/ext-types-research`](https://github.com/vortex-data/vortex/tree/ct/ext-types-research/research/extension-types).

| Topic | Summary for people | Detail for agents |
| --- | --- | --- |
| Decimal and the type roadmap | [`decimal/`](decimal/README.md) | [`decimal/agents/`](decimal/agents/) |

Each summary document links to the agent documents. The agent documents hold file and line
references, reproductions, fix sketches and file lists.

[`decisions.md`](decisions.md) records the decisions that the maintainer made while reviewing these
notes, and the questions that stay open. The other documents follow it.
