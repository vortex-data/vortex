# vortex-scan-plan

Prototype of the lazy scan planner protocol: footer open, file-statistics pruning, and
filter-and-project over natural splits, driven by a single-worker blocking driver. The design,
scope, and stage state tables are in
[`vortex-scan/design/PROTOTYPE.md`](../vortex-scan/design/PROTOTYPE.md); the build order is in
[`vortex-scan/design/PROTOTYPE-PLAN.md`](../vortex-scan/design/PROTOTYPE-PLAN.md). The crate is
reviewed, built, and deleted as a unit; nothing public outside it changes.
