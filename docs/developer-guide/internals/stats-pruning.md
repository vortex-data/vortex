# Stats Pruning

Vortex uses statistics to prove when a filter cannot match a row group, zone, or
file. The proof expression returns `true` when the input can be skipped. It
returns `false` or `null` when pruning is not proven.

The pruning pipeline has two phases:

1. `Expression::falsify(scope, session)` asks the session's
   `StatsRewriteRule`s to rewrite a filter into an abstract proof expression.
   Rules describe semantics in terms of `vortex.stat(input, aggregate_fn)`
   placeholders, so the rewrite does not depend on a particular layout.
2. `bind_stats` lowers those abstract stat placeholders with a `StatBinder`.
   The binder maps stats to the representation used by the caller, such as
   zone-map table fields, file-level stat literals, or typed null literals for
   missing stats.

Missing stats lower to typed null literals. This preserves the three-valued
logic used by pruning: only a non-null `true` value proves that the scope can be
skipped. A missing stat therefore cannot accidentally prune data.

## Binding Targets

Zone maps bind stats to fields in their per-zone stats table. The lowered
expression is evaluated against that table and produces a mask where `true`
means the zone can be skipped.

File-level stats bind stats to literal values from the file footer. The lowered
expression is reduced and evaluated once for the full file. If it evaluates to
`true`, the file stats reader can return an all-false pruning mask without
reading child layouts.

Scan planning uses `checked_pruning_expr` to lower a falsified expression against
the available stats table schema. It returns the stats-table expression and the
set of stat fields still required after expression reduction. If all required
stats are missing, only a constant `true` proof is useful; all other results are
treated as no pruning expression.

For the layout model around these pruning points, see
[Layouts](../../concepts/layouts.md) and [Scanning](../../concepts/scanning.md).
