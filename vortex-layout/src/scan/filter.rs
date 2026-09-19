// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::env;
use std::iter;
use std::sync::LazyLock;

use bit_vec::BitVec;
use itertools::Itertools;
use parking_lot::RwLock;
use sketches_ddsketch::DDSketch;
use vortex_array::dtype::FieldPath;
use vortex_array::expr::BoundExpression;
use vortex_array::expr::analysis::referenced_field_paths;
use vortex_array::expr::bound::and_collect;
use vortex_array::scalar_fn::fns::binary::Binary;
use vortex_array::scalar_fn::fns::dynamic::DynamicExprUpdates;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;

/// The selectivity histogram quantile to use for reordering conjuncts. Where 0 == no rows match.
const DEFAULT_SELECTIVITY_QUANTILE: f64 = 0.1;

/// The grouping applied by [`FilterExpr::new`], read once from `VORTEX_CONJUNCT_GROUPING`.
static ENV_CONJUNCT_GROUPING: LazyLock<ConjunctGrouping> = LazyLock::new(|| {
    let Ok(value) = env::var("VORTEX_CONJUNCT_GROUPING") else {
        return ConjunctGrouping::default();
    };
    match value.as_str() {
        "none" => ConjunctGrouping::None,
        "same" => ConjunctGrouping::SameFields,
        "shared" => ConjunctGrouping::SharedFields,
        other => {
            tracing::warn!(
                "Ignoring unknown VORTEX_CONJUNCT_GROUPING={other}, expected one of \
                 none, same, shared"
            );
            ConjunctGrouping::default()
        }
    }
});

/// How the conjuncts of a filter are regrouped after splitting on `AND`.
///
/// Splitting lets the scan reorder and short-circuit each predicate independently, but two
/// predicates over the same column each decode that column. Regrouping trades scheduling
/// granularity for a single pass over the columns a group shares.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum ConjunctGrouping {
    /// Evaluate every conjunct separately.
    None,
    /// Group conjuncts referencing exactly the same field paths, e.g. `a > 5 AND a < 10`.
    ///
    /// Note that [`Expression::optimize_recursive`] already folds a same-column literal range into
    /// a single `Between`, so this only catches the pairs that rewrite misses.
    ///
    /// [`Expression::optimize_recursive`]: vortex_array::expr::Expression::optimize_recursive
    SameFields,
    /// Group conjuncts sharing any field path, e.g. `a > 5 AND a < b`. The relation is applied
    /// transitively, so `b = 2` joins that same group.
    #[default]
    SharedFields,
}

impl ConjunctGrouping {
    /// Whether conjuncts referencing `lhs` and `rhs` belong to the same group.
    fn relates(self, lhs: &[FieldPath], rhs: &[FieldPath]) -> bool {
        match self {
            // Field paths are prefix-minimal, so set equality is equality of the covering sets.
            Self::SameFields => lhs.len() == rhs.len() && lhs.iter().all(|path| rhs.contains(path)),
            Self::SharedFields => lhs
                .iter()
                .any(|path| rhs.iter().any(|other| path.overlap(other))),
            Self::None => false,
        }
    }
}

/// A [`FilterExpr`] splits boolean expressions into individual conjunctions, tracks
/// statistics about selectivity, and uses this information to reorder the evaluation of the
/// conjunctions in an attempt to minimize the work done.
pub struct FilterExpr {
    /// The conjuncts involved in the filter expression.
    conjuncts: Vec<BoundExpression>,
    /// A histogram for the selectivity of each conjunct.
    conjunct_selectivity: Vec<RwLock<DDSketch>>,
    /// Dynamic expression trackers for each conjunct, incase they contain dynamic expressions.
    dynamic_conjuncts: Vec<Option<DynamicExprUpdates>>,
    /// The preferred ordering of conjuncts.
    ordering: RwLock<Vec<usize>>,
    /// The quantile to use from the selectivity histogram of each conjunct.
    selectivity_quantile: f64,
}

fn bound_conjuncts(expr: &BoundExpression) -> Vec<BoundExpression> {
    let mut conjuncts = Vec::new();
    let mut pending = vec![expr];

    while let Some(expr) = pending.pop() {
        if expr
            .as_scalar()
            .and_then(|scalar_fn| scalar_fn.as_opt::<Binary>())
            .is_some_and(|operator| *operator == Operator::And)
        {
            pending.extend(expr.children().iter().rev());
        } else {
            conjuncts.push(expr.clone());
        }
    }

    conjuncts
}

/// Merges related conjuncts back into single `AND` expressions, preserving input order.
///
/// Relatedness is a graph, not a partition: under [`ConjunctGrouping::SharedFields`] the conjuncts
/// of `a > 5 AND a < b AND b = 2` are all connected, so they form one group. Groups are emitted in
/// the order of their first conjunct.
fn group_conjuncts(
    conjuncts: Vec<BoundExpression>,
    grouping: ConjunctGrouping,
) -> VortexResult<Vec<BoundExpression>> {
    if grouping == ConjunctGrouping::None || conjuncts.len() < 2 {
        return Ok(conjuncts);
    }

    let referenced = conjuncts
        .iter()
        .map(|conjunct| Ok(referenced_field_paths(conjunct)?.into_iter().collect_vec()))
        .collect::<VortexResult<Vec<_>>>()?;

    let mut sets = DisjointSets::new(conjuncts.len());
    for (idx, paths) in referenced.iter().enumerate() {
        for (other, other_paths) in referenced[..idx].iter().enumerate() {
            if grouping.relates(paths, other_paths) {
                sets.union(idx, other);
            }
        }
    }

    // Every conjunct's root is at most its own index, so filling groups in index order leaves both
    // the groups and their members in input order.
    let mut groups = vec![Vec::new(); conjuncts.len()];
    for (idx, conjunct) in conjuncts.into_iter().enumerate() {
        groups[sets.find(idx)].push(conjunct);
    }

    Ok(groups.into_iter().filter_map(and_collect).collect())
}

/// Union-find over conjunct indices, where each set is rooted at its lowest member index.
struct DisjointSets(Vec<usize>);

impl DisjointSets {
    fn new(len: usize) -> Self {
        Self((0..len).collect())
    }

    fn find(&mut self, mut idx: usize) -> usize {
        while self.0[idx] != idx {
            self.0[idx] = self.0[self.0[idx]];
            idx = self.0[idx];
        }
        idx
    }

    fn union(&mut self, lhs: usize, rhs: usize) {
        let (lhs, rhs) = (self.find(lhs), self.find(rhs));
        if lhs < rhs {
            self.0[rhs] = lhs;
        } else {
            self.0[lhs] = rhs;
        }
    }
}

impl FilterExpr {
    /// Build a filter expression using the grouping named by `VORTEX_CONJUNCT_GROUPING`.
    pub fn new(expr: BoundExpression) -> VortexResult<Self> {
        Self::new_with_grouping(expr, *ENV_CONJUNCT_GROUPING)
    }

    pub fn new_with_grouping(
        expr: BoundExpression,
        grouping: ConjunctGrouping,
    ) -> VortexResult<Self> {
        let conjuncts = group_conjuncts(bound_conjuncts(&expr), grouping)?;
        let num_conjuncts = conjuncts.len();

        let dynamic_conjuncts = conjuncts.iter().map(DynamicExprUpdates::new).collect_vec();

        Ok(Self {
            conjuncts,
            conjunct_selectivity: iter::repeat_with(|| RwLock::new(DDSketch::default()))
                .take(num_conjuncts)
                .collect(),
            dynamic_conjuncts,
            // The initial ordering is naive, we could order this by how well we expect each
            // comparison operator to perform. e.g. == might be more selective than <=? Not obvious.
            ordering: RwLock::new((0..num_conjuncts).collect()),
            selectivity_quantile: DEFAULT_SELECTIVITY_QUANTILE,
        })
    }

    /// The conjuncts that make up this filter expression.
    #[inline]
    pub fn conjuncts(&self) -> &[BoundExpression] {
        &self.conjuncts
    }

    /// The dynamic updates for the given conjunct, if any.
    #[inline]
    pub fn dynamic_updates(&self, conjunct_idx: usize) -> Option<&DynamicExprUpdates> {
        self.dynamic_conjuncts[conjunct_idx].as_ref()
    }

    /// Returns the next preferred conjunct to evaluate.
    #[inline]
    pub fn next_conjunct(&self, remaining: &BitVec) -> Option<usize> {
        let read = self.ordering.read();
        // Take the first remaining conjunct in the ordered list.
        read.iter().find(|&idx| remaining[*idx]).copied()
    }

    /// Report the selectivity of a conjunct, i.e. 0 means no rows matched the predicate.
    pub fn report_selectivity(&self, conjunct_idx: usize, selectivity: f64) {
        if !(0.0..=1.0).contains(&selectivity) {
            vortex_panic!(
                "selectivity {} must be in the range [0.0, 1.0]",
                selectivity
            );
        }

        {
            let mut histogram = self.conjunct_selectivity[conjunct_idx].write();

            histogram.add(selectivity);
        }

        // Note: We read from multiple RwLocks here without coordination. This means we might
        // see an inconsistent snapshot where some histograms have been updated more recently
        // than others. This is acceptable because:
        // 1. The ordering is a heuristic optimization, not a correctness requirement
        // 2. The selectivity values are statistical estimates that change gradually
        // 3. Any ordering will produce correct results, just with different performance
        let Some(all_selectivity) = self
            .conjunct_selectivity
            .iter()
            .map(|histogram| {
                histogram
                    .read()
                    .quantile(self.selectivity_quantile)
                    .map_err(|e| vortex_err!("{e}")) // Only errors when the quantile is out of range
                    .vortex_expect("quantile out of range")
            })
            .collect::<Option<Vec<_>>>()
        else {
            // Preserve the input order until every conjunct has been observed. Treating an unseen
            // conjunct as perfectly selective can move expensive predicates ahead of selective
            // predicates based only on which concurrent split finishes first.
            return;
        };

        {
            let ordering = self.ordering.read();
            if ordering.is_sorted_by_key(|&idx| all_selectivity[idx]) {
                return;
            }
        }

        // Re-sort our conjuncts based on the new statistics.
        let mut ordering = self.ordering.write();
        ordering.sort_unstable_by(|&l_idx, &r_idx| {
            all_selectivity[l_idx]
                .partial_cmp(&all_selectivity[r_idx])
                .vortex_expect("Can't compare selectivity values")
        });

        tracing::trace!(
            "Reordered conjuncts based on new selectivity {:?}",
            ordering
                .iter()
                .map(|&idx| format!("({}) => {}", self.conjuncts[idx], all_selectivity[idx]))
                .join(", ")
        );
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::dtype::StructFields;
    use vortex_array::expr::Expression;
    use vortex_array::expr::and;
    use vortex_array::expr::and_collect;
    use vortex_array::expr::col;
    use vortex_array::expr::gt;
    use vortex_array::expr::lit;
    use vortex_array::expr::lt;
    use vortex_array::expr::not;
    use vortex_array::expr::root;
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;

    use super::ConjunctGrouping;
    use super::FilterExpr;

    fn struct_dtype() -> DType {
        DType::Struct(
            StructFields::from_iter([
                ("a", DType::Primitive(PType::I32, Nullability::NonNullable)),
                ("b", DType::Primitive(PType::I32, Nullability::NonNullable)),
                ("c", DType::Primitive(PType::I32, Nullability::NonNullable)),
            ]),
            Nullability::NonNullable,
        )
    }

    /// Asserts that `grouping` splits `expr` into exactly `expected`, each an `AND` of the
    /// conjuncts it groups.
    fn assert_grouped(
        expr: Expression,
        grouping: ConjunctGrouping,
        expected: impl IntoIterator<Item = Vec<Expression>>,
    ) -> VortexResult<()> {
        let dtype = struct_dtype();
        let filter = FilterExpr::new_with_grouping(expr.bind(&dtype)?, grouping)?;
        let expected = expected
            .into_iter()
            .map(|group| {
                and_collect(group)
                    .vortex_expect("expected groups are non-empty")
                    .bind(&dtype)
            })
            .collect::<VortexResult<Vec<_>>>()?;

        assert_eq!(filter.conjuncts(), expected.as_slice());
        Ok(())
    }

    #[test]
    fn bound_conjuncts_preserve_order_and_types() -> VortexResult<()> {
        let expr = and(root(), and(not(root()), lit(true)));
        let dtype = DType::Bool(Nullability::Nullable);
        let bound = expr.bind(&dtype)?;
        let filter = FilterExpr::new_with_grouping(bound, ConjunctGrouping::None)?;
        let conjuncts = filter.conjuncts();

        let expected = vec![
            root().bind(&dtype)?,
            not(root()).bind(&dtype)?,
            lit(true).bind(&dtype)?,
        ];
        assert_eq!(conjuncts, expected.as_slice());
        assert_eq!(
            conjuncts
                .iter()
                .map(|expr| expr.dtype().clone())
                .collect::<Vec<_>>(),
            vec![
                DType::Bool(Nullability::Nullable),
                DType::Bool(Nullability::Nullable),
                DType::Bool(Nullability::NonNullable),
            ]
        );
        Ok(())
    }

    #[test]
    fn waits_for_all_conjuncts_before_reordering() -> VortexResult<()> {
        let dtype = DType::Bool(Nullability::Nullable);
        let bound = and(root(), not(root())).bind(&dtype)?;
        let filter = FilterExpr::new_with_grouping(bound, ConjunctGrouping::None)?;

        filter.report_selectivity(0, 0.9);
        assert_eq!(*filter.ordering.read(), vec![0, 1]);

        filter.report_selectivity(1, 0.1);
        assert_eq!(*filter.ordering.read(), vec![1, 0]);
        Ok(())
    }

    #[rstest]
    #[case::same_fields(ConjunctGrouping::SameFields)]
    #[case::shared_fields(ConjunctGrouping::SharedFields)]
    fn groups_a_range_over_one_field(#[case] grouping: ConjunctGrouping) -> VortexResult<()> {
        let lower = gt(col("a"), lit(5_i32));
        let upper = lt(col("a"), lit(10_i32));

        assert_grouped(
            and(lower.clone(), upper.clone()),
            grouping,
            [vec![lower, upper]],
        )
    }

    #[test]
    fn same_fields_keeps_a_partial_overlap_apart() -> VortexResult<()> {
        let lower = gt(col("a"), lit(5_i32));
        let cross = lt(col("a"), col("b"));

        assert_grouped(
            and(lower.clone(), cross.clone()),
            ConjunctGrouping::SameFields,
            [vec![lower], vec![cross]],
        )
    }

    #[test]
    fn shared_fields_groups_transitively() -> VortexResult<()> {
        // `c > 1` shares nothing with the rest, while `a < b` bridges `a > 5` and `b < 10`.
        let lower = gt(col("a"), lit(5_i32));
        let disjoint = gt(col("c"), lit(1_i32));
        let cross = lt(col("a"), col("b"));
        let upper = lt(col("b"), lit(10_i32));

        assert_grouped(
            and(
                and(lower.clone(), disjoint.clone()),
                and(cross.clone(), upper.clone()),
            ),
            ConjunctGrouping::SharedFields,
            [vec![lower, cross, upper], vec![disjoint]],
        )
    }

    #[rstest]
    #[case::none(ConjunctGrouping::None)]
    #[case::same_fields(ConjunctGrouping::SameFields)]
    #[case::shared_fields(ConjunctGrouping::SharedFields)]
    fn disjoint_fields_are_never_grouped(#[case] grouping: ConjunctGrouping) -> VortexResult<()> {
        let lhs = gt(col("a"), lit(5_i32));
        let rhs = lt(col("b"), lit(10_i32));

        assert_grouped(
            and(lhs.clone(), rhs.clone()),
            grouping,
            [vec![lhs], vec![rhs]],
        )
    }
}
