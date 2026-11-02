// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::iter;

use bit_vec::BitVec;
use itertools::Itertools;
use parking_lot::RwLock;
use sketches_ddsketch::DDSketch;
use vortex_array::dtype::Field;
use vortex_array::expr::BoundExpression;
use vortex_array::expr::bound::and_collect;
use vortex_array::scalar_fn::fns::binary::Binary;
use vortex_array::scalar_fn::fns::dynamic::DynamicExprUpdates;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_error::VortexExpect;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_expr::analysis::referenced_field_paths;
use vortex_utils::aliases::hash_set::HashSet;

/// The selectivity histogram quantile to use for reordering conjuncts. Where 0 == no rows match.
const DEFAULT_SELECTIVITY_QUANTILE: f64 = 0.1;

/// A [`FilterExpr`] splits boolean expressions into individual conjunctions, groups the
/// conjunctions that read a common top-level field into one unit, tracks statistics about the
/// selectivity of each unit, and uses this information to reorder their evaluation in an attempt
/// to minimize the work done.
///
/// A unit is a connected component of the graph linking conjuncts that share a field, so the
/// conjuncts over one field are evaluated together on each batch and units share no fields.
pub struct FilterExpr {
    /// The units of the filter expression: each is the conjunction of its members.
    conjuncts: Vec<BoundExpression>,
    /// The original conjuncts that make up each unit, in input order.
    members: Vec<Vec<BoundExpression>>,
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

/// The top-level fields a conjunct reads, or `None` when it reads the whole scope.
fn top_level_fields(expr: &BoundExpression) -> Option<HashSet<Field>> {
    let paths = referenced_field_paths(expr).ok()?;
    paths
        .iter()
        .map(|path| path.parts().first().cloned())
        .collect()
}

/// Group conjuncts into the connected components of the graph linking conjuncts that read a
/// common top-level field. A conjunct reading the whole scope links with every conjunct that
/// reads any field. Members and groups keep their input order.
fn group_conjuncts(conjuncts: Vec<BoundExpression>) -> Vec<Vec<BoundExpression>> {
    struct Group {
        members: Vec<usize>,
        fields: HashSet<Field>,
        reads_all: bool,
    }

    impl Group {
        fn reads_any(&self) -> bool {
            self.reads_all || !self.fields.is_empty()
        }

        fn links(&self, other: &Group) -> bool {
            (self.reads_all && other.reads_any())
                || (other.reads_all && self.reads_any())
                || !self.fields.is_disjoint(&other.fields)
        }
    }

    let mut groups: Vec<Group> = Vec::new();
    for (idx, conjunct) in conjuncts.iter().enumerate() {
        let fields = top_level_fields(conjunct);
        let mut group = Group {
            members: vec![idx],
            reads_all: fields.is_none(),
            fields: fields.unwrap_or_default(),
        };
        let (linked, mut rest): (Vec<_>, Vec<_>) =
            groups.into_iter().partition(|g| g.links(&group));
        for g in linked {
            group.members.extend(g.members);
            group.fields.extend(g.fields);
            group.reads_all |= g.reads_all;
        }
        group.members.sort_unstable();
        rest.push(group);
        rest.sort_unstable_by_key(|g| g.members[0]);
        groups = rest;
    }

    let mut conjuncts = conjuncts.into_iter().map(Some).collect_vec();
    groups
        .into_iter()
        .map(|g| {
            g.members
                .into_iter()
                .map(|idx| conjuncts[idx].take().vortex_expect("conjunct grouped once"))
                .collect()
        })
        .collect()
}

impl FilterExpr {
    pub fn new(expr: BoundExpression) -> Self {
        let members = group_conjuncts(bound_conjuncts(&expr));
        let conjuncts = members
            .iter()
            .map(|members| and_collect(members.iter().cloned()).vortex_expect("non-empty group"))
            .collect_vec();
        let num_conjuncts = conjuncts.len();

        let dynamic_conjuncts = conjuncts.iter().map(DynamicExprUpdates::new).collect_vec();

        Self {
            conjuncts,
            members,
            conjunct_selectivity: iter::repeat_with(|| RwLock::new(DDSketch::default()))
                .take(num_conjuncts)
                .collect(),
            dynamic_conjuncts,
            // The initial ordering is naive, we could order this by how well we expect each
            // comparison operator to perform. e.g. == might be more selective than <=? Not obvious.
            ordering: RwLock::new((0..num_conjuncts).collect()),
            selectivity_quantile: DEFAULT_SELECTIVITY_QUANTILE,
        }
    }

    /// The units that make up this filter expression, each the conjunction of its members.
    #[inline]
    pub fn conjuncts(&self) -> &[BoundExpression] {
        &self.conjuncts
    }

    /// The original conjuncts grouped into the unit at `conjunct_idx`.
    #[inline]
    pub fn members(&self, conjunct_idx: usize) -> &[BoundExpression] {
        &self.members[conjunct_idx]
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
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::dtype::StructFields;
    use vortex_array::expr::Expression;
    use vortex_array::expr::and;
    use vortex_array::expr::and_collect;
    use vortex_array::expr::col;
    use vortex_array::expr::eq;
    use vortex_array::expr::gt;
    use vortex_array::expr::lit;
    use vortex_array::expr::lt;
    use vortex_array::expr::not;
    use vortex_array::expr::root;
    use vortex_error::VortexResult;

    use super::FilterExpr;
    use super::bound_conjuncts;

    fn struct_dtype() -> DType {
        DType::Struct(
            StructFields::from_iter(
                ["a", "b", "c", "d"]
                    .map(|name| (name, DType::Primitive(PType::I32, Nullability::NonNullable))),
            ),
            Nullability::NonNullable,
        )
    }

    #[test]
    fn bound_conjuncts_preserve_order_and_types() -> VortexResult<()> {
        let expr = and(root(), and(not(root()), lit(true)));
        let dtype = DType::Bool(Nullability::Nullable);
        let conjuncts = bound_conjuncts(&expr.bind(&dtype)?);

        let expected = vec![
            root().bind(&dtype)?,
            not(root()).bind(&dtype)?,
            lit(true).bind(&dtype)?,
        ];
        assert_eq!(conjuncts, expected);
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
    fn groups_conjuncts_connected_by_shared_fields() -> VortexResult<()> {
        let dtype = struct_dtype();
        let a_gt = gt(col("a"), lit(1i32));
        let b_lt = lt(col("b"), lit(2i32));
        let a_lt_c = lt(col("a"), col("c"));
        let c_eq = eq(col("c"), lit(5i32));
        let b_eq = eq(col("b"), lit(3i32));
        let d_eq = eq(col("d"), lit(0i32));
        let filter = FilterExpr::new(
            and_collect([
                a_gt.clone(),
                b_lt.clone(),
                a_lt_c.clone(),
                c_eq.clone(),
                b_eq.clone(),
                d_eq.clone(),
            ])
            .expect("non-empty")
            .bind(&dtype)?,
        );

        // `a` and `c` chain through `a < c`; `b` and `d` stand apart.
        let expected: Vec<Vec<Expression>> =
            vec![vec![a_gt, a_lt_c, c_eq], vec![b_lt, b_eq], vec![d_eq]];
        assert_eq!(filter.conjuncts().len(), expected.len());
        for (idx, members) in expected.into_iter().enumerate() {
            let bound = members
                .iter()
                .map(|expr| expr.bind(&dtype))
                .collect::<VortexResult<Vec<_>>>()?;
            assert_eq!(filter.members(idx), bound.as_slice());
            assert_eq!(
                filter.conjuncts()[idx],
                and_collect(members).expect("non-empty").bind(&dtype)?
            );
        }
        Ok(())
    }

    #[test]
    fn root_conjunct_joins_every_field_reading_conjunct() -> VortexResult<()> {
        let dtype = DType::Bool(Nullability::Nullable);
        let filter = FilterExpr::new(and(root(), and(not(root()), lit(true))).bind(&dtype)?);

        assert_eq!(filter.conjuncts().len(), 2);
        assert_eq!(
            filter.members(0),
            [root().bind(&dtype)?, not(root()).bind(&dtype)?].as_slice()
        );
        assert_eq!(filter.members(1), [lit(true).bind(&dtype)?].as_slice());
        Ok(())
    }

    #[test]
    fn waits_for_all_conjuncts_before_reordering() -> VortexResult<()> {
        let dtype = struct_dtype();
        let bound = and(gt(col("a"), lit(1i32)), gt(col("b"), lit(1i32))).bind(&dtype)?;
        let filter = FilterExpr::new(bound);

        filter.report_selectivity(0, 0.9);
        assert_eq!(*filter.ordering.read(), vec![0, 1]);

        filter.report_selectivity(1, 0.1);
        assert_eq!(*filter.ordering.read(), vec![1, 0]);
        Ok(())
    }
}
