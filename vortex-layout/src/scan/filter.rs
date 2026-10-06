// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use itertools::Itertools;
use vortex_array::dtype::Field;
use vortex_array::dtype::FieldName;
use vortex_array::expr::BoundExpression;
use vortex_array::expr::analysis::referenced_field_paths;
use vortex_array::scalar_fn::fns::binary::Binary;
use vortex_array::scalar_fn::fns::dynamic::DynamicExprUpdates;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_error::VortexExpect;
use vortex_error::vortex_panic;
use vortex_utils::aliases::hash_map::HashMap;

/// Uncompressed bytes per row of each top level field
pub type FieldByteSizes = HashMap<FieldName, f64>;

/// Cumulative row counts for a single conjunct.
#[derive(Default)]
#[repr(align(64))] // Concurrent updates shouldn't contend
struct ConjunctCost {
    rows_in: AtomicU64,
    rows_out: AtomicU64,
}

/// A [`FilterExpr`] splits boolean expressions into individual conjunctions, tracks
/// per-conjunct selectivity, and uses this information to order the evaluation of the
/// conjunctions in an attempt to minimize the work done.
pub struct FilterExpr {
    /// The conjuncts involved in the filter expression.
    conjuncts: Vec<BoundExpression>,
    /// Cumulative row counts for each conjunct
    conjunct_costs: Box<[ConjunctCost]>,
    /// Uncompressed bytes per row read by each conjunct
    bytes_per_row: Option<Box<[f64]>>,
    /// Dynamic expression trackers for each conjunct, incase they contain dynamic expressions.
    dynamic_conjuncts: Vec<Option<DynamicExprUpdates>>,
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

/// Uncompressed bytes per row read by conjunct
fn conjunct_bytes_per_row(expr: &BoundExpression, sizes: &FieldByteSizes) -> Option<f64> {
    referenced_field_paths(expr)
        .ok()?
        .iter()
        .map(|path| match path.parts().first() {
            Some(Field::Name(name)) => sizes.get(name).copied(),
            _ => None,
        })
        .sum()
}

impl FilterExpr {
    pub fn new(expr: BoundExpression, field_sizes: Option<&FieldByteSizes>) -> Self {
        let conjuncts = bound_conjuncts(&expr);
        let num_conjuncts = conjuncts.len();

        let dynamic_conjuncts = conjuncts.iter().map(DynamicExprUpdates::new).collect_vec();
        let bytes_per_row = field_sizes.and_then(|sizes| {
            conjuncts
                .iter()
                .map(|conjunct| conjunct_bytes_per_row(conjunct, sizes))
                .collect::<Option<Box<[_]>>>()
        });

        Self {
            conjuncts,
            conjunct_costs: (0..num_conjuncts)
                .map(|_| ConjunctCost::default())
                .collect(),
            bytes_per_row,
            dynamic_conjuncts,
        }
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

    /// Preferreed evaluation order of conjuncts
    pub fn conjunct_order(&self) -> Vec<usize> {
        let mut order = (0..self.conjuncts.len()).collect_vec();
        if order.len() < 2 {
            return order;
        }

        let all_observed = self
            .conjunct_costs
            .iter()
            .all(|cost| cost.rows_in.load(Ordering::Relaxed) > 0);
        // We can't do reordering before every conjunct has been seen because
        // we may move more selective predicates forward otherwise.
        if !all_observed && self.bytes_per_row.is_none() {
            return order;
        }

        let scores = self
            .conjunct_costs
            .iter()
            .enumerate()
            .map(|(idx, cost)| {
                let bytes_per_row = self.bytes_per_row.as_ref().map_or(1.0, |bytes| bytes[idx]);
                if !all_observed {
                    return bytes_per_row;
                }
                let rows_in = cost.rows_in.load(Ordering::Relaxed);
                let rows_out = cost.rows_out.load(Ordering::Relaxed);
                bytes_per_row * rows_in as f64 / (1 + rows_in - rows_out) as f64
            })
            .collect_vec();

        order.sort_by(|&l_idx, &r_idx| {
            scores[l_idx]
                .partial_cmp(&scores[r_idx])
                .vortex_expect("Can't compare conjunct scores")
        });
        order
    }

    /// Report evaluation of conjunct over rows_in of which rows_out survived
    pub fn report_evaluation(&self, conjunct_idx: usize, rows_in: u64, rows_out: u64) {
        if rows_out > rows_in {
            vortex_panic!("rows_out {rows_out} must not exceed rows_in {rows_in}");
        }

        let cost = &self.conjunct_costs[conjunct_idx];
        cost.rows_in.fetch_add(rows_in, Ordering::Relaxed);
        cost.rows_out.fetch_add(rows_out, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::dtype::StructFields;
    use vortex_array::expr::and;
    use vortex_array::expr::eq;
    use vortex_array::expr::get_item;
    use vortex_array::expr::lit;
    use vortex_array::expr::not;
    use vortex_array::expr::root;
    use vortex_error::VortexResult;

    use super::FieldByteSizes;
    use super::FilterExpr;

    #[test]
    fn bound_conjuncts_preserve_order_and_types() -> VortexResult<()> {
        let expr = and(root(), and(not(root()), lit(true)));
        let dtype = DType::Bool(Nullability::Nullable);
        let bound = expr.bind(&dtype)?;
        let filter = FilterExpr::new(bound, None);
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

    fn filter(sizes: Option<&FieldByteSizes>) -> VortexResult<FilterExpr> {
        let wide_dtype = DType::Utf8(Nullability::NonNullable);
        let narrow_dtype = DType::Primitive(PType::U8, Nullability::NonNullable);
        let dtype = DType::Struct(
            StructFields::from_iter([("wide", wide_dtype), ("narrow", narrow_dtype)]),
            Nullability::NonNullable,
        );
        let expr = and(
            eq(get_item("wide", root()), lit("x")),
            eq(get_item("narrow", root()), lit(5u8)),
        );
        Ok(FilterExpr::new(expr.bind(&dtype)?, sizes))
    }

    fn test_sizes() -> FieldByteSizes {
        FieldByteSizes::from([("wide".into(), 100.0), ("narrow".into(), 1.0)])
    }

    #[test]
    fn not_all_conjuncts_observed() -> VortexResult<()> {
        let filter = filter(Some(&test_sizes()))?;
        assert_eq!(filter.conjunct_order(), vec![1, 0]);
        filter.report_evaluation(0, 1000, 10);
        assert_eq!(filter.conjunct_order(), vec![1, 0]);
        Ok(())
    }

    #[test]
    fn selective_wide() -> VortexResult<()> {
        let filter = filter(Some(&test_sizes()))?;
        filter.report_evaluation(1, 1000, 1000);
        filter.report_evaluation(0, 1000, 10);
        assert_eq!(filter.conjunct_order(), vec![0, 1]);
        Ok(())
    }

    #[test]
    fn missing_sizes() -> VortexResult<()> {
        let filter = filter(None)?;
        assert_eq!(filter.conjunct_order(), vec![0, 1]);

        filter.report_evaluation(0, 1000, 1000);
        assert_eq!(filter.conjunct_order(), vec![0, 1]);
        filter.report_evaluation(1, 1000, 10);
        assert_eq!(filter.conjunct_order(), vec![1, 0]);
        Ok(())
    }
}
