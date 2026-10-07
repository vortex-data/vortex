// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Conversion of DataFusion [`DynamicFilterPhysicalExpr`]s into live Vortex dynamic comparisons.
//!
//! DataFusion operators such as TopK and hash joins push a [`DynamicFilterPhysicalExpr`] into the
//! scan and tighten it while the query runs. The filter's shape is unknown when a file is opened
//! (it usually starts as `true`), but its column children are fixed. For each child column we
//! therefore emit a fixed template of [`dynamic`] comparisons (`<`, `<=`, `>`, `>=`), whose
//! right-hand sides are re-read from the DataFusion filter whenever its generation changes.
//!
//! Every comparison defaults to `true` when the current DataFusion predicate has no matching
//! bound, so the Vortex filter only ever keeps a superset of the rows DataFusion's filter keeps.
//! That is sufficient because the operators that produce dynamic filters still enforce their own
//! semantics; the filter only lets the scan skip data.

use std::sync::Arc;

use datafusion_expr::Operator as DFOperator;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_expr::expressions::DynamicFilterPhysicalExpr;
use datafusion_physical_expr::split_conjunction;
use datafusion_physical_plan::expressions as df_expr;
use parking_lot::Mutex;
use vortex::dtype::DType;
use vortex::dtype::StructFields;
use vortex::expr::Expression;
use vortex::expr::and_collect;
use vortex::expr::dynamic;
use vortex::expr::get_item;
use vortex::expr::is_null;
use vortex::expr::or;
use vortex::expr::root;
use vortex::scalar::ScalarValue;
use vortex::scalar_fn::fns::operators::CompareOperator;
use vortex::session::VortexSession;

use crate::convert::scalar_from_df;

const TEMPLATE_OPS: [CompareOperator; 4] = [
    CompareOperator::Lt,
    CompareOperator::Lte,
    CompareOperator::Gt,
    CompareOperator::Gte,
];

/// Returns the dynamic filter if `expr` is one whose children are all plain columns.
pub(crate) fn as_column_dynamic_filter(
    expr: &Arc<dyn PhysicalExpr>,
) -> Option<&DynamicFilterPhysicalExpr> {
    let dynamic = expr.downcast_ref::<DynamicFilterPhysicalExpr>()?;
    let children = dynamic.children();
    (!children.is_empty()
        && children
            .iter()
            .all(|child| child.downcast_ref::<df_expr::Column>().is_some()))
    .then_some(dynamic)
}

/// Converts a DataFusion dynamic filter into a Vortex expression over a file with `file_fields`.
///
/// Returns `None` when none of the filter's columns can be tracked, in which case the filter is
/// simply not applied by the scan.
pub(crate) fn dynamic_filter_to_vortex(
    expr: &Arc<dyn PhysicalExpr>,
    file_fields: &StructFields,
    session: &VortexSession,
) -> Option<Expression> {
    let dynamic_filter = as_column_dynamic_filter(expr)?;

    let columns: Vec<(String, DType)> = dynamic_filter
        .children()
        .into_iter()
        .filter_map(|child| child.downcast_ref::<df_expr::Column>())
        .filter_map(|col| {
            let dtype = file_fields.field(col.name())?;
            Some((col.name().to_owned(), dtype))
        })
        .collect();
    if columns.is_empty() {
        return None;
    }

    let bounds = Arc::new(LiveBounds {
        filter: Arc::clone(expr),
        columns: columns.clone(),
        session: session.clone(),
        cache: Mutex::new(None),
    });

    let per_column = columns
        .into_iter()
        .enumerate()
        .filter_map(|(idx, (name, dtype))| {
            let lhs = get_item(name, root());
            let comparisons = TEMPLATE_OPS.map(|op| {
                let bounds = Arc::clone(&bounds);
                dynamic(
                    op,
                    move || bounds.value(idx, op),
                    dtype.clone(),
                    true,
                    lhs.clone(),
                )
            });
            let comparisons = and_collect(comparisons)?;
            // Nulls always pass: the DataFusion filter may keep them (e.g. TopK with NULLS FIRST),
            // and keeping extra rows is always safe.
            Some(if dtype.is_nullable() {
                or(is_null(lhs), comparisons)
            } else {
                comparisons
            })
        });

    and_collect(per_column)
}

/// The bounds extracted from the current generation of a dynamic filter.
struct Snapshot {
    generation: u64,
    /// `(column index, operator, value)` with values already cast to the column's file dtype.
    bounds: Vec<(usize, CompareOperator, ScalarValue)>,
}

struct LiveBounds {
    /// Always a [`DynamicFilterPhysicalExpr`]; kept type-erased because that type is not `Clone`.
    filter: Arc<dyn PhysicalExpr>,
    columns: Vec<(String, DType)>,
    session: VortexSession,
    cache: Mutex<Option<Snapshot>>,
}

impl LiveBounds {
    fn value(&self, column: usize, op: CompareOperator) -> Option<ScalarValue> {
        let generation = self.filter.snapshot_generation();
        let mut cache = self.cache.lock();
        if cache.as_ref().is_none_or(|s| s.generation != generation) {
            *cache = Some(Snapshot {
                generation,
                bounds: self.extract_bounds(),
            });
        }
        cache
            .as_ref()?
            .bounds
            .iter()
            .find(|(c, o, _)| *c == column && *o == op)
            .map(|(_, _, value)| value.clone())
    }

    /// Reads `col <op> literal` bounds from the filter's current predicate.
    ///
    /// Only bounds that every row kept by the predicate satisfies are returned; anything else
    /// is dropped, which can only make the Vortex filter keep more rows.
    fn extract_bounds(&self) -> Vec<(usize, CompareOperator, ScalarValue)> {
        let Some(Ok(current)) = self
            .filter
            .downcast_ref::<DynamicFilterPhysicalExpr>()
            .map(DynamicFilterPhysicalExpr::current)
        else {
            return vec![];
        };

        let mut bounds = vec![];
        for conjunct in split_conjunction(&current) {
            if let Some(bound) = self.comparison(conjunct, None) {
                bounds.push(bound);
                continue;
            }

            // `col IS NULL OR <comparisons on col>`: our template always keeps nulls, so the
            // comparisons on the same column are valid bounds for it.
            let Some(binary) = conjunct.downcast_ref::<df_expr::BinaryExpr>() else {
                continue;
            };
            if *binary.op() != DFOperator::Or {
                continue;
            }
            let Some(null_col) = binary
                .left()
                .downcast_ref::<df_expr::IsNullExpr>()
                .and_then(|is_null| is_null.arg().downcast_ref::<df_expr::Column>())
            else {
                continue;
            };
            bounds.extend(
                split_conjunction(binary.right())
                    .into_iter()
                    .filter_map(|expr| self.comparison(expr, Some(null_col.name()))),
            );
        }
        bounds
    }

    /// Parses `col <op> literal` or `literal <op> col` into a bound on a tracked column.
    fn comparison(
        &self,
        expr: &Arc<dyn PhysicalExpr>,
        only_column: Option<&str>,
    ) -> Option<(usize, CompareOperator, ScalarValue)> {
        let binary = expr.downcast_ref::<df_expr::BinaryExpr>()?;
        let op = compare_op(binary.op())?;

        let (col, literal, op) = match (
            binary.left().downcast_ref::<df_expr::Column>(),
            binary.right().downcast_ref::<df_expr::Literal>(),
        ) {
            (Some(col), Some(literal)) => (col, literal, op),
            _ => (
                binary.right().downcast_ref::<df_expr::Column>()?,
                binary.left().downcast_ref::<df_expr::Literal>()?,
                op.swap(),
            ),
        };

        if only_column.is_some_and(|name| name != col.name()) {
            return None;
        }

        let idx = self
            .columns
            .iter()
            .position(|(name, _)| name == col.name())?;
        let dtype = &self.columns[idx].1;

        let scalar = scalar_from_df(literal.value(), &self.session).ok()?;
        let scalar = if scalar.dtype().eq_ignore_nullability(dtype) {
            scalar
        } else {
            scalar.cast(&dtype.as_nullable()).ok()?
        };
        Some((idx, op, scalar.into_value()?))
    }
}

fn compare_op(op: &DFOperator) -> Option<CompareOperator> {
    Some(match op {
        DFOperator::Lt => CompareOperator::Lt,
        DFOperator::LtEq => CompareOperator::Lte,
        DFOperator::Gt => CompareOperator::Gt,
        DFOperator::GtEq => CompareOperator::Gte,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use datafusion_common::ScalarValue;
    use datafusion_expr::Operator as DFOperator;
    use datafusion_physical_expr::PhysicalExpr;
    use datafusion_physical_expr::expressions::DynamicFilterPhysicalExpr;
    use datafusion_physical_plan::expressions as df_expr;
    use rstest::rstest;
    use vortex::VortexSessionDefault;
    use vortex::array::ArrayRef;
    use vortex::array::IntoArray;
    use vortex::array::VortexSessionExecute;
    use vortex::array::arrays::BoolArray;
    use vortex::array::arrays::PrimitiveArray;
    use vortex::array::arrays::StructArray;
    use vortex::array::assert_arrays_eq;
    use vortex::buffer::buffer;
    use vortex::error::VortexResult;
    use vortex::session::VortexSession;

    use super::dynamic_filter_to_vortex;

    type PhysicalExprRef = Arc<dyn PhysicalExpr>;

    fn col_a() -> PhysicalExprRef {
        Arc::new(df_expr::Column::new("a", 0))
    }

    fn lit_i32(v: i32) -> PhysicalExprRef {
        Arc::new(df_expr::Literal::new(ScalarValue::Int32(Some(v))))
    }

    fn binary(l: PhysicalExprRef, op: DFOperator, r: PhysicalExprRef) -> PhysicalExprRef {
        Arc::new(df_expr::BinaryExpr::new(l, op, r))
    }

    fn assert_filter(
        session: &VortexSession,
        input: &ArrayRef,
        filter: &PhysicalExprRef,
        expected: BoolArray,
    ) -> VortexResult<()> {
        let fields = input.dtype().as_struct_fields_opt().expect("struct input");
        let converted =
            dynamic_filter_to_vortex(filter, fields, session).expect("filter should convert");
        let actual = input.clone().apply(&converted)?;
        assert_arrays_eq!(actual, expected, &mut session.create_execution_ctx());
        Ok(())
    }

    /// The Vortex filter tracks updates made to the DataFusion filter after conversion.
    #[test]
    fn tracks_updates() -> anyhow::Result<()> {
        let session = VortexSession::default();
        let input =
            StructArray::from_fields(&[("a", buffer![1i32, 5, 10].into_array())])?.into_array();
        let dynamic = Arc::new(DynamicFilterPhysicalExpr::new(
            vec![col_a()],
            Arc::new(df_expr::Literal::new(ScalarValue::Boolean(Some(true)))),
        ));
        let filter: PhysicalExprRef = Arc::clone(&dynamic) as _;
        let fields = input.dtype().as_struct_fields_opt().expect("struct input");
        let converted =
            dynamic_filter_to_vortex(&filter, fields, &session).expect("filter should convert");
        let mut ctx = session.create_execution_ctx();

        assert_arrays_eq!(
            input.clone().apply(&converted)?,
            BoolArray::from_iter([true, true, true]),
            &mut ctx
        );

        dynamic.update(binary(col_a(), DFOperator::Lt, lit_i32(5)))?;
        assert_arrays_eq!(
            input.clone().apply(&converted)?,
            BoolArray::from_iter([true, false, false]),
            &mut ctx
        );

        dynamic.update(binary(col_a(), DFOperator::Lt, lit_i32(2)))?;
        assert_arrays_eq!(
            input.apply(&converted)?,
            BoolArray::from_iter([true, false, false]),
            &mut ctx
        );
        Ok(())
    }

    #[rstest]
    // Literal on the left flips the operator.
    #[case(binary(lit_i32(10), DFOperator::Gt, col_a()), [true, true, false])]
    // Hash join min/max bounds.
    #[case(
        binary(
            binary(col_a(), DFOperator::GtEq, lit_i32(5)),
            DFOperator::And,
            binary(col_a(), DFOperator::LtEq, lit_i32(10)),
        ),
        [false, true, true],
    )]
    // Disjunctions across bounds (aggregate min/max) are not tracked, so every row passes.
    #[case(
        binary(
            binary(col_a(), DFOperator::Lt, lit_i32(2)),
            DFOperator::Or,
            binary(col_a(), DFOperator::Gt, lit_i32(8)),
        ),
        [true, true, true],
    )]
    // Unsupported conjuncts are dropped while supported ones still apply.
    #[case(
        binary(
            binary(col_a(), DFOperator::Gt, lit_i32(1)),
            DFOperator::And,
            binary(col_a(), DFOperator::NotEq, lit_i32(5)),
        ),
        [false, true, true],
    )]
    fn converts_filter_shapes(
        #[case] current: PhysicalExprRef,
        #[case] expected: [bool; 3],
    ) -> VortexResult<()> {
        let session = VortexSession::default();
        let input =
            StructArray::from_fields(&[("a", buffer![1i32, 5, 10].into_array())])?.into_array();
        let filter: PhysicalExprRef =
            Arc::new(DynamicFilterPhysicalExpr::new(vec![col_a()], current));
        assert_filter(&session, &input, &filter, BoolArray::from_iter(expected))
    }

    /// Nulls always pass, matching TopK's `a IS NULL OR a > v` for NULLS FIRST and staying safe
    /// for NULLS LAST.
    #[rstest]
    #[case(binary(col_a(), DFOperator::Gt, lit_i32(4)))]
    #[case(binary(
        Arc::new(df_expr::IsNullExpr::new(col_a())),
        DFOperator::Or,
        binary(col_a(), DFOperator::Gt, lit_i32(4)),
    ))]
    fn nulls_pass(#[case] current: PhysicalExprRef) -> VortexResult<()> {
        let session = VortexSession::default();
        let a = PrimitiveArray::from_option_iter([Some(1i32), None, Some(10)]).into_array();
        let input = StructArray::from_fields(&[("a", a)])?.into_array();
        let filter: PhysicalExprRef =
            Arc::new(DynamicFilterPhysicalExpr::new(vec![col_a()], current));
        // The comparison is nullable, but the `IS NULL` branch makes every row non-null.
        let expected = BoolArray::from_iter([Some(false), Some(true), Some(true)]);
        assert_filter(&session, &input, &filter, expected)
    }

    #[test]
    fn missing_column_is_not_converted() -> VortexResult<()> {
        let session = VortexSession::default();
        let input = StructArray::from_fields(&[("b", buffer![1i32].into_array())])?.into_array();
        let filter: PhysicalExprRef = Arc::new(DynamicFilterPhysicalExpr::new(
            vec![col_a()],
            binary(col_a(), DFOperator::Lt, lit_i32(5)),
        ));
        let fields = input.dtype().as_struct_fields_opt().expect("struct input");
        assert!(dynamic_filter_to_vortex(&filter, fields, &session).is_none());
        Ok(())
    }
}
