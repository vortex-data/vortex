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
//!
//! Filters that are already complete when the file is opened (typically hash-join filters, whose
//! build side has finished) can never change again. Their bounds are read once and only applied
//! when the file's statistics suggest they will skip a meaningful share of rows, because
//! evaluating a non-selective filter costs more than it saves.

use std::sync::Arc;

use datafusion_expr::Operator as DFOperator;
use datafusion_physical_expr::DynamicFilterTracking;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_expr::expressions::DynamicFilterPhysicalExpr;
use datafusion_physical_expr::split_conjunction;
use datafusion_physical_plan::expressions as df_expr;
use parking_lot::Mutex;
use vortex::dtype::DType;
use vortex::dtype::StructFields;
use vortex::expr::Expression;
use vortex::expr::and_collect;
use vortex::expr::binary;
use vortex::expr::dynamic;
use vortex::expr::get_item;
use vortex::expr::is_null;
use vortex::expr::lit;
use vortex::expr::or;
use vortex::expr::root;
use vortex::expr::stats::Stat;
use vortex::file::FileStatistics;
use vortex::scalar::Scalar;
use vortex::scalar::ScalarValue;
use vortex::scalar_fn::fns::operators::CompareOperator;
use vortex::session::VortexSession;

use crate::convert::scalar_from_df;

/// Complete filters are only applied when they are estimated to keep at most this fraction of a
/// file's value range for some column.
const MAX_KEPT_FRACTION: f64 = 0.5;

const TEMPLATE_OPS: [CompareOperator; 4] = [
    CompareOperator::Lt,
    CompareOperator::Lte,
    CompareOperator::Gt,
    CompareOperator::Gte,
];

/// The column referenced by `expr`, looking through casts the expression adapter adds when a
/// file's type differs from the table's (e.g. a different timestamp unit).
///
/// Bounds on the cast column are converted back to the file type and only kept when that
/// conversion is exact, see [`LiveBounds::comparison`].
fn column_of(expr: &Arc<dyn PhysicalExpr>) -> Option<&df_expr::Column> {
    expr.downcast_ref::<df_expr::Column>().or_else(|| {
        expr.downcast_ref::<df_expr::CastExpr>()?
            .expr()
            .downcast_ref::<df_expr::Column>()
    })
}

/// Returns the dynamic filter if `expr` is one whose children are all (possibly cast) columns.
pub(crate) fn as_column_dynamic_filter(
    expr: &Arc<dyn PhysicalExpr>,
) -> Option<&DynamicFilterPhysicalExpr> {
    let dynamic = expr.downcast_ref::<DynamicFilterPhysicalExpr>()?;
    let children = dynamic.children();
    (!children.is_empty() && children.iter().all(|child| column_of(child).is_some()))
        .then_some(dynamic)
}

/// Converts a DataFusion dynamic filter into a Vortex expression over a file with `file_fields`.
///
/// Returns `None` when the filter is not worth applying to this file, in which case the scan
/// simply ignores it.
pub(crate) fn dynamic_filter_to_vortex(
    expr: &Arc<dyn PhysicalExpr>,
    file_fields: &StructFields,
    file_stats: Option<&FileStatistics>,
    session: &VortexSession,
) -> Option<Expression> {
    let dynamic_filter = as_column_dynamic_filter(expr)?;

    let columns: Vec<(String, DType)> = dynamic_filter
        .children()
        .into_iter()
        .filter_map(|child| column_of(child))
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

    if matches!(
        DynamicFilterTracking::classify(expr),
        DynamicFilterTracking::AllComplete
    ) {
        return bounds.selective_static_filter(file_fields, file_stats?);
    }

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
    /// Builds static comparisons from the filter's current bounds, keeping only columns whose
    /// bounds are estimated to exclude a meaningful share of the file.
    fn selective_static_filter(
        &self,
        file_fields: &StructFields,
        file_stats: &FileStatistics,
    ) -> Option<Expression> {
        let bounds = self.extract_bounds();
        let per_column = self
            .columns
            .iter()
            .enumerate()
            .filter_map(|(idx, (name, dtype))| {
                let column_bounds: Vec<_> = bounds
                    .iter()
                    .filter(|(c, ..)| *c == idx)
                    .map(|(_, op, value)| {
                        (*op, Scalar::try_new(dtype.clone(), Some(value.clone())))
                    })
                    .map(|(op, scalar)| Some((op, scalar.ok()?)))
                    .collect::<Option<_>>()?;
                if column_bounds.is_empty() {
                    return None;
                }

                let (stats, _) = file_stats.get(file_fields.find(name)?);
                let stat = |stat| Scalar::try_new(dtype.clone(), stats.get(stat).into_inner()).ok();
                let kept = kept_fraction(&column_bounds, &stat(Stat::Min)?, &stat(Stat::Max)?)?;
                if kept > MAX_KEPT_FRACTION {
                    return None;
                }

                let lhs = get_item(name.clone(), root());
                let comparisons = and_collect(
                    column_bounds
                        .into_iter()
                        .map(|(op, value)| binary(op.into(), lhs.clone(), lit(value))),
                )?;
                Some(if dtype.is_nullable() {
                    or(is_null(lhs), comparisons)
                } else {
                    comparisons
                })
            });
        and_collect(per_column)
    }

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
                .and_then(|is_null| column_of(is_null.arg()))
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
            column_of(binary.left()),
            binary.right().downcast_ref::<df_expr::Literal>(),
        ) {
            (Some(col), Some(literal)) => (col, literal, op),
            _ => (
                column_of(binary.right())?,
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
            // A lossy cast (e.g. truncating a timestamp to a coarser unit) could make the bound
            // stricter than the original, so only accept casts that round-trip exactly.
            let cast = scalar.cast(&dtype.as_nullable()).ok()?;
            if cast.cast(scalar.dtype()).ok()? != scalar {
                return None;
            }
            cast
        };
        Some((idx, op, scalar.into_value()?))
    }
}

/// Estimates the fraction of the `[min, max]` range kept by `bounds`, assuming values are spread
/// uniformly. Returns `None` for non-numeric columns.
fn kept_fraction(bounds: &[(CompareOperator, Scalar)], min: &Scalar, max: &Scalar) -> Option<f64> {
    let as_f64 = |s: &Scalar| s.as_primitive_opt()?.as_::<f64>();
    let (min, max) = (as_f64(min)?, as_f64(max)?);
    let mut lower = min;
    let mut upper = max;
    for (op, value) in bounds {
        let value = as_f64(value)?;
        match op {
            CompareOperator::Gt | CompareOperator::Gte => lower = lower.max(value),
            CompareOperator::Lt | CompareOperator::Lte => upper = upper.min(value),
            CompareOperator::Eq | CompareOperator::NotEq => {}
        }
    }
    if max <= min {
        return Some(if lower <= upper { 1.0 } else { 0.0 });
    }
    Some(((upper - lower) / (max - min)).clamp(0.0, 1.0))
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

    use arrow_schema::DataType;
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
    use vortex::array::stats::StatsSet;
    use vortex::buffer::buffer;
    use vortex::error::VortexResult;
    use vortex::expr::stats::Precision;
    use vortex::expr::stats::Stat;
    use vortex::file::FileStatistics;
    use vortex::scalar::ScalarValue as VxScalarValue;
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
            dynamic_filter_to_vortex(filter, fields, None, session).expect("filter should convert");
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
        let converted = dynamic_filter_to_vortex(&filter, fields, None, &session)
            .expect("filter should convert");
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

    /// Bounds on a cast column are applied only when the literal converts exactly to the file type.
    #[rstest]
    #[case(DataType::Int64, ScalarValue::Int64(Some(5)), [true, false, false])]
    // 4.5 has no exact INT equivalent; truncating to `a < 4` would wrongly drop `a = 4`.
    #[case(DataType::Float64, ScalarValue::Float64(Some(4.5)), [true, true, true])]
    fn cast_column_bounds(
        #[case] cast_to: DataType,
        #[case] value: ScalarValue,
        #[case] expected: [bool; 3],
    ) -> VortexResult<()> {
        let session = VortexSession::default();
        let input =
            StructArray::from_fields(&[("a", buffer![1i32, 5, 10].into_array())])?.into_array();
        let cast_a: PhysicalExprRef = Arc::new(df_expr::CastExpr::new(col_a(), cast_to, None));
        let current = binary(
            Arc::clone(&cast_a),
            DFOperator::Lt,
            Arc::new(df_expr::Literal::new(value)),
        );
        let filter: PhysicalExprRef =
            Arc::new(DynamicFilterPhysicalExpr::new(vec![cast_a], current));
        assert_filter(&session, &input, &filter, BoolArray::from_iter(expected))
    }

    fn complete_filter(current: PhysicalExprRef) -> PhysicalExprRef {
        let filter = DynamicFilterPhysicalExpr::new(vec![col_a()], current);
        filter.mark_complete();
        Arc::new(filter)
    }

    /// Complete filters are applied as static bounds only when they skip enough of the file.
    #[rstest]
    // Keeps [4, 5] of [1, 10]: applied.
    #[case(4, 5, Some([false, true, false]))]
    // Keeps [2, 10] of [1, 10]: not worth evaluating.
    #[case(2, 10, None)]
    fn complete_filter_applied_when_selective(
        #[case] lower: i32,
        #[case] upper: i32,
        #[case] expected: Option<[bool; 3]>,
    ) -> VortexResult<()> {
        let session = VortexSession::default();
        let input =
            StructArray::from_fields(&[("a", buffer![1i32, 5, 10].into_array())])?.into_array();
        let stats = StatsSet::from_iter([
            (Stat::Min, Precision::exact(VxScalarValue::from(1i32))),
            (Stat::Max, Precision::exact(VxScalarValue::from(10i32))),
        ]);
        let file_stats = FileStatistics::new_with_dtype(Arc::from([stats]), input.dtype());
        let filter = complete_filter(binary(
            binary(col_a(), DFOperator::GtEq, lit_i32(lower)),
            DFOperator::And,
            binary(col_a(), DFOperator::LtEq, lit_i32(upper)),
        ));
        let fields = input.dtype().as_struct_fields_opt().expect("struct input");

        let converted = dynamic_filter_to_vortex(&filter, fields, Some(&file_stats), &session);
        match (converted, expected) {
            (Some(converted), Some(expected)) => assert_arrays_eq!(
                input.apply(&converted)?,
                BoolArray::from_iter(expected),
                &mut session.create_execution_ctx()
            ),
            (None, None) => {}
            (converted, expected) => panic!("expected {expected:?}, got {converted:?}"),
        }
        Ok(())
    }

    #[test]
    fn complete_filter_without_stats_is_not_converted() -> VortexResult<()> {
        let session = VortexSession::default();
        let input = StructArray::from_fields(&[("a", buffer![1i32].into_array())])?.into_array();
        let filter = complete_filter(binary(col_a(), DFOperator::Lt, lit_i32(5)));
        let fields = input.dtype().as_struct_fields_opt().expect("struct input");
        assert!(dynamic_filter_to_vortex(&filter, fields, None, &session).is_none());
        Ok(())
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
        assert!(dynamic_filter_to_vortex(&filter, fields, None, &session).is_none());
        Ok(())
    }
}
