// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Conversion of DataFusion [`DynamicFilterPhysicalExpr`]s into live Vortex dynamic comparisons.
//!
//! DataFusion operators such as TopK and hash joins push a [`DynamicFilterPhysicalExpr`] into the
//! scan and tighten it while the query runs. Its column children are fixed, so for each child we
//! emit [`dynamic`] comparisons whose right-hand sides are re-read from the DataFusion filter
//! whenever its generation changes. If the filter already has bounds when a file is opened, only
//! the comparisons those bounds use are emitted, since a TopK filter keeps its shape as it
//! tightens. Before its first update (it starts as `true`) the shape is unknown, so every column
//! gets the full template (`<`, `<=`, `>`, `>=`).
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

use std::cmp::Ordering;
use std::sync::Arc;

use datafusion_common::ScalarValue as DFScalarValue;
use datafusion_expr::Operator as DFOperator;
use datafusion_physical_expr::DynamicFilterTracking;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_expr::expressions::DynamicFilterPhysicalExpr;
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
/// conversion is exact, see [`LiveBounds::parse_comparison`].
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

    let current = bounds.current_bounds();
    let per_column = columns
        .into_iter()
        .enumerate()
        .filter_map(|(idx, (name, dtype))| {
            // Once the filter has bounds, columns and operators it doesn't bound are skipped.
            let ops: Vec<_> = if current.is_empty() {
                TEMPLATE_OPS.to_vec()
            } else {
                TEMPLATE_OPS
                    .into_iter()
                    .filter(|op| current.iter().any(|(c, o, _)| *c == idx && o == op))
                    .collect()
            };
            let lhs = get_item(name, root());
            let comparisons = and_collect(ops.into_iter().map(|op| {
                let bounds = Arc::clone(&bounds);
                dynamic(
                    op,
                    move || bounds.value(idx, op),
                    dtype.clone(),
                    true,
                    lhs.clone(),
                )
            }))?;
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

/// `(column index, operator, value)`, with the value already cast to the column's file dtype.
type Bound = (usize, CompareOperator, ScalarValue);

/// The bounds extracted from one generation of a dynamic filter.
struct Snapshot {
    generation: u64,
    bounds: Arc<[Bound]>,
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
        self.current_bounds()
            .iter()
            .find(|(c, o, _)| *c == column && *o == op)
            .map(|(_, _, value)| value.clone())
    }

    /// The bounds of the filter's current generation, re-read only when the generation changes.
    fn current_bounds(&self) -> Arc<[Bound]> {
        let generation = self.filter.snapshot_generation();
        let mut cache = self.cache.lock();
        match cache.as_ref() {
            Some(snapshot) if snapshot.generation == generation => Arc::clone(&snapshot.bounds),
            _ => {
                let bounds: Arc<[Bound]> = self.extract_bounds().into();
                *cache = Some(Snapshot {
                    generation,
                    bounds: Arc::clone(&bounds),
                });
                bounds
            }
        }
    }

    /// Reads `col <op> literal` bounds from the filter's current predicate.
    fn extract_bounds(&self) -> Vec<Bound> {
        self.filter
            .downcast_ref::<DynamicFilterPhysicalExpr>()
            .and_then(|filter| filter.current().ok())
            .and_then(|current| self.bounds_of(&current))
            .unwrap_or_default()
    }

    /// Bounds that every row satisfying `expr` also satisfies, unless the bound's column is null,
    /// or `None` if no row can satisfy `expr` (it is `false` or `NULL`).
    ///
    /// Anything not understood yields no bounds, which can only make the Vortex filter keep more
    /// rows.
    fn bounds_of(&self, expr: &Arc<dyn PhysicalExpr>) -> Option<Vec<Bound>> {
        if let Some(literal) = expr.downcast_ref::<df_expr::Literal>() {
            return match literal.value() {
                DFScalarValue::Boolean(Some(true)) => Some(vec![]),
                DFScalarValue::Boolean(Some(false) | None) | DFScalarValue::Null => None,
                _ => Some(vec![]),
            };
        }
        if let Some(case) = expr.downcast_ref::<df_expr::CaseExpr>() {
            // Rows the CASE keeps satisfy one of its results. Without an ELSE, unmatched rows are
            // NULL and so not kept. Partitioned hash joins route each key to its partition's
            // bounds this way.
            return case
                .when_then_expr()
                .iter()
                .map(|(_, then)| then)
                .chain(case.else_expr())
                .map(|branch| self.bounds_of(branch))
                .reduce(|a, b| self.hull(a, b))
                .flatten();
        }
        let Some(binary) = expr.downcast_ref::<df_expr::BinaryExpr>() else {
            return Some(vec![]);
        };
        match binary.op() {
            DFOperator::And => {
                let mut bounds = self.bounds_of(binary.left())?;
                bounds.extend(self.bounds_of(binary.right())?);
                Some(bounds)
            }
            DFOperator::Or => {
                // `col IS NULL OR <rest>`: nulls always pass our template, so bounds `<rest>`
                // places on the same column still hold.
                if let Some(null_col) = binary
                    .left()
                    .downcast_ref::<df_expr::IsNullExpr>()
                    .and_then(|is_null| column_of(is_null.arg()))
                {
                    let idx = self.column_index(null_col.name());
                    return Some(
                        self.bounds_of(binary.right())
                            .unwrap_or_default()
                            .into_iter()
                            .filter(|(c, ..)| Some(*c) == idx)
                            .collect(),
                    );
                }
                // E.g. multi-column TopK, `a < v OR (a = v AND b < w)`, implies `a <= v`.
                self.hull(
                    self.bounds_of(binary.left()),
                    self.bounds_of(binary.right()),
                )
            }
            _ => Some(self.comparison(binary)),
        }
    }

    /// Bounds implied by either side holding: for each column and direction bounded on both
    /// sides, the looser of the two. A side no row can satisfy (`None`) contributes nothing.
    fn hull(&self, left: Option<Vec<Bound>>, right: Option<Vec<Bound>>) -> Option<Vec<Bound>> {
        let (left, right) = match (left, right) {
            (Some(left), Some(right)) => (left, right),
            (side, None) | (None, side) => return side,
        };
        let mut bounds = vec![];
        for (idx, (_, dtype)) in self.columns.iter().enumerate() {
            for upper in [true, false] {
                let tightest = |side: &[Bound]| {
                    side.iter()
                        .filter(|(c, op, _)| *c == idx && is_upper(*op) == Some(upper))
                        .filter_map(|(_, op, value)| {
                            Some((
                                *op,
                                Scalar::try_new(dtype.clone(), Some(value.clone())).ok()?,
                            ))
                        })
                        // Every bound on one side holds, so any of them is safe to keep.
                        .reduce(|a, b| {
                            if looser(&a, &b, upper) == Some(false) {
                                b
                            } else {
                                a
                            }
                        })
                };
                if let (Some(l), Some(r)) = (tightest(&left), tightest(&right))
                    && let Some(r_is_looser) = looser(&l, &r, upper)
                    && let Some(value) = if r_is_looser { r.1 } else { l.1 }.into_value()
                {
                    let op = if r_is_looser { r.0 } else { l.0 };
                    bounds.push((idx, op, value));
                }
            }
        }
        Some(bounds)
    }

    fn column_index(&self, name: &str) -> Option<usize> {
        self.columns.iter().position(|(column, _)| column == name)
    }

    /// Parses `col <op> literal` or `literal <op> col` into bounds on a tracked column, reading
    /// `col = literal` as both `<=` and `>=`.
    fn comparison(&self, binary: &df_expr::BinaryExpr) -> Vec<Bound> {
        let Some(op) = compare_op(binary.op()) else {
            return vec![];
        };
        let Some((idx, op, value)) = self.parse_comparison(binary, op) else {
            return vec![];
        };
        match op {
            CompareOperator::Eq => vec![
                (idx, CompareOperator::Lte, value.clone()),
                (idx, CompareOperator::Gte, value),
            ],
            op => vec![(idx, op, value)],
        }
    }

    fn parse_comparison(&self, binary: &df_expr::BinaryExpr, op: CompareOperator) -> Option<Bound> {
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

        let idx = self.column_index(col.name())?;
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

/// `Some(true)` for upper bounds (`<`, `<=`), `Some(false)` for lower bounds (`>`, `>=`).
fn is_upper(op: CompareOperator) -> Option<bool> {
    match op {
        CompareOperator::Lt | CompareOperator::Lte => Some(true),
        CompareOperator::Gt | CompareOperator::Gte => Some(false),
        CompareOperator::Eq | CompareOperator::NotEq => None,
    }
}

/// Whether bound `b` keeps more values than bound `a` (both upper or both lower), or `None` when
/// their values can't be compared.
fn looser(
    a: &(CompareOperator, Scalar),
    b: &(CompareOperator, Scalar),
    upper: bool,
) -> Option<bool> {
    Some(match a.1.partial_cmp(&b.1)? {
        Ordering::Equal => {
            matches!(a.0, CompareOperator::Lt | CompareOperator::Gt)
                && matches!(b.0, CompareOperator::Lte | CompareOperator::Gte)
        }
        Ordering::Less => upper,
        Ordering::Greater => !upper,
    })
}

fn compare_op(op: &DFOperator) -> Option<CompareOperator> {
    Some(match op {
        DFOperator::Lt => CompareOperator::Lt,
        DFOperator::LtEq => CompareOperator::Lte,
        DFOperator::Gt => CompareOperator::Gt,
        DFOperator::GtEq => CompareOperator::Gte,
        DFOperator::Eq => CompareOperator::Eq,
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

    /// A filter that already has bounds when the file opens only gets the comparisons it uses,
    /// which keep tracking later updates, and stay safe if the filter's shape changes.
    #[test]
    fn narrowed_template_after_first_update() -> anyhow::Result<()> {
        let session = VortexSession::default();
        let input = StructArray::from_fields(&[
            ("a", buffer![1i32, 5, 10].into_array()),
            ("b", buffer![7i32, 0, 0].into_array()),
        ])?
        .into_array();
        let dynamic = Arc::new(DynamicFilterPhysicalExpr::new(
            vec![col_a(), col_b()],
            binary(col_a(), DFOperator::Lt, lit_i32(10)),
        ));
        let filter: PhysicalExprRef = Arc::clone(&dynamic) as _;
        let fields = input.dtype().as_struct_fields_opt().expect("struct input");
        let converted = dynamic_filter_to_vortex(&filter, fields, None, &session)
            .expect("filter should convert");
        assert_eq!(converted.to_string().matches("dynamic(").count(), 1);
        let mut ctx = session.create_execution_ctx();

        dynamic.update(binary(col_a(), DFOperator::Lt, lit_i32(5)))?;
        assert_arrays_eq!(
            input.clone().apply(&converted)?,
            BoolArray::from_iter([true, false, false]),
            &mut ctx
        );

        // `b` was not part of the filter when the file opened, so it is not tracked, and `a` no
        // longer has a bound: every row passes.
        dynamic.update(binary(col_b(), DFOperator::Lt, lit_i32(3)))?;
        assert_arrays_eq!(
            input.apply(&converted)?,
            BoolArray::from_iter([true, true, true]),
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

    fn col_b() -> PhysicalExprRef {
        Arc::new(df_expr::Column::new("b", 1))
    }

    /// Multi-column TopK filters are disjunctions over the sort keys; the leading key still gets
    /// a bound from the loosest side of each disjunction.
    #[rstest]
    // ORDER BY a, b: `a < 5 OR (a = 5 AND b < 3)` implies `a <= 5`.
    #[case(
        binary(
            binary(col_a(), DFOperator::Lt, lit_i32(5)),
            DFOperator::Or,
            binary(
                binary(col_a(), DFOperator::Eq, lit_i32(5)),
                DFOperator::And,
                binary(col_b(), DFOperator::Lt, lit_i32(3)),
            ),
        ),
        [true, true, false],
    )]
    // ORDER BY a DESC, b DESC: `a > 5 OR (a = 5 AND b > 3)` implies `a >= 5`.
    #[case(
        binary(
            binary(col_a(), DFOperator::Gt, lit_i32(5)),
            DFOperator::Or,
            binary(
                binary(col_a(), DFOperator::Eq, lit_i32(5)),
                DFOperator::And,
                binary(col_b(), DFOperator::Gt, lit_i32(3)),
            ),
        ),
        [false, true, true],
    )]
    // The hull keeps the looser bound: `a < 2 OR a <= 5` implies `a <= 5`.
    #[case(
        binary(
            binary(col_a(), DFOperator::Lt, lit_i32(2)),
            DFOperator::Or,
            binary(col_a(), DFOperator::LtEq, lit_i32(5)),
        ),
        [true, true, false],
    )]
    fn disjunction_bounds(
        #[case] current: PhysicalExprRef,
        #[case] expected: [bool; 3],
    ) -> VortexResult<()> {
        let session = VortexSession::default();
        let input = StructArray::from_fields(&[
            ("a", buffer![1i32, 5, 10].into_array()),
            ("b", buffer![0i32, 0, 0].into_array()),
        ])?
        .into_array();
        let filter: PhysicalExprRef = Arc::new(DynamicFilterPhysicalExpr::new(
            vec![col_a(), col_b()],
            current,
        ));
        assert_filter(&session, &input, &filter, BoolArray::from_iter(expected))
    }

    fn range(lower: i32, upper: i32) -> PhysicalExprRef {
        binary(
            binary(col_a(), DFOperator::GtEq, lit_i32(lower)),
            DFOperator::And,
            binary(col_a(), DFOperator::LtEq, lit_i32(upper)),
        )
    }

    fn lit_bool(value: bool) -> PhysicalExprRef {
        Arc::new(df_expr::Literal::new(ScalarValue::Boolean(Some(value))))
    }

    /// Partitioned hash joins route each key to its partition's bounds with a CASE; every kept
    /// row lies within the hull of the partitions' bounds.
    #[rstest]
    // Partitions [2, 4] and [5, 6], plus an empty partition (`false`): `2 <= a <= 6`.
    #[case(vec![range(2, 4), lit_bool(false)], Some(range(5, 6)), [false, true, false])]
    // Without an ELSE, unmatched rows are NULL and not kept.
    #[case(vec![range(2, 4), range(5, 6)], None, [false, true, false])]
    // A partition with unknown contents (`true`) keeps every row.
    #[case(vec![range(2, 4), lit_bool(true)], None, [true, true, true])]
    fn partitioned_join_case_bounds(
        #[case] thens: Vec<PhysicalExprRef>,
        #[case] else_expr: Option<PhysicalExprRef>,
        #[case] expected: [bool; 3],
    ) -> anyhow::Result<()> {
        let session = VortexSession::default();
        let input =
            StructArray::from_fields(&[("a", buffer![1i32, 5, 10].into_array())])?.into_array();
        let when_then = thens
            .into_iter()
            .enumerate()
            .map(|(partition, then)| {
                let partition = u64::try_from(partition).expect("small partition index");
                let when: PhysicalExprRef =
                    Arc::new(df_expr::Literal::new(ScalarValue::UInt64(Some(partition))));
                (when, then)
            })
            .collect();
        let current: PhysicalExprRef = Arc::new(df_expr::CaseExpr::try_new(
            Some(col_a()),
            when_then,
            else_expr,
        )?);
        let filter: PhysicalExprRef =
            Arc::new(DynamicFilterPhysicalExpr::new(vec![col_a()], current));
        assert_filter(&session, &input, &filter, BoolArray::from_iter(expected))?;
        Ok(())
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
