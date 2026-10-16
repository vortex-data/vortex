// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Conversion of DataFusion [`DynamicFilterPhysicalExpr`]s into live Vortex dynamic comparisons.
//!
//! DataFusion operators such as TopK and hash joins push a [`DynamicFilterPhysicalExpr`] into the
//! scan and tighten it while the query runs. Its column children are fixed, so for each child we
//! emit [`dynamic`] comparisons whose right-hand sides are re-read from the DataFusion filter
//! whenever its generation changes. If the filter already has bounds when a file is opened, only
//! the comparisons those bounds use are emitted, since a TopK filter keeps its shape as it
//! tightens. Before its first update (it starts as `true`) the shape is unknown, so the leading
//! column gets the template `<=` and `>=`; a strict bound is applied through its non-strict
//! comparison, which keeps a superset of the rows.
//!
//! Every comparison defaults to `true` when the current DataFusion predicate has no matching
//! bound, so the Vortex filter only ever keeps a superset of the rows DataFusion's filter keeps.
//! That is sufficient because the operators that produce dynamic filters still enforce their own
//! semantics; the filter only lets the scan skip data.
//!
//! Filters that are already complete when the file is opened (typically hash-join filters, whose
//! build side has finished) can never change again. Their bounds are read once and only applied
//! when the file's statistics suggest they will skip a meaningful share of rows, because
//! evaluating a non-selective filter costs more than it saves. A live filter that completes while
//! the file is being scanned gets the same check: bounds on columns where they are not selective
//! are dropped, so their comparisons always pass. The scan then stops evaluating them (see
//! `ScanBuilder::with_skip_unselective_dynamic`).

use std::cmp::Ordering;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering as AtomicOrdering;

use arc_swap::ArcSwapOption;
use datafusion_common::ScalarValue as DFScalarValue;
use datafusion_expr::Operator as DFOperator;
use datafusion_physical_expr::DynamicFilterTracking;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_expr::expressions::DynamicFilterPhysicalExpr;
use datafusion_physical_expr::split_conjunction;
use datafusion_physical_plan::expressions as df_expr;
use vortex::dtype::DType;
use vortex::dtype::Nullability;
use vortex::dtype::StructFields;
use vortex::expr::Expression;
use vortex::expr::and_collect;
use vortex::expr::between;
use vortex::expr::binary;
use vortex::expr::dynamic;
use vortex::expr::get_item;
use vortex::expr::in_list;
use vortex::expr::is_null;
use vortex::expr::lit;
use vortex::expr::or;
use vortex::expr::root;
use vortex::expr::stats::Stat;
use vortex::file::FileStatistics;
use vortex::scalar::PValue;
use vortex::scalar::Scalar;
use vortex::scalar::ScalarValue;
use vortex::scalar_fn::fns::between::BetweenOptions;
use vortex::scalar_fn::fns::between::StrictComparison;
use vortex::scalar_fn::fns::operators::CompareOperator;
use vortex::session::VortexSession;
use vortex_utils::aliases::hash_set::HashSet;

use crate::convert::scalar_from_df;

/// Complete filters are only applied when they are estimated to keep at most this fraction of a
/// file's value range for some column.
const MAX_KEPT_FRACTION: f64 = 0.5;

/// The most values a membership list may hold before it is dropped. `in_list` costs one
/// comparison over the column per value, so beyond this the list is slower than the join probe
/// it saves. DataFusion's own `hash_join_inlist_pushdown_max_distinct_values` is 150.
const MAX_MEMBERS: usize = 256;

/// Comparisons emitted for a live filter's columns. A strict bound is applied through its
/// non-strict comparison, which keeps a superset of the rows, so two comparisons per column
/// serve every bound a TopK or hash-join filter produces.
const TEMPLATE_OPS: [CompareOperator; 2] = [CompareOperator::Lte, CompareOperator::Gte];

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
    let converted = convert_dynamic_filter(expr, file_fields, file_stats, session);
    if tracing::enabled!(tracing::Level::DEBUG) {
        let complete = matches!(
            DynamicFilterTracking::classify(expr),
            DynamicFilterTracking::AllComplete
        );
        let current = expr
            .downcast_ref::<DynamicFilterPhysicalExpr>()
            .and_then(|filter| filter.current().ok())
            .map(|current| current.to_string())
            .unwrap_or_default();
        tracing::debug!(
            complete,
            filter = %expr,
            current = %current,
            converted = converted.as_ref().map(ToString::to_string).unwrap_or_default(),
            "converted dynamic filter"
        );
    }
    converted
}

fn convert_dynamic_filter(
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

    let file_ranges = columns
        .iter()
        .map(|(name, dtype)| {
            file_stats.and_then(|stats| file_range(file_fields, stats, name, dtype))
        })
        .collect();
    let bounds = Arc::new(LiveBounds {
        filter: Arc::clone(expr),
        columns: columns.clone(),
        file_ranges,
        complete: AtomicBool::new(false),
        session: session.clone(),
        cache: ArcSwapOption::empty(),
    });

    if matches!(
        DynamicFilterTracking::classify(expr),
        DynamicFilterTracking::AllComplete
    ) {
        return bounds.selective_static_filter();
    }

    let current = bounds.current_bounds();
    let per_column = columns
        .into_iter()
        .enumerate()
        .filter_map(|(idx, (name, dtype))| {
            // Once the filter has bounds, columns and operators it doesn't bound are skipped.
            // Before that, only the leading column gets the template: a TopK filter over
            // several sort keys is a lexicographic disjunction whose hull only ever bounds the
            // first key, and an aggregate filter bounds one column per accumulator.
            let ops: Vec<_> = if current.is_empty() {
                if idx > 0 {
                    return None;
                }
                TEMPLATE_OPS.to_vec()
            } else {
                TEMPLATE_OPS
                    .into_iter()
                    .filter(|op| {
                        current
                            .iter()
                            .any(|(c, o, _)| *c == idx && non_strict(*o) == *op)
                    })
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
    /// Whether the filter was complete when the bounds were read. Completing a filter does not
    /// change its generation, so an incomplete snapshot is re-read once completion is seen.
    complete: bool,
}

struct LiveBounds {
    /// Always a [`DynamicFilterPhysicalExpr`]; kept type-erased because that type is not `Clone`.
    filter: Arc<dyn PhysicalExpr>,
    columns: Vec<(String, DType)>,
    /// Each column's `(min, max)` in the file, when its statistics are known and numeric.
    file_ranges: Vec<Option<(f64, f64)>>,
    /// Set once the filter is seen complete, after which it can never change again.
    complete: AtomicBool,
    session: VortexSession,
    /// Read on every evaluation of the Vortex filter but replaced only when the DataFusion filter
    /// changes, a few dozen times per query for TopK, so reads must not contend.
    cache: ArcSwapOption<Snapshot>,
}

impl LiveBounds {
    /// Builds static comparisons from the filter's current bounds and membership lists, keeping
    /// only columns whose constraints are estimated to exclude a meaningful share of the file.
    fn selective_static_filter(&self) -> Option<Expression> {
        let current = self
            .filter
            .downcast_ref::<DynamicFilterPhysicalExpr>()?
            .current()
            .ok()?;
        let bounds = self.bounds_of(&current).unwrap_or_default();
        let members = self.members_of(&current).unwrap_or_default();
        let per_column = self
            .columns
            .iter()
            .enumerate()
            .filter_map(|(idx, (name, dtype))| {
                let scalar =
                    |value: &ScalarValue| Scalar::try_new(dtype.clone(), Some(value.clone())).ok();
                let column_bounds: Vec<_> = bounds
                    .iter()
                    .filter(|(c, ..)| *c == idx)
                    .map(|(_, op, value)| Some((*op, scalar(value)?)))
                    .collect::<Option<_>>()?;
                let column_members = match members.iter().find(|(c, _)| *c == idx) {
                    Some((_, values)) => {
                        Some(values.iter().map(scalar).collect::<Option<Vec<_>>>()?)
                    }
                    None => None,
                };
                if column_bounds.is_empty() && column_members.is_none() {
                    return None;
                }
                let range = self.file_ranges[idx]?;
                // A complete filter never changes again, so nulls only need to survive when it
                // can admit them.
                let keep_nulls = dtype.is_nullable() && !rejects_nulls(&current, name);
                column_static_filter(
                    name,
                    dtype,
                    keep_nulls,
                    range,
                    column_bounds,
                    column_members,
                )
            });
        and_collect(per_column)
    }

    fn value(&self, column: usize, op: CompareOperator) -> Option<ScalarValue> {
        find_bound(&self.current_bounds(), column, op)
    }

    /// The bounds of the filter's current generation, re-read only when the generation changes.
    fn current_bounds(&self) -> Arc<[Bound]> {
        let generation = self.filter.snapshot_generation();
        match self.cache.load().as_ref() {
            Some(snapshot)
                if snapshot.generation == generation
                    && (snapshot.complete || !self.is_complete()) =>
            {
                Arc::clone(&snapshot.bounds)
            }
            _ => self.refresh(generation),
        }
    }

    /// Whether the filter is complete. Remembered once seen, as a complete filter never changes.
    fn is_complete(&self) -> bool {
        if self.complete.load(AtomicOrdering::Relaxed) {
            return true;
        }
        let complete = matches!(
            DynamicFilterTracking::classify(&self.filter),
            DynamicFilterTracking::AllComplete
        );
        if complete {
            self.complete.store(true, AtomicOrdering::Relaxed);
        }
        complete
    }

    /// Keeps the bounds of a complete filter only on columns where they are estimated to exclude
    /// a meaningful share of the file, as for a filter that is already complete when the file is
    /// opened. The comparisons on other columns then always pass and stop costing evaluations.
    fn selective_bounds(&self, mut bounds: Vec<Bound>) -> Vec<Bound> {
        let selective: Vec<bool> = self
            .columns
            .iter()
            .zip(&self.file_ranges)
            .enumerate()
            .map(|(idx, ((_, dtype), range))| {
                let Some(range) = *range else {
                    return false;
                };
                let column_bounds = bounds
                    .iter()
                    .filter(|(c, ..)| *c == idx)
                    .map(|(_, op, value)| {
                        Some((
                            *op,
                            Scalar::try_new(dtype.clone(), Some(value.clone())).ok()?,
                        ))
                    })
                    .collect::<Option<Vec<_>>>();
                column_bounds
                    .and_then(|column_bounds| bounded_range(&column_bounds, range))
                    .is_some_and(|bounded| range_fraction(bounded, range) <= MAX_KEPT_FRACTION)
            })
            .collect();
        bounds.retain(|(c, ..)| selective[*c]);
        bounds
    }

    /// Re-reads the bounds for `generation` and caches them, unless a concurrent refresh has
    /// already cached a newer generation.
    fn refresh(&self, generation: u64) -> Arc<[Bound]> {
        // Read completion first: once complete, the bounds read below are final.
        let complete = self.is_complete();
        let bounds = self.extract_bounds();
        let bounds: Arc<[Bound]> = if complete {
            self.selective_bounds(bounds).into()
        } else {
            bounds.into()
        };
        let snapshot = Arc::new(Snapshot {
            generation,
            bounds: Arc::clone(&bounds),
            complete,
        });
        self.cache.rcu(|cached| match cached {
            Some(cached)
                if cached.generation > generation
                    || (cached.generation == generation && cached.complete && !complete) =>
            {
                Some(Arc::clone(cached))
            }
            _ => Some(Arc::clone(&snapshot)),
        });
        bounds
    }

    /// Reads `col <op> literal` bounds from the filter's current predicate.
    fn extract_bounds(&self) -> Vec<Bound> {
        self.filter
            .downcast_ref::<DynamicFilterPhysicalExpr>()
            .and_then(|filter| filter.current().ok())
            .and_then(|current| self.bounds_of(&current))
            .unwrap_or_default()
    }

    /// Per-column value lists that every row satisfying `expr` draws its value from, unless the
    /// column is null, or `None` if no row can satisfy `expr`. Columns without a list are
    /// unconstrained. Lists with more than [`MAX_MEMBERS`] distinct values are dropped.
    fn members_of(&self, expr: &Arc<dyn PhysicalExpr>) -> Option<Vec<Members>> {
        if let Some(literal) = expr.downcast_ref::<df_expr::Literal>() {
            return match literal.value() {
                DFScalarValue::Boolean(Some(false) | None) | DFScalarValue::Null => None,
                _ => Some(vec![]),
            };
        }
        if let Some(case) = expr.downcast_ref::<df_expr::CaseExpr>() {
            // Rows the CASE keeps satisfy one of its results, so a column is constrained only
            // when every reachable branch lists its values.
            return case
                .when_then_expr()
                .iter()
                .map(|(_, then)| then)
                .chain(case.else_expr())
                .map(|branch| self.members_of(branch))
                .reduce(union_members)
                .flatten();
        }
        if let Some(in_list) = expr.downcast_ref::<df_expr::InListExpr>() {
            if in_list.negated() {
                return Some(vec![]);
            }
            let Some(idx) = column_of(in_list.expr()).and_then(|col| self.column_index(col.name()))
            else {
                return Some(vec![]);
            };
            // A hash join's list repeats each build-side key once per build row, so only the
            // distinct values count towards the limit. Deduplicate before converting, as the
            // list can be thousands of entries long.
            let mut distinct = HashSet::new();
            for element in in_list.list() {
                let literal = element
                    .downcast_ref::<df_expr::Literal>()
                    .filter(|literal| !literal.value().is_null());
                // A list we can't convert exactly, or that is too long, constrains nothing.
                let Some(literal) = literal else {
                    return Some(vec![]);
                };
                if distinct.insert(literal.value()) && distinct.len() > MAX_MEMBERS {
                    return Some(vec![]);
                }
            }
            let values = distinct
                .into_iter()
                .map(|value| self.literal_value(idx, value))
                .collect::<Option<Vec<_>>>();
            return Some(values.map_or_else(Vec::new, |values| vec![(idx, values)]));
        }
        let Some(binary) = expr.downcast_ref::<df_expr::BinaryExpr>() else {
            return Some(vec![]);
        };
        match binary.op() {
            DFOperator::And => {
                let mut members = self.members_of(binary.left())?;
                for (idx, values) in self.members_of(binary.right())? {
                    match members.iter_mut().find(|(c, _)| *c == idx) {
                        // Both lists hold, so keep the shorter one.
                        Some((_, existing)) if existing.len() <= values.len() => {}
                        Some((_, existing)) => *existing = values,
                        None => members.push((idx, values)),
                    }
                }
                Some(members)
            }
            DFOperator::Or => {
                if let Some(null_col) = binary
                    .left()
                    .downcast_ref::<df_expr::IsNullExpr>()
                    .and_then(|is_null| column_of(is_null.arg()))
                {
                    let idx = self.column_index(null_col.name());
                    return Some(
                        self.members_of(binary.right())
                            .unwrap_or_default()
                            .into_iter()
                            .filter(|(c, _)| Some(*c) == idx)
                            .collect(),
                    );
                }
                union_members(
                    self.members_of(binary.left()),
                    self.members_of(binary.right()),
                )
            }
            _ => Some(vec![]),
        }
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
        Some((idx, op, self.literal_value(idx, literal.value())?))
    }

    /// Converts a DataFusion literal to the file dtype of tracked column `idx`.
    fn literal_value(&self, idx: usize, literal: &DFScalarValue) -> Option<ScalarValue> {
        let dtype = &self.columns[idx].1;
        let scalar = scalar_from_df(literal, &self.session).ok()?;
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
        scalar.into_value()
    }
}

/// `(column index, values)`: every kept row's value for the column is one of `values`.
type Members = (usize, Vec<ScalarValue>);

/// Lists implied by either side holding: a column is listed only when both sides list it, with
/// the two lists merged. A side no row can satisfy (`None`) contributes nothing.
fn union_members(left: Option<Vec<Members>>, right: Option<Vec<Members>>) -> Option<Vec<Members>> {
    let (left, right) = match (left, right) {
        (Some(left), Some(right)) => (left, right),
        (side, None) | (None, side) => return side,
    };
    Some(
        left.into_iter()
            .filter_map(|(idx, mut values)| {
                let (_, other) = right.iter().find(|(c, _)| *c == idx)?;
                for value in other {
                    if !values.contains(value) {
                        values.push(value.clone());
                    }
                }
                (values.len() <= MAX_MEMBERS).then_some((idx, values))
            })
            .collect(),
    )
}

fn as_f64(scalar: &Scalar) -> Option<f64> {
    scalar.as_primitive_opt()?.as_::<f64>()
}

/// The `(min, max)` of column `name` in the file, when its statistics are known and numeric.
fn file_range(
    file_fields: &StructFields,
    file_stats: &FileStatistics,
    name: &str,
    dtype: &DType,
) -> Option<(f64, f64)> {
    let (stats, _) = file_stats.get(file_fields.find(name)?);
    let stat = |stat| Scalar::try_new(dtype.clone(), stats.get(stat).into_inner()).ok();
    Some((as_f64(&stat(Stat::Min)?)?, as_f64(&stat(Stat::Max)?)?))
}

/// Static comparisons for one column of a complete filter: its bounds, when they are selective
/// against the file's `(min, max)` range, and its membership list, when that list is sparse in
/// the range the bounds keep. With `keep_nulls`, null rows always pass.
fn column_static_filter(
    name: &str,
    dtype: &DType,
    keep_nulls: bool,
    file_range: (f64, f64),
    bounds: Vec<(CompareOperator, Scalar)>,
    members: Option<Vec<Scalar>>,
) -> Option<Expression> {
    let lhs = get_item(name.to_owned(), root());
    // The range the file's rows can lie in after the bounds are applied; the whole file when the
    // bounds aren't selective enough to be worth evaluating.
    let mut range = file_range;
    let mut kept_bounds = None;
    if !bounds.is_empty() {
        let bounded = bounded_range(&bounds, range)?;
        let kept = range_fraction(bounded, range);
        tracing::debug!(column = %name, kept, "complete dynamic filter bounds");
        if kept <= MAX_KEPT_FRACTION {
            range = bounded;
            kept_bounds = Some(bounds);
        }
    }
    // A membership list is only worth its decode of the column when it prunes a meaningful share
    // of the rows the bounds keep, which a list that is dense in that range does not. Assumes an
    // integer column spread uniformly over the range.
    let mut members_list = None;
    if let Some(values) = members
        && dtype.is_int()
    {
        let kept = (values.len() as f64 / (range.1 - range.0 + 1.0)).clamp(0.0, 1.0);
        tracing::debug!(column = %name, kept, members = values.len(), "complete dynamic filter members");
        if kept <= MAX_KEPT_FRACTION {
            members_list = Some(Scalar::list(
                dtype.clone(),
                values,
                Nullability::NonNullable,
            ));
        }
    }
    // Every comparison decodes the column on its own, so the column is referenced once: the list
    // alone when there is one (its values all lie within the bounds), else the bounds as a single
    // `between` where they form a range.
    let comparisons = match (members_list, kept_bounds) {
        (Some(list), _) => in_list(lhs.clone(), lit(list)),
        (None, Some(bounds)) => bounds_expr(&lhs, bounds)?,
        (None, None) => return None,
    };
    Some(if keep_nulls {
        or(is_null(lhs), comparisons)
    } else {
        comparisons
    })
}

/// Whether a top-level conjunct of `current` compares column `name` or tests its membership, so
/// no row with a null `name` can satisfy it. Bounds from an OR or CASE may still admit nulls.
fn rejects_nulls(current: &Arc<dyn PhysicalExpr>, name: &str) -> bool {
    let is_column =
        |expr: &Arc<dyn PhysicalExpr>| column_of(expr).is_some_and(|c| c.name() == name);
    split_conjunction(current).into_iter().any(|expr| {
        if let Some(binary) = expr.downcast_ref::<df_expr::BinaryExpr>() {
            return compare_op(binary.op()).is_some()
                && (is_column(binary.left()) || is_column(binary.right()));
        }
        expr.downcast_ref::<df_expr::InListExpr>()
            .is_some_and(|list| !list.negated() && is_column(list.expr()))
    })
}

/// `lhs` compared against `bounds`, as one `between` when they hold a single lower and a single
/// upper bound.
fn bounds_expr(lhs: &Expression, bounds: Vec<(CompareOperator, Scalar)>) -> Option<Expression> {
    let lower = bounds
        .iter()
        .filter(|(op, _)| matches!(op, CompareOperator::Gt | CompareOperator::Gte))
        .collect::<Vec<_>>();
    let upper = bounds
        .iter()
        .filter(|(op, _)| matches!(op, CompareOperator::Lt | CompareOperator::Lte))
        .collect::<Vec<_>>();
    if let ([(lower_op, lower)], [(upper_op, upper)]) = (lower.as_slice(), upper.as_slice())
        && bounds.len() == 2
    {
        let strictness = |op: &CompareOperator| match op {
            CompareOperator::Gt | CompareOperator::Lt => StrictComparison::Strict,
            _ => StrictComparison::NonStrict,
        };
        return Some(between(
            lhs.clone(),
            lit(lower.clone()),
            lit(upper.clone()),
            BetweenOptions {
                lower_strict: strictness(lower_op),
                upper_strict: strictness(upper_op),
            },
        ));
    }
    and_collect(
        bounds
            .into_iter()
            .map(|(op, value)| binary(op.into(), lhs.clone(), lit(value))),
    )
}

/// Narrows the `[min, max]` range of a column to the part `bounds` keep. Returns `None` for
/// non-numeric bounds.
fn bounded_range(
    bounds: &[(CompareOperator, Scalar)],
    (min, max): (f64, f64),
) -> Option<(f64, f64)> {
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
    Some((lower, upper))
}

/// Estimates the fraction of the `[min, max]` range that `[lower, upper]` covers, assuming values
/// are spread uniformly.
fn range_fraction((lower, upper): (f64, f64), (min, max): (f64, f64)) -> f64 {
    if max <= min {
        return if lower <= upper { 1.0 } else { 0.0 };
    }
    ((upper - lower) / (max - min)).clamp(0.0, 1.0)
}

/// The bound on `column` for comparison `op`, preferring an exact match over a strict bound the
/// non-strict comparison also enforces.
///
/// A strict integer bound is stepped to its equivalent non-strict one (`a < 5` is `a <= 4`). Other
/// strict bounds are applied as they are, which also keeps rows equal to the bound.
fn find_bound(bounds: &[Bound], column: usize, op: CompareOperator) -> Option<ScalarValue> {
    if let Some((_, _, value)) = bounds.iter().find(|(c, o, _)| *c == column && *o == op) {
        return Some(value.clone());
    }
    let (_, strict, value) = bounds
        .iter()
        .find(|(c, o, _)| *c == column && *o != op && non_strict(*o) == op)?;
    Some(step_strict_integer(value, *strict).unwrap_or_else(|| value.clone()))
}

/// The non-strict integer bound equivalent to strict bound `value`, or `None` for non-integers
/// and bounds that would overflow.
fn step_strict_integer(value: &ScalarValue, strict: CompareOperator) -> Option<ScalarValue> {
    let ScalarValue::Primitive(value) = value else {
        return None;
    };
    let up = match strict {
        CompareOperator::Gt => true,
        CompareOperator::Lt => false,
        _ => return None,
    };
    macro_rules! step {
        ($v:expr, $variant:ident) => {
            if up {
                $v.checked_add(1).map(PValue::$variant)
            } else {
                $v.checked_sub(1).map(PValue::$variant)
            }
        };
    }
    let stepped = match *value {
        PValue::U8(v) => step!(v, U8),
        PValue::U16(v) => step!(v, U16),
        PValue::U32(v) => step!(v, U32),
        PValue::U64(v) => step!(v, U64),
        PValue::I8(v) => step!(v, I8),
        PValue::I16(v) => step!(v, I16),
        PValue::I32(v) => step!(v, I32),
        PValue::I64(v) => step!(v, I64),
        _ => None,
    }?;
    Some(ScalarValue::Primitive(stepped))
}

/// The non-strict comparison keeping every row `op` keeps.
fn non_strict(op: CompareOperator) -> CompareOperator {
    match op {
        CompareOperator::Lt => CompareOperator::Lte,
        CompareOperator::Gt => CompareOperator::Gte,
        op => op,
    }
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
    use vortex::scalar_fn::fns::operators::CompareOperator;
    use vortex::session::VortexSession;

    use super::dynamic_filter_to_vortex;
    use super::step_strict_integer;

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

    fn in_list_a(values: &[i32]) -> PhysicalExprRef {
        Arc::new(
            df_expr::InListExpr::try_new(
                col_a(),
                values.iter().map(|v| lit_i32(*v)).collect(),
                false,
                &arrow_schema::Schema::new(vec![arrow_schema::Field::new(
                    "a",
                    DataType::Int32,
                    false,
                )]),
            )
            .expect("valid IN list"),
        )
    }

    /// `CASE a WHEN 0 THEN thens[0] WHEN 1 THEN thens[1] ... ELSE else_expr END`.
    fn case_on_a(
        thens: Vec<PhysicalExprRef>,
        else_expr: Option<PhysicalExprRef>,
    ) -> PhysicalExprRef {
        let when_then = thens
            .into_iter()
            .enumerate()
            .map(|(i, then)| (lit_i32(i32::try_from(i).expect("small index")), then))
            .collect();
        Arc::new(
            df_expr::CaseExpr::try_new(Some(col_a()), when_then, else_expr).expect("valid CASE"),
        )
    }

    fn stats_1_to_10(input: &ArrayRef) -> FileStatistics {
        let stats = StatsSet::from_iter([
            (Stat::Min, Precision::exact(VxScalarValue::from(1i32))),
            (Stat::Max, Precision::exact(VxScalarValue::from(10i32))),
        ]);
        FileStatistics::new_with_dtype(Arc::from([stats]), input.dtype())
    }

    /// A complete join filter's membership list (`col IN (...)`) is applied when it covers a
    /// small share of the file's value range, even where its bounds span the whole range.
    #[rstest]
    // Members 1 and 10 span the range [1, 10] but keep 2 of 10 values: applied.
    #[case(range(1, 10), in_list_a(&[1, 10]), Some([true, false, true]))]
    // Partitioned joins route keys through a CASE; the union of the partition lists applies.
    #[case(
        case_on_a(
            vec![
                binary(range(1, 1), DFOperator::And, in_list_a(&[1])),
                binary(range(5, 5), DFOperator::And, in_list_a(&[5])),
            ],
            Some(lit_bool(false)),
        ),
        lit_bool(true),
        Some([true, true, false]),
    )]
    // A partition without a list leaves the column unconstrained.
    #[case(
        case_on_a(vec![in_list_a(&[1]), lit_bool(true)], None),
        lit_bool(true),
        None,
    )]
    // Six of ten values is not selective enough.
    #[case(lit_bool(true), in_list_a(&[1, 2, 3, 4, 5, 6]), None)]
    // A join's list repeats each key per build row; only distinct values count towards the limit.
    #[case(lit_bool(true), in_list_a(&[1, 5].repeat(200)), Some([true, true, false]))]
    fn complete_filter_membership(
        #[case] left: PhysicalExprRef,
        #[case] right: PhysicalExprRef,
        #[case] expected: Option<[bool; 3]>,
    ) -> VortexResult<()> {
        let session = VortexSession::default();
        let input =
            StructArray::from_fields(&[("a", buffer![1i32, 5, 10].into_array())])?.into_array();
        let file_stats = stats_1_to_10(&input);
        let filter = complete_filter(binary(left, DFOperator::And, right));
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

    /// A completed filter whose top-level conjunct compares the column can drop its nulls; one
    /// whose bounds come from an OR must keep them.
    #[rstest]
    #[case::comparison(false, None)]
    #[case::null_escape(true, Some(true))]
    fn completed_range_nulls(
        #[case] allow_null: bool,
        #[case] expected_null: Option<bool>,
    ) -> VortexResult<()> {
        let session = VortexSession::default();
        let input = StructArray::from_fields(&[(
            "a",
            PrimitiveArray::from_option_iter([Some(1i32), Some(5), Some(10), None]).into_array(),
        )])?
        .into_array();
        let range = binary(col_a(), DFOperator::LtEq, lit_i32(5));
        let current = if allow_null {
            binary(
                Arc::new(df_expr::IsNullExpr::new(col_a())),
                DFOperator::Or,
                range,
            )
        } else {
            range
        };
        let filter = complete_filter(current);
        let file_stats = stats_1_to_10(&input);
        let fields = input.dtype().as_struct_fields_opt().expect("struct input");
        let converted = dynamic_filter_to_vortex(&filter, fields, Some(&file_stats), &session)
            .expect("selective range should be applied");
        assert_arrays_eq!(
            input.apply(&converted)?,
            BoolArray::from_iter([Some(true), Some(true), Some(false), expected_null]),
            &mut session.create_execution_ctx()
        );
        Ok(())
    }

    /// Every comparison decodes the column on its own, so a complete filter references each
    /// column once: a range becomes one `between`, and a membership list replaces the bounds it
    /// lies within.
    #[rstest]
    #[case(range(4, 5), lit_bool(true), [false, true, false])]
    #[case(
        binary(col_a(), DFOperator::Gt, lit_i32(4)),
        binary(col_a(), DFOperator::Lt, lit_i32(6)),
        [false, true, false],
    )]
    #[case(
        binary(col_a(), DFOperator::Gt, lit_i32(4)),
        binary(col_a(), DFOperator::LtEq, lit_i32(5)),
        [false, true, false],
    )]
    #[case(range(1, 10), in_list_a(&[1, 10]), [true, false, true])]
    fn complete_filter_reads_column_once(
        #[case] left: PhysicalExprRef,
        #[case] right: PhysicalExprRef,
        #[case] expected: [bool; 3],
    ) -> VortexResult<()> {
        let session = VortexSession::default();
        let input =
            StructArray::from_fields(&[("a", buffer![1i32, 5, 10].into_array())])?.into_array();
        let file_stats = stats_1_to_10(&input);
        let filter = complete_filter(binary(left, DFOperator::And, right));
        let fields = input.dtype().as_struct_fields_opt().expect("struct input");

        let converted = dynamic_filter_to_vortex(&filter, fields, Some(&file_stats), &session)
            .expect("selective filter should convert");
        assert_eq!(
            converted.to_string().matches("$.a").count(),
            1,
            "{converted}"
        );
        assert_arrays_eq!(
            input.apply(&converted)?,
            BoolArray::from_iter(expected),
            &mut session.create_execution_ctx()
        );
        Ok(())
    }

    /// Before its first update a multi-column filter only tracks the leading column, which is
    /// the only one a TopK or aggregate filter can ever bound.
    #[test]
    fn template_tracks_leading_column_only() -> anyhow::Result<()> {
        let session = VortexSession::default();
        let input = StructArray::from_fields(&[
            ("a", buffer![1i32, 5, 10].into_array()),
            ("b", buffer![7i32, 0, 0].into_array()),
        ])?
        .into_array();
        let dynamic = Arc::new(DynamicFilterPhysicalExpr::new(
            vec![col_a(), col_b()],
            lit_bool(true),
        ));
        let filter: PhysicalExprRef = Arc::clone(&dynamic) as _;
        let fields = input.dtype().as_struct_fields_opt().expect("struct input");
        let converted = dynamic_filter_to_vortex(&filter, fields, None, &session)
            .expect("filter should convert");
        // The leading column's `<=` and `>=` placeholders.
        assert_eq!(converted.to_string().matches("dynamic(").count(), 2);
        assert!(!converted.to_string().contains("$.b"));

        dynamic.update(binary(
            binary(col_a(), DFOperator::Lt, lit_i32(5)),
            DFOperator::Or,
            binary(
                binary(col_a(), DFOperator::Eq, lit_i32(5)),
                DFOperator::And,
                binary(col_b(), DFOperator::Lt, lit_i32(3)),
            ),
        ))?;
        assert_arrays_eq!(
            input.apply(&converted)?,
            BoolArray::from_iter([true, true, false]),
            &mut session.create_execution_ctx()
        );
        Ok(())
    }

    /// A strict integer bound is stepped to its non-strict equivalent; one that would overflow,
    /// or a non-integer bound, is applied as is.
    #[rstest]
    #[case(
        VxScalarValue::from(5i32),
        CompareOperator::Lt,
        Some(VxScalarValue::from(4i32))
    )]
    #[case(
        VxScalarValue::from(5u8),
        CompareOperator::Gt,
        Some(VxScalarValue::from(6u8))
    )]
    #[case(VxScalarValue::from(i32::MIN), CompareOperator::Lt, None)]
    #[case(VxScalarValue::from(u64::MAX), CompareOperator::Gt, None)]
    #[case(VxScalarValue::from(4.5f64), CompareOperator::Lt, None)]
    #[case(VxScalarValue::from(5i32), CompareOperator::Lte, None)]
    fn strict_integer_bounds_step(
        #[case] value: VxScalarValue,
        #[case] op: CompareOperator,
        #[case] expected: Option<VxScalarValue>,
    ) {
        assert_eq!(step_strict_integer(&value, op), expected);
    }

    /// A strict float bound is applied through `<=`, which keeps a superset of the rows.
    #[test]
    fn strict_float_bound_keeps_ties() -> anyhow::Result<()> {
        let session = VortexSession::default();
        let input = StructArray::from_fields(&[("a", buffer![1.0f64, 4.5, 10.0].into_array())])?
            .into_array();
        let dynamic = Arc::new(DynamicFilterPhysicalExpr::new(
            vec![col_a()],
            lit_bool(true),
        ));
        let filter: PhysicalExprRef = Arc::clone(&dynamic) as _;
        let fields = input.dtype().as_struct_fields_opt().expect("struct input");
        let converted = dynamic_filter_to_vortex(&filter, fields, None, &session)
            .expect("filter should convert");
        dynamic.update(binary(
            col_a(),
            DFOperator::Lt,
            Arc::new(df_expr::Literal::new(ScalarValue::Float64(Some(4.5)))),
        ))?;
        assert_arrays_eq!(
            input.apply(&converted)?,
            BoolArray::from_iter([true, true, false]),
            &mut session.create_execution_ctx()
        );
        Ok(())
    }

    /// A live filter that completes mid-scan keeps its bounds only where file statistics predict
    /// they are selective, as a filter already complete at file open would.
    #[rstest]
    // Keeps 1 of the file's 9-wide range: applied.
    #[case(range(4, 5), [false, true, false])]
    // Keeps 8 of 9: dropped, so every row passes.
    #[case(range(2, 10), [true, true, true])]
    fn live_filter_completing_mid_scan_applies_file_stats(
        #[case] current: PhysicalExprRef,
        #[case] expected: [bool; 3],
    ) -> anyhow::Result<()> {
        let session = VortexSession::default();
        let input =
            StructArray::from_fields(&[("a", buffer![1i32, 5, 10].into_array())])?.into_array();
        let file_stats = stats_1_to_10(&input);
        let dynamic = Arc::new(DynamicFilterPhysicalExpr::new(
            vec![col_a()],
            lit_bool(true),
        ));
        let filter: PhysicalExprRef = Arc::clone(&dynamic) as _;
        let fields = input.dtype().as_struct_fields_opt().expect("struct input");
        let converted = dynamic_filter_to_vortex(&filter, fields, Some(&file_stats), &session)
            .expect("live filter should convert");
        let mut ctx = session.create_execution_ctx();

        // Cache the bounds before completion, as a scan would, so completion alone (which
        // keeps the generation) must invalidate them.
        dynamic.update(current)?;
        input
            .clone()
            .apply(&converted)?
            .execute::<BoolArray>(&mut ctx)?;
        dynamic.mark_complete();
        assert_arrays_eq!(
            input.apply(&converted)?,
            BoolArray::from_iter(expected),
            &mut ctx
        );
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
