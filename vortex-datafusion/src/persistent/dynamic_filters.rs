// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Pushes DataFusion dynamic filters into the scans of the files of a table.
//!
//! The conversion of a filter is shared with the Scan API source, through
//! [`DynamicFilter`]. A complete filter is pushed as its current value, without the parts that
//! the statistics of the file show or estimate to keep almost every row. The statistics can also
//! show that the filter keeps no row, and then the scan skips the file. A small file receives
//! only the parts that are estimated to keep few rows, because evaluating a filter costs more
//! than it saves on a small scan.

use std::hash::Hash;
use std::hash::Hasher;
use std::sync::Arc;

use arrow_schema::SchemaRef;
use datafusion_common::Result as DFResult;
use datafusion_physical_expr::DynamicFilterTracking;
use datafusion_physical_expr::PhysicalExpr;
use object_store::path::Path;
use vortex::dtype::DType;
use vortex::dtype::FieldName;
use vortex::dtype::FieldPath;
use vortex::expr::Expression;
use vortex::expr::and_collect;
use vortex::expr::stats::Stat;
use vortex::file::VortexFile;
use vortex::scalar::Scalar;
use vortex::scalar_fn::ScalarFnVTableExt;
use vortex::scalar_fn::fns::binary::Binary;
use vortex::scalar_fn::fns::get_item::GetItem;
use vortex::scalar_fn::fns::list_contains::ListContains;
use vortex::scalar_fn::fns::literal::Literal;
use vortex::scalar_fn::fns::operators::Operator;
use vortex::scalar_fn::fns::switch::Switch;
use vortex_utils::aliases::dash_map::DashMap;
use vortex_utils::aliases::dash_map::Entry;
use vortex_utils::aliases::hash_map::HashMap;

use crate::convert::ExpressionConverter;
use crate::convert::df_predicate::saves_decoding;
use crate::v2::dynamic_filters::DynamicFilter;

/// The fewest rows a file must have for its scan to receive every complete dynamic filter that can
/// pay off. The scan of a smaller file only receives bounds that keep few of its rows.
///
/// A complete filter, such as the keys of a hash join, costs a conversion, a proof over the
/// statistics of the file, filter plans, and a filtered scan that reads the filter columns before
/// the others. DataFusion enforces the filter again in the join, so the scan only saves the
/// decoding of the rows the filter removes. In TPC-DS at SF1 and SF10, scans of the smaller
/// tables, up to the 720,000 rows of `web_sales` at SF1, ran slower with membership tests and with
/// bounds that keep many rows. Bounds that keep few rows of a small dimension table, such as the
/// weeks of `date_dim` in query 59, made the joins above the scan much faster.
const MIN_FILE_ROWS: u64 = 1 << 20;

/// The largest estimated fraction of its rows that a conjunct can keep and still go into the scan
/// of a file with fewer than [`MIN_FILE_ROWS`] rows.
const SMALL_FILE_MAX_KEPT: f64 = 0.1;

/// The estimated fraction of the rows of a file from which a bound is dropped. Such a bound costs
/// an evaluation and saves the decoding of few rows.
const DROPPED_MIN_KEPT: f64 = 0.9;

/// Conversions of the dynamic filters of a scan, shared by its openers.
///
/// Every partition of every file reads the same dynamic filters, and converting one can be costly,
/// such as the IN list of a hash join with thousands of keys. A filter is converted once for each
/// file schema and generation.
#[derive(Default)]
pub(crate) struct DynamicFilterConversions {
    /// Each filter adapted to a file schema, with the filter as the scan received it, kept so that
    /// the address in its key is not reused.
    filters: DashMap<(usize, SchemaKey), AdaptedFilter>,
    /// The switched conjuncts of each complete filter for each file, as of a generation, or `None`
    /// if the statistics of the file show that the filter keeps no row.
    complete: DashMap<(usize, Path), FileConjuncts>,
}

/// A filter as the scan received it, and its conversion for one file schema.
type AdaptedFilter = (Arc<dyn PhysicalExpr>, Arc<DynamicFilter>);

/// The generation of a filter, and its conjuncts for one file or `None` if it keeps no row.
type FileConjuncts = (u64, Option<Vec<Expression>>);

/// The file a scan reads.
pub(crate) struct ScannedFile<'a> {
    pub(crate) path: &'a Path,
    /// The Arrow schema of the file.
    pub(crate) schema: &'a SchemaRef,
    /// The names of the columns the scan outputs that no filter reads.
    pub(crate) projected: &'a [&'a str],
    /// Whether the scan evaluates other filters that Vortex converts.
    pub(crate) filtered: bool,
    pub(crate) file: &'a VortexFile,
}

/// A file schema compared by value, as files that open together hold separate copies of equal
/// schemas.
#[derive(Clone)]
struct SchemaKey(SchemaRef);

impl PartialEq for SchemaKey {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0) || self.0 == other.0
    }
}

impl Eq for SchemaKey {}

impl Hash for SchemaKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.fields().hash(state);
    }
}

impl DynamicFilterConversions {
    /// The conjuncts to push into a scan of `file` for the dynamic filter `predicate`, or `None`
    /// if the filter keeps no row of the file. `adapt` rewrites the filter to the file schema.
    ///
    /// A complete filter, such as the keys of a hash join whose build side has finished, does not
    /// change again, so its current value is pushed. A filter that can still change is pushed as a
    /// dynamic expression, which the scan reads again each time the filter changes.
    ///
    /// Every file of the scan with the same schema gets the same switches, so the files share the
    /// measurements of each conjunct and the decision to skip it. Every partition of a file gets
    /// the same conjuncts, so the partitions share the filter plans of the file.
    pub(crate) fn conjuncts(
        &self,
        predicate: &Arc<dyn PhysicalExpr>,
        file: &ScannedFile<'_>,
        adapt: impl FnOnce(Arc<dyn PhysicalExpr>) -> DFResult<Arc<dyn PhysicalExpr>>,
        converter: &Arc<dyn ExpressionConverter>,
    ) -> DFResult<Option<Vec<Expression>>> {
        let ScannedFile {
            path,
            schema,
            projected,
            filtered,
            file,
        } = *file;
        let filter = self.filter(predicate, schema, adapt, converter)?;
        let key = (Arc::as_ptr(predicate).cast::<()>().addr(), path.clone());
        if !matches!(
            DynamicFilterTracking::classify(filter.expression()),
            DynamicFilterTracking::AllComplete
        ) {
            // The scan of a small file only receives bounds on columns whose statistics estimate
            // the rows they keep, as for a complete filter. Such a bound is cheap to evaluate, and
            // the bounds on a small table on the build side of another join narrow the filter of
            // that join, as with the weeks of `date_dim` in TPC-DS query 59.
            return Ok(Some(if file.row_count() >= MIN_FILE_ROWS {
                filter.dynamic_expressions()
            } else {
                filter
                    .bound_expressions()
                    .into_iter()
                    .filter(|(column, _)| column_min_max(file, column).is_some())
                    .map(|(_, bound)| bound)
                    .collect()
            }));
        }

        let (generation, conjuncts) = filter.current();
        if let Some(entry) = self.complete.get(&key)
            && entry.0 == generation
        {
            return Ok(entry.1.clone());
        }

        // The partitions of a file open it at the same time. One of them reads the statistics of
        // the file while the others wait for its conjuncts.
        let entry = match self.complete.entry(key) {
            Entry::Occupied(entry) if entry.get().0 == generation => {
                return Ok(entry.get().1.clone());
            }
            entry => entry,
        };
        let switched = if conjuncts
            .iter()
            .any(|(conjunct, _)| keeps_no_row(conjunct, file))
        {
            None
        } else {
            Some(
                conjuncts
                    .iter()
                    .filter(|(conjunct, _)| saves_decoding(conjunct, projected, filtered))
                    .filter_map(|(conjunct, state)| {
                        let conjunct = drop_always_true(conjunct, file)?;
                        pays_off(&conjunct, file)
                            .then(|| Switch.new_expr(state.clone(), [conjunct]))
                    })
                    .collect(),
            )
        };
        Ok(entry.insert((generation, switched)).1.clone())
    }

    fn filter(
        &self,
        predicate: &Arc<dyn PhysicalExpr>,
        schema: &SchemaRef,
        adapt: impl FnOnce(Arc<dyn PhysicalExpr>) -> DFResult<Arc<dyn PhysicalExpr>>,
        converter: &Arc<dyn ExpressionConverter>,
    ) -> DFResult<Arc<DynamicFilter>> {
        let key = (
            Arc::as_ptr(predicate).cast::<()>().addr(),
            SchemaKey(Arc::clone(schema)),
        );
        if let Some(entry) = self.filters.get(&key) {
            return Ok(Arc::clone(&entry.1));
        }

        let filter = DynamicFilter::new(
            adapt(Arc::clone(predicate))?,
            Arc::clone(schema),
            Arc::clone(converter),
        );
        Ok(Arc::clone(
            &self
                .filters
                .entry(key)
                .or_insert((Arc::clone(predicate), filter))
                .1,
        ))
    }
}

/// Whether the scan of `file` gains from evaluating `conjunct`.
fn pays_off(conjunct: &Expression, file: &VortexFile) -> bool {
    file.row_count() >= MIN_FILE_ROWS
        || estimated_kept_bounds(conjunct, file).is_some_and(|kept| kept <= SMALL_FILE_MAX_KEPT)
}

/// Whether the statistics of `file` show that `conjunct` keeps none of its rows.
fn keeps_no_row(conjunct: &Expression, file: &VortexFile) -> bool {
    file.can_prune(conjunct).unwrap_or_else(|err| {
        tracing::debug!(%conjunct, %err, "cannot prune the file with a dynamic filter");
        false
    })
}

/// `conjunct` without the parts that the statistics of `file` show keep every non-null row, or
/// `None` if no part remains.
///
/// A hash join publishes the range of its build-side keys, which often covers a whole probe-side
/// column, such as a dimension key that every fact row references. The scan would only pause
/// such a part after evaluating it. Parts that the statistics merely predict to keep most rows
/// are left to the scan, which measures them: values rarely spread evenly between the minimum and
/// the maximum of a column.
fn drop_always_true(conjunct: &Expression, file: &VortexFile) -> Option<Expression> {
    and_collect(
        split_and(conjunct)
            .into_iter()
            .filter(|part| !keeps_every_row(part, file)),
    )
}

/// Whether the statistics of `file` show that `part` keeps every non-null row, or estimate that it
/// keeps at least [`DROPPED_MIN_KEPT`] of the rows.
fn keeps_every_row(part: &Expression, file: &VortexFile) -> bool {
    if holds_every_value(part, file) {
        return true;
    }
    if estimated_kept(part, file).is_some_and(|kept| kept >= DROPPED_MIN_KEPT) {
        return true;
    }

    file.is_satisfied_by_non_null_rows(part)
        .unwrap_or_else(|err| {
            tracing::debug!(%part, %err, "cannot prove a dynamic filter part with file statistics");
            false
        })
}

/// The fraction of the rows of `file` that a comparison of a column with a number keeps, if the
/// values of the column spread evenly between its minimum and maximum.
fn estimated_kept(part: &Expression, file: &VortexFile) -> Option<f64> {
    let operator = *part.as_opt::<Binary>()?;
    let column = column_name(part.child(0))?;
    let value = numeric_literal(part.child(1))?;
    let (min, max) = column_min_max(file, &column)?;
    if max <= min {
        return None;
    }

    let below = ((value - min) / (max - min)).clamp(0.0, 1.0);
    match operator {
        Operator::Gt | Operator::Gte => Some(1.0 - below),
        Operator::Lt | Operator::Lte => Some(below),
        _ => None,
    }
}

/// The fraction of the rows of `file` that a conjunction of comparisons of columns with numbers
/// keeps, if the values of each column spread evenly between its minimum and maximum and the
/// columns are independent.
fn estimated_kept_bounds(conjunct: &Expression, file: &VortexFile) -> Option<f64> {
    let mut bounds = HashMap::<FieldName, (f64, f64)>::default();
    for part in split_and(conjunct) {
        let operator = *part.as_opt::<Binary>()?;
        let column = column_name(part.child(0))?;
        let kept = estimated_kept(&part, file)?;
        let (lower, upper) = bounds.entry(column).or_insert((0.0, 1.0));
        match operator {
            Operator::Gt | Operator::Gte => *lower = lower.max(1.0 - kept),
            Operator::Lt | Operator::Lte => *upper = upper.min(kept),
            _ => return None,
        }
    }

    Some(
        bounds
            .values()
            .map(|(lower, upper)| (upper - lower).max(0.0))
            .product(),
    )
}

/// The value of a numeric literal.
fn numeric_literal(expr: &Expression) -> Option<f64> {
    number(expr.as_opt::<Literal>()?)
}

/// The value of a number, or the stored number of a value of an extension type, such as the days
/// of a date.
fn number(value: &Scalar) -> Option<f64> {
    let storage;
    let value = match value.as_extension_opt() {
        Some(value) => {
            storage = value.to_storage_scalar();
            &storage
        }
        None => value,
    };

    // Viewing a scalar of another type as a primitive builds an error, which is costly.
    if !value.dtype().is_primitive() {
        return None;
    }
    value.as_primitive_opt()?.as_opt::<f64>().flatten()
}

/// The conjuncts of `expr`.
fn split_and(expr: &Expression) -> Vec<Expression> {
    match expr.as_opt::<Binary>() {
        Some(Operator::And) => {
            let mut parts = split_and(expr.child(0));
            parts.extend(split_and(expr.child(1)));
            parts
        }
        _ => vec![expr.clone()],
    }
}

/// The column that `expr` reads, if it reads one top-level column.
fn column_name(expr: &Expression) -> Option<FieldName> {
    let field = expr.as_opt::<GetItem>()?;
    expr.child(0).is_root().then(|| field.clone())
}

/// The minimum and maximum of a primitive column in the statistics of `file`.
fn column_min_max(file: &VortexFile, column: &FieldName) -> Option<(f64, f64)> {
    let (stats, dtype) = file
        .file_stats()?
        .get_by_path(&FieldPath::from_name(column.clone()))?;
    let numeric = dtype.is_primitive()
        || dtype
            .as_extension_opt()
            .is_some_and(|ext| ext.storage_dtype().is_primitive());
    if !numeric {
        return None;
    }
    let stat = |stat: Stat| number(&stats.get(&stat.aggregate_fn()?).as_exact()?);
    Some((stat(Stat::Min)?, stat(Stat::Max)?))
}

/// Whether `expr` is an IN list on an integer column that holds every integer between the
/// minimum and maximum of the column, such as the keys of a hash join on a dense surrogate key.
fn holds_every_value(expr: &Expression, file: &VortexFile) -> bool {
    if !expr.is::<ListContains>() {
        return false;
    }
    let Some(column) = column_name(expr.child(1)) else {
        return false;
    };
    let Some((min, max)) = column_min_max(file, &column) else {
        return false;
    };
    let Some(elements) = expr
        .child(0)
        .as_opt::<Literal>()
        .and_then(|list| list.as_list_opt()?.elements())
    else {
        return false;
    };
    let mut values = elements
        .iter()
        .filter(|element| matches!(element.dtype(), DType::Primitive(ptype, _) if ptype.is_int()))
        .filter_map(|element| element.as_primitive_opt()?.as_opt::<i64>().flatten())
        .filter(|value| (min..=max).contains(&(*value as f64)))
        .collect::<Vec<_>>();
    values.sort_unstable();
    values.dedup();
    #[expect(clippy::cast_precision_loss, reason = "a count of integers")]
    let holds_every_value = values.len() as f64 >= max - min + 1.0;
    holds_every_value
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex::VortexSessionDefault;
    use vortex::array::IntoArray;
    use vortex::array::arrays::PrimitiveArray;
    use vortex::array::arrays::StructArray;
    use vortex::buffer::ByteBufferMut;
    use vortex::dtype::Nullability;
    use vortex::dtype::PType;
    use vortex::error::VortexResult;
    use vortex::expr::eq;
    use vortex::expr::get_item;
    use vortex::expr::gt_eq;
    use vortex::expr::in_list_of;
    use vortex::expr::lit;
    use vortex::expr::lt;
    use vortex::expr::lt_eq;
    use vortex::expr::root;
    use vortex::file::OpenOptionsSessionExt;
    use vortex::file::WriteOptionsSessionExt;
    use vortex::session::VortexSession;

    use super::*;

    fn col(name: &str) -> Expression {
        get_item(name, root())
    }

    fn in_list(value: Expression, elements: Vec<Scalar>) -> Expression {
        in_list_of(
            value,
            DType::Primitive(PType::I32, Nullability::NonNullable),
            elements,
        )
        .expect("elements are i32")
    }

    /// A file whose columns `a` and `b` hold 0 to 99.
    async fn file() -> VortexResult<VortexFile> {
        let session = VortexSession::default();
        let values = StructArray::from_fields(&[
            ("a", PrimitiveArray::from_iter(0i32..100).into_array()),
            ("b", PrimitiveArray::from_iter(0i32..100).into_array()),
        ])?;
        let mut buf = ByteBufferMut::empty();
        session
            .write_options()
            .write(&mut buf, values.into_array().to_array_stream())
            .await?;
        session.open_options().open_buffer(buf)
    }

    #[rstest]
    #[case::whole_column(
        vortex::expr::and(gt_eq(col("a"), lit(0i32)), lt_eq(col("a"), lit(99i32))),
        None
    )]
    #[case::wider_than_the_column(
        vortex::expr::and(gt_eq(col("a"), lit(-5i32)), lt_eq(col("a"), lit(200i32))),
        None
    )]
    #[case::most_of_the_column(
        vortex::expr::and(gt_eq(col("a"), lit(1i32)), lt_eq(col("a"), lit(99i32))),
        None
    )]
    #[case::half_of_the_column(
        vortex::expr::and(gt_eq(col("a"), lit(50i32)), lt_eq(col("a"), lit(99i32))),
        Some(gt_eq(col("a"), lit(50i32)))
    )]
    #[case::strict_bounds_around_the_column(
        vortex::expr::and(gt_eq(col("a"), lit(-1i32)), lt(col("a"), lit(100i32))),
        None
    )]
    #[case::one_selective_column(
        vortex::expr::and(gt_eq(col("a"), lit(0i32)), lt(col("b"), lit(5i32))),
        Some(lt(col("b"), lit(5i32)))
    )]
    #[case::literal_on_the_left(lt(lit(1i32), col("a")), Some(lt(lit(1i32), col("a"))))]
    #[case::other_comparison(eq(col("a"), col("b")), Some(eq(col("a"), col("b"))))]
    #[case::every_value_in_list(
        in_list(col("a"), (0i32..100).rev().filter(|v| v % 3 != 1).chain((0..100).filter(|v| v % 3 == 1)).map(Scalar::from).collect()),
        None
    )]
    #[case::dense_in_list(
        in_list(col("a"), (0i32..95).filter(|v| v % 19 != 7).map(Scalar::from).collect()),
        Some(in_list(col("a"), (0i32..95).filter(|v| v % 19 != 7).map(Scalar::from).collect()))
    )]
    #[case::sparse_in_list(
        in_list(col("a"), (0i32..10).map(|v| Scalar::from(v * 7)).collect()),
        Some(in_list(col("a"), (0i32..10).map(|v| Scalar::from(v * 7)).collect()))
    )]
    #[tokio::test]
    async fn file_statistics_drop_parts_that_keep_every_row(
        #[case] conjunct: Expression,
        #[case] expected: Option<Expression>,
    ) -> VortexResult<()> {
        assert_eq!(drop_always_true(&conjunct, &file().await?), expected);
        Ok(())
    }

    #[rstest]
    #[case::few_rows(
        vortex::expr::and(gt_eq(col("a"), lit(10i32)), lt_eq(col("a"), lit(14i32))),
        true
    )]
    #[case::many_rows(lt(col("b"), lit(50i32)), false)]
    #[case::two_columns(
        vortex::expr::and(lt(col("a"), lit(25i32)), lt(col("b"), lit(25i32))),
        true
    )]
    #[case::no_estimate(
        in_list(col("a"), (0i32..3).map(Scalar::from).collect()),
        false
    )]
    #[tokio::test]
    async fn small_files_only_receive_selective_bounds(
        #[case] conjunct: Expression,
        #[case] expected: bool,
    ) -> VortexResult<()> {
        assert_eq!(pays_off(&conjunct, &file().await?), expected);
        Ok(())
    }

    #[rstest]
    #[case::below_the_column(lt(col("a"), lit(0i32)), true)]
    #[case::inside_the_column(lt(col("a"), lit(5i32)), false)]
    #[tokio::test]
    async fn file_statistics_prune_the_file(
        #[case] conjunct: Expression,
        #[case] expected: bool,
    ) -> VortexResult<()> {
        assert_eq!(keeps_no_row(&conjunct, &file().await?), expected);
        Ok(())
    }
}
