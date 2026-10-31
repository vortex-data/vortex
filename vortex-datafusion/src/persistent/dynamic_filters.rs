// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Pushes DataFusion dynamic filters, such as the build-side keys of a hash join or the threshold
//! of a TopK operator, into Vortex scans.
//!
//! DataFusion keeps enforcing every dynamic filter, so the scan receives each one as an
//! [`switch`](vortex::scalar_fn::fns::switch::Switch) conjunct: it prunes zones with it and removes rows before it decodes the other
//! columns, and it skips the conjunct when it keeps too many rows. Parts of a filter that Vortex
//! cannot express are dropped, apart from the hash table lookup of a join, which DataFusion
//! evaluates inside the scan.

use std::fmt;
use std::hash::Hash;
use std::hash::Hasher;
use std::sync::Arc;
use std::sync::OnceLock;

use arrow_schema::SchemaRef;
use datafusion_common::Result as DFResult;
use datafusion_common::ScalarValue;
use datafusion_expr::Operator as DFOperator;
use datafusion_physical_expr::DynamicFilterTracking;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_expr::expressions as df_expr;
use datafusion_physical_expr::simplifier::PhysicalExprSimplifier;
use datafusion_physical_expr::split_conjunction;
use datafusion_physical_expr::utils::collect_columns;
use datafusion_physical_expr_common::physical_expr::snapshot_generation;
use datafusion_physical_expr_common::physical_expr::snapshot_physical_expr;
use itertools::Itertools;
use object_store::path::Path;
use parking_lot::RwLock;
use vortex::dtype::DType;
use vortex::dtype::FieldName;
use vortex::dtype::FieldPath;
use vortex::dtype::Nullability;
use vortex::error::VortexResult;
use vortex::error::vortex_err;
use vortex::expr::BoundExpression;
use vortex::expr::Expression;
use vortex::expr::and_collect;
use vortex::expr::dynamic_expression;
use vortex::expr::get_item;
use vortex::expr::lit;
use vortex::expr::pack;
use vortex::expr::root;
use vortex::expr::stats::Stat;
use vortex::expr::switch;
use vortex::expr::transform::replace;
use vortex::file::VortexFile;
use vortex::scalar_fn::ScalarFnVTableExt;
use vortex::scalar_fn::fns::between::Between;
use vortex::scalar_fn::fns::binary::Binary;
use vortex::scalar_fn::fns::dynamic_expression::DynamicExpressionSource;
use vortex::scalar_fn::fns::dynamic_expression::input_field_name;
use vortex::scalar_fn::fns::get_item::GetItem;
use vortex::scalar_fn::fns::list_contains::ListContains;
use vortex::scalar_fn::fns::literal::Literal;
use vortex::scalar_fn::fns::operators::Operator;
use vortex::scalar_fn::fns::switch::Switch;
use vortex::scalar_fn::fns::switch::SwitchState;
use vortex_utils::aliases::dash_map::DashMap;
use vortex_utils::aliases::hash_map::HashMap;

use crate::convert::ExpressionConverter;
use crate::convert::df_predicate::DataFusionPredicate;
use crate::convert::df_predicate::saves_decoding;
use crate::convert::exprs::is_infallible;

/// The largest percentage of rows that the range bounds of a dynamic filter can keep and still be
/// evaluated. DuckDB keeps evaluating the min/max filter of a join up to 90%, but in a Vortex scan
/// a mask that keeps more than half of the rows makes the projection cost more than it saves.
const RANGE_MAX_KEPT_PERCENT: u8 = 50;

/// The largest percentage of rows that a membership test of a dynamic filter, an IN list or a hash
/// table lookup, can keep and still be evaluated. DuckDB uses the same limit for its join Bloom
/// filters.
const MEMBERSHIP_MAX_KEPT_PERCENT: u8 = 50;

/// The largest percentage of rows that a dynamic filter that can still change, such as the
/// threshold of a TopK operator, can keep and still be evaluated.
const DYNAMIC_MAX_KEPT_PERCENT: u8 = 50;

/// Conversions of the dynamic filters of a scan, shared by its openers.
///
/// Every partition of every file reads the same dynamic filters, and converting one can be costly,
/// such as the IN list of a hash join with thousands of keys. A filter is converted once for each
/// file schema and generation.
#[derive(Default)]
pub(crate) struct DynamicFilterConversions {
    filters: DashMap<(usize, SchemaKey), Arc<FileDynamicFilter>>,
    /// The switched conjuncts of each complete filter for each file, as of a generation.
    complete: DashMap<(usize, Path), (u64, Vec<Expression>)>,
}

/// The file a scan reads.
pub(crate) struct ScannedFile<'a> {
    pub(crate) path: &'a Path,
    /// The Arrow schema of the file.
    pub(crate) schema: &'a SchemaRef,
    /// The names of the columns the scan outputs.
    pub(crate) projected: &'a [&'a str],
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
    /// The conjuncts to push into a scan of `file` for the dynamic filter `predicate`. `adapt`
    /// rewrites the filter to the file schema.
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
    ) -> DFResult<Vec<Expression>> {
        let ScannedFile {
            path,
            schema,
            projected,
            file,
        } = *file;
        let filter = self.filter(predicate, schema, adapt, converter)?;
        let key = (Arc::as_ptr(predicate).cast::<()>().addr(), path.clone());
        if !matches!(
            DynamicFilterTracking::classify(&filter.adapted),
            DynamicFilterTracking::AllComplete
        ) {
            return Ok(vec![filter.dynamic_expression()]);
        }

        let (generation, conjuncts) = filter.current();
        if let Some(entry) = self.complete.get(&key)
            && entry.0 == generation
        {
            return Ok(entry.1.clone());
        }
        let switched = conjuncts
            .iter()
            .filter(|(conjunct, _)| saves_decoding(conjunct, projected))
            .filter_map(|(conjunct, state)| {
                let conjunct = drop_always_true(conjunct, file)?;
                Some(Switch.new_expr(state.clone(), [conjunct]))
            })
            .collect::<Vec<_>>();
        Ok(self
            .complete
            .entry(key)
            .or_insert((generation, switched))
            .1
            .clone())
    }

    fn filter(
        &self,
        predicate: &Arc<dyn PhysicalExpr>,
        schema: &SchemaRef,
        adapt: impl FnOnce(Arc<dyn PhysicalExpr>) -> DFResult<Arc<dyn PhysicalExpr>>,
        converter: &Arc<dyn ExpressionConverter>,
    ) -> DFResult<Arc<FileDynamicFilter>> {
        let key = (
            Arc::as_ptr(predicate).cast::<()>().addr(),
            SchemaKey(Arc::clone(schema)),
        );
        if let Some(filter) = self.filters.get(&key) {
            return Ok(Arc::clone(&filter));
        }
        let filter = Arc::new(FileDynamicFilter {
            _predicate: Arc::clone(predicate),
            adapted: adapt(Arc::clone(predicate))?,
            schema: Arc::clone(schema),
            converter: Arc::clone(converter),
            conversion: RwLock::default(),
            dynamic: OnceLock::new(),
        });
        Ok(Arc::clone(
            self.filters.entry(key).or_insert(filter).value(),
        ))
    }
}

/// The Vortex conjuncts of a dynamic filter, each with the state of its switch.
type Conjuncts = Arc<[(Expression, SwitchState)]>;

/// A dynamic filter adapted to a file schema, with its conversion as of a generation.
struct FileDynamicFilter {
    /// The filter as the scan received it, kept so that the address in its key is not reused.
    _predicate: Arc<dyn PhysicalExpr>,
    adapted: Arc<dyn PhysicalExpr>,
    schema: SchemaRef,
    converter: Arc<dyn ExpressionConverter>,
    conversion: RwLock<Option<(u64, Conjuncts)>>,
    /// The switched dynamic expression of the filter, built once so that every file shares its
    /// switch.
    dynamic: OnceLock<Expression>,
}

impl FileDynamicFilter {
    /// The generation and the Vortex conjuncts of the current value of the filter, over the file
    /// columns.
    fn current(&self) -> (u64, Conjuncts) {
        // A generation read before the snapshot is at most older than the snapshot, which only
        // makes the next call convert again.
        let generation = snapshot_generation(&self.adapted);
        if let Some((converted, conjuncts)) = self.conversion.read().as_ref()
            && *converted == generation
        {
            return (generation, Arc::clone(conjuncts));
        }

        // Openers that miss convert concurrently rather than wait on each other, and the first
        // conversion stored wins, so that every file shares its switches.
        let conjuncts: Conjuncts =
            match snapshot_conjuncts(&self.adapted, &self.schema, self.converter.as_ref()) {
                Ok(conjuncts) => conjuncts
                    .into_iter()
                    .map(|(conjunct, max_kept_percent)| {
                        (conjunct, SwitchState::new(max_kept_percent))
                    })
                    .collect(),
                Err(err) => {
                    tracing::debug!(filter = %self.adapted, %err, "dropping dynamic filter");
                    Arc::from([])
                }
            };
        let mut conversion = self.conversion.write();
        if let Some((converted, stored)) = conversion.as_ref()
            && *converted == generation
        {
            return (generation, Arc::clone(stored));
        }
        *conversion = Some((generation, Arc::clone(&conjuncts)));
        (generation, conjuncts)
    }

    /// A switched dynamic expression over the columns of the filter that evaluates its current
    /// value.
    fn dynamic_expression(self: &Arc<Self>) -> Expression {
        self.dynamic
            .get_or_init(|| {
                let columns = collect_columns(&self.adapted)
                    .into_iter()
                    .sorted_by_key(|column| column.index())
                    .map(|column| column.name().to_string())
                    .collect::<Vec<_>>();
                let inputs = columns
                    .iter()
                    .map(|column| get_item(column.as_str(), root()))
                    .collect::<Vec<_>>();
                let scope_projection = pack(
                    columns.iter().enumerate().map(|(idx, column)| {
                        (column.as_str(), get_item(input_field_name(idx), root()))
                    }),
                    Nullability::NonNullable,
                );
                let dynamic = dynamic_expression(
                    Arc::new(DataFusionDynamicFilter {
                        filter: Arc::clone(self),
                        scope_projection,
                    }),
                    inputs,
                );
                switch(dynamic, DYNAMIC_MAX_KEPT_PERCENT)
            })
            .clone()
    }
}

/// The source of a dynamic expression for a DataFusion dynamic filter that can still change.
struct DataFusionDynamicFilter {
    filter: Arc<FileDynamicFilter>,
    /// The file columns of the filter, as a struct over the inputs of the dynamic expression.
    scope_projection: Expression,
}

impl fmt::Debug for DataFusionDynamicFilter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.filter.adapted, f)
    }
}

impl fmt::Display for DataFusionDynamicFilter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.filter.adapted, f)
    }
}

impl DynamicExpressionSource for DataFusionDynamicFilter {
    fn generation(&self) -> u64 {
        snapshot_generation(&self.filter.adapted)
    }

    fn current(&self, scope: &DType) -> VortexResult<BoundExpression> {
        let (_, conjuncts) = self.filter.current();
        let predicate = and_collect(conjuncts.iter().map(|(conjunct, _)| conjunct.clone()))
            .map_or_else(
                || lit(true),
                |predicate| replace(predicate, &root(), self.scope_projection.clone()),
            );
        predicate
            .bind(scope)?
            .optimize_recursive()
            .map_err(|err| vortex_err!("Failed to bind dynamic filter {self}: {err}"))
    }
}

/// The Vortex conjuncts of the current value of `filter`, a filter over `schema`.
///
/// The range bounds of the filter form one optional conjunct, so the scan can merge them into
/// betweens, and each membership test forms another. Conjuncts that Vortex cannot express are
/// dropped, apart from those DataFusion can evaluate on the columns they read.
fn snapshot_conjuncts(
    filter: &Arc<dyn PhysicalExpr>,
    schema: &SchemaRef,
    converter: &dyn ExpressionConverter,
) -> DFResult<Vec<(Expression, u8)>> {
    let snapshot = PhysicalExprSimplifier::new(schema)
        .simplify(snapshot_physical_expr(Arc::clone(filter))?)?;

    let mut ranges = Vec::new();
    let mut memberships = Vec::new();
    for conjunct in split_conjunction(&snapshot) {
        // A partitioned join routes each row by its hash to the filter of its partition. The
        // bounds that hold in every partition hold for the whole filter, and are cheap to check
        // before DataFusion evaluates the routed filter on the rows they keep.
        let parts = match conjunct.downcast_ref::<df_expr::CaseExpr>() {
            Some(case) => {
                let mut parts = partition_bounds(case);
                parts.push(Arc::clone(conjunct));
                parts
            }
            None => vec![Arc::clone(conjunct)],
        };
        for part in parts {
            match converter.try_convert(&part, schema) {
                Ok(Some(converted)) if is_infallible(&converted) => {
                    if is_range_bound(&part) {
                        ranges.push(converted);
                    } else {
                        memberships.push((converted, MEMBERSHIP_MAX_KEPT_PERCENT));
                    }
                }
                Err(err) => {
                    tracing::debug!(%part, %err, "dropping dynamic filter conjunct");
                }
                _ => match DataFusionPredicate::expression(&part, schema) {
                    Some(predicate) => {
                        memberships.push((predicate, MEMBERSHIP_MAX_KEPT_PERCENT));
                    }
                    None => tracing::debug!(%part, "dropping dynamic filter conjunct"),
                },
            }
        }
    }

    Ok(and_collect(ranges)
        .map(|ranges| (ranges, RANGE_MAX_KEPT_PERCENT))
        .into_iter()
        .chain(memberships)
        .collect())
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
    let parts = split_and(conjunct);

    let mut ranges: HashMap<FieldName, (f64, f64)> = HashMap::default();
    for (column, lower, upper) in parts.iter().filter_map(range_comparison) {
        let range = ranges
            .entry(column)
            .or_insert((f64::NEG_INFINITY, f64::INFINITY));
        range.0 = range.0.max(lower);
        range.1 = range.1.min(upper);
    }
    let covered = ranges
        .into_iter()
        .filter(|(column, (lower, upper))| {
            column_min_max(file, column).is_some_and(|(min, max)| *lower <= min && *upper >= max)
        })
        .map(|(column, _)| column)
        .collect::<Vec<_>>();

    and_collect(parts.into_iter().filter(|part| {
        range_comparison(part).is_none_or(|(column, ..)| !covered.contains(&column))
            && !holds_every_value(part, file)
    }))
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

/// The value of a numeric literal.
fn numeric_literal(expr: &Expression) -> Option<f64> {
    let value = expr.as_opt::<Literal>()?;
    // Viewing a scalar of another type as a primitive builds an error, which is costly.
    if !value.dtype().is_primitive() {
        return None;
    }
    value.as_primitive_opt()?.as_opt::<f64>().flatten()
}

/// The column and the numeric range that `expr` keeps, if it compares a column with a number.
fn range_comparison(expr: &Expression) -> Option<(FieldName, f64, f64)> {
    if expr.is::<Between>() {
        return Some((
            column_name(expr.child(0))?,
            numeric_literal(expr.child(1))?,
            numeric_literal(expr.child(2))?,
        ));
    }
    let operator = *expr.as_opt::<Binary>()?;
    let column = column_name(expr.child(0))?;
    let value = numeric_literal(expr.child(1))?;
    let (lower, upper) = match operator {
        Operator::Eq => (value, value),
        Operator::Gt | Operator::Gte => (value, f64::INFINITY),
        Operator::Lt | Operator::Lte => (f64::NEG_INFINITY, value),
        _ => return None,
    };
    Some((column, lower, upper))
}

/// The minimum and maximum of a primitive column in the statistics of `file`.
fn column_min_max(file: &VortexFile, column: &FieldName) -> Option<(f64, f64)> {
    let (stats, dtype) = file
        .file_stats()?
        .get_by_path(&FieldPath::from_name(column.clone()))?;
    if !dtype.is_primitive() {
        return None;
    }
    let stat = |stat: Stat| {
        stats
            .get(&stat.aggregate_fn()?)
            .as_exact()?
            .as_primitive_opt()?
            .as_opt::<f64>()
            .flatten()
    };
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

/// Whether `expr` compares an expression with a literal by order or equality.
fn is_range_bound(expr: &Arc<dyn PhysicalExpr>) -> bool {
    expr.downcast_ref::<df_expr::BinaryExpr>()
        .is_some_and(|binary| {
            matches!(
                binary.op(),
                DFOperator::Eq
                    | DFOperator::Lt
                    | DFOperator::LtEq
                    | DFOperator::Gt
                    | DFOperator::GtEq
            ) && binary.right().downcast_ref::<df_expr::Literal>().is_some()
        })
}

/// The bounds of each key that every partition of a partitioned join filter implies, as
/// `key >= min AND key <= max` over the smallest minimum and the largest maximum.
///
/// The filter is `CASE hash(keys) % n WHEN i THEN filter_i ... ELSE false END`, where `filter_i`
/// holds the bounds of partition `i`. A partition without rows has no branch or a `false` branch,
/// so it bounds nothing. A filter whose default is not `false`, as when a build partition was
/// canceled, implies no bounds.
fn partition_bounds(case: &df_expr::CaseExpr) -> Vec<Arc<dyn PhysicalExpr>> {
    let else_is_false = case.else_expr().is_none_or(is_false);
    if !else_is_false {
        return Vec::new();
    }

    // The bounds of each key so far.
    let mut hull: Option<Vec<KeyBounds>> = None;
    for (_, then) in case.when_then_expr() {
        if is_false(then) {
            continue;
        }
        let bounds = branch_bounds(then);
        hull = Some(match hull {
            None => bounds,
            Some(hull) => hull
                .into_iter()
                .filter_map(|(key, min, max)| {
                    let (_, other_min, other_max) =
                        bounds.iter().find(|(other, ..)| other == &key)?;
                    let min = match other_min.partial_cmp(&min)? {
                        std::cmp::Ordering::Less => other_min.clone(),
                        _ => min,
                    };
                    let max = match other_max.partial_cmp(&max)? {
                        std::cmp::Ordering::Greater => other_max.clone(),
                        _ => max,
                    };
                    Some((key, min, max))
                })
                .collect(),
        });
    }

    hull.unwrap_or_default()
        .into_iter()
        .flat_map(|(key, min, max)| {
            [
                Arc::new(df_expr::BinaryExpr::new(
                    Arc::clone(&key),
                    DFOperator::GtEq,
                    df_expr::lit(min),
                )) as Arc<dyn PhysicalExpr>,
                Arc::new(df_expr::BinaryExpr::new(
                    key,
                    DFOperator::LtEq,
                    df_expr::lit(max),
                )),
            ]
        })
        .collect()
}

/// A join key with its minimum and maximum.
type KeyBounds = (Arc<dyn PhysicalExpr>, ScalarValue, ScalarValue);

/// The `key >= min AND key <= max` bounds in the conjuncts of a partition filter.
fn branch_bounds(filter: &Arc<dyn PhysicalExpr>) -> Vec<KeyBounds> {
    let comparisons = split_conjunction(filter)
        .into_iter()
        .filter_map(|conjunct| {
            let binary = conjunct.downcast_ref::<df_expr::BinaryExpr>()?;
            let value = binary.right().downcast_ref::<df_expr::Literal>()?.value();
            (!value.is_null()).then(|| (binary.left(), *binary.op(), value))
        })
        .collect::<Vec<_>>();
    comparisons
        .iter()
        .filter(|(_, op, _)| *op == DFOperator::GtEq)
        .filter_map(|(key, _, min)| {
            let (_, _, max) = comparisons
                .iter()
                .find(|(other, op, _)| *op == DFOperator::LtEq && other == key)?;
            Some((Arc::clone(key), (*min).clone(), (*max).clone()))
        })
        .collect()
}

fn is_false(expr: &Arc<dyn PhysicalExpr>) -> bool {
    expr.downcast_ref::<df_expr::Literal>()
        .is_some_and(|literal| literal.value() == &ScalarValue::Boolean(Some(false)))
}

#[cfg(test)]
mod tests {
    use arrow_schema::DataType;
    use arrow_schema::Field;
    use arrow_schema::Schema;
    use rstest::rstest;
    use vortex::VortexSessionDefault;
    use vortex::array::IntoArray;
    use vortex::array::arrays::PrimitiveArray;
    use vortex::array::arrays::StructArray;
    use vortex::buffer::ByteBufferMut;
    use vortex::dtype::PType;
    use vortex::expr::eq;
    use vortex::expr::gt_eq;
    use vortex::expr::in_list_of;
    use vortex::expr::lt;
    use vortex::expr::lt_eq;
    use vortex::file::OpenOptionsSessionExt;
    use vortex::file::WriteOptionsSessionExt;
    use vortex::scalar::Scalar;
    use vortex::session::VortexSession;

    use super::*;
    use crate::convert::DefaultExpressionConverter;

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
        Some(vortex::expr::and(gt_eq(col("a"), lit(1i32)), lt_eq(col("a"), lit(99i32))))
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

    fn column(name: &str, index: usize) -> Arc<dyn PhysicalExpr> {
        Arc::new(df_expr::Column::new(name, index))
    }

    fn compare(lhs: Arc<dyn PhysicalExpr>, op: DFOperator, value: i32) -> Arc<dyn PhysicalExpr> {
        Arc::new(df_expr::BinaryExpr::new(lhs, op, df_expr::lit(value)))
    }

    fn and(lhs: Arc<dyn PhysicalExpr>, rhs: Arc<dyn PhysicalExpr>) -> Arc<dyn PhysicalExpr> {
        Arc::new(df_expr::BinaryExpr::new(lhs, DFOperator::And, rhs))
    }

    fn bounds(min: i32, max: i32) -> Arc<dyn PhysicalExpr> {
        and(
            compare(column("a", 0), DFOperator::GtEq, min),
            compare(column("a", 0), DFOperator::LtEq, max),
        )
    }

    fn partitioned(
        branches: Vec<Arc<dyn PhysicalExpr>>,
        default: bool,
    ) -> DFResult<df_expr::CaseExpr> {
        let routing = Arc::new(df_expr::BinaryExpr::new(
            column("a", 0),
            DFOperator::Modulo,
            df_expr::lit(branches.len() as u64),
        ));
        df_expr::CaseExpr::try_new(
            Some(routing),
            branches
                .into_iter()
                .enumerate()
                .map(|(idx, then)| (df_expr::lit(idx as u64), then))
                .collect(),
            Some(df_expr::lit(default)),
        )
    }

    #[test]
    fn partition_bounds_cover_every_partition() -> DFResult<()> {
        let case = partitioned(
            vec![bounds(5, 10), df_expr::lit(false), bounds(1, 7)],
            false,
        )?;
        let hull = partition_bounds(&case);
        let expected: Vec<Arc<dyn PhysicalExpr>> = vec![
            compare(column("a", 0), DFOperator::GtEq, 1),
            compare(column("a", 0), DFOperator::LtEq, 10),
        ];
        assert_eq!(hull, expected);
        Ok(())
    }

    #[test]
    fn partition_bounds_need_a_false_default() -> DFResult<()> {
        let case = partitioned(vec![bounds(5, 10), bounds(1, 7)], true)?;
        assert!(partition_bounds(&case).is_empty());
        Ok(())
    }

    #[test]
    fn partition_bounds_need_bounds_in_every_partition() -> DFResult<()> {
        let case = partitioned(vec![bounds(5, 10), df_expr::lit(true)], false)?;
        assert!(partition_bounds(&case).is_empty());
        Ok(())
    }

    #[test]
    fn snapshots_split_into_range_bounds_and_memberships() -> DFResult<()> {
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));
        // `IS NOT DISTINCT FROM` stands in for a hash table lookup: Vortex cannot express it.
        let membership: Arc<dyn PhysicalExpr> = Arc::new(df_expr::BinaryExpr::new(
            column("a", 0),
            DFOperator::IsNotDistinctFrom,
            df_expr::lit(3i32),
        ));
        let filter = and(bounds(1, 7), membership);
        let converter = DefaultExpressionConverter::default();

        let conjuncts = snapshot_conjuncts(&filter, &schema, &converter)?;
        assert_eq!(conjuncts.len(), 2);
        let (ranges, ranges_percent) = &conjuncts[0];
        assert_eq!(
            ranges,
            &vortex::expr::and(
                gt_eq(get_item("a", root()), lit(1i32)),
                lt_eq(get_item("a", root()), lit(7i32))
            )
        );
        assert_eq!(*ranges_percent, RANGE_MAX_KEPT_PERCENT);
        assert!(conjuncts[1].0.is::<DataFusionPredicate>());
        assert!(
            !conjuncts
                .iter()
                .any(|(conjunct, _)| conjunct.is::<Switch>())
        );
        Ok(())
    }
}
