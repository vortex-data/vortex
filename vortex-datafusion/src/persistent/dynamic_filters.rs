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
use vortex::expr::switch;
use vortex::expr::transform::replace;
use vortex::scalar_fn::fns::dynamic_expression::DynamicExpressionSource;
use vortex::scalar_fn::fns::dynamic_expression::input_field_name;
use vortex_utils::aliases::dash_map::DashMap;

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
    /// The switched dynamic expression of each filter that can still change, for each file.
    dynamic: DashMap<(usize, Path), Expression>,
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
    /// The conjuncts to push into a scan of the file at `path`, with `schema`, for the dynamic
    /// filter `predicate`. `adapt` rewrites the filter to the file schema.
    ///
    /// A complete filter, such as the keys of a hash join whose build side has finished, does not
    /// change again, so its current value is pushed. A filter that can still change is pushed as a
    /// dynamic expression, which the scan reads again each time the filter changes.
    ///
    /// Every partition of a file gets the same conjuncts, with the same switches, so they share
    /// the filter plans of the file and the decision to skip a conjunct.
    pub(crate) fn conjuncts(
        &self,
        predicate: &Arc<dyn PhysicalExpr>,
        path: &Path,
        schema: &SchemaRef,
        adapt: impl FnOnce(Arc<dyn PhysicalExpr>) -> DFResult<Arc<dyn PhysicalExpr>>,
        converter: &Arc<dyn ExpressionConverter>,
        projected: &[&str],
    ) -> DFResult<Vec<Expression>> {
        let filter = self.filter(predicate, schema, adapt, converter)?;
        let key = (Arc::as_ptr(predicate).cast::<()>().addr(), path.clone());
        if !matches!(
            DynamicFilterTracking::classify(&filter.adapted),
            DynamicFilterTracking::AllComplete
        ) {
            if let Some(switched) = self.dynamic.get(&key) {
                return Ok(vec![switched.clone()]);
            }
            let switched = switch(filter.dynamic_expression(), DYNAMIC_MAX_KEPT_PERCENT);
            return Ok(vec![self.dynamic.entry(key).or_insert(switched).clone()]);
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
            .map(|(conjunct, max_kept_percent)| switch(conjunct.clone(), *max_kept_percent))
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

/// The Vortex conjuncts of a dynamic filter, each with the largest percentage of rows it can keep
/// and still be evaluated.
type Conjuncts = Arc<[(Expression, u8)]>;

/// A dynamic filter adapted to a file schema, with its conversion as of a generation.
struct FileDynamicFilter {
    /// The filter as the scan received it, kept so that the address in its key is not reused.
    _predicate: Arc<dyn PhysicalExpr>,
    adapted: Arc<dyn PhysicalExpr>,
    schema: SchemaRef,
    converter: Arc<dyn ExpressionConverter>,
    conversion: RwLock<Option<(u64, Conjuncts)>>,
    /// The dynamic expression of the filter, built once so that every file shares its plans.
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

        // Openers that miss convert concurrently rather than wait on each other.
        let conjuncts: Conjuncts =
            match snapshot_conjuncts(&self.adapted, &self.schema, self.converter.as_ref()) {
                Ok(conjuncts) => conjuncts.into(),
                Err(err) => {
                    tracing::debug!(filter = %self.adapted, %err, "dropping dynamic filter");
                    Arc::from([])
                }
            };
        *self.conversion.write() = Some((generation, Arc::clone(&conjuncts)));
        (generation, conjuncts)
    }

    /// A dynamic expression over the columns of the filter that evaluates its current value.
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
                dynamic_expression(
                    Arc::new(DataFusionDynamicFilter {
                        filter: Arc::clone(self),
                        scope_projection,
                    }),
                    inputs,
                )
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
        // bounds that hold in every partition hold for the whole filter.
        let parts = match conjunct.downcast_ref::<df_expr::CaseExpr>() {
            Some(case) => partition_bounds(case),
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
    use vortex::expr::gt_eq;
    use vortex::expr::lt_eq;
    use vortex::scalar_fn::fns::switch::Switch;

    use super::*;
    use crate::convert::DefaultExpressionConverter;

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
