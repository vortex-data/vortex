// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cmp::Ordering;
use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;
use std::sync::Arc;

use arrow_schema::SchemaRef;
use datafusion_common::Result as DFResult;
use datafusion_physical_expr::DynamicFilterTracking;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_expr::simplifier::PhysicalExprSimplifier;
use datafusion_physical_expr::split_conjunction;
use datafusion_physical_expr::utils::collect_columns;
use datafusion_physical_expr_common::physical_expr::snapshot_generation;
use datafusion_physical_expr_common::physical_expr::snapshot_physical_expr;
use itertools::Itertools;
use object_store::path::Path;
use vortex::dtype::DType;
use vortex::dtype::FieldName;
use vortex::error::VortexResult;
use vortex::error::vortex_err;
use vortex::expr::BoundExpression;
use vortex::expr::Expression;
use vortex::expr::binary;
use vortex::expr::bound::and_collect;
use vortex::expr::bound::lit as bound_lit;
use vortex::expr::dynamic_expression;
use vortex::expr::get_item;
use vortex::expr::lit;
use vortex::expr::root;
use vortex::expr::stats::Stat;
use vortex::file::VortexFile;
use vortex::scalar_fn::fns::between::Between;
use vortex::scalar_fn::fns::binary::Binary;
use vortex::scalar_fn::fns::dynamic_expression::DynamicExpressionSource;
use vortex::scalar_fn::fns::dynamic_expression::input_field_name;
use vortex::scalar_fn::fns::get_item::GetItem;
use vortex::scalar_fn::fns::list_contains::ListContains;
use vortex::scalar_fn::fns::literal::Literal;
use vortex::scalar_fn::fns::operators::Operator;
use vortex_utils::aliases::dash_map::DashMap;
use vortex_utils::aliases::hash_map::HashMap;

use crate::convert::ExpressionConverter;
use crate::convert::df_predicate::DataFusionPredicate;

/// Conversions of DataFusion dynamic filters to Vortex conjuncts, shared by the openers of a scan.
///
/// Every partition of every file reads the same dynamic filters, and converting a filter can be
/// costly, such as the IN list of a hash join with thousands of keys. A filter is converted once
/// for each file schema and only again when its generation changes.
#[derive(Default)]
pub(crate) struct DynamicFilterConversions {
    /// Each filter adapted to a file schema, with the filter of the key kept so that its address
    /// is not reused.
    adapted: DashMap<(usize, SchemaKey), AdaptedFilter>,
    conversions: DashMap<(usize, SchemaKey), Arc<Conversion>>,
    /// Filters seen complete, which never change again.
    complete: DashMap<usize, Arc<dyn PhysicalExpr>>,
    /// The conjunction pushed into each file for each complete filter.
    file_filters: DashMap<(usize, Path), Option<Expression>>,
}

/// A dynamic filter of a key and the same filter adapted to a file schema.
type AdaptedFilter = (Arc<dyn PhysicalExpr>, Arc<dyn PhysicalExpr>);

/// The Vortex conjuncts of a dynamic filter as of a generation.
struct Conversion {
    /// The filter of the key, kept so that its address is not reused.
    _predicate: Arc<dyn PhysicalExpr>,
    generation: u64,
    conjuncts: Arc<[Expression]>,
}

/// A file schema compared by value, as files and partitions that open at the same time hold
/// separate copies of equal schemas.
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
    /// Whether `predicate`, adapted as `adapted`, holds only dynamic filters that are complete.
    pub(crate) fn is_complete(
        &self,
        predicate: &Arc<dyn PhysicalExpr>,
        adapted: &Arc<dyn PhysicalExpr>,
    ) -> bool {
        let key = Arc::as_ptr(predicate).cast::<()>().addr();
        if self.complete.contains_key(&key) {
            return true;
        }
        let complete = matches!(
            DynamicFilterTracking::classify(adapted),
            DynamicFilterTracking::AllComplete
        );
        if complete {
            self.complete.insert(key, Arc::clone(predicate));
        }
        complete
    }

    /// The conjunction that `filter` builds for the complete `predicate` and the file at `path`,
    /// built once for each file. The key's filter is kept alive by the completeness record.
    pub(crate) fn file_filter(
        &self,
        predicate: &Arc<dyn PhysicalExpr>,
        path: &Path,
        filter: impl FnOnce() -> DFResult<Option<Expression>>,
    ) -> DFResult<Option<Expression>> {
        let key = (Arc::as_ptr(predicate).cast::<()>().addr(), path.clone());
        if let Some(filter) = self.file_filters.get(&key) {
            return Ok(filter.clone());
        }
        let filter = filter()?;
        self.file_filters.insert(key, filter.clone());
        Ok(filter)
    }

    /// `predicate` adapted to `schema` by `adapt`, which runs once for each schema.
    pub(crate) fn adapted(
        &self,
        predicate: &Arc<dyn PhysicalExpr>,
        schema: &SchemaRef,
        adapt: impl FnOnce(Arc<dyn PhysicalExpr>) -> DFResult<Arc<dyn PhysicalExpr>>,
    ) -> DFResult<Arc<dyn PhysicalExpr>> {
        let key = (
            Arc::as_ptr(predicate).cast::<()>().addr(),
            SchemaKey(Arc::clone(schema)),
        );
        if let Some(entry) = self.adapted.get(&key) {
            return Ok(Arc::clone(&entry.1));
        }
        let adapted = adapt(Arc::clone(predicate))?;
        self.adapted
            .insert(key, (Arc::clone(predicate), Arc::clone(&adapted)));
        Ok(adapted)
    }

    /// The Vortex conjuncts of the current value of `adapted`, which is `predicate` adapted to
    /// `schema`. Conjuncts that Vortex cannot evaluate are dropped: a dynamic filter only discards
    /// rows that cannot contribute to the result.
    pub(crate) fn conjuncts(
        &self,
        predicate: &Arc<dyn PhysicalExpr>,
        adapted: &Arc<dyn PhysicalExpr>,
        converter: &dyn ExpressionConverter,
        schema: &SchemaRef,
    ) -> DFResult<Arc<[Expression]>> {
        let key = (
            Arc::as_ptr(predicate).cast::<()>().addr(),
            SchemaKey(Arc::clone(schema)),
        );
        // A generation read before the snapshot is at most older than the snapshot, which only
        // makes the next lookup convert again.
        let generation = snapshot_generation(adapted);
        if let Some(conversion) = self.conversions.get(&key)
            && conversion.generation == generation
        {
            return Ok(Arc::clone(&conversion.conjuncts));
        }

        // Openers that miss convert concurrently rather than wait on each other: a wait would
        // block their worker threads, and converting costs microseconds.
        let snapshot = PhysicalExprSimplifier::new(schema)
            .simplify(snapshot_physical_expr(Arc::clone(adapted))?)?;
        let conjuncts: Arc<[Expression]> = split_conjunction(&snapshot)
            .into_iter()
            .filter_map(|conjunct| match converter.try_convert(conjunct, schema) {
                Ok(Some(converted)) => Some(converted),
                // DataFusion evaluates what Vortex cannot express, such as the hash table lookup
                // of a join, so the scan still removes the rows it rejects early.
                Ok(None) => DataFusionPredicate::expression(conjunct, schema).or_else(|| {
                    tracing::debug!(%conjunct, "dropping unsupported dynamic filter conjunct");
                    None
                }),
                Err(err) => {
                    tracing::debug!(%conjunct, %err, "dropping dynamic filter conjunct");
                    None
                }
            })
            .collect();
        self.conversions.insert(
            key,
            Arc::new(Conversion {
                _predicate: Arc::clone(predicate),
                generation,
                conjuncts: Arc::clone(&conjuncts),
            }),
        );
        Ok(conjuncts)
    }
}

/// A DataFusion predicate with dynamic filters, as the source of a Vortex dynamic expression.
///
/// The columns of the predicate become the inputs of the dynamic expression, so the scan reads
/// them before the dynamic filters have their first value. Each new generation of the dynamic
/// filters is converted again.
pub(crate) struct DataFusionDynamicFilter {
    /// The predicate as the scan received it, which keys the shared conversions.
    predicate: Arc<dyn PhysicalExpr>,
    /// The predicate adapted to the file schema.
    adapted: Arc<dyn PhysicalExpr>,
    columns: Vec<FieldName>,
    /// Whether this source holds only the bounds that the predicate implies on its one column.
    bounds_only: bool,
    conversions: Arc<DynamicFilterConversions>,
    converter: Arc<dyn ExpressionConverter>,
    schema: SchemaRef,
    /// The file whose statistics predict which ranges keep most rows.
    file: VortexFile,
    /// The selectivity above which ranges are dropped, if the scan skips unselective conjuncts.
    max_selectivity: Option<f64>,
}

impl DataFusionDynamicFilter {
    /// Create Vortex dynamic expressions for `predicate`, adapted to `schema` as `adapted`.
    ///
    /// For a predicate over several columns, such as the threshold of a TopK over several sort
    /// keys, `a < x OR (a = x AND b < y)`, there is first one expression for each column with the
    /// bounds the predicate implies on it, here `a <= x`. A scan evaluates these cheap and selective
    /// conjuncts first, so the full predicate only sees the rows they keep.
    pub(crate) fn expressions(
        predicate: Arc<dyn PhysicalExpr>,
        adapted: Arc<dyn PhysicalExpr>,
        conversions: Arc<DynamicFilterConversions>,
        converter: Arc<dyn ExpressionConverter>,
        schema: SchemaRef,
        file: &VortexFile,
        max_selectivity: Option<f64>,
    ) -> Vec<Expression> {
        let columns = collect_columns(&adapted)
            .into_iter()
            .sorted_by_key(|column| column.index())
            .map(|column| FieldName::from(column.name()))
            .collect::<Vec<_>>();

        let source = |columns: Vec<FieldName>, bounds_only: bool| Self {
            predicate: Arc::clone(&predicate),
            adapted: Arc::clone(&adapted),
            columns,
            bounds_only,
            conversions: Arc::clone(&conversions),
            converter: Arc::clone(&converter),
            schema: Arc::clone(&schema),
            file: file.clone(),
            max_selectivity,
        };
        let mut expressions = Vec::with_capacity(columns.len() + 1);
        if columns.len() > 1 {
            for column in &columns {
                expressions.push(source(vec![column.clone()], true).into_expression());
            }
        }
        expressions.push(source(columns, false).into_expression());
        expressions
    }

    fn into_expression(self) -> Expression {
        let inputs = self
            .columns
            .iter()
            .map(|column| get_item(column.clone(), root()))
            .collect::<Vec<_>>();
        dynamic_expression(Arc::new(self), inputs)
    }

    /// Replace the column references of `expr` by the inputs of the dynamic expression. Returns
    /// `None` if `expr` reads the scope in another way.
    fn read_inputs(&self, expr: &Expression) -> Option<Expression> {
        if let Some(field) = expr.as_opt::<GetItem>()
            && expr.child(0).is_root()
        {
            let input_idx = self.columns.iter().position(|column| column == field)?;
            return Some(get_item(input_field_name(input_idx), root()));
        }

        if expr.is_root() {
            return None;
        }

        let children = expr
            .children()
            .iter()
            .map(|child| self.read_inputs(child))
            .collect::<Option<Vec<_>>>()?;
        expr.clone().with_children(children).ok()
    }
}

impl DynamicExpressionSource for DataFusionDynamicFilter {
    fn generation(&self) -> u64 {
        snapshot_generation(&self.adapted)
    }

    fn current(&self, scope: &DType) -> VortexResult<BoundExpression> {
        let conjuncts = self
            .conversions
            .conjuncts(
                &self.predicate,
                &self.adapted,
                self.converter.as_ref(),
                &self.schema,
            )
            .map_err(|err| vortex_err!("Failed to convert dynamic filter {self}: {err}"))?;

        // Each value can bound a range that keeps most rows of this file, such as the keys of a
        // hash join that cover a whole column.
        let conjuncts = match self.max_selectivity {
            Some(max_selectivity) => {
                drop_unselective_ranges(conjuncts.iter().cloned(), &self.file, max_selectivity)
            }
            None => conjuncts.to_vec(),
        };
        let column = get_item(self.columns[0].clone(), root());
        let conjuncts = conjuncts
            .into_iter()
            .filter_map(|conjunct| {
                if self.bounds_only {
                    implied_bound(&conjunct, &column)
                } else {
                    Some(conjunct)
                }
            })
            .filter_map(|conjunct| self.read_inputs(&conjunct)?.bind(scope).ok())
            .collect::<Vec<_>>();

        let predicate = and_collect(conjuncts).unwrap_or_else(|| bound_lit(true));
        tracing::debug!(filter = %self, %predicate, "converted dynamic filter");
        Ok(predicate)
    }
}

/// Drop the range comparisons on each column whose combined range covers more than
/// `max_selectivity` of the values between the column's minimum and maximum in `file`.
///
/// The estimate assumes values spread evenly between the minimum and maximum. A hash join
/// publishes the range of its build-side keys, which often covers most of a probe-side column,
/// and evaluating such a range in the scan costs more than the few rows it removes.
pub(crate) fn drop_unselective_ranges(
    conjuncts: impl IntoIterator<Item = Expression>,
    file: &VortexFile,
    max_selectivity: f64,
) -> Vec<Expression> {
    let conjuncts = conjuncts.into_iter().collect::<Vec<_>>();
    let mut ranges: HashMap<FieldName, (f64, f64)> = HashMap::default();
    for (column, lower, upper) in conjuncts.iter().filter_map(range_comparison) {
        let range = ranges
            .entry(column)
            .or_insert((f64::NEG_INFINITY, f64::INFINITY));
        range.0 = range.0.max(lower);
        range.1 = range.1.min(upper);
    }
    let unselective = ranges
        .into_iter()
        .filter(|(column, (lower, upper))| {
            range_selectivity(file, column, *lower, *upper)
                .is_some_and(|selectivity| selectivity > max_selectivity)
        })
        .map(|(column, _)| column)
        .collect::<Vec<_>>();

    conjuncts
        .into_iter()
        .filter(|conjunct| {
            range_comparison(conjunct).is_none_or(|(column, ..)| !unselective.contains(&column))
                && in_list_selectivity(conjunct, file)
                    .is_none_or(|selectivity| selectivity <= max_selectivity)
        })
        .collect()
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

/// The fraction of the integers between the minimum and maximum of a column that an IN list on
/// it holds, such as the keys of a hash join on a dense surrogate key.
fn in_list_selectivity(expr: &Expression, file: &VortexFile) -> Option<f64> {
    if !expr.is::<ListContains>() {
        return None;
    }
    let column = column_name(expr.child(1))?;
    let (min, max) = column_min_max(file, &column)?;
    let list = expr.child(0).as_opt::<Literal>()?.as_list_opt()?;
    let mut values = list
        .elements()?
        .iter()
        .filter(|element| matches!(element.dtype(), DType::Primitive(ptype, _) if ptype.is_int()))
        .filter_map(|element| element.as_primitive_opt()?.as_opt::<i64>().flatten())
        .filter(|value| (min..=max).contains(&(*value as f64)))
        .collect::<Vec<_>>();
    values.sort_unstable();
    values.dedup();
    #[expect(clippy::cast_precision_loss, reason = "an estimate")]
    Some((values.len() as f64 / (max - min + 1.0)).min(1.0))
}

/// The minimum and maximum of a primitive column in the file statistics.
fn column_min_max(file: &VortexFile, column: &FieldName) -> Option<(f64, f64)> {
    let field_idx = file.dtype().as_struct_fields_opt()?.find(column)?;
    let (stats, dtype) = file.file_stats()?.get(field_idx);
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

/// Whether the file statistics give `expr` a selectivity estimate, as a numeric range or an IN
/// list on a column with a minimum and maximum. [`drop_unselective_ranges`] already dropped such a
/// conjunct if it keeps every row, so no proof needs to show that.
pub(crate) fn has_selectivity_estimate(expr: &Expression, file: &VortexFile) -> bool {
    let column = match range_comparison(expr) {
        Some((column, ..)) => Some(column),
        None if expr.is::<ListContains>() => column_name(expr.child(1)),
        None => None,
    };
    column.is_some_and(|column| column_min_max(file, &column).is_some())
}

/// The column and the numeric range that `expr` keeps, if it compares a column with a number.
fn range_comparison(expr: &Expression) -> Option<(FieldName, f64, f64)> {
    if expr.is::<Between>() {
        let column = column_name(expr.child(0))?;
        return Some((
            column,
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

/// The fraction of the values between the minimum and maximum of `column` that lie in
/// `lower..=upper`.
fn range_selectivity(file: &VortexFile, column: &FieldName, lower: f64, upper: f64) -> Option<f64> {
    let (min, max) = column_min_max(file, column)?;
    if max <= min {
        return None;
    }
    Some(((upper.min(max) - lower.max(min)) / (max - min)).clamp(0.0, 1.0))
}

/// A bound on `column` that `expr` implies, as a comparison of `column` with a literal.
///
/// Every row for which `expr` is true satisfies the bound. A disjunction implies the hull of the
/// bounds of its sides, so `a < x OR (a = x AND b < y)` implies `a <= x`.
fn implied_bound(expr: &Expression, column: &Expression) -> Option<Expression> {
    let operator = *expr.as_opt::<Binary>()?;
    match operator {
        Operator::And => {
            implied_bound(expr.child(0), column).or_else(|| implied_bound(expr.child(1), column))
        }
        Operator::Or => hull(
            &implied_bound(expr.child(0), column)?,
            &implied_bound(expr.child(1), column)?,
        ),
        Operator::Eq | Operator::Lt | Operator::Lte | Operator::Gt | Operator::Gte => {
            let value = expr.child(1).as_opt::<Literal>()?;
            (expr.child(0) == column && !value.is_null()).then(|| expr.clone())
        }
        _ => None,
    }
}

/// The weakest comparison that both bounds imply, if they bound the same side of the column.
fn hull(lhs: &Expression, rhs: &Expression) -> Option<Expression> {
    let (lhs_op, lhs_value) = (*lhs.as_::<Binary>(), lhs.child(1).as_::<Literal>());
    let (rhs_op, rhs_value) = (*rhs.as_::<Binary>(), rhs.child(1).as_::<Literal>());
    let column = lhs.child(0).clone();

    let is_upper = |op| matches!(op, Operator::Eq | Operator::Lt | Operator::Lte);
    let is_lower = |op| matches!(op, Operator::Eq | Operator::Gt | Operator::Gte);
    let inclusive = |op| matches!(op, Operator::Eq | Operator::Lte | Operator::Gte);

    if lhs_op == Operator::Eq && rhs_op == Operator::Eq {
        return (lhs_value == rhs_value).then(|| lhs.clone());
    }
    let (upper, ordering) = if is_upper(lhs_op) && is_upper(rhs_op) {
        (true, lhs_value.partial_cmp(rhs_value)?)
    } else if is_lower(lhs_op) && is_lower(rhs_op) {
        (false, rhs_value.partial_cmp(lhs_value)?)
    } else {
        return None;
    };

    // The looser side wins; on equal values, the bound is inclusive if either side is.
    let (value, inclusive) = match ordering {
        Ordering::Greater => (lhs_value, inclusive(lhs_op)),
        Ordering::Less => (rhs_value, inclusive(rhs_op)),
        Ordering::Equal => (lhs_value, inclusive(lhs_op) || inclusive(rhs_op)),
    };
    let operator = match (upper, inclusive) {
        (true, true) => Operator::Lte,
        (true, false) => Operator::Lt,
        (false, true) => Operator::Gte,
        (false, false) => Operator::Gt,
    };
    Some(binary(operator, column, lit(value.clone())))
}

impl Display for DataFusionDynamicFilter {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        if self.bounds_only {
            write!(f, "bounds({}) of ", self.columns[0])?;
        }
        Display::fmt(&self.adapted, f)
    }
}

impl Debug for DataFusionDynamicFilter {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DataFusionDynamicFilter")
            .field("predicate", &self.adapted)
            .field("columns", &self.columns)
            .field("bounds_only", &self.bounds_only)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use arrow_schema::DataType;
    use arrow_schema::Field;
    use arrow_schema::Schema;
    use datafusion_expr::Operator as DFOperator;
    use datafusion_physical_expr::expressions as df_expr;
    use rstest::rstest;
    use vortex::VortexSessionDefault;
    use vortex::array::IntoArray;
    use vortex::array::arrays::PrimitiveArray;
    use vortex::array::arrays::StructArray;
    use vortex::buffer::ByteBufferMut;
    use vortex::expr::and;
    use vortex::expr::eq;
    use vortex::expr::gt;
    use vortex::expr::gt_eq;
    use vortex::expr::lt;
    use vortex::expr::lt_eq;
    use vortex::expr::or;
    use vortex::file::OpenOptionsSessionExt;
    use vortex::file::WriteOptionsSessionExt;
    use vortex::scalar::Scalar;
    use vortex::session::VortexSession;

    use super::*;
    use crate::convert::DefaultExpressionConverter;

    fn in_list(value: Expression, elements: Vec<Scalar>) -> Expression {
        vortex::expr::in_list_of(
            value,
            DType::Primitive(
                vortex::dtype::PType::I32,
                vortex::dtype::Nullability::NonNullable,
            ),
            elements,
        )
        .expect("elements are i32")
    }

    fn col(name: &str) -> Expression {
        get_item(name, root())
    }

    #[rstest]
    #[case::topk_threshold(
        or(
            lt(col("a"), lit(5)),
            and(eq(col("a"), lit(5)), lt(col("b"), lit("x")))
        ),
        Some(lt_eq(col("a"), lit(5)))
    )]
    #[case::looser_upper_bound(
        or(lt(col("a"), lit(5)), lt_eq(col("a"), lit(3))),
        Some(lt(col("a"), lit(5)))
    )]
    #[case::lower_bounds(
        or(gt(col("a"), lit(5)), gt_eq(col("a"), lit(7))),
        Some(gt(col("a"), lit(5)))
    )]
    #[case::conjunction(
        and(lt(col("b"), lit(1)), gt_eq(col("a"), lit(2))),
        Some(gt_eq(col("a"), lit(2)))
    )]
    #[case::opposite_sides(or(lt(col("a"), lit(5)), gt(col("a"), lit(9))), None)]
    #[case::other_column(or(lt(col("a"), lit(5)), lt(col("b"), lit(9))), None)]
    fn test_implied_bound(#[case] expr: Expression, #[case] expected: Option<Expression>) {
        assert_eq!(implied_bound(&expr, &col("a")), expected);
    }

    #[test]
    fn test_conversion_is_reused_until_the_generation_changes() -> DFResult<()> {
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));
        let column: Arc<dyn PhysicalExpr> = Arc::new(df_expr::Column::new("a", 0));
        let filter = Arc::new(df_expr::DynamicFilterPhysicalExpr::new(
            vec![Arc::clone(&column)],
            df_expr::lit(true),
        ));
        let predicate: Arc<dyn PhysicalExpr> = Arc::clone(&filter) as _;
        let conversions = DynamicFilterConversions::default();
        let converter = DefaultExpressionConverter::default();

        let first = conversions.conjuncts(&predicate, &predicate, &converter, &schema)?;
        let again = conversions.conjuncts(&predicate, &predicate, &converter, &schema)?;
        assert!(Arc::ptr_eq(&first, &again));

        filter.update(Arc::new(df_expr::BinaryExpr::new(
            column,
            DFOperator::GtEq,
            df_expr::lit(7i32),
        )))?;
        let updated = conversions.conjuncts(&predicate, &predicate, &converter, &schema)?;
        assert_eq!(updated.as_ref(), [gt_eq(col("a"), lit(7i32))].as_slice());
        Ok(())
    }

    #[rstest]
    #[case::wide_range(vec![gt_eq(col("a"), lit(1i32)), lt_eq(col("a"), lit(99i32))], 0)]
    #[case::narrow_range(vec![gt_eq(col("a"), lit(10i32)), lt_eq(col("a"), lit(30i32))], 2)]
    #[case::other_conjunct(vec![gt_eq(col("a"), lit(1i32)), eq(col("b"), lit(1i32))], 1)]
    #[case::literal_on_the_left(vec![lt(lit(1i32), col("a"))], 1)]
    #[case::dense_in_list(
        vec![in_list(col("a"), (0i32..95).filter(|v| v % 19 != 7).map(Scalar::from).collect())],
        0
    )]
    #[case::contiguous_in_list(vec![in_list(col("a"), (0i32..90).map(Scalar::from).collect())], 0)]
    #[case::sparse_in_list(vec![in_list(col("a"), (0i32..10).map(Scalar::from).collect())], 1)]
    #[tokio::test]
    async fn test_drop_unselective_ranges(
        #[case] conjuncts: Vec<Expression>,
        #[case] kept: usize,
    ) -> VortexResult<()> {
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
        let file = session.open_options().open_buffer(buf)?;

        let kept_conjuncts = drop_unselective_ranges(conjuncts, &file, 0.75);
        assert_eq!(kept_conjuncts.len(), kept);
        Ok(())
    }
}
