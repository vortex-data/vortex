// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::NullArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldPath;
use vortex_array::expr::BoundExpression;
use vortex_array::expr::bound::lit;
use vortex_array::expr::traversal::TraversalOrder;
use vortex_array::expr::traversal::pre_order_visit_down;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::fns::binary::Binary;
use vortex_array::scalar_fn::fns::binary::scalar_cmp;
use vortex_array::scalar_fn::fns::cast::Cast;
use vortex_array::scalar_fn::fns::dynamic::DynamicExprUpdates;
use vortex_array::scalar_fn::fns::get_item::GetItem;
use vortex_array::scalar_fn::fns::literal::Literal;
use vortex_array::scalar_fn::fns::operators::CompareOperator;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_array::scalar_fn::fns::stat::StatFn;
use vortex_array::scalar_fn::internal::row_count::RowCount;
use vortex_array::scalar_fn::internal::row_count::substitute_row_count;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;
use vortex_expr::stats::bind::StatBinder;
use vortex_expr::stats::bind::bind_stats;
use vortex_expr::stats::rewrite::BoundExpressionPruneExt;
use vortex_session::VortexSession;

use crate::FileStatistics;
use crate::Footer;
use crate::VortexFile;

/// A static predicate's file-statistics proof, prepared once for a multi-file scan.
///
/// Simple comparisons and boolean combinations read footer scalars directly. Other expressions
/// use the general evaluator. Dynamic predicates are left to the caller so a proof cannot outlive
/// the dynamic values from which it was derived.
#[derive(Debug)]
pub struct FilePruningPredicate {
    root_dtype: Option<DType>,
    proof: StatsProof,
}

impl FilePruningPredicate {
    /// Prepares a pruning proof, or returns `None` for dynamic or unsupported predicates.
    pub fn try_new(filter: BoundExpression, session: &VortexSession) -> VortexResult<Option<Self>> {
        if DynamicExprUpdates::new(&filter).is_some() {
            return Ok(None);
        }
        let Some(proof) = filter.falsify(session)? else {
            return Ok(None);
        };
        let mut root_dtype = None;
        pre_order_visit_down(&filter, |node| {
            if node.is_root() {
                if let Some(dtype) = &root_dtype {
                    vortex_ensure_eq!(
                        node.dtype(),
                        dtype,
                        "File pruning predicate has roots bound to different dtypes"
                    );
                } else {
                    root_dtype = Some(node.dtype().clone());
                }
            }
            Ok(TraversalOrder::Continue)
        })?;
        Ok(Some(Self {
            root_dtype,
            proof: StatsProof::new(&proof),
        }))
    }

    /// Whether this file's statistics prove that the predicate cannot match any rows.
    /// Missing statistics remain unknown and never prove a file can be skipped by themselves.
    pub fn can_prune(&self, file: &VortexFile) -> VortexResult<bool> {
        self.can_prune_footer(file.footer(), file.session())
    }

    /// Evaluates a cached footer without opening the file or creating a segment source.
    pub fn can_prune_footer(&self, footer: &Footer, session: &VortexSession) -> VortexResult<bool> {
        vortex_ensure!(
            self.root_dtype
                .as_ref()
                .is_none_or(|dtype| dtype == footer.dtype()),
            "File pruning predicate is bound to a different dtype"
        );
        let Some(statistics) = footer.statistics() else {
            return Ok(false);
        };
        Ok(self
            .proof
            .evaluate(footer.row_count(), statistics, session)?
            == Some(true))
    }
}

/// Simple proofs read footer scalars directly. Unsupported subtrees retain the general evaluator.
#[derive(Debug)]
enum StatsProof {
    And(Box<Self>, Box<Self>),
    Or(Box<Self>, Box<Self>),
    Compare(StatsValue, StatsValue, CompareOperator),
    Value(StatsValue),
    Fallback(BoundExpression),
}

impl StatsProof {
    fn new(expr: &BoundExpression) -> Self {
        if let Some(operator) = expr.as_opt::<Binary>() {
            match operator {
                Operator::And => {
                    return Self::And(
                        Box::new(Self::new(expr.child(0))),
                        Box::new(Self::new(expr.child(1))),
                    );
                }
                Operator::Or => {
                    return Self::Or(
                        Box::new(Self::new(expr.child(0))),
                        Box::new(Self::new(expr.child(1))),
                    );
                }
                _ => {}
            }
            if let (Some(operator), Some(left), Some(right)) = (
                operator.to_compare_operator(),
                StatsValue::new(expr.child(0)),
                StatsValue::new(expr.child(1)),
            ) {
                return Self::Compare(left, right, operator);
            }
        }
        if let Some(value) = StatsValue::new(expr) {
            return Self::Value(value);
        }
        Self::Fallback(expr.clone())
    }

    fn evaluate(
        &self,
        row_count: u64,
        file_stats: &FileStatistics,
        session: &VortexSession,
    ) -> VortexResult<Option<bool>> {
        match self {
            Self::And(left, right) => {
                let left = left.evaluate(row_count, file_stats, session)?;
                if left == Some(false) {
                    return Ok(Some(false));
                }
                let right = right.evaluate(row_count, file_stats, session)?;
                Ok(match (left, right) {
                    (_, Some(false)) => Some(false),
                    (Some(true), Some(true)) => Some(true),
                    _ => None,
                })
            }
            Self::Or(left, right) => {
                let left = left.evaluate(row_count, file_stats, session)?;
                if left == Some(true) {
                    return Ok(Some(true));
                }
                let right = right.evaluate(row_count, file_stats, session)?;
                Ok(match (left, right) {
                    (_, Some(true)) => Some(true),
                    (Some(false), Some(false)) => Some(false),
                    _ => None,
                })
            }
            Self::Compare(left, right, operator) => Ok(scalar_cmp(
                &left.evaluate(row_count, file_stats),
                &right.evaluate(row_count, file_stats),
                *operator,
            )?
            .as_bool()
            .value()),
            Self::Value(value) => Ok(value.evaluate(row_count, file_stats).as_bool().value()),
            Self::Fallback(expr) => {
                evaluate_file_stats_proof_value(expr.clone(), row_count, file_stats, session)
            }
        }
    }
}

#[derive(Debug)]
enum StatsValue {
    Literal(Scalar),
    Statistic {
        path: FieldPath,
        aggregate: AggregateFnRef,
        dtype: DType,
    },
    RowCount,
}

impl StatsValue {
    fn new(expr: &BoundExpression) -> Option<Self> {
        if let Some(value) = expr.as_opt::<Literal>() {
            return Some(Self::Literal(value.clone()));
        }
        if let Some(stat) = expr.as_opt::<StatFn>() {
            return Some(Self::Statistic {
                path: direct_field_path(expr.child(0))?,
                aggregate: stat.aggregate_fn().clone(),
                dtype: expr.dtype().clone(),
            });
        }
        expr.is::<RowCount>().then_some(Self::RowCount)
    }

    fn evaluate(&self, row_count: u64, file_stats: &FileStatistics) -> Scalar {
        match self {
            Self::Literal(value) => value.clone(),
            Self::Statistic {
                path,
                aggregate,
                dtype,
            } => FileStatsBinder { file_stats }
                .aggregate_value(path, aggregate, dtype)
                .unwrap_or_else(|| Scalar::null(dtype.as_nullable())),
            Self::RowCount => row_count.into(),
        }
    }
}

pub(crate) fn is_falsified_by_file_stats(
    expr: &BoundExpression,
    row_count: u64,
    file_stats: &FileStatistics,
    session: &VortexSession,
) -> VortexResult<bool> {
    let Some(pruning_expr) = expr.falsify(session)? else {
        return Ok(false);
    };

    evaluate_file_stats_proof(pruning_expr, row_count, file_stats, session)
}

pub(crate) fn is_satisfied_by_file_stats(
    expr: &BoundExpression,
    row_count: u64,
    file_stats: &FileStatistics,
    session: &VortexSession,
) -> VortexResult<bool> {
    let Some(proof) = expr.satisfy(session)? else {
        return Ok(false);
    };
    evaluate_file_stats_proof(proof, row_count, file_stats, session)
}

fn evaluate_file_stats_proof(
    pruning_expr: BoundExpression,
    row_count: u64,
    file_stats: &FileStatistics,
    session: &VortexSession,
) -> VortexResult<bool> {
    Ok(
        evaluate_file_stats_proof_value(pruning_expr, row_count, file_stats, session)?
            == Some(true),
    )
}

fn evaluate_file_stats_proof_value(
    pruning_expr: BoundExpression,
    row_count: u64,
    file_stats: &FileStatistics,
    session: &VortexSession,
) -> VortexResult<Option<bool>> {
    let binder = FileStatsBinder { file_stats };
    let pruning_expr = bind_stats(pruning_expr, &binder)?;

    if let Some(result) = pruning_expr.as_opt::<Literal>() {
        return Ok(result.as_bool().value());
    }

    let pruning = NullArray::new(1).into_array().apply_bound(&pruning_expr)?;
    let row_count_replacement = ConstantArray::new(row_count, pruning.len()).into_array();
    let pruning = substitute_row_count(pruning, &row_count_replacement)?;

    let mut ctx = session.create_execution_ctx();
    let result = pruning
        .execute::<Canonical>(&mut ctx)?
        .into_bool()
        .into_array()
        .execute_scalar(0, &mut ctx)?;

    Ok(result.as_bool().value())
}

struct FileStatsBinder<'a> {
    file_stats: &'a FileStatistics,
}

impl StatBinder for FileStatsBinder<'_> {
    fn bind_aggregate(
        &self,
        input: &BoundExpression,
        aggregate_fn: &AggregateFnRef,
        stat_dtype: &DType,
    ) -> VortexResult<Option<BoundExpression>> {
        let Some(field_path) = direct_field_path(input) else {
            return Ok(None);
        };
        Ok(self
            .aggregate_value(&field_path, aggregate_fn, stat_dtype)
            .map(lit))
    }
}

impl FileStatsBinder<'_> {
    /// Binds `aggregate_fn` to the value of the file aggregate that best satisfies it. Pruning
    /// only needs a bound, so an approximate value (e.g. a byte-bounded max for `max`) works too.
    fn aggregate_value(
        &self,
        field_path: &FieldPath,
        aggregate_fn: &AggregateFnRef,
        stat_dtype: &DType,
    ) -> Option<Scalar> {
        let (aggregates, _) = self.file_stats.get_by_path(field_path)?;
        let value = aggregates.get(aggregate_fn).into_inner()?;
        let value = if value.dtype() == stat_dtype {
            value
        } else {
            value.cast(stat_dtype).ok()?
        };
        Some(value)
    }
}

fn direct_field_path(expr: &BoundExpression) -> Option<FieldPath> {
    if expr.is_root() {
        return Some(FieldPath::root());
    }

    if expr.is::<Cast>() {
        return direct_field_path(expr.child(0));
    }

    let field_name = expr.as_opt::<GetItem>()?;
    direct_field_path(expr.child(0)).map(|path| path.push(field_name.clone()))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use rstest::rstest;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::expr::binary;
    use vortex_array::expr::lit;
    use vortex_array::expr::not;
    use vortex_array::scalar::Scalar;
    use vortex_array::scalar_fn::fns::operators::Operator;
    use vortex_error::VortexResult;
    use vortex_layout::session::LayoutSession;
    use vortex_session::VortexSession;

    use super::StatsProof;
    use super::evaluate_file_stats_proof_value;
    use crate::FileStatistics;

    #[rstest]
    #[case(Some(false), Some(false), Some(false), Some(false))]
    #[case(Some(false), Some(true), Some(false), Some(true))]
    #[case(Some(false), None, Some(false), None)]
    #[case(Some(true), Some(false), Some(false), Some(true))]
    #[case(Some(true), Some(true), Some(true), Some(true))]
    #[case(Some(true), None, None, Some(true))]
    #[case(None, Some(false), Some(false), None)]
    #[case(None, Some(true), None, Some(true))]
    #[case(None, None, None, None)]
    fn scalar_proofs_preserve_three_valued_logic(
        #[case] left: Option<bool>,
        #[case] right: Option<bool>,
        #[case] and_expected: Option<bool>,
        #[case] or_expected: Option<bool>,
    ) -> VortexResult<()> {
        let session = VortexSession::empty().with::<LayoutSession>();
        crate::register_default_encodings(&session);
        crate::enable_all_registered_array_encodings(&session);
        let stats = FileStatistics::new(Arc::new([]), Arc::new([]), Arc::new([]));
        let scalar = |value: Option<bool>| match value {
            Some(value) => Scalar::bool(value, Nullability::Nullable),
            None => Scalar::null(DType::Bool(Nullability::Nullable)),
        };
        for (operator, expected) in [(Operator::And, and_expected), (Operator::Or, or_expected)] {
            // Negation uses the fallback evaluator, including when its result is unknown.
            for fallback in [false, true] {
                let rhs = if fallback {
                    not(lit(scalar(right.map(|value| !value))))
                } else {
                    lit(scalar(right))
                };
                let expression = binary(operator, lit(scalar(left)), rhs).bind(&DType::Null)?;
                assert_eq!(
                    StatsProof::new(&expression).evaluate(0, &stats, &session)?,
                    expected
                );
                assert_eq!(
                    evaluate_file_stats_proof_value(expression, 0, &stats, &session)?,
                    expected,
                );
            }
        }
        Ok(())
    }
}
