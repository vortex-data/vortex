// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Stats proofs of dynamic expressions.

use std::fmt::Display;
use std::fmt::Formatter;
use std::sync::Arc;

use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::aggregate_fn::AggregateFnVTableExt;
use vortex_array::aggregate_fn::EmptyOptions as AggregateEmptyOptions;
use vortex_array::aggregate_fn::fns::all_nan::AllNan;
use vortex_array::aggregate_fn::fns::all_non_nan::AllNonNan;
use vortex_array::aggregate_fn::fns::all_non_null::AllNonNull;
use vortex_array::aggregate_fn::fns::all_null::AllNull;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldNames;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::StructFields;
use vortex_array::expr::BoundExpression;
use vortex_array::expr::bound::get_item;
use vortex_array::expr::bound::lit;
use vortex_array::expr::stats::Stat;
use vortex_array::scalar_fn::EmptyOptions;
use vortex_array::scalar_fn::ScalarFnVTableExt;
use vortex_array::scalar_fn::fns::dynamic_expression::DynamicExpression;
use vortex_array::scalar_fn::fns::dynamic_expression::DynamicExpressionOptions;
use vortex_array::scalar_fn::fns::dynamic_expression::DynamicExpressionSource;
use vortex_array::scalar_fn::fns::dynamic_expression::input_field_name;
use vortex_array::scalar_fn::fns::get_item::GetItem;
use vortex_array::scalar_fn::fns::stat::StatFn;
use vortex_array::scalar_fn::internal::row_count::RowCount;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::stats::bound::stat;
use crate::stats::rewrite::BoundExpressionPruneExt;

/// Builds a dynamic stats proof for a dynamic expression.
///
/// Zone pruning builds and lowers the proof of a predicate once, then evaluates it again when a
/// dynamic expression changes. Thus the proof is itself a dynamic expression. Its children are
/// the statistics of the inputs that a proof can use, and the row count, so lowering binds them
/// to the stats of each zone. Each generation proves the current predicate again.
pub(crate) fn dynamic_stats_proof(
    expr: &BoundExpression,
    session: &VortexSession,
) -> VortexResult<Option<BoundExpression>> {
    let options = expr.as_::<DynamicExpression>();
    let input_dtypes = expr
        .children()
        .iter()
        .map(|child| child.dtype().clone())
        .collect::<Vec<_>>();

    let mut stats = Vec::new();
    let mut children = Vec::new();
    for (input_idx, input) in expr.children().iter().enumerate() {
        for aggregate_fn in proof_aggregates() {
            if aggregate_fn.return_dtype(input.dtype()).is_some() {
                children.push(stat(input.clone(), aggregate_fn.clone()));
                stats.push((input_idx, aggregate_fn));
            }
        }
    }
    children.push(RowCount.try_new_bound_expr(EmptyOptions, [])?);

    let source = StatsProofSource {
        predicate: options.clone(),
        input_dtypes,
        stats,
        session: session.clone(),
    };
    Ok(Some(DynamicExpression.try_new_bound_expr(
        DynamicExpressionOptions::new(Arc::new(source)),
        children,
    )?))
}

/// The statistics a proof of a dynamic expression can read.
fn proof_aggregates() -> Vec<AggregateFnRef> {
    [Stat::Min, Stat::Max, Stat::NullCount, Stat::NaNCount]
        .into_iter()
        .filter_map(|stat| stat.aggregate_fn())
        .chain([
            AllNull.bind(AggregateEmptyOptions),
            AllNonNull.bind(AggregateEmptyOptions),
            AllNan.bind(AggregateEmptyOptions),
            AllNonNan.bind(AggregateEmptyOptions),
        ])
        .collect()
}

/// The source of a dynamic stats proof, see [`dynamic_stats_proof`].
#[derive(Debug)]
struct StatsProofSource {
    predicate: DynamicExpressionOptions,
    input_dtypes: Vec<DType>,
    /// The input and aggregate of each statistic child, in child order. The row count follows.
    stats: Vec<(usize, AggregateFnRef)>,
    session: VortexSession,
}

impl StatsProofSource {
    /// Replace the statistics and row count in `proof` by fields of `scope`. Returns `None` when
    /// the proof reads a statistic that is not a child.
    fn bind_children(
        &self,
        proof: &BoundExpression,
        scope: &DType,
    ) -> VortexResult<Option<BoundExpression>> {
        if let Some(options) = proof.as_opt::<StatFn>() {
            let child = proof.child(0);
            let Some(field) = child.as_opt::<GetItem>() else {
                return Ok(None);
            };
            if !child.child(0).is_root() {
                return Ok(None);
            }
            let Some(stat_idx) = self.stats.iter().position(|(input_idx, aggregate_fn)| {
                input_field_name(*input_idx) == *field && aggregate_fn == options.aggregate_fn()
            }) else {
                return Ok(None);
            };
            return Ok(Some(scope_field(stat_idx, scope)?));
        }

        if proof.is::<RowCount>() {
            return Ok(Some(scope_field(self.stats.len(), scope)?));
        }

        if proof.is_root() {
            return Ok(None);
        }

        let mut children = Vec::with_capacity(proof.children().len());
        for child in proof.children() {
            let Some(child) = self.bind_children(child, scope)? else {
                return Ok(None);
            };
            children.push(child);
        }
        Ok(Some(proof.clone().with_children(children)?))
    }
}

fn scope_field(idx: usize, scope: &DType) -> VortexResult<BoundExpression> {
    Ok(get_item(
        input_field_name(idx),
        BoundExpression::new_root(scope.clone()),
    ))
}

impl Display for StatsProofSource {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "falsify({})", self.predicate)
    }
}

impl DynamicExpressionSource for StatsProofSource {
    fn generation(&self) -> u64 {
        self.predicate.source().generation()
    }

    fn current(&self, scope: &DType) -> VortexResult<BoundExpression> {
        let names = (0..self.input_dtypes.len())
            .map(input_field_name)
            .collect::<FieldNames>();
        let input_scope = DType::Struct(
            StructFields::new(names, self.input_dtypes.clone()),
            Nullability::NonNullable,
        );
        let predicate = self.predicate.current(&input_scope)?;
        let proof = predicate.falsify(&self.session)?;

        // A proof that is `false` proves nothing.
        Ok(match proof {
            Some(proof) => self
                .bind_children(&proof, scope)?
                .unwrap_or_else(|| lit(false)),
            None => lit(false),
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicU64;
    use std::sync::atomic::Ordering;

    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::arrays::StructArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::PType;
    use vortex_array::expr::col;
    use vortex_array::expr::dynamic_expression;
    use vortex_array::expr::get_item as unbound_get_item;
    use vortex_array::expr::lit as unbound_lit;
    use vortex_array::expr::lt;
    use vortex_array::expr::root;
    use vortex_array::scalar::Scalar;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;

    use super::*;
    use crate::stats::StatsSession;

    #[derive(Debug, Default)]
    struct Threshold(AtomicU64);

    impl Display for Threshold {
        fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
            write!(f, "threshold")
        }
    }

    impl DynamicExpressionSource for Threshold {
        fn generation(&self) -> u64 {
            self.0.load(Ordering::SeqCst)
        }

        fn current(&self, scope: &DType) -> VortexResult<BoundExpression> {
            if self.generation().is_multiple_of(2) {
                unbound_lit(true).bind(scope)
            } else {
                lt(
                    unbound_get_item(input_field_name(0), root()),
                    unbound_lit(5u64),
                )
                .bind(scope)
            }
        }
    }

    #[test]
    fn existing_falsifier_follows_new_generations() -> VortexResult<()> {
        let session = VortexSession::empty().with::<StatsSession>();
        let threshold = Arc::new(Threshold::default());
        let dtype = DType::struct_(
            [("a", DType::Primitive(PType::U64, Nullability::NonNullable))],
            Nullability::NonNullable,
        );
        let predicate = dynamic_expression(
            Arc::clone(&threshold) as Arc<dyn DynamicExpressionSource>,
            [col("a")],
        )
        .bind(&dtype)?;
        let falsifier = predicate.falsify(&session)?.unwrap();
        let columns = falsifier
            .children()
            .iter()
            .map(|child| {
                if let Some(options) = child.as_opt::<StatFn>() {
                    match Stat::from_aggregate_fn(options.aggregate_fn()) {
                        Some(Stat::Min) => return buffer![7u64].into_array(),
                        Some(Stat::Max) => return buffer![9u64].into_array(),
                        Some(Stat::NullCount) => return buffer![0u64].into_array(),
                        _ => {}
                    }
                }
                ConstantArray::new(Scalar::null(child.dtype().as_nullable()), 1).into_array()
            })
            .collect::<Vec<_>>();
        let names = (0..columns.len())
            .map(input_field_name)
            .collect::<FieldNames>();
        let stats = StructArray::try_new(names, columns, 1, Validity::NonNullable)?.into_array();
        let mut ctx = session.create_execution_ctx();
        for (generation, expected) in [(0, false), (1, true), (2, false)] {
            threshold.0.store(generation, Ordering::SeqCst);
            let proof = falsifier
                .as_::<DynamicExpression>()
                .current(stats.dtype())?;
            assert_arrays_eq!(
                stats.clone().apply_bound(&proof)?,
                BoolArray::from_iter([expected]),
                &mut ctx
            );
        }
        Ok(())
    }
}
