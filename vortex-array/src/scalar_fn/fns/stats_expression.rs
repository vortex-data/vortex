// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Formatter;
use std::marker::PhantomData;

use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::aggregate_fn::AggregateFnRef;
use crate::aggregate_fn::AggregateFnVTable;
use crate::arrays::ScalarFn;
use crate::arrays::scalar_fn::ExactScalarFn;
use crate::arrays::scalar_fn::ScalarFnArrayExt;
use crate::dtype::DType;
use crate::expr::Expression;
use crate::matcher::Matcher;
use crate::scalar_fn::Arity;
use crate::scalar_fn::ChildName;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnVTable;

/// Zero-argument placeholder for an aggregate-derived statistic.
///
/// `StatsExpression` is emitted while building pruning predicates that need a
/// scope-level value which is not stored as a regular stats column, such as the
/// row count of the current file or zone. The layer that owns that scope must
/// replace each placeholder with a concrete array before evaluation.
///
/// Calling [`ScalarFnVTable::execute`] directly returns an error, because this
/// node is only a marker in a lazy expression tree.
#[derive(Clone)]
pub struct StatsExpression;

impl ScalarFnVTable for StatsExpression {
    type Options = AggregateFnRef;

    fn id(&self) -> ScalarFnId {
        ScalarFnId::from("vortex.stats_expression")
    }

    fn arity(&self, _options: &Self::Options) -> Arity {
        Arity::Exact(0)
    }

    fn child_name(&self, _options: &Self::Options, _child_idx: usize) -> ChildName {
        unreachable!("StatsExpression has arity 0")
    }

    fn fmt_sql(
        &self,
        agg: &Self::Options,
        _expr: &Expression,
        f: &mut Formatter<'_>,
    ) -> std::fmt::Result {
        write!(f, "stats_expression({})", agg.id())
    }

    fn return_dtype(&self, agg: &Self::Options, _args: &[DType]) -> VortexResult<DType> {
        // StatsExpression has no children, so we cannot derive a scope dtype. Aggregates whose
        // return type is input-independent (e.g. RowCount) will still produce a valid dtype.
        agg.return_dtype(&DType::Null).ok_or_else(|| {
            vortex_err!(
                "StatsExpression wraps aggregate {} whose return type depends on scope dtype",
                agg.id()
            )
        })
    }

    fn execute(
        &self,
        agg: &Self::Options,
        _args: &dyn ExecutionArgs,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        vortex_bail!(
            "StatsExpression({}) must be substituted before evaluation",
            agg.id()
        )
    }

    fn is_null_sensitive(&self, _options: &Self::Options) -> bool {
        false
    }

    fn is_fallible(&self, _options: &Self::Options) -> bool {
        false
    }
}

/// Matcher for a [`ScalarFnArray`] that contains a [`StatsExpression`] wrapping
/// aggregate `V`.
///
/// [`ScalarFnArray`]: crate::arrays::ScalarFnArray
#[derive(Debug)]
pub struct StatsExpressionOf<V: AggregateFnVTable>(PhantomData<V>);

impl<V: AggregateFnVTable> Matcher for StatsExpressionOf<V> {
    type Match<'a> = &'a ArrayRef;

    fn matches(array: &ArrayRef) -> bool {
        array
            .as_opt::<ExactScalarFn<StatsExpression>>()
            .is_some_and(|view| view.options.is::<V>())
    }

    fn try_match(array: &ArrayRef) -> Option<Self::Match<'_>> {
        Self::matches(array).then_some(array)
    }
}

/// Returns whether `array` contains a [`StatsExpression`] placeholder for
/// aggregate `V`.
///
/// Traversal is limited to lazy [`ScalarFnArray`] nodes produced by
/// [`ArrayRef::apply`][crate::ArrayRef::apply]. Other arrays are evaluation
/// leaves and cannot contain unevaluated placeholders.
///
/// [`ScalarFnArray`]: crate::arrays::ScalarFnArray
pub fn contains_stats_fn_array<V: AggregateFnVTable>(array: &ArrayRef) -> bool {
    if array.is::<StatsExpressionOf<V>>() {
        return true;
    }
    match array.as_opt::<ScalarFn>() {
        Some(view) => view.iter_children().any(contains_stats_fn_array::<V>),
        None => false,
    }
}

/// Replaces every [`StatsExpression`] placeholder for aggregate `V` with
/// `replacement`.
///
/// The replacement must have the same dtype and length as each placeholder.
/// Lazy [`ScalarFnArray`] ancestors are rewritten through slot take/put so
/// unaffected children are preserved, while non-[`ScalarFn`] arrays are returned
/// unchanged.
///
/// [`ScalarFnArray`]: crate::arrays::ScalarFnArray
pub fn substitute_stats_fn_array<V: AggregateFnVTable>(
    array: ArrayRef,
    replacement: &ArrayRef,
) -> VortexResult<ArrayRef> {
    if array.is::<StatsExpressionOf<V>>() {
        vortex_ensure!(
            replacement.len() == array.len(),
            "StatsExpression replacement length {} does not match scope length {}",
            replacement.len(),
            array.len(),
        );
        vortex_ensure!(
            replacement.dtype() == array.dtype(),
            "StatsExpression replacement dtype {} does not match scope dtype {}",
            replacement.dtype(),
            array.dtype(),
        );
        return Ok(replacement.clone());
    }

    if !array.is::<ScalarFn>() {
        return Ok(array);
    }

    let nchildren = array.nchildren();
    let mut array = array;
    for slot_idx in 0..nchildren {
        // SAFETY: `substitute_stats_fn_array` always returns an array with the same dtype and
        // length as its input — `StatsExpression` placeholders are replaced with a checked
        // replacement (same dtype and length), and `ScalarFn` recursion preserves both by
        // operating on each slot in place.
        let (taken, child) = unsafe { array.take_slot_unchecked(slot_idx)? };
        let new_child = substitute_stats_fn_array::<V>(child, replacement)?;
        array = unsafe { taken.put_slot_unchecked(slot_idx, new_child)? };
    }
    Ok(array)
}

#[cfg(test)]
mod tests {
    use crate::aggregate_fn::fns::row_count::RowCount;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::expr::row_count;

    #[test]
    fn row_count_helper_is_stats_expression() {
        let expr = row_count();
        let agg = expr.as_::<super::StatsExpression>();
        assert!(agg.is::<RowCount>());
        assert_eq!(
            expr.return_dtype(&DType::Primitive(PType::I32, Nullability::Nullable))
                .unwrap(),
            DType::Primitive(PType::U64, Nullability::NonNullable),
        );
    }
}
