// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Formatter;

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::ScalarFn;
use vortex_array::arrays::scalar_fn::ExactScalarFn;
use vortex_array::arrays::scalar_fn::ScalarFnArrayExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::expr::Expression;
use vortex_array::expr::lit;
use vortex_array::scalar_fn::Arity;
use vortex_array::scalar_fn::ChildName;
use vortex_array::scalar_fn::EmptyOptions;
use vortex_array::scalar_fn::ExecutionArgs;
use vortex_array::scalar_fn::ScalarFnId;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_session::registry::CachedId;

/// Zero-argument placeholder for the row count of the current evaluation scope.
///
/// This is a legacy pruning hack for readers that only have a `null_count`
/// stat and need to support `is_not_null` pruning. It is currently substituted
/// by the zoned/file stats pruning paths before execution. New stats rewrites
/// should prefer boolean `all_null` and `all_non_null` aggregates instead of
/// depending on this scope-level placeholder.
///
/// This expression *MUST* be replaced with a concrete array before evaluation.
/// Currently, the rewrite only happens in the context of stats pruning.
///
/// `RowCount` is emitted while building pruning predicates that need a
/// scope-level value which is not stored as a regular stats column, such as the
/// row count of the current file or zone. The layer that owns that scope must
/// replace each placeholder with a concrete array via [`substitute_row_count`]
/// before evaluation.
///
/// Calling [`ScalarFnVTable::execute`] directly returns an error because this
/// node is only a marker in a lazy expression tree.
#[derive(Clone)]
pub struct RowCount;

impl ScalarFnVTable for RowCount {
    type Options = EmptyOptions;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.row_count");
        *ID
    }

    fn arity(&self, _options: &Self::Options) -> Arity {
        Arity::Exact(0)
    }

    fn child_name(&self, _options: &Self::Options, _child_idx: usize) -> ChildName {
        unreachable!("RowCount has arity 0")
    }

    fn fmt_sql(
        &self,
        _options: &Self::Options,
        _expr: &Expression,
        f: &mut Formatter<'_>,
    ) -> std::fmt::Result {
        write!(f, "row_count()")
    }

    fn return_dtype(&self, _options: &Self::Options, _args: &[DType]) -> VortexResult<DType> {
        Ok(DType::Primitive(PType::U64, Nullability::NonNullable))
    }

    fn execute(
        &self,
        _options: &Self::Options,
        _args: &dyn ExecutionArgs,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        vortex_bail!("RowCount must be substituted before evaluation")
    }

    fn validity(
        &self,
        _options: &Self::Options,
        _expression: &Expression,
    ) -> VortexResult<Expression> {
        Ok(lit(true))
    }

    fn is_null_sensitive(&self, _options: &Self::Options) -> bool {
        false
    }

    fn is_fallible(&self, _options: &Self::Options) -> bool {
        false
    }
}

/// Returns whether `array` contains a [`RowCount`] placeholder.
///
/// Traversal is limited to lazy [`ScalarFnArray`] nodes produced by
/// [`ArrayRef::apply`][crate::ArrayRef::apply]. Other arrays are evaluation
/// leaves and cannot contain unevaluated placeholders.
///
/// [`ScalarFnArray`]: vortex_array::arrays::ScalarFnArray
pub fn contains_row_count(array: &ArrayRef) -> bool {
    if array.is::<ExactScalarFn<RowCount>>() {
        return true;
    }
    match array.as_opt::<ScalarFn>() {
        Some(view) => view.iter_children().any(contains_row_count),
        None => false,
    }
}

/// Replaces every [`RowCount`] placeholder with `replacement`.
///
/// The replacement must have the same dtype and length as each placeholder.
/// Lazy [`ScalarFnArray`] ancestors are rewritten through slot take/put so
/// unaffected children are preserved, while non-[`ScalarFn`] arrays are returned
/// unchanged.
///
/// [`ScalarFnArray`]: vortex_array::arrays::ScalarFnArray
pub fn substitute_row_count(array: ArrayRef, replacement: &ArrayRef) -> VortexResult<ArrayRef> {
    if array.is::<ExactScalarFn<RowCount>>() {
        vortex_ensure!(
            replacement.len() == array.len(),
            "RowCount replacement length {} does not match scope length {}",
            replacement.len(),
            array.len(),
        );
        vortex_ensure!(
            replacement.dtype() == array.dtype(),
            "RowCount replacement dtype {} does not match scope dtype {}",
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
        // SAFETY: `substitute_row_count` always returns an array with the same dtype and
        // length as its input — `RowCount` placeholders are replaced with a checked
        // replacement (same dtype and length), and `ScalarFn` recursion preserves both by
        // operating on each slot in place.
        let (taken, child) = unsafe { array.take_slot_unchecked(slot_idx)? };
        let new_child = substitute_row_count(child, replacement)?;
        array = unsafe { taken.put_slot_unchecked(slot_idx, new_child)? };
    }
    Ok(array)
}

#[cfg(test)]
mod tests {
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;

    use crate::scalar_fn::EmptyOptions;
    use crate::scalar_fn::internal::row_count::RowCount;
    use crate::scalar_fn::vtable::ScalarFnVTableExt;

    #[test]
    fn row_count_helper_dtype() {
        let expr = RowCount.new_expr(EmptyOptions, []);
        assert_eq!(
            expr.return_dtype(&DType::Primitive(PType::I32, Nullability::Nullable))
                .unwrap(),
            DType::Primitive(PType::U64, Nullability::NonNullable),
        );
    }
}
