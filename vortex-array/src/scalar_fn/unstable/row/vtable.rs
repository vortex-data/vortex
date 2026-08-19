// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Adapts [`RowFn`] implementations to the scalar-function interface.
//!
//! The blanket [`ScalarFnVTable`] implementation supplies common arity, validity, fallibility, and
//! execution behavior. [`row_fn_return_dtype`] and [`execute_rows`] expose the same planning and
//! execution paths to public vtables that delegate to a private row kernel.

use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure_eq;
use vortex_session::VortexSession;

use super::row_fn::RowFn;
use super::visitor::BatchPlanner;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::dtype::DType;
use crate::expr::Expression;
use crate::expr::union_child_validities;
use crate::scalar_fn::Arity;
use crate::scalar_fn::ChildName;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnVTable;

impl<F: RowFn> ScalarFnVTable for F {
    type Options = F::Options;

    fn id(&self) -> ScalarFnId {
        RowFn::id(self)
    }

    fn serialize(&self, options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        RowFn::serialize(self, options)
    }

    fn deserialize(&self, metadata: &[u8], session: &VortexSession) -> VortexResult<Self::Options> {
        RowFn::deserialize(self, metadata, session)
    }

    fn arity(&self, _options: &Self::Options) -> Arity {
        Arity::Exact(F::ARG_NAMES.len())
    }

    fn child_name(&self, _options: &Self::Options, child_index: usize) -> ChildName {
        ChildName::from(F::ARG_NAMES[child_index])
    }

    fn return_dtype(&self, options: &Self::Options, args: &[DType]) -> VortexResult<DType> {
        row_fn_return_dtype(self, options, args)
    }

    fn execute(
        &self,
        options: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        execute_rows(self, options, args, ctx)
    }

    fn validity(
        &self,
        _options: &Self::Options,
        expression: &Expression,
    ) -> VortexResult<Option<Expression>> {
        union_child_validities(expression)
    }

    fn is_strict(&self, _options: &Self::Options) -> bool {
        true
    }

    fn is_fallible(&self, _options: &Self::Options) -> bool {
        !F::INFALLIBLE
    }
}

/// Compute the return dtype of a [`RowFn`] kernel without invoking its blanket vtable.
pub fn row_fn_return_dtype<F: RowFn>(
    function: &F,
    options: &F::Options,
    args: &[DType],
) -> VortexResult<DType> {
    ensure_arity(function, args.len())?;

    let plan = function.dispatch(options, args, BatchPlanner::<F>::new(args, options))?;

    Ok(plan.result_dtype(args))
}

/// Execute a [`RowFn`] without using its blanket [`ScalarFnVTable`] implementation.
///
/// A type cannot implement both [`RowFn`] and [`ScalarFnVTable`] because every `RowFn` receives the
/// standard vtable automatically. Existing vtables can keep their custom hooks on one type and
/// delegate row execution to a private `RowFn` kernel through this function.
pub fn execute_rows<F: RowFn>(
    function: &F,
    _options: &F::Options,
    args: &dyn ExecutionArgs,
    _ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    ensure_arity(function, args.num_inputs())?;

    // TODO(connor)[RowFn]: Replace this temporary error with the execution backend in #9129.
    vortex_bail!(
        "Row function {} does not yet have an execution backend",
        RowFn::id(function)
    )
}

/// Validate the number of arguments before calling user-defined dispatch code.
fn ensure_arity<F: RowFn>(function: &F, actual: usize) -> VortexResult<()> {
    let expected = F::ARG_NAMES.len();
    vortex_ensure_eq!(
        actual,
        expected,
        "row function {} requires arity {expected}, got {actual}",
        RowFn::id(function),
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use vortex_error::VortexError;
    use vortex_error::VortexResult;
    use vortex_session::registry::CachedId;

    use super::execute_rows;
    use super::row_fn_return_dtype;
    use crate::VortexSessionExecute;
    use crate::array_session;
    use crate::dtype::DType;
    use crate::scalar_fn::EmptyOptions;
    use crate::scalar_fn::ScalarFnId;
    use crate::scalar_fn::VecExecutionArgs;
    use crate::scalar_fn::unstable::row::RowFn;
    use crate::scalar_fn::unstable::row::RowVisitor;

    #[derive(Clone)]
    struct IndexingRowFn;

    impl RowFn for IndexingRowFn {
        type Options = EmptyOptions;

        const ARG_NAMES: &'static [&'static str] = &["value"];

        const INFALLIBLE: bool = true;

        fn id(&self) -> ScalarFnId {
            static ID: CachedId = CachedId::new("test.indexing_row_fn");
            *ID
        }

        fn dispatch<V: RowVisitor<Self::Options>>(
            &self,
            _options: &Self::Options,
            args: &[DType],
            visitor: V,
        ) -> VortexResult<V::VisitResult> {
            _ = &args[0];

            visitor.visit::<(i64,), i64>(|(value,)| value)
        }
    }

    #[test]
    fn test_return_dtype_rejects_wrong_arity_before_dispatch() {
        let error = row_fn_return_dtype(&IndexingRowFn, &EmptyOptions, &[])
            .expect_err("wrong arity must fail before dispatch");

        assert_arity_error(error);
    }

    #[test]
    fn test_execute_rejects_wrong_arity_before_dispatch() {
        let args = VecExecutionArgs::new(vec![], 0);
        let mut ctx = array_session().create_execution_ctx();
        let error = execute_rows(&IndexingRowFn, &EmptyOptions, &args, &mut ctx)
            .expect_err("wrong arity must fail before dispatch");

        assert_arity_error(error);
    }

    #[track_caller]
    fn assert_arity_error(error: VortexError) {
        assert!(
            error.to_string().contains("requires arity 1, got 0"),
            "unexpected error: {error}",
        );
    }
}
