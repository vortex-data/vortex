// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Adapts [`RowFn`] implementations to the scalar-function interface.
//!
//! The blanket [`ScalarFnVTable`] implementation supplies common arity, validity, fallibility, and
//! execution behavior. The visitor layer validates and executes the concrete signature selected by
//! dispatch. [`row_fn_return_dtype`] and [`execute_rows`] expose the same paths to public vtables
//! that delegate to a private row kernel.

use vortex_error::VortexResult;
use vortex_error::vortex_ensure_eq;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use super::batch::BorrowedRowFnArgs;
use super::batch::RowFnExecutionArgs;
use super::batch::finalize_kernel_output;
use super::row_fn::RowFn;
use super::visitor::BatchPlanner;
use super::visitor::ExecuteRows;
use super::visitor::ExecuteValidRows;
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

    // `RowFn` is stricter than `ScalarFnVTable::is_strict`: its kernel cannot produce null from
    // valid inputs, so batch execution derives output validity only from input validity.
    fn is_strict(&self, _options: &Self::Options) -> bool {
        true
    }

    fn is_infallible(&self, _options: &Self::Options) -> bool {
        F::INFALLIBLE
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
///
/// Nullary functions execute for `args.row_count()` rows without batch validity handling because
/// they have no input validity to propagate.
pub fn execute_rows<F: RowFn>(
    function: &F,
    options: &F::Options,
    args: &dyn ExecutionArgs,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    ensure_arity(function, args.num_inputs())?;

    if args.num_inputs() == 0 {
        return execute_nullary_rows(function, options, args.row_count(), ctx);
    }

    let batch = prepare_batch(function, options, args)?;
    batch.execute(
        |args, ctx| execute_row_kernel(function, options, args, ctx),
        |args, valid, ctx| try_execute_valid_rows(function, options, args, valid, ctx),
        ctx,
    )
}

/// Execute a nullary kernel without batch validity or constant handling.
fn execute_nullary_rows<F: RowFn>(
    function: &F,
    options: &F::Options,
    row_count: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let plan = function.dispatch(options, &[], BatchPlanner::<F>::new(&[], options))?;
    let result_dtype = plan.result_dtype(&[]);
    let args = BorrowedRowFnArgs::new(&[], row_count, &[], &plan.output_dtype, plan.policy);

    let values = execute_row_kernel(function, options, args, ctx)?;

    finalize_kernel_output(RowFn::id(function), &result_dtype, row_count, values, ctx)
}

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

fn execute_row_kernel<F: RowFn>(
    function: &F,
    options: &F::Options,
    args: BorrowedRowFnArgs<'_>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    function.dispatch(
        options,
        args.dtypes(),
        ExecuteRows::<F>::new(
            &args,
            args.dtypes(),
            options,
            args.output_dtype(),
            args.policy(),
            ctx,
        ),
    )
}

fn try_execute_valid_rows<F: RowFn>(
    function: &F,
    options: &F::Options,
    args: BorrowedRowFnArgs<'_>,
    valid: &Mask,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ArrayRef>> {
    function.dispatch(
        options,
        args.dtypes(),
        ExecuteValidRows::<F>::new(
            &args,
            args.dtypes(),
            options,
            args.output_dtype(),
            args.policy(),
            valid,
            ctx,
        ),
    )
}

fn prepare_batch<F: RowFn>(
    function: &F,
    options: &F::Options,
    args: &dyn ExecutionArgs,
) -> VortexResult<RowFnExecutionArgs> {
    RowFnExecutionArgs::new(RowFn::id(function), args, |arg_dtypes| {
        function.dispatch(
            options,
            arg_dtypes,
            BatchPlanner::<F>::new(arg_dtypes, options),
        )
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;

    use rstest::rstest;
    use vortex_error::VortexError;
    use vortex_error::VortexResult;
    use vortex_session::registry::CachedId;

    use super::execute_rows;
    use super::row_fn_return_dtype;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::array_session;
    use crate::arrays::PrimitiveArray;
    use crate::assert_arrays_eq;
    use crate::dtype::DType;
    use crate::scalar_fn::EmptyOptions;
    use crate::scalar_fn::ScalarFnId;
    use crate::scalar_fn::VecExecutionArgs;
    use crate::scalar_fn::unstable::row::RowFn;
    use crate::scalar_fn::unstable::row::RowVisitor;
    use crate::validity::Validity;

    #[derive(Clone)]
    struct IndexingRowFn;

    #[derive(Clone)]
    struct NullarySeven;

    #[derive(Clone)]
    struct ChangingDispatchRowFn {
        dispatches: Arc<AtomicUsize>,
        change: DispatchChange,
    }

    #[derive(Clone, Copy)]
    enum DispatchChange {
        Policy,
        Element,
    }

    impl RowFn for NullarySeven {
        type Options = EmptyOptions;

        const ARG_NAMES: &'static [&'static str] = &[];
        const INFALLIBLE: bool = true;

        fn id(&self) -> ScalarFnId {
            static ID: CachedId = CachedId::new("test.nullary_seven");
            *ID
        }

        fn dispatch<V: RowVisitor<Self::Options>>(
            &self,
            _options: &Self::Options,
            _args: &[DType],
            visitor: V,
        ) -> VortexResult<V::VisitResult> {
            visitor.visit::<(), i64>(|()| 7)
        }
    }

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

    impl RowFn for ChangingDispatchRowFn {
        type Options = EmptyOptions;

        const ARG_NAMES: &'static [&'static str] = &["value"];
        const INFALLIBLE: bool = false;

        fn id(&self) -> ScalarFnId {
            static ID: CachedId = CachedId::new("test.changing_dispatch_row_fn");
            *ID
        }

        fn dispatch<V: RowVisitor<Self::Options>>(
            &self,
            _options: &Self::Options,
            _args: &[DType],
            visitor: V,
        ) -> VortexResult<V::VisitResult> {
            if self.dispatches.fetch_add(1, Ordering::Relaxed) == 0 {
                visitor.visit::<(i64,), i64>(|(value,)| value)
            } else {
                match self.change {
                    DispatchChange::Policy => visitor
                        .visit_deferred::<(i64,), i64, bool>(|(value,)| (value, false), |_| Ok(())),
                    DispatchChange::Element => visitor.visit::<(u64,), u64>(|(value,)| value),
                }
            }
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

    #[rstest]
    #[case::empty(0)]
    #[case::nonempty(3)]
    fn test_execute_nullary_rows(#[case] row_count: usize) -> VortexResult<()> {
        let args = VecExecutionArgs::new(vec![], row_count);
        let mut ctx = array_session().create_execution_ctx();

        let actual = execute_rows(&NullarySeven, &EmptyOptions, &args, &mut ctx)?;
        let expected = PrimitiveArray::from_iter(vec![7_i64; row_count]).into_array();

        assert_arrays_eq!(&actual, &expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn test_execute_rejects_dispatch_that_changes_after_planning() -> VortexResult<()> {
        let function = ChangingDispatchRowFn {
            dispatches: Arc::new(AtomicUsize::new(0)),
            change: DispatchChange::Policy,
        };
        let input = PrimitiveArray::new(vec![1_i64, 2], Validity::NonNullable).into_array();
        let args = VecExecutionArgs::new(vec![input], 2);
        let mut ctx = array_session().create_execution_ctx();

        let error = match execute_rows(&function, &EmptyOptions, &args, &mut ctx) {
            Err(error) => error,
            Ok(_) => vortex_error::vortex_bail!("dispatch must not change after planning"),
        };
        let message = error.to_string();

        assert!(
            message.contains("row dispatch must select the planned nullable execution policy"),
            "unexpected error: {error}",
        );
        assert!(
            message.contains("planned Dense, got ValidOnly"),
            "unexpected error: {error}",
        );
        Ok(())
    }

    #[test]
    fn test_execute_revalidates_element_types_after_planning() -> VortexResult<()> {
        let function = ChangingDispatchRowFn {
            dispatches: Arc::new(AtomicUsize::new(0)),
            change: DispatchChange::Element,
        };
        let input = PrimitiveArray::new(vec![1_i64, 2], Validity::NonNullable).into_array();
        let args = VecExecutionArgs::new(vec![input], 2);
        let mut ctx = array_session().create_execution_ctx();

        let error = match execute_rows(&function, &EmptyOptions, &args, &mut ctx) {
            Err(error) => error,
            Ok(_) => vortex_error::vortex_bail!("dispatch must preserve its planned element types"),
        };

        assert!(
            error.to_string().contains("expected a u64 column"),
            "unexpected error: {error}",
        );
        Ok(())
    }

    #[track_caller]
    fn assert_arity_error(error: VortexError) {
        assert!(
            error.to_string().contains("requires arity 1, got 0"),
            "unexpected error: {error}",
        );
    }
}
