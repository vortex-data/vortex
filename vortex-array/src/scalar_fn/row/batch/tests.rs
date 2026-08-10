// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use rstest::rstest;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_session::registry::CachedId;

use super::super::execute::RowExecution;
use super::Batch;
use super::BatchPlan;
use super::RowPolicy;
use super::finalize_kernel_output;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::array_session;
use crate::arrays::BoolArray;
use crate::arrays::ConstantArray;
use crate::arrays::PrimitiveArray;
use crate::assert_arrays_eq;
use crate::dtype::DType;
use crate::dtype::NativePType;
use crate::dtype::Nullability;
use crate::scalar_fn::EmptyOptions;
use crate::scalar_fn::OutputSink;
use crate::scalar_fn::RowFn;
use crate::scalar_fn::RowVisitor;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::VecExecutionArgs;
use crate::validity::Validity;

#[derive(Clone)]
struct RetryConstantAdd;

#[derive(Clone)]
struct NullarySeven;

#[derive(Clone)]
struct OriginalInputReducer;

#[derive(Clone)]
struct DeferredOriginalReducer;

#[derive(Clone)]
struct PreparedAdd {
    visit: PreparedVisit,
    prepares: Arc<AtomicUsize>,
}

#[derive(Clone, Copy)]
enum PreparedVisit {
    Owned,
    Sink,
    Deferred,
}

struct I64Sink(BufferMut<i64>);

impl OutputSink for I64Sink {
    type Rows<'a> = &'a mut [i64];
    type Row<'a> = &'a mut i64;
    type WriteToken = ();

    fn sink_dtype(_args: &[DType]) -> VortexResult<DType> {
        Ok(DType::from(i64::PTYPE))
    }

    fn with_capacity(rows: usize, _dtype: &DType) -> VortexResult<Self> {
        Ok(Self(BufferMut::zeroed(rows)))
    }

    fn rows(&mut self) -> Self::Rows<'_> {
        self.0.as_mut_slice()
    }

    fn row_count_matches(rows: &Self::Rows<'_>, row_count: usize) -> bool {
        rows.len() == row_count
    }

    fn row<'a>(rows: &'a mut Self::Rows<'_>, index: usize) -> Self::Row<'a> {
        &mut rows[index]
    }

    unsafe fn finish(self) -> VortexResult<ArrayRef> {
        Ok(PrimitiveArray::new(self.0.freeze(), Validity::NonNullable).into_array())
    }
}

impl RowFn for NullarySeven {
    type Options = EmptyOptions;

    const ARG_NAMES: &'static [&'static str] = &[];

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("test.nullary_seven");
        *ID
    }

    fn dispatch<V: RowVisitor>(
        &self,
        _options: &Self::Options,
        _args: &[DType],
        visitor: V,
    ) -> VortexResult<V::VisitResult> {
        visitor.visit_into::<(), I64Sink, _>(|(), output| {
            *output = 7;
        })
    }
}

impl RowFn for RetryConstantAdd {
    type Options = EmptyOptions;

    const ARG_NAMES: &'static [&'static str] = &["lhs", "rhs"];
    const FALLIBLE: bool = true;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("test.retry_constant_add");
        *ID
    }

    fn dispatch<V: RowVisitor>(
        &self,
        _options: &Self::Options,
        _args: &[DType],
        visitor: V,
    ) -> VortexResult<V::VisitResult> {
        visitor.visit_deferred::<(u8, u8), u8, bool>(
            |(lhs, rhs)| lhs.overflowing_add(rhs),
            |failed| {
                if failed {
                    return Err(vortex_err!(InvalidArgument: "checked add overflowed"));
                }

                Ok(())
            },
        )
    }

    fn reduce_encoded(
        &self,
        _options: &Self::Options,
        args: &[ArrayRef],
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<RowExecution>> {
        if args[0].len() == 1 {
            return Ok(Some(RowExecution::Output(
                ConstantArray::new(0u8, args[0].len()).into_array(),
            )));
        }

        Ok(None)
    }
}

impl RowFn for OriginalInputReducer {
    type Options = EmptyOptions;

    const ARG_NAMES: &'static [&'static str] = &["value"];

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("test.original_input_reducer");
        *ID
    }

    fn dispatch<V: RowVisitor>(
        &self,
        _options: &Self::Options,
        _args: &[DType],
        visitor: V,
    ) -> VortexResult<V::VisitResult> {
        visitor.visit::<(i64,), i64>(|(value,)| value)
    }

    fn reduce_encoded(
        &self,
        _options: &Self::Options,
        args: &[ArrayRef],
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<RowExecution>> {
        if args[0].len() == 3 {
            return Ok(Some(RowExecution::Output(
                ConstantArray::new(42_i64, 3).into_array(),
            )));
        }

        Ok(None)
    }
}

impl RowFn for DeferredOriginalReducer {
    type Options = EmptyOptions;

    const ARG_NAMES: &'static [&'static str] = &["value"];
    const FALLIBLE: bool = true;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("test.deferred_original_reducer");
        *ID
    }

    fn dispatch<V: RowVisitor>(
        &self,
        _options: &Self::Options,
        _args: &[DType],
        visitor: V,
    ) -> VortexResult<V::VisitResult> {
        visitor.visit::<(i64,), i64>(|(value,)| value)
    }

    fn reduce_encoded(
        &self,
        _options: &Self::Options,
        _args: &[ArrayRef],
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<RowExecution>> {
        Ok(Some(RowExecution::DeferredError(vortex_err!(
            InvalidArgument: "encoded payload failed"
        ))))
    }
}

impl RowFn for PreparedAdd {
    type Options = EmptyOptions;

    const ARG_NAMES: &'static [&'static str] = &["lhs", "rhs"];
    const FALLIBLE: bool = true;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("test.prepared_add");
        *ID
    }

    fn dispatch<V: RowVisitor>(
        &self,
        _options: &Self::Options,
        _args: &[DType],
        visitor: V,
    ) -> VortexResult<V::VisitResult> {
        let prepares = Arc::clone(&self.prepares);
        let prepare = move |(_lhs, rhs): (Option<i64>, Option<i64>)| {
            prepares.fetch_add(1, Ordering::Relaxed);
            rhs
        };

        match self.visit {
            PreparedVisit::Owned => visitor
                .visit_prepared::<(i64, i64), i64, _>(prepare, |constant_rhs, (lhs, rhs)| {
                    lhs.wrapping_add(constant_rhs.unwrap_or(rhs))
                }),
            PreparedVisit::Sink => visitor.visit_prepared_into::<(i64, i64), I64Sink, _, ()>(
                prepare,
                |constant_rhs, (lhs, rhs), output| {
                    *output = lhs.wrapping_add(constant_rhs.unwrap_or(rhs));
                },
            ),
            PreparedVisit::Deferred => visitor.visit_prepared_deferred::<(i64, i64), i64, _, bool>(
                prepare,
                |constant_rhs, (lhs, rhs)| lhs.overflowing_add(constant_rhs.unwrap_or(rhs)),
                |failed| {
                    if failed {
                        return Err(vortex_err!(InvalidArgument: "prepared add overflowed"));
                    }

                    Ok(())
                },
            ),
        }
    }
}

#[test]
fn test_batch_rejects_input_length_mismatch() -> VortexResult<()> {
    static ID: CachedId = CachedId::new("test.row_batch");

    let input = PrimitiveArray::new(vec![1i64, 2], Validity::NonNullable).into_array();
    let args = VecExecutionArgs::new(vec![input], 3);
    let result = Batch::new(*ID, &args, |_| {
        Ok(BatchPlan {
            output_dtype: DType::from(i64::PTYPE),
            policy: RowPolicy::Dense,
        })
    });

    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_dense_retry_does_not_reduce_filtered_inputs() -> VortexResult<()> {
    let lhs =
        PrimitiveArray::new(vec![u8::MAX, 1], Validity::from_iter([true, false])).into_array();
    let rhs = ConstantArray::new(1u8, 2).into_array();
    let args = VecExecutionArgs::new(vec![lhs, rhs], 2);
    let mut ctx = array_session().create_execution_ctx();

    let result = ScalarFnVTable::execute(&RetryConstantAdd, &EmptyOptions, &args, &mut ctx);

    assert!(result.is_err());
    Ok(())
}

#[test]
fn test_dense_retry_suppresses_null_row_failure() -> VortexResult<()> {
    let lhs =
        PrimitiveArray::new(vec![1, u8::MAX], Validity::from_iter([true, false])).into_array();
    let rhs = ConstantArray::new(1_u8, 2).into_array();
    let args = VecExecutionArgs::new(vec![lhs, rhs], 2);
    let mut ctx = array_session().create_execution_ctx();

    let actual = ScalarFnVTable::execute(&RetryConstantAdd, &EmptyOptions, &args, &mut ctx)?;
    let expected = PrimitiveArray::new(vec![2_u8, 0], Validity::from_iter([true, false]));

    assert_arrays_eq!(&actual, expected.as_ref(), &mut ctx);
    Ok(())
}

#[test]
fn test_reduce_encoded_defers_errors_behind_nulls() -> VortexResult<()> {
    let input =
        PrimitiveArray::new(vec![10_i64, 20], Validity::from_iter([true, false])).into_array();
    let args = VecExecutionArgs::new(vec![input.clone()], 2);
    let mut ctx = array_session().create_execution_ctx();

    let actual = ScalarFnVTable::execute(&DeferredOriginalReducer, &EmptyOptions, &args, &mut ctx)?;

    assert_arrays_eq!(&actual, &input, &mut ctx);
    Ok(())
}

#[test]
fn test_reduce_encoded_precedes_constant_broadcast() -> VortexResult<()> {
    let input = ConstantArray::new(7_i64, 3).into_array();
    let args = VecExecutionArgs::new(vec![input], 3);
    let mut ctx = array_session().create_execution_ctx();

    let actual = ScalarFnVTable::execute(&OriginalInputReducer, &EmptyOptions, &args, &mut ctx)?;
    let expected = ConstantArray::new(42_i64, 3).into_array();

    assert_arrays_eq!(&actual, &expected, &mut ctx);
    Ok(())
}

#[test]
fn test_constant_input_broadcasts_one_row() -> VortexResult<()> {
    let input = ConstantArray::new(7_i64, 2).into_array();
    let args = VecExecutionArgs::new(vec![input.clone()], 2);
    let mut ctx = array_session().create_execution_ctx();

    let actual = ScalarFnVTable::execute(&OriginalInputReducer, &EmptyOptions, &args, &mut ctx)?;

    assert_arrays_eq!(&actual, &input, &mut ctx);
    Ok(())
}

#[rstest]
#[case::all_valid([true, true])]
#[case::all_invalid([false, false])]
fn test_resolve_validity_array_masks(#[case] validity: [bool; 2]) -> VortexResult<()> {
    static ID: CachedId = CachedId::new("test.resolve_validity");

    let validity = Validity::Array(BoolArray::from_iter(validity).into_array());
    let input = PrimitiveArray::new(vec![4_i64, 5], validity).into_array();
    let args = VecExecutionArgs::new(vec![input.clone()], 2);
    let batch = Batch::new(*ID, &args, |_| {
        Ok(BatchPlan {
            output_dtype: DType::from(i64::PTYPE),
            policy: RowPolicy::ValidOnly,
        })
    })?;
    let mut ctx = array_session().create_execution_ctx();

    let actual = batch.execute(
        |_args, _ctx| Ok(None),
        |args, _ctx| Ok(RowExecution::Output(args.arrays[0].clone())),
        |_args, _valid, _ctx| Ok(None),
        &mut ctx,
    )?;

    assert_arrays_eq!(&actual, &input, &mut ctx);
    Ok(())
}

#[test]
fn test_valid_only_filters_and_scatters() -> VortexResult<()> {
    static ID: CachedId = CachedId::new("test.filter_and_scatter");

    let input = PrimitiveArray::new(
        vec![10_i64, 20, 30, 40],
        Validity::from_iter([true, false, true, false]),
    )
    .into_array();
    let args = VecExecutionArgs::new(vec![input.clone()], 4);
    let batch = Batch::new(*ID, &args, |_| {
        Ok(BatchPlan {
            output_dtype: DType::from(i64::PTYPE),
            policy: RowPolicy::ValidOnly,
        })
    })?;
    let mut ctx = array_session().create_execution_ctx();

    let actual = batch.execute(
        |_args, _ctx| Ok(None),
        |args, _ctx| Ok(RowExecution::Output(args.arrays[0].clone())),
        |_args, _valid, _ctx| Ok(None),
        &mut ctx,
    )?;

    assert_arrays_eq!(&actual, &input, &mut ctx);
    Ok(())
}

#[test]
fn test_finalize_kernel_output_validates_shape_and_dtype() -> VortexResult<()> {
    static ID: CachedId = CachedId::new("test.finalize_kernel_output");

    let values = PrimitiveArray::from_iter([1_i64, 2]).into_array();
    let result_dtype = DType::Primitive(i64::PTYPE, Nullability::Nullable);
    let mut ctx = array_session().create_execution_ctx();

    let actual = finalize_kernel_output(*ID, &result_dtype, 2, values.clone())?;
    let expected = PrimitiveArray::new(vec![1_i64, 2], Validity::AllValid).into_array();
    assert_eq!(actual.dtype(), &result_dtype);
    assert_arrays_eq!(&actual, &expected, &mut ctx);

    assert!(finalize_kernel_output(*ID, &result_dtype, 3, values).is_err());

    let bools = BoolArray::from_iter([true, false]).into_array();
    assert!(finalize_kernel_output(*ID, &result_dtype, 2, bools).is_err());
    Ok(())
}

#[rstest]
#[case::owned_constant(PreparedVisit::Owned, true)]
#[case::owned_varying(PreparedVisit::Owned, false)]
#[case::sink_constant(PreparedVisit::Sink, true)]
#[case::sink_varying(PreparedVisit::Sink, false)]
#[case::deferred_constant(PreparedVisit::Deferred, true)]
#[case::deferred_varying(PreparedVisit::Deferred, false)]
fn test_prepared_visits(
    #[case] visit: PreparedVisit,
    #[case] constant_rhs: bool,
) -> VortexResult<()> {
    let lhs = PrimitiveArray::from_iter([1_i64, 2]).into_array();
    let rhs = if constant_rhs {
        ConstantArray::new(3_i64, 2).into_array()
    } else {
        PrimitiveArray::from_iter([3_i64, 4]).into_array()
    };
    let args = VecExecutionArgs::new(vec![lhs, rhs], 2);
    let prepares = Arc::new(AtomicUsize::new(0));
    let function = PreparedAdd {
        visit,
        prepares: Arc::clone(&prepares),
    };
    let mut ctx = array_session().create_execution_ctx();

    let actual = ScalarFnVTable::execute(&function, &EmptyOptions, &args, &mut ctx)?;
    let expected = if constant_rhs {
        PrimitiveArray::from_iter([4_i64, 5]).into_array()
    } else {
        PrimitiveArray::from_iter([4_i64, 6]).into_array()
    };

    assert_arrays_eq!(&actual, &expected, &mut ctx);
    assert_eq!(prepares.load(Ordering::Relaxed), 1);
    Ok(())
}

#[rstest]
#[case::dense(RowPolicy::Dense)]
#[case::dense_with_retry(RowPolicy::DenseWithRetry)]
#[case::valid_only(RowPolicy::ValidOnly)]
fn test_strategy_matrix(#[case] policy: RowPolicy) -> VortexResult<()> {
    static ID: CachedId = CachedId::new("test.row_strategy");

    let input = PrimitiveArray::new(vec![1i64, 2, 3], Validity::from_iter([true, false, true]))
        .into_array();
    let args = VecExecutionArgs::new(vec![input.clone()], 3);
    let batch = Batch::new(*ID, &args, |_| {
        Ok(BatchPlan {
            output_dtype: DType::from(i64::PTYPE),
            policy,
        })
    })?;
    let mut ctx = array_session().create_execution_ctx();

    let actual = batch.execute(
        |_args, _ctx| Ok(None),
        |args, _ctx| Ok(RowExecution::Output(args.arrays[0].clone())),
        |args, _valid, _ctx| Ok(Some(RowExecution::Output(args.arrays[0].clone()))),
        &mut ctx,
    )?;

    assert_arrays_eq!(&actual, &input, &mut ctx);
    Ok(())
}

#[test]
fn test_nullary_row_function_broadcasts() -> VortexResult<()> {
    let args = VecExecutionArgs::new(vec![], 3);
    let mut ctx = array_session().create_execution_ctx();

    let actual = ScalarFnVTable::execute(&NullarySeven, &EmptyOptions, &args, &mut ctx)?;
    let expected = PrimitiveArray::from_iter([7i64, 7, 7]).into_array();

    assert_arrays_eq!(&actual, &expected, &mut ctx);
    Ok(())
}
