// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use rstest::rstest;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_session::registry::CachedId;

use super::finalize_kernel_output;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::array_session;
use crate::arrays::BoolArray;
use crate::arrays::ConstantArray;
use crate::arrays::ExtensionArray;
use crate::arrays::FixedSizeListArray;
use crate::arrays::PrimitiveArray;
use crate::assert_arrays_eq;
use crate::dtype::DType;
use crate::dtype::NativePType;
use crate::dtype::Nullability;
use crate::dtype::extension::ExtDTypeRef;
use crate::extension::datetime::TimeUnit;
use crate::extension::datetime::Timestamp;
use crate::scalar::Scalar;
use crate::scalar_fn::EmptyOptions;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::VecExecutionArgs;
use crate::scalar_fn::unstable::row::FixedSizeListSink;
use crate::scalar_fn::unstable::row::InitializedRow;
use crate::scalar_fn::unstable::row::InputElement;
use crate::scalar_fn::unstable::row::OutputElement;
use crate::scalar_fn::unstable::row::OutputSink;
use crate::scalar_fn::unstable::row::RowFn;
use crate::scalar_fn::unstable::row::RowVisitor;
use crate::scalar_fn::unstable::row::execute_rows;
use crate::scalar_fn::unstable::row::row_fn_return_dtype;
use crate::validity::Validity;

#[derive(Clone, Default)]
struct DeferredAdd {
    /// The number of preparations across the dense attempt and any valid-row retry.
    prepare_count: Arc<AtomicUsize>,
}

impl DeferredAdd {
    fn prepare_count(&self) -> usize {
        self.prepare_count.load(Ordering::Relaxed)
    }
}

#[derive(Clone)]
struct ValidOnlyIdentity;

#[derive(Clone)]
struct FilterAndScatterIdentity;

#[derive(Clone)]
struct DenseRetryIncrement;

#[derive(Clone)]
struct FilterAndScatterRepeat;

#[derive(Clone)]
struct InvalidKernelOutput;

#[derive(Clone)]
struct PackedPositive;

#[derive(Clone)]
struct PackedGreaterThan;

#[derive(Clone)]
struct ValidOnlyPositive;

#[derive(Clone)]
struct NullaryTrue;

struct ValidOnlyI64;

struct FilterOnlyI64;

struct DenseRetryI64;

// SAFETY: the view is a slice, and its reported length is the buffer length.
unsafe impl InputElement for ValidOnlyI64 {
    type Column = Buffer<i64>;
    type View<'a> = &'a [i64];
    type Elem<'a> = i64;

    const DENSE_SAFE: bool = false;
    const DECODE_INFALLIBLE: bool = true;

    fn validate(dtype: &DType) -> VortexResult<()> {
        <i64 as InputElement>::validate(dtype)
    }

    fn decode(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self::Column> {
        <i64 as InputElement>::decode(array, ctx)
    }

    fn can_decode_null_tolerant(_array: &ArrayRef) -> VortexResult<bool> {
        Ok(true)
    }

    fn get(column: &Self::Column, index: usize) -> Self::Elem<'_> {
        column[index]
    }

    fn view(column: &Self::Column) -> Self::View<'_> {
        column.as_slice()
    }

    fn get_from_view<'a>(view: &Self::View<'a>, index: usize) -> Self::Elem<'a> {
        view[index]
    }
}

// SAFETY: the view is a slice, and its reported length is the buffer length.
unsafe impl InputElement for FilterOnlyI64 {
    type Column = Buffer<i64>;
    type View<'a> = &'a [i64];
    type Elem<'a> = i64;

    const DENSE_SAFE: bool = false;
    const DECODE_INFALLIBLE: bool = false;

    fn validate(dtype: &DType) -> VortexResult<()> {
        <i64 as InputElement>::validate(dtype)
    }

    fn decode(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self::Column> {
        let values = <i64 as InputElement>::decode(array, ctx)?;
        vortex_ensure!(
            !values.as_slice().contains(&i64::MIN),
            "test input contains an invalid payload",
        );

        Ok(values)
    }

    fn get(column: &Self::Column, index: usize) -> Self::Elem<'_> {
        column[index]
    }

    fn view(column: &Self::Column) -> Self::View<'_> {
        column.as_slice()
    }

    fn get_from_view<'a>(view: &Self::View<'a>, index: usize) -> Self::Elem<'a> {
        view[index]
    }
}

// SAFETY: the view is a slice, and its reported length is the buffer length.
unsafe impl InputElement for DenseRetryI64 {
    type Column = Buffer<i64>;
    type View<'a> = &'a [i64];
    type Elem<'a> = i64;

    const DENSE_SAFE: bool = true;
    const DECODE_INFALLIBLE: bool = true;

    fn validate(dtype: &DType) -> VortexResult<()> {
        <i64 as InputElement>::validate(dtype)
    }

    fn decode(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self::Column> {
        <i64 as InputElement>::decode(array, ctx)
    }

    fn get(column: &Self::Column, index: usize) -> Self::Elem<'_> {
        column[index]
    }

    fn view(column: &Self::Column) -> Self::View<'_> {
        column.as_slice()
    }

    fn get_from_view<'a>(view: &Self::View<'a>, index: usize) -> Self::Elem<'a> {
        view[index]
    }
}

/// Produces a null row to exercise output validation at the row-function boundary.
#[derive(Default)]
struct NullProducingI64(i64);

impl OutputElement for NullProducingI64 {
    fn element_dtype() -> DType {
        DType::from(i64::PTYPE)
    }

    fn build(values: Vec<Self>) -> ArrayRef {
        let values: Vec<_> = values.into_iter().map(|value| value.0).collect();
        let validity = Validity::from_iter((0..values.len()).map(|index| index != 0));

        PrimitiveArray::new(values, validity).into_array()
    }
}

struct I64Sink(BufferMut<i64>);

// SAFETY: every row is initialized by `BufferMut::zeroed`, and the sink exposes exactly that
// initialized slice. The `()` write token therefore proves no additional invariant.
unsafe impl OutputSink for I64Sink {
    type Params = ();
    type Rows<'a> = &'a mut [i64];
    type Row<'a> = &'a mut i64;
    type WriteToken = ();

    fn storage_dtype(_params: &Self::Params) -> DType {
        DType::from(i64::PTYPE)
    }

    fn with_capacity(rows: usize, _params: &Self::Params) -> VortexResult<Self> {
        Ok(Self(BufferMut::zeroed(rows)))
    }

    fn rows(&mut self) -> Self::Rows<'_> {
        self.0.as_mut_slice()
    }

    unsafe fn row_unchecked<'a>(rows: &'a mut Self::Rows<'_>, index: usize) -> Self::Row<'a> {
        // SAFETY: required by this method's contract.
        unsafe { rows.get_unchecked_mut(index) }
    }

    unsafe fn finish(self) -> VortexResult<ArrayRef> {
        Ok(PrimitiveArray::new(self.0.freeze(), Validity::NonNullable).into_array())
    }
}

#[derive(Clone)]
struct RepeatValue;

impl RowFn for RepeatValue {
    type Options = usize;

    const ARG_NAMES: &'static [&'static str] = &["value"];
    const INFALLIBLE: bool = true;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("test.repeat_value");
        *ID
    }

    fn dispatch<V: RowVisitor>(
        &self,
        width: &Self::Options,
        _args: &[DType],
        visitor: V,
    ) -> VortexResult<V::VisitResult> {
        vortex_ensure!(
            u32::try_from(*width).is_ok(),
            InvalidArgument:
            "test.repeat_value width must fit in the fixed-size-list u32 list size, got {width}",
        );

        visitor.visit_into::<(i64,), FixedSizeListSink<i64>, _>(*width, |(value,), row| {
            InitializedRow::fill(row, |_| value)
        })
    }
}

impl RowFn for DeferredAdd {
    type Options = EmptyOptions;

    const ARG_NAMES: &'static [&'static str] = &["lhs", "rhs"];
    const INFALLIBLE: bool = false;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("test.deferred_add");
        *ID
    }

    fn dispatch<V: RowVisitor>(
        &self,
        _options: &Self::Options,
        _args: &[DType],
        visitor: V,
    ) -> VortexResult<V::VisitResult> {
        let prepare_count = Arc::clone(&self.prepare_count);

        visitor.visit_prepared_deferred::<(i64, i64), i64, (), bool>(
            move |_| {
                prepare_count.fetch_add(1, Ordering::Relaxed);
            },
            |&(), (lhs, rhs)| lhs.overflowing_add(rhs),
            |overflowed| {
                if overflowed {
                    vortex_bail!(InvalidArgument: "deferred addition overflowed");
                }

                Ok(())
            },
        )
    }
}

impl RowFn for ValidOnlyIdentity {
    type Options = EmptyOptions;

    const ARG_NAMES: &'static [&'static str] = &["value"];
    const INFALLIBLE: bool = false;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("test.valid_only_identity");
        *ID
    }

    fn dispatch<V: RowVisitor>(
        &self,
        _options: &Self::Options,
        _args: &[DType],
        visitor: V,
    ) -> VortexResult<V::VisitResult> {
        visitor.visit_into::<(i64,), I64Sink, VortexResult<()>>((), |(value,), output| {
            *output = value;
            Ok(())
        })
    }
}

impl RowFn for FilterAndScatterIdentity {
    type Options = EmptyOptions;

    const ARG_NAMES: &'static [&'static str] = &["value"];
    const INFALLIBLE: bool = false;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("test.filter_and_scatter_identity");
        *ID
    }

    fn dispatch<V: RowVisitor>(
        &self,
        _options: &Self::Options,
        _args: &[DType],
        visitor: V,
    ) -> VortexResult<V::VisitResult> {
        visitor.visit::<(FilterOnlyI64,), i64>(|(value,)| value)
    }
}

impl RowFn for DenseRetryIncrement {
    type Options = EmptyOptions;

    const ARG_NAMES: &'static [&'static str] = &["value"];
    const INFALLIBLE: bool = false;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("test.dense_retry_increment");
        *ID
    }

    fn dispatch<V: RowVisitor>(
        &self,
        _options: &Self::Options,
        _args: &[DType],
        visitor: V,
    ) -> VortexResult<V::VisitResult> {
        visitor.visit_deferred::<(DenseRetryI64,), i64, bool>(
            |(value,)| value.overflowing_add(1),
            |overflowed| {
                if overflowed {
                    vortex_bail!(InvalidArgument: "deferred increment overflowed");
                }

                Ok(())
            },
        )
    }
}

impl RowFn for FilterAndScatterRepeat {
    type Options = usize;

    const ARG_NAMES: &'static [&'static str] = &["value"];
    const INFALLIBLE: bool = true;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("test.filter_and_scatter_repeat");
        *ID
    }

    fn dispatch<V: RowVisitor>(
        &self,
        width: &Self::Options,
        _args: &[DType],
        visitor: V,
    ) -> VortexResult<V::VisitResult> {
        vortex_ensure!(
            u32::try_from(*width).is_ok(),
            InvalidArgument:
            "test.filter_and_scatter_repeat width must fit in u32, got {width}",
        );

        visitor
            .visit_into::<(FilterOnlyI64,), FixedSizeListSink<i64>, _>(*width, |(value,), row| {
                InitializedRow::fill(row, |_| value)
            })
    }
}

impl RowFn for InvalidKernelOutput {
    type Options = EmptyOptions;

    const ARG_NAMES: &'static [&'static str] = &["value"];
    const INFALLIBLE: bool = true;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("test.invalid_kernel_output");
        *ID
    }

    fn dispatch<V: RowVisitor>(
        &self,
        _options: &Self::Options,
        _args: &[DType],
        visitor: V,
    ) -> VortexResult<V::VisitResult> {
        visitor.visit::<(i64,), NullProducingI64>(|(value,)| NullProducingI64(value))
    }
}

#[rstest]
#[case::dense_width_two(
    vec![1_i64, 2],
    Validity::NonNullable,
    2,
    vec![1_i64, 1, 2, 2],
)]
#[case::dense_width_four(
    vec![3_i64, 4],
    Validity::NonNullable,
    4,
    vec![3_i64, 3, 3, 3, 4, 4, 4, 4],
)]
#[case::empty(vec![], Validity::NonNullable, 3, vec![])]
#[case::zero_width(vec![3_i64, 4], Validity::NonNullable, 0, vec![])]
#[case::all_null(
    vec![5_i64, 6],
    Validity::AllInvalid,
    2,
    vec![0_i64, 0, 0, 0],
)]
#[case::partially_valid(
    vec![7_i64, 8, 9],
    Validity::from_iter([true, false, true]),
    3,
    vec![7_i64, 7, 7, 0, 0, 0, 9, 9, 9],
)]
fn test_fixed_size_list_sink_uses_runtime_width(
    #[case] input_values: Vec<i64>,
    #[case] validity: Validity,
    #[case] width: usize,
    #[case] expected_elements: Vec<i64>,
) -> VortexResult<()> {
    let row_count = input_values.len();
    let input = PrimitiveArray::new(input_values, validity.clone()).into_array();
    let args = VecExecutionArgs::new(vec![input], row_count);
    let mut ctx = array_session().create_execution_ctx();

    let actual = execute_rows(&RepeatValue, &width, &args, &mut ctx)?;
    let expected = FixedSizeListArray::new(
        PrimitiveArray::from_iter(expected_elements).into_array(),
        u32::try_from(width)
            .map_err(|_| vortex_err!(InvalidArgument: "test width must fit in u32, got {width}"))?,
        validity,
        row_count,
    )
    .into_array();

    assert_arrays_eq!(&actual, &expected, &mut ctx);
    Ok(())
}

impl RowFn for PackedPositive {
    type Options = EmptyOptions;

    const ARG_NAMES: &'static [&'static str] = &["value"];
    const INFALLIBLE: bool = true;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("test.packed_positive");
        *ID
    }

    fn dispatch<V: RowVisitor>(
        &self,
        _options: &Self::Options,
        _args: &[DType],
        visitor: V,
    ) -> VortexResult<V::VisitResult> {
        visitor.visit::<(i64,), bool>(|(value,)| value > 0)
    }
}

impl RowFn for PackedGreaterThan {
    type Options = EmptyOptions;

    const ARG_NAMES: &'static [&'static str] = &["lhs", "rhs"];
    const INFALLIBLE: bool = true;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("test.packed_greater_than");
        *ID
    }

    fn dispatch<V: RowVisitor>(
        &self,
        _options: &Self::Options,
        _args: &[DType],
        visitor: V,
    ) -> VortexResult<V::VisitResult> {
        visitor.visit::<(i64, i64), bool>(|(lhs, rhs)| lhs > rhs)
    }
}

impl RowFn for ValidOnlyPositive {
    type Options = EmptyOptions;

    const ARG_NAMES: &'static [&'static str] = &["value"];
    const INFALLIBLE: bool = true;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("test.valid_only_positive");
        *ID
    }

    fn dispatch<V: RowVisitor>(
        &self,
        _options: &Self::Options,
        _args: &[DType],
        visitor: V,
    ) -> VortexResult<V::VisitResult> {
        visitor.visit::<(ValidOnlyI64,), bool>(|(value,)| {
            assert_ne!(value, i64::MIN, "an invalid row was evaluated");
            value > 0
        })
    }
}

impl RowFn for NullaryTrue {
    type Options = EmptyOptions;

    const ARG_NAMES: &'static [&'static str] = &[];
    const INFALLIBLE: bool = true;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("test.nullary_true");
        *ID
    }

    fn dispatch<V: RowVisitor>(
        &self,
        _options: &Self::Options,
        _args: &[DType],
        visitor: V,
    ) -> VortexResult<V::VisitResult> {
        visitor.visit::<(), bool>(|()| true)
    }
}

#[test]
fn test_finalize_kernel_output_rejects_nested_dtype_mismatch() -> VortexResult<()> {
    static ID: CachedId = CachedId::new("test.finalize_kernel_output");

    let element_dtype = DType::Primitive(i64::PTYPE, Nullability::NonNullable);
    let values = ConstantArray::new(
        Scalar::list_empty(Arc::new(element_dtype), Nullability::NonNullable),
        2,
    )
    .into_array();
    let result_dtype = DType::List(
        Arc::new(DType::Primitive(i64::PTYPE, Nullability::Nullable)),
        Nullability::NonNullable,
    );
    let mut ctx = array_session().create_execution_ctx();

    assert!(finalize_kernel_output(*ID, &result_dtype, 2, values, &mut ctx).is_err());
    Ok(())
}

#[test]
fn test_kernel_output_rejects_nulls_at_function_boundary() -> VortexResult<()> {
    let input = PrimitiveArray::new(vec![1_i64, 2], Validity::NonNullable).into_array();
    let args = VecExecutionArgs::new(vec![input], 2);
    let mut ctx = array_session().create_execution_ctx();
    let execution = execute_rows(&InvalidKernelOutput, &EmptyOptions, &args, &mut ctx);
    let error = match execution {
        Err(error) => error,
        Ok(output) => match output.execute::<PrimitiveArray>(&mut ctx) {
            Err(error) => error,
            Ok(_) => vortex_bail!("an invalid row kernel output passed boundary validation"),
        },
    };
    let error = error.to_string();

    assert!(
        error.contains("test.invalid_kernel_output"),
        "the boundary error must name the function, got {error}",
    );
    assert!(
        error.contains("row kernel must produce only valid rows"),
        "the boundary error must identify invalid row output, got {error}",
    );
    Ok(())
}

#[test]
fn test_bool_output_builds_packed_values() -> VortexResult<()> {
    let input = PrimitiveArray::new(
        vec![1_i64, -1, 2, 0, 3],
        Validity::from_iter([true, true, false, true, true]),
    )
    .into_array();
    let args = VecExecutionArgs::new(vec![input], 5);
    let mut ctx = array_session().create_execution_ctx();

    let actual = execute_rows(&PackedPositive, &EmptyOptions, &args, &mut ctx)?;
    let expected =
        BoolArray::from_iter([Some(true), Some(false), None, Some(false), Some(true)]).into_array();

    assert_arrays_eq!(&actual, &expected, &mut ctx);
    Ok(())
}

#[rstest]
#[case::empty(0)]
#[case::one(1)]
#[case::below_word(63)]
#[case::exact_word(64)]
#[case::above_word(65)]
#[case::multiword_remainder(130)]
fn test_bool_output_word_boundaries(#[case] len: usize) -> VortexResult<()> {
    let values: Vec<_> = (0..len).map(|index| index as i64 - 32).collect();
    let input = PrimitiveArray::from_iter(values.iter().copied()).into_array();
    let args = VecExecutionArgs::new(vec![input], len);
    let mut ctx = array_session().create_execution_ctx();

    let actual = execute_rows(&PackedPositive, &EmptyOptions, &args, &mut ctx)?;
    let expected = BoolArray::from_iter(values.iter().map(|value| *value > 0)).into_array();

    assert_arrays_eq!(&actual, &expected, &mut ctx);
    Ok(())
}

#[test]
fn test_bool_output_handles_partial_constants() -> VortexResult<()> {
    let values: Vec<_> = (0_i64..65).collect();
    let varying = PrimitiveArray::from_iter(values.iter().copied()).into_array();
    let constant = ConstantArray::new(32_i64, values.len()).into_array();
    let mut ctx = array_session().create_execution_ctx();

    let lhs_varying = VecExecutionArgs::new(vec![varying.clone(), constant.clone()], values.len());
    let actual = execute_rows(&PackedGreaterThan, &EmptyOptions, &lhs_varying, &mut ctx)?;
    let expected = BoolArray::from_iter(values.iter().map(|value| *value > 32)).into_array();
    assert_arrays_eq!(&actual, &expected, &mut ctx);

    let rhs_varying = VecExecutionArgs::new(vec![constant, varying], values.len());
    let actual = execute_rows(&PackedGreaterThan, &EmptyOptions, &rhs_varying, &mut ctx)?;
    let expected = BoolArray::from_iter(values.iter().map(|value| 32 > *value)).into_array();
    assert_arrays_eq!(&actual, &expected, &mut ctx);

    Ok(())
}

#[test]
fn test_bool_output_handles_all_constant_input() -> VortexResult<()> {
    let input = ConstantArray::new(7_i64, 65).into_array();
    let args = VecExecutionArgs::new(vec![input], 65);
    let mut ctx = array_session().create_execution_ctx();

    let actual = execute_rows(&PackedPositive, &EmptyOptions, &args, &mut ctx)?;
    let expected = BoolArray::from_iter(std::iter::repeat_n(true, 65)).into_array();

    assert_arrays_eq!(&actual, &expected, &mut ctx);
    Ok(())
}

#[test]
fn test_bool_output_handles_nullary_rows() -> VortexResult<()> {
    let args = VecExecutionArgs::new(vec![], 65);
    let mut ctx = array_session().create_execution_ctx();

    let actual = execute_rows(&NullaryTrue, &EmptyOptions, &args, &mut ctx)?;
    let expected = BoolArray::from_iter(std::iter::repeat_n(true, 65)).into_array();

    assert_arrays_eq!(&actual, &expected, &mut ctx);
    Ok(())
}

#[test]
fn test_valid_only_bool_output_skips_invalid_rows() -> VortexResult<()> {
    let input = PrimitiveArray::new(
        vec![1_i64, i64::MIN, -1],
        Validity::from_iter([true, false, true]),
    )
    .into_array();
    let args = VecExecutionArgs::new(vec![input], 3);
    let mut ctx = array_session().create_execution_ctx();

    let actual = execute_rows(&ValidOnlyPositive, &EmptyOptions, &args, &mut ctx)?;
    let expected = BoolArray::from_iter([Some(true), None, Some(false)]).into_array();

    assert_arrays_eq!(&actual, &expected, &mut ctx);
    Ok(())
}

#[test]
fn test_filter_and_scatter_skips_invalid_decode_payloads() -> VortexResult<()> {
    let validity = Validity::from_iter([false, true, false, true]);
    let input =
        PrimitiveArray::new(vec![i64::MIN, 10, i64::MIN, 30], validity.clone()).into_array();
    let args = VecExecutionArgs::new(vec![input], 4);
    let mut ctx = array_session().create_execution_ctx();

    let actual = execute_rows(&FilterAndScatterIdentity, &EmptyOptions, &args, &mut ctx)?;
    let expected = PrimitiveArray::new(vec![0_i64, 10, 0, 30], validity).into_array();

    assert_arrays_eq!(&actual, &expected, &mut ctx);
    Ok(())
}

#[rstest]
#[case::width_two(2, vec![0_i64, 0, 7, 7])]
#[case::zero_width(0, vec![])]
fn test_filter_and_scatter_preserves_runtime_sink_params(
    #[case] width: usize,
    #[case] expected_elements: Vec<i64>,
) -> VortexResult<()> {
    let validity = Validity::from_iter([false, true]);
    let input = PrimitiveArray::new(vec![i64::MIN, 7], validity.clone()).into_array();
    let args = VecExecutionArgs::new(vec![input], 2);
    let mut ctx = array_session().create_execution_ctx();

    let actual = execute_rows(&FilterAndScatterRepeat, &width, &args, &mut ctx)?;
    let expected = FixedSizeListArray::new(
        PrimitiveArray::from_iter(expected_elements).into_array(),
        u32::try_from(width)
            .map_err(|_| vortex_err!(InvalidArgument: "test width must fit in u32, got {width}"))?,
        validity,
        2,
    )
    .into_array();

    assert_arrays_eq!(&actual, &expected, &mut ctx);
    Ok(())
}

#[test]
fn test_deferred_owned_execution_handles_constant_lhs() -> VortexResult<()> {
    let lhs = ConstantArray::new(10_i64, 3).into_array();
    let rhs = PrimitiveArray::from_iter([1_i64, 2, 3]).into_array();
    let args = VecExecutionArgs::new(vec![lhs, rhs], 3);
    let mut ctx = array_session().create_execution_ctx();

    let actual = execute_rows(&DeferredAdd::default(), &EmptyOptions, &args, &mut ctx)?;
    let expected = PrimitiveArray::from_iter([11_i64, 12, 13]).into_array();

    assert_arrays_eq!(&actual, &expected, &mut ctx);
    Ok(())
}

#[test]
fn test_deferred_owned_execution_retries_null_row_failure() -> VortexResult<()> {
    let function = DeferredAdd::default();
    let validity = Validity::from_iter([true, false]);
    let lhs = PrimitiveArray::new(vec![1_i64, i64::MAX], validity.clone()).into_array();
    let rhs = ConstantArray::new(1_i64, 2).into_array();
    let args = VecExecutionArgs::new(vec![lhs, rhs], 2);
    let mut ctx = array_session().create_execution_ctx();

    let actual = execute_rows(&function, &EmptyOptions, &args, &mut ctx)?;
    let expected = PrimitiveArray::new(vec![2_i64, 0], validity).into_array();

    assert_arrays_eq!(&actual, &expected, &mut ctx);
    assert_eq!(function.prepare_count(), 2);
    Ok(())
}

#[test]
fn test_dense_retry_filters_when_direct_valid_rows_are_unavailable() -> VortexResult<()> {
    let validity = Validity::from_iter([true, false, true]);
    let input = PrimitiveArray::new(vec![1_i64, i64::MAX, 3], validity.clone()).into_array();
    let args = VecExecutionArgs::new(vec![input], 3);
    let mut ctx = array_session().create_execution_ctx();

    let actual = execute_rows(&DenseRetryIncrement, &EmptyOptions, &args, &mut ctx)?;
    let expected = PrimitiveArray::new(vec![2_i64, 0, 4], validity).into_array();

    assert_arrays_eq!(&actual, &expected, &mut ctx);
    Ok(())
}

#[test]
fn test_deferred_owned_execution_does_not_retry_partially_valid_success() -> VortexResult<()> {
    let function = DeferredAdd::default();
    let validity = Validity::from_iter([true, false]);
    let lhs = PrimitiveArray::new(vec![1_i64, 2], validity.clone()).into_array();
    let rhs = ConstantArray::new(1_i64, 2).into_array();
    let args = VecExecutionArgs::new(vec![lhs, rhs], 2);
    let mut ctx = array_session().create_execution_ctx();

    let actual = execute_rows(&function, &EmptyOptions, &args, &mut ctx)?;
    let expected = PrimitiveArray::new(vec![2_i64, 0], validity).into_array();

    assert_arrays_eq!(&actual, &expected, &mut ctx);
    assert_eq!(function.prepare_count(), 1);
    Ok(())
}

#[test]
fn test_deferred_owned_execution_retries_and_reports_valid_row_failure() -> VortexResult<()> {
    let function = DeferredAdd::default();
    let validity = Validity::from_iter([true, false]);
    let lhs = PrimitiveArray::new(vec![i64::MAX, 1], validity).into_array();
    let rhs = ConstantArray::new(1_i64, 2).into_array();
    let args = VecExecutionArgs::new(vec![lhs, rhs], 2);
    let mut ctx = array_session().create_execution_ctx();

    let error = execute_rows(&function, &EmptyOptions, &args, &mut ctx)
        .expect_err("a valid-row overflow must remain observable");
    let error = error.to_string();

    assert!(
        error.contains("deferred addition overflowed"),
        "the valid-row retry must report its deferred error, got {error}",
    );
    assert_eq!(function.prepare_count(), 2);
    Ok(())
}

#[test]
fn test_deferred_owned_array_backed_all_valid_error_does_not_retry() -> VortexResult<()> {
    let function = DeferredAdd::default();
    let validity = Validity::Array(ConstantArray::new(true, 2).into_array());
    let lhs = PrimitiveArray::new(vec![i64::MAX, 1], validity).into_array();
    let rhs = ConstantArray::new(1_i64, 2).into_array();
    let args = VecExecutionArgs::new(vec![lhs, rhs], 2);
    let mut ctx = array_session().create_execution_ctx();

    let error = execute_rows(&function, &EmptyOptions, &args, &mut ctx)
        .expect_err("an all-valid deferred error must remain observable");
    let error = error.to_string();

    assert!(
        error.contains("deferred addition overflowed"),
        "the dense attempt must return its original deferred error, got {error}",
    );
    assert_eq!(function.prepare_count(), 1);
    Ok(())
}

#[test]
fn test_valid_only_empty_batch_preserves_nonnullable_dtype() -> VortexResult<()> {
    let input = PrimitiveArray::from_iter(std::iter::empty::<i64>()).into_array();
    let args = VecExecutionArgs::new(vec![input], 0);
    let mut ctx = array_session().create_execution_ctx();

    let actual = execute_rows(&ValidOnlyIdentity, &EmptyOptions, &args, &mut ctx)?;

    assert_eq!(actual.len(), 0);
    assert_eq!(actual.dtype(), &DType::from(i64::PTYPE));
    Ok(())
}

/// A timestamp extension dtype over `i64` storage of the given nullability.
fn timestamp_ext(nullability: Nullability) -> ExtDTypeRef {
    Timestamp::new(TimeUnit::Seconds, nullability).erased()
}

/// The output dtype a declaring dispatch labels onto its `i64` storage.
fn timestamp_dtype(nullability: Nullability) -> DType {
    DType::Extension(timestamp_ext(nullability))
}

/// Build the timestamp column a labelled dispatch is expected to produce.
fn expected_timestamps(values: Vec<i64>, validity: Validity) -> VortexResult<ArrayRef> {
    let storage = PrimitiveArray::new(values, validity).into_array();
    let ext_dtype = timestamp_ext(storage.dtype().nullability());

    Ok(ExtensionArray::try_new(ext_dtype, storage)?.into_array())
}

/// Returns its input unchanged under the output dtype each dispatch declares.
///
/// `declared` is `None` to leave the storage dtype in place, so one function covers both the
/// labelled and unlabelled paths and every rejected label.
#[derive(Clone)]
struct DeclaredOutput {
    declared: Option<DType>,
}

impl DeclaredOutput {
    fn timestamps() -> Self {
        Self {
            declared: Some(timestamp_dtype(Nullability::NonNullable)),
        }
    }
}

impl RowFn for DeclaredOutput {
    type Options = EmptyOptions;

    const ARG_NAMES: &'static [&'static str] = &["value"];
    const INFALLIBLE: bool = true;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("test.declared_output");
        *ID
    }

    fn dispatch<V: RowVisitor>(
        &self,
        _options: &Self::Options,
        _args: &[DType],
        visitor: V,
    ) -> VortexResult<V::VisitResult> {
        let visitor = match &self.declared {
            Some(declared) => visitor.with_output_dtype(declared.clone()),
            None => visitor,
        };

        visitor.visit::<(i64,), i64>(|(value,)| value)
    }
}

/// Labels the output of a sink-writing dispatch, which plans [`RowPolicy::ValidOnly`].
#[derive(Clone)]
struct DeclaredSinkOutput;

impl RowFn for DeclaredSinkOutput {
    type Options = EmptyOptions;

    const ARG_NAMES: &'static [&'static str] = &["value"];
    const INFALLIBLE: bool = false;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("test.declared_sink_output");
        *ID
    }

    fn dispatch<V: RowVisitor>(
        &self,
        _options: &Self::Options,
        _args: &[DType],
        visitor: V,
    ) -> VortexResult<V::VisitResult> {
        visitor
            .with_output_dtype(timestamp_dtype(Nullability::NonNullable))
            .visit_into::<(i64,), I64Sink, VortexResult<()>>((), |(value,), output| {
                *output = value;
                Ok(())
            })
    }
}

/// Declares a different output dtype on its second dispatch, which execution must reject.
#[derive(Clone)]
struct ChangingOutputDType {
    dispatches: Arc<AtomicUsize>,
}

impl RowFn for ChangingOutputDType {
    type Options = EmptyOptions;

    const ARG_NAMES: &'static [&'static str] = &["value"];
    const INFALLIBLE: bool = true;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("test.changing_output_dtype");
        *ID
    }

    fn dispatch<V: RowVisitor>(
        &self,
        _options: &Self::Options,
        _args: &[DType],
        visitor: V,
    ) -> VortexResult<V::VisitResult> {
        let visitor = if self.dispatches.fetch_add(1, Ordering::Relaxed) == 0 {
            visitor.with_output_dtype(timestamp_dtype(Nullability::NonNullable))
        } else {
            visitor
        };

        visitor.visit::<(i64,), i64>(|(value,)| value)
    }
}

#[rstest]
#[case::all_valid(Validity::NonNullable)]
#[case::partially_valid(Validity::from_iter([true, false, true]))]
#[case::array_backed_all_valid(Validity::Array(ConstantArray::new(true, 3).into_array()))]
fn test_declared_output_dtype_labels_batch(#[case] validity: Validity) -> VortexResult<()> {
    let values = vec![1_i64, 2, 3];
    let input = PrimitiveArray::new(values.clone(), validity.clone()).into_array();
    let args = VecExecutionArgs::new(vec![input], 3);
    let mut ctx = array_session().create_execution_ctx();

    let actual = execute_rows(
        &DeclaredOutput::timestamps(),
        &EmptyOptions,
        &args,
        &mut ctx,
    )?;
    let expected = expected_timestamps(values, validity)?;

    assert_arrays_eq!(&actual, &expected, &mut ctx);
    Ok(())
}

#[rstest]
#[case::all_valid(Validity::NonNullable)]
#[case::partially_valid(Validity::from_iter([true, false, true]))]
fn test_declared_output_dtype_labels_sink_output(#[case] validity: Validity) -> VortexResult<()> {
    let values = vec![1_i64, 2, 3];
    let input = PrimitiveArray::new(values.clone(), validity.clone()).into_array();
    let args = VecExecutionArgs::new(vec![input], 3);
    let mut ctx = array_session().create_execution_ctx();

    let actual = execute_rows(&DeclaredSinkOutput, &EmptyOptions, &args, &mut ctx)?;
    let expected = expected_timestamps(values, validity)?;

    assert_arrays_eq!(&actual, &expected, &mut ctx);
    Ok(())
}

#[test]
fn test_declared_output_dtype_labels_empty_batch() -> VortexResult<()> {
    let input = PrimitiveArray::from_iter(std::iter::empty::<i64>()).into_array();
    let args = VecExecutionArgs::new(vec![input], 0);
    let mut ctx = array_session().create_execution_ctx();

    let actual = execute_rows(
        &DeclaredOutput::timestamps(),
        &EmptyOptions,
        &args,
        &mut ctx,
    )?;

    assert_eq!(actual.len(), 0);
    assert_eq!(actual.dtype(), &timestamp_dtype(Nullability::NonNullable));
    Ok(())
}

#[test]
fn test_declared_output_dtype_labels_all_null_batch() -> VortexResult<()> {
    let null_i64 = Scalar::null(DType::Primitive(i64::PTYPE, Nullability::Nullable));
    let input = ConstantArray::new(null_i64, 3).into_array();
    let args = VecExecutionArgs::new(vec![input], 3);
    let mut ctx = array_session().create_execution_ctx();

    let actual = execute_rows(
        &DeclaredOutput::timestamps(),
        &EmptyOptions,
        &args,
        &mut ctx,
    )?;
    let expected =
        ConstantArray::new(Scalar::null(timestamp_dtype(Nullability::Nullable)), 3).into_array();

    assert_arrays_eq!(&actual, &expected, &mut ctx);
    Ok(())
}

#[test]
fn test_declared_output_dtype_labels_constant_batch() -> VortexResult<()> {
    let input = ConstantArray::new(7_i64, 3).into_array();
    let args = VecExecutionArgs::new(vec![input], 3);
    let mut ctx = array_session().create_execution_ctx();

    let actual = execute_rows(
        &DeclaredOutput::timestamps(),
        &EmptyOptions,
        &args,
        &mut ctx,
    )?;
    let expected = expected_timestamps(vec![7, 7, 7], Validity::NonNullable)?;

    assert_arrays_eq!(&actual, &expected, &mut ctx);
    Ok(())
}

#[test]
fn test_declared_output_dtype_reaches_planning() -> VortexResult<()> {
    let args = [DType::Primitive(i64::PTYPE, Nullability::Nullable)];

    let dtype = row_fn_return_dtype(&DeclaredOutput::timestamps(), &EmptyOptions, &args)?;

    assert_eq!(dtype, timestamp_dtype(Nullability::Nullable));
    Ok(())
}

#[rstest]
#[case::nullable(timestamp_dtype(Nullability::Nullable), "must be non-nullable")]
#[case::not_an_extension(DType::from(u64::PTYPE), "must be an extension dtype")]
fn test_declared_output_dtype_rejects_bad_label(
    #[case] declared: DType,
    #[case] expected_message: &str,
) -> VortexResult<()> {
    let function = DeclaredOutput {
        declared: Some(declared),
    };
    let input = PrimitiveArray::from_iter(vec![1_i64, 2]).into_array();
    let args = VecExecutionArgs::new(vec![input], 2);
    let mut ctx = array_session().create_execution_ctx();

    let error = execute_rows(&function, &EmptyOptions, &args, &mut ctx)
        .expect_err("an invalid output dtype must be rejected")
        .to_string();

    assert!(
        error.contains(expected_message),
        "the label error must contain {expected_message:?}, got {error}",
    );
    Ok(())
}

/// Declares a timestamp output dtype over `u64` storage, which its `i64` storage cannot match.
#[derive(Clone)]
struct MismatchedStorage;

impl RowFn for MismatchedStorage {
    type Options = EmptyOptions;

    const ARG_NAMES: &'static [&'static str] = &["value"];
    const INFALLIBLE: bool = true;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("test.mismatched_storage");
        *ID
    }

    fn dispatch<V: RowVisitor>(
        &self,
        _options: &Self::Options,
        _args: &[DType],
        visitor: V,
    ) -> VortexResult<V::VisitResult> {
        visitor
            .with_output_dtype(timestamp_dtype(Nullability::NonNullable))
            .visit::<(u64,), u64>(|(value,)| value)
    }
}

#[test]
fn test_declared_output_dtype_rejects_mismatched_storage() -> VortexResult<()> {
    let input = PrimitiveArray::from_iter(vec![1_u64, 2]).into_array();
    let args = VecExecutionArgs::new(vec![input], 2);
    let mut ctx = array_session().create_execution_ctx();

    let error = execute_rows(&MismatchedStorage, &EmptyOptions, &args, &mut ctx)
        .expect_err("an extension label over another storage dtype must be rejected")
        .to_string();

    assert!(
        error.contains("must store"),
        "the label error must report the expected storage dtype, got {error}",
    );
    Ok(())
}

#[test]
fn test_execution_rejects_a_changed_output_dtype() -> VortexResult<()> {
    let function = ChangingOutputDType {
        dispatches: Arc::new(AtomicUsize::new(0)),
    };
    let input = PrimitiveArray::from_iter(vec![1_i64, 2]).into_array();
    let args = VecExecutionArgs::new(vec![input], 2);
    let mut ctx = array_session().create_execution_ctx();

    let error = execute_rows(&function, &EmptyOptions, &args, &mut ctx)
        .expect_err("an execution dispatch must declare the planned output dtype")
        .to_string();

    assert!(
        error.contains("must declare the planned output dtype"),
        "execution must reject a changed output dtype, got {error}",
    );
    Ok(())
}

#[test]
fn test_undeclared_output_dtype_keeps_the_storage_dtype() -> VortexResult<()> {
    let function = DeclaredOutput { declared: None };
    let values = vec![1_i64, 2];
    let input = PrimitiveArray::from_iter(values.clone()).into_array();
    let args = VecExecutionArgs::new(vec![input], 2);
    let mut ctx = array_session().create_execution_ctx();

    let actual = execute_rows(&function, &EmptyOptions, &args, &mut ctx)?;
    let expected = PrimitiveArray::from_iter(values).into_array();

    assert_arrays_eq!(&actual, &expected, &mut ctx);
    Ok(())
}
