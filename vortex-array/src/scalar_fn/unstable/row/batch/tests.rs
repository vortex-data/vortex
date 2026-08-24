// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use rstest::rstest;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_session::registry::CachedId;

use super::finalize_kernel_output;
use crate::ArrayRef;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::array_session;
use crate::arrays::ConstantArray;
use crate::arrays::ExtensionArray;
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
struct InvalidKernelOutput;

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
    type Rows<'a> = &'a mut [i64];
    type Row<'a> = &'a mut i64;
    type WriteToken = ();

    fn storage_dtype() -> DType {
        DType::from(i64::PTYPE)
    }

    fn with_capacity(rows: usize) -> VortexResult<Self> {
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
        visitor.visit_into::<(i64,), I64Sink, VortexResult<()>>(|(value,), output| {
            *output = value;
            Ok(())
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
            .visit_into::<(i64,), I64Sink, VortexResult<()>>(|(value,), output| {
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

#[test]
fn test_declared_output_dtype_labels_sink_output() -> VortexResult<()> {
    // `I64Sink` declines skip-invalid execution, so this batch stays all-valid. The masked path is
    // covered by the owned-output cases above.
    let values = vec![1_i64, 2, 3];
    let input = PrimitiveArray::new(values.clone(), Validity::NonNullable).into_array();
    let args = VecExecutionArgs::new(vec![input], 3);
    let mut ctx = array_session().create_execution_ctx();

    let actual = execute_rows(&DeclaredSinkOutput, &EmptyOptions, &args, &mut ctx)?;
    let expected = expected_timestamps(values, Validity::NonNullable)?;

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
#[case::not_an_extension(DType::from(u64::PTYPE), "must label the storage dtype")]
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
