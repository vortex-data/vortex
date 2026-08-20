// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

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
use crate::arrays::PrimitiveArray;
use crate::assert_arrays_eq;
use crate::dtype::DType;
use crate::dtype::NativePType;
use crate::dtype::Nullability;
use crate::scalar::Scalar;
use crate::scalar_fn::EmptyOptions;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::VecExecutionArgs;
use crate::scalar_fn::unstable::row::OutputElement;
use crate::scalar_fn::unstable::row::OutputSink;
use crate::scalar_fn::unstable::row::RowFn;
use crate::scalar_fn::unstable::row::RowVisitor;
use crate::scalar_fn::unstable::row::execute_rows;
use crate::validity::Validity;

#[derive(Clone, Default)]
struct DeferredAdd {
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
unsafe impl<Options> OutputSink<Options> for I64Sink {
    type Rows<'a> = &'a mut [i64];
    type Row<'a> = &'a mut i64;
    type WriteToken = ();

    fn return_dtype(_options: &Options) -> VortexResult<DType> {
        Ok(DType::from(i64::PTYPE))
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

    fn dispatch<V: RowVisitor<Self::Options>>(
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

    fn dispatch<V: RowVisitor<Self::Options>>(
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

    fn dispatch<V: RowVisitor<Self::Options>>(
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
fn test_deferred_owned_execution_does_not_retry_success() -> VortexResult<()> {
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
fn test_deferred_owned_execution_reports_valid_failure() -> VortexResult<()> {
    let function = DeferredAdd::default();
    let validity = Validity::from_iter([true, false]);
    let lhs = PrimitiveArray::new(vec![i64::MAX, 1], validity).into_array();
    let rhs = ConstantArray::new(1_i64, 2).into_array();
    let args = VecExecutionArgs::new(vec![lhs, rhs], 2);
    let mut ctx = array_session().create_execution_ctx();

    let error = execute_rows(&function, &EmptyOptions, &args, &mut ctx)
        .expect_err("a valid-row overflow must remain observable");

    assert!(error.to_string().contains("deferred addition overflowed"));
    assert_eq!(function.prepare_count(), 2);
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
