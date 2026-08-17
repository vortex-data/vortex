// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_mask::Mask;

use super::ElementTuple;
use super::batch_constant;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::array_session;
use crate::arrays::Constant;
use crate::arrays::ConstantArray;
use crate::arrays::ExtensionArray;
use crate::arrays::MaskedArray;
use crate::arrays::PrimitiveArray;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::extension::datetime::TimeUnit;
use crate::extension::datetime::Timestamp;
use crate::scalar_fn::VecExecutionArgs;
use crate::scalar_fn::unstable::row::InputElement;
use crate::validity::Validity;

static DECODE_CALLS: AtomicUsize = AtomicUsize::new(0);

macro_rules! i64_test_element {
    ($element:ident, $decode_fallible:literal $(, $can_decode:item)?) => {
        struct $element;

        // SAFETY: the view and unchecked access delegate to the `i64` implementation.
        unsafe impl InputElement for $element {
            type Column = Buffer<i64>;
            type View<'a> = &'a [i64];
            type Elem<'a> = i64;

            const DENSE_SAFE: bool = true;
            const DECODE_FALLIBLE: bool = $decode_fallible;

            fn validate(dtype: &DType) -> VortexResult<()> {
                <i64 as InputElement>::validate(dtype)
            }

            fn decode(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self::Column> {
                DECODE_CALLS.fetch_add(1, Ordering::Relaxed);
                <i64 as InputElement>::decode(array, ctx)
            }

            $($can_decode)?

            fn get(column: &Self::Column, index: usize) -> i64 {
                <i64 as InputElement>::get(column, index)
            }

            fn view(column: &Self::Column) -> Self::View<'_> {
                <i64 as InputElement>::view(column)
            }

            fn view_len(view: &Self::View<'_>) -> usize {
                <i64 as InputElement>::view_len(view)
            }

            fn get_from_view<'a>(view: &Self::View<'a>, index: usize) -> i64
            where
                Self: 'a,
            {
                <i64 as InputElement>::get_from_view(view, index)
            }

            unsafe fn get_from_view_unchecked<'a>(view: &Self::View<'a>, index: usize) -> i64
            where
                Self: 'a,
            {
                // SAFETY: forwarded from this method's contract.
                unsafe { <i64 as InputElement>::get_from_view_unchecked(view, index) }
            }
        }
    };
}

i64_test_element!(
    DecodeProbe,
    false,
    fn can_decode_null_tolerant(_array: &ArrayRef) -> VortexResult<bool> {
        Ok(true)
    }
);
i64_test_element!(DenseFallible, true);

#[test]
fn test_null_tolerant_decline_precedes_decoding() -> VortexResult<()> {
    DECODE_CALLS.store(0, Ordering::Relaxed);
    let first = PrimitiveArray::from_iter([1_i64, 2]).into_array();
    let second = PrimitiveArray::from_iter([3_i64, 4]).into_array();
    let args = VecExecutionArgs::new(vec![first, second], 2);
    let mut ctx = array_session().create_execution_ctx();

    let columns = <(DecodeProbe, DenseFallible)>::decode_null_tolerant(&args, &mut ctx)?;

    assert!(columns.is_none());
    assert_eq!(DECODE_CALLS.load(Ordering::Relaxed), 0);
    Ok(())
}

#[test]
fn test_batch_constant_unwraps_filtered_masked_constant() -> VortexResult<()> {
    let child = ConstantArray::new(7_i64, 3).into_array();
    let masked =
        MaskedArray::try_new(child, Validity::from_iter([true, false, true]))?.into_array();
    let filtered = masked.filter(Mask::from_iter([true, true, false]))?;

    let Some(constant) = batch_constant(&filtered) else {
        vortex_bail!("filtered masked constant must remain batch-constant");
    };

    assert!(constant.is::<Constant>());
    Ok(())
}

#[test]
fn test_batch_constant_preserves_filtered_extension() -> VortexResult<()> {
    let ext_dtype = Timestamp::new(TimeUnit::Milliseconds, Nullability::NonNullable).erased();
    let extension =
        ExtensionArray::new(ext_dtype, ConstantArray::new(7_i64, 3).into_array()).into_array();
    let filtered = extension.filter(Mask::from_iter([true, false, true]))?;

    let Some(constant) = batch_constant(&filtered) else {
        vortex_bail!("filtered extension storage must remain batch-constant");
    };

    assert_eq!(constant.dtype(), extension.dtype());
    Ok(())
}
