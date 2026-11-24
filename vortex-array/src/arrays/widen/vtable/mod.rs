// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Array operations for the `Widen` encoding.
//!
//! Execution produces values at the logical dtype's width. [`WidenPlugin`] handles storage at the
//! child's width.

mod operations;

mod plugin;
pub use plugin::WidenPlugin;
pub use plugin::widen_v2_id;

mod validity;

use vortex_error::VortexResult;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_session::registry::CachedId;

use crate::ArrayRef;
use crate::EmptyArrayData;
use crate::ExecutionCtx;
use crate::ExecutionResult;
use crate::IntoArray;
use crate::array::Array;
use crate::array::ArrayId;
use crate::array::ArrayParts;
use crate::array::ArrayView;
use crate::array::VTable;
use crate::array::ValidityVTableFromChild;
use crate::array::with_empty_buffers;
use crate::arrays::Decimal;
use crate::arrays::DecimalArray;
use crate::arrays::decimal::widened_buffer;
use crate::arrays::widen::WidenArraySlotsExt;
use crate::arrays::widen::WidenSlots;
use crate::arrays::widen::array::validate_dtypes;
use crate::buffer::BufferHandle;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::DecimalDType;
use crate::dtype::NativeDecimalType;
use crate::match_each_decimal_value_type;
use crate::require_child;

/// The encoding of a [`WidenArray`].
#[derive(Clone, Debug)]
pub struct Widen;

/// A [`Widen`]-encoded integer, floating-point, or decimal array.
pub type WidenArray = Array<Widen>;

impl VTable for Widen {
    type TypedArrayData = EmptyArrayData;

    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromChild;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.widen");
        *ID
    }

    fn validate(
        &self,
        _data: &EmptyArrayData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        vortex_ensure_eq!(slots.len(), WidenSlots::COUNT);
        let values = slots[WidenSlots::VALUES]
            .as_ref()
            .ok_or_else(|| vortex_err!("Widen requires a values child"))?;
        vortex_ensure_eq!(values.len(), len);

        validate_dtypes(values.dtype(), dtype)
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        0
    }

    fn buffer(_array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        vortex_panic!("Widen buffer index {idx} out of bounds")
    }

    fn buffer_name(_array: ArrayView<'_, Self>, _idx: usize) -> Option<String> {
        None
    }

    fn with_buffers(
        &self,
        array: ArrayView<'_, Self>,
        buffers: &[BufferHandle],
    ) -> VortexResult<ArrayParts<Self>> {
        with_empty_buffers(self, array, buffers)
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        WidenSlots::NAMES[idx].to_string()
    }

    fn execute(array: Array<Self>, _ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        match array.dtype() {
            DType::Decimal(decimal_dtype, _) => {
                let decimal_dtype = *decimal_dtype;
                let array = require_child!(array, array.values(), WidenSlots::VALUES => Decimal);
                let values = array.values().as_::<Decimal>();
                let target_type = decimal_dtype.canonical_values_type();
                let widened = match_each_decimal_value_type!(target_type, |T| {
                    copy_decimal_values::<T>(values, decimal_dtype)?
                });

                Ok(ExecutionResult::done(widened.into_array()))
            }
            DType::Primitive(..) => {
                // Primitive casts let the child's kernel decode and widen its values in one pass.
                let widened = array.values().cast(array.dtype().clone())?;

                Ok(ExecutionResult::done(widened))
            }
            dtype => {
                vortex_panic!(
                    "Widen validation requires an integer, float, or decimal dtype, got {dtype}"
                )
            }
        }
    }

    fn reduce(array: ArrayView<'_, Self>) -> VortexResult<Option<ArrayRef>> {
        // A physical rewrite keeps only the dtype of each child, so it can put a `Widen` here.
        if !array.values().is::<Widen>() {
            return Ok(None);
        }

        let flattened = WidenArray::try_new(array.values().clone(), array.dtype().clone())?;
        Ok(Some(flattened.into_array()))
    }
}

/// Copies the executed decimal `values` of a [`WidenArray`] into the canonical width of
/// `decimal_dtype`, the logical dtype of the `Widen`.
fn copy_decimal_values<T: NativeDecimalType>(
    values: ArrayView<'_, Decimal>,
    decimal_dtype: DecimalDType,
) -> VortexResult<DecimalArray> {
    let validity = values.validity()?;
    let buffer = widened_buffer::<T>(&values);

    // The wider precision holds every value of the child's native type, at the same scale.
    DecimalArray::try_new_handle_from_trusted_values(
        BufferHandle::new_host(buffer.into_byte_buffer()),
        T::DECIMAL_TYPE,
        decimal_dtype,
        validity,
    )
}
