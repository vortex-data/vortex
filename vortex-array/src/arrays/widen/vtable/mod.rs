// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Vtable and wire adapter for the in-memory `Widen` encoding.
//!
//! Execution restores the logical width. The writer plugin uses an existing wire form because
//! `Widen` has no wire form of its own.

mod operations;

mod plugin;
pub use plugin::WidenPlugin;

mod validity;

use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;
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
use crate::arrays::decimal::DecimalArrayExt;
use crate::arrays::decimal::copy_to_canonical_width;
use crate::arrays::widen::WidenArraySlotsExt;
use crate::arrays::widen::WidenSlots;
use crate::arrays::widen::array::validate_dtypes;
use crate::arrays::widen::compute::rules::PARENT_RULES;
use crate::buffer::BufferHandle;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::DecimalDType;
use crate::match_each_decimal_value_type;
use crate::require_child;
use crate::serde::ArrayChildren;

/// The encoding of a [`WidenArray`].
#[derive(Clone, Debug)]
pub struct Widen;

/// A [`Widen`]-encoded integer or decimal array.
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

    fn serialize(
        _array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        // The default session registers `WidenPlugin`, which writes a storable representation
        // without calling this method. Direct registration of the vtable has no wire form.
        vortex_panic!("TODO(connor)[Widen]: Widen has no wire form yet (#10733)")
    }

    fn deserialize(
        &self,
        _dtype: &DType,
        _len: usize,
        _metadata: &[u8],
        _buffers: &[BufferHandle],
        _children: &dyn ArrayChildren,
        _session: &VortexSession,
    ) -> VortexResult<ArrayParts<Self>> {
        vortex_bail!("Widen deserialization is not supported")
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        WidenSlots::NAMES[idx].to_string()
    }

    fn execute(array: Array<Self>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        // NB: Decimals copy their values instead of yielding a cast. A decimal cast keeps a stored
        // width that already holds the target precision, so it can return values wider than the
        // canonical width of the logical dtype.
        if let DType::Decimal(decimal_dtype, _) = array.dtype() {
            let decimal_dtype = *decimal_dtype;
            let array = require_child!(array, array.values(), WidenSlots::VALUES => Decimal);
            let widened = copy_decimal_values(array.values().as_::<Decimal>(), decimal_dtype, ctx)?;
            return Ok(ExecutionResult::done(widened.into_array()));
        }

        // Yield a cast, not a canonical array, so that one kernel decodes and widens the values.
        let widened = array.values().cast(array.dtype().clone())?;
        Ok(ExecutionResult::done(widened))
    }

    fn reduce(array: ArrayView<'_, Self>) -> VortexResult<Option<ArrayRef>> {
        // A physical rewrite keeps only the dtype of each child, so it can put a `Widen` here.
        if !array.values().is::<Widen>() {
            return Ok(None);
        }

        let flattened = WidenArray::try_new(array.values().clone(), array.dtype().clone())?;
        Ok(Some(flattened.into_array()))
    }

    fn reduce_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_RULES.evaluate(array, parent, child_idx)
    }
}

/// Copies the executed decimal `values` of a [`WidenArray`] into the canonical width of
/// `decimal_dtype`, the logical dtype of the `Widen`.
fn copy_decimal_values(
    values: ArrayView<'_, Decimal>,
    decimal_dtype: DecimalDType,
    ctx: &mut ExecutionCtx,
) -> VortexResult<DecimalArray> {
    let validity = values.validity()?;
    let mask = validity.execute_mask(values.len(), ctx)?;
    let buffer = match_each_decimal_value_type!(values.values_type(), |T| {
        copy_to_canonical_width(values.buffer::<T>().as_slice(), &mask, decimal_dtype)?
    });

    // SAFETY: the copy has the parent canonical width and the child length. Widen construction
    // guarantees the same scale and a parent precision that holds the child, and the unchanged
    // child validity has the same length.
    Ok(unsafe {
        DecimalArray::new_unchecked_handle(
            buffer,
            decimal_dtype.canonical_values_type(),
            decimal_dtype,
            validity,
        )
    })
}
