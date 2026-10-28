// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod operations;
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
use crate::arrays::widen::WidenArraySlotsExt;
use crate::arrays::widen::WidenSlots;
use crate::arrays::widen::array::validate_dtypes;
use crate::buffer::BufferHandle;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::serde::ArrayChildren;

/// The encoding of a [`WidenArray`].
#[derive(Clone, Debug)]
pub struct Widen;

/// A [`Widen`]-encoded integer array.
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
        // `ArraySession` does not register `Widen`, so `array_serialize` fails before it calls
        // this method. Only a session that registers `Widen` itself can reach it.
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

    fn execute(array: Array<Self>, _ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
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
}
