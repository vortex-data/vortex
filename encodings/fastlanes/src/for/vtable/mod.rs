// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::hash::Hash;
use std::hash::Hasher;

use vortex_array::Array;
use vortex_array::ArrayEq;
use vortex_array::ArrayHash;
use vortex_array::ArrayId;
use vortex_array::ArrayParts;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::EqMode;
use vortex_array::ExecutionCtx;
use vortex_array::ExecutionResult;
use vortex_array::IntoArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::scalar::Scalar;
use vortex_array::serde::ArrayChildren;
use vortex_array::smallvec::smallvec;
use vortex_array::vtable::VTable;
use vortex_array::vtable::ValidityVTableFromChild;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;

use crate::FoRData;
use crate::r#for::array::FoRSlots;
use crate::r#for::array::FoRSlotsView;
use crate::r#for::array::for_decompress::decompress;
use crate::r#for::array::num_chunks;
use crate::r#for::vtable::rules::PARENT_RULES;
use crate::for_v2_id;

mod kernels;
mod operations;
mod rules;
mod slice;
mod validity;

/// A [`FoR`]-encoded Vortex array.
pub type FoRArray = Array<FoR>;

pub(crate) fn initialize(session: &VortexSession) {
    kernels::initialize(session);
}

impl ArrayHash for FoRData {
    fn array_hash<H: Hasher>(&self, state: &mut H, _accuracy: EqMode) {
        self.offset.hash(state);
    }
}

impl ArrayEq for FoRData {
    fn array_eq(&self, other: &Self, _accuracy: EqMode) -> bool {
        self.offset == other.offset
    }
}

impl VTable for FoR {
    type TypedArrayData = FoRData;

    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromChild;

    fn id(&self) -> ArrayId {
        for_v2_id()
    }

    fn validate(
        &self,
        data: &Self::TypedArrayData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        let slots = FoRSlotsView::from_slots(slots);
        validate_parts(slots.encoded, slots.references, data.offset, dtype, len)
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        0
    }

    fn buffer(_array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        vortex_panic!("FoRArray buffer index {idx} out of bounds")
    }

    fn buffer_name(_array: ArrayView<'_, Self>, _idx: usize) -> Option<String> {
        None
    }

    fn with_buffers(
        &self,
        array: ArrayView<'_, Self>,
        buffers: &[BufferHandle],
    ) -> VortexResult<ArrayParts<Self>> {
        vortex_array::vtable::with_empty_buffers(self, array, buffers)
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        FoRSlots::NAMES[idx].to_string()
    }

    fn serialize(
        _array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        vortex_bail!("FoR serialization requires FoRPlugin")
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
        vortex_bail!("FoR deserialization requires FoRPlugin")
    }

    fn reduce_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_RULES.evaluate(array, parent, child_idx)
    }

    fn execute(array: Array<Self>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        Ok(ExecutionResult::done(decompress(&array, ctx)?.into_array()))
    }
}

#[derive(Clone, Debug)]
pub struct FoR;

impl FoR {
    /// Construct a new FoR array from an encoded array and a reference scalar.
    pub fn try_new(encoded: ArrayRef, reference: Scalar) -> VortexResult<FoRArray> {
        vortex_ensure!(!reference.is_null(), "Reference value cannot be null");
        let dtype = reference
            .dtype()
            .with_nullability(encoded.dtype().nullability());
        let reference = reference.cast(&dtype.as_nonnullable())?;
        let references = ConstantArray::new(reference, num_chunks(0, encoded.len())).into_array();
        Self::try_new_chunked(encoded, references, 0)
    }

    /// Construct a FoR array with one reference per 1024-element chunk.
    ///
    /// `references` must be a non-nullable integer array of the encoded array's type, with one
    /// entry for each chunk spanned by `offset + encoded.len()` elements. `offset` is the position
    /// of the first element within the first chunk.
    pub fn try_new_chunked(
        encoded: ArrayRef,
        references: ArrayRef,
        offset: u16,
    ) -> VortexResult<FoRArray> {
        let dtype = encoded.dtype().clone();
        let len = encoded.len();
        let data = FoRData::try_new(offset)?;
        let slots = smallvec![Some(encoded), Some(references)];
        Array::try_from_parts(ArrayParts::new(FoR, dtype, len, data).with_slots(slots))
    }

    /// Encode a primitive array using Frame of Reference encoding.
    pub fn encode(array: PrimitiveArray, ctx: &mut ExecutionCtx) -> VortexResult<FoRArray> {
        FoRData::encode(array, ctx)
    }

    /// Encode a primitive array with one Frame of Reference per 1024-element chunk.
    pub fn encode_chunked(array: PrimitiveArray, ctx: &mut ExecutionCtx) -> VortexResult<FoRArray> {
        FoRData::encode_chunked(array, ctx)
    }
}

fn validate_parts(
    encoded: &ArrayRef,
    references: &ArrayRef,
    offset: u16,
    dtype: &DType,
    len: usize,
) -> VortexResult<()> {
    vortex_ensure!(dtype.is_int(), "FoR requires an integer dtype, got {dtype}");
    vortex_ensure!(
        encoded.dtype() == dtype,
        "FoR encoded dtype mismatch: expected {dtype}, got {}",
        encoded.dtype()
    );
    vortex_ensure!(
        encoded.len() == len,
        "FoR encoded length mismatch: expected {len}, got {}",
        encoded.len()
    );
    let references_dtype = dtype.as_nonnullable();
    vortex_ensure!(
        references.dtype() == &references_dtype,
        "FoR references dtype mismatch: expected {references_dtype}, got {}",
        references.dtype()
    );
    let num_chunks = num_chunks(offset, len);
    vortex_ensure!(
        references.len() == num_chunks,
        "FoR expects {num_chunks} references, got {}",
        references.len()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use vortex_array::scalar::ScalarValue;
    use vortex_array::test_harness::check_metadata;

    #[cfg_attr(miri, ignore)]
    #[test]
    fn test_for_metadata() {
        let metadata: Vec<u8> = ScalarValue::to_proto_bytes(Some(&ScalarValue::from(i64::MAX)));
        check_metadata("for.metadata", &metadata);
    }
}
