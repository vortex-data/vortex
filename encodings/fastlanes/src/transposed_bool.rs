// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;
use std::ops::Range;

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
use vortex_array::TypedArrayRef;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::slice::SliceReduce;
use vortex_array::arrays::slice::SliceReduceAdaptor;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::optimizer::rules::ParentRuleSet;
use vortex_array::scalar::Scalar;
use vortex_array::serde::ArrayChildren;
use vortex_array::validity::Validity;
use vortex_array::vtable::OperationsVTable;
use vortex_array::vtable::VTable;
use vortex_array::vtable::ValidityVTable;
use vortex_buffer::BitBuffer;
use vortex_buffer::BitBufferView;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::FL_CHUNK_SIZE;
use crate::bit_transpose::untranspose_bitbuffer;

/// A non-nullable boolean array stored in FastLanes-transposed order.
pub type TransposedBoolArray = Array<TransposedBool>;

/// The array encoding for a boolean bitmap stored in FastLanes-transposed order.
#[derive(Clone, Debug)]
pub struct TransposedBool;

/// Per-array data for a [`TransposedBoolArray`].
#[derive(Clone, Debug)]
pub struct TransposedBoolData {
    bits: BufferHandle,
    offset: usize,
}

impl Display for TransposedBoolData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "offset: {}", self.offset)
    }
}

impl ArrayHash for TransposedBoolData {
    fn array_hash<H: Hasher>(&self, state: &mut H, accuracy: EqMode) {
        self.bits.array_hash(state, accuracy);
        self.offset.hash(state);
    }
}

impl ArrayEq for TransposedBoolData {
    fn array_eq(&self, other: &Self, accuracy: EqMode) -> bool {
        self.bits.array_eq(&other.bits, accuracy) && self.offset == other.offset
    }
}

/// Accessors for a [`TransposedBoolArray`].
pub trait TransposedBoolArrayExt: TypedArrayRef<TransposedBool> {
    /// Returns the logical offset into the first untransposed chunk.
    fn offset(&self) -> usize {
        self.deref().offset
    }

    /// Borrows the full backing bitmap in FastLanes-transposed order.
    ///
    /// For a sliced array, this includes the complete chunks surrounding the logical range.
    fn bit_buffer_view(&self) -> BitBufferView<'_> {
        BitBufferView::new(
            self.deref().bits.as_host().as_slice(),
            self.deref().bits.len() * 8,
        )
    }

    /// Returns the full backing bitmap in FastLanes-transposed order, consuming the array.
    fn to_bit_buffer(self) -> BitBuffer
    where
        Self: Sized,
    {
        let bits = self.deref().bits.clone();
        let len = bits.len() * 8;
        BitBuffer::new(bits.unwrap_host(), len)
    }
}

impl<T: TypedArrayRef<TransposedBool>> TransposedBoolArrayExt for T {}

impl TransposedBool {
    /// Creates an array from a bitmap already stored in FastLanes-transposed order.
    ///
    /// # Errors
    ///
    /// Returns an error if the bitmap does not contain complete 1,024-bit chunks.
    pub fn try_new(bits: BitBuffer) -> VortexResult<TransposedBoolArray> {
        let bits = bits.sliced();
        vortex_ensure!(
            bits.len().is_multiple_of(FL_CHUNK_SIZE),
            "TransposedBoolArray length {} must be a multiple of {FL_CHUNK_SIZE}",
            bits.len()
        );
        let (_, len, bits) = bits.into_inner();
        Self::try_new_view(BufferHandle::new_host(bits), 0, len)
    }

    fn try_new_view(
        bits: BufferHandle,
        offset: usize,
        len: usize,
    ) -> VortexResult<TransposedBoolArray> {
        Array::try_from_parts(ArrayParts::new(
            TransposedBool,
            DType::Bool(Nullability::NonNullable),
            len,
            TransposedBoolData { bits, offset },
        ))
    }
}

impl VTable for TransposedBool {
    type TypedArrayData = TransposedBoolData;
    type OperationsVTable = Self;
    type ValidityVTable = Self;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("fastlanes.transposed_bool");
        *ID
    }

    fn validate(
        &self,
        data: &Self::TypedArrayData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        vortex_ensure!(
            dtype == &DType::Bool(Nullability::NonNullable),
            "TransposedBoolArray must have non-nullable boolean dtype, got {dtype}"
        );
        vortex_ensure!(
            slots.is_empty(),
            "TransposedBoolArray expects no slots, got {}",
            slots.len()
        );
        vortex_ensure!(
            (data.bits.len() * 8).is_multiple_of(FL_CHUNK_SIZE),
            "TransposedBoolArray buffer length {} bits must be a multiple of {FL_CHUNK_SIZE}",
            data.bits.len() * 8
        );
        vortex_ensure!(
            data.offset < FL_CHUNK_SIZE,
            "TransposedBoolArray offset {} must be less than {FL_CHUNK_SIZE}",
            data.offset
        );
        let end = data
            .offset
            .checked_add(len)
            .ok_or_else(|| vortex_error::vortex_err!("TransposedBoolArray range end overflow"))?;
        vortex_ensure!(
            end <= data.bits.len() * 8,
            "TransposedBoolArray range {}..{} exceeds buffer length {} bits",
            data.offset,
            end,
            data.bits.len() * 8
        );
        Ok(())
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        1
    }

    fn buffer(array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        match idx {
            0 => array.bits.clone(),
            _ => vortex_panic!("TransposedBoolArray buffer index {idx} out of bounds"),
        }
    }

    fn buffer_name(_array: ArrayView<'_, Self>, idx: usize) -> Option<String> {
        match idx {
            0 => Some("bits".to_string()),
            _ => None,
        }
    }

    fn with_buffers(
        &self,
        array: ArrayView<'_, Self>,
        buffers: &[BufferHandle],
    ) -> VortexResult<ArrayParts<Self>> {
        vortex_ensure!(
            buffers.len() == 1,
            "TransposedBoolArray expects one buffer, got {}",
            buffers.len()
        );
        Ok(ArrayParts::new(
            self.clone(),
            array.dtype().clone(),
            array.len(),
            TransposedBoolData {
                bits: buffers[0].clone(),
                offset: array.offset(),
            },
        ))
    }

    fn serialize(
        _array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        vortex_bail!("Cannot serialise TransposedBoolArray");
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
        vortex_bail!("Cannot deserialise TransposedBoolArray");
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        vortex_panic!("TransposedBoolArray slot index {idx} out of bounds")
    }

    fn execute(array: Array<Self>, _ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        let len = array.len();
        let offset = array.offset();
        let untransposed = BoolArray::new(
            untranspose_bitbuffer(array.to_bit_buffer()),
            Validity::NonNullable,
        );
        Ok(ExecutionResult::done(
            untransposed.slice(offset..offset + len)?,
        ))
    }

    fn reduce_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        RULES.evaluate(array, parent, child_idx)
    }
}

impl OperationsVTable<TransposedBool> for TransposedBool {
    fn scalar_at(
        array: ArrayView<'_, TransposedBool>,
        index: usize,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let physical_index = array.offset() + index;
        let chunk_start = physical_index / FL_CHUNK_SIZE * FL_CHUNK_SIZE;
        let transposed_index = chunk_start + fastlanes::transpose(physical_index % FL_CHUNK_SIZE);
        Ok(Scalar::bool(
            array.bit_buffer_view().value(transposed_index),
            Nullability::NonNullable,
        ))
    }
}

impl ValidityVTable<TransposedBool> for TransposedBool {
    fn validity(_array: ArrayView<'_, TransposedBool>) -> VortexResult<Validity> {
        Ok(Validity::NonNullable)
    }
}

impl SliceReduce for TransposedBool {
    fn slice(array: ArrayView<'_, Self>, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        let physical_start = array.offset() + range.start;
        let physical_stop = array.offset() + range.end;
        let start_chunk = physical_start / FL_CHUNK_SIZE;
        let stop_chunk = physical_stop.div_ceil(FL_CHUNK_SIZE);
        let bits = array.bits.slice(start_chunk * 128..stop_chunk * 128);

        Ok(Some(
            TransposedBool::try_new_view(bits, physical_start % FL_CHUNK_SIZE, range.len())?
                .into_array(),
        ))
    }
}

static RULES: ParentRuleSet<TransposedBool> =
    ParentRuleSet::new(&[ParentRuleSet::lift(&SliceReduceAdaptor(TransposedBool))]);

#[cfg(test)]
mod tests {
    use vortex_array::VortexSessionExecute;
    use vortex_array::array_session;
    use vortex_array::assert_arrays_eq;
    use vortex_buffer::BitBuffer;
    use vortex_error::VortexResult;

    use super::*;
    use crate::bit_transpose::transpose_bitbuffer;

    fn test_bits() -> BitBuffer {
        BitBuffer::from_iter((0..2 * FL_CHUNK_SIZE).map(|i| i % 3 != 0 && i % 11 != 0))
    }

    #[test]
    fn execute_full_array() -> VortexResult<()> {
        let expected = test_bits();
        let array = TransposedBool::try_new(transpose_bitbuffer(expected.clone()))?;
        let mut ctx = array_session().create_execution_ctx();

        assert_arrays_eq!(array, BoolArray::from(expected), &mut ctx);
        Ok(())
    }

    #[test]
    fn slice_stays_lazy_and_translates_scalars() -> VortexResult<()> {
        let expected = test_bits();
        let array = TransposedBool::try_new(transpose_bitbuffer(expected.clone()))?;
        let sliced = array.slice(1000..1050)?;
        assert!(sliced.is::<TransposedBool>());

        let mut ctx = array_session().create_execution_ctx();
        for index in [0, 23, 49] {
            assert_eq!(
                sliced.execute_scalar(index, &mut ctx)?.as_bool().value(),
                Some(expected.value(1000 + index))
            );
        }
        assert_arrays_eq!(
            sliced,
            BoolArray::from(expected.slice(1000..1050)),
            &mut ctx
        );
        Ok(())
    }
}
