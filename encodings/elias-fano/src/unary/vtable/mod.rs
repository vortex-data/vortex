// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hash;
use std::hash::Hasher;

use num_traits::AsPrimitive;
use prost::Message;
use vortex_array::Array;
use vortex_array::ArrayEq;
use vortex_array::ArrayHash;
use vortex_array::ArrayId;
use vortex_array::ArrayParts;
use vortex_array::ArrayRef;
use vortex_array::ArraySlots;
use vortex_array::ArrayView;
use vortex_array::EqMode;
use vortex_array::ExecutionCtx;
use vortex_array::ExecutionResult;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability::NonNullable;
use vortex_array::dtype::PType;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_array::serde::ArrayChildren;
use vortex_array::validity::Validity;
use vortex_array::vtable::VTable;
use vortex_buffer::BitBuffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use super::array::UnaryData;

mod operations;
mod validity;

/// A [`Unary`]-encoded Vortex array.
pub type UnaryArray = Array<Unary>;

/// Metadata stored alongside the `bits` buffer.
#[derive(Clone, prost::Message)]
pub(super) struct UnaryMetadata {
    /// The number of set bits, which is also the sum of the values. With the array's length it
    /// gives the exact bit length, which the buffer's bytes alone do not.
    #[prost(uint64, tag = "1")]
    pub(super) num_ones: u64,
    /// Where the bits start in the buffer's first byte, so below 8.
    #[prost(uint32, tag = "2")]
    pub(super) offset: u32,
}

impl ArrayHash for UnaryData {
    fn array_hash<H: Hasher>(&self, state: &mut H, accuracy: EqMode) {
        self.bits.array_hash(state, accuracy);
        self.meta.hash(state);
    }
}

impl ArrayEq for UnaryData {
    fn array_eq(&self, other: &Self, accuracy: EqMode) -> bool {
        self.meta == other.meta && self.bits.array_eq(&other.bits, accuracy)
    }
}

/// The number of set bits in `array`, which is its bit length minus its length.
fn num_ones_of(array: ArrayView<'_, Unary>) -> VortexResult<u64> {
    Ok(u64::try_from(array.meta.len() - array.len())?)
}

impl VTable for Unary {
    type TypedArrayData = UnaryData;

    type OperationsVTable = Self;
    type ValidityVTable = Self;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.unary");
        *ID
    }

    fn validate(
        &self,
        data: &Self::TypedArrayData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        vortex_ensure_eq!(slots.len(), 0);
        data.validate(dtype, len)
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        1
    }

    fn buffer(array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        match idx {
            0 => array.bits.clone(),
            _ => vortex_panic!("UnaryArray buffer index {idx} out of bounds"),
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
        vortex_ensure_eq!(buffers.len(), 1);
        let data = UnaryData::from_buffer(
            buffers[0].clone(),
            num_ones_of(array)?,
            array.meta.offset(),
            array.len(),
        )?;
        Ok(ArrayParts::new(
            self.clone(),
            array.dtype().clone(),
            array.len(),
            data,
            ArraySlots::new(),
        ))
    }

    fn serialize(
        array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(
            UnaryMetadata {
                num_ones: num_ones_of(array)?,
                offset: u32::try_from(array.meta.offset())?,
            }
            .encode_to_vec(),
        ))
    }

    fn deserialize(
        &self,
        dtype: &DType,
        len: usize,
        metadata: &[u8],
        buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
        _session: &VortexSession,
    ) -> VortexResult<ArrayParts<Self>> {
        vortex_ensure_eq!(buffers.len(), 1);
        vortex_ensure_eq!(children.len(), 0);
        let metadata = UnaryMetadata::decode(metadata)?;
        let data = UnaryData::from_buffer(
            buffers[0].clone(),
            metadata.num_ones,
            usize::try_from(metadata.offset)?,
            len,
        )?;
        Ok(ArrayParts::new(
            self.clone(),
            dtype.clone(),
            len,
            data,
            ArraySlots::new(),
        ))
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        // Validity, once added, can take an optional slot 0, as in Bool. See `Unary`.
        vortex_panic!("UnaryArray has no slots, got index {idx}")
    }

    fn execute(array: Array<Self>, _ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        Ok(ExecutionResult::done(unary_decode(&array).into_array()))
    }
}

/// Decode every value with one walk over the unset bits.
fn unary_decode(array: &UnaryArray) -> PrimitiveArray {
    let len = array.len();

    // `validate` keeps every value within the ptype, so the loop has nothing to check.
    match_each_unsigned_integer_ptype!(array.dtype().as_ptype(), |T| {
        let mut values = BufferMut::<T>::with_capacity(len);
        array.for_each_value(|value| values.push(AsPrimitive::<T>::as_(value)));
        PrimitiveArray::new(values, Validity::NonNullable)
    })
}

/// Unary encoding for non-nullable unsigned integers, in any order.
///
/// The values are stored as one bit array. Each value `v` is written in unary, as `v` set bits
/// followed by an unset bit that closes it, so the array takes `len + num_ones` bits, where
/// `num_ones`, the number of set bits, is the sum of the values. The value at index `k` is the
/// number of set bits between the `k`-th unset bit and the one before it.
///
/// For `[2, 0, 3]` the bits are `110 0 1110`.
///
/// Reads are deliberately simple for now: `scalar_at` scans from the first bit, and a slice
/// decodes the whole array.
///
/// There is no validity yet, which keeps the encoding simple. Adding it must stay backwards
/// compatible, which an optional validity child allows, as in Bool: arrays written now have no
/// children, so they still read back as non-nullable.
#[derive(Clone, Debug)]
pub struct Unary;

impl Unary {
    /// Assemble an array of `len` values from its bits.
    ///
    /// Prefer [`unary_encode`](crate::unary_encode) unless you already hold the bits.
    pub fn try_new(bits: BitBuffer, ptype: PType, len: usize) -> VortexResult<UnaryArray> {
        let dtype = DType::Primitive(ptype, NonNullable);
        let data = UnaryData::from_bit_buffer(bits);
        Array::try_from_parts(ArrayParts::new(Unary, dtype, len, data, ArraySlots::new()))
    }
}
