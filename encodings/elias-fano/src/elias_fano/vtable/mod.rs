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
use vortex_array::ArrayView;
use vortex_array::EqMode;
use vortex_array::ExecutionCtx;
use vortex_array::ExecutionResult;
use vortex_array::IntoArray;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability::NonNullable;
use vortex_array::dtype::PType;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_array::require_child;
use vortex_array::serde::ArrayChildren;
use vortex_array::validity::Validity;
use vortex_array::vtable::VTable;
use vortex_buffer::BitBuffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use super::array::EliasFanoArraySlotsExt;
use super::array::EliasFanoData;
use super::array::EliasFanoSlots;
use super::array::EliasFanoSlotsView;
use crate::Unary;

mod operations;
mod validity;

/// An [`EliasFano`]-encoded Vortex array.
pub type EliasFanoArray = Array<EliasFano>;

/// Metadata stored alongside the two children.
#[derive(Clone, prost::Message)]
pub(super) struct EliasFanoMetadata {
    /// How many low bits of each value `lower` holds.
    #[prost(uint32, tag = "1")]
    pub(super) lower_bits: u32,
    /// The number of buckets, which is the length of `upper`.
    #[prost(uint64, tag = "2")]
    pub(super) num_buckets: u64,
}

/// The dtype of the `upper` child.
///
/// One bucket can hold every value, so a count can reach the array's length, which only a `u64`
/// holds for every length. The wide dtype does not widen what is stored:
/// [`elias_fano_encode`](crate::elias_fano_encode) holds the counts in a
/// [`Widen`](vortex_array::arrays::Widen) over a narrower type whenever one holds them, and a
/// `Unary` child stores the same bits whatever its dtype.
const UPPER_DTYPE: DType = DType::Primitive(PType::U64, NonNullable);

impl ArrayHash for EliasFanoData {
    fn array_hash<H: Hasher>(&self, state: &mut H, _accuracy: EqMode) {
        self.lower_bits.hash(state);
    }
}

impl ArrayEq for EliasFanoData {
    fn array_eq(&self, other: &Self, _accuracy: EqMode) -> bool {
        self.lower_bits == other.lower_bits
    }
}

impl VTable for EliasFano {
    type TypedArrayData = EliasFanoData;

    type OperationsVTable = Self;
    type ValidityVTable = Self;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.elias_fano");
        *ID
    }

    fn validate(
        &self,
        data: &Self::TypedArrayData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        let DType::Primitive(ptype, NonNullable) = dtype else {
            vortex_bail!(
                "EliasFanoArray requires a non-nullable unsigned integer dtype, got {dtype}"
            );
        };
        vortex_ensure!(
            ptype.is_unsigned_int(),
            "EliasFanoArray requires a non-nullable unsigned integer dtype, got {dtype}"
        );
        vortex_ensure!(
            usize::from(data.lower_bits) < ptype.bit_width(),
            "EliasFanoArray lower_bits {} must be below the bit width of {ptype}",
            data.lower_bits
        );
        vortex_ensure_eq!(slots.len(), EliasFanoSlots::COUNT);
        let slots = EliasFanoSlotsView::from_slots(slots);
        vortex_ensure_eq!(slots.upper.dtype(), &UPPER_DTYPE);
        vortex_ensure_eq!(slots.lower.dtype(), dtype);
        vortex_ensure_eq!(slots.lower.len(), len);
        vortex_ensure!(
            u128::try_from(slots.upper.len())?
                <= 1u128 << (ptype.bit_width() - usize::from(data.lower_bits)),
            "EliasFanoArray has more buckets than {ptype} can hold above its low bits"
        );
        Ok(())
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        0
    }

    fn buffer(_array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        vortex_panic!("EliasFanoArray buffer index {idx} out of bounds")
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

    fn serialize(
        array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(
            EliasFanoMetadata {
                lower_bits: u32::from(array.lower_bits),
                num_buckets: u64::try_from(array.upper().len())?,
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
        vortex_ensure_eq!(buffers.len(), 0);
        vortex_ensure_eq!(children.len(), 2);
        let metadata = EliasFanoMetadata::decode(metadata)?;
        let upper = children.get(
            EliasFanoSlots::UPPER,
            &UPPER_DTYPE,
            usize::try_from(metadata.num_buckets)?,
        )?;
        let lower = children.get(EliasFanoSlots::LOWER, dtype, len)?;
        let data = EliasFanoData {
            lower_bits: u8::try_from(metadata.lower_bits)?,
        };
        Ok(ArrayParts::new(
            self.clone(),
            dtype.clone(),
            len,
            data,
            EliasFanoSlots { upper, lower }.into_slots(),
        ))
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        EliasFanoSlots::NAMES[idx].to_string()
    }

    fn execute(array: Array<Self>, _ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        // A `Unary` upper is decoded from its bits, so its counts are never built.
        let array = if array.upper().is::<Unary>() {
            array
        } else {
            require_child!(array, array.upper(), EliasFanoSlots::UPPER => Primitive)
        };
        let array = require_child!(array, array.lower(), EliasFanoSlots::LOWER => Primitive);
        Ok(ExecutionResult::done(
            elias_fano_decode(&array)?.into_array(),
        ))
    }
}

/// Decode every value. `lower` must be primitive, and `upper` must be [`Unary`], whose bits are
/// read directly, or primitive, whose counts are read bucket by bucket.
fn elias_fano_decode(array: &EliasFanoArray) -> VortexResult<PrimitiveArray> {
    if let Some(upper) = array.upper().as_opt::<Unary>() {
        return decode_bits(array, &upper.to_bit_buffer());
    }
    let counts = array.upper().as_::<Primitive>();
    let counts = counts.as_slice::<u64>();
    // `validate` cannot see the children's values, so a malformed file is caught here.
    let total: u128 = counts.iter().map(|&count| u128::from(count)).sum();
    vortex_ensure_eq!(total, u128::try_from(array.len())?);

    let lower = array.lower().as_::<Primitive>();
    match_each_unsigned_integer_ptype!(array.dtype().as_ptype(), |T| {
        let mut lows = lower.as_slice::<T>().iter();
        let mut values = BufferMut::<T>::with_capacity(array.len());
        let mut stray = 0;
        for (bucket, &count) in counts.iter().enumerate() {
            let high = AsPrimitive::<T>::as_(bucket) << array.lower_bits;
            for &low in lows.by_ref().take(count.as_()) {
                stray |= low;
                values.push(high | low);
            }
        }
        vortex_ensure_eq!(
            stray >> array.lower_bits,
            0,
            "EliasFanoArray low bits must fit in lower_bits"
        );
        Ok(PrimitiveArray::new(values, Validity::NonNullable))
    })
}

/// Decode every value from the bits of a [`Unary`] upper. The value at `index` is the `index`-th
/// set bit, and the unset bits before it, one closing each earlier bucket, give its bucket.
fn decode_bits(array: &EliasFanoArray, bits: &BitBuffer) -> VortexResult<PrimitiveArray> {
    // Each set bit is one value, so a malformed file with too many or too few is caught here.
    vortex_ensure_eq!(bits.true_count(), array.len());

    let lower = array.lower().as_::<Primitive>();
    match_each_unsigned_integer_ptype!(array.dtype().as_ptype(), |T| {
        let lows = lower.as_slice::<T>();
        let mut values = BufferMut::<T>::with_capacity(array.len());
        let mut stray = 0;
        bits.for_each_set_index(|bit| {
            let index = values.len();
            let low = lows[index];
            stray |= low;
            values.push((AsPrimitive::<T>::as_(bit - index) << array.lower_bits) | low);
        });
        vortex_ensure_eq!(
            stray >> array.lower_bits,
            0,
            "EliasFanoArray low bits must fit in lower_bits"
        );
        Ok(PrimitiveArray::new(values, Validity::NonNullable))
    })
}

/// Elias-Fano encoding for sorted, non-nullable unsigned integers.
///
/// Each value is split into its high bits, `value >> lower_bits`, which pick its bucket, and its
/// low `lower_bits` bits. The `upper` child holds the number of values in each bucket, and the
/// `lower` child holds the low bits in order. For `[3, 4, 9]` with `lower_bits = 1`, `upper` is
/// `[0, 1, 1, 0, 1]` and `lower` is `[1, 0, 1]`.
///
/// [`elias_fano_encode`](crate::elias_fano_encode) leaves both children for the compressor to
/// encode as `Unary` and `BitPacked`.
///
/// There is no validity yet, which keeps the encoding simple. Adding it must stay backwards
/// compatible, which an optional validity child allows, as in Bool: arrays written now have only
/// the two children, so they still read back as non-nullable.
#[derive(Clone, Debug)]
pub struct EliasFano;

impl EliasFano {
    /// Assemble an array from its bucket counts, `upper`, and its low bits, `lower`.
    ///
    /// Prefer [`elias_fano_encode`](crate::elias_fano_encode) unless you already hold the
    /// children.
    pub fn try_new(
        upper: ArrayRef,
        lower: ArrayRef,
        lower_bits: u8,
    ) -> VortexResult<EliasFanoArray> {
        let dtype = lower.dtype().clone();
        let len = lower.len();
        let data = EliasFanoData { lower_bits };
        let slots = EliasFanoSlots { upper, lower }.into_slots();
        Array::try_from_parts(ArrayParts::new(EliasFano, dtype, len, data, slots))
    }
}
