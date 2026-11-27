// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The [`VTable`] of [`EliasFano`] arrays.

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
use vortex_array::arrays::Primitive;
use vortex_array::block::BlockOp;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability::NonNullable;
use vortex_array::dtype::PType;
use vortex_array::require_child;
use vortex_array::require_spine;
use vortex_array::vtable::VTable;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_panic;
use vortex_session::registry::CachedId;

use super::array::EliasFanoArraySlotsExt;
use super::array::EliasFanoData;
use super::array::EliasFanoSlots;
use super::array::EliasFanoSlotsView;
use super::decode::elias_fano_decode;
use crate::Unary;

mod operations;
mod plugin;
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
    ///
    /// It is stored rather than derived, because the dtype and `lower_bits` only give an upper
    /// bound, `2^(bit width - lower_bits)`. For example, with 5 low bits, values that fit in 8 bits
    /// need at most `2^3` buckets. That is also the bound for a `u8` array, but a `u16` array
    /// allows `2^11` and a `u32` array `2^27`. The encoder makes buckets only up to the largest
    /// value's high part, so the count follows the data, and `validate` checks it against the
    /// bound.
    #[prost(uint64, tag = "2")]
    pub(super) num_buckets: u64,
}

/// The dtype of the `upper` child.
///
/// One bucket can hold every value, so a count can reach the array's length, which only a `u64`
/// holds for every length. A wide logical dtype costs nothing: a `Unary` child stores the same bits
/// whatever its dtype.
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
            "EliasFanoArray lower_bits must be below the bit width of {ptype}, got {}",
            data.lower_bits
        );

        vortex_ensure_eq!(slots.len(), EliasFanoSlots::COUNT);
        let slots = EliasFanoSlotsView::from_slots(slots);

        vortex_ensure_eq!(slots.upper.dtype(), &UPPER_DTYPE);
        vortex_ensure_eq!(slots.lower.dtype(), dtype);
        vortex_ensure_eq!(slots.lower.len(), len);

        // A `u128`, because a `u64` with no low bits allows `2^64` buckets.
        let max_buckets = 1u128 << (ptype.bit_width() - usize::from(data.lower_bits));
        let num_buckets = slots.upper.len();
        vortex_ensure!(
            u128::try_from(num_buckets)? <= max_buckets,
            "EliasFanoArray can have at most {max_buckets} buckets, got {num_buckets}"
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
        let array = require_spine!(array, array.lower(), EliasFanoSlots::LOWER => Primitive);

        Ok(ExecutionResult::done(
            elias_fano_decode(&array)?.into_array(),
        ))
    }

    fn block_op(
        array: ArrayView<'_, Self>,
        slot_idx: usize,
        _child_dtype: &DType,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Box<dyn BlockOp>>> {
        if slot_idx != EliasFanoSlots::LOWER {
            return Ok(None);
        }
        super::decode::lower_block_op(array)
    }
}

/// Elias-Fano encoding for sorted, non-nullable unsigned integers.
///
/// Each value is split into its high bits, `value >> lower_bits`, which pick its bucket, and its
/// low `lower_bits` bits. The `upper` child holds the number of values in each bucket, and the
/// `lower` child holds the low bits in order. For `[3, 4, 9]` with `lower_bits = 1`, `upper` is
/// `[0, 1, 1, 0, 1]` and `lower` is `[1, 0, 1]`.
///
/// [`elias_fano_encode`](crate::elias_fano_encode) leaves both children uncompressed, for the
/// compressor to encode as `Unary` and `BitPacked`.
///
/// There is no validity yet, which keeps the encoding simple. Adding it must stay backwards
/// compatible, which an optional validity child allows, as in Bool: arrays written now have only
/// the two children, so they still read back as non-nullable.
#[derive(Clone, Debug)]
pub struct EliasFano;

impl EliasFano {
    /// Assemble an array from its bucket counts, `upper`, and its low bits, `lower`.
    ///
    /// This checks the dtypes and lengths of the children, but not their values. If the counts do
    /// not sum to the length of `lower`, or a low value does not fit in `lower_bits` bits, the
    /// array builds, but executing it returns an error.
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
