// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;
use std::ops::Range;

use num_traits::AsPrimitive;
use prost::Message;
use smallvec::smallvec;
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
use vortex_array::array_slots;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability::NonNullable;
use vortex_array::dtype::PType;
use vortex_array::expr::stats::Precision as StatPrecision;
use vortex_array::expr::stats::Stat;
use vortex_array::match_each_integer_ptype;
use vortex_array::proto::scalar::ScalarValue as ProtoScalarValue;
use vortex_array::scalar::PValue;
use vortex_array::scalar::Scalar;
use vortex_array::scalar::ScalarValue;
use vortex_array::serde::ArrayChildren;
use vortex_array::stats::StatsSet;
use vortex_array::validity::Validity;
use vortex_array::vtable::OperationsVTable;
use vortex_array::vtable::VTable;
use vortex_array::vtable::ValidityVTable;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_fastlanes::BitPacked;
use vortex_fastlanes::BitPackedArrayExt;
use vortex_fastlanes::BitWidthsView;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::access::access_at;
use crate::compress::elias_fano_decompress;
use crate::ef;
use crate::malformed;

/// An [`EliasFano`]-encoded Vortex array.
pub type EliasFanoArray = Array<EliasFano>;

/// The dtype of the bit-packed low-bits child, always `u64` whatever the array's own ptype.
///
/// FastLanes packs `128 * bit_width` bytes per block regardless of element type, so a wide child
/// costs no space and every low-bits read monomorphises once with no runtime ptype dispatch.
pub(crate) const LOWER_DTYPE: DType = DType::Primitive(PType::U64, NonNullable);

#[array_slots(EliasFano)]
pub struct EliasFanoSlots {
    /// The low [`EliasFanoData::lower_width`] bits of every element, in element order. Normally
    /// bit-packed, or a constant zero array at width zero. As long as the array: a slice slices it
    /// too.
    #[slot(0)]
    pub lower: ArrayRef,
}

/// Wire-format metadata persisted alongside the buffers `[upper, samples0, samples1]` and
/// [`EliasFanoSlots`].
#[derive(Clone, prost::Message)]
pub struct EliasFanoMetadata {
    /// The value subtracted from every element before encoding.
    #[prost(message, tag = "1")]
    reference: Option<ProtoScalarValue>,
    /// The largest value in the *encoded* sequence, which fixes the universe.
    #[prost(message, tag = "2")]
    max: Option<ProtoScalarValue>,
    /// Number of low bits per element.
    #[prost(uint32, tag = "3")]
    lower_width: u32,
    /// Length in bits of the encoded sequence's upper array.
    #[prost(uint64, tag = "4")]
    upper_len: u64,
    /// Rank of this array's first element within the encoded sequence.
    #[prost(uint64, tag = "5")]
    first_rank: u64,
    /// Number of elements in the encoded sequence, which a slice keeps.
    #[prost(uint64, tag = "6")]
    num_elements: u64,
    /// Absolute position of the `upper` buffer's first bit: the set bit of `first_rank`, or 0.
    #[prost(uint64, tag = "7")]
    upper_base: u64,
}

/// An Elias-Fano encoded monotonically non-decreasing integer sequence.
///
/// Holds only what cannot be re-derived from the layout in the [`crate::ef`] module: the buffers
/// of this array's window of the encoded sequence, the universe bounds, and the numbers that size
/// and place it.
///
/// The buffers are host-resident. The upper array is read bit by bit, so there is no way to serve
/// it one entry at a time from device memory; `with_buffers` and `deserialize` copy to the host
/// once so no accessor below has to ask.
#[derive(Clone, Debug)]
pub struct EliasFanoData {
    /// The window's part of the unary upper array, from byte `upper_base / 8`.
    upper: ByteBuffer,
    /// The window's zero-sample positions, as little-endian `u64`s read unaligned.
    samples0: ByteBuffer,
    /// The window's one-sample positions, as little-endian `u64`s read unaligned.
    samples1: ByteBuffer,
    reference: Scalar,
    max: Scalar,
    lower_width: u8,
    upper_len: u64,
    num_elements: u64,
    first_rank: u64,
    upper_base: u64,
}

impl Display for EliasFanoData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "reference: {}, max: {}, lower_width: {}, upper_len: {}, num_elements: {}, \
             first_rank: {}, upper_base: {}",
            self.reference,
            self.max,
            self.lower_width,
            self.upper_len,
            self.num_elements,
            self.first_rank,
            self.upper_base
        )
    }
}

/// The parts of an [`EliasFanoData`], as [`EliasFanoData::try_new`] takes them.
#[derive(Clone, Debug)]
pub(crate) struct EliasFanoParts {
    pub(crate) upper: ByteBuffer,
    pub(crate) samples0: ByteBuffer,
    pub(crate) samples1: ByteBuffer,
    pub(crate) reference: Scalar,
    pub(crate) max: Scalar,
    pub(crate) lower_width: u8,
    pub(crate) upper_len: u64,
    pub(crate) num_elements: u64,
    pub(crate) first_rank: u64,
    pub(crate) upper_base: u64,
}

impl EliasFanoData {
    /// Construct the per-array data, validating what can be checked without the child slot.
    pub(crate) fn try_new(parts: EliasFanoParts) -> VortexResult<Self> {
        let EliasFanoParts {
            upper,
            samples0,
            samples1,
            reference,
            max,
            lower_width,
            upper_len,
            num_elements,
            first_rank,
            upper_base,
        } = parts;
        vortex_ensure!(
            reference.dtype().is_int() && !reference.dtype().is_nullable(),
            "Elias-Fano reference must be a non-nullable integer, got {}",
            reference.dtype()
        );
        vortex_ensure_eq!(max.dtype(), reference.dtype());
        vortex_ensure!(
            lower_width <= ef::MAX_LOWER_WIDTH,
            "Elias-Fano lower_width {lower_width} exceeds {}",
            ef::MAX_LOWER_WIDTH
        );
        for samples in [&samples0, &samples1] {
            vortex_ensure!(
                samples.len().is_multiple_of(size_of::<u64>()),
                "Elias-Fano samples buffer of {} bytes is not a whole number of u64s",
                samples.len()
            );
        }

        Ok(Self {
            upper,
            samples0,
            samples1,
            reference,
            max,
            lower_width,
            upper_len,
            num_elements,
            first_rank,
            upper_base,
        })
    }

    /// A copy of the parts, for rebuilding with one of them replaced.
    pub(crate) fn parts(&self) -> EliasFanoParts {
        EliasFanoParts {
            upper: self.upper.clone(),
            samples0: self.samples0.clone(),
            samples1: self.samples1.clone(),
            reference: self.reference.clone(),
            max: self.max.clone(),
            lower_width: self.lower_width,
            upper_len: self.upper_len,
            num_elements: self.num_elements,
            first_rank: self.first_rank,
            upper_base: self.upper_base,
        }
    }

    /// The window `bounds` keeps, starting at `first_rank`, sharing this window's buffers.
    pub(crate) fn sliced(&self, bounds: &ef::SliceBounds, first_rank: u64) -> Self {
        let entries =
            |range: &Range<usize>| range.start * size_of::<u64>()..range.end * size_of::<u64>();
        Self {
            upper: self.upper.slice_unaligned(bounds.upper.clone()),
            samples0: self.samples0.slice_unaligned(entries(&bounds.samples0)),
            samples1: self.samples1.slice_unaligned(entries(&bounds.samples1)),
            first_rank,
            upper_base: bounds.upper_base,
            ..self.clone()
        }
    }

    /// The value subtracted from every element before encoding, and added back on read.
    #[inline]
    pub fn reference_scalar(&self) -> &Scalar {
        &self.reference
    }

    /// The largest value of the *encoded* sequence, fixing the universe the upper array was sized
    /// for, so slicing leaves it untouched. Therefore **not** the maximum of a sliced array — read
    /// the element at `len - 1` for that.
    #[inline]
    pub fn max_scalar(&self) -> &Scalar {
        &self.max
    }

    /// Number of low bits stored per element in the child slot.
    #[inline]
    pub fn lower_width(&self) -> u8 {
        self.lower_width
    }

    /// Length in bits of the encoded sequence's upper array.
    #[inline]
    pub fn upper_len(&self) -> u64 {
        self.upper_len
    }

    /// Number of elements in the encoded sequence, which a slice keeps.
    #[inline]
    pub fn num_elements(&self) -> u64 {
        self.num_elements
    }

    /// Rank of this array's element 0 within the encoded sequence: element `i` is rank
    /// `first_rank + i`.
    #[inline]
    pub fn first_rank(&self) -> u64 {
        self.first_rank
    }

    /// Absolute position of the upper buffer's first bit: the set bit of [`Self::first_rank`] in a
    /// slice, or 0.
    #[inline]
    pub fn upper_base(&self) -> u64 {
        self.upper_base
    }

    #[inline]
    pub(crate) fn upper_buffer(&self) -> &ByteBuffer {
        &self.upper
    }

    #[inline]
    pub(crate) fn samples0_buffer(&self) -> &ByteBuffer {
        &self.samples0
    }

    #[inline]
    pub(crate) fn samples1_buffer(&self) -> &ByteBuffer {
        &self.samples1
    }

    /// The codec's view of this array's window of `len` elements.
    pub(crate) fn window(&self, len: usize) -> ef::Window<'_> {
        ef::Window {
            upper: self.upper.as_slice(),
            upper_base: self.upper_base,
            first_rank: self.first_rank,
            len,
            samples0: self.samples0.as_slice(),
            samples1: self.samples1.as_slice(),
        }
    }

    /// The reference value as a sign-extended 64-bit pattern.
    ///
    /// Encoding works in this domain throughout: `element =
    /// value_bits.wrapping_sub(reference_bits)` and back. Sign-extend, wrap, truncate is exactly
    /// two's complement, so one `u64` path serves every integer ptype, signed or not.
    #[inline]
    pub(crate) fn reference_bits(&self) -> u64 {
        scalar_bits(&self.reference)
    }

    /// The span of the encoded universe: `max - reference`, so the universe is `span + 1` values.
    #[inline]
    pub(crate) fn span(&self) -> u64 {
        scalar_bits(&self.max).wrapping_sub(scalar_bits(&self.reference))
    }
}

/// The two's-complement bit pattern of an integer scalar, sign-extended to 64 bits.
pub(crate) fn scalar_bits(scalar: &Scalar) -> u64 {
    let pvalue = scalar
        .as_primitive()
        .pvalue()
        .vortex_expect("Elias-Fano bounds are non-null integers");
    match_each_integer_ptype!(pvalue.ptype(), |P| {
        let value: P = pvalue
            .cast::<P>()
            .vortex_expect("pvalue is already of this ptype");
        value.as_()
    })
}

pub(crate) fn scalar_from_bits(dtype: &DType, bits: u64) -> VortexResult<Scalar> {
    let value = match_each_integer_ptype!(dtype.as_ptype(), |P| {
        // Narrowing the sign-extended pattern back to the array's width is exact.
        let value: P = bits.as_();
        ScalarValue::Primitive(PValue::from(value))
    });
    Scalar::try_new(dtype.clone(), Some(value))
}

impl ArrayHash for EliasFanoData {
    fn array_hash<H: Hasher>(&self, state: &mut H, accuracy: EqMode) {
        self.reference.hash(state);
        self.max.hash(state);
        self.lower_width.hash(state);
        self.upper_len.hash(state);
        self.num_elements.hash(state);
        self.first_rank.hash(state);
        self.upper_base.hash(state);
        self.upper.array_hash(state, accuracy);
        self.samples0.array_hash(state, accuracy);
        self.samples1.array_hash(state, accuracy);
    }
}

impl ArrayEq for EliasFanoData {
    fn array_eq(&self, other: &Self, accuracy: EqMode) -> bool {
        self.reference == other.reference
            && self.max == other.max
            && self.lower_width == other.lower_width
            && self.upper_len == other.upper_len
            && self.num_elements == other.num_elements
            && self.first_rank == other.first_rank
            && self.upper_base == other.upper_base
            && self.upper.array_eq(&other.upper, accuracy)
            && self.samples0.array_eq(&other.samples0, accuracy)
            && self.samples1.array_eq(&other.samples1, accuracy)
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
        let lower = EliasFanoSlotsView::from_slots(slots).lower;
        validate_parts(data, lower, dtype, len)
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        3
    }

    fn buffer(array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        match idx {
            0 => BufferHandle::new_host(array.upper_buffer().clone()),
            1 => BufferHandle::new_host(array.samples0_buffer().clone()),
            2 => BufferHandle::new_host(array.samples1_buffer().clone()),
            _ => vortex_panic!("EliasFanoArray buffer index {idx} out of bounds"),
        }
    }

    fn buffer_name(_array: ArrayView<'_, Self>, idx: usize) -> Option<String> {
        match idx {
            0 => Some("upper".to_string()),
            1 => Some("samples0".to_string()),
            2 => Some("samples1".to_string()),
            _ => None,
        }
    }

    fn with_buffers(
        &self,
        array: ArrayView<'_, Self>,
        buffers: &[BufferHandle],
    ) -> VortexResult<ArrayParts<Self>> {
        vortex_ensure_eq!(buffers.len(), 3);
        // Back through `try_new` rather than assigning the fields, so a replacement buffer still
        // has to satisfy the constructor's invariants.
        let data = EliasFanoData::try_new(EliasFanoParts {
            upper: buffers[0].try_to_host_sync()?,
            samples0: buffers[1].try_to_host_sync()?,
            samples1: buffers[2].try_to_host_sync()?,
            ..array.data().parts()
        })?;
        Ok(ArrayParts::new(
            self.clone(),
            array.dtype().clone(),
            array.len(),
            data,
            array.slots().iter().cloned().collect(),
        ))
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        EliasFanoSlots::NAMES[idx].to_string()
    }

    fn serialize(
        array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(
            EliasFanoMetadata {
                reference: Some(ScalarValue::to_proto(array.reference_scalar().value())),
                max: Some(ScalarValue::to_proto(array.max_scalar().value())),
                lower_width: u32::from(array.lower_width()),
                upper_len: array.upper_len(),
                first_rank: array.first_rank(),
                num_elements: array.num_elements(),
                upper_base: array.upper_base(),
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
        session: &VortexSession,
    ) -> VortexResult<ArrayParts<Self>> {
        vortex_ensure_eq!(buffers.len(), 3);
        vortex_ensure_eq!(children.len(), 1);
        let metadata = EliasFanoMetadata::decode(metadata)?;

        let bound = |value: Option<&ProtoScalarValue>, what: &str| {
            let value = value.ok_or_else(|| vortex_err!("Elias-Fano {what} is required"))?;
            Scalar::from_proto_value(value, dtype, session)
        };

        let lower = children.get(EliasFanoSlots::LOWER, &LOWER_DTYPE, len)?;

        let data = EliasFanoData::try_new(EliasFanoParts {
            upper: buffers[0].try_to_host_sync()?,
            samples0: buffers[1].try_to_host_sync()?,
            samples1: buffers[2].try_to_host_sync()?,
            reference: bound(metadata.reference.as_ref(), "reference")?,
            max: bound(metadata.max.as_ref(), "max")?,
            lower_width: u8::try_from(metadata.lower_width).map_err(|_| {
                vortex_err!("Elias-Fano lower_width {} > 255", metadata.lower_width)
            })?,
            upper_len: metadata.upper_len,
            num_elements: metadata.num_elements,
            first_rank: metadata.first_rank,
            upper_base: metadata.upper_base,
        })?;

        Ok(ArrayParts::new(
            self.clone(),
            dtype.clone(),
            len,
            data,
            smallvec![Some(lower)],
        ))
    }

    fn execute(array: Array<Self>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        Ok(ExecutionResult::done(
            elias_fano_decompress(&array, ctx)?.into_array(),
        ))
    }
}

impl OperationsVTable<EliasFano> for EliasFano {
    // No probe state: like every other encoding in the tree, this stays on the `scalar_at` path
    // that `probe_scalar` defaults to.
    type ProbeState = ();

    /// One sampled `select1` for the high part, and one read of the low-bits child.
    fn scalar_at(
        array: ArrayView<'_, EliasFano>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        access_at(array, index, ctx)
    }
}

impl ValidityVTable<EliasFano> for EliasFano {
    fn validity(_array: ArrayView<'_, EliasFano>) -> VortexResult<Validity> {
        Ok(Validity::NonNullable)
    }
}

/// Elias-Fano encoding for monotonically non-decreasing integer sequences.
#[derive(Clone, Debug)]
pub struct EliasFano;

impl EliasFano {
    /// Assemble an Elias-Fano array from encoded parts.
    ///
    /// Prefer [`elias_fano_encode`](crate::elias_fano_encode) unless you already hold a layout,
    /// which is the case for a slice, a cast, or a rewrite of the low-bits child.
    pub fn try_new(
        data: EliasFanoData,
        lower: ArrayRef,
        len: usize,
    ) -> VortexResult<EliasFanoArray> {
        let dtype = data.reference_scalar().dtype().clone();
        let slots: ArraySlots = smallvec![Some(lower)];
        Array::try_from_parts(ArrayParts::new(EliasFano, dtype, len, data, slots))
            .map(|array| array.with_stats_set(Self::stats()))
    }

    /// Assemble an Elias-Fano array without validating it.
    ///
    /// # Safety
    ///
    /// Caller must ensure the parts satisfy the invariants checked by [`EliasFano::try_new`].
    pub(crate) unsafe fn new_unchecked(
        data: EliasFanoData,
        lower: ArrayRef,
        len: usize,
    ) -> EliasFanoArray {
        let dtype = data.reference_scalar().dtype().clone();
        #[cfg(debug_assertions)]
        if let Err(error) = validate_parts(&data, &lower, &dtype, len) {
            vortex_panic!("invalid Elias-Fano parts: {error}");
        }
        let slots: ArraySlots = smallvec![Some(lower)];
        // SAFETY: upheld by the caller.
        unsafe { Array::from_parts_unchecked(ArrayParts::new(EliasFano, dtype, len, data, slots)) }
            .with_stats_set(Self::stats())
    }

    /// Statistics that hold for every Elias-Fano array by construction.
    ///
    /// `IsSorted` is required, not a nicety: [`ListArray::new`](vortex_array::arrays::ListArray)
    /// refuses offsets that do not report it. `IsStrictSorted` is absent because repeated offsets
    /// are legal — an empty list contributes two identical ones — so it must be computed.
    pub(crate) fn stats() -> StatsSet {
        // SAFETY: a single stat cannot be duplicated.
        unsafe {
            StatsSet::new_unchecked(smallvec![(
                Stat::IsSorted,
                StatPrecision::Exact(true.into()),
            )])
        }
    }
}

fn validate_parts(
    data: &EliasFanoData,
    lower: &ArrayRef,
    dtype: &DType,
    len: usize,
) -> VortexResult<()> {
    vortex_ensure!(
        dtype.is_int(),
        "Elias-Fano requires an integer dtype, got {dtype}"
    );
    vortex_ensure!(
        !dtype.is_nullable(),
        "Elias-Fano requires a non-nullable dtype, got {dtype}"
    );
    vortex_ensure!(
        data.reference_scalar().dtype() == dtype,
        "Elias-Fano reference dtype {} does not match array dtype {dtype}",
        data.reference_scalar().dtype()
    );
    // Any integer array of the right width is acceptable here, not just `BitPacked`: a file
    // roundtrip can hand the slot back wrapped (for example as `Patched(BitPacked)`), and a
    // rewrite may replace it outright.
    vortex_ensure!(
        lower.dtype() == &LOWER_DTYPE,
        "Elias-Fano low-bits child must be {LOWER_DTYPE}, got {}",
        lower.dtype()
    );
    // A globally packed slot's width is metadata, so it is free to check. Narrower is legal, since
    // a rewrite may repack tighter; wider could hold low parts the reader would mask off. Blocked
    // widths live in a child array, so a blocked slot is accepted like any other integer child.
    if let Some(packed) = lower.as_opt::<BitPacked>()
        && let BitWidthsView::Global(bit_width) = packed.bit_widths()
    {
        vortex_ensure!(
            bit_width <= data.lower_width(),
            "Elias-Fano low-bits child is packed at {bit_width} bits, above the {} the layout \
             allows",
            data.lower_width()
        );
    }

    vortex_ensure_eq!(lower.len(), len);

    // Everything below is codec geometry, re-derived from `(span, n)` alone: the widths, the upper
    // length, where the window sits in it, and every sample's count, range and order.
    ef::validate_layout(
        data.span(),
        usize::try_from(data.num_elements())?,
        data.lower_width(),
        data.upper_len(),
        &data.window(len),
    )
    .map_err(malformed)
}

#[cfg(test)]
mod tests {
    use vortex_array::test_harness::check_metadata;

    use super::*;

    #[cfg_attr(miri, ignore)]
    #[test]
    fn test_elias_fano_metadata() {
        check_metadata(
            "elias_fano.metadata",
            &EliasFanoMetadata {
                reference: Some((&ScalarValue::from(i64::MIN)).into()),
                max: Some((&ScalarValue::from(i64::MAX)).into()),
                lower_width: u32::from(u8::MAX),
                upper_len: u64::MAX,
                first_rank: u64::MAX,
                num_elements: u64::MAX,
                upper_base: u64::MAX,
            }
            .encode_to_vec(),
        );
    }
}
