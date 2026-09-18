// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;

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
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_fastlanes::BitPacked;
use vortex_fastlanes::BitPackedArrayExt;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::access::access_at;
use crate::compress::elias_fano_decompress;
use crate::ef;
use crate::malformed;
use crate::rules::RULES;

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
    /// bit-packed, or a constant zero array at width zero. Its length is the *encoded* element
    /// count, which after a slice exceeds the array's own; see [`EliasFanoData::first_rank`].
    #[slot(0)]
    pub lower: ArrayRef,
}

/// Wire-format metadata persisted alongside the buffers `[upper, samples]` and [`EliasFanoSlots`].
///
/// Only what cannot be re-derived: the seam between the two sample tables is absent, because
/// `ef::num_samples0` recovers it from the universe.
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
    /// Length in bits of the `upper` buffer's bit array.
    #[prost(uint64, tag = "4")]
    upper_len: u64,
    /// Rank of this array's first element within the encoded sequence.
    #[prost(uint64, tag = "5")]
    first_rank: u64,
    /// Number of elements in the encoded sequence. Duplicates the low-bits child's length in
    /// memory, but deserialization must declare a child's length before constructing it, and after
    /// a slice that length is not the array's own.
    #[prost(uint64, tag = "6")]
    num_elements: u64,
}

/// An Elias-Fano encoded monotonically non-decreasing integer sequence.
///
/// Holds only what cannot be re-derived from the layout in the [`crate::ef`] module: the two
/// buffers, the universe bounds, and the three numbers that size it.
///
/// Both buffers are host-resident. The upper array is read bit by bit, so there is no way to serve
/// it one entry at a time from device memory; `with_buffers` and `deserialize` copy to the host
/// once so no accessor below has to ask.
#[derive(Clone, Debug)]
pub struct EliasFanoData {
    /// The unary upper array, `upper_len` bits, byte-padded.
    upper: ByteBuffer,
    /// The zero-sample positions followed by the one-sample positions, as little-endian `u64`s.
    /// Where one table ends and the other begins is derived, not stored; see
    /// [`Self::sample_bytes`]. It is read unaligned, because a deserialized buffer carries no
    /// alignment guarantee.
    samples: ByteBuffer,
    reference: Scalar,
    max: Scalar,
    lower_width: u8,
    upper_len: u64,
    first_rank: u64,
}

impl Display for EliasFanoData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "reference: {}, max: {}, lower_width: {}, upper_len: {}, first_rank: {}",
            self.reference, self.max, self.lower_width, self.upper_len, self.first_rank
        )
    }
}

impl EliasFanoData {
    /// Construct the per-array data, validating what can be checked without the child slot.
    // There is one parameter per metadata field, which is what makes the two sides easy to line up.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn try_new(
        upper: ByteBuffer,
        samples: ByteBuffer,
        reference: Scalar,
        max: Scalar,
        lower_width: u8,
        upper_len: u64,
        first_rank: u64,
    ) -> VortexResult<Self> {
        vortex_ensure!(
            reference.dtype().is_int() && !reference.dtype().is_nullable(),
            "Elias-Fano reference must be a non-nullable integer, got {}",
            reference.dtype()
        );
        vortex_ensure!(
            max.dtype() == reference.dtype(),
            "Elias-Fano max dtype {} does not match reference dtype {}",
            max.dtype(),
            reference.dtype()
        );
        vortex_ensure!(
            lower_width <= ef::MAX_LOWER_WIDTH,
            "Elias-Fano lower_width {lower_width} exceeds {}",
            ef::MAX_LOWER_WIDTH
        );
        vortex_ensure!(
            upper.len() == usize::try_from(upper_len.div_ceil(8))?,
            "Elias-Fano upper buffer is {} bytes, expected {} for {upper_len} bits",
            upper.len(),
            upper_len.div_ceil(8)
        );
        vortex_ensure!(
            samples.len().is_multiple_of(size_of::<u64>()),
            "Elias-Fano samples buffer of {} bytes is not a whole number of u64s",
            samples.len()
        );
        // The zero table comes first, so the buffer must reach at least as far as the seam for
        // `sample_bytes` to be able to split there.
        let span = scalar_bits(&max).wrapping_sub(scalar_bits(&reference));
        let num_samples0 = ef::num_samples0(span, lower_width);
        vortex_ensure!(
            (samples.len() / size_of::<u64>()) as u64 >= num_samples0,
            "Elias-Fano samples buffer holds {} entries, fewer than the {num_samples0} \
             zero-samples its universe implies",
            samples.len() / size_of::<u64>()
        );

        Ok(Self {
            upper,
            samples,
            reference,
            max,
            lower_width,
            upper_len,
            first_rank,
        })
    }

    /// Returns the same layout, read from a different starting rank.
    ///
    /// A slice is nothing more than this; see [`Self::first_rank`].
    pub(crate) fn with_first_rank(mut self, first_rank: u64) -> Self {
        self.first_rank = first_rank;
        self
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

    /// Length in bits of the upper array.
    #[inline]
    pub fn upper_len(&self) -> u64 {
        self.upper_len
    }

    /// Rank of this array's element 0 within the encoded sequence.
    ///
    /// Slicing cannot trim the buffers, because the sample tables hold *absolute* bit positions, so
    /// a slice records where it starts and space is reclaimed on rewrite. This offsets both the
    /// upper-array ranks and the low-bits child, so element `i` is rank `first_rank + i` in both.
    #[inline]
    pub fn first_rank(&self) -> u64 {
        self.first_rank
    }

    /// Number of zero-samples stored at the front of the samples buffer.
    ///
    /// This count is derived from the universe rather than stored, and [`ef::num_samples0`]
    /// explains why the element count drops out of the derivation.
    #[inline]
    pub(crate) fn num_samples0(&self) -> u64 {
        ef::num_samples0(self.span(), self.lower_width)
    }

    #[inline]
    pub(crate) fn upper_buffer(&self) -> &ByteBuffer {
        &self.upper
    }

    #[inline]
    pub(crate) fn samples_buffer(&self) -> &ByteBuffer {
        &self.samples
    }

    /// The zero-sample and one-sample tables, still as raw little-endian bytes.
    ///
    /// The two share a buffer; the seam is recomputed here from the universe alone, which buys back
    /// a metadata field for a shift run once per read rather than per element. Deserialized
    /// buffers carry no alignment guarantee, so entries are read one at a time with
    /// [`ef::read_sample`].
    pub(crate) fn sample_bytes(&self) -> VortexResult<(&[u8], &[u8])> {
        let bytes = self.samples.as_slice();
        let num_samples0 = self.num_samples0();
        // `try_new` already proved the buffer reaches the seam; this raises rather than asserting.
        usize::try_from(num_samples0)
            .ok()
            .and_then(|entries| entries.checked_mul(size_of::<u64>()))
            .and_then(|seam| bytes.split_at_checked(seam))
            .ok_or_else(|| {
                vortex_err!(
                    "Elias-Fano samples buffer of {} bytes is too short for the {num_samples0} \
                     zero-samples its universe implies",
                    bytes.len()
                )
            })
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
// The widening is what sign-extends, and it is a no-op only in the `u64` arm the macro also
// expands to, which is the arm the lint sees.
#[expect(clippy::unnecessary_cast)]
pub(crate) fn scalar_bits(scalar: &Scalar) -> u64 {
    let pvalue = scalar
        .as_primitive()
        .pvalue()
        .vortex_expect("Elias-Fano bounds are non-null integers");
    match_each_integer_ptype!(pvalue.ptype(), |P| {
        pvalue
            .cast::<P>()
            .vortex_expect("pvalue is already of this ptype") as u64
    })
}

pub(crate) fn scalar_from_bits(dtype: &DType, bits: u64) -> VortexResult<Scalar> {
    let value = match_each_integer_ptype!(dtype.as_ptype(), |P| {
        ScalarValue::Primitive(PValue::from(bits as P))
    });
    Scalar::try_new(dtype.clone(), Some(value))
}

impl ArrayHash for EliasFanoData {
    fn array_hash<H: Hasher>(&self, state: &mut H, accuracy: EqMode) {
        self.reference.hash(state);
        self.max.hash(state);
        self.lower_width.hash(state);
        self.upper_len.hash(state);
        self.first_rank.hash(state);
        self.upper.array_hash(state, accuracy);
        self.samples.array_hash(state, accuracy);
    }
}

impl ArrayEq for EliasFanoData {
    fn array_eq(&self, other: &Self, accuracy: EqMode) -> bool {
        self.reference == other.reference
            && self.max == other.max
            && self.lower_width == other.lower_width
            && self.upper_len == other.upper_len
            && self.first_rank == other.first_rank
            && self.upper.array_eq(&other.upper, accuracy)
            && self.samples.array_eq(&other.samples, accuracy)
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
        2
    }

    fn buffer(array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        match idx {
            0 => BufferHandle::new_host(array.upper_buffer().clone()),
            1 => BufferHandle::new_host(array.samples_buffer().clone()),
            _ => vortex_panic!("EliasFanoArray buffer index {idx} out of bounds"),
        }
    }

    fn buffer_name(_array: ArrayView<'_, Self>, idx: usize) -> Option<String> {
        match idx {
            0 => Some("upper".to_string()),
            1 => Some("samples".to_string()),
            _ => None,
        }
    }

    fn with_buffers(
        &self,
        array: ArrayView<'_, Self>,
        buffers: &[BufferHandle],
    ) -> VortexResult<ArrayParts<Self>> {
        vortex_ensure!(
            buffers.len() == 2,
            "Expected 2 buffers, got {}",
            buffers.len()
        );
        let previous = array.data();
        // Back through `try_new` rather than assigning the fields, so a replacement buffer still
        // has to satisfy the constructor's invariants.
        let data = EliasFanoData::try_new(
            buffers[0].try_to_host_sync()?,
            buffers[1].try_to_host_sync()?,
            previous.reference.clone(),
            previous.max.clone(),
            previous.lower_width,
            previous.upper_len,
            previous.first_rank,
        )?;
        Ok(
            ArrayParts::new(self.clone(), array.dtype().clone(), array.len(), data)
                .with_slots(array.slots().iter().cloned().collect()),
        )
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
                num_elements: array.lower().len() as u64,
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
        vortex_ensure!(
            buffers.len() == 2,
            "EliasFanoArray expects 2 buffers, got {}",
            buffers.len()
        );
        vortex_ensure!(
            children.len() == 1,
            "EliasFanoArray expects 1 child, got {}",
            children.len()
        );
        let metadata = EliasFanoMetadata::decode(metadata)?;

        let bound = |value: Option<&ProtoScalarValue>, what: &str| {
            let value = value.ok_or_else(|| vortex_err!("Elias-Fano {what} is required"))?;
            Scalar::from_proto_value(value, dtype, session)
        };

        let lower = children.get(
            EliasFanoSlots::LOWER,
            &LOWER_DTYPE,
            usize::try_from(metadata.num_elements)?,
        )?;

        let data = EliasFanoData::try_new(
            buffers[0].try_to_host_sync()?,
            buffers[1].try_to_host_sync()?,
            bound(metadata.reference.as_ref(), "reference")?,
            bound(metadata.max.as_ref(), "max")?,
            u8::try_from(metadata.lower_width).map_err(|_| {
                vortex_err!("Elias-Fano lower_width {} > 255", metadata.lower_width)
            })?,
            metadata.upper_len,
            metadata.first_rank,
        )?;

        Ok(ArrayParts::new(self.clone(), dtype.clone(), len, data)
            .with_slots(smallvec![Some(lower)]))
    }

    fn execute(array: Array<Self>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        Ok(ExecutionResult::done(
            elias_fano_decompress(&array, ctx)?.into_array(),
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
        Array::try_from_parts(ArrayParts::new(EliasFano, dtype, len, data).with_slots(slots))
            .map(|array| array.with_stats_set(Self::stats()))
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
    // A bit-packed slot's width is metadata, so this is free to check and is the only part of the
    // low bits checkable at all. Narrower is legal — a rewrite may repack tighter. Wider is not:
    // the reader ORs the low bits in under `lower_width`, so anything above bleeds into the high
    // part.
    if let Some(packed) = lower.as_opt::<BitPacked>() {
        vortex_ensure!(
            packed.bit_width() <= data.lower_width(),
            "Elias-Fano low-bits child is packed at {} bits, above the {} the layout allows",
            packed.bit_width(),
            data.lower_width()
        );
    }

    let num_elements = lower.len() as u64;
    let end = data
        .first_rank()
        .checked_add(len as u64)
        .ok_or_else(|| vortex_err!("Elias-Fano slice bounds overflow"))?;
    vortex_ensure!(
        end <= num_elements,
        "Elias-Fano slice of {len} from rank {} exceeds the {num_elements} encoded elements",
        data.first_rank()
    );

    // Everything below is codec geometry, re-derived from `(span, n)` alone: the widths, the upper
    // length, both sample counts, and every sample's range and order.
    ef::validate_layout(
        data.span(),
        num_elements as usize,
        data.lower_width(),
        data.upper_len(),
        data.samples_buffer().as_slice(),
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
            }
            .encode_to_vec(),
        );
    }
}
