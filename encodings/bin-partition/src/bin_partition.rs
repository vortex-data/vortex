// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;

use prost::Message;
use vortex_array::Array;
use vortex_array::ArrayEq;
use vortex_array::ArrayHash;
use vortex_array::ArrayId;
use vortex_array::ArrayParts;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::ExecutionResult;
use vortex_array::IntoArray;
use vortex_array::Precision;
use vortex_array::TypedArrayRef;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::scalar::Scalar;
use vortex_array::serde::ArrayChildren;
use vortex_array::smallvec::smallvec;
use vortex_array::validity::Validity;
use vortex_array::vtable::OperationsVTable;
use vortex_array::vtable::VTable;
use vortex_array::vtable::ValidityChild;
use vortex_array::vtable::ValidityVTableFromChild;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::BinInfo;
use crate::BinPartitionMetadata;
use crate::VarWidthBitPacked;

/// A [`BinPartition`]-encoded Vortex array of `i64` values.
///
/// Decomposes the input stream into `(bin_idx, offset)` such that
/// `value[i] = bins[bin_idx[i]].lower + offset[i] as i64`. The offset
/// child is a [`VarWidthBitPackedArray`][crate::VarWidthBitPackedArray] whose per-bin widths match each
/// bin's range.
///
/// # Random access
///
/// `scalar_at(i)` is O(1) modulo the O(64) bound on the inner
/// [`VarWidthBitPackedArray`][crate::VarWidthBitPackedArray].
pub type BinPartitionArray = Array<BinPartition>;

/// Slot holding the bin-index child (`Primitive<u8>`).
pub(crate) const BIN_IDX_SLOT: usize = 0;
/// Slot holding the per-element offsets ([`VarWidthBitPackedArray`][crate::VarWidthBitPackedArray]).
pub(crate) const OFFSET_SLOT: usize = 1;
const NUM_SLOTS: usize = 2;
const SLOT_NAMES: [&str; NUM_SLOTS] = ["bin_idx", "offset"];

/// Default upper bound on the number of bins selected by [`BinPartition::encode`].
pub const DEFAULT_MAX_BINS: usize = 16;

/// Maximum number of bins supported (capped by the `u8` bin_idx width).
pub const MAX_BINS: usize = 256;

/// Marker type implementing [`VTable`] for [`BinPartition`].
#[derive(Clone, Debug)]
pub struct BinPartition;

/// Per-array data for [`BinPartitionArray`].
#[derive(Clone, Debug)]
pub struct BinPartitionData {
    bins: Vec<Bin>,
}

/// One bin in a [`BinPartitionArray`].
///
/// Owns the lower bound and the offset bit width separately from the
/// (prost) [`BinInfo`] wire format.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Bin {
    /// Inclusive lower bound. Each element assigned to this bin satisfies
    /// `lower <= value`.
    pub lower: i64,
    /// Bit width of the offset stored in [`VarWidthBitPackedArray`][crate::VarWidthBitPackedArray]. A
    /// width of `0` means the bin is a single-element bucket.
    pub offset_bits: u8,
}

impl BinPartitionData {
    pub(crate) fn new(bins: Vec<Bin>) -> Self {
        Self { bins }
    }

    /// Per-bin lower bound and offset bit width.
    pub fn bins(&self) -> &[Bin] {
        &self.bins
    }
}

impl Display for BinPartitionData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "n_bins: {}", self.bins.len())
    }
}

impl ArrayHash for BinPartitionData {
    fn array_hash<H: Hasher>(&self, state: &mut H, _precision: Precision) {
        self.bins.hash(state);
    }
}

impl ArrayEq for BinPartitionData {
    fn array_eq(&self, other: &Self, _precision: Precision) -> bool {
        self.bins == other.bins
    }
}

impl VTable for BinPartition {
    type TypedArrayData = BinPartitionData;

    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromChild;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.bin_partition");
        *ID
    }

    fn validate(
        &self,
        data: &Self::TypedArrayData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        let bin_idx = slots[BIN_IDX_SLOT]
            .as_ref()
            .vortex_expect("BinPartitionArray bin_idx slot");
        let offset = slots[OFFSET_SLOT]
            .as_ref()
            .vortex_expect("BinPartitionArray offset slot");
        validate_parts(data, dtype, bin_idx, offset, len)
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        0
    }

    fn buffer(_array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        vortex_panic!("BinPartitionArray buffer index {idx} out of bounds")
    }

    fn buffer_name(_array: ArrayView<'_, Self>, idx: usize) -> Option<String> {
        vortex_panic!("BinPartitionArray buffer_name index {idx} out of bounds")
    }

    fn serialize(
        array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        let bins = array
            .data()
            .bins
            .iter()
            .map(|b| BinInfo {
                lower: b.lower,
                offset_bits: b.offset_bits as u32,
            })
            .collect();
        Ok(Some(BinPartitionMetadata { bins }.encode_to_vec()))
    }

    fn deserialize(
        &self,
        dtype: &DType,
        len: usize,
        metadata: &[u8],
        _buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
        _session: &VortexSession,
    ) -> VortexResult<ArrayParts<Self>> {
        let metadata = BinPartitionMetadata::decode(metadata)?;
        if children.len() != NUM_SLOTS {
            vortex_bail!("Expected {NUM_SLOTS} children, got {}", children.len());
        }
        ensure_i64_dtype(dtype)?;
        let bins = decode_bins(metadata.bins)?;

        let bin_idx_dtype = DType::Primitive(PType::U8, Nullability::NonNullable);
        let bin_idx = children.get(BIN_IDX_SLOT, &bin_idx_dtype, len)?;
        let offset_dtype = DType::Primitive(PType::U64, Nullability::NonNullable);
        let offset = children.get(OFFSET_SLOT, &offset_dtype, len)?;
        let data = BinPartitionData::new(bins);
        validate_parts(&data, dtype, &bin_idx, &offset, len)?;
        let slots = smallvec![Some(bin_idx), Some(offset)];
        Ok(ArrayParts::new(self.clone(), dtype.clone(), len, data).with_slots(slots))
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        SLOT_NAMES[idx].to_string()
    }

    fn execute(array: Array<Self>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        let bins = array.data().bins.clone();
        let bin_idx = array.bin_idx().clone().execute::<PrimitiveArray>(ctx)?;
        let offsets = array.offset().clone().execute::<PrimitiveArray>(ctx)?;
        let values = decode_primitive(bin_idx, offsets, &bins);
        Ok(ExecutionResult::done(
            PrimitiveArray::new(values, Validity::NonNullable).into_array(),
        ))
    }
}

/// Ensure the dtype is `i64` non-nullable.
fn ensure_i64_dtype(dtype: &DType) -> VortexResult<()> {
    let ptype = PType::try_from(dtype)?;
    if ptype != PType::I64 {
        vortex_bail!("BinPartitionArray only supports i64 inputs, got {ptype}");
    }
    if dtype.is_nullable() {
        vortex_bail!("BinPartitionArray is non-nullable in this phase");
    }
    Ok(())
}

fn decode_bins(stored: Vec<BinInfo>) -> VortexResult<Vec<Bin>> {
    if stored.is_empty() {
        vortex_bail!("BinPartitionArray must have at least one bin");
    }
    if stored.len() > MAX_BINS {
        vortex_bail!(
            "BinPartitionArray supports at most {MAX_BINS} bins, got {}",
            stored.len()
        );
    }
    let mut out = Vec::with_capacity(stored.len());
    for bin in stored {
        if bin.offset_bits > 64 {
            vortex_bail!(
                "BinPartitionArray offset_bits {} exceeds 64",
                bin.offset_bits
            );
        }
        out.push(Bin {
            lower: bin.lower,
            offset_bits: u8::try_from(bin.offset_bits)
                .vortex_expect("offset_bits <= 64 fits in u8"),
        });
    }
    Ok(out)
}

fn validate_parts(
    data: &BinPartitionData,
    dtype: &DType,
    bin_idx: &ArrayRef,
    offset: &ArrayRef,
    len: usize,
) -> VortexResult<()> {
    ensure_i64_dtype(dtype)?;
    vortex_ensure!(
        !data.bins.is_empty(),
        "BinPartitionArray must have at least one bin"
    );
    vortex_ensure!(
        data.bins.len() <= MAX_BINS,
        "BinPartitionArray supports at most {MAX_BINS} bins, got {}",
        data.bins.len(),
    );
    let expected_bin_idx_dtype = DType::Primitive(PType::U8, Nullability::NonNullable);
    vortex_ensure!(
        bin_idx.dtype() == &expected_bin_idx_dtype,
        "BinPartitionArray bin_idx dtype {} does not match expected {}",
        bin_idx.dtype(),
        expected_bin_idx_dtype,
    );
    vortex_ensure!(
        bin_idx.len() == len,
        "BinPartitionArray bin_idx len {} does not match array len {len}",
        bin_idx.len(),
    );
    let expected_offset_dtype = DType::Primitive(PType::U64, Nullability::NonNullable);
    vortex_ensure!(
        offset.dtype() == &expected_offset_dtype,
        "BinPartitionArray offset dtype {} does not match expected {}",
        offset.dtype(),
        expected_offset_dtype,
    );
    vortex_ensure!(
        offset.len() == len,
        "BinPartitionArray offset len {} does not match array len {len}",
        offset.len(),
    );
    Ok(())
}

/// Extension methods on any typed reference to a [`BinPartitionArray`].
pub trait BinPartitionArrayExt: TypedArrayRef<BinPartition> {
    /// Per-bin lower bound and offset bit width.
    fn bins(&self) -> &[Bin] {
        BinPartitionData::bins(self)
    }

    /// The `Primitive<u8>` bin-index child.
    fn bin_idx(&self) -> &ArrayRef {
        self.as_ref().slots()[BIN_IDX_SLOT]
            .as_ref()
            .vortex_expect("BinPartitionArray bin_idx slot")
    }

    /// The [`VarWidthBitPackedArray`][crate::VarWidthBitPackedArray] offset child.
    fn offset(&self) -> &ArrayRef {
        self.as_ref().slots()[OFFSET_SLOT]
            .as_ref()
            .vortex_expect("BinPartitionArray offset slot")
    }
}

impl<T: TypedArrayRef<BinPartition>> BinPartitionArrayExt for T {}

impl BinPartition {
    /// Construct a [`BinPartitionArray`] from validated parts.
    pub fn try_new(
        bins: Vec<Bin>,
        bin_idx: ArrayRef,
        offset: ArrayRef,
    ) -> VortexResult<BinPartitionArray> {
        let dtype = DType::Primitive(PType::I64, Nullability::NonNullable);
        let len = bin_idx.len();
        let data = BinPartitionData::new(bins);
        validate_parts(&data, &dtype, &bin_idx, &offset, len)?;
        let slots = smallvec![Some(bin_idx), Some(offset)];
        // SAFETY: validate_parts above checked all type/length invariants.
        Ok(unsafe {
            Array::from_parts_unchecked(
                ArrayParts::new(BinPartition, dtype, len, data).with_slots(slots),
            )
        })
    }

    /// Encode an `i64` primitive array into bins.
    ///
    /// The encoder samples the input, chooses quantile-based bin boundaries
    /// (up to `max_bins`), computes per-bin widths, and emits
    /// `(bin_idx, offset)`.
    ///
    /// `max_bins` must be in `1..=256`.
    ///
    /// # Bin selection
    ///
    /// 1. Compute global min/max.
    /// 2. Sample up to 10 000 elements (by stride).
    /// 3. Sort the samples.
    /// 4. Pick `max_bins - 1` evenly-spaced quantile cuts. Each bin's `lower`
    ///    is its lower-quantile sample; the final bin extends to the global
    ///    max. `offset_bits = ceil(log2(upper - lower + 1))`.
    /// 5. Each input element is assigned by binary search on the bin lowers;
    ///    `offset = value - bin.lower` is encoded as an unsigned u64.
    pub fn encode(
        parray: ArrayView<'_, Primitive>,
        max_bins: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<BinPartitionArray> {
        if !(1..=MAX_BINS).contains(&max_bins) {
            vortex_bail!(
                "BinPartition::encode requires max_bins in 1..={MAX_BINS}, got {max_bins}"
            );
        }
        let ptype = PrimitiveArrayExt::ptype(&parray);
        if ptype != PType::I64 {
            vortex_bail!("BinPartition::encode requires i64 input, got {ptype}");
        }
        let validity = PrimitiveArrayExt::validity(&parray);
        if !matches!(validity, Validity::NonNullable) {
            vortex_bail!(
                "BinPartition::encode requires non-nullable input; nullable streams are not \
                 supported in this phase"
            );
        }
        let parray = parray.into_owned();
        let buf = parray.into_buffer::<i64>();
        let values = buf.as_slice();
        let n = values.len();

        // Empty or singleton input: emit one bin with offset_bits = 0.
        if n == 0 {
            let bins = vec![Bin {
                lower: 0,
                offset_bits: 0,
            }];
            let bin_idx = primitive_u8(Vec::new());
            let widths = vec![0u8];
            let offset = VarWidthBitPacked::encode(bin_idx.clone(), widths, &[], ctx)?;
            return Self::try_new(bins, bin_idx, offset.into_array());
        }

        let bins = choose_bins(values, max_bins);
        let widths: Vec<u8> = bins.iter().map(|b| b.offset_bits).collect();
        let (bin_idx_vec, offsets) = assign(values, &bins);
        let bin_idx = primitive_u8(bin_idx_vec);
        let offset = VarWidthBitPacked::encode(bin_idx.clone(), widths, &offsets, ctx)?;

        Self::try_new(bins, bin_idx, offset.into_array())
    }
}

/// Construct a non-nullable `Primitive<u8>` from a `Vec<u8>`.
fn primitive_u8(values: Vec<u8>) -> ArrayRef {
    PrimitiveArray::new(Buffer::from(values), Validity::NonNullable).into_array()
}

/// Choose bin boundaries from `values`. Returns at least one bin.
fn choose_bins(values: &[i64], max_bins: usize) -> Vec<Bin> {
    debug_assert!(!values.is_empty());
    debug_assert!(max_bins >= 1);

    let (mut min, mut max) = (values[0], values[0]);
    for &v in &values[1..] {
        if v < min {
            min = v;
        }
        if v > max {
            max = v;
        }
    }
    if min == max {
        // All values identical: one bin with offset_bits = 0.
        return vec![Bin {
            lower: min,
            offset_bits: 0,
        }];
    }

    // Sample up to 10_000 elements by stride.
    const MAX_SAMPLES: usize = 10_000;
    let stride = (values.len() / MAX_SAMPLES).max(1);
    let mut samples: Vec<i64> = values.iter().step_by(stride).copied().collect();
    if samples.len() > MAX_SAMPLES {
        samples.truncate(MAX_SAMPLES);
    }
    samples.sort_unstable();
    // Force the min and max into the sample so quantile cuts cover the
    // full range.
    if samples[0] != min {
        samples.insert(0, min);
    }
    if *samples.last().vortex_expect("non-empty samples") != max {
        samples.push(max);
    }

    // Pick `max_bins` lower bounds via evenly-spaced quantile indices.
    let mut lowers: Vec<i64> = Vec::with_capacity(max_bins);
    for b in 0..max_bins {
        let idx = (b * (samples.len() - 1)) / max_bins;
        lowers.push(samples[idx]);
    }
    lowers[0] = min;
    // Deduplicate while preserving order.
    lowers.dedup();
    let n_bins = lowers.len();

    // Compute upper bound per bin and convert to offset_bits.
    let mut bins = Vec::with_capacity(n_bins);
    for i in 0..n_bins {
        let lower = lowers[i];
        let upper = if i + 1 < n_bins {
            lowers[i + 1] - 1
        } else {
            max
        };
        // `upper >= lower` because `lowers` is sorted ascending and
        // deduplicated. The offset range `upper - lower` fits in u64 because
        // `i64` values span at most `2^64 - 1`.
        let span = (upper as i128) - (lower as i128) + 1;
        let offset_bits = if span <= 1 {
            0
        } else {
            // ceil(log2(span)).
            let span_u = span as u128;
            let bits = 128 - (span_u - 1).leading_zeros();
            bits.min(64) as u8
        };
        bins.push(Bin { lower, offset_bits });
    }
    bins
}

/// Assign each value to a bin via binary search on bin lowers; compute the
/// `(value - bin.lower) as u64` offset.
fn assign(values: &[i64], bins: &[Bin]) -> (Vec<u8>, Vec<u64>) {
    let n = values.len();
    let mut bin_idx_vec = Vec::with_capacity(n);
    let mut offsets = Vec::with_capacity(n);
    for &v in values {
        let b = bin_for(bins, v);
        // `bins.len() <= MAX_BINS == 256`, so `b` fits in u8.
        bin_idx_vec.push(u8::try_from(b).vortex_expect("bin index < MAX_BINS fits in u8"));
        // `v >= bins[b].lower` by construction of `bin_for`. The wrapping
        // sub-then-as-u64 gives the unsigned offset directly without
        // touching `i128`.
        let off = (v as u64).wrapping_sub(bins[b].lower as u64);
        offsets.push(off);
    }
    (bin_idx_vec, offsets)
}

/// Binary search for the bin whose `lower <= v` and (if not the last bin)
/// next bin's `lower > v`.
fn bin_for(bins: &[Bin], v: i64) -> usize {
    // partition_point returns the first index where `lower > v`. The bin we
    // want is one before that, clamped to `0..bins.len()`.
    let p = bins.partition_point(|b| b.lower <= v);
    p.saturating_sub(1)
}

/// Reconstruct an `i64` primitive from `bin_idx` + `offset`.
fn decode_primitive(bin_idx: PrimitiveArray, offsets: PrimitiveArray, bins: &[Bin]) -> Buffer<i64> {
    let bi = bin_idx.into_buffer::<u8>();
    let bi = bi.as_slice();
    let off = offsets.into_buffer::<u64>();
    let off = off.as_slice();
    debug_assert_eq!(bi.len(), off.len());
    let mut out = BufferMut::<i64>::with_capacity(bi.len());
    for i in 0..bi.len() {
        let bin = &bins[bi[i] as usize];
        // Wrapping add in u64 mirrors the assign-time wrapping sub. The
        // round-trip is bit-exact because the encoded offset is always
        // exactly `v - bin.lower` mod 2^64.
        let value = (bin.lower as u64).wrapping_add(off[i]) as i64;
        out.push(value);
    }
    out.freeze()
}

impl OperationsVTable<BinPartition> for BinPartition {
    fn scalar_at(
        array: ArrayView<'_, BinPartition>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let bin_idx_scalar = array.bin_idx().execute_scalar(index, ctx)?;
        let offset_scalar = array.offset().execute_scalar(index, ctx)?;
        let bin = bin_idx_scalar
            .as_primitive()
            .typed_value::<u8>()
            .vortex_expect("BinPartition bin_idx scalar must be u8") as usize;
        let off = offset_scalar
            .as_primitive()
            .typed_value::<u64>()
            .vortex_expect("BinPartition offset scalar must be u64");
        let lower = array.data().bins[bin].lower;
        let value = (lower as u64).wrapping_add(off) as i64;
        Ok(Scalar::primitive(value, Nullability::NonNullable))
    }
}

impl ValidityChild<BinPartition> for BinPartition {
    fn validity_child(array: ArrayView<'_, BinPartition>) -> ArrayRef {
        array.bin_idx().clone()
    }
}

#[cfg(test)]
mod tests {
    use rand::RngExt;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;
    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::LEGACY_SESSION;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::scalar::Scalar;
    use vortex_error::VortexResult;

    use super::*;

    fn round_trip(values: Vec<i64>, max_bins: usize) -> VortexResult<BinPartitionArray> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let parray = PrimitiveArray::from_iter(values.iter().copied());
        let encoded = BinPartition::encode(parray.as_view(), max_bins, &mut ctx)?;
        assert_eq!(encoded.len(), parray.len());
        let decoded = encoded
            .clone()
            .into_array()
            .execute::<PrimitiveArray>(&mut ctx)?;
        assert_arrays_eq!(decoded, parray);
        Ok(encoded)
    }

    #[test]
    fn skewed_geometric_round_trip() -> VortexResult<()> {
        // Geometric: x[i] = 2^k for k cycling 0..16. Most values small, few
        // large; bin partition should pick log-density boundaries.
        let n = 4_096;
        let mut rng = SmallRng::seed_from_u64(0xDEAD);
        let values: Vec<i64> = (0..n).map(|_| 1i64 << rng.random_range(0u32..16)).collect();
        let encoded = round_trip(values, 16)?;
        // At least 2 bins should be selected.
        assert!(
            encoded.bins().len() >= 2,
            "expected >=2 bins, got {}",
            encoded.bins().len()
        );
        Ok(())
    }

    #[test]
    fn uniform_random_round_trip() -> VortexResult<()> {
        let n = 4_096;
        let mut rng = SmallRng::seed_from_u64(0xBEEF);
        let values: Vec<i64> = (0..n)
            .map(|_| rng.random_range(-1_000_000i64..1_000_000))
            .collect();
        round_trip(values, 16)?;
        Ok(())
    }

    #[test]
    fn constant_input_single_bin() -> VortexResult<()> {
        let values = vec![42i64; 1024];
        let encoded = round_trip(values, 16)?;
        assert_eq!(encoded.bins().len(), 1);
        assert_eq!(encoded.bins()[0].offset_bits, 0);
        Ok(())
    }

    #[test]
    fn monotone_round_trip() -> VortexResult<()> {
        let n = 1024;
        let values: Vec<i64> = (0..n as i64).collect();
        round_trip(values, 16)?;
        Ok(())
    }

    #[test]
    fn scalar_at_matches_canonical_decode() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let n_elems = 2048;
        let mut rng = SmallRng::seed_from_u64(0xC0DE);
        let values: Vec<i64> = (0..n_elems)
            .map(|_| (1i64 << rng.random_range(0u32..12)) + rng.random_range(-3i64..=3))
            .collect();
        let parray = PrimitiveArray::from_iter(values.iter().copied());
        let encoded = BinPartition::encode(parray.as_view(), 16, &mut ctx)?;
        let arr = encoded.into_array();
        let decoded = arr
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)?
            .into_buffer::<i64>();
        let mut idx_rng = SmallRng::seed_from_u64(0xCAFE);
        for _ in 0..64 {
            let idx = idx_rng.random_range(0..n_elems);
            let scalar = arr.execute_scalar(idx, &mut ctx)?;
            assert_eq!(scalar, Scalar::from(decoded.as_slice()[idx]));
        }
        Ok(())
    }

    #[rstest]
    #[case::zero(0usize)]
    #[case::too_many(257usize)]
    fn max_bins_validation(#[case] max_bins: usize) -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let parray = PrimitiveArray::from_iter([1i64, 2, 3]);
        let err = BinPartition::encode(parray.as_view(), max_bins, &mut ctx);
        assert!(
            err.is_err(),
            "expected error for max_bins={max_bins}, got {err:?}"
        );
        Ok(())
    }

    #[test]
    fn empty_round_trip() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let parray = PrimitiveArray::from_iter(Vec::<i64>::new());
        let encoded = BinPartition::encode(parray.as_view(), 16, &mut ctx)?;
        assert_eq!(encoded.len(), 0);
        let decoded = encoded.into_array().execute::<PrimitiveArray>(&mut ctx)?;
        assert_arrays_eq!(decoded, parray);
        Ok(())
    }

    #[test]
    fn singleton_round_trip() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let parray = PrimitiveArray::from_iter([7i64]);
        let encoded = BinPartition::encode(parray.as_view(), 16, &mut ctx)?;
        assert_eq!(encoded.len(), 1);
        let s = encoded.clone().into_array().execute_scalar(0, &mut ctx)?;
        assert_eq!(s, Scalar::from(7i64));
        let decoded = encoded.into_array().execute::<PrimitiveArray>(&mut ctx)?;
        assert_arrays_eq!(decoded, parray);
        Ok(())
    }
}
