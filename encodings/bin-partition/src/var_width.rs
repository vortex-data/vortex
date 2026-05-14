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
use vortex_array::arrays::PrimitiveArray;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::scalar::Scalar;
use vortex_array::serde::ArrayChildren;
use vortex_array::smallvec::smallvec;
use vortex_array::vtable::OperationsVTable;
use vortex_array::vtable::VTable;
use vortex_array::vtable::ValidityChild;
use vortex_array::vtable::ValidityVTableFromChild;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::VarWidthBitPackedMetadata;

/// A variable-width bit-packed primitive integer array.
///
/// Given a parallel stream of bin assignments `bin_idx[i]` and a per-bin
/// width table `widths[bin]`, element `i` occupies `widths[bin_idx[i]]`
/// bits in a single packed bit buffer. The logical dtype is `u64` and
/// values are interpreted as unsigned offsets; sign-extension and bin
/// reconstruction live one layer up in [`crate::BinPartitionArray`].
///
/// # Random access
///
/// `scalar_at(i)` finds batch `b = i / 64`, looks up its starting
/// bit-offset from a precomputed `Buffer<u64>` of per-batch prefix sums,
/// and scans forward within the batch accumulating widths. This is O(64)
/// in the worst case but element-level: no full-array decode is required.
pub type VarWidthBitPackedArray = Array<VarWidthBitPacked>;

/// Slot holding the per-element bin index child.
pub(crate) const BIN_IDX_SLOT: usize = 0;
const NUM_SLOTS: usize = 1;
const SLOT_NAMES: [&str; NUM_SLOTS] = ["bin_idx"];
const NUM_BUFFERS: usize = 2;
const PACKED_BUFFER_NAME: &str = "packed";
const PREFIX_BUFFER_NAME: &str = "batch_prefix";

/// Elements grouped per batch when computing the prefix-sum index. The
/// scalar_at scan within a batch is O(BATCH) bit reads.
pub(crate) const BATCH: usize = 64;

/// Marker type implementing [`VTable`] for [`VarWidthBitPacked`].
#[derive(Clone, Debug)]
pub struct VarWidthBitPacked;

/// Per-array data for [`VarWidthBitPackedArray`].
#[derive(Clone, Debug)]
pub struct VarWidthBitPackedData {
    /// Per-bin bit widths. Each width is `<= 64`.
    widths: Vec<u8>,
    /// The packed bit buffer.
    packed: ByteBuffer,
    /// Bit-offset of the first element of each `BATCH`-sized batch.
    ///
    /// Length is `ceil(n_elements / BATCH) + 1`; the trailing entry holds
    /// the total bit length used by the array. The buffer is laid out as
    /// native-LE `u64` bytes so it can be served as a `BufferHandle`.
    batch_prefix: ByteBuffer,
    /// Logical number of elements.
    n_elements: u64,
}

impl VarWidthBitPackedData {
    pub(crate) fn new(
        widths: Vec<u8>,
        packed: ByteBuffer,
        batch_prefix: ByteBuffer,
        n_elements: u64,
    ) -> Self {
        Self {
            widths,
            packed,
            batch_prefix,
            n_elements,
        }
    }

    /// Per-bin bit widths (`widths[bin_idx[i]]` = number of bits at `i`).
    pub fn widths(&self) -> &[u8] {
        &self.widths
    }

    /// The packed bit buffer.
    pub fn packed(&self) -> &ByteBuffer {
        &self.packed
    }

    /// Logical number of elements stored.
    pub fn n_elements(&self) -> u64 {
        self.n_elements
    }

    /// Returns the bit-offset of the first element of batch `b`.
    pub(crate) fn batch_bit_offset(&self, b: usize) -> u64 {
        let typed = Buffer::<u64>::from_byte_buffer(self.batch_prefix.clone());
        typed.as_slice()[b]
    }
}

impl Display for VarWidthBitPackedData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "n_elements: {}, n_bins: {}, packed_bytes: {}",
            self.n_elements,
            self.widths.len(),
            self.packed.len()
        )
    }
}

impl ArrayHash for VarWidthBitPackedData {
    fn array_hash<H: Hasher>(&self, state: &mut H, _precision: Precision) {
        self.widths.hash(state);
        self.n_elements.hash(state);
        self.packed.as_slice().hash(state);
    }
}

impl ArrayEq for VarWidthBitPackedData {
    fn array_eq(&self, other: &Self, _precision: Precision) -> bool {
        self.widths == other.widths
            && self.n_elements == other.n_elements
            && self.packed.as_slice() == other.packed.as_slice()
    }
}

impl VTable for VarWidthBitPacked {
    type TypedArrayData = VarWidthBitPackedData;

    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromChild;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.var_width_bitpacked");
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
            .vortex_expect("VarWidthBitPackedArray bin_idx slot");
        validate_parts(data, dtype, bin_idx, len)
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        NUM_BUFFERS
    }

    fn buffer(array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        match idx {
            0 => BufferHandle::new_host(array.data().packed.clone()),
            1 => BufferHandle::new_host(array.data().batch_prefix.clone()),
            _ => vortex_panic!("VarWidthBitPackedArray buffer index {idx} out of bounds"),
        }
    }

    fn buffer_name(_array: ArrayView<'_, Self>, idx: usize) -> Option<String> {
        match idx {
            0 => Some(PACKED_BUFFER_NAME.to_string()),
            1 => Some(PREFIX_BUFFER_NAME.to_string()),
            _ => vortex_panic!("VarWidthBitPackedArray buffer_name index {idx} out of bounds"),
        }
    }

    fn serialize(
        array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(
            VarWidthBitPackedMetadata {
                widths: array.data().widths.iter().map(|w| *w as u32).collect(),
                n_elements: array.data().n_elements,
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
        let metadata = VarWidthBitPackedMetadata::decode(metadata)?;
        if children.len() != NUM_SLOTS {
            vortex_bail!("Expected {NUM_SLOTS} children, got {}", children.len());
        }
        if buffers.len() != NUM_BUFFERS {
            vortex_bail!("Expected {NUM_BUFFERS} buffers, got {}", buffers.len());
        }
        ensure_u64_dtype(dtype)?;
        let widths = decode_widths(&metadata.widths)?;

        let bin_idx_dtype = DType::Primitive(PType::U8, Nullability::NonNullable);
        let bin_idx = children.get(BIN_IDX_SLOT, &bin_idx_dtype, len)?;

        let packed = buffers[0].clone().try_to_host_sync()?;
        let batch_prefix = buffers[1].clone().try_to_host_sync()?;
        let data = VarWidthBitPackedData::new(widths, packed, batch_prefix, metadata.n_elements);
        validate_parts(&data, dtype, &bin_idx, len)?;
        let slots = smallvec![Some(bin_idx)];
        Ok(ArrayParts::new(self.clone(), dtype.clone(), len, data).with_slots(slots))
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        SLOT_NAMES[idx].to_string()
    }

    fn execute(array: Array<Self>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        let bin_idx = array.bin_idx().clone().execute::<PrimitiveArray>(ctx)?;
        let bin_idx_buf = bin_idx.into_buffer::<u8>();
        let widths = array.data().widths.clone();
        let packed = array.data().packed.clone();
        let n = array.len();
        let values = decode_buffer(bin_idx_buf.as_slice(), &widths, packed.as_slice(), n);
        Ok(ExecutionResult::done(
            PrimitiveArray::new(values, vortex_array::validity::Validity::NonNullable).into_array(),
        ))
    }
}

/// Ensure the dtype is `u64` non-nullable.
fn ensure_u64_dtype(dtype: &DType) -> VortexResult<()> {
    let ptype = PType::try_from(dtype)?;
    if ptype != PType::U64 {
        vortex_bail!("VarWidthBitPackedArray only supports u64 in this phase, got {ptype}");
    }
    if dtype.is_nullable() {
        vortex_bail!("VarWidthBitPackedArray is non-nullable in this phase");
    }
    Ok(())
}

/// Decode the prost-stored u32 widths into u8 widths, validating that each
/// fits in `<= 64`.
fn decode_widths(stored: &[u32]) -> VortexResult<Vec<u8>> {
    if stored.is_empty() {
        vortex_bail!("VarWidthBitPackedArray must have at least one bin width");
    }
    if stored.len() > 256 {
        vortex_bail!(
            "VarWidthBitPackedArray supports at most 256 bins, got {}",
            stored.len()
        );
    }
    let mut out = Vec::with_capacity(stored.len());
    for &w in stored {
        if w > 64 {
            vortex_bail!("VarWidthBitPackedArray width {w} exceeds 64 bits");
        }
        out.push(u8::try_from(w).vortex_expect("width <= 64 fits in u8"));
    }
    Ok(out)
}

/// Validate the `bin_idx` child agrees with `len` and the dtype is `u64`.
fn validate_parts(
    data: &VarWidthBitPackedData,
    dtype: &DType,
    bin_idx: &ArrayRef,
    len: usize,
) -> VortexResult<()> {
    ensure_u64_dtype(dtype)?;
    let expected_bin_idx_dtype = DType::Primitive(PType::U8, Nullability::NonNullable);
    vortex_ensure!(
        bin_idx.dtype() == &expected_bin_idx_dtype,
        "VarWidthBitPackedArray bin_idx dtype {} does not match expected {}",
        bin_idx.dtype(),
        expected_bin_idx_dtype,
    );
    vortex_ensure!(
        bin_idx.len() == len,
        "VarWidthBitPackedArray bin_idx len {} does not match array len {len}",
        bin_idx.len(),
    );
    vortex_ensure!(
        usize::try_from(data.n_elements)
            .map(|n| n == len)
            .unwrap_or(false),
        "VarWidthBitPackedArray n_elements {} does not match array len {len}",
        data.n_elements,
    );
    let n_batches = len.div_ceil(BATCH);
    let expected_prefix_bytes = (n_batches + 1) * size_of::<u64>();
    vortex_ensure!(
        data.batch_prefix.len() == expected_prefix_bytes,
        "VarWidthBitPackedArray batch_prefix is {} bytes, expected {expected_prefix_bytes}",
        data.batch_prefix.len(),
    );
    Ok(())
}

/// Extension methods on any typed reference to a [`VarWidthBitPackedArray`].
pub trait VarWidthBitPackedArrayExt: TypedArrayRef<VarWidthBitPacked> {
    /// Per-bin bit widths.
    fn widths(&self) -> &[u8] {
        VarWidthBitPackedData::widths(self)
    }

    /// The packed bit buffer.
    fn packed(&self) -> &ByteBuffer {
        VarWidthBitPackedData::packed(self)
    }

    /// The bin-index child array.
    fn bin_idx(&self) -> &ArrayRef {
        self.as_ref().slots()[BIN_IDX_SLOT]
            .as_ref()
            .vortex_expect("VarWidthBitPackedArray bin_idx slot")
    }
}

impl<T: TypedArrayRef<VarWidthBitPacked>> VarWidthBitPackedArrayExt for T {}

impl VarWidthBitPacked {
    /// Construct a [`VarWidthBitPackedArray`] from validated parts. The
    /// caller is responsible for having packed `values[i]` at the correct
    /// width and computed the per-batch prefix sums.
    pub fn try_new(
        widths: Vec<u8>,
        packed: ByteBuffer,
        batch_prefix: ByteBuffer,
        bin_idx: ArrayRef,
        n_elements: usize,
    ) -> VortexResult<VarWidthBitPackedArray> {
        let dtype = DType::Primitive(PType::U64, Nullability::NonNullable);
        let data = VarWidthBitPackedData::new(
            widths,
            packed,
            batch_prefix,
            u64::try_from(n_elements)
                .map_err(|_| vortex_error::vortex_err!("array length {n_elements} exceeds u64"))?,
        );
        validate_parts(&data, &dtype, &bin_idx, n_elements)?;
        let slots = smallvec![Some(bin_idx)];
        // SAFETY: validate_parts above checked all type/length invariants.
        Ok(unsafe {
            Array::from_parts_unchecked(
                ArrayParts::new(VarWidthBitPacked, dtype, n_elements, data).with_slots(slots),
            )
        })
    }

    /// Pack `values` at per-bin widths, returning a [`VarWidthBitPackedArray`].
    ///
    /// `bin_idx` must be `Primitive<u8>` non-nullable of the same length as
    /// `values`. `widths.len()` must be `>= 1 + max(bin_idx)`. Each `values[i]`
    /// must fit in `widths[bin_idx[i]]` bits as an unsigned integer; an
    /// error is returned if any element overflows.
    ///
    /// The caller is expected to have already biased values into
    /// `[0, 2^width)`; this layer does not subtract bin lowers itself.
    pub fn encode(
        bin_idx: ArrayRef,
        widths: Vec<u8>,
        values: &[u64],
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<VarWidthBitPackedArray> {
        let n = values.len();
        vortex_ensure!(
            bin_idx.len() == n,
            "bin_idx len {} does not match values len {n}",
            bin_idx.len(),
        );
        let expected_dtype = DType::Primitive(PType::U8, Nullability::NonNullable);
        vortex_ensure!(
            bin_idx.dtype() == &expected_dtype,
            "VarWidthBitPacked::encode requires Primitive<u8> non-nullable bin_idx, got {}",
            bin_idx.dtype(),
        );
        for &w in &widths {
            if w > 64 {
                vortex_bail!("VarWidthBitPacked width {w} exceeds 64 bits");
            }
        }
        if widths.is_empty() {
            vortex_bail!("VarWidthBitPacked needs at least one bin width");
        }
        if widths.len() > 256 {
            vortex_bail!(
                "VarWidthBitPacked supports at most 256 bins, got {}",
                widths.len()
            );
        }

        // Resolve bin_idx slice. We execute the bin_idx child via the
        // canonical path so the caller can pass any equivalent array; in
        // P4 the encode is only ever called with a PrimitiveArray<u8>
        // already, so this is a no-op clone.
        let bin_idx_typed = bin_idx.clone().execute::<PrimitiveArray>(ctx)?;
        let bin_idx_buf = bin_idx_typed.into_buffer::<u8>();
        let bin_idx_slice = bin_idx_buf.as_slice();

        // Validate bin indices against widths.len() in one fast pass and
        // build a 256-entry width LUT so the hot packing loop can do a
        // single load per element regardless of n_bins. The LUT also makes
        // a stray out-of-range bin_idx safe to load (any byte indexes a
        // valid slot in width_lut), but we still bail above if any bin_idx
        // exceeds widths.len().
        let widths_len = widths.len();
        let mut width_lut = [u8::MAX; 256];
        for (i, &w) in widths.iter().enumerate() {
            width_lut[i] = w;
        }
        for (i, &bi) in bin_idx_slice.iter().enumerate() {
            if (bi as usize) >= widths_len {
                vortex_bail!(
                    "bin_idx[{i}] = {} >= widths.len() = {widths_len}",
                    bi as usize,
                );
            }
        }

        // Fused validate + pack + prefix pass. Walks `values` exactly once,
        // verifying each value fits in its bin's width while emitting the
        // packed bit stream and the per-batch bit-offset prefix sum.
        let (packed, batch_prefix) = pack_and_prefix(values, bin_idx_slice, &width_lut)?;

        Self::try_new(widths, packed, batch_prefix, bin_idx, n)
    }
}

/// Fused validate + pack + per-batch prefix sum.
///
/// Walks `values` exactly once, validating that each value fits in its bin's
/// width while packing into the bit stream and recording the per-batch
/// bit-offset prefix sum. The packing uses a rolling 64-bit accumulator that
/// flushes a full `u64` word any time at least 64 bits are pending, exactly
/// like the previous two-pass implementation; fusing the validation and
/// prefix-sum construction into the same pass halves the memory traffic on
/// `values` and `bin_idx`.
///
/// `width_lut` is a 256-entry `widths[bin_idx]` lookup; the caller has
/// already verified every `bin_idx` resolves to a real width, so any of the
/// 256 entries is safe to load (entries past `widths.len()` contain
/// `u8::MAX`, which would fail the value-fits-in-width check below).
#[expect(clippy::many_single_char_names)]
fn pack_and_prefix(
    values: &[u64],
    bin_idx: &[u8],
    width_lut: &[u8; 256],
) -> VortexResult<(ByteBuffer, ByteBuffer)> {
    let n = values.len();
    let n_batches = n.div_ceil(BATCH);

    // Compute total_bits in a tight pass over bin_idx so we can size the
    // packed buffer correctly. Single-load-per-element via width_lut.
    let mut total_bits: u64 = 0;
    for &bi in bin_idx {
        total_bits += width_lut[bi as usize] as u64;
    }
    let total_bytes = usize::try_from(total_bits.div_ceil(8))
        .vortex_expect("packed length fits in usize on supported platforms");
    // Reserve at least 8 extra bytes of tail slack so the final flush is
    // always a full `u64` write without a bounds check.
    let alloc_bytes = total_bytes + 8;
    let mut bytes = BufferMut::<u8>::zeroed(alloc_bytes);
    let dst = bytes.as_mut_slice();

    let mut prefix = BufferMut::<u64>::with_capacity(n_batches + 1);
    prefix.push(0);

    let mut acc: u64 = 0;
    let mut acc_bits: u32 = 0;
    let mut byte_pos: usize = 0;
    let mut bit_offset: u64 = 0;

    // Walk batches; flush a prefix entry at every BATCH boundary.
    let mut i = 0;
    while i < n {
        let batch_end = (i + BATCH).min(n);
        // Hot inner pack loop. The compiler is free to keep acc/acc_bits/
        // byte_pos/bit_offset in registers; the only memory traffic per
        // element is one u8 load (bin_idx), one u8 load (width_lut), one
        // u64 load (values), and an occasional 8-byte store.
        for j in i..batch_end {
            let bi = bin_idx[j];
            let w = width_lut[bi as usize] as u32;
            let v = values[j];
            // Overflow check: if w < 64 and v >= 1<<w, the encoding is
            // ambiguous. Validate inline so we do not need a second pass.
            if w < 64 && v >= (1u64 << w) {
                vortex_bail!(
                    "value {v} at index {j} overflows bin {} width {w} (max {})",
                    bi as usize,
                    if w == 0 { 0u64 } else { (1u64 << w) - 1 },
                );
            }
            bit_offset += w as u64;
            if w == 0 {
                continue;
            }
            // Mask the high bits to keep the OR clean; `v` was just
            // validated to fit in `w` bits.
            let masked = if w >= 64 { v } else { v & ((1u64 << w) - 1) };
            acc |= masked << acc_bits;
            let new_bits = acc_bits + w;
            if new_bits >= 64 {
                let end = byte_pos + 8;
                dst[byte_pos..end].copy_from_slice(&acc.to_le_bytes());
                byte_pos = end;
                let leftover = new_bits - 64;
                acc = if leftover == 0 {
                    0
                } else {
                    masked >> (w - leftover)
                };
                acc_bits = leftover;
            } else {
                acc_bits = new_bits;
            }
        }
        prefix.push(bit_offset);
        i = batch_end;
    }
    // Flush remaining bits. The slack of 8 bytes guarantees the write is in
    // bounds; only the first `total_bytes - byte_pos` bytes are logical.
    if acc_bits > 0 || byte_pos < total_bytes {
        let end = byte_pos + 8;
        dst[byte_pos..end].copy_from_slice(&acc.to_le_bytes());
    }

    bytes.truncate(total_bytes);
    Ok((
        bytes.freeze().into_byte_buffer(),
        prefix.freeze().into_byte_buffer(),
    ))
}

/// Read `w` bits from `src` starting at bit `bit_offset`, as an unsigned u64.
#[inline]
#[expect(clippy::cast_possible_truncation)]
fn read_bits(src: &[u8], bit_offset: u64, w: u64) -> u64 {
    debug_assert!(w <= 64);
    if w == 0 {
        return 0;
    }
    let mut remaining = w;
    let mut byte_idx = (bit_offset >> 3) as usize;
    let mut bit_in_byte = (bit_offset & 7) as u32;
    let mut out: u64 = 0;
    let mut shift: u32 = 0;
    while remaining > 0 {
        let space = 8 - bit_in_byte as u64;
        let take = remaining.min(space);
        let mask = if take == 8 {
            0xFFu64
        } else {
            (1u64 << take) - 1
        };
        let chunk = ((src[byte_idx] >> bit_in_byte) as u64) & mask;
        out |= chunk << shift;
        // `take <= 8`, so `take as u32` is exact.
        shift += take as u32;
        remaining -= take;
        bit_in_byte = 0;
        byte_idx += 1;
    }
    out
}

/// Full-array decode: walk the bin_idx + widths, reading each value in
/// sequence from the packed bit buffer.
///
/// Hot path: for every element we do a single unaligned `u128` read from the
/// packed buffer at byte offset `bit_offset / 8`, shift down by
/// `bit_offset & 7`, and mask to `w` bits. The `u128` covers any width
/// `<= 64` bits at any sub-byte alignment, so a single load+shift+mask
/// suffices.
///
/// The output buffer is pre-sized to `n` so the loop writes through a
/// `&mut [u64]` (one store per element) rather than `BufferMut::push`
/// (which performs a per-element capacity check and length bump).
#[expect(clippy::cast_possible_truncation)]
#[expect(clippy::many_single_char_names)]
fn decode_buffer(bin_idx: &[u8], widths: &[u8], packed: &[u8], n: usize) -> Buffer<u64> {
    // 256-entry width lookup so the hot path is a single u8 load.
    let mut width_lut = [0u8; 256];
    for (i, &w) in widths.iter().enumerate() {
        width_lut[i] = w;
    }

    let mut out = BufferMut::<u64>::with_capacity(n);
    if n == 0 {
        return out.freeze();
    }
    // Write directly into the spare capacity (uninit memory) so we avoid
    // `push`'s per-element bookkeeping. The slice we iterate is
    // length-`n`, so the bounds on the zip iterator below are statically
    // matched and Rust elides per-iteration bounds checks.
    let uninit = &mut out.spare_capacity_mut()[..n];
    let packed_len = packed.len();
    let packed_ptr = packed.as_ptr();

    let mut bit_offset: u64 = 0;
    for (dst, &bi) in uninit.iter_mut().zip(bin_idx) {
        let w = width_lut[bi as usize] as u32;
        let value = if w == 0 {
            0
        } else {
            let byte_idx = (bit_offset >> 3) as usize;
            let bit_in_byte = (bit_offset & 7) as u32;
            if byte_idx + 16 <= packed_len {
                // SAFETY: byte_idx + 16 <= packed_len, so we may safely read
                // 16 bytes starting at packed_ptr.add(byte_idx). u128
                // alignment is 1 (byte-wise from_le_bytes).
                let lo16 = unsafe {
                    let p = packed_ptr.add(byte_idx) as *const [u8; 16];
                    u128::from_le_bytes(*p)
                };
                let shifted = lo16 >> bit_in_byte;
                let mask = if w == 64 { u64::MAX } else { (1u64 << w) - 1 };
                (shifted as u64) & mask
            } else {
                read_bits(packed, bit_offset, w as u64)
            }
        };
        dst.write(value);
        bit_offset += w as u64;
    }
    // SAFETY: the zip above wrote every slot in 0..n; commit the length.
    unsafe { out.set_len(n) };
    out.freeze()
}

impl OperationsVTable<VarWidthBitPacked> for VarWidthBitPacked {
    fn scalar_at(
        array: ArrayView<'_, VarWidthBitPacked>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let widths = array.data().widths.clone();
        let packed = array.data().packed.clone();
        let batch = index / BATCH;
        let batch_start_bits = array.data().batch_bit_offset(batch);
        // Read bin_idx for the range [batch * BATCH, index] from the child
        // and accumulate widths to compute the bit offset of `index`. For
        // the same batch we only need at most BATCH bin_idx reads.
        let bin_idx = array
            .bin_idx()
            .clone()
            .execute::<PrimitiveArray>(ctx)?
            .into_buffer::<u8>();
        let bin_idx_slice = bin_idx.as_slice();
        let batch_first = batch * BATCH;
        let mut bit_offset = batch_start_bits;
        for &bi in &bin_idx_slice[batch_first..index] {
            bit_offset += widths[bi as usize] as u64;
        }
        let w = widths[bin_idx_slice[index] as usize] as u64;
        let value = read_bits(packed.as_slice(), bit_offset, w);
        Ok(Scalar::primitive(value, Nullability::NonNullable))
    }
}

impl ValidityChild<VarWidthBitPacked> for VarWidthBitPacked {
    fn validity_child(array: ArrayView<'_, VarWidthBitPacked>) -> ArrayRef {
        array.bin_idx().clone()
    }
}

#[cfg(test)]
#[expect(clippy::cast_possible_truncation)]
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
    use vortex_array::validity::Validity;
    use vortex_error::VortexResult;

    use super::*;

    fn bin_idx_array(values: Vec<u8>) -> ArrayRef {
        PrimitiveArray::new(Buffer::from(values), Validity::NonNullable).into_array()
    }

    fn round_trip(
        widths: Vec<u8>,
        bins: Vec<u8>,
        values: Vec<u64>,
    ) -> VortexResult<VarWidthBitPackedArray> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let bin_idx = bin_idx_array(bins);
        let encoded = VarWidthBitPacked::encode(bin_idx, widths, &values, &mut ctx)?;
        let decoded = encoded
            .clone()
            .into_array()
            .execute::<PrimitiveArray>(&mut ctx)?
            .into_buffer::<u64>();
        assert_eq!(decoded.as_slice(), values.as_slice());
        Ok(encoded)
    }

    #[test]
    fn uniform_width_round_trip() -> VortexResult<()> {
        // All bins share width 8; the kernel collapses to a single fixed-width
        // bitpack.
        let n = 256;
        let widths = vec![8u8, 8, 8, 8];
        let bins: Vec<u8> = (0..n).map(|i| (i % 4) as u8).collect();
        let values: Vec<u64> = (0..n).map(|i| (i as u64 * 17) & 0xFF).collect();
        round_trip(widths, bins, values)?;
        Ok(())
    }

    #[test]
    fn mixed_width_round_trip() -> VortexResult<()> {
        let widths = vec![1u8, 4, 8, 16, 32];
        let mut rng = SmallRng::seed_from_u64(0xBEEF);
        let n = 4_096usize;
        let bins: Vec<u8> = (0..n)
            .map(|_| rng.random_range(0u8..widths.len() as u8))
            .collect();
        let values: Vec<u64> = bins
            .iter()
            .map(|&b| {
                let w = widths[b as usize];
                if w == 0 {
                    0
                } else if w == 64 {
                    rng.random::<u64>()
                } else {
                    rng.random::<u64>() & ((1u64 << w) - 1)
                }
            })
            .collect();
        round_trip(widths, bins, values)?;
        Ok(())
    }

    #[rstest]
    #[case::small(64)]
    #[case::medium(1_000)]
    #[case::large(10_000)]
    fn scalar_at_matches_canonical(#[case] n: usize) -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let widths = vec![3u8, 7, 11, 19];
        let mut rng = SmallRng::seed_from_u64(0xC0FFEE);
        let bins: Vec<u8> = (0..n)
            .map(|_| rng.random_range(0u8..widths.len() as u8))
            .collect();
        let values: Vec<u64> = bins
            .iter()
            .map(|&b| {
                let w = widths[b as usize];
                rng.random::<u64>() & ((1u64 << w) - 1)
            })
            .collect();
        let bin_idx = bin_idx_array(bins);
        let encoded = VarWidthBitPacked::encode(bin_idx, widths, &values, &mut ctx)?;
        let arr = encoded.into_array();
        let decoded = arr
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)?
            .into_buffer::<u64>();
        // Spot-check at indices covering several batches.
        let mut idx_rng = SmallRng::seed_from_u64(0xFEED);
        for _ in 0..64 {
            let i = idx_rng.random_range(0..n);
            let s = arr.execute_scalar(i, &mut ctx)?;
            assert_eq!(s, Scalar::from(decoded.as_slice()[i]));
        }
        Ok(())
    }

    #[test]
    fn empty_round_trip() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let widths = vec![8u8];
        let bin_idx = bin_idx_array(vec![]);
        let encoded = VarWidthBitPacked::encode(bin_idx, widths, &[], &mut ctx)?;
        assert_eq!(encoded.len(), 0);
        let decoded = encoded.into_array().execute::<PrimitiveArray>(&mut ctx)?;
        assert_eq!(decoded.len(), 0);
        Ok(())
    }

    #[test]
    fn singleton_round_trip() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let widths = vec![8u8];
        let bin_idx = bin_idx_array(vec![0]);
        let encoded = VarWidthBitPacked::encode(bin_idx, widths, &[0xA5u64], &mut ctx)?;
        assert_eq!(encoded.len(), 1);
        let s = encoded.clone().into_array().execute_scalar(0, &mut ctx)?;
        assert_eq!(s, Scalar::from(0xA5u64));
        let decoded = encoded
            .into_array()
            .execute::<PrimitiveArray>(&mut ctx)?
            .into_buffer::<u64>();
        assert_eq!(decoded.as_slice(), &[0xA5u64]);
        Ok(())
    }

    #[test]
    fn rejects_overflow_values() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        // Width 4 allows values < 16. 16 should be rejected.
        let widths = vec![4u8];
        let bin_idx = bin_idx_array(vec![0, 0]);
        let err = VarWidthBitPacked::encode(bin_idx, widths, &[1, 16], &mut ctx);
        assert!(err.is_err(), "expected overflow error, got {err:?}");
        Ok(())
    }

    #[test]
    fn slice_round_trip() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let widths = vec![4u8, 8, 12];
        let n = 200usize;
        let bins: Vec<u8> = (0..n).map(|i| (i % 3) as u8).collect();
        let values: Vec<u64> = (0..n)
            .map(|i| (i as u64) & ((1u64 << widths[i % 3]) - 1))
            .collect();
        let bin_idx = bin_idx_array(bins);
        let encoded = VarWidthBitPacked::encode(bin_idx, widths, &values, &mut ctx)?;
        let sliced = encoded.into_array().slice(50..150)?;
        let expected = PrimitiveArray::new(
            Buffer::from(values[50..150].to_vec()),
            Validity::NonNullable,
        );
        assert_arrays_eq!(sliced, expected);
        Ok(())
    }
}
