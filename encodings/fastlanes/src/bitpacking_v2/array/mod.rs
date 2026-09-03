// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;
use std::mem::MaybeUninit;
use std::ops::Range;

use fastlanes::BitPacking;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::TypedArrayRef;
use vortex_array::array_slots;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::PType;
use vortex_array::patches::PatchSlotIndices;
use vortex_array::patches::Patches;
use vortex_array::patches::PatchesData;
use vortex_array::validity::Validity;
use vortex_array::vtable::child_to_validity;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;

pub mod bitpack_compress;
pub mod bitpack_decompress;
pub mod unpack_iter;

use crate::BitPackedV2Array;
use crate::FL_CHUNK_SIZE;
use crate::bitpacking_v2::bitpack_compress::bitpack_encode;
use crate::bitpacking_v2::unpack_iter::BitPackedV2 as BitPackedV2Iter;
use crate::bitpacking_v2::unpack_iter::BitUnpackedChunks;

/// Bytes occupied by one packed FastLanes chunk of `bit_width`-bit values.
#[inline]
pub const fn chunk_packed_bytes(bit_width: u8) -> usize {
    (FL_CHUNK_SIZE / 8) * bit_width as usize
}

/// The bit width of every FastLanes chunk of a bit-packed array.
///
/// Each 1024-element chunk is packed independently at its own width, so an array always carries
/// one width per chunk. The byte offset of every chunk's packed block is cached so chunk lookup
/// stays O(1).
#[derive(Clone, Debug)]
pub struct ChunkWidths {
    widths: Buffer<u8>,
    /// Byte offset of every chunk's packed block, with a trailing entry holding the total.
    byte_offsets: Buffer<u64>,
    max_width: u8,
}

impl ChunkWidths {
    /// Build from one width per chunk.
    pub fn new(widths: Buffer<u8>) -> Self {
        let mut byte_offsets = BufferMut::<u64>::with_capacity(widths.len() + 1);
        let mut total = 0u64;
        let mut max_width = 0u8;
        byte_offsets.push(0);
        for &w in widths.iter() {
            total += chunk_packed_bytes(w) as u64;
            max_width = max_width.max(w);
            byte_offsets.push(total);
        }
        Self {
            widths,
            byte_offsets: byte_offsets.freeze(),
            max_width,
        }
    }

    /// `num_chunks` chunks all packed at `bit_width`.
    pub fn uniform(bit_width: u8, num_chunks: usize) -> Self {
        Self::new(Buffer::from_iter(std::iter::repeat_n(
            bit_width, num_chunks,
        )))
    }

    /// Number of chunks.
    #[inline]
    pub fn len(&self) -> usize {
        self.widths.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.widths.is_empty()
    }

    /// The bit width of `chunk`.
    #[inline]
    pub fn width(&self, chunk: usize) -> u8 {
        self.widths[chunk]
    }

    /// One width per chunk.
    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        self.widths.as_slice()
    }

    /// The widest chunk width.
    #[inline]
    pub fn max_width(&self) -> u8 {
        self.max_width
    }

    /// The single width shared by every chunk, if they all agree.
    pub fn uniform_width(&self) -> Option<u8> {
        let first = *self.widths.first()?;
        self.widths.iter().all(|&w| w == first).then_some(first)
    }

    /// Byte offset of `chunk`'s packed block. Passing the chunk count yields the total size.
    #[inline]
    pub fn byte_offset(&self, chunk: usize) -> usize {
        self.byte_offsets[chunk] as usize
    }

    /// Total packed bytes.
    #[inline]
    pub fn packed_bytes(&self) -> usize {
        self.byte_offset(self.len())
    }

    /// Restrict to the given range of chunks.
    pub fn slice(&self, chunks: Range<usize>) -> Self {
        Self::new(self.widths.slice(chunks))
    }
}

impl PartialEq for ChunkWidths {
    fn eq(&self, other: &Self) -> bool {
        self.widths.as_slice() == other.widths.as_slice()
    }
}

impl Eq for ChunkWidths {}

impl Hash for ChunkWidths {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.widths.as_slice().hash(state);
    }
}

impl Display for ChunkWidths {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self.uniform_width() {
            Some(w) => write!(f, "bit_widths: {} chunks x {w}", self.len()),
            None => write!(
                f,
                "bit_widths: {} chunks, max {}",
                self.len(),
                self.max_width
            ),
        }
    }
}

#[array_slots(crate::BitPackedV2)]
pub struct BitPackedV2Slots {
    /// The indices of exception values that don't fit in the bit-packed representation.
    #[slot(0)]
    pub patch_indices: Option<ArrayRef>,
    /// The exception values that don't fit in the bit-packed representation.
    #[slot(1)]
    pub patch_values: Option<ArrayRef>,
    /// Chunk offsets for the patch indices/values.
    #[slot(2)]
    pub patch_chunk_offsets: Option<ArrayRef>,
    /// The validity bitmap indicating which elements are non-null.
    #[slot(3)]
    pub validity_child: Option<ArrayRef>,
}

pub(crate) const PATCH_SLOTS: PatchSlotIndices = PatchSlotIndices {
    indices: BitPackedV2Slots::PATCH_INDICES,
    values: BitPackedV2Slots::PATCH_VALUES,
    chunk_offsets: BitPackedV2Slots::PATCH_CHUNK_OFFSETS,
};

pub struct BitPackedV2DataParts {
    pub offset: u16,
    pub widths: ChunkWidths,
    pub len: usize,
    pub packed: BufferHandle,
    pub patches: Option<Patches>,
    pub validity: Validity,
}

#[derive(Clone, Debug)]
pub struct BitPackedV2Data {
    /// The offset within the first block (created with a slice).
    /// 0 <= offset < 1024
    pub(super) offset: u16,
    pub(super) widths: ChunkWidths,
    pub(super) packed: BufferHandle,
    /// Patch metadata for reconstructing Patches from slots.
    pub(super) patches_data: Option<PatchesData>,
}

impl Display for BitPackedV2Data {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}, offset: {}", self.widths, self.offset)
    }
}

impl BitPackedV2Data {
    /// Create a new bitpacked array using a buffer of packed data.
    ///
    /// The packed data holds one FastLanes block per 1024-element chunk, each packed at that
    /// chunk's width from `widths` and concatenated in chunk order. The buffer is padded with
    /// zeros to the next multiple of 1024 elements if the length is not divisible by 1024.
    ///
    /// # Safety
    ///
    /// For signed arrays, it is the caller's responsibility to ensure that there are no values
    /// that can be interpreted as negative once unpacked to the provided PType.
    ///
    /// This invariant is upheld by the compressor, but callers must ensure this if they wish to
    /// construct a new `BitPackedV2Array` from parts.
    ///
    /// See also the [`encode`][Self::encode] method on this type for a safe path to create a new
    /// bit-packed array.
    ///
    /// # Validation
    ///
    /// Performed when the array is built from its parts:
    ///
    /// * The `ptype` must be an integer
    /// * `validity` must have `length` len
    /// * Any patches must have any `array_len` equal to `length`
    /// * `widths` must hold one width per chunk, each no wider than the `ptype`
    /// * The `packed` buffer must be exactly the sum of the chunks' packed sizes.
    ///
    /// Any violation of these preconditions will result in an error.
    pub fn try_new(
        packed: BufferHandle,
        patches: Option<Patches>,
        widths: ChunkWidths,
        offset: u16,
    ) -> VortexResult<Self> {
        vortex_ensure!(
            widths.max_width() <= 64,
            "Unsupported bit width {}",
            widths.max_width()
        );
        vortex_ensure!(
            (offset as usize) < FL_CHUNK_SIZE,
            "Offset must be less than the full block i.e., {FL_CHUNK_SIZE}, got {offset}"
        );

        Ok(Self {
            offset,
            widths,
            packed,
            patches_data: patches.as_ref().map(PatchesData::from_patches),
        })
    }

    pub(crate) fn validate(
        packed: &BufferHandle,
        ptype: PType,
        validity: &Validity,
        patches: Option<&Patches>,
        widths: &ChunkWidths,
        length: usize,
        offset: u16,
    ) -> VortexResult<()> {
        vortex_ensure!(ptype.is_int(), MismatchedTypes: "integer", ptype);
        vortex_ensure!(
            widths.max_width() as usize <= ptype.bit_width(),
            "Unsupported bit width {} for {ptype}",
            widths.max_width()
        );

        if let Some(validity_len) = validity.maybe_len() {
            vortex_ensure!(
                validity_len == length,
                "BitPackedV2Array validity length {validity_len} != array length {length}",
            );
        }

        // Validate patches
        if let Some(patches) = patches {
            Self::validate_patches(patches, ptype, length)?;
        }

        // Validate chunk widths and the packed buffer
        let num_chunks = (length + offset as usize).div_ceil(FL_CHUNK_SIZE);
        vortex_ensure!(
            widths.len() == num_chunks,
            "Expected {num_chunks} chunk widths, got {}",
            widths.len()
        );
        let expected_packed_len = widths.packed_bytes();
        vortex_ensure!(
            packed.len() == expected_packed_len,
            "Expected {} packed bytes, got {}",
            expected_packed_len,
            packed.len()
        );

        Ok(())
    }

    fn validate_patches(patches: &Patches, ptype: PType, len: usize) -> VortexResult<()> {
        // Ensure that array and patches have same ptype
        vortex_ensure!(
            patches.dtype().eq_ignore_nullability(ptype.into()),
            "Patches DType {} does not match BitPackedV2Array dtype {}",
            patches.dtype().as_nonnullable(),
            ptype
        );

        vortex_ensure!(
            patches.array_len() == len,
            "BitPackedV2Array patches length {} != expected {len}",
            patches.array_len(),
        );

        Ok(())
    }

    pub fn ptype(&self, dtype: &DType) -> PType {
        dtype.as_ptype()
    }

    /// Underlying bit packed values as byte array
    #[inline]
    pub fn packed(&self) -> &BufferHandle {
        &self.packed
    }

    /// Access the slice of packed values as an array of `T`
    #[inline]
    pub fn packed_slice<T: NativePType + BitPacking>(&self) -> &[T] {
        let packed_bytes = self.packed().as_host();
        let packed_ptr: *const T = packed_bytes.as_ptr().cast();
        // Return number of elements of type `T` packed in the buffer
        let packed_len = packed_bytes.len() / size_of::<T>();

        // SAFETY: as_slice points to buffer memory that outlives the lifetime of `self`.
        //  Unfortunately Rust cannot understand this, so we reconstruct the slice from raw parts
        //  to get it to reinterpret the lifetime.
        unsafe { std::slice::from_raw_parts(packed_ptr, packed_len) }
    }

    /// The packed FastLanes block of `chunk` as `T` words, along with that chunk's bit width.
    #[inline]
    pub fn packed_chunk<T: NativePType + BitPacking>(&self, chunk: usize) -> (&[T], usize) {
        let bit_width = self.widths.width(chunk);
        let start = self.widths.byte_offset(chunk) / size_of::<T>();
        let len = chunk_packed_bytes(bit_width) / size_of::<T>();
        (
            &self.packed_slice::<T>()[start..][..len],
            bit_width as usize,
        )
    }

    /// Accessor for bit unpacked chunks
    pub fn unpacked_chunks<'a, T: BitPackedV2Iter>(
        &'a self,
        dtype: &DType,
        len: usize,
        scratch: &'a mut [MaybeUninit<T>; FL_CHUNK_SIZE],
    ) -> VortexResult<BitUnpackedChunks<'a, T>> {
        assert_eq!(
            T::PTYPE,
            self.ptype(dtype),
            "Requested type doesn't match the array ptype"
        );
        BitUnpackedChunks::try_new(self, len, scratch)
    }

    /// The bit width of every chunk.
    #[inline]
    pub fn chunk_widths(&self) -> &ChunkWidths {
        &self.widths
    }

    /// The widest bit width used by any chunk.
    #[inline]
    pub fn bit_width(&self) -> u8 {
        self.widths.max_width()
    }

    #[inline]
    pub fn offset(&self) -> u16 {
        self.offset
    }

    /// Bit-pack an array of primitive integers down to the target bit-width using the FastLanes
    /// SIMD-accelerated packing kernels.
    ///
    /// # Errors
    ///
    /// If the provided array is not an integer type, an error will be returned.
    ///
    /// If the provided array contains negative values, an error will be returned.
    ///
    /// If the requested bit-width for packing is larger than the array's native width, an
    /// error will be returned.
    pub fn encode(
        array: &ArrayRef,
        bit_width: u8,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<BitPackedV2Array> {
        let parray: PrimitiveArray = array
            .clone()
            .try_downcast::<Primitive>()
            .map_err(|a| vortex_err!(InvalidArgument: "Bitpacking can only encode primitive arrays, got {}", a.encoding_id()))?;
        bitpack_encode(&parray, bit_width, ctx)
    }

    /// Calculate the maximum value that **can** be contained by this array, given its widest
    /// chunk.
    ///
    /// Note that this value need not actually be present in the array.
    #[inline]
    pub fn max_packed_value(&self) -> usize {
        let bit_width = self.bit_width() as u32;
        if bit_width >= usize::BITS {
            usize::MAX
        } else {
            (1usize << bit_width) - 1
        }
    }
}

pub trait BitPackedV2ArrayExt: BitPackedV2ArraySlotsExt {
    #[inline]
    fn packed(&self) -> &BufferHandle {
        BitPackedV2Data::packed(self)
    }

    #[inline]
    fn chunk_widths(&self) -> &ChunkWidths {
        BitPackedV2Data::chunk_widths(self)
    }

    #[inline]
    fn bit_width(&self) -> u8 {
        BitPackedV2Data::bit_width(self)
    }

    #[inline]
    fn offset(&self) -> u16 {
        BitPackedV2Data::offset(self)
    }

    #[inline]
    fn patches(&self) -> Option<Patches> {
        PatchesData::patches_from_slots(
            self.patches_data.as_ref(),
            self.as_ref().len(),
            self.as_ref().slots(),
            PATCH_SLOTS,
        )
    }

    #[inline]
    fn validity(&self) -> Validity {
        child_to_validity(self.validity_child(), self.as_ref().dtype().nullability())
    }

    #[inline]
    fn packed_slice<T: NativePType + BitPacking>(&self) -> &[T] {
        BitPackedV2Data::packed_slice::<T>(self)
    }

    #[inline]
    fn packed_chunk<T: NativePType + BitPacking>(&self, chunk: usize) -> (&[T], usize) {
        BitPackedV2Data::packed_chunk::<T>(self, chunk)
    }

    #[inline]
    fn unpacked_chunks<'a, T: BitPackedV2Iter>(
        &'a self,
        scratch: &'a mut [MaybeUninit<T>; FL_CHUNK_SIZE],
    ) -> VortexResult<BitUnpackedChunks<'a, T>> {
        BitPackedV2Data::unpacked_chunks::<T>(
            self,
            self.as_ref().dtype(),
            self.as_ref().len(),
            scratch,
        )
    }
}

impl<T: TypedArrayRef<crate::BitPackedV2>> BitPackedV2ArrayExt for T {}

#[cfg(test)]
mod test {
    use std::sync::LazyLock;

    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_buffer::Buffer;
    use vortex_buffer::buffer;
    use vortex_session::VortexSession;

    use super::ChunkWidths;
    use crate::BitPackedV2Data;
    use crate::bitpacking_v2::array::BitPackedV2ArrayExt;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = vortex_array::array_session();
        crate::initialize(&session);
        session
    });

    #[test]
    fn test_encode() {
        let mut ctx = SESSION.create_execution_ctx();
        let values = [
            Some(1u64),
            None,
            Some(1),
            None,
            Some(1),
            None,
            Some(u64::MAX),
        ];
        let uncompressed = PrimitiveArray::from_option_iter(values);
        let packed = BitPackedV2Data::encode(&uncompressed.into_array(), 1, &mut ctx).unwrap();
        let expected = PrimitiveArray::from_option_iter(values);
        let packed_primitive = packed
            .as_array()
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)
            .unwrap();
        assert_arrays_eq!(packed_primitive, expected, &mut ctx);
    }

    #[test]
    fn test_encode_too_wide() {
        let mut ctx = SESSION.create_execution_ctx();
        let values = [Some(1u8), None, Some(1), None, Some(1), None];
        let uncompressed = PrimitiveArray::from_option_iter(values);
        let _packed = BitPackedV2Data::encode(&uncompressed.clone().into_array(), 8, &mut ctx)
            .expect_err("Cannot pack value into the same width");
        let _packed = BitPackedV2Data::encode(&uncompressed.into_array(), 9, &mut ctx)
            .expect_err("Cannot pack value into larger width");
    }

    #[test]
    fn signed_with_patches() {
        let mut ctx = SESSION.create_execution_ctx();
        let values: Buffer<i32> = (0i32..=512).collect();
        let parray = values.clone().into_array();

        let packed_with_patches = BitPackedV2Data::encode(&parray, 9, &mut ctx).unwrap();
        assert!(packed_with_patches.patches().is_some());
        let packed_primitive = packed_with_patches
            .as_array()
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)
            .unwrap();
        assert_arrays_eq!(
            packed_primitive,
            PrimitiveArray::new(values, vortex_array::validity::Validity::NonNullable),
            &mut ctx
        );
    }

    #[test]
    fn chunk_widths_offsets() {
        assert_eq!(ChunkWidths::uniform(3, 3).uniform_width(), Some(3));
        assert_eq!(ChunkWidths::new(Buffer::<u8>::empty()).packed_bytes(), 0);

        let widths = ChunkWidths::new(buffer![3u8, 0, 16]);
        assert_eq!(widths.uniform_width(), None);
        assert_eq!(widths.len(), 3);
        assert_eq!(widths.max_width(), 16);
        assert_eq!(widths.width(1), 0);
        assert_eq!(widths.byte_offset(0), 0);
        assert_eq!(widths.byte_offset(1), 128 * 3);
        assert_eq!(widths.byte_offset(2), 128 * 3);
        assert_eq!(widths.packed_bytes(), 128 * 19);
        assert_eq!(widths.slice(1..3), ChunkWidths::new(buffer![0u8, 16]));
        assert_eq!(widths.slice(0..1).uniform_width(), Some(3));
    }
}
