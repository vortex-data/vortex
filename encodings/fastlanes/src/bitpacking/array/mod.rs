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
use vortex_array::IntoArray;
use vortex_array::TypedArrayRef;
use vortex_array::array_slots;
use vortex_array::arrays::Constant;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::Nullability;
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

use crate::BitPackedArray;
use crate::FL_CHUNK_SIZE;
use crate::bitpacking::bitpack_compress::bitpack_encode;
use crate::bitpacking::unpack_iter::BitPacked as BitPackedIter;
use crate::bitpacking::unpack_iter::BitUnpackedChunks;

/// Bytes occupied by one packed FastLanes chunk of `bit_width`-bit values.
#[inline]
pub const fn chunk_packed_bytes(bit_width: u8) -> usize {
    (FL_CHUNK_SIZE / 8) * bit_width as usize
}

/// Chunk widths and byte offsets used while encoding or executing bit-packed data.
/// Execution borrows the materialized children; only encoding computes prefix sums.
#[derive(Clone, Debug)]
pub struct ChunkWidths {
    widths: Widths,
    byte_offsets: Buffer<u64>,
    max_width: u8,
}

#[derive(Clone, Debug)]
enum Widths {
    Uniform { width: u8, len: usize },
    PerChunk(Buffer<u8>),
}

impl ChunkWidths {
    /// Build an encoding plan, computing byte offsets from one width per chunk.
    pub fn new(widths: Buffer<u8>) -> Self {
        let mut byte_offsets = BufferMut::<u64>::with_capacity(widths.len() + 1);
        let mut total = 0u64;
        byte_offsets.push(0);
        for &width in widths.iter() {
            total += chunk_packed_bytes(width) as u64;
            byte_offsets.push(total);
        }
        Self::from_buffers(Widths::PerChunk(widths), byte_offsets.freeze())
    }

    /// `num_chunks` chunks all packed at `bit_width`.
    pub fn uniform(bit_width: u8, num_chunks: usize) -> Self {
        Self::from_buffers(
            Widths::Uniform {
                width: bit_width,
                len: num_chunks,
            },
            Buffer::from_iter((0..=num_chunks).map(|i| (i * chunk_packed_bytes(bit_width)) as u64)),
        )
    }

    fn from_buffers(widths: Widths, byte_offsets: Buffer<u64>) -> Self {
        let max_width = match &widths {
            Widths::Uniform { width, .. } => *width,
            Widths::PerChunk(widths) => widths.iter().copied().max().unwrap_or(0),
        };
        Self {
            widths,
            byte_offsets,
            max_width,
        }
    }

    /// Number of chunks.
    #[inline]
    pub fn len(&self) -> usize {
        match &self.widths {
            Widths::Uniform { len, .. } => *len,
            Widths::PerChunk(widths) => widths.len(),
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The bit width of `chunk`.
    #[inline]
    pub fn width(&self, chunk: usize) -> u8 {
        match &self.widths {
            Widths::Uniform { width, .. } => *width,
            Widths::PerChunk(widths) => widths[chunk],
        }
    }

    /// The widest chunk width.
    #[inline]
    pub fn max_width(&self) -> u8 {
        self.max_width
    }

    /// The single width shared by every chunk, if they all agree.
    pub fn uniform_width(&self) -> Option<u8> {
        if self.is_empty() {
            return None;
        }
        match &self.widths {
            Widths::Uniform { width, .. } => Some(*width),
            Widths::PerChunk(widths) => {
                let first = widths[0];
                widths.iter().all(|&w| w == first).then_some(first)
            }
        }
    }

    /// Whether every chunk shares one width. An empty array counts as uniform.
    pub fn is_uniform(&self) -> bool {
        self.is_empty() || self.uniform_width().is_some()
    }

    /// Materialize the widths as one byte per chunk.
    pub fn as_buffer(&self) -> Buffer<u8> {
        match &self.widths {
            Widths::Uniform { width, len } => Buffer::from_iter(std::iter::repeat_n(*width, *len)),
            Widths::PerChunk(widths) => widths.clone(),
        }
    }

    /// The offsets child, including the trailing boundary. Slices may start at a nonzero offset.
    pub fn offsets_array(&self) -> ArrayRef {
        self.byte_offsets.clone().into_array()
    }

    /// Byte offset relative to the packed buffer. Passing the chunk count yields the total size.
    #[inline]
    pub fn byte_offset(&self, chunk: usize) -> usize {
        (self.byte_offsets[chunk] - self.byte_offsets[0]) as usize
    }

    /// Total packed bytes.
    #[inline]
    pub fn packed_bytes(&self) -> usize {
        self.byte_offset(self.len())
    }

    /// Restrict to chunks without copying or rebasing the offset buffer.
    pub fn slice(&self, chunks: Range<usize>) -> Self {
        let widths = match &self.widths {
            Widths::Uniform { width, .. } => Widths::Uniform {
                width: *width,
                len: chunks.len(),
            },
            Widths::PerChunk(widths) => Widths::PerChunk(widths.slice(chunks.clone())),
        };
        Self::from_buffers(
            widths,
            self.byte_offsets.slice(chunks.start..chunks.end + 1),
        )
    }
}

impl PartialEq for ChunkWidths {
    fn eq(&self, other: &Self) -> bool {
        self.len() == other.len() && (0..self.len()).all(|i| self.width(i) == other.width(i))
    }
}

impl Eq for ChunkWidths {}

impl Hash for ChunkWidths {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.len().hash(state);
        for i in 0..self.len() {
            self.width(i).hash(state);
        }
    }
}

impl Display for ChunkWidths {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self.uniform_width() {
            Some(w) => write!(f, "bit_width: {w}"),
            None => write!(
                f,
                "bit_widths: {} chunks, max {}",
                self.len(),
                self.max_width
            ),
        }
    }
}

#[array_slots(crate::BitPacked)]
pub struct BitPackedSlots {
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
    /// One non-nullable `u8` width per 1024-element chunk. Uniform widths use a constant array.
    #[slot(4)]
    pub width_table: ArrayRef,
    /// Non-nullable `u64` byte boundaries, with one trailing entry after the last chunk.
    /// The first offset is the origin of the packed buffer and may be nonzero after slicing.
    #[slot(5)]
    pub chunk_offsets: ArrayRef,
}

pub(crate) const PATCH_SLOTS: PatchSlotIndices = PatchSlotIndices {
    indices: BitPackedSlots::PATCH_INDICES,
    values: BitPackedSlots::PATCH_VALUES,
    chunk_offsets: BitPackedSlots::PATCH_CHUNK_OFFSETS,
};

/// The dtype of the width table child: one byte per chunk.
pub(crate) const WIDTH_TABLE_DTYPE: DType = DType::Primitive(PType::U8, Nullability::NonNullable);

pub(crate) const CHUNK_OFFSETS_DTYPE: DType =
    DType::Primitive(PType::U64, Nullability::NonNullable);

impl IntoArray for ChunkWidths {
    fn into_array(self) -> ArrayRef {
        if self.is_uniform() {
            ConstantArray::new(self.max_width, self.len()).into_array()
        } else {
            self.as_buffer().into_array()
        }
    }
}

/// Read materialized children without executing them during reduction.
pub(crate) fn materialized_widths(
    table: &ArrayRef,
    offsets: &ArrayRef,
) -> VortexResult<Option<ChunkWidths>> {
    let widths = if let Some(constant) = table.as_opt::<Constant>() {
        Widths::Uniform {
            width: u8::try_from(constant.scalar())?,
            len: table.len(),
        }
    } else if let Some(primitive) = table
        .as_opt::<Primitive>()
        .filter(|a| a.buffer_handle().is_on_host())
    {
        Widths::PerChunk(primitive.to_buffer::<u8>())
    } else {
        return Ok(None);
    };
    Ok(offsets
        .as_opt::<Primitive>()
        .filter(|a| a.buffer_handle().is_on_host())
        .map(|a| ChunkWidths::from_buffers(widths, a.to_buffer::<u64>())))
}

pub struct BitPackedDataParts {
    pub offset: u16,
    pub widths: ArrayRef,
    pub chunk_offsets: ArrayRef,
    pub len: usize,
    pub packed: BufferHandle,
    pub patches: Option<Patches>,
    pub validity: Validity,
}

#[derive(Clone, Debug)]
pub struct BitPackedData {
    /// The offset within the first block (created with a slice).
    /// 0 <= offset < 1024
    pub(super) offset: u16,
    pub(super) packed: BufferHandle,
    /// Patch metadata for reconstructing Patches from slots.
    pub(super) patches_data: Option<PatchesData>,
}

impl Display for BitPackedData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "offset: {}", self.offset)
    }
}

impl BitPackedData {
    /// Create a new bitpacked array using a buffer of packed data.
    ///
    /// The packed data holds one FastLanes block per 1024-element chunk, each packed at that
    /// chunk's width from the width-table child and concatenated in chunk order. The buffer is padded with
    /// zeros to the next multiple of 1024 elements if the length is not divisible by 1024.
    ///
    /// # Safety
    ///
    /// For signed arrays, it is the caller's responsibility to ensure that there are no values
    /// that can be interpreted as negative once unpacked to the provided PType.
    ///
    /// This invariant is upheld by the compressor, but callers must ensure this if they wish to
    /// construct a new `BitPackedArray` from parts.
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
    /// * The width-table child must hold one non-nullable `u8` per chunk.
    /// * The offsets child must hold `num_chunks + 1` non-nullable `u64` byte boundaries.
    ///
    /// Once the widths are materialized, they must be no wider than `ptype`, and the packed
    /// buffer must be exactly the sum of the chunks' packed sizes. Offset differences must
    /// match the widths. Compressed children are checked at execution time, before unpacking.
    ///
    /// Any violation of these preconditions will result in an error.
    pub fn try_new(
        packed: BufferHandle,
        patches: Option<Patches>,
        offset: u16,
    ) -> VortexResult<Self> {
        vortex_ensure!(
            (offset as usize) < FL_CHUNK_SIZE,
            "Offset must be less than the full block i.e., {FL_CHUNK_SIZE}, got {offset}"
        );

        Ok(Self {
            offset,
            packed,
            patches_data: patches.as_ref().map(PatchesData::from_patches),
        })
    }

    pub(crate) fn validate(
        &self,
        ptype: PType,
        validity: &Validity,
        patches: Option<&Patches>,
        table: &ArrayRef,
        offsets: &ArrayRef,
        length: usize,
    ) -> VortexResult<()> {
        vortex_ensure!(ptype.is_int(), MismatchedTypes: "integer", ptype);
        if let Some(validity_len) = validity.maybe_len() {
            vortex_ensure!(
                validity_len == length,
                "BitPackedArray validity length {validity_len} != array length {length}",
            );
        }

        // Validate patches
        if let Some(patches) = patches {
            Self::validate_patches(patches, ptype, length)?;
        }

        let num_chunks = (length + self.offset as usize).div_ceil(FL_CHUNK_SIZE);
        vortex_ensure!(
            table.dtype() == &WIDTH_TABLE_DTYPE,
            "BitPacked width table must be {WIDTH_TABLE_DTYPE}, got {}",
            table.dtype()
        );
        vortex_ensure!(
            table.len() == num_chunks,
            "Expected {num_chunks} chunk widths, got {}",
            table.len()
        );
        vortex_ensure!(
            offsets.dtype() == &CHUNK_OFFSETS_DTYPE,
            "BitPacked chunk offsets must be {CHUNK_OFFSETS_DTYPE}, got {}",
            offsets.dtype()
        );
        vortex_ensure!(
            offsets.len() == num_chunks + 1,
            "Expected {} chunk offsets, got {}",
            num_chunks + 1,
            offsets.len()
        );
        // Compressed children are checked once materialized, before any unchecked unpacking.
        if let Some(widths) = materialized_widths(table, offsets)? {
            Self::validate_widths(&self.packed, ptype, &widths)?;
        }
        Ok(())
    }

    pub(crate) fn validate_widths(
        packed: &BufferHandle,
        ptype: PType,
        widths: &ChunkWidths,
    ) -> VortexResult<()> {
        vortex_ensure!(
            widths.max_width() as usize <= ptype.bit_width(),
            "Unsupported bit width {} for {ptype}",
            widths.max_width()
        );
        for chunk in 0..widths.len() {
            vortex_ensure!(
                widths.byte_offsets[chunk + 1].checked_sub(widths.byte_offsets[chunk])
                    == Some(chunk_packed_bytes(widths.width(chunk)) as u64),
                "Chunk {chunk} offsets do not match its bit width"
            );
        }
        let expected = widths.byte_offsets[widths.len()] - widths.byte_offsets[0];
        vortex_ensure!(
            packed.len() as u64 == expected,
            "Expected {expected} packed bytes, got {}",
            packed.len()
        );
        Ok(())
    }

    fn validate_patches(patches: &Patches, ptype: PType, len: usize) -> VortexResult<()> {
        // Ensure that array and patches have same ptype
        vortex_ensure!(
            patches.dtype().eq_ignore_nullability(ptype.into()),
            "Patches DType {} does not match BitPackedArray dtype {}",
            patches.dtype().as_nonnullable(),
            ptype
        );

        vortex_ensure!(
            patches.array_len() == len,
            "BitPackedArray patches length {} != expected {len}",
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
    pub(crate) fn packed_chunk<T: NativePType + BitPacking>(
        &self,
        widths: &ChunkWidths,
        chunk: usize,
    ) -> (&[T], usize) {
        let bit_width = widths.width(chunk);
        let start = widths.byte_offset(chunk) / size_of::<T>();
        let len = chunk_packed_bytes(bit_width) / size_of::<T>();
        (
            &self.packed_slice::<T>()[start..][..len],
            bit_width as usize,
        )
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
    ) -> VortexResult<BitPackedArray> {
        let parray: PrimitiveArray = array
            .clone()
            .try_downcast::<Primitive>()
            .map_err(|a| vortex_err!(InvalidArgument: "Bitpacking can only encode primitive arrays, got {}", a.encoding_id()))?;
        bitpack_encode(&parray, bit_width, None, ctx)
    }
}

pub trait BitPackedArrayExt: BitPackedArraySlotsExt {
    #[inline]
    fn packed(&self) -> &BufferHandle {
        BitPackedData::packed(self)
    }

    /// Prepare and validate both children once for a bulk operation, without computing prefix sums.
    fn chunk_widths(&self, ctx: &mut ExecutionCtx) -> VortexResult<ChunkWidths> {
        let widths = match materialized_widths(self.width_table(), self.chunk_offsets())? {
            Some(widths) => widths,
            None => {
                let table = self.width_table();
                let widths = if let Some(constant) = table.as_opt::<Constant>() {
                    Widths::Uniform {
                        width: u8::try_from(constant.scalar())?,
                        len: table.len(),
                    }
                } else {
                    Widths::PerChunk(
                        table
                            .clone()
                            .execute::<PrimitiveArray>(ctx)?
                            .to_buffer::<u8>(),
                    )
                };
                let offsets = self
                    .chunk_offsets()
                    .clone()
                    .execute::<PrimitiveArray>(ctx)?
                    .to_buffer::<u64>();
                ChunkWidths::from_buffers(widths, offsets)
            }
        };
        BitPackedData::validate_widths(self.packed(), self.as_ref().dtype().as_ptype(), &widths)?;
        Ok(widths)
    }

    /// Read and validate widths and offsets only when their children are already materialized.
    fn materialized_chunk_widths(&self) -> VortexResult<Option<ChunkWidths>> {
        let widths = materialized_widths(self.width_table(), self.chunk_offsets())?;
        if let Some(widths) = &widths {
            BitPackedData::validate_widths(
                self.packed(),
                self.as_ref().dtype().as_ptype(),
                widths,
            )?;
        }
        Ok(widths)
    }

    /// Read one byte boundary without executing the entire offsets child.
    fn chunk_byte_offset(&self, boundary: usize, ctx: &mut ExecutionCtx) -> VortexResult<u64> {
        vortex_ensure!(
            boundary < self.chunk_offsets().len(),
            "Chunk boundary out of bounds"
        );
        if let Some(offsets) = self
            .chunk_offsets()
            .as_opt::<Primitive>()
            .filter(|a| a.buffer_handle().is_on_host())
        {
            Ok(offsets.as_slice::<u64>()[boundary])
        } else {
            u64::try_from(&self.chunk_offsets().execute_scalar(boundary, ctx)?)
        }
    }

    /// Locate and validate one chunk using scalar child access, without materializing the tables.
    fn chunk_range(
        &self,
        chunk: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<(Range<usize>, u8)> {
        vortex_ensure!(
            chunk < self.width_table().len(),
            "Chunk index out of bounds"
        );
        let width = if let Some(table) = self
            .width_table()
            .as_opt::<Primitive>()
            .filter(|a| a.buffer_handle().is_on_host())
        {
            table.as_slice::<u8>()[chunk]
        } else if let Some(table) = self.width_table().as_opt::<Constant>() {
            u8::try_from(table.scalar())?
        } else {
            u8::try_from(&self.width_table().execute_scalar(chunk, ctx)?)?
        };
        vortex_ensure!(
            width as usize <= self.as_ref().dtype().as_ptype().bit_width(),
            "Unsupported bit width {width}"
        );
        let base = self.chunk_byte_offset(0, ctx)?;
        let start = if chunk == 0 {
            base
        } else {
            self.chunk_byte_offset(chunk, ctx)?
        };
        let end = self.chunk_byte_offset(chunk + 1, ctx)?;
        vortex_ensure!(
            end.checked_sub(start) == Some(chunk_packed_bytes(width) as u64),
            "Chunk {chunk} offsets do not match its bit width"
        );
        let start = start
            .checked_sub(base)
            .ok_or_else(|| vortex_err!("Chunk offset precedes buffer origin"))?;
        let end = end
            .checked_sub(base)
            .ok_or_else(|| vortex_err!("Chunk offset precedes buffer origin"))?;
        vortex_ensure!(
            start % (FL_CHUNK_SIZE / 8) as u64 == 0 && end <= self.packed().len() as u64,
            "Chunk offsets are unaligned or exceed the packed buffer"
        );
        Ok((usize::try_from(start)?..usize::try_from(end)?, width))
    }

    #[inline]
    fn offset(&self) -> u16 {
        BitPackedData::offset(self)
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
        BitPackedData::packed_slice::<T>(self)
    }

    /// Iterate packed chunks using widths materialized for this operation.
    fn unpacked_chunks<'a, T: BitPackedIter>(
        &'a self,
        widths: &'a ChunkWidths,
        scratch: &'a mut [MaybeUninit<T>; FL_CHUNK_SIZE],
    ) -> VortexResult<BitUnpackedChunks<'a, T>> {
        vortex_ensure!(
            T::PTYPE == self.as_ref().dtype().as_ptype(),
            "Requested unpack type does not match array dtype"
        );
        BitUnpackedChunks::try_new(self, self.as_ref().len(), widths, scratch)
    }
}

impl<T: TypedArrayRef<crate::BitPacked>> BitPackedArrayExt for T {}

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
    use crate::BitPackedData;
    use crate::bitpacking::array::BitPackedArrayExt;

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
        let packed = BitPackedData::encode(&uncompressed.into_array(), 1, &mut ctx).unwrap();
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
        let _packed = BitPackedData::encode(&uncompressed.clone().into_array(), 8, &mut ctx)
            .expect_err("Cannot pack value into the same width");
        let _packed = BitPackedData::encode(&uncompressed.into_array(), 9, &mut ctx)
            .expect_err("Cannot pack value into larger width");
    }

    #[test]
    fn signed_with_patches() {
        let mut ctx = SESSION.create_execution_ctx();
        let values: Buffer<i32> = (0i32..=512).collect();
        let parray = values.clone().into_array();

        let packed_with_patches = BitPackedData::encode(&parray, 9, &mut ctx).unwrap();
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
