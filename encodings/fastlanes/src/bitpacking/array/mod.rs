// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;
use std::mem::MaybeUninit;

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
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_array::patches::PatchSlotIndices;
use vortex_array::patches::Patches;
use vortex_array::patches::PatchesData;
use vortex_array::validity::Validity;
use vortex_array::vtable::child_to_validity;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;

pub mod bitpack_compress;
pub mod bitpack_decompress;
pub mod unpack_iter;

#[cfg(test)]
mod tests;

use crate::BitPackedArray;
use crate::FL_CHUNK_SIZE;
use crate::bitpack_compress::bitpack_encode;
use crate::bitpack_compress::bitpack_encode_blocked;
use crate::unpack_iter::BitPacked as BitPackedIter;
use crate::unpack_iter::BitUnpackedChunks;

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
    /// Byte boundaries of the packed blocks as non-nullable unsigned integers, including one
    /// trailing boundary, when the blocks have no global bit width.
    /// Block `i` is packed at `(block_offsets[i + 1] - block_offsets[i]) / 128` bits.
    #[slot(4)]
    pub block_offsets: Option<ArrayRef>,
}

pub(crate) const PATCH_SLOTS: PatchSlotIndices = PatchSlotIndices {
    indices: BitPackedSlots::PATCH_INDICES,
    values: BitPackedSlots::PATCH_VALUES,
    chunk_offsets: BitPackedSlots::PATCH_CHUNK_OFFSETS,
};

/// Check that `offsets` holds `num_blocks + 1` non-nullable unsigned boundaries.
///
/// Debug builds also assert that host-resident boundaries span `packed_len` bytes, each block a
/// whole number of 128-byte rows with a bit width supported by `ptype`. Release builds don't check
/// the boundary values, so decoders must bounds-check them.
pub(crate) fn validate_block_offsets(
    offsets: &ArrayRef,
    ptype: PType,
    num_blocks: usize,
    packed_len: usize,
) -> VortexResult<()> {
    vortex_ensure!(
        offsets.dtype().is_unsigned_int() && !offsets.dtype().is_nullable(),
        "Expected non-nullable unsigned integer block offsets, got {}",
        offsets.dtype()
    );
    vortex_ensure!(
        offsets.len() == num_blocks + 1,
        "Expected {} block boundaries, got {}",
        num_blocks + 1,
        offsets.len()
    );
    if cfg!(debug_assertions)
        && let Some(primitive) = offsets.as_opt::<Primitive>()
        && primitive.buffer_handle().is_on_host()
    {
        let max_bit_width = ptype.bit_width() as u64;
        match_each_unsigned_integer_ptype!(primitive.ptype(), |T| {
            validate_primitive_offsets(primitive.as_slice::<T>(), max_bit_width, packed_len)
                .vortex_expect("invalid BitPacked block offsets")
        })
    }
    Ok(())
}

/// Check that each block between `boundaries` is a whole number of 128-byte rows of at most
/// `max_bit_width` bits, and that the boundaries span `packed_len` bytes.
fn validate_primitive_offsets<T: Copy + Display>(
    boundaries: &[T],
    max_bit_width: u64,
    packed_len: usize,
) -> VortexResult<()>
where
    u64: From<T>,
{
    for pair in boundaries.windows(2) {
        let size = u64::from(pair[1]).checked_sub(u64::from(pair[0]));
        vortex_ensure!(
            size.is_some_and(|size| size % 128 == 0 && size / 128 <= max_bit_width),
            "Block boundaries {} and {} do not hold a supported bit width (at most {max_bit_width} bits)",
            pair[0],
            pair[1]
        );
    }
    let span = match boundaries {
        [first, .., last] => u64::from(*last) - u64::from(*first),
        _ => 0,
    };
    vortex_ensure!(
        span == packed_len as u64,
        "Block offsets span {span} bytes, but the packed buffer has {packed_len}"
    );
    Ok(())
}

/// How the blocks of a [`BitPackedArray`] are packed, borrowing block offsets if present.
#[derive(Clone, Copy, Debug)]
pub enum BitWidthsView<'a> {
    /// Every block is packed at this bit width.
    Global(u8),
    /// Byte boundaries of the packed blocks, from which each block's bit width is derived.
    Blocked(&'a ArrayRef),
}

impl BitWidthsView<'_> {
    /// Returns `true` if every block is packed at one bit width.
    #[inline]
    pub fn is_global(&self) -> bool {
        matches!(self, Self::Global(_))
    }
}

/// How the blocks of a [`BitPackedArray`] are packed, owning block offsets if present.
///
/// This is the owned form of [`BitWidthsView`], held by [`BitPackedDataParts`].
#[derive(Clone, Debug)]
pub enum BitWidths {
    /// Every block is packed at this bit width.
    Global(u8),
    /// Byte boundaries of the packed blocks, from which each block's bit width is derived.
    Blocked(ArrayRef),
}

impl From<BitWidthsView<'_>> for BitWidths {
    fn from(view: BitWidthsView<'_>) -> Self {
        match view {
            BitWidthsView::Global(bit_width) => Self::Global(bit_width),
            BitWidthsView::Blocked(block_offsets) => Self::Blocked(block_offsets.clone()),
        }
    }
}

pub struct BitPackedDataParts {
    pub offset: u16,
    pub bit_widths: BitWidths,
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
    /// The bit width shared by every block, or `None` when the block offsets child holds each
    /// block's boundaries.
    pub(super) global_bit_width: Option<u8>,
    pub(super) packed: BufferHandle,
    /// Patch metadata for reconstructing Patches from slots.
    pub(super) patches_data: Option<PatchesData>,
}

impl Display for BitPackedData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self.global_bit_width {
            Some(bit_width) => write!(f, "bit_width: {}, offset: {}", bit_width, self.offset),
            None => write!(f, "offset: {}", self.offset),
        }
    }
}

impl BitPackedData {
    /// Create a new bitpacked array using a buffer of packed data.
    ///
    /// The packed data should be interpreted as a sequence of values with size `bit_width`.
    ///
    /// # Errors
    ///
    /// This method returns errors if any of the metadata is inconsistent, for example the packed
    /// buffer provided does not have the right size according to the supplied length and target
    /// PType.
    ///
    /// # Safety
    ///
    /// For signed arrays, it is the caller's responsibility to ensure that there are no values
    /// that can be interpreted once unpacked to the provided PType.
    ///
    /// This invariant is upheld by the compressor, but callers must ensure this if they wish to
    /// construct a new `BitPackedArray` from parts.
    ///
    /// See also the [`encode`][Self::encode] method on this type for a safe path to create a new
    /// bit-packed array.
    /// A safe constructor for a `BitPackedArray` from its components:
    ///
    /// * `packed` is ByteBuffer holding the compressed data that was packed with FastLanes
    ///   bit-packing to a `bit_width` bits per value. `length` is the length of the original
    ///   vector. Note that the packed is padded with zeros to the next multiple of 1024 elements
    ///   if `length` is not divisible by 1024.
    /// * `ptype` of the original data
    /// * `validity` to track any nulls
    /// * `patches` optionally provided for values that did not pack
    ///
    /// Any failure in validation will result in an error.
    ///
    /// # Validation
    ///
    /// * The `ptype` must be an integer
    /// * `validity` must have `length` len
    /// * Any patches must have any `array_len` equal to `length`
    /// * The `packed` buffer must be exactly sized to hold `length` values of `bit_width` rounded
    ///   up to the next multiple of 1024.
    ///
    /// Any violation of these preconditions will result in an error.
    pub fn try_new(
        packed: BufferHandle,
        patches: Option<Patches>,
        bit_width: u8,
        offset: u16,
    ) -> VortexResult<Self> {
        vortex_ensure!(bit_width <= 64, "Unsupported bit width {bit_width}");
        vortex_ensure!(
            offset < 1024,
            "Offset must be less than the full block i.e., 1024, got {offset}"
        );

        Ok(Self {
            offset,
            global_bit_width: Some(bit_width),
            packed,
            patches_data: patches.as_ref().map(PatchesData::from_patches),
        })
    }

    /// Create the packed payload for blocks whose bit widths come from a block offsets child.
    pub(crate) fn try_new_blocked(
        packed: BufferHandle,
        patches: Option<Patches>,
        offset: u16,
    ) -> VortexResult<Self> {
        vortex_ensure!(
            offset < 1024,
            "Offset must be less than the full block i.e., 1024, got {offset}"
        );

        Ok(Self {
            offset,
            global_bit_width: None,
            packed,
            patches_data: patches.as_ref().map(PatchesData::from_patches),
        })
    }

    pub(crate) fn validate(
        packed: &BufferHandle,
        ptype: PType,
        validity: &Validity,
        patches: Option<&Patches>,
        bit_width: Option<u8>,
        length: usize,
        offset: u16,
    ) -> VortexResult<()> {
        vortex_ensure!(ptype.is_int(), MismatchedTypes: "integer", ptype);

        if let Some(validity_len) = validity.maybe_len() {
            vortex_ensure_eq!(
                validity_len,
                length,
                "BitPackedArray validity length must match array length"
            );
        }

        // Validate patches
        if let Some(patches) = patches {
            Self::validate_patches(patches, ptype, length)?;
        }

        // Validate packed buffer. Block offsets are only checked against it in debug builds.
        if let Some(bit_width) = bit_width {
            vortex_ensure!(
                usize::from(bit_width) <= ptype.bit_width(),
                "Unsupported bit width {bit_width} for {ptype}"
            );
            let expected_packed_len =
                (length + offset as usize).div_ceil(1024) * (128 * bit_width as usize);
            vortex_ensure_eq!(packed.len(), expected_packed_len);
        }

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

        vortex_ensure_eq!(
            patches.array_len(),
            len,
            "BitPackedArray patches length mismatch"
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

    /// Accessor for bit unpacked chunks
    pub fn unpacked_chunks<'a, T: BitPackedIter>(
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

    /// Bit-pack an array of primitive integers, packing each 1024-value block at its width in
    /// `bit_widths`.
    ///
    /// Values wider than their block's width become patches.
    ///
    /// # Errors
    ///
    /// If the provided array is not a primitive integer array or contains negative values, or if
    /// `bit_widths` does not hold one width per block of at most the array's bit width, an error
    /// will be returned. Nonempty arrays must have at least one block narrower than the array's
    /// native bit width.
    pub fn encode_blocked(
        array: &ArrayRef,
        bit_widths: &[u8],
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<BitPackedArray> {
        let parray: PrimitiveArray = array
            .clone()
            .try_downcast::<Primitive>()
            .map_err(|a| vortex_err!(InvalidArgument: "Bitpacking can only encode primitive arrays, got {}", a.encoding_id()))?;
        bitpack_encode_blocked(&parray, bit_widths, None, ctx)
    }
}

pub trait BitPackedArrayExt: BitPackedArraySlotsExt {
    #[inline]
    fn packed(&self) -> &BufferHandle {
        BitPackedData::packed(self)
    }

    /// How the blocks are packed: at one global bit width, or at the widths implied by the block
    /// offsets.
    #[inline]
    fn bit_widths(&self) -> BitWidthsView<'_> {
        match (self.global_bit_width, self.block_offsets()) {
            (Some(bit_width), None) => BitWidthsView::Global(bit_width),
            (None, Some(block_offsets)) => BitWidthsView::Blocked(block_offsets),
            _ => vortex_panic!(
                "BitPacked must have exactly one of a global bit width and block offsets"
            ),
        }
    }

    #[inline]
    fn offset(&self) -> u16 {
        BitPackedData::offset(self)
    }

    #[inline]
    fn patches(&self) -> Option<Patches> {
        PatchesData::patches_from_slots(
            self.patches_data.as_ref(),
            self.len(),
            self.slots(),
            PATCH_SLOTS,
        )
    }

    #[inline]
    fn validity(&self) -> Validity {
        child_to_validity(self.validity_child(), self.dtype().nullability())
    }

    #[inline]
    fn packed_slice<T: NativePType + BitPacking>(&self) -> &[T] {
        BitPackedData::packed_slice::<T>(self)
    }

    #[inline]
    fn unpacked_chunks<'a, T: BitPackedIter>(
        &'a self,
        scratch: &'a mut [MaybeUninit<T>; FL_CHUNK_SIZE],
    ) -> VortexResult<BitUnpackedChunks<'a, T>> {
        BitPackedData::unpacked_chunks::<T>(self, self.dtype(), self.len(), scratch)
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
    use vortex_session::VortexSession;

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
}
