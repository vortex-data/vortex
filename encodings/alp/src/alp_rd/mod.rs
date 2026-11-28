// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Vortex array support for ALP-RD ("real doubles"), on top of the `alp` crate's encoder.

#![expect(clippy::cast_possible_truncation)]

use std::mem::MaybeUninit;

pub use ::alp::ALPRDFloat;
pub use ::alp::RDEncoder;
use ::alp::alp_rd_combine_codes_inplace;
use ::alp::bit_width;
pub use array::*;
use fearless_simd::Level;
use fearless_simd::Simd;
use fearless_simd::dispatch;
use fearless_simd_macros::simd;
use num_traits::One;
use num_traits::PrimInt;
use num_traits::Zero;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::Bool;
use vortex_array::arrays::Constant;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::bool::BoolArrayExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::PType;
use vortex_array::match_each_integer_ptype;
use vortex_array::patches::PATCH_CHUNK_SIZE;
use vortex_array::patches::Patches;
use vortex_array::validity::Validity;
use vortex_buffer::BitBufferView;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_compute::lane_kernels::for_each_mask_word;
use vortex_compute::lane_kernels::low_bits_mask;
use vortex_error::VortexError;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_fastlanes::bitpack_compress::bitpack_encode_unchecked;
use vortex_mask::AllOr;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use crate::match_each_alp_float_ptype;
use crate::sample_valid_values;

/// Matches the `alp` crate's ALP-RD dictionary search sample budget and run length.
const ALPRD_MAX_SAMPLE: usize = 4096;
const ALPRD_SAMPLE_BLOCK: usize = 64;

mod array;
mod compute;
mod kernel;
mod ops;
mod rules;
mod slice;

pub(crate) fn initialize(session: &VortexSession) {
    kernel::initialize(session);
    rules::register_parent_rules(session);
}

/// Trains an ALP-RD encoder on only the valid values of `array` and encodes it.
///
/// Null slots are excluded from dictionary training and replaced with code `0` and right part `0`
/// during encoding, so dirty garbage bits in null positions cannot pollute the dictionary or emit
/// wasted patches.
///
/// # Errors
///
/// Returns an error if `array` is not `f32` or `f64`, or if its validity mask cannot be executed.
pub fn alp_rd_encode(
    array: ArrayView<'_, Primitive>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ALPRDArray> {
    let validity = array.validity()?;
    let mask = validity.execute_mask(array.len(), ctx)?;

    match array.ptype() {
        PType::F32 => {
            let encoder = rd_encoder_from_masked::<f32>(array.as_slice::<f32>(), &mask);
            Ok(encode_generic_with_mask::<f32>(
                &encoder, array, validity, &mask,
            ))
        }
        PType::F64 => {
            let encoder = rd_encoder_from_masked::<f64>(array.as_slice::<f64>(), &mask);
            Ok(encode_generic_with_mask::<f64>(
                &encoder, array, validity, &mask,
            ))
        }
        ptype => vortex_bail!("ALP-RD can only encode f32 and f64, got {ptype}"),
    }
}

/// Encoding of Vortex arrays with an ALP-RD [`RDEncoder`].
pub trait RDEncoderExt {
    /// Encode a set of floating point values with ALP-RD.
    ///
    /// Each value is split into a left and right component, which are compressed individually.
    // TODO(joe): make fallible
    fn encode(&self, array: ArrayView<'_, Primitive>) -> ALPRDArray;
}

impl RDEncoderExt for RDEncoder {
    fn encode(&self, array: ArrayView<'_, Primitive>) -> ALPRDArray {
        let validity = array
            .validity()
            .vortex_expect("ALP RD validity should be derivable");
        let mask = validity_mask_without_ctx(&validity, array.len());
        match_each_alp_float_ptype!(array.ptype(), |P| {
            encode_generic_with_mask::<P>(self, array, validity, &mask)
        })
    }
}

/// Builds an [`RDEncoder`] from only the valid elements of `values`, handling empty and all-null
/// inputs without panicking.
fn rd_encoder_from_masked<T: ALPRDFloat + NativePType>(values: &[T], mask: &Mask) -> RDEncoder {
    match mask {
        Mask::AllTrue(0) | Mask::AllFalse(_) => RDEncoder::new(&[T::zero()]),
        Mask::AllTrue(_) => RDEncoder::new(values),
        Mask::Values(mask_values) => {
            let sample =
                sample_valid_values(values, mask_values, ALPRD_MAX_SAMPLE, ALPRD_SAMPLE_BLOCK);
            RDEncoder::new(&sample)
        }
    }
}

/// Resolves a [`Validity`] into a [`Mask`] without an [`ExecutionCtx`] for canonical validity
/// encodings, falling back to `AllTrue` for non-canonical validity arrays.
fn validity_mask_without_ctx(validity: &Validity, len: usize) -> Mask {
    match validity {
        Validity::NonNullable | Validity::AllValid => Mask::AllTrue(len),
        Validity::AllInvalid => Mask::AllFalse(len),
        Validity::Array(arr) => {
            if let Some(bool_array) = arr.as_opt::<Bool>() {
                Mask::from_buffer(bool_array.to_bit_buffer())
            } else if let Some(constant) = arr.as_opt::<Constant>() {
                if constant.scalar().as_bool().value().unwrap_or(false) {
                    Mask::AllTrue(len)
                } else {
                    Mask::AllFalse(len)
                }
            } else {
                Mask::AllTrue(len)
            }
        }
    }
}

/// Returns the index of the first exception in each `PATCH_CHUNK_SIZE`-row chunk of a
/// `len`-row array, so a reader can find one chunk's patches without scanning the others.
fn patch_chunk_offsets(exception_positions: &[u64], len: usize) -> Buffer<u64> {
    (0..len.div_ceil(PATCH_CHUNK_SIZE))
        .map(|chunk| {
            let chunk_start = (chunk * PATCH_CHUNK_SIZE) as u64;
            exception_positions.partition_point(|&position| position < chunk_start) as u64
        })
        .collect()
}

fn encode_generic_with_mask<T>(
    encoder: &RDEncoder,
    array: ArrayView<'_, Primitive>,
    validity: Validity,
    mask: &Mask,
) -> ALPRDArray
where
    T: ALPRDFloat + NativePType,
    T::UINT: NativePType,
{
    let values = array.as_slice::<T>();
    let len = values.len();

    let (left_parts, right_parts, exceptions_pos, exceptions) = match mask {
        Mask::AllTrue(_) => {
            let (left, right, exc_pos, exc) = split_parts(encoder, values);
            (Buffer::from(left), Buffer::from(right), exc_pos, exc)
        }
        Mask::AllFalse(_) => {
            assert!(
                !encoder.codes().is_empty(),
                "codes lookup table must be populated before RD encoding"
            );
            (
                Buffer::<u16>::zeroed(len),
                Buffer::<T::UINT>::zeroed(len),
                Vec::new(),
                Vec::new(),
            )
        }
        Mask::Values(mask_values) => {
            split_parts_masked(encoder, values, mask_values.bit_buffer().into())
        }
    };

    // Bit-pack down the encoded left-parts array that have been dictionary encoded.
    let primitive_left = PrimitiveArray::new(left_parts, validity);
    // SAFETY: by construction, all values in left_parts can be packed to left_bit_width.
    let packed_left = unsafe {
        bitpack_encode_unchecked(primitive_left, encoder.left_bit_width())
            .vortex_expect("bitpack_encode_unchecked should succeed for left parts")
            .into_array()
    };

    let primitive_right = PrimitiveArray::new(right_parts, Validity::NonNullable);
    // SAFETY: by construction, all values in right_parts are right_bit_width + leading zeros.
    let packed_right = unsafe {
        bitpack_encode_unchecked(primitive_right, encoder.right_bit_width())
            .vortex_expect("bitpack_encode_unchecked should succeed for right parts")
            .into_array()
    };

    // Patches for the left-parts that could not be dictionary encoded.
    let exceptions = (!exceptions_pos.is_empty()).then(|| {
        let chunk_offsets = patch_chunk_offsets(&exceptions_pos, len);
        let max_exc_pos = exceptions_pos.last().copied().unwrap_or_default();
        let bw = bit_width(max_exc_pos);

        let exc_pos_array =
            PrimitiveArray::new(Buffer::from(exceptions_pos), Validity::NonNullable);
        // SAFETY: We calculate bw such that it is wide enough to hold the largest position index.
        let packed_pos = unsafe {
            bitpack_encode_unchecked(exc_pos_array, bw)
                .vortex_expect("bitpack_encode_unchecked should succeed for exception positions")
                .into_array()
        };

        // SAFETY: the exception positions are collected in increasing order from positions in
        // the input, with one exception value each, and `chunk_offsets` is built from them.
        unsafe {
            Patches::new_unchecked(
                len,
                0,
                packed_pos,
                Buffer::from(exceptions).into_array(),
                Some(chunk_offsets.into_array()),
                Some(0),
            )
        }
    });

    ALPRD::try_new(
        DType::Primitive(T::PTYPE, packed_left.dtype().nullability()),
        packed_left,
        Buffer::<u16>::copy_from(encoder.codes()),
        packed_right,
        encoder.right_bit_width(),
        exceptions,
    )
    .vortex_expect("ALPRDArray construction in encode")
}

/// Decode ALP-RD encoded values back into their original floating point format.
///
/// The decoded values are written over `right_parts`, which shares its layout with the floats they
/// decode to.
///
/// The caller must check that each code is in `left_parts_dict`.
///
/// # Errors
///
/// Returns an error if the patches fail to execute.
///
/// # Panics
///
/// Panics if `left_parts` and `right_parts` differ in length, or if a patch index is out of bounds.
pub fn alp_rd_decode<T: ALPRDFloat + NativePType>(
    left_parts: BufferMut<u16>,
    left_parts_dict: &[u16],
    right_bit_width: u8,
    right_parts: BufferMut<T::UINT>,
    left_parts_patches: Option<Patches>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Buffer<T>>
where
    T::UINT: NativePType,
{
    let left_parts_patches = left_parts_patches
        .map(|patches| PrimitivePatches::execute(patches, ctx))
        .transpose()?;

    Ok(alp_rd_decode_primitive(
        left_parts,
        left_parts_dict,
        right_bit_width,
        right_parts,
        left_parts_patches,
    ))
}

/// Left-part patches whose indices and values are already primitive arrays.
pub(crate) struct PrimitivePatches {
    indices: PrimitiveArray,
    values: PrimitiveArray,
    offset: usize,
}

impl PrimitivePatches {
    fn execute(patches: Patches, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        let parts = patches.into_parts();

        Ok(Self {
            indices: parts.indices.execute::<PrimitiveArray>(ctx)?,
            values: parts.values.execute::<PrimitiveArray>(ctx)?,
            offset: parts.offset,
        })
    }

    /// Downcasts the patch children, which must already be primitive arrays.
    pub(crate) fn downcast(patches: Patches) -> Self {
        let parts = patches.into_parts();

        Self {
            indices: parts.indices.downcast::<Primitive>(),
            values: parts.values.downcast::<Primitive>(),
            offset: parts.offset,
        }
    }
}

/// Decode ALP-RD encoded values with patches that are already primitive arrays.
///
/// The caller must check that each code is in `left_parts_dict`.
///
/// # Panics
///
/// Panics if `left_parts` and `right_parts` differ in length, or if a patch index is out of bounds.
pub(crate) fn alp_rd_decode_primitive<T: ALPRDFloat + NativePType>(
    left_parts: BufferMut<u16>,
    left_parts_dict: &[u16],
    right_bit_width: u8,
    mut right_parts: BufferMut<T::UINT>,
    left_parts_patches: Option<PrimitivePatches>,
) -> Buffer<T>
where
    T::UINT: NativePType,
{
    if left_parts.len() != right_parts.len() {
        vortex_panic!("alp_rd_decode: left_parts.len != right_parts.len");
    }

    // Every code, patched or not, combines through the pre-shifted dictionary. Codes are masked into
    // the dictionary, so the codes at patched positions decode to a wrong left part that the patches
    // then overwrite.
    combine_codes::<T>(
        right_parts.as_mut_slice(),
        left_parts.as_ref(),
        left_parts_dict,
        right_bit_width,
    );

    if let Some(patches) = left_parts_patches {
        match_each_integer_ptype!(patches.indices.ptype(), |I| {
            apply_left_patches::<T, I>(
                right_parts.as_mut_slice(),
                patches.indices.as_slice::<I>(),
                patches.values.as_slice::<u16>(),
                patches.offset,
                right_bit_width,
            )
        });
    }

    // SAFETY: all bit patterns of T::UINT are valid T (u32↔f32 or u64↔f64).
    unsafe { right_parts.transmute::<T>() }.freeze()
}

/// Runs [`RDEncoder::split_parts`] compiled for the best SIMD level the CPU supports.
fn split_parts<T: ALPRDFloat>(
    encoder: &RDEncoder,
    values: &[T],
) -> (Vec<u16>, Vec<T::UINT>, Vec<u64>, Vec<u16>) {
    dispatch!(Level::new(), simd => split_parts_simd(simd, encoder, values))
}

/// The split is generic, so it is instantiated in this crate; whatever of it LLVM inlines here is
/// compiled with the target features of `S`.
#[simd]
fn split_parts_simd<S: Simd, T: ALPRDFloat>(
    _: S,
    encoder: &RDEncoder,
    values: &[T],
) -> (Vec<u16>, Vec<T::UINT>, Vec<u64>, Vec<u16>) {
    encoder.split_parts(values)
}

/// Splits `values` into dictionary-encoded left parts and right parts with a validity mask,
/// zeroing null lanes and skipping exception emission on null positions without copying `values`.
fn split_parts_masked<T: ALPRDFloat + NativePType>(
    encoder: &RDEncoder,
    values: &[T],
    mask: BitBufferView<'_>,
) -> (Buffer<u16>, Buffer<T::UINT>, Vec<u64>, Vec<u16>)
where
    T::UINT: NativePType,
{
    let len = values.len();
    let mut left_parts = BufferMut::<u16>::with_capacity(len);
    let mut right_parts = BufferMut::<T::UINT>::with_capacity(len);
    let mut exception_pos = Vec::with_capacity(len / 4);
    let mut exception_values = Vec::with_capacity(len / 4);

    dispatch!(Level::new(), simd => split_parts_masked_simd(
        simd,
        encoder.codes(),
        encoder.right_bit_width(),
        values,
        mask,
        &mut left_parts.spare_capacity_mut()[..len],
        &mut right_parts.spare_capacity_mut()[..len],
        &mut exception_pos,
        &mut exception_values,
    ));

    // SAFETY: `split_parts_masked_simd` initializes every element in `0..len`.
    unsafe {
        left_parts.set_len(len);
        right_parts.set_len(len);
    }

    (
        left_parts.freeze(),
        right_parts.freeze(),
        exception_pos,
        exception_values,
    )
}

#[simd]
#[allow(clippy::too_many_arguments)]
fn split_parts_masked_simd<S: Simd, T: ALPRDFloat>(
    _: S,
    codes: &[u16],
    right_bit_width: u8,
    values: &[T],
    mask: BitBufferView<'_>,
    left_out: &mut [MaybeUninit<u16>],
    right_out: &mut [MaybeUninit<T::UINT>],
    exception_pos: &mut Vec<u64>,
    exception_values: &mut Vec<u16>,
) {
    assert!(
        !codes.is_empty(),
        "codes lookup table must be populated before RD encoding"
    );
    assert!(
        codes.len() <= 8,
        "ALP-RD dictionary must hold at most 8 entries"
    );
    assert_eq!(left_out.len(), values.len());
    assert_eq!(right_out.len(), values.len());
    assert_eq!(mask.len(), values.len());

    // Pad unused dictionary slots with `codes[0]`; scanning in reverse ensures the lowest matching
    // code wins, matching `alp::RDEncoder::split_parts` while keeping a fixed 8-way loop.
    let mut dict = [codes[0]; 8];
    dict[..codes.len()].copy_from_slice(codes);

    let right_shift = usize::from(right_bit_width);
    let right_mask = (T::UINT::one() << right_shift) - T::UINT::one();

    for_each_mask_word(mask, |valid_word, base, count| {
        let word_values = &values[base..base + count];
        let word_left = &mut left_out[base..base + count];
        let word_right = &mut right_out[base..base + count];

        let mut miss_bits: u64 = 0;
        for (bit_idx, ((&value, left), right)) in word_values
            .iter()
            .zip(word_left.iter_mut())
            .zip(word_right.iter_mut())
            .enumerate()
        {
            let bits = T::to_bits(value);
            right.write(bits & right_mask);

            let pattern = <T as ALPRDFloat>::to_u16(bits >> right_shift);
            let mut matched_code = 0u16;
            let mut matched = false;
            for code in (0..8).rev() {
                if dict[code] == pattern {
                    matched_code = code as u16;
                    matched = true;
                }
            }
            miss_bits |= (!matched as u64) << bit_idx;
            left.write(matched_code);
        }

        let mut valid_misses = miss_bits & valid_word;
        while valid_misses != 0 {
            let bit = valid_misses.trailing_zeros() as usize;
            let bits = T::to_bits(word_values[bit]);
            let pattern = <T as ALPRDFloat>::to_u16(bits >> right_shift);
            exception_pos.push((base + bit) as u64);
            exception_values.push(pattern);
            valid_misses &= valid_misses - 1;
        }

        let mut nulls = !valid_word & low_bits_mask(count);
        while nulls != 0 {
            let bit = nulls.trailing_zeros() as usize;
            word_left[bit].write(0);
            word_right[bit].write(T::UINT::zero());
            nulls &= nulls - 1;
        }
    });
}

/// Combines the dictionary codes into `right_parts` with the `alp` crate's SIMD lookup, compiled
/// for the best SIMD level the CPU supports.
///
/// `alp` selects its explicit AVX-512, AVX2 or NEON kernel itself. Dispatching here lets the
/// scalar loop it uses for short inputs, and for `f32` without AVX2, vectorize with the CPU's
/// target features too.
fn combine_codes<T: ALPRDFloat>(
    right_parts: &mut [T::UINT],
    left_parts: &[u16],
    left_parts_dict: &[u16],
    right_bit_width: u8,
) {
    dispatch!(Level::new(), simd => combine_codes_simd::<_, T>(
        simd,
        right_parts,
        left_parts,
        left_parts_dict,
        right_bit_width,
    ))
}

#[simd]
fn combine_codes_simd<S: Simd, T: ALPRDFloat>(
    _: S,
    right_parts: &mut [T::UINT],
    left_parts: &[u16],
    left_parts_dict: &[u16],
    right_bit_width: u8,
) {
    alp_rd_combine_codes_inplace::<T>(right_parts, left_parts, left_parts_dict, right_bit_width);
}

/// Replaces the left part of each patched value in already combined `right_parts` with its
/// exception left part.
///
/// `offset` is subtracted from every index, to support patches stored relative to the start of an
/// unsliced array.
fn apply_left_patches<T: ALPRDFloat, I: PrimInt>(
    combined: &mut [T::UINT],
    indices: &[I],
    patch_values: &[u16],
    offset: usize,
    right_bit_width: u8,
) {
    assert_eq!(
        indices.len(),
        patch_values.len(),
        "ALP-RD patch indices and values differ in length"
    );

    let shift = usize::from(right_bit_width);
    let right_mask = (T::UINT::one() << shift) - T::UINT::one();
    for (&index, &left) in indices.iter().zip(patch_values) {
        let index = index
            .to_usize()
            .vortex_expect("ALP-RD patch index must fit in usize")
            - offset;
        let value = &mut combined[index];
        *value = (*value & right_mask) | (<T as ALPRDFloat>::from_u16(left) << shift);
    }
}

/// Checks that each valid code is in the dictionary, and sets the codes in null slots to zero.
fn check_left_parts_codes(codes: &mut [u16], validity: &Mask, dict_len: usize) -> VortexResult<()> {
    let max_valid_code = match validity.bit_buffer() {
        AllOr::All => codes.iter().copied().max(),
        AllOr::None => {
            codes.fill(0);
            None
        }
        AllOr::Some(valid) => codes
            .iter_mut()
            .zip(valid.iter())
            .map(|(code, is_valid)| {
                if !is_valid {
                    *code = 0;
                }
                *code
            })
            .max(),
    };

    match max_valid_code {
        Some(code) if usize::from(code) >= dict_len => Err(code_not_in_dictionary(code, dict_len)),
        _ => Ok(()),
    }
}

fn code_not_in_dictionary(code: u16, dict_len: usize) -> VortexError {
    vortex_err!("ALP-RD left-parts code {code} is not in the {dict_len}-entry dictionary")
}

#[cfg(test)]
mod tests {
    use std::ops::Range;
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::ArrayContext;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::NativePType;
    use vortex_array::serde::SerializeOptions;
    use vortex_array::serde::SerializedArray;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
    use vortex_buffer::ByteBufferMut;
    use vortex_error::VortexResult;
    use vortex_mask::Mask;
    use vortex_session::VortexSession;
    use vortex_session::registry::ReadContext;

    use super::ALPRD;
    use super::ALPRDArrayExt;
    use super::ALPRDFloat;
    use super::Primitive;
    use super::RDEncoder;
    use super::RDEncoderExt;
    use super::alp_rd_encode;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty();
        crate::initialize(&session);
        vortex_fastlanes::initialize(&session);
        session
    });

    /// `len` copies of 0.1 with an outlier, which can't be dictionary encoded, at each of `outliers`.
    fn with_outliers(len: usize, outliers: &[usize]) -> PrimitiveArray {
        PrimitiveArray::from_iter(
            (0..len).map(|row| if outliers.contains(&row) { 3e100 } else { 0.1 }),
        )
    }

    /// Reads the patch chunk offsets of an encoded ALP-RD array.
    fn chunk_offsets(array: &super::ALPRDArray) -> VortexResult<Vec<u64>> {
        let patches = array
            .left_parts_patches()
            .expect("outliers produce patches");
        let offsets = patches
            .chunk_offsets()
            .clone()
            .expect("the encoder writes chunk offsets")
            .execute::<PrimitiveArray>(&mut SESSION.create_execution_ctx())?;
        Ok(offsets.as_slice::<u64>().to_vec())
    }

    #[rstest]
    #[case::several_chunks(3000, &[5, 1500, 1600, 2999], &[0, 1, 3])]
    #[case::only_the_last_chunk(3000, &[2500], &[0, 0, 0])]
    #[case::first_and_last_row_of_one_chunk(1024, &[0, 1023], &[0])]
    #[case::first_row_of_the_second_chunk(1025, &[1024], &[0, 0])]
    #[case::either_side_of_a_boundary(2048, &[1023, 1024], &[0, 1])]
    fn test_encode_writes_patch_chunk_offsets(
        #[case] len: usize,
        #[case] outliers: &[usize],
        #[case] expected: &[u64],
    ) -> VortexResult<()> {
        let array = with_outliers(len, outliers);
        let encoded = RDEncoder::new(&[0.1f64]).encode(array.as_view());
        assert_eq!(chunk_offsets(&encoded)?, expected);
        assert_arrays_eq!(encoded, array, &mut SESSION.create_execution_ctx());
        Ok(())
    }

    #[rstest]
    #[case::whole(0..3000)]
    #[case::inside_one_chunk(5..1000)]
    #[case::across_a_boundary(1023..1025)]
    #[case::from_a_chunk_start(1024..2048)]
    #[case::middle(1500..1601)]
    #[case::tail(2000..3000)]
    fn test_slice_with_patch_chunk_offsets(#[case] range: Range<usize>) -> VortexResult<()> {
        // Slicing and single-value lookups both use the chunk offsets to find patches.
        let array = with_outliers(3000, &[5, 1023, 1024, 1500, 1600, 2999]).into_array();
        let encoded = RDEncoder::new(&[0.1f64])
            .encode(array.as_::<Primitive>())
            .into_array();
        let mut ctx = SESSION.create_execution_ctx();
        let sliced = encoded.slice(range.clone())?;
        let expected = array.slice(range)?;
        assert_arrays_eq!(sliced, expected, &mut ctx);
        for index in 0..sliced.len() {
            assert_eq!(
                sliced.execute_scalar(index, &mut ctx)?,
                expected.execute_scalar(index, &mut ctx)?
            );
        }
        Ok(())
    }

    #[test]
    fn test_serde_roundtrip_keeps_patch_chunk_offsets() -> VortexResult<()> {
        let array = with_outliers(3000, &[5, 1500, 1600, 2999]);
        let encoded = RDEncoder::new(&[0.1f64]).encode(array.as_view());
        let expected_offsets = chunk_offsets(&encoded)?;

        let context = ArrayContext::empty();
        let serialized = encoded.clone().into_array().serialize(
            &context,
            &SESSION,
            &SerializeOptions::default(),
        )?;
        let mut bytes = ByteBufferMut::empty();
        for buffer in serialized {
            bytes.extend_from_slice(buffer.as_ref());
        }
        let decoded = SerializedArray::try_from(bytes.freeze())?.decode(
            encoded.dtype(),
            encoded.len(),
            &ReadContext::new(context.to_ids()),
            &SESSION,
        )?;

        let decoded_rd = decoded.as_::<ALPRD>();
        assert_eq!(chunk_offsets(&decoded_rd.into_owned())?, expected_offsets);
        assert_arrays_eq!(decoded, array, &mut SESSION.create_execution_ctx());
        Ok(())
    }

    #[test]
    fn rd_encoder_uses_deterministic_codes_for_tied_prefixes() {
        let values: Vec<_> = (1_000_i64..2_000)
            .map(|row_id| ((row_id * 53) % 36_000) as f64 / 100.0 - 180.0)
            .collect();
        let array = PrimitiveArray::from_iter(values.iter().copied());

        let encoded = RDEncoder::new(&values).encode(array.as_view());

        assert_eq!(encoded.right_bit_width(), 53);
        assert_eq!(
            encoded.left_parts_dictionary().as_slice(),
            &[514, 1538, 515, 1539, 513, 1537, 512, 1536]
        );
    }

    /// Generates >8 distinct dirty left-part bit patterns including NaN, sentinel bits, and extremes.
    fn dirty_null_values_f32() -> Vec<f32> {
        let mut values = vec![
            f32::NAN,
            f32::from_bits(0xdead_beef),
            1e30_f32,
            -1e30_f32,
            f32::INFINITY,
            f32::NEG_INFINITY,
            1e-30_f32,
            f32::from_bits(1),
        ];
        for prefix in 0u32..16 {
            values.push(f32::from_bits((0x2000 + prefix * 0x0333) << 16 | 0x5a5a));
        }
        values
    }

    fn dirty_null_values_f64() -> Vec<f64> {
        let mut values = vec![
            f64::NAN,
            f64::from_bits(0xdead_beef_cafe_babe),
            1e300_f64,
            -1e300_f64,
            f64::INFINITY,
            f64::NEG_INFINITY,
            1e-300_f64,
            f64::from_bits(1),
        ];
        for prefix in 0u64..16 {
            values.push(f64::from_bits(
                (0x2000 + prefix * 0x0333) << 48 | 0x0000_5a5a_1234_5678,
            ));
        }
        values
    }

    fn check_dirty_nulls_alprd<T>(clean_values: &[T], dirty_values: &[T]) -> VortexResult<()>
    where
        T: ALPRDFloat + NativePType,
    {
        let mut ctx = SESSION.create_execution_ctx();
        let len = 10_000;
        let mut raw = Vec::with_capacity(len);
        let mut valid_only = Vec::with_capacity(len);
        let mut is_valid = Vec::with_capacity(len);

        for row in 0..len {
            if row % 10 == 0 {
                raw.push(dirty_values[row % dirty_values.len()]);
                is_valid.push(false);
            } else {
                let value = clean_values[row % clean_values.len()];
                raw.push(value);
                valid_only.push(value);
                is_valid.push(true);
            }
        }

        let array = PrimitiveArray::new(Buffer::from(raw), Validity::from_iter(is_valid));
        let clean_array =
            PrimitiveArray::new(Buffer::from(valid_only.clone()), Validity::NonNullable);

        // Both `alp_rd_encode` and `RDEncoder::encode` must avoid patches on dirty null slots.
        let encoded = alp_rd_encode(array.as_view(), &mut ctx)?;
        let clean_encoded = alp_rd_encode(clean_array.as_view(), &mut ctx)?;

        assert!(encoded.left_parts_patches().is_none());
        assert_eq!(encoded.right_bit_width(), clean_encoded.right_bit_width());
        assert_eq!(
            encoded.left_parts_dictionary().as_slice(),
            clean_encoded.left_parts_dictionary().as_slice()
        );
        assert_arrays_eq!(encoded.clone(), array, &mut ctx);

        let direct_encoded = RDEncoder::new(&valid_only).encode(array.as_view());
        assert!(direct_encoded.left_parts_patches().is_none());
        assert_arrays_eq!(direct_encoded, array, &mut ctx);

        // Verify take, filter, slice, and scalar_at remain sound on dummy null slots.
        let take_indices = PrimitiveArray::from_iter([0u32, 1, 10, 11, 9990, 9999]).into_array();
        assert_arrays_eq!(
            encoded.take(take_indices.clone())?,
            array.take(take_indices)?,
            &mut ctx
        );
        let filter_mask = Mask::from_iter((0..len).map(|row| row % 7 == 0));
        assert_arrays_eq!(
            encoded.filter(filter_mask.clone())?,
            array.filter(filter_mask)?,
            &mut ctx
        );
        assert_arrays_eq!(encoded.slice(5..55)?, array.slice(5..55)?, &mut ctx);
        assert!(encoded.execute_scalar(0, &mut ctx)?.is_null());
        assert_eq!(
            encoded.execute_scalar(1, &mut ctx)?,
            array.execute_scalar(1, &mut ctx)?
        );

        Ok(())
    }

    #[test]
    fn test_dirty_nulls_alprd_f32() -> VortexResult<()> {
        let clean: Vec<f32> = (0..64).map(|idx| 1.0_f32 + (idx as f32) * 1e-4).collect();
        check_dirty_nulls_alprd(&clean, &dirty_null_values_f32())
    }

    #[test]
    fn test_dirty_nulls_alprd_f64() -> VortexResult<()> {
        let clean: Vec<f64> = (0..64).map(|idx| 1.0_f64 + (idx as f64) * 1e-6).collect();
        check_dirty_nulls_alprd(&clean, &dirty_null_values_f64())
    }

    #[rstest]
    #[case::f32(1.123_456_f32, dirty_null_values_f32())]
    #[case::f64(1.123_456_789_012_f64, dirty_null_values_f64())]
    fn test_dirty_nulls_alprd_edge_cases<T>(
        #[case] valid_value: T,
        #[case] dirty_values: Vec<T>,
    ) -> VortexResult<()>
    where
        T: ALPRDFloat + NativePType,
    {
        let mut ctx = SESSION.create_execution_ctx();

        // All-null array with >8 distinct dirty left-part patterns.
        let all_null_raw: Vec<T> = (0..1024)
            .map(|row| dirty_values[row % dirty_values.len()])
            .collect();
        let all_null =
            PrimitiveArray::new(Buffer::from(all_null_raw.clone()), Validity::AllInvalid);
        let all_null_encoded = alp_rd_encode(all_null.as_view(), &mut ctx)?;
        assert!(all_null_encoded.left_parts_patches().is_none());
        assert_arrays_eq!(all_null_encoded.clone(), all_null, &mut ctx);
        assert!(all_null_encoded.execute_scalar(0, &mut ctx)?.is_null());

        // Compressed ConstantArray(false) validity with dirty garbage bits.
        let const_null = PrimitiveArray::new(
            Buffer::from(all_null_raw),
            Validity::Array(ConstantArray::new(false, 1024).into_array()),
        );
        let const_null_encoded = alp_rd_encode(const_null.as_view(), &mut ctx)?;
        assert!(const_null_encoded.left_parts_patches().is_none());
        assert_arrays_eq!(const_null_encoded, const_null, &mut ctx);

        // Empty array must not panic.
        let empty = PrimitiveArray::new(Buffer::<T>::empty(), Validity::AllValid);
        let empty_encoded = alp_rd_encode(empty.as_view(), &mut ctx)?;
        assert!(empty_encoded.left_parts_patches().is_none());
        assert_arrays_eq!(empty_encoded, empty, &mut ctx);

        // Single valid value surrounded by dirty nulls with >8 distinct left parts.
        let mut one_valid_raw: Vec<T> = (0..1024)
            .map(|row| dirty_values[row % dirty_values.len()])
            .collect();
        let mut one_valid_mask = vec![false; 1024];
        one_valid_raw[777] = valid_value;
        one_valid_mask[777] = true;
        let one_valid = PrimitiveArray::new(
            Buffer::from(one_valid_raw),
            Validity::from_iter(one_valid_mask),
        );
        let one_valid_encoded = alp_rd_encode(one_valid.as_view(), &mut ctx)?;
        assert!(one_valid_encoded.left_parts_patches().is_none());
        assert_eq!(one_valid_encoded.left_parts_dictionary().len(), 1);
        assert_arrays_eq!(one_valid_encoded, one_valid, &mut ctx);

        Ok(())
    }
}
