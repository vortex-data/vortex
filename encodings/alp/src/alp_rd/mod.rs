// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Vortex array support for ALP-RD ("real doubles"), on top of the `alp` crate's encoder.

#![expect(clippy::cast_possible_truncation)]

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
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::NativePType;
use vortex_array::match_each_integer_ptype;
use vortex_array::patches::PATCH_CHUNK_SIZE;
use vortex_array::patches::Patches;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexError;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_fastlanes::bitpack_compress::bitpack_encode_unchecked;
use vortex_mask::AllOr;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use crate::match_each_alp_float_ptype;

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
        match_each_alp_float_ptype!(array.ptype(), |P| { encode_generic::<P>(self, array) })
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

fn encode_generic<T>(encoder: &RDEncoder, array: ArrayView<'_, Primitive>) -> ALPRDArray
where
    T: ALPRDFloat + NativePType,
    T::UINT: NativePType,
{
    let (left_parts, right_parts, exceptions_pos, exceptions) =
        split_parts(encoder, array.as_slice::<T>());
    let len = left_parts.len();

    // Bit-pack down the encoded left-parts array that have been dictionary encoded.
    let primitive_left = PrimitiveArray::new(
        Buffer::from(left_parts),
        array
            .validity()
            .vortex_expect("ALP RD validity should be derivable"),
    );
    // SAFETY: by construction, all values in left_parts can be packed to left_bit_width.
    let packed_left = unsafe {
        bitpack_encode_unchecked(primitive_left, encoder.left_bit_width())
            .vortex_expect("bitpack_encode_unchecked should succeed for left parts")
            .into_array()
    };

    let primitive_right = PrimitiveArray::new(Buffer::from(right_parts), Validity::NonNullable);
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
    mut right_parts: BufferMut<T::UINT>,
    left_parts_patches: Option<Patches>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Buffer<T>>
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
        let indices = patches.indices().clone().execute::<PrimitiveArray>(ctx)?;
        let patch_values = patches.values().clone().execute::<PrimitiveArray>(ctx)?;
        match_each_integer_ptype!(indices.ptype(), |I| {
            apply_left_patches::<T, I>(
                right_parts.as_mut_slice(),
                indices.as_slice::<I>(),
                patch_values.as_slice::<u16>(),
                patches.offset(),
                right_bit_width,
            )
        });
    }

    // SAFETY: all bit patterns of T::UINT are valid T (u32↔f32 or u64↔f64).
    Ok(unsafe { right_parts.transmute::<T>() }.freeze())
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
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::serde::SerializeOptions;
    use vortex_array::serde::SerializedArray;
    use vortex_buffer::ByteBufferMut;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;
    use vortex_session::registry::ReadContext;

    use super::ALPRD;
    use super::ALPRDArrayExt;
    use super::Primitive;
    use super::RDEncoder;
    use super::RDEncoderExt;

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
}
