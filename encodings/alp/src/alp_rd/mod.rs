// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Vortex array support for ALP-RD ("real doubles"), on top of the `alp` crate's encoder.

#![expect(clippy::cast_possible_truncation)]

pub use ::alp::ALPRDFloat;
pub use ::alp::RDEncoder;
use ::alp::alp_rd_apply_patches;
use ::alp::alp_rd_combine_codes_inplace;
use ::alp::alp_rd_combine_inplace;
use ::alp::alp_rd_dict_decode_inplace;
use ::alp::bit_width;
pub use array::*;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::NativePType;
use vortex_array::match_each_integer_ptype;
use vortex_array::patches::Patches;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_panic;
use vortex_fastlanes::bitpack_compress::bitpack_encode_unchecked;
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

fn encode_generic<T>(encoder: &RDEncoder, array: ArrayView<'_, Primitive>) -> ALPRDArray
where
    T: ALPRDFloat + NativePType,
    T::UINT: NativePType,
{
    let (left_parts, right_parts, exceptions_pos, exceptions) =
        encoder.split_parts(array.as_slice::<T>());
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

        Patches::new(
            len,
            0,
            packed_pos,
            Buffer::from(exceptions).into_array(),
            // TODO(0ax1): handle chunk offsets
            None,
        )
        .vortex_expect("Patches construction in encode")
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
/// # Panics
///
/// Panics if `left_parts` and `right_parts` differ in length.
pub fn alp_rd_decode<T: ALPRDFloat + NativePType>(
    mut left_parts: BufferMut<u16>,
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

    if let Some(patches) = left_parts_patches {
        // Patched path: some left-part codes map to exception values that live outside
        // the dictionary. We must dictionary-decode first, then overwrite the exceptions,
        // before we can combine with right-parts.
        alp_rd_dict_decode_inplace(left_parts.as_mut_slice(), left_parts_dict);

        let indices = patches.indices().clone().execute::<PrimitiveArray>(ctx)?;
        let patch_values = patches.values().clone().execute::<PrimitiveArray>(ctx)?;
        match_each_integer_ptype!(indices.ptype(), |I| {
            alp_rd_apply_patches(
                left_parts.as_mut_slice(),
                indices.as_slice::<I>(),
                patch_values.as_slice::<u16>(),
                patches.offset(),
            )
        });

        alp_rd_combine_inplace::<T>(
            right_parts.as_mut_slice(),
            left_parts.as_ref(),
            right_bit_width,
        );
    } else {
        // Non-patched fast path: every code maps through the dictionary, so the hot loop is a
        // lookup into the pre-shifted dictionary and an OR.
        alp_rd_combine_codes_inplace::<T>(
            right_parts.as_mut_slice(),
            left_parts.as_ref(),
            left_parts_dict,
            right_bit_width,
        );
    }

    // SAFETY: all bit patterns of T::UINT are valid T (u32↔f32 or u64↔f64).
    Ok(unsafe { right_parts.transmute::<T>() }.freeze())
}

#[cfg(test)]
mod tests {
    use vortex_array::arrays::PrimitiveArray;

    use super::ALPRDArrayExt;
    use super::RDEncoder;
    use super::RDEncoderExt;

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
