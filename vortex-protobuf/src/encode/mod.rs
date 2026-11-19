// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Encoding Vortex arrays of protobuf messages back to the protobuf wire format.
//!
//! The encoder writes the canonical storage columns straight to bytes, without a message object
//! per row, in the [encoding](crate#decoding-and-encoding) that the crate documentation describes.
//!
//! A nested message is written after its length, so the encoder first computes the encoded size
//! of every message from the innermost messages outward, and then writes each row in one pass.

mod columns;
mod function;
mod scalar;
mod value;

use std::sync::Arc;

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::extension::ExtensionArrayExt;
use vortex_array::arrays::varbinview::build_views::MAX_BUFFER_LEN;
use vortex_array::arrays::varbinview::build_views::build_views_from_offsets;
use vortex_array::dtype::DType;
use vortex_array::validity::Validity;
use vortex_buffer::BufferMut;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

pub use self::function::Encode;
pub use self::function::protobuf_encode;
pub(crate) use self::value::encode_value;
use crate::Protobuf;
use crate::encode::columns::MessageEncoder;

/// Serializes each message of a protobuf extension array in the protobuf wire format.
///
/// Returns a binary array of the same length and nullability, with null where the input holds a
/// null message. The [crate documentation](crate#decoding-and-encoding) describes the encoding.
///
/// # Errors
///
/// Returns an error if `array` is not a protobuf extension array, if executing it fails, if a
/// message is larger than a binary array can hold in one value, or if a JSON value in a dedicated
/// form is not valid JSON of its message type.
pub fn encode_messages(array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<VarBinViewArray> {
    let Some(metadata) = array
        .dtype()
        .as_extension_opt()
        .and_then(|ext_dtype| ext_dtype.metadata_opt::<Protobuf>())
    else {
        vortex_bail!(
            "Encoding protobuf messages requires a {} array, got {}",
            Protobuf::ID,
            array.dtype()
        );
    };
    let storage = array
        .clone()
        .execute::<ExtensionArray>(ctx)?
        .storage_array()
        .clone();
    let mut messages = MessageEncoder::new(metadata.plan(), storage, ctx)?;
    messages.compute_sizes();

    let len = array.len();
    let mut total = 0;
    for row in 0..len {
        let size = messages.size(row);
        if size > MAX_BUFFER_LEN {
            vortex_bail!(
                "Protobuf message {row} of {size} bytes exceeds the limit of {MAX_BUFFER_LEN}"
            );
        }
        total += size;
    }
    let mut bytes = Vec::with_capacity(total);
    let mut offsets = BufferMut::<u64>::with_capacity(len + 1);
    offsets.push(0);
    for row in 0..len {
        if messages.validity().value(row) {
            messages.write(row, &mut bytes);
        }
        offsets.push(bytes.len() as u64);
    }
    debug_assert_eq!(bytes.len(), total, "the encoded size of the messages");

    let nullability = array.dtype().nullability();
    let validity = Validity::from_mask(messages.validity().clone(), nullability);
    let (buffers, views) =
        build_views_from_offsets(0, MAX_BUFFER_LEN, ByteBuffer::from(bytes), &offsets);
    // SAFETY: the views come from the offsets of the messages, which were written back to back,
    // and the validity has one entry per message.
    Ok(unsafe {
        VarBinViewArray::new_unchecked(
            views,
            Arc::from(buffers),
            DType::Binary(nullability),
            validity,
        )
    })
}
