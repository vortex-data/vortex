// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::LazyLock;

use prost::Message;
use vortex_array::ArrayContext;
use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::ArrayVTable;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::assert_arrays_eq;
use vortex_array::serde::SerializeOptions;
use vortex_array::serde::SerializedArray;
use vortex_array::session::ArraySessionExt;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_session::VortexSession;
use vortex_session::registry::ReadContext;

use crate::BitPacked;
use crate::BitPackedArray;
use crate::bitpacking::bitpack_compress::bitpack_to_best_bit_width;
use crate::bitpacking::plugin::BitPackedMetadata;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = vortex_array::array_session();
    crate::initialize(&session);
    session
});

fn serde_roundtrip(array: &BitPackedArray) -> VortexResult<(ArrayId, Vec<u8>, ArrayRef)> {
    let array_ref = array.as_array();
    let serialization = SESSION
        .array_serialize(array_ref)?
        .ok_or_else(|| vortex_err!("BitPacked must serialize"))?;
    let array_ctx = ArrayContext::empty();
    let buffers = array_ref.serialize(&array_ctx, &SESSION, &SerializeOptions::default())?;
    let mut bytes = ByteBufferMut::empty();
    for buffer in buffers {
        bytes.extend_from_slice(&buffer);
    }
    let read = SerializedArray::try_from(bytes.freeze())?.decode(
        array_ref.dtype(),
        array_ref.len(),
        &ReadContext::new(array_ctx.to_ids()),
        &SESSION,
    )?;
    Ok((serialization.serialized_id, serialization.metadata, read))
}

#[test]
fn uniform_widths_serialize_as_original_format() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let values: Vec<u32> = (0..3000).map(|i| i % 128).collect();
    let packed =
        bitpack_to_best_bit_width(&PrimitiveArray::from_iter(values.iter().copied()), &mut ctx)?;
    assert_eq!(packed.as_array().children().len(), 1);
    assert!(
        SESSION
            .array_serialize(packed.as_array())?
            .ok_or_else(|| vortex_err!("must serialize"))?
            .children
            .is_empty()
    );
    let (id, metadata, read) = serde_roundtrip(&packed)?;
    assert_eq!(id, ArrayVTable::id(&BitPacked));
    let original = BitPackedMetadata {
        bit_width: 7,
        offset: 0,
        patches: None,
    }
    .encode_to_vec();
    assert_eq!(metadata, original);
    assert_arrays_eq!(
        read,
        PrimitiveArray::from_iter(values.iter().copied()),
        &mut ctx
    );
    Ok(())
}
