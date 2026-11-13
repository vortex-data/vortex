// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::LazyLock;

use prost::Message as _;
use rstest::rstest;
use vortex_array::ArrayContext;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::assert_arrays_eq;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::dtype::PType;
use vortex_array::optimizer::ArrayOptimizer;
use vortex_array::scalar::ScalarValue;
use vortex_array::serde::SerializeOptions;
use vortex_array::serde::SerializedArray;
use vortex_array::session::ArraySessionExt;
use vortex_buffer::ByteBuffer;
use vortex_buffer::ByteBufferMut;
use vortex_session::registry::ReadContext;

use super::*;
use crate::FoRArray;
use crate::for_::array::FoRArraySlotsExt;
use crate::for_::tests::blocked;
use crate::for_::tests::global;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::empty();
    crate::initialize(&session);
    session
});

fn roundtrip(array: &ArrayRef) -> VortexResult<ArrayRef> {
    let array_ctx = ArrayContext::empty();
    let buffers = array.serialize(&array_ctx, &SESSION, &SerializeOptions::default())?;
    let mut bytes = ByteBufferMut::empty();
    for buffer in buffers {
        bytes.extend_from_slice(&buffer);
    }
    SerializedArray::try_from(bytes.freeze())?.decode(
        array.dtype(),
        array.len(),
        &ReadContext::new(array_ctx.to_ids()),
        &SESSION,
    )
}

#[rstest]
#[case::signed(FoR::encode(PrimitiveArray::from_iter([-5i32, 0, 7, 100]), &mut SESSION.create_execution_ctx()))]
#[case::unsigned(FoR::encode(PrimitiveArray::from_iter(1_000u64..3_000), &mut SESSION.create_execution_ctx()))]
#[case::nullable(FoR::encode(
    PrimitiveArray::from_option_iter([Some(10i16), None, Some(12)]),
    &mut SESSION.create_execution_ctx(),
))]
fn serde_roundtrip(#[case] array: VortexResult<FoRArray>) -> VortexResult<()> {
    let array = array?;
    let serialization = SESSION
        .array_serialize(array.as_array())?
        .ok_or_else(|| vortex_err!("FoR must serialize"))?;
    assert_eq!(serialization.serialized_id, for_v1_id());
    assert_eq!(
        serialization.metadata,
        ScalarValue::to_proto_bytes::<Vec<u8>>(global(&array)?.value())
    );

    let read = roundtrip(array.as_array())?;
    assert_eq!(read.encoding_id(), VTable::id(&FoR));
    assert_arrays_eq!(read, array, &mut SESSION.create_execution_ctx());
    Ok(())
}

fn chunked() -> VortexResult<FoRArray> {
    FoR::encode_chunked(
        PrimitiveArray::from_option_iter(
            (0..5000u32).map(|i| (i % 13 != 0).then_some((i / 1024) * 1_000_000 + i % 7)),
        ),
        &mut SESSION.create_execution_ctx(),
    )
}

#[rstest]
#[case::whole(0..5000)]
#[case::sliced(1500..2600)]
fn v2_roundtrip(#[case] range: std::ops::Range<usize>) -> VortexResult<()> {
    let array = chunked()?
        .into_array()
        .slice(range.clone())?
        .optimize_ctx(&SESSION)?;
    let serialization = SESSION
        .array_serialize(&array)?
        .ok_or_else(|| vortex_err!("FoR must serialize"))?;
    assert_eq!(serialization.serialized_id, for_v2_id());
    assert_eq!(
        v2::FoRV2Metadata::decode(serialization.metadata.as_slice())?.offset,
        u32::try_from(range.start % 1024)?
    );

    let read = roundtrip(&array)?;
    assert_eq!(read.encoding_id(), VTable::id(&FoR));
    blocked(&read.as_::<FoR>().into_owned())?;
    assert_arrays_eq!(read, array, &mut SESSION.create_execution_ctx());
    Ok(())
}

fn deserialize_v2_parts(
    dtype: &DType,
    len: usize,
    metadata: &v2::FoRV2Metadata,
    children: &[ArrayRef],
) -> VortexResult<ArrayRef> {
    FoRPlugin.deserialize(
        ArrayDeserialization::new(
            for_v2_id(),
            dtype,
            len,
            &metadata.encode_to_vec(),
            &[],
            &children,
        ),
        &SESSION,
    )
}

#[test]
fn v2_rejects_malformed_parts() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let array = chunked()?;
    let encoded = array.encoded().clone();
    let references = blocked(&array)?;
    let dtype = array.dtype().clone();
    let len = array.len();
    let metadata = v2::FoRV2Metadata { offset: 0 };
    let both = [encoded.clone(), references.clone()];

    let restored = deserialize_v2_parts(&dtype, len, &metadata, &both)?;
    assert_arrays_eq!(restored, array, &mut ctx);

    // Readers ignore children that follow the ones the contract declares.
    let with_trailing = [encoded.clone(), references.clone(), encoded.clone()];
    let restored = deserialize_v2_parts(&dtype, len, &metadata, &with_trailing)?;
    assert_arrays_eq!(restored, array, &mut ctx);

    // The references child is missing.
    assert!(deserialize_v2_parts(&dtype, len, &metadata, std::slice::from_ref(&encoded)).is_err());

    // Buffers are not part of the contract.
    let buffer = BufferHandle::new_host(ByteBuffer::from(vec![0u8; 8]));
    assert!(
        FoRPlugin
            .deserialize(
                ArrayDeserialization::new(
                    for_v2_id(),
                    &dtype,
                    len,
                    &metadata.encode_to_vec(),
                    std::slice::from_ref(&buffer),
                    &both,
                ),
                &SESSION,
            )
            .is_err()
    );

    // A non-integer dtype.
    let float = DType::Primitive(PType::F32, dtype.nullability());
    assert!(deserialize_v2_parts(&float, len, &metadata, &both).is_err());

    // The offset must lie in the first chunk.
    for offset in [1024, u32::MAX] {
        assert!(deserialize_v2_parts(&dtype, len, &v2::FoRV2Metadata { offset }, &both).is_err());
    }

    // The number of chunks spanned by `offset + len` must not overflow.
    let offset_one = v2::FoRV2Metadata { offset: 1 };
    assert!(deserialize_v2_parts(&dtype, usize::MAX, &offset_one, &both).is_err());
    Ok(())
}
