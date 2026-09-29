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
use vortex_array::scalar::ScalarValue;
use vortex_array::serde::SerializeOptions;
use vortex_array::serde::SerializedArray;
use vortex_array::session::ArraySessionExt;
use vortex_buffer::ByteBufferMut;
use vortex_session::registry::ReadContext;

use super::*;
use crate::FoRArray;
use crate::r#for::array::FoRArraySlotsExt;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = vortex_array::array_session();
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
        ScalarValue::to_proto_bytes::<Vec<u8>>(array.constant_reference().unwrap().value())
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
    let array = chunked()?.into_array().slice(range.clone())?;
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
    assert!(read.as_::<FoR>().constant_reference().is_none());
    assert_arrays_eq!(read, array, &mut SESSION.create_execution_ctx());
    Ok(())
}

fn deserialize_v2_parts(
    metadata: &v2::FoRV2Metadata,
    children: &[ArrayRef],
) -> VortexResult<ArrayRef> {
    let dtype = children[0].dtype().clone();
    FoRPlugin.deserialize(
        ArrayDeserialization::new(
            for_v2_id(),
            &dtype,
            children[0].len(),
            &metadata.encode_to_vec(),
            &[],
            &children,
        ),
        &SESSION,
    )
}

#[test]
fn v2_rejects_malformed_parts() -> VortexResult<()> {
    let array = chunked()?;
    let encoded = array.encoded().clone();
    let references = array.references().clone();
    let metadata = v2::FoRV2Metadata { offset: 0 };

    assert!(deserialize_v2_parts(&metadata, &[encoded.clone(), references.clone()]).is_ok());
    assert!(deserialize_v2_parts(&metadata, std::slice::from_ref(&encoded)).is_err());
    assert!(
        deserialize_v2_parts(&v2::FoRV2Metadata { offset: 1024 }, &[encoded, references]).is_err()
    );
    Ok(())
}

#[test]
fn vtable_serde_requires_plugin() -> VortexResult<()> {
    let array = FoR::encode(
        PrimitiveArray::from_iter([1u8, 2, 3]),
        &mut SESSION.create_execution_ctx(),
    )?;
    let session = vortex_array::array_session();
    session.arrays().register(FoR);
    assert!(session.array_serialize(&array.into_array()).is_err());
    Ok(())
}
