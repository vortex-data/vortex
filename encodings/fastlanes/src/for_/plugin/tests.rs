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
use vortex_array::test_harness::check_metadata;
use vortex_buffer::ByteBuffer;
use vortex_buffer::ByteBufferMut;
use vortex_session::registry::ReadContext;

use super::*;
use crate::BitPackedArrayExt;
use crate::BitPackedData;
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

/// Blocked FoR (`u32` logical -> `u32` residuals), which serializes as `fastlanes.for.v2`.
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
    offset: u32,
    children: &[ArrayRef],
) -> VortexResult<ArrayRef> {
    FoRPlugin.deserialize(
        ArrayDeserialization::new(
            for_v2_id(),
            dtype,
            len,
            &v2::FoRV2Metadata { offset }.encode_to_vec(),
            &[],
            &children,
        ),
        &SESSION,
    )
}

#[test]
fn v2_bitpacked_roundtrip() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let raw = PrimitiveArray::from_option_iter((0..3500i64).map(|i| {
        (i % 17 != 0).then_some(1_000_000_000 + (i / 1024) * 10_000_000 + (i % 500) * 100)
    }));
    let encoded_for = FoR::encode_chunked(raw.clone(), &mut ctx)?;
    assert_eq!(encoded_for.encoded().dtype().as_ptype(), PType::I64);
    // Bit-width 10 leaves values >= 1024 as patches.
    let packed = BitPackedData::encode(encoded_for.encoded(), 10, &mut ctx)?;
    assert!(packed.patches().is_some());
    let refs = blocked(&encoded_for)?;
    let array = FoR::try_new_chunked(packed.into_array(), refs, 0)?.into_array();

    // Whole patched array serializes as `fastlanes.for.v2` and round-trips bit-exactly.
    let serialization = SESSION
        .array_serialize(&array)?
        .ok_or_else(|| vortex_err!("FoR must serialize"))?;
    assert_eq!(serialization.serialized_id, for_v2_id());
    let meta = v2::FoRV2Metadata::decode(serialization.metadata.as_slice())?;
    assert_eq!(meta.offset, 0);
    let read = roundtrip(&array)?;
    assert_arrays_eq!(read, raw, &mut ctx);

    // Sliced patched arrays decode and probe accurately at non-zero offsets.
    for range in [100..2600, 1024..2048] {
        let sliced = array.slice(range.clone())?.optimize_ctx(&SESSION)?;
        assert_arrays_eq!(sliced, raw.clone().into_array().slice(range)?, &mut ctx);
    }

    // Sliced unpatched BitPacked serializes as `fastlanes.for.v2` and round-trips.
    let raw_unpatched = PrimitiveArray::from_option_iter((0..3500i64).map(|i| {
        (i % 17 != 0).then_some(1_000_000_000 + (i / 1024) * 10_000_000 + (i % 500) * 60)
    }));
    let encoded_unpatched = FoR::encode_chunked(raw_unpatched.clone(), &mut ctx)?;
    let packed_unpatched = BitPackedData::encode(encoded_unpatched.encoded(), 15, &mut ctx)?;
    assert!(packed_unpatched.patches().is_none());
    let unpatched_for = FoR::try_new_chunked(
        packed_unpatched.into_array(),
        blocked(&encoded_unpatched)?,
        0,
    )?
    .into_array();
    for range in [0..3500, 100..2600, 1024..2048] {
        let sliced = unpatched_for.slice(range.clone())?.optimize_ctx(&SESSION)?;
        let serialization = SESSION
            .array_serialize(&sliced)?
            .ok_or_else(|| vortex_err!("FoR must serialize"))?;
        assert_eq!(serialization.serialized_id, for_v2_id());
        let meta = v2::FoRV2Metadata::decode(serialization.metadata.as_slice())?;
        assert_eq!(meta.offset, u32::try_from(range.start % 1024)?);
        let read = roundtrip(&sliced)?;
        assert_arrays_eq!(
            read,
            raw_unpatched.clone().into_array().slice(range)?,
            &mut ctx
        );
    }
    Ok(())
}

fn assert_err_contains(result: VortexResult<ArrayRef>, expected: &str) {
    let err = result.expect_err("expected deserialization error");
    let msg = err.to_string();
    assert!(
        msg.contains(expected),
        "expected error containing {expected:?}, got {msg:?}"
    );
}

#[test]
fn v2_rejects_malformed_parts() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let array = chunked()?;
    let dtype = array.dtype().clone();
    let len = array.len();
    let encoded = array.encoded().clone();
    let references = blocked(&array)?;
    let both = [encoded.clone(), references.clone()];

    let restored = deserialize_v2_parts(&dtype, len, 0, &both)?;
    assert_arrays_eq!(restored, array, &mut ctx);

    // Readers ignore children that follow the ones the contract declares.
    let with_trailing = [encoded.clone(), references, encoded.clone()];
    let restored = deserialize_v2_parts(&dtype, len, 0, &with_trailing)?;
    assert_arrays_eq!(restored, array, &mut ctx);

    // The references child is missing.
    assert_err_contains(
        deserialize_v2_parts(&dtype, len, 0, std::slice::from_ref(&encoded)),
        "Expected at least 2 children",
    );

    // Buffers are not part of the contract.
    let buffer = BufferHandle::new_host(ByteBuffer::from(vec![0u8; 8]));
    assert_err_contains(
        FoRPlugin.deserialize(
            ArrayDeserialization::new(
                for_v2_id(),
                &dtype,
                len,
                &v2::FoRV2Metadata { offset: 0 }.encode_to_vec(),
                std::slice::from_ref(&buffer),
                &both,
            ),
            &SESSION,
        ),
        "expects 0 buffers",
    );

    // A non-integer dtype.
    let float = DType::Primitive(PType::F32, dtype.nullability());
    assert_err_contains(
        deserialize_v2_parts(&float, len, 0, &both),
        "requires an integer dtype",
    );

    // The offset must lie in the first chunk.
    for offset in [1024, u32::MAX] {
        assert_err_contains(
            deserialize_v2_parts(&dtype, len, offset, &both),
            "FoR offset must be less than",
        );
    }

    // The number of chunks spanned by `offset + len` must not overflow.
    assert_err_contains(
        deserialize_v2_parts(&dtype, usize::MAX, 1, &both),
        "overflowed usize",
    );
    Ok(())
}

#[cfg_attr(miri, ignore)]
#[test]
fn test_for_v2_metadata() {
    check_metadata(
        "for_v2.metadata",
        &v2::FoRV2Metadata { offset: 1023 }.encode_to_vec(),
    );
}
