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
use vortex_array::dtype::DType;
use vortex_array::dtype::PType;
use vortex_array::optimizer::ArrayOptimizer;
use vortex_array::scalar::ScalarValue;
use vortex_array::serde::SerializeOptions;
use vortex_array::serde::SerializedArray;
use vortex_array::session::ArraySessionExt;
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
    metadata: &v2::FoRV2Metadata,
    children: &[ArrayRef],
) -> VortexResult<ArrayRef> {
    FoRPlugin.deserialize(
        ArrayDeserialization::new(
            for_v2_id(),
            dtype,
            children[0].len(),
            &metadata.encode_to_vec(),
            &[],
            &children,
        ),
        &SESSION,
    )
}

#[test]
fn v2_narrowed_bitpacked_roundtrip() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    // i64 values whose per-chunk span fits in u16 (triggers i64 -> u16 narrowing).
    let raw = PrimitiveArray::from_option_iter((0..3500i64).map(|i| {
        (i % 17 != 0).then_some(1_000_000_000 + (i / 1024) * 10_000_000 + (i % 500) * 100)
    }));
    let encoded_for = FoR::encode_chunked(raw.clone(), &mut ctx)?;
    assert_eq!(encoded_for.encoded().dtype().as_ptype(), PType::U16);
    // Bit-width 10 leaves values >= 1024 as u16 patches.
    let packed = BitPackedData::encode(encoded_for.encoded(), 10, &mut ctx)?;
    assert!(packed.patches().is_some());
    let refs = blocked(&encoded_for)?;
    let array = FoR::try_new_chunked(packed.into_array(), refs, 0)?.into_array();

    // Whole patched array serializes cleanly and round-trips bit-exactly.
    let serialization = SESSION
        .array_serialize(&array)?
        .ok_or_else(|| vortex_err!("FoR must serialize"))?;
    assert_eq!(serialization.serialized_id, for_v2_id());
    let meta = v2::FoRV2Metadata::decode(serialization.metadata.as_slice())?;
    assert_eq!(meta.offset, 0);
    assert_eq!(meta.encoded_bytes, 2);
    let read = roundtrip(&array)?;
    assert_arrays_eq!(read, raw, &mut ctx);

    // Sliced patched arrays decode and probe accurately at non-zero offsets.
    for range in [100..2600, 1024..2048] {
        let sliced = array.slice(range.clone())?.optimize_ctx(&SESSION)?;
        assert_arrays_eq!(sliced, raw.clone().into_array().slice(range)?, &mut ctx);
    }

    // Sliced unpatched narrowed BitPacked serializes and round-trips at non-zero offsets.
    let raw_unpatched = PrimitiveArray::from_option_iter((0..3500i64).map(|i| {
        (i % 17 != 0).then_some(1_000_000_000 + (i / 1024) * 10_000_000 + (i % 500) * 60)
    }));
    let encoded_unpatched = FoR::encode_chunked(raw_unpatched.clone(), &mut ctx)?;
    assert_eq!(encoded_unpatched.encoded().dtype().as_ptype(), PType::U16);
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
        let meta = v2::FoRV2Metadata::decode(serialization.metadata.as_slice())?;
        assert_eq!(meta.offset, u32::try_from(range.start % 1024)?);
        assert_eq!(meta.encoded_bytes, 2);
        let read = roundtrip(&sliced)?;
        assert_arrays_eq!(
            read,
            raw_unpatched.clone().into_array().slice(range)?,
            &mut ctx
        );
    }
    Ok(())
}

#[test]
fn v2_rejects_malformed_parts() -> VortexResult<()> {
    let array = chunked()?;
    let dtype = array.dtype().clone();
    let encoded = array.encoded().clone();
    let references = blocked(&array)?;
    let valid_bytes = u32::try_from(encoded.dtype().as_ptype().byte_width())?;
    let metadata = v2::FoRV2Metadata {
        offset: 0,
        encoded_bytes: valid_bytes,
    };

    assert!(
        deserialize_v2_parts(&dtype, &metadata, &[encoded.clone(), references.clone()]).is_ok()
    );
    assert!(deserialize_v2_parts(&dtype, &metadata, std::slice::from_ref(&encoded)).is_err());
    assert!(
        deserialize_v2_parts(
            &dtype,
            &v2::FoRV2Metadata {
                offset: 1024,
                encoded_bytes: valid_bytes,
            },
            &[encoded.clone(), references.clone()],
        )
        .is_err()
    );
    // Reject invalid encoded_bytes values and non-narrower widths.
    assert!(
        deserialize_v2_parts(
            &dtype,
            &v2::FoRV2Metadata {
                offset: 0,
                encoded_bytes: 3,
            },
            &[encoded.clone(), references.clone()],
        )
        .is_err()
    );
    assert!(
        deserialize_v2_parts(
            &dtype,
            &v2::FoRV2Metadata {
                offset: 0,
                encoded_bytes: 4,
            },
            &[encoded, references],
        )
        .is_err()
    );
    Ok(())
}

#[test]
fn v2_unnarrowed_and_legacy_metadata_roundtrip() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    // Spread > u16::MAX in a u32 array keeps same-width residuals (`encoded_bytes == 0`).
    let raw = PrimitiveArray::from_iter((0..2048u32).map(|i| {
        if i % 1024 == 0 {
            (i / 1024) * 500_000
        } else {
            (i / 1024) * 500_000 + 100_000 + (i % 1024)
        }
    }));
    let array = FoR::encode_chunked(raw.clone(), &mut ctx)?;
    assert_eq!(array.encoded().dtype().as_ptype(), PType::U32);

    let serialization = SESSION
        .array_serialize(array.as_array())?
        .ok_or_else(|| vortex_err!("FoR must serialize"))?;
    let meta = v2::FoRV2Metadata::decode(serialization.metadata.as_slice())?;
    assert_eq!(meta.encoded_bytes, 0);

    let read = roundtrip(array.as_array())?;
    assert_arrays_eq!(read, raw, &mut ctx);

    // Pre-narrowing metadata had only tag 1 (`offset`), which decodes with `encoded_bytes == 0`.
    let legacy = deserialize_v2_parts(
        array.dtype(),
        &v2::FoRV2Metadata {
            offset: 0,
            encoded_bytes: 0,
        },
        &[array.encoded().clone(), blocked(&array)?],
    )?;
    assert_arrays_eq!(legacy, raw, &mut ctx);
    Ok(())
}
