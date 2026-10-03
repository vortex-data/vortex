// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::LazyLock;

use rstest::rstest;
use vortex_array::ArrayContext;
use vortex_array::ArrayRef;
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

use crate::VarBitPacked;
use crate::VarBitPackedArrayExt;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = vortex_array::array_session();
    crate::initialize(&session);
    session
});

/// Residuals whose range grows chunk by chunk: 0 bits, then 1, 2, … up to 9.
fn drifting() -> PrimitiveArray {
    PrimitiveArray::from_iter((0..10_000u32).map(|i| {
        let chunk = i / 1024;
        if chunk == 0 { 0 } else { (i * 7919) % (1 << chunk) }
    }))
}

fn nullable() -> PrimitiveArray {
    PrimitiveArray::from_option_iter((0..4000u16).map(|i| (i % 11 != 0).then_some(i % 300)))
}

fn signed() -> PrimitiveArray {
    PrimitiveArray::from_iter((0..3000i64).map(|i| if i < 2048 { i % 100 } else { -i }))
}

fn extremes() -> PrimitiveArray {
    PrimitiveArray::from_iter([u64::MAX, 0, 1, u64::MAX - 1])
}

#[rstest]
#[case::drifting(drifting())]
#[case::nullable(nullable())]
#[case::signed(signed())]
#[case::extremes(extremes())]
#[case::small(PrimitiveArray::from_iter((0..3000u32).map(|i| (i % 256) as u8)))]
#[case::empty(PrimitiveArray::from_iter(Vec::<i32>::new()))]
#[case::all_null(PrimitiveArray::from_option_iter([None::<u16>, None, None]))]
fn roundtrip(#[case] values: PrimitiveArray) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let encoded = VarBitPacked::encode(&values, &mut ctx)?;
    assert_arrays_eq!(encoded, values, &mut ctx);
    Ok(())
}

#[test]
fn widths_follow_each_chunk() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let encoded = VarBitPacked::encode(&drifting(), &mut ctx)?;
    assert_eq!(encoded.chunk_widths(), &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    Ok(())
}

#[rstest]
#[case::start(0..100)]
#[case::within_chunk(1500..1600)]
#[case::across_chunks(1000..3500)]
#[case::to_end(2900..10_000)]
#[case::empty(2048..2048)]
fn slice(#[case] range: std::ops::Range<usize>) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let values = drifting();
    let expected = values.clone().into_array().slice(range.clone())?;
    let sliced = VarBitPacked::encode(&values, &mut ctx)?
        .into_array()
        .slice(range.clone())?;
    assert_arrays_eq!(sliced, expected, &mut ctx);
    // Slicing twice exercises the offset of an already-sliced array.
    let inner = range.len().min(7)..range.len() / 2;
    assert_arrays_eq!(sliced.slice(inner.clone())?, expected.slice(inner)?, &mut ctx);
    Ok(())
}

#[rstest]
#[case::drifting(drifting())]
#[case::nullable(nullable())]
#[case::signed(signed())]
fn scalar_at(#[case] values: PrimitiveArray) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let expected = values.clone().into_array();
    let encoded = VarBitPacked::encode(&values, &mut ctx)?.into_array();
    for index in [0, 1, 11, 1023, 1024, 2047, 2048, 2999] {
        assert_eq!(
            encoded.execute_scalar(index, &mut ctx)?,
            expected.execute_scalar(index, &mut ctx)?
        );
    }
    let sliced = encoded.slice(1000..2500)?;
    let expected = expected.slice(1000..2500)?;
    for index in [0, 23, 24, 1047, 1048, 1499] {
        assert_eq!(
            sliced.execute_scalar(index, &mut ctx)?,
            expected.execute_scalar(index, &mut ctx)?
        );
    }
    Ok(())
}

fn serde_roundtrip(array: &ArrayRef) -> VortexResult<ArrayRef> {
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
#[case::drifting(drifting())]
#[case::nullable(nullable())]
fn serde(#[case] values: PrimitiveArray) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let encoded = VarBitPacked::encode(&values, &mut ctx)?.into_array();
    let read = serde_roundtrip(&encoded)?;
    assert_eq!(read.encoding_id(), encoded.encoding_id());
    assert_arrays_eq!(read, values, &mut ctx);
    let sliced = encoded.slice(1500..2600)?;
    assert_arrays_eq!(serde_roundtrip(&sliced)?, sliced, &mut ctx);
    Ok(())
}
