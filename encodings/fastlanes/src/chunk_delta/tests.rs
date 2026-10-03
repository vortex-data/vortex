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

use crate::BitPackedData;
use crate::ChunkDelta;
use crate::ChunkDeltaArraySlotsExt;
use crate::VarBitPacked;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = vortex_array::array_session();
    crate::initialize(&session);
    session
});

/// A temperature-like walk: small steps up and down.
fn walk() -> PrimitiveArray {
    let mut v = 150i32;
    PrimitiveArray::from_iter((0..5000).map(|i: i32| {
        v += (i * 7919 % 7) - 3;
        v
    }))
}

fn nullable() -> PrimitiveArray {
    PrimitiveArray::from_option_iter((0..4000u16).map(|i| (i % 11 != 0).then_some(1000 + i % 37)))
}

fn extremes() -> PrimitiveArray {
    PrimitiveArray::from_iter([i64::MIN, i64::MAX, 0, -1, 1, i64::MIN + 1, i64::MAX - 1])
}

fn unsigned_extremes() -> PrimitiveArray {
    PrimitiveArray::from_iter((0..2100u64).map(|i| if i % 2 == 0 { u64::MAX - i } else { i }))
}

fn cases() -> Vec<PrimitiveArray> {
    vec![walk(), nullable(), extremes(), unsigned_extremes()]
}

/// The array with its deltas left primitive, per-chunk bit-packed, and bit-packed with and
/// without patches.
fn variants(values: &PrimitiveArray) -> VortexResult<Vec<ArrayRef>> {
    let mut ctx = SESSION.create_execution_ctx();
    let plain = ChunkDelta::encode(values, &mut ctx)?;
    let deltas = plain.deltas().clone().execute::<PrimitiveArray>(&mut ctx)?;
    let width = deltas.ptype().bit_width() as u8 - 1;
    let mut out = vec![plain.clone().into_array()];
    for packed in [
        VarBitPacked::encode(&deltas, &mut ctx)?.into_array(),
        BitPackedData::encode(&deltas.clone().into_array(), width, &mut ctx)?.into_array(),
        // Narrow enough that the larger deltas become patches.
        BitPackedData::encode(&deltas.clone().into_array(), 2, &mut ctx)?.into_array(),
    ] {
        out.push(
            ChunkDelta::try_new(
                packed,
                plain.bases().clone(),
                plain.mins().clone(),
                values.validity()?,
                values.len(),
                0,
            )?
            .into_array(),
        );
    }
    Ok(out)
}

#[rstest]
#[case::walk(walk())]
#[case::nullable(nullable())]
#[case::extremes(extremes())]
#[case::unsigned_extremes(unsigned_extremes())]
#[case::empty(PrimitiveArray::from_iter(Vec::<i64>::new()))]
#[case::all_null(PrimitiveArray::from_option_iter([None::<u8>, None, None]))]
fn roundtrip(#[case] values: PrimitiveArray) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let encoded = ChunkDelta::encode(&values, &mut ctx)?;
    assert_arrays_eq!(encoded, values, &mut ctx);
    Ok(())
}

#[test]
fn fused_and_sliced() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    for values in cases() {
        let expected = values.clone().into_array();
        let n = values.len();
        for array in variants(&values)? {
            assert_arrays_eq!(array, expected, &mut ctx);
            for range in [0..n.min(100), 7..n.min(900), n / 2..n, 1000.min(n)..n.min(2048)] {
                let sliced = array.slice(range.clone())?;
                assert_arrays_eq!(sliced, expected.slice(range.clone())?, &mut ctx);
                // Slicing twice exercises the offset of an already-sliced array.
                let inner = (range.len() / 4)..range.len() / 2;
                assert_arrays_eq!(
                    sliced.slice(inner.clone())?,
                    expected.slice(range.clone())?.slice(inner)?,
                    &mut ctx
                );
            }
        }
    }
    Ok(())
}

#[test]
fn scalar_at() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    for values in cases() {
        let expected = values.clone().into_array();
        let n = values.len();
        for array in variants(&values)? {
            for index in [0, 1, 5, 1023, 1024, 2047, 3999].into_iter().filter(|&i| i < n) {
                assert_eq!(
                    array.execute_scalar(index, &mut ctx)?,
                    expected.execute_scalar(index, &mut ctx)?
                );
            }
        }
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
#[case::walk(walk())]
#[case::nullable(nullable())]
fn serde(#[case] values: PrimitiveArray) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    // A slice of patched bit-packed deltas is a lazy slice, which does not serialize.
    for array in variants(&values)?.into_iter().take(3) {
        let read = serde_roundtrip(&array)?;
        assert_eq!(read.encoding_id(), array.encoding_id());
        assert_arrays_eq!(read, values, &mut ctx);
        let sliced = array.slice(1500..2600)?;
        assert_arrays_eq!(serde_roundtrip(&sliced)?, sliced, &mut ctx);
    }
    Ok(())
}
