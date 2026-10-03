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
use vortex_array::compute::conformance::consistency::test_array_consistency;
use vortex_array::dtype::half::f16;
use vortex_array::serde::SerializeOptions;
use vortex_array::serde::SerializedArray;
use vortex_array::session::ArraySessionExt;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_session::VortexSession;
use vortex_session::registry::ReadContext;

use crate::Binned;
use crate::BinnedArray;
use crate::Config;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = vortex_array::array_session();
    session.arrays().register(Binned);
    session
});

fn encode(values: &PrimitiveArray) -> VortexResult<BinnedArray> {
    Binned::from_primitive(
        values.as_view(),
        &Config::default(),
        &mut SESSION.create_execution_ctx(),
    )
}

fn sample_arrays() -> Vec<PrimitiveArray> {
    vec![
        PrimitiveArray::from_iter((0..5000).map(|i| i64::from(i % 97) * 1000 - 3)),
        PrimitiveArray::from_iter((0..3000).map(|i| f64::from(i % 1000) / 100.0)),
        PrimitiveArray::from_iter((0..2500u32).map(|i| i.wrapping_mul(2_654_435_761))),
        PrimitiveArray::from_iter((0..2000).map(|i| f16::from_f32((i % 50) as f32 / 4.0))),
        PrimitiveArray::from_iter((0..1500).map(|i| (i % 7) as i8 - 3)),
        PrimitiveArray::from_option_iter((0..4000).map(|i| (i % 5 != 0).then_some(i * 3 + 1))),
        PrimitiveArray::from_option_iter((0..1100).map(|i| (i % 3 == 0).then_some(f32::from(i as u16) * 0.5))),
        PrimitiveArray::from_option_iter((0..10).map(|_| None::<u16>)),
        PrimitiveArray::from_iter(Vec::<i32>::new()),
        PrimitiveArray::from_iter([42.42f64]),
    ]
}

#[test]
fn roundtrip_and_slices() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    for values in sample_arrays() {
        let binned = encode(&values)?;
        assert_arrays_eq!(binned, values, &mut ctx);
        let len = values.len();
        for (start, stop) in [(0, len), (len / 3, len / 2), (len.saturating_sub(1), len)] {
            let expected = values.clone().into_array().slice(start..stop)?;
            let sliced = binned.clone().into_array().slice(start..stop)?;
            assert_arrays_eq!(sliced, expected, &mut ctx);
        }
    }
    Ok(())
}

#[test]
fn scalar_reads_match() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    for values in sample_arrays() {
        let binned = encode(&values)?.into_array();
        let values = values.into_array();
        for i in (0..values.len()).step_by(37) {
            assert_eq!(
                binned.execute_scalar(i, &mut ctx)?,
                values.execute_scalar(i, &mut ctx)?,
                "index {i}"
            );
        }
    }
    Ok(())
}

#[test]
fn serde_roundtrip() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    for values in sample_arrays() {
        let binned = encode(&values)?.into_array();
        let context = ArrayContext::empty();
        let bytes = binned
            .serialize(
                &context,
                &SESSION,
                &SerializeOptions {
                    offset: 0,
                    include_padding: true,
                },
            )?
            .into_iter()
            .flat_map(|x| x.into_iter())
            .collect::<BufferMut<u8>>()
            .freeze();
        let decoded: ArrayRef = SerializedArray::try_from(bytes)?.decode(
            values.dtype(),
            values.len(),
            &ReadContext::new(context.to_ids()),
            &SESSION,
        )?;
        assert_arrays_eq!(decoded, values, &mut ctx);
    }
    Ok(())
}

#[rstest]
#[case::i64(0)]
#[case::f64(1)]
#[case::u32(2)]
#[case::f16(3)]
#[case::i8(4)]
#[case::nullable_i32(5)]
#[case::nullable_f32(6)]
#[case::single(9)]
fn consistency(#[case] idx: usize) -> VortexResult<()> {
    let values = sample_arrays().swap_remove(idx);
    test_array_consistency(
        &encode(&values)?.into_array(),
        &mut SESSION.create_execution_ctx(),
    );
    Ok(())
}
