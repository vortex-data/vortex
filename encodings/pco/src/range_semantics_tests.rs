// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Public slice and compact-range behavior, independent of compressed metadata.

use std::iter::repeat_n;
use std::sync::LazyLock;

use rstest::rstest;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PiecewiseSequenceArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::assert_arrays_eq;
use vortex_array::session::ArraySessionExt;
use vortex_array::validity::Validity;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::Pco;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::empty();
    session.arrays().register(Pco);
    session
});

#[rstest]
#[case(vec![0u64, 40, 0, 70], vec![15u64, 20, 3, 0], vec![1u64, 1, 0, 1])]
#[case(vec![100u64, 0, 20], vec![20u64, 20, 20], vec![1u64, 1, 2])]
#[case(vec![99u64, 0], vec![0u64, 0], vec![u64::MAX, 0])]
fn piecewise_ranges(
    #[case] starts: Vec<u64>,
    #[case] lengths: Vec<u64>,
    #[case] strides: Vec<u64>,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let input =
        PrimitiveArray::from_option_iter((0..256i64).map(|i| (i % 5 != 0).then_some(i * i)));
    let encoded = Pco::from_primitive(input.as_view(), 3, 7, &mut ctx)?.into_array();
    let mut expected_indices = Vec::new();
    for ((&start, &length), &stride) in starts.iter().zip(&lengths).zip(&strides) {
        expected_indices.extend((0..length).map(|i| start + i * stride));
    }
    let indices = PiecewiseSequenceArray::try_new(
        PrimitiveArray::from_iter(starts).into_array(),
        PrimitiveArray::from_iter(lengths).into_array(),
        PrimitiveArray::from_iter(strides).into_array(),
        expected_indices.len(),
    )?
    .into_array();
    assert_arrays_eq!(
        encoded.take(indices)?,
        PrimitiveArray::from_option_iter(
            expected_indices
                .into_iter()
                .map(|row| { (row % 5 != 0).then_some((row * row) as i64) })
        ),
        &mut ctx
    );
    Ok(())
}

#[test]
fn sliced_piecewise_coordinates_are_visible_rows() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let input =
        PrimitiveArray::from_option_iter((0..256i64).map(|i| (i % 5 != 0).then_some(i * i)));
    let encoded = Pco::from_primitive(input.as_view(), 3, 7, &mut ctx)?.into_array();
    let indices = PiecewiseSequenceArray::try_new(
        PrimitiveArray::from_iter([0u64, 40, 3]).into_array(),
        PrimitiveArray::from_iter([15u64, 20, 4]).into_array(),
        PrimitiveArray::from_iter([1u64, 1, 0]).into_array(),
        39,
    )?
    .into_array();
    let expected =
        PrimitiveArray::from_option_iter((0..15).chain(40..60).chain(repeat_n(3, 4)).map(|i| {
            let row = i + 17i64;
            (row % 5 != 0).then_some(row * row)
        }));
    assert_arrays_eq!(encoded.slice(17..200)?.take(indices)?, expected, &mut ctx);
    Ok(())
}

#[test]
fn large_public_compressor_selections() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let len = 262_401usize;
    let input = PrimitiveArray::from_iter(
        (0..len).map(|i| u32::try_from(i).vortex_expect("fixture fits u32")),
    );
    let encoded = Pco::from_primitive(input.as_view(), 0, 4096, &mut ctx)?.into_array();
    let indices = PrimitiveArray::from_iter([
        0u32,
        900,
        260,
        3000,
        u32::try_from(len - 1).vortex_expect("fixture fits u32"),
        4095,
        900,
    ])
    .into_array();
    assert_arrays_eq!(
        encoded.take(indices)?,
        PrimitiveArray::from_iter([0u32, 900, 260, 3000, 262400, 4095, 900]),
        &mut ctx
    );
    assert_arrays_eq!(
        encoded.slice(len - 1000..len - 1)?,
        PrimitiveArray::from_iter(
            (len - 1000..len - 1).map(|row| u32::try_from(row).vortex_expect("fixture fits u32"))
        ),
        &mut ctx
    );
    Ok(())
}

#[test]
fn mixed_fragments_narrow_slices_and_compact_repeats() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let input = PrimitiveArray::from_option_iter(
        (0..32768i64).map(|row| (row % 5 != 0).then_some(row * 11)),
    );
    let encoded = Pco::from_primitive(input.as_view(), 3, 1024, &mut ctx)?.into_array();
    let mixed = PrimitiveArray::from_iter(
        (0..4096u64).map(|i| if i < 2 { i } else { (i * 104729) % 32768 }),
    )
    .into_array();
    assert_arrays_eq!(
        encoded.take(mixed)?,
        PrimitiveArray::from_option_iter((0..4096u64).map(|i| {
            let row = if i < 2 { i } else { (i * 104729) % 32768 };
            (row % 5 != 0).then_some(row as i64 * 11)
        })),
        &mut ctx
    );
    let repeated =
        PrimitiveArray::from_iter((0..100000u64).map(|i| if i % 2 == 0 { 0 } else { 127 }))
            .into_array();
    assert_arrays_eq!(
        encoded.slice(30000..30128)?.take(repeated)?,
        PrimitiveArray::from_option_iter((0..100000).map(|i| {
            if i % 2 == 0 {
                None
            } else {
                Some(30127i64 * 11)
            }
        })),
        &mut ctx
    );
    let compact = PiecewiseSequenceArray::try_new(
        PrimitiveArray::from_iter([1u64, 5]).into_array(),
        PrimitiveArray::from_iter([500000u64, 500000]).into_array(),
        PrimitiveArray::from_iter([0u64, 0]).into_array(),
        1000000,
    )?
    .into_array();
    let expected = PrimitiveArray::from_option_iter(
        repeat_n(Some(11i64), 500000).chain(repeat_n(None, 500000)),
    );
    assert_arrays_eq!(encoded.take(compact)?, expected, &mut ctx);
    let nulls = PrimitiveArray::new(vec![0i64; 100000], Validity::AllInvalid);
    let encoded_nulls = Pco::from_primitive(nulls.as_view(), 3, 1024, &mut ctx)?.into_array();
    let two = PrimitiveArray::from_iter([0u64, 99999]).into_array();
    assert_arrays_eq!(
        encoded_nulls.take(two)?,
        PrimitiveArray::from_option_iter([None::<i64>, None]),
        &mut ctx
    );
    Ok(())
}

#[rstest]
fn sliced_values_preserve_float_bits_and_validity(
    #[values(false, true)] dense: bool,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let bits = [
        0u64,
        1u64 << 63,
        0x7ff8000000000042,
        0x7ff0000000000000,
        0xfff0000000000000,
        0x3ff4000000000000,
    ];
    let values = (0..2048)
        .map(|row| f64::from_bits(bits[row % bits.len()]))
        .collect::<Vec<_>>();
    let validity = if dense {
        Validity::NonNullable
    } else {
        Validity::from_iter((0..2048).map(|row| row % 7 != 0))
    };
    let input = PrimitiveArray::new(values.clone(), validity);
    let encoded = Pco::from_primitive(input.as_view(), 3, 31, &mut ctx)?.into_array();
    for range in [1..2047, 89..129, 1000..1001] {
        let actual = encoded
            .slice(range.clone())?
            .execute::<PrimitiveArray>(&mut ctx)?;
        assert_eq!(actual.dtype(), input.dtype());
        let mask = actual.validity()?.execute_mask(actual.len(), &mut ctx)?;
        for (offset, &value) in actual.as_slice::<f64>().iter().enumerate() {
            let row = range.start + offset;
            assert_eq!(mask.value(offset), dense || row % 7 != 0);
            if mask.value(offset) {
                assert_eq!(value.to_bits(), values[row].to_bits());
            }
        }
    }
    Ok(())
}
