// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Public selection contracts, independent of optional PCO metadata or execution strategy.

use rstest::rstest;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PiecewiseSequenceArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::assert_arrays_eq;
use vortex_array::validity::Validity;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::Pco;
use crate::initialize;

#[rstest]
fn selected_values_and_nulls(#[values(false, true)] nullable: bool) -> VortexResult<()> {
    let session = VortexSession::empty();
    initialize(&session);
    let mut ctx = session.create_execution_ctx();
    let input = PrimitiveArray::new(
        (0..4096i64).map(|i| i * 19).collect::<Vec<_>>(),
        if nullable {
            Validity::from_iter((0..4096).map(|i| i % 7 != 0))
        } else {
            Validity::NonNullable
        },
    );
    let compressed = Pco::from_primitive(input.as_view(), 3, 31, &mut ctx)?.into_array();
    let source = compressed.slice(17..4001)?.slice(13..3900)?;
    let indices = PrimitiveArray::from_option_iter([
        Some(3000u64),
        Some(0),
        None,
        Some(127),
        Some(1),
        Some(3000),
        Some(128),
        None,
        Some(5),
    ])
    .into_array();
    let expected = PrimitiveArray::from_option_iter([
        Some(57570i64),
        Some(570),
        None,
        Some(2983),
        Some(589),
        Some(57570),
        Some(3002),
        None,
        if nullable { None } else { Some(665) },
    ]);
    assert_arrays_eq!(source.take(indices)?, expected, &mut ctx);
    let expected_probes = PrimitiveArray::new(
        vec![57570i64, 570, 2983, 589, 57570, 3002, 570, 665],
        if nullable {
            Validity::from_iter([true, true, true, true, true, true, true, false])
        } else {
            Validity::NonNullable
        },
    );
    let mut probe = source.repeated_probe();
    for (position, index) in [3000, 0, 127, 1, 3000, 128, 0, 5].into_iter().enumerate() {
        assert_eq!(
            probe.execute_scalar(index, &mut ctx)?,
            expected_probes.execute_scalar(position, &mut ctx)?,
        );
    }
    Ok(())
}

#[rstest]
#[case(vec![0u64, 40, 0, 70], vec![15u64, 20, 3, 0], vec![1u64, 1, 0, 1])]
#[case(vec![100u64, 0, 20], vec![20u64, 20, 20], vec![1u64, 1, 2])]
fn compact_ranges(
    #[case] starts: Vec<u64>,
    #[case] lengths: Vec<u64>,
    #[case] strides: Vec<u64>,
) -> VortexResult<()> {
    let session = VortexSession::empty();
    initialize(&session);
    let mut ctx = session.create_execution_ctx();
    let input =
        PrimitiveArray::from_option_iter((0..256i64).map(|i| (i % 5 != 0).then_some(i * i)));
    let compressed = Pco::from_primitive(input.as_view(), 3, 7, &mut ctx)?.into_array();
    let expected_indices = starts
        .iter()
        .zip(&lengths)
        .zip(&strides)
        .flat_map(|((&s, &n), &k)| (0..n).map(move |i| s + i * k))
        .collect::<Vec<_>>();
    let indices = PiecewiseSequenceArray::try_new(
        PrimitiveArray::from_iter(starts).into_array(),
        PrimitiveArray::from_iter(lengths).into_array(),
        PrimitiveArray::from_iter(strides).into_array(),
        expected_indices.len(),
    )?
    .into_array();
    assert_arrays_eq!(
        compressed.take(indices)?,
        PrimitiveArray::from_option_iter(
            expected_indices
                .into_iter()
                .map(|i| { (i % 5 != 0).then(|| i64::try_from(i * i).unwrap()) })
        ),
        &mut ctx
    );
    Ok(())
}

#[rstest]
fn null_runs_and_empty(#[values(false, true)] all_null: bool) -> VortexResult<()> {
    let session = VortexSession::empty();
    initialize(&session);
    let mut ctx = session.create_execution_ctx();
    let input = PrimitiveArray::from_option_iter(
        (0..65536i32).map(|i| (!all_null && !(100..65000).contains(&i)).then_some(i)),
    );
    let compressed = Pco::from_primitive(input.as_view(), 3, 128, &mut ctx)?.into_array();
    let indices =
        PrimitiveArray::from_option_iter([Some(0u64), None, Some(65001), Some(30000), Some(0)])
            .into_array();
    assert_arrays_eq!(
        compressed.take(indices)?,
        PrimitiveArray::from_option_iter(if all_null {
            [None, None, None, None, None]
        } else {
            [Some(0i32), None, Some(65001), None, Some(0)]
        }),
        &mut ctx
    );
    assert_arrays_eq!(
        compressed.slice(99..65002)?,
        input.slice(99..65002)?,
        &mut ctx
    );
    assert_arrays_eq!(
        compressed.take(PrimitiveArray::from_iter([] as [u64; 0]).into_array())?,
        PrimitiveArray::from_option_iter([] as [Option<i32>; 0]),
        &mut ctx
    );
    Ok(())
}

#[test]
fn invalid_indices_under_nulls_are_ignored() -> VortexResult<()> {
    let session = VortexSession::empty();
    initialize(&session);
    let mut ctx = session.create_execution_ctx();
    let input = PrimitiveArray::from_iter([11i32, 22, 33]);
    let compressed = Pco::from_primitive(input.as_view(), 3, 2, &mut ctx)?.into_array();
    let indices = PrimitiveArray::new(
        vec![0i64, i64::MIN, i64::MAX, 2],
        Validity::from_iter([true, false, false, true]),
    )
    .into_array();
    assert_arrays_eq!(
        compressed.take(indices)?,
        PrimitiveArray::from_option_iter([Some(11i32), None, None, Some(33)]),
        &mut ctx
    );
    Ok(())
}

#[test]
fn nonnullable_selection_crosses_chunks() -> VortexResult<()> {
    let session = VortexSession::empty();
    initialize(&session);
    let mut ctx = session.create_execution_ctx();
    let input = PrimitiveArray::from_iter((0..300000i64).map(|i| i * 3 + 1));
    let compressed = Pco::from_primitive(input.as_view(), 3, 127, &mut ctx)?.into_array();
    let indices = PrimitiveArray::from_iter([299999u64, 262144, 0, 262143, 262144]).into_array();
    assert_arrays_eq!(
        compressed.take(indices)?,
        PrimitiveArray::from_iter([899998i64, 786433, 1, 786430, 786433]),
        &mut ctx
    );
    Ok(())
}

#[test]
fn nullable_selection_crosses_chunks() -> VortexResult<()> {
    let session = VortexSession::empty();
    initialize(&session);
    let mut ctx = session.create_execution_ctx();
    let input = PrimitiveArray::from_option_iter(
        (0..400_000i64).map(|row| (row % 3 != 0).then_some(row * 3 + 1)),
    );
    let compressed = Pco::from_primitive(input.as_view(), 3, 127, &mut ctx)?.into_array();
    // The dense-value chunk boundary falls between logical rows 393215 and 393217.
    let indices =
        PrimitiveArray::from_iter([393217u64, 393215, 393216, 393214, 393218]).into_array();
    assert_arrays_eq!(
        compressed.take(indices)?,
        PrimitiveArray::from_option_iter([
            Some(1_179_652i64),
            Some(1_179_646),
            None,
            Some(1_179_643),
            Some(1_179_655),
        ]),
        &mut ctx
    );
    let sliced = compressed.slice(393210..393225)?;
    assert_arrays_eq!(
        sliced.clone(),
        PrimitiveArray::from_option_iter(
            (393210..393225i64).map(|row| (row % 3 != 0).then_some(row * 3 + 1)),
        ),
        &mut ctx
    );
    let mut probe = sliced.repeated_probe();
    for (index, expected) in [
        (7, Some(1_179_652i64)),
        (5, Some(1_179_646)),
        (6, None),
        (4, Some(1_179_643)),
        (8, Some(1_179_655)),
        (7, Some(1_179_652)),
    ] {
        let expected = PrimitiveArray::from_option_iter([expected]);
        assert_eq!(
            probe.execute_scalar(index, &mut ctx)?,
            expected.execute_scalar(0, &mut ctx)?
        );
    }
    Ok(())
}
