// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Public take behavior, independent of compressed metadata and execution strategy.

use std::iter::empty;
use std::sync::LazyLock;

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PiecewiseSequenceArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::assert_arrays_eq;
use vortex_array::validity::Validity;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::Pco;
use crate::initialize;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::empty();
    initialize(&session);
    session
});

#[test]
fn reordered_nullable_take() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let input =
        PrimitiveArray::from_option_iter((0..4096i32).map(|i| (i % 7 != 0).then_some(i * 19)));
    let encoded = Pco::from_primitive(input.as_view(), 3, 31, &mut ctx)?.into_array();
    let source = encoded.slice(17..4001)?.slice(13..3900)?;
    let expected_probes = PrimitiveArray::from_option_iter([
        Some(57570i32),
        Some(570),
        Some(2983),
        Some(589),
        Some(57570),
        Some(3002),
        Some(19570),
        Some(570),
    ]);
    let mut probe = source.repeated_probe();
    for (position, index) in [3000, 0, 127, 1, 3000, 128, 1000, 0]
        .into_iter()
        .enumerate()
    {
        assert_eq!(
            probe.execute_scalar(index, &mut ctx)?,
            expected_probes.execute_scalar(position, &mut ctx)?
        );
    }
    let indices = PrimitiveArray::from_option_iter([
        Some(3000u32),
        Some(0),
        None,
        Some(127),
        Some(1),
        Some(3000),
        Some(128),
        None,
        Some(1000),
    ])
    .into_array();
    assert_arrays_eq!(
        source.take(indices)?,
        PrimitiveArray::from_option_iter([
            Some(57570i32),
            Some(570),
            None,
            Some(2983),
            Some(589),
            Some(57570),
            Some(3002),
            None,
            Some(19570),
        ]),
        &mut ctx
    );
    Ok(())
}

#[test]
fn all_null_and_long_null_runs() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    for input in [
        PrimitiveArray::new(vec![0i32; 1024], Validity::AllInvalid),
        PrimitiveArray::from_option_iter(
            (0..100_000).map(|i| [1, 90_000, 99_990].contains(&i).then_some(i)),
        ),
    ] {
        let encoded = Pco::from_primitive(input.as_view(), 3, 1, &mut ctx)?.into_array();
        let indices = PrimitiveArray::from_iter([0u32, 1, 100, 1000, 2, 1]).into_array();
        assert_arrays_eq!(
            encoded.take(indices)?,
            PrimitiveArray::from_option_iter([0i32, 1, 100, 1000, 2, 1].map(|row| {
                if input.len() == 1024 {
                    None
                } else {
                    [1, 90_000, 99_990].contains(&row).then_some(row)
                }
            })),
            &mut ctx
        );
        assert_arrays_eq!(
            encoded.slice(1..1000)?,
            PrimitiveArray::from_option_iter((1..1000).map(|row| {
                if input.len() == 1024 {
                    None
                } else {
                    (row == 1).then_some(row)
                }
            })),
            &mut ctx
        );
    }
    Ok(())
}

#[test]
fn registry_only_session_preserves_take_semantics() -> VortexResult<()> {
    let session = VortexSession::empty();
    initialize(&session);
    let mut ctx = session.create_execution_ctx();
    let input = PrimitiveArray::from_option_iter([Some(5i32), None, Some(9), Some(2)]);
    let encoded = Pco::from_primitive(input.as_view(), 3, 1, &mut ctx)?.into_array();
    let indices: ArrayRef = PrimitiveArray::from_iter([3u32, 1, 0, 3]).into_array();
    assert_arrays_eq!(
        encoded.take(indices)?,
        PrimitiveArray::from_option_iter([Some(2i32), None, Some(5), Some(2)]),
        &mut ctx
    );
    Ok(())
}

#[test]
fn repeated_scattered_requests_preserve_order_and_nulls() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let values = (0..32768i64)
        .map(|row| (row % 7 != 0).then_some(row * 13))
        .collect::<Vec<_>>();
    let input = PrimitiveArray::from_option_iter(values.iter().copied());
    let encoded = Pco::from_primitive(input.as_view(), 3, 1024, &mut ctx)?.into_array();
    let requests = (0..1024usize)
        .map(|i| (i % 11 != 0).then_some((i * 97) % 16))
        .collect::<Vec<_>>();
    let indices = PrimitiveArray::from_option_iter(
        requests
            .iter()
            .map(|row| row.map(|row| u64::try_from(row).vortex_expect("small index"))),
    )
    .into_array();
    let expected = PrimitiveArray::from_option_iter(
        requests.iter().map(|row| row.and_then(|row| values[row])),
    );
    assert_arrays_eq!(encoded.take(indices)?, expected, &mut ctx);
    Ok(())
}

#[test]
fn empty_take_dtype_and_mostly_null_request_storage() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let input = PrimitiveArray::from_iter(0..32768i64);
    let encoded = Pco::from_primitive(input.as_view(), 3, 1024, &mut ctx)?.into_array();
    for source in [encoded.clone(), encoded.slice(0..0)?] {
        for indices in [
            PrimitiveArray::from_iter(empty::<u64>()).into_array(),
            PrimitiveArray::from_option_iter(empty::<Option<u64>>()).into_array(),
        ] {
            let result = source
                .take(indices.clone())?
                .execute::<PrimitiveArray>(&mut ctx)?;
            let expected = if indices.dtype().is_nullable() {
                PrimitiveArray::from_option_iter(empty::<Option<i64>>())
            } else {
                PrimitiveArray::from_iter(empty::<i64>())
            };
            assert_eq!(result.dtype(), expected.dtype());
            assert_arrays_eq!(result, expected, &mut ctx);
        }
    }
    let indices =
        PrimitiveArray::from_option_iter((0..200000).map(|i| (i == 12345).then_some(1u64)))
            .into_array();
    let result = encoded.take(indices)?.execute::<PrimitiveArray>(&mut ctx)?;
    let expected =
        PrimitiveArray::from_option_iter((0..200000).map(|i| (i == 12345).then_some(1i64)));
    assert_eq!(result.dtype(), expected.dtype());
    assert_arrays_eq!(result, expected, &mut ctx);
    Ok(())
}

#[test]
fn logical_null_indices_ignore_invalid_physical_values() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let input = PrimitiveArray::from_iter(0..128i64);
    let encoded = Pco::from_primitive(input.as_view(), 3, 7, &mut ctx)?.into_array();
    let indices = PrimitiveArray::new(
        vec![i64::MIN, 4, i64::MAX],
        Validity::from_iter([false, true, false]),
    )
    .into_array();
    let expected = PrimitiveArray::from_option_iter([None, Some(4i64), None]);
    assert_arrays_eq!(encoded.take(indices.clone())?, expected, &mut ctx);
    // Reuse the same immutable request to check repeated execution.
    assert_arrays_eq!(
        encoded.take(indices)?,
        PrimitiveArray::from_option_iter([None, Some(4i64), None]),
        &mut ctx
    );
    Ok(())
}

#[test]
fn contiguous_take_preserves_nullable_index_dtype() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let input = PrimitiveArray::from_iter(0..32768i64);
    let encoded = Pco::from_primitive(input.as_view(), 3, 1024, &mut ctx)?.into_array();
    let indices = PrimitiveArray::from_option_iter((1..32767u64).map(Some)).into_array();
    let expected = PrimitiveArray::from_option_iter((1..32767i64).map(Some));
    let result = encoded.take(indices)?.execute::<PrimitiveArray>(&mut ctx)?;
    assert_eq!(result.dtype(), expected.dtype());
    assert_arrays_eq!(result, expected, &mut ctx);
    let range = PiecewiseSequenceArray::try_new(
        PrimitiveArray::from_iter([1u64]).into_array(),
        PrimitiveArray::from_iter([32766u64]).into_array(),
        PrimitiveArray::from_iter([1u64]).into_array(),
        32766,
    )?
    .into_array();
    assert_arrays_eq!(
        encoded.take(range)?,
        PrimitiveArray::from_iter(1..32767i64),
        &mut ctx
    );
    Ok(())
}
