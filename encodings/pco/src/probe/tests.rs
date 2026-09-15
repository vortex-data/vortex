// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cell::Cell;

use rstest::rstest;
use vortex_array::ArrayProbe;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::RepeatedArrayProbe;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::assert_arrays_eq;
use vortex_array::builders::builder_with_capacity_in;
use vortex_array::validity::Validity;
use vortex_error::VortexResult;
use vortex_runend::RunEnd;

use super::PcoProbeState;
use crate::Pco;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct ProbeCounts {
    initialized: usize,
    decoded: usize,
    dropped: usize,
}

thread_local! {
    static COUNTS: Cell<ProbeCounts> = Cell::default();
}

pub(super) struct TrackedProbeState;

impl Default for TrackedProbeState {
    fn default() -> Self {
        let counts = COUNTS.get();
        COUNTS.set(ProbeCounts {
            initialized: counts.initialized + 1,
            ..counts
        });
        Self
    }
}

impl TrackedProbeState {
    pub(super) fn record_decode(&self) {
        let counts = COUNTS.get();
        COUNTS.set(ProbeCounts {
            decoded: counts.decoded + 1,
            ..counts
        });
    }
}

impl Drop for TrackedProbeState {
    fn drop(&mut self) {
        let counts = COUNTS.get();
        COUNTS.set(ProbeCounts {
            dropped: counts.dropped + 1,
            ..counts
        });
    }
}

/// A one-off or retained probe over `array`, the retained one living in `retained`.
fn probe_for<'a>(
    array: &'a ArrayRef,
    retained: &'a mut Option<RepeatedArrayProbe>,
    repeated: bool,
) -> ArrayProbe<'a> {
    if repeated {
        retained.insert(array.repeated_probe()).as_probe()
    } else {
        array.probe()
    }
}

fn stacked_runend(
    runs: u32,
    page_size: usize,
    nested: bool,
    nullable: bool,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let ends = PrimitiveArray::from_iter((1..=runs).map(|run| run * 4));
    let ends = Pco::from_primitive(ends.as_view(), 3, page_size, ctx)?.into_array();
    let values = if nested {
        stacked_runend(runs / 4, page_size, false, nullable, ctx)?
    } else {
        let values = PrimitiveArray::new(
            (0..runs).collect::<Vec<_>>(),
            if nullable {
                Validity::from_iter((0..runs).map(|run| run % 11 != 0))
            } else {
                Validity::NonNullable
            },
        );
        Pco::from_primitive(values.as_view(), 3, page_size, ctx)?.into_array()
    };
    Ok(RunEnd::try_new(ends, values, ctx)?.into_array())
}

#[test]
fn stacked_runend_reuses_each_child_state_and_drops_it_once() -> VortexResult<()> {
    let session = vortex_array::array_session();
    vortex_runend::initialize(&session);
    let mut ctx = session.create_execution_ctx();
    // RunEnd(PCO, RunEnd(PCO, PCO)): three leaves, each fitting in one page.
    let array = stacked_runend(256, 512, true, false, &mut ctx)?;
    COUNTS.set(ProbeCounts::default());
    for lifetime in 1..=2 {
        let mut probe = array.repeated_probe();
        assert_eq!(COUNTS.get().initialized, 3 * (lifetime - 1));
        assert!(probe.execute_scalar(array.len(), &mut ctx).is_err());
        assert_eq!(COUNTS.get().initialized, 3 * (lifetime - 1));
        for index in [1, 5, 127, 255, 511, 1023, 0, 1] {
            assert_eq!(
                probe.execute_scalar(index, &mut ctx)?,
                u32::try_from(index / 16)?.into()
            );
            assert_eq!(
                COUNTS.get(),
                ProbeCounts {
                    initialized: 3 * lifetime,
                    decoded: 3 * lifetime,
                    dropped: 3 * (lifetime - 1),
                }
            );
        }
        drop(probe);
        assert_eq!(COUNTS.get().dropped, 3 * lifetime);
    }
    let before = COUNTS.get();
    let mut once = array.probe();
    for index in [1, 17, 511] {
        assert_eq!(
            once.execute_scalar(index, &mut ctx)?,
            u32::try_from(index / 16)?.into()
        );
    }
    assert_eq!(COUNTS.get(), before);
    Ok(())
}

#[rstest]
fn stacked_runend_random_access(
    #[values(false, true)] repeated: bool,
    #[values(false, true)] nested: bool,
    #[values(false, true)] nullable: bool,
    #[values(false, true)] sliced: bool,
) -> VortexResult<()> {
    let session = vortex_array::array_session();
    vortex_runend::initialize(&session);
    let mut ctx = session.create_execution_ctx();
    let encoded = stacked_runend(4096, 128, nested, nullable, &mut ctx)?;
    let range = if sliced { 777..15333 } else { 0..16384 };
    let source = if sliced {
        encoded
            .slice(range.clone())?
            .execute::<ArrayRef>(&mut ctx)?
    } else {
        encoded
    };
    assert!(source.is::<RunEnd>());
    let indices = [
        0,
        1,
        3,
        4,
        15,
        16,
        44,
        176,
        511,
        512,
        1023,
        1024,
        4097,
        source.len() - 1,
        0,
    ];
    let mut actual = builder_with_capacity_in(source.dtype(), indices.len(), ctx.allocator());
    let mut retained = None;
    let mut probe = probe_for(&source, &mut retained, repeated);
    for &index in &indices {
        actual.append_scalar(&probe.execute_scalar(index, &mut ctx)?)?;
    }
    let expected = indices
        .map(|index| u32::try_from((range.start + index) / if nested { 16 } else { 4 }))
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?;
    let validity = if nullable {
        Validity::from_iter(expected.iter().map(|value| value % 11 != 0))
    } else {
        Validity::NonNullable
    };
    let expected = PrimitiveArray::new(expected, validity);
    assert_arrays_eq!(actual.finish(), expected, &mut ctx);
    Ok(())
}

#[rstest]
fn sliced_nullable_random_access(
    #[values(false, true)] repeated: bool,
    #[values(false, true)] sliced: bool,
) -> VortexResult<()> {
    let mut ctx = vortex_array::array_session().create_execution_ctx();
    let input =
        PrimitiveArray::from_option_iter((0..4096i32).map(|i| (i % 7 != 0).then_some(i * 19)));
    let encoded = Pco::from_primitive(input.as_view(), 3, 128, &mut ctx)?.into_array();
    let range = if sliced { 777..3333 } else { 0..4096 };
    let source = encoded.slice(range.clone())?;
    assert!(source.is::<Pco>());
    let input = input.slice(range)?;
    let indices = [0u32, 1, 127, 128, 512, 2048, 129, 2, 1024, 0];
    let mut actual = builder_with_capacity_in(source.dtype(), indices.len(), ctx.allocator());
    let mut retained = None;
    let mut probe = probe_for(&source, &mut retained, repeated);
    for index in indices {
        actual.append_scalar(&probe.execute_scalar(index as usize, &mut ctx)?)?;
    }
    assert_arrays_eq!(
        actual.finish(),
        input.take(PrimitiveArray::from_iter(indices).into_array())?,
        &mut ctx
    );
    assert!(probe.execute_scalar(source.len(), &mut ctx).is_err());
    Ok(())
}

#[rstest]
#[case(PrimitiveArray::from_iter([1u16, 9, 32768, 65535]))]
#[case(PrimitiveArray::from_iter([i64::MIN, -1, 0, i64::MAX]))]
#[case(PrimitiveArray::from_iter([1.25f64, -2.5, 0.0, f64::INFINITY]))]
fn preserves_physical_type(#[case] input: PrimitiveArray) -> VortexResult<()> {
    let mut ctx = vortex_array::array_session().create_execution_ctx();
    let encoded = Pco::from_primitive(input.as_view(), 3, 2, &mut ctx)?.into_array();
    let mut probe = encoded.repeated_probe();
    let mut actual = builder_with_capacity_in(input.dtype(), input.len(), ctx.allocator());
    for i in 0..input.len() {
        actual.append_scalar(&probe.execute_scalar(i, &mut ctx)?)?;
    }
    assert_arrays_eq!(actual.finish(), input, &mut ctx);
    Ok(())
}

#[test]
fn decodes_each_page_once() -> VortexResult<()> {
    let mut ctx = vortex_array::array_session().create_execution_ctx();
    let input = PrimitiveArray::from_iter(0..4096i32);
    let encoded = Pco::from_primitive(input.as_view(), 3, 128, &mut ctx)?;
    let mut state = PcoProbeState::default();
    for index in [1, 5, 2, 100, 7] {
        assert_eq!(
            super::scalar_at(encoded.as_view(), index, Some(&mut state), &mut ctx)?,
            i32::try_from(index)?.into()
        );
    }
    assert_eq!(state.decoded_pages, 1);
    super::scalar_at(encoded.as_view(), 2048, Some(&mut state), &mut ctx)?;
    assert_eq!(state.decoded_pages, 2);
    for index in [1, 2049, 100, 2100] {
        assert_eq!(
            super::scalar_at(encoded.as_view(), index, Some(&mut state), &mut ctx)?,
            i32::try_from(index)?.into()
        );
    }
    assert_eq!(state.decoded_pages, 2);
    Ok(())
}

#[test]
fn all_null_access_does_not_decode() -> VortexResult<()> {
    let mut ctx = vortex_array::array_session().create_execution_ctx();
    let input = PrimitiveArray::new(vec![0i32; 128], Validity::AllInvalid);
    let encoded = Pco::from_primitive(input.as_view(), 3, 128, &mut ctx)?;
    let encoded = encoded.into_array();
    COUNTS.set(ProbeCounts::default());
    let mut probe = encoded.repeated_probe();
    assert!(probe.execute_scalar(42, &mut ctx)?.is_null());
    // The row is null, so the encoding is never entered and no state is built.
    assert_eq!(COUNTS.get(), ProbeCounts::default());
    Ok(())
}

#[test]
fn probe_retains_pages_after_source_is_dropped() -> VortexResult<()> {
    let session = vortex_array::array_session();
    vortex_runend::initialize(&session);
    let mut ctx = session.create_execution_ctx();
    let array = stacked_runend(256, 512, true, false, &mut ctx)?;
    COUNTS.set(ProbeCounts::default());
    let probe = RepeatedArrayProbe::new(array.clone());
    drop(array);
    let mut moved = (probe, ());
    for index in [1, 5, 127, 255, 511, 1023, 0, 1] {
        assert_eq!(
            moved.0.execute_scalar(index, &mut ctx)?,
            u32::try_from(index / 16)?.into()
        );
        assert_eq!(
            COUNTS.get(),
            ProbeCounts {
                initialized: 3,
                decoded: 3,
                dropped: 0
            }
        );
    }
    assert!(moved.0.execute_scalar(4096, &mut ctx).is_err());
    drop(moved);
    assert_eq!(COUNTS.get().dropped, 3);
    Ok(())
}
