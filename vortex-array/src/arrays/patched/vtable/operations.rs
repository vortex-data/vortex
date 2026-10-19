// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::ExecutionCtx;
use crate::array::ArrayView;
use crate::array::OperationsVTable;
use crate::array::ProbeChildren;
use crate::array::ProbeState;
use crate::arrays::Primitive;
use crate::arrays::patched::Patched;
use crate::arrays::patched::PatchedArrayExt;
use crate::arrays::patched::PatchedSlots;
use crate::scalar::Scalar;

// The constructor's 16/32-lane layouts span at most 64 positions; larger lanes stay uncached.
const MAX_CACHED_INDICES: usize = 64;

/// The most recently read lane for a retained Patched scalar probe.
#[derive(Default)]
pub struct PatchedProbeState {
    last_lane: Option<CachedLane>,
}

struct CachedLane {
    index: usize,
    range: Range<usize>,
    decoded_indices: Option<Vec<u16>>,
}

impl OperationsVTable<Patched> for Patched {
    type ProbeState = PatchedProbeState;

    fn probe_scalar(
        state: &mut ProbeState<'_, Patched>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let array = state.array();
        if !state.is_valid(index, ctx)? {
            return Ok(Scalar::null(array.dtype().clone()));
        }

        let physical_index = index + array.offset();
        let chunk = physical_index / 1024;
        #[expect(clippy::cast_possible_truncation, reason = "Index modulo 1024 fits in u16")]
        let chunk_index = (physical_index % 1024) as u16;
        let lane = physical_index % array.n_lanes();
        let lane_index = chunk * array.n_lanes() + lane;
        let (mut cache, mut children) = state.split();
        let cached_range = cache
            .as_ref()
            .and_then(|cache| cache.last_lane.as_ref())
            .filter(|cached| cached.index == lane_index)
            .map(|cached| cached.range.clone());
        let cache_hit = cached_range.is_some();
        let range = if let Some(range) = cached_range {
            range
        } else {
            let range = lane_range(&mut children, lane_index, ctx)?;
            if let Some(cache) = cache.as_mut() {
                cache.last_lane = Some(CachedLane {
                    index: lane_index,
                    range: range.clone(),
                    decoded_indices: None,
                });
            }
            range
        };

        let patch_position = patch_position(&mut children, cache, cache_hit, range, chunk_index, ctx)?;

        if let Some(position) = patch_position {
            return children
                .slot(PatchedSlots::PATCH_VALUES)?
                .ok_or_else(|| vortex_err!("Patched values slot is missing"))?
                .execute_scalar(position, ctx)?
                .cast(array.dtype());
        }

        children
            .slot(PatchedSlots::INNER)?
            .ok_or_else(|| vortex_err!("Patched inner slot is missing"))?
            .execute_scalar(index, ctx)
    }

    fn scalar_at(
        array: ArrayView<'_, Patched>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        Self::probe_scalar(&mut ProbeState::once(array), index, ctx)
    }
}

fn patch_position(
    children: &mut ProbeChildren<'_, '_, Patched>,
    cache: Option<&mut PatchedProbeState>,
    cache_hit: bool,
    range: Range<usize>,
    chunk_index: u16,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<usize>> {
    let mut indices = children
        .slot(PatchedSlots::PATCH_INDICES)?
        .ok_or_else(|| vortex_err!("Patched indices slot is missing"))?;
    if let Some(primitive) = indices.array().as_opt::<Primitive>() {
        return Ok(primitive
            .as_slice::<u16>()
            .get(range.clone())
            .ok_or_else(|| vortex_err!("Patched lane range is out of bounds"))?
            .iter()
            .position(|&value| value == chunk_index)
            .map(|position| range.start + position));
    }
    if range.start > range.end || range.end > indices.array().len() {
        return Err(vortex_err!("Patched lane range is out of bounds"));
    }

    if let Some(decoded) = cache
        .as_ref()
        .and_then(|cache| cache.last_lane.as_ref())
        .and_then(|lane| lane.decoded_indices.as_ref())
    {
        return Ok(decoded
            .iter()
            .position(|&value| value == chunk_index)
            .map(|position| range.start + position));
    }

    // Decode a non-Primitive lane only on its second visit, avoiding allocation on random reads.
    if cache_hit
        && range.len() <= MAX_CACHED_INDICES
        && let Some(cache) = cache
    {
        let decoded = range
            .clone()
            .map(|position| patch_index_at(&mut indices, position, ctx))
            .collect::<VortexResult<Vec<_>>>()?;
        let found = decoded
            .iter()
            .position(|&value| value == chunk_index)
            .map(|position| range.start + position);
        if let Some(lane) = &mut cache.last_lane {
            lane.decoded_indices = Some(decoded);
        }
        return Ok(found);
    }

    for position in range {
        if patch_index_at(&mut indices, position, ctx)? == chunk_index {
            return Ok(Some(position));
        }
    }
    Ok(None)
}

fn patch_index_at(
    indices: &mut crate::array::ArrayProbe<'_>,
    position: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<u16> {
    indices
        .execute_scalar(position, ctx)?
        .as_primitive()
        .as_::<u16>()
        .ok_or_else(|| vortex_err!("Patched index is not a u16"))
}

fn lane_range(
    children: &mut ProbeChildren<'_, '_, Patched>,
    lane_index: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Range<usize>> {
    let mut offsets = children
        .slot(PatchedSlots::LANE_OFFSETS)?
        .ok_or_else(|| vortex_err!("Patched lane offsets slot is missing"))?;
    if let Some(primitive) = offsets.array().as_opt::<Primitive>() {
        let values = primitive.as_slice::<u32>();
        let start = *values
            .get(lane_index)
            .ok_or_else(|| vortex_err!("Patched lane start is out of bounds"))?;
        let end = *values
            .get(lane_index + 1)
            .ok_or_else(|| vortex_err!("Patched lane end is out of bounds"))?;
        let start = usize::try_from(start)
            .map_err(|_| vortex_err!("Patched lane start does not fit usize"))?;
        let end = usize::try_from(end)
            .map_err(|_| vortex_err!("Patched lane end does not fit usize"))?;
        return Ok(start..end);
    }

    let start = offsets
        .execute_scalar(lane_index, ctx)?
        .as_primitive()
        .as_::<usize>()
        .ok_or_else(|| vortex_err!("Patched lane start is not an offset"))?;
    let end = offsets
        .execute_scalar(lane_index + 1, ctx)?
        .as_primitive()
        .as_::<usize>()
        .ok_or_else(|| vortex_err!("Patched lane end is not an offset"))?;
    Ok(start..end)
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::ArrayRef;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::arrays::ChunkedArray;
    use crate::arrays::Patched;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::patched::PatchedArraySlotsExt;
    use crate::arrays::patched::PatchedSlots;
    use crate::dtype::Nullability;
    use crate::optimizer::ArrayOptimizer;
    use crate::patches::Patches;
    use crate::scalar::Scalar;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    #[test]
    fn test_simple() {
        let mut ctx = SESSION.create_execution_ctx();
        let values = buffer![0u16; 1024].into_array();
        let patches = Patches::new(
            1024,
            0,
            buffer![1u32, 2, 3].into_array(),
            buffer![1u16; 3].into_array(),
            None,
            &mut ctx,
        )
        .unwrap();

        let mut ctx = SESSION.create_execution_ctx();

        let array = Patched::from_array_and_patches(values, &patches, &mut ctx)
            .unwrap()
            .into_array();

        assert_eq!(
            array
                .execute_scalar(0, &mut SESSION.create_execution_ctx())
                .unwrap(),
            Scalar::primitive(0u16, Nullability::NonNullable)
        );
        assert_eq!(
            array
                .execute_scalar(1, &mut SESSION.create_execution_ctx())
                .unwrap(),
            Scalar::primitive(1u16, Nullability::NonNullable)
        );
        assert_eq!(
            array
                .execute_scalar(2, &mut SESSION.create_execution_ctx())
                .unwrap(),
            Scalar::primitive(1u16, Nullability::NonNullable)
        );
        assert_eq!(
            array
                .execute_scalar(3, &mut SESSION.create_execution_ctx())
                .unwrap(),
            Scalar::primitive(1u16, Nullability::NonNullable)
        );
    }

    #[test]
    fn test_multi_chunk() {
        let mut ctx = SESSION.create_execution_ctx();
        let values = buffer![0u16; 4096].into_array();
        let patches = Patches::new(
            4096,
            0,
            buffer![1u32, 2, 3].into_array(),
            buffer![1u16; 3].into_array(),
            None,
            &mut ctx,
        )
        .unwrap();

        let mut ctx = SESSION.create_execution_ctx();

        let array = Patched::from_array_and_patches(values, &patches, &mut ctx)
            .unwrap()
            .into_array();

        for index in 0..array.len() {
            let value = array
                .execute_scalar(index, &mut SESSION.create_execution_ctx())
                .unwrap();

            if [1, 2, 3].contains(&index) {
                assert_eq!(value, 1u16.into());
            } else {
                assert_eq!(value, 0u16.into());
            }
        }
    }

    #[test]
    fn test_multi_chunk_sliced() {
        let mut ctx = SESSION.create_execution_ctx();
        let values = buffer![0u16; 4096].into_array();
        let patches = Patches::new(
            4096,
            0,
            buffer![1u32, 2, 3].into_array(),
            buffer![1u16; 3].into_array(),
            None,
            &mut ctx,
        )
        .unwrap();

        let mut ctx = SESSION.create_execution_ctx();

        let array = Patched::from_array_and_patches(values, &patches, &mut ctx)
            .unwrap()
            .into_array()
            .slice(3..4096)
            .unwrap()
            .optimize()
            .unwrap();

        assert!(array.is::<Patched>());

        assert_eq!(
            array
                .execute_scalar(0, &mut SESSION.create_execution_ctx())
                .unwrap(),
            1u16.into()
        );
        for index in 1..array.len() {
            assert_eq!(
                array
                    .execute_scalar(index, &mut SESSION.create_execution_ctx())
                    .unwrap(),
                0u16.into()
            );
        }
    }

    fn probe_fixture() -> VortexResult<ArrayRef> {
        let mut ctx = SESSION.create_execution_ctx();
        let values = PrimitiveArray::from_option_iter((0..3072).map(|index| {
            (![32, 65, 1024].contains(&index)).then_some(index as u32)
        }))
        .into_array();
        let patches = Patches::new(
            3072,
            0,
            buffer![31u32, 32, 1024, 1056, 2048].into_array(),
            buffer![9001u32, 9002, 9003, 9004, 9005].into_array(),
            None,
            &mut ctx,
        )?;
        Ok(Patched::from_array_and_patches(values, &patches, &mut ctx)?.into_array())
    }

    fn expected(index: usize) -> Option<u32> {
        match index {
            32 | 65 | 1024 => None,
            31 => Some(9001),
            1056 => Some(9004),
            2048 => Some(9005),
            _ => Some(index as u32),
        }
    }

    fn check_probe(
        array: &ArrayRef,
        start: usize,
        ctx: &mut crate::ExecutionCtx,
    ) -> VortexResult<()> {
        let mut repeated = array.repeated_probe();
        for absolute in [
            0, 31, 32, 33, 63, 64, 65, 1023, 1024, 1025, 1056, 2048, 2099, 3071, 31,
            1056, 32,
        ] {
            if absolute < start || absolute >= start + array.len() {
                continue;
            }
            let index = absolute - start;
            let expected = Scalar::from(expected(absolute));
            assert_eq!(array.execute_scalar(index, ctx)?, expected);
            assert_eq!(repeated.execute_scalar(index, ctx)?, expected);
            assert_eq!(repeated.execute_is_valid(index, ctx)?, !expected.is_null());
        }
        Ok(())
    }

    #[test]
    fn probe_nullable_patches_and_slices() -> VortexResult<()> {
        let array = probe_fixture()?;
        let mut ctx = SESSION.create_execution_ctx();
        check_probe(&array, 0, &mut ctx)?;
        check_probe(&array.slice(30..1100)?, 30, &mut ctx)?;
        check_probe(&array.slice(1020..2100)?, 1020, &mut ctx)
    }

    fn chunked_child(child: &ArrayRef) -> VortexResult<ArrayRef> {
        let middle = child.len() / 2;
        Ok(ChunkedArray::try_new(
            [child.slice(0..middle)?, child.slice(middle..child.len())?],
            child.dtype().clone(),
        )?
        .into_array())
    }

    #[test]
    fn probe_chunked_offsets_and_indices() -> VortexResult<()> {
        let array = probe_fixture()?;
        let patched = array.as_opt::<Patched>().expect("Patched fixture");
        let slots = PatchedSlots {
            inner: patched.inner().clone(),
            lane_offsets: chunked_child(patched.lane_offsets())?,
            patch_indices: chunked_child(patched.patch_indices())?,
            patch_values: patched.patch_values().clone(),
        };
        // SAFETY: Both replacement children have the same dtype and values as their originals.
        let array = unsafe { array.with_slots(slots.into_slots()) }?;
        let mut ctx = SESSION.create_execution_ctx();
        check_probe(&array, 0, &mut ctx)?;
        check_probe(&array.slice(30..1100)?, 30, &mut ctx)
    }
}
