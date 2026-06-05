// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Row-encode kernels for `RunEndArray`.
//!
//! Like `Dict`, the per-row size and per-row encoded bytes are determined by the column's
//! *values*, so we encode each run-value once and broadcast it across the indices in that
//! run. The per-unique-value cost is amortized over the number of runs rather than the
//! row count.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "row encoding indexes into u32-sized buffers; ends are non-negative"
)]

use num_traits::AsPrimitive;
use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::NativePType;
use vortex_array::match_each_integer_ptype;
use vortex_error::VortexResult;
use vortex_row::RowEncodeRegistration;
use vortex_row::RowSortField;
use vortex_row::dispatch_encode;
use vortex_row::dispatch_size;

use crate::RunEnd;
use crate::RunEndArrayExt;

/// Function pointer registered for the size contribution of a `RunEnd` column.
fn run_end_size_contribution(
    column: &ArrayRef,
    field: RowSortField,
    sizes: &mut [u32],
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<()>> {
    let Some(view) = column.as_opt::<RunEnd>() else {
        return Ok(None);
    };
    let nruns = view.ends().len();
    if nruns > view.len() {
        return Ok(None);
    }

    let mut value_sizes = vec![0u32; view.values().len()];
    dispatch_size(view.values(), field, &mut value_sizes, ctx)?;

    let offset = view.offset() as u64;
    let len = view.len();
    let ends_prim = view.ends().clone().execute::<PrimitiveArray>(ctx)?;

    match_each_integer_ptype!(ends_prim.ptype(), |E| {
        let ends = ends_prim.as_slice::<E>();
        walk_runs::<E>(ends, offset, len, |run_idx, start, stop| {
            let add = value_sizes[run_idx];
            if add == 0 {
                return;
            }
            for s in &mut sizes[start..stop] {
                *s += add;
            }
        });
    });
    Ok(Some(()))
}

/// Function pointer registered for the per-row encode of a `RunEnd` column.
fn run_end_encode_into(
    column: &ArrayRef,
    field: RowSortField,
    offsets: &[u32],
    cursors: &mut [u32],
    out: &mut [u8],
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<()>> {
    let Some(view) = column.as_opt::<RunEnd>() else {
        return Ok(None);
    };
    let nruns = view.ends().len();
    if nruns > view.len() {
        return Ok(None);
    }

    let n_values = view.values().len();
    let mut value_sizes = vec![0u32; n_values];
    dispatch_size(view.values(), field, &mut value_sizes, ctx)?;

    let mut value_offsets = vec![0u32; n_values + 1];
    let mut total: u64 = 0;
    for i in 0..n_values {
        value_offsets[i] = total as u32;
        total += u64::from(value_sizes[i]);
    }
    value_offsets[n_values] = total as u32;
    let mut value_buf = vec![0u8; total as usize];
    let zero_offsets = vec![0u32; n_values];
    let mut inner_cursors = value_offsets[..n_values].to_vec();
    dispatch_encode(
        view.values(),
        field,
        &zero_offsets,
        &mut inner_cursors,
        &mut value_buf,
        ctx,
    )?;

    let offset = view.offset() as u64;
    let len = view.len();
    let ends_prim = view.ends().clone().execute::<PrimitiveArray>(ctx)?;

    match_each_integer_ptype!(ends_prim.ptype(), |E| {
        let ends = ends_prim.as_slice::<E>();
        walk_runs::<E>(ends, offset, len, |run_idx, start, stop| {
            let v_start = value_offsets[run_idx] as usize;
            let v_size = value_sizes[run_idx] as usize;
            if v_size == 0 {
                return;
            }
            let value_bytes = &value_buf[v_start..v_start + v_size];
            let v_size_u32 = v_size as u32;
            for i in start..stop {
                let pos = (offsets[i] + cursors[i]) as usize;
                out[pos..pos + v_size].copy_from_slice(value_bytes);
                cursors[i] += v_size_u32;
            }
        });
    });
    Ok(Some(()))
}

/// For each run, call `f(run_idx, start_logical, stop_logical)` where the logical range is
/// `[max(prev_end - offset, 0), min(curr_end - offset, len))`.
#[inline]
fn walk_runs<E>(ends: &[E], offset: u64, len: usize, mut f: impl FnMut(usize, usize, usize))
where
    E: NativePType + AsPrimitive<u64>,
{
    let mut prev: u64 = offset;
    for (run_idx, &end) in ends.iter().enumerate() {
        let end_u64: u64 = end.as_();
        if end_u64 <= offset {
            prev = end_u64;
            continue;
        }
        let start = (prev.saturating_sub(offset)) as usize;
        let stop_u64 = end_u64 - offset;
        let stop = (stop_u64 as usize).min(len);
        if start < stop {
            f(run_idx, start, stop);
        }
        prev = end_u64;
        if stop >= len {
            break;
        }
    }
}

fn run_end_array_id() -> ArrayId {
    use vortex_session::registry::CachedId;
    static ID: CachedId = CachedId::new("vortex.runend");
    *ID
}

inventory::submit! {
    RowEncodeRegistration {
        id: run_end_array_id,
        size: run_end_size_contribution,
        encode: run_end_encode_into,
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::IntoArray;
    use vortex_array::LEGACY_SESSION;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::ListViewArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::listview::ListViewArrayExt;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_row::RowSortField;
    use vortex_row::convert_columns;

    use crate::RunEnd;

    fn collect_rows(arr: &ListViewArray) -> Vec<Vec<u8>> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let n = arr.len();
        (0..n)
            .map(|i| {
                let slice = arr.list_elements_at(i).unwrap();
                let p = slice.execute::<PrimitiveArray>(&mut ctx).unwrap();
                p.as_slice::<u8>().to_vec()
            })
            .collect()
    }

    #[test]
    fn runend_row_encode_matches_canonical() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let raw = buffer![1i32, 1, 1, 2, 2, 3, 3, 3, 3].into_array();
        let ree = RunEnd::encode(raw.clone(), &mut ctx)?.into_array();

        let by_canonical = convert_columns(&[raw], &[RowSortField::default()], &mut ctx)?;
        let by_ree = convert_columns(&[ree], &[RowSortField::default()], &mut ctx)?;

        assert_eq!(collect_rows(&by_canonical), collect_rows(&by_ree));
        Ok(())
    }
}
