// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
//! Bridges between the [`OnPair`](crate::OnPair) (List-shaped) and
//! [`OnPairView`](crate::OnPairView) (ListView-shaped) encodings, plus the
//! *compacting* take used as the baseline against the metadata-only OnPairView
//! take.

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use crate::OnPair;
use crate::OnPairArray;
use crate::OnPairArrayExt;
use crate::OnPairArraySlotsExt;
use crate::decode::collect_widened;

/// Take rows of an [`OnPairArray`] **into a freshly compacted `OnPairArray`**.
///
/// This is the [`List`](vortex_array::arrays::ListArray)-shaped baseline: it
/// rebuilds the `codes` token stream so the result only holds the tokens for the
/// taken rows (with fresh, monotonically increasing `codes_offsets`). The
/// dictionary blob and `dict_offsets` are shared unchanged. Contrast with
/// [`OnPairView`](crate::OnPairView)'s take, which never touches `codes`.
///
/// Mirrors the codes-rebuild that [`OnPair`](crate::OnPair)'s `filter` kernel
/// performs, so a `filter`-then-`take` chain over `OnPair` compacts at each step.
pub fn onpair_take_compact(
    array: &OnPairArray,
    indices: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<OnPairArray> {
    let idx = collect_widened::<u64>(indices, ctx)?;
    let cumulative = collect_widened::<u32>(array.codes_offsets(), ctx)?;
    let all_codes = collect_widened::<u16>(array.codes(), ctx)?;
    let lengths = collect_widened::<i32>(array.uncompressed_lengths(), ctx)?;

    let mut new_codes: Vec<u16> = Vec::new();
    let mut new_offsets: Vec<u32> = Vec::with_capacity(idx.len() + 1);
    new_offsets.push(0);
    let mut new_lengths: BufferMut<i32> = BufferMut::with_capacity(idx.len());

    for &row in idx.as_slice() {
        let row = row as usize;
        let start = cumulative[row] as usize;
        let end = cumulative[row + 1] as usize;
        vortex_ensure!(
            end <= all_codes.len(),
            "OnPair codes window [{start}, {end}) exceeds codes len {}",
            all_codes.len()
        );
        new_codes.extend_from_slice(&all_codes.as_slice()[start..end]);
        new_offsets.push(new_codes.len() as u32);
        new_lengths.push(lengths[row]);
    }

    let validity = array.array_validity().take(indices)?;

    OnPair::try_new(
        array.dtype().clone(),
        array.dict_bytes_handle().clone(),
        array.dict_offsets().clone(),
        Buffer::from(new_codes).into_array(),
        Buffer::from(new_offsets).into_array(),
        new_lengths.into_array(),
        validity,
        array.bits(),
    )
}
