// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Assertions for block fusion, read from the execution trace. See [`crate::block`].

use vortex_error::VortexResult;
use vortex_error::vortex_ensure_eq;
use vortex_session::VortexSession;

use super::trace::trace_op;
use crate::ArrayRef;
use crate::Canonical;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::assert_arrays_eq;

/// Execute `array` to canonical with block fusion off, then on, assert both give the same values
/// and dtype, and return the fusion steps of the fused run's trace: see [`fusion_lines`].
///
/// The fused run takes `array` itself, so a leaf it alone owns can be decoded in place.
pub fn fusion_trace(array: ArrayRef, session: &VortexSession) -> VortexResult<String> {
    let expected = array
        .clone()
        .execute::<Canonical>(&mut session.create_execution_ctx().with_block_fusion(false))?
        .into_array();
    let mut ctx = session.create_execution_ctx();
    let traced = trace_op(|| array.execute::<Canonical>(&mut ctx))?;
    let output = traced.output.into_array();
    vortex_ensure_eq!(output.dtype(), expected.dtype());
    assert_arrays_eq!(output, expected, &mut ctx);
    Ok(fusion_lines(&traced.trace.to_string()))
}

/// The fusion steps of a rendered trace: each slot request (`ExecuteSpine` when the slot may fuse,
/// `ExecuteSlot` when not), `fuse start`, `fuse extend` and `drive`, and how each op ran over the
/// first block. Each `drive` line is one fused chain, and an op over its leaf's own buffer shows
/// `leaf -> leaf`.
pub fn fusion_lines(trace: &str) -> String {
    trace
        .lines()
        .map(str::trim)
        .filter(|line| {
            ["ExecuteSpine", "ExecuteSlot", "fuse ", "drive ", "block_op"]
                .iter()
                .any(|prefix| line.starts_with(prefix))
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// The number of chains the executor drove in `fusion_lines`.
pub fn fused_chains(fusion_lines: &str) -> usize {
    fusion_lines
        .lines()
        .filter(|line| line.starts_with("drive "))
        .count()
}
