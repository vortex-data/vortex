// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Shared assertions and fixtures for array tests.
//!
//! These helpers are available to this crate's tests and to crates that enable `_test-harness`.

use std::io::Write;

use goldenfile::Mint;
use goldenfile::differs::binary_diff;
use itertools::Itertools;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::arrays::BoolArray;
use crate::arrays::bool::BoolArrayExt;
use crate::optimizer::ArrayOptimizer;
use crate::optimizer::mark_decoded_optimized;

mod execution_paths;
pub use execution_paths::ExpectedOutcome;
pub use execution_paths::assert_execution_paths;

#[cfg(not(codspeed))]
pub mod trace;

/// Check that a named metadata matches its previous versioning.
///
/// Goldenfile takes care of checking for equality against a checked-in file.
#[expect(clippy::unwrap_used)]
pub fn check_metadata(name: &str, metadata: &[u8]) {
    let mut mint = Mint::new("goldenfiles/");
    let mut f = mint
        .new_goldenfile_with_differ(name, Box::new(binary_diff))
        .unwrap();
    f.write_all(metadata).unwrap();
}

/// Outputs the indices of the true values in a BoolArray
pub fn to_int_indices(indices_bits: BoolArray, ctx: &mut ExecutionCtx) -> VortexResult<Vec<u64>> {
    let buffer = indices_bits.to_bit_buffer();
    let mask = indices_bits
        .as_ref()
        .validity()?
        .execute_mask(indices_bits.as_ref().len(), ctx)?;
    Ok(buffer
        .iter()
        .enumerate()
        .filter_map(|(idx, v)| (v && mask.value(idx)).then_some(idx as u64))
        .collect_vec())
}

/// Marks a benchmark input as optimized, as decoding marks the arrays that a reader returns.
///
/// Execution optimizes each array that it reaches unless the array is already marked. An input
/// that a benchmark builds with constructors is not marked, so without this call the benchmark
/// also times the optimizer.
///
/// # Panics
///
/// Panics if the optimizer rewrites `array`. Then the benchmark would time an array other than
/// the one it built.
pub fn mark_optimized(array: &ArrayRef, session: &VortexSession) -> VortexResult<()> {
    let optimized = array.optimize_ctx(session)?;
    assert!(
        ArrayRef::ptr_eq(&optimized, array),
        "the optimizer rewrote the benchmark input:\n{}\ninto:\n{}",
        array.display_tree(),
        optimized.display_tree(),
    );

    Ok(())
}

/// Marks each node of a benchmark input as optimized without trying any rule, as decoding does.
///
/// Use this instead of [`mark_optimized`] when the benchmark times a shape that a file can hold
/// but that the optimizer rewrites, such as an extension array over constant storage. Decoding
/// keeps that shape, so the benchmark must keep it too.
pub fn mark_decoded(array: &ArrayRef) {
    for node in array.depth_first_traversal() {
        mark_decoded_optimized(&node);
    }
}
