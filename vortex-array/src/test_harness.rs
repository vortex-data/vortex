// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::io::Write;

use goldenfile::Mint;
use goldenfile::differs::binary_diff;
use itertools::Itertools;
use vortex_error::VortexResult;
use vortex_session::SessionExt;

use crate::LEGACY_SESSION;
use crate::VortexSessionExecute;
use crate::arrays::BoolArray;
use crate::arrays::bool::BoolArrayExt;
use crate::optimizer::kernels::ArrayKernels;

/// Extension trait for warming session variables before a benchmark's measured region.
///
/// The first access to a session variable inserts its default value. Performing that one-time
/// insertion inside a benchmark's measured loop charges it to the first measured iteration, which
/// is a problem under CodSpeed's instruction-count simulation. Calling [`Self::warm_kernels`]
/// before the bench loop moves the [`ArrayKernels`] insertion into setup instead.
pub trait WarmKernelsExt: SessionExt {
    /// Eagerly initialize the optimizer [`ArrayKernels`] on this session.
    fn warm_kernels(&self) {
        drop(self.get::<ArrayKernels>());
    }
}

impl<S: SessionExt> WarmKernelsExt for S {}

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
pub fn to_int_indices(indices_bits: BoolArray) -> VortexResult<Vec<u64>> {
    let buffer = indices_bits.to_bit_buffer();
    let mask = indices_bits.as_ref().validity()?.execute_mask(
        indices_bits.as_ref().len(),
        &mut LEGACY_SESSION.create_execution_ctx(),
    )?;
    Ok(buffer
        .iter()
        .enumerate()
        .filter_map(|(idx, v)| (v && mask.value(idx)).then_some(idx as u64))
        .collect_vec())
}
