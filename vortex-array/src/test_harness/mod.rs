// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::io::Write;

use goldenfile::Mint;
use goldenfile::differs::binary_diff;
use itertools::Itertools;
use vortex_error::VortexResult;

use crate::ExecutionCtx;
use crate::arrays::BoolArray;
use crate::arrays::bool::BoolArrayExt;
use crate::arrays::fixed_width::take::take_values;
use crate::arrays::fixed_width::take::take_values_fallback;

/// Runs the fixed-width byte take kernel directly for benchmarks.
#[doc(hidden)]
pub fn take_values_u8(
    values: &[u8],
    indices: &[u8],
    allocator: &vortex_buffer::BufferAllocatorRef,
) -> vortex_buffer::Buffer<u8> {
    take_values(values, indices, allocator)
}

/// Runs the fixed-width byte take fallback directly for benchmarks.
#[doc(hidden)]
pub fn take_values_fallback_u8(
    values: &[u8],
    indices: &[u8],
    allocator: &vortex_buffer::BufferAllocatorRef,
) -> vortex_buffer::Buffer<u8> {
    take_values_fallback(values, indices, allocator)
}

macro_rules! export_take_benchmarks {
    ($take:ident, $fallback:ident, $ty:ty) => {
        #[doc(hidden)]
        pub fn $take(
            values: &[$ty],
            indices: &[u8],
            allocator: &vortex_buffer::BufferAllocatorRef,
        ) -> vortex_buffer::Buffer<$ty> {
            take_values(values, indices, allocator)
        }

        #[doc(hidden)]
        pub fn $fallback(
            values: &[$ty],
            indices: &[u8],
            allocator: &vortex_buffer::BufferAllocatorRef,
        ) -> vortex_buffer::Buffer<$ty> {
            take_values_fallback(values, indices, allocator)
        }
    };
}

export_take_benchmarks!(take_values_u16, take_values_fallback_u16, u16);
export_take_benchmarks!(take_values_u32, take_values_fallback_u32, u32);
export_take_benchmarks!(take_values_u64, take_values_fallback_u64, u64);

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
