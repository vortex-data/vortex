// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Byte table lookups: each u8 index selects one byte of a table of at most 32 bytes.
//!
//! A table of 16 bytes fits one lookup instruction in each 16-byte block of a vector (VPSHUFB on
//! x86, TBL on aarch64). A table of 32 bytes takes two lookups and a select. `take_bits` uses the
//! same lookups to find the byte of each bit in a small bitmap.

use std::mem::MaybeUninit;

use fearless_simd::Simd;
use fearless_simd::prelude::*;
use fearless_simd::u8x16;
#[cfg(target_arch = "aarch64")]
use fearless_simd::u8x32;
use fearless_simd::u8x64;
use fearless_simd_macros::simd;

/// Runs `$body` with `$simd` set to the best SIMD level that has fast byte lookups, and evaluates
/// to `Some` of its result. Evaluates to `None` on CPUs without AVX2 or NEON, where the lookups
/// are emulated.
macro_rules! dispatch_byte_lookup {
    ($simd:ident => $body:expr) => {{
        let level = ::fearless_simd::Level::new();

        #[allow(unused_mut, unused_assignments)]
        let mut result = None;

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if let Some($simd) = level.as_avx512() {
            result = Some($body);
        } else if let Some($simd) = level.as_avx2() {
            result = Some($body);
        }

        #[cfg(target_arch = "aarch64")]
        if let Some($simd) = level.as_neon() {
            result = Some($body);
        }

        let _ = level;
        result
    }};
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub(crate) use dispatch_byte_lookup;

/// Writes `table[index]` for each index to `output`.
///
/// Tables of at most 32 bytes use SIMD byte lookups on x86 CPUs with AVX2 and on aarch64. Other
/// tables and CPUs use a scalar loop.
///
/// # Panics
///
/// Panics if `output` and `indices` have different lengths, or if an index is out of bounds of
/// `table`.
pub fn take_bytes(table: &[u8], indices: &[u8], output: &mut [MaybeUninit<u8>]) {
    assert_eq!(
        indices.len(),
        output.len(),
        "output length must equal the number of indices"
    );

    let max_index = match table.len() {
        0 => indices.iter().copied().max(),
        1..=16 => {
            dispatch_byte_lookup!(simd => take_bytes_simd::<_, false>(simd, table, indices, output))
        }
        17..=32 => {
            dispatch_byte_lookup!(simd => take_bytes_simd::<_, true>(simd, table, indices, output))
        }
        _ => None,
    };

    let Some(max_index) = max_index else {
        for (out, &index) in output.iter_mut().zip(indices) {
            out.write(table[usize::from(index)]);
        }
        return;
    };

    assert!(
        usize::from(max_index) < table.len(),
        "take index {max_index} out of bounds for length {}",
        table.len()
    );
}

/// Looks up the bytes of `indices` and returns the largest index, which the caller must check.
///
/// The output for an out-of-bounds index is unspecified. The table holds at most 32 bytes, and at
/// most 16 bytes unless `WIDE`.
#[cfg_attr(
    not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")),
    allow(
        dead_code,
        reason = "only x86 and aarch64 dispatch to SIMD byte lookups"
    )
)]
#[simd]
fn take_bytes_simd<S: Simd, const WIDE: bool>(
    simd: S,
    table: &[u8],
    indices: &[u8],
    output: &mut [MaybeUninit<u8>],
) -> u8 {
    let mut padded = [0u8; 32];
    padded[..table.len()].copy_from_slice(table);
    let (low, high) = block_tables(simd, &padded);

    let (chunks, tail) = indices.as_chunks::<64>();
    let (out_chunks, out_tail) = output.as_chunks_mut::<64>();

    let mut max = u8x64::splat(simd, 0);
    for (chunk, out) in chunks.iter().zip(out_chunks) {
        let indices = u8x64::from_slice(simd, chunk);
        max = max.max(indices);

        let bytes = if WIDE {
            lookup_32(simd, low, high, indices)
        } else {
            low.swizzle_dyn_within_blocks(indices)
        };

        for (out, byte) in out.iter_mut().zip(bytes.as_slice()) {
            out.write(*byte);
        }
    }

    let mut max_index = max.reduce_max();
    for (out, &index) in out_tail.iter_mut().zip(tail) {
        max_index = max_index.max(index);
        out.write(padded[usize::from(index) % 32]);
    }

    max_index
}

/// Splats each 16-byte half of `table` to every block of a vector.
#[cfg_attr(
    not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")),
    allow(
        dead_code,
        reason = "only x86 and aarch64 dispatch to SIMD byte lookups"
    )
)]
#[inline]
pub(crate) fn block_tables<S: Simd>(simd: S, table: &[u8; 32]) -> (u8x64<S>, u8x64<S>) {
    let (low, high) = table.split_at(16);
    (
        u8x64::block_splat(u8x16::from_slice(simd, low)),
        u8x64::block_splat(u8x16::from_slice(simd, high)),
    )
}

/// Looks up each index below 32 in the 32-byte table that `block_tables` splits into `low` and
/// `high`. The result for a larger index is unspecified.
#[cfg_attr(
    not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")),
    allow(
        dead_code,
        reason = "only x86 and aarch64 dispatch to SIMD byte lookups"
    )
)]
#[inline]
pub(crate) fn lookup_32<S: Simd>(
    simd: S,
    low: u8x64<S>,
    high: u8x64<S>,
    indices: u8x64<S>,
) -> u8x64<S> {
    // TBL looks up 16 indices in all 32 bytes at once. The level is known at compile time.
    #[cfg(target_arch = "aarch64")]
    if simd.level().as_neon().is_some() {
        let block = |table: u8x64<S>| simd.split_u8x32(simd.split_u8x64(table).0).0;
        let (low, high) = (block(low), block(high));
        let lookup = |indices: u8x32<S>| {
            let (first, second) = simd.split_u8x32(indices);
            simd.combine_u8x16(
                low.concat_swizzle_dyn(high, first),
                low.concat_swizzle_dyn(high, second),
            )
        };
        let (first, second) = simd.split_u8x64(indices);
        return simd.combine_u8x32(lookup(first), lookup(second));
    }

    // Index 16 and up is out of range for one block. NEON returns zero for it and x86 uses its
    // low 4 bits, so the XOR clears bit 4 for the high half on both.
    let from_low = low.swizzle_dyn_within_blocks(indices);
    let from_high = high.swizzle_dyn_within_blocks(indices ^ u8x64::splat(simd, 0x10));
    let in_high = indices.simd_ge(u8x64::splat(simd, 16));
    in_high.select(from_high, from_low)
}

#[cfg(test)]
mod tests {
    use std::mem::MaybeUninit;

    use hegel::TestCase;
    use hegel::generators as gs;

    use super::take_bytes;

    fn take(table: &[u8], indices: &[u8]) -> Vec<u8> {
        let mut output = vec![MaybeUninit::uninit(); indices.len()];
        take_bytes(table, indices, &mut output);
        output
            .into_iter()
            // SAFETY: take_bytes wrote every byte.
            .map(|byte| unsafe { byte.assume_init() })
            .collect()
    }

    #[hegel::test]
    #[cfg_attr(miri, ignore)]
    fn takes_bytes(tc: TestCase) {
        // Tables of up to 16, up to 32 and more bytes take different paths.
        let table: Vec<u8> = tc.draw(gs::vecs(gs::integers::<u8>()).min_size(1).max_size(40));
        let max_index = u8::try_from(table.len() - 1).unwrap();
        let indices: Vec<u8> =
            tc.draw(gs::vecs(gs::integers::<u8>().max_value(max_index)).max_size(200));

        let expected: Vec<u8> = indices.iter().map(|&i| table[usize::from(i)]).collect();
        assert_eq!(take(&table, &indices), expected);
    }
}
