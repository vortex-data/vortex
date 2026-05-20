// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Microbenchmark isolating the cache-locality question for patched bitpacked arrays:
//! is it faster to unpack every 1024-element block and *then* scatter all patches
//! (`unpack_then_patch`), or to unpack one block and immediately patch it while the
//! freshly-decoded block is still hot in cache (`fused`)?
//!
//! Both strategies perform identical total work (same unpack kernel calls, same number of
//! patch stores); only the loop ordering differs, so any delta is attributable to locality.

#![expect(clippy::cast_possible_truncation)]

use divan::Bencher;
use fastlanes::BitPacking;

fn main() {
    // Correctness guard: both strategies must produce identical output.
    let case = (1usize << 18, 9u8, 50u32);
    let data = Setup::new(case.0, case.1, case.2);
    let a = {
        let mut out = vec![0u32; data.n_padded];
        unpack_then_patch(&data, &mut out);
        out
    };
    let b = {
        let mut out = vec![0u32; data.n_padded];
        fused(&data, &mut out);
        out
    };
    assert_eq!(a, b, "fused and unpack_then_patch must agree");

    divan::main();
}

/// (num_values, bit_width, patch_stride) — one patch every `patch_stride` elements.
const CASES: &[(usize, u8, u32)] = &[
    // 256 KiB output, fits in L2.
    (1 << 16, 9, 200),
    (1 << 16, 9, 20),
    (1 << 16, 9, 5),
    // 1 MiB output, around L2.
    (1 << 18, 9, 200),
    (1 << 18, 9, 20),
    (1 << 18, 9, 5),
    // 4 MiB output, exceeds typical L2.
    (1 << 20, 9, 200),
    (1 << 20, 9, 20),
    (1 << 20, 9, 5),
    // 16 MiB output, far exceeds L2.
    (1 << 22, 9, 200),
    (1 << 22, 9, 20),
    (1 << 22, 9, 5),
];

struct Setup {
    bit_width: usize,
    elems_per_chunk: usize,
    num_chunks: usize,
    n_padded: usize,
    packed: Vec<u32>,
    /// Patch indices, globally sorted.
    indices: Vec<usize>,
    /// Patch values, parallel to `indices`.
    values: Vec<u32>,
    /// `chunk_offsets[c]..chunk_offsets[c + 1]` is the patch range for chunk `c`.
    chunk_offsets: Vec<usize>,
}

impl Setup {
    fn new(n: usize, bit_width: u8, patch_stride: u32) -> Self {
        let bit_width = bit_width as usize;
        let num_chunks = n.div_ceil(1024);
        let n_padded = num_chunks * 1024;
        let elems_per_chunk = 1024 * bit_width / 32;
        let mask = if bit_width == 32 {
            u32::MAX
        } else {
            (1u32 << bit_width) - 1
        };

        // Deterministic, low-entropy values that fit in `bit_width` bits.
        let values_in: Vec<u32> = (0..n_padded as u32)
            .map(|i| i.wrapping_mul(2654435761) & mask)
            .collect();

        let mut packed = vec![0u32; num_chunks * elems_per_chunk];
        for c in 0..num_chunks {
            // SAFETY: input is exactly 1024 elements, output exactly `elems_per_chunk`.
            unsafe {
                BitPacking::unchecked_pack(
                    bit_width,
                    &values_in[c * 1024..][..1024],
                    &mut packed[c * elems_per_chunk..][..elems_per_chunk],
                );
            }
        }

        // Uniformly-spread patches: one every `patch_stride` elements.
        let stride = patch_stride as usize;
        let mut indices = Vec::new();
        let mut values = Vec::new();
        let mut chunk_offsets = vec![0usize; num_chunks + 1];
        let mut idx = 0usize;
        while idx < n_padded {
            indices.push(idx);
            values.push(0xDEAD_BEEF ^ idx as u32);
            idx += stride;
        }
        // Build chunk offsets from the sorted indices.
        let mut p = 0usize;
        for c in 0..num_chunks {
            let chunk_end = (c + 1) * 1024;
            while p < indices.len() && indices[p] < chunk_end {
                p += 1;
            }
            chunk_offsets[c + 1] = p;
        }

        Self {
            bit_width,
            elems_per_chunk,
            num_chunks,
            n_padded,
            packed,
            indices,
            values,
            chunk_offsets,
        }
    }
}

/// Approach A: unpack every block into the output, then scatter all patches in a second pass.
#[inline]
fn unpack_then_patch(s: &Setup, output: &mut [u32]) {
    for c in 0..s.num_chunks {
        // SAFETY: packed slice is `elems_per_chunk`, output range is exactly 1024.
        unsafe {
            BitPacking::unchecked_unpack(
                s.bit_width,
                &s.packed[c * s.elems_per_chunk..][..s.elems_per_chunk],
                &mut output[c * 1024..][..1024],
            );
        }
    }
    for (i, &idx) in s.indices.iter().enumerate() {
        output[idx] = s.values[i];
    }
}

/// Approach B: unpack one block and immediately patch it while still hot in cache.
#[inline]
fn fused(s: &Setup, output: &mut [u32]) {
    for c in 0..s.num_chunks {
        // SAFETY: packed slice is `elems_per_chunk`, output range is exactly 1024.
        unsafe {
            BitPacking::unchecked_unpack(
                s.bit_width,
                &s.packed[c * s.elems_per_chunk..][..s.elems_per_chunk],
                &mut output[c * 1024..][..1024],
            );
        }
        for p in s.chunk_offsets[c]..s.chunk_offsets[c + 1] {
            output[s.indices[p]] = s.values[p];
        }
    }
}

#[divan::bench(args = CASES)]
fn unpack_then_patch_bench(bencher: Bencher, (n, bit_width, stride): (usize, u8, u32)) {
    let setup = Setup::new(n, bit_width, stride);
    bencher
        .with_inputs(|| vec![0u32; setup.n_padded])
        .bench_local_values(|mut output| {
            unpack_then_patch(&setup, &mut output);
            divan::black_box(output);
        });
}

#[divan::bench(args = CASES)]
fn fused_bench(bencher: Bencher, (n, bit_width, stride): (usize, u8, u32)) {
    let setup = Setup::new(n, bit_width, stride);
    bencher
        .with_inputs(|| vec![0u32; setup.n_padded])
        .bench_local_values(|mut output| {
            fused(&setup, &mut output);
            divan::black_box(output);
        });
}
