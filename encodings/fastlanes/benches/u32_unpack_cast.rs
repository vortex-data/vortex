// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compare two ways to materialize a `u32` chunk from bit-packed integers whose values fit
//! in `u16`, using only in-range ("stock") values (no patches/exceptions):
//!
//! * `unpack_u32`          — unpack the bit-packed data directly into a `u32` chunk.
//! * `unpack_u16_cast_u32` — unpack into a narrower `u16` chunk, then widen-cast `u16 -> u32`
//!   (the fused unpack + cast kernel).
//!
//! Each iteration processes a single FastLanes chunk of 1024 elements ("1k per time") at bit
//! widths 3 and 7.
//!
//! Run with `cargo bench -p vortex-fastlanes --bench u32_unpack_cast`.

#![expect(clippy::unwrap_used)]
#![expect(clippy::cast_possible_truncation)]

use divan::Bencher;
use divan::black_box;
use divan::counter::ItemsCount;
use fastlanes::BitPacking;

fn main() {
    divan::main();
}

/// One FastLanes vector / chunk.
const CHUNK: usize = 1024;
/// Small bit widths whose values fit in `u16`.
const BIT_WIDTHS: &[u8] = &[3, 7];

/// Pack one chunk of 1024 `u32` stock values at `bit_width`.
fn pack_u32(bit_width: usize, values: &[u32]) -> Vec<u32> {
    assert_eq!(values.len(), CHUNK);
    let packed_len = 128 * bit_width / size_of::<u32>();
    let mut packed = vec![0u32; packed_len];
    // SAFETY: `values` is exactly 1024 elements and `packed` is exactly `128 * bit_width / 4`.
    unsafe {
        BitPacking::unchecked_pack(bit_width, values, &mut packed);
    }
    packed
}

/// Pack one chunk of 1024 `u16` stock values at `bit_width`.
fn pack_u16(bit_width: usize, values: &[u16]) -> Vec<u16> {
    assert_eq!(values.len(), CHUNK);
    let packed_len = 128 * bit_width / size_of::<u16>();
    let mut packed = vec![0u16; packed_len];
    // SAFETY: `values` is exactly 1024 elements and `packed` is exactly `128 * bit_width / 2`.
    unsafe {
        BitPacking::unchecked_pack(bit_width, values, &mut packed);
    }
    packed
}

/// Unpack a bit-packed chunk straight into `u32`.
#[divan::bench(consts = BIT_WIDTHS)]
fn unpack_u32<const BW: u8>(bencher: Bencher) {
    let bit_width = BW as usize;
    let values: Vec<u32> = (0..CHUNK).map(|i| (i as u32) % (1u32 << BW)).collect();
    let packed = pack_u32(bit_width, &values);
    let mut dst = vec![0u32; CHUNK];

    bencher.counter(ItemsCount::new(CHUNK)).bench_local(|| {
        // SAFETY: `packed` is `128 * bit_width / 4` elements and `dst` is 1024 elements.
        unsafe {
            BitPacking::unchecked_unpack(bit_width, black_box(&packed), &mut dst);
        }
        black_box(&dst);
    });
}

/// Unpack a bit-packed chunk into `u16`, then widen-cast every element to `u32`.
#[divan::bench(consts = BIT_WIDTHS)]
fn unpack_u16_cast_u32<const BW: u8>(bencher: Bencher) {
    let bit_width = BW as usize;
    let values: Vec<u16> = (0..CHUNK).map(|i| (i as u16) % (1u16 << BW)).collect();
    let packed = pack_u16(bit_width, &values);
    let mut tmp = vec![0u16; CHUNK];
    let mut dst = vec![0u32; CHUNK];

    bencher.counter(ItemsCount::new(CHUNK)).bench_local(|| {
        // SAFETY: `packed` is `128 * bit_width / 2` elements and `tmp` is 1024 elements.
        unsafe {
            BitPacking::unchecked_unpack(bit_width, black_box(&packed), &mut tmp);
        }
        for (d, &s) in dst.iter_mut().zip(tmp.iter()) {
            *d = s as u32;
        }
        black_box(&dst);
    });
}
