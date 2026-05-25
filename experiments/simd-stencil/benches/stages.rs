// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Per-stage breakdown of the stack-B decode, to find which kernel dominates
//! before attempting to vectorize it.
//!
//! ```text
//! RUSTFLAGS="-C target-cpu=native" cargo bench -p simd-stencil --bench stages
//! ```

use divan::Bencher;
use divan::counter::ItemsCount;
use simd_stencil::encode::encode_b;
use simd_stencil::encode::gen_f64;
use simd_stencil::kernels::alp_scale_slice;
use simd_stencil::kernels::undelta_u64;
use simd_stencil::kernels::unfor_unpack_u64;
use simd_stencil::kernels::untranspose_u64;

const N: usize = 1 << 20;
const TILE: usize = 1024;
const EXP: i32 = 2;

fn main() {
    divan::main();
}

/// Stage 1: unpack + FoR over the whole column.
#[divan::bench(name = "1_unpack_for")]
fn unpack(bencher: Bencher) {
    let enc = encode_b(&gen_f64(N, EXP, 2), EXP);
    let tiles = N / TILE;
    bencher.counter(ItemsCount::new(N)).bench(|| {
        let mut td = [0u64; TILE];
        for t in 0..tiles {
            let w = enc.width[t] as usize;
            let off = enc.offsets[t];
            let plen = TILE * w / 64;
            unfor_unpack_u64(w, &enc.packed[off..off + plen], enc.reference[t], &mut td);
        }
        td[0]
    });
}

/// Stage 2: undelta over the whole column (input already unpacked).
#[divan::bench(name = "2_undelta")]
fn undelta(bencher: Bencher) {
    let td: Vec<u64> = (0..N)
        .map(|i| (i as u64).wrapping_mul(2654435761))
        .collect();
    let tiles = N / TILE;
    bencher.counter(ItemsCount::new(N)).bench(|| {
        let mut tu = [0u64; TILE];
        for t in 0..tiles {
            let tile: &[u64; TILE] = td[t * TILE..(t + 1) * TILE].try_into().unwrap();
            undelta_u64(tile, &mut tu);
        }
        tu[0]
    });
}

/// Stage 3: untranspose over the whole column.
#[divan::bench(name = "3_untranspose")]
fn untranspose(bencher: Bencher) {
    let tu: Vec<u64> = (0..N)
        .map(|i| (i as u64).wrapping_mul(2654435761))
        .collect();
    let tiles = N / TILE;
    bencher.counter(ItemsCount::new(N)).bench(|| {
        let mut digits = [0u64; TILE];
        for t in 0..tiles {
            let tile: &[u64; TILE] = tu[t * TILE..(t + 1) * TILE].try_into().unwrap();
            untranspose_u64(tile, &mut digits);
        }
        digits[0]
    });
}

/// Stage 4: ALP scale over the whole column (vectorized).
#[divan::bench(name = "4_alp_scale")]
fn scale(bencher: Bencher) {
    let digits: Vec<u64> = (0..N as u64).collect();
    bencher.counter(ItemsCount::new(N)).bench(|| {
        let mut out = vec![0f64; N];
        alp_scale_slice(&digits, 0.01, &mut out);
        out
    });
}
