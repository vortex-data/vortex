// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compare checked, allocation-free dictionary lookup kernels on AArch64.

use std::arch::aarch64::*;
use std::hint::black_box;
use std::time::Instant;

use fearless_simd::Level;
use fearless_simd::prelude::*;
use fearless_simd::u8x16;
use fearless_simd::u8x32;
use fearless_simd::u8x64;
use fearless_simd::u16x8;
use fearless_simd::u16x16;
use fearless_simd::u16x32;

#[path = "../../../vortex-array/src/arrays/fixed_width/take/small_table/neon/table.rs"]
mod neon_table;

#[path = "../../../vortex-array/src/arrays/fixed_width/take/small_table/neon/planes.rs"]
mod neon_planes;

type Kernel<const W: usize> = fn(&[[u8; W]], &[u8], &mut [[u8; W]]);

#[inline(never)]
fn scalar<const W: usize>(values: &[[u8; W]], codes: &[u8], output: &mut [[u8; W]]) {
    assert_eq!(codes.len(), output.len());
    for (out, &code) in output.iter_mut().zip(codes) {
        *out = values[usize::from(code)];
    }
}

#[inline(always)]
fn bytes<const W: usize>(values: &[[u8; W]]) -> &[u8] {
    // SAFETY: Byte arrays have no padding; this covers exactly their initialized bytes.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), values.len() * W) }
}

#[inline(always)]
fn bytes_mut<const W: usize>(values: &mut [[u8; W]]) -> &mut [u8] {
    // SAFETY: Byte arrays have no padding; the exclusive slice covers their initialized bytes.
    unsafe { std::slice::from_raw_parts_mut(values.as_mut_ptr().cast(), values.len() * W) }
}

macro_rules! portable_kernel {
    ($name:ident, $vector:ident, $lanes:expr) => {
        #[inline(always)]
        fn $name<S: Simd, const W: usize, const BANKS: usize>(
            simd: S,
            values: &[[u8; W]],
            codes: &[u8],
            output: &mut [[u8; W]],
        ) {
            assert_eq!(codes.len(), output.len());
            let mut padded = [0u8; 256];
            padded[..values.len() * W].copy_from_slice(bytes(values));
            let a = $vector::from_slice(simd, &padded[..$lanes]);
            let b = $vector::from_slice(simd, &padded[$lanes..2 * $lanes]);
            let c = $vector::from_slice(simd, &padded[2 * $lanes..3 * $lanes]);
            let d = $vector::from_slice(simd, &padded[3 * $lanes..4 * $lanes]);
            let repeat = $vector::from_fn(simd, |i| (i / W) as u8);
            let lane = $vector::from_fn(simd, |i| (i % W) as u8);
            let width = $vector::splat(simd, W as u8);
            let mut maximum = $vector::splat(simd, 0);
            let out_bytes = bytes_mut(output);
            let mut offset = 0;
            while offset + $lanes <= codes.len() {
                let code = $vector::from_slice(simd, &codes[offset..offset + $lanes]);
                maximum = maximum.max(code);
                let index = if W == 1 { code } else {
                    code.swizzle_dyn_precise(repeat) * width + lane
                };
                let mut taken = a.swizzle_dyn_precise(index);
                if BANKS >= 2 {
                    taken |= b.swizzle_dyn_precise(index - $vector::splat(simd, $lanes as u8));
                }
                if BANKS >= 3 {
                    taken |= c.swizzle_dyn_precise(index - $vector::splat(simd, (2 * $lanes) as u8));
                }
                if BANKS >= 4 {
                    taken |= d.swizzle_dyn_precise(index - $vector::splat(simd, (3 * $lanes) as u8));
                }
                taken.store_slice(&mut out_bytes[offset * W..offset * W + $lanes]);
                offset += $lanes / W;
            }
            assert!(usize::from(maximum.reduce_max()) < values.len());
            for i in offset..codes.len() {
                output[i] = values[usize::from(codes[i])];
            }
        }
    };
}

portable_kernel!(portable16, u8x16, 16);
portable_kernel!(portable32, u8x32, 32);
portable_kernel!(portable64, u8x64, 64);

#[inline(never)]
fn portable<const W: usize>(values: &[[u8; W]], codes: &[u8], output: &mut [[u8; W]]) {
    if W == 2 || W == 4 || W == 8 {
        return portable_planar(values, codes, output);
    }
    let level = Level::new();
    match values.len() * W {
        1..=16 => fearless_simd::dispatch!(level, simd => portable16::<_, W, 1>(simd, values, codes, output)),
        17..=32 => fearless_simd::dispatch!(level, simd => portable32::<_, W, 1>(simd, values, codes, output)),
        33..=64 => fearless_simd::dispatch!(level, simd => portable64::<_, W, 1>(simd, values, codes, output)),
        65..=128 => fearless_simd::dispatch!(level, simd => portable64::<_, W, 2>(simd, values, codes, output)),
        129..=192 => fearless_simd::dispatch!(level, simd => portable64::<_, W, 3>(simd, values, codes, output)),
        193..=256 => fearless_simd::dispatch!(level, simd => portable64::<_, W, 4>(simd, values, codes, output)),
        _ => scalar(values, codes, output),
    }
}

#[inline(always)]
fn store_bytes16<S: Simd>(vectors: [u8x16<S>; 4], dest: &mut [u8]) {
    u8x16::store_four_interleaved(vectors, dest);
}

#[inline(always)]
fn store_bytes32<S: Simd>(vectors: [u8x32<S>; 4], dest: &mut [u8]) {
    let parts = vectors.map(|v| v.split());
    store_bytes16(parts.map(|p| p.0), &mut dest[..64]);
    store_bytes16(parts.map(|p| p.1), &mut dest[64..]);
}

#[inline(always)]
fn store_bytes64<S: Simd>(vectors: [u8x64<S>; 4], dest: &mut [u8]) {
    let parts = vectors.map(|v| v.split());
    store_bytes32(parts.map(|p| p.0), &mut dest[..128]);
    store_bytes32(parts.map(|p| p.1), &mut dest[128..]);
}

#[inline(always)]
fn store_words8<S: Simd>(vectors: [u16x8<S>; 4], dest: &mut [u16]) {
    u16x8::store_four_interleaved(vectors, dest);
}

#[inline(always)]
fn store_words16<S: Simd>(vectors: [u16x16<S>; 4], dest: &mut [u16]) {
    let parts = vectors.map(|v| v.split());
    store_words8(parts.map(|p| p.0), &mut dest[..32]);
    store_words8(parts.map(|p| p.1), &mut dest[32..]);
}

#[inline(always)]
fn store_words32<S: Simd>(vectors: [u16x32<S>; 4], dest: &mut [u16]) {
    let parts = vectors.map(|v| v.split());
    store_words16(parts.map(|p| p.0), &mut dest[..64]);
    store_words16(parts.map(|p| p.1), &mut dest[64..]);
}

macro_rules! portable_planes {
    ($name:ident, $vector:ident, $words:ident, $store_bytes:ident, $store_words:ident, $lanes:expr) => {
        #[inline(always)]
        fn $name<S: Simd, const W: usize, const BANKS: usize>(
            simd: S, values: &[[u8; W]], codes: &[u8], output: &mut [[u8; W]],
        ) {
            assert_eq!(codes.len(), output.len());
            let mut planes = [[0u8; 128]; W];
            for (i, value) in values.iter().enumerate() {
                for j in 0..W { planes[j][i] = value[j]; }
            }
            let first = std::array::from_fn::<_, W, _>(|j| $vector::from_slice(simd, &planes[j][..$lanes]));
            let second = std::array::from_fn::<_, W, _>(|j| $vector::from_slice(simd, &planes[j][$lanes..2*$lanes]));
            let mut maximum = $vector::splat(simd, 0);
            let mut offset = 0;
            while offset + $lanes <= codes.len() {
                let code = $vector::from_slice(simd, &codes[offset..offset+$lanes]);
                maximum = maximum.max(code);
                let taken = std::array::from_fn::<_, W, _>(|j| {
                    let mut v = first[j].swizzle_dyn_precise(code);
                    if BANKS == 2 {
                        v |= second[j].swizzle_dyn_precise(code - $vector::splat(simd, $lanes as u8));
                    }
                    v
                });
                let dest = &mut bytes_mut(output)[offset*W..(offset+$lanes)*W];
                if W == 2 {
                    taken[0].zip_low(taken[1]).store_slice(&mut dest[..$lanes]);
                    taken[0].zip_high(taken[1]).store_slice(&mut dest[$lanes..]);
                } else if W == 4 {
                    $store_bytes([taken[0], taken[1], taken[2], taken[3]], dest);
                } else if W == 8 {
                    let low: [$words<S>; 4] = std::array::from_fn(|j| taken[j*2].zip_low(taken[j*2+1]).bitcast());
                    let high: [$words<S>; 4] = std::array::from_fn(|j| taken[j*2].zip_high(taken[j*2+1]).bitcast());
                    assert_eq!(dest.as_ptr().align_offset(align_of::<u16>()), 0);
                    // SAFETY: Alignment is checked, every u16 bit pattern is valid, and the
                    // exclusive slice covers exactly the initialized output bytes.
                    let words = unsafe { std::slice::from_raw_parts_mut(dest.as_mut_ptr().cast::<u16>(), dest.len()/2) };
                    $store_words(low, &mut words[..2*$lanes]);
                    $store_words(high, &mut words[2*$lanes..]);
                }
                offset += $lanes;
            }
            assert!(usize::from(maximum.reduce_max()) < values.len());
            for i in offset..codes.len() { output[i] = values[usize::from(codes[i])]; }
        }
    };
}
portable_planes!(planes16, u8x16, u16x8, store_bytes16, store_words8, 16);
portable_planes!(planes32, u8x32, u16x16, store_bytes32, store_words16, 32);
portable_planes!(planes64, u8x64, u16x32, store_bytes64, store_words32, 64);

#[inline(never)]
fn portable_planar<const W: usize>(values: &[[u8; W]], codes: &[u8], output: &mut [[u8; W]]) {
    let level = Level::new();
    match values.len() {
        1..=16 => fearless_simd::dispatch!(level, simd => planes16::<_, W, 1>(simd, values, codes, output)),
        17..=32 => fearless_simd::dispatch!(level, simd => planes32::<_, W, 1>(simd, values, codes, output)),
        33..=64 => fearless_simd::dispatch!(level, simd => planes64::<_, W, 1>(simd, values, codes, output)),
        65..=128 => fearless_simd::dispatch!(level, simd => planes64::<_, W, 2>(simd, values, codes, output)),
        _ => scalar(values, codes, output),
    }
}

#[inline(always)]
unsafe fn raw_inner<const W: usize, const TABLE: usize>(
    values: &[[u8; W]], codes: &[u8], output: &mut [[u8; W]],
) {
    assert_eq!(codes.len(), output.len());
    let mut padded = [0u8; 256];
    padded[..values.len() * W].copy_from_slice(bytes(values));
    // SAFETY: AArch64 provides NEON; all padded loads are within initialized table storage.
    unsafe {
        let a = vld1q_u8_x4(padded.as_ptr());
        let b = vld1q_u8_x4(padded.as_ptr().add(64));
        let c = vld1q_u8_x4(padded.as_ptr().add(128));
        let d = vld1q_u8_x4(padded.as_ptr().add(192));
        let repeat = std::array::from_fn::<_, 16, _>(|i| (i / W) as u8);
        let lane = std::array::from_fn::<_, 16, _>(|i| (i % W) as u8);
        let repeat = vld1q_u8(repeat.as_ptr());
        let lane = vld1q_u8(lane.as_ptr());
        let width = vdupq_n_u8(W as u8);
        let mut maximum = vdupq_n_u8(0);
        let output_ptr = output.as_mut_ptr().cast::<u8>();
        let mut offset = 0;
        while offset + 16 <= codes.len() {
            let code = vld1q_u8(codes.as_ptr().add(offset));
            maximum = vmaxq_u8(maximum, code);
            let base = if W == 1 { code } else {
                vaddq_u8(vmulq_u8(vqtbl1q_u8(code, repeat), width), lane)
            };
            for half in 0..W.div_ceil(16) {
                let index = vaddq_u8(base, vdupq_n_u8((half * 16) as u8));
                let mut taken = match TABLE {
                    16 => vqtbl1q_u8(a.0, index),
                    32 => vqtbl2q_u8(uint8x16x2_t(a.0, a.1), index),
                    48 => vqtbl3q_u8(uint8x16x3_t(a.0, a.1, a.2), index),
                    _ => vqtbl4q_u8(a, index),
                };
                if TABLE > 64 {
                    taken = vorrq_u8(taken, vqtbl4q_u8(b, vsubq_u8(index, vdupq_n_u8(64))));
                }
                if TABLE > 128 {
                    taken = vorrq_u8(taken, vqtbl4q_u8(c, vsubq_u8(index, vdupq_n_u8(128))));
                }
                if TABLE > 192 {
                    taken = vorrq_u8(taken, vqtbl4q_u8(d, vsubq_u8(index, vdupq_n_u8(192))));
                }
                vst1q_u8(output_ptr.add(offset * W + half * 16), taken);
            }
            offset += (16 / W).max(1);
        }
        assert!(usize::from(vmaxvq_u8(maximum)) < values.len());
        for i in offset..codes.len() {
            output[i] = values[usize::from(codes[i])];
        }
    }
}

#[inline(never)]
fn raw_wide<const W: usize>(values: &[[u8; W]], codes: &[u8], output: &mut [[u8; W]]) {
    // SAFETY: The output has one initialized record per code. Each specialization bounds its
    // vector loads/stores, and checks indices independently of the wrapping byte permutations.
    unsafe {
        match values.len() * W {
            1..=16 => raw_inner::<W, 16>(values, codes, output),
            17..=32 => raw_inner::<W, 32>(values, codes, output),
            33..=48 => raw_inner::<W, 48>(values, codes, output),
            49..=64 => raw_inner::<W, 64>(values, codes, output),
            65..=128 => raw_inner::<W, 128>(values, codes, output),
            129..=192 => raw_inner::<W, 192>(values, codes, output),
            193..=256 => raw_inner::<W, 256>(values, codes, output),
            _ => scalar(values, codes, output),
        }
    }
}

#[inline(never)]
fn raw<const W: usize>(values: &[[u8; W]], codes: &[u8], output: &mut [[u8; W]]) {
    if W > 8 {
        return raw_wide(values, codes, output);
    }
    assert_eq!(output.len(), codes.len());
    if W == 2 || W == 4 || W == 8 {
        // SAFETY: The benchmark limits the dictionary to 256 bytes and reserves output for
        // every code; W is one of the supported plane counts.
        let (offset, maximum) = unsafe {
            neon_planes::take_vectors::<W>(bytes(values), codes, output.as_mut_ptr().cast())
        };
        assert!(usize::from(maximum) < values.len());
        for i in offset..codes.len() {
            output[i] = values[usize::from(codes[i])];
        }
        return;
    }
    let mut table = [0u8; 256];
    let value_bytes = bytes(values);
    table[..value_bytes.len()].copy_from_slice(value_bytes);
    let output_ptr = output.as_mut_ptr().cast::<u8>();
    // SAFETY: W is 1, 2, 4, or 8, table is initialized, and output has one record per code.
    let (offset, maximum) = unsafe {
        match value_bytes.len() {
            1..=16 => neon_table::take_vectors::<[u8; W], 16>(&table, codes, output_ptr),
            17..=32 => neon_table::take_vectors::<[u8; W], 32>(&table, codes, output_ptr),
            33..=48 => neon_table::take_vectors::<[u8; W], 48>(&table, codes, output_ptr),
            49..=64 => neon_table::take_vectors::<[u8; W], 64>(&table, codes, output_ptr),
            65..=128 => neon_table::take_vectors::<[u8; W], 128>(&table, codes, output_ptr),
            129..=192 => neon_table::take_vectors::<[u8; W], 192>(&table, codes, output_ptr),
            _ => neon_table::take_vectors::<[u8; W], 256>(&table, codes, output_ptr),
        }
    };
    assert!(usize::from(maximum) < values.len());
    for i in offset..codes.len() {
        output[i] = values[usize::from(codes[i])];
    }
}

fn dataset<const W: usize>(count: usize, len: usize, skewed: bool) -> (Vec<[u8; W]>, Vec<u8>) {
    let values = (0..count).map(|i| std::array::from_fn(|j| {
        ((i * 37 + j * 53 + 129) ^ (i >> 1)) as u8
    })).collect();
    let mut seed = 0x9e3779b97f4a7c15u64;
    let codes = (0..len).map(|_| {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        if skewed && seed % 10 != 0 { 0 } else { (seed % count as u64) as u8 }
    }).collect();
    (values, codes)
}

fn validate<const W: usize>() {
    for count in [1, 2, 3, 7, 16, 17, 31, 32, 33, 48, 63, 64, 65, 96, 127, 128, 192, 255, 256] {
        if count * W > 256 { continue; }
        if count < 256 {
            for position in [0, 64, 128] {
                for invalid in [count as u8, 255] {
                    let (values, mut codes) = dataset::<W>(count, 129, false);
                    codes[position] = invalid;
                    for kernel in [raw::<W> as Kernel<W>, portable::<W>] {
                        let result = std::panic::catch_unwind(|| {
                            let mut output = vec![[0; W]; codes.len()];
                            kernel(&values, &codes, &mut output);
                        });
                        assert!(result.is_err(), "accepted invalid code: width={W}, count={count}");
                    }
                }
            }
        }
        for len in [0, 1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 1025] {
            let (values, mut codes) = dataset::<W>(count, len + 1, false);
            for (i, code) in codes.iter_mut().enumerate() { *code = (i % count) as u8; }
            let codes = &codes[1..];
            let expected: Vec<_> = codes.iter().map(|&c| values[usize::from(c)]).collect();
            for kernel in [raw::<W> as Kernel<W>, portable::<W>] {
                let mut actual = vec![[0; W]; len];
                kernel(&values, codes, &mut actual);
                assert_eq!(actual, expected, "width={W} count={count} len={len}");
            }
        }
    }
}

fn measure<const W: usize>() {
    let kernels: [(&str, Kernel<W>); 3] = [("scalar", scalar::<W>), ("neon", raw::<W>), ("fearless", portable::<W>)];
    for count in [2, 4, 8, 16, 17, 24, 31, 32, 33, 48, 63, 64, 65, 96, 127, 128, 129, 192, 255, 256] {
        if count * W > 256 { continue; }
        for len in [64, 1024, 65_536, 1_000_000] {
            for skewed in [false, true] {
                let (values, codes) = dataset::<W>(count, len, skewed);
                let expected: Vec<_> = codes.iter().map(|&c| values[usize::from(c)]).collect();
                let mut output = vec![[0; W]; len];
                for (_, kernel) in kernels {
                    kernel(&values, &codes, &mut output);
                    assert_eq!(output, expected);
                }
                let iterations = (2_000_000 / (len * W)).clamp(1, 256);
                let mut samples = [Vec::new(), Vec::new(), Vec::new()];
                for sample in 0..31 {
                    for position in 0..3 {
                        let k = (sample + position) % 3;
                        let kernel = kernels[k].1;
                        let start = Instant::now();
                        for _ in 0..iterations {
                            kernel(black_box(&values), black_box(&codes), black_box(&mut output));
                            black_box(&output);
                        }
                        samples[k].push(start.elapsed().as_secs_f64() * 1e9 / iterations as f64);
                    }
                }
                for k in 0..3 {
                    samples[k].sort_by(f64::total_cmp);
                    println!("{{\"width\":{W},\"count\":{count},\"len\":{len},\"skewed\":{skewed},\"kernel\":\"{}\",\"median_ns\":{},\"min_ns\":{},\"max_ns\":{},\"iterations\":{iterations}}}",
                        kernels[k].0, samples[k][15], samples[k][0], samples[k][30]);
                }
            }
        }
    }
}

fn main() {
    assert!(Level::new().as_neon().is_some());
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    validate::<1>(); validate::<2>(); validate::<4>();
    validate::<8>(); validate::<16>(); validate::<32>();
    std::panic::set_hook(hook);
    eprintln!("Correctness checks passed for all eligible widths, table sizes, and vector tails.");
    measure::<1>(); measure::<2>(); measure::<4>();
    measure::<8>(); measure::<16>(); measure::<32>();
}
