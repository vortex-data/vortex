// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::iter::Iterator;

use arrow_buffer::BooleanBuffer;
use arrow_buffer::BooleanBufferBuilder;
use divan::Bencher;
use vortex_buffer::BitBuffer;
use vortex_buffer::BitBufferMut;
use vortex_buffer::collect_bool_words;

// Sizes spanning L1 -> DRAM for the collect-bool / bitmask-pack benchmarks.
const PACK_SIZES: &[usize] = &[1024, 16_384, 262_144, 1_048_576];

/// Pure-compute baseline: pack `n` truthy bytes (`b != 0`) into a *reused* word
/// buffer via the real `collect_bool_words` (the scalar `packed |= (f(i)) << i`
/// idiom). No allocation in the measured region.
#[divan::bench(args = PACK_SIZES)]
fn pack_truthy_bytes(bencher: Bencher, n: usize) {
    let data: Vec<u8> = (0..n).map(|i| i.is_multiple_of(7) as u8).collect();
    let mut words = vec![0u64; n.div_ceil(64)];
    bencher.bench_local(|| {
        let d = divan::black_box(data.as_slice());
        collect_bool_words(divan::black_box(&mut words), n, |i| d[i] > 0);
        divan::black_box(words.as_slice());
    });
}

/// SIMD fast path: same pack into a *reused* buffer via `pack_nonzero_bytes`.
#[divan::bench(args = PACK_SIZES)]
fn pack_truthy_bytes_simd(bencher: Bencher, n: usize) {
    let data: Vec<u8> = (0..n).map(|i| i.is_multiple_of(7) as u8).collect();
    let mut words = vec![0u64; n.div_ceil(64)];
    bencher.bench_local(|| {
        let d = divan::black_box(data.as_slice());
        vortex_buffer::pack_nonzero_bytes(divan::black_box(&mut words), d);
        divan::black_box(words.as_slice());
    });
}

/// End-to-end real caller: `BitBufferMut::from(&[u8])` (includes allocation).
#[divan::bench(args = PACK_SIZES)]
fn bitbuffer_from_u8(bencher: Bencher, n: usize) {
    let data: Vec<u8> = (0..n).map(|i| i.is_multiple_of(7) as u8).collect();
    bencher
        .with_inputs(|| data.as_slice())
        .bench_refs(|s| BitBufferMut::from(divan::black_box(*s)));
}

// ---- Typed compare -> bitmask (the `primitive between` shape, i32) ----

/// Baseline: exactly what `primitive between` does today — `collect_bool_words`
/// over a contiguous `&[i32]` with the inclusive between predicate.
#[divan::bench(args = PACK_SIZES)]
fn between_i32_scalar(bencher: Bencher, n: usize) {
    let data: Vec<i32> = (0..n).map(|i| (i as i32).wrapping_mul(2_654_435_761u32 as i32)).collect();
    let mut words = vec![0u64; n.div_ceil(64)];
    let (lo, hi) = (-100_000_000i32, 100_000_000i32);
    bencher.bench_local(|| {
        let d = divan::black_box(data.as_slice());
        collect_bool_words(divan::black_box(&mut words), n, |i| lo <= d[i] && d[i] <= hi);
        divan::black_box(words.as_slice());
    });
}

/// AVX-512 between: vpcmpd (>= lo) & vpcmpd (<= hi) -> kmovw, 16 i32/iter.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
fn between_i32_avx512(out: &mut [u16], value: &[i32], lo: i32, hi: i32) {
    use std::arch::x86_64::*;
    let vlo = _mm512_set1_epi32(lo);
    let vhi = _mm512_set1_epi32(hi);
    let p = value.as_ptr() as *const __m512i;
    for (i, w) in out.iter_mut().take(value.len() / 16).enumerate() {
        // SAFETY: i < len/16 keeps the load in bounds.
        let v = unsafe { _mm512_loadu_si512(p.add(i)) };
        let ge = _mm512_cmpge_epi32_mask(v, vlo);
        let le = _mm512_cmple_epi32_mask(v, vhi);
        *w = ge & le;
    }
}

#[divan::bench(args = PACK_SIZES)]
fn between_i32_simd(bencher: Bencher, n: usize) {
    let data: Vec<i32> = (0..n).map(|i| (i as i32).wrapping_mul(2_654_435_761u32 as i32)).collect();
    let mut masks = vec![0u16; n.div_ceil(16)];
    let (lo, hi) = (-100_000_000i32, 100_000_000i32);
    bencher.bench_local(|| {
        let d = divan::black_box(data.as_slice());
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx512f") {
            // SAFETY: avx512f confirmed present at runtime.
            unsafe { between_i32_avx512(divan::black_box(&mut masks), d, lo, hi) };
        }
        divan::black_box(masks.as_slice());
    });
}

fn main() {
    // Pre-warm CPUID feature detection so the one-time probe cost is never
    // included in any benchmark iteration.
    #[cfg(target_arch = "x86_64")]
    {
        let _ = is_x86_feature_detected!("avx2");
        let _ = is_x86_feature_detected!("avx512f");
        let _ = is_x86_feature_detected!("avx512vpopcntdq");
    }

    divan::main();
}

/// Wraps an arrow buffer so Divan can provide a nice name
pub struct Arrow<T>(T);

impl FromIterator<bool> for Arrow<BooleanBuffer> {
    fn from_iter<I: IntoIterator<Item = bool>>(iter: I) -> Self {
        Self(BooleanBuffer::from_iter(iter))
    }
}

const INPUT_SIZE: &[usize] = &[128, 1024, 2048, 16_384, 65_536];

#[inline]
fn true_count_pattern(i: usize) -> bool {
    (i.is_multiple_of(3)) ^ (i.is_multiple_of(11))
}

#[cfg(not(codspeed))]
#[divan::bench(args = INPUT_SIZE)]
fn from_iter_arrow(n: usize) {
    Arrow::<BooleanBuffer>::from_iter((0..n).map(|i| i % 2 == 0));
}

#[divan::bench(args = INPUT_SIZE)]
fn from_iter_bit_buffer(n: usize) {
    BitBuffer::from_iter((0..n).map(|i| i % 2 == 0));
}

#[divan::bench(args = INPUT_SIZE)]
fn append_vortex_buffer(bencher: Bencher, length: usize) {
    bencher
        .with_inputs(|| (BitBufferMut::with_capacity(length), length))
        .bench_refs(|(buffer, length)| {
            for idx in 0..*length {
                buffer.append(idx % 2 == 0);
            }
        });
}

#[divan::bench(args = INPUT_SIZE)]
fn append_arrow_buffer(bencher: Bencher, length: usize) {
    bencher
        .with_inputs(|| (Arrow(BooleanBufferBuilder::new(length)), length))
        .bench_refs(|(buffer, length)| {
            for idx in 0..*length {
                buffer.0.append(idx % 2 == 0);
            }
        });
}

#[divan::bench(args = INPUT_SIZE)]
fn append_n_vortex_buffer(bencher: Bencher, length: usize) {
    bencher
        .with_inputs(|| (BitBufferMut::with_capacity(length), length, true))
        .bench_refs(|(buffer, length, boolean)| {
            for _ in 0..100 {
                buffer.append_n(*boolean, *length / 100);
            }
        });
}

#[divan::bench(args = INPUT_SIZE)]
fn append_n_arrow_buffer(bencher: Bencher, length: usize) {
    bencher
        .with_inputs(|| (Arrow(BooleanBufferBuilder::new(length)), length, true))
        .bench_refs(|(buffer, length, boolean)| {
            for _ in 0..100 {
                buffer.0.append_n(*length / 100, *boolean);
            }
        });
}

#[divan::bench(args = INPUT_SIZE)]
fn append_buffer_vortex_buffer(bencher: Bencher, length: usize) {
    bencher
        .with_inputs(|| {
            let source = BitBuffer::from_iter((0..length / 100).map(|i| i % 2 == 0));
            let dest = BitBufferMut::with_capacity(length);
            (source, dest)
        })
        .bench_refs(|(source, dest)| {
            for _ in 0..100 {
                dest.append_buffer(source);
            }
        });
}

#[divan::bench(args = INPUT_SIZE)]
fn append_buffer_arrow_buffer(bencher: Bencher, length: usize) {
    bencher
        .with_inputs(|| {
            let source = Arrow(BooleanBuffer::from_iter(
                (0..length / 100).map(|i| i % 2 == 0),
            ));
            let dest = Arrow(BooleanBufferBuilder::new(length));
            (source, dest)
        })
        .bench_refs(|(source, dest)| {
            for _ in 0..100 {
                for value in source.0.iter() {
                    dest.0.append(value);
                }
            }
        });
}

#[divan::bench(args = INPUT_SIZE)]
fn value_vortex_buffer(bencher: Bencher, length: usize) {
    let buffer = BitBuffer::from_iter((0..length).map(|i| i % 2 == 0));
    bencher.with_inputs(|| &buffer).bench_refs(|buffer| {
        for idx in 0..length {
            divan::black_box(buffer.value(idx));
        }
    });
}

#[divan::bench(args = INPUT_SIZE)]
fn value_arrow_buffer(bencher: Bencher, length: usize) {
    let buffer = Arrow(BooleanBuffer::from_iter((0..length).map(|i| i % 2 == 0)));
    bencher.with_inputs(|| &buffer).bench_refs(|buffer| {
        for idx in 0..length {
            divan::black_box(buffer.0.value(idx));
        }
    });
}

#[divan::bench(args = INPUT_SIZE)]
fn slice_vortex_buffer(bencher: Bencher, length: usize) {
    let buffer = BitBuffer::from_iter((0..length).map(|i| i % 2 == 0));
    bencher
        .with_inputs(|| (&buffer, length / 2))
        .bench_refs(|(buffer, mid)| {
            let mid = *mid;
            buffer.slice(mid / 2..mid + mid / 2)
        });
}

#[cfg(not(codspeed))]
#[divan::bench(args = INPUT_SIZE)]
fn slice_arrow_buffer(bencher: Bencher, length: usize) {
    let buffer = Arrow(BooleanBuffer::from_iter((0..length).map(|i| i % 2 == 0)));
    bencher
        .with_inputs(|| (&buffer, length / 2))
        .bench_refs(|(buffer, mid)| {
            let mid = *mid;
            buffer.0.slice(mid / 2, mid / 2)
        });
}

#[divan::bench(args = INPUT_SIZE)]
fn true_count_vortex_buffer(bencher: Bencher, length: usize) {
    let buffer = BitBuffer::from_iter((0..length).map(true_count_pattern));

    bencher
        .with_inputs(|| &buffer)
        .bench_refs(|buffer| buffer.true_count())
}

#[divan::bench(args = INPUT_SIZE)]
fn true_count_arrow_buffer(bencher: Bencher, length: usize) {
    let buffer = Arrow(BooleanBuffer::from_iter(
        (0..length).map(true_count_pattern),
    ));

    buffer.0.count_set_bits();

    bencher
        .with_inputs(|| &buffer)
        .bench_refs(|buffer| buffer.0.count_set_bits());
}

#[divan::bench(args = INPUT_SIZE)]
fn bitwise_and_vortex_buffer(bencher: Bencher, length: usize) {
    let a = BitBuffer::from_iter((0..length).map(|i| i % 2 == 0));
    let b = BitBuffer::from_iter((0..length).map(|i| i % 3 == 0));
    bencher
        .with_inputs(|| (&a, &b))
        .bench_values(|(a, b)| a & b);
}

#[divan::bench(args = INPUT_SIZE)]
fn bitwise_and_arrow_buffer(bencher: Bencher, length: usize) {
    let a = Arrow(BooleanBuffer::from_iter((0..length).map(|i| i % 2 == 0)));
    let b = Arrow(BooleanBuffer::from_iter((0..length).map(|i| i % 3 == 0)));
    bencher
        .with_inputs(|| (&a, &b))
        .bench_refs(|(a, b)| &a.0 & &b.0);
}

#[divan::bench(args = INPUT_SIZE)]
fn bitwise_or_vortex_buffer(bencher: Bencher, length: usize) {
    let a = BitBuffer::from_iter((0..length).map(|i| i % 2 == 0));
    let b = BitBuffer::from_iter((0..length).map(|i| i % 3 == 0));
    bencher
        .with_inputs(|| (&a, &b))
        .bench_values(|(a, b)| a | b);
}

#[divan::bench(args = INPUT_SIZE)]
fn bitwise_or_arrow_buffer(bencher: Bencher, length: usize) {
    let a = Arrow(BooleanBuffer::from_iter((0..length).map(|i| i % 2 == 0)));
    let b = Arrow(BooleanBuffer::from_iter((0..length).map(|i| i % 3 == 0)));
    bencher
        .with_inputs(|| (&a, &b))
        .bench_refs(|(a, b)| &a.0 | &b.0);
}

#[divan::bench(args = INPUT_SIZE)]
fn bitwise_not_vortex_buffer(bencher: Bencher, length: usize) {
    bencher
        .with_inputs(|| BitBuffer::from_iter((0..length).map(|i| i % 2 == 0)))
        .bench_values(|buffer| !&buffer);
}

#[divan::bench(args = INPUT_SIZE)]
fn bitwise_not_vortex_buffer_mut(bencher: Bencher, length: usize) {
    bencher
        .with_inputs(|| BitBufferMut::from_iter((0..length).map(|i| i % 2 == 0)))
        .bench_values(|buffer| !buffer);
}

#[divan::bench(args = INPUT_SIZE)]
fn bitwise_not_arrow_buffer(bencher: Bencher, length: usize) {
    bencher
        .with_inputs(|| Arrow(BooleanBuffer::from_iter((0..length).map(|i| i % 2 == 0))))
        .bench_values(|buffer| !&buffer.0);
}

#[divan::bench(args = INPUT_SIZE)]
fn iter_vortex_buffer(bencher: Bencher, length: usize) {
    let buffer = BitBuffer::from_iter((0..length).map(|i| i % 2 == 0));
    bencher.with_inputs(|| &buffer).bench_refs(|buffer| {
        for value in buffer.iter() {
            divan::black_box(value);
        }
    });
}

#[divan::bench(args = INPUT_SIZE)]
fn iter_arrow_buffer(bencher: Bencher, length: usize) {
    let buffer = Arrow(BooleanBuffer::from_iter((0..length).map(|i| i % 2 == 0)));
    bencher.with_inputs(|| &buffer).bench_refs(|buffer| {
        for value in buffer.0.iter() {
            divan::black_box(value);
        }
    });
}

#[divan::bench(args = INPUT_SIZE)]
fn set_indices_vortex_buffer(bencher: Bencher, length: usize) {
    let buffer = BitBuffer::from_iter((0..length).map(|i| i % 2 == 0));
    bencher.with_inputs(|| &buffer).bench_refs(|buffer| {
        for idx in buffer.set_indices() {
            divan::black_box(idx);
        }
    });
}

#[divan::bench(args = INPUT_SIZE)]
fn set_indices_arrow_buffer(bencher: Bencher, length: usize) {
    let buffer = Arrow(BooleanBuffer::from_iter((0..length).map(|i| i % 2 == 0)));
    bencher.with_inputs(|| &buffer).bench_refs(|buffer| {
        for idx in buffer.0.set_indices() {
            divan::black_box(idx);
        }
    });
}
