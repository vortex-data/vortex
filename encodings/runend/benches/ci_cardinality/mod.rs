// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Experimental u32 dictionary lookup and run-end expansion on AArch64.
//! Inputs are pre-encoded dictionaries: dictionary discovery is not timed.

use std::arch::aarch64::*;
use std::hint::black_box;
use std::time::Duration;
use std::time::Instant;

use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use vortex_buffer::BufferMut;

use super::BATCH;
use super::baseline;
use super::current;
use super::exact_only;
use super::forced_exact;
use super::pr;

const ROUNDS: usize = 24;

#[inline(never)]
fn scalar(codes: &[u8], dictionary: &[u32]) -> BufferMut<u32> {
    let mut out = BufferMut::with_capacity(codes.len());
    for &code in codes {
        out.push(dictionary[usize::from(code)]);
    }
    out
}

/// Both kernels receive validated codes; omit bounds checks equally in the gather control.
///
/// # Safety
/// Every code must index the dictionary.
#[inline(never)]
unsafe fn scalar_unchecked(codes: &[u8], dictionary: &[u32]) -> BufferMut<u32> {
    let mut out = BufferMut::<u32>::with_capacity(codes.len());
    // SAFETY: codes are in bounds and capacity equals the number of writes.
    unsafe {
        for (i, &code) in codes.iter().enumerate() {
            out.as_mut_ptr()
                .add(i)
                .write(*dictionary.get_unchecked(usize::from(code)));
        }
        out.set_len(codes.len());
    }
    out
}

/// Transpose a small dictionary into byte planes, then gather 16 u32 values per iteration.
/// Table construction is deliberately inside the timed call.
///
/// # Safety
/// Every code must index a dictionary of 1..=32 elements. NEON must be available.
#[target_feature(enable = "neon")]
unsafe fn neon(codes: &[u8], dictionary: &[u32]) -> BufferMut<u32> {
    assert!(!dictionary.is_empty() && dictionary.len() <= 32);
    let mut planes = [[0u8; 32]; 4];
    for (i, value) in dictionary.iter().enumerate() {
        for (plane, byte) in planes.iter_mut().zip(value.to_le_bytes()) {
            plane[i] = byte;
        }
    }
    let mut out = BufferMut::<u32>::with_capacity(codes.len());
    let mut i = 0;
    // SAFETY: each plane is 32 readable bytes; vector loads/stores cover only complete
    // 16-element chunks. The scalar tail is in bounds by the caller's guarantee.
    unsafe {
        let low = planes.map(|plane| vld1q_u8(plane.as_ptr()));
        if dictionary.len() <= 16 {
            while i + 16 <= codes.len() {
                let code = vld1q_u8(codes.as_ptr().add(i));
                let bytes = uint8x16x4_t(
                    vqtbl1q_u8(low[0], code),
                    vqtbl1q_u8(low[1], code),
                    vqtbl1q_u8(low[2], code),
                    vqtbl1q_u8(low[3], code),
                );
                vst4q_u8(out.as_mut_ptr().add(i).cast(), bytes);
                i += 16;
            }
        } else {
            let high = planes.map(|plane| vld1q_u8(plane.as_ptr().add(16)));
            let tables = [
                uint8x16x2_t(low[0], high[0]),
                uint8x16x2_t(low[1], high[1]),
                uint8x16x2_t(low[2], high[2]),
                uint8x16x2_t(low[3], high[3]),
            ];
            while i + 16 <= codes.len() {
                let code = vld1q_u8(codes.as_ptr().add(i));
                let bytes = uint8x16x4_t(
                    vqtbl2q_u8(tables[0], code),
                    vqtbl2q_u8(tables[1], code),
                    vqtbl2q_u8(tables[2], code),
                    vqtbl2q_u8(tables[3], code),
                );
                vst4q_u8(out.as_mut_ptr().add(i).cast(), bytes);
                i += 16;
            }
        }
        for j in i..codes.len() {
            out.as_mut_ptr()
                .add(j)
                .write(*dictionary.get_unchecked(usize::from(codes[j])));
        }
        out.set_len(codes.len());
    }
    out
}

struct Input {
    ends: Vec<u64>,
    codes: Vec<u8>,
    dictionary: Vec<u32>,
    values: Vec<u32>,
    length: usize,
}

fn materialized(input: &Input) -> BufferMut<u32> {
    current(&input.ends, &input.values, input.length)
}

fn materialize_then_pr(input: &Input) -> BufferMut<u32> {
    // SAFETY: input construction validated all codes.
    let values = unsafe { scalar_unchecked(&input.codes, &input.dictionary) };
    current(&input.ends, values.as_slice(), input.length)
}

fn old_materialized(input: &Input) -> BufferMut<u32> {
    baseline(&input.ends, &input.values, input.length)
}

fn exact_materialized(input: &Input) -> BufferMut<u32> {
    exact_only(&input.ends, &input.values, input.length)
}

fn fused_pr(input: &Input) -> BufferMut<u32> {
    pr::decode_runs(
        input
            .ends
            .iter()
            .zip(&input.codes)
            .map(|(&end, &code)| (end as usize, input.dictionary[usize::from(code)])),
        input.length,
    )
}

fn fused_exact(input: &Input) -> BufferMut<u32> {
    forced_exact::decode_runs(
        input
            .ends
            .iter()
            .zip(&input.codes)
            .map(|(&end, &code)| (end as usize, input.dictionary[usize::from(code)])),
        input.length,
    )
}

fn fused_baseline(input: &Input) -> BufferMut<u32> {
    let mut out = BufferMut::<u32>::with_capacity(input.length);
    for (&end, &code) in input.ends.iter().zip(&input.codes) {
        let end = (end as usize).min(input.length);
        assert!(end >= out.len());
        let value = input.dictionary[usize::from(code)];
        // SAFETY: ends are monotonic and capped at the allocated capacity.
        unsafe { out.push_n_unchecked(value, end - out.len()) };
    }
    out
}

fn neon_pipeline(input: &Input) -> BufferMut<u32> {
    let codes = current(&input.ends, &input.codes, input.length);
    // SAFETY: codes were validated during input construction; local target enables NEON.
    unsafe { neon(codes.as_slice(), &input.dictionary) }
}

fn scalar_lookup(input: &Input) -> BufferMut<u32> {
    // SAFETY: codes were validated during input construction.
    unsafe { scalar_unchecked(&input.codes, &input.dictionary) }
}

fn neon_lookup(input: &Input) -> BufferMut<u32> {
    // SAFETY: codes were validated during input construction; local target enables NEON.
    unsafe { neon(&input.codes, &input.dictionary) }
}

type Kernel = fn(&Input) -> BufferMut<u32>;

fn measure(input: &Input, pattern: &str, variants: &[(&str, Kernel)], expected: &[u32]) {
    for (_, f) in variants {
        assert_eq!(f(input).as_slice(), expected);
        for _ in 0..100 {
            black_box(f(black_box(input)));
        }
    }
    for round in 0..ROUNDS {
        for offset in 0..variants.len() {
            let (variant, f) = variants[(round + offset) % variants.len()];
            let start = Instant::now();
            let mut count = 0;
            while start.elapsed() < Duration::from_millis(8) {
                for _ in 0..BATCH {
                    black_box(f(black_box(input)));
                }
                count += BATCH;
            }
            println!(
                "{},{},{pattern},{variant},{round},{:.3}",
                input.dictionary.len(),
                input.length,
                start.elapsed().as_nanos() as f64 / count as f64
            );
        }
    }
}

pub(super) fn run() {
    assert!(std::arch::is_aarch64_feature_detected!("neon"));
    assert!(cfg!(target_endian = "little"));
    println!("cardinality,length,pattern,variant,round,ns");
    for cardinality in [4, 8, 16, 32] {
        let mut rng = StdRng::seed_from_u64(42);
        let dictionary: Vec<u32> = (0..cardinality).map(|_| rng.random()).collect();
        assert!(
            dictionary
                .iter()
                .enumerate()
                .all(|(i, value)| !dictionary[..i].contains(value))
        );
        // Include every byte code and partial vector tails in correctness checks.
        for length in 0..=79 {
            let codes: Vec<u8> = (0..length).map(|i| (i % cardinality) as u8).collect();
            let expected = scalar(&codes, &dictionary);
            // SAFETY: codes index this dictionary and the target enables NEON.
            assert_eq!(
                unsafe { neon(&codes, &dictionary) }.as_slice(),
                expected.as_slice()
            );
        }
        for length in [1000, 10_000, 100_000] {
            let codes: Vec<u8> = (0..length)
                .map(|_| rng.random_range(0..cardinality) as u8)
                .collect();
            let expected = scalar(&codes, &dictionary);
            let input = Input {
                ends: Vec::new(),
                codes,
                dictionary: dictionary.clone(),
                values: Vec::new(),
                length,
            };
            measure(
                &input,
                "lookup",
                &[("scalar", scalar_lookup), ("neon", neon_lookup)],
                expected.as_slice(),
            );
        }
        for pattern in [
            "fixed1",
            "fixed4",
            "fixed16",
            "fixed1024",
            "alternating1_31",
            "one_long_per1024",
        ] {
            let length = 100_000;
            let mut input = Input {
                ends: Vec::new(),
                codes: Vec::new(),
                dictionary: dictionary.clone(),
                values: Vec::new(),
                length,
            };
            let mut expected = Vec::with_capacity(length);
            let mut pos = 0;
            let mut code = 0;
            while pos < length {
                let n = match pattern {
                    "fixed1" => 1,
                    "fixed4" => 4,
                    "fixed16" => 16,
                    "fixed1024" => 1024,
                    "alternating1_31" => {
                        if input.ends.len() % 2 == 0 {
                            1
                        } else {
                            31
                        }
                    }
                    _ => {
                        if input.ends.len() % 1024 == 1023 {
                            65536
                        } else {
                            1
                        }
                    }
                };
                let end = (pos + n).min(length);
                // Adjacent values differ, as they would after run-end compression.
                code = (code + rng.random_range(1..cardinality)) % cardinality;
                input.ends.push(end as u64);
                input.codes.push(code as u8);
                input.values.push(dictionary[code]);
                expected.extend(std::iter::repeat_n(dictionary[code], end - pos));
                pos = end;
            }
            measure(
                &input,
                pattern,
                &[
                    ("pr_materialized", materialized),
                    ("dict_materialize_pr", materialize_then_pr),
                    ("dict_fused_pr", fused_pr),
                    ("dict_neon_pipeline", neon_pipeline),
                    ("baseline_materialized", old_materialized),
                    ("exact_materialized", exact_materialized),
                    ("dict_fused_exact", fused_exact),
                    ("dict_fused_baseline", fused_baseline),
                ],
                &expected,
            );
        }
    }
}
