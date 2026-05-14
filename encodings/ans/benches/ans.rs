// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Microbenchmarks for `AnsArray` (P5 of the layered pco stack).
//!
//! Two scenarios over `N = 1_000_000` `u8` symbols are exercised:
//!
//! - **A — Zipf-skewed** — alphabet of 16 with `p_k ∝ 1/(k+1)`. Highly
//!   compressible; entropy is well below 4 bits/symbol.
//! - **B — uniform random** — alphabet of 16 with uniform probability.
//!   Lowest compression headroom (entropy = 4 bits/symbol).
//!
//! The bench prints compression ratios up front and runs encode and
//! decode for each scenario. `scalar_at` is not benched because the
//! tANS decoder is sequential by design: each call decodes the full
//! stream.

#![expect(clippy::unwrap_used)]
#![expect(clippy::cast_possible_truncation)]
#![expect(clippy::cast_precision_loss)]

use divan::Bencher;
use divan::counter::BytesCount;
use divan::counter::ItemsCount;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::SmallRng;
use vortex_ans::Ans;
use vortex_ans::AnsArrayExt;
use vortex_array::IntoArray;
use vortex_array::LEGACY_SESSION;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;

const N: usize = 1_000_000;
const SEED: u64 = 42;
const ANS_SIZE_LOG: u8 = 12;

fn main() {
    print_compression_ratio::<A>("A Zipf-skewed (alphabet=16)");
    print_compression_ratio::<B>("B uniform random (alphabet=16)");
    divan::main();
}

// ----- Scenarios -------------------------------------------------------------

trait Scenario {
    fn build() -> Buffer<u8>;
}

/// Scenario A: Zipf-skewed alphabet of 16. `p_k ∝ 1/(k+1)`.
struct A;
/// Scenario B: uniform random alphabet of 16.
struct B;

impl Scenario for A {
    fn build() -> Buffer<u8> {
        let mut rng = SmallRng::seed_from_u64(SEED);
        let weights: Vec<f64> = (0..16).map(|k| 1.0 / (k as f64 + 1.0)).collect();
        let total: f64 = weights.iter().sum();
        let cdf: Vec<f64> = weights
            .iter()
            .scan(0.0, |s, w| {
                *s += w / total;
                Some(*s)
            })
            .collect();
        let mut out = BufferMut::<u8>::with_capacity(N);
        for _ in 0..N {
            let r: f64 = rng.random();
            let mut sym = 15u8;
            for (k, &c) in cdf.iter().enumerate() {
                if r < c {
                    sym = k as u8;
                    break;
                }
            }
            out.push(sym);
        }
        out.freeze()
    }
}

impl Scenario for B {
    fn build() -> Buffer<u8> {
        let mut rng = SmallRng::seed_from_u64(SEED ^ 0xB);
        let mut out = BufferMut::<u8>::with_capacity(N);
        for _ in 0..N {
            out.push(rng.random::<u8>() & 0x0F);
        }
        out.freeze()
    }
}

fn make_primitive<S: Scenario>() -> PrimitiveArray {
    PrimitiveArray::new(S::build(), Validity::NonNullable)
}

// ----- Compression-ratio summary -------------------------------------------

fn print_compression_ratio<S: Scenario>(tag: &str) {
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let parray = make_primitive::<S>();
    let encoded = Ans::encode(parray.as_view(), ANS_SIZE_LOG, &mut ctx).unwrap();
    let raw = N as f64;
    let ans_bytes = encoded.encoded().len() as f64;
    let ratio = raw / ans_bytes.max(1.0);
    println!(
        "compression_ratio[{tag}]: raw={:.2} MiB  ans={:.2} MiB  ratio={:.2}x",
        raw / (1024.0 * 1024.0),
        ans_bytes / (1024.0 * 1024.0),
        ratio,
    );
}

// ----- Benchmarks ----------------------------------------------------------

#[divan::bench(types = [A, B])]
fn encode_ans<S: Scenario>(bencher: Bencher) {
    let parray = make_primitive::<S>();
    bencher
        .counter(BytesCount::new(N))
        .counter(ItemsCount::new(N))
        .bench_local(|| {
            let mut ctx = LEGACY_SESSION.create_execution_ctx();
            Ans::encode(parray.as_view(), ANS_SIZE_LOG, &mut ctx).unwrap()
        });
}

#[divan::bench(types = [A, B])]
fn decode_ans<S: Scenario>(bencher: Bencher) {
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let parray = make_primitive::<S>();
    let encoded = Ans::encode(parray.as_view(), ANS_SIZE_LOG, &mut ctx).unwrap();
    let arr = encoded.into_array();
    bencher
        .counter(BytesCount::new(N))
        .counter(ItemsCount::new(N))
        .bench_local(|| {
            let mut ctx = LEGACY_SESSION.create_execution_ctx();
            arr.clone().execute::<PrimitiveArray>(&mut ctx).unwrap()
        });
}
