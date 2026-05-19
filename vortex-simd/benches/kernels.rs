//! Per-tier throughput for `i32` add and equality. Each tier is benched
//! against the same input so the speedup over scalar is visible directly.
//!
//! The `*_ops_dispatch` benches use the runtime-picked kernel and confirm
//! the fn-pointer table delivers the throughput of the underlying
//! specialized kernel.

use divan::counter::ItemsCount;
use divan::{Bencher, black_box};
use vortex_simd::cpu::{Tier, tier};
use vortex_simd::kernels::scalar;

fn main() {
    divan::main();
}

const N_I32: i32 = 65_536;
const N: usize = N_I32 as usize;

fn inputs() -> (Vec<i32>, Vec<i32>) {
    let lhs: Vec<i32> = (0..N_I32).collect();
    let rhs: Vec<i32> = (0..N_I32).map(|x| x.wrapping_add(1)).collect();
    (lhs, rhs)
}

// L1-resident size: 1024 i32 ≈ 4 KiB per buffer, so all three fit in L1d.
// At this size the kernel speed is compute-bound, not memory-bound, so the
// SIMD speedup is visible (unlike the 65K bench which saturates DRAM).
const SMALL_I32: i32 = 1024;
const SMALL: usize = SMALL_I32 as usize;

fn small_inputs() -> (Vec<i32>, Vec<i32>) {
    let lhs: Vec<i32> = (0..SMALL_I32).collect();
    let rhs: Vec<i32> = (0..SMALL_I32).map(|x| x.wrapping_add(1)).collect();
    (lhs, rhs)
}

// ----- add -----

#[divan::bench]
fn add_scalar(bencher: Bencher) {
    let (lhs, rhs) = inputs();
    let mut out = vec![0_i32; N];
    bencher
        .counter(ItemsCount::new(N))
        .bench_local(|| scalar::add_i32(black_box(&lhs), black_box(&rhs), black_box(&mut out)));
}

#[cfg(target_arch = "x86_64")]
#[divan::bench]
fn add_sse2(bencher: Bencher) {
    use vortex_simd::arch::x86_64 as x;
    if tier() < Tier::SSE42 {
        return;
    }
    let (lhs, rhs) = inputs();
    let mut out = vec![0_i32; N];
    // SAFETY: tier check confirmed SSE2.
    bencher.counter(ItemsCount::new(N)).bench_local(|| unsafe {
        x::add_i32_sse2(black_box(&lhs), black_box(&rhs), black_box(&mut out))
    });
}

#[cfg(target_arch = "x86_64")]
#[divan::bench]
fn add_avx2(bencher: Bencher) {
    use vortex_simd::arch::x86_64 as x;
    if tier() < Tier::AVX2 {
        return;
    }
    let (lhs, rhs) = inputs();
    let mut out = vec![0_i32; N];
    // SAFETY: tier check confirmed AVX2.
    bencher.counter(ItemsCount::new(N)).bench_local(|| unsafe {
        x::add_i32_avx2(black_box(&lhs), black_box(&rhs), black_box(&mut out))
    });
}

#[cfg(target_arch = "x86_64")]
#[divan::bench]
fn add_avx512(bencher: Bencher) {
    use vortex_simd::arch::x86_64 as x;
    if tier() < Tier::AVX512 {
        return;
    }
    let (lhs, rhs) = inputs();
    let mut out = vec![0_i32; N];
    // SAFETY: tier check confirmed AVX-512.
    bencher.counter(ItemsCount::new(N)).bench_local(|| unsafe {
        x::add_i32_avx512(black_box(&lhs), black_box(&rhs), black_box(&mut out))
    });
}

#[cfg(target_arch = "aarch64")]
#[divan::bench]
fn add_neon(bencher: Bencher) {
    use vortex_simd::arch::aarch64 as neon;
    if tier() < Tier::NEON {
        return;
    }
    let (lhs, rhs) = inputs();
    let mut out = vec![0_i32; N];
    // SAFETY: tier check confirmed NEON.
    bencher.counter(ItemsCount::new(N)).bench_local(|| unsafe {
        neon::add_i32_neon(black_box(&lhs), black_box(&rhs), black_box(&mut out))
    });
}

#[divan::bench]
fn add_ops_dispatch(bencher: Bencher) {
    let (lhs, rhs) = inputs();
    let mut out = vec![0_i32; N];
    let kernels = vortex_simd::kernels();
    bencher
        .counter(ItemsCount::new(N))
        .bench_local(|| (kernels.i32_add)(black_box(&lhs), black_box(&rhs), black_box(&mut out)));
}

// ----- add (L1-resident: compute-bound) -----

#[divan::bench]
fn add_small_scalar(bencher: Bencher) {
    let (lhs, rhs) = small_inputs();
    let mut out = vec![0_i32; SMALL];
    bencher
        .counter(ItemsCount::new(SMALL))
        .bench_local(|| scalar::add_i32(black_box(&lhs), black_box(&rhs), black_box(&mut out)));
}

#[cfg(target_arch = "x86_64")]
#[divan::bench]
fn add_small_avx2(bencher: Bencher) {
    use vortex_simd::arch::x86_64 as x;
    if tier() < Tier::AVX2 {
        return;
    }
    let (lhs, rhs) = small_inputs();
    let mut out = vec![0_i32; SMALL];
    // SAFETY: tier check confirmed AVX2.
    bencher
        .counter(ItemsCount::new(SMALL))
        .bench_local(|| unsafe {
            x::add_i32_avx2(black_box(&lhs), black_box(&rhs), black_box(&mut out))
        });
}

#[cfg(target_arch = "x86_64")]
#[divan::bench]
fn add_small_avx512(bencher: Bencher) {
    use vortex_simd::arch::x86_64 as x;
    if tier() < Tier::AVX512 {
        return;
    }
    let (lhs, rhs) = small_inputs();
    let mut out = vec![0_i32; SMALL];
    // SAFETY: tier check confirmed AVX-512.
    bencher
        .counter(ItemsCount::new(SMALL))
        .bench_local(|| unsafe {
            x::add_i32_avx512(black_box(&lhs), black_box(&rhs), black_box(&mut out))
        });
}

#[divan::bench]
fn add_small_ops_dispatch(bencher: Bencher) {
    let (lhs, rhs) = small_inputs();
    let mut out = vec![0_i32; SMALL];
    let kernels = vortex_simd::kernels();
    bencher
        .counter(ItemsCount::new(SMALL))
        .bench_local(|| (kernels.i32_add)(black_box(&lhs), black_box(&rhs), black_box(&mut out)));
}

// ----- eq -----

#[divan::bench]
fn eq_scalar(bencher: Bencher) {
    let (lhs, rhs) = inputs();
    let mut out = vec![0_u8; N / 8];
    bencher
        .counter(ItemsCount::new(N))
        .bench_local(|| scalar::eq_i32(black_box(&lhs), black_box(&rhs), black_box(&mut out)));
}

#[cfg(target_arch = "x86_64")]
#[divan::bench]
fn eq_sse2(bencher: Bencher) {
    use vortex_simd::arch::x86_64 as x;
    if tier() < Tier::SSE42 {
        return;
    }
    let (lhs, rhs) = inputs();
    let mut out = vec![0_u8; N / 8];
    // SAFETY: tier check confirmed SSE2.
    bencher.counter(ItemsCount::new(N)).bench_local(|| unsafe {
        x::eq_i32_sse2(black_box(&lhs), black_box(&rhs), black_box(&mut out))
    });
}

#[cfg(target_arch = "x86_64")]
#[divan::bench]
fn eq_avx2(bencher: Bencher) {
    use vortex_simd::arch::x86_64 as x;
    if tier() < Tier::AVX2 {
        return;
    }
    let (lhs, rhs) = inputs();
    let mut out = vec![0_u8; N / 8];
    // SAFETY: tier check confirmed AVX2.
    bencher.counter(ItemsCount::new(N)).bench_local(|| unsafe {
        x::eq_i32_avx2(black_box(&lhs), black_box(&rhs), black_box(&mut out))
    });
}

#[cfg(target_arch = "x86_64")]
#[divan::bench]
fn eq_avx512(bencher: Bencher) {
    use vortex_simd::arch::x86_64 as x;
    if tier() < Tier::AVX512 {
        return;
    }
    let (lhs, rhs) = inputs();
    let mut out = vec![0_u8; N / 8];
    // SAFETY: tier check confirmed AVX-512.
    bencher.counter(ItemsCount::new(N)).bench_local(|| unsafe {
        x::eq_i32_avx512(black_box(&lhs), black_box(&rhs), black_box(&mut out))
    });
}

#[cfg(target_arch = "aarch64")]
#[divan::bench]
fn eq_neon(bencher: Bencher) {
    use vortex_simd::arch::aarch64 as neon;
    if tier() < Tier::NEON {
        return;
    }
    let (lhs, rhs) = inputs();
    let mut out = vec![0_u8; N / 8];
    // SAFETY: tier check confirmed NEON.
    bencher.counter(ItemsCount::new(N)).bench_local(|| unsafe {
        neon::eq_i32_neon(black_box(&lhs), black_box(&rhs), black_box(&mut out))
    });
}

#[divan::bench]
fn eq_ops_dispatch(bencher: Bencher) {
    let (lhs, rhs) = inputs();
    let mut out = vec![0_u8; N / 8];
    let kernels = vortex_simd::kernels();
    bencher
        .counter(ItemsCount::new(N))
        .bench_local(|| (kernels.i32_eq)(black_box(&lhs), black_box(&rhs), black_box(&mut out)));
}
