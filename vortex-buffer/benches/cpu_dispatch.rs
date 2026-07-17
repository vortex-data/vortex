// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Overhead comparison of one-time CPU-feature dispatch mechanisms.
//!
//! Every variant ends in a call to the same `#[inline(never)]` kernel, so the
//! difference between rows is pure dispatch overhead:
//!
//! * `direct`: plain call — the floor.
//! * `cpu_kernel`: [`CpuKernel`] — relaxed load + null check + indirect call.
//! * `cpu_kernel_unconditional`: a selector with no probes (the aarch64 shape) —
//!   same steady state as `cpu_kernel`.
//! * `lazy_lock`: `std::sync::LazyLock<fn>` — Once-state check + value load +
//!   indirect call.
//! * `feature_detect_each_call_1/_3`: the pre-`CpuKernel` pattern — 1 or 3
//!   `is_x86_feature_detected!` cached lookups per call, then a direct call.

use std::sync::LazyLock;

use divan::Bencher;
use divan::black_box;
use divan::counter::ItemsCount;
use vortex_buffer::CpuKernel;

fn main() {
    // Resolve every dispatcher and warm the std feature-detection cache so no
    // benchmark iteration pays a one-time probe.
    let seed = black_box(7);
    let _ = via_direct(1, seed)
        + via_cpu_kernel(1, seed)
        + via_cpu_kernel_unconditional(1, seed)
        + via_lazy_lock(1, seed);
    #[cfg(target_arch = "x86_64")]
    let _ = via_feature_detect_1(1, seed) + via_feature_detect_3(1, seed);

    divan::main();
}

type Kernel = fn(u64, u64) -> u64;

#[inline(never)]
fn kernel_a(x: u64, seed: u64) -> u64 {
    x.wrapping_mul(31).wrapping_add(seed)
}

#[inline(never)]
fn kernel_b(x: u64, seed: u64) -> u64 {
    x.wrapping_mul(33).wrapping_add(seed)
}

fn has_bmi2() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::arch::is_x86_feature_detected!("bmi2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

// ── dispatch variants ───────────────────────────────────────────────────

#[inline(never)]
fn via_direct(x: u64, seed: u64) -> u64 {
    kernel_a(x, seed)
}

#[cfg(target_arch = "x86_64")]
#[inline(never)]
fn via_feature_detect_1(x: u64, seed: u64) -> u64 {
    if std::arch::is_x86_feature_detected!("bmi2") {
        kernel_a(x, seed)
    } else {
        kernel_b(x, seed)
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(never)]
fn via_feature_detect_3(x: u64, seed: u64) -> u64 {
    // Non-baseline features that are present on any recent x86_64 part, so all
    // three cached lookups actually execute (mirrors the old select_in_chunk).
    if std::arch::is_x86_feature_detected!("avx")
        && std::arch::is_x86_feature_detected!("avx2")
        && std::arch::is_x86_feature_detected!("bmi2")
    {
        kernel_a(x, seed)
    } else {
        kernel_b(x, seed)
    }
}

static LAZY: LazyLock<Kernel> = LazyLock::new(|| if has_bmi2() { kernel_a } else { kernel_b });

#[inline(never)]
fn via_lazy_lock(x: u64, seed: u64) -> u64 {
    (*LAZY)(x, seed)
}

static CPU_KERNEL: CpuKernel<Kernel> =
    CpuKernel::new(|| if has_bmi2() { kernel_a } else { kernel_b });

#[inline(never)]
fn via_cpu_kernel(x: u64, seed: u64) -> u64 {
    CPU_KERNEL.get()(x, seed)
}

static CPU_KERNEL_UNCONDITIONAL: CpuKernel<Kernel> = CpuKernel::new(|| kernel_a);

#[inline(never)]
fn via_cpu_kernel_unconditional(x: u64, seed: u64) -> u64 {
    CPU_KERNEL_UNCONDITIONAL.get()(x, seed)
}

// ── benches ─────────────────────────────────────────────────────────────

const CALLS: u64 = 1024;

fn bench_via(bencher: Bencher, via: fn(u64, u64) -> u64) {
    bencher.counter(ItemsCount::new(CALLS)).bench(|| {
        let mut acc = 0u64;
        for i in 0..CALLS {
            acc = acc.wrapping_add(via(black_box(i), black_box(7)));
        }
        acc
    })
}

#[divan::bench]
fn direct(bencher: Bencher) {
    bench_via(bencher, via_direct);
}

#[divan::bench]
fn cpu_kernel_unconditional(bencher: Bencher) {
    bench_via(bencher, via_cpu_kernel_unconditional);
}

#[divan::bench]
fn cpu_kernel(bencher: Bencher) {
    bench_via(bencher, via_cpu_kernel);
}

#[divan::bench]
fn lazy_lock(bencher: Bencher) {
    bench_via(bencher, via_lazy_lock);
}

#[cfg(target_arch = "x86_64")]
#[divan::bench]
fn feature_detect_each_call_1(bencher: Bencher) {
    bench_via(bencher, via_feature_detect_1);
}

#[cfg(target_arch = "x86_64")]
#[divan::bench]
fn feature_detect_each_call_3(bencher: Bencher) {
    bench_via(bencher, via_feature_detect_3);
}
