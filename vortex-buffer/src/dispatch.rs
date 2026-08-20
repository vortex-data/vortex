// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! One-time CPU-feature-based function dispatch.
//!
//! [`CpuKernel`] holds a function pointer chosen by a *selector* exactly once, on the
//! first call. Every later call is a relaxed atomic load, a never-taken predicted
//! branch, and an indirect call — measured indistinguishable from a direct call (see
//! `benches/cpu_dispatch.rs`).
//!
//! The selector is ordinary code that models both dispatch dimensions:
//!
//! * **Compile time** (x86_64 vs aarch64): one `#[cfg(target_arch = ...)]` block per
//!   architecture, each *returning early* with its chosen kernel.
//! * **Runtime** (AVX-512 vs BMI2 vs ...): an if-chain of feature probes inside that
//!   block.
//! * The **portable default** is the plain tail expression. Because the architecture
//!   arms return early, no `#[cfg(not(any(...)))]` negation is ever needed. An arm
//!   that returns unconditionally (e.g. NEON, architecturally guaranteed on aarch64)
//!   makes the tail unreachable on that architecture — wrap just the tail in an
//!   `#[allow(unreachable_code)]` block.
//!
//! When kernels are `unsafe` `#[target_feature]` functions, make `F` an
//! `unsafe fn(...)` pointer type: the bare kernel names then coerce directly (no
//! closure wrappers), and the one dispatched call is wrapped in `unsafe` with a
//! SAFETY comment stating that the selector probed the required features.
//!
//! Races are benign: the slot only ever holds valid function pointers of the same
//! type, and every candidate must compute the same result.
//!
//! # When NOT to use it
//!
//! Do not put the dispatched call inside a per-element hot loop: the indirect call
//! blocks inlining. Hoist the decision to the outermost per-buffer entry point and
//! monomorphize the loop instead, like the `Bmi2`/`Portable` type-parameter pattern in
//! `vortex-mask::intersect_by_rank`.
//!
//! For the same reason, gate on input size *before* [`get`](CpuKernel::get) when tiny
//! inputs are common: call the portable kernel directly below the size where SIMD pays
//! off, so those calls stay inlinable and skip the dispatch entirely (see
//! `count_ones_aligned`).

use core::mem::transmute_copy;
use core::ptr;
use core::sync::atomic::AtomicPtr;
use core::sync::atomic::Ordering;

/// A function pointer selected by CPU-feature detection once, on first use.
///
/// `F` must be a plain function-pointer type (`fn(...) -> ...`). The selector passed
/// to [`new`](Self::new) runs at most once per process (once per racing thread in the
/// worst case), on the first [`get`](Self::get), and its result is cached.
/// Non-capturing closures coerce to function pointers, so the selector can be written
/// inline in the `static`, like `LazyLock`.
///
/// # Example
///
/// ```
/// use vortex_buffer::CpuKernel;
///
/// /// Sums a slice, using the best kernel for the current CPU.
/// fn sum(values: &[u64]) -> u64 {
///     static KERNEL: CpuKernel<fn(&[u64]) -> u64> = CpuKernel::new(|| {
///         // Compile-time arm per architecture; runtime probes inside it.
///         #[cfg(target_arch = "x86_64")]
///         {
///             if std::arch::is_x86_feature_detected!("avx2") {
///                 // return |values| unsafe { sum_avx2(values) };
///             }
///         }
///         // Portable default: plain tail, no cfg(not(...)) needed.
///         |values| values.iter().sum()
///     });
///     KERNEL.get()(values)
/// }
///
/// assert_eq!(sum(&[1, 2, 3]), 6);
/// ```
pub struct CpuKernel<F> {
    selected: AtomicPtr<()>,
    select: fn() -> F,
}

impl<F: Copy> CpuKernel<F> {
    /// Create a kernel slot whose kernel is chosen by `select` on the first
    /// [`get`](Self::get).
    pub const fn new(select: fn() -> F) -> Self {
        assert!(
            size_of::<F>() == size_of::<*mut ()>(),
            "CpuKernel requires a function-pointer type"
        );
        Self {
            selected: AtomicPtr::new(ptr::null_mut()),
            select,
        }
    }

    /// Return the selected kernel, running the selector on the first call.
    ///
    /// Steady state is a relaxed load plus a never-taken predicted branch.
    #[inline]
    pub fn get(&self) -> F {
        let fn_ptr = self.selected.load(Ordering::Relaxed);
        if fn_ptr.is_null() {
            return self.select_slow();
        }
        // SAFETY: non-null values in `selected` are always the bits of an `F` stored
        // by `select_slow`, and `F` is pointer-sized (asserted in `new`). Function
        // pointers are never null, so the null sentinel stays unambiguous.
        unsafe { transmute_copy::<*mut (), F>(&fn_ptr) }
    }

    #[cold]
    fn select_slow(&self) -> F {
        let kernel = (self.select)();
        // SAFETY: `F` is pointer-sized (asserted in `new`); a bitwise copy into a raw
        // pointer preserves the function pointer for the transmute back in `get`.
        let fn_ptr = unsafe { transmute_copy::<F, *mut ()>(&kernel) };
        self.selected.store(fn_ptr, Ordering::Relaxed);
        kernel
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;

    use super::CpuKernel;

    static SELECT_CALLS: AtomicUsize = AtomicUsize::new(0);

    fn add_one(x: u64) -> u64 {
        static KERNEL: CpuKernel<fn(u64) -> u64> = CpuKernel::new(|| {
            SELECT_CALLS.fetch_add(1, Ordering::Relaxed);
            |x| x + 1
        });
        KERNEL.get()(x)
    }

    #[test]
    fn selects_once_then_dispatches() {
        assert_eq!(add_one(41), 42);
        assert_eq!(add_one(1), 2);
        assert_eq!(add_one(2), 3);
        assert_eq!(SELECT_CALLS.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn selector_early_returns_skip_the_default() {
        static KERNEL: CpuKernel<fn(u64) -> u64> = CpuKernel::new(|| {
            if 1 + 1 == 2 {
                return |x| x + 2;
            }
            |x| x + 100
        });
        assert_eq!(KERNEL.get()(0), 2);
    }

    type XorFn = fn(&[u8; 4], &mut [u8; 4]);

    #[test]
    fn supports_reference_arguments() {
        static KERNEL: CpuKernel<XorFn> = CpuKernel::new(|| {
            |input, output| {
                for (o, i) in output.iter_mut().zip(input) {
                    *o ^= *i;
                }
            }
        });
        let selected = KERNEL.get();
        let input = [1, 2, 3, 4];
        let mut output = [0, 0, 0, 0];
        selected(&input, &mut output);
        assert_eq!(output, input);
    }
}
