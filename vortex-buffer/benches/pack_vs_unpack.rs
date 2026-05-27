// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compare two strategies for handling validity in `try_map_with_mask`:
//!
//! 1. **Unpack the mask** — closure consults `valid` per-lane. Null lanes are
//!    short-circuited inside the closure (return `Some(default)` immediately),
//!    so the checked operation never runs with garbage. The kernel still does
//!    its `fail_bits & src_chunk` post-filter, but it's a no-op because the
//!    closure already produced `Some` at null lanes.
//!
//! 2. **Pack and filter** — closure ignores `_valid`. The checked operation
//!    runs at every lane, including null lanes (where it may produce `None`
//!    on garbage). The kernel's post-loop `fail_bits & src_chunk` filter
//!    drops those null-lane fails. LLVM DCEs the per-lane mask extract since
//!    the closure doesn't consult `valid`.
//!
//! Two ops × two strategies = four vortex benches, plus arrow baselines.
//!
//! - `widen_u16_u32_*` — statically-infallible widening cast. `NumCast::from`
//!   always returns `Some`; LLVM proves it and strips fail-tracking entirely.
//! - `checked_add_u32_*` — genuinely fallible: `u32 + u32` can overflow.

#![expect(clippy::unwrap_used)]

use std::mem::MaybeUninit;
use std::sync::Arc;

use arrow_arith::numeric::add;
use arrow_array::Datum;
use arrow_array::UInt16Array;
use arrow_array::UInt32Array;
use arrow_buffer::NullBuffer;
use arrow_buffer::ScalarBuffer;
use arrow_cast::CastOptions;
use arrow_cast::cast_with_options;
use arrow_schema::DataType;
use divan::Bencher;
use num_traits::NumCast;
use rand::SeedableRng;
use rand::prelude::*;
use rand::rngs::StdRng;
use vortex_buffer::BitBuffer;
use vortex_buffer::BitBufferMut;
use vortex_buffer::Buffer;
use vortex_buffer::lane_ops_indexed::LaneZip;
use vortex_buffer::lane_ops_indexed::try_map_with_mask;

fn main() {
    divan::main();
}

const SIZES: &[usize] = &[4_096, 65_536, 1_048_576];

struct Fixture {
    values_u16: Buffer<u16>,
    lhs_u32: Buffer<u32>,
    rhs_u32: Buffer<u32>,
    mask: BitBuffer,
    arrow_u16: UInt16Array,
    arrow_lhs: Arc<UInt32Array>,
    arrow_rhs: Arc<UInt32Array>,
}

fn fixture(n: usize) -> Fixture {
    let mut rng = StdRng::seed_from_u64(0xC0DE_BEEF);
    // Bounded so `u16 + u16` (as u32) and `u32 + u32` never overflow u32.
    // Both strategies succeed; we measure success-path perf.
    let raw_lhs: Vec<u32> = (0..n)
        .map(|_| rng.random_range(0..(u32::MAX / 2)))
        .collect();
    let raw_rhs: Vec<u32> = (0..n)
        .map(|_| rng.random_range(0..(u32::MAX / 2)))
        .collect();
    let raw_valid: Vec<bool> = (0..n).map(|_| rng.random_bool(0.8)).collect();

    #[expect(clippy::cast_possible_truncation)]
    let values_u16: Buffer<u16> = raw_lhs.iter().map(|&v| v as u16).collect();
    let lhs_u32: Buffer<u32> = raw_lhs.iter().copied().collect();
    let rhs_u32: Buffer<u32> = raw_rhs.iter().copied().collect();

    let mask = {
        let mut m = BitBufferMut::with_capacity(n);
        for &v in &raw_valid {
            m.append(v);
        }
        m.freeze()
    };

    #[expect(clippy::cast_possible_truncation)]
    let arrow_u16 = UInt16Array::new(
        ScalarBuffer::from(raw_lhs.iter().map(|&v| v as u16).collect::<Vec<u16>>()),
        Some(NullBuffer::from(raw_valid.clone())),
    );
    let arrow_lhs = Arc::new(UInt32Array::new(
        ScalarBuffer::from(raw_lhs),
        Some(NullBuffer::from(raw_valid.clone())),
    ));
    let arrow_rhs = Arc::new(UInt32Array::new(
        ScalarBuffer::from(raw_rhs),
        Some(NullBuffer::from(raw_valid)),
    ));

    Fixture {
        values_u16,
        lhs_u32,
        rhs_u32,
        mask,
        arrow_u16,
        arrow_lhs,
        arrow_rhs,
    }
}

fn uninit_out<T>(n: usize) -> Vec<MaybeUninit<T>> {
    let mut out = Vec::with_capacity(n);
    // SAFETY: a `MaybeUninit<T>` does not require initialization.
    unsafe { out.set_len(n) };
    out
}

const CAST_OPTS_CHECKED: CastOptions<'static> = CastOptions {
    safe: false,
    format_options: arrow_cast::display::FormatOptions::new(),
};

// -----------------------------------------------------------------------------
// Widening cast u16 → u32 (statically infallible). NumCast::from never returns
// None for widening, so the failure path is dead in both strategies.
// -----------------------------------------------------------------------------

/// Strategy 1 (unpack mask): closure consults `valid`, short-circuits at null
/// lanes. For widening the short-circuit is dead anyway (no failure possible).
#[divan::bench(args = SIZES)]
fn widen_u16_u32_unpack_mask(bencher: Bencher, n: usize) {
    let f = fixture(n);
    bencher
        .with_inputs(|| (f.values_u16.clone(), f.mask.clone(), uninit_out::<u32>(n)))
        .bench_values(|(values, mask, mut out)| {
            try_map_with_mask(values.as_slice(), &mask, out.as_mut_slice(), |v, valid| {
                if !valid {
                    return Some(0u32);
                }
                <u32 as NumCast>::from(v)
            })
            .unwrap();
            out
        });
}

/// Strategy 2 (pack and filter): closure ignores `_valid`. LLVM DCEs the
/// per-lane mask extract; post-loop `& src_chunk` would filter null-lane fails
/// (none happen for widening).
#[divan::bench(args = SIZES)]
fn widen_u16_u32_pack_and_filter(bencher: Bencher, n: usize) {
    let f = fixture(n);
    bencher
        .with_inputs(|| (f.values_u16.clone(), f.mask.clone(), uninit_out::<u32>(n)))
        .bench_values(|(values, mask, mut out)| {
            try_map_with_mask(values.as_slice(), &mask, out.as_mut_slice(), |v, _valid| {
                <u32 as NumCast>::from(v)
            })
            .unwrap();
            out
        });
}

#[divan::bench(args = SIZES)]
fn widen_u16_u32_arrow(bencher: Bencher, _n: usize) {
    let f = fixture(_n);
    bencher
        .with_inputs(|| f.arrow_u16.clone())
        .bench_refs(|arr| cast_with_options(arr, &DataType::UInt32, &CAST_OPTS_CHECKED).unwrap());
}

// -----------------------------------------------------------------------------
// Checked add u32 + u32 → u32 (genuinely fallible). LaneZip(lhs, rhs) drives
// two-input lanewise.
// -----------------------------------------------------------------------------

/// Strategy 1 (unpack mask): closure short-circuits null lanes; `checked_add`
/// only runs at valid lanes.
#[divan::bench(args = SIZES)]
fn checked_add_u32_unpack_mask(bencher: Bencher, n: usize) {
    let f = fixture(n);
    bencher
        .with_inputs(|| {
            (
                f.lhs_u32.clone(),
                f.rhs_u32.clone(),
                f.mask.clone(),
                uninit_out::<u32>(n),
            )
        })
        .bench_values(|(lhs, rhs, mask, mut out)| {
            try_map_with_mask(
                LaneZip::new(lhs.as_slice(), rhs.as_slice()),
                &mask,
                out.as_mut_slice(),
                |(a, b), valid| {
                    if !valid {
                        return Some(0u32);
                    }
                    a.checked_add(b)
                },
            )
            .unwrap();
            out
        });
}

/// Strategy 2 (pack and filter): `checked_add` runs at every lane (including
/// null lanes with garbage values); kernel's `fail_bits & src_chunk` post-filter
/// drops any null-lane fails.
#[divan::bench(args = SIZES)]
fn checked_add_u32_pack_and_filter(bencher: Bencher, n: usize) {
    let f = fixture(n);
    bencher
        .with_inputs(|| {
            (
                f.lhs_u32.clone(),
                f.rhs_u32.clone(),
                f.mask.clone(),
                uninit_out::<u32>(n),
            )
        })
        .bench_values(|(lhs, rhs, mask, mut out)| {
            try_map_with_mask(
                LaneZip::new(lhs.as_slice(), rhs.as_slice()),
                &mask,
                out.as_mut_slice(),
                |(a, b), _valid| a.checked_add(b),
            )
            .unwrap();
            out
        });
}

// Asm-extraction helpers: `#[unsafe(no_mangle)] #[inline(never)]` so a single
// `cargo rustc --emit=asm` produces clearly-labeled symbols to diff.

#[unsafe(no_mangle)]
#[inline(never)]
pub fn asm_add_unpack_branchy(
    lhs: &[u32],
    rhs: &[u32],
    mask: &BitBuffer,
    out: &mut [MaybeUninit<u32>],
) -> Result<(), usize> {
    try_map_with_mask(
        LaneZip::new(lhs, rhs),
        mask,
        out,
        |(a, b), valid| {
            if !valid {
                return Some(0u32);
            }
            a.checked_add(b)
        },
    )
}

#[unsafe(no_mangle)]
#[inline(never)]
pub fn asm_add_unpack_branchless(
    lhs: &[u32],
    rhs: &[u32],
    mask: &BitBuffer,
    out: &mut [MaybeUninit<u32>],
) -> Result<(), usize> {
    try_map_with_mask(
        LaneZip::new(lhs, rhs),
        mask,
        out,
        |(a, b), valid| {
            // Compute first, then select. No early-return; LLVM may if-convert.
            let r = a.checked_add(b);
            if valid { r } else { Some(0u32) }
        },
    )
}

#[unsafe(no_mangle)]
#[inline(never)]
pub fn asm_add_unpack_multiply(
    lhs: &[u32],
    rhs: &[u32],
    mask: &BitBuffer,
    out: &mut [MaybeUninit<u32>],
) -> Result<(), usize> {
    try_map_with_mask(
        LaneZip::new(lhs, rhs),
        mask,
        out,
        |(a, b), valid| {
            // Neutralize null lanes via multiply (BIC); checked_add runs unconditionally.
            let m = valid as u32;
            (a * m).checked_add(b * m)
        },
    )
}

#[unsafe(no_mangle)]
#[inline(never)]
pub fn asm_add_pack_filter(
    lhs: &[u32],
    rhs: &[u32],
    mask: &BitBuffer,
    out: &mut [MaybeUninit<u32>],
) -> Result<(), usize> {
    try_map_with_mask(
        LaneZip::new(lhs, rhs),
        mask,
        out,
        |(a, b), _valid| a.checked_add(b),
    )
}

/// Branchless-multiply variant of unpack_mask: scale lhs/rhs by `valid as u32` so
/// the checked op runs at every lane (with zeros at null lanes — never overflows)
/// and the kernel's post-loop `& src_chunk` filter still applies.
#[divan::bench(args = SIZES)]
fn checked_add_u32_unpack_multiply(bencher: Bencher, n: usize) {
    let f = fixture(n);
    bencher
        .with_inputs(|| {
            (
                f.lhs_u32.clone(),
                f.rhs_u32.clone(),
                f.mask.clone(),
                uninit_out::<u32>(n),
            )
        })
        .bench_values(|(lhs, rhs, mask, mut out)| {
            try_map_with_mask(
                LaneZip::new(lhs.as_slice(), rhs.as_slice()),
                &mask,
                out.as_mut_slice(),
                |(a, b), valid| {
                    let m = valid as u32;
                    (a * m).checked_add(b * m)
                },
            )
            .unwrap();
            out
        });
}

/// Compute-first-then-select variant of unpack_mask: removes the early `return`,
/// keeps the `valid` consult per-lane. Tests whether LLVM if-converts when both
/// branches are pure expressions.
#[divan::bench(args = SIZES)]
fn checked_add_u32_unpack_branchless(bencher: Bencher, n: usize) {
    let f = fixture(n);
    bencher
        .with_inputs(|| {
            (
                f.lhs_u32.clone(),
                f.rhs_u32.clone(),
                f.mask.clone(),
                uninit_out::<u32>(n),
            )
        })
        .bench_values(|(lhs, rhs, mask, mut out)| {
            try_map_with_mask(
                LaneZip::new(lhs.as_slice(), rhs.as_slice()),
                &mask,
                out.as_mut_slice(),
                |(a, b), valid| {
                    let r = a.checked_add(b);
                    if valid { r } else { Some(0u32) }
                },
            )
            .unwrap();
            out
        });
}

#[divan::bench(args = SIZES)]
fn checked_add_u32_arrow(bencher: Bencher, _n: usize) {
    let f = fixture(_n);
    bencher
        .with_inputs(|| (f.arrow_lhs.clone(), f.arrow_rhs.clone()))
        .bench_refs(|(lhs, rhs)| {
            let lhs_datum: &dyn Datum = lhs.as_ref();
            let rhs_datum: &dyn Datum = rhs.as_ref();
            add(lhs_datum, rhs_datum).unwrap()
        });
}
