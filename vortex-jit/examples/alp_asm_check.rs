// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Pins the two Rust impls behind `#[unsafe(no_mangle)]` so we can extract
//! them from the binary with `objdump --disassemble=<symbol>`.
//!
//! Build:  cargo rustc --release -p vortex-jit --example alp_asm_check
//! Inspect:
//!   objdump --disassemble=vortex_style_alp_decode \
//!           --no-show-raw-insn target/release/examples/alp_asm_check
//!   objdump --disassemble=idealized_alp_decode \
//!           --no-show-raw-insn target/release/examples/alp_asm_check

const F10: [f32; 22] = [
    1.0, 10.0, 100.0, 1_000.0, 10_000.0, 100_000.0, 1_000_000.0, 1e7, 1e8, 1e9, 1e10, 1e11,
    1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18, 1e19, 1e20, 1e21,
];
const IF10: [f32; 22] = [
    1.0, 0.1, 0.01, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13,
    1e-14, 1e-15, 1e-16, 1e-17, 1e-18, 1e-19, 1e-20, 1e-21,
];

/// Per-element helper, kept opaque to the optimizer via `#[inline(never)]`.
#[inline(never)]
#[unsafe(no_mangle)]
pub fn decode_single_alp(encoded: i32, e: u8, f: u8) -> f32 {
    (encoded as f32) * F10[f as usize] * IF10[e as usize]
}

/// Vortex-style: `iter_mut().for_each` + non-inlined helper. Mirrors
/// `encodings/alp/src/alp/mod.rs:253-261`. LLVM's loop vectorizer can't
/// see through the call boundary to fuse load + cvt + mul + store.
#[inline(never)]
#[unsafe(no_mangle)]
pub fn vortex_style_alp_decode(input: &[i32], output: &mut [f32], e: u8, f: u8) {
    output
        .iter_mut()
        .zip(input.iter())
        .for_each(|(o, &x)| {
            *o = decode_single_alp(x, e, f);
        });
}

#[inline(always)]
fn decode_single_alp_inlined(encoded: i32, e: u8, f: u8) -> f32 {
    (encoded as f32) * F10[f as usize] * IF10[e as usize]
}

/// Identical to `vortex_style_alp_decode` except the helper is `#[inline(always)]`.
/// Demonstrates that the ONLY thing blocking LLVM autovec is the call boundary.
#[inline(never)]
#[unsafe(no_mangle)]
pub fn inlined_helper_alp_decode(input: &[i32], output: &mut [f32], e: u8, f: u8) {
    output
        .iter_mut()
        .zip(input.iter())
        .for_each(|(o, &x)| {
            *o = decode_single_alp_inlined(x, e, f);
        });
}

/// Idealized: tight loop with the scale hoisted as a single literal. This
/// is what a competent engineer would write if hand-tuning, and what LLVM's
/// autovec will lift to `vcvtdq2ps` + `vmulps` at the host's native width.
#[inline(never)]
#[unsafe(no_mangle)]
pub fn idealized_alp_decode(input: &[i32], output: &mut [f32], scale: f32) {
    for i in 0..input.len() {
        output[i] = (input[i] as f32) * scale;
    }
}

fn main() {
    // Force-link both symbols so neither gets DCE'd.
    let input: Vec<i32> = (0..1024).collect();
    let mut output = vec![0f32; 1024];
    vortex_style_alp_decode(&input, &mut output, 2, 0);
    std::hint::black_box(&output);
    inlined_helper_alp_decode(&input, &mut output, 2, 0);
    std::hint::black_box(&output);
    idealized_alp_decode(&input, &mut output, 0.01);
    std::hint::black_box(&output);
}
