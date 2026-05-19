// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! JIT fusion benchmarks at 65k elements.
//!
//! Two pipelines, each with three measurement configurations:
//!   - `*_staged_rust`: hand-written Rust that mirrors staged (non-fused)
//!     behavior — stage 1 materializes into a temp buffer; stage 2 reads it.
//!   - `*_jit_warm`: JIT-compiled fused kernel with compile cost outside the
//!     timed region.
//!   - `*_jit_cold`: build a fresh compiler + compile + run, all timed.
//!
//! Pipelines:
//!   - `for_add_only`: LoadIn -> ForAdd -> StoreOut. The simplest fusion case.
//!     Same shape as fastlanes' FFoR over a primitive child. Pure SIMD-friendly.
//!   - `delta_for`: LoadIn -> DeltaPrefixSum -> ForAdd -> StoreOut. Delta is
//!     a serial dependency that doesn't vectorize cleanly; this is the harder
//!     case where v1's extract/insert overhead in Delta still costs us.

use std::sync::Arc;

use divan::Bencher;
use divan::counter::{BytesCount, ItemsCount};
use mimalloc::MiMalloc;
use vortex_jit::stages::{
    AlpDecode, BitPackedLoad, DeltaPrefixSum, DictLookup, ForAdd, LoadIn, StoreOut, pack_dense,
};
use vortex_jit::{Compiler, PType, Pipeline};

#[global_allocator]
static ALLOC: MiMalloc = MiMalloc;

const NUM_VALUES: usize = 65_536;
// "Block" here is the per-emit() chunk count. Empirically tuned: too small
// (16-32) adds loop overhead; too large (>=128) spills SSE registers across
// stage boundaries. 64 is the sweet spot for short i32 pipelines on x86_64.
const BLOCK: usize = 64;
const N_BLOCKS: usize = NUM_VALUES / BLOCK;
const REFERENCE: i32 = 42;

fn build_for_add_pipeline() -> Pipeline {
    let mut p = Pipeline::new(PType::I32, BLOCK);
    p.push(Arc::new(LoadIn { ptype: PType::I32 })).unwrap();
    p.push(Arc::new(ForAdd {
        ptype: PType::I32,
        reference: REFERENCE as i64,
    }))
    .unwrap();
    p.push(Arc::new(StoreOut { ptype: PType::I32 })).unwrap();
    p
}

fn build_delta_for_pipeline() -> Pipeline {
    let mut p = Pipeline::new(PType::I32, BLOCK);
    p.push(Arc::new(LoadIn { ptype: PType::I32 })).unwrap();
    p.push(Arc::new(DeltaPrefixSum { ptype: PType::I32 })).unwrap();
    p.push(Arc::new(ForAdd {
        ptype: PType::I32,
        reference: REFERENCE as i64,
    }))
    .unwrap();
    p.push(Arc::new(StoreOut { ptype: PType::I32 })).unwrap();
    p
}

fn input_and_bases() -> (Vec<i32>, Vec<i32>) {
    let input: Vec<i32> = (0..NUM_VALUES as i32).collect();
    let bases: Vec<i32> = (0..N_BLOCKS).map(|k| (k * BLOCK) as i32).collect();
    (input, bases)
}

fn with_throughput<'a, 'b>(bencher: Bencher<'a, 'b>) -> Bencher<'a, 'b> {
    bencher
        .counter(ItemsCount::new(NUM_VALUES))
        .counter(BytesCount::of_many::<i32>(NUM_VALUES))
}

// ============================================================
// Pipeline 1: LoadIn -> ForAdd -> StoreOut
// ============================================================

#[divan::bench]
fn for_add_only_staged_rust(bencher: Bencher) {
    let (input, _) = input_and_bases();
    with_throughput(bencher)
        .with_inputs(|| (input.clone(), vec![0i32; NUM_VALUES]))
        .bench_local_values(|(input, mut output)| {
            // No intermediate buffer needed — but autovec'd by LLVM.
            for i in 0..NUM_VALUES {
                output[i] = input[i].wrapping_add(REFERENCE);
            }
            divan::black_box(output)
        });
}

#[divan::bench]
fn for_add_only_jit_warm(bencher: Bencher) {
    let (input, _) = input_and_bases();
    let p = build_for_add_pipeline();
    let compiled = Compiler::new(vec![]).unwrap().compile(&p).unwrap();
    with_throughput(bencher)
        .with_inputs(|| (input.clone(), vec![0i32; NUM_VALUES]))
        .bench_local_values(|(input, mut output)| {
            unsafe {
                compiled.call_decompress_only(
                    input.as_ptr().cast(),
                    output.as_mut_ptr().cast(),
                    N_BLOCKS as u64,
                );
            }
            divan::black_box(output)
        });
}

#[divan::bench]
fn for_add_only_jit_cold(bencher: Bencher) {
    let (input, _) = input_and_bases();
    with_throughput(bencher)
        .with_inputs(|| (input.clone(), vec![0i32; NUM_VALUES]))
        .bench_local_values(|(input, mut output)| {
            let p = build_for_add_pipeline();
            let compiled = Compiler::new(vec![]).unwrap().compile(&p).unwrap();
            unsafe {
                compiled.call_decompress_only(
                    input.as_ptr().cast(),
                    output.as_mut_ptr().cast(),
                    N_BLOCKS as u64,
                );
            }
            divan::black_box(output)
        });
}

// ============================================================
// Pipeline 2: LoadIn -> DeltaPrefixSum -> ForAdd -> StoreOut
// ============================================================

#[divan::bench]
fn delta_for_staged_rust(bencher: Bencher) {
    let (input, bases) = input_and_bases();
    with_throughput(bencher)
        .with_inputs(|| (input.clone(), bases.clone(), vec![0i32; NUM_VALUES]))
        .bench_local_values(|(input, bases, mut output)| {
            // Stage 1: prefix sum into a temp buffer.
            let mut tmp = vec![0i32; NUM_VALUES];
            for k in 0..N_BLOCKS {
                let mut running = bases[k];
                for i in 0..BLOCK {
                    running = running.wrapping_add(input[k * BLOCK + i]);
                    tmp[k * BLOCK + i] = running;
                }
            }
            // Stage 2: FoR add into the output buffer.
            for i in 0..NUM_VALUES {
                output[i] = tmp[i].wrapping_add(REFERENCE);
            }
            divan::black_box(output)
        });
}

#[divan::bench]
fn delta_for_jit_warm(bencher: Bencher) {
    let (input, bases) = input_and_bases();
    let p = build_delta_for_pipeline();
    let compiled = Compiler::new(vec![]).unwrap().compile(&p).unwrap();
    with_throughput(bencher)
        .with_inputs(|| (input.clone(), bases.clone(), vec![0i32; NUM_VALUES]))
        .bench_local_values(|(input, bases, mut output)| {
            unsafe {
                compiled.call_with_named(
                    input.as_ptr().cast(),
                    output.as_mut_ptr().cast(),
                    N_BLOCKS as u64,
                    bases.as_ptr().cast(),
                );
            }
            divan::black_box(output)
        });
}

#[divan::bench]
fn delta_for_jit_cold(bencher: Bencher) {
    let (input, bases) = input_and_bases();
    with_throughput(bencher)
        .with_inputs(|| (input.clone(), bases.clone(), vec![0i32; NUM_VALUES]))
        .bench_local_values(|(input, bases, mut output)| {
            let p = build_delta_for_pipeline();
            let compiled = Compiler::new(vec![]).unwrap().compile(&p).unwrap();
            unsafe {
                compiled.call_with_named(
                    input.as_ptr().cast(),
                    output.as_mut_ptr().cast(),
                    N_BLOCKS as u64,
                    bases.as_ptr().cast(),
                );
            }
            divan::black_box(output)
        });
}

// ============================================================
// Pipeline 3: ALP decode — LoadIn(i32) -> AlpDecode -> StoreOut(f32)
//
// This is the case where the JIT is expected to *beat* the current
// Vortex implementation: ALP's actual decode loop at
// `encodings/alp/src/alp/mod.rs:253-261` is `iter_mut().for_each(|v|
// decode_single(transmute, exponents))` with table lookups inside.
// LLVM's autovec doesn't penetrate that shape reliably; the JIT emits
// `vcvtdq2ps` + `vmulps` directly.
//
// `alp_vortex_style`: closely mirrors the current Vortex code shape —
//   non-inlined per-element helper + lookup tables indexed by `e`/`f`.
// `alp_idealized_rust`: tight Rust, scale hoisted as a single literal.
//   This is what a competent engineer would write if hand-tuning. LLVM
//   should autovec this.
// `alp_jit_warm` / `alp_jit_cold`: the JIT.
// ============================================================

const ALP_SCALE: f64 = 0.01;
const F10: [f32; 22] = [
    1.0, 10.0, 100.0, 1_000.0, 10_000.0, 100_000.0, 1_000_000.0, 1e7, 1e8, 1e9, 1e10, 1e11,
    1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18, 1e19, 1e20, 1e21,
];
const IF10: [f32; 22] = [
    1.0, 0.1, 0.01, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13,
    1e-14, 1e-15, 1e-16, 1e-17, 1e-18, 1e-19, 1e-20, 1e-21,
];

/// Mirrors ALP's `decode_single`. `#[inline(never)]` so the function-call
/// shape that blocks LLVM's vectorizer is preserved (without the attribute,
/// LLVM might inline and then autovec the outer loop).
#[inline(never)]
fn decode_single_alp(encoded: i32, e: u8, f: u8) -> f32 {
    (encoded as f32) * F10[f as usize] * IF10[e as usize]
}

fn build_alp_pipeline() -> Pipeline {
    let mut p = Pipeline::new(PType::I32, BLOCK);
    p.push(Arc::new(LoadIn { ptype: PType::I32 })).unwrap();
    p.push(Arc::new(AlpDecode {
        in_ptype: PType::I32,
        out_ptype: PType::F32,
        scale: ALP_SCALE,
    }))
    .unwrap();
    p.push(Arc::new(StoreOut { ptype: PType::F32 })).unwrap();
    p
}

fn alp_input() -> Vec<i32> {
    (0..NUM_VALUES as i32).map(|i| i - 32_768).collect()
}

#[divan::bench]
fn alp_vortex_style(bencher: Bencher) {
    let input = alp_input();
    let (e, f) = (2u8, 0u8); // -> scale = IF10[2] * F10[0] = 0.01
    with_throughput(bencher)
        .with_inputs(|| (input.clone(), vec![0f32; NUM_VALUES]))
        .bench_local_values(|(input, mut output)| {
            // Mirrors `iter_mut().for_each(decode_single)` from
            // encodings/alp/src/alp/mod.rs:253-261. The non-inlined helper
            // + table indexing is what blocks LLVM autovec in practice.
            output
                .iter_mut()
                .zip(input.iter())
                .for_each(|(o, &x)| {
                    *o = decode_single_alp(x, e, f);
                });
            divan::black_box(output)
        });
}

#[divan::bench]
fn alp_idealized_rust(bencher: Bencher) {
    let input = alp_input();
    let scale = ALP_SCALE as f32;
    with_throughput(bencher)
        .with_inputs(|| (input.clone(), vec![0f32; NUM_VALUES]))
        .bench_local_values(|(input, mut output)| {
            // Scale hoisted as a single literal; tight loop. LLVM should
            // autovec this to vcvtdq2ps + vmulps at the host's native width
            // (AVX-512 here).
            for i in 0..NUM_VALUES {
                output[i] = (input[i] as f32) * scale;
            }
            divan::black_box(output)
        });
}

#[divan::bench]
fn alp_jit_warm(bencher: Bencher) {
    let input = alp_input();
    let p = build_alp_pipeline();
    let compiled = Compiler::new(vec![]).unwrap().compile(&p).unwrap();
    with_throughput(bencher)
        .with_inputs(|| (input.clone(), vec![0f32; NUM_VALUES]))
        .bench_local_values(|(input, mut output)| {
            unsafe {
                compiled.call_decompress_only(
                    input.as_ptr().cast(),
                    output.as_mut_ptr().cast(),
                    N_BLOCKS as u64,
                );
            }
            divan::black_box(output)
        });
}

#[divan::bench]
fn alp_jit_cold(bencher: Bencher) {
    let input = alp_input();
    with_throughput(bencher)
        .with_inputs(|| (input.clone(), vec![0f32; NUM_VALUES]))
        .bench_local_values(|(input, mut output)| {
            let p = build_alp_pipeline();
            let compiled = Compiler::new(vec![]).unwrap().compile(&p).unwrap();
            unsafe {
                compiled.call_decompress_only(
                    input.as_ptr().cast(),
                    output.as_mut_ptr().cast(),
                    N_BLOCKS as u64,
                );
            }
            divan::black_box(output)
        });
}

// ============================================================
// Pipeline 4: Full 4-stage chain — BitPacked(W=11) → FoR → ALP → Store(F32).
//
// This is the killer demo. The JIT fuses 4 encoding layers into ONE
// Cranelift function. Comparison points:
//
//   - `chain_vortex_style`: scalar Rust mirroring how current Vortex would
//     execute this chain — bitpack unpack per element with #[inline(never)]
//     helpers, intermediate buffer for the FoR + ALP pass, scalar ALP via
//     iter_mut+for_each. This is the upper-bound on what Vortex's current
//     code shape would produce after LLVM optimizes it.
//   - `chain_jit_warm`: the JIT's fused kernel.
// ============================================================

const W_BP: u8 = 11;
const FOR_REF_CHAIN: i64 = 100;
const ALP_SCALE_CHAIN: f64 = 0.01;
// BitPacked interleaved layout: n_chunks * W must be multiple of 32.
// With simd_lanes=4 and W=11, the chain pipeline uses its own larger
// per-emit block size that satisfies the constraint.
const BLOCK_CHAIN: usize = 128;

fn chain_input() -> (Vec<u32>, Vec<i32>) {
    let originals: Vec<i32> = (0..NUM_VALUES as i32).map(|i| i % (1 << W_BP)).collect();
    let packed = pack_dense(&originals, W_BP);
    (packed, originals)
}

fn build_chain_pipeline() -> Pipeline {
    let mut p = Pipeline::new(PType::I32, BLOCK_CHAIN);
    p.push(Arc::new(BitPackedLoad {
        ptype: PType::I32,
        bit_width: W_BP,
    }))
    .unwrap();
    p.push(Arc::new(ForAdd {
        ptype: PType::I32,
        reference: FOR_REF_CHAIN,
    }))
    .unwrap();
    p.push(Arc::new(AlpDecode {
        in_ptype: PType::I32,
        out_ptype: PType::F32,
        scale: ALP_SCALE_CHAIN,
    }))
    .unwrap();
    p.push(Arc::new(StoreOut { ptype: PType::F32 })).unwrap();
    p
}

/// Scalar per-element bitpack unpack helper for the interleaved layout the
/// JIT consumes. Non-inlined so it mirrors the shape Vortex's actual decode
/// path produces after LLVM gives up.
#[inline(never)]
fn unpack_one_w11(packed: &[u32], idx: usize) -> i32 {
    // Inlined copy of `unpack_one` so the #[inline(never)] applies here.
    const SIMD_LANES: usize = 4;
    const W: u8 = 11;
    let s = idx % SIMD_LANES;
    let row = idx / SIMD_LANES;
    let bit_start = row * W as usize;
    let word_idx = bit_start / 32;
    let bit_off = bit_start % 32;
    let lo = packed[word_idx * SIMD_LANES + s] >> bit_off;
    let val = if bit_off + W as usize <= 32 {
        lo & 0x7FF
    } else {
        let bits_in_lo = 32 - bit_off;
        let hi = packed[(word_idx + 1) * SIMD_LANES + s] & ((1u32 << (11 - bits_in_lo)) - 1);
        (lo | (hi << bits_in_lo)) & 0x7FF
    };
    val as i32
}

#[divan::bench]
fn chain_vortex_style(bencher: Bencher) {
    let (packed, _) = chain_input();
    let scale = ALP_SCALE_CHAIN as f32;
    with_throughput(bencher)
        .with_inputs(|| (packed.clone(), vec![0f32; NUM_VALUES]))
        .bench_local_values(|(packed, mut output)| {
            // Stage 1: scalar bitpack unpack into a tmp buffer.
            let mut tmp = vec![0i32; NUM_VALUES];
            for i in 0..NUM_VALUES {
                tmp[i] = unpack_one_w11(&packed, i);
            }
            // Stage 2: FoR add in-place (autovec'd).
            for v in tmp.iter_mut() {
                *v = v.wrapping_add(FOR_REF_CHAIN as i32);
            }
            // Stage 3: ALP scalar via iter_mut+for_each (this is what blocks autovec).
            output
                .iter_mut()
                .zip(tmp.iter())
                .for_each(|(o, &x)| {
                    *o = decode_single_alp_w11(x);
                });
            divan::black_box(output)
        });
}

/// ALP per-element scalar, non-inlined.
#[inline(never)]
fn decode_single_alp_w11(encoded: i32) -> f32 {
    (encoded as f32) * (ALP_SCALE_CHAIN as f32)
}

#[divan::bench]
fn chain_jit_warm(bencher: Bencher) {
    let (packed, _) = chain_input();
    let p = build_chain_pipeline();
    let compiled = Compiler::new(vec![]).unwrap().compile(&p).unwrap();
    let chain_n_blocks = NUM_VALUES / BLOCK_CHAIN;
    with_throughput(bencher)
        .with_inputs(|| (packed.clone(), vec![0f32; NUM_VALUES]))
        .bench_local_values(|(packed, mut output)| {
            unsafe {
                compiled.call_decompress_only(
                    packed.as_ptr().cast(),
                    output.as_mut_ptr().cast(),
                    chain_n_blocks as u64,
                );
            }
            divan::black_box(output)
        });
}

#[divan::bench]
fn chain_jit_cold(bencher: Bencher) {
    let (packed, _) = chain_input();
    let chain_n_blocks = NUM_VALUES / BLOCK_CHAIN;
    with_throughput(bencher)
        .with_inputs(|| (packed.clone(), vec![0f32; NUM_VALUES]))
        .bench_local_values(|(packed, mut output)| {
            let p = build_chain_pipeline();
            let compiled = Compiler::new(vec![]).unwrap().compile(&p).unwrap();
            unsafe {
                compiled.call_decompress_only(
                    packed.as_ptr().cast(),
                    output.as_mut_ptr().cast(),
                    chain_n_blocks as u64,
                );
            }
            divan::black_box(output)
        });
}

// ============================================================
// Pipeline 5: Dict + ForAdd over BitPacked codes — a fastlanes-style
// composition: codes are bit-packed, decoded via dict, value side gets a FoR
// offset applied. Comparison:
//
//   - `dict_chain_vortex_style`: scalar mock with non-inlined dict lookup.
//   - `dict_chain_jit_warm`: the JIT'd fused kernel.
// ============================================================

const DICT_SIZE: usize = 256;

fn dict_chain_inputs() -> (Vec<u32>, Vec<f32>) {
    // Pack i32 codes at width 8 (fits 256-entry dictionary) in interleaved layout.
    let codes: Vec<i32> = (0..NUM_VALUES as i32).map(|i| i & (DICT_SIZE as i32 - 1)).collect();
    let packed = pack_dense(&codes, 8);
    let values: Vec<f32> = (0..DICT_SIZE).map(|i| (i as f32) * 0.5 + 1.0).collect();
    (packed, values)
}

fn build_dict_chain_pipeline() -> Pipeline {
    // BitPacked(W=8 codes) -> DictLookup -> ForAdd(scaled into f32... wait,
    // ForAdd is integer. So this chain is more naturally:
    //   BitPacked(codes) -> DictLookup -> StoreOut.
    // To make the chain longer, append AlpDecode (which is also an i->f
    // multiply pattern). But DictLookup -> AlpDecode would need both stages
    // accepting f32 in/out which doesn't make semantic sense here.
    //
    // Stick with the simpler 3-stage chain: BitPacked codes -> Dict -> Store f32.
    let mut p = Pipeline::new(PType::I32, 32); // n_chunks*W = 8*8=64, ✓
    p.push(Arc::new(BitPackedLoad {
        ptype: PType::I32,
        bit_width: 8,
    }))
    .unwrap();
    p.push(Arc::new(DictLookup {
        code_ptype: PType::I32,
        value_ptype: PType::F32,
    }))
    .unwrap();
    p.push(Arc::new(StoreOut { ptype: PType::F32 })).unwrap();
    p
}

#[inline(never)]
fn dict_unpack_one_w8(packed: &[u32], idx: usize) -> i32 {
    // Interleaved layout for SIMD_LANES=4.
    const SIMD_LANES: usize = 4;
    let s = idx % SIMD_LANES;
    let row = idx / SIMD_LANES;
    let bit_start = row * 8;
    let word_idx = bit_start / 32;
    let bit_off = bit_start % 32;
    let lo = packed[word_idx * SIMD_LANES + s] >> bit_off;
    (lo & 0xFF) as i32
}

#[inline(never)]
fn dict_lookup_single(values: &[f32], code: i32) -> f32 {
    values[code as usize]
}

#[divan::bench]
fn dict_chain_vortex_style(bencher: Bencher) {
    let (packed, values) = dict_chain_inputs();
    with_throughput(bencher)
        .with_inputs(|| (packed.clone(), vec![0f32; NUM_VALUES]))
        .bench_local_values(|(packed, mut output)| {
            // Scalar shape: per-element unpack + dict lookup, both non-inlined.
            for i in 0..NUM_VALUES {
                let code = dict_unpack_one_w8(&packed, i);
                output[i] = dict_lookup_single(&values, code);
            }
            divan::black_box(output)
        });
}

#[divan::bench]
fn dict_chain_jit_warm(bencher: Bencher) {
    let (packed, values) = dict_chain_inputs();
    let p = build_dict_chain_pipeline();
    let compiled = Compiler::new(vec![]).unwrap().compile(&p).unwrap();
    let n_blocks = NUM_VALUES / 32;
    with_throughput(bencher)
        .with_inputs(|| (packed.clone(), vec![0f32; NUM_VALUES]))
        .bench_local_values(|(packed, mut output)| {
            unsafe {
                compiled.call_with_named(
                    packed.as_ptr().cast(),
                    output.as_mut_ptr().cast(),
                    n_blocks as u64,
                    values.as_ptr().cast(),
                );
            }
            divan::black_box(output)
        });
}

#[divan::bench]
fn dict_chain_jit_cold(bencher: Bencher) {
    let (packed, values) = dict_chain_inputs();
    let n_blocks = NUM_VALUES / 32;
    with_throughput(bencher)
        .with_inputs(|| (packed.clone(), vec![0f32; NUM_VALUES]))
        .bench_local_values(|(packed, mut output)| {
            let p = build_dict_chain_pipeline();
            let compiled = Compiler::new(vec![]).unwrap().compile(&p).unwrap();
            unsafe {
                compiled.call_with_named(
                    packed.as_ptr().cast(),
                    output.as_mut_ptr().cast(),
                    n_blocks as u64,
                    values.as_ptr().cast(),
                );
            }
            divan::black_box(output)
        });
}

fn main() {
    divan::main();
}
