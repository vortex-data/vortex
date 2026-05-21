// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! SIMD-width gap: Cranelift JIT (AVX-128, regalloc only) vs LLVM AOT.
//!
//! Both implement the same fused FoR → ALP decode:
//!   out[i] = ((in[i] + reference) as Float) * scale
//!
//! `aot_*` is a hand-written tight Rust loop — LLVM compiles it AOT. Run the
//! bench twice to see the SIMD-width effect:
//!
//!   # baseline x86-64 (SSE2, 128-bit) — same width as Cranelift
//!   cargo bench -p vortex-jit --bench width_gap
//!
//!   # native (AVX-512, 512-bit on this host) — LLVM goes wide, Cranelift can't
//!   RUSTFLAGS="-C target-cpu=native" cargo bench -p vortex-jit --bench width_gap
//!
//! The delta between the two runs for `aot_*` is the wide-SIMD win LLVM gets
//! and Cranelift (capped at 128-bit) cannot. `jit_*` is unaffected by
//! RUSTFLAGS because Cranelift generates code at runtime.

use std::sync::Arc;

use divan::Bencher;
use divan::counter::{BytesCount, ItemsCount};
use mimalloc::MiMalloc;
use vortex_jit::stages::{AlpDecode, ForAdd, LoadIn, StoreOut};
use vortex_jit::{Compiler, PType, Pipeline};

#[global_allocator]
static ALLOC: MiMalloc = MiMalloc;

const NUM_VALUES: usize = 65_536;
const BLOCK: usize = 64;
const N_BLOCKS: usize = NUM_VALUES / BLOCK;
const REF_U32: i64 = 100;
const REF_U64: i64 = 100;
const SCALE: f64 = 0.01;

fn tp32<'a, 'b>(b: Bencher<'a, 'b>) -> Bencher<'a, 'b> {
    b.counter(ItemsCount::new(NUM_VALUES))
        .counter(BytesCount::of_many::<f32>(NUM_VALUES))
}
fn tp64<'a, 'b>(b: Bencher<'a, 'b>) -> Bencher<'a, 'b> {
    b.counter(ItemsCount::new(NUM_VALUES))
        .counter(BytesCount::of_many::<f64>(NUM_VALUES))
}

// ---------------- u32 -> f32 ----------------

fn build_u32() -> Pipeline {
    let mut p = Pipeline::new(PType::I32, BLOCK);
    p.push(Arc::new(LoadIn { ptype: PType::I32 })).unwrap();
    p.push(Arc::new(ForAdd {
        ptype: PType::I32,
        reference: REF_U32,
    }))
    .unwrap();
    p.push(Arc::new(AlpDecode {
        in_ptype: PType::I32,
        out_ptype: PType::F32,
        scale: SCALE,
    }))
    .unwrap();
    p.push(Arc::new(StoreOut { ptype: PType::F32 })).unwrap();
    p
}

#[divan::bench]
fn jit_u32(bencher: Bencher) {
    let input: Vec<i32> = (0..NUM_VALUES as i32).collect();
    let compiled = Compiler::new(vec![]).unwrap().compile(&build_u32()).unwrap();
    tp32(bencher)
        .with_inputs(|| (input.clone(), vec![0f32; NUM_VALUES]))
        .bench_local_values(|(input, mut out)| {
            unsafe {
                compiled.call_decompress_only(
                    input.as_ptr().cast(),
                    out.as_mut_ptr().cast(),
                    N_BLOCKS as u64,
                );
            }
            divan::black_box(out)
        });
}

#[divan::bench]
fn aot_u32(bencher: Bencher) {
    let input: Vec<i32> = (0..NUM_VALUES as i32).collect();
    let scale = SCALE as f32;
    let reference = REF_U32 as i32;
    tp32(bencher)
        .with_inputs(|| (input.clone(), vec![0f32; NUM_VALUES]))
        .bench_local_values(|(input, mut out)| {
            for i in 0..NUM_VALUES {
                out[i] = (input[i].wrapping_add(reference) as f32) * scale;
            }
            divan::black_box(out)
        });
}

// ---------------- u64 -> f64 ----------------

fn build_u64() -> Pipeline {
    let mut p = Pipeline::new(PType::I64, BLOCK);
    p.push(Arc::new(LoadIn { ptype: PType::I64 })).unwrap();
    p.push(Arc::new(ForAdd {
        ptype: PType::I64,
        reference: REF_U64,
    }))
    .unwrap();
    p.push(Arc::new(AlpDecode {
        in_ptype: PType::I64,
        out_ptype: PType::F64,
        scale: SCALE,
    }))
    .unwrap();
    p.push(Arc::new(StoreOut { ptype: PType::F64 })).unwrap();
    p
}

#[divan::bench]
fn jit_u64(bencher: Bencher) {
    let input: Vec<i64> = (0..NUM_VALUES as i64).collect();
    let compiled = Compiler::new(vec![]).unwrap().compile(&build_u64()).unwrap();
    tp64(bencher)
        .with_inputs(|| (input.clone(), vec![0f64; NUM_VALUES]))
        .bench_local_values(|(input, mut out)| {
            unsafe {
                compiled.call_decompress_only(
                    input.as_ptr().cast(),
                    out.as_mut_ptr().cast(),
                    N_BLOCKS as u64,
                );
            }
            divan::black_box(out)
        });
}

#[divan::bench]
fn aot_u64(bencher: Bencher) {
    let input: Vec<i64> = (0..NUM_VALUES as i64).collect();
    let scale = SCALE;
    let reference = REF_U64;
    tp64(bencher)
        .with_inputs(|| (input.clone(), vec![0f64; NUM_VALUES]))
        .bench_local_values(|(input, mut out)| {
            for i in 0..NUM_VALUES {
                out[i] = (input[i].wrapping_add(reference) as f64) * scale;
            }
            divan::black_box(out)
        });
}

fn main() {
    divan::main();
}
