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
use vortex_jit::stages::{DeltaPrefixSum, ForAdd, LoadIn, StoreOut};
use vortex_jit::{Compiler, PType, Pipeline};

#[global_allocator]
static ALLOC: MiMalloc = MiMalloc;

const NUM_VALUES: usize = 65_536;
const BLOCK: usize = 1024;
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

fn main() {
    divan::main();
}
