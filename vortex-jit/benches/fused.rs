// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! JIT fusion benchmarks at 65k elements.
//!
//! Compares three configurations on identical input:
//!   - `staged_rust`: hand-written Rust that mirrors the staged (non-fused)
//!     reference behavior. Each stage materializes its output into a temp
//!     buffer. This stands in for "current Vortex full chain".
//!   - `jit_warm`: the JIT-compiled fused kernel, with compile cost outside
//!     the timed region. Steady-state.
//!   - `jit_cold`: build a fresh compiler + compile + run, all timed. Shows
//!     the amortization point.
//!
//! Per design-notes §13: at 65k, output is 256 KiB f32 / 512 KiB f64 — L2
//! resident. Throughput is L2-write-bandwidth-bound for the fused path,
//! and ALP-scalar-bound or memory-pass-bound for the staged path.

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

fn build_pipeline() -> Pipeline {
    let mut p = Pipeline::new(PType::I32, BLOCK);
    p.push(Arc::new(LoadIn { ptype: PType::I32 })).unwrap();
    p.push(Arc::new(DeltaPrefixSum { ptype: PType::I32 })).unwrap();
    p.push(Arc::new(ForAdd {
        ptype: PType::I32,
        reference: 42,
    }))
    .unwrap();
    p.push(Arc::new(StoreOut { ptype: PType::I32 })).unwrap();
    p
}

fn input_and_bases() -> (Vec<i32>, Vec<i32>) {
    // Deltas: all 1s. Bases: per-block carry-in.
    let input = vec![1i32; NUM_VALUES];
    let bases: Vec<i32> = (0..N_BLOCKS).map(|k| (k * BLOCK) as i32).collect();
    (input, bases)
}

fn with_throughput<'a, 'b>(bencher: Bencher<'a, 'b>) -> Bencher<'a, 'b> {
    bencher
        .counter(ItemsCount::new(NUM_VALUES))
        .counter(BytesCount::of_many::<i32>(NUM_VALUES))
}

#[divan::bench]
fn staged_rust(bencher: Bencher) {
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
                output[i] = tmp[i].wrapping_add(42);
            }
            divan::black_box(output)
        });
}

#[divan::bench]
fn jit_warm(bencher: Bencher) {
    let (input, bases) = input_and_bases();
    let p = build_pipeline();
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
fn jit_cold(bencher: Bencher) {
    let (input, bases) = input_and_bases();
    with_throughput(bencher)
        .with_inputs(|| (input.clone(), bases.clone(), vec![0i32; NUM_VALUES]))
        .bench_local_values(|(input, bases, mut output)| {
            // Full cold path: build pipeline, compile, run.
            let p = build_pipeline();
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
