// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! End-to-end tests: build pipelines, JIT-compile, run, compare to reference.

use std::sync::Arc;

use cranelift::prelude::types as cl_types;
use vortex_jit::stages::{
    AlpDecode, ApplyPatchesPostLoop, DeltaPrefixSum, ForAdd, LoadIn, StoreOut,
};
use vortex_jit::{Compiler, ExternFn, PType, Pipeline};

/// Helper to instantiate a fresh `Compiler` per test (each test wants its own
/// JIT module so finalize_definitions is one-shot).
fn fresh_compiler(externs: Vec<ExternFn>) -> Compiler {
    Compiler::new(externs).expect("compiler init")
}

#[test]
fn ir_shows_fusion_inline() {
    // Confirm that consecutive Lane-producing stages emit one IR function
    // with no scratch loads/stores between them — the SSA Values flow stage
    // to stage. We assert on the textual IR.
    const BLOCK: usize = 4;
    let mut p = Pipeline::new(PType::I32, BLOCK);
    p.push(Arc::new(LoadIn { ptype: PType::I32 })).unwrap();
    p.push(Arc::new(ForAdd {
        ptype: PType::I32,
        reference: 7,
    }))
    .unwrap();
    p.push(Arc::new(StoreOut { ptype: PType::I32 })).unwrap();

    let compiler = fresh_compiler(vec![]);
    let compiled = compiler.compile(&p).expect("compile");

    let ir = &compiled.ir_dump;
    eprintln!("=== Fused LoadIn -> ForAdd -> StoreOut IR ===\n{ir}");

    // SIMD fusion: BLOCK=4 lanes → 1 chunk of i32x4 → 1 vector load + 1
    // splat + 1 vector iadd + 1 vector store. No intermediate buffers.
    let n_simd_loads = ir.matches("load.i32x4").count();
    let n_splats = ir.matches("splat.i32x4").count();

    assert!(
        n_simd_loads >= 1,
        "expected at least one i32x4 load, got {n_simd_loads}; IR:\n{ir}"
    );
    assert!(
        n_splats >= 1,
        "expected at least one i32x4 splat of the FoR reference, got {n_splats}"
    );
    // The block body should NOT contain scalar i32 loads (those would mean
    // we lost the SIMD path).
    let scalar_block_loads = ir
        .lines()
        .filter(|l| l.contains("load.i32 ") && !l.contains("load.i32x"))
        .count();
    assert_eq!(
        scalar_block_loads, 0,
        "expected zero scalar i32 loads in block body, got {scalar_block_loads}; IR:\n{ir}"
    );
}

#[test]
fn for_add_round_trip_i32() {
    const BLOCK: usize = 16;
    const N_BLOCKS: usize = 4;
    const N: usize = BLOCK * N_BLOCKS;

    let mut p = Pipeline::new(PType::I32, BLOCK);
    p.push(Arc::new(LoadIn { ptype: PType::I32 })).unwrap();
    p.push(Arc::new(ForAdd {
        ptype: PType::I32,
        reference: 1000,
    }))
    .unwrap();
    p.push(Arc::new(StoreOut { ptype: PType::I32 })).unwrap();

    let compiler = fresh_compiler(vec![]);
    let compiled = compiler.compile(&p).expect("compile");

    let input: Vec<i32> = (0..N as i32).collect();
    let mut output: Vec<i32> = vec![0; N];
    unsafe {
        compiled.call_decompress_only(
            input.as_ptr().cast(),
            output.as_mut_ptr().cast(),
            N_BLOCKS as u64,
        );
    }

    let expected: Vec<i32> = input.iter().map(|x| x + 1000).collect();
    assert_eq!(output, expected);
}

#[test]
fn delta_for_pipeline_i32() {
    const BLOCK: usize = 16;
    const N_BLOCKS: usize = 4;
    const N: usize = BLOCK * N_BLOCKS;

    let mut p = Pipeline::new(PType::I32, BLOCK);
    p.push(Arc::new(LoadIn { ptype: PType::I32 })).unwrap();
    p.push(Arc::new(DeltaPrefixSum { ptype: PType::I32 })).unwrap();
    p.push(Arc::new(ForAdd {
        ptype: PType::I32,
        reference: 7,
    }))
    .unwrap();
    p.push(Arc::new(StoreOut { ptype: PType::I32 })).unwrap();

    let compiler = fresh_compiler(vec![]);
    let compiled = compiler.compile(&p).expect("compile");

    // Input is "deltas" — all 1s.
    let input: Vec<i32> = vec![1; N];
    // Bases: per-block carry-in. Block k starts cumulatively from k*BLOCK.
    let bases: Vec<i32> = (0..N_BLOCKS).map(|k| (k * BLOCK) as i32).collect();

    let mut output: Vec<i32> = vec![0; N];

    // We need to call the kernel with the named `delta_bases` arg.
    // Signature is (InPtr, OutPtr, NBlocks, delta_bases) per stable arg order
    // (InPtr/OutPtr/NBlocks pulled to front; then named in declaration order).
    assert_eq!(compiled.args.len(), 4);
    unsafe {
        compiled.call_with_named(
            input.as_ptr().cast(),
            output.as_mut_ptr().cast(),
            N_BLOCKS as u64,
            bases.as_ptr().cast(),
        );
    }

    // Reference: each block is base[k] + 1 + 1 + ... (prefix sum of all 1s),
    // then + 7 (FoR ref).
    let mut expected = vec![0i32; N];
    for k in 0..N_BLOCKS {
        let mut running = bases[k];
        for i in 0..BLOCK {
            running += 1;
            expected[k * BLOCK + i] = running + 7;
        }
    }
    assert_eq!(output, expected);
}

#[test]
fn alp_decode_i32_to_f32() {
    // BLOCK must be a multiple of simd_lanes for i32 (= 4 at 128-bit).
    const BLOCK: usize = 16;
    const N_BLOCKS: usize = 4;
    const N: usize = BLOCK * N_BLOCKS;

    // ALP-style: encoded i32, decoded f32 = (i32 as f32) * scale.
    // Pick a non-power-of-two scale to verify the multiply runs.
    let scale = 0.01f64;

    let mut p = Pipeline::new(PType::I32, BLOCK);
    p.push(Arc::new(LoadIn { ptype: PType::I32 })).unwrap();
    p.push(Arc::new(AlpDecode {
        in_ptype: PType::I32,
        out_ptype: PType::F32,
        scale,
    }))
    .unwrap();
    p.push(Arc::new(StoreOut { ptype: PType::F32 })).unwrap();

    let compiler = fresh_compiler(vec![]);
    let compiled = compiler.compile(&p).expect("compile");

    eprintln!("=== ALP decode IR ===\n{}", compiled.ir_dump);

    let input: Vec<i32> = (-(N as i32 / 2)..(N as i32 / 2)).collect();
    let mut output: Vec<f32> = vec![0.0; N];
    unsafe {
        compiled.call_decompress_only(
            input.as_ptr().cast(),
            output.as_mut_ptr().cast(),
            N_BLOCKS as u64,
        );
    }

    let expected: Vec<f32> = input.iter().map(|x| (*x as f32) * scale as f32).collect();
    // Floating-point equality is exact here because the scale is the same
    // literal in both paths and the convert + multiply has no other rounding.
    for (i, (a, b)) in output.iter().zip(expected.iter()).enumerate() {
        assert_eq!(a, b, "mismatch at {i}: jit={a} expected={b}");
    }
}

/// Apply-patches helper, monomorphized for i32.
///
/// # Safety
/// `out`, `idx`, `val` must be valid for `n` reads.
unsafe extern "C" fn apply_patches_i32(
    out: *mut i32,
    idx: *const u64,
    val: *const i32,
    n: u64,
) {
    let idx = unsafe { std::slice::from_raw_parts(idx, n as usize) };
    let val = unsafe { std::slice::from_raw_parts(val, n as usize) };
    for (k, &i) in idx.iter().enumerate() {
        unsafe { *out.add(i as usize) = val[k] };
    }
}

#[test]
fn for_add_with_patches_i32() {
    const BLOCK: usize = 16;
    const N_BLOCKS: usize = 4;
    const N: usize = BLOCK * N_BLOCKS;

    // Patches override positions 3, 10, 47.
    let patch_idx: Vec<u64> = vec![3, 10, 47];
    let patch_val: Vec<i32> = vec![-1, -2, -3];
    let patch_n: u64 = patch_idx.len() as u64;

    let mut p = Pipeline::new(PType::I32, BLOCK);
    p.push(Arc::new(LoadIn { ptype: PType::I32 })).unwrap();
    p.push(Arc::new(ForAdd {
        ptype: PType::I32,
        reference: 100,
    }))
    .unwrap();
    p.push(Arc::new(StoreOut { ptype: PType::I32 })).unwrap();
    p.push(Arc::new(ApplyPatchesPostLoop {
        ptype: PType::I32,
        helper_name: "apply_patches_i32",
    }))
    .unwrap();

    let compiler = fresh_compiler(vec![ExternFn {
        name: "apply_patches_i32",
        addr: apply_patches_i32 as *const u8,
        // (*mut i32, *const u64, *const i32, u64)
        params: &[cl_types::I64, cl_types::I64, cl_types::I64, cl_types::I64],
        returns: &[],
    }]);
    let compiled = compiler.compile(&p).expect("compile");
    eprintln!(
        "=== LoadIn -> ForAdd -> StoreOut + [PostLoop] ApplyPatches IR ===\n{}",
        compiled.ir_dump
    );

    let input: Vec<i32> = (0..N as i32).collect();
    let mut output: Vec<i32> = vec![0; N];

    // Sig: (InPtr, OutPtr, NBlocks, patch_idx, patch_val, patch_n)
    assert_eq!(compiled.args.len(), 6);
    let patch_n_buf = [patch_n];
    unsafe {
        compiled.call_with_three_named(
            input.as_ptr().cast(),
            output.as_mut_ptr().cast(),
            N_BLOCKS as u64,
            patch_idx.as_ptr().cast(),
            patch_val.as_ptr().cast(),
            patch_n_buf.as_ptr().cast(),
        );
    }

    // Reference: input + 100, then scatter patches.
    let mut expected: Vec<i32> = input.iter().map(|x| x + 100).collect();
    for (k, &i) in patch_idx.iter().enumerate() {
        expected[i as usize] = patch_val[k];
    }
    assert_eq!(output, expected);
}
