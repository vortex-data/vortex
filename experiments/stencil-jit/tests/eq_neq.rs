//! Correctness tests: JIT'd kernel must match a scalar reference for both
//! `eq` and `neq` across every constant in `0..=255`.

#![cfg(all(target_arch = "x86_64", target_os = "linux"))]

use stencil_jit::{CmpOp, Kernel, debug};

/// Scalar oracle: produce the same 32-bit mask the JIT'd kernel should emit.
fn reference(packed: &[u8; 32], constant: u8, op: CmpOp) -> u32 {
    let mut mask = 0u32;
    for (i, &b) in packed.iter().enumerate() {
        let bit = match op {
            CmpOp::Eq => b == constant,
            CmpOp::Neq => b != constant,
        };
        if bit {
            mask |= 1u32 << i;
        }
    }
    mask
}

fn run(kernel: &Kernel, packed: &[u8; 32], constant: u8) -> u32 {
    let mut out: u32 = 0;
    // SAFETY: `packed` is 32 readable bytes; `out` is 4 writable bytes.
    unsafe { kernel.call(packed.as_ptr(), constant, &mut out as *mut u32) };
    out
}

#[test]
fn patch_metadata_is_consistent() {
    let bytes = debug::stencil_bytes();
    let off = debug::patch_offset();
    let len = debug::patch_len();
    assert_eq!(len, 8, "patch slot is exactly 8 bytes by construction");
    assert!(off + len <= bytes.len(), "patch slot inside stencil");
    assert_eq!(
        &bytes[off..off + len],
        debug::eq_patch(),
        "AOT stencil should default to the eq pattern (8 nops)",
    );
}

#[test]
fn eq_kernel_matches_reference_for_all_constants() {
    let kernel = Kernel::compile(CmpOp::Eq).expect("compile eq");
    let packed: [u8; 32] = core::array::from_fn(|i| i as u8);
    for c in 0u8..=255 {
        let got = run(&kernel, &packed, c);
        let want = reference(&packed, c, CmpOp::Eq);
        assert_eq!(got, want, "eq mismatch at constant {c}");
    }
}

#[test]
fn neq_kernel_matches_reference_for_all_constants() {
    let kernel = Kernel::compile(CmpOp::Neq).expect("compile neq");
    let packed: [u8; 32] = core::array::from_fn(|i| i as u8);
    for c in 0u8..=255 {
        let got = run(&kernel, &packed, c);
        let want = reference(&packed, c, CmpOp::Neq);
        assert_eq!(got, want, "neq mismatch at constant {c}");
    }
}

#[test]
fn neq_is_bitwise_inverse_of_eq() {
    let eq = Kernel::compile(CmpOp::Eq).expect("compile eq");
    let neq = Kernel::compile(CmpOp::Neq).expect("compile neq");
    let packed: [u8; 32] = core::array::from_fn(|i| (i as u8).wrapping_mul(7).wrapping_add(13));
    for c in [0u8, 1, 13, 42, 128, 200, 255] {
        let e = run(&eq, &packed, c);
        let n = run(&neq, &packed, c);
        assert_eq!(e ^ n, u32::MAX, "eq XOR neq must cover all 32 lanes (c={c})");
    }
}

#[test]
fn many_compiles_do_not_alias() {
    // Two kernels of the same op must produce identical output. Two kernels
    // of different ops must produce inverted output. This guards against any
    // accidental sharing of the underlying executable page across kernels.
    let eq1 = Kernel::compile(CmpOp::Eq).unwrap();
    let eq2 = Kernel::compile(CmpOp::Eq).unwrap();
    let neq = Kernel::compile(CmpOp::Neq).unwrap();
    let packed: [u8; 32] = core::array::from_fn(|i| i as u8);
    for c in [3u8, 17, 91] {
        assert_eq!(run(&eq1, &packed, c), run(&eq2, &packed, c));
        assert_eq!(run(&eq1, &packed, c) ^ run(&neq, &packed, c), u32::MAX);
    }
}
