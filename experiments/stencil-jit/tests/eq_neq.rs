//! Correctness tests: the JIT'd kernel must match a scalar reference for
//! every op in `CmpOp::ALL` across every constant in `i8::MIN..=i8::MAX`.

#![cfg(all(target_arch = "x86_64", target_os = "linux"))]

use stencil_jit::{CmpOp, Kernel, debug};

/// Scalar oracle. The kernel treats lanes as signed `i8` for ordering
/// comparisons; for eq/neq the sign interpretation is irrelevant.
fn reference(packed: &[u8; 32], constant: u8, op: CmpOp) -> u32 {
    let mut mask = 0u32;
    let c_signed = constant as i8;
    for (i, &b) in packed.iter().enumerate() {
        let a_signed = b as i8;
        let bit = match op {
            CmpOp::Eq => b == constant,
            CmpOp::Neq => b != constant,
            CmpOp::Gt => a_signed > c_signed,
            CmpOp::Lt => a_signed < c_signed,
            CmpOp::Ge => a_signed >= c_signed,
            CmpOp::Le => a_signed <= c_signed,
        };
        if bit {
            mask |= 1u32 << i;
        }
    }
    mask
}

fn run(kernel: &Kernel, packed: &[u8; 32], constant: u8) -> u32 {
    let mut out: u32 = 0;
    // SAFETY: 32 readable bytes, 4 writable bytes.
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
        &[0x90u8; 8],
        "AOT stencil leaves the patch slot as raw nops until the JIT splices an op in",
    );
    for op in CmpOp::ALL {
        assert_eq!(debug::op_patch(op).len(), len, "{op:?} patch must be slot-sized");
    }
}

#[test]
fn all_ops_match_reference() {
    let packed: [u8; 32] = core::array::from_fn(|i| (i as u8).wrapping_mul(31).wrapping_add(7));
    for op in CmpOp::ALL {
        let kernel = Kernel::compile(op).unwrap_or_else(|e| panic!("compile {op:?}: {e}"));
        for c in 0u8..=255 {
            let got = run(&kernel, &packed, c);
            let want = reference(&packed, c, op);
            assert_eq!(got, want, "{op:?} mismatch at constant {c}");
        }
    }
}

#[test]
fn complement_pairs() {
    // eq XOR neq == all-ones; gt XOR le == all-ones; lt XOR ge == all-ones.
    let packed: [u8; 32] = core::array::from_fn(|i| (i as u8).wrapping_mul(7).wrapping_add(13));
    let pairs = [
        (CmpOp::Eq, CmpOp::Neq),
        (CmpOp::Gt, CmpOp::Le),
        (CmpOp::Lt, CmpOp::Ge),
    ];
    for (a, b) in pairs {
        let ka = Kernel::compile(a).unwrap();
        let kb = Kernel::compile(b).unwrap();
        for c in [0u8, 1, 42, 128, 200, 255] {
            assert_eq!(
                run(&ka, &packed, c) ^ run(&kb, &packed, c),
                u32::MAX,
                "{a:?} XOR {b:?} should cover all 32 lanes at c={c}",
            );
        }
    }
}

#[test]
fn distinct_compiles_dont_alias() {
    // Materialize one kernel per op; each must produce the right answer
    // and must not be the same pointer.
    let kernels: Vec<(CmpOp, Kernel)> =
        CmpOp::ALL.iter().map(|&op| (op, Kernel::compile(op).unwrap())).collect();
    let packed: [u8; 32] = core::array::from_fn(|i| i as u8);
    for c in [3u8, 17, 91, 200] {
        for (op, k) in &kernels {
            assert_eq!(run(k, &packed, c), reference(&packed, c, *op));
        }
    }
}
