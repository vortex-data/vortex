//! Correctness tests for the chained stencil. Each JIT'd kernel must match
//! a scalar reference across every `(op, ffor)` configuration and every
//! constant in `0..=255`.

#![cfg(all(target_arch = "x86_64", target_os = "linux"))]

use stencil_jit::{ChainConfig, CmpOp, Kernel, debug};

fn reference(packed: &[u8; 32], constant: u8, ffor_ref: u8, op: CmpOp, ffor: bool) -> u32 {
    let mut mask = 0u32;
    let c_signed = constant as i8;
    for (i, &b) in packed.iter().enumerate() {
        let v = if ffor { b.wrapping_add(ffor_ref) } else { b };
        let a_signed = v as i8;
        let bit = match op {
            CmpOp::Eq => v == constant,
            CmpOp::Neq => v != constant,
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

fn run(kernel: &Kernel, packed: &[u8; 32], constant: u8, ffor_ref: u8) -> u32 {
    let mut out: u32 = 0;
    // SAFETY: 32 readable + 4 writable.
    unsafe { kernel.call(packed.as_ptr(), constant, &mut out as *mut u32, ffor_ref) };
    out
}

#[test]
fn patch_metadata_is_consistent() {
    let bytes = debug::stencil_bytes();
    assert_eq!(debug::ffor_len(), 8);
    assert_eq!(debug::op_len(), 8);
    let f = debug::ffor_offset();
    let o = debug::op_offset();
    assert!(f + 8 <= bytes.len());
    assert!(o + 8 <= bytes.len());
    assert!(f + 8 <= o, "ffor slot must precede op slot");
    assert_eq!(&bytes[f..f + 8], &[0x90u8; 8]);
    assert_eq!(&bytes[o..o + 8], &[0x90u8; 8]);
    assert_eq!(debug::ffor_nop_patch(), &[0x90u8; 8]);
    for op in CmpOp::ALL {
        assert_eq!(debug::op_patch_bytes(op).len(), 8);
    }
}

#[test]
fn all_ops_match_reference_without_ffor() {
    let packed: [u8; 32] = core::array::from_fn(|i| (i as u8).wrapping_mul(31).wrapping_add(7));
    for op in CmpOp::ALL {
        let kernel = Kernel::compile(ChainConfig::compare_only(op)).unwrap();
        for c in 0u8..=255 {
            let got = run(&kernel, &packed, c, 0);
            let want = reference(&packed, c, 0, op, false);
            assert_eq!(got, want, "{op:?} mismatch at constant {c} (no ffor)");
        }
    }
}

#[test]
fn all_ops_match_reference_with_ffor() {
    let packed: [u8; 32] = core::array::from_fn(|i| (i as u8).wrapping_mul(17).wrapping_add(3));
    for op in CmpOp::ALL {
        let kernel = Kernel::compile(ChainConfig::ffor_then_compare(op)).unwrap();
        for ffor_ref in [0u8, 1, 7, 64, 200, 255] {
            for c in 0u8..=255 {
                let got = run(&kernel, &packed, c, ffor_ref);
                let want = reference(&packed, c, ffor_ref, op, true);
                assert_eq!(got, want, "{op:?} mismatch at c={c}, ref={ffor_ref} (ffor on)");
            }
        }
    }
}

#[test]
fn ffor_zero_matches_no_ffor() {
    // Sanity: FFoR-add with reference=0 should equal the no-FFoR kernel.
    let packed: [u8; 32] = core::array::from_fn(|i| (i as u8).wrapping_mul(13));
    for op in CmpOp::ALL {
        let ffor_off = Kernel::compile(ChainConfig::compare_only(op)).unwrap();
        let ffor_on0 = Kernel::compile(ChainConfig::ffor_then_compare(op)).unwrap();
        for c in [0u8, 1, 42, 200, 255] {
            assert_eq!(
                run(&ffor_off, &packed, c, 0),
                run(&ffor_on0, &packed, c, 0),
                "{op:?} mismatch at c={c}",
            );
        }
    }
}
