//! Correctness tests for the chained stencil + bulk kernel.

#![cfg(all(target_arch = "x86_64", target_os = "linux"))]

use stencil_jit::{BulkKernel, ChainConfig, CmpOp, Kernel, debug};

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
    assert_eq!(debug::ffor_len(), 5, "single-block load slot is 5 bytes");
    assert_eq!(debug::op_len(), 8, "single-block compare slot is 8 bytes");
    assert_eq!(debug::single_load_off().len(), 5);
    assert_eq!(debug::single_load_on().len(), 5);
    let f = debug::ffor_offset();
    let o = debug::op_offset();
    assert!(f + 5 <= bytes.len());
    assert!(o + 8 <= bytes.len());
    assert!(f + 5 <= o, "load slot must precede op slot");
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

#[test]
fn bulk_kernel_matches_single_block() {
    // Multi-block input. n_blocks must be even.
    const N: usize = 18;
    let mut packed = vec![0u8; N * 32];
    for (i, b) in packed.iter_mut().enumerate() {
        *b = (i as u8).wrapping_mul(31).wrapping_add(5);
    }
    let constant: u8 = 42;
    let ffor_ref: u8 = 7;

    for op in CmpOp::ALL {
        for ffor in [false, true] {
            let cfg = ChainConfig { ffor, op };
            let single = Kernel::compile(cfg).unwrap();
            let bulk = BulkKernel::compile(cfg).unwrap();

            let mut single_out = vec![0u32; N];
            for i in 0..N {
                let block: &[u8; 32] = (&packed[i * 32..(i + 1) * 32]).try_into().unwrap();
                let mut out: u32 = 0;
                // SAFETY: 32-byte block + 4-byte out.
                unsafe { single.call(block.as_ptr(), constant, &mut out as *mut u32, ffor_ref) };
                single_out[i] = out;
            }

            let mut bulk_out = vec![0u32; N];
            // SAFETY: N*32 readable, N*4 writable, N is even.
            unsafe {
                bulk.call(
                    packed.as_ptr(),
                    constant,
                    bulk_out.as_mut_ptr(),
                    ffor_ref,
                    N,
                )
            };

            assert_eq!(bulk_out, single_out, "bulk vs single disagree for {op:?} ffor={ffor}");
        }
    }
}

#[test]
fn specialized_kernel_matches_reference() {
    use stencil_jit::SpecializedKernel;
    // n_blocks must be a multiple of 4.
    const N: usize = 20;
    let mut packed = vec![0u8; N * 32];
    for (i, b) in packed.iter_mut().enumerate() {
        *b = (i as u8).wrapping_mul(17).wrapping_add(11);
    }
    for &(c, r) in &[(42u8, 7u8), (0, 0), (255, 1), (128, 200), (1, 255)] {
        let kernel = SpecializedKernel::compile_eq(c, r).unwrap();
        let mut out = vec![0u32; N];
        // SAFETY: N*32 readable, N*4 writable, N multiple of 4.
        unsafe { kernel.call(packed.as_ptr(), out.as_mut_ptr(), N) };
        // Reference: (x + r) == c
        for i in 0..N {
            let block: &[u8; 32] = (&packed[i * 32..(i + 1) * 32]).try_into().unwrap();
            let mut want = 0u32;
            for (j, &b) in block.iter().enumerate() {
                if b.wrapping_add(r) == c {
                    want |= 1u32 << j;
                }
            }
            assert_eq!(out[i], want, "specialized mismatch at block {i} c={c} r={r}");
        }
    }
}

#[test]
fn bulk_kernel_zero_blocks_is_noop() {
    let bulk = BulkKernel::compile(ChainConfig::compare_only(CmpOp::Eq)).unwrap();
    // SAFETY: pointers unused when n_blocks == 0.
    unsafe { bulk.call(core::ptr::null(), 0, core::ptr::null_mut(), 0, 0) };
}
