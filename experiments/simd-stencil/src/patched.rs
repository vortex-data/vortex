// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Literal copy-and-patch.
//!
//! The ALP scale stage is the constant-bearing leaf of the f64 stacks, so it is
//! the natural place to demonstrate true copy-and-patch: a template stencil is
//! compiled ahead of time into the binary (the `global_asm!` block below), and
//! at "JIT" time we copy its bytes into a fresh executable page and patch the
//! scale constant directly into the `movabs` immediate. Building a stencil costs
//! one `memcpy`, an 8-byte store, and an `mprotect` — sub-microsecond, no code
//! generation.
//!
//! The integer stages reuse the `fastlanes` per-width unpack/undelta/untranspose
//! stencils, *selected* at run time by bit-width. Selection is the other face of
//! copy-and-patch: the constant (the width) chooses a pre-compiled stencil rather
//! than being patched into one.

#[cfg(all(target_arch = "x86_64", unix))]
mod imp {
    use std::ptr;

    use crate::TILE;
    use crate::encode::EncodedB;
    use crate::encode::EncodedC;
    use crate::kernels::rle_expand;
    use crate::kernels::undelta_u64;
    use crate::kernels::unfor_unpack_u64;
    use crate::kernels::untranspose_u64;
    use crate::strategies::fused;

    // Pre-compiled template stencil. AVX-512: convert 8 i64 -> 8 f64 and multiply
    // by a broadcast scale. The scale lives in the `movabs rax, imm64` immediate,
    // emitted as raw bytes so the patch site is exactly `start + 2`.
    core::arch::global_asm!(
        ".globl alp_scale_stencil_start",
        ".globl alp_scale_stencil_end",
        ".p2align 4",
        "alp_scale_stencil_start:",
        ".byte 0x48, 0xb8",             // REX.W + B8: movabs rax, imm64
        ".byte 0, 0, 0, 0, 0, 0, 0, 0", // imm64 patch site (offset 2)
        "vmovq xmm0, rax",
        "vbroadcastsd zmm0, xmm0",
        "xor ecx, ecx",
        "2:",
        "vcvtqq2pd zmm1, [rdi + rcx*8]",
        "vmulpd zmm1, zmm1, zmm0",
        "vmovupd [rsi + rcx*8], zmm1",
        "add rcx, 8",
        "cmp rcx, 1024",
        "jb 2b",
        "vzeroupper",
        "ret",
        "alp_scale_stencil_end:",
    );

    unsafe extern "C" {
        fn alp_scale_stencil_start();
        fn alp_scale_stencil_end();
    }

    const IMM_OFFSET: usize = 2;

    /// A run-time-emitted ALP scale stencil living in an executable page.
    pub struct AlpScaleStencil {
        code: *mut u8,
        len: usize,
        func: unsafe extern "C" fn(*const i64, *mut f64),
    }

    impl AlpScaleStencil {
        /// Copy the template into a fresh executable page and patch in `scale`.
        pub fn build(scale: f64) -> Self {
            let start = (alp_scale_stencil_start as unsafe extern "C" fn()) as usize;
            let end = (alp_scale_stencil_end as unsafe extern "C" fn()) as usize;
            let len = end - start;

            // SAFETY: standard mmap/mprotect dance to obtain executable code.
            unsafe {
                let code = libc::mmap(
                    ptr::null_mut(),
                    len,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                    -1,
                    0,
                ) as *mut u8;
                assert!(
                    !code.is_null() && code != libc::MAP_FAILED as *mut u8,
                    "mmap failed"
                );

                ptr::copy_nonoverlapping(start as *const u8, code, len);
                // Patch the scale constant into the movabs immediate.
                let bits = scale.to_bits().to_le_bytes();
                ptr::copy_nonoverlapping(bits.as_ptr(), code.add(IMM_OFFSET), 8);

                let rc = libc::mprotect(code.cast(), len, libc::PROT_READ | libc::PROT_EXEC);
                assert_eq!(rc, 0, "mprotect failed");

                AlpScaleStencil {
                    code,
                    len,
                    func: std::mem::transmute::<*mut u8, unsafe extern "C" fn(*const i64, *mut f64)>(
                        code,
                    ),
                }
            }
        }

        /// Decode one tile of `digits` (`i64`) into `out` (`f64`).
        ///
        /// # Safety
        /// `digits` and `out` must each be valid for one full 1024-element tile.
        #[inline(always)]
        pub unsafe fn run_tile(&self, digits: *const i64, out: *mut f64) {
            // SAFETY: caller guarantees `digits`/`out` cover one full tile.
            unsafe { (self.func)(digits, out) }
        }
    }

    impl Drop for AlpScaleStencil {
        fn drop(&mut self) {
            // SAFETY: `code`/`len` came from `mmap`.
            unsafe {
                libc::munmap(self.code.cast(), self.len);
            }
        }
    }

    fn avx512_available() -> bool {
        is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512dq")
    }

    pub fn decode_b(enc: &EncodedB) -> Vec<f64> {
        if !avx512_available() {
            return fused::decode_b(enc);
        }

        // "JIT": build the patched scale stencil once for the column.
        let stencil = AlpScaleStencil::build(enc.scale);

        let n = enc.n;
        let tiles = n / TILE;
        let mut out = vec![0f64; n];

        let mut td = [0u64; TILE];
        let mut tu = [0u64; TILE];
        let mut digits = [0u64; TILE];
        for t in 0..tiles {
            let w = enc.width[t] as usize;
            let off = enc.offsets[t];
            let plen = TILE * w / 64;
            unfor_unpack_u64(w, &enc.packed[off..off + plen], enc.reference[t], &mut td);
            undelta_u64(&td, &mut tu);
            untranspose_u64(&tu, &mut digits);
            // SAFETY: `digits` is one tile; `out[t*TILE..]` has a full tile left.
            unsafe {
                stencil.run_tile(digits.as_ptr().cast::<i64>(), out[t * TILE..].as_mut_ptr());
            }
        }
        out
    }

    pub fn decode_c(enc: &EncodedC) -> Vec<f64> {
        let ends = fused::decode_a(&enc.ends);
        let vals = decode_b(&enc.vals);
        let mut out = vec![0f64; enc.n_logical];
        rle_expand(&ends, &vals, &mut out);
        out
    }
}

#[cfg(all(target_arch = "x86_64", unix))]
pub use imp::AlpScaleStencil;
#[cfg(all(target_arch = "x86_64", unix))]
pub use imp::decode_b;
#[cfg(all(target_arch = "x86_64", unix))]
pub use imp::decode_c;

// Portable fallback: no machine-code patching available, defer to the fused path.
#[cfg(not(all(target_arch = "x86_64", unix)))]
pub fn decode_b(enc: &crate::encode::EncodedB) -> Vec<f64> {
    crate::strategies::fused::decode_b(enc)
}

#[cfg(not(all(target_arch = "x86_64", unix)))]
pub fn decode_c(enc: &crate::encode::EncodedC) -> Vec<f64> {
    crate::strategies::fused::decode_c(enc)
}
