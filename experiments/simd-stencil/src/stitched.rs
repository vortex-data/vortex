// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Body-stitching copy-and-patch.
//!
//! The earlier `patched` result showed that emitting a *single* op as machine
//! code and calling it per tile loses to `fused`: the call boundary and the
//! materialised intermediate cost more than baking the constant saves. The fix
//! is the real Xu/Kjolstad move — stitch the op *bodies* into one loop so the
//! intermediates stay in a register and there are no internal calls.
//!
//! This module implements that for a pipeline of ops `x = (x*a + b).abs()` over
//! an `f64` column (a stand-in for the chained elementwise transforms a decode
//! tail performs: FoR add, ALP scale, ...). The `abs` is deliberate: a *pure*
//! affine chain folds to a single op under constant propagation, which the AOT
//! reference does for free — so the ops are made non-linear to keep the
//! comparison a fair "execute all N ops" test. At build time we:
//!
//! 1. copy a fixed **prologue** that points `r8` at a patched constant pool and
//!    broadcasts every constant into a register once (hoisted out of the loop);
//!    the loop is 8× unrolled (zmm0..7) to keep enough loads in flight,
//! 2. copy one self-contained **body** per op (the bodies are concatenated, not
//!    called),
//! 3. copy a fixed **epilogue** (store, advance, branch back),
//! 4. patch the loop back-edge `rel32` — a genuine relocation, since the branch
//!    distance depends on how many bodies were stitched in, and write the
//!    constants (and abs mask) into the pool.
//!
//! The result is one AVX-512 loop with no per-op call overhead and constants in
//! registers, built in ~`memcpy` time — and it matches the AOT-compiled tail.

/// Maximum ops a stitched pipeline supports (one zmm register pair each in
/// zmm8..19, leaving zmm0..7 as the 8× unrolled accumulators).
pub const MAX_OPS: usize = 6;

#[cfg(all(target_arch = "x86_64", unix))]
mod imp {
    use std::ptr;

    use super::MAX_OPS;

    // Verbatim machine code bracketed by exported symbols. The constants live in
    // a pool appended after the code; the prologue broadcasts every constant
    // into a register *once* (addressed through `r8`), so the loop body is just
    // the `vfmadd`s — exactly what an AOT-compiled tail emits. The loop is
    // 8× unrolled (accumulators zmm0..7, 64 f64 per iteration) so there are
    // enough outstanding loads to saturate memory bandwidth, the actual limit on
    // this workload. Constants live in zmm8..19 (op i → a=zmm(8+2i), b=zmm(9+2i)).
    // Bodies are fall-through (no `ret`).
    core::arch::global_asm!(
        // ---- prologue: rdi=src, rsi=dst, rdx=len; load 6 const pairs into zmm8..19 ----
        ".globl stitch_pro_start",
        ".globl stitch_pro_loop",
        ".globl stitch_pro_end",
        ".p2align 4",
        "stitch_pro_start:",
        ".byte 0x49, 0xb8", // movabs r8, imm64  (constant-pool address, patch at +2)
        ".byte 0, 0, 0, 0, 0, 0, 0, 0",
        "vbroadcastsd zmm8, qword ptr [r8 + 0]",
        "vbroadcastsd zmm9, qword ptr [r8 + 8]",
        "vbroadcastsd zmm10, qword ptr [r8 + 16]",
        "vbroadcastsd zmm11, qword ptr [r8 + 24]",
        "vbroadcastsd zmm12, qword ptr [r8 + 32]",
        "vbroadcastsd zmm13, qword ptr [r8 + 40]",
        "vbroadcastsd zmm14, qword ptr [r8 + 48]",
        "vbroadcastsd zmm15, qword ptr [r8 + 56]",
        "vbroadcastsd zmm16, qword ptr [r8 + 64]",
        "vbroadcastsd zmm17, qword ptr [r8 + 72]",
        "vbroadcastsd zmm18, qword ptr [r8 + 80]",
        "vbroadcastsd zmm19, qword ptr [r8 + 88]",
        "vbroadcastsd zmm20, qword ptr [r8 + 96]", // abs mask (0x7fff...ffff)
        "xor ecx, ecx",
        "stitch_pro_loop:",
        "vmovupd zmm0, [rdi + rcx*8]",
        "vmovupd zmm1, [rdi + rcx*8 + 64]",
        "vmovupd zmm2, [rdi + rcx*8 + 128]",
        "vmovupd zmm3, [rdi + rcx*8 + 192]",
        "vmovupd zmm4, [rdi + rcx*8 + 256]",
        "vmovupd zmm5, [rdi + rcx*8 + 320]",
        "vmovupd zmm6, [rdi + rcx*8 + 384]",
        "vmovupd zmm7, [rdi + rcx*8 + 448]",
        "stitch_pro_end:",
        // ---- bodies: zmm{0..7} = zmm{0..7} * a_i + b_i, constants in registers ----
        ".globl stitch_b0_start",
        ".globl stitch_b0_end",
        "stitch_b0_start:",
        "vfmadd213pd zmm0, zmm8, zmm9",
        "vfmadd213pd zmm1, zmm8, zmm9",
        "vfmadd213pd zmm2, zmm8, zmm9",
        "vfmadd213pd zmm3, zmm8, zmm9",
        "vfmadd213pd zmm4, zmm8, zmm9",
        "vfmadd213pd zmm5, zmm8, zmm9",
        "vfmadd213pd zmm6, zmm8, zmm9",
        "vfmadd213pd zmm7, zmm8, zmm9",
        "vandpd zmm0, zmm0, zmm20",
        "vandpd zmm1, zmm1, zmm20",
        "vandpd zmm2, zmm2, zmm20",
        "vandpd zmm3, zmm3, zmm20",
        "vandpd zmm4, zmm4, zmm20",
        "vandpd zmm5, zmm5, zmm20",
        "vandpd zmm6, zmm6, zmm20",
        "vandpd zmm7, zmm7, zmm20",
        "stitch_b0_end:",
        ".globl stitch_b1_start",
        ".globl stitch_b1_end",
        "stitch_b1_start:",
        "vfmadd213pd zmm0, zmm10, zmm11",
        "vfmadd213pd zmm1, zmm10, zmm11",
        "vfmadd213pd zmm2, zmm10, zmm11",
        "vfmadd213pd zmm3, zmm10, zmm11",
        "vfmadd213pd zmm4, zmm10, zmm11",
        "vfmadd213pd zmm5, zmm10, zmm11",
        "vfmadd213pd zmm6, zmm10, zmm11",
        "vfmadd213pd zmm7, zmm10, zmm11",
        "vandpd zmm0, zmm0, zmm20",
        "vandpd zmm1, zmm1, zmm20",
        "vandpd zmm2, zmm2, zmm20",
        "vandpd zmm3, zmm3, zmm20",
        "vandpd zmm4, zmm4, zmm20",
        "vandpd zmm5, zmm5, zmm20",
        "vandpd zmm6, zmm6, zmm20",
        "vandpd zmm7, zmm7, zmm20",
        "stitch_b1_end:",
        ".globl stitch_b2_start",
        ".globl stitch_b2_end",
        "stitch_b2_start:",
        "vfmadd213pd zmm0, zmm12, zmm13",
        "vfmadd213pd zmm1, zmm12, zmm13",
        "vfmadd213pd zmm2, zmm12, zmm13",
        "vfmadd213pd zmm3, zmm12, zmm13",
        "vfmadd213pd zmm4, zmm12, zmm13",
        "vfmadd213pd zmm5, zmm12, zmm13",
        "vfmadd213pd zmm6, zmm12, zmm13",
        "vfmadd213pd zmm7, zmm12, zmm13",
        "vandpd zmm0, zmm0, zmm20",
        "vandpd zmm1, zmm1, zmm20",
        "vandpd zmm2, zmm2, zmm20",
        "vandpd zmm3, zmm3, zmm20",
        "vandpd zmm4, zmm4, zmm20",
        "vandpd zmm5, zmm5, zmm20",
        "vandpd zmm6, zmm6, zmm20",
        "vandpd zmm7, zmm7, zmm20",
        "stitch_b2_end:",
        ".globl stitch_b3_start",
        ".globl stitch_b3_end",
        "stitch_b3_start:",
        "vfmadd213pd zmm0, zmm14, zmm15",
        "vfmadd213pd zmm1, zmm14, zmm15",
        "vfmadd213pd zmm2, zmm14, zmm15",
        "vfmadd213pd zmm3, zmm14, zmm15",
        "vfmadd213pd zmm4, zmm14, zmm15",
        "vfmadd213pd zmm5, zmm14, zmm15",
        "vfmadd213pd zmm6, zmm14, zmm15",
        "vfmadd213pd zmm7, zmm14, zmm15",
        "vandpd zmm0, zmm0, zmm20",
        "vandpd zmm1, zmm1, zmm20",
        "vandpd zmm2, zmm2, zmm20",
        "vandpd zmm3, zmm3, zmm20",
        "vandpd zmm4, zmm4, zmm20",
        "vandpd zmm5, zmm5, zmm20",
        "vandpd zmm6, zmm6, zmm20",
        "vandpd zmm7, zmm7, zmm20",
        "stitch_b3_end:",
        ".globl stitch_b4_start",
        ".globl stitch_b4_end",
        "stitch_b4_start:",
        "vfmadd213pd zmm0, zmm16, zmm17",
        "vfmadd213pd zmm1, zmm16, zmm17",
        "vfmadd213pd zmm2, zmm16, zmm17",
        "vfmadd213pd zmm3, zmm16, zmm17",
        "vfmadd213pd zmm4, zmm16, zmm17",
        "vfmadd213pd zmm5, zmm16, zmm17",
        "vfmadd213pd zmm6, zmm16, zmm17",
        "vfmadd213pd zmm7, zmm16, zmm17",
        "vandpd zmm0, zmm0, zmm20",
        "vandpd zmm1, zmm1, zmm20",
        "vandpd zmm2, zmm2, zmm20",
        "vandpd zmm3, zmm3, zmm20",
        "vandpd zmm4, zmm4, zmm20",
        "vandpd zmm5, zmm5, zmm20",
        "vandpd zmm6, zmm6, zmm20",
        "vandpd zmm7, zmm7, zmm20",
        "stitch_b4_end:",
        ".globl stitch_b5_start",
        ".globl stitch_b5_end",
        "stitch_b5_start:",
        "vfmadd213pd zmm0, zmm18, zmm19",
        "vfmadd213pd zmm1, zmm18, zmm19",
        "vfmadd213pd zmm2, zmm18, zmm19",
        "vfmadd213pd zmm3, zmm18, zmm19",
        "vfmadd213pd zmm4, zmm18, zmm19",
        "vfmadd213pd zmm5, zmm18, zmm19",
        "vfmadd213pd zmm6, zmm18, zmm19",
        "vfmadd213pd zmm7, zmm18, zmm19",
        "vandpd zmm0, zmm0, zmm20",
        "vandpd zmm1, zmm1, zmm20",
        "vandpd zmm2, zmm2, zmm20",
        "vandpd zmm3, zmm3, zmm20",
        "vandpd zmm4, zmm4, zmm20",
        "vandpd zmm5, zmm5, zmm20",
        "vandpd zmm6, zmm6, zmm20",
        "vandpd zmm7, zmm7, zmm20",
        "stitch_b5_end:",
        // ---- epilogue: store 8 vectors, advance 64, branch back ----
        ".globl stitch_epi_start",
        ".globl stitch_epi_jb",
        ".globl stitch_epi_end",
        "stitch_epi_start:",
        "vmovupd [rsi + rcx*8], zmm0",
        "vmovupd [rsi + rcx*8 + 64], zmm1",
        "vmovupd [rsi + rcx*8 + 128], zmm2",
        "vmovupd [rsi + rcx*8 + 192], zmm3",
        "vmovupd [rsi + rcx*8 + 256], zmm4",
        "vmovupd [rsi + rcx*8 + 320], zmm5",
        "vmovupd [rsi + rcx*8 + 384], zmm6",
        "vmovupd [rsi + rcx*8 + 448], zmm7",
        "add rcx, 64",
        "cmp rcx, rdx", // rdx = element count (3rd arg); loop over the whole buffer
        "stitch_epi_jb:",
        ".byte 0x0f, 0x82, 0, 0, 0, 0", // jb rel32 (back-edge, patched at jb+2)
        "vzeroupper",
        "ret",
        "stitch_epi_end:",
    );

    unsafe extern "C" {
        fn stitch_pro_start();
        fn stitch_pro_loop();
        fn stitch_pro_end();
        fn stitch_b0_start();
        fn stitch_b0_end();
        fn stitch_b1_start();
        fn stitch_b2_start();
        fn stitch_b3_start();
        fn stitch_b4_start();
        fn stitch_b5_start();
        fn stitch_epi_start();
        fn stitch_epi_jb();
        fn stitch_epi_end();
    }

    #[inline]
    fn addr(f: unsafe extern "C" fn()) -> usize {
        f as usize
    }

    /// A stitched affine pipeline living in one executable page.
    pub struct StitchedAffine {
        code: *mut u8,
        len: usize,
        func: unsafe extern "C" fn(*const f64, *mut f64, usize),
    }

    impl StitchedAffine {
        /// Stitch one `vfmadd` body per `(a, b)` op into a single AVX-512 loop,
        /// with the constants placed in a patched pool and hoisted out of the loop.
        pub fn build(ops: &[(f64, f64)]) -> Self {
            assert!(!ops.is_empty() && ops.len() <= MAX_OPS, "1..={MAX_OPS} ops");

            let pro = addr(stitch_pro_start);
            let pro_loop = addr(stitch_pro_loop);
            let pro_end = addr(stitch_pro_end);
            let epi = addr(stitch_epi_start);
            let epi_jb = addr(stitch_epi_jb);
            let epi_end = addr(stitch_epi_end);
            let body_starts = [
                addr(stitch_b0_start),
                addr(stitch_b1_start),
                addr(stitch_b2_start),
                addr(stitch_b3_start),
                addr(stitch_b4_start),
                addr(stitch_b5_start),
            ];
            // All bodies are the same fixed length (8 fmadd + 8 vandpd); reuse b0's.
            let body_len = addr(stitch_b0_end) - body_starts[0];

            let pro_len = pro_end - pro;
            let epi_len = epi_end - epi;
            let pool_slots = 2 * MAX_OPS + 1; // 12 op constants + the abs mask
            let code_len = pro_len + ops.len() * body_len + epi_len;
            let pool_off = code_len.next_multiple_of(8);
            let total = pool_off + pool_slots * 8;

            // SAFETY: map a page, assemble the stitched code + pool, flip to RX.
            unsafe {
                let code = libc::mmap(
                    ptr::null_mut(),
                    total,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                    -1,
                    0,
                ) as *mut u8;
                assert!(
                    !code.is_null() && code != libc::MAP_FAILED as *mut u8,
                    "mmap failed"
                );

                // 1. prologue, then patch r8 with the absolute pool address.
                ptr::copy_nonoverlapping(pro as *const u8, code, pro_len);
                let pool_addr = (code as usize + pool_off) as u64;
                ptr::copy_nonoverlapping(pool_addr.to_le_bytes().as_ptr(), code.add(2), 8);

                // 2. stitch one vfmadd body per op (slot i uses zmm pair 2i+1/2i+2).
                for (i, _) in ops.iter().enumerate() {
                    let dst = code.add(pro_len + i * body_len);
                    ptr::copy_nonoverlapping(body_starts[i] as *const u8, dst, body_len);
                }

                // 3. epilogue + back-edge relocation.
                let epi_page_off = pro_len + ops.len() * body_len;
                ptr::copy_nonoverlapping(epi as *const u8, code.add(epi_page_off), epi_len);
                let loop_target = (pro_loop - pro) as isize;
                let rel32_pos = epi_page_off as isize + (epi_jb - epi) as isize + 2;
                let rel32 = (loop_target - (rel32_pos + 4)) as i32;
                ptr::copy_nonoverlapping(
                    rel32.to_le_bytes().as_ptr(),
                    code.add(rel32_pos as usize),
                    4,
                );

                // 4. constant pool: a_i,b_i for live ops, identity (1,0) elsewhere,
                //    plus the abs mask in the final slot.
                let pool = code.add(pool_off).cast::<f64>();
                for slot in 0..MAX_OPS {
                    let (a, b) = ops.get(slot).copied().unwrap_or((1.0, 0.0));
                    pool.add(2 * slot).write_unaligned(a);
                    pool.add(2 * slot + 1).write_unaligned(b);
                }
                pool.add(2 * MAX_OPS)
                    .write_unaligned(f64::from_bits(0x7fff_ffff_ffff_ffff));

                let rc = libc::mprotect(code.cast(), total, libc::PROT_READ | libc::PROT_EXEC);
                assert_eq!(rc, 0, "mprotect failed");

                StitchedAffine {
                    code,
                    len: total,
                    func: std::mem::transmute::<
                        *mut u8,
                        unsafe extern "C" fn(*const f64, *mut f64, usize),
                    >(code),
                }
            }
        }

        /// Run the stitched pipeline over `len` `f64`s in one call (constants are
        /// loaded once, so this should be called over a whole column, not per tile).
        ///
        /// # Safety
        /// `src`/`dst` must be valid for `len` `f64`s, and `len` must be a
        /// multiple of 64 (the unroll factor).
        #[inline(always)]
        pub unsafe fn run(&self, src: *const f64, dst: *mut f64, len: usize) {
            debug_assert_eq!(len % 64, 0, "len must be a multiple of the 64-wide unroll");
            // SAFETY: caller guarantees validity for `len` elements.
            unsafe { (self.func)(src, dst, len) }
        }
    }

    impl Drop for StitchedAffine {
        fn drop(&mut self) {
            // SAFETY: `code`/`len` came from `mmap`.
            unsafe {
                libc::munmap(self.code.cast(), self.len);
            }
        }
    }
}

#[cfg(all(target_arch = "x86_64", unix))]
pub use imp::StitchedAffine;

/// Reference / AOT pipeline: each op is `x = (x*a + b).abs()`. The `abs` breaks
/// linearity, so — unlike a pure affine chain — the compiler *cannot* fold the
/// ops into one, and both AOT and the stitched JIT must execute all of them.
/// `mul_add` emits the same fused-multiply-add the stitched code uses, so the
/// two agree bit-for-bit.
#[inline(always)]
pub fn affine_aot(ops: &[(f64, f64)], src: &[f64], dst: &mut [f64]) {
    for (s, d) in src.iter().zip(dst.iter_mut()) {
        let mut x = *s;
        for &(a, b) in ops {
            x = x.mul_add(a, b).abs();
        }
        *d = x;
    }
}

/// "Per-op" pipeline: each op is a separate full pass over the buffer, so the
/// intermediate is materialised between ops (what calling one stencil per op
/// forces). Models the non-stitched copy-and-patch backend.
pub fn affine_per_op(ops: &[(f64, f64)], src: &[f64], dst: &mut [f64]) {
    dst.copy_from_slice(src);
    for &(a, b) in ops {
        for d in dst.iter_mut() {
            *d = d.mul_add(a, b).abs();
        }
    }
}
