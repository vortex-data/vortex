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
//! This module implements that for a pipeline of affine ops `x = x*a + b` over
//! an `f64` tile (a faithful stand-in for the chained elementwise transforms a
//! decode tail performs: FoR add, ALP scale, ...). At build time we:
//!
//! 1. copy a fixed **prologue** (set up the loop, load `zmm0`),
//! 2. copy one self-contained **body** per op, patching its two `movabs`
//!    immediates (`a`, `b`) — the stencils are concatenated, not called,
//! 3. copy a fixed **epilogue** (store `zmm0`, advance, branch back),
//! 4. patch the loop back-edge `rel32` — a genuine relocation, since the branch
//!    distance depends on how many bodies were stitched in.
//!
//! The result is one AVX-512 loop with every constant folded as an immediate
//! and zero per-op overhead, built in ~`memcpy` time.

/// Maximum ops a stitched pipeline supports (one zmm register pair each, using
/// zmm1..zmm12 and leaving zmm0 as the accumulator).
pub const MAX_OPS: usize = 6;

#[cfg(all(target_arch = "x86_64", unix))]
mod imp {
    use std::ptr;

    use super::MAX_OPS;

    // Verbatim machine code bracketed by exported symbols. The constants live in
    // a pool appended after the code; the prologue broadcasts every constant
    // into a register *once* (addressed through `r8`), so the loop body is just
    // the `vfmadd`s — exactly what an AOT-compiled tail emits. Bodies are
    // fall-through (no `ret`); execution flows from one into the next.
    core::arch::global_asm!(
        // ---- prologue: rdi=src, rsi=dst; load all 6 constant pairs into zmm1..12 ----
        ".globl stitch_pro_start",
        ".globl stitch_pro_loop",
        ".globl stitch_pro_end",
        ".p2align 4",
        "stitch_pro_start:",
        ".byte 0x49, 0xb8", // movabs r8, imm64  (constant-pool address, patch at +2)
        ".byte 0, 0, 0, 0, 0, 0, 0, 0",
        "vbroadcastsd zmm1, qword ptr [r8 + 0]",
        "vbroadcastsd zmm2, qword ptr [r8 + 8]",
        "vbroadcastsd zmm3, qword ptr [r8 + 16]",
        "vbroadcastsd zmm4, qword ptr [r8 + 24]",
        "vbroadcastsd zmm5, qword ptr [r8 + 32]",
        "vbroadcastsd zmm6, qword ptr [r8 + 40]",
        "vbroadcastsd zmm7, qword ptr [r8 + 48]",
        "vbroadcastsd zmm8, qword ptr [r8 + 56]",
        "vbroadcastsd zmm9, qword ptr [r8 + 64]",
        "vbroadcastsd zmm10, qword ptr [r8 + 72]",
        "vbroadcastsd zmm11, qword ptr [r8 + 80]",
        "vbroadcastsd zmm12, qword ptr [r8 + 88]",
        "xor ecx, ecx",
        "stitch_pro_loop:",
        "vmovupd zmm0, [rdi + rcx*8]",
        "stitch_pro_end:",
        // ---- bodies: zmm0 = zmm0 * a_i + b_i, constants preloaded in registers ----
        ".globl stitch_b0_start",
        ".globl stitch_b0_end",
        "stitch_b0_start:",
        "vfmadd213pd zmm0, zmm1, zmm2",
        "stitch_b0_end:",
        ".globl stitch_b1_start",
        ".globl stitch_b1_end",
        "stitch_b1_start:",
        "vfmadd213pd zmm0, zmm3, zmm4",
        "stitch_b1_end:",
        ".globl stitch_b2_start",
        ".globl stitch_b2_end",
        "stitch_b2_start:",
        "vfmadd213pd zmm0, zmm5, zmm6",
        "stitch_b2_end:",
        ".globl stitch_b3_start",
        ".globl stitch_b3_end",
        "stitch_b3_start:",
        "vfmadd213pd zmm0, zmm7, zmm8",
        "stitch_b3_end:",
        ".globl stitch_b4_start",
        ".globl stitch_b4_end",
        "stitch_b4_start:",
        "vfmadd213pd zmm0, zmm9, zmm10",
        "stitch_b4_end:",
        ".globl stitch_b5_start",
        ".globl stitch_b5_end",
        "stitch_b5_start:",
        "vfmadd213pd zmm0, zmm11, zmm12",
        "stitch_b5_end:",
        // ---- epilogue: store, advance, branch back to the loop ----
        ".globl stitch_epi_start",
        ".globl stitch_epi_jb",
        ".globl stitch_epi_end",
        "stitch_epi_start:",
        "vmovupd [rsi + rcx*8], zmm0",
        "add rcx, 8",
        "cmp rcx, 1024",
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
        func: unsafe extern "C" fn(*const f64, *mut f64),
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
            // Each body is one fixed-length `vfmadd213pd`; reuse b0's length.
            let body_len = addr(stitch_b0_end) - body_starts[0];

            let pro_len = pro_end - pro;
            let epi_len = epi_end - epi;
            let pool_slots = 2 * MAX_OPS; // prologue always loads 12 constants
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

                // 4. constant pool: a_i,b_i for live ops, identity (1,0) elsewhere.
                let pool = code.add(pool_off).cast::<f64>();
                for slot in 0..MAX_OPS {
                    let (a, b) = ops.get(slot).copied().unwrap_or((1.0, 0.0));
                    pool.add(2 * slot).write_unaligned(a);
                    pool.add(2 * slot + 1).write_unaligned(b);
                }

                let rc = libc::mprotect(code.cast(), total, libc::PROT_READ | libc::PROT_EXEC);
                assert_eq!(rc, 0, "mprotect failed");

                StitchedAffine {
                    code,
                    len: total,
                    func: std::mem::transmute::<*mut u8, unsafe extern "C" fn(*const f64, *mut f64)>(
                        code,
                    ),
                }
            }
        }

        /// Run the stitched pipeline over one 1024-element `f64` tile.
        ///
        /// # Safety
        /// `src` and `dst` must each be valid for 1024 `f64`s.
        #[inline(always)]
        pub unsafe fn run_tile(&self, src: *const f64, dst: *mut f64) {
            // SAFETY: caller guarantees full-tile validity.
            unsafe { (self.func)(src, dst) }
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

/// Reference / AOT pipeline: the same affine ops fully inlined by the compiler.
/// `mul_add` emits the same fused-multiply-add the stitched code uses, so the
/// two agree bit-for-bit.
#[inline(always)]
pub fn affine_aot(ops: &[(f64, f64)], src: &[f64], dst: &mut [f64]) {
    for (s, d) in src.iter().zip(dst.iter_mut()) {
        let mut x = *s;
        for &(a, b) in ops {
            x = x.mul_add(a, b);
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
            *d = d.mul_add(a, b);
        }
    }
}
