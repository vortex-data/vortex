//! Copy-and-patch JIT prototype for fused SIMD compare + optional FFoR-add
//! on packed `u8` lanes.
//!
//! Two kernel shapes:
//!
//! * [`Kernel`] processes a single 32-byte block per call.
//! * [`BulkKernel`] processes `n_blocks` 32-byte blocks per call, with a
//!   2x-unrolled inner loop and the load+FFoR-add fused into one memory-
//!   operand `vpaddb`. `n_blocks` must be even.
//!
//! ## Calling convention (SystemV AMD64)
//!
//! ```text
//! Single-block (Kernel):
//!   rdi = const u8*  -- 32 bytes of packed data
//!   rsi =       u64  -- compare constant; SIL is read
//!   rdx =       u32* -- 4 bytes of output mask
//!   rcx =       u64  -- FFoR reference; CL is read
//!
//! Bulk (BulkKernel):
//!   rdi = const u8*  -- n_blocks * 32 bytes
//!   rsi =       u64  -- compare constant; SIL is read
//!   rdx =       u32* -- n_blocks * 4 bytes of output
//!   rcx =       u64  -- FFoR reference; CL is read
//!    r8 =       u64  -- n_blocks; must be even or zero
//! ```
//!
//! ## Register convention
//!
//! ```text
//!   ymm0  = current data lane vector (input and output of each fragment)
//!   ymm1  = broadcast(compare constant)
//!   ymm2  = all-ones (used by invert patches: neq/ge/le)
//!   ymm3  = broadcast(FFoR reference)
//! ```
//!
//! ## Patch slots
//!
//! ```text
//! Single-block:
//!   SLOT 1  (5 B)  vpaddb ymm0, ymm3, [rdi]      (FFoR on)
//!                  vmovdqu ymm0, [rdi] + 1 NOP   (FFoR off)
//!   SLOT 2  (8 B)  compare op (six encodings)
//!
//! Bulk (per loop iteration; two blocks worth of slots):
//!   SLOT 1A (5 B)  same as single-block SLOT 1 against [rdi]
//!   SLOT 2A (8 B)  compare op
//!   SLOT 1B (5 B)  same shape but against [rdi+32]
//!   SLOT 2B (8 B)  compare op (typically identical to SLOT 2A)
//! ```

#![cfg(all(target_arch = "x86_64", target_os = "linux"))]

pub mod delta;
mod stencil;

use core::ptr::NonNull;
use std::io;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum CmpOp {
    Eq,
    Neq,
    Gt,
    Lt,
    Ge,
    Le,
}

impl CmpOp {
    pub const ALL: [Self; 6] = [
        Self::Eq,
        Self::Neq,
        Self::Gt,
        Self::Lt,
        Self::Ge,
        Self::Le,
    ];
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct ChainConfig {
    pub ffor: bool,
    pub op: CmpOp,
}

impl ChainConfig {
    pub const fn compare_only(op: CmpOp) -> Self {
        Self { ffor: false, op }
    }
    pub const fn ffor_then_compare(op: CmpOp) -> Self {
        Self { ffor: true, op }
    }
}

fn op_patch(op: CmpOp) -> &'static [u8; 8] {
    match op {
        CmpOp::Eq => &stencil::EQ_PATCH,
        CmpOp::Neq => &stencil::NEQ_PATCH,
        CmpOp::Gt => &stencil::GT_PATCH,
        CmpOp::Lt => &stencil::LT_PATCH,
        CmpOp::Ge => &stencil::GE_PATCH,
        CmpOp::Le => &stencil::LE_PATCH,
    }
}

/// Block B compare patches act on ymm4 instead of ymm0.
fn op_patch_b(op: CmpOp) -> &'static [u8; 8] {
    match op {
        CmpOp::Eq => &stencil::EQ_PATCH_B,
        CmpOp::Neq => &stencil::NEQ_PATCH_B,
        CmpOp::Gt => &stencil::GT_PATCH_B,
        CmpOp::Lt => &stencil::LT_PATCH_B,
        CmpOp::Ge => &stencil::GE_PATCH_B,
        CmpOp::Le => &stencil::LE_PATCH_B,
    }
}

fn page_size() -> usize {
    // SAFETY: sysconf is reentrant for _SC_PAGESIZE.
    let v = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if v <= 0 { 4096 } else { v as usize }
}

/// Allocate, copy, patch, mprotect-RX. Returns the page pointer.
///
/// # Safety
/// Each `(offset, src)` pair must lie within `bytes`.
unsafe fn materialize(
    bytes: &[u8],
    patches: &[(usize, &[u8])],
    page_len: usize,
) -> io::Result<NonNull<u8>> {
    // SAFETY: standard MAP_ANONYMOUS mmap.
    let raw = unsafe {
        libc::mmap(
            core::ptr::null_mut(),
            page_len,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
            -1,
            0,
        )
    };
    if raw == libc::MAP_FAILED {
        return Err(io::Error::last_os_error());
    }
    let page = NonNull::new(raw as *mut u8).expect("mmap non-null on success");

    // SAFETY: page_len >= bytes.len(); patches lie within bytes per caller.
    unsafe {
        core::ptr::copy_nonoverlapping(bytes.as_ptr(), page.as_ptr(), bytes.len());
        for (off, src) in patches {
            core::ptr::copy_nonoverlapping(src.as_ptr(), page.as_ptr().add(*off), src.len());
        }
    }

    // SAFETY: page came from a successful mmap.
    let rc = unsafe {
        libc::mprotect(
            page.as_ptr().cast(),
            page_len,
            libc::PROT_READ | libc::PROT_EXEC,
        )
    };
    if rc != 0 {
        let err = io::Error::last_os_error();
        // SAFETY: page came from mmap.
        unsafe {
            libc::munmap(page.as_ptr().cast(), page_len);
        }
        return Err(err);
    }
    Ok(page)
}

// =================== Single-block kernel ===================

pub struct Kernel {
    page: NonNull<u8>,
    page_len: usize,
    entry: unsafe extern "sysv64" fn(*const u8, u64, *mut u32, u64),
}

unsafe impl Send for Kernel {}
unsafe impl Sync for Kernel {}

impl Kernel {
    pub fn compile(config: ChainConfig) -> io::Result<Self> {
        let bytes = stencil::stencil_bytes();
        let load_patch: &[u8] = if config.ffor {
            &stencil::SINGLE_LOAD_ON
        } else {
            &stencil::SINGLE_LOAD_OFF
        };
        let patches = [
            (stencil::ffor_offset(), load_patch),
            (stencil::op_offset(), &op_patch(config.op)[..]),
        ];
        let page_len = page_size();
        // SAFETY: patches inside `bytes`.
        let page = unsafe { materialize(bytes, &patches, page_len)? };
        // SAFETY: the page now holds a valid sysv64 function.
        let entry: unsafe extern "sysv64" fn(*const u8, u64, *mut u32, u64) =
            unsafe { core::mem::transmute(page.as_ptr()) };
        Ok(Self {
            page,
            page_len,
            entry,
        })
    }

    /// # Safety
    /// `packed` must point to at least 32 readable bytes; `out` to 4 writable.
    pub unsafe fn call(&self, packed: *const u8, constant: u8, out: *mut u32, ffor_ref: u8) {
        // SAFETY: caller upholds buffer windows.
        unsafe { (self.entry)(packed, u64::from(constant), out, u64::from(ffor_ref)) }
    }
}

impl Drop for Kernel {
    fn drop(&mut self) {
        // SAFETY: page + page_len from materialize.
        unsafe {
            libc::munmap(self.page.as_ptr().cast(), self.page_len);
        }
    }
}

// =================== Bulk kernel (2x unrolled) ===================

pub struct BulkKernel {
    page: NonNull<u8>,
    page_len: usize,
    entry: unsafe extern "sysv64" fn(*const u8, u64, *mut u32, u64, u64),
}

unsafe impl Send for BulkKernel {}
unsafe impl Sync for BulkKernel {}

impl BulkKernel {
    pub fn compile(config: ChainConfig) -> io::Result<Self> {
        let bytes = stencil::bulk_bytes();
        let (load_a, load_b): (&[u8], &[u8]) = if config.ffor {
            (&stencil::BULK_LOAD_A_ON, &stencil::BULK_LOAD_B_ON)
        } else {
            (&stencil::BULK_LOAD_A_OFF, &stencil::BULK_LOAD_B_OFF)
        };
        let patches = [
            (stencil::bulk_ffor_a_offset(), load_a),
            (stencil::bulk_op_a_offset(), &op_patch(config.op)[..]),
            (stencil::bulk_ffor_b_offset(), load_b),
            (stencil::bulk_op_b_offset(), &op_patch_b(config.op)[..]),
        ];
        let page_len = page_size();
        // SAFETY: patches inside `bytes`.
        let page = unsafe { materialize(bytes, &patches, page_len)? };
        // SAFETY: the page now holds a valid sysv64 function.
        let entry: unsafe extern "sysv64" fn(*const u8, u64, *mut u32, u64, u64) =
            unsafe { core::mem::transmute(page.as_ptr()) };
        Ok(Self {
            page,
            page_len,
            entry,
        })
    }

    /// Process `n_blocks` 32-byte blocks.
    ///
    /// # Safety
    /// `packed` must point to at least `n_blocks * 32` readable bytes; `out`
    /// to `n_blocks * 4` writable bytes. `n_blocks` must be even (or zero).
    pub unsafe fn call(
        &self,
        packed: *const u8,
        constant: u8,
        out: *mut u32,
        ffor_ref: u8,
        n_blocks: usize,
    ) {
        debug_assert!(n_blocks.is_multiple_of(2), "bulk kernel requires even n_blocks");
        // SAFETY: caller upholds buffer windows.
        unsafe {
            (self.entry)(
                packed,
                u64::from(constant),
                out,
                u64::from(ffor_ref),
                n_blocks as u64,
            )
        }
    }
}

impl Drop for BulkKernel {
    fn drop(&mut self) {
        // SAFETY: page + page_len from materialize.
        unsafe {
            libc::munmap(self.page.as_ptr().cast(), self.page_len);
        }
    }
}

// =================== Specialized eq kernel (constants baked) ===================
//
// JIT-only win: the constants are known at kernel-compile time (the query
// planner supplies them), so the FFoR-add `(x + r) == c` is algebraically
// folded into `x == (c - r mod 256)`. The kernel body becomes a single
// memory-operand `vpcmpeqb` per block — no separate add, no separate load.
// AOT can't do this if the constants are runtime parameters.

/// A specialized fused kernel for `(x + ffor_ref) == constant` with both
/// constants baked in at JIT-compile time. 4x unrolled bulk loop.
pub struct SpecializedKernel {
    page: NonNull<u8>,
    page_len: usize,
    entry: unsafe extern "sysv64" fn(*const u8, *mut u32, u64),
}

unsafe impl Send for SpecializedKernel {}
unsafe impl Sync for SpecializedKernel {}

impl SpecializedKernel {
    /// Compile a specialized kernel that computes `(x + ffor_ref) == constant`
    /// for each lane. Both arguments are baked into the emitted code.
    pub fn compile_eq(constant: u8, ffor_ref: u8) -> io::Result<Self> {
        let effective_const = constant.wrapping_sub(ffor_ref);
        let bytes = stencil::spec_bytes();
        let patch = stencil::spec_const_patch(effective_const);
        let patches = [(stencil::spec_const_offset(), &patch[..])];
        let page_len = page_size();
        // SAFETY: patch lies inside bytes.
        let page = unsafe { materialize(bytes, &patches, page_len)? };
        // SAFETY: emitted page holds a valid sysv64 function.
        let entry: unsafe extern "sysv64" fn(*const u8, *mut u32, u64) =
            unsafe { core::mem::transmute(page.as_ptr()) };
        Ok(Self {
            page,
            page_len,
            entry,
        })
    }

    /// Run the kernel on `n_blocks` 32-byte blocks. `n_blocks` must be a
    /// multiple of 4 (the unroll factor).
    ///
    /// # Safety
    /// `packed` must point to at least `n_blocks * 32` readable bytes; `out`
    /// to at least `n_blocks * 4` writable bytes; `n_blocks % 4 == 0`.
    pub unsafe fn call(&self, packed: *const u8, out: *mut u32, n_blocks: usize) {
        debug_assert!(n_blocks.is_multiple_of(4), "specialized kernel requires 4-block multiples");
        // SAFETY: caller upholds buffer windows.
        unsafe { (self.entry)(packed, out, n_blocks as u64) }
    }
}

impl Drop for SpecializedKernel {
    fn drop(&mut self) {
        // SAFETY: page + page_len from materialize.
        unsafe {
            libc::munmap(self.page.as_ptr().cast(), self.page_len);
        }
    }
}

// =================== AVX-512 specialized eq kernel ===================
//
// Uses AVX-512BW's `vpcmpeqb k, zmm, [mem]` -> kmask + `kmovq` to write
// 64 bits of mask per 64-byte input block. Doubles the lane width and
// sidesteps the AVX2 `vpmovmskb` port-0 bottleneck. Theoretical max is
// ~224 GB/s at 3.5 GHz (vs ~112 GB/s for AVX2 D-spec).

/// AVX-512 variant of the specialized eq kernel. Caller must have verified
/// `avx512bw + avx512f` at runtime (use `std::is_x86_feature_detected!`).
pub struct SpecializedKernel512 {
    page: NonNull<u8>,
    page_len: usize,
    entry: unsafe extern "sysv64" fn(*const u8, *mut u8, u64),
}

unsafe impl Send for SpecializedKernel512 {}
unsafe impl Sync for SpecializedKernel512 {}

impl SpecializedKernel512 {
    /// Compile a specialized AVX-512 kernel for `(x + ffor_ref) == constant`.
    /// Both constants are baked at JIT-compile time and folded:
    /// `effective_c = constant.wrapping_sub(ffor_ref)`.
    pub fn compile_eq(constant: u8, ffor_ref: u8) -> io::Result<Self> {
        let effective_const = constant.wrapping_sub(ffor_ref);
        let bytes = stencil::spec512_bytes();
        let patch = stencil::spec_const_patch(effective_const);
        let patches = [(stencil::spec512_const_offset(), &patch[..])];
        let page_len = page_size();
        // SAFETY: patch lies inside bytes.
        let page = unsafe { materialize(bytes, &patches, page_len)? };
        // SAFETY: emitted page holds a valid sysv64 function.
        let entry: unsafe extern "sysv64" fn(*const u8, *mut u8, u64) =
            unsafe { core::mem::transmute(page.as_ptr()) };
        Ok(Self {
            page,
            page_len,
            entry,
        })
    }

    /// Run on `n_blocks` 32-byte AVX2-equivalent blocks. The kernel
    /// internally processes 64-byte AVX-512 blocks; `n_blocks` must be a
    /// multiple of 8.
    ///
    /// # Safety
    /// `packed` must point to at least `n_blocks * 32` readable bytes;
    /// `out` to at least `n_blocks * 4` writable bytes; `n_blocks % 8 == 0`.
    /// The caller must have verified AVX-512BW availability before calling.
    pub unsafe fn call(&self, packed: *const u8, out: *mut u32, n_blocks: usize) {
        debug_assert!(n_blocks.is_multiple_of(8), "AVX-512 kernel requires 8-block multiples");
        let n_zmm_iters = (n_blocks / 2) as u64; // each zmm op covers 2 blocks
        // SAFETY: caller upholds buffer windows; mask output is treated as raw bytes
        // (still aligned-compatible since u32 alignment <= u64).
        unsafe { (self.entry)(packed, out.cast::<u8>(), n_zmm_iters) }
    }
}

impl Drop for SpecializedKernel512 {
    fn drop(&mut self) {
        // SAFETY: page + page_len from materialize.
        unsafe {
            libc::munmap(self.page.as_ptr().cast(), self.page_len);
        }
    }
}

pub mod debug {
    use super::{CmpOp, op_patch, stencil};

    pub fn stencil_bytes() -> &'static [u8] {
        stencil::stencil_bytes()
    }
    pub fn ffor_offset() -> usize {
        stencil::ffor_offset()
    }
    pub fn ffor_len() -> usize {
        stencil::ffor_len()
    }
    pub fn op_offset() -> usize {
        stencil::op_offset()
    }
    pub fn op_len() -> usize {
        stencil::op_len()
    }
    pub fn single_load_off() -> &'static [u8] {
        &stencil::SINGLE_LOAD_OFF
    }
    pub fn single_load_on() -> &'static [u8] {
        &stencil::SINGLE_LOAD_ON
    }

    pub fn bulk_bytes() -> &'static [u8] {
        stencil::bulk_bytes()
    }
    pub fn bulk_ffor_a_offset() -> usize {
        stencil::bulk_ffor_a_offset()
    }
    pub fn bulk_op_a_offset() -> usize {
        stencil::bulk_op_a_offset()
    }
    pub fn bulk_ffor_b_offset() -> usize {
        stencil::bulk_ffor_b_offset()
    }
    pub fn bulk_op_b_offset() -> usize {
        stencil::bulk_op_b_offset()
    }

    pub fn op_patch_bytes(op: CmpOp) -> &'static [u8] {
        op_patch(op)
    }
}
