//! Copy-and-patch JIT prototype for fused SIMD compare + optional FFoR-add
//! on packed `u8` lanes. Two splice slots demonstrate fragment chaining.
//!
//! Calling convention for the materialized kernel:
//!   `fn(packed: *const u8, constant: u64, out: *mut u32, ffor_ref: u64)`
//!
//! `constant` and `ffor_ref` are passed as `u64` so the System V AMD64 ABI
//! places them in `rsi` and `rcx`; the stencil reads only the low byte
//! (`sil` / `cl`). `ffor_ref` is ignored when SLOT 1 holds 8 NOPs.

#![cfg(all(target_arch = "x86_64", target_os = "linux"))]

mod stencil;

use core::ptr::NonNull;
use std::io;

/// Signed compare ops supported by SLOT 2.
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

/// Composition for a single materialized kernel.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct ChainConfig {
    /// If `true`, SLOT 1 holds `vpaddb ymm0,ymm0,ymm3` (FFoR-add); else NOPs.
    pub ffor: bool,
    /// SLOT 2 always holds one of these.
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

/// A materialized kernel in an mmap'd executable page.
pub struct Kernel {
    page: NonNull<u8>,
    page_len: usize,
    entry: unsafe extern "sysv64" fn(*const u8, u64, *mut u32, u64),
}

unsafe impl Send for Kernel {}
unsafe impl Sync for Kernel {}

impl Kernel {
    /// Compile a kernel for `config`: copy the AOT stencil, splice both
    /// slots, mprotect to RX.
    pub fn compile(config: ChainConfig) -> io::Result<Self> {
        let bytes = stencil::stencil_bytes();
        let ffor_off = stencil::ffor_offset();
        let ffor_len = stencil::ffor_len();
        let op_off = stencil::op_offset();
        let op_len = stencil::op_len();

        let page_len = page_size();

        // SAFETY: mmap with NULL addr and MAP_ANONYMOUS|MAP_PRIVATE.
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
        let page = NonNull::new(raw as *mut u8).expect("mmap returned non-null on success");

        // SAFETY: page_len >= bytes.len(); both patch slots lie within bytes.
        unsafe {
            core::ptr::copy_nonoverlapping(bytes.as_ptr(), page.as_ptr(), bytes.len());

            let ffor_src: &[u8; 8] = if config.ffor {
                &stencil::FFOR_ADD_PATCH
            } else {
                &stencil::FFOR_NOP_PATCH
            };
            core::ptr::copy_nonoverlapping(ffor_src.as_ptr(), page.as_ptr().add(ffor_off), ffor_len);

            let op_src: &[u8; 8] = op_patch(config.op);
            core::ptr::copy_nonoverlapping(op_src.as_ptr(), page.as_ptr().add(op_off), op_len);
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
            // SAFETY: page came from mmap; same length.
            unsafe {
                libc::munmap(page.as_ptr().cast(), page_len);
            }
            return Err(err);
        }

        // SAFETY: bytes form a valid sysv64 function; relocations are pure
        // byte substitutions that preserve register/stack discipline.
        let entry: unsafe extern "sysv64" fn(*const u8, u64, *mut u32, u64) =
            unsafe { core::mem::transmute(page.as_ptr()) };

        Ok(Self {
            page,
            page_len,
            entry,
        })
    }

    /// Invoke the kernel.
    ///
    /// # Safety
    /// `packed` must point to at least 32 readable bytes; `out` to 4 writable.
    pub unsafe fn call(&self, packed: *const u8, constant: u8, out: *mut u32, ffor_ref: u8) {
        // SAFETY: caller upholds the buffer windows.
        unsafe { (self.entry)(packed, u64::from(constant), out, u64::from(ffor_ref)) }
    }
}

impl Drop for Kernel {
    fn drop(&mut self) {
        // SAFETY: page + page_len came from mmap and were never re-mapped.
        unsafe {
            libc::munmap(self.page.as_ptr().cast(), self.page_len);
        }
    }
}

fn page_size() -> usize {
    // SAFETY: sysconf is reentrant for _SC_PAGESIZE.
    let v = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if v <= 0 { 4096 } else { v as usize }
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

/// Inspection helpers for tests and the dump example.
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

    pub fn op_patch_bytes(op: CmpOp) -> &'static [u8] {
        op_patch(op)
    }

    pub fn ffor_add_patch() -> &'static [u8] {
        &stencil::FFOR_ADD_PATCH
    }

    pub fn ffor_nop_patch() -> &'static [u8] {
        &stencil::FFOR_NOP_PATCH
    }
}
