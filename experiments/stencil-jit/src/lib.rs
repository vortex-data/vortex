//! Copy-and-patch JIT prototype for fused SIMD eq/neq on packed `u8` lanes.
//!
//! See `stencil.rs` for the AOT stencil. This module is the *runtime*: it
//! allocates an anonymous mmap region, copies the stencil bytes in, patches
//! the 8-byte op-slot to select eq vs neq, flips the page to PROT_READ |
//! PROT_EXEC, and exposes the result as a typed function pointer.
//!
//! Scope and non-goals:
//!   * x86-64 Linux, AVX2 only. No Windows, no macOS, no NEON, no AVX-512.
//!   * bit-width = 8 (no actual bit unpacking — that would require a
//!     per-width stencil, which is the obvious next step).
//!   * One patch site, two op kernels. The point is to demonstrate the
//!     mechanism, not to ship a production code generator.
//!
//! Why this isn't snake oil: rustc's monomorphization of fastlanes-rs'
//! `BitPackingCompare::unpack_cmp<W, B, V, F>` already gets you the AOT
//! version of the same fused kernels for free. A runtime JIT only earns its
//! complexity when the op or layout *cannot* be known at compile time —
//! e.g., user-supplied predicates from a query planner, dictionary code
//! mappings only known after a scan starts, or fused chains whose Cartesian
//! product is too large to ship as static code.

#![cfg(all(target_arch = "x86_64", target_os = "linux"))]

mod stencil;

use core::ptr::NonNull;
use std::io;

/// Comparison ops that this prototype can splice in.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum CmpOp {
    /// `lane == constant` for each of 32 `u8` lanes.
    Eq,
    /// `lane != constant` for each of 32 `u8` lanes.
    Neq,
}

/// A compiled kernel sitting in an mmap'd, executable page. The kernel
/// compares 32 packed `u8` lanes to a broadcast constant and writes a
/// 32-bit mask to the output pointer.
pub struct Kernel {
    page: NonNull<u8>,
    page_len: usize,
    entry: unsafe extern "sysv64" fn(*const u8, u64, *mut u32),
}

// SAFETY: the page is owned by the Kernel, mprotect made it RX, and we hand
// out only `&self` to invoke it. The function itself only reads from the
// caller-supplied pointers.
unsafe impl Send for Kernel {}
unsafe impl Sync for Kernel {}

impl Kernel {
    /// Materialize a kernel for `op` by copying the AOT stencil into a
    /// fresh executable page and splicing the patch slot.
    pub fn compile(op: CmpOp) -> io::Result<Self> {
        let bytes = stencil::stencil_bytes();
        let patch_off = stencil::patch_offset();
        let patch_len = stencil::patch_len();

        let page_size = page_size();
        let page_len = page_size; // stencil is ~30 bytes; one page is plenty.

        // mmap PROT_READ | PROT_WRITE. We never mmap as RWX simultaneously,
        // so the W^X invariant is preserved end-to-end.
        // SAFETY: mmap with NULL addr and MAP_ANONYMOUS|MAP_PRIVATE is the
        // standard portable way to allocate fresh pages.
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

        // Copy the stencil and splice in the op bytes.
        // SAFETY: we mmap'd `page_len >= bytes.len()` bytes of writable memory.
        unsafe {
            core::ptr::copy_nonoverlapping(bytes.as_ptr(), page.as_ptr(), bytes.len());
            let patch_src = match op {
                CmpOp::Eq => &stencil::EQ_PATCH,
                CmpOp::Neq => &stencil::NEQ_PATCH,
            };
            core::ptr::copy_nonoverlapping(
                patch_src.as_ptr(),
                page.as_ptr().add(patch_off),
                patch_len,
            );
        }

        // Flip to PROT_READ | PROT_EXEC. The mprotect syscall is a serializing
        // operation; subsequent calls in this thread will observe the patched
        // bytes. x86-64 has coherent instruction caches, so no explicit
        // icache flush is required.
        // SAFETY: page came from a successful mmap of `page_len` bytes.
        let rc = unsafe {
            libc::mprotect(
                page.as_ptr().cast(),
                page_len,
                libc::PROT_READ | libc::PROT_EXEC,
            )
        };
        if rc != 0 {
            let err = io::Error::last_os_error();
            // SAFETY: page came from mmap; munmap the same length we mapped.
            unsafe {
                libc::munmap(page.as_ptr().cast(), page_len);
            }
            return Err(err);
        }

        // SAFETY: the page now holds a valid System V AMD64 function with the
        // signature `extern "sysv64" fn(*const u8, u64, *mut u32)`. The bytes
        // came from a stencil we control, and the patched-in instructions are
        // the only mutation.
        let entry: unsafe extern "sysv64" fn(*const u8, u64, *mut u32) =
            unsafe { core::mem::transmute(page.as_ptr()) };

        Ok(Self {
            page,
            page_len,
            entry,
        })
    }

    /// Run the compiled kernel: read 32 `u8`s from `packed`, compare each to
    /// `constant`, write a 32-bit mask to `out`.
    ///
    /// # Safety
    /// `packed` must point to at least 32 readable bytes. `out` must point to
    /// 4 writable bytes.
    pub unsafe fn call(&self, packed: *const u8, constant: u8, out: *mut u32) {
        // SAFETY: caller upholds the read/write windows; the entry point is
        // a valid function pointer materialized in `compile`.
        unsafe { (self.entry)(packed, u64::from(constant), out) }
    }
}

impl Drop for Kernel {
    fn drop(&mut self) {
        // SAFETY: page + page_len came from a successful mmap and were never
        // remapped or freed.
        unsafe {
            libc::munmap(self.page.as_ptr().cast(), self.page_len);
        }
    }
}

fn page_size() -> usize {
    // SAFETY: sysconf is documented to be reentrant for _SC_PAGESIZE.
    let v = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if v <= 0 { 4096 } else { v as usize }
}

/// Inspection helpers, exposed for tests and curious readers.
pub mod debug {
    use super::stencil;

    /// The raw stencil bytes, exactly as the linker placed them in `.rodata`.
    pub fn stencil_bytes() -> &'static [u8] {
        stencil::stencil_bytes()
    }

    /// Byte offset of the patch slot within the stencil.
    pub fn patch_offset() -> usize {
        stencil::patch_offset()
    }

    /// Byte length of the patch slot.
    pub fn patch_len() -> usize {
        stencil::patch_len()
    }

    /// The 8 bytes the JIT splices in to select `eq`.
    pub fn eq_patch() -> &'static [u8] {
        &stencil::EQ_PATCH
    }

    /// The 8 bytes the JIT splices in to select `neq`.
    pub fn neq_patch() -> &'static [u8] {
        &stencil::NEQ_PATCH
    }
}
