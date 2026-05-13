//! The stencil: a hand-written AVX2 kernel for "compare 32 u8 lanes to a
//! broadcast constant, emit a 32-bit mask," with an 8-byte patch slot that
//! flips the kernel from `eq` to `neq`.
//!
//! Layout (System V AMD64 calling convention):
//!   rdi = const u8*  (32 bytes of packed data, aligned or not)
//!   rsi =       u64  (constant in the low byte SIL)
//!   rdx =       u32* (output: 32-bit mask, one bit per lane)
//!
//! Stencil:
//!   vmovdqu  ymm0, [rdi]              ; load 32 lanes
//!   movzx    eax, sil                 ; isolate constant byte
//!   vmovd    xmm1, eax
//!   vpbroadcastb ymm1, xmm1           ; broadcast to all 32 lanes
//!   vpcmpeqb ymm0, ymm0, ymm1         ; equality mask
//!   ; -------- 8-byte PATCH SLOT --------
//!   ; default = 8x NOP        -> eq behavior
//!   ; patched = vpcmpeqb ymm1,ymm1,ymm1   ; ymm1 := all-ones
//!   ;           vpxor    ymm0,ymm0,ymm1   ; invert mask -> neq
//!   ; -----------------------------------
//!   vpmovmskb eax, ymm0
//!   mov      [rdx], eax
//!   vzeroupper
//!   ret
//!
//! Both the eq form (8 NOPs) and the neq form (the 8-byte two-instruction
//! sequence) occupy exactly the same number of bytes, so the surrounding
//! instructions stay at the same offsets. This is the essence of copy-and-
//! patch: a single AOT-compiled stencil body, byte-spliced at runtime to
//! select one of several closely-related kernels.

use core::ffi::c_void;

// The stencil is emitted as a chunk of `.text` with four labels marking the
// start/end of the stencil and the start/end of the patch slot. We then read
// those bytes at runtime via `core::slice::from_raw_parts`.
//
// We do NOT call the stencil at its static address: x86's W^X policies and
// kernel hardening generally disallow self-modifying the loaded image. The
// JIT instead copies these bytes into an mmap'd page, patches the slot, then
// makes that page executable.
core::arch::global_asm!(
    r#"
    .section .rodata.stencil_jit, "a", @progbits
    .p2align 4
    .globl  __stencil_jit_start
    .hidden __stencil_jit_start
__stencil_jit_start:
    vmovdqu      ymm0, ymmword ptr [rdi]
    movzx        eax, sil
    vmovd        xmm1, eax
    vpbroadcastb ymm1, xmm1
    vpcmpeqb     ymm0, ymm0, ymm1
    .globl  __stencil_jit_patch_start
    .hidden __stencil_jit_patch_start
__stencil_jit_patch_start:
    nop
    nop
    nop
    nop
    nop
    nop
    nop
    nop
    .globl  __stencil_jit_patch_end
    .hidden __stencil_jit_patch_end
__stencil_jit_patch_end:
    vpmovmskb    eax, ymm0
    mov          dword ptr [rdx], eax
    vzeroupper
    ret
    .globl  __stencil_jit_end
    .hidden __stencil_jit_end
__stencil_jit_end:
    .section .text
"#
);

unsafe extern "C" {
    #[link_name = "__stencil_jit_start"]
    static STENCIL_START: c_void;
    #[link_name = "__stencil_jit_patch_start"]
    static PATCH_START: c_void;
    #[link_name = "__stencil_jit_patch_end"]
    static PATCH_END: c_void;
    #[link_name = "__stencil_jit_end"]
    static STENCIL_END: c_void;
}

/// The 8-byte sequence that replaces the NOPs to make the kernel compute
/// `!eq` instead of `eq`.
///
///   `vpcmpeqb ymm1, ymm1, ymm1`  -> `C5 F5 76 C9` (ymm1 := all-ones)
///   `vpxor    ymm0, ymm0, ymm1`  -> `C5 FD EF C1` (invert the eq mask)
pub(crate) const NEQ_PATCH: [u8; 8] = [0xC5, 0xF5, 0x76, 0xC9, 0xC5, 0xFD, 0xEF, 0xC1];

/// 8 x `nop` — the default contents of the patch slot, equivalent to eq.
pub(crate) const EQ_PATCH: [u8; 8] = [0x90; 8];

/// Borrow the AOT stencil bytes from `.rodata`.
pub(crate) fn stencil_bytes() -> &'static [u8] {
    let start = (&raw const STENCIL_START).cast::<u8>();
    let end = (&raw const STENCIL_END).cast::<u8>();
    let len = unsafe { end.offset_from(start) };
    debug_assert!(len > 0, "stencil end must follow start");
    unsafe { core::slice::from_raw_parts(start, len as usize) }
}

/// Byte offset of the patch slot within the stencil.
pub(crate) fn patch_offset() -> usize {
    let start = (&raw const STENCIL_START).cast::<u8>();
    let patch = (&raw const PATCH_START).cast::<u8>();
    unsafe { patch.offset_from(start) as usize }
}

/// Length of the patch slot. Asserted to match the splice constants.
pub(crate) fn patch_len() -> usize {
    let s = (&raw const PATCH_START).cast::<u8>();
    let e = (&raw const PATCH_END).cast::<u8>();
    let len = unsafe { e.offset_from(s) as usize };
    debug_assert_eq!(len, EQ_PATCH.len());
    debug_assert_eq!(len, NEQ_PATCH.len());
    len
}
