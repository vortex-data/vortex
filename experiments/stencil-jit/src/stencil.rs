//! The stencil: a hand-written AVX2 kernel for "compare 32 lanes of i8/u8
//! to a broadcast constant, emit a 32-bit mask," with an 8-byte patch slot
//! that selects which of the 6 comparison ops the kernel computes.
//!
//! Layout (System V AMD64 calling convention):
//!   rdi = const u8*  (32 bytes of packed data, aligned or not)
//!   rsi =       u64  (constant in the low byte SIL)
//!   rdx =       u32* (output: 32-bit mask, one bit per lane)
//!
//! Prologue (identical for every op):
//!   vmovdqu  ymm0, [rdi]              ; load 32 lanes
//!   movzx    eax, sil                 ; isolate constant byte
//!   vmovd    xmm1, eax
//!   vpbroadcastb ymm1, xmm1           ; broadcast to all 32 lanes
//!   vpcmpeqb ymm2, ymm2, ymm2         ; ymm2 := all-ones (for inverts)
//!
//! Patch slot (exactly 8 bytes; each op encodes to <= 8):
//!   eq : vpcmpeqb ymm0,ymm0,ymm1 ; nop4
//!   neq: vpcmpeqb ymm0,ymm0,ymm1 ; vpxor ymm0,ymm0,ymm2
//!   gt : vpcmpgtb ymm0,ymm0,ymm1 ; nop4     (signed)
//!   lt : vpcmpgtb ymm0,ymm1,ymm0 ; nop4     (signed, operands swapped)
//!   ge : vpcmpgtb ymm0,ymm1,ymm0 ; vpxor ymm0,ymm0,ymm2   (!lt)
//!   le : vpcmpgtb ymm0,ymm0,ymm1 ; vpxor ymm0,ymm0,ymm2   (!gt)
//!
//! Epilogue (identical for every op):
//!   vpmovmskb eax, ymm0
//!   mov      [rdx], eax
//!   vzeroupper
//!   ret
//!
//! All six op encodings are exactly 8 bytes, so the surrounding instructions
//! stay at the same offsets. The compare is *signed* — for unsigned data,
//! flip the sign bit of both operands in the prologue (one more `vpxor`).
//! That's a follow-up; the point of this prototype is the splice mechanism.

use core::ffi::c_void;

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
    vpcmpeqb     ymm2, ymm2, ymm2
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

/// `vpcmpeqb ymm0, ymm0, ymm1`
const VPCMPEQB_0_0_1: [u8; 4] = [0xC5, 0xFD, 0x74, 0xC1];
/// `vpcmpgtb ymm0, ymm0, ymm1` (signed: ymm0 > ymm1)
const VPCMPGTB_0_0_1: [u8; 4] = [0xC5, 0xFD, 0x64, 0xC1];
/// `vpcmpgtb ymm0, ymm1, ymm0` (signed: ymm1 > ymm0, i.e., a < constant)
const VPCMPGTB_0_1_0: [u8; 4] = [0xC5, 0xF5, 0x64, 0xC0];
/// `vpxor ymm0, ymm0, ymm2` — invert mask using the precomputed all-ones in ymm2.
const VPXOR_0_0_2: [u8; 4] = [0xC5, 0xFD, 0xEF, 0xC2];
/// 4-byte NOP padding.
const NOP4: [u8; 4] = [0x90; 4];

pub(crate) const EQ_PATCH: [u8; 8] = concat8(VPCMPEQB_0_0_1, NOP4);
pub(crate) const NEQ_PATCH: [u8; 8] = concat8(VPCMPEQB_0_0_1, VPXOR_0_0_2);
pub(crate) const GT_PATCH: [u8; 8] = concat8(VPCMPGTB_0_0_1, NOP4);
pub(crate) const LT_PATCH: [u8; 8] = concat8(VPCMPGTB_0_1_0, NOP4);
pub(crate) const GE_PATCH: [u8; 8] = concat8(VPCMPGTB_0_1_0, VPXOR_0_0_2);
pub(crate) const LE_PATCH: [u8; 8] = concat8(VPCMPGTB_0_0_1, VPXOR_0_0_2);

const fn concat8(a: [u8; 4], b: [u8; 4]) -> [u8; 8] {
    [a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3]]
}

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

/// Length of the patch slot. Asserted to match all six op encodings.
pub(crate) fn patch_len() -> usize {
    let s = (&raw const PATCH_START).cast::<u8>();
    let e = (&raw const PATCH_END).cast::<u8>();
    let len = unsafe { e.offset_from(s) as usize };
    debug_assert_eq!(len, 8);
    len
}
