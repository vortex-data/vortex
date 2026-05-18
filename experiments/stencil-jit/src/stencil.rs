//! The chained stencil: AVX2 kernel for "compare 32 lanes of i8/u8 to a
//! broadcast constant, optionally with a frame-of-reference add applied to
//! every lane first." Two 8-byte splice slots:
//!
//!   SLOT 1: FFoR-add (`vpaddb ymm0,ymm0,ymm3` + nop4), or 8 NOPs.
//!   SLOT 2: the compare op (same 6 encodings as before).
//!
//! Calling convention (System V AMD64):
//!   rdi = const u8*  (32 bytes of packed data)
//!   rsi =       u64  (compare constant in SIL)
//!   rdx =       u32* (output mask, 4 bytes)
//!   rcx =       u64  (FFoR reference in CL; ignored if SLOT 1 is NOPs)
//!
//! Prologue sets up:
//!   ymm0 = data, ymm1 = broadcast(constant), ymm2 = all-ones (for inverts),
//!   ymm3 = broadcast(ffor_ref). The ffor broadcast runs unconditionally;
//!   it's a few cycles and lets one stencil cover both "FFoR" and "no FFoR"
//!   configurations.

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
    movzx        eax, cl
    vmovd        xmm3, eax
    vpbroadcastb ymm3, xmm3
    .globl  __stencil_jit_ffor_start
    .hidden __stencil_jit_ffor_start
__stencil_jit_ffor_start:
    nop
    nop
    nop
    nop
    nop
    nop
    nop
    nop
    .globl  __stencil_jit_ffor_end
    .hidden __stencil_jit_ffor_end
__stencil_jit_ffor_end:
    .globl  __stencil_jit_op_start
    .hidden __stencil_jit_op_start
__stencil_jit_op_start:
    nop
    nop
    nop
    nop
    nop
    nop
    nop
    nop
    .globl  __stencil_jit_op_end
    .hidden __stencil_jit_op_end
__stencil_jit_op_end:
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
    #[link_name = "__stencil_jit_ffor_start"]
    static FFOR_START: c_void;
    #[link_name = "__stencil_jit_ffor_end"]
    static FFOR_END: c_void;
    #[link_name = "__stencil_jit_op_start"]
    static OP_START: c_void;
    #[link_name = "__stencil_jit_op_end"]
    static OP_END: c_void;
    #[link_name = "__stencil_jit_end"]
    static STENCIL_END: c_void;
}

/// `vpcmpeqb ymm0, ymm0, ymm1`
const VPCMPEQB_0_0_1: [u8; 4] = [0xC5, 0xFD, 0x74, 0xC1];
/// `vpcmpgtb ymm0, ymm0, ymm1` (signed: ymm0 > ymm1)
const VPCMPGTB_0_0_1: [u8; 4] = [0xC5, 0xFD, 0x64, 0xC1];
/// `vpcmpgtb ymm0, ymm1, ymm0` (signed: ymm1 > ymm0, i.e., a < constant)
const VPCMPGTB_0_1_0: [u8; 4] = [0xC5, 0xF5, 0x64, 0xC0];
/// `vpxor ymm0, ymm0, ymm2` — invert mask via the precomputed all-ones in ymm2.
const VPXOR_0_0_2: [u8; 4] = [0xC5, 0xFD, 0xEF, 0xC2];
/// `vpaddb ymm0, ymm0, ymm3` — add the broadcast FFoR reference.
const VPADDB_0_0_3: [u8; 4] = [0xC5, 0xFD, 0xFC, 0xC3];
/// 4-byte NOP padding.
const NOP4: [u8; 4] = [0x90; 4];

pub(crate) const EQ_PATCH: [u8; 8] = concat8(VPCMPEQB_0_0_1, NOP4);
pub(crate) const NEQ_PATCH: [u8; 8] = concat8(VPCMPEQB_0_0_1, VPXOR_0_0_2);
pub(crate) const GT_PATCH: [u8; 8] = concat8(VPCMPGTB_0_0_1, NOP4);
pub(crate) const LT_PATCH: [u8; 8] = concat8(VPCMPGTB_0_1_0, NOP4);
pub(crate) const GE_PATCH: [u8; 8] = concat8(VPCMPGTB_0_1_0, VPXOR_0_0_2);
pub(crate) const LE_PATCH: [u8; 8] = concat8(VPCMPGTB_0_0_1, VPXOR_0_0_2);

pub(crate) const FFOR_ADD_PATCH: [u8; 8] = concat8(VPADDB_0_0_3, NOP4);
pub(crate) const FFOR_NOP_PATCH: [u8; 8] = [0x90; 8];

const fn concat8(a: [u8; 4], b: [u8; 4]) -> [u8; 8] {
    [a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3]]
}

pub(crate) fn stencil_bytes() -> &'static [u8] {
    let start = (&raw const STENCIL_START).cast::<u8>();
    let end = (&raw const STENCIL_END).cast::<u8>();
    let len = unsafe { end.offset_from(start) };
    debug_assert!(len > 0);
    unsafe { core::slice::from_raw_parts(start, len as usize) }
}

pub(crate) fn ffor_offset() -> usize {
    let start = (&raw const STENCIL_START).cast::<u8>();
    let s = (&raw const FFOR_START).cast::<u8>();
    unsafe { s.offset_from(start) as usize }
}

pub(crate) fn ffor_len() -> usize {
    let s = (&raw const FFOR_START).cast::<u8>();
    let e = (&raw const FFOR_END).cast::<u8>();
    unsafe { e.offset_from(s) as usize }
}

pub(crate) fn op_offset() -> usize {
    let start = (&raw const STENCIL_START).cast::<u8>();
    let s = (&raw const OP_START).cast::<u8>();
    unsafe { s.offset_from(start) as usize }
}

pub(crate) fn op_len() -> usize {
    let s = (&raw const OP_START).cast::<u8>();
    let e = (&raw const OP_END).cast::<u8>();
    unsafe { e.offset_from(s) as usize }
}
