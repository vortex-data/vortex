//! The chained stencils.
//!
//! `__stencil_jit_*` is the single-block kernel.
//! `__stencil_jit_bulk_*` is the 2x-unrolled bulk kernel with **interleaved**
//! per-block work: block A uses `ymm0` end-to-end, block B uses `ymm4`. The
//! two chains share no architectural register so the OoO core never has to
//! rename through a write-after-write hazard, and both blocks' work issues
//! in parallel rather than back-to-back.
//!
//! Memory-operand `vpaddb` fuses the load with the FFoR-add into one µop.
//! Multi-byte `nopl` is used in the compare-slot padding so single-byte
//! 0x90 µops don't eat decode bandwidth.

use core::ffi::c_void;

core::arch::global_asm!(
    r#"
    .section .rodata.stencil_jit, "a", @progbits

    # ============ single-block stencil ============

    .p2align 4, 0x90
    .globl  __stencil_jit_start
    .hidden __stencil_jit_start
__stencil_jit_start:
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
    # 5 bytes: vmovdqu ymm0,[rdi+0] or vpaddb ymm0,ymm3,[rdi+0]
    nop ; nop ; nop ; nop ; nop
    .globl  __stencil_jit_ffor_end
    .hidden __stencil_jit_ffor_end
__stencil_jit_ffor_end:
    .globl  __stencil_jit_op_start
    .hidden __stencil_jit_op_start
__stencil_jit_op_start:
    nop ; nop ; nop ; nop ; nop ; nop ; nop ; nop
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

    # ============ bulk stencil (2x unrolled, INTERLEAVED) ============
    #
    # Block A flows through ymm0; block B flows through ymm4. The two
    # chains are interleaved instruction-by-instruction:
    #
    #   load A, load B, add+cmp A, add+cmp B, movmsk A, movmsk B, store A, store B
    #
    # Independent registers => the OoO core sees two parallel chains and
    # can pipeline successive loop iterations within the ROB.
    #
    # ABI matches the non-interleaved version:
    #   rdi = packed*  ; rsi = const ; rdx = out* ; rcx = ffor ; r8 = n_blocks (even)

    .p2align 4, 0x90
    .globl  __stencil_jit_bulk_start
    .hidden __stencil_jit_bulk_start
__stencil_jit_bulk_start:
    movzx        eax, sil
    vmovd        xmm1, eax
    vpbroadcastb ymm1, xmm1            # ymm1 = compare const
    vpcmpeqb     ymm2, ymm2, ymm2      # ymm2 = all-ones
    movzx        eax, cl
    vmovd        xmm3, eax
    vpbroadcastb ymm3, xmm3            # ymm3 = ffor ref
    test         r8, r8
    je           __stencil_jit_bulk_end_jmp
    .p2align 5, 0x90
__stencil_jit_bulk_loop:
    # ---- block A load+ffor (5 bytes), interleaved with block B's ----
    .globl  __stencil_jit_bulk_ffor_a_start
    .hidden __stencil_jit_bulk_ffor_a_start
__stencil_jit_bulk_ffor_a_start:
    nop ; nop ; nop ; nop ; nop
    .globl  __stencil_jit_bulk_ffor_a_end
    .hidden __stencil_jit_bulk_ffor_a_end
__stencil_jit_bulk_ffor_a_end:
    # ---- block B load+ffor (5 bytes) ----
    .globl  __stencil_jit_bulk_ffor_b_start
    .hidden __stencil_jit_bulk_ffor_b_start
__stencil_jit_bulk_ffor_b_start:
    nop ; nop ; nop ; nop ; nop
    .globl  __stencil_jit_bulk_ffor_b_end
    .hidden __stencil_jit_bulk_ffor_b_end
    # ---- block A compare (8 bytes) ----
    .globl  __stencil_jit_bulk_op_a_start
    .hidden __stencil_jit_bulk_op_a_start
__stencil_jit_bulk_op_a_start:
    nop ; nop ; nop ; nop ; nop ; nop ; nop ; nop
    .globl  __stencil_jit_bulk_op_a_end
    .hidden __stencil_jit_bulk_op_a_end
    # ---- block B compare (8 bytes) ----
    .globl  __stencil_jit_bulk_op_b_start
    .hidden __stencil_jit_bulk_op_b_start
__stencil_jit_bulk_op_b_start:
    nop ; nop ; nop ; nop ; nop ; nop ; nop ; nop
    .globl  __stencil_jit_bulk_op_b_end
    .hidden __stencil_jit_bulk_op_b_end
    # ---- block A movmsk + store ----
    vpmovmskb    eax, ymm0
    vpmovmskb    r9d, ymm4
    mov          dword ptr [rdx], eax
    mov          dword ptr [rdx + 4], r9d
    add          rdi, 64
    add          rdx, 8
    sub          r8, 2
    jne          __stencil_jit_bulk_loop
__stencil_jit_bulk_end_jmp:
    vzeroupper
    ret
    .globl  __stencil_jit_bulk_end
    .hidden __stencil_jit_bulk_end
__stencil_jit_bulk_end:

    # ============ specialized eq stencil (4x unrolled) ============
    #
    # All constants baked at JIT-compile time. Algebraic simplification
    # folds the FFoR-add into the broadcast constant: the kernel computes
    # `(x == c')` directly, where c' = c - ffor_ref (mod 256). The loop
    # body is one memory-operand `vpcmpeqb` per block — no separate load,
    # no separate add.
    #
    # ABI:
    #   rdi = const u8*  (n_blocks * 32 bytes; n_blocks must be multiple of 4)
    #   rsi =       u32* (n_blocks * 4 bytes of output masks)
    #   rdx =       u64  (n_blocks)
    #
    # Prologue patches: the broadcast of `c'` into ymm1 — patched by the
    # JIT after rustc emits a placeholder.

    .p2align 4, 0x90
    .globl  __stencil_jit_spec_start
    .hidden __stencil_jit_spec_start
__stencil_jit_spec_start:
    # 5-byte slot for `mov al, c'` (the JIT patches the immediate byte).
    .globl  __stencil_jit_spec_const_start
    .hidden __stencil_jit_spec_const_start
__stencil_jit_spec_const_start:
    nop ; nop ; nop ; nop ; nop
    .globl  __stencil_jit_spec_const_end
    .hidden __stencil_jit_spec_const_end
__stencil_jit_spec_const_end:
    vmovd        xmm1, eax
    vpbroadcastb ymm1, xmm1            # ymm1 = broadcast(c')
    test         rdx, rdx
    je           __stencil_jit_spec_end_jmp
    .p2align 5, 0x90
__stencil_jit_spec_loop:
    # 4-way unroll: blocks at [rdi+0], [rdi+32], [rdi+64], [rdi+96].
    # Each is a single memory-operand vpcmpeqb writing to a distinct ymm.
    vpcmpeqb     ymm0, ymm1, ymmword ptr [rdi + 0]
    vpcmpeqb     ymm4, ymm1, ymmword ptr [rdi + 32]
    vpcmpeqb     ymm5, ymm1, ymmword ptr [rdi + 64]
    vpcmpeqb     ymm6, ymm1, ymmword ptr [rdi + 96]
    vpmovmskb    eax, ymm0
    mov          dword ptr [rsi + 0], eax
    vpmovmskb    eax, ymm4
    mov          dword ptr [rsi + 4], eax
    vpmovmskb    eax, ymm5
    mov          dword ptr [rsi + 8], eax
    vpmovmskb    eax, ymm6
    mov          dword ptr [rsi + 12], eax
    add          rdi, 128
    add          rsi, 16
    sub          rdx, 4
    jne          __stencil_jit_spec_loop
__stencil_jit_spec_end_jmp:
    vzeroupper
    ret
    .globl  __stencil_jit_spec_end
    .hidden __stencil_jit_spec_end
__stencil_jit_spec_end:

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

    #[link_name = "__stencil_jit_bulk_start"]
    static BULK_START: c_void;
    #[link_name = "__stencil_jit_bulk_ffor_a_start"]
    static BULK_FFOR_A_START: c_void;
    #[link_name = "__stencil_jit_bulk_ffor_a_end"]
    static BULK_FFOR_A_END: c_void;
    #[link_name = "__stencil_jit_bulk_op_a_start"]
    static BULK_OP_A_START: c_void;
    #[link_name = "__stencil_jit_bulk_op_a_end"]
    static BULK_OP_A_END: c_void;
    #[link_name = "__stencil_jit_bulk_ffor_b_start"]
    static BULK_FFOR_B_START: c_void;
    #[link_name = "__stencil_jit_bulk_ffor_b_end"]
    static BULK_FFOR_B_END: c_void;
    #[link_name = "__stencil_jit_bulk_op_b_start"]
    static BULK_OP_B_START: c_void;
    #[link_name = "__stencil_jit_bulk_op_b_end"]
    static BULK_OP_B_END: c_void;
    #[link_name = "__stencil_jit_bulk_end"]
    static BULK_END: c_void;

    #[link_name = "__stencil_jit_spec_start"]
    static SPEC_START: c_void;
    #[link_name = "__stencil_jit_spec_const_start"]
    static SPEC_CONST_START: c_void;
    #[link_name = "__stencil_jit_spec_const_end"]
    static SPEC_CONST_END: c_void;
    #[link_name = "__stencil_jit_spec_end"]
    static SPEC_END: c_void;
}

// ---- single-block patches (operate on ymm0) ----

/// `vmovdqu ymm0, [rdi+0]` — 5 bytes, no NOP padding.
pub(crate) const SINGLE_LOAD_OFF: [u8; 5] = [0xC5, 0xFE, 0x6F, 0x47, 0x00];
/// `vpaddb ymm0, ymm3, [rdi+0]` — 5 bytes.
pub(crate) const SINGLE_LOAD_ON: [u8; 5] = [0xC5, 0xE5, 0xFC, 0x47, 0x00];

// ---- bulk patches: block A on ymm0, block B on ymm4 ----

/// `vmovdqu ymm0, [rdi+0]` — block A load, no FFoR.
pub(crate) const BULK_LOAD_A_OFF: [u8; 5] = [0xC5, 0xFE, 0x6F, 0x47, 0x00];
/// `vpaddb ymm0, ymm3, [rdi+0]` — block A load+FFoR-add.
pub(crate) const BULK_LOAD_A_ON: [u8; 5] = [0xC5, 0xE5, 0xFC, 0x47, 0x00];
/// `vmovdqu ymm4, [rdi+32]` — block B load, no FFoR. Encoding writes to ymm4
/// (reg=100 in ModR/M), so the slot is 5 bytes: `C5 FE 6F 67 20`.
pub(crate) const BULK_LOAD_B_OFF: [u8; 5] = [0xC5, 0xFE, 0x6F, 0x67, 0x20];
/// `vpaddb ymm4, ymm3, [rdi+32]` — block B load+FFoR-add (dest=ymm4).
pub(crate) const BULK_LOAD_B_ON: [u8; 5] = [0xC5, 0xE5, 0xFC, 0x67, 0x20];

// ---- compare op patches ----

// On ymm0 (block A):
const VPCMPEQB_0_0_1: [u8; 4] = [0xC5, 0xFD, 0x74, 0xC1];
const VPCMPGTB_0_0_1: [u8; 4] = [0xC5, 0xFD, 0x64, 0xC1];
const VPCMPGTB_0_1_0: [u8; 4] = [0xC5, 0xF5, 0x64, 0xC0];
const VPXOR_0_0_2: [u8; 4] = [0xC5, 0xFD, 0xEF, 0xC2];

// On ymm4 (block B):
const VPCMPEQB_4_4_1: [u8; 4] = [0xC5, 0xDD, 0x74, 0xE1];
const VPCMPGTB_4_4_1: [u8; 4] = [0xC5, 0xDD, 0x64, 0xE1];
const VPCMPGTB_4_1_4: [u8; 4] = [0xC5, 0xF5, 0x64, 0xE4];
const VPXOR_4_4_2: [u8; 4] = [0xC5, 0xDD, 0xEF, 0xE2];

/// 4-byte multi-byte NOP: `nopl 0x0(%rax)`. One decoded instruction, zero
/// execution µops.
const NOPL4: [u8; 4] = [0x0F, 0x1F, 0x40, 0x00];

const fn concat8(a: [u8; 4], b: [u8; 4]) -> [u8; 8] {
    [a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3]]
}

// Block A compare patches (act on ymm0):
pub(crate) const EQ_PATCH: [u8; 8] = concat8(VPCMPEQB_0_0_1, NOPL4);
pub(crate) const NEQ_PATCH: [u8; 8] = concat8(VPCMPEQB_0_0_1, VPXOR_0_0_2);
pub(crate) const GT_PATCH: [u8; 8] = concat8(VPCMPGTB_0_0_1, NOPL4);
pub(crate) const LT_PATCH: [u8; 8] = concat8(VPCMPGTB_0_1_0, NOPL4);
pub(crate) const GE_PATCH: [u8; 8] = concat8(VPCMPGTB_0_1_0, VPXOR_0_0_2);
pub(crate) const LE_PATCH: [u8; 8] = concat8(VPCMPGTB_0_0_1, VPXOR_0_0_2);

// Block B compare patches (act on ymm4):
pub(crate) const EQ_PATCH_B: [u8; 8] = concat8(VPCMPEQB_4_4_1, NOPL4);
pub(crate) const NEQ_PATCH_B: [u8; 8] = concat8(VPCMPEQB_4_4_1, VPXOR_4_4_2);
pub(crate) const GT_PATCH_B: [u8; 8] = concat8(VPCMPGTB_4_4_1, NOPL4);
pub(crate) const LT_PATCH_B: [u8; 8] = concat8(VPCMPGTB_4_1_4, NOPL4);
pub(crate) const GE_PATCH_B: [u8; 8] = concat8(VPCMPGTB_4_1_4, VPXOR_4_4_2);
pub(crate) const LE_PATCH_B: [u8; 8] = concat8(VPCMPGTB_4_4_1, VPXOR_4_4_2);

// ---- descriptors ----

pub(crate) fn stencil_bytes() -> &'static [u8] {
    bytes_between(&raw const STENCIL_START, &raw const STENCIL_END)
}
pub(crate) fn ffor_offset() -> usize {
    offset(&raw const STENCIL_START, &raw const FFOR_START)
}
pub(crate) fn ffor_len() -> usize {
    offset(&raw const FFOR_START, &raw const FFOR_END)
}
pub(crate) fn op_offset() -> usize {
    offset(&raw const STENCIL_START, &raw const OP_START)
}
pub(crate) fn op_len() -> usize {
    offset(&raw const OP_START, &raw const OP_END)
}

pub(crate) fn bulk_bytes() -> &'static [u8] {
    bytes_between(&raw const BULK_START, &raw const BULK_END)
}
pub(crate) fn bulk_ffor_a_offset() -> usize {
    offset(&raw const BULK_START, &raw const BULK_FFOR_A_START)
}
pub(crate) fn bulk_op_a_offset() -> usize {
    offset(&raw const BULK_START, &raw const BULK_OP_A_START)
}
pub(crate) fn bulk_ffor_b_offset() -> usize {
    offset(&raw const BULK_START, &raw const BULK_FFOR_B_START)
}
pub(crate) fn bulk_op_b_offset() -> usize {
    offset(&raw const BULK_START, &raw const BULK_OP_B_START)
}

// ---- specialized-stencil descriptors ----

pub(crate) fn spec_bytes() -> &'static [u8] {
    bytes_between(&raw const SPEC_START, &raw const SPEC_END)
}
pub(crate) fn spec_const_offset() -> usize {
    offset(&raw const SPEC_START, &raw const SPEC_CONST_START)
}
pub(crate) fn spec_const_len() -> usize {
    offset(&raw const SPEC_CONST_START, &raw const SPEC_CONST_END)
}

/// Build the 5-byte patch that loads the constant `c` into AL:
/// `mov al, imm8` is 2 bytes (`B0 imm8`); pad with multi-byte NOP.
/// Followed by `vmovd xmm1, eax; vpbroadcastb ymm1, xmm1` in the stencil body.
pub(crate) fn spec_const_patch(c: u8) -> [u8; 5] {
    // mov al, c  =  B0 imm8 (2 bytes)
    // nopl 0x0(%rax)  =  0F 1F 00 ? actually nopw is 66 90, 3-byte nop is 0F 1F 00.
    // We need 3 more bytes of NOP after the 2-byte mov. Use `nop dword ptr [rax]`
    // = 0F 1F 00 (3 bytes, 1 µop).
    [0xB0, c, 0x0F, 0x1F, 0x00]
}

fn bytes_between(start: *const c_void, end: *const c_void) -> &'static [u8] {
    let s = start.cast::<u8>();
    let e = end.cast::<u8>();
    let n = unsafe { e.offset_from(s) };
    debug_assert!(n > 0);
    unsafe { core::slice::from_raw_parts(s, n as usize) }
}

fn offset(from: *const c_void, to: *const c_void) -> usize {
    let f = from.cast::<u8>();
    let t = to.cast::<u8>();
    unsafe { t.offset_from(f) as usize }
}
