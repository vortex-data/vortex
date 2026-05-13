    .text
    .intel_syntax noprefix
    .globl loop_body
loop_body:
    # LLVM-MCA-BEGIN fused_u32_w10
    vpaddd ymm27, ymm0, ymmword ptr [rsp + rbx*4 + 0x298]
    vpslld ymm28, ymmword ptr [rsp + rbx*4 + 0x498], 0xa
    vpaddd ymm28, ymm28, ymm2
    vpandd ymm28, ymm28, ymm3
    vpternlogd ymm28, ymm27, ymm1, 0xf8
    vpslld ymm27, ymmword ptr [rsp + rbx*4 + 0x698], 0x14
    vpaddd ymm27, ymm27, ymm4
    vpandd ymm27, ymm27, ymm5
    vpaddd ymm29, ymm0, ymmword ptr [rsp + rbx*4 + 0x898]
    vpslld ymm30, ymm29, 0x1e
    vpternlogd ymm30, ymm28, ymm27, 0xfe
    vmovdqu64 ymmword ptr [rsp + rbx*4 + 0x1298], ymm30
    vpsrld ymm27, ymm29, 0x2
    vpslld ymm28, ymmword ptr [rsp + rbx*4 + 0xa98], 0x8
    vpaddd ymm28, ymm28, ymm6
    vpandd ymm28, ymm28, ymm7
    vpternlogd ymm28, ymm27, ymm8, 0xf8
    vpslld ymm27, ymmword ptr [rsp + rbx*4 + 0xc98], 0x12
    vpaddd ymm27, ymm27, ymm9
    vpandd ymm27, ymm27, ymm10
    vpaddd ymm29, ymm0, ymmword ptr [rsp + rbx*4 + 0xe98]
    vpslld ymm30, ymm29, 0x1c
    vpternlogd ymm30, ymm28, ymm27, 0xfe
    vmovdqu64 ymmword ptr [rsp + rbx*4 + 0x1318], ymm30
    vpsrld ymm27, ymm29, 0x4
    vpslld ymm28, ymmword ptr [rsp + rbx*4 + 0x1098], 0x6
    vpaddd ymm28, ymm28, ymm12
    vpandd ymm28, ymm28, ymm13
    vpternlogd ymm28, ymm27, ymm11, 0xf8
    vpslld ymm27, ymmword ptr [rsp + rbx*4 + 0x398], 0x10
    vpaddd ymm27, ymm27, ymm14
    vpaddd ymm29, ymm0, ymmword ptr [rsp + rbx*4 + 0x598]
    vpandd ymm27, ymm27, ymm15
    vpslld ymm30, ymm29, 0x1a
    vpternlogd ymm30, ymm28, ymm27, 0xfe
    vmovdqu64 ymmword ptr [rsp + rbx*4 + 0x1398], ymm30
    vpsrld ymm27, ymm29, 0x6
    vpslld ymm28, ymmword ptr [rsp + rbx*4 + 0x798], 0x4
    vpaddd ymm28, ymm28, ymm17
    vpandd ymm28, ymm28, ymm18
    vpslld ymm29, ymmword ptr [rsp + rbx*4 + 0x998], 0xe
    vpternlogd ymm28, ymm27, ymm16, 0xf8
    vpaddd ymm27, ymm29, ymm19
    vpandd ymm27, ymm27, ymm20
    vpaddd ymm29, ymm0, ymmword ptr [rsp + rbx*4 + 0xb98]
    vpslld ymm30, ymm29, 0x18
    vpternlogd ymm30, ymm28, ymm27, 0xfe
    vmovdqu64 ymmword ptr [rsp + rbx*4 + 0x1418], ymm30
    vpsrld ymm27, ymm29, 0x8
    vpslld ymm28, ymmword ptr [rsp + rbx*4 + 0xd98], 0x2
    vpaddd ymm28, ymm28, ymm22
    vpslld ymm29, ymmword ptr [rsp + rbx*4 + 0xf98], 0xc
    vpaddd ymm29, ymm29, ymm24
    vpslld ymm30, ymmword ptr [rsp + rbx*4 + 0x1198], 0x16
    vpternlogd ymm30, ymm27, ymm21, 0xf8
    vpternlogd ymm30, ymm28, ymm23, 0xf8
    vpternlogd ymm30, ymm29, ymm25, 0xf8
    vpaddd ymm27, ymm30, ymm26
    vmovdqu64 ymmword ptr [rsp + rbx*4 + 0x1498], ymm27
    # LLVM-MCA-END
    ret
