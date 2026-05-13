    .text
    .intel_syntax noprefix
    .globl loop_body
loop_body:
    # LLVM-MCA-BEGIN fused_u16_w11
    vpaddw ymm21, ymm0, ymmword ptr [rsp + 2*rax - 0x268]
    vpaddw ymm22, ymm0, ymmword ptr [rsp + 2*rax - 0x168]
    vpsllw ymm23, ymm22, 0xb
    vpternlogq ymm23, ymm21, ymm1, 0xf8
    vmovdqu64 ymmword ptr [rsp + 2*rax + 0x598], ymm23
    vpsrlw ymm21, ymm22, 0x5
    vpaddw ymm22, ymm0, ymmword ptr [rsp + 2*rax - 0x68]
    vpsllw ymm23, ymm22, 0x6
    vpternlogq ymm23, ymm21, ymm2, 0xf8
    vmovdqu64 ymmword ptr [rsp + 2*rax + 0x618], ymm23
    vpsrlw ymm21, ymm22, 0xa
    vpandq ymm21, ymm21, ymm3
    vmovdqu64 ymm22, ymmword ptr [rsp + 2*rax + 0x98]
    vpaddw ymm22, ymm22, ymm22
    vpaddw ymm22, ymm22, ymm4
    vpandq ymm22, ymm22, ymm5
    vpaddw ymm23, ymm0, ymmword ptr [rsp + 2*rax + 0x198]
    vpsllw ymm24, ymm23, 0xc
    vpternlogq ymm24, ymm21, ymm22, 0xfe
    vmovdqu64 ymmword ptr [rsp + 2*rax + 0x698], ymm24
    vpsrlw ymm21, ymm23, 0x4
    vpaddw ymm22, ymm0, ymmword ptr [rsp + 2*rax + 0x298]
    vpsllw ymm23, ymm22, 0x7
    vpternlogq ymm23, ymm21, ymm6, 0xf8
    vmovdqu64 ymmword ptr [rsp + 2*rax + 0x718], ymm23
    vpsrlw ymm21, ymm22, 0x9
    vpandq ymm21, ymm21, ymm7
    vpsllw ymm22, ymmword ptr [rsp + 2*rax + 0x398], 0x2
    vpaddw ymm22, ymm22, ymm8
    vpandq ymm22, ymm22, ymm9
    vpaddw ymm23, ymm0, ymmword ptr [rsp + 2*rax + 0x498]
    vpsllw ymm24, ymm23, 0xd
    vpternlogq ymm24, ymm21, ymm22, 0xfe
    vmovdqu64 ymmword ptr [rsp + 2*rax + 0x798], ymm24
    vpaddw ymm21, ymm0, ymmword ptr [rsp + 2*rax - 0x1e8]
    vpsrlw ymm22, ymm23, 0x3
    vpermt2b ymm22, ymm10, ymm21
    vmovdqu64 ymmword ptr [rsp + 2*rax + 0x818], ymm22
    vpsrlw ymm21, ymm21, 0x8
    vpandq ymm21, ymm21, ymm11
    vpsllw ymm22, ymmword ptr [rsp + 2*rax - 0xe8], 0x3
    vpaddw ymm22, ymm22, ymm12
    vpandq ymm22, ymm22, ymm13
    vpaddw ymm23, ymm0, ymmword ptr [rsp + 2*rax + 0x18]
    vpsllw ymm24, ymm23, 0xe
    vpternlogq ymm24, ymm21, ymm22, 0xfe
    vmovdqu64 ymmword ptr [rsp + 2*rax + 0x898], ymm24
    vpsrlw ymm21, ymm23, 0x2
    vpaddw ymm22, ymm0, ymmword ptr [rsp + 2*rax + 0x118]
    vpsllw ymm23, ymm22, 0x9
    vpternlogq ymm23, ymm21, ymm14, 0xf8
    vmovdqu64 ymmword ptr [rsp + 2*rax + 0x918], ymm23
    vpsrlw ymm21, ymm22, 0x7
    vpandq ymm21, ymm21, ymm15
    vpsllw ymm22, ymmword ptr [rsp + 2*rax + 0x218], 0x4
    vpaddw ymm22, ymm22, ymm16
    vpandq ymm22, ymm22, ymm17
    vpaddw ymm23, ymm0, ymmword ptr [rsp + 2*rax + 0x318]
    vpsllw ymm24, ymm23, 0xf
    vpternlogq ymm24, ymm21, ymm22, 0xfe
    vmovdqu64 ymmword ptr [rsp + 2*rax + 0x998], ymm24
    vpsrlw ymm21, ymm23, 0x1
    vpaddw ymm22, ymm0, ymmword ptr [rsp + 2*rax + 0x418]
    vpsllw ymm23, ymm22, 0xa
    vpternlogq ymm23, ymm21, ymm18, 0xf8
    vmovdqu64 ymmword ptr [rsp + 2*rax + 0xa18], ymm23
    vpsrlw ymm21, ymm22, 0x6
    vpsllw ymm22, ymmword ptr [rsp + 2*rax + 0x518], 0x5
    vpternlogq ymm22, ymm21, ymm19, 0xf8
    vpaddw ymm21, ymm22, ymm20
    vmovdqu64 ymmword ptr [rsp + 2*rax + 0xa98], ymm21
    # LLVM-MCA-END
    ret
