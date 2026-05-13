    .text
    .intel_syntax noprefix
    .globl loop_body
loop_body:
    # LLVM-MCA-BEGIN bare_u32_w10
    vmovdqu ymm1,ymmword ptr [rdi+rax*1+0x80]
    vpand  ymm2,ymm1,ymm0
    vmovdqu ymmword ptr [rsi+rax*1+0x80],ymm2
    vpsrld ymm2,ymm1,0x6
    vpand  ymm2,ymm2,ymm0
    vmovdqu ymmword ptr [rsi+rax*1+0x280],ymm2
    vpsrld ymm2,ymm1,0xc
    vpand  ymm2,ymm2,ymm0
    vmovdqu ymmword ptr [rsi+rax*1+0x480],ymm2
    vpsrld ymm2,ymm1,0x12
    vpand  ymm2,ymm2,ymm0
    vmovdqu ymmword ptr [rsi+rax*1+0x680],ymm2
    vpsrld ymm2,ymm1,0x18
    vpand  ymm2,ymm2,ymm0
    vmovdqu ymmword ptr [rsi+rax*1+0x880],ymm2
    vmovdqu ymm2,ymmword ptr [rdi+rax*1+0x100]
    vpshldd ymm1,ymm2,ymm1,0x2
    vpand  ymm1,ymm1,ymm0
    vmovdqu ymmword ptr [rsi+rax*1+0xa80],ymm1
    vpsrld ymm1,ymm2,0x4
    vpand  ymm1,ymm1,ymm0
    vmovdqu ymmword ptr [rsi+rax*1+0xc80],ymm1
    vpsrld ymm1,ymm2,0xa
    vpand  ymm1,ymm1,ymm0
    vmovdqu ymmword ptr [rsi+rax*1+0xe80],ymm1
    vpsrld ymm1,ymm2,0x10
    vpand  ymm1,ymm1,ymm0
    vmovdqu ymmword ptr [rsi+rax*1+0x180],ymm1
    vpsrld ymm1,ymm2,0x16
    vpand  ymm1,ymm1,ymm0
    vmovdqu ymmword ptr [rsi+rax*1+0x380],ymm1
    vmovdqu ymm1,ymmword ptr [rdi+rax*1+0x180]
    vpshldd ymm2,ymm1,ymm2,0x4
    vpand  ymm2,ymm2,ymm0
    vmovdqu ymmword ptr [rsi+rax*1+0x580],ymm2
    vpsrld ymm2,ymm1,0x2
    vpand  ymm2,ymm2,ymm0
    vmovdqu ymmword ptr [rsi+rax*1+0x780],ymm2
    vpsrld ymm2,ymm1,0x8
    vpand  ymm2,ymm2,ymm0
    vmovdqu ymmword ptr [rsi+rax*1+0x980],ymm2
    vpsrld ymm2,ymm1,0xe
    vpand  ymm2,ymm2,ymm0
    vmovdqu ymmword ptr [rsi+rax*1+0xb80],ymm2
    vpsrld ymm2,ymm1,0x14
    vpand  ymm2,ymm2,ymm0
    vmovdqu ymmword ptr [rsi+rax*1+0xd80],ymm2
    vpsrld ymm1,ymm1,0x1a
    vmovdqu ymm2,ymmword ptr [rdi+rax*1+0x200]
    vmovdqu ymmword ptr [rsi+rax*1+0xf80],ymm1
    vpand  ymm1,ymm2,ymm0
    vmovdqu ymmword ptr [rsi+rax*1+0x100],ymm1
    vpsrld ymm1,ymm2,0x6
    vpand  ymm1,ymm1,ymm0
    vmovdqu ymmword ptr [rsi+rax*1+0x300],ymm1
    vpsrld ymm1,ymm2,0xc
    vpand  ymm1,ymm1,ymm0
    vmovdqu ymmword ptr [rsi+rax*1+0x500],ymm1
    vpsrld ymm1,ymm2,0x12
    vpand  ymm1,ymm1,ymm0
    vmovdqu ymmword ptr [rsi+rax*1+0x700],ymm1
    vpsrld ymm1,ymm2,0x18
    vpand  ymm1,ymm1,ymm0
    vmovdqu ymmword ptr [rsi+rax*1+0x900],ymm1
    vmovdqu ymm1,ymmword ptr [rdi+rax*1+0x280]
    vpshldd ymm2,ymm1,ymm2,0x2
    vpand  ymm2,ymm2,ymm0
    vmovdqu ymmword ptr [rsi+rax*1+0xb00],ymm2
    vpsrld ymm2,ymm1,0x4
    vpand  ymm2,ymm2,ymm0
    vmovdqu ymmword ptr [rsi+rax*1+0xd00],ymm2
    vpsrld ymm2,ymm1,0xa
    vpand  ymm2,ymm2,ymm0
    vmovdqu ymmword ptr [rsi+rax*1+0xf00],ymm2
    vpsrld ymm2,ymm1,0x10
    vpand  ymm2,ymm2,ymm0
    vmovdqu ymmword ptr [rsi+rax*1+0x200],ymm2
    vpsrld ymm2,ymm1,0x16
    vpand  ymm2,ymm2,ymm0
    vmovdqu ymmword ptr [rsi+rax*1+0x400],ymm2
    vmovdqu ymm2,ymmword ptr [rdi+rax*1+0x300]
    vpshldd ymm1,ymm2,ymm1,0x4
    vpand  ymm1,ymm1,ymm0
    vmovdqu ymmword ptr [rsi+rax*1+0x600],ymm1
    vpsrld ymm1,ymm2,0x2
    vpand  ymm1,ymm1,ymm0
    vmovdqu ymmword ptr [rsi+rax*1+0x800],ymm1
    vpsrld ymm1,ymm2,0x8
    vpand  ymm1,ymm1,ymm0
    vmovdqu ymmword ptr [rsi+rax*1+0xa00],ymm1
    vpsrld ymm1,ymm2,0xe
    vpand  ymm1,ymm1,ymm0
    vmovdqu ymmword ptr [rsi+rax*1+0xc00],ymm1
    vpsrld ymm1,ymm2,0x14
    vpand  ymm1,ymm1,ymm0
    vmovdqu ymmword ptr [rsi+rax*1+0xe00],ymm1
    vpsrld ymm1,ymm2,0x1a
    vmovdqu ymmword ptr [rsi+rax*1+0x1000],ymm1
    add    rax,0x20
    jne    3217e0 <_ZN70_$LT$u32$u20$as$u20$fastlanes_kernel_bench..bitpacking..BitPacking$GT$6unpack17he29a2e0bd0f9dc25E+0x10>
    # LLVM-MCA-END
    ret
