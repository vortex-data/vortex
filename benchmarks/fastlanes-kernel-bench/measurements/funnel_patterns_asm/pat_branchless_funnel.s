
/home/user/vortex/target/release/deps/funnel_patterns-b50fc531d4f952e4:     file format elf64-x86-64


Disassembly of section .text:

00000000000599d0 <_ZN15funnel_patterns21pat_branchless_funnel17h1c386f057c3d3ba2E>:
   599d0:	lea    0x8(%rdi),%rax
   599d4:	vpbroadcastq %rsi,%zmm0
   599da:	vmovdqa64 -0x39ae4(%rip),%zmm1        # 1ff00 <anon.3c8f25b610820e10fad07aea247a682a.12.llvm.15771806882712607681+0x10e8>
   599e4:	vpbroadcastq -0x3ac9e(%rip),%zmm2        # 1ed50 <anon.576c71cbdbd6e3deefe97f3cb576ad67.20.llvm.9236128649512152758+0x22c>
   599ee:	vpbroadcastq -0x3acf8(%rip),%zmm3        # 1ed00 <anon.576c71cbdbd6e3deefe97f3cb576ad67.20.llvm.9236128649512152758+0x1dc>
   599f8:	xor    %ecx,%ecx
   599fa:	vpxor  %xmm4,%xmm4,%xmm4
   599fe:	vpbroadcastq -0x3acb0(%rip),%zmm5        # 1ed58 <anon.576c71cbdbd6e3deefe97f3cb576ad67.20.llvm.9236128649512152758+0x234>
   59a08:	vpbroadcastq -0x3ac2a(%rip),%zmm6        # 1ede8 <anon.576c71cbdbd6e3deefe97f3cb576ad67.20.llvm.9236128649512152758+0x2c4>
   59a12:	vpbroadcastq -0x3ae94(%rip),%zmm7        # 1eb88 <anon.576c71cbdbd6e3deefe97f3cb576ad67.20.llvm.9236128649512152758+0x64>
   59a1c:	vpbroadcastq -0x3ae56(%rip),%zmm8        # 1ebd0 <anon.576c71cbdbd6e3deefe97f3cb576ad67.20.llvm.9236128649512152758+0xac>
   59a26:	vpbroadcastq -0x3ac40(%rip),%zmm9        # 1edf0 <anon.576c71cbdbd6e3deefe97f3cb576ad67.20.llvm.9236128649512152758+0x2cc>
   59a30:	vpmullq %zmm2,%zmm1,%zmm10
   59a36:	vpsrlq $0x6,%zmm10,%zmm11
   59a3d:	kxnorw %k0,%k0,%k1
   59a41:	vpxor  %xmm12,%xmm12,%xmm12
   59a46:	vpgatherqq (%rdi,%zmm11,8),%zmm12{%k1}
   59a4d:	vpminuq %zmm3,%zmm11,%zmm11
   59a53:	vpandq %zmm5,%zmm10,%zmm13
   59a59:	vpsrlvq %zmm13,%zmm12,%zmm12
   59a5f:	kxnorw %k0,%k0,%k1
   59a63:	vpxor  %xmm13,%xmm13,%xmm13
   59a68:	vpgatherqq (%rax,%zmm11,8),%zmm13{%k1}
   59a6f:	vpsubq %zmm10,%zmm4,%zmm11
   59a75:	vpandq %zmm5,%zmm11,%zmm11
   59a7b:	vpsllvq %zmm11,%zmm13,%zmm11
   59a81:	vpternlogq $0xc8,%zmm12,%zmm6,%zmm11
   59a88:	vpaddq %zmm0,%zmm11,%zmm11
   59a8e:	vmovdqu64 %zmm11,(%rdx,%rcx,8)
   59a95:	vpaddq %zmm7,%zmm10,%zmm11
   59a9b:	vpsrlq $0x6,%zmm11,%zmm12
   59aa2:	vpminuq %zmm3,%zmm12,%zmm13
   59aa8:	kxnorw %k0,%k0,%k1
   59aac:	vpxor  %xmm14,%xmm14,%xmm14
   59ab1:	vpgatherqq (%rdi,%zmm12,8),%zmm14{%k1}
   59ab8:	kxnorw %k0,%k0,%k1
   59abc:	vpxor  %xmm12,%xmm12,%xmm12
   59ac1:	vpgatherqq (%rax,%zmm13,8),%zmm12{%k1}
   59ac8:	vpandq %zmm5,%zmm11,%zmm11
   59ace:	vpsrlvq %zmm11,%zmm14,%zmm11
   59ad4:	vpsubq %zmm10,%zmm8,%zmm10
   59ada:	vpandq %zmm5,%zmm10,%zmm10
   59ae0:	vpsllvq %zmm10,%zmm12,%zmm10
   59ae6:	vpternlogq $0xc8,%zmm11,%zmm6,%zmm10
   59aed:	vpaddq %zmm0,%zmm10,%zmm10
   59af3:	vmovdqu64 %zmm10,0x40(%rdx,%rcx,8)
   59afb:	add    $0x10,%rcx
   59aff:	vpaddq %zmm9,%zmm1,%zmm1
   59b05:	cmp    $0x400,%rcx
   59b0c:	jne    59a30 <_ZN15funnel_patterns21pat_branchless_funnel17h1c386f057c3d3ba2E+0x60>
   59b12:	vzeroupper
   59b15:	ret

Disassembly of section .init:

Disassembly of section .fini:

Disassembly of section .plt:
