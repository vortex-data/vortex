
/home/user/vortex/target/release/deps/funnel_patterns-b50fc531d4f952e4:     file format elf64-x86-64


Disassembly of section .text:

0000000000059210 <_ZN15funnel_patterns12pat_u128_cat17h8e6044c09a682e20E>:
   59210:	push   %rbp
   59211:	push   %r15
   59213:	push   %r14
   59215:	push   %r13
   59217:	push   %r12
   59219:	push   %rbx
   5921a:	lea    0x8(%rdi),%rax
   5921e:	vmovdqa64 -0x39968(%rip),%zmm0        # 1f8c0 <anon.3c8f25b610820e10fad07aea247a682a.12.llvm.15771806882712607681+0xaa8>
   59228:	vpbroadcastq %rsi,%zmm1
   5922e:	xor    %esi,%esi
   59230:	vpbroadcastq -0x3a4ea(%rip),%zmm2        # 1ed50 <anon.576c71cbdbd6e3deefe97f3cb576ad67.20.llvm.9236128649512152758+0x22c>
   5923a:	vpbroadcastd -0x3a7c3(%rip),%ymm3        # 1ea80 <anon.85b3346390def971eb9b6c1d8d7df145.1.llvm.1722333993647025796+0xc>
   59243:	vpbroadcastq -0x3a54d(%rip),%zmm4        # 1ed00 <anon.576c71cbdbd6e3deefe97f3cb576ad67.20.llvm.9236128649512152758+0x1dc>
   5924d:	vpbroadcastq -0x3a46f(%rip),%zmm5        # 1ede8 <anon.576c71cbdbd6e3deefe97f3cb576ad67.20.llvm.9236128649512152758+0x2c4>
   59257:	vpbroadcastq -0x3a6e1(%rip),%zmm6        # 1eb80 <anon.576c71cbdbd6e3deefe97f3cb576ad67.20.llvm.9236128649512152758+0x5c>
   59261:	data16 data16 data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
   59270:	vpmullq %zmm2,%zmm0,%zmm7
   59276:	vpsrlq $0x6,%zmm7,%zmm8
   5927d:	vpminuq %zmm4,%zmm8,%zmm9
   59283:	kxnorw %k0,%k0,%k1
   59287:	vpxor  %xmm10,%xmm10,%xmm10
   5928c:	kxnorw %k0,%k0,%k2
   59290:	vpxor  %xmm11,%xmm11,%xmm11
   59295:	vpgatherqq (%rax,%zmm9,8),%zmm10{%k1}
   5929c:	vpgatherqq (%rdi,%zmm8,8),%zmm11{%k2}
   592a3:	vpmovqd %zmm7,%ymm7
   592a9:	vpand  %ymm3,%ymm7,%ymm8
   592ad:	vextracti64x4 $0x1,%zmm10,%ymm7
   592b4:	vextracti128 $0x1,%ymm10,%xmm12
   592ba:	vpextrq $0x1,%xmm11,%r10
   592c0:	vmovq  %xmm10,%r8
   592c5:	vextracti64x4 $0x1,%zmm11,%ymm9
   592cc:	vmovq  %xmm11,%r9
   592d1:	vpmovzxdq %xmm8,%ymm13
   592d6:	vmovd  %xmm13,%ecx
   592da:	vpextrq $0x1,%xmm10,%rbx
   592e0:	shrd   %cl,%r8,%r9
   592e4:	vpextrq $0x1,%xmm9,%r11
   592ea:	vextracti128 $0x1,%ymm11,%xmm10
   592f0:	shrx   %rcx,%r8,%r8
   592f5:	test   $0x40,%cl
   592f8:	vpextrb $0x8,%xmm13,%ecx
   592fe:	vmovq  %xmm7,%r14
   59303:	cmove  %r9,%r8
   59307:	shrd   %cl,%rbx,%r10
   5930b:	vmovq  %xmm12,%r12
   59310:	vmovq  %xmm10,%r13
   59315:	vextracti128 $0x1,%ymm13,%xmm11
   5931b:	shrx   %rcx,%rbx,%r9
   59320:	test   $0x40,%cl
   59323:	vmovd  %xmm11,%ecx
   59327:	cmove  %r10,%r9
   5932b:	shrd   %cl,%r12,%r13
   5932f:	vpextrq $0x1,%xmm12,%rbx
   59335:	vmovq  %xmm9,%r15
   5933a:	vpextrq $0x1,%xmm10,%rbp
   59340:	shrx   %rcx,%r12,%r10
   59345:	test   $0x40,%cl
   59348:	cmove  %r13,%r10
   5934c:	vpextrb $0x8,%xmm11,%ecx
   59352:	shrd   %cl,%rbx,%rbp
   59356:	vextracti128 $0x1,%ymm8,%xmm8
   5935c:	vpmovzxdq %xmm8,%ymm8
   59361:	shrx   %rcx,%rbx,%rbx
   59366:	test   $0x40,%cl
   59369:	vmovd  %xmm8,%ecx
   5936d:	cmove  %rbp,%rbx
   59371:	shrd   %cl,%r14,%r15
   59375:	vpextrq $0x1,%xmm7,%r12
   5937b:	vextracti128 $0x1,%ymm9,%xmm9
   59381:	shrx   %rcx,%r14,%r14
   59386:	test   $0x40,%cl
   59389:	vpextrb $0x8,%xmm8,%ecx
   5938f:	vextracti128 $0x1,%ymm7,%xmm7
   59395:	cmove  %r15,%r14
   59399:	shrd   %cl,%r12,%r11
   5939d:	vmovq  %xmm7,%r13
   593a2:	vmovq  %xmm9,%rbp
   593a7:	vextracti128 $0x1,%ymm8,%xmm8
   593ad:	shrx   %rcx,%r12,%r15
   593b2:	test   $0x40,%cl
   593b5:	vmovd  %xmm8,%ecx
   593b9:	cmove  %r11,%r15
   593bd:	shrd   %cl,%r13,%rbp
   593c1:	vpextrq $0x1,%xmm9,%r11
   593c7:	shrx   %rcx,%r13,%r12
   593cc:	test   $0x40,%cl
   593cf:	vpextrb $0x8,%xmm8,%ecx
   593d5:	vpextrq $0x1,%xmm7,%r13
   593db:	cmove  %rbp,%r12
   593df:	shrd   %cl,%r13,%r11
   593e3:	shrx   %rcx,%r13,%r13
   593e8:	test   $0x40,%cl
   593eb:	cmove  %r11,%r13
   593ef:	vmovq  %r12,%xmm7
   593f4:	vmovq  %r15,%xmm8
   593f9:	vmovq  %r14,%xmm9
   593fe:	vmovq  %r13,%xmm10
   59403:	vmovq  %rbx,%xmm11
   59408:	vmovq  %r10,%xmm12
   5940d:	vmovq  %r9,%xmm13
   59412:	vmovq  %r8,%xmm14
   59417:	vpunpcklqdq %xmm8,%xmm9,%xmm8
   5941c:	vpunpcklqdq %xmm10,%xmm7,%xmm7
   59421:	vinserti128 $0x1,%xmm7,%ymm8,%ymm7
   59427:	vpunpcklqdq %xmm11,%xmm12,%xmm8
   5942c:	vpunpcklqdq %xmm13,%xmm14,%xmm9
   59431:	vinserti128 $0x1,%xmm8,%ymm9,%ymm8
   59437:	vinserti64x4 $0x1,%ymm7,%zmm8,%zmm7
   5943e:	vpandq %zmm5,%zmm7,%zmm7
   59444:	vpaddq %zmm1,%zmm7,%zmm7
   5944a:	vmovdqu64 %zmm7,(%rdx,%rsi,8)
   59451:	add    $0x8,%rsi
   59455:	vpaddq %zmm6,%zmm0,%zmm0
   5945b:	cmp    $0x400,%rsi
   59462:	jne    59270 <_ZN15funnel_patterns12pat_u128_cat17h8e6044c09a682e20E+0x60>
   59468:	pop    %rbx
   59469:	pop    %r12
   5946b:	pop    %r13
   5946d:	pop    %r14
   5946f:	pop    %r15
   59471:	pop    %rbp
   59472:	vzeroupper
   59475:	ret

Disassembly of section .init:

Disassembly of section .fini:

Disassembly of section .plt:
