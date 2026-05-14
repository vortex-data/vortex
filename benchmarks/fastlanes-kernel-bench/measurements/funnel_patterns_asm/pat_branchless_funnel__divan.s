
/home/user/vortex/target/release/deps/funnel_patterns-21c1c00107f42b8a:     file format elf64-x86-64


Disassembly of section .text:

000000000004be60 <_ZN15funnel_patterns31pat_branchless_funnel__u64__w5117ha58623d532d5bfdbE>:
   4be60:	push   %rbp
   4be61:	push   %r15
   4be63:	push   %r14
   4be65:	push   %r13
   4be67:	push   %r12
   4be69:	push   %rbx
   4be6a:	sub    $0x1000,%rsp
   4be71:	movq   $0x0,(%rsp)
   4be79:	sub    $0x1000,%rsp
   4be80:	movq   $0x0,(%rsp)
   4be88:	sub    $0x1000,%rsp
   4be8f:	movq   $0x0,(%rsp)
   4be97:	sub    $0xc18,%rsp
   4be9e:	mov    %rdi,%r12
   4bea1:	lea    0x298(%rsp),%rdi
   4bea9:	mov    $0x1980,%edx
   4beae:	xor    %esi,%esi
   4beb0:	call   *0xf040a(%rip)        # 13c2c0 <memset@GLIBC_2.2.5>
   4beb6:	vmovdqa64 -0x3be00(%rip),%zmm0        # 100c0 <__abi_tag+0xfdc4>
   4bec0:	mov    $0x38,%eax
   4bec5:	vpbroadcastq -0x3b73f(%rip),%zmm1        # 10790 <__abi_tag+0x10494>
   4becf:	vpbroadcastq -0x3b869(%rip),%zmm2        # 10670 <__abi_tag+0x10374>
   4bed9:	vpbroadcastq -0x3b74b(%rip),%zmm3        # 10798 <__abi_tag+0x1049c>
   4bee3:	vpbroadcastq -0x3b6cd(%rip),%zmm4        # 10820 <__abi_tag+0x10524>
   4beed:	vpbroadcastq -0x3b6a7(%rip),%zmm5        # 10850 <__abi_tag+0x10554>
   4bef7:	vpbroadcastq -0x3b871(%rip),%zmm6        # 10690 <__abi_tag+0x10394>
   4bf01:	vpbroadcastq -0x3b683(%rip),%zmm7        # 10888 <__abi_tag+0x1058c>
   4bf0b:	vpbroadcastq -0x3b6bd(%rip),%zmm8        # 10858 <__abi_tag+0x1055c>
   4bf15:	vpbroadcastq -0x3b69f(%rip),%zmm9        # 10880 <__abi_tag+0x10584>
   4bf1f:	nop
   4bf20:	vpmullq %zmm1,%zmm0,%zmm10
   4bf26:	vpaddq %zmm2,%zmm10,%zmm11
   4bf2c:	vpaddq %zmm3,%zmm10,%zmm12
   4bf32:	vmovdqu64 %zmm10,0xd8(%rsp,%rax,8)
   4bf3d:	vmovdqu64 %zmm11,0x118(%rsp,%rax,8)
   4bf48:	vpaddq %zmm4,%zmm10,%zmm11
   4bf4e:	vmovdqu64 %zmm12,0x158(%rsp,%rax,8)
   4bf59:	vmovdqu64 %zmm11,0x198(%rsp,%rax,8)
   4bf64:	cmp    $0x338,%rax
   4bf6a:	je     4bfbf <_ZN15funnel_patterns31pat_branchless_funnel__u64__w5117ha58623d532d5bfdbE+0x15f>
   4bf6c:	vpaddq %zmm5,%zmm10,%zmm11
   4bf72:	vpaddq %zmm6,%zmm10,%zmm12
   4bf78:	vpaddq %zmm7,%zmm10,%zmm13
   4bf7e:	vpaddq %zmm8,%zmm10,%zmm10
   4bf84:	vmovdqu64 %zmm11,0x1d8(%rsp,%rax,8)
   4bf8f:	vmovdqu64 %zmm12,0x218(%rsp,%rax,8)
   4bf9a:	vmovdqu64 %zmm13,0x258(%rsp,%rax,8)
   4bfa5:	vmovdqu64 %zmm10,0x298(%rsp,%rax,8)
   4bfb0:	vpaddq %zmm9,%zmm0,%zmm0
   4bfb6:	add    $0x40,%rax
   4bfba:	jmp    4bf20 <_ZN15funnel_patterns31pat_branchless_funnel__u64__w5117ha58623d532d5bfdbE+0xc0>
   4bfbf:	vmovaps -0x3bec9(%rip),%zmm0        # 10100 <__abi_tag+0xfe04>
   4bfc9:	vmovups %zmm0,0x1b98(%rsp)
   4bfd4:	vmovdqa64 -0x3be9e(%rip),%zmm0        # 10140 <__abi_tag+0xfe44>
   4bfde:	vmovdqu64 %zmm0,0x1bd8(%rsp)
   4bfe9:	lea    0x2298(%rsp),%rbx
   4bff1:	lea    0x298(%rsp),%r14
   4bff9:	mov    $0x1980,%edx
   4bffe:	mov    %rbx,%rdi
   4c001:	mov    %r14,%rsi
   4c004:	vzeroupper
   4c007:	call   *0xf02bb(%rip)        # 13c2c8 <memcpy@GLIBC_2.14>
   4c00d:	movq   $0x3b9aca07,0xf0(%rsp)
   4c019:	xor    %ebp,%ebp
   4c01b:	mov    $0x2000,%edx
   4c020:	mov    %r14,%rdi
   4c023:	xor    %esi,%esi
   4c025:	call   *0xf0295(%rip)        # 13c2c0 <memset@GLIBC_2.2.5>
   4c02b:	mov    %rbx,0x208(%rsp)
   4c033:	lea    0xf0(%rsp),%rax
   4c03b:	mov    %rax,0x210(%rsp)
   4c043:	mov    %r14,0x218(%rsp)
   4c04b:	lea    0x208(%rsp),%rax
   4c053:	mov    %rax,0xf8(%rsp)
   4c05b:	lea    0xf8(%rsp),%rax
   4c063:	mov    %rax,0x100(%rsp)
   4c06b:	movq   $0x1,0x100(%r12)
   4c077:	movb   $0x1,0x108(%r12)
   4c080:	mov    0xf0(%r12),%rdx
   4c088:	mov    0xf8(%r12),%rax
   4c090:	movzbl 0x8(%rdx),%ecx
   4c094:	mov    %cl,0x3(%rsp)
   4c098:	cmp    $0x1,%cl
   4c09b:	jne    4c0a3 <_ZN15funnel_patterns31pat_branchless_funnel__u64__w5117ha58623d532d5bfdbE+0x243>
   4c09d:	xor    %esi,%esi
   4c09f:	xor    %ecx,%ecx
   4c0a1:	jmp    4c0cd <_ZN15funnel_patterns31pat_branchless_funnel__u64__w5117ha58623d532d5bfdbE+0x26d>
   4c0a3:	cmpb   $0x0,0x60(%rax)
   4c0a7:	je     4c0bc <_ZN15funnel_patterns31pat_branchless_funnel__u64__w5117ha58623d532d5bfdbE+0x25c>
   4c0a9:	mov    0x64(%rax),%ecx
   4c0ac:	mov    %ecx,0x8(%rsp)
   4c0b0:	mov    $0x2,%ebp
   4c0b5:	mov    $0x1,%sil
   4c0b8:	xor    %ecx,%ecx
   4c0ba:	jmp    4c0cd <_ZN15funnel_patterns31pat_branchless_funnel__u64__w5117ha58623d532d5bfdbE+0x26d>
   4c0bc:	mov    $0x1,%cl
   4c0be:	movl   $0x1,0x8(%rsp)
   4c0c6:	xor    %esi,%esi
   4c0c8:	mov    $0x1,%ebp
   4c0cd:	mov    (%rdx),%r14
   4c0d0:	test   %r14,%r14
   4c0d3:	lea    0x27(%rsp),%rdx
   4c0d8:	mov    %rdx,0x220(%rsp)
   4c0e0:	setne  0x238(%rsp)
   4c0e8:	lea    0x100(%rsp),%rdi
   4c0f0:	mov    %rdi,0x228(%rsp)
   4c0f8:	mov    %rdx,0x230(%rsp)
   4c100:	mov    0x70(%rax),%edi
   4c103:	movq   $0x0,0x88(%rsp)
   4c10f:	mov    $0x0,%edx
   4c114:	mov    %rdx,0x80(%rsp)
   4c11c:	cmp    $0x3b9aca00,%edi
   4c122:	je     4c15d <_ZN15funnel_patterns31pat_branchless_funnel__u64__w5117ha58623d532d5bfdbE+0x2fd>
   4c124:	mov    $0x3b9aca00,%edx
   4c129:	mulx   0x68(%rax),%r8,%r9
   4c12f:	mov    %edi,%edx
   4c131:	add    %r8,%rdx
   4c134:	adc    $0x0,%r9
   4c138:	imul   $0x3e8,%r9,%rdi
   4c13f:	mov    $0x3e8,%r8d
   4c145:	mulx   %r8,%rdx,%r8
   4c14a:	mov    %rdx,0x88(%rsp)
   4c152:	add    %rdi,%r8
   4c155:	mov    %r8,0x80(%rsp)
   4c15d:	movq   $0xffffffffffffffff,0x48(%rsp)
   4c166:	mov    0x80(%rax),%r8d
   4c16d:	movq   $0xffffffffffffffff,0x38(%rsp)
   4c176:	cmp    $0x3b9aca00,%r8d
   4c17d:	je     4c1be <_ZN15funnel_patterns31pat_branchless_funnel__u64__w5117ha58623d532d5bfdbE+0x35e>
   4c17f:	mov    $0x3b9aca00,%edx
   4c184:	mulx   0x78(%rax),%r9,%rdi
   4c18a:	mov    %r8d,%edx
   4c18d:	add    %r9,%rdx
   4c190:	adc    $0x0,%rdi
   4c194:	mov    $0x3e8,%r8d
   4c19a:	mulx   %r8,%r8,%r9
   4c19f:	mov    %r9,0x38(%rsp)
   4c1a4:	mov    %r8,0x48(%rsp)
   4c1a9:	or     %rdi,%rdx
   4c1ac:	je     4cd56 <_ZN15funnel_patterns31pat_branchless_funnel__u64__w5117ha58623d532d5bfdbE+0xef6>
   4c1b2:	imul   $0x3e8,%rdi,%rdx
   4c1b9:	add    %rdx,0x38(%rsp)
   4c1be:	mov    0x58(%rax),%edx
   4c1c1:	cmp    $0x1,%edx
   4c1c4:	jne    4c1d0 <_ZN15funnel_patterns31pat_branchless_funnel__u64__w5117ha58623d532d5bfdbE+0x370>
   4c1c6:	cmpl   $0x0,0x5c(%rax)
   4c1ca:	je     4cd56 <_ZN15funnel_patterns31pat_branchless_funnel__u64__w5117ha58623d532d5bfdbE+0xef6>
   4c1d0:	cmpl   $0x1,0x60(%rax)
   4c1d4:	jne    4c1e0 <_ZN15funnel_patterns31pat_branchless_funnel__u64__w5117ha58623d532d5bfdbE+0x380>
   4c1d6:	cmpl   $0x0,0x64(%rax)
   4c1da:	je     4cd56 <_ZN15funnel_patterns31pat_branchless_funnel__u64__w5117ha58623d532d5bfdbE+0xef6>
   4c1e0:	mov    %r14,0x60(%rsp)
   4c1e5:	test   %dl,%sil
   4c1e8:	je     4c1fb <_ZN15funnel_patterns31pat_branchless_funnel__u64__w5117ha58623d532d5bfdbE+0x39b>
   4c1ea:	mov    0x5c(%rax),%edx
   4c1ed:	mov    %edx,0x18(%rsp)
   4c1f1:	movl   $0x1,0xc(%rsp)
   4c1f9:	jmp    4c20b <_ZN15funnel_patterns31pat_branchless_funnel__u64__w5117ha58623d532d5bfdbE+0x3ab>
   4c1fb:	movzbl %sil,%edx
   4c1ff:	mov    %edx,0xc(%rsp)
   4c203:	movl   $0x64,0x18(%rsp)
   4c20b:	movq   $0x0,0x58(%rsp)
   4c214:	mov    $0x0,%edx
   4c219:	mov    %rdx,0x50(%rsp)
   4c21e:	test   %cl,%cl
   4c220:	je     4c23d <_ZN15funnel_patterns31pat_branchless_funnel__u64__w5117ha58623d532d5bfdbE+0x3dd>
   4c222:	mov    %r14,%rdi
   4c225:	call   *0xf00a5(%rip)        # 13c2d0 <_DYNAMIC+0x240>
   4c22b:	mov    %rax,0x58(%rsp)
   4c230:	mov    %rdx,0x50(%rsp)
   4c235:	mov    0xf8(%r12),%rax
   4c23d:	cmpb   $0x1,0x3(%rsp)
   4c242:	je     4c263 <_ZN15funnel_patterns31pat_branchless_funnel__u64__w5117ha58623d532d5bfdbE+0x403>
   4c244:	mov    $0x1,%edx
   4c249:	cmpb   $0x0,0x58(%rax)
   4c24d:	je     4c252 <_ZN15funnel_patterns31pat_branchless_funnel__u64__w5117ha58623d532d5bfdbE+0x3f2>
   4c24f:	mov    0x5c(%rax),%edx
   4c252:	mov    (%r12),%rcx
   4c256:	mov    0x10(%r12),%rsi
   4c25b:	sub    %rsi,%rcx
   4c25e:	cmp    %rcx,%rdx
   4c261:	ja     4c2a8 <_ZN15funnel_patterns31pat_branchless_funnel__u64__w5117ha58623d532d5bfdbE+0x448>
   4c263:	testb  $0x1,0x88(%rax)
   4c26a:	jne    4c2c1 <_ZN15funnel_patterns31pat_branchless_funnel__u64__w5117ha58623d532d5bfdbE+0x461>
   4c26c:	lock orl $0x0,-0x40(%rsp)
   4c272:	test   %r14,%r14
   4c275:	je     4c293 <_ZN15funnel_patterns31pat_branchless_funnel__u64__w5117ha58623d532d5bfdbE+0x433>
   4c277:	lfence
   4c27a:	rdtsc
   4c27c:	shl    $0x20,%rdx
   4c280:	or     %rax,%rdx
   4c283:	mov    %rdx,0x70(%rsp)
   4c288:	lfence
   4c28b:	mov    $0x3b9aca00,%r13d
   4c291:	jmp    4c2a1 <_ZN15funnel_patterns31pat_branchless_funnel__u64__w5117ha58623d532d5bfdbE+0x441>
   4c293:	call   *0xf003f(%rip)        # 13c2d8 <_DYNAMIC+0x248>
   4c299:	mov    %rax,0x70(%rsp)
   4c29e:	mov    %edx,%r13d
   4c2a1:	mov    0x60(%rsp),%r14
   4c2a6:	jmp    4c2c7 <_ZN15funnel_patterns31pat_branchless_funnel__u64__w5117ha58623d532d5bfdbE+0x467>
   4c2a8:	mov    %r12,%rdi
   4c2ab:	call   51800 <_ZN5alloc7raw_vec20RawVecInner$LT$A$GT$7reserve21do_reserve_and_handle17h46771c9d08372974E>
   4c2b0:	mov    0xf8(%r12),%rax
   4c2b8:	testb  $0x1,0x88(%rax)
   4c2bf:	je     4c26c <_ZN15funnel_patterns31pat_branchless_funnel__u64__w5117ha58623d532d5bfdbE+0x40c>
   4c2c1:	mov    $0x3b9aca01,%r13d
   4c2c7:	mov    %r14,%rdi
   4c2ca:	call   *0xf0010(%rip)        # 13c2e0 <_DYNAMIC+0x250>
   4c2d0:	mov    %rax,0xa8(%rsp)
   4c2d8:	mov    0xf0009(%rip),%r14        # 13c2e8 <_DYNAMIC+0x258>
   4c2df:	mov    0x8(%r14),%eax
   4c2e3:	test   %eax,%eax
   4c2e5:	jne    4cd83 <_ZN15funnel_patterns31pat_branchless_funnel__u64__w5117ha58623d532d5bfdbE+0xf23>
   4c2eb:	mov    (%r14),%rdi
   4c2ee:	call   *0xefffc(%rip)        # 13c2f0 <_DYNAMIC+0x260>
   4c2f4:	mov    0x48(%rsp),%rax
   4c2f9:	or     0x38(%rsp),%rax
   4c2fe:	je     4cd12 <_ZN15funnel_patterns31pat_branchless_funnel__u64__w5117ha58623d532d5bfdbE+0xeb2>
   4c304:	lea    0x50(%r12),%rax
   4c309:	mov    %rax,0x28(%rsp)
   4c30e:	lea    0x18(%r12),%rax
   4c313:	mov    %rax,0x78(%rsp)
   4c318:	mov    $0x10,%r15d
   4c31e:	mov    $0x1,%al
   4c320:	xor    %edx,%edx
   4c322:	xor    %esi,%esi
   4c324:	xor    %r14d,%r14d
   4c327:	mov    %r12,0x30(%rsp)
   4c32c:	mov    %r13d,0x6c(%rsp)
   4c331:	jmp    4c3ab <_ZN15funnel_patterns31pat_branchless_funnel__u64__w5117ha58623d532d5bfdbE+0x54b>
   4c333:	data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
   4c340:	mov    0x90(%rsp),%rdi
   4c348:	cmp    $0x3e9,%rdi
   4c34f:	mov    $0x3e8,%eax
   4c354:	cmovae %rdi,%rax
   4c358:	mov    0x98(%rsp),%rsi
   4c360:	test   %rsi,%rsi
   4c363:	mov    $0x3e8,%ecx
   4c368:	cmove  %rcx,%rdi
   4c36c:	cmove  %rax,%rdi
   4c370:	mov    0xb0(%rsp),%rdx
   4c378:	add    %rdi,%rdx
   4c37b:	adc    %rsi,%r14
   4c37e:	mov    $0xffffffffffffffff,%rax
   4c385:	cmovb  %rax,%r14
   4c389:	cmovb  %rax,%rdx
   4c38d:	mov    %r14,%rsi
   4c390:	mov    $0x1,%r14d
   4c396:	xor    %eax,%eax
   4c398:	cmp    0x48(%rsp),%rdx
   4c39d:	mov    %rsi,%rcx
   4c3a0:	sbb    0x38(%rsp),%rcx
   4c3a5:	jae    4cd1b <_ZN15funnel_patterns31pat_branchless_funnel__u64__w5117ha58623d532d5bfdbE+0xebb>
   4c3ab:	cmp    0x88(%rsp),%rdx
   4c3b3:	mov    %rsi,0xe8(%rsp)
   4c3bb:	mov    %rsi,%rcx
   4c3be:	sbb    0x80(%rsp),%rcx
   4c3c6:	jb     4c3da <_ZN15funnel_patterns31pat_branchless_funnel__u64__w5117ha58623d532d5bfdbE+0x57a>
   4c3c8:	testb  $0x1,0xc(%rsp)
   4c3cd:	je     4c3da <_ZN15funnel_patterns31pat_branchless_funnel__u64__w5117ha58623d532d5bfdbE+0x57a>
   4c3cf:	cmpl   $0x0,0x18(%rsp)
   4c3d4:	je     4cd1b <_ZN15funnel_patterns31pat_branchless_funnel__u64__w5117ha58623d532d5bfdbE+0xebb>
   4c3da:	mov    %rdx,0xb0(%rsp)
   4c3e2:	test   %ebp,%ebp
   4c3e4:	mov    0x8(%rsp),%ecx
   4c3e8:	mov    $0x1,%edx
   4c3ed:	cmove  %edx,%ecx
   4c3f0:	mov    %ecx,0x4(%rsp)
   4c3f4:	mov    %ecx,0x48(%r12)
   4c3f9:	movq   $0x0,0x268(%rsp)
   4c405:	mov    0x28(%rsp),%rcx
   4c40a:	mov    %rcx,0x240(%rsp)
   4c412:	lea    0x220(%rsp),%rcx
   4c41a:	mov    %rcx,0x248(%rsp)
   4c422:	lea    0x4(%rsp),%rcx
   4c427:	mov    %rcx,0x250(%rsp)
   4c42f:	lea    0x268(%rsp),%rcx
   4c437:	mov    %rcx,0x258(%rsp)
   4c43f:	lea    0x60(%rsp),%rcx
   4c444:	mov    %rcx,0x260(%rsp)
   4c44c:	test   $0x1,%al
   4c44e:	jne    4c5c0 <_ZN15funnel_patterns31pat_branchless_funnel__u64__w5117ha58623d532d5bfdbE+0x760>
   4c454:	imul   $0xd0,%r14,%rax
   4c45b:	add    %r15,%rax
   4c45e:	mov    %rax,%rdx
   4c461:	sub    %r15,%rdx
   4c464:	add    $0xffffffffffffff30,%rdx
   4c46b:	movabs $0x4ec4ec4ec4ec4ec5,%rcx
   4c475:	mulx   %rcx,%rcx,%rcx
   4c47a:	mov    %r15,%rsi
   4c47d:	cmp    $0xc30,%rdx
   4c484:	jb     4c610 <_ZN15funnel_patterns31pat_branchless_funnel__u64__w5117ha58623d532d5bfdbE+0x7b0>
   4c48a:	shr    $0x6,%rcx
   4c48e:	inc    %rcx
   4c491:	cmp    $0x3330,%rdx
   4c498:	jae    4c4b0 <_ZN15funnel_patterns31pat_branchless_funnel__u64__w5117ha58623d532d5bfdbE+0x650>
   4c49a:	xor    %edx,%edx
   4c49c:	mov    %r15,%rdi
   4c49f:	jmp    4c55f <_ZN15funnel_patterns31pat_branchless_funnel__u64__w5117ha58623d532d5bfdbE+0x6ff>
   4c4a4:	data16 data16 cs nopw 0x0(%rax,%rax,1)
   4c4b0:	mov    %rcx,%rdx
   4c4b3:	movabs $0x3ffffffffffffc0,%rsi
   4c4bd:	and    %rsi,%rdx
   4c4c0:	imul   $0xd0,%rdx,%rsi
   4c4c7:	lea    (%r15,%rsi,1),%rdi
   4c4cb:	mov    %rdx,%r8
   4c4ce:	mov    %r15,%r9
   4c4d1:	vpbroadcastd -0x3bc07(%rip),%zmm0        # 108d4 <__abi_tag+0x105d8>
   4c4db:	vmovdqa64 -0x3c365(%rip),%zmm1        # 10180 <__abi_tag+0xfe84>
   4c4e5:	vmovdqa64 -0x3c32f(%rip),%zmm2        # 101c0 <__abi_tag+0xfec4>
   4c4ef:	vmovdqa64 -0x3c2f9(%rip),%zmm3        # 10200 <__abi_tag+0xff04>
   4c4f9:	vmovdqa64 -0x3c2c3(%rip),%zmm4        # 10240 <__abi_tag+0xff44>
   4c503:	data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
   4c510:	kxnorw %k0,%k0,%k1
   4c514:	vpscatterdd %zmm0,0x8(%r9,%zmm1,1){%k1}
   4c51c:	kxnorw %k0,%k0,%k1
   4c520:	vpscatterdd %zmm0,0x8(%r9,%zmm2,1){%k1}
   4c528:	kxnorw %k0,%k0,%k1
   4c52c:	vpscatterdd %zmm0,0x8(%r9,%zmm3,1){%k1}
   4c534:	kxnorw %k0,%k0,%k1
   4c538:	vpscatterdd %zmm0,0x8(%r9,%zmm4,1){%k1}
   4c540:	add    $0x3400,%r9
   4c547:	add    $0xffffffffffffffc0,%r8
   4c54b:	jne    4c510 <_ZN15funnel_patterns31pat_branchless_funnel__u64__w5117ha58623d532d5bfdbE+0x6b0>
   4c54d:	cmp    %rdx,%rcx
   4c550:	je     4c623 <_ZN15funnel_patterns31pat_branchless_funnel__u64__w5117ha58623d532d5bfdbE+0x7c3>
   4c556:	test   $0x30,%cl
   4c559:	je     4c60d <_ZN15funnel_patterns31pat_branchless_funnel__u64__w5117ha58623d532d5bfdbE+0x7ad>
   4c55f:	movabs $0x3ffffffffffffc0,%rsi
   4c569:	lea    0x30(%rsi),%r8
   4c56d:	and    %rcx,%r8
   4c570:	imul   $0xd0,%r8,%rsi
   4c577:	add    %r15,%rsi
   4c57a:	sub    %r8,%rdx
   4c57d:	vpbroadcastd -0x3bcb3(%rip),%zmm0        # 108d4 <__abi_tag+0x105d8>
   4c587:	vmovdqa64 -0x3c411(%rip),%zmm1        # 10180 <__abi_tag+0xfe84>
   4c591:	data16 data16 data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
   4c5a0:	kxnorw %k0,%k0,%k1
   4c5a4:	vpscatterdd %zmm0,0x8(%rdi,%zmm1,1){%k1}
   4c5ac:	add    $0xd00,%rdi
   4c5b3:	add    $0x10,%rdx
   4c5b7:	jne    4c5a0 <_ZN15funnel_patterns31pat_branchless_funnel__u64__w5117ha58623d532d5bfdbE+0x740>
   4c5b9:	cmp    %r8,%rcx
   4c5bc:	jne    4c610 <_ZN15funnel_patterns31pat_branchless_funnel__u64__w5117ha58623d532d5bfdbE+0x7b0>
   4c5be:	jmp    4c623 <_ZN15funnel_patterns31pat_branchless_funnel__u64__w5117ha58623d532d5bfdbE+0x7c3>
   4c5c0:	movq   $0x0,0x128(%rsp)
   4c5cc:	mov    $0x10,%esi
   4c5d1:	mov    $0xd0,%edx
   4c5d6:	lea    0x1a0(%rsp),%rdi
   4c5de:	lea    0x120(%rsp),%rcx
   4c5e6:	call   516c0 <_ZN5alloc7raw_vec11finish_grow17hedc133b40cb748a9E>
   4c5eb:	cmpb   $0x0,0x1a0(%rsp)
   4c5f3:	jne    4ce42 <_ZN15funnel_patterns31pat_branchless_funnel__u64__w5117ha58623d532d5bfdbE+0xfe2>
   4c5f9:	mov    0x1a8(%rsp),%r15
   4c601:	lea    0xd0(%r15),%rax
   4c608:	jmp    4c45e <_ZN15funnel_patterns31pat_branchless_funnel__u64__w5117ha58623d532d5bfdbE+0x5fe>
   4c60d:	add    %r15,%rsi
   4c610:	movl   $0x3b9aca01,0x8(%rsi)
   4c617:	add    $0xd0,%rsi
   4c61e:	cmp    %rax,%rsi
   4c621:	jne    4c610 <_ZN15funnel_patterns31pat_branchless_funnel__u64__w5117ha58623d532d5bfdbE+0x7b0>
   4c623:	mov    %r12,%rbx
   4c626:	mov    %r15,0x40(%rsp)
   4c62b:	vzeroupper
   4c62e:	call   *0xefcc4(%rip)        # 13c2f8 <_DYNAMIC+0x268>
   4c634:	mov    %rax,0x120(%rsp)
   4c63c:	movq   $0x0,0x128(%rsp)
   4c648:	lea    0x2a91(%rip),%rax        # 4f0e0 <_ZN30codspeed_divan_compat_walltime11thread_pool19TaskShared$LT$F$GT$3new4call17hd9e3f8b1f7640e9cE>
   4c64f:	mov    %rax,0x130(%rsp)
   4c657:	lea    0x240(%rsp),%rax
   4c65f:	mov    %rax,0x138(%rsp)
   4c667:	mov    %r15,0x140(%rsp)
   4c66f:	mov    0xefc8a(%rip),%rdi        # 13c300 <_DYNAMIC+0x270>
   4c676:	xor    %esi,%esi
   4c678:	lea    0x120(%rsp),%rdx
   4c680:	call   *0xefc82(%rip)        # 13c308 <_DYNAMIC+0x278>
   4c686:	mov    0x120(%rsp),%rax
   4c68e:	lock decq (%rax)
   4c692:	jne    4c6a2 <_ZN15funnel_patterns31pat_branchless_funnel__u64__w5117ha58623d532d5bfdbE+0x842>
   4c694:	lea    0x120(%rsp),%rdi
   4c69c:	call   *0xefc6e(%rip)        # 13c310 <_DYNAMIC+0x280>
   4c6a2:	cmpl   $0x3b9aca01,0x8(%r15)
   4c6aa:	je     4cd9f <_ZN15funnel_patterns31pat_branchless_funnel__u64__w5117ha58623d532d5bfdbE+0xf3f>
   4c6b0:	cmpb   $0x1,0x3(%rsp)
   4c6b5:	je     4cd68 <_ZN15funnel_patterns31pat_branchless_funnel__u64__w5117ha58623d532d5bfdbE+0xf08>
   4c6bb:	vmovups 0x10(%r15),%xmm0
   4c6c1:	vmovaps %xmm0,0x1a0(%rsp)
   4c6ca:	vmovdqu (%r15),%xmm0
   4c6cf:	vmovdqa %xmm0,0x120(%rsp)
   4c6d8:	mov    0xc0(%r15),%rdx
   4c6df:	lea    0x1a0(%rsp),%rdi
   4c6e7:	lea    0x120(%rsp),%rsi
   4c6ef:	call   *0xefc23(%rip)        # 13c318 <_DYNAMIC+0x288>
   4c6f5:	lea    0x10(%r15),%rax
   4c6f9:	mov    %rax,%r12
   4c6fc:	vmovups (%rax),%xmm0
   4c700:	vmovaps %xmm0,0x1a0(%rsp)
   4c709:	vmovdqu (%r15),%xmm0
   4c70e:	vmovdqa %xmm0,0x120(%rsp)
   4c717:	mov    0xc0(%r15),%rdx
   4c71e:	lea    0x1a0(%rsp),%rdi
   4c726:	lea    0x120(%rsp),%rsi
   4c72e:	call   *0xefbe4(%rip)        # 13c318 <_DYNAMIC+0x288>
   4c734:	mov    %rax,%rsi
   4c737:	cmp    $0x1,%ebp
   4c73a:	mov    %rdx,0x98(%rsp)
   4c742:	mov    %rax,0x90(%rsp)
   4c74a:	jne    4c7c0 <_ZN15funnel_patterns31pat_branchless_funnel__u64__w5117ha58623d532d5bfdbE+0x960>
   4c74c:	mov    %rbx,%rcx
   4c74f:	movq   $0x0,0x10(%rbx)
   4c757:	cmpq   $0x0,0x30(%rbx)
   4c75c:	mov    0x58(%rsp),%r13
   4c761:	mov    0x50(%rsp),%rbp
   4c766:	je     4c7e6 <_ZN15funnel_patterns31pat_branchless_funnel__u64__w5117ha58623d532d5bfdbE+0x986>
   4c768:	mov    0x20(%rcx),%r14
   4c76c:	test   %r14,%r14
   4c76f:	je     4c7d5 <_ZN15funnel_patterns31pat_branchless_funnel__u64__w5117ha58623d532d5bfdbE+0x975>
   4c771:	mov    0x78(%rsp),%rax
   4c776:	mov    (%rax),%rdi
   4c779:	lea    0x11(%r14),%rdx
   4c77d:	mov    $0xff,%esi
   4c782:	call   *0xefb38(%rip)        # 13c2c0 <memset@GLIBC_2.2.5>
   4c788:	mov    0x90(%rsp),%rsi
   4c790:	mov    0x98(%rsp),%rdx
   4c798:	lea    0x1(%r14),%rax
   4c79c:	mov    %rax,%rcx
   4c79f:	shr    $0x3,%rcx
   4c7a3:	and    $0xfffffffffffffff8,%rax
   4c7a7:	sub    %rcx,%rax
   4c7aa:	cmp    $0x8,%r14
   4c7ae:	cmovb  %r14,%rax
   4c7b2:	jmp    4c7d7 <_ZN15funnel_patterns31pat_branchless_funnel__u64__w5117ha58623d532d5bfdbE+0x977>
   4c7b4:	data16 data16 cs nopw 0x0(%rax,%rax,1)
   4c7c0:	mov    %ebp,0x14(%rsp)
   4c7c4:	mov    0x4(%rsp),%eax
   4c7c8:	mov    %rax,0xa0(%rsp)
   4c7d0:	jmp    4c877 <_ZN15funnel_patterns31pat_branchless_funnel__u64__w5117ha58623d532d5bfdbE+0xa17>
   4c7d5:	xor    %eax,%eax
   4c7d7:	mov    %rbx,%rcx
   4c7da:	movq   $0x0,0x30(%rbx)
   4c7e2:	mov    %rax,0x28(%rbx)
   4c7e6:	cmpq   $0x0,0x68(%rcx)
   4c7eb:	je     4c7f5 <_ZN15funnel_patterns31pat_branchless_funnel__u64__w5117ha58623d532d5bfdbE+0x995>
   4c7ed:	movq   $0x0,0x60(%rcx)
   4c7f5:	cmpq   $0x0,0x90(%rcx)
   4c7fd:	je     4c80a <_ZN15funnel_patterns31pat_branchless_funnel__u64__w5117ha58623d532d5bfdbE+0x9aa>
   4c7ff:	movq   $0x0,0x88(%rcx)
   4c80a:	cmpq   $0x0,0xb8(%rcx)
   4c812:	je     4c81f <_ZN15funnel_patterns31pat_branchless_funnel__u64__w5117ha58623d532d5bfdbE+0x9bf>
   4c814:	movq   $0x0,0xb0(%rcx)
   4c81f:	cmpq   $0x0,0xe0(%rcx)
   4c827:	je     4c834 <_ZN15funnel_patterns31pat_branchless_funnel__u64__w5117ha58623d532d5bfdbE+0x9d4>
   4c829:	movq   $0x0,0xd8(%rcx)
   4c834:	mov    %r13,%rax
   4c837:	or     %rbp,%rax
   4c83a:	je     4ce33 <_ZN15funnel_patterns31pat_branchless_funnel__u64__w5117ha58623d532d5bfdbE+0xfd3>
   4c840:	mov    %rsi,%rdi
   4c843:	mov    %rdx,%rsi
   4c846:	mov    %r13,%rdx
   4c849:	mov    %rbp,%rcx
   4c84c:	call   *0xeface(%rip)        # 13c320 <_DYNAMIC+0x290>
   4c852:	cmp    $0x65,%rax
   4c856:	sbb    $0x0,%rdx
   4c85a:	mov    0x4(%rsp),%ecx
   4c85e:	mov    %rcx,0xa0(%rsp)
   4c866:	jae    4c890 <_ZN15funnel_patterns31pat_branchless_funnel__u64__w5117ha58623d532d5bfdbE+0xa30>
   4c868:	lea    (%rcx,%rcx,1),%eax
   4c86b:	mov    %eax,0x8(%rsp)
   4c86f:	movl   $0x1,0x14(%rsp)
   4c877:	mov    0x18(%rsp),%eax
   4c87b:	mov    %eax,0x10(%rsp)
   4c87f:	mov    %r12,%rdx
   4c882:	jmp    4c8cd <_ZN15funnel_patterns31pat_branchless_funnel__u64__w5117ha58623d532d5bfdbE+0xa6d>
   4c884:	data16 data16 cs nopw 0x0(%rax,%rax,1)
   4c890:	mov    0xf8(%rbx),%rax
   4c897:	movl   $0x1,0xc(%rsp)
   4c89f:	cmpb   $0x0,0x58(%rax)
   4c8a3:	mov    %r12,%rdx
   4c8a6:	je     4c8b9 <_ZN15funnel_patterns31pat_branchless_funnel__u64__w5117ha58623d532d5bfdbE+0xa59>
   4c8a8:	movl   $0x2,0x14(%rsp)
   4c8b0:	mov    0x5c(%rax),%eax
   4c8b3:	mov    %eax,0x10(%rsp)
   4c8b7:	jmp    4c8c9 <_ZN15funnel_patterns31pat_branchless_funnel__u64__w5117ha58623d532d5bfdbE+0xa69>
   4c8b9:	movl   $0x64,0x10(%rsp)
   4c8c1:	movl   $0x2,0x14(%rsp)
   4c8c9:	mov    %ecx,0x8(%rsp)
   4c8cd:	mov    0xa8(%rsp),%rax
   4c8d5:	mov    (%rax),%rbp
   4c8d8:	mov    0x8(%rax),%r13
   4c8dc:	mov    0x18(%rax),%r14
   4c8e0:	mov    0x10(%rax),%rcx
   4c8e4:	mov    %rcx,0xc0(%rsp)
   4c8ec:	mov    %r15,%rcx
   4c8ef:	mov    0x40(%r15),%r15
   4c8f3:	mov    0x28(%rax),%rsi
   4c8f7:	mov    %rsi,0xc8(%rsp)
   4c8ff:	mov    0x20(%rax),%rsi
   4c903:	mov    %rsi,0xd0(%rsp)
   4c90b:	mov    0x50(%rcx),%rbx
   4c90f:	mov    0x38(%rax),%rsi
   4c913:	mov    %rsi,0xe0(%rsp)
   4c91b:	mov    0x30(%rax),%rax
   4c91f:	mov    %rax,0x18(%rsp)
   4c924:	mov    0x20(%rcx),%rax
   4c928:	mov    %rax,0xd8(%rsp)
   4c930:	mov    0x30(%rcx),%r12
   4c934:	vmovups (%rdx),%xmm0
   4c938:	vmovaps %xmm0,0x1a0(%rsp)
   4c941:	mov    0x30(%rsp),%rax
   4c946:	mov    0x10(%rax),%rax
   4c94a:	mov    %rax,0xb8(%rsp)
   4c952:	vmovdqu (%rcx),%xmm0
   4c956:	vmovdqa %xmm0,0x120(%rsp)
   4c95f:	mov    0xc0(%rcx),%rdx
   4c966:	lea    0x1a0(%rsp),%rdi
   4c96e:	lea    0x120(%rsp),%rsi
   4c976:	call   *0xef99c(%rip)        # 13c318 <_DYNAMIC+0x288>
   4c97c:	mov    %rax,%rsi
   4c97f:	mov    %rdx,%rcx
   4c982:	mov    0xa0(%rsp),%edi
   4c989:	mov    %r13,%rax
   4c98c:	mul    %rdi
   4c98f:	mov    %rbp,%rdx
   4c992:	mulx   %rdi,%r10,%r9
   4c997:	seto   %dl
   4c99a:	add    %rax,%r9
   4c99d:	setb   %r13b
   4c9a1:	or     %dl,%r13b
   4c9a4:	mov    %r14,%rax
   4c9a7:	mul    %r15
   4c9aa:	seto   %r11b
   4c9ae:	mov    0xc0(%rsp),%rdx
   4c9b6:	mulx   %r15,%r8,%rdi
   4c9bb:	add    %rax,%rdi
   4c9be:	setb   %r14b
   4c9c2:	or     %r11b,%r14b
   4c9c5:	mov    0xc8(%rsp),%rax
   4c9cd:	mul    %rbx
   4c9d0:	mov    0xd0(%rsp),%rdx
   4c9d8:	mulx   %rbx,%rbx,%r11
   4c9dd:	seto   %dl
   4c9e0:	add    %rax,%r11
   4c9e3:	setb   %bpl
   4c9e7:	or     %dl,%bpl
   4c9ea:	or     %r14b,%bpl
   4c9ed:	or     %r13b,%bpl
   4c9f0:	xor    %r14d,%r14d
   4c9f3:	add    0xd8(%rsp),%r12
   4c9fb:	setb   %r14b
   4c9ff:	mov    0xe0(%rsp),%rax
   4ca07:	test   %rax,%rax
   4ca0a:	setne  %dl
   4ca0d:	mov    %r14d,%r15d
   4ca10:	and    %dl,%r15b
   4ca13:	mul    %r12
   4ca16:	seto   %r13b
   4ca1a:	or     %r15b,%r13b
   4ca1d:	mov    0x18(%rsp),%rdx
   4ca22:	imul   %rdx,%r14
   4ca26:	add    %rax,%r14
   4ca29:	mulx   %r12,%rdx,%rax
   4ca2e:	add    %r14,%rax
   4ca31:	setb   %r14b
   4ca35:	or     %r13b,%r14b
   4ca38:	or     %bpl,%r14b
   4ca3b:	add    %r10,%r8
   4ca3e:	adc    %r9,%rdi
   4ca41:	mov    $0xffffffffffffffff,%r15
   4ca48:	cmovb  %r15,%rdi
   4ca4c:	cmovb  %r15,%r8
   4ca50:	add    %rbx,%r8
   4ca53:	adc    %r11,%rdi
   4ca56:	cmovb  %r15,%rdi
   4ca5a:	cmovb  %r15,%r8
   4ca5e:	mov    $0xffffffffffffffff,%r10
   4ca65:	mov    $0xffffffffffffffff,%r9
   4ca6c:	test   $0x1,%r14b
   4ca70:	jne    4ca86 <_ZN15funnel_patterns31pat_branchless_funnel__u64__w5117ha58623d532d5bfdbE+0xc26>
   4ca72:	add    %rdx,%r8
   4ca75:	adc    %rax,%rdi
   4ca78:	cmovb  %r15,%rdi
   4ca7c:	cmovb  %r15,%r8
   4ca80:	mov    %r8,%r10
   4ca83:	mov    %rdi,%r9
   4ca86:	mov    %rsi,%rax
   4ca89:	or     %rcx,%rax
   4ca8c:	mov    0x50(%rsp),%rdx
   4ca91:	cmove  %rdx,%rcx
   4ca95:	mov    0x58(%rsp),%rax
   4ca9a:	cmove  %rax,%rsi
   4ca9e:	mov    %rsi,%rbx
   4caa1:	sub    %r10,%rbx
   4caa4:	mov    %rcx,%r14
   4caa7:	sbb    %r9,%r14
   4caaa:	cmp    %rsi,%r10
   4caad:	sbb    %rcx,%r9
   4cab0:	cmovae %rdx,%r14
   4cab4:	cmovae %rax,%rbx
   4cab8:	mov    0x30(%rsp),%rax
   4cabd:	mov    0x10(%rax),%r15
   4cac1:	cmp    (%rax),%r15
   4cac4:	jne    4cad0 <_ZN15funnel_patterns31pat_branchless_funnel__u64__w5117ha58623d532d5bfdbE+0xc70>
   4cac6:	mov    0x30(%rsp),%rdi
   4cacb:	call   51730 <_ZN5alloc7raw_vec19RawVec$LT$T$C$A$GT$8grow_one17hfe527167958a2dadE>
   4cad0:	mov    0x30(%rsp),%r12
   4cad5:	mov    0x8(%r12),%rax
   4cada:	mov    %r15,%rcx
   4cadd:	shl    $0x4,%rcx
   4cae1:	mov    %r14,0x8(%rax,%rcx,1)
   4cae6:	mov    %rbx,(%rax,%rcx,1)
   4caea:	inc    %r15
   4caed:	mov    %r15,0x10(%r12)
   4caf2:	mov    0x40(%rsp),%r15
   4caf7:	mov    0x28(%r15),%rax
   4cafb:	or     0x20(%r15),%rax
   4caff:	jne    4cb20 <_ZN15funnel_patterns31pat_branchless_funnel__u64__w5117ha58623d532d5bfdbE+0xcc0>
   4cb01:	mov    0x38(%r15),%rax
   4cb05:	or     0x30(%r15),%rax
   4cb09:	jne    4cb20 <_ZN15funnel_patterns31pat_branchless_funnel__u64__w5117ha58623d532d5bfdbE+0xcc0>
   4cb0b:	mov    0x48(%r15),%rax
   4cb0f:	or     0x40(%r15),%rax
   4cb13:	jne    4cb20 <_ZN15funnel_patterns31pat_branchless_funnel__u64__w5117ha58623d532d5bfdbE+0xcc0>
   4cb15:	mov    0x58(%r15),%rax
   4cb19:	or     0x50(%r15),%rax
   4cb1d:	je     4cb76 <_ZN15funnel_patterns31pat_branchless_funnel__u64__w5117ha58623d532d5bfdbE+0xd16>
   4cb1f:	nop
   4cb20:	lea    0x20(%r15),%rax
   4cb24:	vmovdqu64 (%rax),%zmm0
   4cb2a:	vmovdqu64 0x20(%rax),%zmm1
   4cb34:	vmovdqu64 %zmm1,0x1c0(%rsp)
   4cb3c:	vmovdqu64 %zmm0,0x1a0(%rsp)
   4cb47:	lea    0x120(%rsp),%rdi
   4cb4f:	mov    0x78(%rsp),%rsi
   4cb54:	mov    0xb8(%rsp),%rdx
   4cb5c:	lea    0x1a0(%rsp),%rcx
   4cb64:	vzeroupper
   4cb67:	call   51900 <_ZN9hashbrown3map28HashMap$LT$K$C$V$C$S$C$A$GT$6insert17he36ed9c6ee1bc3e6E>
   4cb6c:	mov    0x30(%rsp),%r12
   4cb71:	mov    0x40(%rsp),%r15
   4cb76:	cmpq   $0x0,0x68(%r12)
   4cb7c:	mov    0x14(%rsp),%ebp
   4cb80:	mov    0x6c(%rsp),%r13d
   4cb85:	mov    0xef79c(%rip),%rbx        # 13c328 <_DYNAMIC+0x298>
   4cb8c:	mov    0xe8(%rsp),%r14
   4cb94:	je     4cbc6 <_ZN15funnel_patterns31pat_branchless_funnel__u64__w5117ha58623d532d5bfdbE+0xd66>
   4cb96:	mov    0x4(%rsp),%eax
   4cb9a:	test   %eax,%eax
   4cb9c:	je     4cd8d <_ZN15funnel_patterns31pat_branchless_funnel__u64__w5117ha58623d532d5bfdbE+0xf2d>
   4cba2:	mov    %eax,%edx
   4cba4:	mov    0x80(%r15),%rdi
   4cbab:	mov    0x88(%r15),%rsi
   4cbb2:	xor    %ecx,%ecx
   4cbb4:	call   *0xef766(%rip)        # 13c320 <_DYNAMIC+0x290>
   4cbba:	mov    0x28(%rsp),%rdi
   4cbbf:	mov    %rax,%rsi
   4cbc2:	xor    %edx,%edx
   4cbc4:	call   *%rbx
   4cbc6:	cmpq   $0x0,0x90(%r12)
   4cbcf:	je     4cc04 <_ZN15funnel_patterns31pat_branchless_funnel__u64__w5117ha58623d532d5bfdbE+0xda4>
   4cbd1:	mov    0x4(%rsp),%eax
   4cbd5:	test   %eax,%eax
   4cbd7:	je     4cd8d <_ZN15funnel_patterns31pat_branchless_funnel__u64__w5117ha58623d532d5bfdbE+0xf2d>
   4cbdd:	mov    %eax,%edx
   4cbdf:	mov    0x90(%r15),%rdi
   4cbe6:	mov    0x98(%r15),%rsi
   4cbed:	xor    %ecx,%ecx
   4cbef:	call   *0xef72b(%rip)        # 13c320 <_DYNAMIC+0x290>
   4cbf5:	mov    0x28(%rsp),%rdi
   4cbfa:	mov    %rax,%rsi
   4cbfd:	mov    $0x1,%edx
   4cc02:	call   *%rbx
   4cc04:	cmpq   $0x0,0xb8(%r12)
   4cc0d:	je     4cc42 <_ZN15funnel_patterns31pat_branchless_funnel__u64__w5117ha58623d532d5bfdbE+0xde2>
   4cc0f:	mov    0x4(%rsp),%eax
   4cc13:	test   %eax,%eax
   4cc15:	je     4cd8d <_ZN15funnel_patterns31pat_branchless_funnel__u64__w5117ha58623d532d5bfdbE+0xf2d>
   4cc1b:	mov    %eax,%edx
   4cc1d:	mov    0xa0(%r15),%rdi
   4cc24:	mov    0xa8(%r15),%rsi
   4cc2b:	xor    %ecx,%ecx
   4cc2d:	call   *0xef6ed(%rip)        # 13c320 <_DYNAMIC+0x290>
   4cc33:	mov    0x28(%rsp),%rdi
   4cc38:	mov    %rax,%rsi
   4cc3b:	mov    $0x2,%edx
   4cc40:	call   *%rbx
   4cc42:	cmpq   $0x0,0xe0(%r12)
   4cc4b:	je     4cc80 <_ZN15funnel_patterns31pat_branchless_funnel__u64__w5117ha58623d532d5bfdbE+0xe20>
   4cc4d:	mov    0x4(%rsp),%eax
   4cc51:	test   %eax,%eax
   4cc53:	je     4cd8d <_ZN15funnel_patterns31pat_branchless_funnel__u64__w5117ha58623d532d5bfdbE+0xf2d>
   4cc59:	mov    %eax,%edx
   4cc5b:	mov    0xb0(%r15),%rdi
   4cc62:	mov    0xb8(%r15),%rsi
   4cc69:	xor    %ecx,%ecx
   4cc6b:	call   *0xef6af(%rip)        # 13c320 <_DYNAMIC+0x290>
   4cc71:	mov    0x28(%rsp),%rdi
   4cc76:	mov    %rax,%rsi
   4cc79:	mov    $0x3,%edx
   4cc7e:	call   *%rbx
   4cc80:	mov    0x10(%rsp),%ecx
   4cc84:	mov    %ecx,%edx
   4cc86:	sub    $0x1,%edx
   4cc89:	mov    $0x0,%eax
   4cc8e:	cmovb  %eax,%edx
   4cc91:	testb  $0x1,0xc(%rsp)
   4cc96:	cmove  %ecx,%edx
   4cc99:	mov    %edx,0x18(%rsp)
   4cc9d:	cmp    $0x3b9aca01,%r13d
   4cca4:	je     4c340 <_ZN15funnel_patterns31pat_branchless_funnel__u64__w5117ha58623d532d5bfdbE+0x4e0>
   4ccaa:	mov    0x70(%rsp),%rax
   4ccaf:	mov    %rax,0x190(%rsp)
   4ccb7:	mov    %r13d,0x198(%rsp)
   4ccbf:	mov    0x18(%r15),%eax
   4ccc3:	cmp    $0x3b9aca01,%eax
   4ccc8:	je     4ce24 <_ZN15funnel_patterns31pat_branchless_funnel__u64__w5117ha58623d532d5bfdbE+0xfc4>
   4ccce:	mov    0x1c(%r15),%ecx
   4ccd2:	mov    0x10(%r15),%rdx
   4ccd6:	mov    %rdx,0x110(%rsp)
   4ccde:	mov    %eax,0x118(%rsp)
   4cce5:	mov    %ecx,0x11c(%rsp)
   4ccec:	mov    0x60(%rsp),%rdx
   4ccf1:	lea    0x110(%rsp),%rdi
   4ccf9:	lea    0x190(%rsp),%rsi
   4cd01:	call   *0xef611(%rip)        # 13c318 <_DYNAMIC+0x288>
   4cd07:	mov    %rdx,%rsi
   4cd0a:	mov    %rax,%rdx
   4cd0d:	jmp    4c390 <_ZN15funnel_patterns31pat_branchless_funnel__u64__w5117ha58623d532d5bfdbE+0x530>
   4cd12:	mov    $0x10,%r15d
   4cd18:	xor    %r14d,%r14d
   4cd1b:	mov    0xef5c6(%rip),%rbx        # 13c2e8 <_DYNAMIC+0x258>
   4cd22:	mov    0x8(%rbx),%eax
   4cd25:	test   %eax,%eax
   4cd27:	jne    4cd7c <_ZN15funnel_patterns31pat_branchless_funnel__u64__w5117ha58623d532d5bfdbE+0xf1c>
   4cd29:	mov    (%rbx),%rdi
   4cd2c:	call   *0xef5fe(%rip)        # 13c330 <_DYNAMIC+0x2a0>
   4cd32:	mov    0xef5ff(%rip),%rax        # 13c338 <_DYNAMIC+0x2a8>
   4cd39:	movb   $0x0,(%rax)
   4cd3c:	test   %r14,%r14
   4cd3f:	je     4cd56 <_ZN15funnel_patterns31pat_branchless_funnel__u64__w5117ha58623d532d5bfdbE+0xef6>
   4cd41:	imul   $0xd0,%r14,%rsi
   4cd48:	mov    $0x10,%edx
   4cd4d:	mov    %r15,%rdi
   4cd50:	call   *0xef5ea(%rip)        # 13c340 <_DYNAMIC+0x2b0>
   4cd56:	add    $0x3c18,%rsp
   4cd5d:	pop    %rbx
   4cd5e:	pop    %r12
   4cd60:	pop    %r13
   4cd62:	pop    %r14
   4cd64:	pop    %r15
   4cd66:	pop    %rbp
   4cd67:	ret
   4cd68:	mov    $0x1,%r14d
   4cd6e:	mov    0xef573(%rip),%rbx        # 13c2e8 <_DYNAMIC+0x258>
   4cd75:	mov    0x8(%rbx),%eax
   4cd78:	test   %eax,%eax
   4cd7a:	je     4cd29 <_ZN15funnel_patterns31pat_branchless_funnel__u64__w5117ha58623d532d5bfdbE+0xec9>
   4cd7c:	call   5152e <_ZN3std4sync9once_lock17OnceLock$LT$T$GT$10initialize17h13c8d01c1aad6442E>
   4cd81:	jmp    4cd29 <_ZN15funnel_patterns31pat_branchless_funnel__u64__w5117ha58623d532d5bfdbE+0xec9>
   4cd83:	call   5152e <_ZN3std4sync9once_lock17OnceLock$LT$T$GT$10initialize17h13c8d01c1aad6442E>
   4cd88:	jmp    4c2eb <_ZN15funnel_patterns31pat_branchless_funnel__u64__w5117ha58623d532d5bfdbE+0x48b>
   4cd8d:	lea    0xe6d94(%rip),%rdi        # 133b28 <_ZN15funnel_patterns46__DIVAN_BENCH_PAT_U128_CAT_UNROLLED4__U64__W514PUSH17h68d55424f84a52b4E+0xe0>
   4cd94:	call   *0xef5ae(%rip)        # 13c348 <_DYNAMIC+0x2b8>
   4cd9a:	jmp    4ce40 <_ZN15funnel_patterns31pat_branchless_funnel__u64__w5117ha58623d532d5bfdbE+0xfe0>
   4cd9f:	movq   $0x0,0x108(%rsp)
   4cdab:	lea    0x108(%rsp),%rax
   4cdb3:	mov    %rax,0x1a0(%rsp)
   4cdbb:	mov    0xef58e(%rip),%rax        # 13c350 <_DYNAMIC+0x2c0>
   4cdc2:	mov    %rax,0x1a8(%rsp)
   4cdca:	lea    0xe6cd7(%rip),%rax        # 133aa8 <_ZN15funnel_patterns46__DIVAN_BENCH_PAT_U128_CAT_UNROLLED4__U64__W514PUSH17h68d55424f84a52b4E+0x60>
   4cdd1:	mov    %rax,0x120(%rsp)
   4cdd9:	movq   $0x2,0x128(%rsp)
   4cde5:	movq   $0x0,0x140(%rsp)
   4cdf1:	lea    0x1a0(%rsp),%rax
   4cdf9:	mov    %rax,0x130(%rsp)
   4ce01:	movq   $0x1,0x138(%rsp)
   4ce0d:	lea    0xe6cb4(%rip),%rsi        # 133ac8 <_ZN15funnel_patterns46__DIVAN_BENCH_PAT_U128_CAT_UNROLLED4__U64__W514PUSH17h68d55424f84a52b4E+0x80>
   4ce14:	lea    0x120(%rsp),%rdi
   4ce1c:	call   *0xef536(%rip)        # 13c358 <_DYNAMIC+0x2c8>
   4ce22:	jmp    4ce40 <_ZN15funnel_patterns31pat_branchless_funnel__u64__w5117ha58623d532d5bfdbE+0xfe0>
   4ce24:	lea    0xe6ccd(%rip),%rdi        # 133af8 <_ZN15funnel_patterns46__DIVAN_BENCH_PAT_U128_CAT_UNROLLED4__U64__W514PUSH17h68d55424f84a52b4E+0xb0>
   4ce2b:	call   *0xef52f(%rip)        # 13c360 <_DYNAMIC+0x2d0>
   4ce31:	jmp    4ce40 <_ZN15funnel_patterns31pat_branchless_funnel__u64__w5117ha58623d532d5bfdbE+0xfe0>
   4ce33:	lea    0xe6ca6(%rip),%rdi        # 133ae0 <_ZN15funnel_patterns46__DIVAN_BENCH_PAT_U128_CAT_UNROLLED4__U64__W514PUSH17h68d55424f84a52b4E+0x98>
   4ce3a:	call   *0xef508(%rip)        # 13c348 <_DYNAMIC+0x2b8>
   4ce40:	ud2
   4ce42:	mov    0x1a8(%rsp),%rdi
   4ce4a:	mov    0x1b0(%rsp),%rsi
   4ce52:	lea    0xe6c37(%rip),%rdx        # 133a90 <_ZN15funnel_patterns46__DIVAN_BENCH_PAT_U128_CAT_UNROLLED4__U64__W514PUSH17h68d55424f84a52b4E+0x48>
   4ce59:	call   *0xef509(%rip)        # 13c368 <_DYNAMIC+0x2d8>
   4ce5f:	mov    %r15,0x40(%rsp)
   4ce64:	mov    %rax,%rbx
   4ce67:	test   %r14,%r14
   4ce6a:	jne    4cea0 <_ZN15funnel_patterns31pat_branchless_funnel__u64__w5117ha58623d532d5bfdbE+0x1040>
   4ce6c:	jmp    4ceb5 <_ZN15funnel_patterns31pat_branchless_funnel__u64__w5117ha58623d532d5bfdbE+0x1055>
   4ce6e:	mov    %rax,%rbx
   4ce71:	mov    0x120(%rsp),%rax
   4ce79:	lock decq (%rax)
   4ce7d:	jne    4cea0 <_ZN15funnel_patterns31pat_branchless_funnel__u64__w5117ha58623d532d5bfdbE+0x1040>
   4ce7f:	lea    0x120(%rsp),%rdi
   4ce87:	call   *0xef483(%rip)        # 13c310 <_DYNAMIC+0x280>
   4ce8d:	jmp    4cea0 <_ZN15funnel_patterns31pat_branchless_funnel__u64__w5117ha58623d532d5bfdbE+0x1040>
   4ce8f:	call   *0xef4db(%rip)        # 13c370 <_DYNAMIC+0x2e0>
   4ce95:	jmp    4ce9d <_ZN15funnel_patterns31pat_branchless_funnel__u64__w5117ha58623d532d5bfdbE+0x103d>
   4ce97:	jmp    4ce9d <_ZN15funnel_patterns31pat_branchless_funnel__u64__w5117ha58623d532d5bfdbE+0x103d>
   4ce99:	jmp    4ce9d <_ZN15funnel_patterns31pat_branchless_funnel__u64__w5117ha58623d532d5bfdbE+0x103d>
   4ce9b:	jmp    4ce9d <_ZN15funnel_patterns31pat_branchless_funnel__u64__w5117ha58623d532d5bfdbE+0x103d>
   4ce9d:	mov    %rax,%rbx
   4cea0:	mov    $0xd0,%esi
   4cea5:	mov    $0x10,%edx
   4ceaa:	mov    0x40(%rsp),%rdi
   4ceaf:	call   *0xef48b(%rip)        # 13c340 <_DYNAMIC+0x2b0>
   4ceb5:	mov    %rbx,%rdi
   4ceb8:	call   1328b0 <_Unwind_Resume@plt>

Disassembly of section .init:

Disassembly of section .fini:

Disassembly of section .plt:
