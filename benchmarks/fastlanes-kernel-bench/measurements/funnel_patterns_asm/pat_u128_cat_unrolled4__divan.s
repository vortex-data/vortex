
/home/user/vortex/target/release/deps/funnel_patterns-21c1c00107f42b8a:     file format elf64-x86-64


Disassembly of section .text:

000000000004df20 <_ZN15funnel_patterns32pat_u128_cat_unrolled4__u64__w5117hc0066f52d5fc329cE>:
   4df20:	push   %rbp
   4df21:	push   %r15
   4df23:	push   %r14
   4df25:	push   %r13
   4df27:	push   %r12
   4df29:	push   %rbx
   4df2a:	sub    $0x1000,%rsp
   4df31:	movq   $0x0,(%rsp)
   4df39:	sub    $0x1000,%rsp
   4df40:	movq   $0x0,(%rsp)
   4df48:	sub    $0x1000,%rsp
   4df4f:	movq   $0x0,(%rsp)
   4df57:	sub    $0xc18,%rsp
   4df5e:	mov    %rdi,%r12
   4df61:	lea    0x298(%rsp),%rdi
   4df69:	mov    $0x1980,%edx
   4df6e:	xor    %esi,%esi
   4df70:	call   *0xee34a(%rip)        # 13c2c0 <memset@GLIBC_2.2.5>
   4df76:	vmovdqa64 -0x3db40(%rip),%zmm0        # 10440 <__abi_tag+0x10144>
   4df80:	mov    $0x38,%eax
   4df85:	vpbroadcastq -0x3d7ff(%rip),%zmm1        # 10790 <__abi_tag+0x10494>
   4df8f:	vpbroadcastq -0x3d929(%rip),%zmm2        # 10670 <__abi_tag+0x10374>
   4df99:	vpbroadcastq -0x3d80b(%rip),%zmm3        # 10798 <__abi_tag+0x1049c>
   4dfa3:	vpbroadcastq -0x3d78d(%rip),%zmm4        # 10820 <__abi_tag+0x10524>
   4dfad:	vpbroadcastq -0x3d767(%rip),%zmm5        # 10850 <__abi_tag+0x10554>
   4dfb7:	vpbroadcastq -0x3d931(%rip),%zmm6        # 10690 <__abi_tag+0x10394>
   4dfc1:	vpbroadcastq -0x3d743(%rip),%zmm7        # 10888 <__abi_tag+0x1058c>
   4dfcb:	vpbroadcastq -0x3d77d(%rip),%zmm8        # 10858 <__abi_tag+0x1055c>
   4dfd5:	vpbroadcastq -0x3d75f(%rip),%zmm9        # 10880 <__abi_tag+0x10584>
   4dfdf:	nop
   4dfe0:	vpmullq %zmm1,%zmm0,%zmm10
   4dfe6:	vpaddq %zmm2,%zmm10,%zmm11
   4dfec:	vpaddq %zmm3,%zmm10,%zmm12
   4dff2:	vmovdqu64 %zmm10,0xd8(%rsp,%rax,8)
   4dffd:	vmovdqu64 %zmm11,0x118(%rsp,%rax,8)
   4e008:	vpaddq %zmm4,%zmm10,%zmm11
   4e00e:	vmovdqu64 %zmm12,0x158(%rsp,%rax,8)
   4e019:	vmovdqu64 %zmm11,0x198(%rsp,%rax,8)
   4e024:	cmp    $0x338,%rax
   4e02a:	je     4e07f <_ZN15funnel_patterns32pat_u128_cat_unrolled4__u64__w5117hc0066f52d5fc329cE+0x15f>
   4e02c:	vpaddq %zmm5,%zmm10,%zmm11
   4e032:	vpaddq %zmm6,%zmm10,%zmm12
   4e038:	vpaddq %zmm7,%zmm10,%zmm13
   4e03e:	vpaddq %zmm8,%zmm10,%zmm10
   4e044:	vmovdqu64 %zmm11,0x1d8(%rsp,%rax,8)
   4e04f:	vmovdqu64 %zmm12,0x218(%rsp,%rax,8)
   4e05a:	vmovdqu64 %zmm13,0x258(%rsp,%rax,8)
   4e065:	vmovdqu64 %zmm10,0x298(%rsp,%rax,8)
   4e070:	vpaddq %zmm9,%zmm0,%zmm0
   4e076:	add    $0x40,%rax
   4e07a:	jmp    4dfe0 <_ZN15funnel_patterns32pat_u128_cat_unrolled4__u64__w5117hc0066f52d5fc329cE+0xc0>
   4e07f:	vmovaps -0x3dc09(%rip),%zmm0        # 10480 <__abi_tag+0x10184>
   4e089:	vmovups %zmm0,0x1b98(%rsp)
   4e094:	vmovdqa64 -0x3dbde(%rip),%zmm0        # 104c0 <__abi_tag+0x101c4>
   4e09e:	vmovdqu64 %zmm0,0x1bd8(%rsp)
   4e0a9:	lea    0x2298(%rsp),%rbx
   4e0b1:	lea    0x298(%rsp),%r14
   4e0b9:	mov    $0x1980,%edx
   4e0be:	mov    %rbx,%rdi
   4e0c1:	mov    %r14,%rsi
   4e0c4:	vzeroupper
   4e0c7:	call   *0xee1fb(%rip)        # 13c2c8 <memcpy@GLIBC_2.14>
   4e0cd:	movq   $0x3b9aca07,0xf0(%rsp)
   4e0d9:	xor    %ebp,%ebp
   4e0db:	mov    $0x2000,%edx
   4e0e0:	mov    %r14,%rdi
   4e0e3:	xor    %esi,%esi
   4e0e5:	call   *0xee1d5(%rip)        # 13c2c0 <memset@GLIBC_2.2.5>
   4e0eb:	mov    %rbx,0x208(%rsp)
   4e0f3:	lea    0xf0(%rsp),%rax
   4e0fb:	mov    %rax,0x210(%rsp)
   4e103:	mov    %r14,0x218(%rsp)
   4e10b:	lea    0x208(%rsp),%rax
   4e113:	mov    %rax,0xf8(%rsp)
   4e11b:	lea    0xf8(%rsp),%rax
   4e123:	mov    %rax,0x100(%rsp)
   4e12b:	movq   $0x1,0x100(%r12)
   4e137:	movb   $0x1,0x108(%r12)
   4e140:	mov    0xf0(%r12),%rdx
   4e148:	mov    0xf8(%r12),%rax
   4e150:	movzbl 0x8(%rdx),%ecx
   4e154:	mov    %cl,0x3(%rsp)
   4e158:	cmp    $0x1,%cl
   4e15b:	jne    4e163 <_ZN15funnel_patterns32pat_u128_cat_unrolled4__u64__w5117hc0066f52d5fc329cE+0x243>
   4e15d:	xor    %esi,%esi
   4e15f:	xor    %ecx,%ecx
   4e161:	jmp    4e18d <_ZN15funnel_patterns32pat_u128_cat_unrolled4__u64__w5117hc0066f52d5fc329cE+0x26d>
   4e163:	cmpb   $0x0,0x60(%rax)
   4e167:	je     4e17c <_ZN15funnel_patterns32pat_u128_cat_unrolled4__u64__w5117hc0066f52d5fc329cE+0x25c>
   4e169:	mov    0x64(%rax),%ecx
   4e16c:	mov    %ecx,0x8(%rsp)
   4e170:	mov    $0x2,%ebp
   4e175:	mov    $0x1,%sil
   4e178:	xor    %ecx,%ecx
   4e17a:	jmp    4e18d <_ZN15funnel_patterns32pat_u128_cat_unrolled4__u64__w5117hc0066f52d5fc329cE+0x26d>
   4e17c:	mov    $0x1,%cl
   4e17e:	movl   $0x1,0x8(%rsp)
   4e186:	xor    %esi,%esi
   4e188:	mov    $0x1,%ebp
   4e18d:	mov    (%rdx),%r14
   4e190:	test   %r14,%r14
   4e193:	lea    0x27(%rsp),%rdx
   4e198:	mov    %rdx,0x220(%rsp)
   4e1a0:	setne  0x238(%rsp)
   4e1a8:	lea    0x100(%rsp),%rdi
   4e1b0:	mov    %rdi,0x228(%rsp)
   4e1b8:	mov    %rdx,0x230(%rsp)
   4e1c0:	mov    0x70(%rax),%edi
   4e1c3:	movq   $0x0,0x88(%rsp)
   4e1cf:	mov    $0x0,%edx
   4e1d4:	mov    %rdx,0x80(%rsp)
   4e1dc:	cmp    $0x3b9aca00,%edi
   4e1e2:	je     4e21d <_ZN15funnel_patterns32pat_u128_cat_unrolled4__u64__w5117hc0066f52d5fc329cE+0x2fd>
   4e1e4:	mov    $0x3b9aca00,%edx
   4e1e9:	mulx   0x68(%rax),%r8,%r9
   4e1ef:	mov    %edi,%edx
   4e1f1:	add    %r8,%rdx
   4e1f4:	adc    $0x0,%r9
   4e1f8:	imul   $0x3e8,%r9,%rdi
   4e1ff:	mov    $0x3e8,%r8d
   4e205:	mulx   %r8,%rdx,%r8
   4e20a:	mov    %rdx,0x88(%rsp)
   4e212:	add    %rdi,%r8
   4e215:	mov    %r8,0x80(%rsp)
   4e21d:	movq   $0xffffffffffffffff,0x48(%rsp)
   4e226:	mov    0x80(%rax),%r8d
   4e22d:	movq   $0xffffffffffffffff,0x38(%rsp)
   4e236:	cmp    $0x3b9aca00,%r8d
   4e23d:	je     4e27e <_ZN15funnel_patterns32pat_u128_cat_unrolled4__u64__w5117hc0066f52d5fc329cE+0x35e>
   4e23f:	mov    $0x3b9aca00,%edx
   4e244:	mulx   0x78(%rax),%r9,%rdi
   4e24a:	mov    %r8d,%edx
   4e24d:	add    %r9,%rdx
   4e250:	adc    $0x0,%rdi
   4e254:	mov    $0x3e8,%r8d
   4e25a:	mulx   %r8,%r8,%r9
   4e25f:	mov    %r9,0x38(%rsp)
   4e264:	mov    %r8,0x48(%rsp)
   4e269:	or     %rdi,%rdx
   4e26c:	je     4ee16 <_ZN15funnel_patterns32pat_u128_cat_unrolled4__u64__w5117hc0066f52d5fc329cE+0xef6>
   4e272:	imul   $0x3e8,%rdi,%rdx
   4e279:	add    %rdx,0x38(%rsp)
   4e27e:	mov    0x58(%rax),%edx
   4e281:	cmp    $0x1,%edx
   4e284:	jne    4e290 <_ZN15funnel_patterns32pat_u128_cat_unrolled4__u64__w5117hc0066f52d5fc329cE+0x370>
   4e286:	cmpl   $0x0,0x5c(%rax)
   4e28a:	je     4ee16 <_ZN15funnel_patterns32pat_u128_cat_unrolled4__u64__w5117hc0066f52d5fc329cE+0xef6>
   4e290:	cmpl   $0x1,0x60(%rax)
   4e294:	jne    4e2a0 <_ZN15funnel_patterns32pat_u128_cat_unrolled4__u64__w5117hc0066f52d5fc329cE+0x380>
   4e296:	cmpl   $0x0,0x64(%rax)
   4e29a:	je     4ee16 <_ZN15funnel_patterns32pat_u128_cat_unrolled4__u64__w5117hc0066f52d5fc329cE+0xef6>
   4e2a0:	mov    %r14,0x60(%rsp)
   4e2a5:	test   %dl,%sil
   4e2a8:	je     4e2bb <_ZN15funnel_patterns32pat_u128_cat_unrolled4__u64__w5117hc0066f52d5fc329cE+0x39b>
   4e2aa:	mov    0x5c(%rax),%edx
   4e2ad:	mov    %edx,0x18(%rsp)
   4e2b1:	movl   $0x1,0xc(%rsp)
   4e2b9:	jmp    4e2cb <_ZN15funnel_patterns32pat_u128_cat_unrolled4__u64__w5117hc0066f52d5fc329cE+0x3ab>
   4e2bb:	movzbl %sil,%edx
   4e2bf:	mov    %edx,0xc(%rsp)
   4e2c3:	movl   $0x64,0x18(%rsp)
   4e2cb:	movq   $0x0,0x58(%rsp)
   4e2d4:	mov    $0x0,%edx
   4e2d9:	mov    %rdx,0x50(%rsp)
   4e2de:	test   %cl,%cl
   4e2e0:	je     4e2fd <_ZN15funnel_patterns32pat_u128_cat_unrolled4__u64__w5117hc0066f52d5fc329cE+0x3dd>
   4e2e2:	mov    %r14,%rdi
   4e2e5:	call   *0xedfe5(%rip)        # 13c2d0 <_DYNAMIC+0x240>
   4e2eb:	mov    %rax,0x58(%rsp)
   4e2f0:	mov    %rdx,0x50(%rsp)
   4e2f5:	mov    0xf8(%r12),%rax
   4e2fd:	cmpb   $0x1,0x3(%rsp)
   4e302:	je     4e323 <_ZN15funnel_patterns32pat_u128_cat_unrolled4__u64__w5117hc0066f52d5fc329cE+0x403>
   4e304:	mov    $0x1,%edx
   4e309:	cmpb   $0x0,0x58(%rax)
   4e30d:	je     4e312 <_ZN15funnel_patterns32pat_u128_cat_unrolled4__u64__w5117hc0066f52d5fc329cE+0x3f2>
   4e30f:	mov    0x5c(%rax),%edx
   4e312:	mov    (%r12),%rcx
   4e316:	mov    0x10(%r12),%rsi
   4e31b:	sub    %rsi,%rcx
   4e31e:	cmp    %rcx,%rdx
   4e321:	ja     4e368 <_ZN15funnel_patterns32pat_u128_cat_unrolled4__u64__w5117hc0066f52d5fc329cE+0x448>
   4e323:	testb  $0x1,0x88(%rax)
   4e32a:	jne    4e381 <_ZN15funnel_patterns32pat_u128_cat_unrolled4__u64__w5117hc0066f52d5fc329cE+0x461>
   4e32c:	lock orl $0x0,-0x40(%rsp)
   4e332:	test   %r14,%r14
   4e335:	je     4e353 <_ZN15funnel_patterns32pat_u128_cat_unrolled4__u64__w5117hc0066f52d5fc329cE+0x433>
   4e337:	lfence
   4e33a:	rdtsc
   4e33c:	shl    $0x20,%rdx
   4e340:	or     %rax,%rdx
   4e343:	mov    %rdx,0x70(%rsp)
   4e348:	lfence
   4e34b:	mov    $0x3b9aca00,%r13d
   4e351:	jmp    4e361 <_ZN15funnel_patterns32pat_u128_cat_unrolled4__u64__w5117hc0066f52d5fc329cE+0x441>
   4e353:	call   *0xedf7f(%rip)        # 13c2d8 <_DYNAMIC+0x248>
   4e359:	mov    %rax,0x70(%rsp)
   4e35e:	mov    %edx,%r13d
   4e361:	mov    0x60(%rsp),%r14
   4e366:	jmp    4e387 <_ZN15funnel_patterns32pat_u128_cat_unrolled4__u64__w5117hc0066f52d5fc329cE+0x467>
   4e368:	mov    %r12,%rdi
   4e36b:	call   51800 <_ZN5alloc7raw_vec20RawVecInner$LT$A$GT$7reserve21do_reserve_and_handle17h46771c9d08372974E>
   4e370:	mov    0xf8(%r12),%rax
   4e378:	testb  $0x1,0x88(%rax)
   4e37f:	je     4e32c <_ZN15funnel_patterns32pat_u128_cat_unrolled4__u64__w5117hc0066f52d5fc329cE+0x40c>
   4e381:	mov    $0x3b9aca01,%r13d
   4e387:	mov    %r14,%rdi
   4e38a:	call   *0xedf50(%rip)        # 13c2e0 <_DYNAMIC+0x250>
   4e390:	mov    %rax,0xa8(%rsp)
   4e398:	mov    0xedf49(%rip),%r14        # 13c2e8 <_DYNAMIC+0x258>
   4e39f:	mov    0x8(%r14),%eax
   4e3a3:	test   %eax,%eax
   4e3a5:	jne    4ee43 <_ZN15funnel_patterns32pat_u128_cat_unrolled4__u64__w5117hc0066f52d5fc329cE+0xf23>
   4e3ab:	mov    (%r14),%rdi
   4e3ae:	call   *0xedf3c(%rip)        # 13c2f0 <_DYNAMIC+0x260>
   4e3b4:	mov    0x48(%rsp),%rax
   4e3b9:	or     0x38(%rsp),%rax
   4e3be:	je     4edd2 <_ZN15funnel_patterns32pat_u128_cat_unrolled4__u64__w5117hc0066f52d5fc329cE+0xeb2>
   4e3c4:	lea    0x50(%r12),%rax
   4e3c9:	mov    %rax,0x28(%rsp)
   4e3ce:	lea    0x18(%r12),%rax
   4e3d3:	mov    %rax,0x78(%rsp)
   4e3d8:	mov    $0x10,%r15d
   4e3de:	mov    $0x1,%al
   4e3e0:	xor    %edx,%edx
   4e3e2:	xor    %esi,%esi
   4e3e4:	xor    %r14d,%r14d
   4e3e7:	mov    %r12,0x30(%rsp)
   4e3ec:	mov    %r13d,0x6c(%rsp)
   4e3f1:	jmp    4e46b <_ZN15funnel_patterns32pat_u128_cat_unrolled4__u64__w5117hc0066f52d5fc329cE+0x54b>
   4e3f3:	data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
   4e400:	mov    0x90(%rsp),%rdi
   4e408:	cmp    $0x3e9,%rdi
   4e40f:	mov    $0x3e8,%eax
   4e414:	cmovae %rdi,%rax
   4e418:	mov    0x98(%rsp),%rsi
   4e420:	test   %rsi,%rsi
   4e423:	mov    $0x3e8,%ecx
   4e428:	cmove  %rcx,%rdi
   4e42c:	cmove  %rax,%rdi
   4e430:	mov    0xb0(%rsp),%rdx
   4e438:	add    %rdi,%rdx
   4e43b:	adc    %rsi,%r14
   4e43e:	mov    $0xffffffffffffffff,%rax
   4e445:	cmovb  %rax,%r14
   4e449:	cmovb  %rax,%rdx
   4e44d:	mov    %r14,%rsi
   4e450:	mov    $0x1,%r14d
   4e456:	xor    %eax,%eax
   4e458:	cmp    0x48(%rsp),%rdx
   4e45d:	mov    %rsi,%rcx
   4e460:	sbb    0x38(%rsp),%rcx
   4e465:	jae    4eddb <_ZN15funnel_patterns32pat_u128_cat_unrolled4__u64__w5117hc0066f52d5fc329cE+0xebb>
   4e46b:	cmp    0x88(%rsp),%rdx
   4e473:	mov    %rsi,0xe8(%rsp)
   4e47b:	mov    %rsi,%rcx
   4e47e:	sbb    0x80(%rsp),%rcx
   4e486:	jb     4e49a <_ZN15funnel_patterns32pat_u128_cat_unrolled4__u64__w5117hc0066f52d5fc329cE+0x57a>
   4e488:	testb  $0x1,0xc(%rsp)
   4e48d:	je     4e49a <_ZN15funnel_patterns32pat_u128_cat_unrolled4__u64__w5117hc0066f52d5fc329cE+0x57a>
   4e48f:	cmpl   $0x0,0x18(%rsp)
   4e494:	je     4eddb <_ZN15funnel_patterns32pat_u128_cat_unrolled4__u64__w5117hc0066f52d5fc329cE+0xebb>
   4e49a:	mov    %rdx,0xb0(%rsp)
   4e4a2:	test   %ebp,%ebp
   4e4a4:	mov    0x8(%rsp),%ecx
   4e4a8:	mov    $0x1,%edx
   4e4ad:	cmove  %edx,%ecx
   4e4b0:	mov    %ecx,0x4(%rsp)
   4e4b4:	mov    %ecx,0x48(%r12)
   4e4b9:	movq   $0x0,0x268(%rsp)
   4e4c5:	mov    0x28(%rsp),%rcx
   4e4ca:	mov    %rcx,0x240(%rsp)
   4e4d2:	lea    0x220(%rsp),%rcx
   4e4da:	mov    %rcx,0x248(%rsp)
   4e4e2:	lea    0x4(%rsp),%rcx
   4e4e7:	mov    %rcx,0x250(%rsp)
   4e4ef:	lea    0x268(%rsp),%rcx
   4e4f7:	mov    %rcx,0x258(%rsp)
   4e4ff:	lea    0x60(%rsp),%rcx
   4e504:	mov    %rcx,0x260(%rsp)
   4e50c:	test   $0x1,%al
   4e50e:	jne    4e680 <_ZN15funnel_patterns32pat_u128_cat_unrolled4__u64__w5117hc0066f52d5fc329cE+0x760>
   4e514:	imul   $0xd0,%r14,%rax
   4e51b:	add    %r15,%rax
   4e51e:	mov    %rax,%rdx
   4e521:	sub    %r15,%rdx
   4e524:	add    $0xffffffffffffff30,%rdx
   4e52b:	movabs $0x4ec4ec4ec4ec4ec5,%rcx
   4e535:	mulx   %rcx,%rcx,%rcx
   4e53a:	mov    %r15,%rsi
   4e53d:	cmp    $0xc30,%rdx
   4e544:	jb     4e6d0 <_ZN15funnel_patterns32pat_u128_cat_unrolled4__u64__w5117hc0066f52d5fc329cE+0x7b0>
   4e54a:	shr    $0x6,%rcx
   4e54e:	inc    %rcx
   4e551:	cmp    $0x3330,%rdx
   4e558:	jae    4e570 <_ZN15funnel_patterns32pat_u128_cat_unrolled4__u64__w5117hc0066f52d5fc329cE+0x650>
   4e55a:	xor    %edx,%edx
   4e55c:	mov    %r15,%rdi
   4e55f:	jmp    4e61f <_ZN15funnel_patterns32pat_u128_cat_unrolled4__u64__w5117hc0066f52d5fc329cE+0x6ff>
   4e564:	data16 data16 cs nopw 0x0(%rax,%rax,1)
   4e570:	mov    %rcx,%rdx
   4e573:	movabs $0x3ffffffffffffc0,%rsi
   4e57d:	and    %rsi,%rdx
   4e580:	imul   $0xd0,%rdx,%rsi
   4e587:	lea    (%r15,%rsi,1),%rdi
   4e58b:	mov    %rdx,%r8
   4e58e:	mov    %r15,%r9
   4e591:	vpbroadcastd -0x3dcc7(%rip),%zmm0        # 108d4 <__abi_tag+0x105d8>
   4e59b:	vmovdqa64 -0x3e0a5(%rip),%zmm1        # 10500 <__abi_tag+0x10204>
   4e5a5:	vmovdqa64 -0x3e06f(%rip),%zmm2        # 10540 <__abi_tag+0x10244>
   4e5af:	vmovdqa64 -0x3e039(%rip),%zmm3        # 10580 <__abi_tag+0x10284>
   4e5b9:	vmovdqa64 -0x3e003(%rip),%zmm4        # 105c0 <__abi_tag+0x102c4>
   4e5c3:	data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
   4e5d0:	kxnorw %k0,%k0,%k1
   4e5d4:	vpscatterdd %zmm0,0x8(%r9,%zmm1,1){%k1}
   4e5dc:	kxnorw %k0,%k0,%k1
   4e5e0:	vpscatterdd %zmm0,0x8(%r9,%zmm2,1){%k1}
   4e5e8:	kxnorw %k0,%k0,%k1
   4e5ec:	vpscatterdd %zmm0,0x8(%r9,%zmm3,1){%k1}
   4e5f4:	kxnorw %k0,%k0,%k1
   4e5f8:	vpscatterdd %zmm0,0x8(%r9,%zmm4,1){%k1}
   4e600:	add    $0x3400,%r9
   4e607:	add    $0xffffffffffffffc0,%r8
   4e60b:	jne    4e5d0 <_ZN15funnel_patterns32pat_u128_cat_unrolled4__u64__w5117hc0066f52d5fc329cE+0x6b0>
   4e60d:	cmp    %rdx,%rcx
   4e610:	je     4e6e3 <_ZN15funnel_patterns32pat_u128_cat_unrolled4__u64__w5117hc0066f52d5fc329cE+0x7c3>
   4e616:	test   $0x30,%cl
   4e619:	je     4e6cd <_ZN15funnel_patterns32pat_u128_cat_unrolled4__u64__w5117hc0066f52d5fc329cE+0x7ad>
   4e61f:	movabs $0x3ffffffffffffc0,%rsi
   4e629:	lea    0x30(%rsi),%r8
   4e62d:	and    %rcx,%r8
   4e630:	imul   $0xd0,%r8,%rsi
   4e637:	add    %r15,%rsi
   4e63a:	sub    %r8,%rdx
   4e63d:	vpbroadcastd -0x3dd73(%rip),%zmm0        # 108d4 <__abi_tag+0x105d8>
   4e647:	vmovdqa64 -0x3e151(%rip),%zmm1        # 10500 <__abi_tag+0x10204>
   4e651:	data16 data16 data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
   4e660:	kxnorw %k0,%k0,%k1
   4e664:	vpscatterdd %zmm0,0x8(%rdi,%zmm1,1){%k1}
   4e66c:	add    $0xd00,%rdi
   4e673:	add    $0x10,%rdx
   4e677:	jne    4e660 <_ZN15funnel_patterns32pat_u128_cat_unrolled4__u64__w5117hc0066f52d5fc329cE+0x740>
   4e679:	cmp    %r8,%rcx
   4e67c:	jne    4e6d0 <_ZN15funnel_patterns32pat_u128_cat_unrolled4__u64__w5117hc0066f52d5fc329cE+0x7b0>
   4e67e:	jmp    4e6e3 <_ZN15funnel_patterns32pat_u128_cat_unrolled4__u64__w5117hc0066f52d5fc329cE+0x7c3>
   4e680:	movq   $0x0,0x128(%rsp)
   4e68c:	mov    $0x10,%esi
   4e691:	mov    $0xd0,%edx
   4e696:	lea    0x1a0(%rsp),%rdi
   4e69e:	lea    0x120(%rsp),%rcx
   4e6a6:	call   516c0 <_ZN5alloc7raw_vec11finish_grow17hedc133b40cb748a9E>
   4e6ab:	cmpb   $0x0,0x1a0(%rsp)
   4e6b3:	jne    4ef02 <_ZN15funnel_patterns32pat_u128_cat_unrolled4__u64__w5117hc0066f52d5fc329cE+0xfe2>
   4e6b9:	mov    0x1a8(%rsp),%r15
   4e6c1:	lea    0xd0(%r15),%rax
   4e6c8:	jmp    4e51e <_ZN15funnel_patterns32pat_u128_cat_unrolled4__u64__w5117hc0066f52d5fc329cE+0x5fe>
   4e6cd:	add    %r15,%rsi
   4e6d0:	movl   $0x3b9aca01,0x8(%rsi)
   4e6d7:	add    $0xd0,%rsi
   4e6de:	cmp    %rax,%rsi
   4e6e1:	jne    4e6d0 <_ZN15funnel_patterns32pat_u128_cat_unrolled4__u64__w5117hc0066f52d5fc329cE+0x7b0>
   4e6e3:	mov    %r12,%rbx
   4e6e6:	mov    %r15,0x40(%rsp)
   4e6eb:	vzeroupper
   4e6ee:	call   *0xedc04(%rip)        # 13c2f8 <_DYNAMIC+0x268>
   4e6f4:	mov    %rax,0x120(%rsp)
   4e6fc:	movq   $0x0,0x128(%rsp)
   4e708:	lea    0x9f1(%rip),%rax        # 4f100 <_ZN30codspeed_divan_compat_walltime11thread_pool19TaskShared$LT$F$GT$3new4call17he8a1023ba3b23565E>
   4e70f:	mov    %rax,0x130(%rsp)
   4e717:	lea    0x240(%rsp),%rax
   4e71f:	mov    %rax,0x138(%rsp)
   4e727:	mov    %r15,0x140(%rsp)
   4e72f:	mov    0xedbca(%rip),%rdi        # 13c300 <_DYNAMIC+0x270>
   4e736:	xor    %esi,%esi
   4e738:	lea    0x120(%rsp),%rdx
   4e740:	call   *0xedbc2(%rip)        # 13c308 <_DYNAMIC+0x278>
   4e746:	mov    0x120(%rsp),%rax
   4e74e:	lock decq (%rax)
   4e752:	jne    4e762 <_ZN15funnel_patterns32pat_u128_cat_unrolled4__u64__w5117hc0066f52d5fc329cE+0x842>
   4e754:	lea    0x120(%rsp),%rdi
   4e75c:	call   *0xedbae(%rip)        # 13c310 <_DYNAMIC+0x280>
   4e762:	cmpl   $0x3b9aca01,0x8(%r15)
   4e76a:	je     4ee5f <_ZN15funnel_patterns32pat_u128_cat_unrolled4__u64__w5117hc0066f52d5fc329cE+0xf3f>
   4e770:	cmpb   $0x1,0x3(%rsp)
   4e775:	je     4ee28 <_ZN15funnel_patterns32pat_u128_cat_unrolled4__u64__w5117hc0066f52d5fc329cE+0xf08>
   4e77b:	vmovups 0x10(%r15),%xmm0
   4e781:	vmovaps %xmm0,0x1a0(%rsp)
   4e78a:	vmovdqu (%r15),%xmm0
   4e78f:	vmovdqa %xmm0,0x120(%rsp)
   4e798:	mov    0xc0(%r15),%rdx
   4e79f:	lea    0x1a0(%rsp),%rdi
   4e7a7:	lea    0x120(%rsp),%rsi
   4e7af:	call   *0xedb63(%rip)        # 13c318 <_DYNAMIC+0x288>
   4e7b5:	lea    0x10(%r15),%rax
   4e7b9:	mov    %rax,%r12
   4e7bc:	vmovups (%rax),%xmm0
   4e7c0:	vmovaps %xmm0,0x1a0(%rsp)
   4e7c9:	vmovdqu (%r15),%xmm0
   4e7ce:	vmovdqa %xmm0,0x120(%rsp)
   4e7d7:	mov    0xc0(%r15),%rdx
   4e7de:	lea    0x1a0(%rsp),%rdi
   4e7e6:	lea    0x120(%rsp),%rsi
   4e7ee:	call   *0xedb24(%rip)        # 13c318 <_DYNAMIC+0x288>
   4e7f4:	mov    %rax,%rsi
   4e7f7:	cmp    $0x1,%ebp
   4e7fa:	mov    %rdx,0x98(%rsp)
   4e802:	mov    %rax,0x90(%rsp)
   4e80a:	jne    4e880 <_ZN15funnel_patterns32pat_u128_cat_unrolled4__u64__w5117hc0066f52d5fc329cE+0x960>
   4e80c:	mov    %rbx,%rcx
   4e80f:	movq   $0x0,0x10(%rbx)
   4e817:	cmpq   $0x0,0x30(%rbx)
   4e81c:	mov    0x58(%rsp),%r13
   4e821:	mov    0x50(%rsp),%rbp
   4e826:	je     4e8a6 <_ZN15funnel_patterns32pat_u128_cat_unrolled4__u64__w5117hc0066f52d5fc329cE+0x986>
   4e828:	mov    0x20(%rcx),%r14
   4e82c:	test   %r14,%r14
   4e82f:	je     4e895 <_ZN15funnel_patterns32pat_u128_cat_unrolled4__u64__w5117hc0066f52d5fc329cE+0x975>
   4e831:	mov    0x78(%rsp),%rax
   4e836:	mov    (%rax),%rdi
   4e839:	lea    0x11(%r14),%rdx
   4e83d:	mov    $0xff,%esi
   4e842:	call   *0xeda78(%rip)        # 13c2c0 <memset@GLIBC_2.2.5>
   4e848:	mov    0x90(%rsp),%rsi
   4e850:	mov    0x98(%rsp),%rdx
   4e858:	lea    0x1(%r14),%rax
   4e85c:	mov    %rax,%rcx
   4e85f:	shr    $0x3,%rcx
   4e863:	and    $0xfffffffffffffff8,%rax
   4e867:	sub    %rcx,%rax
   4e86a:	cmp    $0x8,%r14
   4e86e:	cmovb  %r14,%rax
   4e872:	jmp    4e897 <_ZN15funnel_patterns32pat_u128_cat_unrolled4__u64__w5117hc0066f52d5fc329cE+0x977>
   4e874:	data16 data16 cs nopw 0x0(%rax,%rax,1)
   4e880:	mov    %ebp,0x14(%rsp)
   4e884:	mov    0x4(%rsp),%eax
   4e888:	mov    %rax,0xa0(%rsp)
   4e890:	jmp    4e937 <_ZN15funnel_patterns32pat_u128_cat_unrolled4__u64__w5117hc0066f52d5fc329cE+0xa17>
   4e895:	xor    %eax,%eax
   4e897:	mov    %rbx,%rcx
   4e89a:	movq   $0x0,0x30(%rbx)
   4e8a2:	mov    %rax,0x28(%rbx)
   4e8a6:	cmpq   $0x0,0x68(%rcx)
   4e8ab:	je     4e8b5 <_ZN15funnel_patterns32pat_u128_cat_unrolled4__u64__w5117hc0066f52d5fc329cE+0x995>
   4e8ad:	movq   $0x0,0x60(%rcx)
   4e8b5:	cmpq   $0x0,0x90(%rcx)
   4e8bd:	je     4e8ca <_ZN15funnel_patterns32pat_u128_cat_unrolled4__u64__w5117hc0066f52d5fc329cE+0x9aa>
   4e8bf:	movq   $0x0,0x88(%rcx)
   4e8ca:	cmpq   $0x0,0xb8(%rcx)
   4e8d2:	je     4e8df <_ZN15funnel_patterns32pat_u128_cat_unrolled4__u64__w5117hc0066f52d5fc329cE+0x9bf>
   4e8d4:	movq   $0x0,0xb0(%rcx)
   4e8df:	cmpq   $0x0,0xe0(%rcx)
   4e8e7:	je     4e8f4 <_ZN15funnel_patterns32pat_u128_cat_unrolled4__u64__w5117hc0066f52d5fc329cE+0x9d4>
   4e8e9:	movq   $0x0,0xd8(%rcx)
   4e8f4:	mov    %r13,%rax
   4e8f7:	or     %rbp,%rax
   4e8fa:	je     4eef3 <_ZN15funnel_patterns32pat_u128_cat_unrolled4__u64__w5117hc0066f52d5fc329cE+0xfd3>
   4e900:	mov    %rsi,%rdi
   4e903:	mov    %rdx,%rsi
   4e906:	mov    %r13,%rdx
   4e909:	mov    %rbp,%rcx
   4e90c:	call   *0xeda0e(%rip)        # 13c320 <_DYNAMIC+0x290>
   4e912:	cmp    $0x65,%rax
   4e916:	sbb    $0x0,%rdx
   4e91a:	mov    0x4(%rsp),%ecx
   4e91e:	mov    %rcx,0xa0(%rsp)
   4e926:	jae    4e950 <_ZN15funnel_patterns32pat_u128_cat_unrolled4__u64__w5117hc0066f52d5fc329cE+0xa30>
   4e928:	lea    (%rcx,%rcx,1),%eax
   4e92b:	mov    %eax,0x8(%rsp)
   4e92f:	movl   $0x1,0x14(%rsp)
   4e937:	mov    0x18(%rsp),%eax
   4e93b:	mov    %eax,0x10(%rsp)
   4e93f:	mov    %r12,%rdx
   4e942:	jmp    4e98d <_ZN15funnel_patterns32pat_u128_cat_unrolled4__u64__w5117hc0066f52d5fc329cE+0xa6d>
   4e944:	data16 data16 cs nopw 0x0(%rax,%rax,1)
   4e950:	mov    0xf8(%rbx),%rax
   4e957:	movl   $0x1,0xc(%rsp)
   4e95f:	cmpb   $0x0,0x58(%rax)
   4e963:	mov    %r12,%rdx
   4e966:	je     4e979 <_ZN15funnel_patterns32pat_u128_cat_unrolled4__u64__w5117hc0066f52d5fc329cE+0xa59>
   4e968:	movl   $0x2,0x14(%rsp)
   4e970:	mov    0x5c(%rax),%eax
   4e973:	mov    %eax,0x10(%rsp)
   4e977:	jmp    4e989 <_ZN15funnel_patterns32pat_u128_cat_unrolled4__u64__w5117hc0066f52d5fc329cE+0xa69>
   4e979:	movl   $0x64,0x10(%rsp)
   4e981:	movl   $0x2,0x14(%rsp)
   4e989:	mov    %ecx,0x8(%rsp)
   4e98d:	mov    0xa8(%rsp),%rax
   4e995:	mov    (%rax),%rbp
   4e998:	mov    0x8(%rax),%r13
   4e99c:	mov    0x18(%rax),%r14
   4e9a0:	mov    0x10(%rax),%rcx
   4e9a4:	mov    %rcx,0xc0(%rsp)
   4e9ac:	mov    %r15,%rcx
   4e9af:	mov    0x40(%r15),%r15
   4e9b3:	mov    0x28(%rax),%rsi
   4e9b7:	mov    %rsi,0xc8(%rsp)
   4e9bf:	mov    0x20(%rax),%rsi
   4e9c3:	mov    %rsi,0xd0(%rsp)
   4e9cb:	mov    0x50(%rcx),%rbx
   4e9cf:	mov    0x38(%rax),%rsi
   4e9d3:	mov    %rsi,0xe0(%rsp)
   4e9db:	mov    0x30(%rax),%rax
   4e9df:	mov    %rax,0x18(%rsp)
   4e9e4:	mov    0x20(%rcx),%rax
   4e9e8:	mov    %rax,0xd8(%rsp)
   4e9f0:	mov    0x30(%rcx),%r12
   4e9f4:	vmovups (%rdx),%xmm0
   4e9f8:	vmovaps %xmm0,0x1a0(%rsp)
   4ea01:	mov    0x30(%rsp),%rax
   4ea06:	mov    0x10(%rax),%rax
   4ea0a:	mov    %rax,0xb8(%rsp)
   4ea12:	vmovdqu (%rcx),%xmm0
   4ea16:	vmovdqa %xmm0,0x120(%rsp)
   4ea1f:	mov    0xc0(%rcx),%rdx
   4ea26:	lea    0x1a0(%rsp),%rdi
   4ea2e:	lea    0x120(%rsp),%rsi
   4ea36:	call   *0xed8dc(%rip)        # 13c318 <_DYNAMIC+0x288>
   4ea3c:	mov    %rax,%rsi
   4ea3f:	mov    %rdx,%rcx
   4ea42:	mov    0xa0(%rsp),%edi
   4ea49:	mov    %r13,%rax
   4ea4c:	mul    %rdi
   4ea4f:	mov    %rbp,%rdx
   4ea52:	mulx   %rdi,%r10,%r9
   4ea57:	seto   %dl
   4ea5a:	add    %rax,%r9
   4ea5d:	setb   %r13b
   4ea61:	or     %dl,%r13b
   4ea64:	mov    %r14,%rax
   4ea67:	mul    %r15
   4ea6a:	seto   %r11b
   4ea6e:	mov    0xc0(%rsp),%rdx
   4ea76:	mulx   %r15,%r8,%rdi
   4ea7b:	add    %rax,%rdi
   4ea7e:	setb   %r14b
   4ea82:	or     %r11b,%r14b
   4ea85:	mov    0xc8(%rsp),%rax
   4ea8d:	mul    %rbx
   4ea90:	mov    0xd0(%rsp),%rdx
   4ea98:	mulx   %rbx,%rbx,%r11
   4ea9d:	seto   %dl
   4eaa0:	add    %rax,%r11
   4eaa3:	setb   %bpl
   4eaa7:	or     %dl,%bpl
   4eaaa:	or     %r14b,%bpl
   4eaad:	or     %r13b,%bpl
   4eab0:	xor    %r14d,%r14d
   4eab3:	add    0xd8(%rsp),%r12
   4eabb:	setb   %r14b
   4eabf:	mov    0xe0(%rsp),%rax
   4eac7:	test   %rax,%rax
   4eaca:	setne  %dl
   4eacd:	mov    %r14d,%r15d
   4ead0:	and    %dl,%r15b
   4ead3:	mul    %r12
   4ead6:	seto   %r13b
   4eada:	or     %r15b,%r13b
   4eadd:	mov    0x18(%rsp),%rdx
   4eae2:	imul   %rdx,%r14
   4eae6:	add    %rax,%r14
   4eae9:	mulx   %r12,%rdx,%rax
   4eaee:	add    %r14,%rax
   4eaf1:	setb   %r14b
   4eaf5:	or     %r13b,%r14b
   4eaf8:	or     %bpl,%r14b
   4eafb:	add    %r10,%r8
   4eafe:	adc    %r9,%rdi
   4eb01:	mov    $0xffffffffffffffff,%r15
   4eb08:	cmovb  %r15,%rdi
   4eb0c:	cmovb  %r15,%r8
   4eb10:	add    %rbx,%r8
   4eb13:	adc    %r11,%rdi
   4eb16:	cmovb  %r15,%rdi
   4eb1a:	cmovb  %r15,%r8
   4eb1e:	mov    $0xffffffffffffffff,%r10
   4eb25:	mov    $0xffffffffffffffff,%r9
   4eb2c:	test   $0x1,%r14b
   4eb30:	jne    4eb46 <_ZN15funnel_patterns32pat_u128_cat_unrolled4__u64__w5117hc0066f52d5fc329cE+0xc26>
   4eb32:	add    %rdx,%r8
   4eb35:	adc    %rax,%rdi
   4eb38:	cmovb  %r15,%rdi
   4eb3c:	cmovb  %r15,%r8
   4eb40:	mov    %r8,%r10
   4eb43:	mov    %rdi,%r9
   4eb46:	mov    %rsi,%rax
   4eb49:	or     %rcx,%rax
   4eb4c:	mov    0x50(%rsp),%rdx
   4eb51:	cmove  %rdx,%rcx
   4eb55:	mov    0x58(%rsp),%rax
   4eb5a:	cmove  %rax,%rsi
   4eb5e:	mov    %rsi,%rbx
   4eb61:	sub    %r10,%rbx
   4eb64:	mov    %rcx,%r14
   4eb67:	sbb    %r9,%r14
   4eb6a:	cmp    %rsi,%r10
   4eb6d:	sbb    %rcx,%r9
   4eb70:	cmovae %rdx,%r14
   4eb74:	cmovae %rax,%rbx
   4eb78:	mov    0x30(%rsp),%rax
   4eb7d:	mov    0x10(%rax),%r15
   4eb81:	cmp    (%rax),%r15
   4eb84:	jne    4eb90 <_ZN15funnel_patterns32pat_u128_cat_unrolled4__u64__w5117hc0066f52d5fc329cE+0xc70>
   4eb86:	mov    0x30(%rsp),%rdi
   4eb8b:	call   51730 <_ZN5alloc7raw_vec19RawVec$LT$T$C$A$GT$8grow_one17hfe527167958a2dadE>
   4eb90:	mov    0x30(%rsp),%r12
   4eb95:	mov    0x8(%r12),%rax
   4eb9a:	mov    %r15,%rcx
   4eb9d:	shl    $0x4,%rcx
   4eba1:	mov    %r14,0x8(%rax,%rcx,1)
   4eba6:	mov    %rbx,(%rax,%rcx,1)
   4ebaa:	inc    %r15
   4ebad:	mov    %r15,0x10(%r12)
   4ebb2:	mov    0x40(%rsp),%r15
   4ebb7:	mov    0x28(%r15),%rax
   4ebbb:	or     0x20(%r15),%rax
   4ebbf:	jne    4ebe0 <_ZN15funnel_patterns32pat_u128_cat_unrolled4__u64__w5117hc0066f52d5fc329cE+0xcc0>
   4ebc1:	mov    0x38(%r15),%rax
   4ebc5:	or     0x30(%r15),%rax
   4ebc9:	jne    4ebe0 <_ZN15funnel_patterns32pat_u128_cat_unrolled4__u64__w5117hc0066f52d5fc329cE+0xcc0>
   4ebcb:	mov    0x48(%r15),%rax
   4ebcf:	or     0x40(%r15),%rax
   4ebd3:	jne    4ebe0 <_ZN15funnel_patterns32pat_u128_cat_unrolled4__u64__w5117hc0066f52d5fc329cE+0xcc0>
   4ebd5:	mov    0x58(%r15),%rax
   4ebd9:	or     0x50(%r15),%rax
   4ebdd:	je     4ec36 <_ZN15funnel_patterns32pat_u128_cat_unrolled4__u64__w5117hc0066f52d5fc329cE+0xd16>
   4ebdf:	nop
   4ebe0:	lea    0x20(%r15),%rax
   4ebe4:	vmovdqu64 (%rax),%zmm0
   4ebea:	vmovdqu64 0x20(%rax),%zmm1
   4ebf4:	vmovdqu64 %zmm1,0x1c0(%rsp)
   4ebfc:	vmovdqu64 %zmm0,0x1a0(%rsp)
   4ec07:	lea    0x120(%rsp),%rdi
   4ec0f:	mov    0x78(%rsp),%rsi
   4ec14:	mov    0xb8(%rsp),%rdx
   4ec1c:	lea    0x1a0(%rsp),%rcx
   4ec24:	vzeroupper
   4ec27:	call   51900 <_ZN9hashbrown3map28HashMap$LT$K$C$V$C$S$C$A$GT$6insert17he36ed9c6ee1bc3e6E>
   4ec2c:	mov    0x30(%rsp),%r12
   4ec31:	mov    0x40(%rsp),%r15
   4ec36:	cmpq   $0x0,0x68(%r12)
   4ec3c:	mov    0x14(%rsp),%ebp
   4ec40:	mov    0x6c(%rsp),%r13d
   4ec45:	mov    0xed6dc(%rip),%rbx        # 13c328 <_DYNAMIC+0x298>
   4ec4c:	mov    0xe8(%rsp),%r14
   4ec54:	je     4ec86 <_ZN15funnel_patterns32pat_u128_cat_unrolled4__u64__w5117hc0066f52d5fc329cE+0xd66>
   4ec56:	mov    0x4(%rsp),%eax
   4ec5a:	test   %eax,%eax
   4ec5c:	je     4ee4d <_ZN15funnel_patterns32pat_u128_cat_unrolled4__u64__w5117hc0066f52d5fc329cE+0xf2d>
   4ec62:	mov    %eax,%edx
   4ec64:	mov    0x80(%r15),%rdi
   4ec6b:	mov    0x88(%r15),%rsi
   4ec72:	xor    %ecx,%ecx
   4ec74:	call   *0xed6a6(%rip)        # 13c320 <_DYNAMIC+0x290>
   4ec7a:	mov    0x28(%rsp),%rdi
   4ec7f:	mov    %rax,%rsi
   4ec82:	xor    %edx,%edx
   4ec84:	call   *%rbx
   4ec86:	cmpq   $0x0,0x90(%r12)
   4ec8f:	je     4ecc4 <_ZN15funnel_patterns32pat_u128_cat_unrolled4__u64__w5117hc0066f52d5fc329cE+0xda4>
   4ec91:	mov    0x4(%rsp),%eax
   4ec95:	test   %eax,%eax
   4ec97:	je     4ee4d <_ZN15funnel_patterns32pat_u128_cat_unrolled4__u64__w5117hc0066f52d5fc329cE+0xf2d>
   4ec9d:	mov    %eax,%edx
   4ec9f:	mov    0x90(%r15),%rdi
   4eca6:	mov    0x98(%r15),%rsi
   4ecad:	xor    %ecx,%ecx
   4ecaf:	call   *0xed66b(%rip)        # 13c320 <_DYNAMIC+0x290>
   4ecb5:	mov    0x28(%rsp),%rdi
   4ecba:	mov    %rax,%rsi
   4ecbd:	mov    $0x1,%edx
   4ecc2:	call   *%rbx
   4ecc4:	cmpq   $0x0,0xb8(%r12)
   4eccd:	je     4ed02 <_ZN15funnel_patterns32pat_u128_cat_unrolled4__u64__w5117hc0066f52d5fc329cE+0xde2>
   4eccf:	mov    0x4(%rsp),%eax
   4ecd3:	test   %eax,%eax
   4ecd5:	je     4ee4d <_ZN15funnel_patterns32pat_u128_cat_unrolled4__u64__w5117hc0066f52d5fc329cE+0xf2d>
   4ecdb:	mov    %eax,%edx
   4ecdd:	mov    0xa0(%r15),%rdi
   4ece4:	mov    0xa8(%r15),%rsi
   4eceb:	xor    %ecx,%ecx
   4eced:	call   *0xed62d(%rip)        # 13c320 <_DYNAMIC+0x290>
   4ecf3:	mov    0x28(%rsp),%rdi
   4ecf8:	mov    %rax,%rsi
   4ecfb:	mov    $0x2,%edx
   4ed00:	call   *%rbx
   4ed02:	cmpq   $0x0,0xe0(%r12)
   4ed0b:	je     4ed40 <_ZN15funnel_patterns32pat_u128_cat_unrolled4__u64__w5117hc0066f52d5fc329cE+0xe20>
   4ed0d:	mov    0x4(%rsp),%eax
   4ed11:	test   %eax,%eax
   4ed13:	je     4ee4d <_ZN15funnel_patterns32pat_u128_cat_unrolled4__u64__w5117hc0066f52d5fc329cE+0xf2d>
   4ed19:	mov    %eax,%edx
   4ed1b:	mov    0xb0(%r15),%rdi
   4ed22:	mov    0xb8(%r15),%rsi
   4ed29:	xor    %ecx,%ecx
   4ed2b:	call   *0xed5ef(%rip)        # 13c320 <_DYNAMIC+0x290>
   4ed31:	mov    0x28(%rsp),%rdi
   4ed36:	mov    %rax,%rsi
   4ed39:	mov    $0x3,%edx
   4ed3e:	call   *%rbx
   4ed40:	mov    0x10(%rsp),%ecx
   4ed44:	mov    %ecx,%edx
   4ed46:	sub    $0x1,%edx
   4ed49:	mov    $0x0,%eax
   4ed4e:	cmovb  %eax,%edx
   4ed51:	testb  $0x1,0xc(%rsp)
   4ed56:	cmove  %ecx,%edx
   4ed59:	mov    %edx,0x18(%rsp)
   4ed5d:	cmp    $0x3b9aca01,%r13d
   4ed64:	je     4e400 <_ZN15funnel_patterns32pat_u128_cat_unrolled4__u64__w5117hc0066f52d5fc329cE+0x4e0>
   4ed6a:	mov    0x70(%rsp),%rax
   4ed6f:	mov    %rax,0x190(%rsp)
   4ed77:	mov    %r13d,0x198(%rsp)
   4ed7f:	mov    0x18(%r15),%eax
   4ed83:	cmp    $0x3b9aca01,%eax
   4ed88:	je     4eee4 <_ZN15funnel_patterns32pat_u128_cat_unrolled4__u64__w5117hc0066f52d5fc329cE+0xfc4>
   4ed8e:	mov    0x1c(%r15),%ecx
   4ed92:	mov    0x10(%r15),%rdx
   4ed96:	mov    %rdx,0x110(%rsp)
   4ed9e:	mov    %eax,0x118(%rsp)
   4eda5:	mov    %ecx,0x11c(%rsp)
   4edac:	mov    0x60(%rsp),%rdx
   4edb1:	lea    0x110(%rsp),%rdi
   4edb9:	lea    0x190(%rsp),%rsi
   4edc1:	call   *0xed551(%rip)        # 13c318 <_DYNAMIC+0x288>
   4edc7:	mov    %rdx,%rsi
   4edca:	mov    %rax,%rdx
   4edcd:	jmp    4e450 <_ZN15funnel_patterns32pat_u128_cat_unrolled4__u64__w5117hc0066f52d5fc329cE+0x530>
   4edd2:	mov    $0x10,%r15d
   4edd8:	xor    %r14d,%r14d
   4eddb:	mov    0xed506(%rip),%rbx        # 13c2e8 <_DYNAMIC+0x258>
   4ede2:	mov    0x8(%rbx),%eax
   4ede5:	test   %eax,%eax
   4ede7:	jne    4ee3c <_ZN15funnel_patterns32pat_u128_cat_unrolled4__u64__w5117hc0066f52d5fc329cE+0xf1c>
   4ede9:	mov    (%rbx),%rdi
   4edec:	call   *0xed53e(%rip)        # 13c330 <_DYNAMIC+0x2a0>
   4edf2:	mov    0xed53f(%rip),%rax        # 13c338 <_DYNAMIC+0x2a8>
   4edf9:	movb   $0x0,(%rax)
   4edfc:	test   %r14,%r14
   4edff:	je     4ee16 <_ZN15funnel_patterns32pat_u128_cat_unrolled4__u64__w5117hc0066f52d5fc329cE+0xef6>
   4ee01:	imul   $0xd0,%r14,%rsi
   4ee08:	mov    $0x10,%edx
   4ee0d:	mov    %r15,%rdi
   4ee10:	call   *0xed52a(%rip)        # 13c340 <_DYNAMIC+0x2b0>
   4ee16:	add    $0x3c18,%rsp
   4ee1d:	pop    %rbx
   4ee1e:	pop    %r12
   4ee20:	pop    %r13
   4ee22:	pop    %r14
   4ee24:	pop    %r15
   4ee26:	pop    %rbp
   4ee27:	ret
   4ee28:	mov    $0x1,%r14d
   4ee2e:	mov    0xed4b3(%rip),%rbx        # 13c2e8 <_DYNAMIC+0x258>
   4ee35:	mov    0x8(%rbx),%eax
   4ee38:	test   %eax,%eax
   4ee3a:	je     4ede9 <_ZN15funnel_patterns32pat_u128_cat_unrolled4__u64__w5117hc0066f52d5fc329cE+0xec9>
   4ee3c:	call   5152e <_ZN3std4sync9once_lock17OnceLock$LT$T$GT$10initialize17h13c8d01c1aad6442E>
   4ee41:	jmp    4ede9 <_ZN15funnel_patterns32pat_u128_cat_unrolled4__u64__w5117hc0066f52d5fc329cE+0xec9>
   4ee43:	call   5152e <_ZN3std4sync9once_lock17OnceLock$LT$T$GT$10initialize17h13c8d01c1aad6442E>
   4ee48:	jmp    4e3ab <_ZN15funnel_patterns32pat_u128_cat_unrolled4__u64__w5117hc0066f52d5fc329cE+0x48b>
   4ee4d:	lea    0xe4cd4(%rip),%rdi        # 133b28 <_ZN15funnel_patterns46__DIVAN_BENCH_PAT_U128_CAT_UNROLLED4__U64__W514PUSH17h68d55424f84a52b4E+0xe0>
   4ee54:	call   *0xed4ee(%rip)        # 13c348 <_DYNAMIC+0x2b8>
   4ee5a:	jmp    4ef00 <_ZN15funnel_patterns32pat_u128_cat_unrolled4__u64__w5117hc0066f52d5fc329cE+0xfe0>
   4ee5f:	movq   $0x0,0x108(%rsp)
   4ee6b:	lea    0x108(%rsp),%rax
   4ee73:	mov    %rax,0x1a0(%rsp)
   4ee7b:	mov    0xed4ce(%rip),%rax        # 13c350 <_DYNAMIC+0x2c0>
   4ee82:	mov    %rax,0x1a8(%rsp)
   4ee8a:	lea    0xe4c17(%rip),%rax        # 133aa8 <_ZN15funnel_patterns46__DIVAN_BENCH_PAT_U128_CAT_UNROLLED4__U64__W514PUSH17h68d55424f84a52b4E+0x60>
   4ee91:	mov    %rax,0x120(%rsp)
   4ee99:	movq   $0x2,0x128(%rsp)
   4eea5:	movq   $0x0,0x140(%rsp)
   4eeb1:	lea    0x1a0(%rsp),%rax
   4eeb9:	mov    %rax,0x130(%rsp)
   4eec1:	movq   $0x1,0x138(%rsp)
   4eecd:	lea    0xe4bf4(%rip),%rsi        # 133ac8 <_ZN15funnel_patterns46__DIVAN_BENCH_PAT_U128_CAT_UNROLLED4__U64__W514PUSH17h68d55424f84a52b4E+0x80>
   4eed4:	lea    0x120(%rsp),%rdi
   4eedc:	call   *0xed476(%rip)        # 13c358 <_DYNAMIC+0x2c8>
   4eee2:	jmp    4ef00 <_ZN15funnel_patterns32pat_u128_cat_unrolled4__u64__w5117hc0066f52d5fc329cE+0xfe0>
   4eee4:	lea    0xe4c0d(%rip),%rdi        # 133af8 <_ZN15funnel_patterns46__DIVAN_BENCH_PAT_U128_CAT_UNROLLED4__U64__W514PUSH17h68d55424f84a52b4E+0xb0>
   4eeeb:	call   *0xed46f(%rip)        # 13c360 <_DYNAMIC+0x2d0>
   4eef1:	jmp    4ef00 <_ZN15funnel_patterns32pat_u128_cat_unrolled4__u64__w5117hc0066f52d5fc329cE+0xfe0>
   4eef3:	lea    0xe4be6(%rip),%rdi        # 133ae0 <_ZN15funnel_patterns46__DIVAN_BENCH_PAT_U128_CAT_UNROLLED4__U64__W514PUSH17h68d55424f84a52b4E+0x98>
   4eefa:	call   *0xed448(%rip)        # 13c348 <_DYNAMIC+0x2b8>
   4ef00:	ud2
   4ef02:	mov    0x1a8(%rsp),%rdi
   4ef0a:	mov    0x1b0(%rsp),%rsi
   4ef12:	lea    0xe4b77(%rip),%rdx        # 133a90 <_ZN15funnel_patterns46__DIVAN_BENCH_PAT_U128_CAT_UNROLLED4__U64__W514PUSH17h68d55424f84a52b4E+0x48>
   4ef19:	call   *0xed449(%rip)        # 13c368 <_DYNAMIC+0x2d8>
   4ef1f:	mov    %r15,0x40(%rsp)
   4ef24:	mov    %rax,%rbx
   4ef27:	test   %r14,%r14
   4ef2a:	jne    4ef60 <_ZN15funnel_patterns32pat_u128_cat_unrolled4__u64__w5117hc0066f52d5fc329cE+0x1040>
   4ef2c:	jmp    4ef75 <_ZN15funnel_patterns32pat_u128_cat_unrolled4__u64__w5117hc0066f52d5fc329cE+0x1055>
   4ef2e:	mov    %rax,%rbx
   4ef31:	mov    0x120(%rsp),%rax
   4ef39:	lock decq (%rax)
   4ef3d:	jne    4ef60 <_ZN15funnel_patterns32pat_u128_cat_unrolled4__u64__w5117hc0066f52d5fc329cE+0x1040>
   4ef3f:	lea    0x120(%rsp),%rdi
   4ef47:	call   *0xed3c3(%rip)        # 13c310 <_DYNAMIC+0x280>
   4ef4d:	jmp    4ef60 <_ZN15funnel_patterns32pat_u128_cat_unrolled4__u64__w5117hc0066f52d5fc329cE+0x1040>
   4ef4f:	call   *0xed41b(%rip)        # 13c370 <_DYNAMIC+0x2e0>
   4ef55:	jmp    4ef5d <_ZN15funnel_patterns32pat_u128_cat_unrolled4__u64__w5117hc0066f52d5fc329cE+0x103d>
   4ef57:	jmp    4ef5d <_ZN15funnel_patterns32pat_u128_cat_unrolled4__u64__w5117hc0066f52d5fc329cE+0x103d>
   4ef59:	jmp    4ef5d <_ZN15funnel_patterns32pat_u128_cat_unrolled4__u64__w5117hc0066f52d5fc329cE+0x103d>
   4ef5b:	jmp    4ef5d <_ZN15funnel_patterns32pat_u128_cat_unrolled4__u64__w5117hc0066f52d5fc329cE+0x103d>
   4ef5d:	mov    %rax,%rbx
   4ef60:	mov    $0xd0,%esi
   4ef65:	mov    $0x10,%edx
   4ef6a:	mov    0x40(%rsp),%rdi
   4ef6f:	call   *0xed3cb(%rip)        # 13c340 <_DYNAMIC+0x2b0>
   4ef75:	mov    %rbx,%rdi
   4ef78:	call   1328b0 <_Unwind_Resume@plt>

Disassembly of section .init:

Disassembly of section .fini:

Disassembly of section .plt:
