
/home/user/vortex/target/release/deps/funnel_patterns-21c1c00107f42b8a:     file format elf64-x86-64


Disassembly of section .text:

000000000004cec0 <_ZN15funnel_patterns31pat_combine_then_mask__u64__w5117h51620081b03c5800E>:
   4cec0:	push   %rbp
   4cec1:	push   %r15
   4cec3:	push   %r14
   4cec5:	push   %r13
   4cec7:	push   %r12
   4cec9:	push   %rbx
   4ceca:	sub    $0x1000,%rsp
   4ced1:	movq   $0x0,(%rsp)
   4ced9:	sub    $0x1000,%rsp
   4cee0:	movq   $0x0,(%rsp)
   4cee8:	sub    $0x1000,%rsp
   4ceef:	movq   $0x0,(%rsp)
   4cef7:	sub    $0xc18,%rsp
   4cefe:	mov    %rdi,%r12
   4cf01:	lea    0x298(%rsp),%rdi
   4cf09:	mov    $0x1980,%edx
   4cf0e:	xor    %esi,%esi
   4cf10:	call   *0xef3aa(%rip)        # 13c2c0 <memset@GLIBC_2.2.5>
   4cf16:	vmovdqa64 -0x3cca0(%rip),%zmm0        # 10280 <__abi_tag+0xff84>
   4cf20:	mov    $0x38,%eax
   4cf25:	vpbroadcastq -0x3c79f(%rip),%zmm1        # 10790 <__abi_tag+0x10494>
   4cf2f:	vpbroadcastq -0x3c8c9(%rip),%zmm2        # 10670 <__abi_tag+0x10374>
   4cf39:	vpbroadcastq -0x3c7ab(%rip),%zmm3        # 10798 <__abi_tag+0x1049c>
   4cf43:	vpbroadcastq -0x3c72d(%rip),%zmm4        # 10820 <__abi_tag+0x10524>
   4cf4d:	vpbroadcastq -0x3c707(%rip),%zmm5        # 10850 <__abi_tag+0x10554>
   4cf57:	vpbroadcastq -0x3c8d1(%rip),%zmm6        # 10690 <__abi_tag+0x10394>
   4cf61:	vpbroadcastq -0x3c6e3(%rip),%zmm7        # 10888 <__abi_tag+0x1058c>
   4cf6b:	vpbroadcastq -0x3c71d(%rip),%zmm8        # 10858 <__abi_tag+0x1055c>
   4cf75:	vpbroadcastq -0x3c6ff(%rip),%zmm9        # 10880 <__abi_tag+0x10584>
   4cf7f:	nop
   4cf80:	vpmullq %zmm1,%zmm0,%zmm10
   4cf86:	vpaddq %zmm2,%zmm10,%zmm11
   4cf8c:	vpaddq %zmm3,%zmm10,%zmm12
   4cf92:	vmovdqu64 %zmm10,0xd8(%rsp,%rax,8)
   4cf9d:	vmovdqu64 %zmm11,0x118(%rsp,%rax,8)
   4cfa8:	vpaddq %zmm4,%zmm10,%zmm11
   4cfae:	vmovdqu64 %zmm12,0x158(%rsp,%rax,8)
   4cfb9:	vmovdqu64 %zmm11,0x198(%rsp,%rax,8)
   4cfc4:	cmp    $0x338,%rax
   4cfca:	je     4d01f <_ZN15funnel_patterns31pat_combine_then_mask__u64__w5117h51620081b03c5800E+0x15f>
   4cfcc:	vpaddq %zmm5,%zmm10,%zmm11
   4cfd2:	vpaddq %zmm6,%zmm10,%zmm12
   4cfd8:	vpaddq %zmm7,%zmm10,%zmm13
   4cfde:	vpaddq %zmm8,%zmm10,%zmm10
   4cfe4:	vmovdqu64 %zmm11,0x1d8(%rsp,%rax,8)
   4cfef:	vmovdqu64 %zmm12,0x218(%rsp,%rax,8)
   4cffa:	vmovdqu64 %zmm13,0x258(%rsp,%rax,8)
   4d005:	vmovdqu64 %zmm10,0x298(%rsp,%rax,8)
   4d010:	vpaddq %zmm9,%zmm0,%zmm0
   4d016:	add    $0x40,%rax
   4d01a:	jmp    4cf80 <_ZN15funnel_patterns31pat_combine_then_mask__u64__w5117h51620081b03c5800E+0xc0>
   4d01f:	vmovaps -0x3cd69(%rip),%zmm0        # 102c0 <__abi_tag+0xffc4>
   4d029:	vmovups %zmm0,0x1b98(%rsp)
   4d034:	vmovdqa64 -0x3cd3e(%rip),%zmm0        # 10300 <__abi_tag+0x10004>
   4d03e:	vmovdqu64 %zmm0,0x1bd8(%rsp)
   4d049:	lea    0x2298(%rsp),%rbx
   4d051:	lea    0x298(%rsp),%r14
   4d059:	mov    $0x1980,%edx
   4d05e:	mov    %rbx,%rdi
   4d061:	mov    %r14,%rsi
   4d064:	vzeroupper
   4d067:	call   *0xef25b(%rip)        # 13c2c8 <memcpy@GLIBC_2.14>
   4d06d:	movq   $0x3b9aca07,0xf0(%rsp)
   4d079:	xor    %ebp,%ebp
   4d07b:	mov    $0x2000,%edx
   4d080:	mov    %r14,%rdi
   4d083:	xor    %esi,%esi
   4d085:	call   *0xef235(%rip)        # 13c2c0 <memset@GLIBC_2.2.5>
   4d08b:	mov    %rbx,0x208(%rsp)
   4d093:	lea    0xf0(%rsp),%rax
   4d09b:	mov    %rax,0x210(%rsp)
   4d0a3:	mov    %r14,0x218(%rsp)
   4d0ab:	lea    0x208(%rsp),%rax
   4d0b3:	mov    %rax,0xf8(%rsp)
   4d0bb:	lea    0xf8(%rsp),%rax
   4d0c3:	mov    %rax,0x100(%rsp)
   4d0cb:	movq   $0x1,0x100(%r12)
   4d0d7:	movb   $0x1,0x108(%r12)
   4d0e0:	mov    0xf0(%r12),%rdx
   4d0e8:	mov    0xf8(%r12),%rax
   4d0f0:	movzbl 0x8(%rdx),%ecx
   4d0f4:	mov    %cl,0x3(%rsp)
   4d0f8:	cmp    $0x1,%cl
   4d0fb:	jne    4d103 <_ZN15funnel_patterns31pat_combine_then_mask__u64__w5117h51620081b03c5800E+0x243>
   4d0fd:	xor    %esi,%esi
   4d0ff:	xor    %ecx,%ecx
   4d101:	jmp    4d12d <_ZN15funnel_patterns31pat_combine_then_mask__u64__w5117h51620081b03c5800E+0x26d>
   4d103:	cmpb   $0x0,0x60(%rax)
   4d107:	je     4d11c <_ZN15funnel_patterns31pat_combine_then_mask__u64__w5117h51620081b03c5800E+0x25c>
   4d109:	mov    0x64(%rax),%ecx
   4d10c:	mov    %ecx,0x8(%rsp)
   4d110:	mov    $0x2,%ebp
   4d115:	mov    $0x1,%sil
   4d118:	xor    %ecx,%ecx
   4d11a:	jmp    4d12d <_ZN15funnel_patterns31pat_combine_then_mask__u64__w5117h51620081b03c5800E+0x26d>
   4d11c:	mov    $0x1,%cl
   4d11e:	movl   $0x1,0x8(%rsp)
   4d126:	xor    %esi,%esi
   4d128:	mov    $0x1,%ebp
   4d12d:	mov    (%rdx),%r14
   4d130:	test   %r14,%r14
   4d133:	lea    0x27(%rsp),%rdx
   4d138:	mov    %rdx,0x220(%rsp)
   4d140:	setne  0x238(%rsp)
   4d148:	lea    0x100(%rsp),%rdi
   4d150:	mov    %rdi,0x228(%rsp)
   4d158:	mov    %rdx,0x230(%rsp)
   4d160:	mov    0x70(%rax),%edi
   4d163:	movq   $0x0,0x88(%rsp)
   4d16f:	mov    $0x0,%edx
   4d174:	mov    %rdx,0x80(%rsp)
   4d17c:	cmp    $0x3b9aca00,%edi
   4d182:	je     4d1bd <_ZN15funnel_patterns31pat_combine_then_mask__u64__w5117h51620081b03c5800E+0x2fd>
   4d184:	mov    $0x3b9aca00,%edx
   4d189:	mulx   0x68(%rax),%r8,%r9
   4d18f:	mov    %edi,%edx
   4d191:	add    %r8,%rdx
   4d194:	adc    $0x0,%r9
   4d198:	imul   $0x3e8,%r9,%rdi
   4d19f:	mov    $0x3e8,%r8d
   4d1a5:	mulx   %r8,%rdx,%r8
   4d1aa:	mov    %rdx,0x88(%rsp)
   4d1b2:	add    %rdi,%r8
   4d1b5:	mov    %r8,0x80(%rsp)
   4d1bd:	movq   $0xffffffffffffffff,0x48(%rsp)
   4d1c6:	mov    0x80(%rax),%r8d
   4d1cd:	movq   $0xffffffffffffffff,0x38(%rsp)
   4d1d6:	cmp    $0x3b9aca00,%r8d
   4d1dd:	je     4d21e <_ZN15funnel_patterns31pat_combine_then_mask__u64__w5117h51620081b03c5800E+0x35e>
   4d1df:	mov    $0x3b9aca00,%edx
   4d1e4:	mulx   0x78(%rax),%r9,%rdi
   4d1ea:	mov    %r8d,%edx
   4d1ed:	add    %r9,%rdx
   4d1f0:	adc    $0x0,%rdi
   4d1f4:	mov    $0x3e8,%r8d
   4d1fa:	mulx   %r8,%r8,%r9
   4d1ff:	mov    %r9,0x38(%rsp)
   4d204:	mov    %r8,0x48(%rsp)
   4d209:	or     %rdi,%rdx
   4d20c:	je     4ddb6 <_ZN15funnel_patterns31pat_combine_then_mask__u64__w5117h51620081b03c5800E+0xef6>
   4d212:	imul   $0x3e8,%rdi,%rdx
   4d219:	add    %rdx,0x38(%rsp)
   4d21e:	mov    0x58(%rax),%edx
   4d221:	cmp    $0x1,%edx
   4d224:	jne    4d230 <_ZN15funnel_patterns31pat_combine_then_mask__u64__w5117h51620081b03c5800E+0x370>
   4d226:	cmpl   $0x0,0x5c(%rax)
   4d22a:	je     4ddb6 <_ZN15funnel_patterns31pat_combine_then_mask__u64__w5117h51620081b03c5800E+0xef6>
   4d230:	cmpl   $0x1,0x60(%rax)
   4d234:	jne    4d240 <_ZN15funnel_patterns31pat_combine_then_mask__u64__w5117h51620081b03c5800E+0x380>
   4d236:	cmpl   $0x0,0x64(%rax)
   4d23a:	je     4ddb6 <_ZN15funnel_patterns31pat_combine_then_mask__u64__w5117h51620081b03c5800E+0xef6>
   4d240:	mov    %r14,0x60(%rsp)
   4d245:	test   %dl,%sil
   4d248:	je     4d25b <_ZN15funnel_patterns31pat_combine_then_mask__u64__w5117h51620081b03c5800E+0x39b>
   4d24a:	mov    0x5c(%rax),%edx
   4d24d:	mov    %edx,0x18(%rsp)
   4d251:	movl   $0x1,0xc(%rsp)
   4d259:	jmp    4d26b <_ZN15funnel_patterns31pat_combine_then_mask__u64__w5117h51620081b03c5800E+0x3ab>
   4d25b:	movzbl %sil,%edx
   4d25f:	mov    %edx,0xc(%rsp)
   4d263:	movl   $0x64,0x18(%rsp)
   4d26b:	movq   $0x0,0x58(%rsp)
   4d274:	mov    $0x0,%edx
   4d279:	mov    %rdx,0x50(%rsp)
   4d27e:	test   %cl,%cl
   4d280:	je     4d29d <_ZN15funnel_patterns31pat_combine_then_mask__u64__w5117h51620081b03c5800E+0x3dd>
   4d282:	mov    %r14,%rdi
   4d285:	call   *0xef045(%rip)        # 13c2d0 <_DYNAMIC+0x240>
   4d28b:	mov    %rax,0x58(%rsp)
   4d290:	mov    %rdx,0x50(%rsp)
   4d295:	mov    0xf8(%r12),%rax
   4d29d:	cmpb   $0x1,0x3(%rsp)
   4d2a2:	je     4d2c3 <_ZN15funnel_patterns31pat_combine_then_mask__u64__w5117h51620081b03c5800E+0x403>
   4d2a4:	mov    $0x1,%edx
   4d2a9:	cmpb   $0x0,0x58(%rax)
   4d2ad:	je     4d2b2 <_ZN15funnel_patterns31pat_combine_then_mask__u64__w5117h51620081b03c5800E+0x3f2>
   4d2af:	mov    0x5c(%rax),%edx
   4d2b2:	mov    (%r12),%rcx
   4d2b6:	mov    0x10(%r12),%rsi
   4d2bb:	sub    %rsi,%rcx
   4d2be:	cmp    %rcx,%rdx
   4d2c1:	ja     4d308 <_ZN15funnel_patterns31pat_combine_then_mask__u64__w5117h51620081b03c5800E+0x448>
   4d2c3:	testb  $0x1,0x88(%rax)
   4d2ca:	jne    4d321 <_ZN15funnel_patterns31pat_combine_then_mask__u64__w5117h51620081b03c5800E+0x461>
   4d2cc:	lock orl $0x0,-0x40(%rsp)
   4d2d2:	test   %r14,%r14
   4d2d5:	je     4d2f3 <_ZN15funnel_patterns31pat_combine_then_mask__u64__w5117h51620081b03c5800E+0x433>
   4d2d7:	lfence
   4d2da:	rdtsc
   4d2dc:	shl    $0x20,%rdx
   4d2e0:	or     %rax,%rdx
   4d2e3:	mov    %rdx,0x70(%rsp)
   4d2e8:	lfence
   4d2eb:	mov    $0x3b9aca00,%r13d
   4d2f1:	jmp    4d301 <_ZN15funnel_patterns31pat_combine_then_mask__u64__w5117h51620081b03c5800E+0x441>
   4d2f3:	call   *0xeefdf(%rip)        # 13c2d8 <_DYNAMIC+0x248>
   4d2f9:	mov    %rax,0x70(%rsp)
   4d2fe:	mov    %edx,%r13d
   4d301:	mov    0x60(%rsp),%r14
   4d306:	jmp    4d327 <_ZN15funnel_patterns31pat_combine_then_mask__u64__w5117h51620081b03c5800E+0x467>
   4d308:	mov    %r12,%rdi
   4d30b:	call   51800 <_ZN5alloc7raw_vec20RawVecInner$LT$A$GT$7reserve21do_reserve_and_handle17h46771c9d08372974E>
   4d310:	mov    0xf8(%r12),%rax
   4d318:	testb  $0x1,0x88(%rax)
   4d31f:	je     4d2cc <_ZN15funnel_patterns31pat_combine_then_mask__u64__w5117h51620081b03c5800E+0x40c>
   4d321:	mov    $0x3b9aca01,%r13d
   4d327:	mov    %r14,%rdi
   4d32a:	call   *0xeefb0(%rip)        # 13c2e0 <_DYNAMIC+0x250>
   4d330:	mov    %rax,0xa8(%rsp)
   4d338:	mov    0xeefa9(%rip),%r14        # 13c2e8 <_DYNAMIC+0x258>
   4d33f:	mov    0x8(%r14),%eax
   4d343:	test   %eax,%eax
   4d345:	jne    4dde3 <_ZN15funnel_patterns31pat_combine_then_mask__u64__w5117h51620081b03c5800E+0xf23>
   4d34b:	mov    (%r14),%rdi
   4d34e:	call   *0xeef9c(%rip)        # 13c2f0 <_DYNAMIC+0x260>
   4d354:	mov    0x48(%rsp),%rax
   4d359:	or     0x38(%rsp),%rax
   4d35e:	je     4dd72 <_ZN15funnel_patterns31pat_combine_then_mask__u64__w5117h51620081b03c5800E+0xeb2>
   4d364:	lea    0x50(%r12),%rax
   4d369:	mov    %rax,0x28(%rsp)
   4d36e:	lea    0x18(%r12),%rax
   4d373:	mov    %rax,0x78(%rsp)
   4d378:	mov    $0x10,%r15d
   4d37e:	mov    $0x1,%al
   4d380:	xor    %edx,%edx
   4d382:	xor    %esi,%esi
   4d384:	xor    %r14d,%r14d
   4d387:	mov    %r12,0x30(%rsp)
   4d38c:	mov    %r13d,0x6c(%rsp)
   4d391:	jmp    4d40b <_ZN15funnel_patterns31pat_combine_then_mask__u64__w5117h51620081b03c5800E+0x54b>
   4d393:	data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
   4d3a0:	mov    0x90(%rsp),%rdi
   4d3a8:	cmp    $0x3e9,%rdi
   4d3af:	mov    $0x3e8,%eax
   4d3b4:	cmovae %rdi,%rax
   4d3b8:	mov    0x98(%rsp),%rsi
   4d3c0:	test   %rsi,%rsi
   4d3c3:	mov    $0x3e8,%ecx
   4d3c8:	cmove  %rcx,%rdi
   4d3cc:	cmove  %rax,%rdi
   4d3d0:	mov    0xb0(%rsp),%rdx
   4d3d8:	add    %rdi,%rdx
   4d3db:	adc    %rsi,%r14
   4d3de:	mov    $0xffffffffffffffff,%rax
   4d3e5:	cmovb  %rax,%r14
   4d3e9:	cmovb  %rax,%rdx
   4d3ed:	mov    %r14,%rsi
   4d3f0:	mov    $0x1,%r14d
   4d3f6:	xor    %eax,%eax
   4d3f8:	cmp    0x48(%rsp),%rdx
   4d3fd:	mov    %rsi,%rcx
   4d400:	sbb    0x38(%rsp),%rcx
   4d405:	jae    4dd7b <_ZN15funnel_patterns31pat_combine_then_mask__u64__w5117h51620081b03c5800E+0xebb>
   4d40b:	cmp    0x88(%rsp),%rdx
   4d413:	mov    %rsi,0xe8(%rsp)
   4d41b:	mov    %rsi,%rcx
   4d41e:	sbb    0x80(%rsp),%rcx
   4d426:	jb     4d43a <_ZN15funnel_patterns31pat_combine_then_mask__u64__w5117h51620081b03c5800E+0x57a>
   4d428:	testb  $0x1,0xc(%rsp)
   4d42d:	je     4d43a <_ZN15funnel_patterns31pat_combine_then_mask__u64__w5117h51620081b03c5800E+0x57a>
   4d42f:	cmpl   $0x0,0x18(%rsp)
   4d434:	je     4dd7b <_ZN15funnel_patterns31pat_combine_then_mask__u64__w5117h51620081b03c5800E+0xebb>
   4d43a:	mov    %rdx,0xb0(%rsp)
   4d442:	test   %ebp,%ebp
   4d444:	mov    0x8(%rsp),%ecx
   4d448:	mov    $0x1,%edx
   4d44d:	cmove  %edx,%ecx
   4d450:	mov    %ecx,0x4(%rsp)
   4d454:	mov    %ecx,0x48(%r12)
   4d459:	movq   $0x0,0x268(%rsp)
   4d465:	mov    0x28(%rsp),%rcx
   4d46a:	mov    %rcx,0x240(%rsp)
   4d472:	lea    0x220(%rsp),%rcx
   4d47a:	mov    %rcx,0x248(%rsp)
   4d482:	lea    0x4(%rsp),%rcx
   4d487:	mov    %rcx,0x250(%rsp)
   4d48f:	lea    0x268(%rsp),%rcx
   4d497:	mov    %rcx,0x258(%rsp)
   4d49f:	lea    0x60(%rsp),%rcx
   4d4a4:	mov    %rcx,0x260(%rsp)
   4d4ac:	test   $0x1,%al
   4d4ae:	jne    4d620 <_ZN15funnel_patterns31pat_combine_then_mask__u64__w5117h51620081b03c5800E+0x760>
   4d4b4:	imul   $0xd0,%r14,%rax
   4d4bb:	add    %r15,%rax
   4d4be:	mov    %rax,%rdx
   4d4c1:	sub    %r15,%rdx
   4d4c4:	add    $0xffffffffffffff30,%rdx
   4d4cb:	movabs $0x4ec4ec4ec4ec4ec5,%rcx
   4d4d5:	mulx   %rcx,%rcx,%rcx
   4d4da:	mov    %r15,%rsi
   4d4dd:	cmp    $0xc30,%rdx
   4d4e4:	jb     4d670 <_ZN15funnel_patterns31pat_combine_then_mask__u64__w5117h51620081b03c5800E+0x7b0>
   4d4ea:	shr    $0x6,%rcx
   4d4ee:	inc    %rcx
   4d4f1:	cmp    $0x3330,%rdx
   4d4f8:	jae    4d510 <_ZN15funnel_patterns31pat_combine_then_mask__u64__w5117h51620081b03c5800E+0x650>
   4d4fa:	xor    %edx,%edx
   4d4fc:	mov    %r15,%rdi
   4d4ff:	jmp    4d5bf <_ZN15funnel_patterns31pat_combine_then_mask__u64__w5117h51620081b03c5800E+0x6ff>
   4d504:	data16 data16 cs nopw 0x0(%rax,%rax,1)
   4d510:	mov    %rcx,%rdx
   4d513:	movabs $0x3ffffffffffffc0,%rsi
   4d51d:	and    %rsi,%rdx
   4d520:	imul   $0xd0,%rdx,%rsi
   4d527:	lea    (%r15,%rsi,1),%rdi
   4d52b:	mov    %rdx,%r8
   4d52e:	mov    %r15,%r9
   4d531:	vpbroadcastd -0x3cc67(%rip),%zmm0        # 108d4 <__abi_tag+0x105d8>
   4d53b:	vmovdqa64 -0x3d205(%rip),%zmm1        # 10340 <__abi_tag+0x10044>
   4d545:	vmovdqa64 -0x3d1cf(%rip),%zmm2        # 10380 <__abi_tag+0x10084>
   4d54f:	vmovdqa64 -0x3d199(%rip),%zmm3        # 103c0 <__abi_tag+0x100c4>
   4d559:	vmovdqa64 -0x3d163(%rip),%zmm4        # 10400 <__abi_tag+0x10104>
   4d563:	data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
   4d570:	kxnorw %k0,%k0,%k1
   4d574:	vpscatterdd %zmm0,0x8(%r9,%zmm1,1){%k1}
   4d57c:	kxnorw %k0,%k0,%k1
   4d580:	vpscatterdd %zmm0,0x8(%r9,%zmm2,1){%k1}
   4d588:	kxnorw %k0,%k0,%k1
   4d58c:	vpscatterdd %zmm0,0x8(%r9,%zmm3,1){%k1}
   4d594:	kxnorw %k0,%k0,%k1
   4d598:	vpscatterdd %zmm0,0x8(%r9,%zmm4,1){%k1}
   4d5a0:	add    $0x3400,%r9
   4d5a7:	add    $0xffffffffffffffc0,%r8
   4d5ab:	jne    4d570 <_ZN15funnel_patterns31pat_combine_then_mask__u64__w5117h51620081b03c5800E+0x6b0>
   4d5ad:	cmp    %rdx,%rcx
   4d5b0:	je     4d683 <_ZN15funnel_patterns31pat_combine_then_mask__u64__w5117h51620081b03c5800E+0x7c3>
   4d5b6:	test   $0x30,%cl
   4d5b9:	je     4d66d <_ZN15funnel_patterns31pat_combine_then_mask__u64__w5117h51620081b03c5800E+0x7ad>
   4d5bf:	movabs $0x3ffffffffffffc0,%rsi
   4d5c9:	lea    0x30(%rsi),%r8
   4d5cd:	and    %rcx,%r8
   4d5d0:	imul   $0xd0,%r8,%rsi
   4d5d7:	add    %r15,%rsi
   4d5da:	sub    %r8,%rdx
   4d5dd:	vpbroadcastd -0x3cd13(%rip),%zmm0        # 108d4 <__abi_tag+0x105d8>
   4d5e7:	vmovdqa64 -0x3d2b1(%rip),%zmm1        # 10340 <__abi_tag+0x10044>
   4d5f1:	data16 data16 data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
   4d600:	kxnorw %k0,%k0,%k1
   4d604:	vpscatterdd %zmm0,0x8(%rdi,%zmm1,1){%k1}
   4d60c:	add    $0xd00,%rdi
   4d613:	add    $0x10,%rdx
   4d617:	jne    4d600 <_ZN15funnel_patterns31pat_combine_then_mask__u64__w5117h51620081b03c5800E+0x740>
   4d619:	cmp    %r8,%rcx
   4d61c:	jne    4d670 <_ZN15funnel_patterns31pat_combine_then_mask__u64__w5117h51620081b03c5800E+0x7b0>
   4d61e:	jmp    4d683 <_ZN15funnel_patterns31pat_combine_then_mask__u64__w5117h51620081b03c5800E+0x7c3>
   4d620:	movq   $0x0,0x128(%rsp)
   4d62c:	mov    $0x10,%esi
   4d631:	mov    $0xd0,%edx
   4d636:	lea    0x1a0(%rsp),%rdi
   4d63e:	lea    0x120(%rsp),%rcx
   4d646:	call   516c0 <_ZN5alloc7raw_vec11finish_grow17hedc133b40cb748a9E>
   4d64b:	cmpb   $0x0,0x1a0(%rsp)
   4d653:	jne    4dea2 <_ZN15funnel_patterns31pat_combine_then_mask__u64__w5117h51620081b03c5800E+0xfe2>
   4d659:	mov    0x1a8(%rsp),%r15
   4d661:	lea    0xd0(%r15),%rax
   4d668:	jmp    4d4be <_ZN15funnel_patterns31pat_combine_then_mask__u64__w5117h51620081b03c5800E+0x5fe>
   4d66d:	add    %r15,%rsi
   4d670:	movl   $0x3b9aca01,0x8(%rsi)
   4d677:	add    $0xd0,%rsi
   4d67e:	cmp    %rax,%rsi
   4d681:	jne    4d670 <_ZN15funnel_patterns31pat_combine_then_mask__u64__w5117h51620081b03c5800E+0x7b0>
   4d683:	mov    %r12,%rbx
   4d686:	mov    %r15,0x40(%rsp)
   4d68b:	vzeroupper
   4d68e:	call   *0xeec64(%rip)        # 13c2f8 <_DYNAMIC+0x268>
   4d694:	mov    %rax,0x120(%rsp)
   4d69c:	movq   $0x0,0x128(%rsp)
   4d6a8:	lea    0x1a21(%rip),%rax        # 4f0d0 <_ZN30codspeed_divan_compat_walltime11thread_pool19TaskShared$LT$F$GT$3new4call17h9fadf0f50d13f6c9E>
   4d6af:	mov    %rax,0x130(%rsp)
   4d6b7:	lea    0x240(%rsp),%rax
   4d6bf:	mov    %rax,0x138(%rsp)
   4d6c7:	mov    %r15,0x140(%rsp)
   4d6cf:	mov    0xeec2a(%rip),%rdi        # 13c300 <_DYNAMIC+0x270>
   4d6d6:	xor    %esi,%esi
   4d6d8:	lea    0x120(%rsp),%rdx
   4d6e0:	call   *0xeec22(%rip)        # 13c308 <_DYNAMIC+0x278>
   4d6e6:	mov    0x120(%rsp),%rax
   4d6ee:	lock decq (%rax)
   4d6f2:	jne    4d702 <_ZN15funnel_patterns31pat_combine_then_mask__u64__w5117h51620081b03c5800E+0x842>
   4d6f4:	lea    0x120(%rsp),%rdi
   4d6fc:	call   *0xeec0e(%rip)        # 13c310 <_DYNAMIC+0x280>
   4d702:	cmpl   $0x3b9aca01,0x8(%r15)
   4d70a:	je     4ddff <_ZN15funnel_patterns31pat_combine_then_mask__u64__w5117h51620081b03c5800E+0xf3f>
   4d710:	cmpb   $0x1,0x3(%rsp)
   4d715:	je     4ddc8 <_ZN15funnel_patterns31pat_combine_then_mask__u64__w5117h51620081b03c5800E+0xf08>
   4d71b:	vmovups 0x10(%r15),%xmm0
   4d721:	vmovaps %xmm0,0x1a0(%rsp)
   4d72a:	vmovdqu (%r15),%xmm0
   4d72f:	vmovdqa %xmm0,0x120(%rsp)
   4d738:	mov    0xc0(%r15),%rdx
   4d73f:	lea    0x1a0(%rsp),%rdi
   4d747:	lea    0x120(%rsp),%rsi
   4d74f:	call   *0xeebc3(%rip)        # 13c318 <_DYNAMIC+0x288>
   4d755:	lea    0x10(%r15),%rax
   4d759:	mov    %rax,%r12
   4d75c:	vmovups (%rax),%xmm0
   4d760:	vmovaps %xmm0,0x1a0(%rsp)
   4d769:	vmovdqu (%r15),%xmm0
   4d76e:	vmovdqa %xmm0,0x120(%rsp)
   4d777:	mov    0xc0(%r15),%rdx
   4d77e:	lea    0x1a0(%rsp),%rdi
   4d786:	lea    0x120(%rsp),%rsi
   4d78e:	call   *0xeeb84(%rip)        # 13c318 <_DYNAMIC+0x288>
   4d794:	mov    %rax,%rsi
   4d797:	cmp    $0x1,%ebp
   4d79a:	mov    %rdx,0x98(%rsp)
   4d7a2:	mov    %rax,0x90(%rsp)
   4d7aa:	jne    4d820 <_ZN15funnel_patterns31pat_combine_then_mask__u64__w5117h51620081b03c5800E+0x960>
   4d7ac:	mov    %rbx,%rcx
   4d7af:	movq   $0x0,0x10(%rbx)
   4d7b7:	cmpq   $0x0,0x30(%rbx)
   4d7bc:	mov    0x58(%rsp),%r13
   4d7c1:	mov    0x50(%rsp),%rbp
   4d7c6:	je     4d846 <_ZN15funnel_patterns31pat_combine_then_mask__u64__w5117h51620081b03c5800E+0x986>
   4d7c8:	mov    0x20(%rcx),%r14
   4d7cc:	test   %r14,%r14
   4d7cf:	je     4d835 <_ZN15funnel_patterns31pat_combine_then_mask__u64__w5117h51620081b03c5800E+0x975>
   4d7d1:	mov    0x78(%rsp),%rax
   4d7d6:	mov    (%rax),%rdi
   4d7d9:	lea    0x11(%r14),%rdx
   4d7dd:	mov    $0xff,%esi
   4d7e2:	call   *0xeead8(%rip)        # 13c2c0 <memset@GLIBC_2.2.5>
   4d7e8:	mov    0x90(%rsp),%rsi
   4d7f0:	mov    0x98(%rsp),%rdx
   4d7f8:	lea    0x1(%r14),%rax
   4d7fc:	mov    %rax,%rcx
   4d7ff:	shr    $0x3,%rcx
   4d803:	and    $0xfffffffffffffff8,%rax
   4d807:	sub    %rcx,%rax
   4d80a:	cmp    $0x8,%r14
   4d80e:	cmovb  %r14,%rax
   4d812:	jmp    4d837 <_ZN15funnel_patterns31pat_combine_then_mask__u64__w5117h51620081b03c5800E+0x977>
   4d814:	data16 data16 cs nopw 0x0(%rax,%rax,1)
   4d820:	mov    %ebp,0x14(%rsp)
   4d824:	mov    0x4(%rsp),%eax
   4d828:	mov    %rax,0xa0(%rsp)
   4d830:	jmp    4d8d7 <_ZN15funnel_patterns31pat_combine_then_mask__u64__w5117h51620081b03c5800E+0xa17>
   4d835:	xor    %eax,%eax
   4d837:	mov    %rbx,%rcx
   4d83a:	movq   $0x0,0x30(%rbx)
   4d842:	mov    %rax,0x28(%rbx)
   4d846:	cmpq   $0x0,0x68(%rcx)
   4d84b:	je     4d855 <_ZN15funnel_patterns31pat_combine_then_mask__u64__w5117h51620081b03c5800E+0x995>
   4d84d:	movq   $0x0,0x60(%rcx)
   4d855:	cmpq   $0x0,0x90(%rcx)
   4d85d:	je     4d86a <_ZN15funnel_patterns31pat_combine_then_mask__u64__w5117h51620081b03c5800E+0x9aa>
   4d85f:	movq   $0x0,0x88(%rcx)
   4d86a:	cmpq   $0x0,0xb8(%rcx)
   4d872:	je     4d87f <_ZN15funnel_patterns31pat_combine_then_mask__u64__w5117h51620081b03c5800E+0x9bf>
   4d874:	movq   $0x0,0xb0(%rcx)
   4d87f:	cmpq   $0x0,0xe0(%rcx)
   4d887:	je     4d894 <_ZN15funnel_patterns31pat_combine_then_mask__u64__w5117h51620081b03c5800E+0x9d4>
   4d889:	movq   $0x0,0xd8(%rcx)
   4d894:	mov    %r13,%rax
   4d897:	or     %rbp,%rax
   4d89a:	je     4de93 <_ZN15funnel_patterns31pat_combine_then_mask__u64__w5117h51620081b03c5800E+0xfd3>
   4d8a0:	mov    %rsi,%rdi
   4d8a3:	mov    %rdx,%rsi
   4d8a6:	mov    %r13,%rdx
   4d8a9:	mov    %rbp,%rcx
   4d8ac:	call   *0xeea6e(%rip)        # 13c320 <_DYNAMIC+0x290>
   4d8b2:	cmp    $0x65,%rax
   4d8b6:	sbb    $0x0,%rdx
   4d8ba:	mov    0x4(%rsp),%ecx
   4d8be:	mov    %rcx,0xa0(%rsp)
   4d8c6:	jae    4d8f0 <_ZN15funnel_patterns31pat_combine_then_mask__u64__w5117h51620081b03c5800E+0xa30>
   4d8c8:	lea    (%rcx,%rcx,1),%eax
   4d8cb:	mov    %eax,0x8(%rsp)
   4d8cf:	movl   $0x1,0x14(%rsp)
   4d8d7:	mov    0x18(%rsp),%eax
   4d8db:	mov    %eax,0x10(%rsp)
   4d8df:	mov    %r12,%rdx
   4d8e2:	jmp    4d92d <_ZN15funnel_patterns31pat_combine_then_mask__u64__w5117h51620081b03c5800E+0xa6d>
   4d8e4:	data16 data16 cs nopw 0x0(%rax,%rax,1)
   4d8f0:	mov    0xf8(%rbx),%rax
   4d8f7:	movl   $0x1,0xc(%rsp)
   4d8ff:	cmpb   $0x0,0x58(%rax)
   4d903:	mov    %r12,%rdx
   4d906:	je     4d919 <_ZN15funnel_patterns31pat_combine_then_mask__u64__w5117h51620081b03c5800E+0xa59>
   4d908:	movl   $0x2,0x14(%rsp)
   4d910:	mov    0x5c(%rax),%eax
   4d913:	mov    %eax,0x10(%rsp)
   4d917:	jmp    4d929 <_ZN15funnel_patterns31pat_combine_then_mask__u64__w5117h51620081b03c5800E+0xa69>
   4d919:	movl   $0x64,0x10(%rsp)
   4d921:	movl   $0x2,0x14(%rsp)
   4d929:	mov    %ecx,0x8(%rsp)
   4d92d:	mov    0xa8(%rsp),%rax
   4d935:	mov    (%rax),%rbp
   4d938:	mov    0x8(%rax),%r13
   4d93c:	mov    0x18(%rax),%r14
   4d940:	mov    0x10(%rax),%rcx
   4d944:	mov    %rcx,0xc0(%rsp)
   4d94c:	mov    %r15,%rcx
   4d94f:	mov    0x40(%r15),%r15
   4d953:	mov    0x28(%rax),%rsi
   4d957:	mov    %rsi,0xc8(%rsp)
   4d95f:	mov    0x20(%rax),%rsi
   4d963:	mov    %rsi,0xd0(%rsp)
   4d96b:	mov    0x50(%rcx),%rbx
   4d96f:	mov    0x38(%rax),%rsi
   4d973:	mov    %rsi,0xe0(%rsp)
   4d97b:	mov    0x30(%rax),%rax
   4d97f:	mov    %rax,0x18(%rsp)
   4d984:	mov    0x20(%rcx),%rax
   4d988:	mov    %rax,0xd8(%rsp)
   4d990:	mov    0x30(%rcx),%r12
   4d994:	vmovups (%rdx),%xmm0
   4d998:	vmovaps %xmm0,0x1a0(%rsp)
   4d9a1:	mov    0x30(%rsp),%rax
   4d9a6:	mov    0x10(%rax),%rax
   4d9aa:	mov    %rax,0xb8(%rsp)
   4d9b2:	vmovdqu (%rcx),%xmm0
   4d9b6:	vmovdqa %xmm0,0x120(%rsp)
   4d9bf:	mov    0xc0(%rcx),%rdx
   4d9c6:	lea    0x1a0(%rsp),%rdi
   4d9ce:	lea    0x120(%rsp),%rsi
   4d9d6:	call   *0xee93c(%rip)        # 13c318 <_DYNAMIC+0x288>
   4d9dc:	mov    %rax,%rsi
   4d9df:	mov    %rdx,%rcx
   4d9e2:	mov    0xa0(%rsp),%edi
   4d9e9:	mov    %r13,%rax
   4d9ec:	mul    %rdi
   4d9ef:	mov    %rbp,%rdx
   4d9f2:	mulx   %rdi,%r10,%r9
   4d9f7:	seto   %dl
   4d9fa:	add    %rax,%r9
   4d9fd:	setb   %r13b
   4da01:	or     %dl,%r13b
   4da04:	mov    %r14,%rax
   4da07:	mul    %r15
   4da0a:	seto   %r11b
   4da0e:	mov    0xc0(%rsp),%rdx
   4da16:	mulx   %r15,%r8,%rdi
   4da1b:	add    %rax,%rdi
   4da1e:	setb   %r14b
   4da22:	or     %r11b,%r14b
   4da25:	mov    0xc8(%rsp),%rax
   4da2d:	mul    %rbx
   4da30:	mov    0xd0(%rsp),%rdx
   4da38:	mulx   %rbx,%rbx,%r11
   4da3d:	seto   %dl
   4da40:	add    %rax,%r11
   4da43:	setb   %bpl
   4da47:	or     %dl,%bpl
   4da4a:	or     %r14b,%bpl
   4da4d:	or     %r13b,%bpl
   4da50:	xor    %r14d,%r14d
   4da53:	add    0xd8(%rsp),%r12
   4da5b:	setb   %r14b
   4da5f:	mov    0xe0(%rsp),%rax
   4da67:	test   %rax,%rax
   4da6a:	setne  %dl
   4da6d:	mov    %r14d,%r15d
   4da70:	and    %dl,%r15b
   4da73:	mul    %r12
   4da76:	seto   %r13b
   4da7a:	or     %r15b,%r13b
   4da7d:	mov    0x18(%rsp),%rdx
   4da82:	imul   %rdx,%r14
   4da86:	add    %rax,%r14
   4da89:	mulx   %r12,%rdx,%rax
   4da8e:	add    %r14,%rax
   4da91:	setb   %r14b
   4da95:	or     %r13b,%r14b
   4da98:	or     %bpl,%r14b
   4da9b:	add    %r10,%r8
   4da9e:	adc    %r9,%rdi
   4daa1:	mov    $0xffffffffffffffff,%r15
   4daa8:	cmovb  %r15,%rdi
   4daac:	cmovb  %r15,%r8
   4dab0:	add    %rbx,%r8
   4dab3:	adc    %r11,%rdi
   4dab6:	cmovb  %r15,%rdi
   4daba:	cmovb  %r15,%r8
   4dabe:	mov    $0xffffffffffffffff,%r10
   4dac5:	mov    $0xffffffffffffffff,%r9
   4dacc:	test   $0x1,%r14b
   4dad0:	jne    4dae6 <_ZN15funnel_patterns31pat_combine_then_mask__u64__w5117h51620081b03c5800E+0xc26>
   4dad2:	add    %rdx,%r8
   4dad5:	adc    %rax,%rdi
   4dad8:	cmovb  %r15,%rdi
   4dadc:	cmovb  %r15,%r8
   4dae0:	mov    %r8,%r10
   4dae3:	mov    %rdi,%r9
   4dae6:	mov    %rsi,%rax
   4dae9:	or     %rcx,%rax
   4daec:	mov    0x50(%rsp),%rdx
   4daf1:	cmove  %rdx,%rcx
   4daf5:	mov    0x58(%rsp),%rax
   4dafa:	cmove  %rax,%rsi
   4dafe:	mov    %rsi,%rbx
   4db01:	sub    %r10,%rbx
   4db04:	mov    %rcx,%r14
   4db07:	sbb    %r9,%r14
   4db0a:	cmp    %rsi,%r10
   4db0d:	sbb    %rcx,%r9
   4db10:	cmovae %rdx,%r14
   4db14:	cmovae %rax,%rbx
   4db18:	mov    0x30(%rsp),%rax
   4db1d:	mov    0x10(%rax),%r15
   4db21:	cmp    (%rax),%r15
   4db24:	jne    4db30 <_ZN15funnel_patterns31pat_combine_then_mask__u64__w5117h51620081b03c5800E+0xc70>
   4db26:	mov    0x30(%rsp),%rdi
   4db2b:	call   51730 <_ZN5alloc7raw_vec19RawVec$LT$T$C$A$GT$8grow_one17hfe527167958a2dadE>
   4db30:	mov    0x30(%rsp),%r12
   4db35:	mov    0x8(%r12),%rax
   4db3a:	mov    %r15,%rcx
   4db3d:	shl    $0x4,%rcx
   4db41:	mov    %r14,0x8(%rax,%rcx,1)
   4db46:	mov    %rbx,(%rax,%rcx,1)
   4db4a:	inc    %r15
   4db4d:	mov    %r15,0x10(%r12)
   4db52:	mov    0x40(%rsp),%r15
   4db57:	mov    0x28(%r15),%rax
   4db5b:	or     0x20(%r15),%rax
   4db5f:	jne    4db80 <_ZN15funnel_patterns31pat_combine_then_mask__u64__w5117h51620081b03c5800E+0xcc0>
   4db61:	mov    0x38(%r15),%rax
   4db65:	or     0x30(%r15),%rax
   4db69:	jne    4db80 <_ZN15funnel_patterns31pat_combine_then_mask__u64__w5117h51620081b03c5800E+0xcc0>
   4db6b:	mov    0x48(%r15),%rax
   4db6f:	or     0x40(%r15),%rax
   4db73:	jne    4db80 <_ZN15funnel_patterns31pat_combine_then_mask__u64__w5117h51620081b03c5800E+0xcc0>
   4db75:	mov    0x58(%r15),%rax
   4db79:	or     0x50(%r15),%rax
   4db7d:	je     4dbd6 <_ZN15funnel_patterns31pat_combine_then_mask__u64__w5117h51620081b03c5800E+0xd16>
   4db7f:	nop
   4db80:	lea    0x20(%r15),%rax
   4db84:	vmovdqu64 (%rax),%zmm0
   4db8a:	vmovdqu64 0x20(%rax),%zmm1
   4db94:	vmovdqu64 %zmm1,0x1c0(%rsp)
   4db9c:	vmovdqu64 %zmm0,0x1a0(%rsp)
   4dba7:	lea    0x120(%rsp),%rdi
   4dbaf:	mov    0x78(%rsp),%rsi
   4dbb4:	mov    0xb8(%rsp),%rdx
   4dbbc:	lea    0x1a0(%rsp),%rcx
   4dbc4:	vzeroupper
   4dbc7:	call   51900 <_ZN9hashbrown3map28HashMap$LT$K$C$V$C$S$C$A$GT$6insert17he36ed9c6ee1bc3e6E>
   4dbcc:	mov    0x30(%rsp),%r12
   4dbd1:	mov    0x40(%rsp),%r15
   4dbd6:	cmpq   $0x0,0x68(%r12)
   4dbdc:	mov    0x14(%rsp),%ebp
   4dbe0:	mov    0x6c(%rsp),%r13d
   4dbe5:	mov    0xee73c(%rip),%rbx        # 13c328 <_DYNAMIC+0x298>
   4dbec:	mov    0xe8(%rsp),%r14
   4dbf4:	je     4dc26 <_ZN15funnel_patterns31pat_combine_then_mask__u64__w5117h51620081b03c5800E+0xd66>
   4dbf6:	mov    0x4(%rsp),%eax
   4dbfa:	test   %eax,%eax
   4dbfc:	je     4dded <_ZN15funnel_patterns31pat_combine_then_mask__u64__w5117h51620081b03c5800E+0xf2d>
   4dc02:	mov    %eax,%edx
   4dc04:	mov    0x80(%r15),%rdi
   4dc0b:	mov    0x88(%r15),%rsi
   4dc12:	xor    %ecx,%ecx
   4dc14:	call   *0xee706(%rip)        # 13c320 <_DYNAMIC+0x290>
   4dc1a:	mov    0x28(%rsp),%rdi
   4dc1f:	mov    %rax,%rsi
   4dc22:	xor    %edx,%edx
   4dc24:	call   *%rbx
   4dc26:	cmpq   $0x0,0x90(%r12)
   4dc2f:	je     4dc64 <_ZN15funnel_patterns31pat_combine_then_mask__u64__w5117h51620081b03c5800E+0xda4>
   4dc31:	mov    0x4(%rsp),%eax
   4dc35:	test   %eax,%eax
   4dc37:	je     4dded <_ZN15funnel_patterns31pat_combine_then_mask__u64__w5117h51620081b03c5800E+0xf2d>
   4dc3d:	mov    %eax,%edx
   4dc3f:	mov    0x90(%r15),%rdi
   4dc46:	mov    0x98(%r15),%rsi
   4dc4d:	xor    %ecx,%ecx
   4dc4f:	call   *0xee6cb(%rip)        # 13c320 <_DYNAMIC+0x290>
   4dc55:	mov    0x28(%rsp),%rdi
   4dc5a:	mov    %rax,%rsi
   4dc5d:	mov    $0x1,%edx
   4dc62:	call   *%rbx
   4dc64:	cmpq   $0x0,0xb8(%r12)
   4dc6d:	je     4dca2 <_ZN15funnel_patterns31pat_combine_then_mask__u64__w5117h51620081b03c5800E+0xde2>
   4dc6f:	mov    0x4(%rsp),%eax
   4dc73:	test   %eax,%eax
   4dc75:	je     4dded <_ZN15funnel_patterns31pat_combine_then_mask__u64__w5117h51620081b03c5800E+0xf2d>
   4dc7b:	mov    %eax,%edx
   4dc7d:	mov    0xa0(%r15),%rdi
   4dc84:	mov    0xa8(%r15),%rsi
   4dc8b:	xor    %ecx,%ecx
   4dc8d:	call   *0xee68d(%rip)        # 13c320 <_DYNAMIC+0x290>
   4dc93:	mov    0x28(%rsp),%rdi
   4dc98:	mov    %rax,%rsi
   4dc9b:	mov    $0x2,%edx
   4dca0:	call   *%rbx
   4dca2:	cmpq   $0x0,0xe0(%r12)
   4dcab:	je     4dce0 <_ZN15funnel_patterns31pat_combine_then_mask__u64__w5117h51620081b03c5800E+0xe20>
   4dcad:	mov    0x4(%rsp),%eax
   4dcb1:	test   %eax,%eax
   4dcb3:	je     4dded <_ZN15funnel_patterns31pat_combine_then_mask__u64__w5117h51620081b03c5800E+0xf2d>
   4dcb9:	mov    %eax,%edx
   4dcbb:	mov    0xb0(%r15),%rdi
   4dcc2:	mov    0xb8(%r15),%rsi
   4dcc9:	xor    %ecx,%ecx
   4dccb:	call   *0xee64f(%rip)        # 13c320 <_DYNAMIC+0x290>
   4dcd1:	mov    0x28(%rsp),%rdi
   4dcd6:	mov    %rax,%rsi
   4dcd9:	mov    $0x3,%edx
   4dcde:	call   *%rbx
   4dce0:	mov    0x10(%rsp),%ecx
   4dce4:	mov    %ecx,%edx
   4dce6:	sub    $0x1,%edx
   4dce9:	mov    $0x0,%eax
   4dcee:	cmovb  %eax,%edx
   4dcf1:	testb  $0x1,0xc(%rsp)
   4dcf6:	cmove  %ecx,%edx
   4dcf9:	mov    %edx,0x18(%rsp)
   4dcfd:	cmp    $0x3b9aca01,%r13d
   4dd04:	je     4d3a0 <_ZN15funnel_patterns31pat_combine_then_mask__u64__w5117h51620081b03c5800E+0x4e0>
   4dd0a:	mov    0x70(%rsp),%rax
   4dd0f:	mov    %rax,0x190(%rsp)
   4dd17:	mov    %r13d,0x198(%rsp)
   4dd1f:	mov    0x18(%r15),%eax
   4dd23:	cmp    $0x3b9aca01,%eax
   4dd28:	je     4de84 <_ZN15funnel_patterns31pat_combine_then_mask__u64__w5117h51620081b03c5800E+0xfc4>
   4dd2e:	mov    0x1c(%r15),%ecx
   4dd32:	mov    0x10(%r15),%rdx
   4dd36:	mov    %rdx,0x110(%rsp)
   4dd3e:	mov    %eax,0x118(%rsp)
   4dd45:	mov    %ecx,0x11c(%rsp)
   4dd4c:	mov    0x60(%rsp),%rdx
   4dd51:	lea    0x110(%rsp),%rdi
   4dd59:	lea    0x190(%rsp),%rsi
   4dd61:	call   *0xee5b1(%rip)        # 13c318 <_DYNAMIC+0x288>
   4dd67:	mov    %rdx,%rsi
   4dd6a:	mov    %rax,%rdx
   4dd6d:	jmp    4d3f0 <_ZN15funnel_patterns31pat_combine_then_mask__u64__w5117h51620081b03c5800E+0x530>
   4dd72:	mov    $0x10,%r15d
   4dd78:	xor    %r14d,%r14d
   4dd7b:	mov    0xee566(%rip),%rbx        # 13c2e8 <_DYNAMIC+0x258>
   4dd82:	mov    0x8(%rbx),%eax
   4dd85:	test   %eax,%eax
   4dd87:	jne    4dddc <_ZN15funnel_patterns31pat_combine_then_mask__u64__w5117h51620081b03c5800E+0xf1c>
   4dd89:	mov    (%rbx),%rdi
   4dd8c:	call   *0xee59e(%rip)        # 13c330 <_DYNAMIC+0x2a0>
   4dd92:	mov    0xee59f(%rip),%rax        # 13c338 <_DYNAMIC+0x2a8>
   4dd99:	movb   $0x0,(%rax)
   4dd9c:	test   %r14,%r14
   4dd9f:	je     4ddb6 <_ZN15funnel_patterns31pat_combine_then_mask__u64__w5117h51620081b03c5800E+0xef6>
   4dda1:	imul   $0xd0,%r14,%rsi
   4dda8:	mov    $0x10,%edx
   4ddad:	mov    %r15,%rdi
   4ddb0:	call   *0xee58a(%rip)        # 13c340 <_DYNAMIC+0x2b0>
   4ddb6:	add    $0x3c18,%rsp
   4ddbd:	pop    %rbx
   4ddbe:	pop    %r12
   4ddc0:	pop    %r13
   4ddc2:	pop    %r14
   4ddc4:	pop    %r15
   4ddc6:	pop    %rbp
   4ddc7:	ret
   4ddc8:	mov    $0x1,%r14d
   4ddce:	mov    0xee513(%rip),%rbx        # 13c2e8 <_DYNAMIC+0x258>
   4ddd5:	mov    0x8(%rbx),%eax
   4ddd8:	test   %eax,%eax
   4ddda:	je     4dd89 <_ZN15funnel_patterns31pat_combine_then_mask__u64__w5117h51620081b03c5800E+0xec9>
   4dddc:	call   5152e <_ZN3std4sync9once_lock17OnceLock$LT$T$GT$10initialize17h13c8d01c1aad6442E>
   4dde1:	jmp    4dd89 <_ZN15funnel_patterns31pat_combine_then_mask__u64__w5117h51620081b03c5800E+0xec9>
   4dde3:	call   5152e <_ZN3std4sync9once_lock17OnceLock$LT$T$GT$10initialize17h13c8d01c1aad6442E>
   4dde8:	jmp    4d34b <_ZN15funnel_patterns31pat_combine_then_mask__u64__w5117h51620081b03c5800E+0x48b>
   4dded:	lea    0xe5d34(%rip),%rdi        # 133b28 <_ZN15funnel_patterns46__DIVAN_BENCH_PAT_U128_CAT_UNROLLED4__U64__W514PUSH17h68d55424f84a52b4E+0xe0>
   4ddf4:	call   *0xee54e(%rip)        # 13c348 <_DYNAMIC+0x2b8>
   4ddfa:	jmp    4dea0 <_ZN15funnel_patterns31pat_combine_then_mask__u64__w5117h51620081b03c5800E+0xfe0>
   4ddff:	movq   $0x0,0x108(%rsp)
   4de0b:	lea    0x108(%rsp),%rax
   4de13:	mov    %rax,0x1a0(%rsp)
   4de1b:	mov    0xee52e(%rip),%rax        # 13c350 <_DYNAMIC+0x2c0>
   4de22:	mov    %rax,0x1a8(%rsp)
   4de2a:	lea    0xe5c77(%rip),%rax        # 133aa8 <_ZN15funnel_patterns46__DIVAN_BENCH_PAT_U128_CAT_UNROLLED4__U64__W514PUSH17h68d55424f84a52b4E+0x60>
   4de31:	mov    %rax,0x120(%rsp)
   4de39:	movq   $0x2,0x128(%rsp)
   4de45:	movq   $0x0,0x140(%rsp)
   4de51:	lea    0x1a0(%rsp),%rax
   4de59:	mov    %rax,0x130(%rsp)
   4de61:	movq   $0x1,0x138(%rsp)
   4de6d:	lea    0xe5c54(%rip),%rsi        # 133ac8 <_ZN15funnel_patterns46__DIVAN_BENCH_PAT_U128_CAT_UNROLLED4__U64__W514PUSH17h68d55424f84a52b4E+0x80>
   4de74:	lea    0x120(%rsp),%rdi
   4de7c:	call   *0xee4d6(%rip)        # 13c358 <_DYNAMIC+0x2c8>
   4de82:	jmp    4dea0 <_ZN15funnel_patterns31pat_combine_then_mask__u64__w5117h51620081b03c5800E+0xfe0>
   4de84:	lea    0xe5c6d(%rip),%rdi        # 133af8 <_ZN15funnel_patterns46__DIVAN_BENCH_PAT_U128_CAT_UNROLLED4__U64__W514PUSH17h68d55424f84a52b4E+0xb0>
   4de8b:	call   *0xee4cf(%rip)        # 13c360 <_DYNAMIC+0x2d0>
   4de91:	jmp    4dea0 <_ZN15funnel_patterns31pat_combine_then_mask__u64__w5117h51620081b03c5800E+0xfe0>
   4de93:	lea    0xe5c46(%rip),%rdi        # 133ae0 <_ZN15funnel_patterns46__DIVAN_BENCH_PAT_U128_CAT_UNROLLED4__U64__W514PUSH17h68d55424f84a52b4E+0x98>
   4de9a:	call   *0xee4a8(%rip)        # 13c348 <_DYNAMIC+0x2b8>
   4dea0:	ud2
   4dea2:	mov    0x1a8(%rsp),%rdi
   4deaa:	mov    0x1b0(%rsp),%rsi
   4deb2:	lea    0xe5bd7(%rip),%rdx        # 133a90 <_ZN15funnel_patterns46__DIVAN_BENCH_PAT_U128_CAT_UNROLLED4__U64__W514PUSH17h68d55424f84a52b4E+0x48>
   4deb9:	call   *0xee4a9(%rip)        # 13c368 <_DYNAMIC+0x2d8>
   4debf:	mov    %r15,0x40(%rsp)
   4dec4:	mov    %rax,%rbx
   4dec7:	test   %r14,%r14
   4deca:	jne    4df00 <_ZN15funnel_patterns31pat_combine_then_mask__u64__w5117h51620081b03c5800E+0x1040>
   4decc:	jmp    4df15 <_ZN15funnel_patterns31pat_combine_then_mask__u64__w5117h51620081b03c5800E+0x1055>
   4dece:	mov    %rax,%rbx
   4ded1:	mov    0x120(%rsp),%rax
   4ded9:	lock decq (%rax)
   4dedd:	jne    4df00 <_ZN15funnel_patterns31pat_combine_then_mask__u64__w5117h51620081b03c5800E+0x1040>
   4dedf:	lea    0x120(%rsp),%rdi
   4dee7:	call   *0xee423(%rip)        # 13c310 <_DYNAMIC+0x280>
   4deed:	jmp    4df00 <_ZN15funnel_patterns31pat_combine_then_mask__u64__w5117h51620081b03c5800E+0x1040>
   4deef:	call   *0xee47b(%rip)        # 13c370 <_DYNAMIC+0x2e0>
   4def5:	jmp    4defd <_ZN15funnel_patterns31pat_combine_then_mask__u64__w5117h51620081b03c5800E+0x103d>
   4def7:	jmp    4defd <_ZN15funnel_patterns31pat_combine_then_mask__u64__w5117h51620081b03c5800E+0x103d>
   4def9:	jmp    4defd <_ZN15funnel_patterns31pat_combine_then_mask__u64__w5117h51620081b03c5800E+0x103d>
   4defb:	jmp    4defd <_ZN15funnel_patterns31pat_combine_then_mask__u64__w5117h51620081b03c5800E+0x103d>
   4defd:	mov    %rax,%rbx
   4df00:	mov    $0xd0,%esi
   4df05:	mov    $0x10,%edx
   4df0a:	mov    0x40(%rsp),%rdi
   4df0f:	call   *0xee42b(%rip)        # 13c340 <_DYNAMIC+0x2b0>
   4df15:	mov    %rbx,%rdi
   4df18:	call   1328b0 <_Unwind_Resume@plt>

Disassembly of section .init:

Disassembly of section .fini:

Disassembly of section .plt:
