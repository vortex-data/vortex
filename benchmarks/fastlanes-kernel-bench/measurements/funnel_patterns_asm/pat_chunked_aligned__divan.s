
/home/user/vortex/target/release/deps/funnel_patterns-21c1c00107f42b8a:     file format elf64-x86-64


Disassembly of section .text:

000000000004ae00 <_ZN15funnel_patterns29pat_chunked_aligned__u64__w5117h3114e5c745d20737E>:
   4ae00:	push   %rbp
   4ae01:	push   %r15
   4ae03:	push   %r14
   4ae05:	push   %r13
   4ae07:	push   %r12
   4ae09:	push   %rbx
   4ae0a:	sub    $0x1000,%rsp
   4ae11:	movq   $0x0,(%rsp)
   4ae19:	sub    $0x1000,%rsp
   4ae20:	movq   $0x0,(%rsp)
   4ae28:	sub    $0x1000,%rsp
   4ae2f:	movq   $0x0,(%rsp)
   4ae37:	sub    $0xc18,%rsp
   4ae3e:	mov    %rdi,%r12
   4ae41:	lea    0x298(%rsp),%rdi
   4ae49:	mov    $0x1980,%edx
   4ae4e:	xor    %esi,%esi
   4ae50:	call   *0xf146a(%rip)        # 13c2c0 <memset@GLIBC_2.2.5>
   4ae56:	vmovdqa64 -0x3af60(%rip),%zmm0        # ff00 <__abi_tag+0xfc04>
   4ae60:	mov    $0x38,%eax
   4ae65:	vpbroadcastq -0x3a6df(%rip),%zmm1        # 10790 <__abi_tag+0x10494>
   4ae6f:	vpbroadcastq -0x3a809(%rip),%zmm2        # 10670 <__abi_tag+0x10374>
   4ae79:	vpbroadcastq -0x3a6eb(%rip),%zmm3        # 10798 <__abi_tag+0x1049c>
   4ae83:	vpbroadcastq -0x3a66d(%rip),%zmm4        # 10820 <__abi_tag+0x10524>
   4ae8d:	vpbroadcastq -0x3a647(%rip),%zmm5        # 10850 <__abi_tag+0x10554>
   4ae97:	vpbroadcastq -0x3a811(%rip),%zmm6        # 10690 <__abi_tag+0x10394>
   4aea1:	vpbroadcastq -0x3a623(%rip),%zmm7        # 10888 <__abi_tag+0x1058c>
   4aeab:	vpbroadcastq -0x3a65d(%rip),%zmm8        # 10858 <__abi_tag+0x1055c>
   4aeb5:	vpbroadcastq -0x3a63f(%rip),%zmm9        # 10880 <__abi_tag+0x10584>
   4aebf:	nop
   4aec0:	vpmullq %zmm1,%zmm0,%zmm10
   4aec6:	vpaddq %zmm2,%zmm10,%zmm11
   4aecc:	vpaddq %zmm3,%zmm10,%zmm12
   4aed2:	vmovdqu64 %zmm10,0xd8(%rsp,%rax,8)
   4aedd:	vmovdqu64 %zmm11,0x118(%rsp,%rax,8)
   4aee8:	vpaddq %zmm4,%zmm10,%zmm11
   4aeee:	vmovdqu64 %zmm12,0x158(%rsp,%rax,8)
   4aef9:	vmovdqu64 %zmm11,0x198(%rsp,%rax,8)
   4af04:	cmp    $0x338,%rax
   4af0a:	je     4af5f <_ZN15funnel_patterns29pat_chunked_aligned__u64__w5117h3114e5c745d20737E+0x15f>
   4af0c:	vpaddq %zmm5,%zmm10,%zmm11
   4af12:	vpaddq %zmm6,%zmm10,%zmm12
   4af18:	vpaddq %zmm7,%zmm10,%zmm13
   4af1e:	vpaddq %zmm8,%zmm10,%zmm10
   4af24:	vmovdqu64 %zmm11,0x1d8(%rsp,%rax,8)
   4af2f:	vmovdqu64 %zmm12,0x218(%rsp,%rax,8)
   4af3a:	vmovdqu64 %zmm13,0x258(%rsp,%rax,8)
   4af45:	vmovdqu64 %zmm10,0x298(%rsp,%rax,8)
   4af50:	vpaddq %zmm9,%zmm0,%zmm0
   4af56:	add    $0x40,%rax
   4af5a:	jmp    4aec0 <_ZN15funnel_patterns29pat_chunked_aligned__u64__w5117h3114e5c745d20737E+0xc0>
   4af5f:	vmovaps -0x3b029(%rip),%zmm0        # ff40 <__abi_tag+0xfc44>
   4af69:	vmovups %zmm0,0x1b98(%rsp)
   4af74:	vmovdqa64 -0x3affe(%rip),%zmm0        # ff80 <__abi_tag+0xfc84>
   4af7e:	vmovdqu64 %zmm0,0x1bd8(%rsp)
   4af89:	lea    0x2298(%rsp),%rbx
   4af91:	lea    0x298(%rsp),%r14
   4af99:	mov    $0x1980,%edx
   4af9e:	mov    %rbx,%rdi
   4afa1:	mov    %r14,%rsi
   4afa4:	vzeroupper
   4afa7:	call   *0xf131b(%rip)        # 13c2c8 <memcpy@GLIBC_2.14>
   4afad:	movq   $0x3b9aca07,0xf0(%rsp)
   4afb9:	xor    %ebp,%ebp
   4afbb:	mov    $0x2000,%edx
   4afc0:	mov    %r14,%rdi
   4afc3:	xor    %esi,%esi
   4afc5:	call   *0xf12f5(%rip)        # 13c2c0 <memset@GLIBC_2.2.5>
   4afcb:	mov    %rbx,0x208(%rsp)
   4afd3:	lea    0xf0(%rsp),%rax
   4afdb:	mov    %rax,0x210(%rsp)
   4afe3:	mov    %r14,0x218(%rsp)
   4afeb:	lea    0x208(%rsp),%rax
   4aff3:	mov    %rax,0xf8(%rsp)
   4affb:	lea    0xf8(%rsp),%rax
   4b003:	mov    %rax,0x100(%rsp)
   4b00b:	movq   $0x1,0x100(%r12)
   4b017:	movb   $0x1,0x108(%r12)
   4b020:	mov    0xf0(%r12),%rdx
   4b028:	mov    0xf8(%r12),%rax
   4b030:	movzbl 0x8(%rdx),%ecx
   4b034:	mov    %cl,0x3(%rsp)
   4b038:	cmp    $0x1,%cl
   4b03b:	jne    4b043 <_ZN15funnel_patterns29pat_chunked_aligned__u64__w5117h3114e5c745d20737E+0x243>
   4b03d:	xor    %esi,%esi
   4b03f:	xor    %ecx,%ecx
   4b041:	jmp    4b06d <_ZN15funnel_patterns29pat_chunked_aligned__u64__w5117h3114e5c745d20737E+0x26d>
   4b043:	cmpb   $0x0,0x60(%rax)
   4b047:	je     4b05c <_ZN15funnel_patterns29pat_chunked_aligned__u64__w5117h3114e5c745d20737E+0x25c>
   4b049:	mov    0x64(%rax),%ecx
   4b04c:	mov    %ecx,0x8(%rsp)
   4b050:	mov    $0x2,%ebp
   4b055:	mov    $0x1,%sil
   4b058:	xor    %ecx,%ecx
   4b05a:	jmp    4b06d <_ZN15funnel_patterns29pat_chunked_aligned__u64__w5117h3114e5c745d20737E+0x26d>
   4b05c:	mov    $0x1,%cl
   4b05e:	movl   $0x1,0x8(%rsp)
   4b066:	xor    %esi,%esi
   4b068:	mov    $0x1,%ebp
   4b06d:	mov    (%rdx),%r14
   4b070:	test   %r14,%r14
   4b073:	lea    0x27(%rsp),%rdx
   4b078:	mov    %rdx,0x220(%rsp)
   4b080:	setne  0x238(%rsp)
   4b088:	lea    0x100(%rsp),%rdi
   4b090:	mov    %rdi,0x228(%rsp)
   4b098:	mov    %rdx,0x230(%rsp)
   4b0a0:	mov    0x70(%rax),%edi
   4b0a3:	movq   $0x0,0x88(%rsp)
   4b0af:	mov    $0x0,%edx
   4b0b4:	mov    %rdx,0x80(%rsp)
   4b0bc:	cmp    $0x3b9aca00,%edi
   4b0c2:	je     4b0fd <_ZN15funnel_patterns29pat_chunked_aligned__u64__w5117h3114e5c745d20737E+0x2fd>
   4b0c4:	mov    $0x3b9aca00,%edx
   4b0c9:	mulx   0x68(%rax),%r8,%r9
   4b0cf:	mov    %edi,%edx
   4b0d1:	add    %r8,%rdx
   4b0d4:	adc    $0x0,%r9
   4b0d8:	imul   $0x3e8,%r9,%rdi
   4b0df:	mov    $0x3e8,%r8d
   4b0e5:	mulx   %r8,%rdx,%r8
   4b0ea:	mov    %rdx,0x88(%rsp)
   4b0f2:	add    %rdi,%r8
   4b0f5:	mov    %r8,0x80(%rsp)
   4b0fd:	movq   $0xffffffffffffffff,0x48(%rsp)
   4b106:	mov    0x80(%rax),%r8d
   4b10d:	movq   $0xffffffffffffffff,0x38(%rsp)
   4b116:	cmp    $0x3b9aca00,%r8d
   4b11d:	je     4b15e <_ZN15funnel_patterns29pat_chunked_aligned__u64__w5117h3114e5c745d20737E+0x35e>
   4b11f:	mov    $0x3b9aca00,%edx
   4b124:	mulx   0x78(%rax),%r9,%rdi
   4b12a:	mov    %r8d,%edx
   4b12d:	add    %r9,%rdx
   4b130:	adc    $0x0,%rdi
   4b134:	mov    $0x3e8,%r8d
   4b13a:	mulx   %r8,%r8,%r9
   4b13f:	mov    %r9,0x38(%rsp)
   4b144:	mov    %r8,0x48(%rsp)
   4b149:	or     %rdi,%rdx
   4b14c:	je     4bcf6 <_ZN15funnel_patterns29pat_chunked_aligned__u64__w5117h3114e5c745d20737E+0xef6>
   4b152:	imul   $0x3e8,%rdi,%rdx
   4b159:	add    %rdx,0x38(%rsp)
   4b15e:	mov    0x58(%rax),%edx
   4b161:	cmp    $0x1,%edx
   4b164:	jne    4b170 <_ZN15funnel_patterns29pat_chunked_aligned__u64__w5117h3114e5c745d20737E+0x370>
   4b166:	cmpl   $0x0,0x5c(%rax)
   4b16a:	je     4bcf6 <_ZN15funnel_patterns29pat_chunked_aligned__u64__w5117h3114e5c745d20737E+0xef6>
   4b170:	cmpl   $0x1,0x60(%rax)
   4b174:	jne    4b180 <_ZN15funnel_patterns29pat_chunked_aligned__u64__w5117h3114e5c745d20737E+0x380>
   4b176:	cmpl   $0x0,0x64(%rax)
   4b17a:	je     4bcf6 <_ZN15funnel_patterns29pat_chunked_aligned__u64__w5117h3114e5c745d20737E+0xef6>
   4b180:	mov    %r14,0x60(%rsp)
   4b185:	test   %dl,%sil
   4b188:	je     4b19b <_ZN15funnel_patterns29pat_chunked_aligned__u64__w5117h3114e5c745d20737E+0x39b>
   4b18a:	mov    0x5c(%rax),%edx
   4b18d:	mov    %edx,0x18(%rsp)
   4b191:	movl   $0x1,0xc(%rsp)
   4b199:	jmp    4b1ab <_ZN15funnel_patterns29pat_chunked_aligned__u64__w5117h3114e5c745d20737E+0x3ab>
   4b19b:	movzbl %sil,%edx
   4b19f:	mov    %edx,0xc(%rsp)
   4b1a3:	movl   $0x64,0x18(%rsp)
   4b1ab:	movq   $0x0,0x58(%rsp)
   4b1b4:	mov    $0x0,%edx
   4b1b9:	mov    %rdx,0x50(%rsp)
   4b1be:	test   %cl,%cl
   4b1c0:	je     4b1dd <_ZN15funnel_patterns29pat_chunked_aligned__u64__w5117h3114e5c745d20737E+0x3dd>
   4b1c2:	mov    %r14,%rdi
   4b1c5:	call   *0xf1105(%rip)        # 13c2d0 <_DYNAMIC+0x240>
   4b1cb:	mov    %rax,0x58(%rsp)
   4b1d0:	mov    %rdx,0x50(%rsp)
   4b1d5:	mov    0xf8(%r12),%rax
   4b1dd:	cmpb   $0x1,0x3(%rsp)
   4b1e2:	je     4b203 <_ZN15funnel_patterns29pat_chunked_aligned__u64__w5117h3114e5c745d20737E+0x403>
   4b1e4:	mov    $0x1,%edx
   4b1e9:	cmpb   $0x0,0x58(%rax)
   4b1ed:	je     4b1f2 <_ZN15funnel_patterns29pat_chunked_aligned__u64__w5117h3114e5c745d20737E+0x3f2>
   4b1ef:	mov    0x5c(%rax),%edx
   4b1f2:	mov    (%r12),%rcx
   4b1f6:	mov    0x10(%r12),%rsi
   4b1fb:	sub    %rsi,%rcx
   4b1fe:	cmp    %rcx,%rdx
   4b201:	ja     4b248 <_ZN15funnel_patterns29pat_chunked_aligned__u64__w5117h3114e5c745d20737E+0x448>
   4b203:	testb  $0x1,0x88(%rax)
   4b20a:	jne    4b261 <_ZN15funnel_patterns29pat_chunked_aligned__u64__w5117h3114e5c745d20737E+0x461>
   4b20c:	lock orl $0x0,-0x40(%rsp)
   4b212:	test   %r14,%r14
   4b215:	je     4b233 <_ZN15funnel_patterns29pat_chunked_aligned__u64__w5117h3114e5c745d20737E+0x433>
   4b217:	lfence
   4b21a:	rdtsc
   4b21c:	shl    $0x20,%rdx
   4b220:	or     %rax,%rdx
   4b223:	mov    %rdx,0x70(%rsp)
   4b228:	lfence
   4b22b:	mov    $0x3b9aca00,%r13d
   4b231:	jmp    4b241 <_ZN15funnel_patterns29pat_chunked_aligned__u64__w5117h3114e5c745d20737E+0x441>
   4b233:	call   *0xf109f(%rip)        # 13c2d8 <_DYNAMIC+0x248>
   4b239:	mov    %rax,0x70(%rsp)
   4b23e:	mov    %edx,%r13d
   4b241:	mov    0x60(%rsp),%r14
   4b246:	jmp    4b267 <_ZN15funnel_patterns29pat_chunked_aligned__u64__w5117h3114e5c745d20737E+0x467>
   4b248:	mov    %r12,%rdi
   4b24b:	call   51800 <_ZN5alloc7raw_vec20RawVecInner$LT$A$GT$7reserve21do_reserve_and_handle17h46771c9d08372974E>
   4b250:	mov    0xf8(%r12),%rax
   4b258:	testb  $0x1,0x88(%rax)
   4b25f:	je     4b20c <_ZN15funnel_patterns29pat_chunked_aligned__u64__w5117h3114e5c745d20737E+0x40c>
   4b261:	mov    $0x3b9aca01,%r13d
   4b267:	mov    %r14,%rdi
   4b26a:	call   *0xf1070(%rip)        # 13c2e0 <_DYNAMIC+0x250>
   4b270:	mov    %rax,0xa8(%rsp)
   4b278:	mov    0xf1069(%rip),%r14        # 13c2e8 <_DYNAMIC+0x258>
   4b27f:	mov    0x8(%r14),%eax
   4b283:	test   %eax,%eax
   4b285:	jne    4bd23 <_ZN15funnel_patterns29pat_chunked_aligned__u64__w5117h3114e5c745d20737E+0xf23>
   4b28b:	mov    (%r14),%rdi
   4b28e:	call   *0xf105c(%rip)        # 13c2f0 <_DYNAMIC+0x260>
   4b294:	mov    0x48(%rsp),%rax
   4b299:	or     0x38(%rsp),%rax
   4b29e:	je     4bcb2 <_ZN15funnel_patterns29pat_chunked_aligned__u64__w5117h3114e5c745d20737E+0xeb2>
   4b2a4:	lea    0x50(%r12),%rax
   4b2a9:	mov    %rax,0x28(%rsp)
   4b2ae:	lea    0x18(%r12),%rax
   4b2b3:	mov    %rax,0x78(%rsp)
   4b2b8:	mov    $0x10,%r15d
   4b2be:	mov    $0x1,%al
   4b2c0:	xor    %edx,%edx
   4b2c2:	xor    %esi,%esi
   4b2c4:	xor    %r14d,%r14d
   4b2c7:	mov    %r12,0x30(%rsp)
   4b2cc:	mov    %r13d,0x6c(%rsp)
   4b2d1:	jmp    4b34b <_ZN15funnel_patterns29pat_chunked_aligned__u64__w5117h3114e5c745d20737E+0x54b>
   4b2d3:	data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
   4b2e0:	mov    0x90(%rsp),%rdi
   4b2e8:	cmp    $0x3e9,%rdi
   4b2ef:	mov    $0x3e8,%eax
   4b2f4:	cmovae %rdi,%rax
   4b2f8:	mov    0x98(%rsp),%rsi
   4b300:	test   %rsi,%rsi
   4b303:	mov    $0x3e8,%ecx
   4b308:	cmove  %rcx,%rdi
   4b30c:	cmove  %rax,%rdi
   4b310:	mov    0xb0(%rsp),%rdx
   4b318:	add    %rdi,%rdx
   4b31b:	adc    %rsi,%r14
   4b31e:	mov    $0xffffffffffffffff,%rax
   4b325:	cmovb  %rax,%r14
   4b329:	cmovb  %rax,%rdx
   4b32d:	mov    %r14,%rsi
   4b330:	mov    $0x1,%r14d
   4b336:	xor    %eax,%eax
   4b338:	cmp    0x48(%rsp),%rdx
   4b33d:	mov    %rsi,%rcx
   4b340:	sbb    0x38(%rsp),%rcx
   4b345:	jae    4bcbb <_ZN15funnel_patterns29pat_chunked_aligned__u64__w5117h3114e5c745d20737E+0xebb>
   4b34b:	cmp    0x88(%rsp),%rdx
   4b353:	mov    %rsi,0xe8(%rsp)
   4b35b:	mov    %rsi,%rcx
   4b35e:	sbb    0x80(%rsp),%rcx
   4b366:	jb     4b37a <_ZN15funnel_patterns29pat_chunked_aligned__u64__w5117h3114e5c745d20737E+0x57a>
   4b368:	testb  $0x1,0xc(%rsp)
   4b36d:	je     4b37a <_ZN15funnel_patterns29pat_chunked_aligned__u64__w5117h3114e5c745d20737E+0x57a>
   4b36f:	cmpl   $0x0,0x18(%rsp)
   4b374:	je     4bcbb <_ZN15funnel_patterns29pat_chunked_aligned__u64__w5117h3114e5c745d20737E+0xebb>
   4b37a:	mov    %rdx,0xb0(%rsp)
   4b382:	test   %ebp,%ebp
   4b384:	mov    0x8(%rsp),%ecx
   4b388:	mov    $0x1,%edx
   4b38d:	cmove  %edx,%ecx
   4b390:	mov    %ecx,0x4(%rsp)
   4b394:	mov    %ecx,0x48(%r12)
   4b399:	movq   $0x0,0x268(%rsp)
   4b3a5:	mov    0x28(%rsp),%rcx
   4b3aa:	mov    %rcx,0x240(%rsp)
   4b3b2:	lea    0x220(%rsp),%rcx
   4b3ba:	mov    %rcx,0x248(%rsp)
   4b3c2:	lea    0x4(%rsp),%rcx
   4b3c7:	mov    %rcx,0x250(%rsp)
   4b3cf:	lea    0x268(%rsp),%rcx
   4b3d7:	mov    %rcx,0x258(%rsp)
   4b3df:	lea    0x60(%rsp),%rcx
   4b3e4:	mov    %rcx,0x260(%rsp)
   4b3ec:	test   $0x1,%al
   4b3ee:	jne    4b560 <_ZN15funnel_patterns29pat_chunked_aligned__u64__w5117h3114e5c745d20737E+0x760>
   4b3f4:	imul   $0xd0,%r14,%rax
   4b3fb:	add    %r15,%rax
   4b3fe:	mov    %rax,%rdx
   4b401:	sub    %r15,%rdx
   4b404:	add    $0xffffffffffffff30,%rdx
   4b40b:	movabs $0x4ec4ec4ec4ec4ec5,%rcx
   4b415:	mulx   %rcx,%rcx,%rcx
   4b41a:	mov    %r15,%rsi
   4b41d:	cmp    $0xc30,%rdx
   4b424:	jb     4b5b0 <_ZN15funnel_patterns29pat_chunked_aligned__u64__w5117h3114e5c745d20737E+0x7b0>
   4b42a:	shr    $0x6,%rcx
   4b42e:	inc    %rcx
   4b431:	cmp    $0x3330,%rdx
   4b438:	jae    4b450 <_ZN15funnel_patterns29pat_chunked_aligned__u64__w5117h3114e5c745d20737E+0x650>
   4b43a:	xor    %edx,%edx
   4b43c:	mov    %r15,%rdi
   4b43f:	jmp    4b4ff <_ZN15funnel_patterns29pat_chunked_aligned__u64__w5117h3114e5c745d20737E+0x6ff>
   4b444:	data16 data16 cs nopw 0x0(%rax,%rax,1)
   4b450:	mov    %rcx,%rdx
   4b453:	movabs $0x3ffffffffffffc0,%rsi
   4b45d:	and    %rsi,%rdx
   4b460:	imul   $0xd0,%rdx,%rsi
   4b467:	lea    (%r15,%rsi,1),%rdi
   4b46b:	mov    %rdx,%r8
   4b46e:	mov    %r15,%r9
   4b471:	vpbroadcastd -0x3aba7(%rip),%zmm0        # 108d4 <__abi_tag+0x105d8>
   4b47b:	vmovdqa64 -0x3b4c5(%rip),%zmm1        # ffc0 <__abi_tag+0xfcc4>
   4b485:	vmovdqa64 -0x3b48f(%rip),%zmm2        # 10000 <__abi_tag+0xfd04>
   4b48f:	vmovdqa64 -0x3b459(%rip),%zmm3        # 10040 <__abi_tag+0xfd44>
   4b499:	vmovdqa64 -0x3b423(%rip),%zmm4        # 10080 <__abi_tag+0xfd84>
   4b4a3:	data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
   4b4b0:	kxnorw %k0,%k0,%k1
   4b4b4:	vpscatterdd %zmm0,0x8(%r9,%zmm1,1){%k1}
   4b4bc:	kxnorw %k0,%k0,%k1
   4b4c0:	vpscatterdd %zmm0,0x8(%r9,%zmm2,1){%k1}
   4b4c8:	kxnorw %k0,%k0,%k1
   4b4cc:	vpscatterdd %zmm0,0x8(%r9,%zmm3,1){%k1}
   4b4d4:	kxnorw %k0,%k0,%k1
   4b4d8:	vpscatterdd %zmm0,0x8(%r9,%zmm4,1){%k1}
   4b4e0:	add    $0x3400,%r9
   4b4e7:	add    $0xffffffffffffffc0,%r8
   4b4eb:	jne    4b4b0 <_ZN15funnel_patterns29pat_chunked_aligned__u64__w5117h3114e5c745d20737E+0x6b0>
   4b4ed:	cmp    %rdx,%rcx
   4b4f0:	je     4b5c3 <_ZN15funnel_patterns29pat_chunked_aligned__u64__w5117h3114e5c745d20737E+0x7c3>
   4b4f6:	test   $0x30,%cl
   4b4f9:	je     4b5ad <_ZN15funnel_patterns29pat_chunked_aligned__u64__w5117h3114e5c745d20737E+0x7ad>
   4b4ff:	movabs $0x3ffffffffffffc0,%rsi
   4b509:	lea    0x30(%rsi),%r8
   4b50d:	and    %rcx,%r8
   4b510:	imul   $0xd0,%r8,%rsi
   4b517:	add    %r15,%rsi
   4b51a:	sub    %r8,%rdx
   4b51d:	vpbroadcastd -0x3ac53(%rip),%zmm0        # 108d4 <__abi_tag+0x105d8>
   4b527:	vmovdqa64 -0x3b571(%rip),%zmm1        # ffc0 <__abi_tag+0xfcc4>
   4b531:	data16 data16 data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
   4b540:	kxnorw %k0,%k0,%k1
   4b544:	vpscatterdd %zmm0,0x8(%rdi,%zmm1,1){%k1}
   4b54c:	add    $0xd00,%rdi
   4b553:	add    $0x10,%rdx
   4b557:	jne    4b540 <_ZN15funnel_patterns29pat_chunked_aligned__u64__w5117h3114e5c745d20737E+0x740>
   4b559:	cmp    %r8,%rcx
   4b55c:	jne    4b5b0 <_ZN15funnel_patterns29pat_chunked_aligned__u64__w5117h3114e5c745d20737E+0x7b0>
   4b55e:	jmp    4b5c3 <_ZN15funnel_patterns29pat_chunked_aligned__u64__w5117h3114e5c745d20737E+0x7c3>
   4b560:	movq   $0x0,0x128(%rsp)
   4b56c:	mov    $0x10,%esi
   4b571:	mov    $0xd0,%edx
   4b576:	lea    0x1a0(%rsp),%rdi
   4b57e:	lea    0x120(%rsp),%rcx
   4b586:	call   516c0 <_ZN5alloc7raw_vec11finish_grow17hedc133b40cb748a9E>
   4b58b:	cmpb   $0x0,0x1a0(%rsp)
   4b593:	jne    4bde2 <_ZN15funnel_patterns29pat_chunked_aligned__u64__w5117h3114e5c745d20737E+0xfe2>
   4b599:	mov    0x1a8(%rsp),%r15
   4b5a1:	lea    0xd0(%r15),%rax
   4b5a8:	jmp    4b3fe <_ZN15funnel_patterns29pat_chunked_aligned__u64__w5117h3114e5c745d20737E+0x5fe>
   4b5ad:	add    %r15,%rsi
   4b5b0:	movl   $0x3b9aca01,0x8(%rsi)
   4b5b7:	add    $0xd0,%rsi
   4b5be:	cmp    %rax,%rsi
   4b5c1:	jne    4b5b0 <_ZN15funnel_patterns29pat_chunked_aligned__u64__w5117h3114e5c745d20737E+0x7b0>
   4b5c3:	mov    %r12,%rbx
   4b5c6:	mov    %r15,0x40(%rsp)
   4b5cb:	vzeroupper
   4b5ce:	call   *0xf0d24(%rip)        # 13c2f8 <_DYNAMIC+0x268>
   4b5d4:	mov    %rax,0x120(%rsp)
   4b5dc:	movq   $0x0,0x128(%rsp)
   4b5e8:	lea    0x3ad1(%rip),%rax        # 4f0c0 <_ZN30codspeed_divan_compat_walltime11thread_pool19TaskShared$LT$F$GT$3new4call17h8d3e3abb24b6f223E>
   4b5ef:	mov    %rax,0x130(%rsp)
   4b5f7:	lea    0x240(%rsp),%rax
   4b5ff:	mov    %rax,0x138(%rsp)
   4b607:	mov    %r15,0x140(%rsp)
   4b60f:	mov    0xf0cea(%rip),%rdi        # 13c300 <_DYNAMIC+0x270>
   4b616:	xor    %esi,%esi
   4b618:	lea    0x120(%rsp),%rdx
   4b620:	call   *0xf0ce2(%rip)        # 13c308 <_DYNAMIC+0x278>
   4b626:	mov    0x120(%rsp),%rax
   4b62e:	lock decq (%rax)
   4b632:	jne    4b642 <_ZN15funnel_patterns29pat_chunked_aligned__u64__w5117h3114e5c745d20737E+0x842>
   4b634:	lea    0x120(%rsp),%rdi
   4b63c:	call   *0xf0cce(%rip)        # 13c310 <_DYNAMIC+0x280>
   4b642:	cmpl   $0x3b9aca01,0x8(%r15)
   4b64a:	je     4bd3f <_ZN15funnel_patterns29pat_chunked_aligned__u64__w5117h3114e5c745d20737E+0xf3f>
   4b650:	cmpb   $0x1,0x3(%rsp)
   4b655:	je     4bd08 <_ZN15funnel_patterns29pat_chunked_aligned__u64__w5117h3114e5c745d20737E+0xf08>
   4b65b:	vmovups 0x10(%r15),%xmm0
   4b661:	vmovaps %xmm0,0x1a0(%rsp)
   4b66a:	vmovdqu (%r15),%xmm0
   4b66f:	vmovdqa %xmm0,0x120(%rsp)
   4b678:	mov    0xc0(%r15),%rdx
   4b67f:	lea    0x1a0(%rsp),%rdi
   4b687:	lea    0x120(%rsp),%rsi
   4b68f:	call   *0xf0c83(%rip)        # 13c318 <_DYNAMIC+0x288>
   4b695:	lea    0x10(%r15),%rax
   4b699:	mov    %rax,%r12
   4b69c:	vmovups (%rax),%xmm0
   4b6a0:	vmovaps %xmm0,0x1a0(%rsp)
   4b6a9:	vmovdqu (%r15),%xmm0
   4b6ae:	vmovdqa %xmm0,0x120(%rsp)
   4b6b7:	mov    0xc0(%r15),%rdx
   4b6be:	lea    0x1a0(%rsp),%rdi
   4b6c6:	lea    0x120(%rsp),%rsi
   4b6ce:	call   *0xf0c44(%rip)        # 13c318 <_DYNAMIC+0x288>
   4b6d4:	mov    %rax,%rsi
   4b6d7:	cmp    $0x1,%ebp
   4b6da:	mov    %rdx,0x98(%rsp)
   4b6e2:	mov    %rax,0x90(%rsp)
   4b6ea:	jne    4b760 <_ZN15funnel_patterns29pat_chunked_aligned__u64__w5117h3114e5c745d20737E+0x960>
   4b6ec:	mov    %rbx,%rcx
   4b6ef:	movq   $0x0,0x10(%rbx)
   4b6f7:	cmpq   $0x0,0x30(%rbx)
   4b6fc:	mov    0x58(%rsp),%r13
   4b701:	mov    0x50(%rsp),%rbp
   4b706:	je     4b786 <_ZN15funnel_patterns29pat_chunked_aligned__u64__w5117h3114e5c745d20737E+0x986>
   4b708:	mov    0x20(%rcx),%r14
   4b70c:	test   %r14,%r14
   4b70f:	je     4b775 <_ZN15funnel_patterns29pat_chunked_aligned__u64__w5117h3114e5c745d20737E+0x975>
   4b711:	mov    0x78(%rsp),%rax
   4b716:	mov    (%rax),%rdi
   4b719:	lea    0x11(%r14),%rdx
   4b71d:	mov    $0xff,%esi
   4b722:	call   *0xf0b98(%rip)        # 13c2c0 <memset@GLIBC_2.2.5>
   4b728:	mov    0x90(%rsp),%rsi
   4b730:	mov    0x98(%rsp),%rdx
   4b738:	lea    0x1(%r14),%rax
   4b73c:	mov    %rax,%rcx
   4b73f:	shr    $0x3,%rcx
   4b743:	and    $0xfffffffffffffff8,%rax
   4b747:	sub    %rcx,%rax
   4b74a:	cmp    $0x8,%r14
   4b74e:	cmovb  %r14,%rax
   4b752:	jmp    4b777 <_ZN15funnel_patterns29pat_chunked_aligned__u64__w5117h3114e5c745d20737E+0x977>
   4b754:	data16 data16 cs nopw 0x0(%rax,%rax,1)
   4b760:	mov    %ebp,0x14(%rsp)
   4b764:	mov    0x4(%rsp),%eax
   4b768:	mov    %rax,0xa0(%rsp)
   4b770:	jmp    4b817 <_ZN15funnel_patterns29pat_chunked_aligned__u64__w5117h3114e5c745d20737E+0xa17>
   4b775:	xor    %eax,%eax
   4b777:	mov    %rbx,%rcx
   4b77a:	movq   $0x0,0x30(%rbx)
   4b782:	mov    %rax,0x28(%rbx)
   4b786:	cmpq   $0x0,0x68(%rcx)
   4b78b:	je     4b795 <_ZN15funnel_patterns29pat_chunked_aligned__u64__w5117h3114e5c745d20737E+0x995>
   4b78d:	movq   $0x0,0x60(%rcx)
   4b795:	cmpq   $0x0,0x90(%rcx)
   4b79d:	je     4b7aa <_ZN15funnel_patterns29pat_chunked_aligned__u64__w5117h3114e5c745d20737E+0x9aa>
   4b79f:	movq   $0x0,0x88(%rcx)
   4b7aa:	cmpq   $0x0,0xb8(%rcx)
   4b7b2:	je     4b7bf <_ZN15funnel_patterns29pat_chunked_aligned__u64__w5117h3114e5c745d20737E+0x9bf>
   4b7b4:	movq   $0x0,0xb0(%rcx)
   4b7bf:	cmpq   $0x0,0xe0(%rcx)
   4b7c7:	je     4b7d4 <_ZN15funnel_patterns29pat_chunked_aligned__u64__w5117h3114e5c745d20737E+0x9d4>
   4b7c9:	movq   $0x0,0xd8(%rcx)
   4b7d4:	mov    %r13,%rax
   4b7d7:	or     %rbp,%rax
   4b7da:	je     4bdd3 <_ZN15funnel_patterns29pat_chunked_aligned__u64__w5117h3114e5c745d20737E+0xfd3>
   4b7e0:	mov    %rsi,%rdi
   4b7e3:	mov    %rdx,%rsi
   4b7e6:	mov    %r13,%rdx
   4b7e9:	mov    %rbp,%rcx
   4b7ec:	call   *0xf0b2e(%rip)        # 13c320 <_DYNAMIC+0x290>
   4b7f2:	cmp    $0x65,%rax
   4b7f6:	sbb    $0x0,%rdx
   4b7fa:	mov    0x4(%rsp),%ecx
   4b7fe:	mov    %rcx,0xa0(%rsp)
   4b806:	jae    4b830 <_ZN15funnel_patterns29pat_chunked_aligned__u64__w5117h3114e5c745d20737E+0xa30>
   4b808:	lea    (%rcx,%rcx,1),%eax
   4b80b:	mov    %eax,0x8(%rsp)
   4b80f:	movl   $0x1,0x14(%rsp)
   4b817:	mov    0x18(%rsp),%eax
   4b81b:	mov    %eax,0x10(%rsp)
   4b81f:	mov    %r12,%rdx
   4b822:	jmp    4b86d <_ZN15funnel_patterns29pat_chunked_aligned__u64__w5117h3114e5c745d20737E+0xa6d>
   4b824:	data16 data16 cs nopw 0x0(%rax,%rax,1)
   4b830:	mov    0xf8(%rbx),%rax
   4b837:	movl   $0x1,0xc(%rsp)
   4b83f:	cmpb   $0x0,0x58(%rax)
   4b843:	mov    %r12,%rdx
   4b846:	je     4b859 <_ZN15funnel_patterns29pat_chunked_aligned__u64__w5117h3114e5c745d20737E+0xa59>
   4b848:	movl   $0x2,0x14(%rsp)
   4b850:	mov    0x5c(%rax),%eax
   4b853:	mov    %eax,0x10(%rsp)
   4b857:	jmp    4b869 <_ZN15funnel_patterns29pat_chunked_aligned__u64__w5117h3114e5c745d20737E+0xa69>
   4b859:	movl   $0x64,0x10(%rsp)
   4b861:	movl   $0x2,0x14(%rsp)
   4b869:	mov    %ecx,0x8(%rsp)
   4b86d:	mov    0xa8(%rsp),%rax
   4b875:	mov    (%rax),%rbp
   4b878:	mov    0x8(%rax),%r13
   4b87c:	mov    0x18(%rax),%r14
   4b880:	mov    0x10(%rax),%rcx
   4b884:	mov    %rcx,0xc0(%rsp)
   4b88c:	mov    %r15,%rcx
   4b88f:	mov    0x40(%r15),%r15
   4b893:	mov    0x28(%rax),%rsi
   4b897:	mov    %rsi,0xc8(%rsp)
   4b89f:	mov    0x20(%rax),%rsi
   4b8a3:	mov    %rsi,0xd0(%rsp)
   4b8ab:	mov    0x50(%rcx),%rbx
   4b8af:	mov    0x38(%rax),%rsi
   4b8b3:	mov    %rsi,0xe0(%rsp)
   4b8bb:	mov    0x30(%rax),%rax
   4b8bf:	mov    %rax,0x18(%rsp)
   4b8c4:	mov    0x20(%rcx),%rax
   4b8c8:	mov    %rax,0xd8(%rsp)
   4b8d0:	mov    0x30(%rcx),%r12
   4b8d4:	vmovups (%rdx),%xmm0
   4b8d8:	vmovaps %xmm0,0x1a0(%rsp)
   4b8e1:	mov    0x30(%rsp),%rax
   4b8e6:	mov    0x10(%rax),%rax
   4b8ea:	mov    %rax,0xb8(%rsp)
   4b8f2:	vmovdqu (%rcx),%xmm0
   4b8f6:	vmovdqa %xmm0,0x120(%rsp)
   4b8ff:	mov    0xc0(%rcx),%rdx
   4b906:	lea    0x1a0(%rsp),%rdi
   4b90e:	lea    0x120(%rsp),%rsi
   4b916:	call   *0xf09fc(%rip)        # 13c318 <_DYNAMIC+0x288>
   4b91c:	mov    %rax,%rsi
   4b91f:	mov    %rdx,%rcx
   4b922:	mov    0xa0(%rsp),%edi
   4b929:	mov    %r13,%rax
   4b92c:	mul    %rdi
   4b92f:	mov    %rbp,%rdx
   4b932:	mulx   %rdi,%r10,%r9
   4b937:	seto   %dl
   4b93a:	add    %rax,%r9
   4b93d:	setb   %r13b
   4b941:	or     %dl,%r13b
   4b944:	mov    %r14,%rax
   4b947:	mul    %r15
   4b94a:	seto   %r11b
   4b94e:	mov    0xc0(%rsp),%rdx
   4b956:	mulx   %r15,%r8,%rdi
   4b95b:	add    %rax,%rdi
   4b95e:	setb   %r14b
   4b962:	or     %r11b,%r14b
   4b965:	mov    0xc8(%rsp),%rax
   4b96d:	mul    %rbx
   4b970:	mov    0xd0(%rsp),%rdx
   4b978:	mulx   %rbx,%rbx,%r11
   4b97d:	seto   %dl
   4b980:	add    %rax,%r11
   4b983:	setb   %bpl
   4b987:	or     %dl,%bpl
   4b98a:	or     %r14b,%bpl
   4b98d:	or     %r13b,%bpl
   4b990:	xor    %r14d,%r14d
   4b993:	add    0xd8(%rsp),%r12
   4b99b:	setb   %r14b
   4b99f:	mov    0xe0(%rsp),%rax
   4b9a7:	test   %rax,%rax
   4b9aa:	setne  %dl
   4b9ad:	mov    %r14d,%r15d
   4b9b0:	and    %dl,%r15b
   4b9b3:	mul    %r12
   4b9b6:	seto   %r13b
   4b9ba:	or     %r15b,%r13b
   4b9bd:	mov    0x18(%rsp),%rdx
   4b9c2:	imul   %rdx,%r14
   4b9c6:	add    %rax,%r14
   4b9c9:	mulx   %r12,%rdx,%rax
   4b9ce:	add    %r14,%rax
   4b9d1:	setb   %r14b
   4b9d5:	or     %r13b,%r14b
   4b9d8:	or     %bpl,%r14b
   4b9db:	add    %r10,%r8
   4b9de:	adc    %r9,%rdi
   4b9e1:	mov    $0xffffffffffffffff,%r15
   4b9e8:	cmovb  %r15,%rdi
   4b9ec:	cmovb  %r15,%r8
   4b9f0:	add    %rbx,%r8
   4b9f3:	adc    %r11,%rdi
   4b9f6:	cmovb  %r15,%rdi
   4b9fa:	cmovb  %r15,%r8
   4b9fe:	mov    $0xffffffffffffffff,%r10
   4ba05:	mov    $0xffffffffffffffff,%r9
   4ba0c:	test   $0x1,%r14b
   4ba10:	jne    4ba26 <_ZN15funnel_patterns29pat_chunked_aligned__u64__w5117h3114e5c745d20737E+0xc26>
   4ba12:	add    %rdx,%r8
   4ba15:	adc    %rax,%rdi
   4ba18:	cmovb  %r15,%rdi
   4ba1c:	cmovb  %r15,%r8
   4ba20:	mov    %r8,%r10
   4ba23:	mov    %rdi,%r9
   4ba26:	mov    %rsi,%rax
   4ba29:	or     %rcx,%rax
   4ba2c:	mov    0x50(%rsp),%rdx
   4ba31:	cmove  %rdx,%rcx
   4ba35:	mov    0x58(%rsp),%rax
   4ba3a:	cmove  %rax,%rsi
   4ba3e:	mov    %rsi,%rbx
   4ba41:	sub    %r10,%rbx
   4ba44:	mov    %rcx,%r14
   4ba47:	sbb    %r9,%r14
   4ba4a:	cmp    %rsi,%r10
   4ba4d:	sbb    %rcx,%r9
   4ba50:	cmovae %rdx,%r14
   4ba54:	cmovae %rax,%rbx
   4ba58:	mov    0x30(%rsp),%rax
   4ba5d:	mov    0x10(%rax),%r15
   4ba61:	cmp    (%rax),%r15
   4ba64:	jne    4ba70 <_ZN15funnel_patterns29pat_chunked_aligned__u64__w5117h3114e5c745d20737E+0xc70>
   4ba66:	mov    0x30(%rsp),%rdi
   4ba6b:	call   51730 <_ZN5alloc7raw_vec19RawVec$LT$T$C$A$GT$8grow_one17hfe527167958a2dadE>
   4ba70:	mov    0x30(%rsp),%r12
   4ba75:	mov    0x8(%r12),%rax
   4ba7a:	mov    %r15,%rcx
   4ba7d:	shl    $0x4,%rcx
   4ba81:	mov    %r14,0x8(%rax,%rcx,1)
   4ba86:	mov    %rbx,(%rax,%rcx,1)
   4ba8a:	inc    %r15
   4ba8d:	mov    %r15,0x10(%r12)
   4ba92:	mov    0x40(%rsp),%r15
   4ba97:	mov    0x28(%r15),%rax
   4ba9b:	or     0x20(%r15),%rax
   4ba9f:	jne    4bac0 <_ZN15funnel_patterns29pat_chunked_aligned__u64__w5117h3114e5c745d20737E+0xcc0>
   4baa1:	mov    0x38(%r15),%rax
   4baa5:	or     0x30(%r15),%rax
   4baa9:	jne    4bac0 <_ZN15funnel_patterns29pat_chunked_aligned__u64__w5117h3114e5c745d20737E+0xcc0>
   4baab:	mov    0x48(%r15),%rax
   4baaf:	or     0x40(%r15),%rax
   4bab3:	jne    4bac0 <_ZN15funnel_patterns29pat_chunked_aligned__u64__w5117h3114e5c745d20737E+0xcc0>
   4bab5:	mov    0x58(%r15),%rax
   4bab9:	or     0x50(%r15),%rax
   4babd:	je     4bb16 <_ZN15funnel_patterns29pat_chunked_aligned__u64__w5117h3114e5c745d20737E+0xd16>
   4babf:	nop
   4bac0:	lea    0x20(%r15),%rax
   4bac4:	vmovdqu64 (%rax),%zmm0
   4baca:	vmovdqu64 0x20(%rax),%zmm1
   4bad4:	vmovdqu64 %zmm1,0x1c0(%rsp)
   4badc:	vmovdqu64 %zmm0,0x1a0(%rsp)
   4bae7:	lea    0x120(%rsp),%rdi
   4baef:	mov    0x78(%rsp),%rsi
   4baf4:	mov    0xb8(%rsp),%rdx
   4bafc:	lea    0x1a0(%rsp),%rcx
   4bb04:	vzeroupper
   4bb07:	call   51900 <_ZN9hashbrown3map28HashMap$LT$K$C$V$C$S$C$A$GT$6insert17he36ed9c6ee1bc3e6E>
   4bb0c:	mov    0x30(%rsp),%r12
   4bb11:	mov    0x40(%rsp),%r15
   4bb16:	cmpq   $0x0,0x68(%r12)
   4bb1c:	mov    0x14(%rsp),%ebp
   4bb20:	mov    0x6c(%rsp),%r13d
   4bb25:	mov    0xf07fc(%rip),%rbx        # 13c328 <_DYNAMIC+0x298>
   4bb2c:	mov    0xe8(%rsp),%r14
   4bb34:	je     4bb66 <_ZN15funnel_patterns29pat_chunked_aligned__u64__w5117h3114e5c745d20737E+0xd66>
   4bb36:	mov    0x4(%rsp),%eax
   4bb3a:	test   %eax,%eax
   4bb3c:	je     4bd2d <_ZN15funnel_patterns29pat_chunked_aligned__u64__w5117h3114e5c745d20737E+0xf2d>
   4bb42:	mov    %eax,%edx
   4bb44:	mov    0x80(%r15),%rdi
   4bb4b:	mov    0x88(%r15),%rsi
   4bb52:	xor    %ecx,%ecx
   4bb54:	call   *0xf07c6(%rip)        # 13c320 <_DYNAMIC+0x290>
   4bb5a:	mov    0x28(%rsp),%rdi
   4bb5f:	mov    %rax,%rsi
   4bb62:	xor    %edx,%edx
   4bb64:	call   *%rbx
   4bb66:	cmpq   $0x0,0x90(%r12)
   4bb6f:	je     4bba4 <_ZN15funnel_patterns29pat_chunked_aligned__u64__w5117h3114e5c745d20737E+0xda4>
   4bb71:	mov    0x4(%rsp),%eax
   4bb75:	test   %eax,%eax
   4bb77:	je     4bd2d <_ZN15funnel_patterns29pat_chunked_aligned__u64__w5117h3114e5c745d20737E+0xf2d>
   4bb7d:	mov    %eax,%edx
   4bb7f:	mov    0x90(%r15),%rdi
   4bb86:	mov    0x98(%r15),%rsi
   4bb8d:	xor    %ecx,%ecx
   4bb8f:	call   *0xf078b(%rip)        # 13c320 <_DYNAMIC+0x290>
   4bb95:	mov    0x28(%rsp),%rdi
   4bb9a:	mov    %rax,%rsi
   4bb9d:	mov    $0x1,%edx
   4bba2:	call   *%rbx
   4bba4:	cmpq   $0x0,0xb8(%r12)
   4bbad:	je     4bbe2 <_ZN15funnel_patterns29pat_chunked_aligned__u64__w5117h3114e5c745d20737E+0xde2>
   4bbaf:	mov    0x4(%rsp),%eax
   4bbb3:	test   %eax,%eax
   4bbb5:	je     4bd2d <_ZN15funnel_patterns29pat_chunked_aligned__u64__w5117h3114e5c745d20737E+0xf2d>
   4bbbb:	mov    %eax,%edx
   4bbbd:	mov    0xa0(%r15),%rdi
   4bbc4:	mov    0xa8(%r15),%rsi
   4bbcb:	xor    %ecx,%ecx
   4bbcd:	call   *0xf074d(%rip)        # 13c320 <_DYNAMIC+0x290>
   4bbd3:	mov    0x28(%rsp),%rdi
   4bbd8:	mov    %rax,%rsi
   4bbdb:	mov    $0x2,%edx
   4bbe0:	call   *%rbx
   4bbe2:	cmpq   $0x0,0xe0(%r12)
   4bbeb:	je     4bc20 <_ZN15funnel_patterns29pat_chunked_aligned__u64__w5117h3114e5c745d20737E+0xe20>
   4bbed:	mov    0x4(%rsp),%eax
   4bbf1:	test   %eax,%eax
   4bbf3:	je     4bd2d <_ZN15funnel_patterns29pat_chunked_aligned__u64__w5117h3114e5c745d20737E+0xf2d>
   4bbf9:	mov    %eax,%edx
   4bbfb:	mov    0xb0(%r15),%rdi
   4bc02:	mov    0xb8(%r15),%rsi
   4bc09:	xor    %ecx,%ecx
   4bc0b:	call   *0xf070f(%rip)        # 13c320 <_DYNAMIC+0x290>
   4bc11:	mov    0x28(%rsp),%rdi
   4bc16:	mov    %rax,%rsi
   4bc19:	mov    $0x3,%edx
   4bc1e:	call   *%rbx
   4bc20:	mov    0x10(%rsp),%ecx
   4bc24:	mov    %ecx,%edx
   4bc26:	sub    $0x1,%edx
   4bc29:	mov    $0x0,%eax
   4bc2e:	cmovb  %eax,%edx
   4bc31:	testb  $0x1,0xc(%rsp)
   4bc36:	cmove  %ecx,%edx
   4bc39:	mov    %edx,0x18(%rsp)
   4bc3d:	cmp    $0x3b9aca01,%r13d
   4bc44:	je     4b2e0 <_ZN15funnel_patterns29pat_chunked_aligned__u64__w5117h3114e5c745d20737E+0x4e0>
   4bc4a:	mov    0x70(%rsp),%rax
   4bc4f:	mov    %rax,0x190(%rsp)
   4bc57:	mov    %r13d,0x198(%rsp)
   4bc5f:	mov    0x18(%r15),%eax
   4bc63:	cmp    $0x3b9aca01,%eax
   4bc68:	je     4bdc4 <_ZN15funnel_patterns29pat_chunked_aligned__u64__w5117h3114e5c745d20737E+0xfc4>
   4bc6e:	mov    0x1c(%r15),%ecx
   4bc72:	mov    0x10(%r15),%rdx
   4bc76:	mov    %rdx,0x110(%rsp)
   4bc7e:	mov    %eax,0x118(%rsp)
   4bc85:	mov    %ecx,0x11c(%rsp)
   4bc8c:	mov    0x60(%rsp),%rdx
   4bc91:	lea    0x110(%rsp),%rdi
   4bc99:	lea    0x190(%rsp),%rsi
   4bca1:	call   *0xf0671(%rip)        # 13c318 <_DYNAMIC+0x288>
   4bca7:	mov    %rdx,%rsi
   4bcaa:	mov    %rax,%rdx
   4bcad:	jmp    4b330 <_ZN15funnel_patterns29pat_chunked_aligned__u64__w5117h3114e5c745d20737E+0x530>
   4bcb2:	mov    $0x10,%r15d
   4bcb8:	xor    %r14d,%r14d
   4bcbb:	mov    0xf0626(%rip),%rbx        # 13c2e8 <_DYNAMIC+0x258>
   4bcc2:	mov    0x8(%rbx),%eax
   4bcc5:	test   %eax,%eax
   4bcc7:	jne    4bd1c <_ZN15funnel_patterns29pat_chunked_aligned__u64__w5117h3114e5c745d20737E+0xf1c>
   4bcc9:	mov    (%rbx),%rdi
   4bccc:	call   *0xf065e(%rip)        # 13c330 <_DYNAMIC+0x2a0>
   4bcd2:	mov    0xf065f(%rip),%rax        # 13c338 <_DYNAMIC+0x2a8>
   4bcd9:	movb   $0x0,(%rax)
   4bcdc:	test   %r14,%r14
   4bcdf:	je     4bcf6 <_ZN15funnel_patterns29pat_chunked_aligned__u64__w5117h3114e5c745d20737E+0xef6>
   4bce1:	imul   $0xd0,%r14,%rsi
   4bce8:	mov    $0x10,%edx
   4bced:	mov    %r15,%rdi
   4bcf0:	call   *0xf064a(%rip)        # 13c340 <_DYNAMIC+0x2b0>
   4bcf6:	add    $0x3c18,%rsp
   4bcfd:	pop    %rbx
   4bcfe:	pop    %r12
   4bd00:	pop    %r13
   4bd02:	pop    %r14
   4bd04:	pop    %r15
   4bd06:	pop    %rbp
   4bd07:	ret
   4bd08:	mov    $0x1,%r14d
   4bd0e:	mov    0xf05d3(%rip),%rbx        # 13c2e8 <_DYNAMIC+0x258>
   4bd15:	mov    0x8(%rbx),%eax
   4bd18:	test   %eax,%eax
   4bd1a:	je     4bcc9 <_ZN15funnel_patterns29pat_chunked_aligned__u64__w5117h3114e5c745d20737E+0xec9>
   4bd1c:	call   5152e <_ZN3std4sync9once_lock17OnceLock$LT$T$GT$10initialize17h13c8d01c1aad6442E>
   4bd21:	jmp    4bcc9 <_ZN15funnel_patterns29pat_chunked_aligned__u64__w5117h3114e5c745d20737E+0xec9>
   4bd23:	call   5152e <_ZN3std4sync9once_lock17OnceLock$LT$T$GT$10initialize17h13c8d01c1aad6442E>
   4bd28:	jmp    4b28b <_ZN15funnel_patterns29pat_chunked_aligned__u64__w5117h3114e5c745d20737E+0x48b>
   4bd2d:	lea    0xe7df4(%rip),%rdi        # 133b28 <_ZN15funnel_patterns46__DIVAN_BENCH_PAT_U128_CAT_UNROLLED4__U64__W514PUSH17h68d55424f84a52b4E+0xe0>
   4bd34:	call   *0xf060e(%rip)        # 13c348 <_DYNAMIC+0x2b8>
   4bd3a:	jmp    4bde0 <_ZN15funnel_patterns29pat_chunked_aligned__u64__w5117h3114e5c745d20737E+0xfe0>
   4bd3f:	movq   $0x0,0x108(%rsp)
   4bd4b:	lea    0x108(%rsp),%rax
   4bd53:	mov    %rax,0x1a0(%rsp)
   4bd5b:	mov    0xf05ee(%rip),%rax        # 13c350 <_DYNAMIC+0x2c0>
   4bd62:	mov    %rax,0x1a8(%rsp)
   4bd6a:	lea    0xe7d37(%rip),%rax        # 133aa8 <_ZN15funnel_patterns46__DIVAN_BENCH_PAT_U128_CAT_UNROLLED4__U64__W514PUSH17h68d55424f84a52b4E+0x60>
   4bd71:	mov    %rax,0x120(%rsp)
   4bd79:	movq   $0x2,0x128(%rsp)
   4bd85:	movq   $0x0,0x140(%rsp)
   4bd91:	lea    0x1a0(%rsp),%rax
   4bd99:	mov    %rax,0x130(%rsp)
   4bda1:	movq   $0x1,0x138(%rsp)
   4bdad:	lea    0xe7d14(%rip),%rsi        # 133ac8 <_ZN15funnel_patterns46__DIVAN_BENCH_PAT_U128_CAT_UNROLLED4__U64__W514PUSH17h68d55424f84a52b4E+0x80>
   4bdb4:	lea    0x120(%rsp),%rdi
   4bdbc:	call   *0xf0596(%rip)        # 13c358 <_DYNAMIC+0x2c8>
   4bdc2:	jmp    4bde0 <_ZN15funnel_patterns29pat_chunked_aligned__u64__w5117h3114e5c745d20737E+0xfe0>
   4bdc4:	lea    0xe7d2d(%rip),%rdi        # 133af8 <_ZN15funnel_patterns46__DIVAN_BENCH_PAT_U128_CAT_UNROLLED4__U64__W514PUSH17h68d55424f84a52b4E+0xb0>
   4bdcb:	call   *0xf058f(%rip)        # 13c360 <_DYNAMIC+0x2d0>
   4bdd1:	jmp    4bde0 <_ZN15funnel_patterns29pat_chunked_aligned__u64__w5117h3114e5c745d20737E+0xfe0>
   4bdd3:	lea    0xe7d06(%rip),%rdi        # 133ae0 <_ZN15funnel_patterns46__DIVAN_BENCH_PAT_U128_CAT_UNROLLED4__U64__W514PUSH17h68d55424f84a52b4E+0x98>
   4bdda:	call   *0xf0568(%rip)        # 13c348 <_DYNAMIC+0x2b8>
   4bde0:	ud2
   4bde2:	mov    0x1a8(%rsp),%rdi
   4bdea:	mov    0x1b0(%rsp),%rsi
   4bdf2:	lea    0xe7c97(%rip),%rdx        # 133a90 <_ZN15funnel_patterns46__DIVAN_BENCH_PAT_U128_CAT_UNROLLED4__U64__W514PUSH17h68d55424f84a52b4E+0x48>
   4bdf9:	call   *0xf0569(%rip)        # 13c368 <_DYNAMIC+0x2d8>
   4bdff:	mov    %r15,0x40(%rsp)
   4be04:	mov    %rax,%rbx
   4be07:	test   %r14,%r14
   4be0a:	jne    4be40 <_ZN15funnel_patterns29pat_chunked_aligned__u64__w5117h3114e5c745d20737E+0x1040>
   4be0c:	jmp    4be55 <_ZN15funnel_patterns29pat_chunked_aligned__u64__w5117h3114e5c745d20737E+0x1055>
   4be0e:	mov    %rax,%rbx
   4be11:	mov    0x120(%rsp),%rax
   4be19:	lock decq (%rax)
   4be1d:	jne    4be40 <_ZN15funnel_patterns29pat_chunked_aligned__u64__w5117h3114e5c745d20737E+0x1040>
   4be1f:	lea    0x120(%rsp),%rdi
   4be27:	call   *0xf04e3(%rip)        # 13c310 <_DYNAMIC+0x280>
   4be2d:	jmp    4be40 <_ZN15funnel_patterns29pat_chunked_aligned__u64__w5117h3114e5c745d20737E+0x1040>
   4be2f:	call   *0xf053b(%rip)        # 13c370 <_DYNAMIC+0x2e0>
   4be35:	jmp    4be3d <_ZN15funnel_patterns29pat_chunked_aligned__u64__w5117h3114e5c745d20737E+0x103d>
   4be37:	jmp    4be3d <_ZN15funnel_patterns29pat_chunked_aligned__u64__w5117h3114e5c745d20737E+0x103d>
   4be39:	jmp    4be3d <_ZN15funnel_patterns29pat_chunked_aligned__u64__w5117h3114e5c745d20737E+0x103d>
   4be3b:	jmp    4be3d <_ZN15funnel_patterns29pat_chunked_aligned__u64__w5117h3114e5c745d20737E+0x103d>
   4be3d:	mov    %rax,%rbx
   4be40:	mov    $0xd0,%esi
   4be45:	mov    $0x10,%edx
   4be4a:	mov    0x40(%rsp),%rdi
   4be4f:	call   *0xf04eb(%rip)        # 13c340 <_DYNAMIC+0x2b0>
   4be55:	mov    %rbx,%rdi
   4be58:	call   1328b0 <_Unwind_Resume@plt>

Disassembly of section .init:

Disassembly of section .fini:

Disassembly of section .plt:
