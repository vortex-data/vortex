
/home/user/vortex/target/release/deps/funnel_patterns-21c1c00107f42b8a:     file format elf64-x86-64


Disassembly of section .text:

0000000000049da0 <_ZN15funnel_patterns25pat_macro_shape__u64__w5117he54d8f6ffc7df352E>:
   49da0:	push   %rbp
   49da1:	push   %r15
   49da3:	push   %r14
   49da5:	push   %r13
   49da7:	push   %r12
   49da9:	push   %rbx
   49daa:	sub    $0x1000,%rsp
   49db1:	movq   $0x0,(%rsp)
   49db9:	sub    $0x1000,%rsp
   49dc0:	movq   $0x0,(%rsp)
   49dc8:	sub    $0x1000,%rsp
   49dcf:	movq   $0x0,(%rsp)
   49dd7:	sub    $0xc18,%rsp
   49dde:	mov    %rdi,%r12
   49de1:	lea    0x298(%rsp),%rdi
   49de9:	mov    $0x1980,%edx
   49dee:	xor    %esi,%esi
   49df0:	call   *0xf24ca(%rip)        # 13c2c0 <memset@GLIBC_2.2.5>
   49df6:	vmovdqa64 -0x3a0c0(%rip),%zmm0        # fd40 <__abi_tag+0xfa44>
   49e00:	mov    $0x38,%eax
   49e05:	vpbroadcastq -0x3967f(%rip),%zmm1        # 10790 <__abi_tag+0x10494>
   49e0f:	vpbroadcastq -0x397a9(%rip),%zmm2        # 10670 <__abi_tag+0x10374>
   49e19:	vpbroadcastq -0x3968b(%rip),%zmm3        # 10798 <__abi_tag+0x1049c>
   49e23:	vpbroadcastq -0x3960d(%rip),%zmm4        # 10820 <__abi_tag+0x10524>
   49e2d:	vpbroadcastq -0x395e7(%rip),%zmm5        # 10850 <__abi_tag+0x10554>
   49e37:	vpbroadcastq -0x397b1(%rip),%zmm6        # 10690 <__abi_tag+0x10394>
   49e41:	vpbroadcastq -0x395c3(%rip),%zmm7        # 10888 <__abi_tag+0x1058c>
   49e4b:	vpbroadcastq -0x395fd(%rip),%zmm8        # 10858 <__abi_tag+0x1055c>
   49e55:	vpbroadcastq -0x395df(%rip),%zmm9        # 10880 <__abi_tag+0x10584>
   49e5f:	nop
   49e60:	vpmullq %zmm1,%zmm0,%zmm10
   49e66:	vpaddq %zmm2,%zmm10,%zmm11
   49e6c:	vpaddq %zmm3,%zmm10,%zmm12
   49e72:	vmovdqu64 %zmm10,0xd8(%rsp,%rax,8)
   49e7d:	vmovdqu64 %zmm11,0x118(%rsp,%rax,8)
   49e88:	vpaddq %zmm4,%zmm10,%zmm11
   49e8e:	vmovdqu64 %zmm12,0x158(%rsp,%rax,8)
   49e99:	vmovdqu64 %zmm11,0x198(%rsp,%rax,8)
   49ea4:	cmp    $0x338,%rax
   49eaa:	je     49eff <_ZN15funnel_patterns25pat_macro_shape__u64__w5117he54d8f6ffc7df352E+0x15f>
   49eac:	vpaddq %zmm5,%zmm10,%zmm11
   49eb2:	vpaddq %zmm6,%zmm10,%zmm12
   49eb8:	vpaddq %zmm7,%zmm10,%zmm13
   49ebe:	vpaddq %zmm8,%zmm10,%zmm10
   49ec4:	vmovdqu64 %zmm11,0x1d8(%rsp,%rax,8)
   49ecf:	vmovdqu64 %zmm12,0x218(%rsp,%rax,8)
   49eda:	vmovdqu64 %zmm13,0x258(%rsp,%rax,8)
   49ee5:	vmovdqu64 %zmm10,0x298(%rsp,%rax,8)
   49ef0:	vpaddq %zmm9,%zmm0,%zmm0
   49ef6:	add    $0x40,%rax
   49efa:	jmp    49e60 <_ZN15funnel_patterns25pat_macro_shape__u64__w5117he54d8f6ffc7df352E+0xc0>
   49eff:	vmovaps -0x3a189(%rip),%zmm0        # fd80 <__abi_tag+0xfa84>
   49f09:	vmovups %zmm0,0x1b98(%rsp)
   49f14:	vmovdqa64 -0x3a15e(%rip),%zmm0        # fdc0 <__abi_tag+0xfac4>
   49f1e:	vmovdqu64 %zmm0,0x1bd8(%rsp)
   49f29:	lea    0x2298(%rsp),%rbx
   49f31:	lea    0x298(%rsp),%r14
   49f39:	mov    $0x1980,%edx
   49f3e:	mov    %rbx,%rdi
   49f41:	mov    %r14,%rsi
   49f44:	vzeroupper
   49f47:	call   *0xf237b(%rip)        # 13c2c8 <memcpy@GLIBC_2.14>
   49f4d:	movq   $0x3b9aca07,0xf0(%rsp)
   49f59:	xor    %ebp,%ebp
   49f5b:	mov    $0x2000,%edx
   49f60:	mov    %r14,%rdi
   49f63:	xor    %esi,%esi
   49f65:	call   *0xf2355(%rip)        # 13c2c0 <memset@GLIBC_2.2.5>
   49f6b:	mov    %rbx,0x208(%rsp)
   49f73:	lea    0xf0(%rsp),%rax
   49f7b:	mov    %rax,0x210(%rsp)
   49f83:	mov    %r14,0x218(%rsp)
   49f8b:	lea    0x208(%rsp),%rax
   49f93:	mov    %rax,0xf8(%rsp)
   49f9b:	lea    0xf8(%rsp),%rax
   49fa3:	mov    %rax,0x100(%rsp)
   49fab:	movq   $0x1,0x100(%r12)
   49fb7:	movb   $0x1,0x108(%r12)
   49fc0:	mov    0xf0(%r12),%rdx
   49fc8:	mov    0xf8(%r12),%rax
   49fd0:	movzbl 0x8(%rdx),%ecx
   49fd4:	mov    %cl,0x3(%rsp)
   49fd8:	cmp    $0x1,%cl
   49fdb:	jne    49fe3 <_ZN15funnel_patterns25pat_macro_shape__u64__w5117he54d8f6ffc7df352E+0x243>
   49fdd:	xor    %esi,%esi
   49fdf:	xor    %ecx,%ecx
   49fe1:	jmp    4a00d <_ZN15funnel_patterns25pat_macro_shape__u64__w5117he54d8f6ffc7df352E+0x26d>
   49fe3:	cmpb   $0x0,0x60(%rax)
   49fe7:	je     49ffc <_ZN15funnel_patterns25pat_macro_shape__u64__w5117he54d8f6ffc7df352E+0x25c>
   49fe9:	mov    0x64(%rax),%ecx
   49fec:	mov    %ecx,0x8(%rsp)
   49ff0:	mov    $0x2,%ebp
   49ff5:	mov    $0x1,%sil
   49ff8:	xor    %ecx,%ecx
   49ffa:	jmp    4a00d <_ZN15funnel_patterns25pat_macro_shape__u64__w5117he54d8f6ffc7df352E+0x26d>
   49ffc:	mov    $0x1,%cl
   49ffe:	movl   $0x1,0x8(%rsp)
   4a006:	xor    %esi,%esi
   4a008:	mov    $0x1,%ebp
   4a00d:	mov    (%rdx),%r14
   4a010:	test   %r14,%r14
   4a013:	lea    0x27(%rsp),%rdx
   4a018:	mov    %rdx,0x220(%rsp)
   4a020:	setne  0x238(%rsp)
   4a028:	lea    0x100(%rsp),%rdi
   4a030:	mov    %rdi,0x228(%rsp)
   4a038:	mov    %rdx,0x230(%rsp)
   4a040:	mov    0x70(%rax),%edi
   4a043:	movq   $0x0,0x88(%rsp)
   4a04f:	mov    $0x0,%edx
   4a054:	mov    %rdx,0x80(%rsp)
   4a05c:	cmp    $0x3b9aca00,%edi
   4a062:	je     4a09d <_ZN15funnel_patterns25pat_macro_shape__u64__w5117he54d8f6ffc7df352E+0x2fd>
   4a064:	mov    $0x3b9aca00,%edx
   4a069:	mulx   0x68(%rax),%r8,%r9
   4a06f:	mov    %edi,%edx
   4a071:	add    %r8,%rdx
   4a074:	adc    $0x0,%r9
   4a078:	imul   $0x3e8,%r9,%rdi
   4a07f:	mov    $0x3e8,%r8d
   4a085:	mulx   %r8,%rdx,%r8
   4a08a:	mov    %rdx,0x88(%rsp)
   4a092:	add    %rdi,%r8
   4a095:	mov    %r8,0x80(%rsp)
   4a09d:	movq   $0xffffffffffffffff,0x48(%rsp)
   4a0a6:	mov    0x80(%rax),%r8d
   4a0ad:	movq   $0xffffffffffffffff,0x38(%rsp)
   4a0b6:	cmp    $0x3b9aca00,%r8d
   4a0bd:	je     4a0fe <_ZN15funnel_patterns25pat_macro_shape__u64__w5117he54d8f6ffc7df352E+0x35e>
   4a0bf:	mov    $0x3b9aca00,%edx
   4a0c4:	mulx   0x78(%rax),%r9,%rdi
   4a0ca:	mov    %r8d,%edx
   4a0cd:	add    %r9,%rdx
   4a0d0:	adc    $0x0,%rdi
   4a0d4:	mov    $0x3e8,%r8d
   4a0da:	mulx   %r8,%r8,%r9
   4a0df:	mov    %r9,0x38(%rsp)
   4a0e4:	mov    %r8,0x48(%rsp)
   4a0e9:	or     %rdi,%rdx
   4a0ec:	je     4ac96 <_ZN15funnel_patterns25pat_macro_shape__u64__w5117he54d8f6ffc7df352E+0xef6>
   4a0f2:	imul   $0x3e8,%rdi,%rdx
   4a0f9:	add    %rdx,0x38(%rsp)
   4a0fe:	mov    0x58(%rax),%edx
   4a101:	cmp    $0x1,%edx
   4a104:	jne    4a110 <_ZN15funnel_patterns25pat_macro_shape__u64__w5117he54d8f6ffc7df352E+0x370>
   4a106:	cmpl   $0x0,0x5c(%rax)
   4a10a:	je     4ac96 <_ZN15funnel_patterns25pat_macro_shape__u64__w5117he54d8f6ffc7df352E+0xef6>
   4a110:	cmpl   $0x1,0x60(%rax)
   4a114:	jne    4a120 <_ZN15funnel_patterns25pat_macro_shape__u64__w5117he54d8f6ffc7df352E+0x380>
   4a116:	cmpl   $0x0,0x64(%rax)
   4a11a:	je     4ac96 <_ZN15funnel_patterns25pat_macro_shape__u64__w5117he54d8f6ffc7df352E+0xef6>
   4a120:	mov    %r14,0x60(%rsp)
   4a125:	test   %dl,%sil
   4a128:	je     4a13b <_ZN15funnel_patterns25pat_macro_shape__u64__w5117he54d8f6ffc7df352E+0x39b>
   4a12a:	mov    0x5c(%rax),%edx
   4a12d:	mov    %edx,0x18(%rsp)
   4a131:	movl   $0x1,0xc(%rsp)
   4a139:	jmp    4a14b <_ZN15funnel_patterns25pat_macro_shape__u64__w5117he54d8f6ffc7df352E+0x3ab>
   4a13b:	movzbl %sil,%edx
   4a13f:	mov    %edx,0xc(%rsp)
   4a143:	movl   $0x64,0x18(%rsp)
   4a14b:	movq   $0x0,0x58(%rsp)
   4a154:	mov    $0x0,%edx
   4a159:	mov    %rdx,0x50(%rsp)
   4a15e:	test   %cl,%cl
   4a160:	je     4a17d <_ZN15funnel_patterns25pat_macro_shape__u64__w5117he54d8f6ffc7df352E+0x3dd>
   4a162:	mov    %r14,%rdi
   4a165:	call   *0xf2165(%rip)        # 13c2d0 <_DYNAMIC+0x240>
   4a16b:	mov    %rax,0x58(%rsp)
   4a170:	mov    %rdx,0x50(%rsp)
   4a175:	mov    0xf8(%r12),%rax
   4a17d:	cmpb   $0x1,0x3(%rsp)
   4a182:	je     4a1a3 <_ZN15funnel_patterns25pat_macro_shape__u64__w5117he54d8f6ffc7df352E+0x403>
   4a184:	mov    $0x1,%edx
   4a189:	cmpb   $0x0,0x58(%rax)
   4a18d:	je     4a192 <_ZN15funnel_patterns25pat_macro_shape__u64__w5117he54d8f6ffc7df352E+0x3f2>
   4a18f:	mov    0x5c(%rax),%edx
   4a192:	mov    (%r12),%rcx
   4a196:	mov    0x10(%r12),%rsi
   4a19b:	sub    %rsi,%rcx
   4a19e:	cmp    %rcx,%rdx
   4a1a1:	ja     4a1e8 <_ZN15funnel_patterns25pat_macro_shape__u64__w5117he54d8f6ffc7df352E+0x448>
   4a1a3:	testb  $0x1,0x88(%rax)
   4a1aa:	jne    4a201 <_ZN15funnel_patterns25pat_macro_shape__u64__w5117he54d8f6ffc7df352E+0x461>
   4a1ac:	lock orl $0x0,-0x40(%rsp)
   4a1b2:	test   %r14,%r14
   4a1b5:	je     4a1d3 <_ZN15funnel_patterns25pat_macro_shape__u64__w5117he54d8f6ffc7df352E+0x433>
   4a1b7:	lfence
   4a1ba:	rdtsc
   4a1bc:	shl    $0x20,%rdx
   4a1c0:	or     %rax,%rdx
   4a1c3:	mov    %rdx,0x70(%rsp)
   4a1c8:	lfence
   4a1cb:	mov    $0x3b9aca00,%r13d
   4a1d1:	jmp    4a1e1 <_ZN15funnel_patterns25pat_macro_shape__u64__w5117he54d8f6ffc7df352E+0x441>
   4a1d3:	call   *0xf20ff(%rip)        # 13c2d8 <_DYNAMIC+0x248>
   4a1d9:	mov    %rax,0x70(%rsp)
   4a1de:	mov    %edx,%r13d
   4a1e1:	mov    0x60(%rsp),%r14
   4a1e6:	jmp    4a207 <_ZN15funnel_patterns25pat_macro_shape__u64__w5117he54d8f6ffc7df352E+0x467>
   4a1e8:	mov    %r12,%rdi
   4a1eb:	call   51800 <_ZN5alloc7raw_vec20RawVecInner$LT$A$GT$7reserve21do_reserve_and_handle17h46771c9d08372974E>
   4a1f0:	mov    0xf8(%r12),%rax
   4a1f8:	testb  $0x1,0x88(%rax)
   4a1ff:	je     4a1ac <_ZN15funnel_patterns25pat_macro_shape__u64__w5117he54d8f6ffc7df352E+0x40c>
   4a201:	mov    $0x3b9aca01,%r13d
   4a207:	mov    %r14,%rdi
   4a20a:	call   *0xf20d0(%rip)        # 13c2e0 <_DYNAMIC+0x250>
   4a210:	mov    %rax,0xa8(%rsp)
   4a218:	mov    0xf20c9(%rip),%r14        # 13c2e8 <_DYNAMIC+0x258>
   4a21f:	mov    0x8(%r14),%eax
   4a223:	test   %eax,%eax
   4a225:	jne    4acc3 <_ZN15funnel_patterns25pat_macro_shape__u64__w5117he54d8f6ffc7df352E+0xf23>
   4a22b:	mov    (%r14),%rdi
   4a22e:	call   *0xf20bc(%rip)        # 13c2f0 <_DYNAMIC+0x260>
   4a234:	mov    0x48(%rsp),%rax
   4a239:	or     0x38(%rsp),%rax
   4a23e:	je     4ac52 <_ZN15funnel_patterns25pat_macro_shape__u64__w5117he54d8f6ffc7df352E+0xeb2>
   4a244:	lea    0x50(%r12),%rax
   4a249:	mov    %rax,0x28(%rsp)
   4a24e:	lea    0x18(%r12),%rax
   4a253:	mov    %rax,0x78(%rsp)
   4a258:	mov    $0x10,%r15d
   4a25e:	mov    $0x1,%al
   4a260:	xor    %edx,%edx
   4a262:	xor    %esi,%esi
   4a264:	xor    %r14d,%r14d
   4a267:	mov    %r12,0x30(%rsp)
   4a26c:	mov    %r13d,0x6c(%rsp)
   4a271:	jmp    4a2eb <_ZN15funnel_patterns25pat_macro_shape__u64__w5117he54d8f6ffc7df352E+0x54b>
   4a273:	data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
   4a280:	mov    0x90(%rsp),%rdi
   4a288:	cmp    $0x3e9,%rdi
   4a28f:	mov    $0x3e8,%eax
   4a294:	cmovae %rdi,%rax
   4a298:	mov    0x98(%rsp),%rsi
   4a2a0:	test   %rsi,%rsi
   4a2a3:	mov    $0x3e8,%ecx
   4a2a8:	cmove  %rcx,%rdi
   4a2ac:	cmove  %rax,%rdi
   4a2b0:	mov    0xb0(%rsp),%rdx
   4a2b8:	add    %rdi,%rdx
   4a2bb:	adc    %rsi,%r14
   4a2be:	mov    $0xffffffffffffffff,%rax
   4a2c5:	cmovb  %rax,%r14
   4a2c9:	cmovb  %rax,%rdx
   4a2cd:	mov    %r14,%rsi
   4a2d0:	mov    $0x1,%r14d
   4a2d6:	xor    %eax,%eax
   4a2d8:	cmp    0x48(%rsp),%rdx
   4a2dd:	mov    %rsi,%rcx
   4a2e0:	sbb    0x38(%rsp),%rcx
   4a2e5:	jae    4ac5b <_ZN15funnel_patterns25pat_macro_shape__u64__w5117he54d8f6ffc7df352E+0xebb>
   4a2eb:	cmp    0x88(%rsp),%rdx
   4a2f3:	mov    %rsi,0xe8(%rsp)
   4a2fb:	mov    %rsi,%rcx
   4a2fe:	sbb    0x80(%rsp),%rcx
   4a306:	jb     4a31a <_ZN15funnel_patterns25pat_macro_shape__u64__w5117he54d8f6ffc7df352E+0x57a>
   4a308:	testb  $0x1,0xc(%rsp)
   4a30d:	je     4a31a <_ZN15funnel_patterns25pat_macro_shape__u64__w5117he54d8f6ffc7df352E+0x57a>
   4a30f:	cmpl   $0x0,0x18(%rsp)
   4a314:	je     4ac5b <_ZN15funnel_patterns25pat_macro_shape__u64__w5117he54d8f6ffc7df352E+0xebb>
   4a31a:	mov    %rdx,0xb0(%rsp)
   4a322:	test   %ebp,%ebp
   4a324:	mov    0x8(%rsp),%ecx
   4a328:	mov    $0x1,%edx
   4a32d:	cmove  %edx,%ecx
   4a330:	mov    %ecx,0x4(%rsp)
   4a334:	mov    %ecx,0x48(%r12)
   4a339:	movq   $0x0,0x268(%rsp)
   4a345:	mov    0x28(%rsp),%rcx
   4a34a:	mov    %rcx,0x240(%rsp)
   4a352:	lea    0x220(%rsp),%rcx
   4a35a:	mov    %rcx,0x248(%rsp)
   4a362:	lea    0x4(%rsp),%rcx
   4a367:	mov    %rcx,0x250(%rsp)
   4a36f:	lea    0x268(%rsp),%rcx
   4a377:	mov    %rcx,0x258(%rsp)
   4a37f:	lea    0x60(%rsp),%rcx
   4a384:	mov    %rcx,0x260(%rsp)
   4a38c:	test   $0x1,%al
   4a38e:	jne    4a500 <_ZN15funnel_patterns25pat_macro_shape__u64__w5117he54d8f6ffc7df352E+0x760>
   4a394:	imul   $0xd0,%r14,%rax
   4a39b:	add    %r15,%rax
   4a39e:	mov    %rax,%rdx
   4a3a1:	sub    %r15,%rdx
   4a3a4:	add    $0xffffffffffffff30,%rdx
   4a3ab:	movabs $0x4ec4ec4ec4ec4ec5,%rcx
   4a3b5:	mulx   %rcx,%rcx,%rcx
   4a3ba:	mov    %r15,%rsi
   4a3bd:	cmp    $0xc30,%rdx
   4a3c4:	jb     4a550 <_ZN15funnel_patterns25pat_macro_shape__u64__w5117he54d8f6ffc7df352E+0x7b0>
   4a3ca:	shr    $0x6,%rcx
   4a3ce:	inc    %rcx
   4a3d1:	cmp    $0x3330,%rdx
   4a3d8:	jae    4a3f0 <_ZN15funnel_patterns25pat_macro_shape__u64__w5117he54d8f6ffc7df352E+0x650>
   4a3da:	xor    %edx,%edx
   4a3dc:	mov    %r15,%rdi
   4a3df:	jmp    4a49f <_ZN15funnel_patterns25pat_macro_shape__u64__w5117he54d8f6ffc7df352E+0x6ff>
   4a3e4:	data16 data16 cs nopw 0x0(%rax,%rax,1)
   4a3f0:	mov    %rcx,%rdx
   4a3f3:	movabs $0x3ffffffffffffc0,%rsi
   4a3fd:	and    %rsi,%rdx
   4a400:	imul   $0xd0,%rdx,%rsi
   4a407:	lea    (%r15,%rsi,1),%rdi
   4a40b:	mov    %rdx,%r8
   4a40e:	mov    %r15,%r9
   4a411:	vpbroadcastd -0x39b47(%rip),%zmm0        # 108d4 <__abi_tag+0x105d8>
   4a41b:	vmovdqa64 -0x3a625(%rip),%zmm1        # fe00 <__abi_tag+0xfb04>
   4a425:	vmovdqa64 -0x3a5ef(%rip),%zmm2        # fe40 <__abi_tag+0xfb44>
   4a42f:	vmovdqa64 -0x3a5b9(%rip),%zmm3        # fe80 <__abi_tag+0xfb84>
   4a439:	vmovdqa64 -0x3a583(%rip),%zmm4        # fec0 <__abi_tag+0xfbc4>
   4a443:	data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
   4a450:	kxnorw %k0,%k0,%k1
   4a454:	vpscatterdd %zmm0,0x8(%r9,%zmm1,1){%k1}
   4a45c:	kxnorw %k0,%k0,%k1
   4a460:	vpscatterdd %zmm0,0x8(%r9,%zmm2,1){%k1}
   4a468:	kxnorw %k0,%k0,%k1
   4a46c:	vpscatterdd %zmm0,0x8(%r9,%zmm3,1){%k1}
   4a474:	kxnorw %k0,%k0,%k1
   4a478:	vpscatterdd %zmm0,0x8(%r9,%zmm4,1){%k1}
   4a480:	add    $0x3400,%r9
   4a487:	add    $0xffffffffffffffc0,%r8
   4a48b:	jne    4a450 <_ZN15funnel_patterns25pat_macro_shape__u64__w5117he54d8f6ffc7df352E+0x6b0>
   4a48d:	cmp    %rdx,%rcx
   4a490:	je     4a563 <_ZN15funnel_patterns25pat_macro_shape__u64__w5117he54d8f6ffc7df352E+0x7c3>
   4a496:	test   $0x30,%cl
   4a499:	je     4a54d <_ZN15funnel_patterns25pat_macro_shape__u64__w5117he54d8f6ffc7df352E+0x7ad>
   4a49f:	movabs $0x3ffffffffffffc0,%rsi
   4a4a9:	lea    0x30(%rsi),%r8
   4a4ad:	and    %rcx,%r8
   4a4b0:	imul   $0xd0,%r8,%rsi
   4a4b7:	add    %r15,%rsi
   4a4ba:	sub    %r8,%rdx
   4a4bd:	vpbroadcastd -0x39bf3(%rip),%zmm0        # 108d4 <__abi_tag+0x105d8>
   4a4c7:	vmovdqa64 -0x3a6d1(%rip),%zmm1        # fe00 <__abi_tag+0xfb04>
   4a4d1:	data16 data16 data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
   4a4e0:	kxnorw %k0,%k0,%k1
   4a4e4:	vpscatterdd %zmm0,0x8(%rdi,%zmm1,1){%k1}
   4a4ec:	add    $0xd00,%rdi
   4a4f3:	add    $0x10,%rdx
   4a4f7:	jne    4a4e0 <_ZN15funnel_patterns25pat_macro_shape__u64__w5117he54d8f6ffc7df352E+0x740>
   4a4f9:	cmp    %r8,%rcx
   4a4fc:	jne    4a550 <_ZN15funnel_patterns25pat_macro_shape__u64__w5117he54d8f6ffc7df352E+0x7b0>
   4a4fe:	jmp    4a563 <_ZN15funnel_patterns25pat_macro_shape__u64__w5117he54d8f6ffc7df352E+0x7c3>
   4a500:	movq   $0x0,0x128(%rsp)
   4a50c:	mov    $0x10,%esi
   4a511:	mov    $0xd0,%edx
   4a516:	lea    0x1a0(%rsp),%rdi
   4a51e:	lea    0x120(%rsp),%rcx
   4a526:	call   516c0 <_ZN5alloc7raw_vec11finish_grow17hedc133b40cb748a9E>
   4a52b:	cmpb   $0x0,0x1a0(%rsp)
   4a533:	jne    4ad82 <_ZN15funnel_patterns25pat_macro_shape__u64__w5117he54d8f6ffc7df352E+0xfe2>
   4a539:	mov    0x1a8(%rsp),%r15
   4a541:	lea    0xd0(%r15),%rax
   4a548:	jmp    4a39e <_ZN15funnel_patterns25pat_macro_shape__u64__w5117he54d8f6ffc7df352E+0x5fe>
   4a54d:	add    %r15,%rsi
   4a550:	movl   $0x3b9aca01,0x8(%rsi)
   4a557:	add    $0xd0,%rsi
   4a55e:	cmp    %rax,%rsi
   4a561:	jne    4a550 <_ZN15funnel_patterns25pat_macro_shape__u64__w5117he54d8f6ffc7df352E+0x7b0>
   4a563:	mov    %r12,%rbx
   4a566:	mov    %r15,0x40(%rsp)
   4a56b:	vzeroupper
   4a56e:	call   *0xf1d84(%rip)        # 13c2f8 <_DYNAMIC+0x268>
   4a574:	mov    %rax,0x120(%rsp)
   4a57c:	movq   $0x0,0x128(%rsp)
   4a588:	lea    0x4b61(%rip),%rax        # 4f0f0 <_ZN30codspeed_divan_compat_walltime11thread_pool19TaskShared$LT$F$GT$3new4call17hdef8ccc415b76681E>
   4a58f:	mov    %rax,0x130(%rsp)
   4a597:	lea    0x240(%rsp),%rax
   4a59f:	mov    %rax,0x138(%rsp)
   4a5a7:	mov    %r15,0x140(%rsp)
   4a5af:	mov    0xf1d4a(%rip),%rdi        # 13c300 <_DYNAMIC+0x270>
   4a5b6:	xor    %esi,%esi
   4a5b8:	lea    0x120(%rsp),%rdx
   4a5c0:	call   *0xf1d42(%rip)        # 13c308 <_DYNAMIC+0x278>
   4a5c6:	mov    0x120(%rsp),%rax
   4a5ce:	lock decq (%rax)
   4a5d2:	jne    4a5e2 <_ZN15funnel_patterns25pat_macro_shape__u64__w5117he54d8f6ffc7df352E+0x842>
   4a5d4:	lea    0x120(%rsp),%rdi
   4a5dc:	call   *0xf1d2e(%rip)        # 13c310 <_DYNAMIC+0x280>
   4a5e2:	cmpl   $0x3b9aca01,0x8(%r15)
   4a5ea:	je     4acdf <_ZN15funnel_patterns25pat_macro_shape__u64__w5117he54d8f6ffc7df352E+0xf3f>
   4a5f0:	cmpb   $0x1,0x3(%rsp)
   4a5f5:	je     4aca8 <_ZN15funnel_patterns25pat_macro_shape__u64__w5117he54d8f6ffc7df352E+0xf08>
   4a5fb:	vmovups 0x10(%r15),%xmm0
   4a601:	vmovaps %xmm0,0x1a0(%rsp)
   4a60a:	vmovdqu (%r15),%xmm0
   4a60f:	vmovdqa %xmm0,0x120(%rsp)
   4a618:	mov    0xc0(%r15),%rdx
   4a61f:	lea    0x1a0(%rsp),%rdi
   4a627:	lea    0x120(%rsp),%rsi
   4a62f:	call   *0xf1ce3(%rip)        # 13c318 <_DYNAMIC+0x288>
   4a635:	lea    0x10(%r15),%rax
   4a639:	mov    %rax,%r12
   4a63c:	vmovups (%rax),%xmm0
   4a640:	vmovaps %xmm0,0x1a0(%rsp)
   4a649:	vmovdqu (%r15),%xmm0
   4a64e:	vmovdqa %xmm0,0x120(%rsp)
   4a657:	mov    0xc0(%r15),%rdx
   4a65e:	lea    0x1a0(%rsp),%rdi
   4a666:	lea    0x120(%rsp),%rsi
   4a66e:	call   *0xf1ca4(%rip)        # 13c318 <_DYNAMIC+0x288>
   4a674:	mov    %rax,%rsi
   4a677:	cmp    $0x1,%ebp
   4a67a:	mov    %rdx,0x98(%rsp)
   4a682:	mov    %rax,0x90(%rsp)
   4a68a:	jne    4a700 <_ZN15funnel_patterns25pat_macro_shape__u64__w5117he54d8f6ffc7df352E+0x960>
   4a68c:	mov    %rbx,%rcx
   4a68f:	movq   $0x0,0x10(%rbx)
   4a697:	cmpq   $0x0,0x30(%rbx)
   4a69c:	mov    0x58(%rsp),%r13
   4a6a1:	mov    0x50(%rsp),%rbp
   4a6a6:	je     4a726 <_ZN15funnel_patterns25pat_macro_shape__u64__w5117he54d8f6ffc7df352E+0x986>
   4a6a8:	mov    0x20(%rcx),%r14
   4a6ac:	test   %r14,%r14
   4a6af:	je     4a715 <_ZN15funnel_patterns25pat_macro_shape__u64__w5117he54d8f6ffc7df352E+0x975>
   4a6b1:	mov    0x78(%rsp),%rax
   4a6b6:	mov    (%rax),%rdi
   4a6b9:	lea    0x11(%r14),%rdx
   4a6bd:	mov    $0xff,%esi
   4a6c2:	call   *0xf1bf8(%rip)        # 13c2c0 <memset@GLIBC_2.2.5>
   4a6c8:	mov    0x90(%rsp),%rsi
   4a6d0:	mov    0x98(%rsp),%rdx
   4a6d8:	lea    0x1(%r14),%rax
   4a6dc:	mov    %rax,%rcx
   4a6df:	shr    $0x3,%rcx
   4a6e3:	and    $0xfffffffffffffff8,%rax
   4a6e7:	sub    %rcx,%rax
   4a6ea:	cmp    $0x8,%r14
   4a6ee:	cmovb  %r14,%rax
   4a6f2:	jmp    4a717 <_ZN15funnel_patterns25pat_macro_shape__u64__w5117he54d8f6ffc7df352E+0x977>
   4a6f4:	data16 data16 cs nopw 0x0(%rax,%rax,1)
   4a700:	mov    %ebp,0x14(%rsp)
   4a704:	mov    0x4(%rsp),%eax
   4a708:	mov    %rax,0xa0(%rsp)
   4a710:	jmp    4a7b7 <_ZN15funnel_patterns25pat_macro_shape__u64__w5117he54d8f6ffc7df352E+0xa17>
   4a715:	xor    %eax,%eax
   4a717:	mov    %rbx,%rcx
   4a71a:	movq   $0x0,0x30(%rbx)
   4a722:	mov    %rax,0x28(%rbx)
   4a726:	cmpq   $0x0,0x68(%rcx)
   4a72b:	je     4a735 <_ZN15funnel_patterns25pat_macro_shape__u64__w5117he54d8f6ffc7df352E+0x995>
   4a72d:	movq   $0x0,0x60(%rcx)
   4a735:	cmpq   $0x0,0x90(%rcx)
   4a73d:	je     4a74a <_ZN15funnel_patterns25pat_macro_shape__u64__w5117he54d8f6ffc7df352E+0x9aa>
   4a73f:	movq   $0x0,0x88(%rcx)
   4a74a:	cmpq   $0x0,0xb8(%rcx)
   4a752:	je     4a75f <_ZN15funnel_patterns25pat_macro_shape__u64__w5117he54d8f6ffc7df352E+0x9bf>
   4a754:	movq   $0x0,0xb0(%rcx)
   4a75f:	cmpq   $0x0,0xe0(%rcx)
   4a767:	je     4a774 <_ZN15funnel_patterns25pat_macro_shape__u64__w5117he54d8f6ffc7df352E+0x9d4>
   4a769:	movq   $0x0,0xd8(%rcx)
   4a774:	mov    %r13,%rax
   4a777:	or     %rbp,%rax
   4a77a:	je     4ad73 <_ZN15funnel_patterns25pat_macro_shape__u64__w5117he54d8f6ffc7df352E+0xfd3>
   4a780:	mov    %rsi,%rdi
   4a783:	mov    %rdx,%rsi
   4a786:	mov    %r13,%rdx
   4a789:	mov    %rbp,%rcx
   4a78c:	call   *0xf1b8e(%rip)        # 13c320 <_DYNAMIC+0x290>
   4a792:	cmp    $0x65,%rax
   4a796:	sbb    $0x0,%rdx
   4a79a:	mov    0x4(%rsp),%ecx
   4a79e:	mov    %rcx,0xa0(%rsp)
   4a7a6:	jae    4a7d0 <_ZN15funnel_patterns25pat_macro_shape__u64__w5117he54d8f6ffc7df352E+0xa30>
   4a7a8:	lea    (%rcx,%rcx,1),%eax
   4a7ab:	mov    %eax,0x8(%rsp)
   4a7af:	movl   $0x1,0x14(%rsp)
   4a7b7:	mov    0x18(%rsp),%eax
   4a7bb:	mov    %eax,0x10(%rsp)
   4a7bf:	mov    %r12,%rdx
   4a7c2:	jmp    4a80d <_ZN15funnel_patterns25pat_macro_shape__u64__w5117he54d8f6ffc7df352E+0xa6d>
   4a7c4:	data16 data16 cs nopw 0x0(%rax,%rax,1)
   4a7d0:	mov    0xf8(%rbx),%rax
   4a7d7:	movl   $0x1,0xc(%rsp)
   4a7df:	cmpb   $0x0,0x58(%rax)
   4a7e3:	mov    %r12,%rdx
   4a7e6:	je     4a7f9 <_ZN15funnel_patterns25pat_macro_shape__u64__w5117he54d8f6ffc7df352E+0xa59>
   4a7e8:	movl   $0x2,0x14(%rsp)
   4a7f0:	mov    0x5c(%rax),%eax
   4a7f3:	mov    %eax,0x10(%rsp)
   4a7f7:	jmp    4a809 <_ZN15funnel_patterns25pat_macro_shape__u64__w5117he54d8f6ffc7df352E+0xa69>
   4a7f9:	movl   $0x64,0x10(%rsp)
   4a801:	movl   $0x2,0x14(%rsp)
   4a809:	mov    %ecx,0x8(%rsp)
   4a80d:	mov    0xa8(%rsp),%rax
   4a815:	mov    (%rax),%rbp
   4a818:	mov    0x8(%rax),%r13
   4a81c:	mov    0x18(%rax),%r14
   4a820:	mov    0x10(%rax),%rcx
   4a824:	mov    %rcx,0xc0(%rsp)
   4a82c:	mov    %r15,%rcx
   4a82f:	mov    0x40(%r15),%r15
   4a833:	mov    0x28(%rax),%rsi
   4a837:	mov    %rsi,0xc8(%rsp)
   4a83f:	mov    0x20(%rax),%rsi
   4a843:	mov    %rsi,0xd0(%rsp)
   4a84b:	mov    0x50(%rcx),%rbx
   4a84f:	mov    0x38(%rax),%rsi
   4a853:	mov    %rsi,0xe0(%rsp)
   4a85b:	mov    0x30(%rax),%rax
   4a85f:	mov    %rax,0x18(%rsp)
   4a864:	mov    0x20(%rcx),%rax
   4a868:	mov    %rax,0xd8(%rsp)
   4a870:	mov    0x30(%rcx),%r12
   4a874:	vmovups (%rdx),%xmm0
   4a878:	vmovaps %xmm0,0x1a0(%rsp)
   4a881:	mov    0x30(%rsp),%rax
   4a886:	mov    0x10(%rax),%rax
   4a88a:	mov    %rax,0xb8(%rsp)
   4a892:	vmovdqu (%rcx),%xmm0
   4a896:	vmovdqa %xmm0,0x120(%rsp)
   4a89f:	mov    0xc0(%rcx),%rdx
   4a8a6:	lea    0x1a0(%rsp),%rdi
   4a8ae:	lea    0x120(%rsp),%rsi
   4a8b6:	call   *0xf1a5c(%rip)        # 13c318 <_DYNAMIC+0x288>
   4a8bc:	mov    %rax,%rsi
   4a8bf:	mov    %rdx,%rcx
   4a8c2:	mov    0xa0(%rsp),%edi
   4a8c9:	mov    %r13,%rax
   4a8cc:	mul    %rdi
   4a8cf:	mov    %rbp,%rdx
   4a8d2:	mulx   %rdi,%r10,%r9
   4a8d7:	seto   %dl
   4a8da:	add    %rax,%r9
   4a8dd:	setb   %r13b
   4a8e1:	or     %dl,%r13b
   4a8e4:	mov    %r14,%rax
   4a8e7:	mul    %r15
   4a8ea:	seto   %r11b
   4a8ee:	mov    0xc0(%rsp),%rdx
   4a8f6:	mulx   %r15,%r8,%rdi
   4a8fb:	add    %rax,%rdi
   4a8fe:	setb   %r14b
   4a902:	or     %r11b,%r14b
   4a905:	mov    0xc8(%rsp),%rax
   4a90d:	mul    %rbx
   4a910:	mov    0xd0(%rsp),%rdx
   4a918:	mulx   %rbx,%rbx,%r11
   4a91d:	seto   %dl
   4a920:	add    %rax,%r11
   4a923:	setb   %bpl
   4a927:	or     %dl,%bpl
   4a92a:	or     %r14b,%bpl
   4a92d:	or     %r13b,%bpl
   4a930:	xor    %r14d,%r14d
   4a933:	add    0xd8(%rsp),%r12
   4a93b:	setb   %r14b
   4a93f:	mov    0xe0(%rsp),%rax
   4a947:	test   %rax,%rax
   4a94a:	setne  %dl
   4a94d:	mov    %r14d,%r15d
   4a950:	and    %dl,%r15b
   4a953:	mul    %r12
   4a956:	seto   %r13b
   4a95a:	or     %r15b,%r13b
   4a95d:	mov    0x18(%rsp),%rdx
   4a962:	imul   %rdx,%r14
   4a966:	add    %rax,%r14
   4a969:	mulx   %r12,%rdx,%rax
   4a96e:	add    %r14,%rax
   4a971:	setb   %r14b
   4a975:	or     %r13b,%r14b
   4a978:	or     %bpl,%r14b
   4a97b:	add    %r10,%r8
   4a97e:	adc    %r9,%rdi
   4a981:	mov    $0xffffffffffffffff,%r15
   4a988:	cmovb  %r15,%rdi
   4a98c:	cmovb  %r15,%r8
   4a990:	add    %rbx,%r8
   4a993:	adc    %r11,%rdi
   4a996:	cmovb  %r15,%rdi
   4a99a:	cmovb  %r15,%r8
   4a99e:	mov    $0xffffffffffffffff,%r10
   4a9a5:	mov    $0xffffffffffffffff,%r9
   4a9ac:	test   $0x1,%r14b
   4a9b0:	jne    4a9c6 <_ZN15funnel_patterns25pat_macro_shape__u64__w5117he54d8f6ffc7df352E+0xc26>
   4a9b2:	add    %rdx,%r8
   4a9b5:	adc    %rax,%rdi
   4a9b8:	cmovb  %r15,%rdi
   4a9bc:	cmovb  %r15,%r8
   4a9c0:	mov    %r8,%r10
   4a9c3:	mov    %rdi,%r9
   4a9c6:	mov    %rsi,%rax
   4a9c9:	or     %rcx,%rax
   4a9cc:	mov    0x50(%rsp),%rdx
   4a9d1:	cmove  %rdx,%rcx
   4a9d5:	mov    0x58(%rsp),%rax
   4a9da:	cmove  %rax,%rsi
   4a9de:	mov    %rsi,%rbx
   4a9e1:	sub    %r10,%rbx
   4a9e4:	mov    %rcx,%r14
   4a9e7:	sbb    %r9,%r14
   4a9ea:	cmp    %rsi,%r10
   4a9ed:	sbb    %rcx,%r9
   4a9f0:	cmovae %rdx,%r14
   4a9f4:	cmovae %rax,%rbx
   4a9f8:	mov    0x30(%rsp),%rax
   4a9fd:	mov    0x10(%rax),%r15
   4aa01:	cmp    (%rax),%r15
   4aa04:	jne    4aa10 <_ZN15funnel_patterns25pat_macro_shape__u64__w5117he54d8f6ffc7df352E+0xc70>
   4aa06:	mov    0x30(%rsp),%rdi
   4aa0b:	call   51730 <_ZN5alloc7raw_vec19RawVec$LT$T$C$A$GT$8grow_one17hfe527167958a2dadE>
   4aa10:	mov    0x30(%rsp),%r12
   4aa15:	mov    0x8(%r12),%rax
   4aa1a:	mov    %r15,%rcx
   4aa1d:	shl    $0x4,%rcx
   4aa21:	mov    %r14,0x8(%rax,%rcx,1)
   4aa26:	mov    %rbx,(%rax,%rcx,1)
   4aa2a:	inc    %r15
   4aa2d:	mov    %r15,0x10(%r12)
   4aa32:	mov    0x40(%rsp),%r15
   4aa37:	mov    0x28(%r15),%rax
   4aa3b:	or     0x20(%r15),%rax
   4aa3f:	jne    4aa60 <_ZN15funnel_patterns25pat_macro_shape__u64__w5117he54d8f6ffc7df352E+0xcc0>
   4aa41:	mov    0x38(%r15),%rax
   4aa45:	or     0x30(%r15),%rax
   4aa49:	jne    4aa60 <_ZN15funnel_patterns25pat_macro_shape__u64__w5117he54d8f6ffc7df352E+0xcc0>
   4aa4b:	mov    0x48(%r15),%rax
   4aa4f:	or     0x40(%r15),%rax
   4aa53:	jne    4aa60 <_ZN15funnel_patterns25pat_macro_shape__u64__w5117he54d8f6ffc7df352E+0xcc0>
   4aa55:	mov    0x58(%r15),%rax
   4aa59:	or     0x50(%r15),%rax
   4aa5d:	je     4aab6 <_ZN15funnel_patterns25pat_macro_shape__u64__w5117he54d8f6ffc7df352E+0xd16>
   4aa5f:	nop
   4aa60:	lea    0x20(%r15),%rax
   4aa64:	vmovdqu64 (%rax),%zmm0
   4aa6a:	vmovdqu64 0x20(%rax),%zmm1
   4aa74:	vmovdqu64 %zmm1,0x1c0(%rsp)
   4aa7c:	vmovdqu64 %zmm0,0x1a0(%rsp)
   4aa87:	lea    0x120(%rsp),%rdi
   4aa8f:	mov    0x78(%rsp),%rsi
   4aa94:	mov    0xb8(%rsp),%rdx
   4aa9c:	lea    0x1a0(%rsp),%rcx
   4aaa4:	vzeroupper
   4aaa7:	call   51900 <_ZN9hashbrown3map28HashMap$LT$K$C$V$C$S$C$A$GT$6insert17he36ed9c6ee1bc3e6E>
   4aaac:	mov    0x30(%rsp),%r12
   4aab1:	mov    0x40(%rsp),%r15
   4aab6:	cmpq   $0x0,0x68(%r12)
   4aabc:	mov    0x14(%rsp),%ebp
   4aac0:	mov    0x6c(%rsp),%r13d
   4aac5:	mov    0xf185c(%rip),%rbx        # 13c328 <_DYNAMIC+0x298>
   4aacc:	mov    0xe8(%rsp),%r14
   4aad4:	je     4ab06 <_ZN15funnel_patterns25pat_macro_shape__u64__w5117he54d8f6ffc7df352E+0xd66>
   4aad6:	mov    0x4(%rsp),%eax
   4aada:	test   %eax,%eax
   4aadc:	je     4accd <_ZN15funnel_patterns25pat_macro_shape__u64__w5117he54d8f6ffc7df352E+0xf2d>
   4aae2:	mov    %eax,%edx
   4aae4:	mov    0x80(%r15),%rdi
   4aaeb:	mov    0x88(%r15),%rsi
   4aaf2:	xor    %ecx,%ecx
   4aaf4:	call   *0xf1826(%rip)        # 13c320 <_DYNAMIC+0x290>
   4aafa:	mov    0x28(%rsp),%rdi
   4aaff:	mov    %rax,%rsi
   4ab02:	xor    %edx,%edx
   4ab04:	call   *%rbx
   4ab06:	cmpq   $0x0,0x90(%r12)
   4ab0f:	je     4ab44 <_ZN15funnel_patterns25pat_macro_shape__u64__w5117he54d8f6ffc7df352E+0xda4>
   4ab11:	mov    0x4(%rsp),%eax
   4ab15:	test   %eax,%eax
   4ab17:	je     4accd <_ZN15funnel_patterns25pat_macro_shape__u64__w5117he54d8f6ffc7df352E+0xf2d>
   4ab1d:	mov    %eax,%edx
   4ab1f:	mov    0x90(%r15),%rdi
   4ab26:	mov    0x98(%r15),%rsi
   4ab2d:	xor    %ecx,%ecx
   4ab2f:	call   *0xf17eb(%rip)        # 13c320 <_DYNAMIC+0x290>
   4ab35:	mov    0x28(%rsp),%rdi
   4ab3a:	mov    %rax,%rsi
   4ab3d:	mov    $0x1,%edx
   4ab42:	call   *%rbx
   4ab44:	cmpq   $0x0,0xb8(%r12)
   4ab4d:	je     4ab82 <_ZN15funnel_patterns25pat_macro_shape__u64__w5117he54d8f6ffc7df352E+0xde2>
   4ab4f:	mov    0x4(%rsp),%eax
   4ab53:	test   %eax,%eax
   4ab55:	je     4accd <_ZN15funnel_patterns25pat_macro_shape__u64__w5117he54d8f6ffc7df352E+0xf2d>
   4ab5b:	mov    %eax,%edx
   4ab5d:	mov    0xa0(%r15),%rdi
   4ab64:	mov    0xa8(%r15),%rsi
   4ab6b:	xor    %ecx,%ecx
   4ab6d:	call   *0xf17ad(%rip)        # 13c320 <_DYNAMIC+0x290>
   4ab73:	mov    0x28(%rsp),%rdi
   4ab78:	mov    %rax,%rsi
   4ab7b:	mov    $0x2,%edx
   4ab80:	call   *%rbx
   4ab82:	cmpq   $0x0,0xe0(%r12)
   4ab8b:	je     4abc0 <_ZN15funnel_patterns25pat_macro_shape__u64__w5117he54d8f6ffc7df352E+0xe20>
   4ab8d:	mov    0x4(%rsp),%eax
   4ab91:	test   %eax,%eax
   4ab93:	je     4accd <_ZN15funnel_patterns25pat_macro_shape__u64__w5117he54d8f6ffc7df352E+0xf2d>
   4ab99:	mov    %eax,%edx
   4ab9b:	mov    0xb0(%r15),%rdi
   4aba2:	mov    0xb8(%r15),%rsi
   4aba9:	xor    %ecx,%ecx
   4abab:	call   *0xf176f(%rip)        # 13c320 <_DYNAMIC+0x290>
   4abb1:	mov    0x28(%rsp),%rdi
   4abb6:	mov    %rax,%rsi
   4abb9:	mov    $0x3,%edx
   4abbe:	call   *%rbx
   4abc0:	mov    0x10(%rsp),%ecx
   4abc4:	mov    %ecx,%edx
   4abc6:	sub    $0x1,%edx
   4abc9:	mov    $0x0,%eax
   4abce:	cmovb  %eax,%edx
   4abd1:	testb  $0x1,0xc(%rsp)
   4abd6:	cmove  %ecx,%edx
   4abd9:	mov    %edx,0x18(%rsp)
   4abdd:	cmp    $0x3b9aca01,%r13d
   4abe4:	je     4a280 <_ZN15funnel_patterns25pat_macro_shape__u64__w5117he54d8f6ffc7df352E+0x4e0>
   4abea:	mov    0x70(%rsp),%rax
   4abef:	mov    %rax,0x190(%rsp)
   4abf7:	mov    %r13d,0x198(%rsp)
   4abff:	mov    0x18(%r15),%eax
   4ac03:	cmp    $0x3b9aca01,%eax
   4ac08:	je     4ad64 <_ZN15funnel_patterns25pat_macro_shape__u64__w5117he54d8f6ffc7df352E+0xfc4>
   4ac0e:	mov    0x1c(%r15),%ecx
   4ac12:	mov    0x10(%r15),%rdx
   4ac16:	mov    %rdx,0x110(%rsp)
   4ac1e:	mov    %eax,0x118(%rsp)
   4ac25:	mov    %ecx,0x11c(%rsp)
   4ac2c:	mov    0x60(%rsp),%rdx
   4ac31:	lea    0x110(%rsp),%rdi
   4ac39:	lea    0x190(%rsp),%rsi
   4ac41:	call   *0xf16d1(%rip)        # 13c318 <_DYNAMIC+0x288>
   4ac47:	mov    %rdx,%rsi
   4ac4a:	mov    %rax,%rdx
   4ac4d:	jmp    4a2d0 <_ZN15funnel_patterns25pat_macro_shape__u64__w5117he54d8f6ffc7df352E+0x530>
   4ac52:	mov    $0x10,%r15d
   4ac58:	xor    %r14d,%r14d
   4ac5b:	mov    0xf1686(%rip),%rbx        # 13c2e8 <_DYNAMIC+0x258>
   4ac62:	mov    0x8(%rbx),%eax
   4ac65:	test   %eax,%eax
   4ac67:	jne    4acbc <_ZN15funnel_patterns25pat_macro_shape__u64__w5117he54d8f6ffc7df352E+0xf1c>
   4ac69:	mov    (%rbx),%rdi
   4ac6c:	call   *0xf16be(%rip)        # 13c330 <_DYNAMIC+0x2a0>
   4ac72:	mov    0xf16bf(%rip),%rax        # 13c338 <_DYNAMIC+0x2a8>
   4ac79:	movb   $0x0,(%rax)
   4ac7c:	test   %r14,%r14
   4ac7f:	je     4ac96 <_ZN15funnel_patterns25pat_macro_shape__u64__w5117he54d8f6ffc7df352E+0xef6>
   4ac81:	imul   $0xd0,%r14,%rsi
   4ac88:	mov    $0x10,%edx
   4ac8d:	mov    %r15,%rdi
   4ac90:	call   *0xf16aa(%rip)        # 13c340 <_DYNAMIC+0x2b0>
   4ac96:	add    $0x3c18,%rsp
   4ac9d:	pop    %rbx
   4ac9e:	pop    %r12
   4aca0:	pop    %r13
   4aca2:	pop    %r14
   4aca4:	pop    %r15
   4aca6:	pop    %rbp
   4aca7:	ret
   4aca8:	mov    $0x1,%r14d
   4acae:	mov    0xf1633(%rip),%rbx        # 13c2e8 <_DYNAMIC+0x258>
   4acb5:	mov    0x8(%rbx),%eax
   4acb8:	test   %eax,%eax
   4acba:	je     4ac69 <_ZN15funnel_patterns25pat_macro_shape__u64__w5117he54d8f6ffc7df352E+0xec9>
   4acbc:	call   5152e <_ZN3std4sync9once_lock17OnceLock$LT$T$GT$10initialize17h13c8d01c1aad6442E>
   4acc1:	jmp    4ac69 <_ZN15funnel_patterns25pat_macro_shape__u64__w5117he54d8f6ffc7df352E+0xec9>
   4acc3:	call   5152e <_ZN3std4sync9once_lock17OnceLock$LT$T$GT$10initialize17h13c8d01c1aad6442E>
   4acc8:	jmp    4a22b <_ZN15funnel_patterns25pat_macro_shape__u64__w5117he54d8f6ffc7df352E+0x48b>
   4accd:	lea    0xe8e54(%rip),%rdi        # 133b28 <_ZN15funnel_patterns46__DIVAN_BENCH_PAT_U128_CAT_UNROLLED4__U64__W514PUSH17h68d55424f84a52b4E+0xe0>
   4acd4:	call   *0xf166e(%rip)        # 13c348 <_DYNAMIC+0x2b8>
   4acda:	jmp    4ad80 <_ZN15funnel_patterns25pat_macro_shape__u64__w5117he54d8f6ffc7df352E+0xfe0>
   4acdf:	movq   $0x0,0x108(%rsp)
   4aceb:	lea    0x108(%rsp),%rax
   4acf3:	mov    %rax,0x1a0(%rsp)
   4acfb:	mov    0xf164e(%rip),%rax        # 13c350 <_DYNAMIC+0x2c0>
   4ad02:	mov    %rax,0x1a8(%rsp)
   4ad0a:	lea    0xe8d97(%rip),%rax        # 133aa8 <_ZN15funnel_patterns46__DIVAN_BENCH_PAT_U128_CAT_UNROLLED4__U64__W514PUSH17h68d55424f84a52b4E+0x60>
   4ad11:	mov    %rax,0x120(%rsp)
   4ad19:	movq   $0x2,0x128(%rsp)
   4ad25:	movq   $0x0,0x140(%rsp)
   4ad31:	lea    0x1a0(%rsp),%rax
   4ad39:	mov    %rax,0x130(%rsp)
   4ad41:	movq   $0x1,0x138(%rsp)
   4ad4d:	lea    0xe8d74(%rip),%rsi        # 133ac8 <_ZN15funnel_patterns46__DIVAN_BENCH_PAT_U128_CAT_UNROLLED4__U64__W514PUSH17h68d55424f84a52b4E+0x80>
   4ad54:	lea    0x120(%rsp),%rdi
   4ad5c:	call   *0xf15f6(%rip)        # 13c358 <_DYNAMIC+0x2c8>
   4ad62:	jmp    4ad80 <_ZN15funnel_patterns25pat_macro_shape__u64__w5117he54d8f6ffc7df352E+0xfe0>
   4ad64:	lea    0xe8d8d(%rip),%rdi        # 133af8 <_ZN15funnel_patterns46__DIVAN_BENCH_PAT_U128_CAT_UNROLLED4__U64__W514PUSH17h68d55424f84a52b4E+0xb0>
   4ad6b:	call   *0xf15ef(%rip)        # 13c360 <_DYNAMIC+0x2d0>
   4ad71:	jmp    4ad80 <_ZN15funnel_patterns25pat_macro_shape__u64__w5117he54d8f6ffc7df352E+0xfe0>
   4ad73:	lea    0xe8d66(%rip),%rdi        # 133ae0 <_ZN15funnel_patterns46__DIVAN_BENCH_PAT_U128_CAT_UNROLLED4__U64__W514PUSH17h68d55424f84a52b4E+0x98>
   4ad7a:	call   *0xf15c8(%rip)        # 13c348 <_DYNAMIC+0x2b8>
   4ad80:	ud2
   4ad82:	mov    0x1a8(%rsp),%rdi
   4ad8a:	mov    0x1b0(%rsp),%rsi
   4ad92:	lea    0xe8cf7(%rip),%rdx        # 133a90 <_ZN15funnel_patterns46__DIVAN_BENCH_PAT_U128_CAT_UNROLLED4__U64__W514PUSH17h68d55424f84a52b4E+0x48>
   4ad99:	call   *0xf15c9(%rip)        # 13c368 <_DYNAMIC+0x2d8>
   4ad9f:	mov    %r15,0x40(%rsp)
   4ada4:	mov    %rax,%rbx
   4ada7:	test   %r14,%r14
   4adaa:	jne    4ade0 <_ZN15funnel_patterns25pat_macro_shape__u64__w5117he54d8f6ffc7df352E+0x1040>
   4adac:	jmp    4adf5 <_ZN15funnel_patterns25pat_macro_shape__u64__w5117he54d8f6ffc7df352E+0x1055>
   4adae:	mov    %rax,%rbx
   4adb1:	mov    0x120(%rsp),%rax
   4adb9:	lock decq (%rax)
   4adbd:	jne    4ade0 <_ZN15funnel_patterns25pat_macro_shape__u64__w5117he54d8f6ffc7df352E+0x1040>
   4adbf:	lea    0x120(%rsp),%rdi
   4adc7:	call   *0xf1543(%rip)        # 13c310 <_DYNAMIC+0x280>
   4adcd:	jmp    4ade0 <_ZN15funnel_patterns25pat_macro_shape__u64__w5117he54d8f6ffc7df352E+0x1040>
   4adcf:	call   *0xf159b(%rip)        # 13c370 <_DYNAMIC+0x2e0>
   4add5:	jmp    4addd <_ZN15funnel_patterns25pat_macro_shape__u64__w5117he54d8f6ffc7df352E+0x103d>
   4add7:	jmp    4addd <_ZN15funnel_patterns25pat_macro_shape__u64__w5117he54d8f6ffc7df352E+0x103d>
   4add9:	jmp    4addd <_ZN15funnel_patterns25pat_macro_shape__u64__w5117he54d8f6ffc7df352E+0x103d>
   4addb:	jmp    4addd <_ZN15funnel_patterns25pat_macro_shape__u64__w5117he54d8f6ffc7df352E+0x103d>
   4addd:	mov    %rax,%rbx
   4ade0:	mov    $0xd0,%esi
   4ade5:	mov    $0x10,%edx
   4adea:	mov    0x40(%rsp),%rdi
   4adef:	call   *0xf154b(%rip)        # 13c340 <_DYNAMIC+0x2b0>
   4adf5:	mov    %rbx,%rdi
   4adf8:	call   1328b0 <_Unwind_Resume@plt>

Disassembly of section .init:

Disassembly of section .fini:

Disassembly of section .plt:
