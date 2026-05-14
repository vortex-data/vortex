
/home/user/vortex/target/release/deps/funnel_patterns-21c1c00107f42b8a:     file format elf64-x86-64


Disassembly of section .text:

00000000000483c0 <_ZN15funnel_patterns22pat_u128_cat__u64__w5117h72fda062540fe8b8E>:
   483c0:	push   %rbp
   483c1:	push   %r15
   483c3:	push   %r14
   483c5:	push   %r13
   483c7:	push   %r12
   483c9:	push   %rbx
   483ca:	sub    $0x1000,%rsp
   483d1:	movq   $0x0,(%rsp)
   483d9:	sub    $0x1000,%rsp
   483e0:	movq   $0x0,(%rsp)
   483e8:	sub    $0x1000,%rsp
   483ef:	movq   $0x0,(%rsp)
   483f7:	sub    $0xc18,%rsp
   483fe:	mov    %rdi,%r12
   48401:	lea    0x298(%rsp),%rdi
   48409:	mov    $0x1980,%edx
   4840e:	xor    %esi,%esi
   48410:	call   *0xf3eaa(%rip)        # 13c2c0 <memset@GLIBC_2.2.5>
   48416:	vmovdqa64 -0x38ae0(%rip),%zmm0        # f940 <__abi_tag+0xf644>
   48420:	mov    $0x38,%eax
   48425:	vpbroadcastq -0x37c9f(%rip),%zmm1        # 10790 <__abi_tag+0x10494>
   4842f:	vpbroadcastq -0x37dc9(%rip),%zmm2        # 10670 <__abi_tag+0x10374>
   48439:	vpbroadcastq -0x37cab(%rip),%zmm3        # 10798 <__abi_tag+0x1049c>
   48443:	vpbroadcastq -0x37c2d(%rip),%zmm4        # 10820 <__abi_tag+0x10524>
   4844d:	vpbroadcastq -0x37c07(%rip),%zmm5        # 10850 <__abi_tag+0x10554>
   48457:	vpbroadcastq -0x37dd1(%rip),%zmm6        # 10690 <__abi_tag+0x10394>
   48461:	vpbroadcastq -0x37be3(%rip),%zmm7        # 10888 <__abi_tag+0x1058c>
   4846b:	vpbroadcastq -0x37c1d(%rip),%zmm8        # 10858 <__abi_tag+0x1055c>
   48475:	vpbroadcastq -0x37bff(%rip),%zmm9        # 10880 <__abi_tag+0x10584>
   4847f:	nop
   48480:	vpmullq %zmm1,%zmm0,%zmm10
   48486:	vpaddq %zmm2,%zmm10,%zmm11
   4848c:	vpaddq %zmm3,%zmm10,%zmm12
   48492:	vmovdqu64 %zmm10,0xd8(%rsp,%rax,8)
   4849d:	vmovdqu64 %zmm11,0x118(%rsp,%rax,8)
   484a8:	vpaddq %zmm4,%zmm10,%zmm11
   484ae:	vmovdqu64 %zmm12,0x158(%rsp,%rax,8)
   484b9:	vmovdqu64 %zmm11,0x198(%rsp,%rax,8)
   484c4:	cmp    $0x338,%rax
   484ca:	je     4851f <_ZN15funnel_patterns22pat_u128_cat__u64__w5117h72fda062540fe8b8E+0x15f>
   484cc:	vpaddq %zmm5,%zmm10,%zmm11
   484d2:	vpaddq %zmm6,%zmm10,%zmm12
   484d8:	vpaddq %zmm7,%zmm10,%zmm13
   484de:	vpaddq %zmm8,%zmm10,%zmm10
   484e4:	vmovdqu64 %zmm11,0x1d8(%rsp,%rax,8)
   484ef:	vmovdqu64 %zmm12,0x218(%rsp,%rax,8)
   484fa:	vmovdqu64 %zmm13,0x258(%rsp,%rax,8)
   48505:	vmovdqu64 %zmm10,0x298(%rsp,%rax,8)
   48510:	vpaddq %zmm9,%zmm0,%zmm0
   48516:	add    $0x40,%rax
   4851a:	jmp    48480 <_ZN15funnel_patterns22pat_u128_cat__u64__w5117h72fda062540fe8b8E+0xc0>
   4851f:	vmovaps -0x38ba9(%rip),%zmm0        # f980 <__abi_tag+0xf684>
   48529:	vmovups %zmm0,0x1b98(%rsp)
   48534:	vmovdqa64 -0x38b7e(%rip),%zmm0        # f9c0 <__abi_tag+0xf6c4>
   4853e:	vmovdqu64 %zmm0,0x1bd8(%rsp)
   48549:	lea    0x2298(%rsp),%rbx
   48551:	lea    0x298(%rsp),%r14
   48559:	mov    $0x1980,%edx
   4855e:	mov    %rbx,%rdi
   48561:	mov    %r14,%rsi
   48564:	vzeroupper
   48567:	call   *0xf3d5b(%rip)        # 13c2c8 <memcpy@GLIBC_2.14>
   4856d:	movq   $0x3b9aca07,0xf0(%rsp)
   48579:	xor    %ebp,%ebp
   4857b:	mov    $0x2000,%edx
   48580:	mov    %r14,%rdi
   48583:	xor    %esi,%esi
   48585:	call   *0xf3d35(%rip)        # 13c2c0 <memset@GLIBC_2.2.5>
   4858b:	mov    %rbx,0x208(%rsp)
   48593:	lea    0xf0(%rsp),%rax
   4859b:	mov    %rax,0x210(%rsp)
   485a3:	mov    %r14,0x218(%rsp)
   485ab:	lea    0x208(%rsp),%rax
   485b3:	mov    %rax,0xf8(%rsp)
   485bb:	lea    0xf8(%rsp),%rax
   485c3:	mov    %rax,0x100(%rsp)
   485cb:	movq   $0x1,0x100(%r12)
   485d7:	movb   $0x1,0x108(%r12)
   485e0:	mov    0xf0(%r12),%rdx
   485e8:	mov    0xf8(%r12),%rax
   485f0:	movzbl 0x8(%rdx),%ecx
   485f4:	mov    %cl,0x3(%rsp)
   485f8:	cmp    $0x1,%cl
   485fb:	jne    48603 <_ZN15funnel_patterns22pat_u128_cat__u64__w5117h72fda062540fe8b8E+0x243>
   485fd:	xor    %esi,%esi
   485ff:	xor    %ecx,%ecx
   48601:	jmp    4862d <_ZN15funnel_patterns22pat_u128_cat__u64__w5117h72fda062540fe8b8E+0x26d>
   48603:	cmpb   $0x0,0x60(%rax)
   48607:	je     4861c <_ZN15funnel_patterns22pat_u128_cat__u64__w5117h72fda062540fe8b8E+0x25c>
   48609:	mov    0x64(%rax),%ecx
   4860c:	mov    %ecx,0x8(%rsp)
   48610:	mov    $0x2,%ebp
   48615:	mov    $0x1,%sil
   48618:	xor    %ecx,%ecx
   4861a:	jmp    4862d <_ZN15funnel_patterns22pat_u128_cat__u64__w5117h72fda062540fe8b8E+0x26d>
   4861c:	mov    $0x1,%cl
   4861e:	movl   $0x1,0x8(%rsp)
   48626:	xor    %esi,%esi
   48628:	mov    $0x1,%ebp
   4862d:	mov    (%rdx),%r14
   48630:	test   %r14,%r14
   48633:	lea    0x27(%rsp),%rdx
   48638:	mov    %rdx,0x220(%rsp)
   48640:	setne  0x238(%rsp)
   48648:	lea    0x100(%rsp),%rdi
   48650:	mov    %rdi,0x228(%rsp)
   48658:	mov    %rdx,0x230(%rsp)
   48660:	mov    0x70(%rax),%edi
   48663:	movq   $0x0,0x88(%rsp)
   4866f:	mov    $0x0,%edx
   48674:	mov    %rdx,0x80(%rsp)
   4867c:	cmp    $0x3b9aca00,%edi
   48682:	je     486bd <_ZN15funnel_patterns22pat_u128_cat__u64__w5117h72fda062540fe8b8E+0x2fd>
   48684:	mov    $0x3b9aca00,%edx
   48689:	mulx   0x68(%rax),%r8,%r9
   4868f:	mov    %edi,%edx
   48691:	add    %r8,%rdx
   48694:	adc    $0x0,%r9
   48698:	imul   $0x3e8,%r9,%rdi
   4869f:	mov    $0x3e8,%r8d
   486a5:	mulx   %r8,%rdx,%r8
   486aa:	mov    %rdx,0x88(%rsp)
   486b2:	add    %rdi,%r8
   486b5:	mov    %r8,0x80(%rsp)
   486bd:	movq   $0xffffffffffffffff,0x48(%rsp)
   486c6:	mov    0x80(%rax),%r8d
   486cd:	movq   $0xffffffffffffffff,0x38(%rsp)
   486d6:	cmp    $0x3b9aca00,%r8d
   486dd:	je     4871e <_ZN15funnel_patterns22pat_u128_cat__u64__w5117h72fda062540fe8b8E+0x35e>
   486df:	mov    $0x3b9aca00,%edx
   486e4:	mulx   0x78(%rax),%r9,%rdi
   486ea:	mov    %r8d,%edx
   486ed:	add    %r9,%rdx
   486f0:	adc    $0x0,%rdi
   486f4:	mov    $0x3e8,%r8d
   486fa:	mulx   %r8,%r8,%r9
   486ff:	mov    %r9,0x38(%rsp)
   48704:	mov    %r8,0x48(%rsp)
   48709:	or     %rdi,%rdx
   4870c:	je     492b6 <_ZN15funnel_patterns22pat_u128_cat__u64__w5117h72fda062540fe8b8E+0xef6>
   48712:	imul   $0x3e8,%rdi,%rdx
   48719:	add    %rdx,0x38(%rsp)
   4871e:	mov    0x58(%rax),%edx
   48721:	cmp    $0x1,%edx
   48724:	jne    48730 <_ZN15funnel_patterns22pat_u128_cat__u64__w5117h72fda062540fe8b8E+0x370>
   48726:	cmpl   $0x0,0x5c(%rax)
   4872a:	je     492b6 <_ZN15funnel_patterns22pat_u128_cat__u64__w5117h72fda062540fe8b8E+0xef6>
   48730:	cmpl   $0x1,0x60(%rax)
   48734:	jne    48740 <_ZN15funnel_patterns22pat_u128_cat__u64__w5117h72fda062540fe8b8E+0x380>
   48736:	cmpl   $0x0,0x64(%rax)
   4873a:	je     492b6 <_ZN15funnel_patterns22pat_u128_cat__u64__w5117h72fda062540fe8b8E+0xef6>
   48740:	mov    %r14,0x60(%rsp)
   48745:	test   %dl,%sil
   48748:	je     4875b <_ZN15funnel_patterns22pat_u128_cat__u64__w5117h72fda062540fe8b8E+0x39b>
   4874a:	mov    0x5c(%rax),%edx
   4874d:	mov    %edx,0x18(%rsp)
   48751:	movl   $0x1,0xc(%rsp)
   48759:	jmp    4876b <_ZN15funnel_patterns22pat_u128_cat__u64__w5117h72fda062540fe8b8E+0x3ab>
   4875b:	movzbl %sil,%edx
   4875f:	mov    %edx,0xc(%rsp)
   48763:	movl   $0x64,0x18(%rsp)
   4876b:	movq   $0x0,0x58(%rsp)
   48774:	mov    $0x0,%edx
   48779:	mov    %rdx,0x50(%rsp)
   4877e:	test   %cl,%cl
   48780:	je     4879d <_ZN15funnel_patterns22pat_u128_cat__u64__w5117h72fda062540fe8b8E+0x3dd>
   48782:	mov    %r14,%rdi
   48785:	call   *0xf3b45(%rip)        # 13c2d0 <_DYNAMIC+0x240>
   4878b:	mov    %rax,0x58(%rsp)
   48790:	mov    %rdx,0x50(%rsp)
   48795:	mov    0xf8(%r12),%rax
   4879d:	cmpb   $0x1,0x3(%rsp)
   487a2:	je     487c3 <_ZN15funnel_patterns22pat_u128_cat__u64__w5117h72fda062540fe8b8E+0x403>
   487a4:	mov    $0x1,%edx
   487a9:	cmpb   $0x0,0x58(%rax)
   487ad:	je     487b2 <_ZN15funnel_patterns22pat_u128_cat__u64__w5117h72fda062540fe8b8E+0x3f2>
   487af:	mov    0x5c(%rax),%edx
   487b2:	mov    (%r12),%rcx
   487b6:	mov    0x10(%r12),%rsi
   487bb:	sub    %rsi,%rcx
   487be:	cmp    %rcx,%rdx
   487c1:	ja     48808 <_ZN15funnel_patterns22pat_u128_cat__u64__w5117h72fda062540fe8b8E+0x448>
   487c3:	testb  $0x1,0x88(%rax)
   487ca:	jne    48821 <_ZN15funnel_patterns22pat_u128_cat__u64__w5117h72fda062540fe8b8E+0x461>
   487cc:	lock orl $0x0,-0x40(%rsp)
   487d2:	test   %r14,%r14
   487d5:	je     487f3 <_ZN15funnel_patterns22pat_u128_cat__u64__w5117h72fda062540fe8b8E+0x433>
   487d7:	lfence
   487da:	rdtsc
   487dc:	shl    $0x20,%rdx
   487e0:	or     %rax,%rdx
   487e3:	mov    %rdx,0x70(%rsp)
   487e8:	lfence
   487eb:	mov    $0x3b9aca00,%r13d
   487f1:	jmp    48801 <_ZN15funnel_patterns22pat_u128_cat__u64__w5117h72fda062540fe8b8E+0x441>
   487f3:	call   *0xf3adf(%rip)        # 13c2d8 <_DYNAMIC+0x248>
   487f9:	mov    %rax,0x70(%rsp)
   487fe:	mov    %edx,%r13d
   48801:	mov    0x60(%rsp),%r14
   48806:	jmp    48827 <_ZN15funnel_patterns22pat_u128_cat__u64__w5117h72fda062540fe8b8E+0x467>
   48808:	mov    %r12,%rdi
   4880b:	call   51800 <_ZN5alloc7raw_vec20RawVecInner$LT$A$GT$7reserve21do_reserve_and_handle17h46771c9d08372974E>
   48810:	mov    0xf8(%r12),%rax
   48818:	testb  $0x1,0x88(%rax)
   4881f:	je     487cc <_ZN15funnel_patterns22pat_u128_cat__u64__w5117h72fda062540fe8b8E+0x40c>
   48821:	mov    $0x3b9aca01,%r13d
   48827:	mov    %r14,%rdi
   4882a:	call   *0xf3ab0(%rip)        # 13c2e0 <_DYNAMIC+0x250>
   48830:	mov    %rax,0xa8(%rsp)
   48838:	mov    0xf3aa9(%rip),%r14        # 13c2e8 <_DYNAMIC+0x258>
   4883f:	mov    0x8(%r14),%eax
   48843:	test   %eax,%eax
   48845:	jne    492e3 <_ZN15funnel_patterns22pat_u128_cat__u64__w5117h72fda062540fe8b8E+0xf23>
   4884b:	mov    (%r14),%rdi
   4884e:	call   *0xf3a9c(%rip)        # 13c2f0 <_DYNAMIC+0x260>
   48854:	mov    0x48(%rsp),%rax
   48859:	or     0x38(%rsp),%rax
   4885e:	je     49272 <_ZN15funnel_patterns22pat_u128_cat__u64__w5117h72fda062540fe8b8E+0xeb2>
   48864:	lea    0x50(%r12),%rax
   48869:	mov    %rax,0x28(%rsp)
   4886e:	lea    0x18(%r12),%rax
   48873:	mov    %rax,0x78(%rsp)
   48878:	mov    $0x10,%r15d
   4887e:	mov    $0x1,%al
   48880:	xor    %edx,%edx
   48882:	xor    %esi,%esi
   48884:	xor    %r14d,%r14d
   48887:	mov    %r12,0x30(%rsp)
   4888c:	mov    %r13d,0x6c(%rsp)
   48891:	jmp    4890b <_ZN15funnel_patterns22pat_u128_cat__u64__w5117h72fda062540fe8b8E+0x54b>
   48893:	data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
   488a0:	mov    0x90(%rsp),%rdi
   488a8:	cmp    $0x3e9,%rdi
   488af:	mov    $0x3e8,%eax
   488b4:	cmovae %rdi,%rax
   488b8:	mov    0x98(%rsp),%rsi
   488c0:	test   %rsi,%rsi
   488c3:	mov    $0x3e8,%ecx
   488c8:	cmove  %rcx,%rdi
   488cc:	cmove  %rax,%rdi
   488d0:	mov    0xb0(%rsp),%rdx
   488d8:	add    %rdi,%rdx
   488db:	adc    %rsi,%r14
   488de:	mov    $0xffffffffffffffff,%rax
   488e5:	cmovb  %rax,%r14
   488e9:	cmovb  %rax,%rdx
   488ed:	mov    %r14,%rsi
   488f0:	mov    $0x1,%r14d
   488f6:	xor    %eax,%eax
   488f8:	cmp    0x48(%rsp),%rdx
   488fd:	mov    %rsi,%rcx
   48900:	sbb    0x38(%rsp),%rcx
   48905:	jae    4927b <_ZN15funnel_patterns22pat_u128_cat__u64__w5117h72fda062540fe8b8E+0xebb>
   4890b:	cmp    0x88(%rsp),%rdx
   48913:	mov    %rsi,0xe8(%rsp)
   4891b:	mov    %rsi,%rcx
   4891e:	sbb    0x80(%rsp),%rcx
   48926:	jb     4893a <_ZN15funnel_patterns22pat_u128_cat__u64__w5117h72fda062540fe8b8E+0x57a>
   48928:	testb  $0x1,0xc(%rsp)
   4892d:	je     4893a <_ZN15funnel_patterns22pat_u128_cat__u64__w5117h72fda062540fe8b8E+0x57a>
   4892f:	cmpl   $0x0,0x18(%rsp)
   48934:	je     4927b <_ZN15funnel_patterns22pat_u128_cat__u64__w5117h72fda062540fe8b8E+0xebb>
   4893a:	mov    %rdx,0xb0(%rsp)
   48942:	test   %ebp,%ebp
   48944:	mov    0x8(%rsp),%ecx
   48948:	mov    $0x1,%edx
   4894d:	cmove  %edx,%ecx
   48950:	mov    %ecx,0x4(%rsp)
   48954:	mov    %ecx,0x48(%r12)
   48959:	movq   $0x0,0x268(%rsp)
   48965:	mov    0x28(%rsp),%rcx
   4896a:	mov    %rcx,0x240(%rsp)
   48972:	lea    0x220(%rsp),%rcx
   4897a:	mov    %rcx,0x248(%rsp)
   48982:	lea    0x4(%rsp),%rcx
   48987:	mov    %rcx,0x250(%rsp)
   4898f:	lea    0x268(%rsp),%rcx
   48997:	mov    %rcx,0x258(%rsp)
   4899f:	lea    0x60(%rsp),%rcx
   489a4:	mov    %rcx,0x260(%rsp)
   489ac:	test   $0x1,%al
   489ae:	jne    48b20 <_ZN15funnel_patterns22pat_u128_cat__u64__w5117h72fda062540fe8b8E+0x760>
   489b4:	imul   $0xd0,%r14,%rax
   489bb:	add    %r15,%rax
   489be:	mov    %rax,%rdx
   489c1:	sub    %r15,%rdx
   489c4:	add    $0xffffffffffffff30,%rdx
   489cb:	movabs $0x4ec4ec4ec4ec4ec5,%rcx
   489d5:	mulx   %rcx,%rcx,%rcx
   489da:	mov    %r15,%rsi
   489dd:	cmp    $0xc30,%rdx
   489e4:	jb     48b70 <_ZN15funnel_patterns22pat_u128_cat__u64__w5117h72fda062540fe8b8E+0x7b0>
   489ea:	shr    $0x6,%rcx
   489ee:	inc    %rcx
   489f1:	cmp    $0x3330,%rdx
   489f8:	jae    48a10 <_ZN15funnel_patterns22pat_u128_cat__u64__w5117h72fda062540fe8b8E+0x650>
   489fa:	xor    %edx,%edx
   489fc:	mov    %r15,%rdi
   489ff:	jmp    48abf <_ZN15funnel_patterns22pat_u128_cat__u64__w5117h72fda062540fe8b8E+0x6ff>
   48a04:	data16 data16 cs nopw 0x0(%rax,%rax,1)
   48a10:	mov    %rcx,%rdx
   48a13:	movabs $0x3ffffffffffffc0,%rsi
   48a1d:	and    %rsi,%rdx
   48a20:	imul   $0xd0,%rdx,%rsi
   48a27:	lea    (%r15,%rsi,1),%rdi
   48a2b:	mov    %rdx,%r8
   48a2e:	mov    %r15,%r9
   48a31:	vpbroadcastd -0x38167(%rip),%zmm0        # 108d4 <__abi_tag+0x105d8>
   48a3b:	vmovdqa64 -0x39045(%rip),%zmm1        # fa00 <__abi_tag+0xf704>
   48a45:	vmovdqa64 -0x3900f(%rip),%zmm2        # fa40 <__abi_tag+0xf744>
   48a4f:	vmovdqa64 -0x38fd9(%rip),%zmm3        # fa80 <__abi_tag+0xf784>
   48a59:	vmovdqa64 -0x38fa3(%rip),%zmm4        # fac0 <__abi_tag+0xf7c4>
   48a63:	data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
   48a70:	kxnorw %k0,%k0,%k1
   48a74:	vpscatterdd %zmm0,0x8(%r9,%zmm1,1){%k1}
   48a7c:	kxnorw %k0,%k0,%k1
   48a80:	vpscatterdd %zmm0,0x8(%r9,%zmm2,1){%k1}
   48a88:	kxnorw %k0,%k0,%k1
   48a8c:	vpscatterdd %zmm0,0x8(%r9,%zmm3,1){%k1}
   48a94:	kxnorw %k0,%k0,%k1
   48a98:	vpscatterdd %zmm0,0x8(%r9,%zmm4,1){%k1}
   48aa0:	add    $0x3400,%r9
   48aa7:	add    $0xffffffffffffffc0,%r8
   48aab:	jne    48a70 <_ZN15funnel_patterns22pat_u128_cat__u64__w5117h72fda062540fe8b8E+0x6b0>
   48aad:	cmp    %rdx,%rcx
   48ab0:	je     48b83 <_ZN15funnel_patterns22pat_u128_cat__u64__w5117h72fda062540fe8b8E+0x7c3>
   48ab6:	test   $0x30,%cl
   48ab9:	je     48b6d <_ZN15funnel_patterns22pat_u128_cat__u64__w5117h72fda062540fe8b8E+0x7ad>
   48abf:	movabs $0x3ffffffffffffc0,%rsi
   48ac9:	lea    0x30(%rsi),%r8
   48acd:	and    %rcx,%r8
   48ad0:	imul   $0xd0,%r8,%rsi
   48ad7:	add    %r15,%rsi
   48ada:	sub    %r8,%rdx
   48add:	vpbroadcastd -0x38213(%rip),%zmm0        # 108d4 <__abi_tag+0x105d8>
   48ae7:	vmovdqa64 -0x390f1(%rip),%zmm1        # fa00 <__abi_tag+0xf704>
   48af1:	data16 data16 data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
   48b00:	kxnorw %k0,%k0,%k1
   48b04:	vpscatterdd %zmm0,0x8(%rdi,%zmm1,1){%k1}
   48b0c:	add    $0xd00,%rdi
   48b13:	add    $0x10,%rdx
   48b17:	jne    48b00 <_ZN15funnel_patterns22pat_u128_cat__u64__w5117h72fda062540fe8b8E+0x740>
   48b19:	cmp    %r8,%rcx
   48b1c:	jne    48b70 <_ZN15funnel_patterns22pat_u128_cat__u64__w5117h72fda062540fe8b8E+0x7b0>
   48b1e:	jmp    48b83 <_ZN15funnel_patterns22pat_u128_cat__u64__w5117h72fda062540fe8b8E+0x7c3>
   48b20:	movq   $0x0,0x128(%rsp)
   48b2c:	mov    $0x10,%esi
   48b31:	mov    $0xd0,%edx
   48b36:	lea    0x1a0(%rsp),%rdi
   48b3e:	lea    0x120(%rsp),%rcx
   48b46:	call   516c0 <_ZN5alloc7raw_vec11finish_grow17hedc133b40cb748a9E>
   48b4b:	cmpb   $0x0,0x1a0(%rsp)
   48b53:	jne    493a2 <_ZN15funnel_patterns22pat_u128_cat__u64__w5117h72fda062540fe8b8E+0xfe2>
   48b59:	mov    0x1a8(%rsp),%r15
   48b61:	lea    0xd0(%r15),%rax
   48b68:	jmp    489be <_ZN15funnel_patterns22pat_u128_cat__u64__w5117h72fda062540fe8b8E+0x5fe>
   48b6d:	add    %r15,%rsi
   48b70:	movl   $0x3b9aca01,0x8(%rsi)
   48b77:	add    $0xd0,%rsi
   48b7e:	cmp    %rax,%rsi
   48b81:	jne    48b70 <_ZN15funnel_patterns22pat_u128_cat__u64__w5117h72fda062540fe8b8E+0x7b0>
   48b83:	mov    %r12,%rbx
   48b86:	mov    %r15,0x40(%rsp)
   48b8b:	vzeroupper
   48b8e:	call   *0xf3764(%rip)        # 13c2f8 <_DYNAMIC+0x268>
   48b94:	mov    %rax,0x120(%rsp)
   48b9c:	movq   $0x0,0x128(%rsp)
   48ba8:	lea    0x6501(%rip),%rax        # 4f0b0 <_ZN30codspeed_divan_compat_walltime11thread_pool19TaskShared$LT$F$GT$3new4call17h21f1025d8ba7895aE>
   48baf:	mov    %rax,0x130(%rsp)
   48bb7:	lea    0x240(%rsp),%rax
   48bbf:	mov    %rax,0x138(%rsp)
   48bc7:	mov    %r15,0x140(%rsp)
   48bcf:	mov    0xf372a(%rip),%rdi        # 13c300 <_DYNAMIC+0x270>
   48bd6:	xor    %esi,%esi
   48bd8:	lea    0x120(%rsp),%rdx
   48be0:	call   *0xf3722(%rip)        # 13c308 <_DYNAMIC+0x278>
   48be6:	mov    0x120(%rsp),%rax
   48bee:	lock decq (%rax)
   48bf2:	jne    48c02 <_ZN15funnel_patterns22pat_u128_cat__u64__w5117h72fda062540fe8b8E+0x842>
   48bf4:	lea    0x120(%rsp),%rdi
   48bfc:	call   *0xf370e(%rip)        # 13c310 <_DYNAMIC+0x280>
   48c02:	cmpl   $0x3b9aca01,0x8(%r15)
   48c0a:	je     492ff <_ZN15funnel_patterns22pat_u128_cat__u64__w5117h72fda062540fe8b8E+0xf3f>
   48c10:	cmpb   $0x1,0x3(%rsp)
   48c15:	je     492c8 <_ZN15funnel_patterns22pat_u128_cat__u64__w5117h72fda062540fe8b8E+0xf08>
   48c1b:	vmovups 0x10(%r15),%xmm0
   48c21:	vmovaps %xmm0,0x1a0(%rsp)
   48c2a:	vmovdqu (%r15),%xmm0
   48c2f:	vmovdqa %xmm0,0x120(%rsp)
   48c38:	mov    0xc0(%r15),%rdx
   48c3f:	lea    0x1a0(%rsp),%rdi
   48c47:	lea    0x120(%rsp),%rsi
   48c4f:	call   *0xf36c3(%rip)        # 13c318 <_DYNAMIC+0x288>
   48c55:	lea    0x10(%r15),%rax
   48c59:	mov    %rax,%r12
   48c5c:	vmovups (%rax),%xmm0
   48c60:	vmovaps %xmm0,0x1a0(%rsp)
   48c69:	vmovdqu (%r15),%xmm0
   48c6e:	vmovdqa %xmm0,0x120(%rsp)
   48c77:	mov    0xc0(%r15),%rdx
   48c7e:	lea    0x1a0(%rsp),%rdi
   48c86:	lea    0x120(%rsp),%rsi
   48c8e:	call   *0xf3684(%rip)        # 13c318 <_DYNAMIC+0x288>
   48c94:	mov    %rax,%rsi
   48c97:	cmp    $0x1,%ebp
   48c9a:	mov    %rdx,0x98(%rsp)
   48ca2:	mov    %rax,0x90(%rsp)
   48caa:	jne    48d20 <_ZN15funnel_patterns22pat_u128_cat__u64__w5117h72fda062540fe8b8E+0x960>
   48cac:	mov    %rbx,%rcx
   48caf:	movq   $0x0,0x10(%rbx)
   48cb7:	cmpq   $0x0,0x30(%rbx)
   48cbc:	mov    0x58(%rsp),%r13
   48cc1:	mov    0x50(%rsp),%rbp
   48cc6:	je     48d46 <_ZN15funnel_patterns22pat_u128_cat__u64__w5117h72fda062540fe8b8E+0x986>
   48cc8:	mov    0x20(%rcx),%r14
   48ccc:	test   %r14,%r14
   48ccf:	je     48d35 <_ZN15funnel_patterns22pat_u128_cat__u64__w5117h72fda062540fe8b8E+0x975>
   48cd1:	mov    0x78(%rsp),%rax
   48cd6:	mov    (%rax),%rdi
   48cd9:	lea    0x11(%r14),%rdx
   48cdd:	mov    $0xff,%esi
   48ce2:	call   *0xf35d8(%rip)        # 13c2c0 <memset@GLIBC_2.2.5>
   48ce8:	mov    0x90(%rsp),%rsi
   48cf0:	mov    0x98(%rsp),%rdx
   48cf8:	lea    0x1(%r14),%rax
   48cfc:	mov    %rax,%rcx
   48cff:	shr    $0x3,%rcx
   48d03:	and    $0xfffffffffffffff8,%rax
   48d07:	sub    %rcx,%rax
   48d0a:	cmp    $0x8,%r14
   48d0e:	cmovb  %r14,%rax
   48d12:	jmp    48d37 <_ZN15funnel_patterns22pat_u128_cat__u64__w5117h72fda062540fe8b8E+0x977>
   48d14:	data16 data16 cs nopw 0x0(%rax,%rax,1)
   48d20:	mov    %ebp,0x14(%rsp)
   48d24:	mov    0x4(%rsp),%eax
   48d28:	mov    %rax,0xa0(%rsp)
   48d30:	jmp    48dd7 <_ZN15funnel_patterns22pat_u128_cat__u64__w5117h72fda062540fe8b8E+0xa17>
   48d35:	xor    %eax,%eax
   48d37:	mov    %rbx,%rcx
   48d3a:	movq   $0x0,0x30(%rbx)
   48d42:	mov    %rax,0x28(%rbx)
   48d46:	cmpq   $0x0,0x68(%rcx)
   48d4b:	je     48d55 <_ZN15funnel_patterns22pat_u128_cat__u64__w5117h72fda062540fe8b8E+0x995>
   48d4d:	movq   $0x0,0x60(%rcx)
   48d55:	cmpq   $0x0,0x90(%rcx)
   48d5d:	je     48d6a <_ZN15funnel_patterns22pat_u128_cat__u64__w5117h72fda062540fe8b8E+0x9aa>
   48d5f:	movq   $0x0,0x88(%rcx)
   48d6a:	cmpq   $0x0,0xb8(%rcx)
   48d72:	je     48d7f <_ZN15funnel_patterns22pat_u128_cat__u64__w5117h72fda062540fe8b8E+0x9bf>
   48d74:	movq   $0x0,0xb0(%rcx)
   48d7f:	cmpq   $0x0,0xe0(%rcx)
   48d87:	je     48d94 <_ZN15funnel_patterns22pat_u128_cat__u64__w5117h72fda062540fe8b8E+0x9d4>
   48d89:	movq   $0x0,0xd8(%rcx)
   48d94:	mov    %r13,%rax
   48d97:	or     %rbp,%rax
   48d9a:	je     49393 <_ZN15funnel_patterns22pat_u128_cat__u64__w5117h72fda062540fe8b8E+0xfd3>
   48da0:	mov    %rsi,%rdi
   48da3:	mov    %rdx,%rsi
   48da6:	mov    %r13,%rdx
   48da9:	mov    %rbp,%rcx
   48dac:	call   *0xf356e(%rip)        # 13c320 <_DYNAMIC+0x290>
   48db2:	cmp    $0x65,%rax
   48db6:	sbb    $0x0,%rdx
   48dba:	mov    0x4(%rsp),%ecx
   48dbe:	mov    %rcx,0xa0(%rsp)
   48dc6:	jae    48df0 <_ZN15funnel_patterns22pat_u128_cat__u64__w5117h72fda062540fe8b8E+0xa30>
   48dc8:	lea    (%rcx,%rcx,1),%eax
   48dcb:	mov    %eax,0x8(%rsp)
   48dcf:	movl   $0x1,0x14(%rsp)
   48dd7:	mov    0x18(%rsp),%eax
   48ddb:	mov    %eax,0x10(%rsp)
   48ddf:	mov    %r12,%rdx
   48de2:	jmp    48e2d <_ZN15funnel_patterns22pat_u128_cat__u64__w5117h72fda062540fe8b8E+0xa6d>
   48de4:	data16 data16 cs nopw 0x0(%rax,%rax,1)
   48df0:	mov    0xf8(%rbx),%rax
   48df7:	movl   $0x1,0xc(%rsp)
   48dff:	cmpb   $0x0,0x58(%rax)
   48e03:	mov    %r12,%rdx
   48e06:	je     48e19 <_ZN15funnel_patterns22pat_u128_cat__u64__w5117h72fda062540fe8b8E+0xa59>
   48e08:	movl   $0x2,0x14(%rsp)
   48e10:	mov    0x5c(%rax),%eax
   48e13:	mov    %eax,0x10(%rsp)
   48e17:	jmp    48e29 <_ZN15funnel_patterns22pat_u128_cat__u64__w5117h72fda062540fe8b8E+0xa69>
   48e19:	movl   $0x64,0x10(%rsp)
   48e21:	movl   $0x2,0x14(%rsp)
   48e29:	mov    %ecx,0x8(%rsp)
   48e2d:	mov    0xa8(%rsp),%rax
   48e35:	mov    (%rax),%rbp
   48e38:	mov    0x8(%rax),%r13
   48e3c:	mov    0x18(%rax),%r14
   48e40:	mov    0x10(%rax),%rcx
   48e44:	mov    %rcx,0xc0(%rsp)
   48e4c:	mov    %r15,%rcx
   48e4f:	mov    0x40(%r15),%r15
   48e53:	mov    0x28(%rax),%rsi
   48e57:	mov    %rsi,0xc8(%rsp)
   48e5f:	mov    0x20(%rax),%rsi
   48e63:	mov    %rsi,0xd0(%rsp)
   48e6b:	mov    0x50(%rcx),%rbx
   48e6f:	mov    0x38(%rax),%rsi
   48e73:	mov    %rsi,0xe0(%rsp)
   48e7b:	mov    0x30(%rax),%rax
   48e7f:	mov    %rax,0x18(%rsp)
   48e84:	mov    0x20(%rcx),%rax
   48e88:	mov    %rax,0xd8(%rsp)
   48e90:	mov    0x30(%rcx),%r12
   48e94:	vmovups (%rdx),%xmm0
   48e98:	vmovaps %xmm0,0x1a0(%rsp)
   48ea1:	mov    0x30(%rsp),%rax
   48ea6:	mov    0x10(%rax),%rax
   48eaa:	mov    %rax,0xb8(%rsp)
   48eb2:	vmovdqu (%rcx),%xmm0
   48eb6:	vmovdqa %xmm0,0x120(%rsp)
   48ebf:	mov    0xc0(%rcx),%rdx
   48ec6:	lea    0x1a0(%rsp),%rdi
   48ece:	lea    0x120(%rsp),%rsi
   48ed6:	call   *0xf343c(%rip)        # 13c318 <_DYNAMIC+0x288>
   48edc:	mov    %rax,%rsi
   48edf:	mov    %rdx,%rcx
   48ee2:	mov    0xa0(%rsp),%edi
   48ee9:	mov    %r13,%rax
   48eec:	mul    %rdi
   48eef:	mov    %rbp,%rdx
   48ef2:	mulx   %rdi,%r10,%r9
   48ef7:	seto   %dl
   48efa:	add    %rax,%r9
   48efd:	setb   %r13b
   48f01:	or     %dl,%r13b
   48f04:	mov    %r14,%rax
   48f07:	mul    %r15
   48f0a:	seto   %r11b
   48f0e:	mov    0xc0(%rsp),%rdx
   48f16:	mulx   %r15,%r8,%rdi
   48f1b:	add    %rax,%rdi
   48f1e:	setb   %r14b
   48f22:	or     %r11b,%r14b
   48f25:	mov    0xc8(%rsp),%rax
   48f2d:	mul    %rbx
   48f30:	mov    0xd0(%rsp),%rdx
   48f38:	mulx   %rbx,%rbx,%r11
   48f3d:	seto   %dl
   48f40:	add    %rax,%r11
   48f43:	setb   %bpl
   48f47:	or     %dl,%bpl
   48f4a:	or     %r14b,%bpl
   48f4d:	or     %r13b,%bpl
   48f50:	xor    %r14d,%r14d
   48f53:	add    0xd8(%rsp),%r12
   48f5b:	setb   %r14b
   48f5f:	mov    0xe0(%rsp),%rax
   48f67:	test   %rax,%rax
   48f6a:	setne  %dl
   48f6d:	mov    %r14d,%r15d
   48f70:	and    %dl,%r15b
   48f73:	mul    %r12
   48f76:	seto   %r13b
   48f7a:	or     %r15b,%r13b
   48f7d:	mov    0x18(%rsp),%rdx
   48f82:	imul   %rdx,%r14
   48f86:	add    %rax,%r14
   48f89:	mulx   %r12,%rdx,%rax
   48f8e:	add    %r14,%rax
   48f91:	setb   %r14b
   48f95:	or     %r13b,%r14b
   48f98:	or     %bpl,%r14b
   48f9b:	add    %r10,%r8
   48f9e:	adc    %r9,%rdi
   48fa1:	mov    $0xffffffffffffffff,%r15
   48fa8:	cmovb  %r15,%rdi
   48fac:	cmovb  %r15,%r8
   48fb0:	add    %rbx,%r8
   48fb3:	adc    %r11,%rdi
   48fb6:	cmovb  %r15,%rdi
   48fba:	cmovb  %r15,%r8
   48fbe:	mov    $0xffffffffffffffff,%r10
   48fc5:	mov    $0xffffffffffffffff,%r9
   48fcc:	test   $0x1,%r14b
   48fd0:	jne    48fe6 <_ZN15funnel_patterns22pat_u128_cat__u64__w5117h72fda062540fe8b8E+0xc26>
   48fd2:	add    %rdx,%r8
   48fd5:	adc    %rax,%rdi
   48fd8:	cmovb  %r15,%rdi
   48fdc:	cmovb  %r15,%r8
   48fe0:	mov    %r8,%r10
   48fe3:	mov    %rdi,%r9
   48fe6:	mov    %rsi,%rax
   48fe9:	or     %rcx,%rax
   48fec:	mov    0x50(%rsp),%rdx
   48ff1:	cmove  %rdx,%rcx
   48ff5:	mov    0x58(%rsp),%rax
   48ffa:	cmove  %rax,%rsi
   48ffe:	mov    %rsi,%rbx
   49001:	sub    %r10,%rbx
   49004:	mov    %rcx,%r14
   49007:	sbb    %r9,%r14
   4900a:	cmp    %rsi,%r10
   4900d:	sbb    %rcx,%r9
   49010:	cmovae %rdx,%r14
   49014:	cmovae %rax,%rbx
   49018:	mov    0x30(%rsp),%rax
   4901d:	mov    0x10(%rax),%r15
   49021:	cmp    (%rax),%r15
   49024:	jne    49030 <_ZN15funnel_patterns22pat_u128_cat__u64__w5117h72fda062540fe8b8E+0xc70>
   49026:	mov    0x30(%rsp),%rdi
   4902b:	call   51730 <_ZN5alloc7raw_vec19RawVec$LT$T$C$A$GT$8grow_one17hfe527167958a2dadE>
   49030:	mov    0x30(%rsp),%r12
   49035:	mov    0x8(%r12),%rax
   4903a:	mov    %r15,%rcx
   4903d:	shl    $0x4,%rcx
   49041:	mov    %r14,0x8(%rax,%rcx,1)
   49046:	mov    %rbx,(%rax,%rcx,1)
   4904a:	inc    %r15
   4904d:	mov    %r15,0x10(%r12)
   49052:	mov    0x40(%rsp),%r15
   49057:	mov    0x28(%r15),%rax
   4905b:	or     0x20(%r15),%rax
   4905f:	jne    49080 <_ZN15funnel_patterns22pat_u128_cat__u64__w5117h72fda062540fe8b8E+0xcc0>
   49061:	mov    0x38(%r15),%rax
   49065:	or     0x30(%r15),%rax
   49069:	jne    49080 <_ZN15funnel_patterns22pat_u128_cat__u64__w5117h72fda062540fe8b8E+0xcc0>
   4906b:	mov    0x48(%r15),%rax
   4906f:	or     0x40(%r15),%rax
   49073:	jne    49080 <_ZN15funnel_patterns22pat_u128_cat__u64__w5117h72fda062540fe8b8E+0xcc0>
   49075:	mov    0x58(%r15),%rax
   49079:	or     0x50(%r15),%rax
   4907d:	je     490d6 <_ZN15funnel_patterns22pat_u128_cat__u64__w5117h72fda062540fe8b8E+0xd16>
   4907f:	nop
   49080:	lea    0x20(%r15),%rax
   49084:	vmovdqu64 (%rax),%zmm0
   4908a:	vmovdqu64 0x20(%rax),%zmm1
   49094:	vmovdqu64 %zmm1,0x1c0(%rsp)
   4909c:	vmovdqu64 %zmm0,0x1a0(%rsp)
   490a7:	lea    0x120(%rsp),%rdi
   490af:	mov    0x78(%rsp),%rsi
   490b4:	mov    0xb8(%rsp),%rdx
   490bc:	lea    0x1a0(%rsp),%rcx
   490c4:	vzeroupper
   490c7:	call   51900 <_ZN9hashbrown3map28HashMap$LT$K$C$V$C$S$C$A$GT$6insert17he36ed9c6ee1bc3e6E>
   490cc:	mov    0x30(%rsp),%r12
   490d1:	mov    0x40(%rsp),%r15
   490d6:	cmpq   $0x0,0x68(%r12)
   490dc:	mov    0x14(%rsp),%ebp
   490e0:	mov    0x6c(%rsp),%r13d
   490e5:	mov    0xf323c(%rip),%rbx        # 13c328 <_DYNAMIC+0x298>
   490ec:	mov    0xe8(%rsp),%r14
   490f4:	je     49126 <_ZN15funnel_patterns22pat_u128_cat__u64__w5117h72fda062540fe8b8E+0xd66>
   490f6:	mov    0x4(%rsp),%eax
   490fa:	test   %eax,%eax
   490fc:	je     492ed <_ZN15funnel_patterns22pat_u128_cat__u64__w5117h72fda062540fe8b8E+0xf2d>
   49102:	mov    %eax,%edx
   49104:	mov    0x80(%r15),%rdi
   4910b:	mov    0x88(%r15),%rsi
   49112:	xor    %ecx,%ecx
   49114:	call   *0xf3206(%rip)        # 13c320 <_DYNAMIC+0x290>
   4911a:	mov    0x28(%rsp),%rdi
   4911f:	mov    %rax,%rsi
   49122:	xor    %edx,%edx
   49124:	call   *%rbx
   49126:	cmpq   $0x0,0x90(%r12)
   4912f:	je     49164 <_ZN15funnel_patterns22pat_u128_cat__u64__w5117h72fda062540fe8b8E+0xda4>
   49131:	mov    0x4(%rsp),%eax
   49135:	test   %eax,%eax
   49137:	je     492ed <_ZN15funnel_patterns22pat_u128_cat__u64__w5117h72fda062540fe8b8E+0xf2d>
   4913d:	mov    %eax,%edx
   4913f:	mov    0x90(%r15),%rdi
   49146:	mov    0x98(%r15),%rsi
   4914d:	xor    %ecx,%ecx
   4914f:	call   *0xf31cb(%rip)        # 13c320 <_DYNAMIC+0x290>
   49155:	mov    0x28(%rsp),%rdi
   4915a:	mov    %rax,%rsi
   4915d:	mov    $0x1,%edx
   49162:	call   *%rbx
   49164:	cmpq   $0x0,0xb8(%r12)
   4916d:	je     491a2 <_ZN15funnel_patterns22pat_u128_cat__u64__w5117h72fda062540fe8b8E+0xde2>
   4916f:	mov    0x4(%rsp),%eax
   49173:	test   %eax,%eax
   49175:	je     492ed <_ZN15funnel_patterns22pat_u128_cat__u64__w5117h72fda062540fe8b8E+0xf2d>
   4917b:	mov    %eax,%edx
   4917d:	mov    0xa0(%r15),%rdi
   49184:	mov    0xa8(%r15),%rsi
   4918b:	xor    %ecx,%ecx
   4918d:	call   *0xf318d(%rip)        # 13c320 <_DYNAMIC+0x290>
   49193:	mov    0x28(%rsp),%rdi
   49198:	mov    %rax,%rsi
   4919b:	mov    $0x2,%edx
   491a0:	call   *%rbx
   491a2:	cmpq   $0x0,0xe0(%r12)
   491ab:	je     491e0 <_ZN15funnel_patterns22pat_u128_cat__u64__w5117h72fda062540fe8b8E+0xe20>
   491ad:	mov    0x4(%rsp),%eax
   491b1:	test   %eax,%eax
   491b3:	je     492ed <_ZN15funnel_patterns22pat_u128_cat__u64__w5117h72fda062540fe8b8E+0xf2d>
   491b9:	mov    %eax,%edx
   491bb:	mov    0xb0(%r15),%rdi
   491c2:	mov    0xb8(%r15),%rsi
   491c9:	xor    %ecx,%ecx
   491cb:	call   *0xf314f(%rip)        # 13c320 <_DYNAMIC+0x290>
   491d1:	mov    0x28(%rsp),%rdi
   491d6:	mov    %rax,%rsi
   491d9:	mov    $0x3,%edx
   491de:	call   *%rbx
   491e0:	mov    0x10(%rsp),%ecx
   491e4:	mov    %ecx,%edx
   491e6:	sub    $0x1,%edx
   491e9:	mov    $0x0,%eax
   491ee:	cmovb  %eax,%edx
   491f1:	testb  $0x1,0xc(%rsp)
   491f6:	cmove  %ecx,%edx
   491f9:	mov    %edx,0x18(%rsp)
   491fd:	cmp    $0x3b9aca01,%r13d
   49204:	je     488a0 <_ZN15funnel_patterns22pat_u128_cat__u64__w5117h72fda062540fe8b8E+0x4e0>
   4920a:	mov    0x70(%rsp),%rax
   4920f:	mov    %rax,0x190(%rsp)
   49217:	mov    %r13d,0x198(%rsp)
   4921f:	mov    0x18(%r15),%eax
   49223:	cmp    $0x3b9aca01,%eax
   49228:	je     49384 <_ZN15funnel_patterns22pat_u128_cat__u64__w5117h72fda062540fe8b8E+0xfc4>
   4922e:	mov    0x1c(%r15),%ecx
   49232:	mov    0x10(%r15),%rdx
   49236:	mov    %rdx,0x110(%rsp)
   4923e:	mov    %eax,0x118(%rsp)
   49245:	mov    %ecx,0x11c(%rsp)
   4924c:	mov    0x60(%rsp),%rdx
   49251:	lea    0x110(%rsp),%rdi
   49259:	lea    0x190(%rsp),%rsi
   49261:	call   *0xf30b1(%rip)        # 13c318 <_DYNAMIC+0x288>
   49267:	mov    %rdx,%rsi
   4926a:	mov    %rax,%rdx
   4926d:	jmp    488f0 <_ZN15funnel_patterns22pat_u128_cat__u64__w5117h72fda062540fe8b8E+0x530>
   49272:	mov    $0x10,%r15d
   49278:	xor    %r14d,%r14d
   4927b:	mov    0xf3066(%rip),%rbx        # 13c2e8 <_DYNAMIC+0x258>
   49282:	mov    0x8(%rbx),%eax
   49285:	test   %eax,%eax
   49287:	jne    492dc <_ZN15funnel_patterns22pat_u128_cat__u64__w5117h72fda062540fe8b8E+0xf1c>
   49289:	mov    (%rbx),%rdi
   4928c:	call   *0xf309e(%rip)        # 13c330 <_DYNAMIC+0x2a0>
   49292:	mov    0xf309f(%rip),%rax        # 13c338 <_DYNAMIC+0x2a8>
   49299:	movb   $0x0,(%rax)
   4929c:	test   %r14,%r14
   4929f:	je     492b6 <_ZN15funnel_patterns22pat_u128_cat__u64__w5117h72fda062540fe8b8E+0xef6>
   492a1:	imul   $0xd0,%r14,%rsi
   492a8:	mov    $0x10,%edx
   492ad:	mov    %r15,%rdi
   492b0:	call   *0xf308a(%rip)        # 13c340 <_DYNAMIC+0x2b0>
   492b6:	add    $0x3c18,%rsp
   492bd:	pop    %rbx
   492be:	pop    %r12
   492c0:	pop    %r13
   492c2:	pop    %r14
   492c4:	pop    %r15
   492c6:	pop    %rbp
   492c7:	ret
   492c8:	mov    $0x1,%r14d
   492ce:	mov    0xf3013(%rip),%rbx        # 13c2e8 <_DYNAMIC+0x258>
   492d5:	mov    0x8(%rbx),%eax
   492d8:	test   %eax,%eax
   492da:	je     49289 <_ZN15funnel_patterns22pat_u128_cat__u64__w5117h72fda062540fe8b8E+0xec9>
   492dc:	call   5152e <_ZN3std4sync9once_lock17OnceLock$LT$T$GT$10initialize17h13c8d01c1aad6442E>
   492e1:	jmp    49289 <_ZN15funnel_patterns22pat_u128_cat__u64__w5117h72fda062540fe8b8E+0xec9>
   492e3:	call   5152e <_ZN3std4sync9once_lock17OnceLock$LT$T$GT$10initialize17h13c8d01c1aad6442E>
   492e8:	jmp    4884b <_ZN15funnel_patterns22pat_u128_cat__u64__w5117h72fda062540fe8b8E+0x48b>
   492ed:	lea    0xea834(%rip),%rdi        # 133b28 <_ZN15funnel_patterns46__DIVAN_BENCH_PAT_U128_CAT_UNROLLED4__U64__W514PUSH17h68d55424f84a52b4E+0xe0>
   492f4:	call   *0xf304e(%rip)        # 13c348 <_DYNAMIC+0x2b8>
   492fa:	jmp    493a0 <_ZN15funnel_patterns22pat_u128_cat__u64__w5117h72fda062540fe8b8E+0xfe0>
   492ff:	movq   $0x0,0x108(%rsp)
   4930b:	lea    0x108(%rsp),%rax
   49313:	mov    %rax,0x1a0(%rsp)
   4931b:	mov    0xf302e(%rip),%rax        # 13c350 <_DYNAMIC+0x2c0>
   49322:	mov    %rax,0x1a8(%rsp)
   4932a:	lea    0xea777(%rip),%rax        # 133aa8 <_ZN15funnel_patterns46__DIVAN_BENCH_PAT_U128_CAT_UNROLLED4__U64__W514PUSH17h68d55424f84a52b4E+0x60>
   49331:	mov    %rax,0x120(%rsp)
   49339:	movq   $0x2,0x128(%rsp)
   49345:	movq   $0x0,0x140(%rsp)
   49351:	lea    0x1a0(%rsp),%rax
   49359:	mov    %rax,0x130(%rsp)
   49361:	movq   $0x1,0x138(%rsp)
   4936d:	lea    0xea754(%rip),%rsi        # 133ac8 <_ZN15funnel_patterns46__DIVAN_BENCH_PAT_U128_CAT_UNROLLED4__U64__W514PUSH17h68d55424f84a52b4E+0x80>
   49374:	lea    0x120(%rsp),%rdi
   4937c:	call   *0xf2fd6(%rip)        # 13c358 <_DYNAMIC+0x2c8>
   49382:	jmp    493a0 <_ZN15funnel_patterns22pat_u128_cat__u64__w5117h72fda062540fe8b8E+0xfe0>
   49384:	lea    0xea76d(%rip),%rdi        # 133af8 <_ZN15funnel_patterns46__DIVAN_BENCH_PAT_U128_CAT_UNROLLED4__U64__W514PUSH17h68d55424f84a52b4E+0xb0>
   4938b:	call   *0xf2fcf(%rip)        # 13c360 <_DYNAMIC+0x2d0>
   49391:	jmp    493a0 <_ZN15funnel_patterns22pat_u128_cat__u64__w5117h72fda062540fe8b8E+0xfe0>
   49393:	lea    0xea746(%rip),%rdi        # 133ae0 <_ZN15funnel_patterns46__DIVAN_BENCH_PAT_U128_CAT_UNROLLED4__U64__W514PUSH17h68d55424f84a52b4E+0x98>
   4939a:	call   *0xf2fa8(%rip)        # 13c348 <_DYNAMIC+0x2b8>
   493a0:	ud2
   493a2:	mov    0x1a8(%rsp),%rdi
   493aa:	mov    0x1b0(%rsp),%rsi
   493b2:	lea    0xea6d7(%rip),%rdx        # 133a90 <_ZN15funnel_patterns46__DIVAN_BENCH_PAT_U128_CAT_UNROLLED4__U64__W514PUSH17h68d55424f84a52b4E+0x48>
   493b9:	call   *0xf2fa9(%rip)        # 13c368 <_DYNAMIC+0x2d8>
   493bf:	mov    %r15,0x40(%rsp)
   493c4:	mov    %rax,%rbx
   493c7:	test   %r14,%r14
   493ca:	jne    49400 <_ZN15funnel_patterns22pat_u128_cat__u64__w5117h72fda062540fe8b8E+0x1040>
   493cc:	jmp    49415 <_ZN15funnel_patterns22pat_u128_cat__u64__w5117h72fda062540fe8b8E+0x1055>
   493ce:	mov    %rax,%rbx
   493d1:	mov    0x120(%rsp),%rax
   493d9:	lock decq (%rax)
   493dd:	jne    49400 <_ZN15funnel_patterns22pat_u128_cat__u64__w5117h72fda062540fe8b8E+0x1040>
   493df:	lea    0x120(%rsp),%rdi
   493e7:	call   *0xf2f23(%rip)        # 13c310 <_DYNAMIC+0x280>
   493ed:	jmp    49400 <_ZN15funnel_patterns22pat_u128_cat__u64__w5117h72fda062540fe8b8E+0x1040>
   493ef:	call   *0xf2f7b(%rip)        # 13c370 <_DYNAMIC+0x2e0>
   493f5:	jmp    493fd <_ZN15funnel_patterns22pat_u128_cat__u64__w5117h72fda062540fe8b8E+0x103d>
   493f7:	jmp    493fd <_ZN15funnel_patterns22pat_u128_cat__u64__w5117h72fda062540fe8b8E+0x103d>
   493f9:	jmp    493fd <_ZN15funnel_patterns22pat_u128_cat__u64__w5117h72fda062540fe8b8E+0x103d>
   493fb:	jmp    493fd <_ZN15funnel_patterns22pat_u128_cat__u64__w5117h72fda062540fe8b8E+0x103d>
   493fd:	mov    %rax,%rbx
   49400:	mov    $0xd0,%esi
   49405:	mov    $0x10,%edx
   4940a:	mov    0x40(%rsp),%rdi
   4940f:	call   *0xf2f2b(%rip)        # 13c340 <_DYNAMIC+0x2b0>
   49415:	mov    %rbx,%rdi
   49418:	call   1328b0 <_Unwind_Resume@plt>

Disassembly of section .init:

Disassembly of section .fini:

Disassembly of section .plt:
