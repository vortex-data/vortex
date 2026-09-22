
/tmp/row-fn-bool-linux-state-avx2-target/x86_64-unknown-linux-gnu/release/deps/row_fn_bool_retry-3e4c315823f8823b:	file format elf64-x86-64

Disassembly of section .text:

0000000000334450 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack23collect_bool_words_avx2NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1g_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB35_EENtNtB3j_11row_visitor10RowVisitor19visit_deferred_boolB2Y_bKB35_NCINvXB3V_B3S_NtNtB1g_6row_fn5RowFn8dispatchB3e_E0NCB5N_s_0E0NCB3a_s_0B6z_E0EB3V_>:
  334450:      	pushq	%rbp
  334451:      	movq	%rsp, %rbp
  334454:      	pushq	%r15
  334456:      	pushq	%r14
  334458:      	pushq	%r13
  33445a:      	pushq	%r12
  33445c:      	pushq	%rbx
  33445d:      	subq	$0x38, %rsp
  334461:      	movq	%rsi, %r15
  334464:      	movq	%rdx, %rsi
  334467:      	shrq	$0x6, %rsi
  33446b:      	cmpq	%r15, %rsi
  33446e:      	ja	0x335a63 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack23collect_bool_words_avx2NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1g_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB35_EENtNtB3j_11row_visitor10RowVisitor19visit_deferred_boolB2Y_bKB35_NCINvXB3V_B3S_NtNtB1g_6row_fn5RowFn8dispatchB3e_E0NCB5N_s_0E0NCB3a_s_0B6z_E0EB3V_+0x1613>
  334474:      	movq	%rcx, %r13
  334477:      	movl	%edx, %r11d
  33447a:      	andl	$0x3f, %r11d
  33447e:      	movabsq	$0x7fffffffffffffff, %r10 # imm = 0x7FFFFFFFFFFFFFFF
  334488:      	leaq	(%rdi,%rsi,8), %rax
  33448c:      	movq	%rax, -0x30(%rbp)
  334490:      	testq	%rsi, %rsi
  334493:      	je	0x3353f9 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack23collect_bool_words_avx2NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1g_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB35_EENtNtB3j_11row_visitor10RowVisitor19visit_deferred_boolB2Y_bKB35_NCINvXB3V_B3S_NtNtB1g_6row_fn5RowFn8dispatchB3e_E0NCB5N_s_0E0NCB3a_s_0B6z_E0EB3V_+0xfa9>
  334499:      	movq	%r15, -0x60(%rbp)
  33449d:      	leaq	(,%rsi,8), %rcx
  3344a5:      	movzbl	0x18(%r13), %r15d
  3344aa:      	movq	0x20(%r13), %rbx
  3344ae:      	movq	0x28(%r13), %r14
  3344b2:      	movzbl	0x38(%r13), %eax
  3344b7:      	cmpb	$0x0, (%r13)
  3344bc:      	movq	%rsi, -0x58(%rbp)
  3344c0:      	movq	%r13, -0x50(%rbp)
  3344c4:      	movq	%rdx, -0x48(%rbp)
  3344c8:      	je	0x3345bd <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack23collect_bool_words_avx2NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1g_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB35_EENtNtB3j_11row_visitor10RowVisitor19visit_deferred_boolB2Y_bKB35_NCINvXB3V_B3S_NtNtB1g_6row_fn5RowFn8dispatchB3e_E0NCB5N_s_0E0NCB3a_s_0B6z_E0EB3V_+0x16d>
  3344ce:      	movq	0x10(%r13), %r8
  3344d2:      	movq	(%r8), %rsi
  3344d5:      	testb	%r15b, %r15b
  3344d8:      	je	0x334edc <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack23collect_bool_words_avx2NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1g_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB35_EENtNtB3j_11row_visitor10RowVisitor19visit_deferred_boolB2Y_bKB35_NCINvXB3V_B3S_NtNtB1g_6row_fn5RowFn8dispatchB3e_E0NCB5N_s_0E0NCB3a_s_0B6z_E0EB3V_+0xa8c>
  3344de:      	movq	(%r14), %rdx
  3344e1:      	xorl	%r9d, %r9d
  3344e4:      	movq	%rdx, -0x40(%rbp)
  3344e8:      	cmpq	%rdx, %rsi
  3344eb:      	movabsq	$0x101010101010101, %r15 # imm = 0x101010101010101
  3344f5:      	cmovgeq	%r9, %r15
  3344f9:      	movq	%r15, %r9
  3344fc:      	shrq	$0x38, %r9
  334500:      	movq	%r15, %r14
  334503:      	shrq	$0x30, %r14
  334507:      	movq	%r15, %r12
  33450a:      	shrq	$0x28, %r12
  33450e:      	movq	%r15, %r13
  334511:      	shrq	$0x20, %r13
  334515:      	movl	%r15d, %ebx
  334518:      	shrl	$0x18, %ebx
  33451b:      	movl	%r15d, %r8d
  33451e:      	shrl	$0x10, %r8d
  334522:      	movl	%r15d, %edx
  334525:      	shrl	$0x8, %edx
  334528:      	vmovd	%r15d, %xmm0
  33452d:      	vpinsrb	$0x1, %edx, %xmm0, %xmm0
  334533:      	vpinsrb	$0x2, %r8d, %xmm0, %xmm0
  334539:      	vpinsrb	$0x3, %ebx, %xmm0, %xmm0
  33453f:      	vpinsrb	$0x4, %r13d, %xmm0, %xmm0
  334545:      	vpinsrb	$0x5, %r12d, %xmm0, %xmm0
  33454b:      	vpinsrb	$0x6, %r14d, %xmm0, %xmm0
  334551:      	vpinsrb	$0x7, %r9d, %xmm0, %xmm0
  334557:      	vpinsrb	$0x8, %r15d, %xmm0, %xmm0
  33455d:      	vpinsrb	$0x9, %edx, %xmm0, %xmm0
  334563:      	vpinsrb	$0xa, %r8d, %xmm0, %xmm0
  334569:      	vpinsrb	$0xb, %ebx, %xmm0, %xmm0
  33456f:      	vpinsrb	$0xc, %r13d, %xmm0, %xmm0
  334575:      	vpinsrb	$0xd, %r12d, %xmm0, %xmm0
  33457b:      	vpinsrb	$0xe, %r14d, %xmm0, %xmm0
  334581:      	vpinsrb	$0xf, %r9d, %xmm0, %xmm0
  334587:      	vinserti128	$0x1, %xmm0, %ymm0, %ymm0
  33458d:      	vpxor	%xmm1, %xmm1, %xmm1
  334591:      	vpcmpgtb	%ymm1, %ymm0, %ymm0
  334595:      	vpmovmskb	%ymm0, %edx
  334599:      	movq	%rdx, %r14
  33459c:      	shlq	$0x20, %r14
  3345a0:      	orq	%rdx, %r14
  3345a3:      	addq	$-0x8, %rcx
  3345a7:      	cmpq	$0x18, %rcx
  3345ab:      	jae	0x335311 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack23collect_bool_words_avx2NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1g_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB35_EENtNtB3j_11row_visitor10RowVisitor19visit_deferred_boolB2Y_bKB35_NCINvXB3V_B3S_NtNtB1g_6row_fn5RowFn8dispatchB3e_E0NCB5N_s_0E0NCB3a_s_0B6z_E0EB3V_+0xec1>
  3345b1:      	movq	%rdi, %r12
  3345b4:      	movq	-0x30(%rbp), %rdx
  3345b8:      	jmp	0x3353c0 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack23collect_bool_words_avx2NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1g_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB35_EENtNtB3j_11row_visitor10RowVisitor19visit_deferred_boolB2Y_bKB35_NCINvXB3V_B3S_NtNtB1g_6row_fn5RowFn8dispatchB3e_E0NCB5N_s_0E0NCB3a_s_0B6z_E0EB3V_+0xf70>
  3345bd:      	movq	0x8(%r13), %r12
  3345c1:      	movl	$0x1e0, %r13d           # imm = 0x1E0
  3345c7:      	xorl	%r8d, %r8d
  3345ca:      	vpbroadcastq	-0x2986bb(%rip), %ymm0 # 0x9bf18 <anon.ad39dab31d93760b58da730830a5ab31.633.llvm.6451718222114420760+0x50>
  3345d3:      	vpbroadcastd	-0x2963fc(%rip), %xmm1 # 0x9e1e0 <anon.c60d04bc1ccb4c59159856f942bde85e.7.llvm.13399503513443445546+0x340>
  3345dc:      	vmovdqa	-0x2af8d4(%rip), %xmm2  # 0x84d10 <anon.6c1fb0dbcc3470c819cb5f40076e76af.65.llvm.10840910162848374496+0x640>
  3345e4:      	movl	%eax, %r9d
  3345e7:      	jmp	0x3349f4 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack23collect_bool_words_avx2NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1g_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB35_EENtNtB3j_11row_visitor10RowVisitor19visit_deferred_boolB2Y_bKB35_NCINvXB3V_B3S_NtNtB1g_6row_fn5RowFn8dispatchB3e_E0NCB5N_s_0E0NCB3a_s_0B6z_E0EB3V_+0x5a4>
  3345ec:      	nopl	(%rax)
  3345f0:      	movq	(%r14), %rdx
  3345f3:      	cmpq	%r10, %rdx
  3345f6:      	sete	%r9b
  3345fa:      	vmovq	%rdx, %xmm3
  3345ff:      	vpbroadcastq	%xmm3, %ymm6
  334604:      	vmovd	%eax, %xmm3
  334608:      	vmovdqu	-0x1e0(%r12,%r13), %ymm4
  334612:      	vmovdqu	-0x1c0(%r12,%r13), %ymm9
  33461c:      	vmovdqu	-0x1a0(%r12,%r13), %ymm11
  334626:      	vmovdqu	-0x180(%r12,%r13), %ymm12
  334630:      	vpcmpeqq	%ymm0, %ymm4, %ymm5
  334635:      	vextracti128	$0x1, %ymm5, %xmm7
  33463b:      	vpackssdw	%xmm7, %xmm5, %xmm5
  33463f:      	vpor	%xmm3, %xmm5, %xmm10
  334643:      	vpcmpeqq	%ymm0, %ymm9, %ymm8
  334648:      	vpcmpeqq	%ymm0, %ymm11, %ymm7
  33464d:      	vpcmpeqq	%ymm0, %ymm12, %ymm5
  334652:      	vpcmpgtq	%ymm4, %ymm6, %ymm3
  334657:      	vextracti128	$0x1, %ymm3, %xmm4
  33465d:      	vpackssdw	%xmm4, %xmm3, %xmm3
  334661:      	vpackssdw	%xmm3, %xmm3, %xmm3
  334665:      	vpcmpgtq	%ymm9, %ymm6, %ymm4
  33466a:      	vextracti128	$0x1, %ymm4, %xmm9
  334670:      	vpackssdw	%xmm9, %xmm4, %xmm4
  334675:      	vpackssdw	%xmm4, %xmm4, %xmm4
  334679:      	vpacksswb	%xmm4, %xmm3, %xmm3
  33467d:      	vpcmpgtq	%ymm11, %ymm6, %ymm4
  334682:      	vextracti128	$0x1, %ymm4, %xmm9
  334688:      	vpackssdw	%xmm9, %xmm4, %xmm4
  33468d:      	vpackssdw	%xmm4, %xmm4, %xmm4
  334691:      	vpacksswb	%xmm4, %xmm4, %xmm4
  334695:      	vpand	%xmm2, %xmm4, %xmm4
  334699:      	vpcmpgtq	%ymm12, %ymm6, %ymm9
  33469e:      	vextracti128	$0x1, %ymm9, %xmm11
  3346a4:      	vpackssdw	%xmm11, %xmm9, %xmm9
  3346a9:      	vpackssdw	%xmm9, %xmm9, %xmm9
  3346ae:      	vpacksswb	%xmm9, %xmm9, %xmm9
  3346b3:      	vpshufd	$0xd8, %xmm3, %xmm3     # xmm3 = xmm3[0,2,1,3]
  3346b8:      	vpand	%xmm1, %xmm3, %xmm3
  3346bc:      	vpunpcklqdq	%xmm4, %xmm3, %xmm3 # xmm3 = xmm3[0],xmm4[0]
  3346c0:      	vpbroadcastd	%xmm9, %xmm4
  3346c5:      	vpand	%xmm1, %xmm4, %xmm4
  3346c9:      	vpblendd	$0x8, %xmm4, %xmm3, %xmm11 # xmm11 = xmm3[0,1,2],xmm4[3]
  3346cf:      	vmovdqu	-0x160(%r12,%r13), %ymm3
  3346d9:      	vmovdqu	-0x140(%r12,%r13), %ymm4
  3346e3:      	vmovdqu	-0x120(%r12,%r13), %ymm12
  3346ed:      	vmovdqu	-0x100(%r12,%r13), %ymm13
  3346f7:      	vpcmpeqq	%ymm0, %ymm3, %ymm9
  3346fc:      	vextracti128	$0x1, %ymm9, %xmm14
  334702:      	vpackssdw	%xmm14, %xmm9, %xmm14
  334707:      	vpcmpeqq	%ymm0, %ymm4, %ymm9
  33470c:      	vpor	%ymm9, %ymm8, %ymm9
  334711:      	vpcmpeqq	%ymm0, %ymm12, %ymm8
  334716:      	vpor	%ymm7, %ymm8, %ymm8
  33471a:      	vpcmpeqq	%ymm0, %ymm13, %ymm7
  33471f:      	vpor	%ymm7, %ymm5, %ymm7
  334723:      	vpcmpgtq	%ymm3, %ymm6, %ymm3
  334728:      	vextracti128	$0x1, %ymm3, %xmm5
  33472e:      	vpackssdw	%xmm5, %xmm3, %xmm3
  334732:      	vpackssdw	%xmm3, %xmm3, %xmm3
  334736:      	vpacksswb	%xmm3, %xmm3, %xmm3
  33473a:      	vpand	%xmm2, %xmm3, %xmm3
  33473e:      	vpcmpgtq	%ymm4, %ymm6, %ymm4
  334743:      	vextracti128	$0x1, %ymm4, %xmm5
  334749:      	vpackssdw	%xmm5, %xmm4, %xmm4
  33474d:      	vpackssdw	%xmm4, %xmm4, %xmm4
  334751:      	vpacksswb	%xmm4, %xmm4, %xmm4
  334755:      	vpand	%xmm2, %xmm4, %xmm4
  334759:      	vpcmpgtq	%ymm12, %ymm6, %ymm5
  33475e:      	vextracti128	$0x1, %ymm5, %xmm12
  334764:      	vpackssdw	%xmm12, %xmm5, %xmm5
  334769:      	vpackssdw	%xmm5, %xmm5, %xmm5
  33476d:      	vpacksswb	%xmm5, %xmm5, %xmm5
  334771:      	vpand	%xmm2, %xmm5, %xmm5
  334775:      	vpcmpgtq	%ymm13, %ymm6, %ymm12
  33477a:      	vextracti128	$0x1, %ymm12, %xmm13
  334780:      	vpackssdw	%xmm13, %xmm12, %xmm12
  334785:      	vpackssdw	%xmm12, %xmm12, %xmm12
  33478a:      	vpacksswb	%xmm12, %xmm12, %xmm12
  33478f:      	vpand	%xmm2, %xmm12, %xmm12
  334793:      	vinserti128	$0x1, %xmm3, %ymm11, %ymm3
  334799:      	vpbroadcastd	%xmm4, %ymm4
  33479e:      	vpblendd	$0x20, %ymm4, %ymm3, %ymm3 # ymm3 = ymm3[0,1,2,3,4],ymm4[5],ymm3[6,7]
  3347a4:      	vpbroadcastq	%xmm5, %ymm4
  3347a9:      	vpblendd	$0xc0, %ymm4, %ymm3, %ymm3 # ymm3 = ymm3[0,1,2,3,4,5],ymm4[6,7]
  3347af:      	vpbroadcastd	%xmm12, %ymm4
  3347b4:      	vpblendd	$0x80, %ymm4, %ymm3, %ymm5 # ymm5 = ymm3[0,1,2,3,4,5,6],ymm4[7]
  3347ba:      	vmovdqu	-0xe0(%r12,%r13), %ymm3
  3347c4:      	vmovdqu	-0xc0(%r12,%r13), %ymm11
  3347ce:      	vmovdqu	-0xa0(%r12,%r13), %ymm12
  3347d8:      	vmovdqu	-0x80(%r12,%r13), %ymm13
  3347df:      	vpcmpeqq	%ymm0, %ymm3, %ymm4
  3347e4:      	vextracti128	$0x1, %ymm4, %xmm15
  3347ea:      	vpackssdw	%xmm15, %xmm4, %xmm4
  3347ef:      	vpor	%xmm4, %xmm14, %xmm4
  3347f3:      	vpor	%xmm4, %xmm10, %xmm14
  3347f7:      	vpcmpgtq	%ymm3, %ymm6, %ymm3
  3347fc:      	vextracti128	$0x1, %ymm3, %xmm4
  334802:      	vpackssdw	%xmm4, %xmm3, %xmm3
  334806:      	vpackssdw	%xmm3, %xmm3, %xmm3
  33480a:      	vpcmpgtq	%ymm11, %ymm6, %ymm4
  33480f:      	vextracti128	$0x1, %ymm4, %xmm10
  334815:      	vpackssdw	%xmm10, %xmm4, %xmm4
  33481a:      	vpackssdw	%xmm4, %xmm4, %xmm4
  33481e:      	vpacksswb	%xmm4, %xmm3, %xmm3
  334822:      	vpcmpgtq	%ymm12, %ymm6, %ymm4
  334827:      	vextracti128	$0x1, %ymm4, %xmm10
  33482d:      	vpackssdw	%xmm10, %xmm4, %xmm4
  334832:      	vpackssdw	%xmm4, %xmm4, %xmm4
  334836:      	vpacksswb	%xmm4, %xmm4, %xmm4
  33483a:      	vpand	%xmm2, %xmm4, %xmm4
  33483e:      	vpcmpgtq	%ymm13, %ymm6, %ymm10
  334843:      	vextracti128	$0x1, %ymm10, %xmm15
  334849:      	vpackssdw	%xmm15, %xmm10, %xmm10
  33484e:      	vpackssdw	%xmm10, %xmm10, %xmm10
  334853:      	vpacksswb	%xmm10, %xmm10, %xmm10
  334858:      	vpshufd	$0xd8, %xmm3, %xmm3     # xmm3 = xmm3[0,2,1,3]
  33485d:      	vpand	%xmm1, %xmm3, %xmm3
  334861:      	vpunpcklqdq	%xmm4, %xmm3, %xmm3 # xmm3 = xmm3[0],xmm4[0]
  334865:      	vpbroadcastd	%xmm10, %xmm4
  33486a:      	vpand	%xmm1, %xmm4, %xmm4
  33486e:      	vpblendd	$0x8, %xmm4, %xmm3, %xmm10 # xmm10 = xmm3[0,1,2],xmm4[3]
  334874:      	vmovdqu	-0x60(%r12,%r13), %ymm15
  33487b:      	vpcmpeqq	%ymm0, %ymm15, %ymm3
  334880:      	vextracti128	$0x1, %ymm3, %xmm4
  334886:      	vpackssdw	%xmm4, %xmm3, %xmm3
  33488a:      	vmovd	%r9d, %xmm4
  33488f:      	vpbroadcastb	%xmm4, %xmm4
  334894:      	vpcmpeqq	%ymm0, %ymm11, %ymm11
  334899:      	vpcmpeqq	%ymm0, %ymm12, %ymm12
  33489e:      	vpcmpeqq	%ymm0, %ymm13, %ymm13
  3348a3:      	vpor	%xmm4, %xmm3, %xmm3
  3348a7:      	vmovdqu	-0x40(%r12,%r13), %ymm4
  3348ae:      	vpor	%xmm3, %xmm14, %xmm3
  3348b2:      	vpcmpeqq	%ymm0, %ymm4, %ymm14
  3348b7:      	vpor	%ymm14, %ymm11, %ymm14
  3348bc:      	vmovdqu	-0x20(%r12,%r13), %ymm11
  3348c3:      	vpor	%ymm14, %ymm9, %ymm14
  3348c8:      	vpcmpeqq	%ymm0, %ymm11, %ymm9
  3348cd:      	vpor	%ymm9, %ymm12, %ymm12
  3348d2:      	vmovdqu	(%r12,%r13), %ymm9
  3348d8:      	vpor	%ymm12, %ymm8, %ymm8
  3348dd:      	vpcmpeqq	%ymm0, %ymm9, %ymm12
  3348e2:      	vpor	%ymm12, %ymm13, %ymm12
  3348e7:      	vpor	%ymm7, %ymm12, %ymm7
  3348eb:      	vextracti128	$0x1, %ymm14, %xmm12
  3348f1:      	vpackssdw	%xmm12, %xmm14, %xmm12
  3348f6:      	vextracti128	$0x1, %ymm8, %xmm13
  3348fc:      	vpackssdw	%xmm13, %xmm8, %xmm8
  334901:      	vpor	%xmm12, %xmm8, %xmm8
  334906:      	vpor	%xmm3, %xmm8, %xmm3
  33490a:      	vextracti128	$0x1, %ymm7, %xmm8
  334910:      	vpackssdw	%xmm8, %xmm7, %xmm7
  334915:      	vpor	%xmm3, %xmm7, %xmm7
  334919:      	vpcmpgtq	%ymm15, %ymm6, %ymm3
  33491e:      	vextracti128	$0x1, %ymm3, %xmm8
  334924:      	vpackssdw	%xmm8, %xmm3, %xmm3
  334929:      	vpcmpgtq	%ymm4, %ymm6, %ymm4
  33492e:      	vextracti128	$0x1, %ymm4, %xmm8
  334934:      	vpackssdw	%xmm8, %xmm4, %xmm4
  334939:      	vpcmpgtq	%ymm11, %ymm6, %ymm8
  33493e:      	vextracti128	$0x1, %ymm8, %xmm11
  334944:      	vpackssdw	%xmm11, %xmm8, %xmm8
  334949:      	vpcmpgtq	%ymm9, %ymm6, %ymm6
  33494e:      	vextracti128	$0x1, %ymm6, %xmm9
  334954:      	vpackssdw	%xmm9, %xmm6, %xmm6
  334959:      	vpackssdw	%xmm3, %xmm3, %xmm3
  33495d:      	vpacksswb	%xmm3, %xmm3, %xmm3
  334961:      	vpand	%xmm2, %xmm3, %xmm3
  334965:      	vinserti128	$0x1, %xmm3, %ymm10, %ymm3
  33496b:      	vpackssdw	%xmm4, %xmm4, %xmm4
  33496f:      	vpacksswb	%xmm4, %xmm4, %xmm4
  334973:      	vpand	%xmm2, %xmm4, %xmm4
  334977:      	vpbroadcastd	%xmm4, %ymm4
  33497c:      	vpblendd	$0x20, %ymm4, %ymm3, %ymm3 # ymm3 = ymm3[0,1,2,3,4],ymm4[5],ymm3[6,7]
  334982:      	vpackssdw	%xmm8, %xmm8, %xmm4
  334987:      	vpacksswb	%xmm4, %xmm4, %xmm4
  33498b:      	vpand	%xmm2, %xmm4, %xmm4
  33498f:      	vpbroadcastq	%xmm4, %ymm4
  334994:      	vpblendd	$0xc0, %ymm4, %ymm3, %ymm3 # ymm3 = ymm3[0,1,2,3,4,5],ymm4[6,7]
  33499a:      	vpackssdw	%xmm6, %xmm6, %xmm4
  33499e:      	vpacksswb	%xmm4, %xmm4, %xmm4
  3349a2:      	vpand	%xmm2, %xmm4, %xmm4
  3349a6:      	vpbroadcastd	%xmm4, %ymm4
  3349ab:      	vpblendd	$0x80, %ymm4, %ymm3, %ymm6 # ymm6 = ymm3[0,1,2,3,4,5,6],ymm4[7]
  3349b1:      	vptest	-0x2afa4a(%rip), %xmm7  # 0x84f70 <anon.6c1fb0dbcc3470c819cb5f40076e76af.71.llvm.10840910162848374496+0x90>
  3349ba:      	setne	%r9b
  3349be:      	vpxor	%xmm4, %xmm4, %xmm4
  3349c2:      	vpcmpeqb	%ymm4, %ymm5, %ymm3
  3349c6:      	vpmovmskb	%ymm3, %eax
  3349ca:      	vpcmpeqb	%ymm4, %ymm6, %ymm3
  3349ce:      	vpmovmskb	%ymm3, %edx
  3349d2:      	shlq	$0x20, %rdx
  3349d6:      	orq	%rax, %rdx
  3349d9:      	notq	%rdx
  3349dc:      	movq	%rdx, (%rdi,%r8)
  3349e0:      	addq	$0x8, %r8
  3349e4:      	addq	$0x200, %r13            # imm = 0x200
  3349eb:      	cmpq	%r8, %rcx
  3349ee:      	je	0x3353e1 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack23collect_bool_words_avx2NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1g_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB35_EENtNtB3j_11row_visitor10RowVisitor19visit_deferred_boolB2Y_bKB35_NCINvXB3V_B3S_NtNtB1g_6row_fn5RowFn8dispatchB3e_E0NCB5N_s_0E0NCB3a_s_0B6z_E0EB3V_+0xf91>
  3349f4:      	movzbl	%r9b, %eax
  3349f8:      	testb	%r15b, %r15b
  3349fb:      	jne	0x3345f0 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack23collect_bool_words_avx2NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1g_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB35_EENtNtB3j_11row_visitor10RowVisitor19visit_deferred_boolB2Y_bKB35_NCINvXB3V_B3S_NtNtB1g_6row_fn5RowFn8dispatchB3e_E0NCB5N_s_0E0NCB3a_s_0B6z_E0EB3V_+0x1a0>
  334a01:      	vmovd	%eax, %xmm5
  334a05:      	vmovdqu	-0x1e0(%r12,%r13), %ymm10
  334a0f:      	vmovdqu	-0x1c0(%r12,%r13), %ymm11
  334a19:      	vmovdqu	-0x1a0(%r12,%r13), %ymm8
  334a23:      	vmovdqu	-0x180(%r12,%r13), %ymm7
  334a2d:      	vpcmpeqq	%ymm0, %ymm10, %ymm6
  334a32:      	vpcmpeqq	%ymm0, %ymm11, %ymm12
  334a37:      	vmovdqu	-0x1e0(%rbx,%r13), %ymm13
  334a41:      	vmovdqu	-0x1c0(%rbx,%r13), %ymm14
  334a4b:      	vmovdqu	-0x1a0(%rbx,%r13), %ymm15
  334a55:      	vmovdqu	-0x180(%rbx,%r13), %ymm9
  334a5f:      	vpcmpeqq	%ymm0, %ymm13, %ymm3
  334a64:      	vpor	%ymm3, %ymm6, %ymm3
  334a68:      	vpcmpeqq	%ymm0, %ymm14, %ymm6
  334a6d:      	vpor	%ymm6, %ymm12, %ymm12
  334a71:      	vextracti128	$0x1, %ymm3, %xmm6
  334a77:      	vpackssdw	%xmm6, %xmm3, %xmm3
  334a7b:      	vpor	%xmm3, %xmm5, %xmm6
  334a7f:      	vpcmpgtq	%ymm10, %ymm13, %ymm3
  334a84:      	vextracti128	$0x1, %ymm3, %xmm5
  334a8a:      	vpackssdw	%xmm5, %xmm3, %xmm3
  334a8e:      	vpackssdw	%xmm3, %xmm3, %xmm3
  334a92:      	vpcmpgtq	%ymm11, %ymm14, %ymm5
  334a97:      	vextracti128	$0x1, %ymm5, %xmm10
  334a9d:      	vpackssdw	%xmm10, %xmm5, %xmm5
  334aa2:      	vpackssdw	%xmm5, %xmm5, %xmm5
  334aa6:      	vpacksswb	%xmm5, %xmm3, %xmm3
  334aaa:      	vpcmpeqq	%ymm0, %ymm8, %ymm5
  334aaf:      	vpcmpeqq	%ymm0, %ymm7, %ymm10
  334ab4:      	vpcmpeqq	%ymm0, %ymm15, %ymm11
  334ab9:      	vpor	%ymm5, %ymm11, %ymm11
  334abd:      	vpcmpeqq	%ymm0, %ymm9, %ymm5
  334ac2:      	vpor	%ymm5, %ymm10, %ymm5
  334ac6:      	vpor	%ymm5, %ymm12, %ymm5
  334aca:      	vextracti128	$0x1, %ymm11, %xmm10
  334ad0:      	vpackssdw	%xmm10, %xmm11, %xmm10
  334ad5:      	vpcmpgtq	%ymm8, %ymm15, %ymm8
  334ada:      	vextracti128	$0x1, %ymm8, %xmm11
  334ae0:      	vpackssdw	%xmm11, %xmm8, %xmm8
  334ae5:      	vpackssdw	%xmm8, %xmm8, %xmm8
  334aea:      	vpacksswb	%xmm8, %xmm8, %xmm8
  334aef:      	vpand	%xmm2, %xmm8, %xmm8
  334af3:      	vpcmpgtq	%ymm7, %ymm9, %ymm7
  334af8:      	vextracti128	$0x1, %ymm7, %xmm9
  334afe:      	vpackssdw	%xmm9, %xmm7, %xmm7
  334b03:      	vpackssdw	%xmm7, %xmm7, %xmm7
  334b07:      	vpacksswb	%xmm7, %xmm7, %xmm7
  334b0b:      	vpshufd	$0xd8, %xmm3, %xmm3     # xmm3 = xmm3[0,2,1,3]
  334b10:      	vpand	%xmm1, %xmm3, %xmm3
  334b14:      	vpunpcklqdq	%xmm8, %xmm3, %xmm3 # xmm3 = xmm3[0],xmm8[0]
  334b19:      	vpbroadcastd	%xmm7, %xmm7
  334b1e:      	vpand	%xmm1, %xmm7, %xmm7
  334b22:      	vpblendd	$0x8, %xmm7, %xmm3, %xmm3 # xmm3 = xmm3[0,1,2],xmm7[3]
  334b28:      	vmovdqu	-0x160(%r12,%r13), %ymm9
  334b32:      	vmovdqu	-0x140(%r12,%r13), %ymm11
  334b3c:      	vmovdqu	-0x160(%rbx,%r13), %ymm12
  334b46:      	vmovdqu	-0x140(%rbx,%r13), %ymm13
  334b50:      	vpcmpeqq	%ymm0, %ymm9, %ymm7
  334b55:      	vpcmpeqq	%ymm0, %ymm11, %ymm8
  334b5a:      	vpcmpeqq	%ymm0, %ymm12, %ymm14
  334b5f:      	vpor	%ymm7, %ymm14, %ymm7
  334b63:      	vpcmpeqq	%ymm0, %ymm13, %ymm14
  334b68:      	vpor	%ymm14, %ymm8, %ymm8
  334b6d:      	vextracti128	$0x1, %ymm7, %xmm14
  334b73:      	vpackssdw	%xmm14, %xmm7, %xmm7
  334b78:      	vpor	%xmm7, %xmm10, %xmm7
  334b7c:      	vpor	%xmm7, %xmm6, %xmm7
  334b80:      	vpcmpgtq	%ymm9, %ymm12, %ymm6
  334b85:      	vextracti128	$0x1, %ymm6, %xmm9
  334b8b:      	vpackssdw	%xmm9, %xmm6, %xmm6
  334b90:      	vpackssdw	%xmm6, %xmm6, %xmm6
  334b94:      	vpacksswb	%xmm6, %xmm6, %xmm6
  334b98:      	vpand	%xmm2, %xmm6, %xmm6
  334b9c:      	vpcmpgtq	%ymm11, %ymm13, %ymm9
  334ba1:      	vextracti128	$0x1, %ymm9, %xmm10
  334ba7:      	vpackssdw	%xmm10, %xmm9, %xmm9
  334bac:      	vpackssdw	%xmm9, %xmm9, %xmm9
  334bb1:      	vpacksswb	%xmm9, %xmm9, %xmm9
  334bb6:      	vpand	%xmm2, %xmm9, %xmm9
  334bba:      	vinserti128	$0x1, %xmm6, %ymm3, %ymm3
  334bc0:      	vpbroadcastd	%xmm9, %ymm6
  334bc5:      	vpblendd	$0x20, %ymm6, %ymm3, %ymm3 # ymm3 = ymm3[0,1,2,3,4],ymm6[5],ymm3[6,7]
  334bcb:      	vmovdqu	-0x120(%r12,%r13), %ymm9
  334bd5:      	vmovdqu	-0x100(%r12,%r13), %ymm10
  334bdf:      	vmovdqu	-0x120(%rbx,%r13), %ymm11
  334be9:      	vmovdqu	-0x100(%rbx,%r13), %ymm12
  334bf3:      	vpcmpeqq	%ymm0, %ymm9, %ymm6
  334bf8:      	vpcmpeqq	%ymm0, %ymm10, %ymm13
  334bfd:      	vpcmpeqq	%ymm0, %ymm11, %ymm14
  334c02:      	vpor	%ymm6, %ymm14, %ymm14
  334c06:      	vpcmpeqq	%ymm0, %ymm12, %ymm6
  334c0b:      	vpor	%ymm6, %ymm13, %ymm6
  334c0f:      	vpor	%ymm6, %ymm8, %ymm6
  334c13:      	vpor	%ymm6, %ymm5, %ymm6
  334c17:      	vextracti128	$0x1, %ymm14, %xmm5
  334c1d:      	vpackssdw	%xmm5, %xmm14, %xmm8
  334c21:      	vpcmpgtq	%ymm9, %ymm11, %ymm5
  334c26:      	vextracti128	$0x1, %ymm5, %xmm9
  334c2c:      	vpackssdw	%xmm9, %xmm5, %xmm5
  334c31:      	vpackssdw	%xmm5, %xmm5, %xmm5
  334c35:      	vpacksswb	%xmm5, %xmm5, %xmm5
  334c39:      	vpand	%xmm2, %xmm5, %xmm5
  334c3d:      	vpcmpgtq	%ymm10, %ymm12, %ymm9
  334c42:      	vextracti128	$0x1, %ymm9, %xmm10
  334c48:      	vpackssdw	%xmm10, %xmm9, %xmm9
  334c4d:      	vpackssdw	%xmm9, %xmm9, %xmm9
  334c52:      	vpacksswb	%xmm9, %xmm9, %xmm9
  334c57:      	vpand	%xmm2, %xmm9, %xmm9
  334c5b:      	vpbroadcastq	%xmm5, %ymm5
  334c60:      	vpblendd	$0xc0, %ymm5, %ymm3, %ymm3 # ymm3 = ymm3[0,1,2,3,4,5],ymm5[6,7]
  334c66:      	vpbroadcastd	%xmm9, %ymm5
  334c6b:      	vpblendd	$0x80, %ymm5, %ymm3, %ymm5 # ymm5 = ymm3[0,1,2,3,4,5,6],ymm5[7]
  334c71:      	vmovdqu	-0xe0(%r12,%r13), %ymm3
  334c7b:      	vmovdqu	-0xc0(%r12,%r13), %ymm9
  334c85:      	vmovdqu	-0xe0(%rbx,%r13), %ymm10
  334c8f:      	vmovdqu	-0xc0(%rbx,%r13), %ymm11
  334c99:      	vpcmpeqq	%ymm0, %ymm3, %ymm12
  334c9e:      	vpcmpeqq	%ymm0, %ymm9, %ymm13
  334ca3:      	vpcmpeqq	%ymm0, %ymm10, %ymm14
  334ca8:      	vpor	%ymm14, %ymm12, %ymm12
  334cad:      	vpcmpeqq	%ymm0, %ymm11, %ymm14
  334cb2:      	vpor	%ymm14, %ymm13, %ymm13
  334cb7:      	vextracti128	$0x1, %ymm12, %xmm14
  334cbd:      	vpackssdw	%xmm14, %xmm12, %xmm12
  334cc2:      	vpor	%xmm12, %xmm8, %xmm12
  334cc7:      	vpcmpgtq	%ymm3, %ymm10, %ymm3
  334ccc:      	vextracti128	$0x1, %ymm3, %xmm8
  334cd2:      	vpackssdw	%xmm8, %xmm3, %xmm3
  334cd7:      	vpackssdw	%xmm3, %xmm3, %xmm3
  334cdb:      	vpcmpgtq	%ymm9, %ymm11, %ymm8
  334ce0:      	vextracti128	$0x1, %ymm8, %xmm9
  334ce6:      	vpackssdw	%xmm9, %xmm8, %xmm8
  334ceb:      	vpackssdw	%xmm8, %xmm8, %xmm8
  334cf0:      	vpacksswb	%xmm8, %xmm3, %xmm3
  334cf5:      	vmovdqu	-0xa0(%r12,%r13), %ymm9
  334cff:      	vmovdqu	-0x80(%r12,%r13), %ymm10
  334d06:      	vmovdqu	-0xa0(%rbx,%r13), %ymm11
  334d10:      	vmovdqu	-0x80(%rbx,%r13), %ymm14
  334d17:      	vpcmpeqq	%ymm0, %ymm9, %ymm8
  334d1c:      	vpcmpeqq	%ymm0, %ymm10, %ymm15
  334d21:      	vpcmpeqq	%ymm0, %ymm11, %ymm4
  334d26:      	vpor	%ymm4, %ymm8, %ymm4
  334d2a:      	vpcmpeqq	%ymm0, %ymm14, %ymm8
  334d2f:      	vpor	%ymm8, %ymm15, %ymm8
  334d34:      	vpor	%ymm8, %ymm13, %ymm8
  334d39:      	vextracti128	$0x1, %ymm4, %xmm13
  334d3f:      	vpackssdw	%xmm13, %xmm4, %xmm4
  334d44:      	vpor	%xmm4, %xmm12, %xmm4
  334d48:      	vpor	%xmm4, %xmm7, %xmm7
  334d4c:      	vpcmpgtq	%ymm9, %ymm11, %ymm4
  334d51:      	vextracti128	$0x1, %ymm4, %xmm9
  334d57:      	vpackssdw	%xmm9, %xmm4, %xmm4
  334d5c:      	vpackssdw	%xmm4, %xmm4, %xmm4
  334d60:      	vpacksswb	%xmm4, %xmm4, %xmm4
  334d64:      	vpand	%xmm2, %xmm4, %xmm4
  334d68:      	vpcmpgtq	%ymm10, %ymm14, %ymm9
  334d6d:      	vextracti128	$0x1, %ymm9, %xmm10
  334d73:      	vpackssdw	%xmm10, %xmm9, %xmm9
  334d78:      	vpackssdw	%xmm9, %xmm9, %xmm9
  334d7d:      	vpacksswb	%xmm9, %xmm9, %xmm9
  334d82:      	vpshufd	$0xd8, %xmm3, %xmm3     # xmm3 = xmm3[0,2,1,3]
  334d87:      	vpand	%xmm1, %xmm3, %xmm3
  334d8b:      	vpunpcklqdq	%xmm4, %xmm3, %xmm3 # xmm3 = xmm3[0],xmm4[0]
  334d8f:      	vpbroadcastd	%xmm9, %xmm4
  334d94:      	vpand	%xmm1, %xmm4, %xmm4
  334d98:      	vpblendd	$0x8, %xmm4, %xmm3, %xmm3 # xmm3 = xmm3[0,1,2],xmm4[3]
  334d9e:      	vmovdqu	-0x60(%r12,%r13), %ymm4
  334da5:      	vmovdqu	-0x40(%r12,%r13), %ymm9
  334dac:      	vmovdqu	-0x60(%rbx,%r13), %ymm10
  334db3:      	vmovdqu	-0x40(%rbx,%r13), %ymm11
  334dba:      	vpcmpeqq	%ymm0, %ymm4, %ymm12
  334dbf:      	vpcmpeqq	%ymm0, %ymm9, %ymm13
  334dc4:      	vpcmpeqq	%ymm0, %ymm10, %ymm14
  334dc9:      	vpor	%ymm14, %ymm12, %ymm12
  334dce:      	vpcmpeqq	%ymm0, %ymm11, %ymm14
  334dd3:      	vpor	%ymm14, %ymm13, %ymm13
  334dd8:      	vpor	%ymm13, %ymm8, %ymm8
  334ddd:      	vpor	%ymm6, %ymm8, %ymm13
  334de1:      	vextracti128	$0x1, %ymm12, %xmm6
  334de7:      	vpackssdw	%xmm6, %xmm12, %xmm8
  334deb:      	vpcmpgtq	%ymm4, %ymm10, %ymm4
  334df0:      	vextracti128	$0x1, %ymm4, %xmm6
  334df6:      	vpackssdw	%xmm6, %xmm4, %xmm4
  334dfa:      	vpackssdw	%xmm4, %xmm4, %xmm4
  334dfe:      	vpacksswb	%xmm4, %xmm4, %xmm4
  334e02:      	vpand	%xmm2, %xmm4, %xmm4
  334e06:      	vpcmpgtq	%ymm9, %ymm11, %ymm6
  334e0b:      	vextracti128	$0x1, %ymm6, %xmm9
  334e11:      	vpackssdw	%xmm9, %xmm6, %xmm6
  334e16:      	vpackssdw	%xmm6, %xmm6, %xmm6
  334e1a:      	vpacksswb	%xmm6, %xmm6, %xmm6
  334e1e:      	vpand	%xmm2, %xmm6, %xmm6
  334e22:      	vinserti128	$0x1, %xmm4, %ymm3, %ymm3
  334e28:      	vpbroadcastd	%xmm6, %ymm4
  334e2d:      	vpblendd	$0x20, %ymm4, %ymm3, %ymm6 # ymm6 = ymm3[0,1,2,3,4],ymm4[5],ymm3[6,7]
  334e33:      	vmovdqu	-0x20(%r12,%r13), %ymm3
  334e3a:      	vmovdqu	(%r12,%r13), %ymm4
  334e40:      	vmovdqu	-0x20(%rbx,%r13), %ymm9
  334e47:      	vmovdqu	(%rbx,%r13), %ymm10
  334e4d:      	vpcmpeqq	%ymm0, %ymm3, %ymm11
  334e52:      	vpcmpeqq	%ymm0, %ymm4, %ymm12
  334e57:      	vpcmpeqq	%ymm0, %ymm9, %ymm14
  334e5c:      	vpor	%ymm14, %ymm11, %ymm11
  334e61:      	vpcmpeqq	%ymm0, %ymm10, %ymm14
  334e66:      	vpor	%ymm14, %ymm12, %ymm12
  334e6b:      	vpor	%ymm12, %ymm13, %ymm12
  334e70:      	vextracti128	$0x1, %ymm11, %xmm13
  334e76:      	vpackssdw	%xmm13, %xmm11, %xmm11
  334e7b:      	vpor	%xmm11, %xmm8, %xmm8
  334e80:      	vpor	%xmm7, %xmm8, %xmm7
  334e84:      	vextracti128	$0x1, %ymm12, %xmm8
  334e8a:      	vshufps	$0x88, %xmm8, %xmm12, %xmm8 # xmm8 = xmm12[0,2],xmm8[0,2]
  334e90:      	vorps	%xmm7, %xmm8, %xmm7
  334e94:      	vpcmpgtq	%ymm3, %ymm9, %ymm3
  334e99:      	vextracti128	$0x1, %ymm3, %xmm8
  334e9f:      	vpackssdw	%xmm8, %xmm3, %xmm3
  334ea4:      	vpackssdw	%xmm3, %xmm3, %xmm3
  334ea8:      	vpacksswb	%xmm3, %xmm3, %xmm3
  334eac:      	vpand	%xmm2, %xmm3, %xmm3
  334eb0:      	vpcmpgtq	%ymm4, %ymm10, %ymm4
  334eb5:      	vextracti128	$0x1, %ymm4, %xmm8
  334ebb:      	vpackssdw	%xmm8, %xmm4, %xmm4
  334ec0:      	vpackssdw	%xmm4, %xmm4, %xmm4
  334ec4:      	vpacksswb	%xmm4, %xmm4, %xmm4
  334ec8:      	vpand	%xmm2, %xmm4, %xmm4
  334ecc:      	vpbroadcastq	%xmm3, %ymm3
  334ed1:      	vpblendd	$0xc0, %ymm3, %ymm6, %ymm3 # ymm3 = ymm6[0,1,2,3,4,5],ymm3[6,7]
  334ed7:      	jmp	0x3349a6 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack23collect_bool_words_avx2NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1g_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB35_EENtNtB3j_11row_visitor10RowVisitor19visit_deferred_boolB2Y_bKB35_NCINvXB3V_B3S_NtNtB1g_6row_fn5RowFn8dispatchB3e_E0NCB5N_s_0E0NCB3a_s_0B6z_E0EB3V_+0x556>
  334edc:      	cmpq	%r10, %rsi
  334edf:      	sete	%r9b
  334ee3:      	vmovd	%r9d, %xmm0
  334ee8:      	vpbroadcastb	%xmm0, %xmm0
  334eed:      	vmovdqa	%xmm0, -0x40(%rbp)
  334ef2:      	vmovq	%rsi, %xmm0
  334ef7:      	vpbroadcastq	%xmm0, %ymm1
  334efc:      	addq	$0x1e0, %rbx            # imm = 0x1E0
  334f03:      	xorl	%r8d, %r8d
  334f06:      	vpbroadcastq	-0x298ff7(%rip), %ymm2 # 0x9bf18 <anon.ad39dab31d93760b58da730830a5ab31.633.llvm.6451718222114420760+0x50>
  334f0f:      	vmovdqa	-0x2b0207(%rip), %xmm3  # 0x84d10 <anon.6c1fb0dbcc3470c819cb5f40076e76af.65.llvm.10840910162848374496+0x640>
  334f17:      	vpbroadcastd	-0x296d40(%rip), %xmm4 # 0x9e1e0 <anon.c60d04bc1ccb4c59159856f942bde85e.7.llvm.13399503513443445546+0x340>
  334f20:      	movl	%eax, %r9d
  334f23:      	nopw	%cs:(%rax,%rax)
  334f30:      	movzbl	%r9b, %eax
  334f34:      	vmovd	%eax, %xmm0
  334f38:      	vmovdqu	-0x1e0(%rbx), %ymm9
  334f40:      	vmovdqu	-0x1c0(%rbx), %ymm11
  334f48:      	vmovdqu	-0x1a0(%rbx), %ymm12
  334f50:      	vmovdqu	-0x180(%rbx), %ymm13
  334f58:      	vpcmpeqq	%ymm2, %ymm9, %ymm6
  334f5d:      	vextracti128	$0x1, %ymm6, %xmm7
  334f63:      	vpackssdw	%xmm7, %xmm6, %xmm6
  334f67:      	vpor	%xmm6, %xmm0, %xmm10
  334f6b:      	vpcmpeqq	%ymm2, %ymm11, %ymm8
  334f70:      	vpcmpeqq	%ymm2, %ymm12, %ymm7
  334f75:      	vpcmpeqq	%ymm2, %ymm13, %ymm6
  334f7a:      	vpcmpgtq	%ymm1, %ymm9, %ymm0
  334f7f:      	vextracti128	$0x1, %ymm0, %xmm9
  334f85:      	vpackssdw	%xmm9, %xmm0, %xmm0
  334f8a:      	vpackssdw	%xmm0, %xmm0, %xmm0
  334f8e:      	vpcmpgtq	%ymm1, %ymm11, %ymm9
  334f93:      	vextracti128	$0x1, %ymm9, %xmm11
  334f99:      	vpackssdw	%xmm11, %xmm9, %xmm9
  334f9e:      	vpackssdw	%xmm9, %xmm9, %xmm9
  334fa3:      	vpacksswb	%xmm9, %xmm0, %xmm0
  334fa8:      	vpcmpgtq	%ymm1, %ymm12, %ymm9
  334fad:      	vextracti128	$0x1, %ymm9, %xmm11
  334fb3:      	vpackssdw	%xmm11, %xmm9, %xmm9
  334fb8:      	vpackssdw	%xmm9, %xmm9, %xmm9
  334fbd:      	vpacksswb	%xmm9, %xmm9, %xmm9
  334fc2:      	vpand	%xmm3, %xmm9, %xmm9
  334fc6:      	vpcmpgtq	%ymm1, %ymm13, %ymm11
  334fcb:      	vextracti128	$0x1, %ymm11, %xmm12
  334fd1:      	vpackssdw	%xmm12, %xmm11, %xmm11
  334fd6:      	vpackssdw	%xmm11, %xmm11, %xmm11
  334fdb:      	vpacksswb	%xmm11, %xmm11, %xmm11
  334fe0:      	vpshufd	$0xd8, %xmm0, %xmm0     # xmm0 = xmm0[0,2,1,3]
  334fe5:      	vpand	%xmm4, %xmm0, %xmm0
  334fe9:      	vpunpcklqdq	%xmm9, %xmm0, %xmm0 # xmm0 = xmm0[0],xmm9[0]
  334fee:      	vpbroadcastd	%xmm11, %xmm9
  334ff3:      	vpand	%xmm4, %xmm9, %xmm9
  334ff7:      	vpblendd	$0x8, %xmm9, %xmm0, %xmm12 # xmm12 = xmm0[0,1,2],xmm9[3]
  334ffd:      	vmovdqu	-0x160(%rbx), %ymm0
  335005:      	vmovdqu	-0x140(%rbx), %ymm13
  33500d:      	vmovdqu	-0x120(%rbx), %ymm14
  335015:      	vmovdqu	-0x100(%rbx), %ymm15
  33501d:      	vpcmpeqq	%ymm2, %ymm0, %ymm9
  335022:      	vextracti128	$0x1, %ymm9, %xmm11
  335028:      	vpackssdw	%xmm11, %xmm9, %xmm11
  33502d:      	vpcmpeqq	%ymm2, %ymm13, %ymm9
  335032:      	vpor	%ymm9, %ymm8, %ymm9
  335037:      	vpcmpeqq	%ymm2, %ymm14, %ymm8
  33503c:      	vpor	%ymm7, %ymm8, %ymm8
  335040:      	vpcmpeqq	%ymm2, %ymm15, %ymm7
  335045:      	vpor	%ymm7, %ymm6, %ymm7
  335049:      	vpcmpgtq	%ymm1, %ymm0, %ymm0
  33504e:      	vextracti128	$0x1, %ymm0, %xmm6
  335054:      	vpackssdw	%xmm6, %xmm0, %xmm0
  335058:      	vpackssdw	%xmm0, %xmm0, %xmm0
  33505c:      	vpacksswb	%xmm0, %xmm0, %xmm0
  335060:      	vpand	%xmm3, %xmm0, %xmm0
  335064:      	vpcmpgtq	%ymm1, %ymm13, %ymm6
  335069:      	vextracti128	$0x1, %ymm6, %xmm13
  33506f:      	vpackssdw	%xmm13, %xmm6, %xmm6
  335074:      	vpackssdw	%xmm6, %xmm6, %xmm6
  335078:      	vpacksswb	%xmm6, %xmm6, %xmm6
  33507c:      	vpand	%xmm3, %xmm6, %xmm6
  335080:      	vpcmpgtq	%ymm1, %ymm14, %ymm13
  335085:      	vextracti128	$0x1, %ymm13, %xmm14
  33508b:      	vpackssdw	%xmm14, %xmm13, %xmm13
  335090:      	vpackssdw	%xmm13, %xmm13, %xmm13
  335095:      	vpacksswb	%xmm13, %xmm13, %xmm13
  33509a:      	vpand	%xmm3, %xmm13, %xmm13
  33509e:      	vpcmpgtq	%ymm1, %ymm15, %ymm14
  3350a3:      	vextracti128	$0x1, %ymm14, %xmm15
  3350a9:      	vpackssdw	%xmm15, %xmm14, %xmm14
  3350ae:      	vpackssdw	%xmm14, %xmm14, %xmm14
  3350b3:      	vpacksswb	%xmm14, %xmm14, %xmm14
  3350b8:      	vpand	%xmm3, %xmm14, %xmm14
  3350bc:      	vinserti128	$0x1, %xmm0, %ymm12, %ymm0
  3350c2:      	vpbroadcastd	%xmm6, %ymm6
  3350c7:      	vpblendd	$0x20, %ymm6, %ymm0, %ymm0 # ymm0 = ymm0[0,1,2,3,4],ymm6[5],ymm0[6,7]
  3350cd:      	vpbroadcastq	%xmm13, %ymm6
  3350d2:      	vpblendd	$0xc0, %ymm6, %ymm0, %ymm0 # ymm0 = ymm0[0,1,2,3,4,5],ymm6[6,7]
  3350d8:      	vpbroadcastd	%xmm14, %ymm6
  3350dd:      	vpblendd	$0x80, %ymm6, %ymm0, %ymm6 # ymm6 = ymm0[0,1,2,3,4,5,6],ymm6[7]
  3350e3:      	vmovdqu	-0xe0(%rbx), %ymm0
  3350eb:      	vmovdqu	-0xc0(%rbx), %ymm14
  3350f3:      	vmovdqu	-0xa0(%rbx), %ymm15
  3350fb:      	vmovdqu	-0x80(%rbx), %ymm5
  335100:      	vpcmpeqq	%ymm2, %ymm0, %ymm12
  335105:      	vextracti128	$0x1, %ymm12, %xmm13
  33510b:      	vpackssdw	%xmm13, %xmm12, %xmm12
  335110:      	vpor	%xmm12, %xmm11, %xmm11
  335115:      	vpor	%xmm11, %xmm10, %xmm13
  33511a:      	vpcmpeqq	%ymm2, %ymm14, %ymm12
  33511f:      	vpcmpeqq	%ymm2, %ymm15, %ymm11
  335124:      	vpcmpgtq	%ymm1, %ymm0, %ymm0
  335129:      	vextracti128	$0x1, %ymm0, %xmm10
  33512f:      	vpackssdw	%xmm10, %xmm0, %xmm0
  335134:      	vpcmpgtq	%ymm1, %ymm14, %ymm10
  335139:      	vextracti128	$0x1, %ymm10, %xmm14
  33513f:      	vpackssdw	%xmm14, %xmm10, %xmm10
  335144:      	vpcmpeqq	%ymm2, %ymm5, %ymm14
  335149:      	vpackssdw	%xmm0, %xmm0, %xmm0
  33514d:      	vpackssdw	%xmm10, %xmm10, %xmm10
  335152:      	vpacksswb	%xmm10, %xmm0, %xmm0
  335157:      	vpcmpgtq	%ymm1, %ymm15, %ymm10
  33515c:      	vextracti128	$0x1, %ymm10, %xmm15
  335162:      	vpackssdw	%xmm15, %xmm10, %xmm10
  335167:      	vpackssdw	%xmm10, %xmm10, %xmm10
  33516c:      	vpacksswb	%xmm10, %xmm10, %xmm10
  335171:      	vpand	%xmm3, %xmm10, %xmm10
  335175:      	vpcmpgtq	%ymm1, %ymm5, %ymm5
  33517a:      	vextracti128	$0x1, %ymm5, %xmm15
  335180:      	vpackssdw	%xmm15, %xmm5, %xmm5
  335185:      	vpackssdw	%xmm5, %xmm5, %xmm5
  335189:      	vpacksswb	%xmm5, %xmm5, %xmm5
  33518d:      	vpshufd	$0xd8, %xmm0, %xmm0     # xmm0 = xmm0[0,2,1,3]
  335192:      	vpand	%xmm4, %xmm0, %xmm0
  335196:      	vpunpcklqdq	%xmm10, %xmm0, %xmm0 # xmm0 = xmm0[0],xmm10[0]
  33519b:      	vpbroadcastd	%xmm5, %xmm5
  3351a0:      	vpand	%xmm4, %xmm5, %xmm5
  3351a4:      	vpblendd	$0x8, %xmm5, %xmm0, %xmm10 # xmm10 = xmm0[0,1,2],xmm5[3]
  3351aa:      	vmovdqu	-0x60(%rbx), %ymm15
  3351af:      	vpcmpeqq	%ymm2, %ymm15, %ymm0
  3351b4:      	vextracti128	$0x1, %ymm0, %xmm5
  3351ba:      	vpackssdw	%xmm5, %xmm0, %xmm5
  3351be:      	vmovdqu	-0x40(%rbx), %ymm0
  3351c3:      	vpor	-0x40(%rbp), %xmm5, %xmm5
  3351c8:      	vpor	%xmm5, %xmm13, %xmm5
  3351cc:      	vpcmpeqq	%ymm2, %ymm0, %ymm13
  3351d1:      	vpor	%ymm13, %ymm12, %ymm13
  3351d6:      	vmovdqu	-0x20(%rbx), %ymm12
  3351db:      	vpor	%ymm13, %ymm9, %ymm13
  3351e0:      	vpcmpeqq	%ymm2, %ymm12, %ymm9
  3351e5:      	vpor	%ymm9, %ymm11, %ymm11
  3351ea:      	vmovdqu	(%rbx), %ymm9
  3351ee:      	vpor	%ymm11, %ymm8, %ymm8
  3351f3:      	vpcmpeqq	%ymm2, %ymm9, %ymm11
  3351f8:      	vpor	%ymm11, %ymm14, %ymm11
  3351fd:      	vpor	%ymm7, %ymm11, %ymm7
  335201:      	vextracti128	$0x1, %ymm13, %xmm11
  335207:      	vpackssdw	%xmm11, %xmm13, %xmm11
  33520c:      	vextracti128	$0x1, %ymm8, %xmm13
  335212:      	vpackssdw	%xmm13, %xmm8, %xmm8
  335217:      	vpor	%xmm11, %xmm8, %xmm8
  33521c:      	vpor	%xmm5, %xmm8, %xmm5
  335220:      	vextracti128	$0x1, %ymm7, %xmm8
  335226:      	vpackssdw	%xmm8, %xmm7, %xmm7
  33522b:      	vpor	%xmm5, %xmm7, %xmm7
  33522f:      	vpcmpgtq	%ymm1, %ymm15, %ymm5
  335234:      	vextracti128	$0x1, %ymm5, %xmm8
  33523a:      	vpackssdw	%xmm8, %xmm5, %xmm5
  33523f:      	vpcmpgtq	%ymm1, %ymm0, %ymm0
  335244:      	vextracti128	$0x1, %ymm0, %xmm8
  33524a:      	vpackssdw	%xmm8, %xmm0, %xmm0
  33524f:      	vpcmpgtq	%ymm1, %ymm12, %ymm8
  335254:      	vextracti128	$0x1, %ymm8, %xmm11
  33525a:      	vpackssdw	%xmm11, %xmm8, %xmm8
  33525f:      	vpcmpgtq	%ymm1, %ymm9, %ymm9
  335264:      	vextracti128	$0x1, %ymm9, %xmm11
  33526a:      	vpackssdw	%xmm11, %xmm9, %xmm9
  33526f:      	vpackssdw	%xmm5, %xmm5, %xmm5
  335273:      	vpacksswb	%xmm5, %xmm5, %xmm5
  335277:      	vpand	%xmm3, %xmm5, %xmm5
  33527b:      	vinserti128	$0x1, %xmm5, %ymm10, %ymm5
  335281:      	vpackssdw	%xmm0, %xmm0, %xmm0
  335285:      	vpacksswb	%xmm0, %xmm0, %xmm0
  335289:      	vpand	%xmm3, %xmm0, %xmm0
  33528d:      	vpbroadcastd	%xmm0, %ymm0
  335292:      	vpblendd	$0x20, %ymm0, %ymm5, %ymm0 # ymm0 = ymm5[0,1,2,3,4],ymm0[5],ymm5[6,7]
  335298:      	vpackssdw	%xmm8, %xmm8, %xmm5
  33529d:      	vpacksswb	%xmm5, %xmm5, %xmm5
  3352a1:      	vpand	%xmm3, %xmm5, %xmm5
  3352a5:      	vpbroadcastq	%xmm5, %ymm5
  3352aa:      	vpblendd	$0xc0, %ymm5, %ymm0, %ymm0 # ymm0 = ymm0[0,1,2,3,4,5],ymm5[6,7]
  3352b0:      	vpackssdw	%xmm9, %xmm9, %xmm5
  3352b5:      	vpacksswb	%xmm5, %xmm5, %xmm5
  3352b9:      	vpand	%xmm3, %xmm5, %xmm5
  3352bd:      	vpbroadcastd	%xmm5, %ymm5
  3352c2:      	vpblendd	$0x80, %ymm5, %ymm0, %ymm0 # ymm0 = ymm0[0,1,2,3,4,5,6],ymm5[7]
  3352c8:      	vpslld	$0x1f, %xmm7, %xmm5
  3352cd:      	vtestps	%xmm5, %xmm5
  3352d2:      	vpxor	%xmm7, %xmm7, %xmm7
  3352d6:      	vpcmpeqb	%ymm7, %ymm6, %ymm5
  3352da:      	vpmovmskb	%ymm5, %eax
  3352de:      	vpcmpeqb	%ymm7, %ymm0, %ymm0
  3352e2:      	vpmovmskb	%ymm0, %edx
  3352e6:      	setne	%r9b
  3352ea:      	shlq	$0x20, %rdx
  3352ee:      	orq	%rax, %rdx
  3352f1:      	notq	%rdx
  3352f4:      	movq	%rdx, (%rdi,%r8)
  3352f8:      	addq	$0x8, %r8
  3352fc:      	addq	$0x200, %rbx            # imm = 0x200
  335303:      	cmpq	%r8, %rcx
  335306:      	jne	0x334f30 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack23collect_bool_words_avx2NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1g_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB35_EENtNtB3j_11row_visitor10RowVisitor19visit_deferred_boolB2Y_bKB35_NCINvXB3V_B3S_NtNtB1g_6row_fn5RowFn8dispatchB3e_E0NCB5N_s_0E0NCB3a_s_0B6z_E0EB3V_+0xae0>
  33530c:      	jmp	0x3353e1 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack23collect_bool_words_avx2NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1g_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB35_EENtNtB3j_11row_visitor10RowVisitor19visit_deferred_boolB2Y_bKB35_NCINvXB3V_B3S_NtNtB1g_6row_fn5RowFn8dispatchB3e_E0NCB5N_s_0E0NCB3a_s_0B6z_E0EB3V_+0xf91>
  335311:      	movq	%rcx, %r9
  335314:      	shrq	$0x3, %r9
  335318:      	incq	%r9
  33531b:      	movabsq	$0x3ffffffffffffff0, %r15 # imm = 0x3FFFFFFFFFFFFFF0
  335325:      	vmovq	%r14, %xmm0
  33532a:      	cmpq	$0x78, %rcx
  33532e:      	movq	-0x30(%rbp), %rdx
  335332:      	jae	0x335338 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack23collect_bool_words_avx2NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1g_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB35_EENtNtB3j_11row_visitor10RowVisitor19visit_deferred_boolB2Y_bKB35_NCINvXB3V_B3S_NtNtB1g_6row_fn5RowFn8dispatchB3e_E0NCB5N_s_0E0NCB3a_s_0B6z_E0EB3V_+0xee8>
  335334:      	xorl	%ecx, %ecx
  335336:      	jmp	0x335383 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack23collect_bool_words_avx2NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1g_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB35_EENtNtB3j_11row_visitor10RowVisitor19visit_deferred_boolB2Y_bKB35_NCINvXB3V_B3S_NtNtB1g_6row_fn5RowFn8dispatchB3e_E0NCB5N_s_0E0NCB3a_s_0B6z_E0EB3V_+0xf33>
  335338:      	movq	%r9, %rcx
  33533b:      	andq	%r15, %rcx
  33533e:      	vpbroadcastq	%xmm0, %ymm1
  335343:      	xorl	%r12d, %r12d
  335346:      	nopw	%cs:(%rax,%rax)
  335350:      	vmovdqu	%ymm1, (%rdi,%r12,8)
  335356:      	vmovdqu	%ymm1, 0x20(%rdi,%r12,8)
  33535d:      	vmovdqu	%ymm1, 0x40(%rdi,%r12,8)
  335364:      	vmovdqu	%ymm1, 0x60(%rdi,%r12,8)
  33536b:      	addq	$0x10, %r12
  33536f:      	cmpq	%r12, %rcx
  335372:      	jne	0x335350 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack23collect_bool_words_avx2NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1g_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB35_EENtNtB3j_11row_visitor10RowVisitor19visit_deferred_boolB2Y_bKB35_NCINvXB3V_B3S_NtNtB1g_6row_fn5RowFn8dispatchB3e_E0NCB5N_s_0E0NCB3a_s_0B6z_E0EB3V_+0xf00>
  335374:      	cmpq	%rcx, %r9
  335377:      	je	0x3353cd <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack23collect_bool_words_avx2NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1g_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB35_EENtNtB3j_11row_visitor10RowVisitor19visit_deferred_boolB2Y_bKB35_NCINvXB3V_B3S_NtNtB1g_6row_fn5RowFn8dispatchB3e_E0NCB5N_s_0E0NCB3a_s_0B6z_E0EB3V_+0xf7d>
  335379:      	testb	$0xc, %r9b
  33537d:      	je	0x335a5a <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack23collect_bool_words_avx2NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1g_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB35_EENtNtB3j_11row_visitor10RowVisitor19visit_deferred_boolB2Y_bKB35_NCINvXB3V_B3S_NtNtB1g_6row_fn5RowFn8dispatchB3e_E0NCB5N_s_0E0NCB3a_s_0B6z_E0EB3V_+0x160a>
  335383:      	addq	$0xc, %r15
  335387:      	andq	%r9, %r15
  33538a:      	leaq	(%rdi,%r15,8), %r12
  33538e:      	vpbroadcastq	%xmm0, %ymm0
  335393:      	nopw	%cs:(%rax,%rax)
  3353a0:      	vmovdqu	%ymm0, (%rdi,%rcx,8)
  3353a5:      	addq	$0x4, %rcx
  3353a9:      	cmpq	%rcx, %r15
  3353ac:      	jne	0x3353a0 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack23collect_bool_words_avx2NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1g_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB35_EENtNtB3j_11row_visitor10RowVisitor19visit_deferred_boolB2Y_bKB35_NCINvXB3V_B3S_NtNtB1g_6row_fn5RowFn8dispatchB3e_E0NCB5N_s_0E0NCB3a_s_0B6z_E0EB3V_+0xf50>
  3353ae:      	cmpq	%r15, %r9
  3353b1:      	je	0x3353cd <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack23collect_bool_words_avx2NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1g_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB35_EENtNtB3j_11row_visitor10RowVisitor19visit_deferred_boolB2Y_bKB35_NCINvXB3V_B3S_NtNtB1g_6row_fn5RowFn8dispatchB3e_E0NCB5N_s_0E0NCB3a_s_0B6z_E0EB3V_+0xf7d>
  3353b3:      	nopw	%cs:(%rax,%rax)
  3353c0:      	movq	%r14, (%r12)
  3353c4:      	addq	$0x8, %r12
  3353c8:      	cmpq	%rdx, %r12
  3353cb:      	jne	0x3353c0 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack23collect_bool_words_avx2NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1g_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB35_EENtNtB3j_11row_visitor10RowVisitor19visit_deferred_boolB2Y_bKB35_NCINvXB3V_B3S_NtNtB1g_6row_fn5RowFn8dispatchB3e_E0NCB5N_s_0E0NCB3a_s_0B6z_E0EB3V_+0xf70>
  3353cd:      	cmpq	%r10, %rsi
  3353d0:      	sete	%cl
  3353d3:      	cmpq	%r10, -0x40(%rbp)
  3353d7:      	sete	%r9b
  3353db:      	orb	%cl, %r9b
  3353de:      	orb	%al, %r9b
  3353e1:      	andb	$0x1, %r9b
  3353e5:      	movq	-0x50(%rbp), %r13
  3353e9:      	movb	%r9b, 0x38(%r13)
  3353ed:      	movq	-0x60(%rbp), %r15
  3353f1:      	movq	-0x58(%rbp), %rsi
  3353f5:      	movq	-0x48(%rbp), %rdx
  3353f9:      	testq	%r11, %r11
  3353fc:      	je	0x335a48 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack23collect_bool_words_avx2NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1g_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB35_EENtNtB3j_11row_visitor10RowVisitor19visit_deferred_boolB2Y_bKB35_NCINvXB3V_B3S_NtNtB1g_6row_fn5RowFn8dispatchB3e_E0NCB5N_s_0E0NCB3a_s_0B6z_E0EB3V_+0x15f8>
  335402:      	movzbl	0x18(%r13), %eax
  335407:      	movzbl	0x38(%r13), %ebx
  33540c:      	cmpl	$0x1, (%r13)
  335411:      	jne	0x33543e <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack23collect_bool_words_avx2NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1g_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB35_EENtNtB3j_11row_visitor10RowVisitor19visit_deferred_boolB2Y_bKB35_NCINvXB3V_B3S_NtNtB1g_6row_fn5RowFn8dispatchB3e_E0NCB5N_s_0E0NCB3a_s_0B6z_E0EB3V_+0xfee>
  335413:      	movq	0x10(%r13), %rcx
  335417:      	testb	%al, %al
  335419:      	je	0x33545f <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack23collect_bool_words_avx2NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1g_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB35_EENtNtB3j_11row_visitor10RowVisitor19visit_deferred_boolB2Y_bKB35_NCINvXB3V_B3S_NtNtB1g_6row_fn5RowFn8dispatchB3e_E0NCB5N_s_0E0NCB3a_s_0B6z_E0EB3V_+0x100f>
  33541b:      	movq	0x28(%r13), %rax
  33541f:      	movq	(%rcx), %r9
  335422:      	movq	(%rax), %r8
  335425:      	xorl	%r14d, %r14d
  335428:      	cmpq	%r8, %r9
  33542b:      	setl	%r14b
  33542f:      	cmpl	$0x4, %r11d
  335433:      	jae	0x335496 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack23collect_bool_words_avx2NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1g_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB35_EENtNtB3j_11row_visitor10RowVisitor19visit_deferred_boolB2Y_bKB35_NCINvXB3V_B3S_NtNtB1g_6row_fn5RowFn8dispatchB3e_E0NCB5N_s_0E0NCB3a_s_0B6z_E0EB3V_+0x1046>
  335435:      	xorl	%edi, %edi
  335437:      	xorl	%ecx, %ecx
  335439:      	jmp	0x335a10 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack23collect_bool_words_avx2NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1g_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB35_EENtNtB3j_11row_visitor10RowVisitor19visit_deferred_boolB2Y_bKB35_NCINvXB3V_B3S_NtNtB1g_6row_fn5RowFn8dispatchB3e_E0NCB5N_s_0E0NCB3a_s_0B6z_E0EB3V_+0x15c0>
  33543e:      	movq	0x8(%r13), %r8
  335442:      	testb	%al, %al
  335444:      	je	0x33547c <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack23collect_bool_words_avx2NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1g_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB35_EENtNtB3j_11row_visitor10RowVisitor19visit_deferred_boolB2Y_bKB35_NCINvXB3V_B3S_NtNtB1g_6row_fn5RowFn8dispatchB3e_E0NCB5N_s_0E0NCB3a_s_0B6z_E0EB3V_+0x102c>
  335446:      	movq	0x28(%r13), %rax
  33544a:      	movq	(%rax), %r14
  33544d:      	cmpl	$0x8, %r11d
  335451:      	jae	0x3354ae <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack23collect_bool_words_avx2NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1g_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB35_EENtNtB3j_11row_visitor10RowVisitor19visit_deferred_boolB2Y_bKB35_NCINvXB3V_B3S_NtNtB1g_6row_fn5RowFn8dispatchB3e_E0NCB5N_s_0E0NCB3a_s_0B6z_E0EB3V_+0x105e>
  335453:      	xorl	%edi, %edi
  335455:      	xorl	%ecx, %ecx
  335457:      	movl	%ebx, %r9d
  33545a:      	jmp	0x3355cc <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack23collect_bool_words_avx2NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1g_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB35_EENtNtB3j_11row_visitor10RowVisitor19visit_deferred_boolB2Y_bKB35_NCINvXB3V_B3S_NtNtB1g_6row_fn5RowFn8dispatchB3e_E0NCB5N_s_0E0NCB3a_s_0B6z_E0EB3V_+0x117c>
  33545f:      	movq	0x20(%r13), %r14
  335463:      	movq	(%rcx), %r8
  335466:      	cmpl	$0x8, %r11d
  33546a:      	jae	0x33561a <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack23collect_bool_words_avx2NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1g_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB35_EENtNtB3j_11row_visitor10RowVisitor19visit_deferred_boolB2Y_bKB35_NCINvXB3V_B3S_NtNtB1g_6row_fn5RowFn8dispatchB3e_E0NCB5N_s_0E0NCB3a_s_0B6z_E0EB3V_+0x11ca>
  335470:      	xorl	%edi, %edi
  335472:      	xorl	%ecx, %ecx
  335474:      	movl	%ebx, %r9d
  335477:      	jmp	0x33572c <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack23collect_bool_words_avx2NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1g_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB35_EENtNtB3j_11row_visitor10RowVisitor19visit_deferred_boolB2Y_bKB35_NCINvXB3V_B3S_NtNtB1g_6row_fn5RowFn8dispatchB3e_E0NCB5N_s_0E0NCB3a_s_0B6z_E0EB3V_+0x12dc>
  33547c:      	movq	0x20(%r13), %r14
  335480:      	cmpl	$0x8, %r11d
  335484:      	jae	0x33577b <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack23collect_bool_words_avx2NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1g_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB35_EENtNtB3j_11row_visitor10RowVisitor19visit_deferred_boolB2Y_bKB35_NCINvXB3V_B3S_NtNtB1g_6row_fn5RowFn8dispatchB3e_E0NCB5N_s_0E0NCB3a_s_0B6z_E0EB3V_+0x132b>
  33548a:      	xorl	%edi, %edi
  33548c:      	xorl	%ecx, %ecx
  33548e:      	movl	%ebx, %r9d
  335491:      	jmp	0x33589a <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack23collect_bool_words_avx2NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1g_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB35_EENtNtB3j_11row_visitor10RowVisitor19visit_deferred_boolB2Y_bKB35_NCINvXB3V_B3S_NtNtB1g_6row_fn5RowFn8dispatchB3e_E0NCB5N_s_0E0NCB3a_s_0B6z_E0EB3V_+0x144a>
  335496:      	vmovq	%r14, %xmm0
  33549b:      	cmpl	$0x10, %r11d
  33549f:      	jae	0x3358e8 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack23collect_bool_words_avx2NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1g_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB35_EENtNtB3j_11row_visitor10RowVisitor19visit_deferred_boolB2Y_bKB35_NCINvXB3V_B3S_NtNtB1g_6row_fn5RowFn8dispatchB3e_E0NCB5N_s_0E0NCB3a_s_0B6z_E0EB3V_+0x1498>
  3354a5:      	xorl	%ecx, %ecx
  3354a7:      	xorl	%edi, %edi
  3354a9:      	jmp	0x3359ab <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack23collect_bool_words_avx2NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1g_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB35_EENtNtB3j_11row_visitor10RowVisitor19visit_deferred_boolB2Y_bKB35_NCINvXB3V_B3S_NtNtB1g_6row_fn5RowFn8dispatchB3e_E0NCB5N_s_0E0NCB3a_s_0B6z_E0EB3V_+0x155b>
  3354ae:      	cmpq	%r10, %r14
  3354b1:      	sete	%al
  3354b4:      	movl	%edx, %ecx
  3354b6:      	andl	$0x38, %ecx
  3354b9:      	movzbl	%bl, %edi
  3354bc:      	vmovd	%edi, %xmm0
  3354c0:      	vmovq	%r14, %xmm1
  3354c5:      	vpbroadcastq	%xmm1, %ymm1
  3354ca:      	vmovd	%eax, %xmm2
  3354ce:      	vpbroadcastb	%xmm2, %xmm2
  3354d3:      	movq	%rdx, %rax
  3354d6:      	andq	$-0x40, %rax
  3354da:      	leaq	(%r8,%rax,8), %rax
  3354de:      	addq	$0x20, %rax
  3354e2:      	vpxor	%xmm5, %xmm5, %xmm5
  3354e6:      	vmovdqa	-0x297d8e(%rip), %ymm4  # 0x9d760 <anon.1b4110e42284c5f75ac48cf7fa14c04d.13.llvm.7054906275604583804+0x80>
  3354ee:      	vpxor	%xmm3, %xmm3, %xmm3
  3354f2:      	vpbroadcastq	-0x298feb(%rip), %ymm6 # 0x9c510 <anon.5278ce255300ec7b361bbf944ee0b803.26.llvm.754055615369681222>
  3354fb:      	vpbroadcastq	-0x2995ec(%rip), %ymm7 # 0x9bf18 <anon.ad39dab31d93760b58da730830a5ab31.633.llvm.6451718222114420760+0x50>
  335504:      	xorl	%edi, %edi
  335506:      	vpbroadcastq	-0x299b4f(%rip), %ymm8 # 0x9b9c0 <anon.5278ce255300ec7b361bbf944ee0b803.25.llvm.754055615369681222>
  33550f:      	vpxor	%xmm9, %xmm9, %xmm9
  335514:      	nopw	%cs:(%rax,%rax)
  335520:      	vpaddq	%ymm6, %ymm4, %ymm11
  335524:      	vmovdqu	-0x20(%rax,%rdi,8), %ymm12
  33552a:      	vmovdqu	(%rax,%rdi,8), %ymm13
  33552f:      	vpcmpeqq	%ymm7, %ymm12, %ymm10
  335534:      	vextracti128	$0x1, %ymm10, %xmm14
  33553a:      	vpackssdw	%xmm14, %xmm10, %xmm14
  33553f:      	vpcmpeqq	%ymm7, %ymm13, %ymm10
  335544:      	vextracti128	$0x1, %ymm10, %xmm15
  33554a:      	vpackssdw	%xmm15, %xmm10, %xmm10
  33554f:      	vpor	%xmm5, %xmm10, %xmm10
  335553:      	vpor	%xmm2, %xmm0, %xmm0
  335557:      	vpor	%xmm0, %xmm14, %xmm0
  33555b:      	vpor	%xmm2, %xmm10, %xmm5
  33555f:      	vpcmpgtq	%ymm12, %ymm1, %ymm12
  335564:      	vpsrlq	$0x3f, %ymm12, %ymm12
  33556a:      	vpcmpgtq	%ymm13, %ymm1, %ymm13
  33556f:      	vpsrlq	$0x3f, %ymm13, %ymm13
  335575:      	vpsllvq	%ymm11, %ymm13, %ymm11
  33557a:      	vpor	%ymm9, %ymm11, %ymm9
  33557f:      	vpsllvq	%ymm4, %ymm12, %ymm11
  335584:      	vpor	%ymm3, %ymm11, %ymm3
  335588:      	addq	$0x8, %rdi
  33558c:      	vpaddq	%ymm4, %ymm8, %ymm4
  335590:      	cmpq	%rdi, %rcx
  335593:      	jne	0x335520 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack23collect_bool_words_avx2NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1g_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB35_EENtNtB3j_11row_visitor10RowVisitor19visit_deferred_boolB2Y_bKB35_NCINvXB3V_B3S_NtNtB1g_6row_fn5RowFn8dispatchB3e_E0NCB5N_s_0E0NCB3a_s_0B6z_E0EB3V_+0x10d0>
  335595:      	vpor	%ymm3, %ymm9, %ymm1
  335599:      	vextracti128	$0x1, %ymm1, %xmm2
  33559f:      	vpor	%xmm2, %xmm1, %xmm1
  3355a3:      	vpshufd	$0xee, %xmm1, %xmm2     # xmm2 = xmm1[2,3,2,3]
  3355a8:      	vpor	%xmm2, %xmm1, %xmm1
  3355ac:      	vmovq	%xmm1, %rdi
  3355b1:      	vpor	%xmm0, %xmm10, %xmm0
  3355b5:      	vpslld	$0x1f, %xmm0, %xmm0
  3355ba:      	vtestps	%xmm0, %xmm0
  3355bf:      	setne	%r9b
  3355c3:      	cmpl	%ecx, %r11d
  3355c6:      	je	0x335a34 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack23collect_bool_words_avx2NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1g_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB35_EENtNtB3j_11row_visitor10RowVisitor19visit_deferred_boolB2Y_bKB35_NCINvXB3V_B3S_NtNtB1g_6row_fn5RowFn8dispatchB3e_E0NCB5N_s_0E0NCB3a_s_0B6z_E0EB3V_+0x15e4>
  3355cc:      	andq	$-0x40, %rdx
  3355d0:      	leaq	(%r8,%rdx,8), %rax
  3355d4:      	movl	%r9d, %edx
  3355d7:      	nopw	(%rax,%rax)
  3355e0:      	cmpq	%r10, %r14
  3355e3:      	sete	%r9b
  3355e7:      	movq	(%rax,%rcx,8), %r12
  3355eb:      	cmpq	%r10, %r12
  3355ee:      	sete	%r8b
  3355f2:      	xorl	%ebx, %ebx
  3355f4:      	cmpq	%r14, %r12
  3355f7:      	setl	%bl
  3355fa:      	orb	%dl, %r9b
  3355fd:      	orb	%r8b, %r9b
  335600:      	leaq	0x1(%rcx), %r8
  335604:      	shlq	%cl, %rbx
  335607:      	orq	%rbx, %rdi
  33560a:      	movl	%r9d, %edx
  33560d:      	cmpq	%r8, %r11
  335610:      	movq	%r8, %rcx
  335613:      	jne	0x3355e0 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack23collect_bool_words_avx2NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1g_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB35_EENtNtB3j_11row_visitor10RowVisitor19visit_deferred_boolB2Y_bKB35_NCINvXB3V_B3S_NtNtB1g_6row_fn5RowFn8dispatchB3e_E0NCB5N_s_0E0NCB3a_s_0B6z_E0EB3V_+0x1190>
  335615:      	jmp	0x335a34 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack23collect_bool_words_avx2NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1g_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB35_EENtNtB3j_11row_visitor10RowVisitor19visit_deferred_boolB2Y_bKB35_NCINvXB3V_B3S_NtNtB1g_6row_fn5RowFn8dispatchB3e_E0NCB5N_s_0E0NCB3a_s_0B6z_E0EB3V_+0x15e4>
  33561a:      	cmpq	%r10, %r8
  33561d:      	sete	%al
  335620:      	movl	%edx, %ecx
  335622:      	andl	$0x38, %ecx
  335625:      	movzbl	%bl, %edi
  335628:      	vmovd	%edi, %xmm0
  33562c:      	vmovq	%r8, %xmm1
  335631:      	vpbroadcastq	%xmm1, %ymm1
  335636:      	vmovd	%eax, %xmm2
  33563a:      	vpbroadcastb	%xmm2, %xmm2
  33563f:      	movq	%rdx, %rax
  335642:      	andq	$-0x40, %rax
  335646:      	leaq	(%r14,%rax,8), %rax
  33564a:      	addq	$0x20, %rax
  33564e:      	vpxor	%xmm5, %xmm5, %xmm5
  335652:      	vmovdqa	-0x297efa(%rip), %ymm4  # 0x9d760 <anon.1b4110e42284c5f75ac48cf7fa14c04d.13.llvm.7054906275604583804+0x80>
  33565a:      	vpxor	%xmm3, %xmm3, %xmm3
  33565e:      	vpbroadcastq	-0x299157(%rip), %ymm6 # 0x9c510 <anon.5278ce255300ec7b361bbf944ee0b803.26.llvm.754055615369681222>
  335667:      	vpbroadcastq	-0x299758(%rip), %ymm7 # 0x9bf18 <anon.ad39dab31d93760b58da730830a5ab31.633.llvm.6451718222114420760+0x50>
  335670:      	xorl	%edi, %edi
  335672:      	vpbroadcastq	-0x299cbb(%rip), %ymm8 # 0x9b9c0 <anon.5278ce255300ec7b361bbf944ee0b803.25.llvm.754055615369681222>
  33567b:      	vpxor	%xmm9, %xmm9, %xmm9
  335680:      	vpaddq	%ymm6, %ymm4, %ymm11
  335684:      	vmovdqu	-0x20(%rax,%rdi,8), %ymm12
  33568a:      	vmovdqu	(%rax,%rdi,8), %ymm13
  33568f:      	vpcmpeqq	%ymm7, %ymm12, %ymm10
  335694:      	vextracti128	$0x1, %ymm10, %xmm14
  33569a:      	vpackssdw	%xmm14, %xmm10, %xmm14
  33569f:      	vpcmpeqq	%ymm7, %ymm13, %ymm10
  3356a4:      	vextracti128	$0x1, %ymm10, %xmm15
  3356aa:      	vpackssdw	%xmm15, %xmm10, %xmm10
  3356af:      	vpor	%xmm5, %xmm10, %xmm10
  3356b3:      	vpor	%xmm2, %xmm0, %xmm0
  3356b7:      	vpor	%xmm0, %xmm14, %xmm0
  3356bb:      	vpor	%xmm2, %xmm10, %xmm5
  3356bf:      	vpcmpgtq	%ymm1, %ymm12, %ymm12
  3356c4:      	vpsrlq	$0x3f, %ymm12, %ymm12
  3356ca:      	vpcmpgtq	%ymm1, %ymm13, %ymm13
  3356cf:      	vpsrlq	$0x3f, %ymm13, %ymm13
  3356d5:      	vpsllvq	%ymm11, %ymm13, %ymm11
  3356da:      	vpor	%ymm9, %ymm11, %ymm9
  3356df:      	vpsllvq	%ymm4, %ymm12, %ymm11
  3356e4:      	vpor	%ymm3, %ymm11, %ymm3
  3356e8:      	addq	$0x8, %rdi
  3356ec:      	vpaddq	%ymm4, %ymm8, %ymm4
  3356f0:      	cmpq	%rdi, %rcx
  3356f3:      	jne	0x335680 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack23collect_bool_words_avx2NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1g_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB35_EENtNtB3j_11row_visitor10RowVisitor19visit_deferred_boolB2Y_bKB35_NCINvXB3V_B3S_NtNtB1g_6row_fn5RowFn8dispatchB3e_E0NCB5N_s_0E0NCB3a_s_0B6z_E0EB3V_+0x1230>
  3356f5:      	vpor	%ymm3, %ymm9, %ymm1
  3356f9:      	vextracti128	$0x1, %ymm1, %xmm2
  3356ff:      	vpor	%xmm2, %xmm1, %xmm1
  335703:      	vpshufd	$0xee, %xmm1, %xmm2     # xmm2 = xmm1[2,3,2,3]
  335708:      	vpor	%xmm2, %xmm1, %xmm1
  33570c:      	vmovq	%xmm1, %rdi
  335711:      	vpor	%xmm0, %xmm10, %xmm0
  335715:      	vpslld	$0x1f, %xmm0, %xmm0
  33571a:      	vtestps	%xmm0, %xmm0
  33571f:      	setne	%r9b
  335723:      	cmpl	%ecx, %r11d
  335726:      	je	0x335a34 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack23collect_bool_words_avx2NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1g_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB35_EENtNtB3j_11row_visitor10RowVisitor19visit_deferred_boolB2Y_bKB35_NCINvXB3V_B3S_NtNtB1g_6row_fn5RowFn8dispatchB3e_E0NCB5N_s_0E0NCB3a_s_0B6z_E0EB3V_+0x15e4>
  33572c:      	andq	$-0x40, %rdx
  335730:      	leaq	(%r14,%rdx,8), %rax
  335734:      	movl	%r9d, %edx
  335737:      	nopw	(%rax,%rax)
  335740:      	cmpq	%r10, %r8
  335743:      	sete	%r9b
  335747:      	movq	(%rax,%rcx,8), %r12
  33574b:      	cmpq	%r10, %r12
  33574e:      	sete	%bl
  335751:      	xorl	%r14d, %r14d
  335754:      	cmpq	%r12, %r8
  335757:      	setl	%r14b
  33575b:      	orb	%dl, %r9b
  33575e:      	orb	%bl, %r9b
  335761:      	leaq	0x1(%rcx), %rbx
  335765:      	shlq	%cl, %r14
  335768:      	orq	%r14, %rdi
  33576b:      	movl	%r9d, %edx
  33576e:      	cmpq	%rbx, %r11
  335771:      	movq	%rbx, %rcx
  335774:      	jne	0x335740 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack23collect_bool_words_avx2NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1g_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB35_EENtNtB3j_11row_visitor10RowVisitor19visit_deferred_boolB2Y_bKB35_NCINvXB3V_B3S_NtNtB1g_6row_fn5RowFn8dispatchB3e_E0NCB5N_s_0E0NCB3a_s_0B6z_E0EB3V_+0x12f0>
  335776:      	jmp	0x335a34 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack23collect_bool_words_avx2NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1g_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB35_EENtNtB3j_11row_visitor10RowVisitor19visit_deferred_boolB2Y_bKB35_NCINvXB3V_B3S_NtNtB1g_6row_fn5RowFn8dispatchB3e_E0NCB5N_s_0E0NCB3a_s_0B6z_E0EB3V_+0x15e4>
  33577b:      	movl	%edx, %ecx
  33577d:      	andl	$0x38, %ecx
  335780:      	movzbl	%bl, %eax
  335783:      	vmovd	%eax, %xmm0
  335787:      	movq	%rdx, %rdi
  33578a:      	andq	$-0x40, %rdi
  33578e:      	leaq	(%r14,%rdi,8), %rax
  335792:      	addq	$0x20, %rax
  335796:      	leaq	0x20(%r8,%rdi,8), %rdi
  33579b:      	vpxor	%xmm1, %xmm1, %xmm1
  33579f:      	vmovdqa	-0x298047(%rip), %ymm3  # 0x9d760 <anon.1b4110e42284c5f75ac48cf7fa14c04d.13.llvm.7054906275604583804+0x80>
  3357a7:      	vpxor	%xmm2, %xmm2, %xmm2
  3357ab:      	vpbroadcastq	-0x2992a4(%rip), %ymm4 # 0x9c510 <anon.5278ce255300ec7b361bbf944ee0b803.26.llvm.754055615369681222>
  3357b4:      	vpbroadcastq	-0x2998a5(%rip), %ymm5 # 0x9bf18 <anon.ad39dab31d93760b58da730830a5ab31.633.llvm.6451718222114420760+0x50>
  3357bd:      	xorl	%r9d, %r9d
  3357c0:      	vpbroadcastq	-0x299e09(%rip), %ymm6 # 0x9b9c0 <anon.5278ce255300ec7b361bbf944ee0b803.25.llvm.754055615369681222>
  3357c9:      	vpxor	%xmm7, %xmm7, %xmm7
  3357cd:      	nopl	(%rax)
  3357d0:      	vpaddq	%ymm4, %ymm3, %ymm8
  3357d4:      	vmovdqu	-0x20(%rdi,%r9,8), %ymm9
  3357db:      	vmovdqu	(%rdi,%r9,8), %ymm10
  3357e1:      	vmovdqu	-0x20(%rax,%r9,8), %ymm11
  3357e8:      	vmovdqu	(%rax,%r9,8), %ymm12
  3357ee:      	vpcmpeqq	%ymm5, %ymm9, %ymm13
  3357f3:      	vpcmpeqq	%ymm5, %ymm10, %ymm14
  3357f8:      	vpcmpeqq	%ymm5, %ymm11, %ymm15
  3357fd:      	vpor	%ymm15, %ymm13, %ymm13
  335802:      	vpcmpeqq	%ymm5, %ymm12, %ymm15
  335807:      	vpor	%ymm15, %ymm14, %ymm14
  33580c:      	vextracti128	$0x1, %ymm13, %xmm15
  335812:      	vpackssdw	%xmm15, %xmm13, %xmm13
  335817:      	vpor	%xmm0, %xmm13, %xmm0
  33581b:      	vextracti128	$0x1, %ymm14, %xmm13
  335821:      	vpackssdw	%xmm13, %xmm14, %xmm13
  335826:      	vpor	%xmm1, %xmm13, %xmm1
  33582a:      	vpcmpgtq	%ymm9, %ymm11, %ymm9
  33582f:      	vpsrlq	$0x3f, %ymm9, %ymm9
  335835:      	vpcmpgtq	%ymm10, %ymm12, %ymm10
  33583a:      	vpsrlq	$0x3f, %ymm10, %ymm10
  335840:      	vpsllvq	%ymm8, %ymm10, %ymm8
  335845:      	vpor	%ymm7, %ymm8, %ymm7
  335849:      	vpsllvq	%ymm3, %ymm9, %ymm8
  33584e:      	vpor	%ymm2, %ymm8, %ymm2
  335852:      	addq	$0x8, %r9
  335856:      	vpaddq	%ymm6, %ymm3, %ymm3
  33585a:      	cmpq	%r9, %rcx
  33585d:      	jne	0x3357d0 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack23collect_bool_words_avx2NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1g_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB35_EENtNtB3j_11row_visitor10RowVisitor19visit_deferred_boolB2Y_bKB35_NCINvXB3V_B3S_NtNtB1g_6row_fn5RowFn8dispatchB3e_E0NCB5N_s_0E0NCB3a_s_0B6z_E0EB3V_+0x1380>
  335863:      	vpor	%ymm2, %ymm7, %ymm2
  335867:      	vextracti128	$0x1, %ymm2, %xmm3
  33586d:      	vpor	%xmm3, %xmm2, %xmm2
  335871:      	vpshufd	$0xee, %xmm2, %xmm3     # xmm3 = xmm2[2,3,2,3]
  335876:      	vpor	%xmm3, %xmm2, %xmm2
  33587a:      	vmovq	%xmm2, %rdi
  33587f:      	vpor	%xmm0, %xmm1, %xmm0
  335883:      	vpslld	$0x1f, %xmm0, %xmm0
  335888:      	vtestps	%xmm0, %xmm0
  33588d:      	setne	%r9b
  335891:      	cmpl	%ecx, %r11d
  335894:      	je	0x335a34 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack23collect_bool_words_avx2NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1g_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB35_EENtNtB3j_11row_visitor10RowVisitor19visit_deferred_boolB2Y_bKB35_NCINvXB3V_B3S_NtNtB1g_6row_fn5RowFn8dispatchB3e_E0NCB5N_s_0E0NCB3a_s_0B6z_E0EB3V_+0x15e4>
  33589a:      	andq	$-0x40, %rdx
  33589e:      	shlq	$0x3, %rdx
  3358a2:      	addq	%rdx, %r14
  3358a5:      	addq	%rdx, %r8
  3358a8:      	nopl	(%rax,%rax)
  3358b0:      	movq	(%r8,%rcx,8), %rax
  3358b4:      	movq	(%r14,%rcx,8), %rdx
  3358b8:      	cmpq	%r10, %rax
  3358bb:      	sete	%r12b
  3358bf:      	cmpq	%r10, %rdx
  3358c2:      	sete	%bl
  3358c5:      	orb	%r12b, %bl
  3358c8:      	xorl	%r12d, %r12d
  3358cb:      	cmpq	%rdx, %rax
  3358ce:      	setl	%r12b
  3358d2:      	orb	%bl, %r9b
  3358d5:      	shlq	%cl, %r12
  3358d8:      	incq	%rcx
  3358db:      	orq	%r12, %rdi
  3358de:      	cmpq	%rcx, %r11
  3358e1:      	jne	0x3358b0 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack23collect_bool_words_avx2NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1g_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB35_EENtNtB3j_11row_visitor10RowVisitor19visit_deferred_boolB2Y_bKB35_NCINvXB3V_B3S_NtNtB1g_6row_fn5RowFn8dispatchB3e_E0NCB5N_s_0E0NCB3a_s_0B6z_E0EB3V_+0x1460>
  3358e3:      	jmp	0x335a34 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack23collect_bool_words_avx2NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1g_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB35_EENtNtB3j_11row_visitor10RowVisitor19visit_deferred_boolB2Y_bKB35_NCINvXB3V_B3S_NtNtB1g_6row_fn5RowFn8dispatchB3e_E0NCB5N_s_0E0NCB3a_s_0B6z_E0EB3V_+0x15e4>
  3358e8:      	movl	%edx, %ecx
  3358ea:      	andl	$0x30, %ecx
  3358ed:      	vpbroadcastq	%xmm0, %ymm1
  3358f2:      	vmovdqa	-0x29819a(%rip), %ymm3  # 0x9d760 <anon.1b4110e42284c5f75ac48cf7fa14c04d.13.llvm.7054906275604583804+0x80>
  3358fa:      	vpxor	%xmm2, %xmm2, %xmm2
  3358fe:      	vpbroadcastq	-0x2993f7(%rip), %ymm4 # 0x9c510 <anon.5278ce255300ec7b361bbf944ee0b803.26.llvm.754055615369681222>
  335907:      	vpbroadcastq	-0x299f50(%rip), %ymm5 # 0x9b9c0 <anon.5278ce255300ec7b361bbf944ee0b803.25.llvm.754055615369681222>
  335910:      	vpbroadcastq	-0x299f51(%rip), %ymm6 # 0x9b9c8 <anon.5278ce255300ec7b361bbf944ee0b803.25.llvm.754055615369681222+0x8>
  335919:      	vpbroadcastq	-0x2993aa(%rip), %ymm7 # 0x9c578 <anon.ad39dab31d93760b58da730830a5ab31.728.llvm.6451718222114420760+0x40>
  335922:      	movq	%rcx, %rax
  335925:      	vpxor	%xmm8, %xmm8, %xmm8
  33592a:      	vpxor	%xmm9, %xmm9, %xmm9
  33592f:      	vpxor	%xmm10, %xmm10, %xmm10
  335934:      	nopw	%cs:(%rax,%rax)
  335940:      	vpaddq	%ymm4, %ymm3, %ymm11
  335944:      	vpaddq	%ymm5, %ymm3, %ymm12
  335948:      	vpaddq	%ymm6, %ymm3, %ymm13
  33594c:      	vpsllvq	%ymm3, %ymm1, %ymm14
  335951:      	vpor	%ymm2, %ymm14, %ymm2
  335955:      	vpsllvq	%ymm11, %ymm1, %ymm11
  33595a:      	vpor	%ymm8, %ymm11, %ymm8
  33595f:      	vpsllvq	%ymm12, %ymm1, %ymm11
  335964:      	vpor	%ymm9, %ymm11, %ymm9
  335969:      	vpsllvq	%ymm13, %ymm1, %ymm11
  33596e:      	vpor	%ymm10, %ymm11, %ymm10
  335973:      	vpaddq	%ymm7, %ymm3, %ymm3
  335977:      	addq	$-0x10, %rax
  33597b:      	jne	0x335940 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack23collect_bool_words_avx2NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1g_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB35_EENtNtB3j_11row_visitor10RowVisitor19visit_deferred_boolB2Y_bKB35_NCINvXB3V_B3S_NtNtB1g_6row_fn5RowFn8dispatchB3e_E0NCB5N_s_0E0NCB3a_s_0B6z_E0EB3V_+0x14f0>
  33597d:      	vpor	%ymm2, %ymm8, %ymm1
  335981:      	vpor	%ymm1, %ymm9, %ymm1
  335985:      	vpor	%ymm1, %ymm10, %ymm1
  335989:      	vextracti128	$0x1, %ymm1, %xmm2
  33598f:      	vpor	%xmm2, %xmm1, %xmm1
  335993:      	vpshufd	$0xee, %xmm1, %xmm2     # xmm2 = xmm1[2,3,2,3]
  335998:      	vpor	%xmm2, %xmm1, %xmm1
  33599c:      	vmovq	%xmm1, %rdi
  3359a1:      	cmpl	%ecx, %r11d
  3359a4:      	je	0x335a21 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack23collect_bool_words_avx2NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1g_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB35_EENtNtB3j_11row_visitor10RowVisitor19visit_deferred_boolB2Y_bKB35_NCINvXB3V_B3S_NtNtB1g_6row_fn5RowFn8dispatchB3e_E0NCB5N_s_0E0NCB3a_s_0B6z_E0EB3V_+0x15d1>
  3359a6:      	testb	$0xc, %dl
  3359a9:      	je	0x335a10 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack23collect_bool_words_avx2NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1g_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB35_EENtNtB3j_11row_visitor10RowVisitor19visit_deferred_boolB2Y_bKB35_NCINvXB3V_B3S_NtNtB1g_6row_fn5RowFn8dispatchB3e_E0NCB5N_s_0E0NCB3a_s_0B6z_E0EB3V_+0x15c0>
  3359ab:      	movq	%rcx, %rax
  3359ae:      	movl	%edx, %ecx
  3359b0:      	andl	$0x3c, %ecx
  3359b3:      	vmovq	%rdi, %xmm1
  3359b8:      	vmovq	%rax, %xmm2
  3359bd:      	vpbroadcastq	%xmm2, %ymm2
  3359c2:      	vpor	-0x29826a(%rip), %ymm2, %ymm2 # 0x9d760 <anon.1b4110e42284c5f75ac48cf7fa14c04d.13.llvm.7054906275604583804+0x80>
  3359ca:      	vpbroadcastq	%xmm0, %ymm0
  3359cf:      	subq	%rcx, %rax
  3359d2:      	vpbroadcastq	-0x2994cb(%rip), %ymm3 # 0x9c510 <anon.5278ce255300ec7b361bbf944ee0b803.26.llvm.754055615369681222>
  3359db:      	nopl	(%rax,%rax)
  3359e0:      	vpsllvq	%ymm2, %ymm0, %ymm4
  3359e5:      	vpor	%ymm1, %ymm4, %ymm1
  3359e9:      	vpaddq	%ymm3, %ymm2, %ymm2
  3359ed:      	addq	$0x4, %rax
  3359f1:      	jne	0x3359e0 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack23collect_bool_words_avx2NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1g_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB35_EENtNtB3j_11row_visitor10RowVisitor19visit_deferred_boolB2Y_bKB35_NCINvXB3V_B3S_NtNtB1g_6row_fn5RowFn8dispatchB3e_E0NCB5N_s_0E0NCB3a_s_0B6z_E0EB3V_+0x1590>
  3359f3:      	vextracti128	$0x1, %ymm1, %xmm0
  3359f9:      	vpor	%xmm0, %xmm1, %xmm0
  3359fd:      	vpshufd	$0xee, %xmm0, %xmm1     # xmm1 = xmm0[2,3,2,3]
  335a02:      	vpor	%xmm1, %xmm0, %xmm0
  335a06:      	vmovq	%xmm0, %rdi
  335a0b:      	cmpl	%ecx, %r11d
  335a0e:      	je	0x335a21 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack23collect_bool_words_avx2NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1g_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB35_EENtNtB3j_11row_visitor10RowVisitor19visit_deferred_boolB2Y_bKB35_NCINvXB3V_B3S_NtNtB1g_6row_fn5RowFn8dispatchB3e_E0NCB5N_s_0E0NCB3a_s_0B6z_E0EB3V_+0x15d1>
  335a10:      	movq	%r14, %rax
  335a13:      	shlq	%cl, %rax
  335a16:      	incq	%rcx
  335a19:      	orq	%rax, %rdi
  335a1c:      	cmpq	%rcx, %r11
  335a1f:      	jne	0x335a10 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack23collect_bool_words_avx2NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1g_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB35_EENtNtB3j_11row_visitor10RowVisitor19visit_deferred_boolB2Y_bKB35_NCINvXB3V_B3S_NtNtB1g_6row_fn5RowFn8dispatchB3e_E0NCB5N_s_0E0NCB3a_s_0B6z_E0EB3V_+0x15c0>
  335a21:      	cmpq	%r10, %r9
  335a24:      	sete	%al
  335a27:      	cmpq	%r10, %r8
  335a2a:      	sete	%r9b
  335a2e:      	orb	%al, %r9b
  335a31:      	orb	%bl, %r9b
  335a34:      	andb	$0x1, %r9b
  335a38:      	movb	%r9b, 0x38(%r13)
  335a3c:      	cmpq	%r15, %rsi
  335a3f:      	jae	0x335a75 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack23collect_bool_words_avx2NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1g_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB35_EENtNtB3j_11row_visitor10RowVisitor19visit_deferred_boolB2Y_bKB35_NCINvXB3V_B3S_NtNtB1g_6row_fn5RowFn8dispatchB3e_E0NCB5N_s_0E0NCB3a_s_0B6z_E0EB3V_+0x1625>
  335a41:      	movq	-0x30(%rbp), %rax
  335a45:      	movq	%rdi, (%rax)
  335a48:      	addq	$0x38, %rsp
  335a4c:      	popq	%rbx
  335a4d:      	popq	%r12
  335a4f:      	popq	%r13
  335a51:      	popq	%r14
  335a53:      	popq	%r15
  335a55:      	popq	%rbp
  335a56:      	vzeroupper
  335a59:      	retq
  335a5a:      	leaq	(%rdi,%rcx,8), %r12
  335a5e:      	jmp	0x3353c0 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack23collect_bool_words_avx2NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1g_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB35_EENtNtB3j_11row_visitor10RowVisitor19visit_deferred_boolB2Y_bKB35_NCINvXB3V_B3S_NtNtB1g_6row_fn5RowFn8dispatchB3e_E0NCB5N_s_0E0NCB3a_s_0B6z_E0EB3V_+0xf70>
  335a63:      	leaq	0xfb1416(%rip), %rcx    # 0x12e6e80 <vtable.4.llvm.9723113569546643380+0x38>
  335a6a:      	xorl	%edi, %edi
  335a6c:      	movq	%r15, %rdx
  335a6f:      	callq	*0x100d9f3(%rip)        # 0x1343468 <writev+0x1343468>
  335a75:      	leaq	0xfb13ec(%rip), %rdx    # 0x12e6e68 <vtable.4.llvm.9723113569546643380+0x20>
  335a7c:      	movq	%rsi, %rdi
  335a7f:      	movq	%r15, %rsi
  335a82:      	vzeroupper
  335a85:      	callq	*0x100d525(%rip)        # 0x1342fb0 <writev+0x1342fb0>
