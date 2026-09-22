
/tmp/row-fn-bool-linux-old-avx2-target/x86_64-unknown-linux-gnu/release/deps/row_fn_bool_retry-3e4c315823f8823b:	file format elf64-x86-64

Disassembly of section .text:

0000000000326520 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_>:
  326520:      	pushq	%rbp
  326521:      	movq	%rsp, %rbp
  326524:      	pushq	%r15
  326526:      	pushq	%r14
  326528:      	pushq	%r13
  32652a:      	pushq	%r12
  32652c:      	pushq	%rbx
  32652d:      	subq	$0x58, %rsp
  326531:      	movq	%rdx, %rax
  326534:      	shrq	$0x6, %rax
  326538:      	cmpq	%rsi, %rax
  32653b:      	ja	0x3266e9 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x1c9>
  326541:      	movq	%rsi, -0x40(%rbp)
  326545:      	movl	%edx, %r9d
  326548:      	andl	$0x3f, %r9d
  32654c:      	movabsq	$0x7fffffffffffffff, %r10 # imm = 0x7FFFFFFFFFFFFFFF
  326556:      	leaq	(%rdi,%rax,8), %rsi
  32655a:      	movq	%rsi, -0x30(%rbp)
  32655e:      	movq	%rax, -0x38(%rbp)
  326562:      	testq	%rax, %rax
  326565:      	je	0x326627 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x107>
  32656b:      	movq	(%rcx), %r11
  32656e:      	movq	0x18(%rcx), %rbx
  326572:      	xorl	%r14d, %r14d
  326575:      	vxorps	%xmm0, %xmm0, %xmm0
  326579:      	jmp	0x3265a7 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x87>
  32657b:      	nopl	(%rax,%rax)
  326580:      	vmovdqu64	-0x80(%rbp), %zmm1
  326587:      	vptestmb	%zmm1, %zmm1, %k0
  32658d:      	kmovq	%k0, (%rdi)
  326592:      	addq	$0x8, %rdi
  326596:      	addq	$0x200, %r14            # imm = 0x200
  32659d:      	cmpq	-0x30(%rbp), %rdi
  3265a1:      	je	0x326627 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x107>
  3265a7:      	vmovups	%zmm0, -0x80(%rbp)
  3265ae:      	movq	%r14, %r15
  3265b1:      	xorl	%r12d, %r12d
  3265b4:      	jmp	0x3265f5 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0xd5>
  3265b6:      	nopw	%cs:(%rax,%rax)
  3265c0:      	movq	0x28(%r11), %rax
  3265c4:      	movq	(%r13), %r13
  3265c8:      	movq	(%rax), %rax
  3265cb:      	cmpq	%r10, %r13
  3265ce:      	sete	%r8b
  3265d2:      	cmpq	%r10, %rax
  3265d5:      	sete	%sil
  3265d9:      	orb	%r8b, %sil
  3265dc:      	orb	%sil, (%rbx)
  3265df:      	cmpq	%rax, %r13
  3265e2:      	setl	-0x80(%rbp,%r12)
  3265e8:      	incq	%r12
  3265eb:      	addq	$0x8, %r15
  3265ef:      	cmpq	$0x40, %r12
  3265f3:      	je	0x326580 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x60>
  3265f5:      	cmpl	$0x1, (%r11)
  3265f9:      	jne	0x326610 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0xf0>
  3265fb:      	movq	0x10(%r11), %r13
  3265ff:      	cmpl	$0x1, 0x18(%r11)
  326604:      	je	0x3265c0 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0xa0>
  326606:      	jmp	0x32661e <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0xfe>
  326608:      	nopl	(%rax,%rax)
  326610:      	movq	0x8(%r11), %r13
  326614:      	addq	%r15, %r13
  326617:      	cmpl	$0x1, 0x18(%r11)
  32661c:      	je	0x3265c0 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0xa0>
  32661e:      	movq	0x20(%r11), %rax
  326622:      	addq	%r15, %rax
  326625:      	jmp	0x3265c4 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0xa4>
  326627:      	testq	%r9, %r9
  32662a:      	je	0x3266d7 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x1b7>
  326630:      	movq	(%rcx), %rdi
  326633:      	movq	0x18(%rcx), %r11
  326637:      	andq	$-0x40, %rdx
  32663b:      	shlq	$0x3, %rdx
  32663f:      	xorl	%ecx, %ecx
  326641:      	xorl	%ebx, %ebx
  326643:      	jmp	0x326691 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x171>
  326645:      	nopw	%cs:(%rax,%rax)
  326650:      	movq	0x20(%rdi), %rax
  326654:      	addq	%rdx, %rax
  326657:      	movq	(%r14), %r14
  32665a:      	movq	(%rax), %rax
  32665d:      	cmpq	%r10, %r14
  326660:      	sete	%r15b
  326664:      	cmpq	%r10, %rax
  326667:      	sete	%r12b
  32666b:      	orb	%r15b, %r12b
  32666e:      	xorl	%r15d, %r15d
  326671:      	cmpq	%rax, %r14
  326674:      	setl	%r15b
  326678:      	orb	%r12b, (%r11)
  32667b:      	leaq	0x1(%rcx), %rax
  32667f:      	shlq	%cl, %r15
  326682:      	orq	%r15, %rbx
  326685:      	addq	$0x8, %rdx
  326689:      	cmpq	%rax, %r9
  32668c:      	movq	%rax, %rcx
  32668f:      	je	0x3266c3 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x1a3>
  326691:      	cmpl	$0x1, (%rdi)
  326694:      	jne	0x3266b0 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x190>
  326696:      	movq	0x10(%rdi), %r14
  32669a:      	cmpl	$0x1, 0x18(%rdi)
  32669e:      	jne	0x326650 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x130>
  3266a0:      	jmp	0x3266bd <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x19d>
  3266a2:      	nopw	%cs:(%rax,%rax)
  3266b0:      	movq	0x8(%rdi), %r14
  3266b4:      	addq	%rdx, %r14
  3266b7:      	cmpl	$0x1, 0x18(%rdi)
  3266bb:      	jne	0x326650 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x130>
  3266bd:      	movq	0x28(%rdi), %rax
  3266c1:      	jmp	0x326657 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x137>
  3266c3:      	movq	-0x40(%rbp), %rsi
  3266c7:      	movq	-0x38(%rbp), %rdi
  3266cb:      	cmpq	%rsi, %rdi
  3266ce:      	jae	0x3266fe <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x1de>
  3266d0:      	movq	-0x30(%rbp), %rax
  3266d4:      	movq	%rbx, (%rax)
  3266d7:      	addq	$0x58, %rsp
  3266db:      	popq	%rbx
  3266dc:      	popq	%r12
  3266de:      	popq	%r13
  3266e0:      	popq	%r14
  3266e2:      	popq	%r15
  3266e4:      	popq	%rbp
  3266e5:      	vzeroupper
  3266e8:      	retq
  3266e9:      	leaq	0xfd26d0(%rip), %rcx    # 0x12f8dc0 <alloc_ebbd31bf42d91f579e6aa57e93569175.llvm.523936492659896562>
  3266f0:      	xorl	%edi, %edi
  3266f2:      	movq	%rsi, %rdx
  3266f5:      	movq	%rax, %rsi
  3266f8:      	callq	*0x102fc1a(%rip)        # 0x1356318 <writev+0x1356318>
  3266fe:      	leaq	0xfd26a3(%rip), %rdx    # 0x12f8da8 <alloc_1b2922caae1b461da04857d3d02eae5b.llvm.523936492659896562>
  326705:      	vzeroupper
  326708:      	callq	*0x102f9da(%rip)        # 0x13560e8 <writev+0x13560e8>
