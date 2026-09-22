
/tmp/row-fn-bool-linux-old-avx2-target/x86_64-unknown-linux-gnu/release/deps/row_fn_bool_retry-3e4c315823f8823b:	file format elf64-x86-64

Disassembly of section .text:

0000000000326320 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack23collect_bool_words_avx2NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1g_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB35_EENtNtB3j_11row_visitor10RowVisitor19visit_deferred_boolB2Y_bKB35_NCINvXB3V_B3S_NtNtB1g_6row_fn5RowFn8dispatchB3e_E0NCB5N_s_0E0NCB3a_s_0B6z_E0EB3V_>:
  326320:      	pushq	%rbp
  326321:      	movq	%rsp, %rbp
  326324:      	pushq	%r15
  326326:      	pushq	%r14
  326328:      	pushq	%r13
  32632a:      	pushq	%r12
  32632c:      	pushq	%rbx
  32632d:      	subq	$0x58, %rsp
  326331:      	movq	%rdx, %rax
  326334:      	shrq	$0x6, %rax
  326338:      	cmpq	%rsi, %rax
  32633b:      	ja	0x3264f9 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack23collect_bool_words_avx2NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1g_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB35_EENtNtB3j_11row_visitor10RowVisitor19visit_deferred_boolB2Y_bKB35_NCINvXB3V_B3S_NtNtB1g_6row_fn5RowFn8dispatchB3e_E0NCB5N_s_0E0NCB3a_s_0B6z_E0EB3V_+0x1d9>
  326341:      	movq	%rsi, -0x40(%rbp)
  326345:      	movl	%edx, %r9d
  326348:      	andl	$0x3f, %r9d
  32634c:      	movabsq	$0x7fffffffffffffff, %r10 # imm = 0x7FFFFFFFFFFFFFFF
  326356:      	leaq	(%rdi,%rax,8), %rsi
  32635a:      	movq	%rsi, -0x30(%rbp)
  32635e:      	movq	%rax, -0x38(%rbp)
  326362:      	testq	%rax, %rax
  326365:      	je	0x326437 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack23collect_bool_words_avx2NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1g_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB35_EENtNtB3j_11row_visitor10RowVisitor19visit_deferred_boolB2Y_bKB35_NCINvXB3V_B3S_NtNtB1g_6row_fn5RowFn8dispatchB3e_E0NCB5N_s_0E0NCB3a_s_0B6z_E0EB3V_+0x117>
  32636b:      	movq	(%rcx), %r11
  32636e:      	movq	0x18(%rcx), %rbx
  326372:      	xorl	%r14d, %r14d
  326375:      	vpxor	%xmm0, %xmm0, %xmm0
  326379:      	jmp	0x3263b4 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack23collect_bool_words_avx2NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1g_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB35_EENtNtB3j_11row_visitor10RowVisitor19visit_deferred_boolB2Y_bKB35_NCINvXB3V_B3S_NtNtB1g_6row_fn5RowFn8dispatchB3e_E0NCB5N_s_0E0NCB3a_s_0B6z_E0EB3V_+0x94>
  32637b:      	nopl	(%rax,%rax)
  326380:      	vpcmpeqb	-0x60(%rbp), %ymm0, %ymm1
  326385:      	vpmovmskb	%ymm1, %eax
  326389:      	vpcmpeqb	-0x80(%rbp), %ymm0, %ymm1
  32638e:      	shlq	$0x20, %rax
  326392:      	vpmovmskb	%ymm1, %esi
  326396:      	orq	%rax, %rsi
  326399:      	notq	%rsi
  32639c:      	movq	%rsi, (%rdi)
  32639f:      	addq	$0x8, %rdi
  3263a3:      	addq	$0x200, %r14            # imm = 0x200
  3263aa:      	cmpq	-0x30(%rbp), %rdi
  3263ae:      	je	0x326437 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack23collect_bool_words_avx2NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1g_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB35_EENtNtB3j_11row_visitor10RowVisitor19visit_deferred_boolB2Y_bKB35_NCINvXB3V_B3S_NtNtB1g_6row_fn5RowFn8dispatchB3e_E0NCB5N_s_0E0NCB3a_s_0B6z_E0EB3V_+0x117>
  3263b4:      	vmovdqu	%ymm0, -0x60(%rbp)
  3263b9:      	vmovdqu	%ymm0, -0x80(%rbp)
  3263be:      	movq	%r14, %r15
  3263c1:      	xorl	%r12d, %r12d
  3263c4:      	jmp	0x326409 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack23collect_bool_words_avx2NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1g_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB35_EENtNtB3j_11row_visitor10RowVisitor19visit_deferred_boolB2Y_bKB35_NCINvXB3V_B3S_NtNtB1g_6row_fn5RowFn8dispatchB3e_E0NCB5N_s_0E0NCB3a_s_0B6z_E0EB3V_+0xe9>
  3263c6:      	nopw	%cs:(%rax,%rax)
  3263d0:      	movq	0x28(%r11), %rax
  3263d4:      	movq	(%r13), %r13
  3263d8:      	movq	(%rax), %rax
  3263db:      	cmpq	%r10, %r13
  3263de:      	sete	%r8b
  3263e2:      	cmpq	%r10, %rax
  3263e5:      	sete	%sil
  3263e9:      	orb	%r8b, %sil
  3263ec:      	orb	%sil, (%rbx)
  3263ef:      	cmpq	%rax, %r13
  3263f2:      	setl	-0x80(%rbp,%r12)
  3263f8:      	incq	%r12
  3263fb:      	addq	$0x8, %r15
  3263ff:      	cmpq	$0x40, %r12
  326403:      	je	0x326380 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack23collect_bool_words_avx2NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1g_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB35_EENtNtB3j_11row_visitor10RowVisitor19visit_deferred_boolB2Y_bKB35_NCINvXB3V_B3S_NtNtB1g_6row_fn5RowFn8dispatchB3e_E0NCB5N_s_0E0NCB3a_s_0B6z_E0EB3V_+0x60>
  326409:      	cmpl	$0x1, (%r11)
  32640d:      	jne	0x326420 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack23collect_bool_words_avx2NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1g_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB35_EENtNtB3j_11row_visitor10RowVisitor19visit_deferred_boolB2Y_bKB35_NCINvXB3V_B3S_NtNtB1g_6row_fn5RowFn8dispatchB3e_E0NCB5N_s_0E0NCB3a_s_0B6z_E0EB3V_+0x100>
  32640f:      	movq	0x10(%r11), %r13
  326413:      	cmpl	$0x1, 0x18(%r11)
  326418:      	je	0x3263d0 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack23collect_bool_words_avx2NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1g_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB35_EENtNtB3j_11row_visitor10RowVisitor19visit_deferred_boolB2Y_bKB35_NCINvXB3V_B3S_NtNtB1g_6row_fn5RowFn8dispatchB3e_E0NCB5N_s_0E0NCB3a_s_0B6z_E0EB3V_+0xb0>
  32641a:      	jmp	0x32642e <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack23collect_bool_words_avx2NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1g_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB35_EENtNtB3j_11row_visitor10RowVisitor19visit_deferred_boolB2Y_bKB35_NCINvXB3V_B3S_NtNtB1g_6row_fn5RowFn8dispatchB3e_E0NCB5N_s_0E0NCB3a_s_0B6z_E0EB3V_+0x10e>
  32641c:      	nopl	(%rax)
  326420:      	movq	0x8(%r11), %r13
  326424:      	addq	%r15, %r13
  326427:      	cmpl	$0x1, 0x18(%r11)
  32642c:      	je	0x3263d0 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack23collect_bool_words_avx2NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1g_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB35_EENtNtB3j_11row_visitor10RowVisitor19visit_deferred_boolB2Y_bKB35_NCINvXB3V_B3S_NtNtB1g_6row_fn5RowFn8dispatchB3e_E0NCB5N_s_0E0NCB3a_s_0B6z_E0EB3V_+0xb0>
  32642e:      	movq	0x20(%r11), %rax
  326432:      	addq	%r15, %rax
  326435:      	jmp	0x3263d4 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack23collect_bool_words_avx2NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1g_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB35_EENtNtB3j_11row_visitor10RowVisitor19visit_deferred_boolB2Y_bKB35_NCINvXB3V_B3S_NtNtB1g_6row_fn5RowFn8dispatchB3e_E0NCB5N_s_0E0NCB3a_s_0B6z_E0EB3V_+0xb4>
  326437:      	testq	%r9, %r9
  32643a:      	je	0x3264e7 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack23collect_bool_words_avx2NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1g_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB35_EENtNtB3j_11row_visitor10RowVisitor19visit_deferred_boolB2Y_bKB35_NCINvXB3V_B3S_NtNtB1g_6row_fn5RowFn8dispatchB3e_E0NCB5N_s_0E0NCB3a_s_0B6z_E0EB3V_+0x1c7>
  326440:      	movq	(%rcx), %rdi
  326443:      	movq	0x18(%rcx), %r11
  326447:      	andq	$-0x40, %rdx
  32644b:      	shlq	$0x3, %rdx
  32644f:      	xorl	%ecx, %ecx
  326451:      	xorl	%ebx, %ebx
  326453:      	jmp	0x3264a1 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack23collect_bool_words_avx2NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1g_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB35_EENtNtB3j_11row_visitor10RowVisitor19visit_deferred_boolB2Y_bKB35_NCINvXB3V_B3S_NtNtB1g_6row_fn5RowFn8dispatchB3e_E0NCB5N_s_0E0NCB3a_s_0B6z_E0EB3V_+0x181>
  326455:      	nopw	%cs:(%rax,%rax)
  326460:      	movq	0x20(%rdi), %rax
  326464:      	addq	%rdx, %rax
  326467:      	movq	(%r14), %r14
  32646a:      	movq	(%rax), %rax
  32646d:      	cmpq	%r10, %r14
  326470:      	sete	%r15b
  326474:      	cmpq	%r10, %rax
  326477:      	sete	%r12b
  32647b:      	orb	%r15b, %r12b
  32647e:      	xorl	%r15d, %r15d
  326481:      	cmpq	%rax, %r14
  326484:      	setl	%r15b
  326488:      	orb	%r12b, (%r11)
  32648b:      	leaq	0x1(%rcx), %rax
  32648f:      	shlq	%cl, %r15
  326492:      	orq	%r15, %rbx
  326495:      	addq	$0x8, %rdx
  326499:      	cmpq	%rax, %r9
  32649c:      	movq	%rax, %rcx
  32649f:      	je	0x3264d3 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack23collect_bool_words_avx2NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1g_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB35_EENtNtB3j_11row_visitor10RowVisitor19visit_deferred_boolB2Y_bKB35_NCINvXB3V_B3S_NtNtB1g_6row_fn5RowFn8dispatchB3e_E0NCB5N_s_0E0NCB3a_s_0B6z_E0EB3V_+0x1b3>
  3264a1:      	cmpl	$0x1, (%rdi)
  3264a4:      	jne	0x3264c0 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack23collect_bool_words_avx2NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1g_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB35_EENtNtB3j_11row_visitor10RowVisitor19visit_deferred_boolB2Y_bKB35_NCINvXB3V_B3S_NtNtB1g_6row_fn5RowFn8dispatchB3e_E0NCB5N_s_0E0NCB3a_s_0B6z_E0EB3V_+0x1a0>
  3264a6:      	movq	0x10(%rdi), %r14
  3264aa:      	cmpl	$0x1, 0x18(%rdi)
  3264ae:      	jne	0x326460 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack23collect_bool_words_avx2NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1g_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB35_EENtNtB3j_11row_visitor10RowVisitor19visit_deferred_boolB2Y_bKB35_NCINvXB3V_B3S_NtNtB1g_6row_fn5RowFn8dispatchB3e_E0NCB5N_s_0E0NCB3a_s_0B6z_E0EB3V_+0x140>
  3264b0:      	jmp	0x3264cd <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack23collect_bool_words_avx2NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1g_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB35_EENtNtB3j_11row_visitor10RowVisitor19visit_deferred_boolB2Y_bKB35_NCINvXB3V_B3S_NtNtB1g_6row_fn5RowFn8dispatchB3e_E0NCB5N_s_0E0NCB3a_s_0B6z_E0EB3V_+0x1ad>
  3264b2:      	nopw	%cs:(%rax,%rax)
  3264c0:      	movq	0x8(%rdi), %r14
  3264c4:      	addq	%rdx, %r14
  3264c7:      	cmpl	$0x1, 0x18(%rdi)
  3264cb:      	jne	0x326460 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack23collect_bool_words_avx2NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1g_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB35_EENtNtB3j_11row_visitor10RowVisitor19visit_deferred_boolB2Y_bKB35_NCINvXB3V_B3S_NtNtB1g_6row_fn5RowFn8dispatchB3e_E0NCB5N_s_0E0NCB3a_s_0B6z_E0EB3V_+0x140>
  3264cd:      	movq	0x28(%rdi), %rax
  3264d1:      	jmp	0x326467 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack23collect_bool_words_avx2NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1g_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB35_EENtNtB3j_11row_visitor10RowVisitor19visit_deferred_boolB2Y_bKB35_NCINvXB3V_B3S_NtNtB1g_6row_fn5RowFn8dispatchB3e_E0NCB5N_s_0E0NCB3a_s_0B6z_E0EB3V_+0x147>
  3264d3:      	movq	-0x40(%rbp), %rsi
  3264d7:      	movq	-0x38(%rbp), %rdi
  3264db:      	cmpq	%rsi, %rdi
  3264de:      	jae	0x32650e <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack23collect_bool_words_avx2NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1g_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB35_EENtNtB3j_11row_visitor10RowVisitor19visit_deferred_boolB2Y_bKB35_NCINvXB3V_B3S_NtNtB1g_6row_fn5RowFn8dispatchB3e_E0NCB5N_s_0E0NCB3a_s_0B6z_E0EB3V_+0x1ee>
  3264e0:      	movq	-0x30(%rbp), %rax
  3264e4:      	movq	%rbx, (%rax)
  3264e7:      	addq	$0x58, %rsp
  3264eb:      	popq	%rbx
  3264ec:      	popq	%r12
  3264ee:      	popq	%r13
  3264f0:      	popq	%r14
  3264f2:      	popq	%r15
  3264f4:      	popq	%rbp
  3264f5:      	vzeroupper
  3264f8:      	retq
  3264f9:      	leaq	0xfd28c0(%rip), %rcx    # 0x12f8dc0 <alloc_ebbd31bf42d91f579e6aa57e93569175.llvm.523936492659896562>
  326500:      	xorl	%edi, %edi
  326502:      	movq	%rsi, %rdx
  326505:      	movq	%rax, %rsi
  326508:      	callq	*0x102fe0a(%rip)        # 0x1356318 <writev+0x1356318>
  32650e:      	leaq	0xfd2893(%rip), %rdx    # 0x12f8da8 <alloc_1b2922caae1b461da04857d3d02eae5b.llvm.523936492659896562>
  326515:      	vzeroupper
  326518:      	callq	*0x102fbca(%rip)        # 0x13560e8 <writev+0x13560e8>
