
/tmp/row-fn-bool-ci-artifact-after/codspeed/walltime/vortex-array/row_fn_bool_retry:	file format elf64-x86-64

Disassembly of section .text:

000000000034ce70 <_RINvNtNtCsdxhMBRXnOVY_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_>:
  34ce70:      	pushq	%rbp
  34ce71:      	movq	%rsp, %rbp
  34ce74:      	pushq	%r15
  34ce76:      	pushq	%r14
  34ce78:      	pushq	%r13
  34ce7a:      	pushq	%r12
  34ce7c:      	pushq	%rbx
  34ce7d:      	subq	$0x58, %rsp
  34ce81:      	movq	%rdx, %rax
  34ce84:      	shrq	$0x6, %rax
  34ce88:      	cmpq	%rsi, %rax
  34ce8b:      	ja	0x34d039 <_RINvNtNtCsdxhMBRXnOVY_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x1c9>
  34ce91:      	movq	%rsi, -0x40(%rbp)
  34ce95:      	movl	%edx, %r9d
  34ce98:      	andl	$0x3f, %r9d
  34ce9c:      	movabsq	$0x7fffffffffffffff, %r10 # imm = 0x7FFFFFFFFFFFFFFF
  34cea6:      	leaq	(%rdi,%rax,8), %rsi
  34ceaa:      	movq	%rsi, -0x30(%rbp)
  34ceae:      	movq	%rax, -0x38(%rbp)
  34ceb2:      	testq	%rax, %rax
  34ceb5:      	je	0x34cf77 <_RINvNtNtCsdxhMBRXnOVY_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x107>
  34cebb:      	movq	(%rcx), %r11
  34cebe:      	movq	0x18(%rcx), %rbx
  34cec2:      	xorl	%r14d, %r14d
  34cec5:      	vxorps	%xmm0, %xmm0, %xmm0
  34cec9:      	jmp	0x34cef7 <_RINvNtNtCsdxhMBRXnOVY_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x87>
  34cecb:      	nopl	(%rax,%rax)
  34ced0:      	vmovdqu64	-0x80(%rbp), %zmm1
  34ced7:      	vptestmb	%zmm1, %zmm1, %k0
  34cedd:      	kmovq	%k0, (%rdi)
  34cee2:      	addq	$0x8, %rdi
  34cee6:      	addq	$0x200, %r14            # imm = 0x200
  34ceed:      	cmpq	-0x30(%rbp), %rdi
  34cef1:      	je	0x34cf77 <_RINvNtNtCsdxhMBRXnOVY_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x107>
  34cef7:      	vmovups	%zmm0, -0x80(%rbp)
  34cefe:      	movq	%r14, %r15
  34cf01:      	xorl	%r12d, %r12d
  34cf04:      	jmp	0x34cf45 <_RINvNtNtCsdxhMBRXnOVY_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0xd5>
  34cf06:      	nopw	%cs:(%rax,%rax)
  34cf10:      	movq	0x28(%r11), %rax
  34cf14:      	movq	(%r13), %r13
  34cf18:      	movq	(%rax), %rax
  34cf1b:      	cmpq	%r10, %r13
  34cf1e:      	sete	%r8b
  34cf22:      	cmpq	%r10, %rax
  34cf25:      	sete	%sil
  34cf29:      	orb	%r8b, %sil
  34cf2c:      	orb	%sil, (%rbx)
  34cf2f:      	cmpq	%rax, %r13
  34cf32:      	setl	-0x80(%rbp,%r12)
  34cf38:      	incq	%r12
  34cf3b:      	addq	$0x8, %r15
  34cf3f:      	cmpq	$0x40, %r12
  34cf43:      	je	0x34ced0 <_RINvNtNtCsdxhMBRXnOVY_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x60>
  34cf45:      	cmpl	$0x1, (%r11)
  34cf49:      	jne	0x34cf60 <_RINvNtNtCsdxhMBRXnOVY_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0xf0>
  34cf4b:      	movq	0x10(%r11), %r13
  34cf4f:      	cmpl	$0x1, 0x18(%r11)
  34cf54:      	je	0x34cf10 <_RINvNtNtCsdxhMBRXnOVY_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0xa0>
  34cf56:      	jmp	0x34cf6e <_RINvNtNtCsdxhMBRXnOVY_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0xfe>
  34cf58:      	nopl	(%rax,%rax)
  34cf60:      	movq	0x8(%r11), %r13
  34cf64:      	addq	%r15, %r13
  34cf67:      	cmpl	$0x1, 0x18(%r11)
  34cf6c:      	je	0x34cf10 <_RINvNtNtCsdxhMBRXnOVY_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0xa0>
  34cf6e:      	movq	0x20(%r11), %rax
  34cf72:      	addq	%r15, %rax
  34cf75:      	jmp	0x34cf14 <_RINvNtNtCsdxhMBRXnOVY_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0xa4>
  34cf77:      	testq	%r9, %r9
  34cf7a:      	je	0x34d027 <_RINvNtNtCsdxhMBRXnOVY_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x1b7>
  34cf80:      	movq	(%rcx), %rdi
  34cf83:      	movq	0x18(%rcx), %r11
  34cf87:      	andq	$-0x40, %rdx
  34cf8b:      	shlq	$0x3, %rdx
  34cf8f:      	xorl	%ecx, %ecx
  34cf91:      	xorl	%ebx, %ebx
  34cf93:      	jmp	0x34cfe1 <_RINvNtNtCsdxhMBRXnOVY_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x171>
  34cf95:      	nopw	%cs:(%rax,%rax)
  34cfa0:      	movq	0x20(%rdi), %rax
  34cfa4:      	addq	%rdx, %rax
  34cfa7:      	movq	(%r14), %r14
  34cfaa:      	movq	(%rax), %rax
  34cfad:      	cmpq	%r10, %r14
  34cfb0:      	sete	%r15b
  34cfb4:      	cmpq	%r10, %rax
  34cfb7:      	sete	%r12b
  34cfbb:      	orb	%r15b, %r12b
  34cfbe:      	xorl	%r15d, %r15d
  34cfc1:      	cmpq	%rax, %r14
  34cfc4:      	setl	%r15b
  34cfc8:      	orb	%r12b, (%r11)
  34cfcb:      	leaq	0x1(%rcx), %rax
  34cfcf:      	shlq	%cl, %r15
  34cfd2:      	orq	%r15, %rbx
  34cfd5:      	addq	$0x8, %rdx
  34cfd9:      	cmpq	%rax, %r9
  34cfdc:      	movq	%rax, %rcx
  34cfdf:      	je	0x34d013 <_RINvNtNtCsdxhMBRXnOVY_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x1a3>
  34cfe1:      	cmpl	$0x1, (%rdi)
  34cfe4:      	jne	0x34d000 <_RINvNtNtCsdxhMBRXnOVY_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x190>
  34cfe6:      	movq	0x10(%rdi), %r14
  34cfea:      	cmpl	$0x1, 0x18(%rdi)
  34cfee:      	jne	0x34cfa0 <_RINvNtNtCsdxhMBRXnOVY_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x130>
  34cff0:      	jmp	0x34d00d <_RINvNtNtCsdxhMBRXnOVY_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x19d>
  34cff2:      	nopw	%cs:(%rax,%rax)
  34d000:      	movq	0x8(%rdi), %r14
  34d004:      	addq	%rdx, %r14
  34d007:      	cmpl	$0x1, 0x18(%rdi)
  34d00b:      	jne	0x34cfa0 <_RINvNtNtCsdxhMBRXnOVY_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x130>
  34d00d:      	movq	0x28(%rdi), %rax
  34d011:      	jmp	0x34cfa7 <_RINvNtNtCsdxhMBRXnOVY_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x137>
  34d013:      	movq	-0x40(%rbp), %rsi
  34d017:      	movq	-0x38(%rbp), %rdi
  34d01b:      	cmpq	%rsi, %rdi
  34d01e:      	jae	0x34d04e <_RINvNtNtCsdxhMBRXnOVY_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x1de>
  34d020:      	movq	-0x30(%rbp), %rax
  34d024:      	movq	%rbx, (%rax)
  34d027:      	addq	$0x58, %rsp
  34d02b:      	popq	%rbx
  34d02c:      	popq	%r12
  34d02e:      	popq	%r13
  34d030:      	popq	%r14
  34d032:      	popq	%r15
  34d034:      	popq	%rbp
  34d035:      	vzeroupper
  34d038:      	retq
  34d039:      	leaq	0xf8dff8(%rip), %rcx    # 0x12db038 <anon.495be8ba92c8f6abd2ddc2fd80a1ac9b.2.llvm.13825541008927929928>
  34d040:      	xorl	%edi, %edi
  34d042:      	movq	%rsi, %rdx
  34d045:      	movq	%rax, %rsi
  34d048:      	callq	*0xfead52(%rip)         # 0x1337da0 <writev+0x1337da0>
  34d04e:      	leaq	0xf8dfcb(%rip), %rdx    # 0x12db020 <anon.495be8ba92c8f6abd2ddc2fd80a1ac9b.1.llvm.13825541008927929928>
  34d055:      	vzeroupper
  34d058:      	callq	*0xfeaaba(%rip)         # 0x1337b18 <writev+0x1337b18>
