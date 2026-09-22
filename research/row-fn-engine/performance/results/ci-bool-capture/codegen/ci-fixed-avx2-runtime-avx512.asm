
/tmp/row-fn-bool-ci-artifact-fixed-avx2/codspeed/walltime/vortex-array/row_fn_bool_retry:	file format elf64-x86-64

Disassembly of section .text:

0000000000338cf0 <_RINvNtNtCsdxhMBRXnOVY_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_>:
  338cf0: 55                           	pushq	%rbp
  338cf1: 48 89 e5                     	movq	%rsp, %rbp
  338cf4: 41 57                        	pushq	%r15
  338cf6: 41 56                        	pushq	%r14
  338cf8: 41 55                        	pushq	%r13
  338cfa: 41 54                        	pushq	%r12
  338cfc: 53                           	pushq	%rbx
  338cfd: 48 83 ec 58                  	subq	$0x58, %rsp
  338d01: 48 89 4d c0                  	movq	%rcx, -0x40(%rbp)
  338d05: 49 89 d0                     	movq	%rdx, %r8
  338d08: 49 c1 e8 06                  	shrq	$0x6, %r8
  338d0c: 49 39 f0                     	cmpq	%rsi, %r8
  338d0f: 0f 87 9c 02 00 00            	ja	0x338fb1 <_RINvNtNtCsdxhMBRXnOVY_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x2c1>
  338d15: 48 89 f8                     	movq	%rdi, %rax
  338d18: 48 89 75 c8                  	movq	%rsi, -0x38(%rbp)
  338d1c: 89 d7                        	movl	%edx, %edi
  338d1e: 83 e7 3f                     	andl	$0x3f, %edi
  338d21: 4e 8d 3c c0                  	leaq	(%rax,%r8,8), %r15
  338d25: 4c 89 45 d0                  	movq	%r8, -0x30(%rbp)
  338d29: 4d 85 c0                     	testq	%r8, %r8
  338d2c: 0f 84 40 02 00 00            	je	0x338f72 <_RINvNtNtCsdxhMBRXnOVY_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x282>
  338d32: 48 be ff ff ff ff ff ff ff 7f	movabsq	$0x7fffffffffffffff, %rsi # imm = 0x7FFFFFFFFFFFFFFF
  338d3c: 44 0f b6 61 18               	movzbl	0x18(%rcx), %r12d
  338d41: 4c 8b 51 20                  	movq	0x20(%rcx), %r10
  338d45: 4c 8b 49 28                  	movq	0x28(%rcx), %r9
  338d49: 44 0f b6 41 38               	movzbl	0x38(%rcx), %r8d
  338d4e: 83 39 01                     	cmpl	$0x1, (%rcx)
  338d51: 0f 85 28 01 00 00            	jne	0x338e7f <_RINvNtNtCsdxhMBRXnOVY_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x18f>
  338d57: 4c 8b 59 10                  	movq	0x10(%rcx), %r11
  338d5b: c5 f8 57 c0                  	vxorps	%xmm0, %xmm0, %xmm0
  338d5f: 45 84 e4                     	testb	%r12b, %r12b
  338d62: 0f 84 a8 00 00 00            	je	0x338e10 <_RINvNtNtCsdxhMBRXnOVY_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x120>
  338d68: 0f 1f 84 00 00 00 00 00      	nopl	(%rax,%rax)
  338d70: 62 f1 7c 48 11 45 fe         	vmovups	%zmm0, -0x80(%rbp)
  338d77: 45 31 d2                     	xorl	%r10d, %r10d
  338d7a: 66 0f 1f 44 00 00            	nopw	(%rax,%rax)
  338d80: 4d 8b 23                     	movq	(%r11), %r12
  338d83: 4d 8b 29                     	movq	(%r9), %r13
  338d86: 49 39 f4                     	cmpq	%rsi, %r12
  338d89: 0f 94 c3                     	sete	%bl
  338d8c: 49 39 f5                     	cmpq	%rsi, %r13
  338d8f: 41 0f 94 c6                  	sete	%r14b
  338d93: 41 08 de                     	orb	%bl, %r14b
  338d96: 45 08 c6                     	orb	%r8b, %r14b
  338d99: 45 89 f0                     	movl	%r14d, %r8d
  338d9c: 41 80 e0 01                  	andb	$0x1, %r8b
  338da0: 4d 39 ec                     	cmpq	%r13, %r12
  338da3: 42 0f 9c 44 15 80            	setl	-0x80(%rbp,%r10)
  338da9: 44 88 41 38                  	movb	%r8b, 0x38(%rcx)
  338dad: 49 8b 1b                     	movq	(%r11), %rbx
  338db0: 4d 8b 21                     	movq	(%r9), %r12
  338db3: 48 39 f3                     	cmpq	%rsi, %rbx
  338db6: 41 0f 94 c5                  	sete	%r13b
  338dba: 49 39 f4                     	cmpq	%rsi, %r12
  338dbd: 41 0f 94 c0                  	sete	%r8b
  338dc1: 45 08 e8                     	orb	%r13b, %r8b
  338dc4: 45 08 f0                     	orb	%r14b, %r8b
  338dc7: 45 89 c6                     	movl	%r8d, %r14d
  338dca: 41 80 e6 01                  	andb	$0x1, %r14b
  338dce: 4c 39 e3                     	cmpq	%r12, %rbx
  338dd1: 42 0f 9c 44 15 81            	setl	-0x7f(%rbp,%r10)
  338dd7: 49 83 c2 02                  	addq	$0x2, %r10
  338ddb: 44 88 71 38                  	movb	%r14b, 0x38(%rcx)
  338ddf: 49 83 fa 40                  	cmpq	$0x40, %r10
  338de3: 75 9b                        	jne	0x338d80 <_RINvNtNtCsdxhMBRXnOVY_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x90>
  338de5: 62 f1 fe 48 6f 4d fe         	vmovdqu64	-0x80(%rbp), %zmm1
  338dec: 62 f2 75 48 26 c1            	vptestmb	%zmm1, %zmm1, %k0
  338df2: c4 e1 f8 91 00               	kmovq	%k0, (%rax)
  338df7: 48 83 c0 08                  	addq	$0x8, %rax
  338dfb: 4c 39 f8                     	cmpq	%r15, %rax
  338dfe: 0f 85 6c ff ff ff            	jne	0x338d70 <_RINvNtNtCsdxhMBRXnOVY_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x80>
  338e04: e9 69 01 00 00               	jmp	0x338f72 <_RINvNtNtCsdxhMBRXnOVY_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x282>
  338e09: 0f 1f 80 00 00 00 00         	nopl	(%rax)
  338e10: 62 f1 7c 48 11 45 fe         	vmovups	%zmm0, -0x80(%rbp)
  338e17: 45 31 c9                     	xorl	%r9d, %r9d
  338e1a: 66 0f 1f 44 00 00            	nopw	(%rax,%rax)
  338e20: 49 8b 1b                     	movq	(%r11), %rbx
  338e23: 4f 8b 34 ca                  	movq	(%r10,%r9,8), %r14
  338e27: 48 39 f3                     	cmpq	%rsi, %rbx
  338e2a: 41 0f 94 c4                  	sete	%r12b
  338e2e: 49 39 f6                     	cmpq	%rsi, %r14
  338e31: 41 0f 94 c5                  	sete	%r13b
  338e35: 45 08 e5                     	orb	%r12b, %r13b
  338e38: 45 08 e8                     	orb	%r13b, %r8b
  338e3b: 45 89 c4                     	movl	%r8d, %r12d
  338e3e: 41 80 e4 01                  	andb	$0x1, %r12b
  338e42: 4c 39 f3                     	cmpq	%r14, %rbx
  338e45: 42 0f 9c 44 0d 80            	setl	-0x80(%rbp,%r9)
  338e4b: 49 ff c1                     	incq	%r9
  338e4e: 44 88 61 38                  	movb	%r12b, 0x38(%rcx)
  338e52: 49 83 f9 40                  	cmpq	$0x40, %r9
  338e56: 75 c8                        	jne	0x338e20 <_RINvNtNtCsdxhMBRXnOVY_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x130>
  338e58: 62 f1 fe 48 6f 4d fe         	vmovdqu64	-0x80(%rbp), %zmm1
  338e5f: 62 f2 75 48 26 c1            	vptestmb	%zmm1, %zmm1, %k0
  338e65: c4 e1 f8 91 00               	kmovq	%k0, (%rax)
  338e6a: 48 83 c0 08                  	addq	$0x8, %rax
  338e6e: 49 81 c2 00 02 00 00         	addq	$0x200, %r10            # imm = 0x200
  338e75: 4c 39 f8                     	cmpq	%r15, %rax
  338e78: 75 96                        	jne	0x338e10 <_RINvNtNtCsdxhMBRXnOVY_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x120>
  338e7a: e9 f3 00 00 00               	jmp	0x338f72 <_RINvNtNtCsdxhMBRXnOVY_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x282>
  338e7f: 4c 8b 59 08                  	movq	0x8(%rcx), %r11
  338e83: c5 f8 57 c0                  	vxorps	%xmm0, %xmm0, %xmm0
  338e87: 45 84 e4                     	testb	%r12b, %r12b
  338e8a: 74 74                        	je	0x338f00 <_RINvNtNtCsdxhMBRXnOVY_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x210>
  338e8c: 0f 1f 40 00                  	nopl	(%rax)
  338e90: 62 f1 7c 48 11 45 fe         	vmovups	%zmm0, -0x80(%rbp)
  338e97: 45 31 d2                     	xorl	%r10d, %r10d
  338e9a: 66 0f 1f 44 00 00            	nopw	(%rax,%rax)
  338ea0: 4b 8b 1c d3                  	movq	(%r11,%r10,8), %rbx
  338ea4: 4d 8b 31                     	movq	(%r9), %r14
  338ea7: 48 39 f3                     	cmpq	%rsi, %rbx
  338eaa: 41 0f 94 c4                  	sete	%r12b
  338eae: 49 39 f6                     	cmpq	%rsi, %r14
  338eb1: 41 0f 94 c5                  	sete	%r13b
  338eb5: 45 08 e5                     	orb	%r12b, %r13b
  338eb8: 45 08 e8                     	orb	%r13b, %r8b
  338ebb: 45 89 c4                     	movl	%r8d, %r12d
  338ebe: 41 80 e4 01                  	andb	$0x1, %r12b
  338ec2: 4c 39 f3                     	cmpq	%r14, %rbx
  338ec5: 42 0f 9c 44 15 80            	setl	-0x80(%rbp,%r10)
  338ecb: 49 ff c2                     	incq	%r10
  338ece: 44 88 61 38                  	movb	%r12b, 0x38(%rcx)
  338ed2: 49 83 fa 40                  	cmpq	$0x40, %r10
  338ed6: 75 c8                        	jne	0x338ea0 <_RINvNtNtCsdxhMBRXnOVY_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x1b0>
  338ed8: 62 f1 fe 48 6f 4d fe         	vmovdqu64	-0x80(%rbp), %zmm1
  338edf: 62 f2 75 48 26 c1            	vptestmb	%zmm1, %zmm1, %k0
  338ee5: c4 e1 f8 91 00               	kmovq	%k0, (%rax)
  338eea: 48 83 c0 08                  	addq	$0x8, %rax
  338eee: 49 81 c3 00 02 00 00         	addq	$0x200, %r11            # imm = 0x200
  338ef5: 4c 39 f8                     	cmpq	%r15, %rax
  338ef8: 75 96                        	jne	0x338e90 <_RINvNtNtCsdxhMBRXnOVY_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x1a0>
  338efa: eb 76                        	jmp	0x338f72 <_RINvNtNtCsdxhMBRXnOVY_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x282>
  338efc: 0f 1f 40 00                  	nopl	(%rax)
  338f00: 62 f1 7c 48 11 45 fe         	vmovups	%zmm0, -0x80(%rbp)
  338f07: 45 31 c9                     	xorl	%r9d, %r9d
  338f0a: 66 0f 1f 44 00 00            	nopw	(%rax,%rax)
  338f10: 4b 8b 1c cb                  	movq	(%r11,%r9,8), %rbx
  338f14: 4f 8b 34 ca                  	movq	(%r10,%r9,8), %r14
  338f18: 48 39 f3                     	cmpq	%rsi, %rbx
  338f1b: 41 0f 94 c4                  	sete	%r12b
  338f1f: 49 39 f6                     	cmpq	%rsi, %r14
  338f22: 41 0f 94 c5                  	sete	%r13b
  338f26: 45 08 e5                     	orb	%r12b, %r13b
  338f29: 45 08 e8                     	orb	%r13b, %r8b
  338f2c: 45 89 c4                     	movl	%r8d, %r12d
  338f2f: 41 80 e4 01                  	andb	$0x1, %r12b
  338f33: 4c 39 f3                     	cmpq	%r14, %rbx
  338f36: 42 0f 9c 44 0d 80            	setl	-0x80(%rbp,%r9)
  338f3c: 49 ff c1                     	incq	%r9
  338f3f: 44 88 61 38                  	movb	%r12b, 0x38(%rcx)
  338f43: 49 83 f9 40                  	cmpq	$0x40, %r9
  338f47: 75 c7                        	jne	0x338f10 <_RINvNtNtCsdxhMBRXnOVY_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x220>
  338f49: 62 f1 fe 48 6f 4d fe         	vmovdqu64	-0x80(%rbp), %zmm1
  338f50: 62 f2 75 48 26 c1            	vptestmb	%zmm1, %zmm1, %k0
  338f56: c4 e1 f8 91 00               	kmovq	%k0, (%rax)
  338f5b: 48 83 c0 08                  	addq	$0x8, %rax
  338f5f: 49 81 c2 00 02 00 00         	addq	$0x200, %r10            # imm = 0x200
  338f66: 49 81 c3 00 02 00 00         	addq	$0x200, %r11            # imm = 0x200
  338f6d: 4c 39 f8                     	cmpq	%r15, %rax
  338f70: 75 8e                        	jne	0x338f00 <_RINvNtNtCsdxhMBRXnOVY_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x210>
  338f72: 48 85 ff                     	testq	%rdi, %rdi
  338f75: 74 28                        	je	0x338f9f <_RINvNtNtCsdxhMBRXnOVY_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x2af>
  338f77: 48 83 e2 c0                  	andq	$-0x40, %rdx
  338f7b: 48 89 55 80                  	movq	%rdx, -0x80(%rbp)
  338f7f: 48 8d 75 c0                  	leaq	-0x40(%rbp), %rsi
  338f83: 48 8d 55 80                  	leaq	-0x80(%rbp), %rdx
  338f87: c5 f8 77                     	vzeroupper
  338f8a: e8 31 fb ff ff               	callq	0x338ac0 <_RINvNtNtCsdxhMBRXnOVY_13vortex_buffer3bit4pack24collect_bool_word_scalarNCINvB2_23collect_bool_words_withNCINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1O_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB3D_EENtNtB3R_11row_visitor10RowVisitor19visit_deferred_boolB3w_bKB3D_NCINvXB4t_B4q_NtNtB1O_6row_fn5RowFn8dispatchB3M_E0NCB6l_s_0E0NCB3I_s_0B77_E0NCINvB2_25collect_bool_words_avx512B1F_E0E0EB4t_.llvm.17614297580313611986>
  338f8f: 48 8b 75 c8                  	movq	-0x38(%rbp), %rsi
  338f93: 48 8b 7d d0                  	movq	-0x30(%rbp), %rdi
  338f97: 48 39 f7                     	cmpq	%rsi, %rdi
  338f9a: 73 2a                        	jae	0x338fc6 <_RINvNtNtCsdxhMBRXnOVY_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x2d6>
  338f9c: 49 89 07                     	movq	%rax, (%r15)
  338f9f: 48 83 c4 58                  	addq	$0x58, %rsp
  338fa3: 5b                           	popq	%rbx
  338fa4: 41 5c                        	popq	%r12
  338fa6: 41 5d                        	popq	%r13
  338fa8: 41 5e                        	popq	%r14
  338faa: 41 5f                        	popq	%r15
  338fac: 5d                           	popq	%rbp
  338fad: c5 f8 77                     	vzeroupper
  338fb0: c3                           	retq
  338fb1: 48 8d 0d c8 72 fd 00         	leaq	0xfd72c8(%rip), %rcx    # 0x1310280 <anon.d23762f469f1c8cdb7907e1b370362a2.2.llvm.17614297580313611986>
  338fb8: 31 ff                        	xorl	%edi, %edi
  338fba: 48 89 f2                     	movq	%rsi, %rdx
  338fbd: 4c 89 c6                     	movq	%r8, %rsi
  338fc0: ff 15 f2 43 03 01            	callq	*0x10343f2(%rip)        # 0x136d3b8 <writev+0x136d3b8>
  338fc6: 48 8d 15 9b 72 fd 00         	leaq	0xfd729b(%rip), %rdx    # 0x1310268 <anon.d23762f469f1c8cdb7907e1b370362a2.1.llvm.17614297580313611986>
  338fcd: ff 15 35 40 03 01            	callq	*0x1034035(%rip)        # 0x136d008 <writev+0x136d008>
