
/tmp/row-fn-bool-ci-artifact-final-avx2/codspeed/walltime/vortex-array/row_fn_bool_retry:	file format elf64-x86-64

Disassembly of section .text:

0000000000338b90 <_RINvNtNtCsdxhMBRXnOVY_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_>:
  338b90: 55                           	pushq	%rbp
  338b91: 48 89 e5                     	movq	%rsp, %rbp
  338b94: 41 57                        	pushq	%r15
  338b96: 41 56                        	pushq	%r14
  338b98: 41 55                        	pushq	%r13
  338b9a: 41 54                        	pushq	%r12
  338b9c: 53                           	pushq	%rbx
  338b9d: 48 83 ec 18                  	subq	$0x18, %rsp
  338ba1: 49 89 f0                     	movq	%rsi, %r8
  338ba4: 48 89 d6                     	movq	%rdx, %rsi
  338ba7: 48 c1 ee 06                  	shrq	$0x6, %rsi
  338bab: 4c 39 c6                     	cmpq	%r8, %rsi
  338bae: 0f 87 16 10 00 00            	ja	0x339bca <_RINvNtNtCsdxhMBRXnOVY_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x103a>
  338bb4: 48 89 c8                     	movq	%rcx, %rax
  338bb7: 4c 89 45 c8                  	movq	%r8, -0x38(%rbp)
  338bbb: 41 89 d2                     	movl	%edx, %r10d
  338bbe: 41 83 e2 3f                  	andl	$0x3f, %r10d
  338bc2: 49 bb ff ff ff ff ff ff ff 7f	movabsq	$0x7fffffffffffffff, %r11 # imm = 0x7FFFFFFFFFFFFFFF
  338bcc: 4c 8d 0c f7                  	leaq	(%rdi,%rsi,8), %r9
  338bd0: 48 85 f6                     	testq	%rsi, %rsi
  338bd3: 0f 84 da 06 00 00            	je	0x3392b3 <_RINvNtNtCsdxhMBRXnOVY_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x723>
  338bd9: 48 8d 0c f5 00 00 00 00      	leaq	(,%rsi,8), %rcx
  338be1: 44 0f b6 68 18               	movzbl	0x18(%rax), %r13d
  338be6: 48 8b 58 20                  	movq	0x20(%rax), %rbx
  338bea: 4c 8b 78 28                  	movq	0x28(%rax), %r15
  338bee: 44 0f b6 40 38               	movzbl	0x38(%rax), %r8d
  338bf3: 83 38 01                     	cmpl	$0x1, (%rax)
  338bf6: 0f 85 31 01 00 00            	jne	0x338d2d <_RINvNtNtCsdxhMBRXnOVY_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x19d>
  338bfc: 4c 8b 70 10                  	movq	0x10(%rax), %r14
  338c00: 4d 8b 26                     	movq	(%r14), %r12
  338c03: 45 84 ed                     	testb	%r13b, %r13b
  338c06: 0f 84 d7 02 00 00            	je	0x338ee3 <_RINvNtNtCsdxhMBRXnOVY_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x353>
  338c0c: 4d 39 dc                     	cmpq	%r11, %r12
  338c0f: 0f 94 c3                     	sete	%bl
  338c12: 4d 8b 37                     	movq	(%r15), %r14
  338c15: 4d 39 de                     	cmpq	%r11, %r14
  338c18: 41 0f 94 c7                  	sete	%r15b
  338c1c: 45 31 ed                     	xorl	%r13d, %r13d
  338c1f: 4d 39 f4                     	cmpq	%r14, %r12
  338c22: 41 be 01 01 00 00            	movl	$0x101, %r14d           # imm = 0x101
  338c28: 4d 0f 4d f5                  	cmovgeq	%r13, %r14
  338c2c: 45 89 f4                     	movl	%r14d, %r12d
  338c2f: 41 c1 e4 10                  	shll	$0x10, %r12d
  338c33: 41 c1 ec 18                  	shrl	$0x18, %r12d
  338c37: 45 89 f5                     	movl	%r14d, %r13d
  338c3a: 41 c1 ed 08                  	shrl	$0x8, %r13d
  338c3e: c4 c1 79 6e c6               	vmovd	%r14d, %xmm0
  338c43: c4 c3 79 20 c5 01            	vpinsrb	$0x1, %r13d, %xmm0, %xmm0
  338c49: c4 c3 79 20 c6 02            	vpinsrb	$0x2, %r14d, %xmm0, %xmm0
  338c4f: c4 c3 79 20 c4 03            	vpinsrb	$0x3, %r12d, %xmm0, %xmm0
  338c55: c4 c3 79 20 c6 04            	vpinsrb	$0x4, %r14d, %xmm0, %xmm0
  338c5b: c4 c3 79 20 c5 05            	vpinsrb	$0x5, %r13d, %xmm0, %xmm0
  338c61: c4 c3 79 20 c6 06            	vpinsrb	$0x6, %r14d, %xmm0, %xmm0
  338c67: c4 c3 79 20 c4 07            	vpinsrb	$0x7, %r12d, %xmm0, %xmm0
  338c6d: c4 c3 79 20 c6 08            	vpinsrb	$0x8, %r14d, %xmm0, %xmm0
  338c73: c4 c3 79 20 c5 09            	vpinsrb	$0x9, %r13d, %xmm0, %xmm0
  338c79: c4 c3 79 20 c6 0a            	vpinsrb	$0xa, %r14d, %xmm0, %xmm0
  338c7f: c4 c3 79 20 c4 0b            	vpinsrb	$0xb, %r12d, %xmm0, %xmm0
  338c85: c4 c3 79 20 c6 0c            	vpinsrb	$0xc, %r14d, %xmm0, %xmm0
  338c8b: c4 c3 79 20 c5 0d            	vpinsrb	$0xd, %r13d, %xmm0, %xmm0
  338c91: c4 c3 79 20 c6 0e            	vpinsrb	$0xe, %r14d, %xmm0, %xmm0
  338c97: c4 c3 79 20 c4 0f            	vpinsrb	$0xf, %r12d, %xmm0, %xmm0
  338c9d: 62 f3 fd 48 43 c0 00         	vshufi64x2	$0x0, %zmm0, %zmm0, %zmm0 # zmm0 = zmm0[0,1,0,1,0,1,0,1]
  338ca4: 45 08 c7                     	orb	%r8b, %r15b
  338ca7: 48 83 c1 f8                  	addq	$-0x8, %rcx
  338cab: 41 89 c8                     	movl	%ecx, %r8d
  338cae: 41 f7 d0                     	notl	%r8d
  338cb1: 62 f2 7d 48 26 c0            	vptestmb	%zmm0, %zmm0, %k0
  338cb7: 41 f6 c0 38                  	testb	$0x38, %r8b
  338cbb: 74 21                        	je	0x338cde <_RINvNtNtCsdxhMBRXnOVY_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x14e>
  338cbd: 41 89 c8                     	movl	%ecx, %r8d
  338cc0: 41 c1 e8 03                  	shrl	$0x3, %r8d
  338cc4: 41 ff c0                     	incl	%r8d
  338cc7: 41 83 e0 07                  	andl	$0x7, %r8d
  338ccb: 0f 1f 44 00 00               	nopl	(%rax,%rax)
  338cd0: c4 e1 f8 91 07               	kmovq	%k0, (%rdi)
  338cd5: 48 83 c7 08                  	addq	$0x8, %rdi
  338cd9: 49 ff c8                     	decq	%r8
  338cdc: 75 f2                        	jne	0x338cd0 <_RINvNtNtCsdxhMBRXnOVY_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x140>
  338cde: 41 08 df                     	orb	%bl, %r15b
  338ce1: 48 83 f9 38                  	cmpq	$0x38, %rcx
  338ce5: 0f 82 c0 05 00 00            	jb	0x3392ab <_RINvNtNtCsdxhMBRXnOVY_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x71b>
  338ceb: 0f 1f 44 00 00               	nopl	(%rax,%rax)
  338cf0: c4 e1 f8 91 07               	kmovq	%k0, (%rdi)
  338cf5: c4 e1 f8 91 47 08            	kmovq	%k0, 0x8(%rdi)
  338cfb: c4 e1 f8 91 47 10            	kmovq	%k0, 0x10(%rdi)
  338d01: c4 e1 f8 91 47 18            	kmovq	%k0, 0x18(%rdi)
  338d07: c4 e1 f8 91 47 20            	kmovq	%k0, 0x20(%rdi)
  338d0d: c4 e1 f8 91 47 28            	kmovq	%k0, 0x28(%rdi)
  338d13: c4 e1 f8 91 47 30            	kmovq	%k0, 0x30(%rdi)
  338d19: c4 e1 f8 91 47 38            	kmovq	%k0, 0x38(%rdi)
  338d1f: 48 83 c7 40                  	addq	$0x40, %rdi
  338d23: 4c 39 cf                     	cmpq	%r9, %rdi
  338d26: 75 c8                        	jne	0x338cf0 <_RINvNtNtCsdxhMBRXnOVY_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x160>
  338d28: e9 7e 05 00 00               	jmp	0x3392ab <_RINvNtNtCsdxhMBRXnOVY_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x71b>
  338d2d: 4c 8b 70 08                  	movq	0x8(%rax), %r14
  338d31: 45 84 ed                     	testb	%r13b, %r13b
  338d34: 0f 84 49 03 00 00            	je	0x339083 <_RINvNtNtCsdxhMBRXnOVY_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x4f3>
  338d3a: 4d 8b 3f                     	movq	(%r15), %r15
  338d3d: 31 db                        	xorl	%ebx, %ebx
  338d3f: 4d 39 df                     	cmpq	%r11, %r15
  338d42: 62 d2 fd 48 7c c7            	vpbroadcastq	%r15, %zmm0
  338d48: 41 bf ff 00 00 00            	movl	$0xff, %r15d
  338d4e: 44 0f 45 fb                  	cmovnel	%ebx, %r15d
  338d52: c4 c1 7b 92 c7               	kmovd	%r15d, %k0
  338d57: 49 81 c6 c0 01 00 00         	addq	$0x1c0, %r14            # imm = 0x1C0
  338d5e: 62 f2 fd 48 59 0d a0 3c d6 ff	vpbroadcastq	-0x29c360(%rip), %zmm1 # 0x9ca08 <anon.4e66a2221d34052adc0f362485be06ac.633.llvm.17808730978696754556+0x50>
  338d68: 62 f1 fd 48 6f 15 ce 9c d6 ff	vmovdqa64	-0x296332(%rip), %zmm2 # 0xa2a40 <anon.5f9230b22d292b428d6dc1056b12f2b5.6.llvm.14864309431291582878+0xb4>
  338d72: 62 f1 fd 48 6f 1d 04 9d d6 ff	vmovdqa64	-0x2962fc(%rip), %zmm3 # 0xa2a80 <anon.5f9230b22d292b428d6dc1056b12f2b5.6.llvm.14864309431291582878+0xf4>
  338d7c: 62 f1 fd 48 6f 25 3a 9d d6 ff	vmovdqa64	-0x2962c6(%rip), %zmm4 # 0xa2ac0 <anon.5f9230b22d292b428d6dc1056b12f2b5.6.llvm.14864309431291582878+0x134>
  338d86: 45 89 c7                     	movl	%r8d, %r15d
  338d89: 0f 1f 80 00 00 00 00         	nopl	(%rax)
  338d90: 41 83 e7 01                  	andl	$0x1, %r15d
  338d94: c4 c1 78 92 e7               	kmovw	%r15d, %k4
  338d99: 62 d1 fe 48 6f 6e f9         	vmovdqu64	-0x1c0(%r14), %zmm5
  338da0: 62 d1 fe 48 6f 76 fa         	vmovdqu64	-0x180(%r14), %zmm6
  338da7: 62 d1 fe 48 6f 7e fb         	vmovdqu64	-0x140(%r14), %zmm7
  338dae: 62 51 fe 48 6f 46 fc         	vmovdqu64	-0x100(%r14), %zmm8
  338db5: 62 f2 d5 48 29 e9            	vpcmpeqq	%zmm1, %zmm5, %k5
  338dbb: 62 f2 cd 48 29 d9            	vpcmpeqq	%zmm1, %zmm6, %k3
  338dc1: 62 f2 c5 48 29 d1            	vpcmpeqq	%zmm1, %zmm7, %k2
  338dc7: 62 f2 bd 48 29 c9            	vpcmpeqq	%zmm1, %zmm8, %k1
  338dcd: c5 d4 45 e4                  	korw	%k4, %k5, %k4
  338dd1: 62 f2 fd 48 37 ed            	vpcmpgtq	%zmm5, %zmm0, %k5
  338dd7: 62 f1 7f cd 6f ea            	vmovdqu8	%zmm2, %zmm5 {%k5} {z}
  338ddd: 62 f2 fd 48 37 ee            	vpcmpgtq	%zmm6, %zmm0, %k5
  338de3: 62 f1 7f cd 6f f2            	vmovdqu8	%zmm2, %zmm6 {%k5} {z}
  338de9: 62 f2 fd 48 37 ef            	vpcmpgtq	%zmm7, %zmm0, %k5
  338def: 62 f1 7f cd 6f fa            	vmovdqu8	%zmm2, %zmm7 {%k5} {z}
  338df5: 62 d2 fd 48 37 e8            	vpcmpgtq	%zmm8, %zmm0, %k5
  338dfb: 62 71 7f cd 6f c2            	vmovdqu8	%zmm2, %zmm8 {%k5} {z}
  338e01: 62 51 fe 48 6f 4e fd         	vmovdqu64	-0xc0(%r14), %zmm9
  338e08: 62 51 fe 48 6f 56 fe         	vmovdqu64	-0x80(%r14), %zmm10
  338e0f: 62 51 fe 48 6f 5e ff         	vmovdqu64	-0x40(%r14), %zmm11
  338e16: 62 51 fe 48 6f 26            	vmovdqu64	(%r14), %zmm12
  338e1c: 62 f2 b5 48 29 e9            	vpcmpeqq	%zmm1, %zmm9, %k5
  338e22: c5 dc 45 e5                  	korw	%k5, %k4, %k4
  338e26: 62 f2 ad 48 29 e9            	vpcmpeqq	%zmm1, %zmm10, %k5
  338e2c: c5 e4 45 dd                  	korw	%k5, %k3, %k3
  338e30: 62 f2 a5 48 29 e9            	vpcmpeqq	%zmm1, %zmm11, %k5
  338e36: c5 ec 45 d5                  	korw	%k5, %k2, %k2
  338e3a: 62 f2 9d 48 29 e9            	vpcmpeqq	%zmm1, %zmm12, %k5
  338e40: c5 f4 45 cd                  	korw	%k5, %k1, %k1
  338e44: 62 d2 fd 48 37 e9            	vpcmpgtq	%zmm9, %zmm0, %k5
  338e4a: 62 71 7f cd 6f ca            	vmovdqu8	%zmm2, %zmm9 {%k5} {z}
  338e50: 62 d2 fd 48 37 ea            	vpcmpgtq	%zmm10, %zmm0, %k5
  338e56: 62 71 7f cd 6f d2            	vmovdqu8	%zmm2, %zmm10 {%k5} {z}
  338e5c: 62 d2 fd 48 37 eb            	vpcmpgtq	%zmm11, %zmm0, %k5
  338e62: 62 71 7f cd 6f da            	vmovdqu8	%zmm2, %zmm11 {%k5} {z}
  338e68: 62 d2 fd 48 37 ec            	vpcmpgtq	%zmm12, %zmm0, %k5
  338e6e: 62 71 7f cd 6f e2            	vmovdqu8	%zmm2, %zmm12 {%k5} {z}
  338e74: c5 dc 45 e0                  	korw	%k0, %k4, %k4
  338e78: c5 e4 45 dc                  	korw	%k4, %k3, %k3
  338e7c: c5 ec 45 d3                  	korw	%k3, %k2, %k2
  338e80: c5 f4 45 ca                  	korw	%k2, %k1, %k1
  338e84: c5 d1 6c ee                  	vpunpcklqdq	%xmm6, %xmm5, %xmm5 # xmm5 = xmm5[0],xmm6[0]
  338e88: c4 e3 55 38 ef 01            	vinserti128	$0x1, %xmm7, %ymm5, %ymm5
  338e8e: c4 c2 7d 59 f0               	vpbroadcastq	%xmm8, %ymm6
  338e93: c4 e3 55 02 ee c0            	vpblendd	$0xc0, %ymm6, %ymm5, %ymm5 # ymm5 = ymm5[0,1,2,3,4,5],ymm6[6,7]
  338e99: 62 d3 d5 48 3a e9 01         	vinserti64x4	$0x1, %ymm9, %zmm5, %zmm5
  338ea0: 62 d2 e5 48 7e ea            	vpermt2q	%zmm10, %zmm3, %zmm5
  338ea6: 62 d3 55 48 38 eb 03         	vinserti32x4	$0x3, %xmm11, %zmm5, %zmm5
  338ead: 62 d2 dd 48 7e ec            	vpermt2q	%zmm12, %zmm4, %zmm5
  338eb3: c5 7b 93 c1                  	kmovd	%k1, %r8d
  338eb7: 62 f2 55 48 26 cd            	vptestmb	%zmm5, %zmm5, %k1
  338ebd: c4 e1 f8 91 0c 1f            	kmovq	%k1, (%rdi,%rbx)
  338ec3: 45 84 c0                     	testb	%r8b, %r8b
  338ec6: 41 0f 95 c7                  	setne	%r15b
  338eca: 48 83 c3 08                  	addq	$0x8, %rbx
  338ece: 49 81 c6 00 02 00 00         	addq	$0x200, %r14            # imm = 0x200
  338ed5: 48 39 d9                     	cmpq	%rbx, %rcx
  338ed8: 0f 85 b2 fe ff ff            	jne	0x338d90 <_RINvNtNtCsdxhMBRXnOVY_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x200>
  338ede: e9 c8 03 00 00               	jmp	0x3392ab <_RINvNtNtCsdxhMBRXnOVY_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x71b>
  338ee3: 45 31 f6                     	xorl	%r14d, %r14d
  338ee6: 4d 39 dc                     	cmpq	%r11, %r12
  338ee9: 41 bf ff 00 00 00            	movl	$0xff, %r15d
  338eef: 45 0f 45 fe                  	cmovnel	%r14d, %r15d
  338ef3: c4 c1 7b 92 c7               	kmovd	%r15d, %k0
  338ef8: 62 d2 fd 48 7c c4            	vpbroadcastq	%r12, %zmm0
  338efe: 48 81 c3 c0 01 00 00         	addq	$0x1c0, %rbx            # imm = 0x1C0
  338f05: 62 f2 fd 48 59 0d f9 3a d6 ff	vpbroadcastq	-0x29c507(%rip), %zmm1 # 0x9ca08 <anon.4e66a2221d34052adc0f362485be06ac.633.llvm.17808730978696754556+0x50>
  338f0f: 62 f1 fd 48 6f 15 27 9b d6 ff	vmovdqa64	-0x2964d9(%rip), %zmm2 # 0xa2a40 <anon.5f9230b22d292b428d6dc1056b12f2b5.6.llvm.14864309431291582878+0xb4>
  338f19: 62 f1 fd 48 6f 1d 5d 9b d6 ff	vmovdqa64	-0x2964a3(%rip), %zmm3 # 0xa2a80 <anon.5f9230b22d292b428d6dc1056b12f2b5.6.llvm.14864309431291582878+0xf4>
  338f23: 62 f1 fd 48 6f 25 93 9b d6 ff	vmovdqa64	-0x29646d(%rip), %zmm4 # 0xa2ac0 <anon.5f9230b22d292b428d6dc1056b12f2b5.6.llvm.14864309431291582878+0x134>
  338f2d: 45 89 c7                     	movl	%r8d, %r15d
  338f30: 41 83 e7 01                  	andl	$0x1, %r15d
  338f34: c4 c1 78 92 e7               	kmovw	%r15d, %k4
  338f39: 62 f1 fe 48 6f 6b f9         	vmovdqu64	-0x1c0(%rbx), %zmm5
  338f40: 62 f1 fe 48 6f 73 fa         	vmovdqu64	-0x180(%rbx), %zmm6
  338f47: 62 f1 fe 48 6f 7b fb         	vmovdqu64	-0x140(%rbx), %zmm7
  338f4e: 62 71 fe 48 6f 43 fc         	vmovdqu64	-0x100(%rbx), %zmm8
  338f55: 62 f2 d5 48 29 e9            	vpcmpeqq	%zmm1, %zmm5, %k5
  338f5b: 62 f2 cd 48 29 d9            	vpcmpeqq	%zmm1, %zmm6, %k3
  338f61: 62 f2 c5 48 29 d1            	vpcmpeqq	%zmm1, %zmm7, %k2
  338f67: 62 f2 bd 48 29 c9            	vpcmpeqq	%zmm1, %zmm8, %k1
  338f6d: c5 d4 45 e4                  	korw	%k4, %k5, %k4
  338f71: 62 f2 d5 48 37 e8            	vpcmpgtq	%zmm0, %zmm5, %k5
  338f77: 62 f1 7f cd 6f ea            	vmovdqu8	%zmm2, %zmm5 {%k5} {z}
  338f7d: 62 f2 cd 48 37 e8            	vpcmpgtq	%zmm0, %zmm6, %k5
  338f83: 62 f1 7f cd 6f f2            	vmovdqu8	%zmm2, %zmm6 {%k5} {z}
  338f89: 62 f2 c5 48 37 e8            	vpcmpgtq	%zmm0, %zmm7, %k5
  338f8f: 62 f1 7f cd 6f fa            	vmovdqu8	%zmm2, %zmm7 {%k5} {z}
  338f95: 62 f2 bd 48 37 e8            	vpcmpgtq	%zmm0, %zmm8, %k5
  338f9b: 62 71 7f cd 6f c2            	vmovdqu8	%zmm2, %zmm8 {%k5} {z}
  338fa1: 62 71 fe 48 6f 4b fd         	vmovdqu64	-0xc0(%rbx), %zmm9
  338fa8: 62 71 fe 48 6f 53 fe         	vmovdqu64	-0x80(%rbx), %zmm10
  338faf: 62 71 fe 48 6f 5b ff         	vmovdqu64	-0x40(%rbx), %zmm11
  338fb6: 62 71 fe 48 6f 23            	vmovdqu64	(%rbx), %zmm12
  338fbc: 62 f2 b5 48 29 e9            	vpcmpeqq	%zmm1, %zmm9, %k5
  338fc2: c5 dc 45 e5                  	korw	%k5, %k4, %k4
  338fc6: 62 f2 ad 48 29 e9            	vpcmpeqq	%zmm1, %zmm10, %k5
  338fcc: c5 e4 45 dd                  	korw	%k5, %k3, %k3
  338fd0: 62 f2 a5 48 29 e9            	vpcmpeqq	%zmm1, %zmm11, %k5
  338fd6: c5 ec 45 d5                  	korw	%k5, %k2, %k2
  338fda: 62 f2 9d 48 29 e9            	vpcmpeqq	%zmm1, %zmm12, %k5
  338fe0: c5 f4 45 cd                  	korw	%k5, %k1, %k1
  338fe4: 62 f2 b5 48 37 e8            	vpcmpgtq	%zmm0, %zmm9, %k5
  338fea: 62 71 7f cd 6f ca            	vmovdqu8	%zmm2, %zmm9 {%k5} {z}
  338ff0: 62 f2 ad 48 37 e8            	vpcmpgtq	%zmm0, %zmm10, %k5
  338ff6: 62 71 7f cd 6f d2            	vmovdqu8	%zmm2, %zmm10 {%k5} {z}
  338ffc: 62 f2 a5 48 37 e8            	vpcmpgtq	%zmm0, %zmm11, %k5
  339002: 62 71 7f cd 6f da            	vmovdqu8	%zmm2, %zmm11 {%k5} {z}
  339008: 62 f2 9d 48 37 e8            	vpcmpgtq	%zmm0, %zmm12, %k5
  33900e: 62 71 7f cd 6f e2            	vmovdqu8	%zmm2, %zmm12 {%k5} {z}
  339014: c5 dc 45 e0                  	korw	%k0, %k4, %k4
  339018: c5 e4 45 dc                  	korw	%k4, %k3, %k3
  33901c: c5 ec 45 d3                  	korw	%k3, %k2, %k2
  339020: c5 f4 45 ca                  	korw	%k2, %k1, %k1
  339024: c5 d1 6c ee                  	vpunpcklqdq	%xmm6, %xmm5, %xmm5 # xmm5 = xmm5[0],xmm6[0]
  339028: c4 e3 55 38 ef 01            	vinserti128	$0x1, %xmm7, %ymm5, %ymm5
  33902e: c4 c2 7d 59 f0               	vpbroadcastq	%xmm8, %ymm6
  339033: c4 e3 55 02 ee c0            	vpblendd	$0xc0, %ymm6, %ymm5, %ymm5 # ymm5 = ymm5[0,1,2,3,4,5],ymm6[6,7]
  339039: 62 d3 d5 48 3a e9 01         	vinserti64x4	$0x1, %ymm9, %zmm5, %zmm5
  339040: 62 d2 e5 48 7e ea            	vpermt2q	%zmm10, %zmm3, %zmm5
  339046: 62 d3 55 48 38 eb 03         	vinserti32x4	$0x3, %xmm11, %zmm5, %zmm5
  33904d: 62 d2 dd 48 7e ec            	vpermt2q	%zmm12, %zmm4, %zmm5
  339053: c5 7b 93 c1                  	kmovd	%k1, %r8d
  339057: 62 f2 55 48 26 cd            	vptestmb	%zmm5, %zmm5, %k1
  33905d: c4 a1 f8 91 0c 37            	kmovq	%k1, (%rdi,%r14)
  339063: 45 84 c0                     	testb	%r8b, %r8b
  339066: 41 0f 95 c7                  	setne	%r15b
  33906a: 49 83 c6 08                  	addq	$0x8, %r14
  33906e: 48 81 c3 00 02 00 00         	addq	$0x200, %rbx            # imm = 0x200
  339075: 4c 39 f1                     	cmpq	%r14, %rcx
  339078: 0f 85 b2 fe ff ff            	jne	0x338f30 <_RINvNtNtCsdxhMBRXnOVY_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x3a0>
  33907e: e9 28 02 00 00               	jmp	0x3392ab <_RINvNtNtCsdxhMBRXnOVY_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x71b>
  339083: 41 bc c0 01 00 00            	movl	$0x1c0, %r12d           # imm = 0x1C0
  339089: 45 31 ed                     	xorl	%r13d, %r13d
  33908c: 62 f2 fd 48 59 05 72 39 d6 ff	vpbroadcastq	-0x29c68e(%rip), %zmm0 # 0x9ca08 <anon.4e66a2221d34052adc0f362485be06ac.633.llvm.17808730978696754556+0x50>
  339096: 62 f1 fd 48 6f 0d a0 99 d6 ff	vmovdqa64	-0x296660(%rip), %zmm1 # 0xa2a40 <anon.5f9230b22d292b428d6dc1056b12f2b5.6.llvm.14864309431291582878+0xb4>
  3390a0: 62 f1 fd 48 6f 15 d6 99 d6 ff	vmovdqa64	-0x29662a(%rip), %zmm2 # 0xa2a80 <anon.5f9230b22d292b428d6dc1056b12f2b5.6.llvm.14864309431291582878+0xf4>
  3390aa: 62 f1 fd 48 6f 1d 0c 9a d6 ff	vmovdqa64	-0x2965f4(%rip), %zmm3 # 0xa2ac0 <anon.5f9230b22d292b428d6dc1056b12f2b5.6.llvm.14864309431291582878+0x134>
  3390b4: 45 89 c7                     	movl	%r8d, %r15d
  3390b7: 66 0f 1f 84 00 00 00 00 00   	nopw	(%rax,%rax)
  3390c0: 41 83 e7 01                  	andl	$0x1, %r15d
  3390c4: c4 c1 78 92 df               	kmovw	%r15d, %k3
  3390c9: 62 91 fe 48 6f 64 26 f9      	vmovdqu64	-0x1c0(%r14,%r12), %zmm4
  3390d1: 62 91 fe 48 6f 6c 26 fa      	vmovdqu64	-0x180(%r14,%r12), %zmm5
  3390d9: 62 91 fe 48 6f 74 26 fb      	vmovdqu64	-0x140(%r14,%r12), %zmm6
  3390e1: 62 91 fe 48 6f 7c 26 fc      	vmovdqu64	-0x100(%r14,%r12), %zmm7
  3390e9: 62 31 fe 48 6f 44 23 f9      	vmovdqu64	-0x1c0(%rbx,%r12), %zmm8
  3390f1: 62 31 fe 48 6f 4c 23 fa      	vmovdqu64	-0x180(%rbx,%r12), %zmm9
  3390f9: 62 31 fe 48 6f 54 23 fb      	vmovdqu64	-0x140(%rbx,%r12), %zmm10
  339101: 62 31 fe 48 6f 5c 23 fc      	vmovdqu64	-0x100(%rbx,%r12), %zmm11
  339109: 62 f2 dd 48 29 c0            	vpcmpeqq	%zmm0, %zmm4, %k0
  33910f: 62 f2 d5 48 29 c8            	vpcmpeqq	%zmm0, %zmm5, %k1
  339115: 62 f2 cd 48 29 d0            	vpcmpeqq	%zmm0, %zmm6, %k2
  33911b: 62 f2 c5 48 29 e0            	vpcmpeqq	%zmm0, %zmm7, %k4
  339121: 62 f2 bd 48 29 e8            	vpcmpeqq	%zmm0, %zmm8, %k5
  339127: 62 f2 b5 48 29 f0            	vpcmpeqq	%zmm0, %zmm9, %k6
  33912d: 62 f2 ad 48 29 f8            	vpcmpeqq	%zmm0, %zmm10, %k7
  339133: c5 fc 45 ed                  	korw	%k5, %k0, %k5
  339137: 62 f2 a5 48 29 c0            	vpcmpeqq	%zmm0, %zmm11, %k0
  33913d: c5 f4 45 f6                  	korw	%k6, %k1, %k6
  339141: c5 ec 45 cf                  	korw	%k7, %k2, %k1
  339145: c5 f8 91 4d d6               	kmovw	%k1, -0x2a(%rbp)
  33914a: c5 dc 45 d0                  	korw	%k0, %k4, %k2
  33914e: 62 f2 bd 48 37 e4            	vpcmpgtq	%zmm4, %zmm8, %k4
  339154: 62 f2 b5 48 37 fd            	vpcmpgtq	%zmm5, %zmm9, %k7
  33915a: 62 f2 ad 48 37 ce            	vpcmpgtq	%zmm6, %zmm10, %k1
  339160: c5 d4 45 db                  	korw	%k3, %k5, %k3
  339164: 62 f2 a5 48 37 ef            	vpcmpgtq	%zmm7, %zmm11, %k5
  33916a: 62 f1 7f cc 6f e1            	vmovdqu8	%zmm1, %zmm4 {%k4} {z}
  339170: 62 f1 7f cf 6f e9            	vmovdqu8	%zmm1, %zmm5 {%k7} {z}
  339176: 62 f1 7f c9 6f f1            	vmovdqu8	%zmm1, %zmm6 {%k1} {z}
  33917c: 62 f1 7f cd 6f f9            	vmovdqu8	%zmm1, %zmm7 {%k5} {z}
  339182: 62 11 fe 48 6f 44 26 fd      	vmovdqu64	-0xc0(%r14,%r12), %zmm8
  33918a: 62 11 fe 48 6f 4c 26 fe      	vmovdqu64	-0x80(%r14,%r12), %zmm9
  339192: 62 11 fe 48 6f 54 26 ff      	vmovdqu64	-0x40(%r14,%r12), %zmm10
  33919a: 62 11 fe 48 6f 1c 26         	vmovdqu64	(%r14,%r12), %zmm11
  3391a1: 62 31 fe 48 6f 64 23 fd      	vmovdqu64	-0xc0(%rbx,%r12), %zmm12
  3391a9: 62 31 fe 48 6f 6c 23 fe      	vmovdqu64	-0x80(%rbx,%r12), %zmm13
  3391b1: 62 31 fe 48 6f 74 23 ff      	vmovdqu64	-0x40(%rbx,%r12), %zmm14
  3391b9: 62 31 fe 48 6f 3c 23         	vmovdqu64	(%rbx,%r12), %zmm15
  3391c0: 62 f2 bd 48 29 c0            	vpcmpeqq	%zmm0, %zmm8, %k0
  3391c6: 62 f2 b5 48 29 c8            	vpcmpeqq	%zmm0, %zmm9, %k1
  3391cc: 62 f2 ad 48 29 e0            	vpcmpeqq	%zmm0, %zmm10, %k4
  3391d2: 62 f2 9d 48 29 e8            	vpcmpeqq	%zmm0, %zmm12, %k5
  3391d8: c5 fc 45 c5                  	korw	%k5, %k0, %k0
  3391dc: 62 f2 95 48 29 e8            	vpcmpeqq	%zmm0, %zmm13, %k5
  3391e2: c5 f4 45 cd                  	korw	%k5, %k1, %k1
  3391e6: 62 f2 8d 48 29 e8            	vpcmpeqq	%zmm0, %zmm14, %k5
  3391ec: c5 dc 45 e5                  	korw	%k5, %k4, %k4
  3391f0: 62 f2 a5 48 29 e8            	vpcmpeqq	%zmm0, %zmm11, %k5
  3391f6: 62 f2 85 48 29 f8            	vpcmpeqq	%zmm0, %zmm15, %k7
  3391fc: c5 d4 45 ef                  	korw	%k7, %k5, %k5
  339200: c5 fc 45 db                  	korw	%k3, %k0, %k3
  339204: c5 f4 45 c6                  	korw	%k6, %k1, %k0
  339208: c5 f8 90 4d d6               	kmovw	-0x2a(%rbp), %k1
  33920d: c5 dc 45 c9                  	korw	%k1, %k4, %k1
  339211: c5 d4 45 d2                  	korw	%k2, %k5, %k2
  339215: 62 d2 9d 48 37 e0            	vpcmpgtq	%zmm8, %zmm12, %k4
  33921b: 62 71 7f cc 6f c1            	vmovdqu8	%zmm1, %zmm8 {%k4} {z}
  339221: 62 d2 95 48 37 e1            	vpcmpgtq	%zmm9, %zmm13, %k4
  339227: 62 71 7f cc 6f c9            	vmovdqu8	%zmm1, %zmm9 {%k4} {z}
  33922d: 62 d2 8d 48 37 e2            	vpcmpgtq	%zmm10, %zmm14, %k4
  339233: 62 71 7f cc 6f d1            	vmovdqu8	%zmm1, %zmm10 {%k4} {z}
  339239: 62 d2 85 48 37 e3            	vpcmpgtq	%zmm11, %zmm15, %k4
  33923f: 62 71 7f cc 6f d9            	vmovdqu8	%zmm1, %zmm11 {%k4} {z}
  339245: c5 fc 45 c3                  	korw	%k3, %k0, %k0
  339249: c5 f4 45 c0                  	korw	%k0, %k1, %k0
  33924d: c5 ec 45 c0                  	korw	%k0, %k2, %k0
  339251: c5 d9 6c e5                  	vpunpcklqdq	%xmm5, %xmm4, %xmm4 # xmm4 = xmm4[0],xmm5[0]
  339255: c4 e3 5d 38 e6 01            	vinserti128	$0x1, %xmm6, %ymm4, %ymm4
  33925b: c4 e2 7d 59 ef               	vpbroadcastq	%xmm7, %ymm5
  339260: c4 e3 5d 02 e5 c0            	vpblendd	$0xc0, %ymm5, %ymm4, %ymm4 # ymm4 = ymm4[0,1,2,3,4,5],ymm5[6,7]
  339266: 62 d3 dd 48 3a e0 01         	vinserti64x4	$0x1, %ymm8, %zmm4, %zmm4
  33926d: 62 d2 ed 48 7e e1            	vpermt2q	%zmm9, %zmm2, %zmm4
  339273: 62 d3 5d 48 38 e2 03         	vinserti32x4	$0x3, %xmm10, %zmm4, %zmm4
  33927a: 62 d2 e5 48 7e e3            	vpermt2q	%zmm11, %zmm3, %zmm4
  339280: c5 7b 93 c0                  	kmovd	%k0, %r8d
  339284: 62 f2 5d 48 26 c4            	vptestmb	%zmm4, %zmm4, %k0
  33928a: c4 a1 f8 91 04 2f            	kmovq	%k0, (%rdi,%r13)
  339290: 45 84 c0                     	testb	%r8b, %r8b
  339293: 41 0f 95 c7                  	setne	%r15b
  339297: 49 83 c5 08                  	addq	$0x8, %r13
  33929b: 49 81 c4 00 02 00 00         	addq	$0x200, %r12            # imm = 0x200
  3392a2: 4c 39 e9                     	cmpq	%r13, %rcx
  3392a5: 0f 85 15 fe ff ff            	jne	0x3390c0 <_RINvNtNtCsdxhMBRXnOVY_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x530>
  3392ab: 41 80 e7 01                  	andb	$0x1, %r15b
  3392af: 44 88 78 38                  	movb	%r15b, 0x38(%rax)
  3392b3: 4d 85 d2                     	testq	%r10, %r10
  3392b6: 0f 84 fc 08 00 00            	je	0x339bb8 <_RINvNtNtCsdxhMBRXnOVY_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x1028>
  3392bc: 49 89 d6                     	movq	%rdx, %r14
  3392bf: 49 83 e6 c0                  	andq	$-0x40, %r14
  3392c3: 44 0f b6 78 18               	movzbl	0x18(%rax), %r15d
  3392c8: 44 0f b6 40 38               	movzbl	0x38(%rax), %r8d
  3392cd: 48 8b 78 20                  	movq	0x20(%rax), %rdi
  3392d1: 48 8b 48 28                  	movq	0x28(%rax), %rcx
  3392d5: 83 38 01                     	cmpl	$0x1, (%rax)
  3392d8: 75 3c                        	jne	0x339316 <_RINvNtNtCsdxhMBRXnOVY_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x786>
  3392da: 48 8b 58 10                  	movq	0x10(%rax), %rbx
  3392de: 48 8b 1b                     	movq	(%rbx), %rbx
  3392e1: 45 84 ff                     	testb	%r15b, %r15b
  3392e4: 74 4f                        	je	0x339335 <_RINvNtNtCsdxhMBRXnOVY_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x7a5>
  3392e6: 4c 39 db                     	cmpq	%r11, %rbx
  3392e9: 41 0f 94 c6                  	sete	%r14b
  3392ed: 48 8b 09                     	movq	(%rcx), %rcx
  3392f0: 4c 39 d9                     	cmpq	%r11, %rcx
  3392f3: 41 0f 94 c7                  	sete	%r15b
  3392f7: 31 ff                        	xorl	%edi, %edi
  3392f9: 48 39 cb                     	cmpq	%rcx, %rbx
  3392fc: 40 0f 9c c7                  	setl	%dil
  339300: 45 08 c7                     	orb	%r8b, %r15b
  339303: 45 08 f7                     	orb	%r14b, %r15b
  339306: 41 83 fa 04                  	cmpl	$0x4, %r10d
  33930a: 73 4f                        	jae	0x33935b <_RINvNtNtCsdxhMBRXnOVY_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x7cb>
  33930c: 45 31 e4                     	xorl	%r12d, %r12d
  33930f: 31 c9                        	xorl	%ecx, %ecx
  339311: e9 ba 01 00 00               	jmp	0x3394d0 <_RINvNtNtCsdxhMBRXnOVY_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x940>
  339316: 48 8b 58 08                  	movq	0x8(%rax), %rbx
  33931a: 45 84 ff                     	testb	%r15b, %r15b
  33931d: 74 29                        	je	0x339348 <_RINvNtNtCsdxhMBRXnOVY_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x7b8>
  33931f: 48 8b 39                     	movq	(%rcx), %rdi
  339322: 41 83 fa 04                  	cmpl	$0x4, %r10d
  339326: 73 43                        	jae	0x33936b <_RINvNtNtCsdxhMBRXnOVY_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x7db>
  339328: 45 31 e4                     	xorl	%r12d, %r12d
  33932b: 45 89 c7                     	movl	%r8d, %r15d
  33932e: 31 c9                        	xorl	%ecx, %ecx
  339330: e9 ad 03 00 00               	jmp	0x3396e2 <_RINvNtNtCsdxhMBRXnOVY_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0xb52>
  339335: 41 83 fa 04                  	cmpl	$0x4, %r10d
  339339: 73 4a                        	jae	0x339385 <_RINvNtNtCsdxhMBRXnOVY_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x7f5>
  33933b: 45 31 e4                     	xorl	%r12d, %r12d
  33933e: 45 89 c7                     	movl	%r8d, %r15d
  339341: 31 c9                        	xorl	%ecx, %ecx
  339343: e9 da 05 00 00               	jmp	0x339922 <_RINvNtNtCsdxhMBRXnOVY_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0xd92>
  339348: 41 83 fa 04                  	cmpl	$0x4, %r10d
  33934c: 73 51                        	jae	0x33939f <_RINvNtNtCsdxhMBRXnOVY_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x80f>
  33934e: 45 31 e4                     	xorl	%r12d, %r12d
  339351: 45 89 c7                     	movl	%r8d, %r15d
  339354: 31 c9                        	xorl	%ecx, %ecx
  339356: e9 00 08 00 00               	jmp	0x339b5b <_RINvNtNtCsdxhMBRXnOVY_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0xfcb>
  33935b: 41 83 fa 10                  	cmpl	$0x10, %r10d
  33935f: 73 55                        	jae	0x3393b6 <_RINvNtNtCsdxhMBRXnOVY_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x826>
  339361: 31 c9                        	xorl	%ecx, %ecx
  339363: 45 31 e4                     	xorl	%r12d, %r12d
  339366: e9 e8 00 00 00               	jmp	0x339453 <_RINvNtNtCsdxhMBRXnOVY_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x8c3>
  33936b: 45 31 ed                     	xorl	%r13d, %r13d
  33936e: 41 83 fa 10                  	cmpl	$0x10, %r10d
  339372: 0f 83 6e 01 00 00            	jae	0x3394e6 <_RINvNtNtCsdxhMBRXnOVY_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x956>
  339378: 31 c9                        	xorl	%ecx, %ecx
  33937a: 45 89 c7                     	movl	%r8d, %r15d
  33937d: 45 31 e4                     	xorl	%r12d, %r12d
  339380: e9 9d 02 00 00               	jmp	0x339622 <_RINvNtNtCsdxhMBRXnOVY_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0xa92>
  339385: 45 31 ed                     	xorl	%r13d, %r13d
  339388: 41 83 fa 10                  	cmpl	$0x10, %r10d
  33938c: 0f 83 9a 03 00 00            	jae	0x33972c <_RINvNtNtCsdxhMBRXnOVY_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0xb9c>
  339392: 31 c9                        	xorl	%ecx, %ecx
  339394: 45 89 c7                     	movl	%r8d, %r15d
  339397: 45 31 e4                     	xorl	%r12d, %r12d
  33939a: e9 c3 04 00 00               	jmp	0x339862 <_RINvNtNtCsdxhMBRXnOVY_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0xcd2>
  33939f: 41 83 fa 10                  	cmpl	$0x10, %r10d
  3393a3: 0f 83 c3 05 00 00            	jae	0x33996c <_RINvNtNtCsdxhMBRXnOVY_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0xddc>
  3393a9: 31 c9                        	xorl	%ecx, %ecx
  3393ab: 45 89 c7                     	movl	%r8d, %r15d
  3393ae: 45 31 e4                     	xorl	%r12d, %r12d
  3393b1: e9 fa 06 00 00               	jmp	0x339ab0 <_RINvNtNtCsdxhMBRXnOVY_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0xf20>
  3393b6: 89 d1                        	movl	%edx, %ecx
  3393b8: 83 e1 30                     	andl	$0x30, %ecx
  3393bb: 62 f2 fd 48 7c c7            	vpbroadcastq	%rdi, %zmm0
  3393c1: 62 f1 fd 48 6f 15 35 97 d6 ff	vmovdqa64	-0x2968cb(%rip), %zmm2 # 0xa2b00 <anon.5f9230b22d292b428d6dc1056b12f2b5.6.llvm.14864309431291582878+0x174>
  3393cb: c5 f1 ef c9                  	vpxor	%xmm1, %xmm1, %xmm1
  3393cf: 62 f2 fd 48 59 1d a7 30 d6 ff	vpbroadcastq	-0x29cf59(%rip), %zmm3 # 0x9c480 <anon.5e4eb8cc45d99afa7f400ea3fba78f81.168.llvm.15372149864199885434>
  3393d9: 62 f2 fd 48 59 25 b5 3c d6 ff	vpbroadcastq	-0x29c34b(%rip), %zmm4 # 0x9d098 <anon.4e66a2221d34052adc0f362485be06ac.728.llvm.17808730978696754556+0x48>
  3393e3: 49 89 c8                     	movq	%rcx, %r8
  3393e6: c5 d1 ef ed                  	vpxor	%xmm5, %xmm5, %xmm5
  3393ea: 66 0f 1f 44 00 00            	nopw	(%rax,%rax)
  3393f0: 62 f1 ed 48 d4 f3            	vpaddq	%zmm3, %zmm2, %zmm6
  3393f6: 62 f2 fd 48 47 fa            	vpsllvq	%zmm2, %zmm0, %zmm7
  3393fc: 62 f1 c5 48 eb c9            	vporq	%zmm1, %zmm7, %zmm1
  339402: 62 f2 fd 48 47 f6            	vpsllvq	%zmm6, %zmm0, %zmm6
  339408: 62 f1 cd 48 eb ed            	vporq	%zmm5, %zmm6, %zmm5
  33940e: 62 f1 ed 48 d4 d4            	vpaddq	%zmm4, %zmm2, %zmm2
  339414: 49 83 c0 f0                  	addq	$-0x10, %r8
  339418: 75 d6                        	jne	0x3393f0 <_RINvNtNtCsdxhMBRXnOVY_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x860>
  33941a: 62 f1 d5 48 eb c1            	vporq	%zmm1, %zmm5, %zmm0
  339420: 62 f3 fd 48 3b c1 01         	vextracti64x4	$0x1, %zmm0, %ymm1
  339427: 62 f1 fd 48 eb c1            	vporq	%zmm1, %zmm0, %zmm0
  33942d: c4 e3 7d 39 c1 01            	vextracti128	$0x1, %ymm0, %xmm1
  339433: c5 f9 eb c1                  	vpor	%xmm1, %xmm0, %xmm0
  339437: c5 f9 70 c8 ee               	vpshufd	$0xee, %xmm0, %xmm1     # xmm1 = xmm0[2,3,2,3]
  33943c: c5 f9 eb c1                  	vpor	%xmm1, %xmm0, %xmm0
  339440: c4 c1 f9 7e c4               	vmovq	%xmm0, %r12
  339445: 41 39 ca                     	cmpl	%ecx, %r10d
  339448: 0f 84 56 07 00 00            	je	0x339ba4 <_RINvNtNtCsdxhMBRXnOVY_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x1014>
  33944e: f6 c2 0c                     	testb	$0xc, %dl
  339451: 74 7d                        	je	0x3394d0 <_RINvNtNtCsdxhMBRXnOVY_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x940>
  339453: 49 89 c8                     	movq	%rcx, %r8
  339456: 89 d1                        	movl	%edx, %ecx
  339458: 83 e1 3c                     	andl	$0x3c, %ecx
  33945b: c4 c1 f9 6e c4               	vmovq	%r12, %xmm0
  339460: c4 e1 f9 6e d7               	vmovq	%rdi, %xmm2
  339465: c4 c1 f9 6e c8               	vmovq	%r8, %xmm1
  33946a: c4 e2 7d 59 c9               	vpbroadcastq	%xmm1, %ymm1
  33946f: c5 f5 eb 0d 09 4f d6 ff      	vpor	-0x29b0f7(%rip), %ymm1, %ymm1 # 0x9e380 <anon.6f50054b7ece9027416a5bb6e70b105d.10.llvm.9732915856748287937+0x80>
  339477: c4 e2 7d 59 d2               	vpbroadcastq	%xmm2, %ymm2
  33947c: 49 29 c8                     	subq	%rcx, %r8
  33947f: c4 e2 7d 59 1d a0 3b d6 ff   	vpbroadcastq	-0x29c460(%rip), %ymm3 # 0x9d028 <anon.5e4eb8cc45d99afa7f400ea3fba78f81.169.llvm.15372149864199885434>
  339488: 0f 1f 84 00 00 00 00 00      	nopl	(%rax,%rax)
  339490: c4 e2 ed 47 e1               	vpsllvq	%ymm1, %ymm2, %ymm4
  339495: c5 dd eb c0                  	vpor	%ymm0, %ymm4, %ymm0
  339499: c5 f5 d4 cb                  	vpaddq	%ymm3, %ymm1, %ymm1
  33949d: 49 83 c0 04                  	addq	$0x4, %r8
  3394a1: 75 ed                        	jne	0x339490 <_RINvNtNtCsdxhMBRXnOVY_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x900>
  3394a3: c4 e3 7d 39 c1 01            	vextracti128	$0x1, %ymm0, %xmm1
  3394a9: c5 f9 eb c1                  	vpor	%xmm1, %xmm0, %xmm0
  3394ad: c5 f9 70 c8 ee               	vpshufd	$0xee, %xmm0, %xmm1     # xmm1 = xmm0[2,3,2,3]
  3394b2: c5 f9 eb c1                  	vpor	%xmm1, %xmm0, %xmm0
  3394b6: c4 c1 f9 7e c4               	vmovq	%xmm0, %r12
  3394bb: 41 39 ca                     	cmpl	%ecx, %r10d
  3394be: 0f 84 e0 06 00 00            	je	0x339ba4 <_RINvNtNtCsdxhMBRXnOVY_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x1014>
  3394c4: 66 66 66 2e 0f 1f 84 00 00 00 00 00  	nopw	%cs:(%rax,%rax)
  3394d0: 48 89 fa                     	movq	%rdi, %rdx
  3394d3: 48 d3 e2                     	shlq	%cl, %rdx
  3394d6: 48 ff c1                     	incq	%rcx
  3394d9: 49 09 d4                     	orq	%rdx, %r12
  3394dc: 49 39 ca                     	cmpq	%rcx, %r10
  3394df: 75 ef                        	jne	0x3394d0 <_RINvNtNtCsdxhMBRXnOVY_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x940>
  3394e1: e9 be 06 00 00               	jmp	0x339ba4 <_RINvNtNtCsdxhMBRXnOVY_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x1014>
  3394e6: 89 d1                        	movl	%edx, %ecx
  3394e8: 83 e1 30                     	andl	$0x30, %ecx
  3394eb: 41 83 e0 01                  	andl	$0x1, %r8d
  3394ef: 4c 39 df                     	cmpq	%r11, %rdi
  3394f2: c4 c1 78 92 c0               	kmovw	%r8d, %k0
  3394f7: 62 f2 fd 48 7c c7            	vpbroadcastq	%rdi, %zmm0
  3394fd: 41 b8 ff 00 00 00            	movl	$0xff, %r8d
  339503: 45 0f 45 c5                  	cmovnel	%r13d, %r8d
  339507: c4 c1 7b 92 c8               	kmovd	%r8d, %k1
  33950c: 4e 8d 04 f3                  	leaq	(%rbx,%r14,8), %r8
  339510: c5 fc 47 d0                  	kxorw	%k0, %k0, %k2
  339514: 62 f1 fd 48 6f 15 e2 95 d6 ff	vmovdqa64	-0x296a1e(%rip), %zmm2 # 0xa2b00 <anon.5f9230b22d292b428d6dc1056b12f2b5.6.llvm.14864309431291582878+0x174>
  33951e: c5 f1 ef c9                  	vpxor	%xmm1, %xmm1, %xmm1
  339522: 62 f2 fd 48 59 25 54 2f d6 ff	vpbroadcastq	-0x29d0ac(%rip), %zmm4 # 0x9c480 <anon.5e4eb8cc45d99afa7f400ea3fba78f81.168.llvm.15372149864199885434>
  33952c: 62 f2 fd 48 59 2d d2 34 d6 ff	vpbroadcastq	-0x29cb2e(%rip), %zmm5 # 0x9ca08 <anon.4e66a2221d34052adc0f362485be06ac.633.llvm.17808730978696754556+0x50>
  339536: 62 f2 fd 48 59 35 58 3b d6 ff	vpbroadcastq	-0x29c4a8(%rip), %zmm6 # 0x9d098 <anon.4e66a2221d34052adc0f362485be06ac.728.llvm.17808730978696754556+0x48>
  339540: 49 89 cf                     	movq	%rcx, %r15
  339543: c5 e1 ef db                  	vpxor	%xmm3, %xmm3, %xmm3
  339547: 66 0f 1f 84 00 00 00 00 00   	nopw	(%rax,%rax)
  339550: 62 f1 ed 48 d4 fc            	vpaddq	%zmm4, %zmm2, %zmm7
  339556: c4 c1 f9 7e d4               	vmovq	%xmm2, %r12
  33955b: 62 11 fe 48 6f 04 e0         	vmovdqu64	(%r8,%r12,8), %zmm8
  339562: 62 11 fe 48 6f 4c e0 01      	vmovdqu64	0x40(%r8,%r12,8), %zmm9
  33956a: 62 f2 bd 48 29 dd            	vpcmpeqq	%zmm5, %zmm8, %k3
  339570: 62 f2 b5 48 29 e5            	vpcmpeqq	%zmm5, %zmm9, %k4
  339576: 62 d2 fd 48 37 e8            	vpcmpgtq	%zmm8, %zmm0, %k5
  33957c: 62 d2 fd 48 37 f1            	vpcmpgtq	%zmm9, %zmm0, %k6
  339582: c5 e4 45 c0                  	korw	%k0, %k3, %k0
  339586: c5 fc 45 c1                  	korw	%k1, %k0, %k0
  33958a: c5 dc 45 da                  	korw	%k2, %k4, %k3
  33958e: c5 e4 45 d1                  	korw	%k1, %k3, %k2
  339592: 62 53 bd cd 25 c0 ff         	vpternlogq	$0xff, %zmm8, %zmm8, %zmm8 {%k5} {z} # zmm8 {%k5} {z} = -1
  339599: 62 d1 bd 48 73 d0 3f         	vpsrlq	$0x3f, %zmm8, %zmm8
  3395a0: 62 53 b5 ce 25 c9 ff         	vpternlogq	$0xff, %zmm9, %zmm9, %zmm9 {%k6} {z} # zmm9 {%k6} {z} = -1
  3395a7: 62 d1 b5 48 73 d1 3f         	vpsrlq	$0x3f, %zmm9, %zmm9
  3395ae: 62 f2 b5 48 47 ff            	vpsllvq	%zmm7, %zmm9, %zmm7
  3395b4: 62 f1 c5 48 eb db            	vporq	%zmm3, %zmm7, %zmm3
  3395ba: 62 f2 bd 48 47 fa            	vpsllvq	%zmm2, %zmm8, %zmm7
  3395c0: 62 f1 c5 48 eb c9            	vporq	%zmm1, %zmm7, %zmm1
  3395c6: 62 f1 ed 48 d4 d6            	vpaddq	%zmm6, %zmm2, %zmm2
  3395cc: 49 83 c7 f0                  	addq	$-0x10, %r15
  3395d0: 0f 85 7a ff ff ff            	jne	0x339550 <_RINvNtNtCsdxhMBRXnOVY_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x9c0>
  3395d6: c5 e4 45 c0                  	korw	%k0, %k3, %k0
  3395da: c5 7b 93 c0                  	kmovd	%k0, %r8d
  3395de: 45 84 c0                     	testb	%r8b, %r8b
  3395e1: 41 0f 95 c7                  	setne	%r15b
  3395e5: 62 f1 e5 48 eb c1            	vporq	%zmm1, %zmm3, %zmm0
  3395eb: 62 f3 fd 48 3b c1 01         	vextracti64x4	$0x1, %zmm0, %ymm1
  3395f2: 62 f1 fd 48 eb c1            	vporq	%zmm1, %zmm0, %zmm0
  3395f8: c4 e3 7d 39 c1 01            	vextracti128	$0x1, %ymm0, %xmm1
  3395fe: c5 f9 eb c1                  	vpor	%xmm1, %xmm0, %xmm0
  339602: c5 f9 70 c8 ee               	vpshufd	$0xee, %xmm0, %xmm1     # xmm1 = xmm0[2,3,2,3]
  339607: c5 f9 eb c1                  	vpor	%xmm1, %xmm0, %xmm0
  33960b: c4 c1 f9 7e c4               	vmovq	%xmm0, %r12
  339610: 41 39 ca                     	cmpl	%ecx, %r10d
  339613: 0f 84 8b 05 00 00            	je	0x339ba4 <_RINvNtNtCsdxhMBRXnOVY_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x1014>
  339619: f6 c2 0c                     	testb	$0xc, %dl
  33961c: 0f 84 c0 00 00 00            	je	0x3396e2 <_RINvNtNtCsdxhMBRXnOVY_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0xb52>
  339622: 49 89 c8                     	movq	%rcx, %r8
  339625: 89 d1                        	movl	%edx, %ecx
  339627: 83 e1 3c                     	andl	$0x3c, %ecx
  33962a: 41 83 e7 01                  	andl	$0x1, %r15d
  33962e: 4c 39 df                     	cmpq	%r11, %rdi
  339631: c4 c1 78 92 c7               	kmovw	%r15d, %k0
  339636: c4 c1 f9 6e c4               	vmovq	%r12, %xmm0
  33963b: c4 e1 f9 6e cf               	vmovq	%rdi, %xmm1
  339640: ba ff 00 00 00               	movl	$0xff, %edx
  339645: 44 0f 44 ea                  	cmovel	%edx, %r13d
  339649: c4 e2 7d 59 c9               	vpbroadcastq	%xmm1, %ymm1
  33964e: c4 c1 7b 92 cd               	kmovd	%r13d, %k1
  339653: c4 c1 f9 6e d0               	vmovq	%r8, %xmm2
  339658: c4 e2 7d 59 d2               	vpbroadcastq	%xmm2, %ymm2
  33965d: c5 ed eb 15 1b 4d d6 ff      	vpor	-0x29b2e5(%rip), %ymm2, %ymm2 # 0x9e380 <anon.6f50054b7ece9027416a5bb6e70b105d.10.llvm.9732915856748287937+0x80>
  339665: 4a 8d 14 f3                  	leaq	(%rbx,%r14,8), %rdx
  339669: 49 29 c8                     	subq	%rcx, %r8
  33966c: 62 f2 fd 48 59 1d 92 33 d6 ff	vpbroadcastq	-0x29cc6e(%rip), %zmm3 # 0x9ca08 <anon.4e66a2221d34052adc0f362485be06ac.633.llvm.17808730978696754556+0x50>
  339676: c4 e2 7d 59 25 a9 39 d6 ff   	vpbroadcastq	-0x29c657(%rip), %ymm4 # 0x9d028 <anon.5e4eb8cc45d99afa7f400ea3fba78f81.169.llvm.15372149864199885434>
  33967f: 90                           	nop
  339680: c4 c1 f9 7e d6               	vmovq	%xmm2, %r14
  339685: c4 a1 7e 6f 2c f2            	vmovdqu	(%rdx,%r14,8), %ymm5
  33968b: 62 f2 d5 48 29 d3            	vpcmpeqq	%zmm3, %zmm5, %k2
  339691: c5 ec 45 d1                  	korw	%k1, %k2, %k2
  339695: c5 ec 45 c0                  	korw	%k0, %k2, %k0
  339699: c4 e2 75 37 ed               	vpcmpgtq	%ymm5, %ymm1, %ymm5
  33969e: c5 d5 73 d5 3f               	vpsrlq	$0x3f, %ymm5, %ymm5
  3396a3: c4 e2 d5 47 ea               	vpsllvq	%ymm2, %ymm5, %ymm5
  3396a8: c5 d5 eb c0                  	vpor	%ymm0, %ymm5, %ymm0
  3396ac: c5 ed d4 d4                  	vpaddq	%ymm4, %ymm2, %ymm2
  3396b0: 49 83 c0 04                  	addq	$0x4, %r8
  3396b4: 75 ca                        	jne	0x339680 <_RINvNtNtCsdxhMBRXnOVY_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0xaf0>
  3396b6: c5 fb 93 d0                  	kmovd	%k0, %edx
  3396ba: f6 c2 0f                     	testb	$0xf, %dl
  3396bd: 41 0f 95 c7                  	setne	%r15b
  3396c1: c4 e3 7d 39 c1 01            	vextracti128	$0x1, %ymm0, %xmm1
  3396c7: c5 f9 eb c1                  	vpor	%xmm1, %xmm0, %xmm0
  3396cb: c5 f9 70 c8 ee               	vpshufd	$0xee, %xmm0, %xmm1     # xmm1 = xmm0[2,3,2,3]
  3396d0: c5 f9 eb c1                  	vpor	%xmm1, %xmm0, %xmm0
  3396d4: c4 c1 f9 7e c4               	vmovq	%xmm0, %r12
  3396d9: 41 39 ca                     	cmpl	%ecx, %r10d
  3396dc: 0f 84 c2 04 00 00            	je	0x339ba4 <_RINvNtNtCsdxhMBRXnOVY_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x1014>
  3396e2: 48 89 f2                     	movq	%rsi, %rdx
  3396e5: 48 c1 e2 09                  	shlq	$0x9, %rdx
  3396e9: 48 01 d3                     	addq	%rdx, %rbx
  3396ec: 44 89 fa                     	movl	%r15d, %edx
  3396ef: 90                           	nop
  3396f0: 4c 39 df                     	cmpq	%r11, %rdi
  3396f3: 41 0f 94 c7                  	sete	%r15b
  3396f7: 4c 8b 04 cb                  	movq	(%rbx,%rcx,8), %r8
  3396fb: 4d 39 d8                     	cmpq	%r11, %r8
  3396fe: 41 0f 94 c6                  	sete	%r14b
  339702: 45 31 ed                     	xorl	%r13d, %r13d
  339705: 49 39 f8                     	cmpq	%rdi, %r8
  339708: 41 0f 9c c5                  	setl	%r13b
  33970c: 41 08 d7                     	orb	%dl, %r15b
  33970f: 45 08 f7                     	orb	%r14b, %r15b
  339712: 4c 8d 41 01                  	leaq	0x1(%rcx), %r8
  339716: 49 d3 e5                     	shlq	%cl, %r13
  339719: 4d 09 ec                     	orq	%r13, %r12
  33971c: 44 89 fa                     	movl	%r15d, %edx
  33971f: 4d 39 c2                     	cmpq	%r8, %r10
  339722: 4c 89 c1                     	movq	%r8, %rcx
  339725: 75 c9                        	jne	0x3396f0 <_RINvNtNtCsdxhMBRXnOVY_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0xb60>
  339727: e9 78 04 00 00               	jmp	0x339ba4 <_RINvNtNtCsdxhMBRXnOVY_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x1014>
  33972c: 89 d1                        	movl	%edx, %ecx
  33972e: 83 e1 30                     	andl	$0x30, %ecx
  339731: 41 83 e0 01                  	andl	$0x1, %r8d
  339735: 4c 39 db                     	cmpq	%r11, %rbx
  339738: c4 c1 78 92 c0               	kmovw	%r8d, %k0
  33973d: 62 f2 fd 48 7c c3            	vpbroadcastq	%rbx, %zmm0
  339743: 41 b8 ff 00 00 00            	movl	$0xff, %r8d
  339749: 45 0f 45 c5                  	cmovnel	%r13d, %r8d
  33974d: c4 c1 7b 92 c8               	kmovd	%r8d, %k1
  339752: 4e 8d 04 f7                  	leaq	(%rdi,%r14,8), %r8
  339756: c5 fc 47 d0                  	kxorw	%k0, %k0, %k2
  33975a: 62 f1 fd 48 6f 15 9c 93 d6 ff	vmovdqa64	-0x296c64(%rip), %zmm2 # 0xa2b00 <anon.5f9230b22d292b428d6dc1056b12f2b5.6.llvm.14864309431291582878+0x174>
  339764: c5 f1 ef c9                  	vpxor	%xmm1, %xmm1, %xmm1
  339768: 62 f2 fd 48 59 25 0e 2d d6 ff	vpbroadcastq	-0x29d2f2(%rip), %zmm4 # 0x9c480 <anon.5e4eb8cc45d99afa7f400ea3fba78f81.168.llvm.15372149864199885434>
  339772: 62 f2 fd 48 59 2d 8c 32 d6 ff	vpbroadcastq	-0x29cd74(%rip), %zmm5 # 0x9ca08 <anon.4e66a2221d34052adc0f362485be06ac.633.llvm.17808730978696754556+0x50>
  33977c: 62 f2 fd 48 59 35 12 39 d6 ff	vpbroadcastq	-0x29c6ee(%rip), %zmm6 # 0x9d098 <anon.4e66a2221d34052adc0f362485be06ac.728.llvm.17808730978696754556+0x48>
  339786: 49 89 cf                     	movq	%rcx, %r15
  339789: c5 e1 ef db                  	vpxor	%xmm3, %xmm3, %xmm3
  33978d: 0f 1f 00                     	nopl	(%rax)
  339790: 62 f1 ed 48 d4 fc            	vpaddq	%zmm4, %zmm2, %zmm7
  339796: c4 c1 f9 7e d4               	vmovq	%xmm2, %r12
  33979b: 62 11 fe 48 6f 04 e0         	vmovdqu64	(%r8,%r12,8), %zmm8
  3397a2: 62 11 fe 48 6f 4c e0 01      	vmovdqu64	0x40(%r8,%r12,8), %zmm9
  3397aa: 62 f2 bd 48 29 dd            	vpcmpeqq	%zmm5, %zmm8, %k3
  3397b0: 62 f2 b5 48 29 e5            	vpcmpeqq	%zmm5, %zmm9, %k4
  3397b6: 62 f2 bd 48 37 e8            	vpcmpgtq	%zmm0, %zmm8, %k5
  3397bc: 62 f2 b5 48 37 f0            	vpcmpgtq	%zmm0, %zmm9, %k6
  3397c2: c5 e4 45 c0                  	korw	%k0, %k3, %k0
  3397c6: c5 dc 45 da                  	korw	%k2, %k4, %k3
  3397ca: c5 fc 45 c1                  	korw	%k1, %k0, %k0
  3397ce: c5 e4 45 d1                  	korw	%k1, %k3, %k2
  3397d2: 62 53 bd cd 25 c0 ff         	vpternlogq	$0xff, %zmm8, %zmm8, %zmm8 {%k5} {z} # zmm8 {%k5} {z} = -1
  3397d9: 62 d1 bd 48 73 d0 3f         	vpsrlq	$0x3f, %zmm8, %zmm8
  3397e0: 62 53 b5 ce 25 c9 ff         	vpternlogq	$0xff, %zmm9, %zmm9, %zmm9 {%k6} {z} # zmm9 {%k6} {z} = -1
  3397e7: 62 d1 b5 48 73 d1 3f         	vpsrlq	$0x3f, %zmm9, %zmm9
  3397ee: 62 f2 b5 48 47 ff            	vpsllvq	%zmm7, %zmm9, %zmm7
  3397f4: 62 f1 c5 48 eb db            	vporq	%zmm3, %zmm7, %zmm3
  3397fa: 62 f2 bd 48 47 fa            	vpsllvq	%zmm2, %zmm8, %zmm7
  339800: 62 f1 c5 48 eb c9            	vporq	%zmm1, %zmm7, %zmm1
  339806: 62 f1 ed 48 d4 d6            	vpaddq	%zmm6, %zmm2, %zmm2
  33980c: 49 83 c7 f0                  	addq	$-0x10, %r15
  339810: 0f 85 7a ff ff ff            	jne	0x339790 <_RINvNtNtCsdxhMBRXnOVY_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0xc00>
  339816: c5 e4 45 c0                  	korw	%k0, %k3, %k0
  33981a: c5 7b 93 c0                  	kmovd	%k0, %r8d
  33981e: 45 84 c0                     	testb	%r8b, %r8b
  339821: 41 0f 95 c7                  	setne	%r15b
  339825: 62 f1 e5 48 eb c1            	vporq	%zmm1, %zmm3, %zmm0
  33982b: 62 f3 fd 48 3b c1 01         	vextracti64x4	$0x1, %zmm0, %ymm1
  339832: 62 f1 fd 48 eb c1            	vporq	%zmm1, %zmm0, %zmm0
  339838: c4 e3 7d 39 c1 01            	vextracti128	$0x1, %ymm0, %xmm1
  33983e: c5 f9 eb c1                  	vpor	%xmm1, %xmm0, %xmm0
  339842: c5 f9 70 c8 ee               	vpshufd	$0xee, %xmm0, %xmm1     # xmm1 = xmm0[2,3,2,3]
  339847: c5 f9 eb c1                  	vpor	%xmm1, %xmm0, %xmm0
  33984b: c4 c1 f9 7e c4               	vmovq	%xmm0, %r12
  339850: 41 39 ca                     	cmpl	%ecx, %r10d
  339853: 0f 84 4b 03 00 00            	je	0x339ba4 <_RINvNtNtCsdxhMBRXnOVY_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x1014>
  339859: f6 c2 0c                     	testb	$0xc, %dl
  33985c: 0f 84 c0 00 00 00            	je	0x339922 <_RINvNtNtCsdxhMBRXnOVY_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0xd92>
  339862: 49 89 c8                     	movq	%rcx, %r8
  339865: 89 d1                        	movl	%edx, %ecx
  339867: 83 e1 3c                     	andl	$0x3c, %ecx
  33986a: 41 83 e7 01                  	andl	$0x1, %r15d
  33986e: 4c 39 db                     	cmpq	%r11, %rbx
  339871: c4 c1 78 92 c7               	kmovw	%r15d, %k0
  339876: c4 c1 f9 6e c4               	vmovq	%r12, %xmm0
  33987b: c4 e1 f9 6e cb               	vmovq	%rbx, %xmm1
  339880: ba ff 00 00 00               	movl	$0xff, %edx
  339885: 44 0f 44 ea                  	cmovel	%edx, %r13d
  339889: c4 e2 7d 59 c9               	vpbroadcastq	%xmm1, %ymm1
  33988e: c4 c1 7b 92 cd               	kmovd	%r13d, %k1
  339893: c4 c1 f9 6e d0               	vmovq	%r8, %xmm2
  339898: c4 e2 7d 59 d2               	vpbroadcastq	%xmm2, %ymm2
  33989d: c5 ed eb 15 db 4a d6 ff      	vpor	-0x29b525(%rip), %ymm2, %ymm2 # 0x9e380 <anon.6f50054b7ece9027416a5bb6e70b105d.10.llvm.9732915856748287937+0x80>
  3398a5: 4a 8d 14 f7                  	leaq	(%rdi,%r14,8), %rdx
  3398a9: 49 29 c8                     	subq	%rcx, %r8
  3398ac: 62 f2 fd 48 59 1d 52 31 d6 ff	vpbroadcastq	-0x29ceae(%rip), %zmm3 # 0x9ca08 <anon.4e66a2221d34052adc0f362485be06ac.633.llvm.17808730978696754556+0x50>
  3398b6: c4 e2 7d 59 25 69 37 d6 ff   	vpbroadcastq	-0x29c897(%rip), %ymm4 # 0x9d028 <anon.5e4eb8cc45d99afa7f400ea3fba78f81.169.llvm.15372149864199885434>
  3398bf: 90                           	nop
  3398c0: c4 c1 f9 7e d6               	vmovq	%xmm2, %r14
  3398c5: c4 a1 7e 6f 2c f2            	vmovdqu	(%rdx,%r14,8), %ymm5
  3398cb: 62 f2 d5 48 29 d3            	vpcmpeqq	%zmm3, %zmm5, %k2
  3398d1: c5 ec 45 c0                  	korw	%k0, %k2, %k0
  3398d5: c5 fc 45 c1                  	korw	%k1, %k0, %k0
  3398d9: c4 e2 55 37 e9               	vpcmpgtq	%ymm1, %ymm5, %ymm5
  3398de: c5 d5 73 d5 3f               	vpsrlq	$0x3f, %ymm5, %ymm5
  3398e3: c4 e2 d5 47 ea               	vpsllvq	%ymm2, %ymm5, %ymm5
  3398e8: c5 d5 eb c0                  	vpor	%ymm0, %ymm5, %ymm0
  3398ec: c5 ed d4 d4                  	vpaddq	%ymm4, %ymm2, %ymm2
  3398f0: 49 83 c0 04                  	addq	$0x4, %r8
  3398f4: 75 ca                        	jne	0x3398c0 <_RINvNtNtCsdxhMBRXnOVY_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0xd30>
  3398f6: c5 fb 93 d0                  	kmovd	%k0, %edx
  3398fa: f6 c2 0f                     	testb	$0xf, %dl
  3398fd: 41 0f 95 c7                  	setne	%r15b
  339901: c4 e3 7d 39 c1 01            	vextracti128	$0x1, %ymm0, %xmm1
  339907: c5 f9 eb c1                  	vpor	%xmm1, %xmm0, %xmm0
  33990b: c5 f9 70 c8 ee               	vpshufd	$0xee, %xmm0, %xmm1     # xmm1 = xmm0[2,3,2,3]
  339910: c5 f9 eb c1                  	vpor	%xmm1, %xmm0, %xmm0
  339914: c4 c1 f9 7e c4               	vmovq	%xmm0, %r12
  339919: 41 39 ca                     	cmpl	%ecx, %r10d
  33991c: 0f 84 82 02 00 00            	je	0x339ba4 <_RINvNtNtCsdxhMBRXnOVY_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x1014>
  339922: 48 89 f2                     	movq	%rsi, %rdx
  339925: 48 c1 e2 09                  	shlq	$0x9, %rdx
  339929: 48 01 d7                     	addq	%rdx, %rdi
  33992c: 44 89 fa                     	movl	%r15d, %edx
  33992f: 90                           	nop
  339930: 4c 39 db                     	cmpq	%r11, %rbx
  339933: 41 0f 94 c7                  	sete	%r15b
  339937: 4c 8b 04 cf                  	movq	(%rdi,%rcx,8), %r8
  33993b: 4d 39 d8                     	cmpq	%r11, %r8
  33993e: 41 0f 94 c6                  	sete	%r14b
  339942: 45 31 ed                     	xorl	%r13d, %r13d
  339945: 4c 39 c3                     	cmpq	%r8, %rbx
  339948: 41 0f 9c c5                  	setl	%r13b
  33994c: 41 08 d7                     	orb	%dl, %r15b
  33994f: 45 08 f7                     	orb	%r14b, %r15b
  339952: 4c 8d 41 01                  	leaq	0x1(%rcx), %r8
  339956: 49 d3 e5                     	shlq	%cl, %r13
  339959: 4d 09 ec                     	orq	%r13, %r12
  33995c: 44 89 fa                     	movl	%r15d, %edx
  33995f: 4d 39 c2                     	cmpq	%r8, %r10
  339962: 4c 89 c1                     	movq	%r8, %rcx
  339965: 75 c9                        	jne	0x339930 <_RINvNtNtCsdxhMBRXnOVY_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0xda0>
  339967: e9 38 02 00 00               	jmp	0x339ba4 <_RINvNtNtCsdxhMBRXnOVY_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x1014>
  33996c: 89 d1                        	movl	%edx, %ecx
  33996e: 83 e1 30                     	andl	$0x30, %ecx
  339971: 41 83 e0 01                  	andl	$0x1, %r8d
  339975: c4 c1 78 92 c0               	kmovw	%r8d, %k0
  33997a: c5 fc 47 c8                  	kxorw	%k0, %k0, %k1
  33997e: 62 f1 fd 48 6f 0d 78 91 d6 ff	vmovdqa64	-0x296e88(%rip), %zmm1 # 0xa2b00 <anon.5f9230b22d292b428d6dc1056b12f2b5.6.llvm.14864309431291582878+0x174>
  339988: c5 f9 ef c0                  	vpxor	%xmm0, %xmm0, %xmm0
  33998c: 62 f2 fd 48 59 1d ea 2a d6 ff	vpbroadcastq	-0x29d516(%rip), %zmm3 # 0x9c480 <anon.5e4eb8cc45d99afa7f400ea3fba78f81.168.llvm.15372149864199885434>
  339996: 62 f2 fd 48 59 25 68 30 d6 ff	vpbroadcastq	-0x29cf98(%rip), %zmm4 # 0x9ca08 <anon.4e66a2221d34052adc0f362485be06ac.633.llvm.17808730978696754556+0x50>
  3399a0: 62 f2 fd 48 59 2d ee 36 d6 ff	vpbroadcastq	-0x29c912(%rip), %zmm5 # 0x9d098 <anon.4e66a2221d34052adc0f362485be06ac.728.llvm.17808730978696754556+0x48>
  3399aa: 49 89 c8                     	movq	%rcx, %r8
  3399ad: c5 e9 ef d2                  	vpxor	%xmm2, %xmm2, %xmm2
  3399b1: 66 66 66 66 66 66 2e 0f 1f 84 00 00 00 00 00 	nopw	%cs:(%rax,%rax)
  3399c0: 62 f1 f5 48 d4 f3            	vpaddq	%zmm3, %zmm1, %zmm6
  3399c6: c4 c1 f9 7e cf               	vmovq	%xmm1, %r15
  3399cb: 4d 01 f7                     	addq	%r14, %r15
  3399ce: 62 b1 fe 48 6f 3c fb         	vmovdqu64	(%rbx,%r15,8), %zmm7
  3399d5: 62 31 fe 48 6f 44 fb 01      	vmovdqu64	0x40(%rbx,%r15,8), %zmm8
  3399dd: 62 31 fe 48 6f 0c ff         	vmovdqu64	(%rdi,%r15,8), %zmm9
  3399e4: 62 31 fe 48 6f 54 ff 01      	vmovdqu64	0x40(%rdi,%r15,8), %zmm10
  3399ec: 62 f2 c5 48 29 d4            	vpcmpeqq	%zmm4, %zmm7, %k2
  3399f2: 62 f2 bd 48 29 dc            	vpcmpeqq	%zmm4, %zmm8, %k3
  3399f8: 62 f2 b5 48 29 e4            	vpcmpeqq	%zmm4, %zmm9, %k4
  3399fe: 62 f2 ad 48 29 ec            	vpcmpeqq	%zmm4, %zmm10, %k5
  339a04: c5 ec 45 d4                  	korw	%k4, %k2, %k2
  339a08: c5 e4 45 dd                  	korw	%k5, %k3, %k3
  339a0c: 62 f2 b5 48 37 e7            	vpcmpgtq	%zmm7, %zmm9, %k4
  339a12: 62 d2 ad 48 37 e8            	vpcmpgtq	%zmm8, %zmm10, %k5
  339a18: c5 ec 45 c0                  	korw	%k0, %k2, %k0
  339a1c: c5 e4 45 c9                  	korw	%k1, %k3, %k1
  339a20: 62 f3 c5 cc 25 ff ff         	vpternlogq	$0xff, %zmm7, %zmm7, %zmm7 {%k4} {z} # zmm7 {%k4} {z} = -1
  339a27: 62 f1 c5 48 73 d7 3f         	vpsrlq	$0x3f, %zmm7, %zmm7
  339a2e: 62 53 bd cd 25 c0 ff         	vpternlogq	$0xff, %zmm8, %zmm8, %zmm8 {%k5} {z} # zmm8 {%k5} {z} = -1
  339a35: 62 d1 bd 48 73 d0 3f         	vpsrlq	$0x3f, %zmm8, %zmm8
  339a3c: 62 f2 bd 48 47 f6            	vpsllvq	%zmm6, %zmm8, %zmm6
  339a42: 62 f1 cd 48 eb d2            	vporq	%zmm2, %zmm6, %zmm2
  339a48: 62 f2 c5 48 47 f1            	vpsllvq	%zmm1, %zmm7, %zmm6
  339a4e: 62 f1 cd 48 eb c0            	vporq	%zmm0, %zmm6, %zmm0
  339a54: 62 f1 f5 48 d4 cd            	vpaddq	%zmm5, %zmm1, %zmm1
  339a5a: 49 83 c0 f0                  	addq	$-0x10, %r8
  339a5e: 0f 85 5c ff ff ff            	jne	0x3399c0 <_RINvNtNtCsdxhMBRXnOVY_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0xe30>
  339a64: c5 f4 45 c0                  	korw	%k0, %k1, %k0
  339a68: c5 7b 93 c0                  	kmovd	%k0, %r8d
  339a6c: 45 84 c0                     	testb	%r8b, %r8b
  339a6f: 41 0f 95 c7                  	setne	%r15b
  339a73: 62 f1 ed 48 eb c0            	vporq	%zmm0, %zmm2, %zmm0
  339a79: 62 f3 fd 48 3b c1 01         	vextracti64x4	$0x1, %zmm0, %ymm1
  339a80: 62 f1 fd 48 eb c1            	vporq	%zmm1, %zmm0, %zmm0
  339a86: c4 e3 7d 39 c1 01            	vextracti128	$0x1, %ymm0, %xmm1
  339a8c: c5 f9 eb c1                  	vpor	%xmm1, %xmm0, %xmm0
  339a90: c5 f9 70 c8 ee               	vpshufd	$0xee, %xmm0, %xmm1     # xmm1 = xmm0[2,3,2,3]
  339a95: c5 f9 eb c1                  	vpor	%xmm1, %xmm0, %xmm0
  339a99: c4 c1 f9 7e c4               	vmovq	%xmm0, %r12
  339a9e: 41 39 ca                     	cmpl	%ecx, %r10d
  339aa1: 0f 84 fd 00 00 00            	je	0x339ba4 <_RINvNtNtCsdxhMBRXnOVY_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x1014>
  339aa7: f6 c2 0c                     	testb	$0xc, %dl
  339aaa: 0f 84 ab 00 00 00            	je	0x339b5b <_RINvNtNtCsdxhMBRXnOVY_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0xfcb>
  339ab0: 49 89 c8                     	movq	%rcx, %r8
  339ab3: 89 d1                        	movl	%edx, %ecx
  339ab5: 83 e1 3c                     	andl	$0x3c, %ecx
  339ab8: 41 83 e7 01                  	andl	$0x1, %r15d
  339abc: c4 c1 78 92 c7               	kmovw	%r15d, %k0
  339ac1: c4 c1 f9 6e c4               	vmovq	%r12, %xmm0
  339ac6: c4 c1 f9 6e c8               	vmovq	%r8, %xmm1
  339acb: c4 e2 7d 59 c9               	vpbroadcastq	%xmm1, %ymm1
  339ad0: c5 f5 eb 0d a8 48 d6 ff      	vpor	-0x29b758(%rip), %ymm1, %ymm1 # 0x9e380 <anon.6f50054b7ece9027416a5bb6e70b105d.10.llvm.9732915856748287937+0x80>
  339ad8: 49 29 c8                     	subq	%rcx, %r8
  339adb: c4 e2 7d 59 15 24 2f d6 ff   	vpbroadcastq	-0x29d0dc(%rip), %ymm2 # 0x9ca08 <anon.4e66a2221d34052adc0f362485be06ac.633.llvm.17808730978696754556+0x50>
  339ae4: c4 e2 7d 59 1d 3b 35 d6 ff   	vpbroadcastq	-0x29cac5(%rip), %ymm3 # 0x9d028 <anon.5e4eb8cc45d99afa7f400ea3fba78f81.169.llvm.15372149864199885434>
  339aed: 0f 1f 00                     	nopl	(%rax)
  339af0: c4 e1 f9 7e ca               	vmovq	%xmm1, %rdx
  339af5: 4c 01 f2                     	addq	%r14, %rdx
  339af8: c5 fe 6f 24 d3               	vmovdqu	(%rbx,%rdx,8), %ymm4
  339afd: c5 fe 6f 2c d7               	vmovdqu	(%rdi,%rdx,8), %ymm5
  339b02: 62 f2 dd 48 29 ca            	vpcmpeqq	%zmm2, %zmm4, %k1
  339b08: 62 f2 d5 48 29 d2            	vpcmpeqq	%zmm2, %zmm5, %k2
  339b0e: c5 f4 45 ca                  	korw	%k2, %k1, %k1
  339b12: c5 f4 45 c0                  	korw	%k0, %k1, %k0
  339b16: c4 e2 55 37 e4               	vpcmpgtq	%ymm4, %ymm5, %ymm4
  339b1b: c5 dd 73 d4 3f               	vpsrlq	$0x3f, %ymm4, %ymm4
  339b20: c4 e2 dd 47 e1               	vpsllvq	%ymm1, %ymm4, %ymm4
  339b25: c5 dd eb c0                  	vpor	%ymm0, %ymm4, %ymm0
  339b29: c5 f5 d4 cb                  	vpaddq	%ymm3, %ymm1, %ymm1
  339b2d: 49 83 c0 04                  	addq	$0x4, %r8
  339b31: 75 bd                        	jne	0x339af0 <_RINvNtNtCsdxhMBRXnOVY_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0xf60>
  339b33: c5 fb 93 d0                  	kmovd	%k0, %edx
  339b37: f6 c2 0f                     	testb	$0xf, %dl
  339b3a: 41 0f 95 c7                  	setne	%r15b
  339b3e: c4 e3 7d 39 c1 01            	vextracti128	$0x1, %ymm0, %xmm1
  339b44: c5 f9 eb c1                  	vpor	%xmm1, %xmm0, %xmm0
  339b48: c5 f9 70 c8 ee               	vpshufd	$0xee, %xmm0, %xmm1     # xmm1 = xmm0[2,3,2,3]
  339b4d: c5 f9 eb c1                  	vpor	%xmm1, %xmm0, %xmm0
  339b51: c4 c1 f9 7e c4               	vmovq	%xmm0, %r12
  339b56: 41 39 ca                     	cmpl	%ecx, %r10d
  339b59: 74 49                        	je	0x339ba4 <_RINvNtNtCsdxhMBRXnOVY_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x1014>
  339b5b: 48 89 f2                     	movq	%rsi, %rdx
  339b5e: 48 c1 e2 09                  	shlq	$0x9, %rdx
  339b62: 48 01 d7                     	addq	%rdx, %rdi
  339b65: 48 01 d3                     	addq	%rdx, %rbx
  339b68: 0f 1f 84 00 00 00 00 00      	nopl	(%rax,%rax)
  339b70: 48 8b 14 cb                  	movq	(%rbx,%rcx,8), %rdx
  339b74: 4c 8b 04 cf                  	movq	(%rdi,%rcx,8), %r8
  339b78: 4c 39 da                     	cmpq	%r11, %rdx
  339b7b: 41 0f 94 c6                  	sete	%r14b
  339b7f: 4d 39 d8                     	cmpq	%r11, %r8
  339b82: 41 0f 94 c5                  	sete	%r13b
  339b86: 45 08 f5                     	orb	%r14b, %r13b
  339b89: 45 31 f6                     	xorl	%r14d, %r14d
  339b8c: 4c 39 c2                     	cmpq	%r8, %rdx
  339b8f: 41 0f 9c c6                  	setl	%r14b
  339b93: 45 08 ef                     	orb	%r13b, %r15b
  339b96: 49 d3 e6                     	shlq	%cl, %r14
  339b99: 48 ff c1                     	incq	%rcx
  339b9c: 4d 09 f4                     	orq	%r14, %r12
  339b9f: 49 39 ca                     	cmpq	%rcx, %r10
  339ba2: 75 cc                        	jne	0x339b70 <_RINvNtNtCsdxhMBRXnOVY_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0xfe0>
  339ba4: 41 80 e7 01                  	andb	$0x1, %r15b
  339ba8: 44 88 78 38                  	movb	%r15b, 0x38(%rax)
  339bac: 48 8b 45 c8                  	movq	-0x38(%rbp), %rax
  339bb0: 48 39 c6                     	cmpq	%rax, %rsi
  339bb3: 73 27                        	jae	0x339bdc <_RINvNtNtCsdxhMBRXnOVY_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x104c>
  339bb5: 4d 89 21                     	movq	%r12, (%r9)
  339bb8: 48 83 c4 18                  	addq	$0x18, %rsp
  339bbc: 5b                           	popq	%rbx
  339bbd: 41 5c                        	popq	%r12
  339bbf: 41 5d                        	popq	%r13
  339bc1: 41 5e                        	popq	%r14
  339bc3: 41 5f                        	popq	%r15
  339bc5: 5d                           	popq	%rbp
  339bc6: c5 f8 77                     	vzeroupper
  339bc9: c3                           	retq
  339bca: 48 8d 0d ef 72 fd 00         	leaq	0xfd72ef(%rip), %rcx    # 0x1310ec0 <anon.d23762f469f1c8cdb7907e1b370362a2.2.llvm.9169974397394244368>
  339bd1: 31 ff                        	xorl	%edi, %edi
  339bd3: 4c 89 c2                     	movq	%r8, %rdx
  339bd6: ff 15 1c 44 03 01            	callq	*0x103441c(%rip)        # 0x136dff8 <writev+0x136dff8>
  339bdc: 48 8d 15 c5 72 fd 00         	leaq	0xfd72c5(%rip), %rdx    # 0x1310ea8 <anon.d23762f469f1c8cdb7907e1b370362a2.1.llvm.9169974397394244368>
  339be3: 48 89 f7                     	movq	%rsi, %rdi
  339be6: 48 89 c6                     	movq	%rax, %rsi
  339be9: c5 f8 77                     	vzeroupper
  339bec: ff 15 56 40 03 01            	callq	*0x1034056(%rip)        # 0x136dc48 <writev+0x136dc48>
