
/tmp/row-fn-bool-linux-state-avx2-target/x86_64-unknown-linux-gnu/release/deps/row_fn_bool_retry-3e4c315823f8823b:	file format elf64-x86-64

Disassembly of section .text:

000000000032d250 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_>:
  32d250: 55                           	pushq	%rbp
  32d251: 48 89 e5                     	movq	%rsp, %rbp
  32d254: 41 57                        	pushq	%r15
  32d256: 41 56                        	pushq	%r14
  32d258: 41 55                        	pushq	%r13
  32d25a: 41 54                        	pushq	%r12
  32d25c: 53                           	pushq	%rbx
  32d25d: 48 83 ec 18                  	subq	$0x18, %rsp
  32d261: 49 89 f0                     	movq	%rsi, %r8
  32d264: 48 89 d6                     	movq	%rdx, %rsi
  32d267: 48 c1 ee 06                  	shrq	$0x6, %rsi
  32d26b: 4c 39 c6                     	cmpq	%r8, %rsi
  32d26e: 0f 87 06 10 00 00            	ja	0x32e27a <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x102a>
  32d274: 48 89 c8                     	movq	%rcx, %rax
  32d277: 4c 89 45 c8                  	movq	%r8, -0x38(%rbp)
  32d27b: 41 89 d2                     	movl	%edx, %r10d
  32d27e: 41 83 e2 3f                  	andl	$0x3f, %r10d
  32d282: 49 bb ff ff ff ff ff ff ff 7f	movabsq	$0x7fffffffffffffff, %r11 # imm = 0x7FFFFFFFFFFFFFFF
  32d28c: 4c 8d 0c f7                  	leaq	(%rdi,%rsi,8), %r9
  32d290: 48 85 f6                     	testq	%rsi, %rsi
  32d293: 0f 84 da 06 00 00            	je	0x32d973 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x723>
  32d299: 48 8d 0c f5 00 00 00 00      	leaq	(,%rsi,8), %rcx
  32d2a1: 44 0f b6 68 18               	movzbl	0x18(%rax), %r13d
  32d2a6: 48 8b 58 20                  	movq	0x20(%rax), %rbx
  32d2aa: 4c 8b 78 28                  	movq	0x28(%rax), %r15
  32d2ae: 44 0f b6 40 38               	movzbl	0x38(%rax), %r8d
  32d2b3: 83 38 01                     	cmpl	$0x1, (%rax)
  32d2b6: 0f 85 31 01 00 00            	jne	0x32d3ed <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x19d>
  32d2bc: 4c 8b 70 10                  	movq	0x10(%rax), %r14
  32d2c0: 4d 8b 26                     	movq	(%r14), %r12
  32d2c3: 45 84 ed                     	testb	%r13b, %r13b
  32d2c6: 0f 84 d7 02 00 00            	je	0x32d5a3 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x353>
  32d2cc: 4d 39 dc                     	cmpq	%r11, %r12
  32d2cf: 0f 94 c3                     	sete	%bl
  32d2d2: 4d 8b 37                     	movq	(%r15), %r14
  32d2d5: 4d 39 de                     	cmpq	%r11, %r14
  32d2d8: 41 0f 94 c7                  	sete	%r15b
  32d2dc: 45 31 ed                     	xorl	%r13d, %r13d
  32d2df: 4d 39 f4                     	cmpq	%r14, %r12
  32d2e2: 41 be 01 01 00 00            	movl	$0x101, %r14d           # imm = 0x101
  32d2e8: 4d 0f 4d f5                  	cmovgeq	%r13, %r14
  32d2ec: 45 89 f4                     	movl	%r14d, %r12d
  32d2ef: 41 c1 e4 10                  	shll	$0x10, %r12d
  32d2f3: 41 c1 ec 18                  	shrl	$0x18, %r12d
  32d2f7: 45 89 f5                     	movl	%r14d, %r13d
  32d2fa: 41 c1 ed 08                  	shrl	$0x8, %r13d
  32d2fe: c4 c1 79 6e c6               	vmovd	%r14d, %xmm0
  32d303: c4 c3 79 20 c5 01            	vpinsrb	$0x1, %r13d, %xmm0, %xmm0
  32d309: c4 c3 79 20 c6 02            	vpinsrb	$0x2, %r14d, %xmm0, %xmm0
  32d30f: c4 c3 79 20 c4 03            	vpinsrb	$0x3, %r12d, %xmm0, %xmm0
  32d315: c4 c3 79 20 c6 04            	vpinsrb	$0x4, %r14d, %xmm0, %xmm0
  32d31b: c4 c3 79 20 c5 05            	vpinsrb	$0x5, %r13d, %xmm0, %xmm0
  32d321: c4 c3 79 20 c6 06            	vpinsrb	$0x6, %r14d, %xmm0, %xmm0
  32d327: c4 c3 79 20 c4 07            	vpinsrb	$0x7, %r12d, %xmm0, %xmm0
  32d32d: c4 c3 79 20 c6 08            	vpinsrb	$0x8, %r14d, %xmm0, %xmm0
  32d333: c4 c3 79 20 c5 09            	vpinsrb	$0x9, %r13d, %xmm0, %xmm0
  32d339: c4 c3 79 20 c6 0a            	vpinsrb	$0xa, %r14d, %xmm0, %xmm0
  32d33f: c4 c3 79 20 c4 0b            	vpinsrb	$0xb, %r12d, %xmm0, %xmm0
  32d345: c4 c3 79 20 c6 0c            	vpinsrb	$0xc, %r14d, %xmm0, %xmm0
  32d34b: c4 c3 79 20 c5 0d            	vpinsrb	$0xd, %r13d, %xmm0, %xmm0
  32d351: c4 c3 79 20 c6 0e            	vpinsrb	$0xe, %r14d, %xmm0, %xmm0
  32d357: c4 c3 79 20 c4 0f            	vpinsrb	$0xf, %r12d, %xmm0, %xmm0
  32d35d: 62 f3 fd 48 43 c0 00         	vshufi64x2	$0x0, %zmm0, %zmm0, %zmm0 # zmm0 = zmm0[0,1,0,1,0,1,0,1]
  32d364: 45 08 c7                     	orb	%r8b, %r15b
  32d367: 48 83 c1 f8                  	addq	$-0x8, %rcx
  32d36b: 41 89 c8                     	movl	%ecx, %r8d
  32d36e: 41 f7 d0                     	notl	%r8d
  32d371: 62 f2 7d 48 26 c0            	vptestmb	%zmm0, %zmm0, %k0
  32d377: 41 f6 c0 38                  	testb	$0x38, %r8b
  32d37b: 74 21                        	je	0x32d39e <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x14e>
  32d37d: 41 89 c8                     	movl	%ecx, %r8d
  32d380: 41 c1 e8 03                  	shrl	$0x3, %r8d
  32d384: 41 ff c0                     	incl	%r8d
  32d387: 41 83 e0 07                  	andl	$0x7, %r8d
  32d38b: 0f 1f 44 00 00               	nopl	(%rax,%rax)
  32d390: c4 e1 f8 91 07               	kmovq	%k0, (%rdi)
  32d395: 48 83 c7 08                  	addq	$0x8, %rdi
  32d399: 49 ff c8                     	decq	%r8
  32d39c: 75 f2                        	jne	0x32d390 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x140>
  32d39e: 41 08 df                     	orb	%bl, %r15b
  32d3a1: 48 83 f9 38                  	cmpq	$0x38, %rcx
  32d3a5: 0f 82 c0 05 00 00            	jb	0x32d96b <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x71b>
  32d3ab: 0f 1f 44 00 00               	nopl	(%rax,%rax)
  32d3b0: c4 e1 f8 91 07               	kmovq	%k0, (%rdi)
  32d3b5: c4 e1 f8 91 47 08            	kmovq	%k0, 0x8(%rdi)
  32d3bb: c4 e1 f8 91 47 10            	kmovq	%k0, 0x10(%rdi)
  32d3c1: c4 e1 f8 91 47 18            	kmovq	%k0, 0x18(%rdi)
  32d3c7: c4 e1 f8 91 47 20            	kmovq	%k0, 0x20(%rdi)
  32d3cd: c4 e1 f8 91 47 28            	kmovq	%k0, 0x28(%rdi)
  32d3d3: c4 e1 f8 91 47 30            	kmovq	%k0, 0x30(%rdi)
  32d3d9: c4 e1 f8 91 47 38            	kmovq	%k0, 0x38(%rdi)
  32d3df: 48 83 c7 40                  	addq	$0x40, %rdi
  32d3e3: 4c 39 cf                     	cmpq	%r9, %rdi
  32d3e6: 75 c8                        	jne	0x32d3b0 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x160>
  32d3e8: e9 7e 05 00 00               	jmp	0x32d96b <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x71b>
  32d3ed: 4c 8b 70 08                  	movq	0x8(%rax), %r14
  32d3f1: 45 84 ed                     	testb	%r13b, %r13b
  32d3f4: 0f 84 49 03 00 00            	je	0x32d743 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x4f3>
  32d3fa: 4d 8b 3f                     	movq	(%r15), %r15
  32d3fd: 31 db                        	xorl	%ebx, %ebx
  32d3ff: 4d 39 df                     	cmpq	%r11, %r15
  32d402: 62 d2 fd 48 7c c7            	vpbroadcastq	%r15, %zmm0
  32d408: 41 bf ff 00 00 00            	movl	$0xff, %r15d
  32d40e: 44 0f 45 fb                  	cmovnel	%ebx, %r15d
  32d412: c4 c1 7b 92 c7               	kmovd	%r15d, %k0
  32d417: 49 81 c6 c0 01 00 00         	addq	$0x1c0, %r14            # imm = 0x1C0
  32d41e: 62 f2 fd 48 59 0d 40 e8 d6 ff	vpbroadcastq	-0x2917c0(%rip), %zmm1 # 0x9bc68 <anon.ad39dab31d93760b58da730830a5ab31.633.llvm.6451718222114420760+0x50>
  32d428: 62 f1 fd 48 6f 15 8e 39 d7 ff	vmovdqa64	-0x28c672(%rip), %zmm2 # 0xa0dc0 <anon.ad39dab31d93760b58da730830a5ab31.2427.llvm.6451718222114420760+0x84>
  32d432: 62 f1 fd 48 6f 1d c4 39 d7 ff	vmovdqa64	-0x28c63c(%rip), %zmm3 # 0xa0e00 <anon.ad39dab31d93760b58da730830a5ab31.2427.llvm.6451718222114420760+0xc4>
  32d43c: 62 f1 fd 48 6f 25 fa 39 d7 ff	vmovdqa64	-0x28c606(%rip), %zmm4 # 0xa0e40 <anon.ad39dab31d93760b58da730830a5ab31.2427.llvm.6451718222114420760+0x104>
  32d446: 45 89 c7                     	movl	%r8d, %r15d
  32d449: 0f 1f 80 00 00 00 00         	nopl	(%rax)
  32d450: 41 83 e7 01                  	andl	$0x1, %r15d
  32d454: c4 c1 78 92 e7               	kmovw	%r15d, %k4
  32d459: 62 d1 fe 48 6f 6e f9         	vmovdqu64	-0x1c0(%r14), %zmm5
  32d460: 62 d1 fe 48 6f 76 fa         	vmovdqu64	-0x180(%r14), %zmm6
  32d467: 62 d1 fe 48 6f 7e fb         	vmovdqu64	-0x140(%r14), %zmm7
  32d46e: 62 51 fe 48 6f 46 fc         	vmovdqu64	-0x100(%r14), %zmm8
  32d475: 62 f2 d5 48 29 e9            	vpcmpeqq	%zmm1, %zmm5, %k5
  32d47b: 62 f2 cd 48 29 d9            	vpcmpeqq	%zmm1, %zmm6, %k3
  32d481: 62 f2 c5 48 29 d1            	vpcmpeqq	%zmm1, %zmm7, %k2
  32d487: 62 f2 bd 48 29 c9            	vpcmpeqq	%zmm1, %zmm8, %k1
  32d48d: c5 d4 45 e4                  	korw	%k4, %k5, %k4
  32d491: 62 f2 fd 48 37 ed            	vpcmpgtq	%zmm5, %zmm0, %k5
  32d497: 62 f1 7f cd 6f ea            	vmovdqu8	%zmm2, %zmm5 {%k5} {z}
  32d49d: 62 f2 fd 48 37 ee            	vpcmpgtq	%zmm6, %zmm0, %k5
  32d4a3: 62 f1 7f cd 6f f2            	vmovdqu8	%zmm2, %zmm6 {%k5} {z}
  32d4a9: 62 f2 fd 48 37 ef            	vpcmpgtq	%zmm7, %zmm0, %k5
  32d4af: 62 f1 7f cd 6f fa            	vmovdqu8	%zmm2, %zmm7 {%k5} {z}
  32d4b5: 62 d2 fd 48 37 e8            	vpcmpgtq	%zmm8, %zmm0, %k5
  32d4bb: 62 71 7f cd 6f c2            	vmovdqu8	%zmm2, %zmm8 {%k5} {z}
  32d4c1: 62 51 fe 48 6f 4e fd         	vmovdqu64	-0xc0(%r14), %zmm9
  32d4c8: 62 51 fe 48 6f 56 fe         	vmovdqu64	-0x80(%r14), %zmm10
  32d4cf: 62 51 fe 48 6f 5e ff         	vmovdqu64	-0x40(%r14), %zmm11
  32d4d6: 62 51 fe 48 6f 26            	vmovdqu64	(%r14), %zmm12
  32d4dc: 62 f2 b5 48 29 e9            	vpcmpeqq	%zmm1, %zmm9, %k5
  32d4e2: c5 dc 45 e5                  	korw	%k5, %k4, %k4
  32d4e6: 62 f2 ad 48 29 e9            	vpcmpeqq	%zmm1, %zmm10, %k5
  32d4ec: c5 e4 45 dd                  	korw	%k5, %k3, %k3
  32d4f0: 62 f2 a5 48 29 e9            	vpcmpeqq	%zmm1, %zmm11, %k5
  32d4f6: c5 ec 45 d5                  	korw	%k5, %k2, %k2
  32d4fa: 62 f2 9d 48 29 e9            	vpcmpeqq	%zmm1, %zmm12, %k5
  32d500: c5 f4 45 cd                  	korw	%k5, %k1, %k1
  32d504: 62 d2 fd 48 37 e9            	vpcmpgtq	%zmm9, %zmm0, %k5
  32d50a: 62 71 7f cd 6f ca            	vmovdqu8	%zmm2, %zmm9 {%k5} {z}
  32d510: 62 d2 fd 48 37 ea            	vpcmpgtq	%zmm10, %zmm0, %k5
  32d516: 62 71 7f cd 6f d2            	vmovdqu8	%zmm2, %zmm10 {%k5} {z}
  32d51c: 62 d2 fd 48 37 eb            	vpcmpgtq	%zmm11, %zmm0, %k5
  32d522: 62 71 7f cd 6f da            	vmovdqu8	%zmm2, %zmm11 {%k5} {z}
  32d528: 62 d2 fd 48 37 ec            	vpcmpgtq	%zmm12, %zmm0, %k5
  32d52e: 62 71 7f cd 6f e2            	vmovdqu8	%zmm2, %zmm12 {%k5} {z}
  32d534: c5 dc 45 e0                  	korw	%k0, %k4, %k4
  32d538: c5 e4 45 dc                  	korw	%k4, %k3, %k3
  32d53c: c5 ec 45 d3                  	korw	%k3, %k2, %k2
  32d540: c5 f4 45 ca                  	korw	%k2, %k1, %k1
  32d544: c5 d1 6c ee                  	vpunpcklqdq	%xmm6, %xmm5, %xmm5 # xmm5 = xmm5[0],xmm6[0]
  32d548: c4 e3 55 38 ef 01            	vinserti128	$0x1, %xmm7, %ymm5, %ymm5
  32d54e: c4 c2 7d 59 f0               	vpbroadcastq	%xmm8, %ymm6
  32d553: c4 e3 55 02 ee c0            	vpblendd	$0xc0, %ymm6, %ymm5, %ymm5 # ymm5 = ymm5[0,1,2,3,4,5],ymm6[6,7]
  32d559: 62 d3 d5 48 3a e9 01         	vinserti64x4	$0x1, %ymm9, %zmm5, %zmm5
  32d560: 62 d2 e5 48 7e ea            	vpermt2q	%zmm10, %zmm3, %zmm5
  32d566: 62 d3 55 48 38 eb 03         	vinserti32x4	$0x3, %xmm11, %zmm5, %zmm5
  32d56d: 62 d2 dd 48 7e ec            	vpermt2q	%zmm12, %zmm4, %zmm5
  32d573: c5 7b 93 c1                  	kmovd	%k1, %r8d
  32d577: 62 f2 55 48 26 cd            	vptestmb	%zmm5, %zmm5, %k1
  32d57d: c4 e1 f8 91 0c 1f            	kmovq	%k1, (%rdi,%rbx)
  32d583: 45 84 c0                     	testb	%r8b, %r8b
  32d586: 41 0f 95 c7                  	setne	%r15b
  32d58a: 48 83 c3 08                  	addq	$0x8, %rbx
  32d58e: 49 81 c6 00 02 00 00         	addq	$0x200, %r14            # imm = 0x200
  32d595: 48 39 d9                     	cmpq	%rbx, %rcx
  32d598: 0f 85 b2 fe ff ff            	jne	0x32d450 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x200>
  32d59e: e9 c8 03 00 00               	jmp	0x32d96b <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x71b>
  32d5a3: 45 31 f6                     	xorl	%r14d, %r14d
  32d5a6: 4d 39 dc                     	cmpq	%r11, %r12
  32d5a9: 41 bf ff 00 00 00            	movl	$0xff, %r15d
  32d5af: 45 0f 45 fe                  	cmovnel	%r14d, %r15d
  32d5b3: c4 c1 7b 92 c7               	kmovd	%r15d, %k0
  32d5b8: 62 d2 fd 48 7c c4            	vpbroadcastq	%r12, %zmm0
  32d5be: 48 81 c3 c0 01 00 00         	addq	$0x1c0, %rbx            # imm = 0x1C0
  32d5c5: 62 f2 fd 48 59 0d 99 e6 d6 ff	vpbroadcastq	-0x291967(%rip), %zmm1 # 0x9bc68 <anon.ad39dab31d93760b58da730830a5ab31.633.llvm.6451718222114420760+0x50>
  32d5cf: 62 f1 fd 48 6f 15 e7 37 d7 ff	vmovdqa64	-0x28c819(%rip), %zmm2 # 0xa0dc0 <anon.ad39dab31d93760b58da730830a5ab31.2427.llvm.6451718222114420760+0x84>
  32d5d9: 62 f1 fd 48 6f 1d 1d 38 d7 ff	vmovdqa64	-0x28c7e3(%rip), %zmm3 # 0xa0e00 <anon.ad39dab31d93760b58da730830a5ab31.2427.llvm.6451718222114420760+0xc4>
  32d5e3: 62 f1 fd 48 6f 25 53 38 d7 ff	vmovdqa64	-0x28c7ad(%rip), %zmm4 # 0xa0e40 <anon.ad39dab31d93760b58da730830a5ab31.2427.llvm.6451718222114420760+0x104>
  32d5ed: 45 89 c7                     	movl	%r8d, %r15d
  32d5f0: 41 83 e7 01                  	andl	$0x1, %r15d
  32d5f4: c4 c1 78 92 e7               	kmovw	%r15d, %k4
  32d5f9: 62 f1 fe 48 6f 6b f9         	vmovdqu64	-0x1c0(%rbx), %zmm5
  32d600: 62 f1 fe 48 6f 73 fa         	vmovdqu64	-0x180(%rbx), %zmm6
  32d607: 62 f1 fe 48 6f 7b fb         	vmovdqu64	-0x140(%rbx), %zmm7
  32d60e: 62 71 fe 48 6f 43 fc         	vmovdqu64	-0x100(%rbx), %zmm8
  32d615: 62 f2 d5 48 29 e9            	vpcmpeqq	%zmm1, %zmm5, %k5
  32d61b: 62 f2 cd 48 29 d9            	vpcmpeqq	%zmm1, %zmm6, %k3
  32d621: 62 f2 c5 48 29 d1            	vpcmpeqq	%zmm1, %zmm7, %k2
  32d627: 62 f2 bd 48 29 c9            	vpcmpeqq	%zmm1, %zmm8, %k1
  32d62d: c5 d4 45 e4                  	korw	%k4, %k5, %k4
  32d631: 62 f2 d5 48 37 e8            	vpcmpgtq	%zmm0, %zmm5, %k5
  32d637: 62 f1 7f cd 6f ea            	vmovdqu8	%zmm2, %zmm5 {%k5} {z}
  32d63d: 62 f2 cd 48 37 e8            	vpcmpgtq	%zmm0, %zmm6, %k5
  32d643: 62 f1 7f cd 6f f2            	vmovdqu8	%zmm2, %zmm6 {%k5} {z}
  32d649: 62 f2 c5 48 37 e8            	vpcmpgtq	%zmm0, %zmm7, %k5
  32d64f: 62 f1 7f cd 6f fa            	vmovdqu8	%zmm2, %zmm7 {%k5} {z}
  32d655: 62 f2 bd 48 37 e8            	vpcmpgtq	%zmm0, %zmm8, %k5
  32d65b: 62 71 7f cd 6f c2            	vmovdqu8	%zmm2, %zmm8 {%k5} {z}
  32d661: 62 71 fe 48 6f 4b fd         	vmovdqu64	-0xc0(%rbx), %zmm9
  32d668: 62 71 fe 48 6f 53 fe         	vmovdqu64	-0x80(%rbx), %zmm10
  32d66f: 62 71 fe 48 6f 5b ff         	vmovdqu64	-0x40(%rbx), %zmm11
  32d676: 62 71 fe 48 6f 23            	vmovdqu64	(%rbx), %zmm12
  32d67c: 62 f2 b5 48 29 e9            	vpcmpeqq	%zmm1, %zmm9, %k5
  32d682: c5 dc 45 e5                  	korw	%k5, %k4, %k4
  32d686: 62 f2 ad 48 29 e9            	vpcmpeqq	%zmm1, %zmm10, %k5
  32d68c: c5 e4 45 dd                  	korw	%k5, %k3, %k3
  32d690: 62 f2 a5 48 29 e9            	vpcmpeqq	%zmm1, %zmm11, %k5
  32d696: c5 ec 45 d5                  	korw	%k5, %k2, %k2
  32d69a: 62 f2 9d 48 29 e9            	vpcmpeqq	%zmm1, %zmm12, %k5
  32d6a0: c5 f4 45 cd                  	korw	%k5, %k1, %k1
  32d6a4: 62 f2 b5 48 37 e8            	vpcmpgtq	%zmm0, %zmm9, %k5
  32d6aa: 62 71 7f cd 6f ca            	vmovdqu8	%zmm2, %zmm9 {%k5} {z}
  32d6b0: 62 f2 ad 48 37 e8            	vpcmpgtq	%zmm0, %zmm10, %k5
  32d6b6: 62 71 7f cd 6f d2            	vmovdqu8	%zmm2, %zmm10 {%k5} {z}
  32d6bc: 62 f2 a5 48 37 e8            	vpcmpgtq	%zmm0, %zmm11, %k5
  32d6c2: 62 71 7f cd 6f da            	vmovdqu8	%zmm2, %zmm11 {%k5} {z}
  32d6c8: 62 f2 9d 48 37 e8            	vpcmpgtq	%zmm0, %zmm12, %k5
  32d6ce: 62 71 7f cd 6f e2            	vmovdqu8	%zmm2, %zmm12 {%k5} {z}
  32d6d4: c5 dc 45 e0                  	korw	%k0, %k4, %k4
  32d6d8: c5 e4 45 dc                  	korw	%k4, %k3, %k3
  32d6dc: c5 ec 45 d3                  	korw	%k3, %k2, %k2
  32d6e0: c5 f4 45 ca                  	korw	%k2, %k1, %k1
  32d6e4: c5 d1 6c ee                  	vpunpcklqdq	%xmm6, %xmm5, %xmm5 # xmm5 = xmm5[0],xmm6[0]
  32d6e8: c4 e3 55 38 ef 01            	vinserti128	$0x1, %xmm7, %ymm5, %ymm5
  32d6ee: c4 c2 7d 59 f0               	vpbroadcastq	%xmm8, %ymm6
  32d6f3: c4 e3 55 02 ee c0            	vpblendd	$0xc0, %ymm6, %ymm5, %ymm5 # ymm5 = ymm5[0,1,2,3,4,5],ymm6[6,7]
  32d6f9: 62 d3 d5 48 3a e9 01         	vinserti64x4	$0x1, %ymm9, %zmm5, %zmm5
  32d700: 62 d2 e5 48 7e ea            	vpermt2q	%zmm10, %zmm3, %zmm5
  32d706: 62 d3 55 48 38 eb 03         	vinserti32x4	$0x3, %xmm11, %zmm5, %zmm5
  32d70d: 62 d2 dd 48 7e ec            	vpermt2q	%zmm12, %zmm4, %zmm5
  32d713: c5 7b 93 c1                  	kmovd	%k1, %r8d
  32d717: 62 f2 55 48 26 cd            	vptestmb	%zmm5, %zmm5, %k1
  32d71d: c4 a1 f8 91 0c 37            	kmovq	%k1, (%rdi,%r14)
  32d723: 45 84 c0                     	testb	%r8b, %r8b
  32d726: 41 0f 95 c7                  	setne	%r15b
  32d72a: 49 83 c6 08                  	addq	$0x8, %r14
  32d72e: 48 81 c3 00 02 00 00         	addq	$0x200, %rbx            # imm = 0x200
  32d735: 4c 39 f1                     	cmpq	%r14, %rcx
  32d738: 0f 85 b2 fe ff ff            	jne	0x32d5f0 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x3a0>
  32d73e: e9 28 02 00 00               	jmp	0x32d96b <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x71b>
  32d743: 41 bc c0 01 00 00            	movl	$0x1c0, %r12d           # imm = 0x1C0
  32d749: 45 31 ed                     	xorl	%r13d, %r13d
  32d74c: 62 f2 fd 48 59 05 12 e5 d6 ff	vpbroadcastq	-0x291aee(%rip), %zmm0 # 0x9bc68 <anon.ad39dab31d93760b58da730830a5ab31.633.llvm.6451718222114420760+0x50>
  32d756: 62 f1 fd 48 6f 0d 60 36 d7 ff	vmovdqa64	-0x28c9a0(%rip), %zmm1 # 0xa0dc0 <anon.ad39dab31d93760b58da730830a5ab31.2427.llvm.6451718222114420760+0x84>
  32d760: 62 f1 fd 48 6f 15 96 36 d7 ff	vmovdqa64	-0x28c96a(%rip), %zmm2 # 0xa0e00 <anon.ad39dab31d93760b58da730830a5ab31.2427.llvm.6451718222114420760+0xc4>
  32d76a: 62 f1 fd 48 6f 1d cc 36 d7 ff	vmovdqa64	-0x28c934(%rip), %zmm3 # 0xa0e40 <anon.ad39dab31d93760b58da730830a5ab31.2427.llvm.6451718222114420760+0x104>
  32d774: 45 89 c7                     	movl	%r8d, %r15d
  32d777: 66 0f 1f 84 00 00 00 00 00   	nopw	(%rax,%rax)
  32d780: 41 83 e7 01                  	andl	$0x1, %r15d
  32d784: c4 c1 78 92 df               	kmovw	%r15d, %k3
  32d789: 62 91 fe 48 6f 64 26 f9      	vmovdqu64	-0x1c0(%r14,%r12), %zmm4
  32d791: 62 91 fe 48 6f 6c 26 fa      	vmovdqu64	-0x180(%r14,%r12), %zmm5
  32d799: 62 91 fe 48 6f 74 26 fb      	vmovdqu64	-0x140(%r14,%r12), %zmm6
  32d7a1: 62 91 fe 48 6f 7c 26 fc      	vmovdqu64	-0x100(%r14,%r12), %zmm7
  32d7a9: 62 31 fe 48 6f 44 23 f9      	vmovdqu64	-0x1c0(%rbx,%r12), %zmm8
  32d7b1: 62 31 fe 48 6f 4c 23 fa      	vmovdqu64	-0x180(%rbx,%r12), %zmm9
  32d7b9: 62 31 fe 48 6f 54 23 fb      	vmovdqu64	-0x140(%rbx,%r12), %zmm10
  32d7c1: 62 31 fe 48 6f 5c 23 fc      	vmovdqu64	-0x100(%rbx,%r12), %zmm11
  32d7c9: 62 f2 dd 48 29 c0            	vpcmpeqq	%zmm0, %zmm4, %k0
  32d7cf: 62 f2 d5 48 29 c8            	vpcmpeqq	%zmm0, %zmm5, %k1
  32d7d5: 62 f2 cd 48 29 d0            	vpcmpeqq	%zmm0, %zmm6, %k2
  32d7db: 62 f2 c5 48 29 e0            	vpcmpeqq	%zmm0, %zmm7, %k4
  32d7e1: 62 f2 bd 48 29 e8            	vpcmpeqq	%zmm0, %zmm8, %k5
  32d7e7: 62 f2 b5 48 29 f0            	vpcmpeqq	%zmm0, %zmm9, %k6
  32d7ed: 62 f2 ad 48 29 f8            	vpcmpeqq	%zmm0, %zmm10, %k7
  32d7f3: c5 fc 45 ed                  	korw	%k5, %k0, %k5
  32d7f7: 62 f2 a5 48 29 c0            	vpcmpeqq	%zmm0, %zmm11, %k0
  32d7fd: c5 f4 45 f6                  	korw	%k6, %k1, %k6
  32d801: c5 ec 45 cf                  	korw	%k7, %k2, %k1
  32d805: c5 f8 91 4d d6               	kmovw	%k1, -0x2a(%rbp)
  32d80a: c5 dc 45 d0                  	korw	%k0, %k4, %k2
  32d80e: 62 f2 bd 48 37 e4            	vpcmpgtq	%zmm4, %zmm8, %k4
  32d814: 62 f2 b5 48 37 fd            	vpcmpgtq	%zmm5, %zmm9, %k7
  32d81a: 62 f2 ad 48 37 ce            	vpcmpgtq	%zmm6, %zmm10, %k1
  32d820: c5 d4 45 db                  	korw	%k3, %k5, %k3
  32d824: 62 f2 a5 48 37 ef            	vpcmpgtq	%zmm7, %zmm11, %k5
  32d82a: 62 f1 7f cc 6f e1            	vmovdqu8	%zmm1, %zmm4 {%k4} {z}
  32d830: 62 f1 7f cf 6f e9            	vmovdqu8	%zmm1, %zmm5 {%k7} {z}
  32d836: 62 f1 7f c9 6f f1            	vmovdqu8	%zmm1, %zmm6 {%k1} {z}
  32d83c: 62 f1 7f cd 6f f9            	vmovdqu8	%zmm1, %zmm7 {%k5} {z}
  32d842: 62 11 fe 48 6f 44 26 fd      	vmovdqu64	-0xc0(%r14,%r12), %zmm8
  32d84a: 62 11 fe 48 6f 4c 26 fe      	vmovdqu64	-0x80(%r14,%r12), %zmm9
  32d852: 62 11 fe 48 6f 54 26 ff      	vmovdqu64	-0x40(%r14,%r12), %zmm10
  32d85a: 62 11 fe 48 6f 1c 26         	vmovdqu64	(%r14,%r12), %zmm11
  32d861: 62 31 fe 48 6f 64 23 fd      	vmovdqu64	-0xc0(%rbx,%r12), %zmm12
  32d869: 62 31 fe 48 6f 6c 23 fe      	vmovdqu64	-0x80(%rbx,%r12), %zmm13
  32d871: 62 31 fe 48 6f 74 23 ff      	vmovdqu64	-0x40(%rbx,%r12), %zmm14
  32d879: 62 31 fe 48 6f 3c 23         	vmovdqu64	(%rbx,%r12), %zmm15
  32d880: 62 f2 bd 48 29 c0            	vpcmpeqq	%zmm0, %zmm8, %k0
  32d886: 62 f2 b5 48 29 c8            	vpcmpeqq	%zmm0, %zmm9, %k1
  32d88c: 62 f2 ad 48 29 e0            	vpcmpeqq	%zmm0, %zmm10, %k4
  32d892: 62 f2 9d 48 29 e8            	vpcmpeqq	%zmm0, %zmm12, %k5
  32d898: c5 fc 45 c5                  	korw	%k5, %k0, %k0
  32d89c: 62 f2 95 48 29 e8            	vpcmpeqq	%zmm0, %zmm13, %k5
  32d8a2: c5 f4 45 cd                  	korw	%k5, %k1, %k1
  32d8a6: 62 f2 8d 48 29 e8            	vpcmpeqq	%zmm0, %zmm14, %k5
  32d8ac: c5 dc 45 e5                  	korw	%k5, %k4, %k4
  32d8b0: 62 f2 a5 48 29 e8            	vpcmpeqq	%zmm0, %zmm11, %k5
  32d8b6: 62 f2 85 48 29 f8            	vpcmpeqq	%zmm0, %zmm15, %k7
  32d8bc: c5 d4 45 ef                  	korw	%k7, %k5, %k5
  32d8c0: c5 fc 45 db                  	korw	%k3, %k0, %k3
  32d8c4: c5 f4 45 c6                  	korw	%k6, %k1, %k0
  32d8c8: c5 f8 90 4d d6               	kmovw	-0x2a(%rbp), %k1
  32d8cd: c5 dc 45 c9                  	korw	%k1, %k4, %k1
  32d8d1: c5 d4 45 d2                  	korw	%k2, %k5, %k2
  32d8d5: 62 d2 9d 48 37 e0            	vpcmpgtq	%zmm8, %zmm12, %k4
  32d8db: 62 71 7f cc 6f c1            	vmovdqu8	%zmm1, %zmm8 {%k4} {z}
  32d8e1: 62 d2 95 48 37 e1            	vpcmpgtq	%zmm9, %zmm13, %k4
  32d8e7: 62 71 7f cc 6f c9            	vmovdqu8	%zmm1, %zmm9 {%k4} {z}
  32d8ed: 62 d2 8d 48 37 e2            	vpcmpgtq	%zmm10, %zmm14, %k4
  32d8f3: 62 71 7f cc 6f d1            	vmovdqu8	%zmm1, %zmm10 {%k4} {z}
  32d8f9: 62 d2 85 48 37 e3            	vpcmpgtq	%zmm11, %zmm15, %k4
  32d8ff: 62 71 7f cc 6f d9            	vmovdqu8	%zmm1, %zmm11 {%k4} {z}
  32d905: c5 fc 45 c3                  	korw	%k3, %k0, %k0
  32d909: c5 f4 45 c0                  	korw	%k0, %k1, %k0
  32d90d: c5 ec 45 c0                  	korw	%k0, %k2, %k0
  32d911: c5 d9 6c e5                  	vpunpcklqdq	%xmm5, %xmm4, %xmm4 # xmm4 = xmm4[0],xmm5[0]
  32d915: c4 e3 5d 38 e6 01            	vinserti128	$0x1, %xmm6, %ymm4, %ymm4
  32d91b: c4 e2 7d 59 ef               	vpbroadcastq	%xmm7, %ymm5
  32d920: c4 e3 5d 02 e5 c0            	vpblendd	$0xc0, %ymm5, %ymm4, %ymm4 # ymm4 = ymm4[0,1,2,3,4,5],ymm5[6,7]
  32d926: 62 d3 dd 48 3a e0 01         	vinserti64x4	$0x1, %ymm8, %zmm4, %zmm4
  32d92d: 62 d2 ed 48 7e e1            	vpermt2q	%zmm9, %zmm2, %zmm4
  32d933: 62 d3 5d 48 38 e2 03         	vinserti32x4	$0x3, %xmm10, %zmm4, %zmm4
  32d93a: 62 d2 e5 48 7e e3            	vpermt2q	%zmm11, %zmm3, %zmm4
  32d940: c5 7b 93 c0                  	kmovd	%k0, %r8d
  32d944: 62 f2 5d 48 26 c4            	vptestmb	%zmm4, %zmm4, %k0
  32d94a: c4 a1 f8 91 04 2f            	kmovq	%k0, (%rdi,%r13)
  32d950: 45 84 c0                     	testb	%r8b, %r8b
  32d953: 41 0f 95 c7                  	setne	%r15b
  32d957: 49 83 c5 08                  	addq	$0x8, %r13
  32d95b: 49 81 c4 00 02 00 00         	addq	$0x200, %r12            # imm = 0x200
  32d962: 4c 39 e9                     	cmpq	%r13, %rcx
  32d965: 0f 85 15 fe ff ff            	jne	0x32d780 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x530>
  32d96b: 41 80 e7 01                  	andb	$0x1, %r15b
  32d96f: 44 88 78 38                  	movb	%r15b, 0x38(%rax)
  32d973: 4d 85 d2                     	testq	%r10, %r10
  32d976: 0f 84 ec 08 00 00            	je	0x32e268 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x1018>
  32d97c: 48 89 d7                     	movq	%rdx, %rdi
  32d97f: 48 83 e7 c0                  	andq	$-0x40, %rdi
  32d983: 44 0f b6 70 18               	movzbl	0x18(%rax), %r14d
  32d988: 44 0f b6 40 38               	movzbl	0x38(%rax), %r8d
  32d98d: 48 8b 58 20                  	movq	0x20(%rax), %rbx
  32d991: 48 8b 48 28                  	movq	0x28(%rax), %rcx
  32d995: 83 38 01                     	cmpl	$0x1, (%rax)
  32d998: 75 3b                        	jne	0x32d9d5 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x785>
  32d99a: 4c 8b 78 10                  	movq	0x10(%rax), %r15
  32d99e: 4d 8b 3f                     	movq	(%r15), %r15
  32d9a1: 45 84 f6                     	testb	%r14b, %r14b
  32d9a4: 74 4e                        	je	0x32d9f4 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x7a4>
  32d9a6: 4d 39 df                     	cmpq	%r11, %r15
  32d9a9: 0f 94 c3                     	sete	%bl
  32d9ac: 48 8b 09                     	movq	(%rcx), %rcx
  32d9af: 4c 39 d9                     	cmpq	%r11, %rcx
  32d9b2: 41 0f 94 c6                  	sete	%r14b
  32d9b6: 31 ff                        	xorl	%edi, %edi
  32d9b8: 49 39 cf                     	cmpq	%rcx, %r15
  32d9bb: 40 0f 9c c7                  	setl	%dil
  32d9bf: 45 08 c6                     	orb	%r8b, %r14b
  32d9c2: 41 08 de                     	orb	%bl, %r14b
  32d9c5: 41 83 fa 04                  	cmpl	$0x4, %r10d
  32d9c9: 73 4f                        	jae	0x32da1a <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x7ca>
  32d9cb: 45 31 e4                     	xorl	%r12d, %r12d
  32d9ce: 31 c9                        	xorl	%ecx, %ecx
  32d9d0: e9 bb 01 00 00               	jmp	0x32db90 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x940>
  32d9d5: 4c 8b 78 08                  	movq	0x8(%rax), %r15
  32d9d9: 45 84 f6                     	testb	%r14b, %r14b
  32d9dc: 74 29                        	je	0x32da07 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x7b7>
  32d9de: 48 8b 19                     	movq	(%rcx), %rbx
  32d9e1: 41 83 fa 04                  	cmpl	$0x4, %r10d
  32d9e5: 73 43                        	jae	0x32da2a <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x7da>
  32d9e7: 45 31 e4                     	xorl	%r12d, %r12d
  32d9ea: 45 89 c6                     	movl	%r8d, %r14d
  32d9ed: 31 c9                        	xorl	%ecx, %ecx
  32d9ef: e9 ae 03 00 00               	jmp	0x32dda2 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0xb52>
  32d9f4: 41 83 fa 04                  	cmpl	$0x4, %r10d
  32d9f8: 73 4a                        	jae	0x32da44 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x7f4>
  32d9fa: 45 31 e4                     	xorl	%r12d, %r12d
  32d9fd: 45 89 c6                     	movl	%r8d, %r14d
  32da00: 31 c9                        	xorl	%ecx, %ecx
  32da02: e9 db 05 00 00               	jmp	0x32dfe2 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0xd92>
  32da07: 41 83 fa 04                  	cmpl	$0x4, %r10d
  32da0b: 73 51                        	jae	0x32da5e <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x80e>
  32da0d: 45 31 e4                     	xorl	%r12d, %r12d
  32da10: 45 89 c6                     	movl	%r8d, %r14d
  32da13: 31 c9                        	xorl	%ecx, %ecx
  32da15: e9 f2 07 00 00               	jmp	0x32e20c <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0xfbc>
  32da1a: 41 83 fa 10                  	cmpl	$0x10, %r10d
  32da1e: 73 55                        	jae	0x32da75 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x825>
  32da20: 31 c9                        	xorl	%ecx, %ecx
  32da22: 45 31 e4                     	xorl	%r12d, %r12d
  32da25: e9 e9 00 00 00               	jmp	0x32db13 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x8c3>
  32da2a: 45 31 ed                     	xorl	%r13d, %r13d
  32da2d: 41 83 fa 10                  	cmpl	$0x10, %r10d
  32da31: 0f 83 6f 01 00 00            	jae	0x32dba6 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x956>
  32da37: 31 c9                        	xorl	%ecx, %ecx
  32da39: 45 89 c6                     	movl	%r8d, %r14d
  32da3c: 45 31 e4                     	xorl	%r12d, %r12d
  32da3f: e9 9e 02 00 00               	jmp	0x32dce2 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0xa92>
  32da44: 45 31 ed                     	xorl	%r13d, %r13d
  32da47: 41 83 fa 10                  	cmpl	$0x10, %r10d
  32da4b: 0f 83 9b 03 00 00            	jae	0x32ddec <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0xb9c>
  32da51: 31 c9                        	xorl	%ecx, %ecx
  32da53: 45 89 c6                     	movl	%r8d, %r14d
  32da56: 45 31 e4                     	xorl	%r12d, %r12d
  32da59: e9 c4 04 00 00               	jmp	0x32df22 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0xcd2>
  32da5e: 41 83 fa 10                  	cmpl	$0x10, %r10d
  32da62: 0f 83 c3 05 00 00            	jae	0x32e02b <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0xddb>
  32da68: 31 c9                        	xorl	%ecx, %ecx
  32da6a: 45 89 c6                     	movl	%r8d, %r14d
  32da6d: 45 31 e4                     	xorl	%r12d, %r12d
  32da70: e9 eb 06 00 00               	jmp	0x32e160 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0xf10>
  32da75: 89 d1                        	movl	%edx, %ecx
  32da77: 83 e1 30                     	andl	$0x30, %ecx
  32da7a: 62 f2 fd 48 7c c7            	vpbroadcastq	%rdi, %zmm0
  32da80: 62 f1 fd 48 6f 15 f6 33 d7 ff	vmovdqa64	-0x28cc0a(%rip), %zmm2 # 0xa0e80 <anon.ad39dab31d93760b58da730830a5ab31.2427.llvm.6451718222114420760+0x144>
  32da8a: c5 f1 ef c9                  	vpxor	%xmm1, %xmm1, %xmm1
  32da8e: 62 f2 fd 48 59 1d 70 dc d6 ff	vpbroadcastq	-0x292390(%rip), %zmm3 # 0x9b708 <anon.5278ce255300ec7b361bbf944ee0b803.25.llvm.754055615369681222>
  32da98: 62 f2 fd 48 59 25 26 e8 d6 ff	vpbroadcastq	-0x2917da(%rip), %zmm4 # 0x9c2c8 <anon.ad39dab31d93760b58da730830a5ab31.728.llvm.6451718222114420760+0x40>
  32daa2: 49 89 c8                     	movq	%rcx, %r8
  32daa5: c5 d1 ef ed                  	vpxor	%xmm5, %xmm5, %xmm5
  32daa9: 0f 1f 80 00 00 00 00         	nopl	(%rax)
  32dab0: 62 f1 ed 48 d4 f3            	vpaddq	%zmm3, %zmm2, %zmm6
  32dab6: 62 f2 fd 48 47 fa            	vpsllvq	%zmm2, %zmm0, %zmm7
  32dabc: 62 f1 c5 48 eb c9            	vporq	%zmm1, %zmm7, %zmm1
  32dac2: 62 f2 fd 48 47 f6            	vpsllvq	%zmm6, %zmm0, %zmm6
  32dac8: 62 f1 cd 48 eb ed            	vporq	%zmm5, %zmm6, %zmm5
  32dace: 62 f1 ed 48 d4 d4            	vpaddq	%zmm4, %zmm2, %zmm2
  32dad4: 49 83 c0 f0                  	addq	$-0x10, %r8
  32dad8: 75 d6                        	jne	0x32dab0 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x860>
  32dada: 62 f1 d5 48 eb c1            	vporq	%zmm1, %zmm5, %zmm0
  32dae0: 62 f3 fd 48 3b c1 01         	vextracti64x4	$0x1, %zmm0, %ymm1
  32dae7: 62 f1 fd 48 eb c1            	vporq	%zmm1, %zmm0, %zmm0
  32daed: c4 e3 7d 39 c1 01            	vextracti128	$0x1, %ymm0, %xmm1
  32daf3: c5 f9 eb c1                  	vpor	%xmm1, %xmm0, %xmm0
  32daf7: c5 f9 70 c8 ee               	vpshufd	$0xee, %xmm0, %xmm1     # xmm1 = xmm0[2,3,2,3]
  32dafc: c5 f9 eb c1                  	vpor	%xmm1, %xmm0, %xmm0
  32db00: c4 c1 f9 7e c4               	vmovq	%xmm0, %r12
  32db05: 41 39 ca                     	cmpl	%ecx, %r10d
  32db08: 0f 84 46 07 00 00            	je	0x32e254 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x1004>
  32db0e: f6 c2 0c                     	testb	$0xc, %dl
  32db11: 74 7d                        	je	0x32db90 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x940>
  32db13: 49 89 c8                     	movq	%rcx, %r8
  32db16: 89 d1                        	movl	%edx, %ecx
  32db18: 83 e1 3c                     	andl	$0x3c, %ecx
  32db1b: c4 c1 f9 6e c4               	vmovq	%r12, %xmm0
  32db20: c4 e1 f9 6e d7               	vmovq	%rdi, %xmm2
  32db25: c4 c1 f9 6e c8               	vmovq	%r8, %xmm1
  32db2a: c4 e2 7d 59 c9               	vpbroadcastq	%xmm1, %ymm1
  32db2f: c5 f5 eb 0d 09 fb d6 ff      	vpor	-0x2904f7(%rip), %ymm1, %ymm1 # 0x9d640 <anon.1b4110e42284c5f75ac48cf7fa14c04d.13.llvm.1722327191986239410+0x80>
  32db37: c4 e2 7d 59 d2               	vpbroadcastq	%xmm2, %ymm2
  32db3c: 49 29 c8                     	subq	%rcx, %r8
  32db3f: c4 e2 7d 59 1d 18 e7 d6 ff   	vpbroadcastq	-0x2918e8(%rip), %ymm3 # 0x9c260 <anon.5278ce255300ec7b361bbf944ee0b803.26.llvm.754055615369681222>
  32db48: 0f 1f 84 00 00 00 00 00      	nopl	(%rax,%rax)
  32db50: c4 e2 ed 47 e1               	vpsllvq	%ymm1, %ymm2, %ymm4
  32db55: c5 dd eb c0                  	vpor	%ymm0, %ymm4, %ymm0
  32db59: c5 f5 d4 cb                  	vpaddq	%ymm3, %ymm1, %ymm1
  32db5d: 49 83 c0 04                  	addq	$0x4, %r8
  32db61: 75 ed                        	jne	0x32db50 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x900>
  32db63: c4 e3 7d 39 c1 01            	vextracti128	$0x1, %ymm0, %xmm1
  32db69: c5 f9 eb c1                  	vpor	%xmm1, %xmm0, %xmm0
  32db6d: c5 f9 70 c8 ee               	vpshufd	$0xee, %xmm0, %xmm1     # xmm1 = xmm0[2,3,2,3]
  32db72: c5 f9 eb c1                  	vpor	%xmm1, %xmm0, %xmm0
  32db76: c4 c1 f9 7e c4               	vmovq	%xmm0, %r12
  32db7b: 41 39 ca                     	cmpl	%ecx, %r10d
  32db7e: 0f 84 d0 06 00 00            	je	0x32e254 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x1004>
  32db84: 66 66 66 2e 0f 1f 84 00 00 00 00 00  	nopw	%cs:(%rax,%rax)
  32db90: 48 89 fa                     	movq	%rdi, %rdx
  32db93: 48 d3 e2                     	shlq	%cl, %rdx
  32db96: 48 ff c1                     	incq	%rcx
  32db99: 49 09 d4                     	orq	%rdx, %r12
  32db9c: 49 39 ca                     	cmpq	%rcx, %r10
  32db9f: 75 ef                        	jne	0x32db90 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x940>
  32dba1: e9 ae 06 00 00               	jmp	0x32e254 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x1004>
  32dba6: 89 d1                        	movl	%edx, %ecx
  32dba8: 83 e1 30                     	andl	$0x30, %ecx
  32dbab: 41 83 e0 01                  	andl	$0x1, %r8d
  32dbaf: 4c 39 db                     	cmpq	%r11, %rbx
  32dbb2: c4 c1 78 92 c0               	kmovw	%r8d, %k0
  32dbb7: 62 f2 fd 48 7c c3            	vpbroadcastq	%rbx, %zmm0
  32dbbd: 41 b8 ff 00 00 00            	movl	$0xff, %r8d
  32dbc3: 45 0f 45 c5                  	cmovnel	%r13d, %r8d
  32dbc7: c4 c1 7b 92 c8               	kmovd	%r8d, %k1
  32dbcc: 4d 8d 04 ff                  	leaq	(%r15,%rdi,8), %r8
  32dbd0: c5 fc 47 d0                  	kxorw	%k0, %k0, %k2
  32dbd4: 62 f1 fd 48 6f 15 a2 32 d7 ff	vmovdqa64	-0x28cd5e(%rip), %zmm2 # 0xa0e80 <anon.ad39dab31d93760b58da730830a5ab31.2427.llvm.6451718222114420760+0x144>
  32dbde: c5 f1 ef c9                  	vpxor	%xmm1, %xmm1, %xmm1
  32dbe2: 62 f2 fd 48 59 25 1c db d6 ff	vpbroadcastq	-0x2924e4(%rip), %zmm4 # 0x9b708 <anon.5278ce255300ec7b361bbf944ee0b803.25.llvm.754055615369681222>
  32dbec: 62 f2 fd 48 59 2d 72 e0 d6 ff	vpbroadcastq	-0x291f8e(%rip), %zmm5 # 0x9bc68 <anon.ad39dab31d93760b58da730830a5ab31.633.llvm.6451718222114420760+0x50>
  32dbf6: 62 f2 fd 48 59 35 c8 e6 d6 ff	vpbroadcastq	-0x291938(%rip), %zmm6 # 0x9c2c8 <anon.ad39dab31d93760b58da730830a5ab31.728.llvm.6451718222114420760+0x40>
  32dc00: 49 89 ce                     	movq	%rcx, %r14
  32dc03: c5 e1 ef db                  	vpxor	%xmm3, %xmm3, %xmm3
  32dc07: 66 0f 1f 84 00 00 00 00 00   	nopw	(%rax,%rax)
  32dc10: 62 f1 ed 48 d4 fc            	vpaddq	%zmm4, %zmm2, %zmm7
  32dc16: c4 c1 f9 7e d4               	vmovq	%xmm2, %r12
  32dc1b: 62 11 fe 48 6f 04 e0         	vmovdqu64	(%r8,%r12,8), %zmm8
  32dc22: 62 11 fe 48 6f 4c e0 01      	vmovdqu64	0x40(%r8,%r12,8), %zmm9
  32dc2a: 62 f2 bd 48 29 dd            	vpcmpeqq	%zmm5, %zmm8, %k3
  32dc30: 62 f2 b5 48 29 e5            	vpcmpeqq	%zmm5, %zmm9, %k4
  32dc36: 62 d2 fd 48 37 e8            	vpcmpgtq	%zmm8, %zmm0, %k5
  32dc3c: 62 d2 fd 48 37 f1            	vpcmpgtq	%zmm9, %zmm0, %k6
  32dc42: c5 e4 45 c0                  	korw	%k0, %k3, %k0
  32dc46: c5 fc 45 c1                  	korw	%k1, %k0, %k0
  32dc4a: c5 dc 45 da                  	korw	%k2, %k4, %k3
  32dc4e: c5 e4 45 d1                  	korw	%k1, %k3, %k2
  32dc52: 62 53 bd cd 25 c0 ff         	vpternlogq	$0xff, %zmm8, %zmm8, %zmm8 {%k5} {z} # zmm8 {%k5} {z} = -1
  32dc59: 62 d1 bd 48 73 d0 3f         	vpsrlq	$0x3f, %zmm8, %zmm8
  32dc60: 62 53 b5 ce 25 c9 ff         	vpternlogq	$0xff, %zmm9, %zmm9, %zmm9 {%k6} {z} # zmm9 {%k6} {z} = -1
  32dc67: 62 d1 b5 48 73 d1 3f         	vpsrlq	$0x3f, %zmm9, %zmm9
  32dc6e: 62 f2 b5 48 47 ff            	vpsllvq	%zmm7, %zmm9, %zmm7
  32dc74: 62 f1 c5 48 eb db            	vporq	%zmm3, %zmm7, %zmm3
  32dc7a: 62 f2 bd 48 47 fa            	vpsllvq	%zmm2, %zmm8, %zmm7
  32dc80: 62 f1 c5 48 eb c9            	vporq	%zmm1, %zmm7, %zmm1
  32dc86: 62 f1 ed 48 d4 d6            	vpaddq	%zmm6, %zmm2, %zmm2
  32dc8c: 49 83 c6 f0                  	addq	$-0x10, %r14
  32dc90: 0f 85 7a ff ff ff            	jne	0x32dc10 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x9c0>
  32dc96: c5 e4 45 c0                  	korw	%k0, %k3, %k0
  32dc9a: c5 7b 93 c0                  	kmovd	%k0, %r8d
  32dc9e: 45 84 c0                     	testb	%r8b, %r8b
  32dca1: 41 0f 95 c6                  	setne	%r14b
  32dca5: 62 f1 e5 48 eb c1            	vporq	%zmm1, %zmm3, %zmm0
  32dcab: 62 f3 fd 48 3b c1 01         	vextracti64x4	$0x1, %zmm0, %ymm1
  32dcb2: 62 f1 fd 48 eb c1            	vporq	%zmm1, %zmm0, %zmm0
  32dcb8: c4 e3 7d 39 c1 01            	vextracti128	$0x1, %ymm0, %xmm1
  32dcbe: c5 f9 eb c1                  	vpor	%xmm1, %xmm0, %xmm0
  32dcc2: c5 f9 70 c8 ee               	vpshufd	$0xee, %xmm0, %xmm1     # xmm1 = xmm0[2,3,2,3]
  32dcc7: c5 f9 eb c1                  	vpor	%xmm1, %xmm0, %xmm0
  32dccb: c4 c1 f9 7e c4               	vmovq	%xmm0, %r12
  32dcd0: 41 39 ca                     	cmpl	%ecx, %r10d
  32dcd3: 0f 84 7b 05 00 00            	je	0x32e254 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x1004>
  32dcd9: f6 c2 0c                     	testb	$0xc, %dl
  32dcdc: 0f 84 c0 00 00 00            	je	0x32dda2 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0xb52>
  32dce2: 49 89 c8                     	movq	%rcx, %r8
  32dce5: 89 d1                        	movl	%edx, %ecx
  32dce7: 83 e1 3c                     	andl	$0x3c, %ecx
  32dcea: 41 83 e6 01                  	andl	$0x1, %r14d
  32dcee: 4c 39 db                     	cmpq	%r11, %rbx
  32dcf1: c4 c1 78 92 c6               	kmovw	%r14d, %k0
  32dcf6: c4 c1 f9 6e c4               	vmovq	%r12, %xmm0
  32dcfb: c4 e1 f9 6e cb               	vmovq	%rbx, %xmm1
  32dd00: ba ff 00 00 00               	movl	$0xff, %edx
  32dd05: 44 0f 44 ea                  	cmovel	%edx, %r13d
  32dd09: c4 e2 7d 59 c9               	vpbroadcastq	%xmm1, %ymm1
  32dd0e: c4 c1 7b 92 cd               	kmovd	%r13d, %k1
  32dd13: c4 c1 f9 6e d0               	vmovq	%r8, %xmm2
  32dd18: c4 e2 7d 59 d2               	vpbroadcastq	%xmm2, %ymm2
  32dd1d: c5 ed eb 15 1b f9 d6 ff      	vpor	-0x2906e5(%rip), %ymm2, %ymm2 # 0x9d640 <anon.1b4110e42284c5f75ac48cf7fa14c04d.13.llvm.1722327191986239410+0x80>
  32dd25: 49 8d 14 ff                  	leaq	(%r15,%rdi,8), %rdx
  32dd29: 49 29 c8                     	subq	%rcx, %r8
  32dd2c: 62 f2 fd 48 59 1d 32 df d6 ff	vpbroadcastq	-0x2920ce(%rip), %zmm3 # 0x9bc68 <anon.ad39dab31d93760b58da730830a5ab31.633.llvm.6451718222114420760+0x50>
  32dd36: c4 e2 7d 59 25 21 e5 d6 ff   	vpbroadcastq	-0x291adf(%rip), %ymm4 # 0x9c260 <anon.5278ce255300ec7b361bbf944ee0b803.26.llvm.754055615369681222>
  32dd3f: 90                           	nop
  32dd40: c4 c1 f9 7e d6               	vmovq	%xmm2, %r14
  32dd45: c4 a1 7e 6f 2c f2            	vmovdqu	(%rdx,%r14,8), %ymm5
  32dd4b: 62 f2 d5 48 29 d3            	vpcmpeqq	%zmm3, %zmm5, %k2
  32dd51: c5 ec 45 d1                  	korw	%k1, %k2, %k2
  32dd55: c5 ec 45 c0                  	korw	%k0, %k2, %k0
  32dd59: c4 e2 75 37 ed               	vpcmpgtq	%ymm5, %ymm1, %ymm5
  32dd5e: c5 d5 73 d5 3f               	vpsrlq	$0x3f, %ymm5, %ymm5
  32dd63: c4 e2 d5 47 ea               	vpsllvq	%ymm2, %ymm5, %ymm5
  32dd68: c5 d5 eb c0                  	vpor	%ymm0, %ymm5, %ymm0
  32dd6c: c5 ed d4 d4                  	vpaddq	%ymm4, %ymm2, %ymm2
  32dd70: 49 83 c0 04                  	addq	$0x4, %r8
  32dd74: 75 ca                        	jne	0x32dd40 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0xaf0>
  32dd76: c5 fb 93 d0                  	kmovd	%k0, %edx
  32dd7a: f6 c2 0f                     	testb	$0xf, %dl
  32dd7d: 41 0f 95 c6                  	setne	%r14b
  32dd81: c4 e3 7d 39 c1 01            	vextracti128	$0x1, %ymm0, %xmm1
  32dd87: c5 f9 eb c1                  	vpor	%xmm1, %xmm0, %xmm0
  32dd8b: c5 f9 70 c8 ee               	vpshufd	$0xee, %xmm0, %xmm1     # xmm1 = xmm0[2,3,2,3]
  32dd90: c5 f9 eb c1                  	vpor	%xmm1, %xmm0, %xmm0
  32dd94: c4 c1 f9 7e c4               	vmovq	%xmm0, %r12
  32dd99: 41 39 ca                     	cmpl	%ecx, %r10d
  32dd9c: 0f 84 b2 04 00 00            	je	0x32e254 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x1004>
  32dda2: 49 8d 14 ff                  	leaq	(%r15,%rdi,8), %rdx
  32dda6: 44 89 f7                     	movl	%r14d, %edi
  32dda9: 0f 1f 80 00 00 00 00         	nopl	(%rax)
  32ddb0: 4c 39 db                     	cmpq	%r11, %rbx
  32ddb3: 41 0f 94 c6                  	sete	%r14b
  32ddb7: 4c 8b 04 ca                  	movq	(%rdx,%rcx,8), %r8
  32ddbb: 4d 39 d8                     	cmpq	%r11, %r8
  32ddbe: 41 0f 94 c7                  	sete	%r15b
  32ddc2: 45 31 ed                     	xorl	%r13d, %r13d
  32ddc5: 49 39 d8                     	cmpq	%rbx, %r8
  32ddc8: 41 0f 9c c5                  	setl	%r13b
  32ddcc: 41 08 fe                     	orb	%dil, %r14b
  32ddcf: 45 08 fe                     	orb	%r15b, %r14b
  32ddd2: 4c 8d 41 01                  	leaq	0x1(%rcx), %r8
  32ddd6: 49 d3 e5                     	shlq	%cl, %r13
  32ddd9: 4d 09 ec                     	orq	%r13, %r12
  32dddc: 44 89 f7                     	movl	%r14d, %edi
  32dddf: 4d 39 c2                     	cmpq	%r8, %r10
  32dde2: 4c 89 c1                     	movq	%r8, %rcx
  32dde5: 75 c9                        	jne	0x32ddb0 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0xb60>
  32dde7: e9 68 04 00 00               	jmp	0x32e254 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x1004>
  32ddec: 89 d1                        	movl	%edx, %ecx
  32ddee: 83 e1 30                     	andl	$0x30, %ecx
  32ddf1: 41 83 e0 01                  	andl	$0x1, %r8d
  32ddf5: 4d 39 df                     	cmpq	%r11, %r15
  32ddf8: c4 c1 78 92 c0               	kmovw	%r8d, %k0
  32ddfd: 62 d2 fd 48 7c c7            	vpbroadcastq	%r15, %zmm0
  32de03: 41 b8 ff 00 00 00            	movl	$0xff, %r8d
  32de09: 45 0f 45 c5                  	cmovnel	%r13d, %r8d
  32de0d: c4 c1 7b 92 c8               	kmovd	%r8d, %k1
  32de12: 4c 8d 04 fb                  	leaq	(%rbx,%rdi,8), %r8
  32de16: c5 fc 47 d0                  	kxorw	%k0, %k0, %k2
  32de1a: 62 f1 fd 48 6f 15 5c 30 d7 ff	vmovdqa64	-0x28cfa4(%rip), %zmm2 # 0xa0e80 <anon.ad39dab31d93760b58da730830a5ab31.2427.llvm.6451718222114420760+0x144>
  32de24: c5 f1 ef c9                  	vpxor	%xmm1, %xmm1, %xmm1
  32de28: 62 f2 fd 48 59 25 d6 d8 d6 ff	vpbroadcastq	-0x29272a(%rip), %zmm4 # 0x9b708 <anon.5278ce255300ec7b361bbf944ee0b803.25.llvm.754055615369681222>
  32de32: 62 f2 fd 48 59 2d 2c de d6 ff	vpbroadcastq	-0x2921d4(%rip), %zmm5 # 0x9bc68 <anon.ad39dab31d93760b58da730830a5ab31.633.llvm.6451718222114420760+0x50>
  32de3c: 62 f2 fd 48 59 35 82 e4 d6 ff	vpbroadcastq	-0x291b7e(%rip), %zmm6 # 0x9c2c8 <anon.ad39dab31d93760b58da730830a5ab31.728.llvm.6451718222114420760+0x40>
  32de46: 49 89 ce                     	movq	%rcx, %r14
  32de49: c5 e1 ef db                  	vpxor	%xmm3, %xmm3, %xmm3
  32de4d: 0f 1f 00                     	nopl	(%rax)
  32de50: 62 f1 ed 48 d4 fc            	vpaddq	%zmm4, %zmm2, %zmm7
  32de56: c4 c1 f9 7e d4               	vmovq	%xmm2, %r12
  32de5b: 62 11 fe 48 6f 04 e0         	vmovdqu64	(%r8,%r12,8), %zmm8
  32de62: 62 11 fe 48 6f 4c e0 01      	vmovdqu64	0x40(%r8,%r12,8), %zmm9
  32de6a: 62 f2 bd 48 29 dd            	vpcmpeqq	%zmm5, %zmm8, %k3
  32de70: 62 f2 b5 48 29 e5            	vpcmpeqq	%zmm5, %zmm9, %k4
  32de76: 62 f2 bd 48 37 e8            	vpcmpgtq	%zmm0, %zmm8, %k5
  32de7c: 62 f2 b5 48 37 f0            	vpcmpgtq	%zmm0, %zmm9, %k6
  32de82: c5 e4 45 c0                  	korw	%k0, %k3, %k0
  32de86: c5 dc 45 da                  	korw	%k2, %k4, %k3
  32de8a: c5 fc 45 c1                  	korw	%k1, %k0, %k0
  32de8e: c5 e4 45 d1                  	korw	%k1, %k3, %k2
  32de92: 62 53 bd cd 25 c0 ff         	vpternlogq	$0xff, %zmm8, %zmm8, %zmm8 {%k5} {z} # zmm8 {%k5} {z} = -1
  32de99: 62 d1 bd 48 73 d0 3f         	vpsrlq	$0x3f, %zmm8, %zmm8
  32dea0: 62 53 b5 ce 25 c9 ff         	vpternlogq	$0xff, %zmm9, %zmm9, %zmm9 {%k6} {z} # zmm9 {%k6} {z} = -1
  32dea7: 62 d1 b5 48 73 d1 3f         	vpsrlq	$0x3f, %zmm9, %zmm9
  32deae: 62 f2 b5 48 47 ff            	vpsllvq	%zmm7, %zmm9, %zmm7
  32deb4: 62 f1 c5 48 eb db            	vporq	%zmm3, %zmm7, %zmm3
  32deba: 62 f2 bd 48 47 fa            	vpsllvq	%zmm2, %zmm8, %zmm7
  32dec0: 62 f1 c5 48 eb c9            	vporq	%zmm1, %zmm7, %zmm1
  32dec6: 62 f1 ed 48 d4 d6            	vpaddq	%zmm6, %zmm2, %zmm2
  32decc: 49 83 c6 f0                  	addq	$-0x10, %r14
  32ded0: 0f 85 7a ff ff ff            	jne	0x32de50 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0xc00>
  32ded6: c5 e4 45 c0                  	korw	%k0, %k3, %k0
  32deda: c5 7b 93 c0                  	kmovd	%k0, %r8d
  32dede: 45 84 c0                     	testb	%r8b, %r8b
  32dee1: 41 0f 95 c6                  	setne	%r14b
  32dee5: 62 f1 e5 48 eb c1            	vporq	%zmm1, %zmm3, %zmm0
  32deeb: 62 f3 fd 48 3b c1 01         	vextracti64x4	$0x1, %zmm0, %ymm1
  32def2: 62 f1 fd 48 eb c1            	vporq	%zmm1, %zmm0, %zmm0
  32def8: c4 e3 7d 39 c1 01            	vextracti128	$0x1, %ymm0, %xmm1
  32defe: c5 f9 eb c1                  	vpor	%xmm1, %xmm0, %xmm0
  32df02: c5 f9 70 c8 ee               	vpshufd	$0xee, %xmm0, %xmm1     # xmm1 = xmm0[2,3,2,3]
  32df07: c5 f9 eb c1                  	vpor	%xmm1, %xmm0, %xmm0
  32df0b: c4 c1 f9 7e c4               	vmovq	%xmm0, %r12
  32df10: 41 39 ca                     	cmpl	%ecx, %r10d
  32df13: 0f 84 3b 03 00 00            	je	0x32e254 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x1004>
  32df19: f6 c2 0c                     	testb	$0xc, %dl
  32df1c: 0f 84 c0 00 00 00            	je	0x32dfe2 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0xd92>
  32df22: 49 89 c8                     	movq	%rcx, %r8
  32df25: 89 d1                        	movl	%edx, %ecx
  32df27: 83 e1 3c                     	andl	$0x3c, %ecx
  32df2a: 41 83 e6 01                  	andl	$0x1, %r14d
  32df2e: 4d 39 df                     	cmpq	%r11, %r15
  32df31: c4 c1 78 92 c6               	kmovw	%r14d, %k0
  32df36: c4 c1 f9 6e c4               	vmovq	%r12, %xmm0
  32df3b: c4 c1 f9 6e cf               	vmovq	%r15, %xmm1
  32df40: ba ff 00 00 00               	movl	$0xff, %edx
  32df45: 44 0f 44 ea                  	cmovel	%edx, %r13d
  32df49: c4 e2 7d 59 c9               	vpbroadcastq	%xmm1, %ymm1
  32df4e: c4 c1 7b 92 cd               	kmovd	%r13d, %k1
  32df53: c4 c1 f9 6e d0               	vmovq	%r8, %xmm2
  32df58: c4 e2 7d 59 d2               	vpbroadcastq	%xmm2, %ymm2
  32df5d: c5 ed eb 15 db f6 d6 ff      	vpor	-0x290925(%rip), %ymm2, %ymm2 # 0x9d640 <anon.1b4110e42284c5f75ac48cf7fa14c04d.13.llvm.1722327191986239410+0x80>
  32df65: 48 8d 14 fb                  	leaq	(%rbx,%rdi,8), %rdx
  32df69: 49 29 c8                     	subq	%rcx, %r8
  32df6c: 62 f2 fd 48 59 1d f2 dc d6 ff	vpbroadcastq	-0x29230e(%rip), %zmm3 # 0x9bc68 <anon.ad39dab31d93760b58da730830a5ab31.633.llvm.6451718222114420760+0x50>
  32df76: c4 e2 7d 59 25 e1 e2 d6 ff   	vpbroadcastq	-0x291d1f(%rip), %ymm4 # 0x9c260 <anon.5278ce255300ec7b361bbf944ee0b803.26.llvm.754055615369681222>
  32df7f: 90                           	nop
  32df80: c4 c1 f9 7e d6               	vmovq	%xmm2, %r14
  32df85: c4 a1 7e 6f 2c f2            	vmovdqu	(%rdx,%r14,8), %ymm5
  32df8b: 62 f2 d5 48 29 d3            	vpcmpeqq	%zmm3, %zmm5, %k2
  32df91: c5 ec 45 c0                  	korw	%k0, %k2, %k0
  32df95: c5 fc 45 c1                  	korw	%k1, %k0, %k0
  32df99: c4 e2 55 37 e9               	vpcmpgtq	%ymm1, %ymm5, %ymm5
  32df9e: c5 d5 73 d5 3f               	vpsrlq	$0x3f, %ymm5, %ymm5
  32dfa3: c4 e2 d5 47 ea               	vpsllvq	%ymm2, %ymm5, %ymm5
  32dfa8: c5 d5 eb c0                  	vpor	%ymm0, %ymm5, %ymm0
  32dfac: c5 ed d4 d4                  	vpaddq	%ymm4, %ymm2, %ymm2
  32dfb0: 49 83 c0 04                  	addq	$0x4, %r8
  32dfb4: 75 ca                        	jne	0x32df80 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0xd30>
  32dfb6: c5 fb 93 d0                  	kmovd	%k0, %edx
  32dfba: f6 c2 0f                     	testb	$0xf, %dl
  32dfbd: 41 0f 95 c6                  	setne	%r14b
  32dfc1: c4 e3 7d 39 c1 01            	vextracti128	$0x1, %ymm0, %xmm1
  32dfc7: c5 f9 eb c1                  	vpor	%xmm1, %xmm0, %xmm0
  32dfcb: c5 f9 70 c8 ee               	vpshufd	$0xee, %xmm0, %xmm1     # xmm1 = xmm0[2,3,2,3]
  32dfd0: c5 f9 eb c1                  	vpor	%xmm1, %xmm0, %xmm0
  32dfd4: c4 c1 f9 7e c4               	vmovq	%xmm0, %r12
  32dfd9: 41 39 ca                     	cmpl	%ecx, %r10d
  32dfdc: 0f 84 72 02 00 00            	je	0x32e254 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x1004>
  32dfe2: 48 8d 14 fb                  	leaq	(%rbx,%rdi,8), %rdx
  32dfe6: 44 89 f7                     	movl	%r14d, %edi
  32dfe9: 0f 1f 80 00 00 00 00         	nopl	(%rax)
  32dff0: 4d 39 df                     	cmpq	%r11, %r15
  32dff3: 41 0f 94 c6                  	sete	%r14b
  32dff7: 4c 8b 04 ca                  	movq	(%rdx,%rcx,8), %r8
  32dffb: 4d 39 d8                     	cmpq	%r11, %r8
  32dffe: 0f 94 c3                     	sete	%bl
  32e001: 45 31 ed                     	xorl	%r13d, %r13d
  32e004: 4d 39 c7                     	cmpq	%r8, %r15
  32e007: 41 0f 9c c5                  	setl	%r13b
  32e00b: 41 08 fe                     	orb	%dil, %r14b
  32e00e: 41 08 de                     	orb	%bl, %r14b
  32e011: 4c 8d 41 01                  	leaq	0x1(%rcx), %r8
  32e015: 49 d3 e5                     	shlq	%cl, %r13
  32e018: 4d 09 ec                     	orq	%r13, %r12
  32e01b: 44 89 f7                     	movl	%r14d, %edi
  32e01e: 4d 39 c2                     	cmpq	%r8, %r10
  32e021: 4c 89 c1                     	movq	%r8, %rcx
  32e024: 75 ca                        	jne	0x32dff0 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0xda0>
  32e026: e9 29 02 00 00               	jmp	0x32e254 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x1004>
  32e02b: 89 d1                        	movl	%edx, %ecx
  32e02d: 83 e1 30                     	andl	$0x30, %ecx
  32e030: 41 83 e0 01                  	andl	$0x1, %r8d
  32e034: c4 c1 78 92 c0               	kmovw	%r8d, %k0
  32e039: c5 fc 47 c8                  	kxorw	%k0, %k0, %k1
  32e03d: 62 f1 fd 48 6f 0d 39 2e d7 ff	vmovdqa64	-0x28d1c7(%rip), %zmm1 # 0xa0e80 <anon.ad39dab31d93760b58da730830a5ab31.2427.llvm.6451718222114420760+0x144>
  32e047: c5 f9 ef c0                  	vpxor	%xmm0, %xmm0, %xmm0
  32e04b: 62 f2 fd 48 59 1d b3 d6 d6 ff	vpbroadcastq	-0x29294d(%rip), %zmm3 # 0x9b708 <anon.5278ce255300ec7b361bbf944ee0b803.25.llvm.754055615369681222>
  32e055: 62 f2 fd 48 59 25 09 dc d6 ff	vpbroadcastq	-0x2923f7(%rip), %zmm4 # 0x9bc68 <anon.ad39dab31d93760b58da730830a5ab31.633.llvm.6451718222114420760+0x50>
  32e05f: 62 f2 fd 48 59 2d 5f e2 d6 ff	vpbroadcastq	-0x291da1(%rip), %zmm5 # 0x9c2c8 <anon.ad39dab31d93760b58da730830a5ab31.728.llvm.6451718222114420760+0x40>
  32e069: 49 89 c8                     	movq	%rcx, %r8
  32e06c: c5 e9 ef d2                  	vpxor	%xmm2, %xmm2, %xmm2
  32e070: 62 f1 f5 48 d4 f3            	vpaddq	%zmm3, %zmm1, %zmm6
  32e076: c4 c1 f9 7e ce               	vmovq	%xmm1, %r14
  32e07b: 49 01 fe                     	addq	%rdi, %r14
  32e07e: 62 91 fe 48 6f 3c f7         	vmovdqu64	(%r15,%r14,8), %zmm7
  32e085: 62 11 fe 48 6f 44 f7 01      	vmovdqu64	0x40(%r15,%r14,8), %zmm8
  32e08d: 62 31 fe 48 6f 0c f3         	vmovdqu64	(%rbx,%r14,8), %zmm9
  32e094: 62 31 fe 48 6f 54 f3 01      	vmovdqu64	0x40(%rbx,%r14,8), %zmm10
  32e09c: 62 f2 c5 48 29 d4            	vpcmpeqq	%zmm4, %zmm7, %k2
  32e0a2: 62 f2 bd 48 29 dc            	vpcmpeqq	%zmm4, %zmm8, %k3
  32e0a8: 62 f2 b5 48 29 e4            	vpcmpeqq	%zmm4, %zmm9, %k4
  32e0ae: 62 f2 ad 48 29 ec            	vpcmpeqq	%zmm4, %zmm10, %k5
  32e0b4: c5 ec 45 d4                  	korw	%k4, %k2, %k2
  32e0b8: c5 e4 45 dd                  	korw	%k5, %k3, %k3
  32e0bc: 62 f2 b5 48 37 e7            	vpcmpgtq	%zmm7, %zmm9, %k4
  32e0c2: 62 d2 ad 48 37 e8            	vpcmpgtq	%zmm8, %zmm10, %k5
  32e0c8: c5 ec 45 c0                  	korw	%k0, %k2, %k0
  32e0cc: c5 e4 45 c9                  	korw	%k1, %k3, %k1
  32e0d0: 62 f3 c5 cc 25 ff ff         	vpternlogq	$0xff, %zmm7, %zmm7, %zmm7 {%k4} {z} # zmm7 {%k4} {z} = -1
  32e0d7: 62 f1 c5 48 73 d7 3f         	vpsrlq	$0x3f, %zmm7, %zmm7
  32e0de: 62 53 bd cd 25 c0 ff         	vpternlogq	$0xff, %zmm8, %zmm8, %zmm8 {%k5} {z} # zmm8 {%k5} {z} = -1
  32e0e5: 62 d1 bd 48 73 d0 3f         	vpsrlq	$0x3f, %zmm8, %zmm8
  32e0ec: 62 f2 bd 48 47 f6            	vpsllvq	%zmm6, %zmm8, %zmm6
  32e0f2: 62 f1 cd 48 eb d2            	vporq	%zmm2, %zmm6, %zmm2
  32e0f8: 62 f2 c5 48 47 f1            	vpsllvq	%zmm1, %zmm7, %zmm6
  32e0fe: 62 f1 cd 48 eb c0            	vporq	%zmm0, %zmm6, %zmm0
  32e104: 62 f1 f5 48 d4 cd            	vpaddq	%zmm5, %zmm1, %zmm1
  32e10a: 49 83 c0 f0                  	addq	$-0x10, %r8
  32e10e: 0f 85 5c ff ff ff            	jne	0x32e070 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0xe20>
  32e114: c5 f4 45 c0                  	korw	%k0, %k1, %k0
  32e118: c5 7b 93 c0                  	kmovd	%k0, %r8d
  32e11c: 45 84 c0                     	testb	%r8b, %r8b
  32e11f: 41 0f 95 c6                  	setne	%r14b
  32e123: 62 f1 ed 48 eb c0            	vporq	%zmm0, %zmm2, %zmm0
  32e129: 62 f3 fd 48 3b c1 01         	vextracti64x4	$0x1, %zmm0, %ymm1
  32e130: 62 f1 fd 48 eb c1            	vporq	%zmm1, %zmm0, %zmm0
  32e136: c4 e3 7d 39 c1 01            	vextracti128	$0x1, %ymm0, %xmm1
  32e13c: c5 f9 eb c1                  	vpor	%xmm1, %xmm0, %xmm0
  32e140: c5 f9 70 c8 ee               	vpshufd	$0xee, %xmm0, %xmm1     # xmm1 = xmm0[2,3,2,3]
  32e145: c5 f9 eb c1                  	vpor	%xmm1, %xmm0, %xmm0
  32e149: c4 c1 f9 7e c4               	vmovq	%xmm0, %r12
  32e14e: 41 39 ca                     	cmpl	%ecx, %r10d
  32e151: 0f 84 fd 00 00 00            	je	0x32e254 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x1004>
  32e157: f6 c2 0c                     	testb	$0xc, %dl
  32e15a: 0f 84 ac 00 00 00            	je	0x32e20c <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0xfbc>
  32e160: 49 89 c8                     	movq	%rcx, %r8
  32e163: 89 d1                        	movl	%edx, %ecx
  32e165: 83 e1 3c                     	andl	$0x3c, %ecx
  32e168: 41 83 e6 01                  	andl	$0x1, %r14d
  32e16c: c4 c1 78 92 c6               	kmovw	%r14d, %k0
  32e171: c4 c1 f9 6e c4               	vmovq	%r12, %xmm0
  32e176: c4 c1 f9 6e c8               	vmovq	%r8, %xmm1
  32e17b: c4 e2 7d 59 c9               	vpbroadcastq	%xmm1, %ymm1
  32e180: c5 f5 eb 0d b8 f4 d6 ff      	vpor	-0x290b48(%rip), %ymm1, %ymm1 # 0x9d640 <anon.1b4110e42284c5f75ac48cf7fa14c04d.13.llvm.1722327191986239410+0x80>
  32e188: 49 29 c8                     	subq	%rcx, %r8
  32e18b: c4 e2 7d 59 15 d4 da d6 ff   	vpbroadcastq	-0x29252c(%rip), %ymm2 # 0x9bc68 <anon.ad39dab31d93760b58da730830a5ab31.633.llvm.6451718222114420760+0x50>
  32e194: c4 e2 7d 59 1d c3 e0 d6 ff   	vpbroadcastq	-0x291f3d(%rip), %ymm3 # 0x9c260 <anon.5278ce255300ec7b361bbf944ee0b803.26.llvm.754055615369681222>
  32e19d: 0f 1f 00                     	nopl	(%rax)
  32e1a0: c4 e1 f9 7e ca               	vmovq	%xmm1, %rdx
  32e1a5: 48 01 fa                     	addq	%rdi, %rdx
  32e1a8: c4 c1 7e 6f 24 d7            	vmovdqu	(%r15,%rdx,8), %ymm4
  32e1ae: c5 fe 6f 2c d3               	vmovdqu	(%rbx,%rdx,8), %ymm5
  32e1b3: 62 f2 dd 48 29 ca            	vpcmpeqq	%zmm2, %zmm4, %k1
  32e1b9: 62 f2 d5 48 29 d2            	vpcmpeqq	%zmm2, %zmm5, %k2
  32e1bf: c5 f4 45 ca                  	korw	%k2, %k1, %k1
  32e1c3: c5 f4 45 c0                  	korw	%k0, %k1, %k0
  32e1c7: c4 e2 55 37 e4               	vpcmpgtq	%ymm4, %ymm5, %ymm4
  32e1cc: c5 dd 73 d4 3f               	vpsrlq	$0x3f, %ymm4, %ymm4
  32e1d1: c4 e2 dd 47 e1               	vpsllvq	%ymm1, %ymm4, %ymm4
  32e1d6: c5 dd eb c0                  	vpor	%ymm0, %ymm4, %ymm0
  32e1da: c5 f5 d4 cb                  	vpaddq	%ymm3, %ymm1, %ymm1
  32e1de: 49 83 c0 04                  	addq	$0x4, %r8
  32e1e2: 75 bc                        	jne	0x32e1a0 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0xf50>
  32e1e4: c5 fb 93 d0                  	kmovd	%k0, %edx
  32e1e8: f6 c2 0f                     	testb	$0xf, %dl
  32e1eb: 41 0f 95 c6                  	setne	%r14b
  32e1ef: c4 e3 7d 39 c1 01            	vextracti128	$0x1, %ymm0, %xmm1
  32e1f5: c5 f9 eb c1                  	vpor	%xmm1, %xmm0, %xmm0
  32e1f9: c5 f9 70 c8 ee               	vpshufd	$0xee, %xmm0, %xmm1     # xmm1 = xmm0[2,3,2,3]
  32e1fe: c5 f9 eb c1                  	vpor	%xmm1, %xmm0, %xmm0
  32e202: c4 c1 f9 7e c4               	vmovq	%xmm0, %r12
  32e207: 41 39 ca                     	cmpl	%ecx, %r10d
  32e20a: 74 48                        	je	0x32e254 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x1004>
  32e20c: 48 c1 e7 03                  	shlq	$0x3, %rdi
  32e210: 48 01 fb                     	addq	%rdi, %rbx
  32e213: 49 01 ff                     	addq	%rdi, %r15
  32e216: 66 2e 0f 1f 84 00 00 00 00 00	nopw	%cs:(%rax,%rax)
  32e220: 49 8b 14 cf                  	movq	(%r15,%rcx,8), %rdx
  32e224: 48 8b 3c cb                  	movq	(%rbx,%rcx,8), %rdi
  32e228: 4c 39 da                     	cmpq	%r11, %rdx
  32e22b: 41 0f 94 c0                  	sete	%r8b
  32e22f: 4c 39 df                     	cmpq	%r11, %rdi
  32e232: 41 0f 94 c5                  	sete	%r13b
  32e236: 45 08 c5                     	orb	%r8b, %r13b
  32e239: 45 31 c0                     	xorl	%r8d, %r8d
  32e23c: 48 39 fa                     	cmpq	%rdi, %rdx
  32e23f: 41 0f 9c c0                  	setl	%r8b
  32e243: 45 08 ee                     	orb	%r13b, %r14b
  32e246: 49 d3 e0                     	shlq	%cl, %r8
  32e249: 48 ff c1                     	incq	%rcx
  32e24c: 4d 09 c4                     	orq	%r8, %r12
  32e24f: 49 39 ca                     	cmpq	%rcx, %r10
  32e252: 75 cc                        	jne	0x32e220 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0xfd0>
  32e254: 41 80 e6 01                  	andb	$0x1, %r14b
  32e258: 44 88 70 38                  	movb	%r14b, 0x38(%rax)
  32e25c: 48 8b 45 c8                  	movq	-0x38(%rbp), %rax
  32e260: 48 39 c6                     	cmpq	%rax, %rsi
  32e263: 73 27                        	jae	0x32e28c <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x103c>
  32e265: 4d 89 21                     	movq	%r12, (%r9)
  32e268: 48 83 c4 18                  	addq	$0x18, %rsp
  32e26c: 5b                           	popq	%rbx
  32e26d: 41 5c                        	popq	%r12
  32e26f: 41 5d                        	popq	%r13
  32e271: 41 5e                        	popq	%r14
  32e273: 41 5f                        	popq	%r15
  32e275: 5d                           	popq	%rbp
  32e276: c5 f8 77                     	vzeroupper
  32e279: c3                           	retq
  32e27a: 48 8d 0d 4f 9f fb 00         	leaq	0xfb9f4f(%rip), %rcx    # 0x12e81d0 <alloc_44f32c0500caa70dfe7fcfa87250b31f.llvm.5180523300074625071>
  32e281: 31 ff                        	xorl	%edi, %edi
  32e283: 4c 89 c2                     	movq	%r8, %rdx
  32e286: ff 15 24 6d 01 01            	callq	*0x1016d24(%rip)        # 0x1344fb0 <writev+0x1344fb0>
  32e28c: 48 8d 15 25 9f fb 00         	leaq	0xfb9f25(%rip), %rdx    # 0x12e81b8 <alloc_1fbb1658c3e837d8e13e0cd32c0c5d29.llvm.5180523300074625071>
  32e293: 48 89 f7                     	movq	%rsi, %rdi
  32e296: 48 89 c6                     	movq	%rax, %rsi
  32e299: c5 f8 77                     	vzeroupper
  32e29c: ff 15 7e 6a 01 01            	callq	*0x1016a7e(%rip)        # 0x1344d20 <writev+0x1344d20>
