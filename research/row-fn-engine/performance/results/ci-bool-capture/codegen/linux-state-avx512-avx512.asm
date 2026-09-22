
/tmp/row-fn-bool-linux-state-avx512-target/x86_64-unknown-linux-gnu/release/deps/row_fn_bool_retry-4282a15eb71e334d:	file format elf64-x86-64

Disassembly of section .text:

000000000034bb70 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_>:
  34bb70:      	pushq	%rbp
  34bb71:      	movq	%rsp, %rbp
  34bb74:      	pushq	%r15
  34bb76:      	pushq	%r14
  34bb78:      	pushq	%r13
  34bb7a:      	pushq	%r12
  34bb7c:      	pushq	%rbx
  34bb7d:      	subq	$0x18, %rsp
  34bb81:      	movq	%rsi, %r8
  34bb84:      	movq	%rdx, %rsi
  34bb87:      	shrq	$0x6, %rsi
  34bb8b:      	cmpq	%r8, %rsi
  34bb8e:      	ja	0x34cbcb <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x105b>
  34bb94:      	movq	%rcx, %rax
  34bb97:      	movq	%r8, -0x38(%rbp)
  34bb9b:      	movl	%edx, %r11d
  34bb9e:      	andl	$0x3f, %r11d
  34bba2:      	movabsq	$0x7fffffffffffffff, %r10 # imm = 0x7FFFFFFFFFFFFFFF
  34bbac:      	leaq	(%rdi,%rsi,8), %r13
  34bbb0:      	testq	%rsi, %rsi
  34bbb3:      	movq	%r13, -0x30(%rbp)
  34bbb7:      	je	0x34c2a1 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x731>
  34bbbd:      	leaq	(,%rsi,8), %rcx
  34bbc5:      	movzbl	0x38(%rax), %r15d
  34bbca:      	movzbl	0x18(%rax), %r9d
  34bbcf:      	movq	0x20(%rax), %rbx
  34bbd3:      	movq	0x28(%rax), %r8
  34bbd7:      	cmpl	$0x1, (%rax)
  34bbda:      	jne	0x34bd10 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x1a0>
  34bbe0:      	movq	0x10(%rax), %r14
  34bbe4:      	movq	(%r14), %r14
  34bbe7:      	testb	%r9b, %r9b
  34bbea:      	je	0x34becb <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x35b>
  34bbf0:      	movq	(%r8), %r8
  34bbf3:      	xorl	%r9d, %r9d
  34bbf6:      	cmpq	%r8, %r14
  34bbf9:      	movl	$0x101, %ebx            # imm = 0x101
  34bbfe:      	cmovgeq	%r9, %rbx
  34bc02:      	movl	%ebx, %r9d
  34bc05:      	shll	$0x10, %r9d
  34bc09:      	shrl	$0x18, %r9d
  34bc0d:      	movl	%ebx, %r12d
  34bc10:      	shrl	$0x8, %r12d
  34bc14:      	vmovd	%ebx, %xmm0
  34bc18:      	vpinsrb	$0x1, %r12d, %xmm0, %xmm0
  34bc1e:      	vpinsrb	$0x2, %ebx, %xmm0, %xmm0
  34bc24:      	vpinsrb	$0x3, %r9d, %xmm0, %xmm0
  34bc2a:      	vpinsrb	$0x4, %ebx, %xmm0, %xmm0
  34bc30:      	vpinsrb	$0x5, %r12d, %xmm0, %xmm0
  34bc36:      	vpinsrb	$0x6, %ebx, %xmm0, %xmm0
  34bc3c:      	vpinsrb	$0x7, %r9d, %xmm0, %xmm0
  34bc42:      	vpinsrb	$0x8, %ebx, %xmm0, %xmm0
  34bc48:      	vpinsrb	$0x9, %r12d, %xmm0, %xmm0
  34bc4e:      	vpinsrb	$0xa, %ebx, %xmm0, %xmm0
  34bc54:      	vpinsrb	$0xb, %r9d, %xmm0, %xmm0
  34bc5a:      	vpinsrb	$0xc, %ebx, %xmm0, %xmm0
  34bc60:      	vpinsrb	$0xd, %r12d, %xmm0, %xmm0
  34bc66:      	vpinsrb	$0xe, %ebx, %xmm0, %xmm0
  34bc6c:      	vpinsrb	$0xf, %r9d, %xmm0, %xmm0
  34bc72:      	vshufi64x2	$0x0, %zmm0, %zmm0, %zmm0 # zmm0 = zmm0[0,1,0,1,0,1,0,1]
  34bc79:      	addq	$-0x8, %rcx
  34bc7d:      	movl	%ecx, %r9d
  34bc80:      	notl	%r9d
  34bc83:      	vptestmb	%zmm0, %zmm0, %k0
  34bc89:      	testb	$0x38, %r9b
  34bc8d:      	je	0x34bcae <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x13e>
  34bc8f:      	movl	%ecx, %r9d
  34bc92:      	shrl	$0x3, %r9d
  34bc96:      	incl	%r9d
  34bc99:      	andl	$0x7, %r9d
  34bc9d:      	nopl	(%rax)
  34bca0:      	kmovq	%k0, (%rdi)
  34bca5:      	addq	$0x8, %rdi
  34bca9:      	decq	%r9
  34bcac:      	jne	0x34bca0 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x130>
  34bcae:      	cmpq	$0x38, %rcx
  34bcb2:      	jb	0x34bcf8 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x188>
  34bcb4:      	nopw	%cs:(%rax,%rax)
  34bcc0:      	kmovq	%k0, (%rdi)
  34bcc5:      	kmovq	%k0, 0x8(%rdi)
  34bccb:      	kmovq	%k0, 0x10(%rdi)
  34bcd1:      	kmovq	%k0, 0x18(%rdi)
  34bcd7:      	kmovq	%k0, 0x20(%rdi)
  34bcdd:      	kmovq	%k0, 0x28(%rdi)
  34bce3:      	kmovq	%k0, 0x30(%rdi)
  34bce9:      	kmovq	%k0, 0x38(%rdi)
  34bcef:      	addq	$0x40, %rdi
  34bcf3:      	cmpq	%r13, %rdi
  34bcf6:      	jne	0x34bcc0 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x150>
  34bcf8:      	cmpq	%r10, %r14
  34bcfb:      	sete	%cl
  34bcfe:      	cmpq	%r10, %r8
  34bd01:      	sete	%r9b
  34bd05:      	orb	%cl, %r9b
  34bd08:      	orb	%r15b, %r9b
  34bd0b:      	jmp	0x34c299 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x729>
  34bd10:      	movq	0x8(%rax), %r14
  34bd14:      	testb	%r9b, %r9b
  34bd17:      	je	0x34c07b <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x50b>
  34bd1d:      	movq	(%r8), %r9
  34bd20:      	xorl	%r8d, %r8d
  34bd23:      	cmpq	%r10, %r9
  34bd26:      	vpbroadcastq	%r9, %zmm0
  34bd2c:      	movl	$0xff, %r9d
  34bd32:      	cmovnel	%r8d, %r9d
  34bd36:      	kmovd	%r9d, %k0
  34bd3b:      	addq	$0x1c0, %r14            # imm = 0x1C0
  34bd42:      	vpbroadcastq	-0x2aed74(%rip), %zmm1 # 0x9cfd8 <anon.ad39dab31d93760b58da730830a5ab31.633.llvm.3251032610986828358+0x50>
  34bd4c:      	vmovdqa	-0x2c64a4(%rip), %xmm2  # 0x858b0 <anon.dfb41d6ae96b09f07f21f4b37f1918c9.68.llvm.16825879777302589835+0x280>
  34bd54:      	vmovdqa64	-0x2aac5e(%rip), %zmm3 # 0xa1100 <anon.ad39dab31d93760b58da730830a5ab31.2427.llvm.3251032610986828358+0xc20>
  34bd5e:      	vmovdqa64	-0x2aac28(%rip), %zmm4 # 0xa1140 <anon.ad39dab31d93760b58da730830a5ab31.2427.llvm.3251032610986828358+0xc60>
  34bd68:      	movl	%r15d, %r9d
  34bd6b:      	nopl	(%rax,%rax)
  34bd70:      	kmovd	%r9d, %k1
  34bd75:      	kshiftlb	$0x7, %k1, %k1
  34bd7b:      	kshiftrb	$0x7, %k1, %k1
  34bd81:      	vmovdqu64	-0x1c0(%r14), %zmm5
  34bd88:      	vmovdqu64	-0x180(%r14), %zmm6
  34bd8f:      	vmovdqu64	-0x140(%r14), %zmm7
  34bd96:      	vmovdqu64	-0x100(%r14), %zmm8
  34bd9d:      	vpcmpeqq	%zmm1, %zmm5, %k2
  34bda3:      	korb	%k1, %k2, %k1
  34bda7:      	vpcmpeqq	%zmm1, %zmm6, %k4
  34bdad:      	vpcmpeqq	%zmm1, %zmm7, %k3
  34bdb3:      	vpcmpeqq	%zmm1, %zmm8, %k2
  34bdb9:      	vmovdqu64	-0xc0(%r14), %zmm9
  34bdc0:      	vmovdqu64	-0x80(%r14), %zmm10
  34bdc7:      	vmovdqu64	-0x40(%r14), %zmm11
  34bdce:      	vmovdqu64	(%r14), %zmm12
  34bdd4:      	vpcmpeqq	%zmm1, %zmm10, %k5
  34bdda:      	korb	%k5, %k4, %k4
  34bdde:      	vpcmpeqq	%zmm1, %zmm11, %k5
  34bde4:      	korb	%k5, %k3, %k3
  34bde8:      	vpcmpeqq	%zmm1, %zmm12, %k5
  34bdee:      	korb	%k5, %k2, %k2
  34bdf2:      	vpcmpeqq	%zmm1, %zmm9, %k5
  34bdf8:      	korb	%k0, %k5, %k5
  34bdfc:      	korb	%k5, %k1, %k1
  34be00:      	korb	%k1, %k4, %k1
  34be04:      	korb	%k1, %k3, %k1
  34be08:      	kortestb	%k1, %k2
  34be0c:      	vpcmpgtq	%zmm5, %zmm0, %k1
  34be12:      	vmovdqu8	%xmm2, %xmm5 {%k1} {z}
  34be18:      	vpcmpgtq	%zmm6, %zmm0, %k1
  34be1e:      	vmovdqu8	%xmm2, %xmm6 {%k1} {z}
  34be24:      	vpcmpgtq	%zmm7, %zmm0, %k1
  34be2a:      	vmovdqu8	%xmm2, %xmm7 {%k1} {z}
  34be30:      	vpcmpgtq	%zmm8, %zmm0, %k1
  34be36:      	vpunpcklqdq	%xmm6, %xmm5, %xmm5 # xmm5 = xmm5[0],xmm6[0]
  34be3a:      	vinserti128	$0x1, %xmm7, %ymm5, %ymm5
  34be40:      	vmovdqu8	%xmm2, %xmm6 {%k1} {z}
  34be46:      	vpbroadcastq	%xmm6, %ymm6
  34be4b:      	vpblendd	$0xc0, %ymm6, %ymm5, %ymm5 # ymm5 = ymm5[0,1,2,3,4,5],ymm6[6,7]
  34be51:      	vpcmpgtq	%zmm9, %zmm0, %k1
  34be57:      	vmovdqu8	%xmm2, %xmm6 {%k1} {z}
  34be5d:      	vpcmpgtq	%zmm10, %zmm0, %k1
  34be63:      	vmovdqu8	%xmm2, %xmm7 {%k1} {z}
  34be69:      	vpcmpgtq	%zmm11, %zmm0, %k1
  34be6f:      	vmovdqu8	%xmm2, %xmm8 {%k1} {z}
  34be75:      	vpcmpgtq	%zmm12, %zmm0, %k1
  34be7b:      	vmovdqu8	%xmm2, %xmm9 {%k1} {z}
  34be81:      	vinserti32x4	$0x2, %xmm6, %zmm0, %zmm6
  34be88:      	vinserti64x4	$0x0, %ymm5, %zmm6, %zmm5
  34be8f:      	vpermt2q	%zmm7, %zmm3, %zmm5
  34be95:      	vinserti32x4	$0x3, %xmm8, %zmm5, %zmm5
  34be9c:      	vpermt2q	%zmm9, %zmm4, %zmm5
  34bea2:      	vptestmb	%zmm5, %zmm5, %k1
  34bea8:      	kmovq	%k1, (%rdi,%r8)
  34beae:      	setne	%r9b
  34beb2:      	addq	$0x8, %r8
  34beb6:      	addq	$0x200, %r14            # imm = 0x200
  34bebd:      	cmpq	%r8, %rcx
  34bec0:      	jne	0x34bd70 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x200>
  34bec6:      	jmp	0x34c299 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x729>
  34becb:      	xorl	%r8d, %r8d
  34bece:      	cmpq	%r10, %r14
  34bed1:      	movl	$0xff, %r9d
  34bed7:      	cmovnel	%r8d, %r9d
  34bedb:      	kmovd	%r9d, %k0
  34bee0:      	vpbroadcastq	%r14, %zmm0
  34bee6:      	addq	$0x1c0, %rbx            # imm = 0x1C0
  34beed:      	vpbroadcastq	-0x2aef1f(%rip), %zmm1 # 0x9cfd8 <anon.ad39dab31d93760b58da730830a5ab31.633.llvm.3251032610986828358+0x50>
  34bef7:      	vmovdqa	-0x2c664f(%rip), %xmm2  # 0x858b0 <anon.dfb41d6ae96b09f07f21f4b37f1918c9.68.llvm.16825879777302589835+0x280>
  34beff:      	vmovdqa64	-0x2aae09(%rip), %zmm3 # 0xa1100 <anon.ad39dab31d93760b58da730830a5ab31.2427.llvm.3251032610986828358+0xc20>
  34bf09:      	vmovdqa64	-0x2aadd3(%rip), %zmm4 # 0xa1140 <anon.ad39dab31d93760b58da730830a5ab31.2427.llvm.3251032610986828358+0xc60>
  34bf13:      	movl	%r15d, %r9d
  34bf16:      	nopw	%cs:(%rax,%rax)
  34bf20:      	kmovd	%r9d, %k1
  34bf25:      	kshiftlb	$0x7, %k1, %k1
  34bf2b:      	kshiftrb	$0x7, %k1, %k1
  34bf31:      	vmovdqu64	-0x1c0(%rbx), %zmm5
  34bf38:      	vmovdqu64	-0x180(%rbx), %zmm6
  34bf3f:      	vmovdqu64	-0x140(%rbx), %zmm7
  34bf46:      	vmovdqu64	-0x100(%rbx), %zmm8
  34bf4d:      	vpcmpeqq	%zmm1, %zmm5, %k2
  34bf53:      	korb	%k1, %k2, %k1
  34bf57:      	vpcmpeqq	%zmm1, %zmm6, %k4
  34bf5d:      	vpcmpeqq	%zmm1, %zmm7, %k3
  34bf63:      	vpcmpeqq	%zmm1, %zmm8, %k2
  34bf69:      	vmovdqu64	-0xc0(%rbx), %zmm9
  34bf70:      	vmovdqu64	-0x80(%rbx), %zmm10
  34bf77:      	vmovdqu64	-0x40(%rbx), %zmm11
  34bf7e:      	vmovdqu64	(%rbx), %zmm12
  34bf84:      	vpcmpeqq	%zmm1, %zmm10, %k5
  34bf8a:      	korb	%k5, %k4, %k4
  34bf8e:      	vpcmpeqq	%zmm1, %zmm11, %k5
  34bf94:      	korb	%k5, %k3, %k3
  34bf98:      	vpcmpeqq	%zmm1, %zmm12, %k5
  34bf9e:      	korb	%k5, %k2, %k2
  34bfa2:      	vpcmpeqq	%zmm1, %zmm9, %k5
  34bfa8:      	korb	%k0, %k5, %k5
  34bfac:      	korb	%k5, %k1, %k1
  34bfb0:      	korb	%k1, %k4, %k1
  34bfb4:      	korb	%k1, %k3, %k1
  34bfb8:      	kortestb	%k1, %k2
  34bfbc:      	vpcmpgtq	%zmm0, %zmm5, %k1
  34bfc2:      	vmovdqu8	%xmm2, %xmm5 {%k1} {z}
  34bfc8:      	vpcmpgtq	%zmm0, %zmm6, %k1
  34bfce:      	vmovdqu8	%xmm2, %xmm6 {%k1} {z}
  34bfd4:      	vpcmpgtq	%zmm0, %zmm7, %k1
  34bfda:      	vmovdqu8	%xmm2, %xmm7 {%k1} {z}
  34bfe0:      	vpcmpgtq	%zmm0, %zmm8, %k1
  34bfe6:      	vpunpcklqdq	%xmm6, %xmm5, %xmm5 # xmm5 = xmm5[0],xmm6[0]
  34bfea:      	vinserti128	$0x1, %xmm7, %ymm5, %ymm5
  34bff0:      	vmovdqu8	%xmm2, %xmm6 {%k1} {z}
  34bff6:      	vpbroadcastq	%xmm6, %ymm6
  34bffb:      	vpblendd	$0xc0, %ymm6, %ymm5, %ymm5 # ymm5 = ymm5[0,1,2,3,4,5],ymm6[6,7]
  34c001:      	vpcmpgtq	%zmm0, %zmm9, %k1
  34c007:      	vmovdqu8	%xmm2, %xmm6 {%k1} {z}
  34c00d:      	vpcmpgtq	%zmm0, %zmm10, %k1
  34c013:      	vmovdqu8	%xmm2, %xmm7 {%k1} {z}
  34c019:      	vpcmpgtq	%zmm0, %zmm11, %k1
  34c01f:      	vmovdqu8	%xmm2, %xmm8 {%k1} {z}
  34c025:      	vpcmpgtq	%zmm0, %zmm12, %k1
  34c02b:      	vmovdqu8	%xmm2, %xmm9 {%k1} {z}
  34c031:      	vinserti32x4	$0x2, %xmm6, %zmm0, %zmm6
  34c038:      	vinserti64x4	$0x0, %ymm5, %zmm6, %zmm5
  34c03f:      	vpermt2q	%zmm7, %zmm3, %zmm5
  34c045:      	vinserti32x4	$0x3, %xmm8, %zmm5, %zmm5
  34c04c:      	vpermt2q	%zmm9, %zmm4, %zmm5
  34c052:      	vptestmb	%zmm5, %zmm5, %k1
  34c058:      	kmovq	%k1, (%rdi,%r8)
  34c05e:      	setne	%r9b
  34c062:      	addq	$0x8, %r8
  34c066:      	addq	$0x200, %rbx            # imm = 0x200
  34c06d:      	cmpq	%r8, %rcx
  34c070:      	jne	0x34bf20 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x3b0>
  34c076:      	jmp	0x34c299 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x729>
  34c07b:      	movl	$0x1c0, %r12d           # imm = 0x1C0
  34c081:      	xorl	%r13d, %r13d
  34c084:      	vpbroadcastq	-0x2af0b6(%rip), %zmm0 # 0x9cfd8 <anon.ad39dab31d93760b58da730830a5ab31.633.llvm.3251032610986828358+0x50>
  34c08e:      	vmovdqa	-0x2c67e6(%rip), %xmm1  # 0x858b0 <anon.dfb41d6ae96b09f07f21f4b37f1918c9.68.llvm.16825879777302589835+0x280>
  34c096:      	vmovdqa64	-0x2aafa0(%rip), %zmm2 # 0xa1100 <anon.ad39dab31d93760b58da730830a5ab31.2427.llvm.3251032610986828358+0xc20>
  34c0a0:      	vmovdqa64	-0x2aaf6a(%rip), %zmm3 # 0xa1140 <anon.ad39dab31d93760b58da730830a5ab31.2427.llvm.3251032610986828358+0xc60>
  34c0aa:      	movl	%r15d, %r9d
  34c0ad:      	nopl	(%rax)
  34c0b0:      	kmovd	%r9d, %k0
  34c0b5:      	kshiftlb	$0x7, %k0, %k0
  34c0bb:      	kshiftrb	$0x7, %k0, %k0
  34c0c1:      	vmovdqu64	-0x1c0(%r14,%r12), %zmm7
  34c0c9:      	vmovdqu64	-0x180(%r14,%r12), %zmm6
  34c0d1:      	vmovdqu64	-0x140(%r14,%r12), %zmm5
  34c0d9:      	vmovdqu64	-0x100(%r14,%r12), %zmm4
  34c0e1:      	vmovdqu64	-0x1c0(%rbx,%r12), %zmm11
  34c0e9:      	vmovdqu64	-0x180(%rbx,%r12), %zmm10
  34c0f1:      	vmovdqu64	-0x140(%rbx,%r12), %zmm9
  34c0f9:      	vmovdqu64	-0x100(%rbx,%r12), %zmm8
  34c101:      	vpcmpeqq	%zmm0, %zmm7, %k1
  34c107:      	vpcmpeqq	%zmm0, %zmm6, %k2
  34c10d:      	vpcmpeqq	%zmm0, %zmm5, %k4
  34c113:      	vpcmpeqq	%zmm0, %zmm4, %k5
  34c119:      	vpcmpeqq	%zmm0, %zmm11, %k3
  34c11f:      	korb	%k3, %k1, %k1
  34c123:      	korb	%k1, %k0, %k3
  34c127:      	vpcmpeqq	%zmm0, %zmm10, %k0
  34c12d:      	korb	%k0, %k2, %k2
  34c131:      	vpcmpeqq	%zmm0, %zmm9, %k0
  34c137:      	korb	%k0, %k4, %k1
  34c13b:      	vpcmpeqq	%zmm0, %zmm8, %k0
  34c141:      	korb	%k0, %k5, %k0
  34c145:      	vmovdqu64	-0xc0(%r14,%r12), %zmm12
  34c14d:      	vmovdqu64	-0x80(%r14,%r12), %zmm13
  34c155:      	vmovdqu64	-0xc0(%rbx,%r12), %zmm14
  34c15d:      	vmovdqu64	-0x80(%rbx,%r12), %zmm15
  34c165:      	vpcmpeqq	%zmm0, %zmm12, %k4
  34c16b:      	vpcmpeqq	%zmm0, %zmm14, %k5
  34c171:      	korb	%k5, %k4, %k4
  34c175:      	vpcmpeqq	%zmm0, %zmm13, %k5
  34c17b:      	korb	%k4, %k3, %k3
  34c17f:      	vpcmpeqq	%zmm0, %zmm15, %k4
  34c185:      	korb	%k4, %k5, %k4
  34c189:      	vmovdqu64	-0x40(%r14,%r12), %zmm16
  34c191:      	vmovdqu64	-0x40(%rbx,%r12), %zmm17
  34c199:      	korb	%k4, %k2, %k2
  34c19d:      	vpcmpeqq	%zmm0, %zmm16, %k4
  34c1a3:      	korb	%k3, %k2, %k2
  34c1a7:      	vpcmpeqq	%zmm0, %zmm17, %k3
  34c1ad:      	korb	%k3, %k4, %k3
  34c1b1:      	vmovdqu64	(%r14,%r12), %zmm18
  34c1b8:      	vmovdqu64	(%rbx,%r12), %zmm19
  34c1bf:      	korb	%k3, %k1, %k1
  34c1c3:      	vpcmpeqq	%zmm0, %zmm18, %k3
  34c1c9:      	korb	%k2, %k1, %k1
  34c1cd:      	vpcmpeqq	%zmm0, %zmm19, %k2
  34c1d3:      	korb	%k2, %k3, %k2
  34c1d7:      	korb	%k2, %k0, %k0
  34c1db:      	kortestb	%k1, %k0
  34c1df:      	vpcmpgtq	%zmm7, %zmm11, %k1
  34c1e5:      	vmovdqu8	%xmm1, %xmm7 {%k1} {z}
  34c1eb:      	vpcmpgtq	%zmm6, %zmm10, %k1
  34c1f1:      	vmovdqu8	%xmm1, %xmm6 {%k1} {z}
  34c1f7:      	vpcmpgtq	%zmm5, %zmm9, %k1
  34c1fd:      	vmovdqu8	%xmm1, %xmm5 {%k1} {z}
  34c203:      	vpcmpgtq	%zmm4, %zmm8, %k1
  34c209:      	vpunpcklqdq	%xmm6, %xmm7, %xmm4 # xmm4 = xmm7[0],xmm6[0]
  34c20d:      	vinserti128	$0x1, %xmm5, %ymm4, %ymm4
  34c213:      	vmovdqu8	%xmm1, %xmm5 {%k1} {z}
  34c219:      	vpbroadcastq	%xmm5, %ymm5
  34c21e:      	vpblendd	$0xc0, %ymm5, %ymm4, %ymm4 # ymm4 = ymm4[0,1,2,3,4,5],ymm5[6,7]
  34c224:      	vpcmpgtq	%zmm12, %zmm14, %k1
  34c22a:      	vmovdqu8	%xmm1, %xmm5 {%k1} {z}
  34c230:      	vpcmpgtq	%zmm13, %zmm15, %k1
  34c236:      	vmovdqu8	%xmm1, %xmm6 {%k1} {z}
  34c23c:      	vpcmpgtq	%zmm16, %zmm17, %k1
  34c242:      	vmovdqu8	%xmm1, %xmm7 {%k1} {z}
  34c248:      	vpcmpgtq	%zmm18, %zmm19, %k1
  34c24e:      	vmovdqu8	%xmm1, %xmm8 {%k1} {z}
  34c254:      	vinserti32x4	$0x2, %xmm5, %zmm0, %zmm5
  34c25b:      	vinserti64x4	$0x0, %ymm4, %zmm5, %zmm4
  34c262:      	vpermt2q	%zmm6, %zmm2, %zmm4
  34c268:      	vinserti32x4	$0x3, %xmm7, %zmm4, %zmm4
  34c26f:      	vpermt2q	%zmm8, %zmm3, %zmm4
  34c275:      	vptestmb	%zmm4, %zmm4, %k0
  34c27b:      	kmovq	%k0, (%rdi,%r13)
  34c281:      	setne	%r9b
  34c285:      	addq	$0x8, %r13
  34c289:      	addq	$0x200, %r12            # imm = 0x200
  34c290:      	cmpq	%r13, %rcx
  34c293:      	jne	0x34c0b0 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x540>
  34c299:      	andb	$0x1, %r9b
  34c29d:      	movb	%r9b, 0x38(%rax)
  34c2a1:      	testq	%r11, %r11
  34c2a4:      	je	0x34cbb9 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x1049>
  34c2aa:      	movq	%rdx, %rdi
  34c2ad:      	andq	$-0x40, %rdi
  34c2b1:      	movzbl	0x38(%rax), %r13d
  34c2b6:      	movzbl	0x18(%rax), %r8d
  34c2bb:      	movq	0x20(%rax), %rbx
  34c2bf:      	movq	0x28(%rax), %rcx
  34c2c3:      	cmpl	$0x1, (%rax)
  34c2c6:      	jne	0x34c2ef <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x77f>
  34c2c8:      	movq	0x10(%rax), %r9
  34c2cc:      	movq	(%r9), %r14
  34c2cf:      	testb	%r8b, %r8b
  34c2d2:      	je	0x34c312 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x7a2>
  34c2d4:      	movq	(%rcx), %rdi
  34c2d7:      	xorl	%ebx, %ebx
  34c2d9:      	cmpq	%rdi, %r14
  34c2dc:      	setl	%bl
  34c2df:      	cmpl	$0x4, %r11d
  34c2e3:      	jae	0x34c340 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x7d0>
  34c2e5:      	xorl	%r15d, %r15d
  34c2e8:      	xorl	%ecx, %ecx
  34c2ea:      	jmp	0x34c4a0 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x930>
  34c2ef:      	movq	0x8(%rax), %r14
  34c2f3:      	testb	%r8b, %r8b
  34c2f6:      	je	0x34c329 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x7b9>
  34c2f8:      	movq	(%rcx), %rbx
  34c2fb:      	cmpl	$0x4, %r11d
  34c2ff:      	jae	0x34c350 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x7e0>
  34c301:      	xorl	%r15d, %r15d
  34c304:      	movl	%r13d, %r12d
  34c307:      	xorl	%ecx, %ecx
  34c309:      	movq	-0x30(%rbp), %r13
  34c30d:      	jmp	0x34c6d2 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0xb62>
  34c312:      	cmpl	$0x4, %r11d
  34c316:      	jae	0x34c36e <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x7fe>
  34c318:      	xorl	%r15d, %r15d
  34c31b:      	movl	%r13d, %r12d
  34c31e:      	xorl	%ecx, %ecx
  34c320:      	movq	-0x30(%rbp), %r13
  34c324:      	jmp	0x34c922 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0xdb2>
  34c329:      	cmpl	$0x4, %r11d
  34c32d:      	jae	0x34c38c <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x81c>
  34c32f:      	xorl	%r15d, %r15d
  34c332:      	movl	%r13d, %r12d
  34c335:      	xorl	%ecx, %ecx
  34c337:      	movq	-0x30(%rbp), %r13
  34c33b:      	jmp	0x34cb5c <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0xfec>
  34c340:      	cmpl	$0x10, %r11d
  34c344:      	jae	0x34c3a7 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x837>
  34c346:      	xorl	%ecx, %ecx
  34c348:      	xorl	%r15d, %r15d
  34c34b:      	jmp	0x34c43f <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x8cf>
  34c350:      	xorl	%r8d, %r8d
  34c353:      	cmpl	$0x10, %r11d
  34c357:      	jae	0x34c4cd <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x95d>
  34c35d:      	xorl	%ecx, %ecx
  34c35f:      	movl	%r13d, %r12d
  34c362:      	xorl	%r15d, %r15d
  34c365:      	movq	-0x30(%rbp), %r13
  34c369:      	jmp	0x34c609 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0xa99>
  34c36e:      	xorl	%r8d, %r8d
  34c371:      	cmpl	$0x10, %r11d
  34c375:      	jae	0x34c71c <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0xbac>
  34c37b:      	xorl	%ecx, %ecx
  34c37d:      	movl	%r13d, %r12d
  34c380:      	xorl	%r15d, %r15d
  34c383:      	movq	-0x30(%rbp), %r13
  34c387:      	jmp	0x34c859 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0xce9>
  34c38c:      	cmpl	$0x10, %r11d
  34c390:      	jae	0x34c96a <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0xdfa>
  34c396:      	xorl	%ecx, %ecx
  34c398:      	movl	%r13d, %r12d
  34c39b:      	xorl	%r15d, %r15d
  34c39e:      	movq	-0x30(%rbp), %r13
  34c3a2:      	jmp	0x34caab <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0xf3b>
  34c3a7:      	movl	%edx, %ecx
  34c3a9:      	andl	$0x30, %ecx
  34c3ac:      	vpbroadcastq	%rbx, %zmm0
  34c3b2:      	vmovdqa64	-0x2ab23c(%rip), %zmm2 # 0xa1180 <anon.ad39dab31d93760b58da730830a5ab31.2427.llvm.3251032610986828358+0xca0>
  34c3bc:      	vpxor	%xmm1, %xmm1, %xmm1
  34c3c0:      	vpbroadcastq	-0x2af942(%rip), %zmm3 # 0x9ca88 <anon.488c77b9dbf8e19ed178bcd3c2303ecd.415.llvm.3849972704292181595>
  34c3ca:      	vpbroadcastq	-0x2aed4c(%rip), %zmm4 # 0x9d688 <anon.ad39dab31d93760b58da730830a5ab31.728.llvm.3251032610986828358+0x40>
  34c3d4:      	movq	%rcx, %r8
  34c3d7:      	vpxor	%xmm5, %xmm5, %xmm5
  34c3db:      	nopl	(%rax,%rax)
  34c3e0:      	vpaddq	%zmm3, %zmm2, %zmm6
  34c3e6:      	vpsllvq	%zmm2, %zmm0, %zmm7
  34c3ec:      	vporq	%zmm1, %zmm7, %zmm1
  34c3f2:      	vpsllvq	%zmm6, %zmm0, %zmm6
  34c3f8:      	vporq	%zmm5, %zmm6, %zmm5
  34c3fe:      	vpaddq	%zmm4, %zmm2, %zmm2
  34c404:      	addq	$-0x10, %r8
  34c408:      	jne	0x34c3e0 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x870>
  34c40a:      	vporq	%zmm1, %zmm5, %zmm0
  34c410:      	vextracti64x4	$0x1, %zmm0, %ymm1
  34c417:      	vporq	%zmm1, %zmm0, %zmm0
  34c41d:      	vextracti128	$0x1, %ymm0, %xmm1
  34c423:      	vpor	%xmm1, %xmm0, %xmm0
  34c427:      	vpshufd	$0xee, %xmm0, %xmm1     # xmm1 = xmm0[2,3,2,3]
  34c42c:      	vpor	%xmm1, %xmm0, %xmm0
  34c430:      	vmovq	%xmm0, %r15
  34c435:      	cmpl	%ecx, %r11d
  34c438:      	je	0x34c4b1 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x941>
  34c43a:      	testb	$0xc, %dl
  34c43d:      	je	0x34c4a0 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x930>
  34c43f:      	movq	%rcx, %r8
  34c442:      	movl	%edx, %ecx
  34c444:      	andl	$0x3c, %ecx
  34c447:      	vmovq	%r15, %xmm0
  34c44c:      	vpbroadcastq	%r8, %ymm1
  34c452:      	vpor	-0x2aa7da(%rip), %ymm1, %ymm1 # 0xa1c80 <anon.52176a6184fe66ae916e776d42656f76.6.llvm.2286782273685313781+0xa0>
  34c45a:      	vpbroadcastq	%rbx, %ymm2
  34c460:      	subq	%rcx, %r8
  34c463:      	vpbroadcastq	-0x2aee4c(%rip), %ymm3 # 0x9d620 <anon.488c77b9dbf8e19ed178bcd3c2303ecd.416.llvm.3849972704292181595>
  34c46c:      	nopl	(%rax)
  34c470:      	vpsllvq	%ymm1, %ymm2, %ymm4
  34c475:      	vpor	%ymm0, %ymm4, %ymm0
  34c479:      	vpaddq	%ymm3, %ymm1, %ymm1
  34c47d:      	addq	$0x4, %r8
  34c481:      	jne	0x34c470 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x900>
  34c483:      	vextracti128	$0x1, %ymm0, %xmm1
  34c489:      	vpor	%xmm1, %xmm0, %xmm0
  34c48d:      	vpshufd	$0xee, %xmm0, %xmm1     # xmm1 = xmm0[2,3,2,3]
  34c492:      	vpor	%xmm1, %xmm0, %xmm0
  34c496:      	vmovq	%xmm0, %r15
  34c49b:      	cmpl	%ecx, %r11d
  34c49e:      	je	0x34c4b1 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x941>
  34c4a0:      	movq	%rbx, %rdx
  34c4a3:      	shlq	%cl, %rdx
  34c4a6:      	incq	%rcx
  34c4a9:      	orq	%rdx, %r15
  34c4ac:      	cmpq	%rcx, %r11
  34c4af:      	jne	0x34c4a0 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x930>
  34c4b1:      	cmpq	%r10, %r14
  34c4b4:      	sete	%cl
  34c4b7:      	cmpq	%r10, %rdi
  34c4ba:      	sete	%r12b
  34c4be:      	orb	%cl, %r12b
  34c4c1:      	orb	%r13b, %r12b
  34c4c4:      	movq	-0x30(%rbp), %r13
  34c4c8:      	jmp	0x34cba4 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x1034>
  34c4cd:      	movl	%edx, %ecx
  34c4cf:      	andl	$0x30, %ecx
  34c4d2:      	cmpq	%r10, %rbx
  34c4d5:      	kmovd	%r13d, %k0
  34c4da:      	kshiftlb	$0x7, %k0, %k0
  34c4e0:      	kshiftrb	$0x7, %k0, %k0
  34c4e6:      	vpbroadcastq	%rbx, %zmm0
  34c4ec:      	movl	$0xff, %r9d
  34c4f2:      	cmovnel	%r8d, %r9d
  34c4f6:      	kmovd	%r9d, %k1
  34c4fb:      	leaq	(%r14,%rdi,8), %r9
  34c4ff:      	vmovdqa64	-0x2ab389(%rip), %zmm2 # 0xa1180 <anon.ad39dab31d93760b58da730830a5ab31.2427.llvm.3251032610986828358+0xca0>
  34c509:      	vpxor	%xmm1, %xmm1, %xmm1
  34c50d:      	kxorb	%k0, %k0, %k2
  34c511:      	vpbroadcastq	-0x2afa93(%rip), %zmm3 # 0x9ca88 <anon.488c77b9dbf8e19ed178bcd3c2303ecd.415.llvm.3849972704292181595>
  34c51b:      	vpbroadcastq	-0x2af54d(%rip), %zmm4 # 0x9cfd8 <anon.ad39dab31d93760b58da730830a5ab31.633.llvm.3251032610986828358+0x50>
  34c525:      	vpbroadcastq	-0x2aeea7(%rip), %zmm5 # 0x9d688 <anon.ad39dab31d93760b58da730830a5ab31.728.llvm.3251032610986828358+0x40>
  34c52f:      	movq	%rcx, %r15
  34c532:      	vpxor	%xmm6, %xmm6, %xmm6
  34c536:      	nopw	%cs:(%rax,%rax)
  34c540:      	vpaddq	%zmm3, %zmm2, %zmm7
  34c546:      	vmovq	%xmm2, %r12
  34c54b:      	vmovdqu64	(%r9,%r12,8), %zmm8
  34c552:      	vmovdqu64	0x40(%r9,%r12,8), %zmm9
  34c55a:      	vpcmpeqq	%zmm4, %zmm8, %k4
  34c560:      	vpcmpeqq	%zmm4, %zmm9, %k3
  34c566:      	korb	%k2, %k3, %k3
  34c56a:      	vpcmpgtq	%zmm8, %zmm0, %k5
  34c570:      	vpcmpgtq	%zmm9, %zmm0, %k6
  34c576:      	korb	%k1, %k0, %k0
  34c57a:      	korb	%k0, %k4, %k0
  34c57e:      	korb	%k1, %k3, %k2
  34c582:      	vpmovm2q	%k5, %zmm8
  34c588:      	vpsrlq	$0x3f, %zmm8, %zmm8
  34c58f:      	vpmovm2q	%k6, %zmm9
  34c595:      	vpsrlq	$0x3f, %zmm9, %zmm9
  34c59c:      	vpsllvq	%zmm7, %zmm9, %zmm7
  34c5a2:      	vporq	%zmm6, %zmm7, %zmm6
  34c5a8:      	vpsllvq	%zmm2, %zmm8, %zmm7
  34c5ae:      	vporq	%zmm1, %zmm7, %zmm1
  34c5b4:      	vpaddq	%zmm5, %zmm2, %zmm2
  34c5ba:      	addq	$-0x10, %r15
  34c5be:      	jne	0x34c540 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x9d0>
  34c5c0:      	kortestb	%k0, %k3
  34c5c4:      	setne	%r12b
  34c5c8:      	vporq	%zmm1, %zmm6, %zmm0
  34c5ce:      	vextracti64x4	$0x1, %zmm0, %ymm1
  34c5d5:      	vporq	%zmm1, %zmm0, %zmm0
  34c5db:      	vextracti128	$0x1, %ymm0, %xmm1
  34c5e1:      	vpor	%xmm1, %xmm0, %xmm0
  34c5e5:      	vpshufd	$0xee, %xmm0, %xmm1     # xmm1 = xmm0[2,3,2,3]
  34c5ea:      	vpor	%xmm1, %xmm0, %xmm0
  34c5ee:      	vmovq	%xmm0, %r15
  34c5f3:      	cmpl	%ecx, %r11d
  34c5f6:      	movq	-0x30(%rbp), %r13
  34c5fa:      	je	0x34cba4 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x1034>
  34c600:      	testb	$0xc, %dl
  34c603:      	je	0x34c6d2 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0xb62>
  34c609:      	movq	%rcx, %r9
  34c60c:      	movl	%edx, %ecx
  34c60e:      	andl	$0x3c, %ecx
  34c611:      	cmpq	%r10, %rbx
  34c614:      	kmovd	%r12d, %k0
  34c619:      	kshiftlb	$0x7, %k0, %k0
  34c61f:      	kshiftrb	$0x7, %k0, %k0
  34c625:      	vmovq	%r15, %xmm0
  34c62a:      	vpbroadcastq	%rbx, %ymm1
  34c630:      	movl	$0xff, %edx
  34c635:      	cmovel	%edx, %r8d
  34c639:      	kmovd	%r8d, %k1
  34c63e:      	vpbroadcastq	%r9, %ymm2
  34c644:      	vpor	-0x2aa9cc(%rip), %ymm2, %ymm2 # 0xa1c80 <anon.52176a6184fe66ae916e776d42656f76.6.llvm.2286782273685313781+0xa0>
  34c64c:      	leaq	(%r14,%rdi,8), %rdx
  34c650:      	subq	%rcx, %r9
  34c653:      	vpbroadcastq	-0x2af684(%rip), %ymm3 # 0x9cfd8 <anon.ad39dab31d93760b58da730830a5ab31.633.llvm.3251032610986828358+0x50>
  34c65c:      	vpbroadcastq	-0x2af045(%rip), %ymm4 # 0x9d620 <anon.488c77b9dbf8e19ed178bcd3c2303ecd.416.llvm.3849972704292181595>
  34c665:      	nopw	%cs:(%rax,%rax)
  34c670:      	vmovq	%xmm2, %r8
  34c675:      	vmovdqu	(%rdx,%r8,8), %ymm5
  34c67b:      	vpcmpeqq	%ymm3, %ymm5, %k2
  34c681:      	korw	%k1, %k2, %k2
  34c685:      	korw	%k2, %k0, %k0
  34c689:      	vpcmpgtq	%ymm5, %ymm1, %ymm5
  34c68e:      	vpsrlq	$0x3f, %ymm5, %ymm5
  34c693:      	vpsllvq	%ymm2, %ymm5, %ymm5
  34c698:      	vpor	%ymm0, %ymm5, %ymm0
  34c69c:      	vpaddq	%ymm4, %ymm2, %ymm2
  34c6a0:      	addq	$0x4, %r9
  34c6a4:      	jne	0x34c670 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0xb00>
  34c6a6:      	kmovd	%k0, %edx
  34c6aa:      	testb	$0xf, %dl
  34c6ad:      	setne	%r12b
  34c6b1:      	vextracti128	$0x1, %ymm0, %xmm1
  34c6b7:      	vpor	%xmm1, %xmm0, %xmm0
  34c6bb:      	vpshufd	$0xee, %xmm0, %xmm1     # xmm1 = xmm0[2,3,2,3]
  34c6c0:      	vpor	%xmm1, %xmm0, %xmm0
  34c6c4:      	vmovq	%xmm0, %r15
  34c6c9:      	cmpl	%ecx, %r11d
  34c6cc:      	je	0x34cba4 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x1034>
  34c6d2:      	leaq	(%r14,%rdi,8), %rdx
  34c6d6:      	movl	%r12d, %edi
  34c6d9:      	nopl	(%rax)
  34c6e0:      	cmpq	%r10, %rbx
  34c6e3:      	sete	%r12b
  34c6e7:      	movq	(%rdx,%rcx,8), %r8
  34c6eb:      	cmpq	%r10, %r8
  34c6ee:      	sete	%r9b
  34c6f2:      	xorl	%r14d, %r14d
  34c6f5:      	cmpq	%rbx, %r8
  34c6f8:      	setl	%r14b
  34c6fc:      	orb	%dil, %r12b
  34c6ff:      	orb	%r9b, %r12b
  34c702:      	leaq	0x1(%rcx), %r8
  34c706:      	shlq	%cl, %r14
  34c709:      	orq	%r14, %r15
  34c70c:      	movl	%r12d, %edi
  34c70f:      	cmpq	%r8, %r11
  34c712:      	movq	%r8, %rcx
  34c715:      	jne	0x34c6e0 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0xb70>
  34c717:      	jmp	0x34cba4 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x1034>
  34c71c:      	movl	%edx, %ecx
  34c71e:      	andl	$0x30, %ecx
  34c721:      	cmpq	%r10, %r14
  34c724:      	kmovd	%r13d, %k0
  34c729:      	kshiftlb	$0x7, %k0, %k0
  34c72f:      	kshiftrb	$0x7, %k0, %k0
  34c735:      	vpbroadcastq	%r14, %zmm0
  34c73b:      	movl	$0xff, %r9d
  34c741:      	cmovnel	%r8d, %r9d
  34c745:      	kmovd	%r9d, %k1
  34c74a:      	leaq	(%rbx,%rdi,8), %r9
  34c74e:      	vmovdqa64	-0x2ab5d8(%rip), %zmm2 # 0xa1180 <anon.ad39dab31d93760b58da730830a5ab31.2427.llvm.3251032610986828358+0xca0>
  34c758:      	vpxor	%xmm1, %xmm1, %xmm1
  34c75c:      	kxorb	%k0, %k0, %k2
  34c760:      	vpbroadcastq	-0x2afce2(%rip), %zmm3 # 0x9ca88 <anon.488c77b9dbf8e19ed178bcd3c2303ecd.415.llvm.3849972704292181595>
  34c76a:      	vpbroadcastq	-0x2af79c(%rip), %zmm4 # 0x9cfd8 <anon.ad39dab31d93760b58da730830a5ab31.633.llvm.3251032610986828358+0x50>
  34c774:      	vpbroadcastq	-0x2af0f6(%rip), %zmm5 # 0x9d688 <anon.ad39dab31d93760b58da730830a5ab31.728.llvm.3251032610986828358+0x40>
  34c77e:      	movq	%rcx, %r15
  34c781:      	vpxor	%xmm6, %xmm6, %xmm6
  34c785:      	nopw	%cs:(%rax,%rax)
  34c790:      	vpaddq	%zmm3, %zmm2, %zmm7
  34c796:      	vmovq	%xmm2, %r12
  34c79b:      	vmovdqu64	(%r9,%r12,8), %zmm8
  34c7a2:      	vmovdqu64	0x40(%r9,%r12,8), %zmm9
  34c7aa:      	vpcmpeqq	%zmm4, %zmm8, %k4
  34c7b0:      	vpcmpeqq	%zmm4, %zmm9, %k3
  34c7b6:      	korb	%k2, %k3, %k3
  34c7ba:      	vpcmpgtq	%zmm0, %zmm8, %k5
  34c7c0:      	vpcmpgtq	%zmm0, %zmm9, %k6
  34c7c6:      	korb	%k1, %k0, %k0
  34c7ca:      	korb	%k0, %k4, %k0
  34c7ce:      	korb	%k1, %k3, %k2
  34c7d2:      	vpmovm2q	%k5, %zmm8
  34c7d8:      	vpsrlq	$0x3f, %zmm8, %zmm8
  34c7df:      	vpmovm2q	%k6, %zmm9
  34c7e5:      	vpsrlq	$0x3f, %zmm9, %zmm9
  34c7ec:      	vpsllvq	%zmm7, %zmm9, %zmm7
  34c7f2:      	vporq	%zmm6, %zmm7, %zmm6
  34c7f8:      	vpsllvq	%zmm2, %zmm8, %zmm7
  34c7fe:      	vporq	%zmm1, %zmm7, %zmm1
  34c804:      	vpaddq	%zmm5, %zmm2, %zmm2
  34c80a:      	addq	$-0x10, %r15
  34c80e:      	jne	0x34c790 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0xc20>
  34c810:      	kortestb	%k0, %k3
  34c814:      	setne	%r12b
  34c818:      	vporq	%zmm1, %zmm6, %zmm0
  34c81e:      	vextracti64x4	$0x1, %zmm0, %ymm1
  34c825:      	vporq	%zmm1, %zmm0, %zmm0
  34c82b:      	vextracti128	$0x1, %ymm0, %xmm1
  34c831:      	vpor	%xmm1, %xmm0, %xmm0
  34c835:      	vpshufd	$0xee, %xmm0, %xmm1     # xmm1 = xmm0[2,3,2,3]
  34c83a:      	vpor	%xmm1, %xmm0, %xmm0
  34c83e:      	vmovq	%xmm0, %r15
  34c843:      	cmpl	%ecx, %r11d
  34c846:      	movq	-0x30(%rbp), %r13
  34c84a:      	je	0x34cba4 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x1034>
  34c850:      	testb	$0xc, %dl
  34c853:      	je	0x34c922 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0xdb2>
  34c859:      	movq	%rcx, %r9
  34c85c:      	movl	%edx, %ecx
  34c85e:      	andl	$0x3c, %ecx
  34c861:      	cmpq	%r10, %r14
  34c864:      	kmovd	%r12d, %k0
  34c869:      	kshiftlb	$0x7, %k0, %k0
  34c86f:      	kshiftrb	$0x7, %k0, %k0
  34c875:      	vmovq	%r15, %xmm0
  34c87a:      	vpbroadcastq	%r14, %ymm1
  34c880:      	movl	$0xff, %edx
  34c885:      	cmovel	%edx, %r8d
  34c889:      	kmovd	%r8d, %k1
  34c88e:      	vpbroadcastq	%r9, %ymm2
  34c894:      	vpor	-0x2aac1c(%rip), %ymm2, %ymm2 # 0xa1c80 <anon.52176a6184fe66ae916e776d42656f76.6.llvm.2286782273685313781+0xa0>
  34c89c:      	leaq	(%rbx,%rdi,8), %rdx
  34c8a0:      	subq	%rcx, %r9
  34c8a3:      	vpbroadcastq	-0x2af8d4(%rip), %ymm3 # 0x9cfd8 <anon.ad39dab31d93760b58da730830a5ab31.633.llvm.3251032610986828358+0x50>
  34c8ac:      	vpbroadcastq	-0x2af295(%rip), %ymm4 # 0x9d620 <anon.488c77b9dbf8e19ed178bcd3c2303ecd.416.llvm.3849972704292181595>
  34c8b5:      	nopw	%cs:(%rax,%rax)
  34c8c0:      	vmovq	%xmm2, %r8
  34c8c5:      	vmovdqu	(%rdx,%r8,8), %ymm5
  34c8cb:      	vpcmpeqq	%ymm3, %ymm5, %k2
  34c8d1:      	korw	%k0, %k2, %k0
  34c8d5:      	korw	%k1, %k0, %k0
  34c8d9:      	vpcmpgtq	%ymm1, %ymm5, %ymm5
  34c8de:      	vpsrlq	$0x3f, %ymm5, %ymm5
  34c8e3:      	vpsllvq	%ymm2, %ymm5, %ymm5
  34c8e8:      	vpor	%ymm0, %ymm5, %ymm0
  34c8ec:      	vpaddq	%ymm4, %ymm2, %ymm2
  34c8f0:      	addq	$0x4, %r9
  34c8f4:      	jne	0x34c8c0 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0xd50>
  34c8f6:      	kmovd	%k0, %edx
  34c8fa:      	testb	$0xf, %dl
  34c8fd:      	setne	%r12b
  34c901:      	vextracti128	$0x1, %ymm0, %xmm1
  34c907:      	vpor	%xmm1, %xmm0, %xmm0
  34c90b:      	vpshufd	$0xee, %xmm0, %xmm1     # xmm1 = xmm0[2,3,2,3]
  34c910:      	vpor	%xmm1, %xmm0, %xmm0
  34c914:      	vmovq	%xmm0, %r15
  34c919:      	cmpl	%ecx, %r11d
  34c91c:      	je	0x34cba4 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x1034>
  34c922:      	leaq	(%rbx,%rdi,8), %rdx
  34c926:      	movl	%r12d, %edi
  34c929:      	nopl	(%rax)
  34c930:      	cmpq	%r10, %r14
  34c933:      	sete	%r12b
  34c937:      	movq	(%rdx,%rcx,8), %r8
  34c93b:      	cmpq	%r10, %r8
  34c93e:      	sete	%r9b
  34c942:      	xorl	%ebx, %ebx
  34c944:      	cmpq	%r8, %r14
  34c947:      	setl	%bl
  34c94a:      	orb	%dil, %r12b
  34c94d:      	orb	%r9b, %r12b
  34c950:      	leaq	0x1(%rcx), %r8
  34c954:      	shlq	%cl, %rbx
  34c957:      	orq	%rbx, %r15
  34c95a:      	movl	%r12d, %edi
  34c95d:      	cmpq	%r8, %r11
  34c960:      	movq	%r8, %rcx
  34c963:      	jne	0x34c930 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0xdc0>
  34c965:      	jmp	0x34cba4 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x1034>
  34c96a:      	movl	%edx, %ecx
  34c96c:      	andl	$0x30, %ecx
  34c96f:      	kmovd	%r13d, %k0
  34c974:      	kshiftlb	$0x7, %k0, %k0
  34c97a:      	kshiftrb	$0x7, %k0, %k0
  34c980:      	vmovdqa64	-0x2ab80a(%rip), %zmm1 # 0xa1180 <anon.ad39dab31d93760b58da730830a5ab31.2427.llvm.3251032610986828358+0xca0>
  34c98a:      	vpxor	%xmm0, %xmm0, %xmm0
  34c98e:      	kxorb	%k0, %k0, %k1
  34c992:      	vpbroadcastq	-0x2aff14(%rip), %zmm2 # 0x9ca88 <anon.488c77b9dbf8e19ed178bcd3c2303ecd.415.llvm.3849972704292181595>
  34c99c:      	vpbroadcastq	-0x2af9ce(%rip), %zmm3 # 0x9cfd8 <anon.ad39dab31d93760b58da730830a5ab31.633.llvm.3251032610986828358+0x50>
  34c9a6:      	vpbroadcastq	-0x2af328(%rip), %zmm4 # 0x9d688 <anon.ad39dab31d93760b58da730830a5ab31.728.llvm.3251032610986828358+0x40>
  34c9b0:      	movq	%rcx, %r8
  34c9b3:      	vpxor	%xmm5, %xmm5, %xmm5
  34c9b7:      	nopw	(%rax,%rax)
  34c9c0:      	vpaddq	%zmm2, %zmm1, %zmm6
  34c9c6:      	vmovq	%xmm1, %r9
  34c9cb:      	addq	%rdi, %r9
  34c9ce:      	vmovdqu64	(%r14,%r9,8), %zmm7
  34c9d5:      	vmovdqu64	0x40(%r14,%r9,8), %zmm8
  34c9dd:      	vmovdqu64	(%rbx,%r9,8), %zmm9
  34c9e4:      	vmovdqu64	0x40(%rbx,%r9,8), %zmm10
  34c9ec:      	vpcmpeqq	%zmm3, %zmm7, %k2
  34c9f2:      	vpcmpeqq	%zmm3, %zmm8, %k3
  34c9f8:      	vpcmpeqq	%zmm3, %zmm9, %k4
  34c9fe:      	korb	%k4, %k2, %k2
  34ca02:      	korb	%k2, %k0, %k0
  34ca06:      	vpcmpeqq	%zmm3, %zmm10, %k2
  34ca0c:      	korb	%k2, %k3, %k2
  34ca10:      	korb	%k2, %k1, %k1
  34ca14:      	vpcmpgtq	%zmm7, %zmm9, %k2
  34ca1a:      	vpcmpgtq	%zmm8, %zmm10, %k3
  34ca20:      	vpmovm2q	%k2, %zmm7
  34ca26:      	vpsrlq	$0x3f, %zmm7, %zmm7
  34ca2d:      	vpmovm2q	%k3, %zmm8
  34ca33:      	vpsrlq	$0x3f, %zmm8, %zmm8
  34ca3a:      	vpsllvq	%zmm6, %zmm8, %zmm6
  34ca40:      	vporq	%zmm5, %zmm6, %zmm5
  34ca46:      	vpsllvq	%zmm1, %zmm7, %zmm6
  34ca4c:      	vporq	%zmm0, %zmm6, %zmm0
  34ca52:      	vpaddq	%zmm4, %zmm1, %zmm1
  34ca58:      	addq	$-0x10, %r8
  34ca5c:      	jne	0x34c9c0 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0xe50>
  34ca62:      	kortestb	%k0, %k1
  34ca66:      	setne	%r12b
  34ca6a:      	vporq	%zmm0, %zmm5, %zmm0
  34ca70:      	vextracti64x4	$0x1, %zmm0, %ymm1
  34ca77:      	vporq	%zmm1, %zmm0, %zmm0
  34ca7d:      	vextracti128	$0x1, %ymm0, %xmm1
  34ca83:      	vpor	%xmm1, %xmm0, %xmm0
  34ca87:      	vpshufd	$0xee, %xmm0, %xmm1     # xmm1 = xmm0[2,3,2,3]
  34ca8c:      	vpor	%xmm1, %xmm0, %xmm0
  34ca90:      	vmovq	%xmm0, %r15
  34ca95:      	cmpl	%ecx, %r11d
  34ca98:      	movq	-0x30(%rbp), %r13
  34ca9c:      	je	0x34cba4 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x1034>
  34caa2:      	testb	$0xc, %dl
  34caa5:      	je	0x34cb5c <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0xfec>
  34caab:      	movq	%rcx, %r8
  34caae:      	movl	%edx, %ecx
  34cab0:      	andl	$0x3c, %ecx
  34cab3:      	kmovd	%r12d, %k0
  34cab8:      	kshiftlb	$0x7, %k0, %k0
  34cabe:      	kshiftrb	$0x7, %k0, %k0
  34cac4:      	vmovq	%r15, %xmm0
  34cac9:      	vpbroadcastq	%r8, %ymm1
  34cacf:      	vpor	-0x2aae57(%rip), %ymm1, %ymm1 # 0xa1c80 <anon.52176a6184fe66ae916e776d42656f76.6.llvm.2286782273685313781+0xa0>
  34cad7:      	subq	%rcx, %r8
  34cada:      	vpbroadcastq	-0x2afb0b(%rip), %ymm2 # 0x9cfd8 <anon.ad39dab31d93760b58da730830a5ab31.633.llvm.3251032610986828358+0x50>
  34cae3:      	vpbroadcastq	-0x2af4cc(%rip), %ymm3 # 0x9d620 <anon.488c77b9dbf8e19ed178bcd3c2303ecd.416.llvm.3849972704292181595>
  34caec:      	nopl	(%rax)
  34caf0:      	vmovq	%xmm1, %rdx
  34caf5:      	addq	%rdi, %rdx
  34caf8:      	vmovdqu	(%r14,%rdx,8), %ymm4
  34cafe:      	vmovdqu	(%rbx,%rdx,8), %ymm5
  34cb03:      	vpcmpeqq	%ymm2, %ymm4, %k1
  34cb09:      	vpcmpeqq	%ymm2, %ymm5, %k2
  34cb0f:      	korw	%k2, %k1, %k1
  34cb13:      	korw	%k1, %k0, %k0
  34cb17:      	vpcmpgtq	%ymm4, %ymm5, %ymm4
  34cb1c:      	vpsrlq	$0x3f, %ymm4, %ymm4
  34cb21:      	vpsllvq	%ymm1, %ymm4, %ymm4
  34cb26:      	vpor	%ymm0, %ymm4, %ymm0
  34cb2a:      	vpaddq	%ymm3, %ymm1, %ymm1
  34cb2e:      	addq	$0x4, %r8
  34cb32:      	jne	0x34caf0 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0xf80>
  34cb34:      	kmovd	%k0, %edx
  34cb38:      	testb	$0xf, %dl
  34cb3b:      	setne	%r12b
  34cb3f:      	vextracti128	$0x1, %ymm0, %xmm1
  34cb45:      	vpor	%xmm1, %xmm0, %xmm0
  34cb49:      	vpshufd	$0xee, %xmm0, %xmm1     # xmm1 = xmm0[2,3,2,3]
  34cb4e:      	vpor	%xmm1, %xmm0, %xmm0
  34cb52:      	vmovq	%xmm0, %r15
  34cb57:      	cmpl	%ecx, %r11d
  34cb5a:      	je	0x34cba4 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x1034>
  34cb5c:      	shlq	$0x3, %rdi
  34cb60:      	addq	%rdi, %rbx
  34cb63:      	addq	%rdi, %r14
  34cb66:      	nopw	%cs:(%rax,%rax)
  34cb70:      	movq	(%r14,%rcx,8), %rdx
  34cb74:      	movq	(%rbx,%rcx,8), %rdi
  34cb78:      	cmpq	%r10, %rdx
  34cb7b:      	sete	%r8b
  34cb7f:      	cmpq	%r10, %rdi
  34cb82:      	sete	%r9b
  34cb86:      	orb	%r8b, %r9b
  34cb89:      	xorl	%r8d, %r8d
  34cb8c:      	cmpq	%rdi, %rdx
  34cb8f:      	setl	%r8b
  34cb93:      	orb	%r9b, %r12b
  34cb96:      	shlq	%cl, %r8
  34cb99:      	incq	%rcx
  34cb9c:      	orq	%r8, %r15
  34cb9f:      	cmpq	%rcx, %r11
  34cba2:      	jne	0x34cb70 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x1000>
  34cba4:      	andb	$0x1, %r12b
  34cba8:      	movb	%r12b, 0x38(%rax)
  34cbac:      	movq	-0x38(%rbp), %rax
  34cbb0:      	cmpq	%rax, %rsi
  34cbb3:      	jae	0x34cbdd <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x106d>
  34cbb5:      	movq	%r15, (%r13)
  34cbb9:      	addq	$0x18, %rsp
  34cbbd:      	popq	%rbx
  34cbbe:      	popq	%r12
  34cbc0:      	popq	%r13
  34cbc2:      	popq	%r14
  34cbc4:      	popq	%r15
  34cbc6:      	popq	%rbp
  34cbc7:      	vzeroupper
  34cbca:      	retq
  34cbcb:      	leaq	0xf76d2e(%rip), %rcx    # 0x12c3900 <alloc_ebbd31bf42d91f579e6aa57e93569175.llvm.7192105927927146806>
  34cbd2:      	xorl	%edi, %edi
  34cbd4:      	movq	%r8, %rdx
  34cbd7:      	callq	*0xfd3bf3(%rip)         # 0x13207d0 <writev+0x13207d0>
  34cbdd:      	leaq	0xf76d04(%rip), %rdx    # 0x12c38e8 <alloc_1b2922caae1b461da04857d3d02eae5b.llvm.7192105927927146806>
  34cbe4:      	movq	%rsi, %rdi
  34cbe7:      	movq	%rax, %rsi
  34cbea:      	vzeroupper
  34cbed:      	callq	*0xfd39b5(%rip)         # 0x13205a8 <writev+0x13205a8>
