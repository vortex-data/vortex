
/tmp/row-fn-bool-linux-state-avx2-target/x86_64-unknown-linux-gnu/release/deps/row_fn_bool_retry-3e4c315823f8823b:	file format elf64-x86-64

Disassembly of section .text:

0000000000335a90 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_>:
  335a90:      	pushq	%rbp
  335a91:      	movq	%rsp, %rbp
  335a94:      	pushq	%r15
  335a96:      	pushq	%r14
  335a98:      	pushq	%r13
  335a9a:      	pushq	%r12
  335a9c:      	pushq	%rbx
  335a9d:      	subq	$0x38, %rsp
  335aa1:      	movq	%rsi, %r9
  335aa4:      	movq	%rdx, %rsi
  335aa7:      	shrq	$0x6, %rsi
  335aab:      	cmpq	%r9, %rsi
  335aae:      	ja	0x336b62 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x10d2>
  335ab4:      	movq	%rcx, %r8
  335ab7:      	movl	%edx, %r11d
  335aba:      	andl	$0x3f, %r11d
  335abe:      	movabsq	$0x7fffffffffffffff, %r10 # imm = 0x7FFFFFFFFFFFFFFF
  335ac8:      	leaq	(%rdi,%rsi,8), %rax
  335acc:      	movq	%rax, -0x38(%rbp)
  335ad0:      	testq	%rsi, %rsi
  335ad3:      	movq	%rsi, -0x58(%rbp)
  335ad7:      	movq	%rcx, -0x48(%rbp)
  335adb:      	je	0x336210 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x780>
  335ae1:      	movq	%rdx, -0x30(%rbp)
  335ae5:      	leaq	(,%rsi,8), %rcx
  335aed:      	movzbl	0x18(%r8), %edx
  335af2:      	movq	0x20(%r8), %rbx
  335af6:      	movq	0x28(%r8), %rax
  335afa:      	movzbl	0x38(%r8), %r15d
  335aff:      	cmpl	$0x1, (%r8)
  335b03:      	movq	%r9, -0x50(%rbp)
  335b07:      	jne	0x335be4 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x154>
  335b0d:      	movq	0x10(%r8), %rsi
  335b11:      	movq	(%rsi), %r14
  335b14:      	testb	%dl, %dl
  335b16:      	je	0x335da2 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x312>
  335b1c:      	movq	(%rax), %rdx
  335b1f:      	xorl	%eax, %eax
  335b21:      	movq	%rdx, -0x40(%rbp)
  335b25:      	cmpq	%rdx, %r14
  335b28:      	movabsq	$0x101010101010101, %rbx # imm = 0x101010101010101
  335b32:      	cmovgeq	%rax, %rbx
  335b36:      	movq	%rbx, %r9
  335b39:      	shrq	$0x38, %r9
  335b3d:      	movq	%rbx, %r12
  335b40:      	shrq	$0x30, %r12
  335b44:      	movq	%rbx, %rdx
  335b47:      	shrq	$0x20, %rdx
  335b4b:      	movzbl	%dh, %eax
  335b4e:      	movl	%eax, %r8d
  335b51:      	movl	%ebx, %r13d
  335b54:      	shrl	$0x18, %r13d
  335b58:      	movl	%ebx, %eax
  335b5a:      	shrl	$0x10, %eax
  335b5d:      	movzbl	%bh, %esi
  335b60:      	vmovd	%ebx, %xmm0
  335b64:      	vpinsrb	$0x1, %esi, %xmm0, %xmm0
  335b6a:      	vpinsrb	$0x2, %eax, %xmm0, %xmm0
  335b70:      	vpinsrb	$0x3, %r13d, %xmm0, %xmm0
  335b76:      	vpinsrb	$0x4, %edx, %xmm0, %xmm0
  335b7c:      	vpinsrb	$0x5, %r8d, %xmm0, %xmm0
  335b82:      	vpinsrb	$0x6, %r12d, %xmm0, %xmm0
  335b88:      	vpinsrb	$0x7, %r9d, %xmm0, %xmm0
  335b8e:      	vpinsrb	$0x8, %ebx, %xmm0, %xmm0
  335b94:      	vpinsrb	$0x9, %esi, %xmm0, %xmm0
  335b9a:      	vpinsrb	$0xa, %eax, %xmm0, %xmm0
  335ba0:      	vpinsrb	$0xb, %r13d, %xmm0, %xmm0
  335ba6:      	vpinsrb	$0xc, %edx, %xmm0, %xmm0
  335bac:      	vpinsrb	$0xd, %r8d, %xmm0, %xmm0
  335bb2:      	vpinsrb	$0xe, %r12d, %xmm0, %xmm0
  335bb8:      	vpinsrb	$0xf, %r9d, %xmm0, %xmm0
  335bbe:      	vshufi64x2	$0x0, %zmm0, %zmm0, %zmm0 # zmm0 = zmm0[0,1,0,1,0,1,0,1]
  335bc5:      	addq	$-0x8, %rcx
  335bc9:      	movl	%ecx, %eax
  335bcb:      	notl	%eax
  335bcd:      	vptestmb	%zmm0, %zmm0, %k0
  335bd3:      	testb	$0x38, %al
  335bd5:      	jne	0x33617f <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x6ef>
  335bdb:      	movq	-0x38(%rbp), %rdx
  335bdf:      	jmp	0x33619e <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x70e>
  335be4:      	movq	0x8(%r8), %r14
  335be8:      	testb	%dl, %dl
  335bea:      	je	0x335f52 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x4c2>
  335bf0:      	movq	(%rax), %rax
  335bf3:      	xorl	%r8d, %r8d
  335bf6:      	cmpq	%r10, %rax
  335bf9:      	vpbroadcastq	%rax, %zmm0
  335bff:      	movl	$0xff, %eax
  335c04:      	cmovnel	%r8d, %eax
  335c08:      	kmovd	%eax, %k0
  335c0c:      	addq	$0x1c0, %r14            # imm = 0x1C0
  335c13:      	vpbroadcastq	-0x299d05(%rip), %zmm1 # 0x9bf18 <anon.ad39dab31d93760b58da730830a5ab31.633.llvm.6451718222114420760+0x50>
  335c1d:      	vmovdqa64	-0x294527(%rip), %zmm2 # 0xa1700 <alloc_cbda3acf5ff4808ffe121d88bd143ef4.llvm.9723113569546643380+0x72>
  335c27:      	vmovdqa64	-0x2944f1(%rip), %zmm3 # 0xa1740 <alloc_cbda3acf5ff4808ffe121d88bd143ef4.llvm.9723113569546643380+0xb2>
  335c31:      	vmovdqa64	-0x2944bb(%rip), %zmm4 # 0xa1780 <alloc_cbda3acf5ff4808ffe121d88bd143ef4.llvm.9723113569546643380+0xf2>
  335c3b:      	movl	%r15d, %r9d
  335c3e:      	movq	-0x30(%rbp), %rdx
  335c42:      	nopw	%cs:(%rax,%rax)
  335c50:      	andl	$0x1, %r9d
  335c54:      	kmovw	%r9d, %k4
  335c59:      	vmovdqu64	-0x1c0(%r14), %zmm5
  335c60:      	vmovdqu64	-0x180(%r14), %zmm6
  335c67:      	vmovdqu64	-0x140(%r14), %zmm7
  335c6e:      	vmovdqu64	-0x100(%r14), %zmm8
  335c75:      	vpcmpeqq	%zmm1, %zmm5, %k5
  335c7b:      	vpcmpeqq	%zmm1, %zmm6, %k3
  335c81:      	vpcmpeqq	%zmm1, %zmm7, %k2
  335c87:      	vpcmpeqq	%zmm1, %zmm8, %k1
  335c8d:      	korw	%k4, %k5, %k4
  335c91:      	vpcmpgtq	%zmm5, %zmm0, %k5
  335c97:      	vmovdqu8	%zmm2, %zmm5 {%k5} {z}
  335c9d:      	vpcmpgtq	%zmm6, %zmm0, %k5
  335ca3:      	vmovdqu8	%zmm2, %zmm6 {%k5} {z}
  335ca9:      	vpcmpgtq	%zmm7, %zmm0, %k5
  335caf:      	vmovdqu8	%zmm2, %zmm7 {%k5} {z}
  335cb5:      	vpcmpgtq	%zmm8, %zmm0, %k5
  335cbb:      	vmovdqu8	%zmm2, %zmm8 {%k5} {z}
  335cc1:      	vmovdqu64	-0xc0(%r14), %zmm9
  335cc8:      	vmovdqu64	-0x80(%r14), %zmm10
  335ccf:      	vmovdqu64	-0x40(%r14), %zmm11
  335cd6:      	vmovdqu64	(%r14), %zmm12
  335cdc:      	vpcmpeqq	%zmm1, %zmm9, %k5
  335ce2:      	korw	%k5, %k4, %k4
  335ce6:      	vpcmpeqq	%zmm1, %zmm10, %k5
  335cec:      	korw	%k5, %k3, %k3
  335cf0:      	vpcmpeqq	%zmm1, %zmm11, %k5
  335cf6:      	korw	%k5, %k2, %k2
  335cfa:      	vpcmpeqq	%zmm1, %zmm12, %k5
  335d00:      	korw	%k5, %k1, %k1
  335d04:      	vpcmpgtq	%zmm9, %zmm0, %k5
  335d0a:      	vmovdqu8	%zmm2, %zmm9 {%k5} {z}
  335d10:      	vpcmpgtq	%zmm10, %zmm0, %k5
  335d16:      	vmovdqu8	%zmm2, %zmm10 {%k5} {z}
  335d1c:      	vpcmpgtq	%zmm11, %zmm0, %k5
  335d22:      	vmovdqu8	%zmm2, %zmm11 {%k5} {z}
  335d28:      	vpcmpgtq	%zmm12, %zmm0, %k5
  335d2e:      	vmovdqu8	%zmm2, %zmm12 {%k5} {z}
  335d34:      	korw	%k0, %k4, %k4
  335d38:      	korw	%k4, %k3, %k3
  335d3c:      	korw	%k3, %k2, %k2
  335d40:      	korw	%k2, %k1, %k1
  335d44:      	vpunpcklqdq	%xmm6, %xmm5, %xmm5 # xmm5 = xmm5[0],xmm6[0]
  335d48:      	vinserti128	$0x1, %xmm7, %ymm5, %ymm5
  335d4e:      	vpbroadcastq	%xmm8, %ymm6
  335d53:      	vpblendd	$0xc0, %ymm6, %ymm5, %ymm5 # ymm5 = ymm5[0,1,2,3,4,5],ymm6[6,7]
  335d59:      	vinserti64x4	$0x1, %ymm9, %zmm5, %zmm5
  335d60:      	vpermt2q	%zmm10, %zmm3, %zmm5
  335d66:      	vinserti32x4	$0x3, %xmm11, %zmm5, %zmm5
  335d6d:      	vpermt2q	%zmm12, %zmm4, %zmm5
  335d73:      	kmovd	%k1, %eax
  335d77:      	vptestmb	%zmm5, %zmm5, %k1
  335d7d:      	kmovq	%k1, (%rdi,%r8)
  335d83:      	testb	%al, %al
  335d85:      	setne	%r9b
  335d89:      	addq	$0x8, %r8
  335d8d:      	addq	$0x200, %r14            # imm = 0x200
  335d94:      	cmpq	%r8, %rcx
  335d97:      	jne	0x335c50 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x1c0>
  335d9d:      	jmp	0x336200 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x770>
  335da2:      	xorl	%r8d, %r8d
  335da5:      	cmpq	%r10, %r14
  335da8:      	movl	$0xff, %eax
  335dad:      	cmovnel	%r8d, %eax
  335db1:      	kmovd	%eax, %k0
  335db5:      	vpbroadcastq	%r14, %zmm0
  335dbb:      	addq	$0x1c0, %rbx            # imm = 0x1C0
  335dc2:      	vpbroadcastq	-0x299eb4(%rip), %zmm1 # 0x9bf18 <anon.ad39dab31d93760b58da730830a5ab31.633.llvm.6451718222114420760+0x50>
  335dcc:      	vmovdqa64	-0x2946d6(%rip), %zmm2 # 0xa1700 <alloc_cbda3acf5ff4808ffe121d88bd143ef4.llvm.9723113569546643380+0x72>
  335dd6:      	vmovdqa64	-0x2946a0(%rip), %zmm3 # 0xa1740 <alloc_cbda3acf5ff4808ffe121d88bd143ef4.llvm.9723113569546643380+0xb2>
  335de0:      	vmovdqa64	-0x29466a(%rip), %zmm4 # 0xa1780 <alloc_cbda3acf5ff4808ffe121d88bd143ef4.llvm.9723113569546643380+0xf2>
  335dea:      	movl	%r15d, %r9d
  335ded:      	movq	-0x30(%rbp), %rdx
  335df1:      	nopw	%cs:(%rax,%rax)
  335e00:      	andl	$0x1, %r9d
  335e04:      	kmovw	%r9d, %k4
  335e09:      	vmovdqu64	-0x1c0(%rbx), %zmm5
  335e10:      	vmovdqu64	-0x180(%rbx), %zmm6
  335e17:      	vmovdqu64	-0x140(%rbx), %zmm7
  335e1e:      	vmovdqu64	-0x100(%rbx), %zmm8
  335e25:      	vpcmpeqq	%zmm1, %zmm5, %k5
  335e2b:      	vpcmpeqq	%zmm1, %zmm6, %k3
  335e31:      	vpcmpeqq	%zmm1, %zmm7, %k2
  335e37:      	vpcmpeqq	%zmm1, %zmm8, %k1
  335e3d:      	korw	%k5, %k4, %k4
  335e41:      	vpcmpgtq	%zmm0, %zmm5, %k5
  335e47:      	vmovdqu8	%zmm2, %zmm5 {%k5} {z}
  335e4d:      	vpcmpgtq	%zmm0, %zmm6, %k5
  335e53:      	vmovdqu8	%zmm2, %zmm6 {%k5} {z}
  335e59:      	vpcmpgtq	%zmm0, %zmm7, %k5
  335e5f:      	vmovdqu8	%zmm2, %zmm7 {%k5} {z}
  335e65:      	vpcmpgtq	%zmm0, %zmm8, %k5
  335e6b:      	vmovdqu8	%zmm2, %zmm8 {%k5} {z}
  335e71:      	vmovdqu64	-0xc0(%rbx), %zmm9
  335e78:      	vmovdqu64	-0x80(%rbx), %zmm10
  335e7f:      	vmovdqu64	-0x40(%rbx), %zmm11
  335e86:      	vmovdqu64	(%rbx), %zmm12
  335e8c:      	vpcmpeqq	%zmm1, %zmm9, %k5
  335e92:      	korw	%k5, %k4, %k4
  335e96:      	vpcmpeqq	%zmm1, %zmm10, %k5
  335e9c:      	korw	%k5, %k3, %k3
  335ea0:      	vpcmpeqq	%zmm1, %zmm11, %k5
  335ea6:      	korw	%k5, %k2, %k2
  335eaa:      	vpcmpeqq	%zmm1, %zmm12, %k5
  335eb0:      	korw	%k5, %k1, %k1
  335eb4:      	vpcmpgtq	%zmm0, %zmm9, %k5
  335eba:      	vmovdqu8	%zmm2, %zmm9 {%k5} {z}
  335ec0:      	vpcmpgtq	%zmm0, %zmm10, %k5
  335ec6:      	vmovdqu8	%zmm2, %zmm10 {%k5} {z}
  335ecc:      	vpcmpgtq	%zmm0, %zmm11, %k5
  335ed2:      	vmovdqu8	%zmm2, %zmm11 {%k5} {z}
  335ed8:      	vpcmpgtq	%zmm0, %zmm12, %k5
  335ede:      	vmovdqu8	%zmm2, %zmm12 {%k5} {z}
  335ee4:      	korw	%k0, %k4, %k4
  335ee8:      	korw	%k4, %k3, %k3
  335eec:      	korw	%k3, %k2, %k2
  335ef0:      	korw	%k2, %k1, %k1
  335ef4:      	vpunpcklqdq	%xmm6, %xmm5, %xmm5 # xmm5 = xmm5[0],xmm6[0]
  335ef8:      	vinserti128	$0x1, %xmm7, %ymm5, %ymm5
  335efe:      	vpbroadcastq	%xmm8, %ymm6
  335f03:      	vpblendd	$0xc0, %ymm6, %ymm5, %ymm5 # ymm5 = ymm5[0,1,2,3,4,5],ymm6[6,7]
  335f09:      	vinserti64x4	$0x1, %ymm9, %zmm5, %zmm5
  335f10:      	vpermt2q	%zmm10, %zmm3, %zmm5
  335f16:      	vinserti32x4	$0x3, %xmm11, %zmm5, %zmm5
  335f1d:      	vpermt2q	%zmm12, %zmm4, %zmm5
  335f23:      	kmovd	%k1, %eax
  335f27:      	vptestmb	%zmm5, %zmm5, %k1
  335f2d:      	kmovq	%k1, (%rdi,%r8)
  335f33:      	testb	%al, %al
  335f35:      	setne	%r9b
  335f39:      	addq	$0x8, %r8
  335f3d:      	addq	$0x200, %rbx            # imm = 0x200
  335f44:      	cmpq	%r8, %rcx
  335f47:      	jne	0x335e00 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x370>
  335f4d:      	jmp	0x336200 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x770>
  335f52:      	movl	$0x1c0, %r12d           # imm = 0x1C0
  335f58:      	xorl	%r13d, %r13d
  335f5b:      	vpbroadcastq	-0x29a04d(%rip), %zmm0 # 0x9bf18 <anon.ad39dab31d93760b58da730830a5ab31.633.llvm.6451718222114420760+0x50>
  335f65:      	vmovdqa64	-0x29486f(%rip), %zmm1 # 0xa1700 <alloc_cbda3acf5ff4808ffe121d88bd143ef4.llvm.9723113569546643380+0x72>
  335f6f:      	vmovdqa64	-0x294839(%rip), %zmm2 # 0xa1740 <alloc_cbda3acf5ff4808ffe121d88bd143ef4.llvm.9723113569546643380+0xb2>
  335f79:      	vmovdqa64	-0x294803(%rip), %zmm3 # 0xa1780 <alloc_cbda3acf5ff4808ffe121d88bd143ef4.llvm.9723113569546643380+0xf2>
  335f83:      	movl	%r15d, %r9d
  335f86:      	movq	-0x30(%rbp), %rdx
  335f8a:      	nopw	(%rax,%rax)
  335f90:      	andl	$0x1, %r9d
  335f94:      	kmovw	%r9d, %k3
  335f99:      	vmovdqu64	-0x1c0(%r14,%r12), %zmm4
  335fa1:      	vmovdqu64	-0x180(%r14,%r12), %zmm5
  335fa9:      	vmovdqu64	-0x140(%r14,%r12), %zmm6
  335fb1:      	vmovdqu64	-0x100(%r14,%r12), %zmm7
  335fb9:      	vmovdqu64	-0x1c0(%rbx,%r12), %zmm8
  335fc1:      	vmovdqu64	-0x180(%rbx,%r12), %zmm9
  335fc9:      	vmovdqu64	-0x140(%rbx,%r12), %zmm10
  335fd1:      	vmovdqu64	-0x100(%rbx,%r12), %zmm11
  335fd9:      	vpcmpeqq	%zmm0, %zmm4, %k0
  335fdf:      	vpcmpeqq	%zmm0, %zmm5, %k1
  335fe5:      	vpcmpeqq	%zmm0, %zmm6, %k2
  335feb:      	vpcmpeqq	%zmm0, %zmm7, %k4
  335ff1:      	vpcmpeqq	%zmm0, %zmm8, %k5
  335ff7:      	vpcmpeqq	%zmm0, %zmm9, %k6
  335ffd:      	vpcmpeqq	%zmm0, %zmm10, %k7
  336003:      	korw	%k5, %k0, %k5
  336007:      	vpcmpeqq	%zmm0, %zmm11, %k0
  33600d:      	korw	%k6, %k1, %k6
  336011:      	korw	%k7, %k2, %k1
  336015:      	kmovw	%k1, -0x40(%rbp)
  33601a:      	korw	%k0, %k4, %k2
  33601e:      	vpcmpgtq	%zmm4, %zmm8, %k4
  336024:      	vpcmpgtq	%zmm5, %zmm9, %k7
  33602a:      	vpcmpgtq	%zmm6, %zmm10, %k1
  336030:      	korw	%k5, %k3, %k3
  336034:      	vpcmpgtq	%zmm7, %zmm11, %k5
  33603a:      	vmovdqu8	%zmm1, %zmm4 {%k4} {z}
  336040:      	vmovdqu8	%zmm1, %zmm5 {%k7} {z}
  336046:      	vmovdqu8	%zmm1, %zmm6 {%k1} {z}
  33604c:      	vmovdqu8	%zmm1, %zmm7 {%k5} {z}
  336052:      	vmovdqu64	-0xc0(%r14,%r12), %zmm8
  33605a:      	vmovdqu64	-0x80(%r14,%r12), %zmm9
  336062:      	vmovdqu64	-0x40(%r14,%r12), %zmm10
  33606a:      	vmovdqu64	(%r14,%r12), %zmm11
  336071:      	vmovdqu64	-0xc0(%rbx,%r12), %zmm12
  336079:      	vmovdqu64	-0x80(%rbx,%r12), %zmm13
  336081:      	vmovdqu64	-0x40(%rbx,%r12), %zmm14
  336089:      	vmovdqu64	(%rbx,%r12), %zmm15
  336090:      	vpcmpeqq	%zmm0, %zmm8, %k0
  336096:      	vpcmpeqq	%zmm0, %zmm9, %k1
  33609c:      	vpcmpeqq	%zmm0, %zmm10, %k4
  3360a2:      	vpcmpeqq	%zmm0, %zmm12, %k5
  3360a8:      	korw	%k5, %k0, %k0
  3360ac:      	vpcmpeqq	%zmm0, %zmm13, %k5
  3360b2:      	korw	%k5, %k1, %k1
  3360b6:      	vpcmpeqq	%zmm0, %zmm14, %k5
  3360bc:      	korw	%k5, %k4, %k4
  3360c0:      	vpcmpeqq	%zmm0, %zmm11, %k5
  3360c6:      	vpcmpeqq	%zmm0, %zmm15, %k7
  3360cc:      	korw	%k7, %k5, %k5
  3360d0:      	korw	%k0, %k3, %k3
  3360d4:      	korw	%k1, %k6, %k0
  3360d8:      	kmovw	-0x40(%rbp), %k1
  3360dd:      	korw	%k4, %k1, %k1
  3360e1:      	korw	%k5, %k2, %k2
  3360e5:      	vpcmpgtq	%zmm8, %zmm12, %k4
  3360eb:      	vmovdqu8	%zmm1, %zmm8 {%k4} {z}
  3360f1:      	vpcmpgtq	%zmm9, %zmm13, %k4
  3360f7:      	vmovdqu8	%zmm1, %zmm9 {%k4} {z}
  3360fd:      	vpcmpgtq	%zmm10, %zmm14, %k4
  336103:      	vmovdqu8	%zmm1, %zmm10 {%k4} {z}
  336109:      	vpcmpgtq	%zmm11, %zmm15, %k4
  33610f:      	vmovdqu8	%zmm1, %zmm11 {%k4} {z}
  336115:      	korw	%k3, %k0, %k0
  336119:      	korw	%k0, %k1, %k0
  33611d:      	korw	%k0, %k2, %k0
  336121:      	vpunpcklqdq	%xmm5, %xmm4, %xmm4 # xmm4 = xmm4[0],xmm5[0]
  336125:      	vinserti128	$0x1, %xmm6, %ymm4, %ymm4
  33612b:      	vpbroadcastq	%xmm7, %ymm5
  336130:      	vpblendd	$0xc0, %ymm5, %ymm4, %ymm4 # ymm4 = ymm4[0,1,2,3,4,5],ymm5[6,7]
  336136:      	vinserti64x4	$0x1, %ymm8, %zmm4, %zmm4
  33613d:      	vpermt2q	%zmm9, %zmm2, %zmm4
  336143:      	vinserti32x4	$0x3, %xmm10, %zmm4, %zmm4
  33614a:      	vpermt2q	%zmm11, %zmm3, %zmm4
  336150:      	kmovd	%k0, %eax
  336154:      	vptestmb	%zmm4, %zmm4, %k0
  33615a:      	kmovq	%k0, (%rdi,%r13)
  336160:      	testb	%al, %al
  336162:      	setne	%r9b
  336166:      	addq	$0x8, %r13
  33616a:      	addq	$0x200, %r12            # imm = 0x200
  336171:      	cmpq	%r13, %rcx
  336174:      	jne	0x335f90 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x500>
  33617a:      	jmp	0x336200 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x770>
  33617f:      	movl	%ecx, %eax
  336181:      	shrl	$0x3, %eax
  336184:      	incl	%eax
  336186:      	andl	$0x7, %eax
  336189:      	movq	-0x38(%rbp), %rdx
  33618d:      	nopl	(%rax)
  336190:      	kmovq	%k0, (%rdi)
  336195:      	addq	$0x8, %rdi
  336199:      	decq	%rax
  33619c:      	jne	0x336190 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x700>
  33619e:      	cmpq	$0x38, %rcx
  3361a2:      	jb	0x3361e8 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x758>
  3361a4:      	nopw	%cs:(%rax,%rax)
  3361b0:      	kmovq	%k0, (%rdi)
  3361b5:      	kmovq	%k0, 0x8(%rdi)
  3361bb:      	kmovq	%k0, 0x10(%rdi)
  3361c1:      	kmovq	%k0, 0x18(%rdi)
  3361c7:      	kmovq	%k0, 0x20(%rdi)
  3361cd:      	kmovq	%k0, 0x28(%rdi)
  3361d3:      	kmovq	%k0, 0x30(%rdi)
  3361d9:      	kmovq	%k0, 0x38(%rdi)
  3361df:      	addq	$0x40, %rdi
  3361e3:      	cmpq	%rdx, %rdi
  3361e6:      	jne	0x3361b0 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x720>
  3361e8:      	cmpq	%r10, %r14
  3361eb:      	sete	%al
  3361ee:      	cmpq	%r10, -0x40(%rbp)
  3361f2:      	sete	%r9b
  3361f6:      	orb	%r15b, %r9b
  3361f9:      	orb	%al, %r9b
  3361fc:      	movq	-0x30(%rbp), %rdx
  336200:      	andb	$0x1, %r9b
  336204:      	movq	-0x48(%rbp), %r8
  336208:      	movb	%r9b, 0x38(%r8)
  33620c:      	movq	-0x50(%rbp), %r9
  336210:      	testq	%r11, %r11
  336213:      	je	0x336b50 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x10c0>
  336219:      	movq	%rdx, %rdi
  33621c:      	andq	$-0x40, %rdi
  336220:      	movzbl	0x18(%r8), %ecx
  336225:      	movzbl	0x38(%r8), %r12d
  33622a:      	cmpl	$0x1, (%r8)
  33622e:      	jne	0x33625c <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x7cc>
  336230:      	movq	0x10(%r8), %rax
  336234:      	testb	%cl, %cl
  336236:      	je	0x33627e <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x7ee>
  336238:      	movq	0x28(%r8), %rcx
  33623c:      	movq	(%rax), %rbx
  33623f:      	movq	(%rcx), %rdi
  336242:      	xorl	%r15d, %r15d
  336245:      	cmpq	%rdi, %rbx
  336248:      	setl	%r15b
  33624c:      	cmpl	$0x4, %r11d
  336250:      	jae	0x3362af <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x81f>
  336252:      	xorl	%r14d, %r14d
  336255:      	xorl	%ecx, %ecx
  336257:      	jmp	0x336410 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x980>
  33625c:      	movq	0x8(%r8), %rbx
  336260:      	testb	%cl, %cl
  336262:      	je	0x336298 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x808>
  336264:      	movq	0x28(%r8), %rax
  336268:      	movq	(%rax), %r13
  33626b:      	cmpl	$0x4, %r11d
  33626f:      	jae	0x3362bf <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x82f>
  336271:      	xorl	%r14d, %r14d
  336274:      	xorl	%ecx, %ecx
  336276:      	movl	%r12d, %r15d
  336279:      	jmp	0x336643 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0xbb3>
  33627e:      	movq	0x20(%r8), %r13
  336282:      	movq	(%rax), %rbx
  336285:      	cmpl	$0x4, %r11d
  336289:      	jae	0x3362d9 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x849>
  33628b:      	xorl	%r14d, %r14d
  33628e:      	xorl	%ecx, %ecx
  336290:      	movl	%r12d, %r15d
  336293:      	jmp	0x3368a3 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0xe13>
  336298:      	movq	0x20(%r8), %r13
  33629c:      	cmpl	$0x4, %r11d
  3362a0:      	jae	0x3362f3 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x863>
  3362a2:      	xorl	%r14d, %r14d
  3362a5:      	xorl	%ecx, %ecx
  3362a7:      	movl	%r12d, %r15d
  3362aa:      	jmp	0x336aec <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x105c>
  3362af:      	cmpl	$0x10, %r11d
  3362b3:      	jae	0x33630a <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x87a>
  3362b5:      	xorl	%ecx, %ecx
  3362b7:      	xorl	%r14d, %r14d
  3362ba:      	jmp	0x3363a3 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x913>
  3362bf:      	xorl	%r8d, %r8d
  3362c2:      	cmpl	$0x10, %r11d
  3362c6:      	jae	0x336439 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x9a9>
  3362cc:      	xorl	%ecx, %ecx
  3362ce:      	xorl	%r14d, %r14d
  3362d1:      	movl	%r12d, %r15d
  3362d4:      	jmp	0x336577 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0xae7>
  3362d9:      	xorl	%r8d, %r8d
  3362dc:      	cmpl	$0x10, %r11d
  3362e0:      	jae	0x33668c <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0xbfc>
  3362e6:      	xorl	%ecx, %ecx
  3362e8:      	xorl	%r14d, %r14d
  3362eb:      	movl	%r12d, %r15d
  3362ee:      	jmp	0x3367d7 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0xd47>
  3362f3:      	cmpl	$0x10, %r11d
  3362f7:      	jae	0x3368fc <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0xe6c>
  3362fd:      	xorl	%ecx, %ecx
  3362ff:      	xorl	%r14d, %r14d
  336302:      	movl	%r12d, %r15d
  336305:      	jmp	0x336a40 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0xfb0>
  33630a:      	movl	%edx, %ecx
  33630c:      	andl	$0x30, %ecx
  33630f:      	vpbroadcastq	%r15, %zmm0
  336315:      	vmovdqa64	-0x294b5f(%rip), %zmm2 # 0xa17c0 <alloc_cbda3acf5ff4808ffe121d88bd143ef4.llvm.9723113569546643380+0x132>
  33631f:      	vpxor	%xmm1, %xmm1, %xmm1
  336323:      	vpbroadcastq	-0x29a96d(%rip), %zmm3 # 0x9b9c0 <anon.5278ce255300ec7b361bbf944ee0b803.25.llvm.754055615369681222>
  33632d:      	vpbroadcastq	-0x299dbf(%rip), %zmm4 # 0x9c578 <anon.ad39dab31d93760b58da730830a5ab31.728.llvm.6451718222114420760+0x40>
  336337:      	movq	%rcx, %rax
  33633a:      	vpxor	%xmm5, %xmm5, %xmm5
  33633e:      	nop
  336340:      	vpaddq	%zmm3, %zmm2, %zmm6
  336346:      	vpsllvq	%zmm2, %zmm0, %zmm7
  33634c:      	vporq	%zmm1, %zmm7, %zmm1
  336352:      	vpsllvq	%zmm6, %zmm0, %zmm6
  336358:      	vporq	%zmm5, %zmm6, %zmm5
  33635e:      	vpaddq	%zmm4, %zmm2, %zmm2
  336364:      	addq	$-0x10, %rax
  336368:      	jne	0x336340 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x8b0>
  33636a:      	vporq	%zmm1, %zmm5, %zmm0
  336370:      	vextracti64x4	$0x1, %zmm0, %ymm1
  336377:      	vporq	%zmm1, %zmm0, %zmm0
  33637d:      	vextracti128	$0x1, %ymm0, %xmm1
  336383:      	vpor	%xmm1, %xmm0, %xmm0
  336387:      	vpshufd	$0xee, %xmm0, %xmm1     # xmm1 = xmm0[2,3,2,3]
  33638c:      	vpor	%xmm1, %xmm0, %xmm0
  336390:      	vmovq	%xmm0, %r14
  336395:      	cmpl	%ecx, %r11d
  336398:      	je	0x336421 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x991>
  33639e:      	testb	$0xc, %dl
  3363a1:      	je	0x336410 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x980>
  3363a3:      	movq	%rcx, %r8
  3363a6:      	movl	%edx, %ecx
  3363a8:      	andl	$0x3c, %ecx
  3363ab:      	vmovq	%r14, %xmm0
  3363b0:      	vmovq	%r15, %xmm2
  3363b5:      	vmovq	%r8, %xmm1
  3363ba:      	vpbroadcastq	%xmm1, %ymm1
  3363bf:      	vpor	-0x298c67(%rip), %ymm1, %ymm1 # 0x9d760 <anon.1b4110e42284c5f75ac48cf7fa14c04d.13.llvm.7054906275604583804+0x80>
  3363c7:      	vpbroadcastq	%xmm2, %ymm2
  3363cc:      	subq	%rcx, %r8
  3363cf:      	vpbroadcastq	-0x299ec8(%rip), %ymm3 # 0x9c510 <anon.5278ce255300ec7b361bbf944ee0b803.26.llvm.754055615369681222>
  3363d8:      	nopl	(%rax,%rax)
  3363e0:      	vpsllvq	%ymm1, %ymm2, %ymm4
  3363e5:      	vpor	%ymm0, %ymm4, %ymm0
  3363e9:      	vpaddq	%ymm3, %ymm1, %ymm1
  3363ed:      	addq	$0x4, %r8
  3363f1:      	jne	0x3363e0 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x950>
  3363f3:      	vextracti128	$0x1, %ymm0, %xmm1
  3363f9:      	vpor	%xmm1, %xmm0, %xmm0
  3363fd:      	vpshufd	$0xee, %xmm0, %xmm1     # xmm1 = xmm0[2,3,2,3]
  336402:      	vpor	%xmm1, %xmm0, %xmm0
  336406:      	vmovq	%xmm0, %r14
  33640b:      	cmpl	%ecx, %r11d
  33640e:      	je	0x336421 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x991>
  336410:      	movq	%r15, %rax
  336413:      	shlq	%cl, %rax
  336416:      	incq	%rcx
  336419:      	orq	%rax, %r14
  33641c:      	cmpq	%rcx, %r11
  33641f:      	jne	0x336410 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x980>
  336421:      	cmpq	%r10, %rbx
  336424:      	sete	%al
  336427:      	cmpq	%r10, %rdi
  33642a:      	sete	%r15b
  33642e:      	orb	%al, %r15b
  336431:      	orb	%r12b, %r15b
  336434:      	jmp	0x336b34 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x10a4>
  336439:      	movq	%rdx, %rcx
  33643c:      	movq	%r9, %rdx
  33643f:      	movq	%rcx, %rsi
  336442:      	andl	$0x30, %ecx
  336445:      	andl	$0x1, %r12d
  336449:      	cmpq	%r10, %r13
  33644c:      	kmovw	%r12d, %k0
  336451:      	vpbroadcastq	%r13, %zmm0
  336457:      	movl	$0xff, %eax
  33645c:      	cmovnel	%r8d, %eax
  336460:      	kmovd	%eax, %k1
  336464:      	leaq	(%rbx,%rdi,8), %r9
  336468:      	kxorw	%k0, %k0, %k2
  33646c:      	vmovdqa64	-0x294cb6(%rip), %zmm2 # 0xa17c0 <alloc_cbda3acf5ff4808ffe121d88bd143ef4.llvm.9723113569546643380+0x132>
  336476:      	vpxor	%xmm1, %xmm1, %xmm1
  33647a:      	vpbroadcastq	-0x29aac4(%rip), %zmm3 # 0x9b9c0 <anon.5278ce255300ec7b361bbf944ee0b803.25.llvm.754055615369681222>
  336484:      	vpbroadcastq	-0x29a576(%rip), %zmm4 # 0x9bf18 <anon.ad39dab31d93760b58da730830a5ab31.633.llvm.6451718222114420760+0x50>
  33648e:      	vpbroadcastq	-0x299f20(%rip), %zmm5 # 0x9c578 <anon.ad39dab31d93760b58da730830a5ab31.728.llvm.6451718222114420760+0x40>
  336498:      	movq	%rcx, %r14
  33649b:      	vpxor	%xmm6, %xmm6, %xmm6
  33649f:      	nop
  3364a0:      	vpaddq	%zmm3, %zmm2, %zmm7
  3364a6:      	vmovq	%xmm2, %rax
  3364ab:      	vmovdqu64	(%r9,%rax,8), %zmm8
  3364b2:      	vmovdqu64	0x40(%r9,%rax,8), %zmm9
  3364ba:      	vpcmpeqq	%zmm4, %zmm8, %k3
  3364c0:      	vpcmpeqq	%zmm4, %zmm9, %k4
  3364c6:      	vpcmpgtq	%zmm8, %zmm0, %k5
  3364cc:      	vpcmpgtq	%zmm9, %zmm0, %k6
  3364d2:      	korw	%k0, %k3, %k0
  3364d6:      	korw	%k1, %k0, %k0
  3364da:      	korw	%k2, %k4, %k3
  3364de:      	korw	%k1, %k3, %k2
  3364e2:      	vpternlogq	$0xff, %zmm8, %zmm8, %zmm8 {%k5} {z} # zmm8 {%k5} {z} = -1
  3364e9:      	vpsrlq	$0x3f, %zmm8, %zmm8
  3364f0:      	vpternlogq	$0xff, %zmm9, %zmm9, %zmm9 {%k6} {z} # zmm9 {%k6} {z} = -1
  3364f7:      	vpsrlq	$0x3f, %zmm9, %zmm9
  3364fe:      	vpsllvq	%zmm7, %zmm9, %zmm7
  336504:      	vporq	%zmm6, %zmm7, %zmm6
  33650a:      	vpsllvq	%zmm2, %zmm8, %zmm7
  336510:      	vporq	%zmm1, %zmm7, %zmm1
  336516:      	vpaddq	%zmm5, %zmm2, %zmm2
  33651c:      	addq	$-0x10, %r14
  336520:      	jne	0x3364a0 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0xa10>
  336526:      	vporq	%zmm1, %zmm6, %zmm0
  33652c:      	vextracti64x4	$0x1, %zmm0, %ymm1
  336533:      	vporq	%zmm1, %zmm0, %zmm0
  336539:      	vextracti128	$0x1, %ymm0, %xmm1
  33653f:      	vpor	%xmm1, %xmm0, %xmm0
  336543:      	vpshufd	$0xee, %xmm0, %xmm1     # xmm1 = xmm0[2,3,2,3]
  336548:      	vpor	%xmm1, %xmm0, %xmm0
  33654c:      	vmovq	%xmm0, %r14
  336551:      	korw	%k0, %k3, %k0
  336555:      	kmovd	%k0, %eax
  336559:      	testb	%al, %al
  33655b:      	setne	%r15b
  33655f:      	cmpl	%ecx, %r11d
  336562:      	movq	%rdx, %r9
  336565:      	je	0x336b34 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x10a4>
  33656b:      	movq	%rsi, %rdx
  33656e:      	testb	$0xc, %dl
  336571:      	je	0x336643 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0xbb3>
  336577:      	movq	%r9, %rsi
  33657a:      	movq	%rcx, %r9
  33657d:      	movl	%edx, %ecx
  33657f:      	andl	$0x3c, %ecx
  336582:      	andl	$0x1, %r15d
  336586:      	cmpq	%r10, %r13
  336589:      	vmovq	%r14, %xmm0
  33658e:      	kmovw	%r15d, %k0
  336593:      	vmovq	%r13, %xmm1
  336598:      	movl	$0xff, %eax
  33659d:      	cmovel	%eax, %r8d
  3365a1:      	vpbroadcastq	%xmm1, %ymm1
  3365a6:      	kmovd	%r8d, %k1
  3365ab:      	vmovq	%r9, %xmm2
  3365b0:      	vpbroadcastq	%xmm2, %ymm2
  3365b5:      	vpor	-0x298e5d(%rip), %ymm2, %ymm2 # 0x9d760 <anon.1b4110e42284c5f75ac48cf7fa14c04d.13.llvm.7054906275604583804+0x80>
  3365bd:      	leaq	(%rbx,%rdi,8), %rdx
  3365c1:      	subq	%rcx, %r9
  3365c4:      	vpbroadcastq	-0x29a6b6(%rip), %zmm3 # 0x9bf18 <anon.ad39dab31d93760b58da730830a5ab31.633.llvm.6451718222114420760+0x50>
  3365ce:      	vpbroadcastq	-0x29a0c7(%rip), %ymm4 # 0x9c510 <anon.5278ce255300ec7b361bbf944ee0b803.26.llvm.754055615369681222>
  3365d7:      	nopw	(%rax,%rax)
  3365e0:      	vmovq	%xmm2, %rax
  3365e5:      	vmovdqu	(%rdx,%rax,8), %ymm5
  3365ea:      	vpcmpeqq	%zmm3, %zmm5, %k2
  3365f0:      	korw	%k2, %k1, %k2
  3365f4:      	korw	%k2, %k0, %k0
  3365f8:      	vpcmpgtq	%ymm5, %ymm1, %ymm5
  3365fd:      	vpsrlq	$0x3f, %ymm5, %ymm5
  336602:      	vpsllvq	%ymm2, %ymm5, %ymm5
  336607:      	vpor	%ymm0, %ymm5, %ymm0
  33660b:      	vpaddq	%ymm4, %ymm2, %ymm2
  33660f:      	addq	$0x4, %r9
  336613:      	jne	0x3365e0 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0xb50>
  336615:      	vextracti128	$0x1, %ymm0, %xmm1
  33661b:      	vpor	%xmm1, %xmm0, %xmm0
  33661f:      	vpshufd	$0xee, %xmm0, %xmm1     # xmm1 = xmm0[2,3,2,3]
  336624:      	vpor	%xmm1, %xmm0, %xmm0
  336628:      	vmovq	%xmm0, %r14
  33662d:      	kmovd	%k0, %eax
  336631:      	testb	$0xf, %al
  336633:      	setne	%r15b
  336637:      	cmpl	%ecx, %r11d
  33663a:      	movq	%rsi, %r9
  33663d:      	je	0x336b34 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x10a4>
  336643:      	leaq	(%rbx,%rdi,8), %rdx
  336647:      	movl	%r15d, %eax
  33664a:      	nopw	(%rax,%rax)
  336650:      	cmpq	%r10, %r13
  336653:      	sete	%r15b
  336657:      	movq	(%rdx,%rcx,8), %rsi
  33665b:      	cmpq	%r10, %rsi
  33665e:      	sete	%dil
  336662:      	xorl	%r8d, %r8d
  336665:      	cmpq	%r13, %rsi
  336668:      	setl	%r8b
  33666c:      	orb	%al, %r15b
  33666f:      	orb	%dil, %r15b
  336672:      	leaq	0x1(%rcx), %rsi
  336676:      	shlq	%cl, %r8
  336679:      	orq	%r8, %r14
  33667c:      	movl	%r15d, %eax
  33667f:      	cmpq	%rsi, %r11
  336682:      	movq	%rsi, %rcx
  336685:      	jne	0x336650 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0xbc0>
  336687:      	jmp	0x336b34 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x10a4>
  33668c:      	movq	%rdx, %rcx
  33668f:      	movq	%r9, %rdx
  336692:      	movq	%rcx, %rsi
  336695:      	andl	$0x30, %ecx
  336698:      	andl	$0x1, %r12d
  33669c:      	cmpq	%r10, %rbx
  33669f:      	kmovw	%r12d, %k0
  3366a4:      	vpbroadcastq	%rbx, %zmm0
  3366aa:      	movl	$0xff, %eax
  3366af:      	cmovnel	%r8d, %eax
  3366b3:      	kmovd	%eax, %k1
  3366b7:      	leaq	(%r13,%rdi,8), %r9
  3366bc:      	kxorw	%k0, %k0, %k2
  3366c0:      	vmovdqa64	-0x294f0a(%rip), %zmm2 # 0xa17c0 <alloc_cbda3acf5ff4808ffe121d88bd143ef4.llvm.9723113569546643380+0x132>
  3366ca:      	vpxor	%xmm1, %xmm1, %xmm1
  3366ce:      	vpbroadcastq	-0x29ad18(%rip), %zmm3 # 0x9b9c0 <anon.5278ce255300ec7b361bbf944ee0b803.25.llvm.754055615369681222>
  3366d8:      	vpbroadcastq	-0x29a7ca(%rip), %zmm4 # 0x9bf18 <anon.ad39dab31d93760b58da730830a5ab31.633.llvm.6451718222114420760+0x50>
  3366e2:      	vpbroadcastq	-0x29a174(%rip), %zmm5 # 0x9c578 <anon.ad39dab31d93760b58da730830a5ab31.728.llvm.6451718222114420760+0x40>
  3366ec:      	movq	%rcx, %r14
  3366ef:      	vpxor	%xmm6, %xmm6, %xmm6
  3366f3:      	nopw	%cs:(%rax,%rax)
  336700:      	vpaddq	%zmm3, %zmm2, %zmm7
  336706:      	vmovq	%xmm2, %rax
  33670b:      	vmovdqu64	(%r9,%rax,8), %zmm8
  336712:      	vmovdqu64	0x40(%r9,%rax,8), %zmm9
  33671a:      	vpcmpeqq	%zmm4, %zmm8, %k3
  336720:      	vpcmpeqq	%zmm4, %zmm9, %k4
  336726:      	vpcmpgtq	%zmm0, %zmm8, %k5
  33672c:      	vpcmpgtq	%zmm0, %zmm9, %k6
  336732:      	korw	%k0, %k3, %k0
  336736:      	korw	%k1, %k0, %k0
  33673a:      	korw	%k2, %k4, %k3
  33673e:      	korw	%k1, %k3, %k2
  336742:      	vpternlogq	$0xff, %zmm8, %zmm8, %zmm8 {%k5} {z} # zmm8 {%k5} {z} = -1
  336749:      	vpsrlq	$0x3f, %zmm8, %zmm8
  336750:      	vpternlogq	$0xff, %zmm9, %zmm9, %zmm9 {%k6} {z} # zmm9 {%k6} {z} = -1
  336757:      	vpsrlq	$0x3f, %zmm9, %zmm9
  33675e:      	vpsllvq	%zmm7, %zmm9, %zmm7
  336764:      	vporq	%zmm6, %zmm7, %zmm6
  33676a:      	vpsllvq	%zmm2, %zmm8, %zmm7
  336770:      	vporq	%zmm1, %zmm7, %zmm1
  336776:      	vpaddq	%zmm5, %zmm2, %zmm2
  33677c:      	addq	$-0x10, %r14
  336780:      	jne	0x336700 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0xc70>
  336786:      	vporq	%zmm1, %zmm6, %zmm0
  33678c:      	vextracti64x4	$0x1, %zmm0, %ymm1
  336793:      	vporq	%zmm1, %zmm0, %zmm0
  336799:      	vextracti128	$0x1, %ymm0, %xmm1
  33679f:      	vpor	%xmm1, %xmm0, %xmm0
  3367a3:      	vpshufd	$0xee, %xmm0, %xmm1     # xmm1 = xmm0[2,3,2,3]
  3367a8:      	vpor	%xmm1, %xmm0, %xmm0
  3367ac:      	vmovq	%xmm0, %r14
  3367b1:      	korw	%k0, %k3, %k0
  3367b5:      	kmovd	%k0, %eax
  3367b9:      	testb	%al, %al
  3367bb:      	setne	%r15b
  3367bf:      	cmpl	%ecx, %r11d
  3367c2:      	movq	%rdx, %r9
  3367c5:      	je	0x336b34 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x10a4>
  3367cb:      	movq	%rsi, %rdx
  3367ce:      	testb	$0xc, %dl
  3367d1:      	je	0x3368a3 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0xe13>
  3367d7:      	movq	%r9, %rsi
  3367da:      	movq	%rcx, %r9
  3367dd:      	movl	%edx, %ecx
  3367df:      	andl	$0x3c, %ecx
  3367e2:      	andl	$0x1, %r15d
  3367e6:      	cmpq	%r10, %rbx
  3367e9:      	vmovq	%r14, %xmm0
  3367ee:      	kmovw	%r15d, %k0
  3367f3:      	vmovq	%rbx, %xmm1
  3367f8:      	movl	$0xff, %eax
  3367fd:      	cmovel	%eax, %r8d
  336801:      	vpbroadcastq	%xmm1, %ymm1
  336806:      	kmovd	%r8d, %k1
  33680b:      	vmovq	%r9, %xmm2
  336810:      	vpbroadcastq	%xmm2, %ymm2
  336815:      	vpor	-0x2990bd(%rip), %ymm2, %ymm2 # 0x9d760 <anon.1b4110e42284c5f75ac48cf7fa14c04d.13.llvm.7054906275604583804+0x80>
  33681d:      	leaq	(,%rdi,8), %rdx
  336825:      	addq	%r13, %rdx
  336828:      	subq	%rcx, %r9
  33682b:      	vpbroadcastq	-0x29a91d(%rip), %zmm3 # 0x9bf18 <anon.ad39dab31d93760b58da730830a5ab31.633.llvm.6451718222114420760+0x50>
  336835:      	vpbroadcastq	-0x29a32e(%rip), %ymm4 # 0x9c510 <anon.5278ce255300ec7b361bbf944ee0b803.26.llvm.754055615369681222>
  33683e:      	nop
  336840:      	vmovq	%xmm2, %rax
  336845:      	vmovdqu	(%rdx,%rax,8), %ymm5
  33684a:      	vpcmpeqq	%zmm3, %zmm5, %k2
  336850:      	korw	%k2, %k1, %k2
  336854:      	korw	%k2, %k0, %k0
  336858:      	vpcmpgtq	%ymm1, %ymm5, %ymm5
  33685d:      	vpsrlq	$0x3f, %ymm5, %ymm5
  336862:      	vpsllvq	%ymm2, %ymm5, %ymm5
  336867:      	vpor	%ymm0, %ymm5, %ymm0
  33686b:      	vpaddq	%ymm4, %ymm2, %ymm2
  33686f:      	addq	$0x4, %r9
  336873:      	jne	0x336840 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0xdb0>
  336875:      	vextracti128	$0x1, %ymm0, %xmm1
  33687b:      	vpor	%xmm1, %xmm0, %xmm0
  33687f:      	vpshufd	$0xee, %xmm0, %xmm1     # xmm1 = xmm0[2,3,2,3]
  336884:      	vpor	%xmm1, %xmm0, %xmm0
  336888:      	vmovq	%xmm0, %r14
  33688d:      	kmovd	%k0, %eax
  336891:      	testb	$0xf, %al
  336893:      	setne	%r15b
  336897:      	cmpl	%ecx, %r11d
  33689a:      	movq	%rsi, %r9
  33689d:      	je	0x336b34 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x10a4>
  3368a3:      	leaq	(,%rdi,8), %rdx
  3368ab:      	addq	%r13, %rdx
  3368ae:      	movl	%r15d, %eax
  3368b1:      	nopw	%cs:(%rax,%rax)
  3368c0:      	cmpq	%r10, %rbx
  3368c3:      	sete	%r15b
  3368c7:      	movq	(%rdx,%rcx,8), %rsi
  3368cb:      	cmpq	%r10, %rsi
  3368ce:      	sete	%dil
  3368d2:      	xorl	%r8d, %r8d
  3368d5:      	cmpq	%rsi, %rbx
  3368d8:      	setl	%r8b
  3368dc:      	orb	%al, %r15b
  3368df:      	orb	%dil, %r15b
  3368e2:      	leaq	0x1(%rcx), %rsi
  3368e6:      	shlq	%cl, %r8
  3368e9:      	orq	%r8, %r14
  3368ec:      	movl	%r15d, %eax
  3368ef:      	cmpq	%rsi, %r11
  3368f2:      	movq	%rsi, %rcx
  3368f5:      	jne	0x3368c0 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0xe30>
  3368f7:      	jmp	0x336b34 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x10a4>
  3368fc:      	movl	%edx, %ecx
  3368fe:      	andl	$0x30, %ecx
  336901:      	andl	$0x1, %r12d
  336905:      	kmovw	%r12d, %k0
  33690a:      	kxorw	%k0, %k0, %k1
  33690e:      	vmovdqa64	-0x295158(%rip), %zmm1 # 0xa17c0 <alloc_cbda3acf5ff4808ffe121d88bd143ef4.llvm.9723113569546643380+0x132>
  336918:      	vpxor	%xmm0, %xmm0, %xmm0
  33691c:      	vpbroadcastq	-0x29af66(%rip), %zmm2 # 0x9b9c0 <anon.5278ce255300ec7b361bbf944ee0b803.25.llvm.754055615369681222>
  336926:      	vpbroadcastq	-0x29aa18(%rip), %zmm3 # 0x9bf18 <anon.ad39dab31d93760b58da730830a5ab31.633.llvm.6451718222114420760+0x50>
  336930:      	vpbroadcastq	-0x29a3c2(%rip), %zmm4 # 0x9c578 <anon.ad39dab31d93760b58da730830a5ab31.728.llvm.6451718222114420760+0x40>
  33693a:      	movq	%rcx, %r8
  33693d:      	vpxor	%xmm5, %xmm5, %xmm5
  336941:      	nopw	%cs:(%rax,%rax)
  336950:      	vpaddq	%zmm2, %zmm1, %zmm6
  336956:      	vmovq	%xmm1, %rax
  33695b:      	addq	%rdi, %rax
  33695e:      	vmovdqu64	(%rbx,%rax,8), %zmm7
  336965:      	vmovdqu64	0x40(%rbx,%rax,8), %zmm8
  33696d:      	vmovdqu64	(%r13,%rax,8), %zmm9
  336975:      	vmovdqu64	0x40(%r13,%rax,8), %zmm10
  33697d:      	vpcmpeqq	%zmm3, %zmm7, %k2
  336983:      	vpcmpeqq	%zmm3, %zmm8, %k3
  336989:      	vpcmpeqq	%zmm3, %zmm9, %k4
  33698f:      	vpcmpeqq	%zmm3, %zmm10, %k5
  336995:      	korw	%k4, %k2, %k2
  336999:      	korw	%k5, %k3, %k3
  33699d:      	vpcmpgtq	%zmm7, %zmm9, %k4
  3369a3:      	vpcmpgtq	%zmm8, %zmm10, %k5
  3369a9:      	korw	%k2, %k0, %k0
  3369ad:      	korw	%k3, %k1, %k1
  3369b1:      	vpternlogq	$0xff, %zmm7, %zmm7, %zmm7 {%k4} {z} # zmm7 {%k4} {z} = -1
  3369b8:      	vpsrlq	$0x3f, %zmm7, %zmm7
  3369bf:      	vpternlogq	$0xff, %zmm8, %zmm8, %zmm8 {%k5} {z} # zmm8 {%k5} {z} = -1
  3369c6:      	vpsrlq	$0x3f, %zmm8, %zmm8
  3369cd:      	vpsllvq	%zmm6, %zmm8, %zmm6
  3369d3:      	vporq	%zmm5, %zmm6, %zmm5
  3369d9:      	vpsllvq	%zmm1, %zmm7, %zmm6
  3369df:      	vporq	%zmm0, %zmm6, %zmm0
  3369e5:      	vpaddq	%zmm4, %zmm1, %zmm1
  3369eb:      	addq	$-0x10, %r8
  3369ef:      	jne	0x336950 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0xec0>
  3369f5:      	vporq	%zmm0, %zmm5, %zmm0
  3369fb:      	vextracti64x4	$0x1, %zmm0, %ymm1
  336a02:      	vporq	%zmm1, %zmm0, %zmm0
  336a08:      	vextracti128	$0x1, %ymm0, %xmm1
  336a0e:      	vpor	%xmm1, %xmm0, %xmm0
  336a12:      	vpshufd	$0xee, %xmm0, %xmm1     # xmm1 = xmm0[2,3,2,3]
  336a17:      	vpor	%xmm1, %xmm0, %xmm0
  336a1b:      	vmovq	%xmm0, %r14
  336a20:      	korw	%k0, %k1, %k0
  336a24:      	kmovd	%k0, %eax
  336a28:      	testb	%al, %al
  336a2a:      	setne	%r15b
  336a2e:      	cmpl	%ecx, %r11d
  336a31:      	je	0x336b34 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x10a4>
  336a37:      	testb	$0xc, %dl
  336a3a:      	je	0x336aec <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x105c>
  336a40:      	movq	%rcx, %r8
  336a43:      	movl	%edx, %ecx
  336a45:      	andl	$0x3c, %ecx
  336a48:      	vmovq	%r14, %xmm0
  336a4d:      	andl	$0x1, %r15d
  336a51:      	kmovw	%r15d, %k0
  336a56:      	vmovq	%r8, %xmm1
  336a5b:      	vpbroadcastq	%xmm1, %ymm1
  336a60:      	vpor	-0x299308(%rip), %ymm1, %ymm1 # 0x9d760 <anon.1b4110e42284c5f75ac48cf7fa14c04d.13.llvm.7054906275604583804+0x80>
  336a68:      	subq	%rcx, %r8
  336a6b:      	vpbroadcastq	-0x29ab5c(%rip), %ymm2 # 0x9bf18 <anon.ad39dab31d93760b58da730830a5ab31.633.llvm.6451718222114420760+0x50>
  336a74:      	vpbroadcastq	-0x29a56d(%rip), %ymm3 # 0x9c510 <anon.5278ce255300ec7b361bbf944ee0b803.26.llvm.754055615369681222>
  336a7d:      	nopl	(%rax)
  336a80:      	vmovq	%xmm1, %rax
  336a85:      	addq	%rdi, %rax
  336a88:      	vmovdqu	(%rbx,%rax,8), %ymm4
  336a8d:      	vmovdqu	(%r13,%rax,8), %ymm5
  336a94:      	vpcmpeqq	%zmm2, %zmm4, %k1
  336a9a:      	vpcmpeqq	%zmm2, %zmm5, %k2
  336aa0:      	korw	%k2, %k1, %k1
  336aa4:      	korw	%k1, %k0, %k0
  336aa8:      	vpcmpgtq	%ymm4, %ymm5, %ymm4
  336aad:      	vpsrlq	$0x3f, %ymm4, %ymm4
  336ab2:      	vpsllvq	%ymm1, %ymm4, %ymm4
  336ab7:      	vpor	%ymm0, %ymm4, %ymm0
  336abb:      	vpaddq	%ymm3, %ymm1, %ymm1
  336abf:      	addq	$0x4, %r8
  336ac3:      	jne	0x336a80 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0xff0>
  336ac5:      	vextracti128	$0x1, %ymm0, %xmm1
  336acb:      	vpor	%xmm1, %xmm0, %xmm0
  336acf:      	vpshufd	$0xee, %xmm0, %xmm1     # xmm1 = xmm0[2,3,2,3]
  336ad4:      	vpor	%xmm1, %xmm0, %xmm0
  336ad8:      	vmovq	%xmm0, %r14
  336add:      	kmovd	%k0, %eax
  336ae1:      	testb	$0xf, %al
  336ae3:      	setne	%r15b
  336ae7:      	cmpl	%ecx, %r11d
  336aea:      	je	0x336b34 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x10a4>
  336aec:      	shlq	$0x3, %rdi
  336af0:      	addq	%rdi, %r13
  336af3:      	addq	%rdi, %rbx
  336af6:      	nopw	%cs:(%rax,%rax)
  336b00:      	movq	(%rbx,%rcx,8), %rax
  336b04:      	movq	(%r13,%rcx,8), %rdx
  336b09:      	cmpq	%r10, %rax
  336b0c:      	sete	%sil
  336b10:      	cmpq	%r10, %rdx
  336b13:      	sete	%dil
  336b17:      	orb	%sil, %dil
  336b1a:      	xorl	%esi, %esi
  336b1c:      	cmpq	%rdx, %rax
  336b1f:      	setl	%sil
  336b23:      	orb	%dil, %r15b
  336b26:      	shlq	%cl, %rsi
  336b29:      	incq	%rcx
  336b2c:      	orq	%rsi, %r14
  336b2f:      	cmpq	%rcx, %r11
  336b32:      	jne	0x336b00 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x1070>
  336b34:      	andb	$0x1, %r15b
  336b38:      	movq	-0x48(%rbp), %rax
  336b3c:      	movb	%r15b, 0x38(%rax)
  336b40:      	movq	-0x58(%rbp), %rdi
  336b44:      	cmpq	%r9, %rdi
  336b47:      	jae	0x336b74 <_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor7execute11ExecuteRowsINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB3X_B3U_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5P_s_0E0NCB3c_s_0B6B_E0EB3X_+0x10e4>
  336b49:      	movq	-0x38(%rbp), %rax
  336b4d:      	movq	%r14, (%rax)
  336b50:      	addq	$0x38, %rsp
  336b54:      	popq	%rbx
  336b55:      	popq	%r12
  336b57:      	popq	%r13
  336b59:      	popq	%r14
  336b5b:      	popq	%r15
  336b5d:      	popq	%rbp
  336b5e:      	vzeroupper
  336b61:      	retq
  336b62:      	leaq	0xfb0317(%rip), %rcx    # 0x12e6e80 <vtable.4.llvm.9723113569546643380+0x38>
  336b69:      	xorl	%edi, %edi
  336b6b:      	movq	%r9, %rdx
  336b6e:      	callq	*0x100c8f4(%rip)        # 0x1343468 <writev+0x1343468>
  336b74:      	leaq	0xfb02ed(%rip), %rdx    # 0x12e6e68 <vtable.4.llvm.9723113569546643380+0x20>
  336b7b:      	movq	%r9, %rsi
  336b7e:      	vzeroupper
  336b81:      	callq	*0x100c429(%rip)        # 0x1342fb0 <writev+0x1342fb0>
