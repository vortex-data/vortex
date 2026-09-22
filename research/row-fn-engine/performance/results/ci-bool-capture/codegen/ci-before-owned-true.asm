
/tmp/row-fn-bool-ci-artifact-before/codspeed/walltime/vortex-array/row_fn_bool_retry:	file format elf64-x86-64

Disassembly of section .text:

0000000000353b70 <_RINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool18execute_owned_boolTxxEubKb1_NCINvYINtNtNtB6_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB1N_EENtNtB21_11row_visitor10RowVisitor19visit_deferred_boolB1G_bKB1N_NCINvXB2C_B2z_NtNtB6_6row_fn5RowFn8dispatchB1W_E0NCB4u_s_0E0NCB1S_s_0B5f_EB2C_>:
  353b70:      	pushq	%rbp
  353b71:      	movq	%rsp, %rbp
  353b74:      	pushq	%r15
  353b76:      	pushq	%r14
  353b78:      	pushq	%r13
  353b7a:      	pushq	%r12
  353b7c:      	pushq	%rbx
  353b7d:      	subq	$0x168, %rsp            # imm = 0x168
  353b84:      	movq	%rdx, %r14
  353b87:      	movq	%rsi, %r15
  353b8a:      	movq	%rdi, %r12
  353b8d:      	leaq	-0xb0(%rbp), %rdi
  353b94:      	callq	0x356e80 <_RNvXs4_NtNtNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row5types7element5tuple13element_tupleTxxENtB5_12ElementTuple6decodeCslTNGN9sYpGe_17row_fn_bool_retry>
  353b99:      	movq	-0xb0(%rbp), %rax
  353ba0:      	vmovups	-0xa8(%rbp), %zmm0
  353baa:      	vmovups	%zmm0, -0x110(%rbp)
  353bb4:      	cmpq	$-0x1, %rax
  353bb8:      	je	0x353bf4 <_RINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool18execute_owned_boolTxxEubKb1_NCINvYINtNtNtB6_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB1N_EENtNtB21_11row_visitor10RowVisitor19visit_deferred_boolB1G_bKB1N_NCINvXB2C_B2z_NtNtB6_6row_fn5RowFn8dispatchB1W_E0NCB4u_s_0E0NCB1S_s_0B5f_EB2C_+0x84>
  353bba:      	movq	-0x68(%rbp), %rcx
  353bbe:      	vmovups	-0x110(%rbp), %zmm0
  353bc8:      	vmovups	%zmm0, 0x8(%r12)
  353bd3:      	movq	%rax, (%r12)
  353bd7:      	movq	%rcx, 0x48(%r12)
  353bdc:      	movq	%r12, %rax
  353bdf:      	addq	$0x168, %rsp            # imm = 0x168
  353be6:      	popq	%rbx
  353be7:      	popq	%r12
  353be9:      	popq	%r13
  353beb:      	popq	%r14
  353bed:      	popq	%r15
  353bef:      	popq	%rbp
  353bf0:      	vzeroupper
  353bf3:      	retq
  353bf4:      	vmovups	-0x110(%rbp), %zmm0
  353bfe:      	vmovups	%zmm0, -0x160(%rbp)
  353c08:      	movq	%r15, %rdi
  353c0b:      	vzeroupper
  353c0e:      	callq	*0x28(%r14)
  353c12:      	movq	%rax, -0x180(%rbp)
  353c19:      	movq	-0x160(%rbp), %rcx
  353c20:      	testq	%rcx, %rcx
  353c23:      	je	0x353d11 <_RINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool18execute_owned_boolTxxEubKb1_NCINvYINtNtNtB6_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB1N_EENtNtB21_11row_visitor10RowVisitor19visit_deferred_boolB1G_bKB1N_NCINvXB2C_B2z_NtNtB6_6row_fn5RowFn8dispatchB1W_E0NCB4u_s_0E0NCB1S_s_0B5f_EB2C_+0x1a1>
  353c29:      	movq	%rcx, %r13
  353c2c:      	movq	%rax, %rdx
  353c2f:      	cmpq	%rax, -0x158(%rbp)
  353c36:      	jne	0x353c5b <_RINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool18execute_owned_boolTxxEubKb1_NCINvYINtNtNtB6_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB1N_EENtNtB21_11row_visitor10RowVisitor19visit_deferred_boolB1G_bKB1N_NCINvXB2C_B2z_NtNtB6_6row_fn5RowFn8dispatchB1W_E0NCB4u_s_0E0NCB1S_s_0B5f_EB2C_+0xeb>
  353c38:      	movq	-0x140(%rbp), %rsi
  353c3f:      	testq	%rsi, %rsi
  353c42:      	je	0x353d2b <_RINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool18execute_owned_boolTxxEubKb1_NCINvYINtNtNtB6_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB1N_EENtNtB21_11row_visitor10RowVisitor19visit_deferred_boolB1G_bKB1N_NCINvXB2C_B2z_NtNtB6_6row_fn5RowFn8dispatchB1W_E0NCB4u_s_0E0NCB1S_s_0B5f_EB2C_+0x1bb>
  353c48:      	movq	%rsi, %rbx
  353c4b:      	movq	%rax, %rdi
  353c4e:      	cmpq	%rax, -0x138(%rbp)
  353c55:      	je	0x353d35 <_RINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool18execute_owned_boolTxxEubKb1_NCINvYINtNtNtB6_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB1N_EENtNtB21_11row_visitor10RowVisitor19visit_deferred_boolB1G_bKB1N_NCINvXB2C_B2z_NtNtB6_6row_fn5RowFn8dispatchB1W_E0NCB4u_s_0E0NCB1S_s_0B5f_EB2C_+0x1c5>
  353c5b:      	leaq	-0x180(%rbp), %rax
  353c62:      	movq	%rax, -0x110(%rbp)
  353c69:      	movq	0xfe28b8(%rip), %rax    # 0x1336528 <writev+0x1336528>
  353c70:      	movq	%rax, -0x108(%rbp)
  353c77:      	leaq	-0x2c8225(%rip), %rax   # 0x8ba59 <anon.90ed1ae6b5658b2a871c3b59a02a3489.21.llvm.3791648811283390273+0x215>
  353c7e:      	movq	%rax, -0xd0(%rbp)
  353c85:      	leaq	-0x110(%rbp), %rax
  353c8c:      	movq	%rax, -0xc8(%rbp)
  353c93:      	leaq	0x1656(%rip), %rsi      # 0x3552f0 <_RNcNtNtCscHxMUXUtRzx_12vortex_error11VortexError5Other0>
  353c9a:      	leaq	-0xb0(%rbp), %rdi
  353ca1:      	leaq	-0xd0(%rbp), %rdx
  353ca8:      	callq	*0xfe289a(%rip)         # 0x1336548 <writev+0x1336548>
  353cae:      	vmovups	-0xb0(%rbp), %zmm0
  353cb8:      	vmovdqu64	-0xa0(%rbp), %zmm1
  353cc2:      	vmovdqu64	%zmm1, 0x10(%r12)
  353ccd:      	vmovups	%zmm0, (%r12)
  353cd4:      	cmpq	$0x0, -0x160(%rbp)
  353cdc:      	je	0x354fe7 <_RINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool18execute_owned_boolTxxEubKb1_NCINvYINtNtNtB6_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB1N_EENtNtB21_11row_visitor10RowVisitor19visit_deferred_boolB1G_bKB1N_NCINvXB2C_B2z_NtNtB6_6row_fn5RowFn8dispatchB1W_E0NCB4u_s_0E0NCB1S_s_0B5f_EB2C_+0x1477>
  353ce2:      	movq	-0x150(%rbp), %rax
  353ce9:      	testq	%rax, %rax
  353cec:      	je	0x354fe7 <_RINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool18execute_owned_boolTxxEubKb1_NCINvYINtNtNtB6_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB1N_EENtNtB21_11row_visitor10RowVisitor19visit_deferred_boolB1G_bKB1N_NCINvXB2C_B2z_NtNtB6_6row_fn5RowFn8dispatchB1W_E0NCB4u_s_0E0NCB1S_s_0B5f_EB2C_+0x1477>
  353cf2:      	lock
  353cf3:      	decq	(%rax)
  353cf6:      	jne	0x354fe7 <_RINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool18execute_owned_boolTxxEubKb1_NCINvYINtNtNtB6_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB1N_EENtNtB21_11row_visitor10RowVisitor19visit_deferred_boolB1G_bKB1N_NCINvXB2C_B2z_NtNtB6_6row_fn5RowFn8dispatchB1W_E0NCB4u_s_0E0NCB1S_s_0B5f_EB2C_+0x1477>
  353cfc:      	leaq	-0x150(%rbp), %rdi
  353d03:      	vzeroupper
  353d06:      	callq	*0xfe28ac(%rip)         # 0x13365b8 <writev+0x13365b8>
  353d0c:      	jmp	0x354fe7 <_RINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool18execute_owned_boolTxxEubKb1_NCINvYINtNtNtB6_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB1N_EENtNtB21_11row_visitor10RowVisitor19visit_deferred_boolB1G_bKB1N_NCINvXB2C_B2z_NtNtB6_6row_fn5RowFn8dispatchB1W_E0NCB4u_s_0E0NCB1S_s_0B5f_EB2C_+0x1477>
  353d11:      	leaq	-0x158(%rbp), %rdx
  353d18:      	movq	%rax, %r13
  353d1b:      	movq	-0x140(%rbp), %rsi
  353d22:      	testq	%rsi, %rsi
  353d25:      	jne	0x353c48 <_RINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool18execute_owned_boolTxxEubKb1_NCINvYINtNtNtB6_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB1N_EENtNtB21_11row_visitor10RowVisitor19visit_deferred_boolB1G_bKB1N_NCINvXB2C_B2z_NtNtB6_6row_fn5RowFn8dispatchB1W_E0NCB4u_s_0E0NCB1S_s_0B5f_EB2C_+0xd8>
  353d2b:      	leaq	-0x138(%rbp), %rdi
  353d32:      	movq	%rax, %rbx
  353d35:      	movq	%rdi, -0x30(%rbp)
  353d39:      	movq	%rsi, -0x48(%rbp)
  353d3d:      	movq	%rdx, -0x50(%rbp)
  353d41:      	movq	%rcx, -0x38(%rbp)
  353d45:      	movq	%rax, %rcx
  353d48:      	shrq	$0x6, %rcx
  353d4c:      	xorl	%r14d, %r14d
  353d4f:      	movq	%rax, %rdx
  353d52:      	andq	$0x3f, %rdx
  353d56:      	movq	%rdx, -0x120(%rbp)
  353d5d:      	setne	%r14b
  353d61:      	movq	%rcx, -0x40(%rbp)
  353d65:      	addq	%rcx, %r14
  353d68:      	testq	%rax, %rax
  353d6b:      	leaq	0x100(,%r14,8), %rdx
  353d73:      	cmoveq	%rax, %rdx
  353d77:      	movl	$0x100, %ecx            # imm = 0x100
  353d7c:      	movl	$0x1, %esi
  353d81:      	cmoveq	%rcx, %rsi
  353d85:      	leaq	-0xb0(%rbp), %rdi
  353d8c:      	xorl	%ecx, %ecx
  353d8e:      	movq	%rax, %r15
  353d91:      	callq	*0xfe2d31(%rip)         # 0x1336ac8 <writev+0x1336ac8>
  353d97:      	shlq	$0x3, %r14
  353d9b:      	movq	-0xa0(%rbp), %rax
  353da2:      	leaq	0xff(%rax), %rcx
  353da9:      	andq	$-0x100, %rcx
  353db0:      	vmovups	-0xb0(%rbp), %xmm0
  353db8:      	vmovaps	%xmm0, -0xd0(%rbp)
  353dc0:      	vmovdqu	-0x98(%rbp), %xmm0
  353dc8:      	vmovdqa	%xmm0, -0x170(%rbp)
  353dd0:      	cmpq	$0x40, %r15
  353dd4:      	movq	%r15, %r11
  353dd7:      	movq	%r12, -0x178(%rbp)
  353dde:      	jae	0x353e1a <_RINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool18execute_owned_boolTxxEubKb1_NCINvYINtNtNtB6_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB1N_EENtNtB21_11row_visitor10RowVisitor19visit_deferred_boolB1G_bKB1N_NCINvXB2C_B2z_NtNtB6_6row_fn5RowFn8dispatchB1W_E0NCB4u_s_0E0NCB1S_s_0B5f_EB2C_+0x2aa>
  353de0:      	testq	%r11, %r11
  353de3:      	movq	-0x50(%rbp), %r8
  353de7:      	movq	-0x48(%rbp), %r9
  353deb:      	je	0x35407f <_RINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool18execute_owned_boolTxxEubKb1_NCINvYINtNtNtB6_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB1N_EENtNtB21_11row_visitor10RowVisitor19visit_deferred_boolB1G_bKB1N_NCINvXB2C_B2z_NtNtB6_6row_fn5RowFn8dispatchB1W_E0NCB4u_s_0E0NCB1S_s_0B5f_EB2C_+0x50f>
  353df1:      	cmpq	$0x4, %r11
  353df5:      	movq	%rcx, %r15
  353df8:      	movq	%r14, -0x60(%rbp)
  353dfc:      	movq	%rax, -0x58(%rbp)
  353e00:      	jae	0x354256 <_RINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool18execute_owned_boolTxxEubKb1_NCINvYINtNtNtB6_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB1N_EENtNtB21_11row_visitor10RowVisitor19visit_deferred_boolB1G_bKB1N_NCINvXB2C_B2z_NtNtB6_6row_fn5RowFn8dispatchB1W_E0NCB4u_s_0E0NCB1S_s_0B5f_EB2C_+0x6e6>
  353e06:      	xorl	%r12d, %r12d
  353e09:      	xorl	%edx, %edx
  353e0b:      	xorl	%ecx, %ecx
  353e0d:      	movq	-0x38(%rbp), %r14
  353e11:      	movq	-0x30(%rbp), %rax
  353e15:      	jmp	0x354859 <_RINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool18execute_owned_boolTxxEubKb1_NCINvYINtNtNtB6_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB1N_EENtNtB21_11row_visitor10RowVisitor19visit_deferred_boolB1G_bKB1N_NCINvXB2C_B2z_NtNtB6_6row_fn5RowFn8dispatchB1W_E0NCB4u_s_0E0NCB1S_s_0B5f_EB2C_+0xce9>
  353e1a:      	movq	%r14, -0x60(%rbp)
  353e1e:      	shlq	$0x3, -0x40(%rbp)
  353e23:      	movabsq	$0x7fffffffffffffff, %rdx # imm = 0x7FFFFFFFFFFFFFFF
  353e2d:      	movq	-0x38(%rbp), %r14
  353e31:      	testq	%r14, %r14
  353e34:      	movq	-0x50(%rbp), %r10
  353e38:      	movq	-0x48(%rbp), %rsi
  353e3c:      	movq	%rax, -0x58(%rbp)
  353e40:      	je	0x35408a <_RINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool18execute_owned_boolTxxEubKb1_NCINvYINtNtNtB6_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB1N_EENtNtB21_11row_visitor10RowVisitor19visit_deferred_boolB1G_bKB1N_NCINvXB2C_B2z_NtNtB6_6row_fn5RowFn8dispatchB1W_E0NCB4u_s_0E0NCB1S_s_0B5f_EB2C_+0x51a>
  353e46:      	testq	%rsi, %rsi
  353e49:      	movq	%rcx, %r15
  353e4c:      	je	0x354270 <_RINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool18execute_owned_boolTxxEubKb1_NCINvYINtNtNtB6_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB1N_EENtNtB21_11row_visitor10RowVisitor19visit_deferred_boolB1G_bKB1N_NCINvXB2C_B2z_NtNtB6_6row_fn5RowFn8dispatchB1W_E0NCB4u_s_0E0NCB1S_s_0B5f_EB2C_+0x700>
  353e52:      	movl	$0x1c0, %ecx            # imm = 0x1C0
  353e57:      	xorl	%esi, %esi
  353e59:      	vpbroadcastq	-0x2b7813(%rip), %zmm0 # 0x9c650 <anon.4e66a2221d34052adc0f362485be06ac.633.llvm.27319815200592487+0x50>
  353e63:      	vmovdqa	-0x2ce1db(%rip), %xmm1  # 0x85c90 <anon.37e9284aed57eb9304064780576275d5.103.llvm.4630057869998642792+0x90>
  353e6b:      	vmovdqa64	-0x2b1ab5(%rip), %zmm2 # 0xa23c0 <anon.4e8d97506139486fadb1e849b133b59b.19.llvm.16316955986290925330+0x2ed>
  353e75:      	vmovdqa64	-0x2b1a7f(%rip), %zmm3 # 0xa2400 <anon.4e8d97506139486fadb1e849b133b59b.19.llvm.16316955986290925330+0x32d>
  353e7f:      	xorl	%r12d, %r12d
  353e82:      	movq	-0x40(%rbp), %rax
  353e86:      	nopw	%cs:(%rax,%rax)
  353e90:      	kmovd	%r12d, %k0
  353e95:      	kshiftlb	$0x7, %k0, %k0
  353e9b:      	kshiftrb	$0x7, %k0, %k0
  353ea1:      	vmovdqu64	-0x1c0(%r13,%rcx), %zmm7
  353ea9:      	vmovdqu64	-0x180(%r13,%rcx), %zmm6
  353eb1:      	vmovdqu64	-0x140(%r13,%rcx), %zmm5
  353eb9:      	vmovdqu64	-0x100(%r13,%rcx), %zmm4
  353ec1:      	vmovdqu64	-0x1c0(%rbx,%rcx), %zmm11
  353ec9:      	vmovdqu64	-0x180(%rbx,%rcx), %zmm10
  353ed1:      	vmovdqu64	-0x140(%rbx,%rcx), %zmm9
  353ed9:      	vmovdqu64	-0x100(%rbx,%rcx), %zmm8
  353ee1:      	vpcmpeqq	%zmm0, %zmm7, %k1
  353ee7:      	vpcmpeqq	%zmm0, %zmm6, %k2
  353eed:      	vpcmpeqq	%zmm0, %zmm5, %k4
  353ef3:      	vpcmpeqq	%zmm0, %zmm4, %k5
  353ef9:      	vpcmpeqq	%zmm0, %zmm11, %k3
  353eff:      	korb	%k3, %k1, %k1
  353f03:      	korb	%k1, %k0, %k3
  353f07:      	vpcmpeqq	%zmm0, %zmm10, %k0
  353f0d:      	korb	%k0, %k2, %k2
  353f11:      	vpcmpeqq	%zmm0, %zmm9, %k0
  353f17:      	korb	%k0, %k4, %k1
  353f1b:      	vpcmpeqq	%zmm0, %zmm8, %k0
  353f21:      	korb	%k0, %k5, %k0
  353f25:      	vmovdqu64	-0xc0(%r13,%rcx), %zmm12
  353f2d:      	vmovdqu64	-0x80(%r13,%rcx), %zmm13
  353f35:      	vmovdqu64	-0xc0(%rbx,%rcx), %zmm14
  353f3d:      	vmovdqu64	-0x80(%rbx,%rcx), %zmm15
  353f45:      	vpcmpeqq	%zmm0, %zmm12, %k4
  353f4b:      	vpcmpeqq	%zmm0, %zmm14, %k5
  353f51:      	korb	%k5, %k4, %k4
  353f55:      	vpcmpeqq	%zmm0, %zmm13, %k5
  353f5b:      	korb	%k4, %k3, %k3
  353f5f:      	vpcmpeqq	%zmm0, %zmm15, %k4
  353f65:      	korb	%k4, %k5, %k4
  353f69:      	vmovdqu64	-0x40(%r13,%rcx), %zmm16
  353f71:      	vmovdqu64	-0x40(%rbx,%rcx), %zmm17
  353f79:      	korb	%k4, %k2, %k2
  353f7d:      	vpcmpeqq	%zmm0, %zmm16, %k4
  353f83:      	korb	%k3, %k2, %k2
  353f87:      	vpcmpeqq	%zmm0, %zmm17, %k3
  353f8d:      	korb	%k3, %k4, %k3
  353f91:      	vmovdqu64	(%r13,%rcx), %zmm18
  353f99:      	vmovdqu64	(%rbx,%rcx), %zmm19
  353fa0:      	korb	%k3, %k1, %k1
  353fa4:      	vpcmpeqq	%zmm0, %zmm18, %k3
  353faa:      	korb	%k2, %k1, %k1
  353fae:      	vpcmpeqq	%zmm0, %zmm19, %k2
  353fb4:      	korb	%k2, %k3, %k2
  353fb8:      	korb	%k2, %k0, %k0
  353fbc:      	kortestb	%k1, %k0
  353fc0:      	vpcmpgtq	%zmm7, %zmm11, %k1
  353fc6:      	vmovdqu8	%xmm1, %xmm7 {%k1} {z}
  353fcc:      	vpcmpgtq	%zmm6, %zmm10, %k1
  353fd2:      	vmovdqu8	%xmm1, %xmm6 {%k1} {z}
  353fd8:      	vpcmpgtq	%zmm5, %zmm9, %k1
  353fde:      	vmovdqu8	%xmm1, %xmm5 {%k1} {z}
  353fe4:      	vpcmpgtq	%zmm4, %zmm8, %k1
  353fea:      	vpunpcklqdq	%xmm6, %xmm7, %xmm4 # xmm4 = xmm7[0],xmm6[0]
  353fee:      	vinserti128	$0x1, %xmm5, %ymm4, %ymm4
  353ff4:      	vmovdqu8	%xmm1, %xmm5 {%k1} {z}
  353ffa:      	vpbroadcastq	%xmm5, %ymm5
  353fff:      	vpblendd	$0xc0, %ymm5, %ymm4, %ymm4 # ymm4 = ymm4[0,1,2,3,4,5],ymm5[6,7]
  354005:      	vpcmpgtq	%zmm12, %zmm14, %k1
  35400b:      	vmovdqu8	%xmm1, %xmm5 {%k1} {z}
  354011:      	vpcmpgtq	%zmm13, %zmm15, %k1
  354017:      	vmovdqu8	%xmm1, %xmm6 {%k1} {z}
  35401d:      	vpcmpgtq	%zmm16, %zmm17, %k1
  354023:      	vmovdqu8	%xmm1, %xmm7 {%k1} {z}
  354029:      	vpcmpgtq	%zmm18, %zmm19, %k1
  35402f:      	vmovdqu8	%xmm1, %xmm8 {%k1} {z}
  354035:      	vinserti32x4	$0x2, %xmm5, %zmm0, %zmm5
  35403c:      	vinserti64x4	$0x0, %ymm4, %zmm5, %zmm4
  354043:      	vpermt2q	%zmm6, %zmm2, %zmm4
  354049:      	vinserti32x4	$0x3, %xmm7, %zmm4, %zmm4
  354050:      	vpermt2q	%zmm8, %zmm3, %zmm4
  354056:      	vptestmb	%zmm4, %zmm4, %k0
  35405c:      	kmovq	%k0, (%r15,%rsi)
  354062:      	setne	%r12b
  354066:      	addq	$0x8, %rsi
  35406a:      	addq	$0x200, %rcx            # imm = 0x200
  354071:      	cmpq	%rsi, %rax
  354074:      	jne	0x353e90 <_RINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool18execute_owned_boolTxxEubKb1_NCINvYINtNtNtB6_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB1N_EENtNtB21_11row_visitor10RowVisitor19visit_deferred_boolB1G_bKB1N_NCINvXB2C_B2z_NtNtB6_6row_fn5RowFn8dispatchB1W_E0NCB4u_s_0E0NCB1S_s_0B5f_EB2C_+0x320>
  35407a:      	jmp	0x35499c <_RINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool18execute_owned_boolTxxEubKb1_NCINvYINtNtNtB6_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB1N_EENtNtB21_11row_visitor10RowVisitor19visit_deferred_boolB1G_bKB1N_NCINvXB2C_B2z_NtNtB6_6row_fn5RowFn8dispatchB1W_E0NCB4u_s_0E0NCB1S_s_0B5f_EB2C_+0xe2c>
  35407f:      	xorl	%r12d, %r12d
  354082:      	movq	%rcx, %r15
  354085:      	jmp	0x354d9d <_RINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool18execute_owned_boolTxxEubKb1_NCINvYINtNtNtB6_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB1N_EENtNtB21_11row_visitor10RowVisitor19visit_deferred_boolB1G_bKB1N_NCINvXB2C_B2z_NtNtB6_6row_fn5RowFn8dispatchB1W_E0NCB4u_s_0E0NCB1S_s_0B5f_EB2C_+0x122d>
  35408a:      	testq	%rsi, %rsi
  35408d:      	movq	%rcx, %r15
  354090:      	je	0x3548ce <_RINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool18execute_owned_boolTxxEubKb1_NCINvYINtNtNtB6_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB1N_EENtNtB21_11row_visitor10RowVisitor19visit_deferred_boolB1G_bKB1N_NCINvXB2C_B2z_NtNtB6_6row_fn5RowFn8dispatchB1W_E0NCB4u_s_0E0NCB1S_s_0B5f_EB2C_+0xd5e>
  354096:      	leaq	0x1c0(%rbx), %rcx
  35409d:      	xorl	%esi, %esi
  35409f:      	movl	$0xff, %edi
  3540a4:      	vpbroadcastq	-0x2b7a5e(%rip), %zmm0 # 0x9c650 <anon.4e66a2221d34052adc0f362485be06ac.633.llvm.27319815200592487+0x50>
  3540ae:      	vmovdqa	-0x2ce426(%rip), %xmm1  # 0x85c90 <anon.37e9284aed57eb9304064780576275d5.103.llvm.4630057869998642792+0x90>
  3540b6:      	vmovdqa64	-0x2b1d00(%rip), %zmm2 # 0xa23c0 <anon.4e8d97506139486fadb1e849b133b59b.19.llvm.16316955986290925330+0x2ed>
  3540c0:      	vmovdqa64	-0x2b1cca(%rip), %zmm3 # 0xa2400 <anon.4e8d97506139486fadb1e849b133b59b.19.llvm.16316955986290925330+0x32d>
  3540ca:      	xorl	%r12d, %r12d
  3540cd:      	movq	-0x40(%rbp), %rax
  3540d1:      	nopw	%cs:(%rax,%rax)
  3540e0:      	movq	(%r10), %r8
  3540e3:      	cmpq	%rdx, %r8
  3540e6:      	movl	$0x0, %r9d
  3540ec:      	cmovel	%edi, %r9d
  3540f0:      	kmovd	%r9d, %k0
  3540f5:      	kmovd	%r12d, %k1
  3540fa:      	kshiftlb	$0x7, %k1, %k1
  354100:      	kshiftrb	$0x7, %k1, %k1
  354106:      	vmovdqu64	-0x1c0(%rcx), %zmm4
  35410d:      	vmovdqu64	-0x180(%rcx), %zmm5
  354114:      	vmovdqu64	-0x140(%rcx), %zmm6
  35411b:      	vmovdqu64	-0x100(%rcx), %zmm7
  354122:      	vpcmpeqq	%zmm0, %zmm4, %k2
  354128:      	korb	%k1, %k2, %k1
  35412c:      	vpcmpeqq	%zmm0, %zmm5, %k4
  354132:      	vpcmpeqq	%zmm0, %zmm6, %k3
  354138:      	vpcmpeqq	%zmm0, %zmm7, %k2
  35413e:      	vmovdqu64	-0xc0(%rcx), %zmm8
  354145:      	vmovdqu64	-0x80(%rcx), %zmm9
  35414c:      	vmovdqu64	-0x40(%rcx), %zmm10
  354153:      	vmovdqu64	(%rcx), %zmm11
  354159:      	vpcmpeqq	%zmm0, %zmm9, %k5
  35415f:      	korb	%k5, %k4, %k4
  354163:      	vpcmpeqq	%zmm0, %zmm10, %k5
  354169:      	korb	%k5, %k3, %k3
  35416d:      	vpcmpeqq	%zmm0, %zmm11, %k5
  354173:      	korb	%k5, %k2, %k2
  354177:      	vpcmpeqq	%zmm0, %zmm8, %k5
  35417d:      	korb	%k0, %k5, %k0
  354181:      	korb	%k0, %k1, %k0
  354185:      	korb	%k0, %k4, %k0
  354189:      	korb	%k0, %k3, %k0
  35418d:      	kortestb	%k0, %k2
  354191:      	vpbroadcastq	%r8, %zmm12
  354197:      	vpcmpgtq	%zmm12, %zmm4, %k1
  35419d:      	vmovdqu8	%xmm1, %xmm4 {%k1} {z}
  3541a3:      	vpcmpgtq	%zmm12, %zmm5, %k1
  3541a9:      	vmovdqu8	%xmm1, %xmm5 {%k1} {z}
  3541af:      	vpcmpgtq	%zmm12, %zmm6, %k1
  3541b5:      	vmovdqu8	%xmm1, %xmm6 {%k1} {z}
  3541bb:      	vpcmpgtq	%zmm12, %zmm7, %k1
  3541c1:      	vpunpcklqdq	%xmm5, %xmm4, %xmm4 # xmm4 = xmm4[0],xmm5[0]
  3541c5:      	vinserti128	$0x1, %xmm6, %ymm4, %ymm4
  3541cb:      	vmovdqu8	%xmm1, %xmm5 {%k1} {z}
  3541d1:      	vpbroadcastq	%xmm5, %ymm5
  3541d6:      	vpblendd	$0xc0, %ymm5, %ymm4, %ymm4 # ymm4 = ymm4[0,1,2,3,4,5],ymm5[6,7]
  3541dc:      	vpcmpgtq	%zmm12, %zmm8, %k1
  3541e2:      	vmovdqu8	%xmm1, %xmm5 {%k1} {z}
  3541e8:      	vpcmpgtq	%zmm12, %zmm9, %k1
  3541ee:      	vmovdqu8	%xmm1, %xmm6 {%k1} {z}
  3541f4:      	vpcmpgtq	%zmm12, %zmm10, %k1
  3541fa:      	vmovdqu8	%xmm1, %xmm7 {%k1} {z}
  354200:      	vpcmpgtq	%zmm12, %zmm11, %k1
  354206:      	vmovdqu8	%xmm1, %xmm8 {%k1} {z}
  35420c:      	vinserti32x4	$0x2, %xmm5, %zmm0, %zmm5
  354213:      	vinserti64x4	$0x0, %ymm4, %zmm5, %zmm4
  35421a:      	vpermt2q	%zmm6, %zmm2, %zmm4
  354220:      	vinserti32x4	$0x3, %xmm7, %zmm4, %zmm4
  354227:      	vpermt2q	%zmm8, %zmm3, %zmm4
  35422d:      	vptestmb	%zmm4, %zmm4, %k0
  354233:      	kmovq	%k0, (%r15,%rsi)
  354239:      	setne	%r12b
  35423d:      	addq	$0x8, %rsi
  354241:      	addq	$0x200, %rcx            # imm = 0x200
  354248:      	cmpq	%rsi, %rax
  35424b:      	jne	0x3540e0 <_RINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool18execute_owned_boolTxxEubKb1_NCINvYINtNtNtB6_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB1N_EENtNtB21_11row_visitor10RowVisitor19visit_deferred_boolB1G_bKB1N_NCINvXB2C_B2z_NtNtB6_6row_fn5RowFn8dispatchB1W_E0NCB4u_s_0E0NCB1S_s_0B5f_EB2C_+0x570>
  354251:      	jmp	0x35499c <_RINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool18execute_owned_boolTxxEubKb1_NCINvYINtNtNtB6_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB1N_EENtNtB21_11row_visitor10RowVisitor19visit_deferred_boolB1G_bKB1N_NCINvXB2C_B2z_NtNtB6_6row_fn5RowFn8dispatchB1W_E0NCB4u_s_0E0NCB1S_s_0B5f_EB2C_+0xe2c>
  354256:      	cmpq	$0x10, %r11
  35425a:      	jae	0x354427 <_RINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool18execute_owned_boolTxxEubKb1_NCINvYINtNtNtB6_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB1N_EENtNtB21_11row_visitor10RowVisitor19visit_deferred_boolB1G_bKB1N_NCINvXB2C_B2z_NtNtB6_6row_fn5RowFn8dispatchB1W_E0NCB4u_s_0E0NCB1S_s_0B5f_EB2C_+0x8b7>
  354260:      	xorl	%ecx, %ecx
  354262:      	xorl	%r12d, %r12d
  354265:      	xorl	%edx, %edx
  354267:      	movq	-0x30(%rbp), %rax
  35426b:      	jmp	0x35474d <_RINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool18execute_owned_boolTxxEubKb1_NCINvYINtNtNtB6_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB1N_EENtNtB21_11row_visitor10RowVisitor19visit_deferred_boolB1G_bKB1N_NCINvXB2C_B2z_NtNtB6_6row_fn5RowFn8dispatchB1W_E0NCB4u_s_0E0NCB1S_s_0B5f_EB2C_+0xbdd>
  354270:      	leaq	0x1c0(%r13), %rcx
  354277:      	xorl	%esi, %esi
  354279:      	movl	$0xff, %edi
  35427e:      	vpbroadcastq	-0x2b7c38(%rip), %zmm0 # 0x9c650 <anon.4e66a2221d34052adc0f362485be06ac.633.llvm.27319815200592487+0x50>
  354288:      	vmovdqa	-0x2ce600(%rip), %xmm1  # 0x85c90 <anon.37e9284aed57eb9304064780576275d5.103.llvm.4630057869998642792+0x90>
  354290:      	vmovdqa64	-0x2b1eda(%rip), %zmm2 # 0xa23c0 <anon.4e8d97506139486fadb1e849b133b59b.19.llvm.16316955986290925330+0x2ed>
  35429a:      	vmovdqa64	-0x2b1ea4(%rip), %zmm3 # 0xa2400 <anon.4e8d97506139486fadb1e849b133b59b.19.llvm.16316955986290925330+0x32d>
  3542a4:      	xorl	%r12d, %r12d
  3542a7:      	movq	-0x30(%rbp), %rax
  3542ab:      	nopl	(%rax,%rax)
  3542b0:      	movq	(%rax), %r8
  3542b3:      	cmpq	%rdx, %r8
  3542b6:      	movl	$0x0, %r9d
  3542bc:      	cmovel	%edi, %r9d
  3542c0:      	kmovd	%r9d, %k0
  3542c5:      	kmovd	%r12d, %k1
  3542ca:      	kshiftlb	$0x7, %k1, %k1
  3542d0:      	kshiftrb	$0x7, %k1, %k1
  3542d6:      	vmovdqu64	-0x1c0(%rcx), %zmm4
  3542dd:      	vmovdqu64	-0x180(%rcx), %zmm5
  3542e4:      	vmovdqu64	-0x140(%rcx), %zmm6
  3542eb:      	vmovdqu64	-0x100(%rcx), %zmm7
  3542f2:      	vpcmpeqq	%zmm0, %zmm4, %k2
  3542f8:      	korb	%k1, %k2, %k1
  3542fc:      	vpcmpeqq	%zmm0, %zmm5, %k4
  354302:      	vpcmpeqq	%zmm0, %zmm6, %k3
  354308:      	vpcmpeqq	%zmm0, %zmm7, %k2
  35430e:      	vmovdqu64	-0xc0(%rcx), %zmm8
  354315:      	vmovdqu64	-0x80(%rcx), %zmm9
  35431c:      	vmovdqu64	-0x40(%rcx), %zmm10
  354323:      	vmovdqu64	(%rcx), %zmm11
  354329:      	vpcmpeqq	%zmm0, %zmm9, %k5
  35432f:      	korb	%k5, %k4, %k4
  354333:      	vpcmpeqq	%zmm0, %zmm10, %k5
  354339:      	korb	%k5, %k3, %k3
  35433d:      	vpcmpeqq	%zmm0, %zmm11, %k5
  354343:      	korb	%k5, %k2, %k2
  354347:      	vpcmpeqq	%zmm0, %zmm8, %k5
  35434d:      	korb	%k0, %k5, %k0
  354351:      	korb	%k0, %k1, %k0
  354355:      	korb	%k0, %k4, %k0
  354359:      	korb	%k0, %k3, %k0
  35435d:      	kortestb	%k0, %k2
  354361:      	vpbroadcastq	%r8, %zmm12
  354367:      	vpcmpgtq	%zmm4, %zmm12, %k1
  35436d:      	vmovdqu8	%xmm1, %xmm4 {%k1} {z}
  354373:      	vpcmpgtq	%zmm5, %zmm12, %k1
  354379:      	vmovdqu8	%xmm1, %xmm5 {%k1} {z}
  35437f:      	vpcmpgtq	%zmm6, %zmm12, %k1
  354385:      	vmovdqu8	%xmm1, %xmm6 {%k1} {z}
  35438b:      	vpcmpgtq	%zmm7, %zmm12, %k1
  354391:      	vpunpcklqdq	%xmm5, %xmm4, %xmm4 # xmm4 = xmm4[0],xmm5[0]
  354395:      	vinserti128	$0x1, %xmm6, %ymm4, %ymm4
  35439b:      	vmovdqu8	%xmm1, %xmm5 {%k1} {z}
  3543a1:      	vpbroadcastq	%xmm5, %ymm5
  3543a6:      	vpblendd	$0xc0, %ymm5, %ymm4, %ymm4 # ymm4 = ymm4[0,1,2,3,4,5],ymm5[6,7]
  3543ac:      	vpcmpgtq	%zmm8, %zmm12, %k1
  3543b2:      	vmovdqu8	%xmm1, %xmm5 {%k1} {z}
  3543b8:      	vpcmpgtq	%zmm9, %zmm12, %k1
  3543be:      	vmovdqu8	%xmm1, %xmm6 {%k1} {z}
  3543c4:      	vpcmpgtq	%zmm10, %zmm12, %k1
  3543ca:      	vmovdqu8	%xmm1, %xmm7 {%k1} {z}
  3543d0:      	vpcmpgtq	%zmm11, %zmm12, %k1
  3543d6:      	vmovdqu8	%xmm1, %xmm8 {%k1} {z}
  3543dc:      	vinserti32x4	$0x2, %xmm5, %zmm0, %zmm5
  3543e3:      	vinserti64x4	$0x0, %ymm4, %zmm5, %zmm4
  3543ea:      	vpermt2q	%zmm6, %zmm2, %zmm4
  3543f0:      	vinserti32x4	$0x3, %xmm7, %zmm4, %zmm4
  3543f7:      	vpermt2q	%zmm8, %zmm3, %zmm4
  3543fd:      	vptestmb	%zmm4, %zmm4, %k0
  354403:      	kmovq	%k0, (%r15,%rsi)
  354409:      	setne	%r12b
  35440d:      	addq	$0x8, %rsi
  354411:      	addq	$0x200, %rcx            # imm = 0x200
  354418:      	cmpq	%rsi, -0x40(%rbp)
  35441c:      	jne	0x3542b0 <_RINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool18execute_owned_boolTxxEubKb1_NCINvYINtNtNtB6_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB1N_EENtNtB21_11row_visitor10RowVisitor19visit_deferred_boolB1G_bKB1N_NCINvXB2C_B2z_NtNtB6_6row_fn5RowFn8dispatchB1W_E0NCB4u_s_0E0NCB1S_s_0B5f_EB2C_+0x740>
  354422:      	jmp	0x35499c <_RINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool18execute_owned_boolTxxEubKb1_NCINvYINtNtNtB6_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB1N_EENtNtB21_11row_visitor10RowVisitor19visit_deferred_boolB1G_bKB1N_NCINvXB2C_B2z_NtNtB6_6row_fn5RowFn8dispatchB1W_E0NCB4u_s_0E0NCB1S_s_0B5f_EB2C_+0xe2c>
  354427:      	movl	%r11d, %ecx
  35442a:      	andl	$0x30, %ecx
  35442d:      	vpbroadcastq	%r8, %zmm2
  354433:      	vpbroadcastq	%r13, %zmm4
  354439:      	vmovdqa64	-0x2b1fc3(%rip), %zmm6 # 0xa2480 <anon.4e8d97506139486fadb1e849b133b59b.19.llvm.16316955986290925330+0x3ad>
  354443:      	vmovdqa64	-0x2b1f8d(%rip), %zmm7 # 0xa24c0 <anon.4e8d97506139486fadb1e849b133b59b.19.llvm.16316955986290925330+0x3ed>
  35444d:      	vmovdqa64	%zmm2, %zmm8
  354453:      	vmovdqa64	%zmm2, %zmm5
  354459:      	movq	-0x38(%rbp), %rsi
  35445d:      	testq	%rsi, %rsi
  354460:      	movq	-0x30(%rbp), %rdi
  354464:      	je	0x354472 <_RINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool18execute_owned_boolTxxEubKb1_NCINvYINtNtNtB6_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB1N_EENtNtB21_11row_visitor10RowVisitor19visit_deferred_boolB1G_bKB1N_NCINvXB2C_B2z_NtNtB6_6row_fn5RowFn8dispatchB1W_E0NCB4u_s_0E0NCB1S_s_0B5f_EB2C_+0x902>
  354466:      	vpaddq	%zmm6, %zmm4, %zmm8
  35446c:      	vpaddq	%zmm7, %zmm4, %zmm5
  354472:      	vpbroadcastq	%rdi, %zmm3
  354478:      	kxnorb	%k0, %k0, %k1
  35447c:      	vpxor	%xmm0, %xmm0, %xmm0
  354480:      	kxnorb	%k0, %k0, %k2
  354484:      	vpxor	%xmm1, %xmm1, %xmm1
  354488:      	vpgatherqq	(,%zmm8), %zmm1 {%k2}
  354493:      	kxnorb	%k0, %k0, %k2
  354497:      	vpxor	%xmm8, %xmm8, %xmm8
  35449c:      	vpgatherqq	(,%zmm5), %zmm8 {%k2}
  3544a7:      	vpbroadcastq	%rbx, %zmm5
  3544ad:      	vmovdqa64	%zmm3, %zmm10
  3544b3:      	vmovdqa64	%zmm3, %zmm9
  3544b9:      	testq	%r9, %r9
  3544bc:      	je	0x3544ca <_RINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool18execute_owned_boolTxxEubKb1_NCINvYINtNtNtB6_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB1N_EENtNtB21_11row_visitor10RowVisitor19visit_deferred_boolB1G_bKB1N_NCINvXB2C_B2z_NtNtB6_6row_fn5RowFn8dispatchB1W_E0NCB4u_s_0E0NCB1S_s_0B5f_EB2C_+0x95a>
  3544be:      	vpaddq	%zmm6, %zmm5, %zmm10
  3544c4:      	vpaddq	%zmm7, %zmm5, %zmm9
  3544ca:      	kxnorb	%k0, %k0, %k2
  3544ce:      	vpxor	%xmm7, %xmm7, %xmm7
  3544d2:      	vpgatherqq	(,%zmm10), %zmm7 {%k2}
  3544dd:      	vpbroadcastq	-0x2b7e97(%rip), %zmm6 # 0x9c650 <anon.4e66a2221d34052adc0f362485be06ac.633.llvm.27319815200592487+0x50>
  3544e7:      	vpgatherqq	(,%zmm9), %zmm0 {%k1}
  3544f2:      	vpcmpeqq	%zmm6, %zmm1, %k0
  3544f8:      	vpcmpeqq	%zmm6, %zmm8, %k1
  3544fe:      	vpcmpeqq	%zmm6, %zmm7, %k2
  354504:      	korb	%k2, %k0, %k0
  354508:      	vpcmpeqq	%zmm6, %zmm0, %k2
  35450e:      	korb	%k2, %k1, %k1
  354512:      	vpcmpgtq	%zmm1, %zmm7, %k2
  354518:      	vpcmpgtq	%zmm8, %zmm0, %k3
  35451e:      	vmovdqa64	-0x2b2028(%rip), %zmm0 {%k2} {z} # 0xa2500 <anon.4e8d97506139486fadb1e849b133b59b.19.llvm.16316955986290925330+0x42d>
  354528:      	vmovdqa64	-0x2b1ff2(%rip), %zmm1 {%k3} {z} # 0xa2540 <anon.4e8d97506139486fadb1e849b133b59b.19.llvm.16316955986290925330+0x46d>
  354532:      	cmpq	$0x10, %rcx
  354536:      	je	0x354700 <_RINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool18execute_owned_boolTxxEubKb1_NCINvYINtNtNtB6_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB1N_EENtNtB21_11row_visitor10RowVisitor19visit_deferred_boolB1G_bKB1N_NCINvXB2C_B2z_NtNtB6_6row_fn5RowFn8dispatchB1W_E0NCB4u_s_0E0NCB1S_s_0B5f_EB2C_+0xb90>
  35453c:      	vmovdqa64	-0x2b1fc6(%rip), %zmm9 # 0xa2580 <anon.4e8d97506139486fadb1e849b133b59b.19.llvm.16316955986290925330+0x4ad>
  354546:      	vmovdqa64	-0x2b1f90(%rip), %zmm10 # 0xa25c0 <anon.4e8d97506139486fadb1e849b133b59b.19.llvm.16316955986290925330+0x4ed>
  354550:      	vmovdqa64	%zmm2, %zmm11
  354556:      	vmovdqa64	%zmm2, %zmm12
  35455c:      	testq	%rsi, %rsi
  35455f:      	je	0x35456d <_RINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool18execute_owned_boolTxxEubKb1_NCINvYINtNtNtB6_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB1N_EENtNtB21_11row_visitor10RowVisitor19visit_deferred_boolB1G_bKB1N_NCINvXB2C_B2z_NtNtB6_6row_fn5RowFn8dispatchB1W_E0NCB4u_s_0E0NCB1S_s_0B5f_EB2C_+0x9fd>
  354561:      	vpaddq	%zmm9, %zmm4, %zmm11
  354567:      	vpaddq	%zmm10, %zmm4, %zmm12
  35456d:      	kxnorb	%k0, %k0, %k2
  354571:      	vpxor	%xmm7, %xmm7, %xmm7
  354575:      	kxnorb	%k0, %k0, %k3
  354579:      	vpxor	%xmm8, %xmm8, %xmm8
  35457e:      	vpgatherqq	(,%zmm11), %zmm8 {%k3}
  354589:      	kxnorb	%k0, %k0, %k3
  35458d:      	vpxor	%xmm11, %xmm11, %xmm11
  354592:      	vpgatherqq	(,%zmm12), %zmm11 {%k3}
  35459d:      	vmovdqa64	%zmm3, %zmm13
  3545a3:      	vmovdqa64	%zmm3, %zmm12
  3545a9:      	testq	%r9, %r9
  3545ac:      	je	0x3545ba <_RINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool18execute_owned_boolTxxEubKb1_NCINvYINtNtNtB6_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB1N_EENtNtB21_11row_visitor10RowVisitor19visit_deferred_boolB1G_bKB1N_NCINvXB2C_B2z_NtNtB6_6row_fn5RowFn8dispatchB1W_E0NCB4u_s_0E0NCB1S_s_0B5f_EB2C_+0xa4a>
  3545ae:      	vpaddq	%zmm9, %zmm5, %zmm13
  3545b4:      	vpaddq	%zmm10, %zmm5, %zmm12
  3545ba:      	kxnorb	%k0, %k0, %k3
  3545be:      	vpxor	%xmm9, %xmm9, %xmm9
  3545c3:      	vpgatherqq	(,%zmm13), %zmm9 {%k3}
  3545ce:      	vpgatherqq	(,%zmm12), %zmm7 {%k2}
  3545d9:      	vpcmpeqq	%zmm6, %zmm8, %k2
  3545df:      	vpcmpeqq	%zmm6, %zmm11, %k3
  3545e5:      	vpcmpeqq	%zmm6, %zmm9, %k4
  3545eb:      	korb	%k4, %k2, %k2
  3545ef:      	vpcmpeqq	%zmm6, %zmm7, %k4
  3545f5:      	korb	%k4, %k3, %k3
  3545f9:      	vpcmpgtq	%zmm8, %zmm9, %k4
  3545ff:      	vpcmpgtq	%zmm11, %zmm7, %k5
  354605:      	korb	%k2, %k0, %k0
  354609:      	vporq	-0x2b2013(%rip), %zmm0, %zmm0 {%k4} # 0xa2600 <anon.4e8d97506139486fadb1e849b133b59b.19.llvm.16316955986290925330+0x52d>
  354613:      	korb	%k3, %k1, %k1
  354617:      	vporq	-0x2b1fe1(%rip), %zmm1, %zmm1 {%k5} # 0xa2640 <anon.4e8d97506139486fadb1e849b133b59b.19.llvm.16316955986290925330+0x56d>
  354621:      	cmpl	$0x20, %ecx
  354624:      	je	0x354700 <_RINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool18execute_owned_boolTxxEubKb1_NCINvYINtNtNtB6_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB1N_EENtNtB21_11row_visitor10RowVisitor19visit_deferred_boolB1G_bKB1N_NCINvXB2C_B2z_NtNtB6_6row_fn5RowFn8dispatchB1W_E0NCB4u_s_0E0NCB1S_s_0B5f_EB2C_+0xb90>
  35462a:      	vmovdqa64	-0x2b1fb4(%rip), %zmm8 # 0xa2680 <anon.4e8d97506139486fadb1e849b133b59b.19.llvm.16316955986290925330+0x5ad>
  354634:      	vmovdqa64	-0x2b1f7e(%rip), %zmm9 # 0xa26c0 <anon.4e8d97506139486fadb1e849b133b59b.19.llvm.16316955986290925330+0x5ed>
  35463e:      	vmovdqa64	%zmm2, %zmm10
  354644:      	testq	%rsi, %rsi
  354647:      	je	0x354655 <_RINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool18execute_owned_boolTxxEubKb1_NCINvYINtNtNtB6_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB1N_EENtNtB21_11row_visitor10RowVisitor19visit_deferred_boolB1G_bKB1N_NCINvXB2C_B2z_NtNtB6_6row_fn5RowFn8dispatchB1W_E0NCB4u_s_0E0NCB1S_s_0B5f_EB2C_+0xae5>
  354649:      	vpaddq	%zmm8, %zmm4, %zmm2
  35464f:      	vpaddq	%zmm9, %zmm4, %zmm10
  354655:      	kxnorb	%k0, %k0, %k2
  354659:      	vpxor	%xmm4, %xmm4, %xmm4
  35465d:      	vpxor	%xmm7, %xmm7, %xmm7
  354661:      	kxnorb	%k0, %k0, %k3
  354665:      	vpgatherqq	(,%zmm2), %zmm7 {%k3}
  354670:      	vpxor	%xmm2, %xmm2, %xmm2
  354674:      	kxnorb	%k0, %k0, %k3
  354678:      	vpgatherqq	(,%zmm10), %zmm2 {%k3}
  354683:      	vmovdqa64	%zmm3, %zmm10
  354689:      	testq	%r9, %r9
  35468c:      	je	0x35469a <_RINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool18execute_owned_boolTxxEubKb1_NCINvYINtNtNtB6_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB1N_EENtNtB21_11row_visitor10RowVisitor19visit_deferred_boolB1G_bKB1N_NCINvXB2C_B2z_NtNtB6_6row_fn5RowFn8dispatchB1W_E0NCB4u_s_0E0NCB1S_s_0B5f_EB2C_+0xb2a>
  35468e:      	vpaddq	%zmm8, %zmm5, %zmm3
  354694:      	vpaddq	%zmm9, %zmm5, %zmm10
  35469a:      	vpxor	%xmm5, %xmm5, %xmm5
  35469e:      	kxnorb	%k0, %k0, %k3
  3546a2:      	vpgatherqq	(,%zmm3), %zmm5 {%k3}
  3546ad:      	vpgatherqq	(,%zmm10), %zmm4 {%k2}
  3546b8:      	vpcmpeqq	%zmm6, %zmm7, %k2
  3546be:      	vpcmpeqq	%zmm6, %zmm2, %k3
  3546c4:      	vpcmpeqq	%zmm6, %zmm5, %k4
  3546ca:      	korb	%k4, %k2, %k2
  3546ce:      	vpcmpeqq	%zmm6, %zmm4, %k4
  3546d4:      	korb	%k4, %k3, %k3
  3546d8:      	vpcmpgtq	%zmm7, %zmm5, %k4
  3546de:      	vpcmpgtq	%zmm2, %zmm4, %k5
  3546e4:      	korb	%k2, %k0, %k0
  3546e8:      	korb	%k3, %k1, %k1
  3546ec:      	vporq	-0x2b1ff6(%rip), %zmm0, %zmm0 {%k4} # 0xa2700 <anon.4e8d97506139486fadb1e849b133b59b.19.llvm.16316955986290925330+0x62d>
  3546f6:      	vporq	-0x2b1fc0(%rip), %zmm1, %zmm1 {%k5} # 0xa2740 <anon.4e8d97506139486fadb1e849b133b59b.19.llvm.16316955986290925330+0x66d>
  354700:      	vporq	%zmm0, %zmm1, %zmm0
  354706:      	kortestb	%k0, %k1
  35470a:      	setne	%r12b
  35470e:      	vextracti64x4	$0x1, %zmm0, %ymm1
  354715:      	vporq	%zmm1, %zmm0, %zmm0
  35471b:      	vextracti128	$0x1, %ymm0, %xmm1
  354721:      	vpor	%xmm1, %xmm0, %xmm0
  354725:      	vpshufd	$0xee, %xmm0, %xmm1     # xmm1 = xmm0[2,3,2,3]
  35472a:      	vpor	%xmm1, %xmm0, %xmm0
  35472e:      	vmovq	%xmm0, %rdx
  354733:      	cmpq	%rcx, %r11
  354736:      	je	0x3548c6 <_RINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool18execute_owned_boolTxxEubKb1_NCINvYINtNtNtB6_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB1N_EENtNtB21_11row_visitor10RowVisitor19visit_deferred_boolB1G_bKB1N_NCINvXB2C_B2z_NtNtB6_6row_fn5RowFn8dispatchB1W_E0NCB4u_s_0E0NCB1S_s_0B5f_EB2C_+0xd56>
  35473c:      	testb	$0xc, %r11b
  354740:      	movq	-0x38(%rbp), %r14
  354744:      	movq	%rdi, %rax
  354747:      	je	0x354859 <_RINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool18execute_owned_boolTxxEubKb1_NCINvYINtNtNtB6_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB1N_EENtNtB21_11row_visitor10RowVisitor19visit_deferred_boolB1G_bKB1N_NCINvXB2C_B2z_NtNtB6_6row_fn5RowFn8dispatchB1W_E0NCB4u_s_0E0NCB1S_s_0B5f_EB2C_+0xce9>
  35474d:      	movq	%rcx, %rsi
  354750:      	movl	%r11d, %ecx
  354753:      	andl	$0x3c, %ecx
  354756:      	kmovd	%r12d, %k0
  35475b:      	kshiftlb	$0x7, %k0, %k0
  354761:      	kshiftrb	$0x7, %k0, %k0
  354767:      	vmovq	%rdx, %xmm0
  35476c:      	vpbroadcastq	%r8, %ymm1
  354772:      	vpbroadcastq	%rsi, %ymm2
  354778:      	vpor	-0x2b3460(%rip), %ymm2, %ymm2 # 0xa1320 <anon.e6fb86fb181e0aaaed06fff69d344575.10.llvm.18170190781415527360+0xa0>
  354780:      	vpbroadcastq	%rax, %ymm3
  354786:      	subq	%rcx, %rsi
  354789:      	vpbroadcastq	%r13, %ymm4
  35478f:      	vpbroadcastq	%rbx, %ymm5
  354795:      	vpbroadcastq	-0x2b814e(%rip), %ymm6 # 0x9c650 <anon.4e66a2221d34052adc0f362485be06ac.633.llvm.27319815200592487+0x50>
  35479e:      	vpbroadcastq	-0x2b7b0f(%rip), %ymm7 # 0x9cc98 <anon.4d90d9de9367fb21f144d13924f001fb.148.llvm.1315488742485634063>
  3547a7:      	jmp	0x35480a <_RINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool18execute_owned_boolTxxEubKb1_NCINvYINtNtNtB6_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB1N_EENtNtB21_11row_visitor10RowVisitor19visit_deferred_boolB1G_bKB1N_NCINvXB2C_B2z_NtNtB6_6row_fn5RowFn8dispatchB1W_E0NCB4u_s_0E0NCB1S_s_0B5f_EB2C_+0xc9a>
  3547a9:      	nopl	(%rax)
  3547b0:      	kxnorw	%k0, %k0, %k1
  3547b4:      	vpxor	%xmm9, %xmm9, %xmm9
  3547b9:      	vpgatherqq	(,%ymm8), %ymm9 {%k1}
  3547c4:      	kxnorw	%k0, %k0, %k1
  3547c8:      	vpxor	%xmm8, %xmm8, %xmm8
  3547cd:      	vpgatherqq	(,%ymm10), %ymm8 {%k1}
  3547d8:      	vpcmpeqq	%ymm6, %ymm9, %k1
  3547de:      	vpcmpeqq	%ymm6, %ymm8, %k2
  3547e4:      	korw	%k2, %k1, %k1
  3547e8:      	korw	%k1, %k0, %k0
  3547ec:      	vpcmpgtq	%ymm9, %ymm8, %ymm8
  3547f1:      	vpsrlq	$0x3f, %ymm8, %ymm8
  3547f7:      	vpsllvq	%ymm2, %ymm8, %ymm8
  3547fc:      	vpor	%ymm0, %ymm8, %ymm0
  354800:      	vpaddq	%ymm7, %ymm2, %ymm2
  354804:      	addq	$0x4, %rsi
  354808:      	je	0x35482d <_RINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool18execute_owned_boolTxxEubKb1_NCINvYINtNtNtB6_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB1N_EENtNtB21_11row_visitor10RowVisitor19visit_deferred_boolB1G_bKB1N_NCINvXB2C_B2z_NtNtB6_6row_fn5RowFn8dispatchB1W_E0NCB4u_s_0E0NCB1S_s_0B5f_EB2C_+0xcbd>
  35480a:      	vpsllq	$0x3, %ymm2, %ymm9
  35480f:      	vmovdqa	%ymm1, %ymm8
  354813:      	cmpq	$0x0, -0x38(%rbp)
  354818:      	je	0x35481e <_RINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool18execute_owned_boolTxxEubKb1_NCINvYINtNtNtB6_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB1N_EENtNtB21_11row_visitor10RowVisitor19visit_deferred_boolB1G_bKB1N_NCINvXB2C_B2z_NtNtB6_6row_fn5RowFn8dispatchB1W_E0NCB4u_s_0E0NCB1S_s_0B5f_EB2C_+0xcae>
  35481a:      	vpaddq	%ymm4, %ymm9, %ymm8
  35481e:      	vmovdqa	%ymm3, %ymm10
  354822:      	testq	%r9, %r9
  354825:      	je	0x3547b0 <_RINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool18execute_owned_boolTxxEubKb1_NCINvYINtNtNtB6_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB1N_EENtNtB21_11row_visitor10RowVisitor19visit_deferred_boolB1G_bKB1N_NCINvXB2C_B2z_NtNtB6_6row_fn5RowFn8dispatchB1W_E0NCB4u_s_0E0NCB1S_s_0B5f_EB2C_+0xc40>
  354827:      	vpaddq	%ymm5, %ymm9, %ymm10
  35482b:      	jmp	0x3547b0 <_RINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool18execute_owned_boolTxxEubKb1_NCINvYINtNtNtB6_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB1N_EENtNtB21_11row_visitor10RowVisitor19visit_deferred_boolB1G_bKB1N_NCINvXB2C_B2z_NtNtB6_6row_fn5RowFn8dispatchB1W_E0NCB4u_s_0E0NCB1S_s_0B5f_EB2C_+0xc40>
  35482d:      	kmovd	%k0, %edx
  354831:      	testb	$0xf, %dl
  354834:      	setne	%r12b
  354838:      	vextracti128	$0x1, %ymm0, %xmm1
  35483e:      	vpor	%xmm1, %xmm0, %xmm0
  354842:      	vpshufd	$0xee, %xmm0, %xmm1     # xmm1 = xmm0[2,3,2,3]
  354847:      	vpor	%xmm1, %xmm0, %xmm0
  35484b:      	vmovq	%xmm0, %rdx
  354850:      	cmpq	%rcx, %r11
  354853:      	movq	-0x38(%rbp), %r14
  354857:      	je	0x3548c6 <_RINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool18execute_owned_boolTxxEubKb1_NCINvYINtNtNtB6_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB1N_EENtNtB21_11row_visitor10RowVisitor19visit_deferred_boolB1G_bKB1N_NCINvXB2C_B2z_NtNtB6_6row_fn5RowFn8dispatchB1W_E0NCB4u_s_0E0NCB1S_s_0B5f_EB2C_+0xd56>
  354859:      	movabsq	$0x7fffffffffffffff, %rsi # imm = 0x7FFFFFFFFFFFFFFF
  354863:      	leal	(,%rcx,8), %edi
  35486a:      	addq	%rdi, %rbx
  35486d:      	addq	%rdi, %r13
  354870:      	testq	%r14, %r14
  354873:      	movq	%r13, %rdi
  354876:      	cmoveq	%r8, %rdi
  35487a:      	testq	%r9, %r9
  35487d:      	movq	(%rdi), %rdi
  354880:      	movq	%rbx, %r8
  354883:      	cmoveq	%rax, %r8
  354887:      	movq	(%r8), %r8
  35488a:      	cmpq	%rsi, %rdi
  35488d:      	sete	%r9b
  354891:      	cmpq	%rsi, %r8
  354894:      	sete	%r10b
  354898:      	orb	%r9b, %r10b
  35489b:      	xorl	%r9d, %r9d
  35489e:      	cmpq	%r8, %rdi
  3548a1:      	setl	%r9b
  3548a5:      	shlq	%cl, %r9
  3548a8:      	orq	%r9, %rdx
  3548ab:      	movq	-0x48(%rbp), %r9
  3548af:      	orb	%r10b, %r12b
  3548b2:      	movq	-0x50(%rbp), %r8
  3548b6:      	incq	%rcx
  3548b9:      	addq	$0x8, %rbx
  3548bd:      	addq	$0x8, %r13
  3548c1:      	cmpq	%rcx, %r11
  3548c4:      	jne	0x354870 <_RINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool18execute_owned_boolTxxEubKb1_NCINvYINtNtNtB6_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB1N_EENtNtB21_11row_visitor10RowVisitor19visit_deferred_boolB1G_bKB1N_NCINvXB2C_B2z_NtNtB6_6row_fn5RowFn8dispatchB1W_E0NCB4u_s_0E0NCB1S_s_0B5f_EB2C_+0xd00>
  3548c6:      	movq	%rdx, (%r15)
  3548c9:      	jmp	0x3549d0 <_RINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool18execute_owned_boolTxxEubKb1_NCINvYINtNtNtB6_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB1N_EENtNtB21_11row_visitor10RowVisitor19visit_deferred_boolB1G_bKB1N_NCINvXB2C_B2z_NtNtB6_6row_fn5RowFn8dispatchB1W_E0NCB4u_s_0E0NCB1S_s_0B5f_EB2C_+0xe60>
  3548ce:      	xorl	%ecx, %ecx
  3548d0:      	movl	$0x101, %esi            # imm = 0x101
  3548d5:      	xorl	%r12d, %r12d
  3548d8:      	movq	-0x30(%rbp), %rax
  3548dc:      	nopl	(%rax)
  3548e0:      	movq	(%r10), %rdi
  3548e3:      	movq	(%rax), %r8
  3548e6:      	cmpq	%rdx, %rdi
  3548e9:      	sete	%r9b
  3548ed:      	cmpq	%rdx, %r8
  3548f0:      	sete	%r10b
  3548f4:      	orb	%r9b, %r10b
  3548f7:      	orb	%r10b, %r12b
  3548fa:      	movq	-0x50(%rbp), %r10
  3548fe:      	movl	$0x0, %r9d
  354904:      	cmpq	%r8, %rdi
  354907:      	cmovlq	%rsi, %r9
  35490b:      	movl	%r9d, %edi
  35490e:      	shrl	$0x8, %edi
  354911:      	vmovd	%r9d, %xmm0
  354916:      	vpinsrb	$0x1, %edi, %xmm0, %xmm0
  35491c:      	movl	%r9d, %r8d
  35491f:      	shll	$0x10, %r8d
  354923:      	vpinsrb	$0x2, %r9d, %xmm0, %xmm0
  354929:      	shrl	$0x18, %r8d
  35492d:      	vpinsrb	$0x3, %r8d, %xmm0, %xmm0
  354933:      	vpinsrb	$0x4, %r9d, %xmm0, %xmm0
  354939:      	vpinsrb	$0x5, %edi, %xmm0, %xmm0
  35493f:      	vpinsrb	$0x6, %r9d, %xmm0, %xmm0
  354945:      	vpinsrb	$0x7, %r8d, %xmm0, %xmm0
  35494b:      	vpinsrb	$0x8, %r9d, %xmm0, %xmm0
  354951:      	vpinsrb	$0x9, %edi, %xmm0, %xmm0
  354957:      	vpinsrb	$0xa, %r9d, %xmm0, %xmm0
  35495d:      	vpinsrb	$0xb, %r8d, %xmm0, %xmm0
  354963:      	vpinsrb	$0xc, %r9d, %xmm0, %xmm0
  354969:      	vpinsrb	$0xd, %edi, %xmm0, %xmm0
  35496f:      	vpinsrb	$0xe, %r9d, %xmm0, %xmm0
  354975:      	vpinsrb	$0xf, %r8d, %xmm0, %xmm0
  35497b:      	vshufi64x2	$0x0, %zmm0, %zmm0, %zmm0 # zmm0 = zmm0[0,1,0,1,0,1,0,1]
  354982:      	vptestmb	%zmm0, %zmm0, %k0
  354988:      	kmovq	%k0, (%r15,%rcx)
  35498e:      	addq	$0x8, %rcx
  354992:      	cmpq	%rcx, -0x40(%rbp)
  354996:      	jne	0x3548e0 <_RINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool18execute_owned_boolTxxEubKb1_NCINvYINtNtNtB6_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB1N_EENtNtB21_11row_visitor10RowVisitor19visit_deferred_boolB1G_bKB1N_NCINvXB2C_B2z_NtNtB6_6row_fn5RowFn8dispatchB1W_E0NCB4u_s_0E0NCB1S_s_0B5f_EB2C_+0xd70>
  35499c:      	movq	-0x120(%rbp), %r8
  3549a3:      	testq	%r8, %r8
  3549a6:      	je	0x3549d0 <_RINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool18execute_owned_boolTxxEubKb1_NCINvYINtNtNtB6_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB1N_EENtNtB21_11row_visitor10RowVisitor19visit_deferred_boolB1G_bKB1N_NCINvXB2C_B2z_NtNtB6_6row_fn5RowFn8dispatchB1W_E0NCB4u_s_0E0NCB1S_s_0B5f_EB2C_+0xe60>
  3549a8:      	movq	%r11, %rsi
  3549ab:      	andq	$-0x40, %rsi
  3549af:      	cmpl	$0x4, %r8d
  3549b3:      	movq	%r11, -0x118(%rbp)
  3549ba:      	jae	0x3549dd <_RINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool18execute_owned_boolTxxEubKb1_NCINvYINtNtNtB6_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB1N_EENtNtB21_11row_visitor10RowVisitor19visit_deferred_boolB1G_bKB1N_NCINvXB2C_B2z_NtNtB6_6row_fn5RowFn8dispatchB1W_E0NCB4u_s_0E0NCB1S_s_0B5f_EB2C_+0xe6d>
  3549bc:      	xorl	%edi, %edi
  3549be:      	xorl	%ecx, %ecx
  3549c0:      	movq	-0x48(%rbp), %r9
  3549c4:      	movq	-0x30(%rbp), %rax
  3549c8:      	movq	%r8, %r11
  3549cb:      	jmp	0x354d10 <_RINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool18execute_owned_boolTxxEubKb1_NCINvYINtNtNtB6_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB1N_EENtNtB21_11row_visitor10RowVisitor19visit_deferred_boolB1G_bKB1N_NCINvXB2C_B2z_NtNtB6_6row_fn5RowFn8dispatchB1W_E0NCB4u_s_0E0NCB1S_s_0B5f_EB2C_+0x11a0>
  3549d0:      	movq	-0x60(%rbp), %r14
  3549d4:      	movq	-0x58(%rbp), %rax
  3549d8:      	jmp	0x354d9d <_RINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool18execute_owned_boolTxxEubKb1_NCINvYINtNtNtB6_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB1N_EENtNtB21_11row_visitor10RowVisitor19visit_deferred_boolB1G_bKB1N_NCINvXB2C_B2z_NtNtB6_6row_fn5RowFn8dispatchB1W_E0NCB4u_s_0E0NCB1S_s_0B5f_EB2C_+0x122d>
  3549dd:      	cmpl	$0x10, %r8d
  3549e1:      	movq	-0x48(%rbp), %r9
  3549e5:      	jae	0x3549f4 <_RINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool18execute_owned_boolTxxEubKb1_NCINvYINtNtNtB6_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB1N_EENtNtB21_11row_visitor10RowVisitor19visit_deferred_boolB1G_bKB1N_NCINvXB2C_B2z_NtNtB6_6row_fn5RowFn8dispatchB1W_E0NCB4u_s_0E0NCB1S_s_0B5f_EB2C_+0xe84>
  3549e7:      	xorl	%ecx, %ecx
  3549e9:      	xorl	%edi, %edi
  3549eb:      	movq	-0x30(%rbp), %rax
  3549ef:      	jmp	0x354bf4 <_RINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool18execute_owned_boolTxxEubKb1_NCINvYINtNtNtB6_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB1N_EENtNtB21_11row_visitor10RowVisitor19visit_deferred_boolB1G_bKB1N_NCINvXB2C_B2z_NtNtB6_6row_fn5RowFn8dispatchB1W_E0NCB4u_s_0E0NCB1S_s_0B5f_EB2C_+0x1084>
  3549f4:      	movl	%r11d, %ecx
  3549f7:      	andl	$0x30, %ecx
  3549fa:      	kmovd	%r12d, %k0
  3549ff:      	kshiftlb	$0x7, %k0, %k0
  354a05:      	kshiftrb	$0x7, %k0, %k0
  354a0b:      	vpbroadcastq	%rsi, %zmm0
  354a11:      	vpbroadcastq	%r10, %zmm1
  354a17:      	movq	-0x30(%rbp), %rax
  354a1b:      	vpbroadcastq	%rax, %zmm3
  354a21:      	vpaddq	-0x2b892b(%rip){1to8}, %zmm0, %zmm4 # 0x9c100 <anon.4d90d9de9367fb21f144d13924f001fb.147.llvm.1315488742485634063>
  354a2b:      	vmovdqa64	-0x2b25f5(%rip), %zmm5 # 0xa2440 <anon.4e8d97506139486fadb1e849b133b59b.19.llvm.16316955986290925330+0x36d>
  354a35:      	vpxor	%xmm2, %xmm2, %xmm2
  354a39:      	kxorb	%k0, %k0, %k1
  354a3d:      	vpbroadcastq	-0x2b8947(%rip), %zmm6 # 0x9c100 <anon.4d90d9de9367fb21f144d13924f001fb.147.llvm.1315488742485634063>
  354a47:      	vpbroadcastq	%r13, %zmm7
  354a4d:      	vpbroadcastq	%rbx, %zmm8
  354a53:      	vpbroadcastq	-0x2b840d(%rip), %zmm9 # 0x9c650 <anon.4e66a2221d34052adc0f362485be06ac.633.llvm.27319815200592487+0x50>
  354a5d:      	vpbroadcastq	-0x2b7d5f(%rip), %zmm10 # 0x9cd08 <anon.4e66a2221d34052adc0f362485be06ac.728.llvm.27319815200592487+0x48>
  354a67:      	movq	%rcx, %rdi
  354a6a:      	vpxor	%xmm11, %xmm11, %xmm11
  354a6f:      	jmp	0x354b4a <_RINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool18execute_owned_boolTxxEubKb1_NCINvYINtNtNtB6_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB1N_EENtNtB21_11row_visitor10RowVisitor19visit_deferred_boolB1G_bKB1N_NCINvXB2C_B2z_NtNtB6_6row_fn5RowFn8dispatchB1W_E0NCB4u_s_0E0NCB1S_s_0B5f_EB2C_+0xfda>
  354a74:      	nopw	%cs:(%rax,%rax)
  354a80:      	vpaddq	%zmm6, %zmm5, %zmm15
  354a86:      	kxnorb	%k0, %k0, %k2
  354a8a:      	vpxord	%xmm16, %xmm16, %xmm16
  354a90:      	vpgatherqq	(,%zmm13), %zmm16 {%k2}
  354a9b:      	kxnorb	%k0, %k0, %k2
  354a9f:      	vpxor	%xmm13, %xmm13, %xmm13
  354aa4:      	vpgatherqq	(,%zmm12), %zmm13 {%k2}
  354aaf:      	kxnorb	%k0, %k0, %k2
  354ab3:      	vpxor	%xmm12, %xmm12, %xmm12
  354ab8:      	vpgatherqq	(,%zmm17), %zmm12 {%k2}
  354ac3:      	kxnorb	%k0, %k0, %k2
  354ac7:      	vpxord	%xmm17, %xmm17, %xmm17
  354acd:      	vpgatherqq	(,%zmm14), %zmm17 {%k2}
  354ad8:      	vpcmpeqq	%zmm9, %zmm16, %k2
  354ade:      	vpcmpeqq	%zmm9, %zmm13, %k3
  354ae4:      	vpcmpeqq	%zmm9, %zmm12, %k4
  354aea:      	korb	%k4, %k2, %k2
  354aee:      	korb	%k2, %k0, %k0
  354af2:      	vpcmpeqq	%zmm9, %zmm17, %k2
  354af8:      	korb	%k2, %k3, %k2
  354afc:      	korb	%k2, %k1, %k1
  354b00:      	vpcmpgtq	%zmm16, %zmm12, %k2
  354b06:      	vpcmpgtq	%zmm13, %zmm17, %k3
  354b0c:      	vpmovm2q	%k2, %zmm12
  354b12:      	vpsrlq	$0x3f, %zmm12, %zmm12
  354b19:      	vpmovm2q	%k3, %zmm13
  354b1f:      	vpsrlq	$0x3f, %zmm13, %zmm13
  354b26:      	vpsllvq	%zmm15, %zmm13, %zmm13
  354b2c:      	vporq	%zmm11, %zmm13, %zmm11
  354b32:      	vpsllvq	%zmm5, %zmm12, %zmm12
  354b38:      	vporq	%zmm2, %zmm12, %zmm2
  354b3e:      	vpaddq	%zmm10, %zmm5, %zmm5
  354b44:      	addq	$-0x10, %rdi
  354b48:      	je	0x354ba7 <_RINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool18execute_owned_boolTxxEubKb1_NCINvYINtNtNtB6_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB1N_EENtNtB21_11row_visitor10RowVisitor19visit_deferred_boolB1G_bKB1N_NCINvXB2C_B2z_NtNtB6_6row_fn5RowFn8dispatchB1W_E0NCB4u_s_0E0NCB1S_s_0B5f_EB2C_+0x1037>
  354b4a:      	vpaddq	%zmm0, %zmm5, %zmm12
  354b50:      	vpaddq	%zmm4, %zmm5, %zmm13
  354b56:      	vpsllq	$0x3, %zmm12, %zmm15
  354b5d:      	vpsllq	$0x3, %zmm13, %zmm16
  354b64:      	vmovdqa64	%zmm1, %zmm13
  354b6a:      	vmovdqa64	%zmm1, %zmm12
  354b70:      	testq	%r14, %r14
  354b73:      	je	0x354b81 <_RINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool18execute_owned_boolTxxEubKb1_NCINvYINtNtNtB6_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB1N_EENtNtB21_11row_visitor10RowVisitor19visit_deferred_boolB1G_bKB1N_NCINvXB2C_B2z_NtNtB6_6row_fn5RowFn8dispatchB1W_E0NCB4u_s_0E0NCB1S_s_0B5f_EB2C_+0x1011>
  354b75:      	vpaddq	%zmm15, %zmm7, %zmm13
  354b7b:      	vpaddq	%zmm16, %zmm7, %zmm12
  354b81:      	vmovdqa64	%zmm3, %zmm17
  354b87:      	vmovdqa64	%zmm3, %zmm14
  354b8d:      	testq	%r9, %r9
  354b90:      	je	0x354a80 <_RINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool18execute_owned_boolTxxEubKb1_NCINvYINtNtNtB6_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB1N_EENtNtB21_11row_visitor10RowVisitor19visit_deferred_boolB1G_bKB1N_NCINvXB2C_B2z_NtNtB6_6row_fn5RowFn8dispatchB1W_E0NCB4u_s_0E0NCB1S_s_0B5f_EB2C_+0xf10>
  354b96:      	vpaddq	%zmm15, %zmm8, %zmm17
  354b9c:      	vpaddq	%zmm16, %zmm8, %zmm14
  354ba2:      	jmp	0x354a80 <_RINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool18execute_owned_boolTxxEubKb1_NCINvYINtNtNtB6_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB1N_EENtNtB21_11row_visitor10RowVisitor19visit_deferred_boolB1G_bKB1N_NCINvXB2C_B2z_NtNtB6_6row_fn5RowFn8dispatchB1W_E0NCB4u_s_0E0NCB1S_s_0B5f_EB2C_+0xf10>
  354ba7:      	kortestb	%k0, %k1
  354bab:      	setne	%r12b
  354baf:      	vporq	%zmm2, %zmm11, %zmm0
  354bb5:      	vextracti64x4	$0x1, %zmm0, %ymm1
  354bbc:      	vporq	%zmm1, %zmm0, %zmm0
  354bc2:      	vextracti128	$0x1, %ymm0, %xmm1
  354bc8:      	vpor	%xmm1, %xmm0, %xmm0
  354bcc:      	vpshufd	$0xee, %xmm0, %xmm1     # xmm1 = xmm0[2,3,2,3]
  354bd1:      	vpor	%xmm1, %xmm0, %xmm0
  354bd5:      	vmovq	%xmm0, %rdi
  354bda:      	cmpl	%ecx, %r8d
  354bdd:      	je	0x354d86 <_RINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool18execute_owned_boolTxxEubKb1_NCINvYINtNtNtB6_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB1N_EENtNtB21_11row_visitor10RowVisitor19visit_deferred_boolB1G_bKB1N_NCINvXB2C_B2z_NtNtB6_6row_fn5RowFn8dispatchB1W_E0NCB4u_s_0E0NCB1S_s_0B5f_EB2C_+0x1216>
  354be3:      	testb	$0xc, %r11b
  354be7:      	movq	-0x30(%rbp), %rax
  354beb:      	movq	%r8, %r11
  354bee:      	je	0x354d10 <_RINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool18execute_owned_boolTxxEubKb1_NCINvYINtNtNtB6_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB1N_EENtNtB21_11row_visitor10RowVisitor19visit_deferred_boolB1G_bKB1N_NCINvXB2C_B2z_NtNtB6_6row_fn5RowFn8dispatchB1W_E0NCB4u_s_0E0NCB1S_s_0B5f_EB2C_+0x11a0>
  354bf4:      	movq	%rcx, %r8
  354bf7:      	movq	-0x118(%rbp), %rcx
  354bfe:      	andl	$0x3c, %ecx
  354c01:      	kmovd	%r12d, %k0
  354c06:      	kshiftlb	$0x7, %k0, %k0
  354c0c:      	kshiftrb	$0x7, %k0, %k0
  354c12:      	vmovq	%rdi, %xmm0
  354c17:      	vpbroadcastq	%rsi, %ymm1
  354c1d:      	vpbroadcastq	%r10, %ymm2
  354c23:      	vpbroadcastq	%r8, %ymm3
  354c29:      	vpor	-0x2b3911(%rip), %ymm3, %ymm3 # 0xa1320 <anon.e6fb86fb181e0aaaed06fff69d344575.10.llvm.18170190781415527360+0xa0>
  354c31:      	vpbroadcastq	%rax, %ymm4
  354c37:      	subq	%rcx, %r8
  354c3a:      	vpbroadcastq	%r13, %ymm5
  354c40:      	vpbroadcastq	%rbx, %ymm6
  354c46:      	vpbroadcastq	-0x2b85ff(%rip), %ymm7 # 0x9c650 <anon.4e66a2221d34052adc0f362485be06ac.633.llvm.27319815200592487+0x50>
  354c4f:      	vpbroadcastq	-0x2b7fc0(%rip), %ymm8 # 0x9cc98 <anon.4d90d9de9367fb21f144d13924f001fb.148.llvm.1315488742485634063>
  354c58:      	jmp	0x354cba <_RINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool18execute_owned_boolTxxEubKb1_NCINvYINtNtNtB6_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB1N_EENtNtB21_11row_visitor10RowVisitor19visit_deferred_boolB1G_bKB1N_NCINvXB2C_B2z_NtNtB6_6row_fn5RowFn8dispatchB1W_E0NCB4u_s_0E0NCB1S_s_0B5f_EB2C_+0x114a>
  354c5a:      	nopw	(%rax,%rax)
  354c60:      	kxnorw	%k0, %k0, %k1
  354c64:      	vpxor	%xmm10, %xmm10, %xmm10
  354c69:      	vpgatherqq	(,%ymm9), %ymm10 {%k1}
  354c74:      	kxnorw	%k0, %k0, %k1
  354c78:      	vpxor	%xmm9, %xmm9, %xmm9
  354c7d:      	vpgatherqq	(,%ymm11), %ymm9 {%k1}
  354c88:      	vpcmpeqq	%ymm7, %ymm10, %k1
  354c8e:      	vpcmpeqq	%ymm7, %ymm9, %k2
  354c94:      	korw	%k2, %k1, %k1
  354c98:      	korw	%k1, %k0, %k0
  354c9c:      	vpcmpgtq	%ymm10, %ymm9, %ymm9
  354ca1:      	vpsrlq	$0x3f, %ymm9, %ymm9
  354ca7:      	vpsllvq	%ymm3, %ymm9, %ymm9
  354cac:      	vpor	%ymm0, %ymm9, %ymm0
  354cb0:      	vpaddq	%ymm3, %ymm8, %ymm3
  354cb4:      	addq	$0x4, %r8
  354cb8:      	je	0x354ce0 <_RINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool18execute_owned_boolTxxEubKb1_NCINvYINtNtNtB6_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB1N_EENtNtB21_11row_visitor10RowVisitor19visit_deferred_boolB1G_bKB1N_NCINvXB2C_B2z_NtNtB6_6row_fn5RowFn8dispatchB1W_E0NCB4u_s_0E0NCB1S_s_0B5f_EB2C_+0x1170>
  354cba:      	vpaddq	%ymm1, %ymm3, %ymm9
  354cbe:      	vpsllq	$0x3, %ymm9, %ymm10
  354cc4:      	vmovdqa	%ymm2, %ymm9
  354cc8:      	testq	%r14, %r14
  354ccb:      	je	0x354cd1 <_RINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool18execute_owned_boolTxxEubKb1_NCINvYINtNtNtB6_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB1N_EENtNtB21_11row_visitor10RowVisitor19visit_deferred_boolB1G_bKB1N_NCINvXB2C_B2z_NtNtB6_6row_fn5RowFn8dispatchB1W_E0NCB4u_s_0E0NCB1S_s_0B5f_EB2C_+0x1161>
  354ccd:      	vpaddq	%ymm5, %ymm10, %ymm9
  354cd1:      	vmovdqa	%ymm4, %ymm11
  354cd5:      	testq	%r9, %r9
  354cd8:      	je	0x354c60 <_RINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool18execute_owned_boolTxxEubKb1_NCINvYINtNtNtB6_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB1N_EENtNtB21_11row_visitor10RowVisitor19visit_deferred_boolB1G_bKB1N_NCINvXB2C_B2z_NtNtB6_6row_fn5RowFn8dispatchB1W_E0NCB4u_s_0E0NCB1S_s_0B5f_EB2C_+0x10f0>
  354cda:      	vpaddq	%ymm6, %ymm10, %ymm11
  354cde:      	jmp	0x354c60 <_RINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool18execute_owned_boolTxxEubKb1_NCINvYINtNtNtB6_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB1N_EENtNtB21_11row_visitor10RowVisitor19visit_deferred_boolB1G_bKB1N_NCINvXB2C_B2z_NtNtB6_6row_fn5RowFn8dispatchB1W_E0NCB4u_s_0E0NCB1S_s_0B5f_EB2C_+0x10f0>
  354ce0:      	kmovd	%k0, %edi
  354ce4:      	testb	$0xf, %dil
  354ce8:      	setne	%r12b
  354cec:      	vextracti128	$0x1, %ymm0, %xmm1
  354cf2:      	vpor	%xmm1, %xmm0, %xmm0
  354cf6:      	vpshufd	$0xee, %xmm0, %xmm1     # xmm1 = xmm0[2,3,2,3]
  354cfb:      	vpor	%xmm1, %xmm0, %xmm0
  354cff:      	vmovq	%xmm0, %rdi
  354d04:      	movq	-0x120(%rbp), %r11
  354d0b:      	cmpl	%ecx, %r11d
  354d0e:      	je	0x354d86 <_RINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool18execute_owned_boolTxxEubKb1_NCINvYINtNtNtB6_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB1N_EENtNtB21_11row_visitor10RowVisitor19visit_deferred_boolB1G_bKB1N_NCINvXB2C_B2z_NtNtB6_6row_fn5RowFn8dispatchB1W_E0NCB4u_s_0E0NCB1S_s_0B5f_EB2C_+0x1216>
  354d10:      	leaq	(,%rcx,8), %r8
  354d18:      	leaq	(%r8,%rsi,8), %rsi
  354d1c:      	addq	%rsi, %rbx
  354d1f:      	addq	%rsi, %r13
  354d22:      	nopw	%cs:(%rax,%rax)
  354d30:      	testq	%r14, %r14
  354d33:      	movq	%r13, %rsi
  354d36:      	cmoveq	%r10, %rsi
  354d3a:      	testq	%r9, %r9
  354d3d:      	movq	(%rsi), %rsi
  354d40:      	movq	%rbx, %r8
  354d43:      	cmoveq	%rax, %r8
  354d47:      	movq	(%r8), %r8
  354d4a:      	cmpq	%rdx, %rsi
  354d4d:      	sete	%r9b
  354d51:      	cmpq	%rdx, %r8
  354d54:      	sete	%r10b
  354d58:      	orb	%r9b, %r10b
  354d5b:      	xorl	%r9d, %r9d
  354d5e:      	cmpq	%r8, %rsi
  354d61:      	setl	%r9b
  354d65:      	shlq	%cl, %r9
  354d68:      	orq	%r9, %rdi
  354d6b:      	movq	-0x48(%rbp), %r9
  354d6f:      	orb	%r10b, %r12b
  354d72:      	movq	-0x50(%rbp), %r10
  354d76:      	incq	%rcx
  354d79:      	addq	$0x8, %rbx
  354d7d:      	addq	$0x8, %r13
  354d81:      	cmpq	%rcx, %r11
  354d84:      	jne	0x354d30 <_RINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool18execute_owned_boolTxxEubKb1_NCINvYINtNtNtB6_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB1N_EENtNtB21_11row_visitor10RowVisitor19visit_deferred_boolB1G_bKB1N_NCINvXB2C_B2z_NtNtB6_6row_fn5RowFn8dispatchB1W_E0NCB4u_s_0E0NCB1S_s_0B5f_EB2C_+0x11c0>
  354d86:      	movq	-0x40(%rbp), %rax
  354d8a:      	movq	%rdi, (%r15,%rax)
  354d8e:      	movq	-0x60(%rbp), %r14
  354d92:      	movq	-0x58(%rbp), %rax
  354d96:      	movq	-0x118(%rbp), %r11
  354d9d:      	vmovaps	-0xd0(%rbp), %xmm0
  354da5:      	vmovups	%xmm0, -0xa0(%rbp)
  354dad:      	movq	%r11, %rbx
  354db0:      	shrq	$0x3, %rbx
  354db4:      	movl	%r11d, %ecx
  354db7:      	andl	$0x7, %ecx
  354dba:      	cmpq	$0x1, %rcx
  354dbe:      	sbbq	$-0x1, %rbx
  354dc2:      	vmovaps	-0x170(%rbp), %xmm0
  354dca:      	cmpq	%r14, %rbx
  354dcd:      	cmovaeq	%r14, %rbx
  354dd1:      	vmovups	%xmm0, -0x88(%rbp)
  354dd9:      	movq	$0x1, -0xb0(%rbp)
  354de4:      	movq	$0x1, -0xa8(%rbp)
  354def:      	movq	%rax, -0x90(%rbp)
  354df6:      	movq	%r11, %r14
  354df9:      	vzeroupper
  354dfc:      	callq	*0xfe1806(%rip)         # 0x1336608 <writev+0x1336608>
  354e02:      	movl	$0x38, %edi
  354e07:      	movl	$0x8, %esi
  354e0c:      	callq	*0xfe17fe(%rip)         # 0x1336610 <writev+0x1336610>
  354e12:      	testq	%rax, %rax
  354e15:      	je	0x355024 <_RINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool18execute_owned_boolTxxEubKb1_NCINvYINtNtNtB6_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB1N_EENtNtB21_11row_visitor10RowVisitor19visit_deferred_boolB1G_bKB1N_NCINvXB2C_B2z_NtNtB6_6row_fn5RowFn8dispatchB1W_E0NCB4u_s_0E0NCB1S_s_0B5f_EB2C_+0x14b4>
  354e1b:      	vmovups	-0xb0(%rbp), %ymm0
  354e23:      	vmovdqu	-0x98(%rbp), %ymm1
  354e2b:      	vmovdqu	%ymm1, 0x18(%rax)
  354e30:      	vmovups	%ymm0, (%rax)
  354e34:      	movq	%r15, -0xd0(%rbp)
  354e3b:      	movq	%rbx, -0xc8(%rbp)
  354e42:      	movb	$0x3, -0xb8(%rbp)
  354e49:      	movq	%rax, -0xc0(%rbp)
  354e50:      	movq	%r14, -0x188(%rbp)
  354e57:      	movq	$0x0, -0x190(%rbp)
  354e62:      	leaq	(,%rbx,8), %rcx
  354e6a:      	movq	%rbx, %rdx
  354e6d:      	shrq	$0x3d, %rdx
  354e71:      	cmpq	%rcx, %r14
  354e74:      	seta	%cl
  354e77:      	testq	%rdx, %rdx
  354e7a:      	je	0x354e7e <_RINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool18execute_owned_boolTxxEubKb1_NCINvYINtNtNtB6_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB1N_EENtNtB21_11row_visitor10RowVisitor19visit_deferred_boolB1G_bKB1N_NCINvXB2C_B2z_NtNtB6_6row_fn5RowFn8dispatchB1W_E0NCB4u_s_0E0NCB1S_s_0B5f_EB2C_+0x130e>
  354e7c:      	xorl	%ecx, %ecx
  354e7e:      	testb	%cl, %cl
  354e80:      	jne	0x355036 <_RINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool18execute_owned_boolTxxEubKb1_NCINvYINtNtNtB6_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB1N_EENtNtB21_11row_visitor10RowVisitor19visit_deferred_boolB1G_bKB1N_NCINvXB2C_B2z_NtNtB6_6row_fn5RowFn8dispatchB1W_E0NCB4u_s_0E0NCB1S_s_0B5f_EB2C_+0x14c6>
  354e86:      	leaq	-0xc8(%rbp), %rax
  354e8d:      	vmovups	(%rax), %xmm0
  354e91:      	vmovups	%xmm0, -0x108(%rbp)
  354e99:      	movq	%r15, -0x110(%rbp)
  354ea0:      	movb	$0x0, -0xf8(%rbp)
  354ea7:      	movq	$0x0, -0xf0(%rbp)
  354eb2:      	movq	%r14, -0xe8(%rbp)
  354eb9:      	testb	$0x1, %r12b
  354ebd:      	movq	-0x178(%rbp), %r12
  354ec4:      	jne	0x3550ad <_RINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool18execute_owned_boolTxxEubKb1_NCINvYINtNtNtB6_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB1N_EENtNtB21_11row_visitor10RowVisitor19visit_deferred_boolB1G_bKB1N_NCINvXB2C_B2z_NtNtB6_6row_fn5RowFn8dispatchB1W_E0NCB4u_s_0E0NCB1S_s_0B5f_EB2C_+0x153d>
  354eca:      	movq	-0x110(%rbp), %rax
  354ed1:      	vmovups	-0x108(%rbp), %xmm0
  354ed9:      	movzbl	-0xf8(%rbp), %ecx
  354ee0:      	movl	-0xf7(%rbp), %edx
  354ee6:      	movzwl	-0xf3(%rbp), %esi
  354eed:      	movzbl	-0xf1(%rbp), %edi
  354ef4:      	movq	-0x100(%rbp), %r8
  354efb:      	movq	%r8, -0xa0(%rbp)
  354f02:      	movzbl	-0xf8(%rbp), %r8d
  354f0a:      	movb	%r8b, -0x98(%rbp)
  354f11:      	movl	-0xf7(%rbp), %r8d
  354f18:      	movl	%r8d, -0x97(%rbp)
  354f1f:      	movzwl	-0xf3(%rbp), %r8d
  354f27:      	movw	%r8w, -0x93(%rbp)
  354f2f:      	movzbl	-0xf1(%rbp), %r8d
  354f37:      	movb	%r8b, -0x91(%rbp)
  354f3e:      	movq	-0xf0(%rbp), %r8
  354f45:      	movq	%r8, -0x90(%rbp)
  354f4c:      	movq	-0xe8(%rbp), %r8
  354f53:      	movq	%r8, -0x88(%rbp)
  354f5a:      	movq	%rax, -0xb0(%rbp)
  354f61:      	vmovups	%xmm0, -0xa8(%rbp)
  354f69:      	movb	%cl, -0x98(%rbp)
  354f6f:      	movl	%edx, -0x97(%rbp)
  354f75:      	movw	%si, -0x93(%rbp)
  354f7c:      	movb	%dil, -0x91(%rbp)
  354f83:      	movq	$0x0, -0xd0(%rbp)
  354f8e:      	leaq	-0xb0(%rbp), %rdi
  354f95:      	leaq	-0xd0(%rbp), %rsi
  354f9c:      	vzeroupper
  354f9f:      	callq	*0xfe15c3(%rip)         # 0x1336568 <writev+0x1336568>
  354fa5:      	movq	%rax, 0x8(%r12)
  354faa:      	movq	%rdx, 0x10(%r12)
  354faf:      	movq	$-0x1, (%r12)
  354fb7:      	cmpq	$0x0, -0x160(%rbp)
  354fbf:      	je	0x354fe7 <_RINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool18execute_owned_boolTxxEubKb1_NCINvYINtNtNtB6_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB1N_EENtNtB21_11row_visitor10RowVisitor19visit_deferred_boolB1G_bKB1N_NCINvXB2C_B2z_NtNtB6_6row_fn5RowFn8dispatchB1W_E0NCB4u_s_0E0NCB1S_s_0B5f_EB2C_+0x1477>
  354fc1:      	movq	-0x150(%rbp), %rax
  354fc8:      	testq	%rax, %rax
  354fcb:      	je	0x354fe7 <_RINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool18execute_owned_boolTxxEubKb1_NCINvYINtNtNtB6_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB1N_EENtNtB21_11row_visitor10RowVisitor19visit_deferred_boolB1G_bKB1N_NCINvXB2C_B2z_NtNtB6_6row_fn5RowFn8dispatchB1W_E0NCB4u_s_0E0NCB1S_s_0B5f_EB2C_+0x1477>
  354fcd:      	lock
  354fce:      	decq	(%rax)
  354fd1:      	jne	0x354fe7 <_RINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool18execute_owned_boolTxxEubKb1_NCINvYINtNtNtB6_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB1N_EENtNtB21_11row_visitor10RowVisitor19visit_deferred_boolB1G_bKB1N_NCINvXB2C_B2z_NtNtB6_6row_fn5RowFn8dispatchB1W_E0NCB4u_s_0E0NCB1S_s_0B5f_EB2C_+0x1477>
  354fd3:      	leaq	-0x140(%rbp), %r14
  354fda:      	leaq	-0x150(%rbp), %rdi
  354fe1:      	callq	*0xfe15d1(%rip)         # 0x13365b8 <writev+0x13365b8>
  354fe7:      	cmpq	$0x0, -0x140(%rbp)
  354fef:      	je	0x353bdc <_RINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool18execute_owned_boolTxxEubKb1_NCINvYINtNtNtB6_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB1N_EENtNtB21_11row_visitor10RowVisitor19visit_deferred_boolB1G_bKB1N_NCINvXB2C_B2z_NtNtB6_6row_fn5RowFn8dispatchB1W_E0NCB4u_s_0E0NCB1S_s_0B5f_EB2C_+0x6c>
  354ff5:      	movq	-0x130(%rbp), %rax
  354ffc:      	testq	%rax, %rax
  354fff:      	je	0x353bdc <_RINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool18execute_owned_boolTxxEubKb1_NCINvYINtNtNtB6_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB1N_EENtNtB21_11row_visitor10RowVisitor19visit_deferred_boolB1G_bKB1N_NCINvXB2C_B2z_NtNtB6_6row_fn5RowFn8dispatchB1W_E0NCB4u_s_0E0NCB1S_s_0B5f_EB2C_+0x6c>
  355005:      	lock
  355006:      	decq	(%rax)
  355009:      	jne	0x353bdc <_RINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool18execute_owned_boolTxxEubKb1_NCINvYINtNtNtB6_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB1N_EENtNtB21_11row_visitor10RowVisitor19visit_deferred_boolB1G_bKB1N_NCINvXB2C_B2z_NtNtB6_6row_fn5RowFn8dispatchB1W_E0NCB4u_s_0E0NCB1S_s_0B5f_EB2C_+0x6c>
  35500f:      	leaq	-0x130(%rbp), %rdi
  355016:      	vzeroupper
  355019:      	callq	*0xfe1599(%rip)         # 0x13365b8 <writev+0x13365b8>
  35501f:      	jmp	0x353bdc <_RINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool18execute_owned_boolTxxEubKb1_NCINvYINtNtNtB6_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB1N_EENtNtB21_11row_visitor10RowVisitor19visit_deferred_boolB1G_bKB1N_NCINvXB2C_B2z_NtNtB6_6row_fn5RowFn8dispatchB1W_E0NCB4u_s_0E0NCB1S_s_0B5f_EB2C_+0x6c>
  355024:      	movl	$0x8, %edi
  355029:      	movl	$0x38, %esi
  35502e:      	callq	*0xfe160c(%rip)         # 0x1336640 <writev+0x1336640>
  355034:      	jmp	0x3550ab <_RINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool18execute_owned_boolTxxEubKb1_NCINvYINtNtNtB6_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB1N_EENtNtB21_11row_visitor10RowVisitor19visit_deferred_boolB1G_bKB1N_NCINvXB2C_B2z_NtNtB6_6row_fn5RowFn8dispatchB1W_E0NCB4u_s_0E0NCB1S_s_0B5f_EB2C_+0x153b>
  355036:      	movq	%rax, %r15
  355039:      	leaq	-0xc0(%rbp), %r14
  355040:      	movq	%rbx, -0x170(%rbp)
  355047:      	leaq	-0x170(%rbp), %rax
  35504e:      	movq	%rax, -0xb0(%rbp)
  355055:      	movq	0xfe14cc(%rip), %rax    # 0x1336528 <writev+0x1336528>
  35505c:      	movq	%rax, -0xa8(%rbp)
  355063:      	leaq	-0x190(%rbp), %rcx
  35506a:      	movq	%rcx, -0xa0(%rbp)
  355071:      	movq	%rax, -0x98(%rbp)
  355078:      	leaq	-0x188(%rbp), %rcx
  35507f:      	movq	%rcx, -0x90(%rbp)
  355086:      	movq	%rax, -0x88(%rbp)
  35508d:      	leaq	-0x2b9be3(%rip), %rdi   # 0x9b4b1 <anon.06f529d2af3b7eeb8ea026828d8ed908.2.llvm.4550720677393736157+0x116>
  355094:      	leaq	0xf8514d(%rip), %rdx    # 0x12da1e8 <anon.4e8d97506139486fadb1e849b133b59b.21.llvm.16316955986290925330+0x1f8>
  35509b:      	leaq	-0xb0(%rbp), %rsi
  3550a2:      	vzeroupper
  3550a5:      	callq	*0xfe15f5(%rip)         # 0x13366a0 <writev+0x13366a0>
  3550ab:      	ud2
  3550ad:      	leaq	-0x2b2911(%rip), %rax   # 0xa27a3 <anon.0025ab1289b848272e8054c1be5b9203.3.llvm.12724177022795420273+0x1>
  3550b4:      	movq	%rax, -0xd0(%rbp)
  3550bb:      	movq	$0x37, -0xc8(%rbp)
  3550c6:      	leaq	0x223(%rip), %rsi       # 0x3552f0 <_RNcNtNtCscHxMUXUtRzx_12vortex_error11VortexError5Other0>
  3550cd:      	leaq	-0xb0(%rbp), %rdi
  3550d4:      	leaq	-0xd0(%rbp), %rdx
  3550db:      	vzeroupper
  3550de:      	callq	*0xfe1464(%rip)         # 0x1336548 <writev+0x1336548>
  3550e4:      	cmpq	$-0x1, -0xb0(%rbp)
  3550ec:      	je	0x354eca <_RINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool18execute_owned_boolTxxEubKb1_NCINvYINtNtNtB6_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB1N_EENtNtB21_11row_visitor10RowVisitor19visit_deferred_boolB1G_bKB1N_NCINvXB2C_B2z_NtNtB6_6row_fn5RowFn8dispatchB1W_E0NCB4u_s_0E0NCB1S_s_0B5f_EB2C_+0x135a>
  3550f2:      	vmovups	-0xb0(%rbp), %zmm0
  3550fc:      	vmovdqu64	-0xa0(%rbp), %zmm1
  355106:      	vmovdqu64	%zmm1, 0x10(%r12)
  355111:      	vmovups	%zmm0, (%r12)
  355118:      	movq	-0x100(%rbp), %rax
  35511f:      	testq	%rax, %rax
  355122:      	je	0x353cd4 <_RINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool18execute_owned_boolTxxEubKb1_NCINvYINtNtNtB6_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB1N_EENtNtB21_11row_visitor10RowVisitor19visit_deferred_boolB1G_bKB1N_NCINvXB2C_B2z_NtNtB6_6row_fn5RowFn8dispatchB1W_E0NCB4u_s_0E0NCB1S_s_0B5f_EB2C_+0x164>
  355128:      	lock
  355129:      	decq	(%rax)
  35512c:      	jne	0x353cd4 <_RINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool18execute_owned_boolTxxEubKb1_NCINvYINtNtNtB6_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB1N_EENtNtB21_11row_visitor10RowVisitor19visit_deferred_boolB1G_bKB1N_NCINvXB2C_B2z_NtNtB6_6row_fn5RowFn8dispatchB1W_E0NCB4u_s_0E0NCB1S_s_0B5f_EB2C_+0x164>
  355132:      	leaq	-0x100(%rbp), %rdi
  355139:      	vzeroupper
  35513c:      	callq	*0xfe1476(%rip)         # 0x13365b8 <writev+0x13365b8>
  355142:      	jmp	0x353cd4 <_RINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool18execute_owned_boolTxxEubKb1_NCINvYINtNtNtB6_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB1N_EENtNtB21_11row_visitor10RowVisitor19visit_deferred_boolB1G_bKB1N_NCINvXB2C_B2z_NtNtB6_6row_fn5RowFn8dispatchB1W_E0NCB4u_s_0E0NCB1S_s_0B5f_EB2C_+0x164>
  355147:      	movq	%rax, %rbx
  35514a:      	movq	-0x100(%rbp), %rax
  355151:      	testq	%rax, %rax
  355154:      	je	0x3551c7 <_RINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool18execute_owned_boolTxxEubKb1_NCINvYINtNtNtB6_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB1N_EENtNtB21_11row_visitor10RowVisitor19visit_deferred_boolB1G_bKB1N_NCINvXB2C_B2z_NtNtB6_6row_fn5RowFn8dispatchB1W_E0NCB4u_s_0E0NCB1S_s_0B5f_EB2C_+0x1657>
  355156:      	lock
  355157:      	decq	(%rax)
  35515a:      	jne	0x3551c7 <_RINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool18execute_owned_boolTxxEubKb1_NCINvYINtNtNtB6_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB1N_EENtNtB21_11row_visitor10RowVisitor19visit_deferred_boolB1G_bKB1N_NCINvXB2C_B2z_NtNtB6_6row_fn5RowFn8dispatchB1W_E0NCB4u_s_0E0NCB1S_s_0B5f_EB2C_+0x1657>
  35515c:      	leaq	-0x100(%rbp), %rdi
  355163:      	callq	*0xfe144f(%rip)         # 0x13365b8 <writev+0x13365b8>
  355169:      	jmp	0x3551c7 <_RINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool18execute_owned_boolTxxEubKb1_NCINvYINtNtNtB6_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB1N_EENtNtB21_11row_visitor10RowVisitor19visit_deferred_boolB1G_bKB1N_NCINvXB2C_B2z_NtNtB6_6row_fn5RowFn8dispatchB1W_E0NCB4u_s_0E0NCB1S_s_0B5f_EB2C_+0x1657>
  35516b:      	movq	%rax, %rbx
  35516e:      	movq	%r14, %rdi
  355171:      	callq	0x352340 <_RINvNtCsc36rpYXAlPq_4core3ptr9drop_glueINtNtNtNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row5types7element5tuple13element_tuple9ArgColumnxEECslTNGN9sYpGe_17row_fn_bool_retry>
  355176:      	jmp	0x3551d3 <_RINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool18execute_owned_boolTxxEubKb1_NCINvYINtNtNtB6_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB1N_EENtNtB21_11row_visitor10RowVisitor19visit_deferred_boolB1G_bKB1N_NCINvXB2C_B2z_NtNtB6_6row_fn5RowFn8dispatchB1W_E0NCB4u_s_0E0NCB1S_s_0B5f_EB2C_+0x1663>
  355178:      	callq	*0xfe13ba(%rip)         # 0x1336538 <writev+0x1336538>
  35517e:      	movq	%rax, %rbx
  355181:      	leaq	-0x140(%rbp), %rdi
  355188:      	callq	0x352340 <_RINvNtCsc36rpYXAlPq_4core3ptr9drop_glueINtNtNtNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row5types7element5tuple13element_tuple9ArgColumnxEECslTNGN9sYpGe_17row_fn_bool_retry>
  35518d:      	jmp	0x3551d3 <_RINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool18execute_owned_boolTxxEubKb1_NCINvYINtNtNtB6_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB1N_EENtNtB21_11row_visitor10RowVisitor19visit_deferred_boolB1G_bKB1N_NCINvXB2C_B2z_NtNtB6_6row_fn5RowFn8dispatchB1W_E0NCB4u_s_0E0NCB1S_s_0B5f_EB2C_+0x1663>
  35518f:      	callq	*0xfe13a3(%rip)         # 0x1336538 <writev+0x1336538>
  355195:      	jmp	0x355199 <_RINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool18execute_owned_boolTxxEubKb1_NCINvYINtNtNtB6_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB1N_EENtNtB21_11row_visitor10RowVisitor19visit_deferred_boolB1G_bKB1N_NCINvXB2C_B2z_NtNtB6_6row_fn5RowFn8dispatchB1W_E0NCB4u_s_0E0NCB1S_s_0B5f_EB2C_+0x1629>
  355197:      	jmp	0x355199 <_RINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool18execute_owned_boolTxxEubKb1_NCINvYINtNtNtB6_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB1N_EENtNtB21_11row_visitor10RowVisitor19visit_deferred_boolB1G_bKB1N_NCINvXB2C_B2z_NtNtB6_6row_fn5RowFn8dispatchB1W_E0NCB4u_s_0E0NCB1S_s_0B5f_EB2C_+0x1629>
  355199:      	movq	%rax, %rbx
  35519c:      	jmp	0x3551c7 <_RINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool18execute_owned_boolTxxEubKb1_NCINvYINtNtNtB6_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB1N_EENtNtB21_11row_visitor10RowVisitor19visit_deferred_boolB1G_bKB1N_NCINvXB2C_B2z_NtNtB6_6row_fn5RowFn8dispatchB1W_E0NCB4u_s_0E0NCB1S_s_0B5f_EB2C_+0x1657>
  35519e:      	movq	%rax, %rbx
  3551a1:      	lock
  3551a2:      	decq	(%r15)
  3551a5:      	jne	0x3551c7 <_RINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool18execute_owned_boolTxxEubKb1_NCINvYINtNtNtB6_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB1N_EENtNtB21_11row_visitor10RowVisitor19visit_deferred_boolB1G_bKB1N_NCINvXB2C_B2z_NtNtB6_6row_fn5RowFn8dispatchB1W_E0NCB4u_s_0E0NCB1S_s_0B5f_EB2C_+0x1657>
  3551a7:      	movq	%r14, %rdi
  3551aa:      	callq	*0xfe1408(%rip)         # 0x13365b8 <writev+0x13365b8>
  3551b0:      	jmp	0x3551c7 <_RINvNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row7execute11packed_bool18execute_owned_boolTxxEubKb1_NCINvYINtNtNtB6_7visitor7execute11ExecuteRowsINtCslTNGN9sYpGe_17row_fn_bool_retry9PredicateKB1N_EENtNtB21_11row_visitor10RowVisitor19visit_deferred_boolB1G_bKB1N_NCINvXB2C_B2z_NtNtB6_6row_fn5RowFn8dispatchB1W_E0NCB4u_s_0E0NCB1S_s_0B5f_EB2C_+0x1657>
  3551b2:      	callq	*0xfe1380(%rip)         # 0x1336538 <writev+0x1336538>
  3551b8:      	movq	%rax, %rbx
  3551bb:      	leaq	-0xb0(%rbp), %rdi
  3551c2:      	callq	0x35b2d0 <_RINvNtCsc36rpYXAlPq_4core3ptr9drop_glueINtNtCscHgRw1M2fX5_5alloc4sync8ArcInnerNtNtCsdxhMBRXnOVY_13vortex_buffer10allocation13BufferBackingEECslTNGN9sYpGe_17row_fn_bool_retry.llvm.8584182929843863652>
  3551c7:      	leaq	-0x160(%rbp), %rdi
  3551ce:      	callq	0x352440 <_RINvNtCsc36rpYXAlPq_4core3ptr9drop_glueTINtNtNtNtNtNtNtNtCs4NpyjxtQ07h_12vortex_array9scalar_fn8unstable3row5types7element5tuple13element_tuple9ArgColumnxEBC_EECslTNGN9sYpGe_17row_fn_bool_retry>
  3551d3:      	movq	%rbx, %rdi
  3551d6:      	callq	0x12d7920 <_Unwind_Resume@plt>
  3551db:      	callq	*0xfe1357(%rip)         # 0x1336538 <writev+0x1336538>
  3551e1:      	callq	*0xfe1351(%rip)         # 0x1336538 <writev+0x1336538>
