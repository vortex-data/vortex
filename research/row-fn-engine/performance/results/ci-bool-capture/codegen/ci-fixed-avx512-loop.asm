
/tmp/row-fn-bool-ci-artifact-fixed/codspeed/walltime/vortex-array/row_fn_bool_retry:	file format elf64-x86-64

Disassembly of section .text:

0000000000353980 <vortex_buffer::bit::pack::collect_bool_words_avx512::<vortex_array::scalar_fn::unstable::row::execute::packed_bool::execute_bool_dense_attempt<(i64, i64), (), bool, true, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#1}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}>>:
  353980: 55                           	pushq	%rbp
  353981: 48 89 e5                     	movq	%rsp, %rbp
  353984: 41 57                        	pushq	%r15
  353986: 41 56                        	pushq	%r14
  353988: 41 55                        	pushq	%r13
  35398a: 41 54                        	pushq	%r12
  35398c: 53                           	pushq	%rbx
  35398d: 48 83 ec 18                  	subq	$0x18, %rsp
  353991: 49 89 f0                     	movq	%rsi, %r8
  353994: 48 89 d6                     	movq	%rdx, %rsi
  353997: 48 c1 ee 06                  	shrq	$0x6, %rsi
  35399b: 4c 39 c6                     	cmpq	%r8, %rsi
  35399e: 0f 87 37 10 00 00            	ja	0x3549db <vortex_buffer::bit::pack::collect_bool_words_avx512::<vortex_array::scalar_fn::unstable::row::execute::packed_bool::execute_bool_dense_attempt<(i64, i64), (), bool, true, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#1}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}>+0x105b>
  3539a4: 48 89 c8                     	movq	%rcx, %rax
  3539a7: 4c 89 45 c8                  	movq	%r8, -0x38(%rbp)
  3539ab: 41 89 d3                     	movl	%edx, %r11d
  3539ae: 41 83 e3 3f                  	andl	$0x3f, %r11d
  3539b2: 49 ba ff ff ff ff ff ff ff 7f	movabsq	$0x7fffffffffffffff, %r10 # imm = 0x7FFFFFFFFFFFFFFF
  3539bc: 4c 8d 2c f7                  	leaq	(%rdi,%rsi,8), %r13
  3539c0: 48 85 f6                     	testq	%rsi, %rsi
  3539c3: 4c 89 6d d0                  	movq	%r13, -0x30(%rbp)
  3539c7: 0f 84 e4 06 00 00            	je	0x3540b1 <vortex_buffer::bit::pack::collect_bool_words_avx512::<vortex_array::scalar_fn::unstable::row::execute::packed_bool::execute_bool_dense_attempt<(i64, i64), (), bool, true, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#1}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}>+0x731>
  3539cd: 48 8d 0c f5 00 00 00 00      	leaq	(,%rsi,8), %rcx
  3539d5: 44 0f b6 78 38               	movzbl	0x38(%rax), %r15d
  3539da: 44 0f b6 48 18               	movzbl	0x18(%rax), %r9d
  3539df: 48 8b 58 20                  	movq	0x20(%rax), %rbx
  3539e3: 4c 8b 40 28                  	movq	0x28(%rax), %r8
  3539e7: 83 38 01                     	cmpl	$0x1, (%rax)
  3539ea: 0f 85 30 01 00 00            	jne	0x353b20 <vortex_buffer::bit::pack::collect_bool_words_avx512::<vortex_array::scalar_fn::unstable::row::execute::packed_bool::execute_bool_dense_attempt<(i64, i64), (), bool, true, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#1}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}>+0x1a0>
  3539f0: 4c 8b 70 10                  	movq	0x10(%rax), %r14
  3539f4: 4d 8b 36                     	movq	(%r14), %r14
  3539f7: 45 84 c9                     	testb	%r9b, %r9b
  3539fa: 0f 84 db 02 00 00            	je	0x353cdb <vortex_buffer::bit::pack::collect_bool_words_avx512::<vortex_array::scalar_fn::unstable::row::execute::packed_bool::execute_bool_dense_attempt<(i64, i64), (), bool, true, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#1}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}>+0x35b>
  353a00: 4d 8b 00                     	movq	(%r8), %r8
  353a03: 45 31 c9                     	xorl	%r9d, %r9d
  353a06: 4d 39 c6                     	cmpq	%r8, %r14
  353a09: bb 01 01 00 00               	movl	$0x101, %ebx            # imm = 0x101
  353a0e: 49 0f 4d d9                  	cmovgeq	%r9, %rbx
  353a12: 41 89 d9                     	movl	%ebx, %r9d
  353a15: 41 c1 e1 10                  	shll	$0x10, %r9d
  353a19: 41 c1 e9 18                  	shrl	$0x18, %r9d
  353a1d: 41 89 dc                     	movl	%ebx, %r12d
  353a20: 41 c1 ec 08                  	shrl	$0x8, %r12d
  353a24: c5 f9 6e c3                  	vmovd	%ebx, %xmm0
  353a28: c4 c3 79 20 c4 01            	vpinsrb	$0x1, %r12d, %xmm0, %xmm0
  353a2e: c4 e3 79 20 c3 02            	vpinsrb	$0x2, %ebx, %xmm0, %xmm0
  353a34: c4 c3 79 20 c1 03            	vpinsrb	$0x3, %r9d, %xmm0, %xmm0
  353a3a: c4 e3 79 20 c3 04            	vpinsrb	$0x4, %ebx, %xmm0, %xmm0
  353a40: c4 c3 79 20 c4 05            	vpinsrb	$0x5, %r12d, %xmm0, %xmm0
  353a46: c4 e3 79 20 c3 06            	vpinsrb	$0x6, %ebx, %xmm0, %xmm0
  353a4c: c4 c3 79 20 c1 07            	vpinsrb	$0x7, %r9d, %xmm0, %xmm0
  353a52: c4 e3 79 20 c3 08            	vpinsrb	$0x8, %ebx, %xmm0, %xmm0
  353a58: c4 c3 79 20 c4 09            	vpinsrb	$0x9, %r12d, %xmm0, %xmm0
  353a5e: c4 e3 79 20 c3 0a            	vpinsrb	$0xa, %ebx, %xmm0, %xmm0
  353a64: c4 c3 79 20 c1 0b            	vpinsrb	$0xb, %r9d, %xmm0, %xmm0
  353a6a: c4 e3 79 20 c3 0c            	vpinsrb	$0xc, %ebx, %xmm0, %xmm0
  353a70: c4 c3 79 20 c4 0d            	vpinsrb	$0xd, %r12d, %xmm0, %xmm0
  353a76: c4 e3 79 20 c3 0e            	vpinsrb	$0xe, %ebx, %xmm0, %xmm0
  353a7c: c4 c3 79 20 c1 0f            	vpinsrb	$0xf, %r9d, %xmm0, %xmm0
  353a82: 62 f3 fd 48 43 c0 00         	vshufi64x2	$0x0, %zmm0, %zmm0, %zmm0 # zmm0 = zmm0[0,1,0,1,0,1,0,1]
  353a89: 48 83 c1 f8                  	addq	$-0x8, %rcx
  353a8d: 41 89 c9                     	movl	%ecx, %r9d
  353a90: 41 f7 d1                     	notl	%r9d
  353a93: 62 f2 7d 48 26 c0            	vptestmb	%zmm0, %zmm0, %k0
  353a99: 41 f6 c1 38                  	testb	$0x38, %r9b
  353a9d: 74 1f                        	je	0x353abe <vortex_buffer::bit::pack::collect_bool_words_avx512::<vortex_array::scalar_fn::unstable::row::execute::packed_bool::execute_bool_dense_attempt<(i64, i64), (), bool, true, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#1}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}>+0x13e>
  353a9f: 41 89 c9                     	movl	%ecx, %r9d
  353aa2: 41 c1 e9 03                  	shrl	$0x3, %r9d
  353aa6: 41 ff c1                     	incl	%r9d
  353aa9: 41 83 e1 07                  	andl	$0x7, %r9d
  353aad: 0f 1f 00                     	nopl	(%rax)
  353ab0: c4 e1 f8 91 07               	kmovq	%k0, (%rdi)
  353ab5: 48 83 c7 08                  	addq	$0x8, %rdi
  353ab9: 49 ff c9                     	decq	%r9
  353abc: 75 f2                        	jne	0x353ab0 <vortex_buffer::bit::pack::collect_bool_words_avx512::<vortex_array::scalar_fn::unstable::row::execute::packed_bool::execute_bool_dense_attempt<(i64, i64), (), bool, true, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#1}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}>+0x130>
  353abe: 48 83 f9 38                  	cmpq	$0x38, %rcx
  353ac2: 72 44                        	jb	0x353b08 <vortex_buffer::bit::pack::collect_bool_words_avx512::<vortex_array::scalar_fn::unstable::row::execute::packed_bool::execute_bool_dense_attempt<(i64, i64), (), bool, true, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#1}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}>+0x188>
  353ac4: 66 66 66 2e 0f 1f 84 00 00 00 00 00  	nopw	%cs:(%rax,%rax)
  353ad0: c4 e1 f8 91 07               	kmovq	%k0, (%rdi)
  353ad5: c4 e1 f8 91 47 08            	kmovq	%k0, 0x8(%rdi)
  353adb: c4 e1 f8 91 47 10            	kmovq	%k0, 0x10(%rdi)
  353ae1: c4 e1 f8 91 47 18            	kmovq	%k0, 0x18(%rdi)
  353ae7: c4 e1 f8 91 47 20            	kmovq	%k0, 0x20(%rdi)
  353aed: c4 e1 f8 91 47 28            	kmovq	%k0, 0x28(%rdi)
  353af3: c4 e1 f8 91 47 30            	kmovq	%k0, 0x30(%rdi)
  353af9: c4 e1 f8 91 47 38            	kmovq	%k0, 0x38(%rdi)
  353aff: 48 83 c7 40                  	addq	$0x40, %rdi
  353b03: 4c 39 ef                     	cmpq	%r13, %rdi
  353b06: 75 c8                        	jne	0x353ad0 <vortex_buffer::bit::pack::collect_bool_words_avx512::<vortex_array::scalar_fn::unstable::row::execute::packed_bool::execute_bool_dense_attempt<(i64, i64), (), bool, true, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#1}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}>+0x150>
  353b08: 4d 39 d6                     	cmpq	%r10, %r14
  353b0b: 0f 94 c1                     	sete	%cl
  353b0e: 4d 39 d0                     	cmpq	%r10, %r8
  353b11: 41 0f 94 c1                  	sete	%r9b
  353b15: 41 08 c9                     	orb	%cl, %r9b
  353b18: 45 08 f9                     	orb	%r15b, %r9b
  353b1b: e9 89 05 00 00               	jmp	0x3540a9 <vortex_buffer::bit::pack::collect_bool_words_avx512::<vortex_array::scalar_fn::unstable::row::execute::packed_bool::execute_bool_dense_attempt<(i64, i64), (), bool, true, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#1}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}>+0x729>
  353b20: 4c 8b 70 08                  	movq	0x8(%rax), %r14
  353b24: 45 84 c9                     	testb	%r9b, %r9b
  353b27: 0f 84 5e 03 00 00            	je	0x353e8b <vortex_buffer::bit::pack::collect_bool_words_avx512::<vortex_array::scalar_fn::unstable::row::execute::packed_bool::execute_bool_dense_attempt<(i64, i64), (), bool, true, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#1}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}>+0x50b>
  353b2d: 4d 8b 08                     	movq	(%r8), %r9
  353b30: 45 31 c0                     	xorl	%r8d, %r8d
  353b33: 4d 39 d1                     	cmpq	%r10, %r9
  353b36: 62 d2 fd 48 7c c1            	vpbroadcastq	%r9, %zmm0
  353b3c: 41 b9 ff 00 00 00            	movl	$0xff, %r9d
  353b42: 45 0f 45 c8                  	cmovnel	%r8d, %r9d
  353b46: c4 c1 7b 92 c1               	kmovd	%r9d, %k0
  353b4b: 49 81 c6 c0 01 00 00         	addq	$0x1c0, %r14            # imm = 0x1C0
  353b52: 62 f2 fd 48 59 0d c4 85 d4 ff	vpbroadcastq	-0x2b7a3c(%rip), %zmm1 # 0x9c120 <anon.4e66a2221d34052adc0f362485be06ac.633.llvm.27319815200592487+0x50>
  353b5c: c5 f9 6f 15 2c 1d d3 ff      	vmovdqa	-0x2ce2d4(%rip), %xmm2  # 0x85890 <anon.37e9284aed57eb9304064780576275d5.103.llvm.15049800457202559735+0x90>
  353b64: 62 f1 fd 48 6f 1d 52 e3 d4 ff	vmovdqa64	-0x2b1cae(%rip), %zmm3 # 0xa1ec0 <anon.96e61e178d3ba3a8159b8fb9481c24f2.0.llvm.3084270292863525209+0x140>
  353b6e: 62 f1 fd 48 6f 25 88 e3 d4 ff	vmovdqa64	-0x2b1c78(%rip), %zmm4 # 0xa1f00 <anon.96e61e178d3ba3a8159b8fb9481c24f2.0.llvm.3084270292863525209+0x180>
  353b78: 45 89 f9                     	movl	%r15d, %r9d
  353b7b: 0f 1f 44 00 00               	nopl	(%rax,%rax)
  353b80: c4 c1 7b 92 c9               	kmovd	%r9d, %k1
  353b85: c4 e3 79 32 c9 07            	kshiftlb	$0x7, %k1, %k1
  353b8b: c4 e3 79 30 c9 07            	kshiftrb	$0x7, %k1, %k1
  353b91: 62 d1 fe 48 6f 6e f9         	vmovdqu64	-0x1c0(%r14), %zmm5
  353b98: 62 d1 fe 48 6f 76 fa         	vmovdqu64	-0x180(%r14), %zmm6
  353b9f: 62 d1 fe 48 6f 7e fb         	vmovdqu64	-0x140(%r14), %zmm7
  353ba6: 62 51 fe 48 6f 46 fc         	vmovdqu64	-0x100(%r14), %zmm8
  353bad: 62 f2 d5 48 29 d1            	vpcmpeqq	%zmm1, %zmm5, %k2
  353bb3: c5 ed 45 c9                  	korb	%k1, %k2, %k1
  353bb7: 62 f2 cd 48 29 e1            	vpcmpeqq	%zmm1, %zmm6, %k4
  353bbd: 62 f2 c5 48 29 d9            	vpcmpeqq	%zmm1, %zmm7, %k3
  353bc3: 62 f2 bd 48 29 d1            	vpcmpeqq	%zmm1, %zmm8, %k2
  353bc9: 62 51 fe 48 6f 4e fd         	vmovdqu64	-0xc0(%r14), %zmm9
  353bd0: 62 51 fe 48 6f 56 fe         	vmovdqu64	-0x80(%r14), %zmm10
  353bd7: 62 51 fe 48 6f 5e ff         	vmovdqu64	-0x40(%r14), %zmm11
  353bde: 62 51 fe 48 6f 26            	vmovdqu64	(%r14), %zmm12
  353be4: 62 f2 ad 48 29 e9            	vpcmpeqq	%zmm1, %zmm10, %k5
  353bea: c5 dd 45 e5                  	korb	%k5, %k4, %k4
  353bee: 62 f2 a5 48 29 e9            	vpcmpeqq	%zmm1, %zmm11, %k5
  353bf4: c5 e5 45 dd                  	korb	%k5, %k3, %k3
  353bf8: 62 f2 9d 48 29 e9            	vpcmpeqq	%zmm1, %zmm12, %k5
  353bfe: c5 ed 45 d5                  	korb	%k5, %k2, %k2
  353c02: 62 f2 b5 48 29 e9            	vpcmpeqq	%zmm1, %zmm9, %k5
  353c08: c5 d5 45 e8                  	korb	%k0, %k5, %k5
  353c0c: c5 f5 45 cd                  	korb	%k5, %k1, %k1
  353c10: c5 dd 45 c9                  	korb	%k1, %k4, %k1
  353c14: c5 e5 45 c9                  	korb	%k1, %k3, %k1
  353c18: c5 f9 98 d1                  	kortestb	%k1, %k2
  353c1c: 62 f2 fd 48 37 cd            	vpcmpgtq	%zmm5, %zmm0, %k1
  353c22: 62 f1 7f 89 6f ea            	vmovdqu8	%xmm2, %xmm5 {%k1} {z}
  353c28: 62 f2 fd 48 37 ce            	vpcmpgtq	%zmm6, %zmm0, %k1
  353c2e: 62 f1 7f 89 6f f2            	vmovdqu8	%xmm2, %xmm6 {%k1} {z}
  353c34: 62 f2 fd 48 37 cf            	vpcmpgtq	%zmm7, %zmm0, %k1
  353c3a: 62 f1 7f 89 6f fa            	vmovdqu8	%xmm2, %xmm7 {%k1} {z}
  353c40: 62 d2 fd 48 37 c8            	vpcmpgtq	%zmm8, %zmm0, %k1
  353c46: c5 d1 6c ee                  	vpunpcklqdq	%xmm6, %xmm5, %xmm5 # xmm5 = xmm5[0],xmm6[0]
  353c4a: c4 e3 55 38 ef 01            	vinserti128	$0x1, %xmm7, %ymm5, %ymm5
  353c50: 62 f1 7f 89 6f f2            	vmovdqu8	%xmm2, %xmm6 {%k1} {z}
  353c56: c4 e2 7d 59 f6               	vpbroadcastq	%xmm6, %ymm6
  353c5b: c4 e3 55 02 ee c0            	vpblendd	$0xc0, %ymm6, %ymm5, %ymm5 # ymm5 = ymm5[0,1,2,3,4,5],ymm6[6,7]
  353c61: 62 d2 fd 48 37 c9            	vpcmpgtq	%zmm9, %zmm0, %k1
  353c67: 62 f1 7f 89 6f f2            	vmovdqu8	%xmm2, %xmm6 {%k1} {z}
  353c6d: 62 d2 fd 48 37 ca            	vpcmpgtq	%zmm10, %zmm0, %k1
  353c73: 62 f1 7f 89 6f fa            	vmovdqu8	%xmm2, %xmm7 {%k1} {z}
  353c79: 62 d2 fd 48 37 cb            	vpcmpgtq	%zmm11, %zmm0, %k1
  353c7f: 62 71 7f 89 6f c2            	vmovdqu8	%xmm2, %xmm8 {%k1} {z}
  353c85: 62 d2 fd 48 37 cc            	vpcmpgtq	%zmm12, %zmm0, %k1
  353c8b: 62 71 7f 89 6f ca            	vmovdqu8	%xmm2, %xmm9 {%k1} {z}
  353c91: 62 f3 7d 48 38 f6 02         	vinserti32x4	$0x2, %xmm6, %zmm0, %zmm6
  353c98: 62 f3 cd 48 3a ed 00         	vinserti64x4	$0x0, %ymm5, %zmm6, %zmm5
  353c9f: 62 f2 e5 48 7e ef            	vpermt2q	%zmm7, %zmm3, %zmm5
  353ca5: 62 d3 55 48 38 e8 03         	vinserti32x4	$0x3, %xmm8, %zmm5, %zmm5
  353cac: 62 d2 dd 48 7e e9            	vpermt2q	%zmm9, %zmm4, %zmm5
  353cb2: 62 f2 55 48 26 cd            	vptestmb	%zmm5, %zmm5, %k1
  353cb8: c4 a1 f8 91 0c 07            	kmovq	%k1, (%rdi,%r8)
  353cbe: 41 0f 95 c1                  	setne	%r9b
  353cc2: 49 83 c0 08                  	addq	$0x8, %r8
  353cc6: 49 81 c6 00 02 00 00         	addq	$0x200, %r14            # imm = 0x200
  353ccd: 4c 39 c1                     	cmpq	%r8, %rcx
  353cd0: 0f 85 aa fe ff ff            	jne	0x353b80 <vortex_buffer::bit::pack::collect_bool_words_avx512::<vortex_array::scalar_fn::unstable::row::execute::packed_bool::execute_bool_dense_attempt<(i64, i64), (), bool, true, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#1}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}>+0x200>
  353cd6: e9 ce 03 00 00               	jmp	0x3540a9 <vortex_buffer::bit::pack::collect_bool_words_avx512::<vortex_array::scalar_fn::unstable::row::execute::packed_bool::execute_bool_dense_attempt<(i64, i64), (), bool, true, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#1}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}>+0x729>
  353cdb: 45 31 c0                     	xorl	%r8d, %r8d
  353cde: 4d 39 d6                     	cmpq	%r10, %r14
  353ce1: 41 b9 ff 00 00 00            	movl	$0xff, %r9d
  353ce7: 45 0f 45 c8                  	cmovnel	%r8d, %r9d
  353ceb: c4 c1 7b 92 c1               	kmovd	%r9d, %k0
  353cf0: 62 d2 fd 48 7c c6            	vpbroadcastq	%r14, %zmm0
  353cf6: 48 81 c3 c0 01 00 00         	addq	$0x1c0, %rbx            # imm = 0x1C0
  353cfd: 62 f2 fd 48 59 0d 19 84 d4 ff	vpbroadcastq	-0x2b7be7(%rip), %zmm1 # 0x9c120 <anon.4e66a2221d34052adc0f362485be06ac.633.llvm.27319815200592487+0x50>
  353d07: c5 f9 6f 15 81 1b d3 ff      	vmovdqa	-0x2ce47f(%rip), %xmm2  # 0x85890 <anon.37e9284aed57eb9304064780576275d5.103.llvm.15049800457202559735+0x90>
  353d0f: 62 f1 fd 48 6f 1d a7 e1 d4 ff	vmovdqa64	-0x2b1e59(%rip), %zmm3 # 0xa1ec0 <anon.96e61e178d3ba3a8159b8fb9481c24f2.0.llvm.3084270292863525209+0x140>
  353d19: 62 f1 fd 48 6f 25 dd e1 d4 ff	vmovdqa64	-0x2b1e23(%rip), %zmm4 # 0xa1f00 <anon.96e61e178d3ba3a8159b8fb9481c24f2.0.llvm.3084270292863525209+0x180>
  353d23: 45 89 f9                     	movl	%r15d, %r9d
  353d26: 66 2e 0f 1f 84 00 00 00 00 00	nopw	%cs:(%rax,%rax)
  353d30: c4 c1 7b 92 c9               	kmovd	%r9d, %k1
  353d35: c4 e3 79 32 c9 07            	kshiftlb	$0x7, %k1, %k1
  353d3b: c4 e3 79 30 c9 07            	kshiftrb	$0x7, %k1, %k1
  353d41: 62 f1 fe 48 6f 6b f9         	vmovdqu64	-0x1c0(%rbx), %zmm5
  353d48: 62 f1 fe 48 6f 73 fa         	vmovdqu64	-0x180(%rbx), %zmm6
  353d4f: 62 f1 fe 48 6f 7b fb         	vmovdqu64	-0x140(%rbx), %zmm7
  353d56: 62 71 fe 48 6f 43 fc         	vmovdqu64	-0x100(%rbx), %zmm8
  353d5d: 62 f2 d5 48 29 d1            	vpcmpeqq	%zmm1, %zmm5, %k2
  353d63: c5 ed 45 c9                  	korb	%k1, %k2, %k1
  353d67: 62 f2 cd 48 29 e1            	vpcmpeqq	%zmm1, %zmm6, %k4
  353d6d: 62 f2 c5 48 29 d9            	vpcmpeqq	%zmm1, %zmm7, %k3
  353d73: 62 f2 bd 48 29 d1            	vpcmpeqq	%zmm1, %zmm8, %k2
  353d79: 62 71 fe 48 6f 4b fd         	vmovdqu64	-0xc0(%rbx), %zmm9
  353d80: 62 71 fe 48 6f 53 fe         	vmovdqu64	-0x80(%rbx), %zmm10
  353d87: 62 71 fe 48 6f 5b ff         	vmovdqu64	-0x40(%rbx), %zmm11
  353d8e: 62 71 fe 48 6f 23            	vmovdqu64	(%rbx), %zmm12
  353d94: 62 f2 ad 48 29 e9            	vpcmpeqq	%zmm1, %zmm10, %k5
  353d9a: c5 dd 45 e5                  	korb	%k5, %k4, %k4
  353d9e: 62 f2 a5 48 29 e9            	vpcmpeqq	%zmm1, %zmm11, %k5
  353da4: c5 e5 45 dd                  	korb	%k5, %k3, %k3
  353da8: 62 f2 9d 48 29 e9            	vpcmpeqq	%zmm1, %zmm12, %k5
  353dae: c5 ed 45 d5                  	korb	%k5, %k2, %k2
  353db2: 62 f2 b5 48 29 e9            	vpcmpeqq	%zmm1, %zmm9, %k5
  353db8: c5 d5 45 e8                  	korb	%k0, %k5, %k5
  353dbc: c5 f5 45 cd                  	korb	%k5, %k1, %k1
  353dc0: c5 dd 45 c9                  	korb	%k1, %k4, %k1
  353dc4: c5 e5 45 c9                  	korb	%k1, %k3, %k1
  353dc8: c5 f9 98 d1                  	kortestb	%k1, %k2
  353dcc: 62 f2 d5 48 37 c8            	vpcmpgtq	%zmm0, %zmm5, %k1
  353dd2: 62 f1 7f 89 6f ea            	vmovdqu8	%xmm2, %xmm5 {%k1} {z}
  353dd8: 62 f2 cd 48 37 c8            	vpcmpgtq	%zmm0, %zmm6, %k1
  353dde: 62 f1 7f 89 6f f2            	vmovdqu8	%xmm2, %xmm6 {%k1} {z}
  353de4: 62 f2 c5 48 37 c8            	vpcmpgtq	%zmm0, %zmm7, %k1
  353dea: 62 f1 7f 89 6f fa            	vmovdqu8	%xmm2, %xmm7 {%k1} {z}
  353df0: 62 f2 bd 48 37 c8            	vpcmpgtq	%zmm0, %zmm8, %k1
  353df6: c5 d1 6c ee                  	vpunpcklqdq	%xmm6, %xmm5, %xmm5 # xmm5 = xmm5[0],xmm6[0]
  353dfa: c4 e3 55 38 ef 01            	vinserti128	$0x1, %xmm7, %ymm5, %ymm5
  353e00: 62 f1 7f 89 6f f2            	vmovdqu8	%xmm2, %xmm6 {%k1} {z}
  353e06: c4 e2 7d 59 f6               	vpbroadcastq	%xmm6, %ymm6
  353e0b: c4 e3 55 02 ee c0            	vpblendd	$0xc0, %ymm6, %ymm5, %ymm5 # ymm5 = ymm5[0,1,2,3,4,5],ymm6[6,7]
  353e11: 62 f2 b5 48 37 c8            	vpcmpgtq	%zmm0, %zmm9, %k1
  353e17: 62 f1 7f 89 6f f2            	vmovdqu8	%xmm2, %xmm6 {%k1} {z}
  353e1d: 62 f2 ad 48 37 c8            	vpcmpgtq	%zmm0, %zmm10, %k1
  353e23: 62 f1 7f 89 6f fa            	vmovdqu8	%xmm2, %xmm7 {%k1} {z}
  353e29: 62 f2 a5 48 37 c8            	vpcmpgtq	%zmm0, %zmm11, %k1
  353e2f: 62 71 7f 89 6f c2            	vmovdqu8	%xmm2, %xmm8 {%k1} {z}
  353e35: 62 f2 9d 48 37 c8            	vpcmpgtq	%zmm0, %zmm12, %k1
  353e3b: 62 71 7f 89 6f ca            	vmovdqu8	%xmm2, %xmm9 {%k1} {z}
  353e41: 62 f3 7d 48 38 f6 02         	vinserti32x4	$0x2, %xmm6, %zmm0, %zmm6
  353e48: 62 f3 cd 48 3a ed 00         	vinserti64x4	$0x0, %ymm5, %zmm6, %zmm5
  353e4f: 62 f2 e5 48 7e ef            	vpermt2q	%zmm7, %zmm3, %zmm5
  353e55: 62 d3 55 48 38 e8 03         	vinserti32x4	$0x3, %xmm8, %zmm5, %zmm5
  353e5c: 62 d2 dd 48 7e e9            	vpermt2q	%zmm9, %zmm4, %zmm5
  353e62: 62 f2 55 48 26 cd            	vptestmb	%zmm5, %zmm5, %k1
  353e68: c4 a1 f8 91 0c 07            	kmovq	%k1, (%rdi,%r8)
  353e6e: 41 0f 95 c1                  	setne	%r9b
  353e72: 49 83 c0 08                  	addq	$0x8, %r8
  353e76: 48 81 c3 00 02 00 00         	addq	$0x200, %rbx            # imm = 0x200
  353e7d: 4c 39 c1                     	cmpq	%r8, %rcx
  353e80: 0f 85 aa fe ff ff            	jne	0x353d30 <vortex_buffer::bit::pack::collect_bool_words_avx512::<vortex_array::scalar_fn::unstable::row::execute::packed_bool::execute_bool_dense_attempt<(i64, i64), (), bool, true, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#1}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}>+0x3b0>
  353e86: e9 1e 02 00 00               	jmp	0x3540a9 <vortex_buffer::bit::pack::collect_bool_words_avx512::<vortex_array::scalar_fn::unstable::row::execute::packed_bool::execute_bool_dense_attempt<(i64, i64), (), bool, true, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#1}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}>+0x729>
  353e8b: 41 bc c0 01 00 00            	movl	$0x1c0, %r12d           # imm = 0x1C0
  353e91: 45 31 ed                     	xorl	%r13d, %r13d
  353e94: 62 f2 fd 48 59 05 82 82 d4 ff	vpbroadcastq	-0x2b7d7e(%rip), %zmm0 # 0x9c120 <anon.4e66a2221d34052adc0f362485be06ac.633.llvm.27319815200592487+0x50>
  353e9e: c5 f9 6f 0d ea 19 d3 ff      	vmovdqa	-0x2ce616(%rip), %xmm1  # 0x85890 <anon.37e9284aed57eb9304064780576275d5.103.llvm.15049800457202559735+0x90>
  353ea6: 62 f1 fd 48 6f 15 10 e0 d4 ff	vmovdqa64	-0x2b1ff0(%rip), %zmm2 # 0xa1ec0 <anon.96e61e178d3ba3a8159b8fb9481c24f2.0.llvm.3084270292863525209+0x140>
  353eb0: 62 f1 fd 48 6f 1d 46 e0 d4 ff	vmovdqa64	-0x2b1fba(%rip), %zmm3 # 0xa1f00 <anon.96e61e178d3ba3a8159b8fb9481c24f2.0.llvm.3084270292863525209+0x180>
  353eba: 45 89 f9                     	movl	%r15d, %r9d
  353ebd: 0f 1f 00                     	nopl	(%rax)
  353ec0: c4 c1 7b 92 c1               	kmovd	%r9d, %k0
  353ec5: c4 e3 79 32 c0 07            	kshiftlb	$0x7, %k0, %k0
  353ecb: c4 e3 79 30 c0 07            	kshiftrb	$0x7, %k0, %k0
  353ed1: 62 91 fe 48 6f 7c 26 f9      	vmovdqu64	-0x1c0(%r14,%r12), %zmm7
  353ed9: 62 91 fe 48 6f 74 26 fa      	vmovdqu64	-0x180(%r14,%r12), %zmm6
  353ee1: 62 91 fe 48 6f 6c 26 fb      	vmovdqu64	-0x140(%r14,%r12), %zmm5
  353ee9: 62 91 fe 48 6f 64 26 fc      	vmovdqu64	-0x100(%r14,%r12), %zmm4
  353ef1: 62 31 fe 48 6f 5c 23 f9      	vmovdqu64	-0x1c0(%rbx,%r12), %zmm11
  353ef9: 62 31 fe 48 6f 54 23 fa      	vmovdqu64	-0x180(%rbx,%r12), %zmm10
  353f01: 62 31 fe 48 6f 4c 23 fb      	vmovdqu64	-0x140(%rbx,%r12), %zmm9
  353f09: 62 31 fe 48 6f 44 23 fc      	vmovdqu64	-0x100(%rbx,%r12), %zmm8
  353f11: 62 f2 c5 48 29 c8            	vpcmpeqq	%zmm0, %zmm7, %k1
  353f17: 62 f2 cd 48 29 d0            	vpcmpeqq	%zmm0, %zmm6, %k2
  353f1d: 62 f2 d5 48 29 e0            	vpcmpeqq	%zmm0, %zmm5, %k4
  353f23: 62 f2 dd 48 29 e8            	vpcmpeqq	%zmm0, %zmm4, %k5
  353f29: 62 f2 a5 48 29 d8            	vpcmpeqq	%zmm0, %zmm11, %k3
  353f2f: c5 f5 45 cb                  	korb	%k3, %k1, %k1
  353f33: c5 fd 45 d9                  	korb	%k1, %k0, %k3
  353f37: 62 f2 ad 48 29 c0            	vpcmpeqq	%zmm0, %zmm10, %k0
  353f3d: c5 ed 45 d0                  	korb	%k0, %k2, %k2
  353f41: 62 f2 b5 48 29 c0            	vpcmpeqq	%zmm0, %zmm9, %k0
  353f47: c5 dd 45 c8                  	korb	%k0, %k4, %k1
  353f4b: 62 f2 bd 48 29 c0            	vpcmpeqq	%zmm0, %zmm8, %k0
  353f51: c5 d5 45 c0                  	korb	%k0, %k5, %k0
  353f55: 62 11 fe 48 6f 64 26 fd      	vmovdqu64	-0xc0(%r14,%r12), %zmm12
  353f5d: 62 11 fe 48 6f 6c 26 fe      	vmovdqu64	-0x80(%r14,%r12), %zmm13
  353f65: 62 31 fe 48 6f 74 23 fd      	vmovdqu64	-0xc0(%rbx,%r12), %zmm14
  353f6d: 62 31 fe 48 6f 7c 23 fe      	vmovdqu64	-0x80(%rbx,%r12), %zmm15
  353f75: 62 f2 9d 48 29 e0            	vpcmpeqq	%zmm0, %zmm12, %k4
  353f7b: 62 f2 8d 48 29 e8            	vpcmpeqq	%zmm0, %zmm14, %k5
  353f81: c5 dd 45 e5                  	korb	%k5, %k4, %k4
  353f85: 62 f2 95 48 29 e8            	vpcmpeqq	%zmm0, %zmm13, %k5
  353f8b: c5 e5 45 dc                  	korb	%k4, %k3, %k3
  353f8f: 62 f2 85 48 29 e0            	vpcmpeqq	%zmm0, %zmm15, %k4
  353f95: c5 d5 45 e4                  	korb	%k4, %k5, %k4
  353f99: 62 81 fe 48 6f 44 26 ff      	vmovdqu64	-0x40(%r14,%r12), %zmm16
  353fa1: 62 a1 fe 48 6f 4c 23 ff      	vmovdqu64	-0x40(%rbx,%r12), %zmm17
  353fa9: c5 ed 45 d4                  	korb	%k4, %k2, %k2
  353fad: 62 f2 fd 40 29 e0            	vpcmpeqq	%zmm0, %zmm16, %k4
  353fb3: c5 ed 45 d3                  	korb	%k3, %k2, %k2
  353fb7: 62 f2 f5 40 29 d8            	vpcmpeqq	%zmm0, %zmm17, %k3
  353fbd: c5 dd 45 db                  	korb	%k3, %k4, %k3
  353fc1: 62 81 fe 48 6f 14 26         	vmovdqu64	(%r14,%r12), %zmm18
  353fc8: 62 a1 fe 48 6f 1c 23         	vmovdqu64	(%rbx,%r12), %zmm19
  353fcf: c5 f5 45 cb                  	korb	%k3, %k1, %k1
  353fd3: 62 f2 ed 40 29 d8            	vpcmpeqq	%zmm0, %zmm18, %k3
  353fd9: c5 f5 45 ca                  	korb	%k2, %k1, %k1
  353fdd: 62 f2 e5 40 29 d0            	vpcmpeqq	%zmm0, %zmm19, %k2
  353fe3: c5 e5 45 d2                  	korb	%k2, %k3, %k2
  353fe7: c5 fd 45 c2                  	korb	%k2, %k0, %k0
  353feb: c5 f9 98 c1                  	kortestb	%k1, %k0
  353fef: 62 f2 a5 48 37 cf            	vpcmpgtq	%zmm7, %zmm11, %k1
  353ff5: 62 f1 7f 89 6f f9            	vmovdqu8	%xmm1, %xmm7 {%k1} {z}
  353ffb: 62 f2 ad 48 37 ce            	vpcmpgtq	%zmm6, %zmm10, %k1
  354001: 62 f1 7f 89 6f f1            	vmovdqu8	%xmm1, %xmm6 {%k1} {z}
  354007: 62 f2 b5 48 37 cd            	vpcmpgtq	%zmm5, %zmm9, %k1
  35400d: 62 f1 7f 89 6f e9            	vmovdqu8	%xmm1, %xmm5 {%k1} {z}
  354013: 62 f2 bd 48 37 cc            	vpcmpgtq	%zmm4, %zmm8, %k1
  354019: c5 c1 6c e6                  	vpunpcklqdq	%xmm6, %xmm7, %xmm4 # xmm4 = xmm7[0],xmm6[0]
  35401d: c4 e3 5d 38 e5 01            	vinserti128	$0x1, %xmm5, %ymm4, %ymm4
  354023: 62 f1 7f 89 6f e9            	vmovdqu8	%xmm1, %xmm5 {%k1} {z}
  354029: c4 e2 7d 59 ed               	vpbroadcastq	%xmm5, %ymm5
  35402e: c4 e3 5d 02 e5 c0            	vpblendd	$0xc0, %ymm5, %ymm4, %ymm4 # ymm4 = ymm4[0,1,2,3,4,5],ymm5[6,7]
  354034: 62 d2 8d 48 37 cc            	vpcmpgtq	%zmm12, %zmm14, %k1
  35403a: 62 f1 7f 89 6f e9            	vmovdqu8	%xmm1, %xmm5 {%k1} {z}
  354040: 62 d2 85 48 37 cd            	vpcmpgtq	%zmm13, %zmm15, %k1
  354046: 62 f1 7f 89 6f f1            	vmovdqu8	%xmm1, %xmm6 {%k1} {z}
  35404c: 62 b2 f5 40 37 c8            	vpcmpgtq	%zmm16, %zmm17, %k1
  354052: 62 f1 7f 89 6f f9            	vmovdqu8	%xmm1, %xmm7 {%k1} {z}
  354058: 62 b2 e5 40 37 ca            	vpcmpgtq	%zmm18, %zmm19, %k1
  35405e: 62 71 7f 89 6f c1            	vmovdqu8	%xmm1, %xmm8 {%k1} {z}
  354064: 62 f3 7d 48 38 ed 02         	vinserti32x4	$0x2, %xmm5, %zmm0, %zmm5
  35406b: 62 f3 d5 48 3a e4 00         	vinserti64x4	$0x0, %ymm4, %zmm5, %zmm4
  354072: 62 f2 ed 48 7e e6            	vpermt2q	%zmm6, %zmm2, %zmm4
  354078: 62 f3 5d 48 38 e7 03         	vinserti32x4	$0x3, %xmm7, %zmm4, %zmm4
  35407f: 62 d2 e5 48 7e e0            	vpermt2q	%zmm8, %zmm3, %zmm4
  354085: 62 f2 5d 48 26 c4            	vptestmb	%zmm4, %zmm4, %k0
  35408b: c4 a1 f8 91 04 2f            	kmovq	%k0, (%rdi,%r13)
  354091: 41 0f 95 c1                  	setne	%r9b
  354095: 49 83 c5 08                  	addq	$0x8, %r13
  354099: 49 81 c4 00 02 00 00         	addq	$0x200, %r12            # imm = 0x200
  3540a0: 4c 39 e9                     	cmpq	%r13, %rcx
  3540a3: 0f 85 17 fe ff ff            	jne	0x353ec0 <vortex_buffer::bit::pack::collect_bool_words_avx512::<vortex_array::scalar_fn::unstable::row::execute::packed_bool::execute_bool_dense_attempt<(i64, i64), (), bool, true, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#1}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}>+0x540>
  3540a9: 41 80 e1 01                  	andb	$0x1, %r9b
  3540ad: 44 88 48 38                  	movb	%r9b, 0x38(%rax)
  3540b1: 4d 85 db                     	testq	%r11, %r11
  3540b4: 0f 84 0f 09 00 00            	je	0x3549c9 <vortex_buffer::bit::pack::collect_bool_words_avx512::<vortex_array::scalar_fn::unstable::row::execute::packed_bool::execute_bool_dense_attempt<(i64, i64), (), bool, true, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#1}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}>+0x1049>
  3540ba: 49 89 d6                     	movq	%rdx, %r14
  3540bd: 49 83 e6 c0                  	andq	$-0x40, %r14
  3540c1: 44 0f b6 68 38               	movzbl	0x38(%rax), %r13d
  3540c6: 44 0f b6 40 18               	movzbl	0x18(%rax), %r8d
  3540cb: 48 8b 78 20                  	movq	0x20(%rax), %rdi
  3540cf: 48 8b 48 28                  	movq	0x28(%rax), %rcx
  3540d3: 83 38 01                     	cmpl	$0x1, (%rax)
  3540d6: 75 29                        	jne	0x354101 <vortex_buffer::bit::pack::collect_bool_words_avx512::<vortex_array::scalar_fn::unstable::row::execute::packed_bool::execute_bool_dense_attempt<(i64, i64), (), bool, true, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#1}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}>+0x781>
  3540d8: 4c 8b 48 10                  	movq	0x10(%rax), %r9
  3540dc: 49 8b 19                     	movq	(%r9), %rbx
  3540df: 45 84 c0                     	testb	%r8b, %r8b
  3540e2: 74 40                        	je	0x354124 <vortex_buffer::bit::pack::collect_bool_words_avx512::<vortex_array::scalar_fn::unstable::row::execute::packed_bool::execute_bool_dense_attempt<(i64, i64), (), bool, true, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#1}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}>+0x7a4>
  3540e4: 48 8b 39                     	movq	(%rcx), %rdi
  3540e7: 45 31 f6                     	xorl	%r14d, %r14d
  3540ea: 48 39 fb                     	cmpq	%rdi, %rbx
  3540ed: 41 0f 9c c6                  	setl	%r14b
  3540f1: 41 83 fb 04                  	cmpl	$0x4, %r11d
  3540f5: 73 5b                        	jae	0x354152 <vortex_buffer::bit::pack::collect_bool_words_avx512::<vortex_array::scalar_fn::unstable::row::execute::packed_bool::execute_bool_dense_attempt<(i64, i64), (), bool, true, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#1}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}>+0x7d2>
  3540f7: 45 31 ff                     	xorl	%r15d, %r15d
  3540fa: 31 c9                        	xorl	%ecx, %ecx
  3540fc: e9 af 01 00 00               	jmp	0x3542b0 <vortex_buffer::bit::pack::collect_bool_words_avx512::<vortex_array::scalar_fn::unstable::row::execute::packed_bool::execute_bool_dense_attempt<(i64, i64), (), bool, true, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#1}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}>+0x930>
  354101: 48 8b 58 08                  	movq	0x8(%rax), %rbx
  354105: 45 84 c0                     	testb	%r8b, %r8b
  354108: 74 31                        	je	0x35413b <vortex_buffer::bit::pack::collect_bool_words_avx512::<vortex_array::scalar_fn::unstable::row::execute::packed_bool::execute_bool_dense_attempt<(i64, i64), (), bool, true, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#1}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}>+0x7bb>
  35410a: 48 8b 39                     	movq	(%rcx), %rdi
  35410d: 41 83 fb 04                  	cmpl	$0x4, %r11d
  354111: 73 4f                        	jae	0x354162 <vortex_buffer::bit::pack::collect_bool_words_avx512::<vortex_array::scalar_fn::unstable::row::execute::packed_bool::execute_bool_dense_attempt<(i64, i64), (), bool, true, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#1}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}>+0x7e2>
  354113: 45 31 ff                     	xorl	%r15d, %r15d
  354116: 45 89 ec                     	movl	%r13d, %r12d
  354119: 31 c9                        	xorl	%ecx, %ecx
  35411b: 4c 8b 6d d0                  	movq	-0x30(%rbp), %r13
  35411f: e9 be 03 00 00               	jmp	0x3544e2 <vortex_buffer::bit::pack::collect_bool_words_avx512::<vortex_array::scalar_fn::unstable::row::execute::packed_bool::execute_bool_dense_attempt<(i64, i64), (), bool, true, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#1}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}>+0xb62>
  354124: 41 83 fb 04                  	cmpl	$0x4, %r11d
  354128: 73 56                        	jae	0x354180 <vortex_buffer::bit::pack::collect_bool_words_avx512::<vortex_array::scalar_fn::unstable::row::execute::packed_bool::execute_bool_dense_attempt<(i64, i64), (), bool, true, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#1}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}>+0x800>
  35412a: 45 31 ff                     	xorl	%r15d, %r15d
  35412d: 45 89 ec                     	movl	%r13d, %r12d
  354130: 31 c9                        	xorl	%ecx, %ecx
  354132: 4c 8b 6d d0                  	movq	-0x30(%rbp), %r13
  354136: e9 f7 05 00 00               	jmp	0x354732 <vortex_buffer::bit::pack::collect_bool_words_avx512::<vortex_array::scalar_fn::unstable::row::execute::packed_bool::execute_bool_dense_attempt<(i64, i64), (), bool, true, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#1}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}>+0xdb2>
  35413b: 41 83 fb 04                  	cmpl	$0x4, %r11d
  35413f: 73 5d                        	jae	0x35419e <vortex_buffer::bit::pack::collect_bool_words_avx512::<vortex_array::scalar_fn::unstable::row::execute::packed_bool::execute_bool_dense_attempt<(i64, i64), (), bool, true, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#1}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}>+0x81e>
  354141: 45 31 ff                     	xorl	%r15d, %r15d
  354144: 45 89 ec                     	movl	%r13d, %r12d
  354147: 31 c9                        	xorl	%ecx, %ecx
  354149: 4c 8b 6d d0                  	movq	-0x30(%rbp), %r13
  35414d: e9 19 08 00 00               	jmp	0x35496b <vortex_buffer::bit::pack::collect_bool_words_avx512::<vortex_array::scalar_fn::unstable::row::execute::packed_bool::execute_bool_dense_attempt<(i64, i64), (), bool, true, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#1}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}>+0xfeb>
  354152: 41 83 fb 10                  	cmpl	$0x10, %r11d
  354156: 73 61                        	jae	0x3541b9 <vortex_buffer::bit::pack::collect_bool_words_avx512::<vortex_array::scalar_fn::unstable::row::execute::packed_bool::execute_bool_dense_attempt<(i64, i64), (), bool, true, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#1}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}>+0x839>
  354158: 31 c9                        	xorl	%ecx, %ecx
  35415a: 45 31 ff                     	xorl	%r15d, %r15d
  35415d: e9 ed 00 00 00               	jmp	0x35424f <vortex_buffer::bit::pack::collect_bool_words_avx512::<vortex_array::scalar_fn::unstable::row::execute::packed_bool::execute_bool_dense_attempt<(i64, i64), (), bool, true, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#1}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}>+0x8cf>
  354162: 45 31 c0                     	xorl	%r8d, %r8d
  354165: 41 83 fb 10                  	cmpl	$0x10, %r11d
  354169: 0f 83 6e 01 00 00            	jae	0x3542dd <vortex_buffer::bit::pack::collect_bool_words_avx512::<vortex_array::scalar_fn::unstable::row::execute::packed_bool::execute_bool_dense_attempt<(i64, i64), (), bool, true, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#1}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}>+0x95d>
  35416f: 31 c9                        	xorl	%ecx, %ecx
  354171: 45 89 ec                     	movl	%r13d, %r12d
  354174: 45 31 ff                     	xorl	%r15d, %r15d
  354177: 4c 8b 6d d0                  	movq	-0x30(%rbp), %r13
  35417b: e9 99 02 00 00               	jmp	0x354419 <vortex_buffer::bit::pack::collect_bool_words_avx512::<vortex_array::scalar_fn::unstable::row::execute::packed_bool::execute_bool_dense_attempt<(i64, i64), (), bool, true, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#1}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}>+0xa99>
  354180: 45 31 c0                     	xorl	%r8d, %r8d
  354183: 41 83 fb 10                  	cmpl	$0x10, %r11d
  354187: 0f 83 9f 03 00 00            	jae	0x35452c <vortex_buffer::bit::pack::collect_bool_words_avx512::<vortex_array::scalar_fn::unstable::row::execute::packed_bool::execute_bool_dense_attempt<(i64, i64), (), bool, true, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#1}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}>+0xbac>
  35418d: 31 c9                        	xorl	%ecx, %ecx
  35418f: 45 89 ec                     	movl	%r13d, %r12d
  354192: 45 31 ff                     	xorl	%r15d, %r15d
  354195: 4c 8b 6d d0                  	movq	-0x30(%rbp), %r13
  354199: e9 cb 04 00 00               	jmp	0x354669 <vortex_buffer::bit::pack::collect_bool_words_avx512::<vortex_array::scalar_fn::unstable::row::execute::packed_bool::execute_bool_dense_attempt<(i64, i64), (), bool, true, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#1}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}>+0xce9>
  35419e: 41 83 fb 10                  	cmpl	$0x10, %r11d
  3541a2: 0f 83 d4 05 00 00            	jae	0x35477c <vortex_buffer::bit::pack::collect_bool_words_avx512::<vortex_array::scalar_fn::unstable::row::execute::packed_bool::execute_bool_dense_attempt<(i64, i64), (), bool, true, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#1}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}>+0xdfc>
  3541a8: 31 c9                        	xorl	%ecx, %ecx
  3541aa: 45 89 ec                     	movl	%r13d, %r12d
  3541ad: 45 31 ff                     	xorl	%r15d, %r15d
  3541b0: 4c 8b 6d d0                  	movq	-0x30(%rbp), %r13
  3541b4: e9 02 07 00 00               	jmp	0x3548bb <vortex_buffer::bit::pack::collect_bool_words_avx512::<vortex_array::scalar_fn::unstable::row::execute::packed_bool::execute_bool_dense_attempt<(i64, i64), (), bool, true, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#1}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}>+0xf3b>
  3541b9: 89 d1                        	movl	%edx, %ecx
  3541bb: 83 e1 30                     	andl	$0x30, %ecx
  3541be: 62 d2 fd 48 7c c6            	vpbroadcastq	%r14, %zmm0
  3541c4: 62 f1 fd 48 6f 15 72 dd d4 ff	vmovdqa64	-0x2b228e(%rip), %zmm2 # 0xa1f40 <anon.96e61e178d3ba3a8159b8fb9481c24f2.0.llvm.3084270292863525209+0x1c0>
  3541ce: c5 f1 ef c9                  	vpxor	%xmm1, %xmm1, %xmm1
  3541d2: 62 f2 fd 48 59 1d ec 79 d4 ff	vpbroadcastq	-0x2b8614(%rip), %zmm3 # 0x9bbc8 <anon.4d90d9de9367fb21f144d13924f001fb.147.llvm.1315488742485634063>
  3541dc: 62 f2 fd 48 59 25 f2 85 d4 ff	vpbroadcastq	-0x2b7a0e(%rip), %zmm4 # 0x9c7d8 <anon.4e66a2221d34052adc0f362485be06ac.728.llvm.27319815200592487+0x48>
  3541e6: 49 89 c8                     	movq	%rcx, %r8
  3541e9: c5 d1 ef ed                  	vpxor	%xmm5, %xmm5, %xmm5
  3541ed: 0f 1f 00                     	nopl	(%rax)
  3541f0: 62 f1 ed 48 d4 f3            	vpaddq	%zmm3, %zmm2, %zmm6
  3541f6: 62 f2 fd 48 47 fa            	vpsllvq	%zmm2, %zmm0, %zmm7
  3541fc: 62 f1 c5 48 eb c9            	vporq	%zmm1, %zmm7, %zmm1
  354202: 62 f2 fd 48 47 f6            	vpsllvq	%zmm6, %zmm0, %zmm6
  354208: 62 f1 cd 48 eb ed            	vporq	%zmm5, %zmm6, %zmm5
  35420e: 62 f1 ed 48 d4 d4            	vpaddq	%zmm4, %zmm2, %zmm2
  354214: 49 83 c0 f0                  	addq	$-0x10, %r8
  354218: 75 d6                        	jne	0x3541f0 <vortex_buffer::bit::pack::collect_bool_words_avx512::<vortex_array::scalar_fn::unstable::row::execute::packed_bool::execute_bool_dense_attempt<(i64, i64), (), bool, true, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#1}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}>+0x870>
  35421a: 62 f1 d5 48 eb c1            	vporq	%zmm1, %zmm5, %zmm0
  354220: 62 f3 fd 48 3b c1 01         	vextracti64x4	$0x1, %zmm0, %ymm1
  354227: 62 f1 fd 48 eb c1            	vporq	%zmm1, %zmm0, %zmm0
  35422d: c4 e3 7d 39 c1 01            	vextracti128	$0x1, %ymm0, %xmm1
  354233: c5 f9 eb c1                  	vpor	%xmm1, %xmm0, %xmm0
  354237: c5 f9 70 c8 ee               	vpshufd	$0xee, %xmm0, %xmm1     # xmm1 = xmm0[2,3,2,3]
  35423c: c5 f9 eb c1                  	vpor	%xmm1, %xmm0, %xmm0
  354240: c4 c1 f9 7e c7               	vmovq	%xmm0, %r15
  354245: 41 39 cb                     	cmpl	%ecx, %r11d
  354248: 74 77                        	je	0x3542c1 <vortex_buffer::bit::pack::collect_bool_words_avx512::<vortex_array::scalar_fn::unstable::row::execute::packed_bool::execute_bool_dense_attempt<(i64, i64), (), bool, true, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#1}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}>+0x941>
  35424a: f6 c2 0c                     	testb	$0xc, %dl
  35424d: 74 61                        	je	0x3542b0 <vortex_buffer::bit::pack::collect_bool_words_avx512::<vortex_array::scalar_fn::unstable::row::execute::packed_bool::execute_bool_dense_attempt<(i64, i64), (), bool, true, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#1}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}>+0x930>
  35424f: 49 89 c8                     	movq	%rcx, %r8
  354252: 89 d1                        	movl	%edx, %ecx
  354254: 83 e1 3c                     	andl	$0x3c, %ecx
  354257: c4 c1 f9 6e c7               	vmovq	%r15, %xmm0
  35425c: 62 d2 fd 28 7c c8            	vpbroadcastq	%r8, %ymm1
  354262: c5 f5 eb 0d f6 cc d4 ff      	vpor	-0x2b330a(%rip), %ymm1, %ymm1 # 0xa0f60 <anon.e6fb86fb181e0aaaed06fff69d344575.10.llvm.18170190781415527360+0xa0>
  35426a: 62 d2 fd 28 7c d6            	vpbroadcastq	%r14, %ymm2
  354270: 49 29 c8                     	subq	%rcx, %r8
  354273: c4 e2 7d 59 1d ec 84 d4 ff   	vpbroadcastq	-0x2b7b14(%rip), %ymm3 # 0x9c768 <anon.4d90d9de9367fb21f144d13924f001fb.148.llvm.1315488742485634063>
  35427c: 0f 1f 40 00                  	nopl	(%rax)
  354280: c4 e2 ed 47 e1               	vpsllvq	%ymm1, %ymm2, %ymm4
  354285: c5 dd eb c0                  	vpor	%ymm0, %ymm4, %ymm0
  354289: c5 f5 d4 cb                  	vpaddq	%ymm3, %ymm1, %ymm1
  35428d: 49 83 c0 04                  	addq	$0x4, %r8
  354291: 75 ed                        	jne	0x354280 <vortex_buffer::bit::pack::collect_bool_words_avx512::<vortex_array::scalar_fn::unstable::row::execute::packed_bool::execute_bool_dense_attempt<(i64, i64), (), bool, true, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#1}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}>+0x900>
  354293: c4 e3 7d 39 c1 01            	vextracti128	$0x1, %ymm0, %xmm1
  354299: c5 f9 eb c1                  	vpor	%xmm1, %xmm0, %xmm0
  35429d: c5 f9 70 c8 ee               	vpshufd	$0xee, %xmm0, %xmm1     # xmm1 = xmm0[2,3,2,3]
  3542a2: c5 f9 eb c1                  	vpor	%xmm1, %xmm0, %xmm0
  3542a6: c4 c1 f9 7e c7               	vmovq	%xmm0, %r15
  3542ab: 41 39 cb                     	cmpl	%ecx, %r11d
  3542ae: 74 11                        	je	0x3542c1 <vortex_buffer::bit::pack::collect_bool_words_avx512::<vortex_array::scalar_fn::unstable::row::execute::packed_bool::execute_bool_dense_attempt<(i64, i64), (), bool, true, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#1}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}>+0x941>
  3542b0: 4c 89 f2                     	movq	%r14, %rdx
  3542b3: 48 d3 e2                     	shlq	%cl, %rdx
  3542b6: 48 ff c1                     	incq	%rcx
  3542b9: 49 09 d7                     	orq	%rdx, %r15
  3542bc: 49 39 cb                     	cmpq	%rcx, %r11
  3542bf: 75 ef                        	jne	0x3542b0 <vortex_buffer::bit::pack::collect_bool_words_avx512::<vortex_array::scalar_fn::unstable::row::execute::packed_bool::execute_bool_dense_attempt<(i64, i64), (), bool, true, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#1}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}>+0x930>
  3542c1: 4c 39 d3                     	cmpq	%r10, %rbx
  3542c4: 0f 94 c1                     	sete	%cl
  3542c7: 4c 39 d7                     	cmpq	%r10, %rdi
  3542ca: 41 0f 94 c4                  	sete	%r12b
  3542ce: 41 08 cc                     	orb	%cl, %r12b
  3542d1: 45 08 ec                     	orb	%r13b, %r12b
  3542d4: 4c 8b 6d d0                  	movq	-0x30(%rbp), %r13
  3542d8: e9 d7 06 00 00               	jmp	0x3549b4 <vortex_buffer::bit::pack::collect_bool_words_avx512::<vortex_array::scalar_fn::unstable::row::execute::packed_bool::execute_bool_dense_attempt<(i64, i64), (), bool, true, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#1}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}>+0x1034>
  3542dd: 89 d1                        	movl	%edx, %ecx
  3542df: 83 e1 30                     	andl	$0x30, %ecx
  3542e2: 4c 39 d7                     	cmpq	%r10, %rdi
  3542e5: c4 c1 7b 92 c5               	kmovd	%r13d, %k0
  3542ea: c4 e3 79 32 c0 07            	kshiftlb	$0x7, %k0, %k0
  3542f0: c4 e3 79 30 c0 07            	kshiftrb	$0x7, %k0, %k0
  3542f6: 62 f2 fd 48 7c c7            	vpbroadcastq	%rdi, %zmm0
  3542fc: 41 b9 ff 00 00 00            	movl	$0xff, %r9d
  354302: 45 0f 45 c8                  	cmovnel	%r8d, %r9d
  354306: c4 c1 7b 92 c9               	kmovd	%r9d, %k1
  35430b: 4e 8d 0c f3                  	leaq	(%rbx,%r14,8), %r9
  35430f: 62 f1 fd 48 6f 15 27 dc d4 ff	vmovdqa64	-0x2b23d9(%rip), %zmm2 # 0xa1f40 <anon.96e61e178d3ba3a8159b8fb9481c24f2.0.llvm.3084270292863525209+0x1c0>
  354319: c5 f1 ef c9                  	vpxor	%xmm1, %xmm1, %xmm1
  35431d: c5 fd 47 d0                  	kxorb	%k0, %k0, %k2
  354321: 62 f2 fd 48 59 1d 9d 78 d4 ff	vpbroadcastq	-0x2b8763(%rip), %zmm3 # 0x9bbc8 <anon.4d90d9de9367fb21f144d13924f001fb.147.llvm.1315488742485634063>
  35432b: 62 f2 fd 48 59 25 eb 7d d4 ff	vpbroadcastq	-0x2b8215(%rip), %zmm4 # 0x9c120 <anon.4e66a2221d34052adc0f362485be06ac.633.llvm.27319815200592487+0x50>
  354335: 62 f2 fd 48 59 2d 99 84 d4 ff	vpbroadcastq	-0x2b7b67(%rip), %zmm5 # 0x9c7d8 <anon.4e66a2221d34052adc0f362485be06ac.728.llvm.27319815200592487+0x48>
  35433f: 49 89 cf                     	movq	%rcx, %r15
  354342: c5 c9 ef f6                  	vpxor	%xmm6, %xmm6, %xmm6
  354346: 66 2e 0f 1f 84 00 00 00 00 00	nopw	%cs:(%rax,%rax)
  354350: 62 f1 ed 48 d4 fb            	vpaddq	%zmm3, %zmm2, %zmm7
  354356: c4 c1 f9 7e d4               	vmovq	%xmm2, %r12
  35435b: 62 11 fe 48 6f 04 e1         	vmovdqu64	(%r9,%r12,8), %zmm8
  354362: 62 11 fe 48 6f 4c e1 01      	vmovdqu64	0x40(%r9,%r12,8), %zmm9
  35436a: 62 f2 bd 48 29 e4            	vpcmpeqq	%zmm4, %zmm8, %k4
  354370: 62 f2 b5 48 29 dc            	vpcmpeqq	%zmm4, %zmm9, %k3
  354376: c5 e5 45 da                  	korb	%k2, %k3, %k3
  35437a: 62 d2 fd 48 37 e8            	vpcmpgtq	%zmm8, %zmm0, %k5
  354380: 62 d2 fd 48 37 f1            	vpcmpgtq	%zmm9, %zmm0, %k6
  354386: c5 fd 45 c1                  	korb	%k1, %k0, %k0
  35438a: c5 dd 45 c0                  	korb	%k0, %k4, %k0
  35438e: c5 e5 45 d1                  	korb	%k1, %k3, %k2
  354392: 62 72 fe 48 38 c5            	vpmovm2q	%k5, %zmm8
  354398: 62 d1 bd 48 73 d0 3f         	vpsrlq	$0x3f, %zmm8, %zmm8
  35439f: 62 72 fe 48 38 ce            	vpmovm2q	%k6, %zmm9
  3543a5: 62 d1 b5 48 73 d1 3f         	vpsrlq	$0x3f, %zmm9, %zmm9
  3543ac: 62 f2 b5 48 47 ff            	vpsllvq	%zmm7, %zmm9, %zmm7
  3543b2: 62 f1 c5 48 eb f6            	vporq	%zmm6, %zmm7, %zmm6
  3543b8: 62 f2 bd 48 47 fa            	vpsllvq	%zmm2, %zmm8, %zmm7
  3543be: 62 f1 c5 48 eb c9            	vporq	%zmm1, %zmm7, %zmm1
  3543c4: 62 f1 ed 48 d4 d5            	vpaddq	%zmm5, %zmm2, %zmm2
  3543ca: 49 83 c7 f0                  	addq	$-0x10, %r15
  3543ce: 75 80                        	jne	0x354350 <vortex_buffer::bit::pack::collect_bool_words_avx512::<vortex_array::scalar_fn::unstable::row::execute::packed_bool::execute_bool_dense_attempt<(i64, i64), (), bool, true, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#1}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}>+0x9d0>
  3543d0: c5 f9 98 d8                  	kortestb	%k0, %k3
  3543d4: 41 0f 95 c4                  	setne	%r12b
  3543d8: 62 f1 cd 48 eb c1            	vporq	%zmm1, %zmm6, %zmm0
  3543de: 62 f3 fd 48 3b c1 01         	vextracti64x4	$0x1, %zmm0, %ymm1
  3543e5: 62 f1 fd 48 eb c1            	vporq	%zmm1, %zmm0, %zmm0
  3543eb: c4 e3 7d 39 c1 01            	vextracti128	$0x1, %ymm0, %xmm1
  3543f1: c5 f9 eb c1                  	vpor	%xmm1, %xmm0, %xmm0
  3543f5: c5 f9 70 c8 ee               	vpshufd	$0xee, %xmm0, %xmm1     # xmm1 = xmm0[2,3,2,3]
  3543fa: c5 f9 eb c1                  	vpor	%xmm1, %xmm0, %xmm0
  3543fe: c4 c1 f9 7e c7               	vmovq	%xmm0, %r15
  354403: 41 39 cb                     	cmpl	%ecx, %r11d
  354406: 4c 8b 6d d0                  	movq	-0x30(%rbp), %r13
  35440a: 0f 84 a4 05 00 00            	je	0x3549b4 <vortex_buffer::bit::pack::collect_bool_words_avx512::<vortex_array::scalar_fn::unstable::row::execute::packed_bool::execute_bool_dense_attempt<(i64, i64), (), bool, true, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#1}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}>+0x1034>
  354410: f6 c2 0c                     	testb	$0xc, %dl
  354413: 0f 84 c9 00 00 00            	je	0x3544e2 <vortex_buffer::bit::pack::collect_bool_words_avx512::<vortex_array::scalar_fn::unstable::row::execute::packed_bool::execute_bool_dense_attempt<(i64, i64), (), bool, true, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#1}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}>+0xb62>
  354419: 49 89 c9                     	movq	%rcx, %r9
  35441c: 89 d1                        	movl	%edx, %ecx
  35441e: 83 e1 3c                     	andl	$0x3c, %ecx
  354421: 4c 39 d7                     	cmpq	%r10, %rdi
  354424: c4 c1 7b 92 c4               	kmovd	%r12d, %k0
  354429: c4 e3 79 32 c0 07            	kshiftlb	$0x7, %k0, %k0
  35442f: c4 e3 79 30 c0 07            	kshiftrb	$0x7, %k0, %k0
  354435: c4 c1 f9 6e c7               	vmovq	%r15, %xmm0
  35443a: 62 f2 fd 28 7c cf            	vpbroadcastq	%rdi, %ymm1
  354440: ba ff 00 00 00               	movl	$0xff, %edx
  354445: 44 0f 44 c2                  	cmovel	%edx, %r8d
  354449: c4 c1 7b 92 c8               	kmovd	%r8d, %k1
  35444e: 62 d2 fd 28 7c d1            	vpbroadcastq	%r9, %ymm2
  354454: c5 ed eb 15 04 cb d4 ff      	vpor	-0x2b34fc(%rip), %ymm2, %ymm2 # 0xa0f60 <anon.e6fb86fb181e0aaaed06fff69d344575.10.llvm.18170190781415527360+0xa0>
  35445c: 4a 8d 14 f3                  	leaq	(%rbx,%r14,8), %rdx
  354460: 49 29 c9                     	subq	%rcx, %r9
  354463: c4 e2 7d 59 1d b4 7c d4 ff   	vpbroadcastq	-0x2b834c(%rip), %ymm3 # 0x9c120 <anon.4e66a2221d34052adc0f362485be06ac.633.llvm.27319815200592487+0x50>
  35446c: c4 e2 7d 59 25 f3 82 d4 ff   	vpbroadcastq	-0x2b7d0d(%rip), %ymm4 # 0x9c768 <anon.4d90d9de9367fb21f144d13924f001fb.148.llvm.1315488742485634063>
  354475: 66 66 2e 0f 1f 84 00 00 00 00 00     	nopw	%cs:(%rax,%rax)
  354480: c4 c1 f9 7e d0               	vmovq	%xmm2, %r8
  354485: c4 a1 7e 6f 2c c2            	vmovdqu	(%rdx,%r8,8), %ymm5
  35448b: 62 f2 d5 28 29 d3            	vpcmpeqq	%ymm3, %ymm5, %k2
  354491: c5 ec 45 d1                  	korw	%k1, %k2, %k2
  354495: c5 fc 45 c2                  	korw	%k2, %k0, %k0
  354499: c4 e2 75 37 ed               	vpcmpgtq	%ymm5, %ymm1, %ymm5
  35449e: c5 d5 73 d5 3f               	vpsrlq	$0x3f, %ymm5, %ymm5
  3544a3: c4 e2 d5 47 ea               	vpsllvq	%ymm2, %ymm5, %ymm5
  3544a8: c5 d5 eb c0                  	vpor	%ymm0, %ymm5, %ymm0
  3544ac: c5 ed d4 d4                  	vpaddq	%ymm4, %ymm2, %ymm2
  3544b0: 49 83 c1 04                  	addq	$0x4, %r9
  3544b4: 75 ca                        	jne	0x354480 <vortex_buffer::bit::pack::collect_bool_words_avx512::<vortex_array::scalar_fn::unstable::row::execute::packed_bool::execute_bool_dense_attempt<(i64, i64), (), bool, true, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#1}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}>+0xb00>
  3544b6: c5 fb 93 d0                  	kmovd	%k0, %edx
  3544ba: f6 c2 0f                     	testb	$0xf, %dl
  3544bd: 41 0f 95 c4                  	setne	%r12b
  3544c1: c4 e3 7d 39 c1 01            	vextracti128	$0x1, %ymm0, %xmm1
  3544c7: c5 f9 eb c1                  	vpor	%xmm1, %xmm0, %xmm0
  3544cb: c5 f9 70 c8 ee               	vpshufd	$0xee, %xmm0, %xmm1     # xmm1 = xmm0[2,3,2,3]
  3544d0: c5 f9 eb c1                  	vpor	%xmm1, %xmm0, %xmm0
  3544d4: c4 c1 f9 7e c7               	vmovq	%xmm0, %r15
  3544d9: 41 39 cb                     	cmpl	%ecx, %r11d
  3544dc: 0f 84 d2 04 00 00            	je	0x3549b4 <vortex_buffer::bit::pack::collect_bool_words_avx512::<vortex_array::scalar_fn::unstable::row::execute::packed_bool::execute_bool_dense_attempt<(i64, i64), (), bool, true, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#1}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}>+0x1034>
  3544e2: 48 89 f2                     	movq	%rsi, %rdx
  3544e5: 48 c1 e2 09                  	shlq	$0x9, %rdx
  3544e9: 48 01 d3                     	addq	%rdx, %rbx
  3544ec: 44 89 e2                     	movl	%r12d, %edx
  3544ef: 90                           	nop
  3544f0: 4c 39 d7                     	cmpq	%r10, %rdi
  3544f3: 41 0f 94 c4                  	sete	%r12b
  3544f7: 4c 8b 04 cb                  	movq	(%rbx,%rcx,8), %r8
  3544fb: 4d 39 d0                     	cmpq	%r10, %r8
  3544fe: 41 0f 94 c1                  	sete	%r9b
  354502: 45 31 f6                     	xorl	%r14d, %r14d
  354505: 49 39 f8                     	cmpq	%rdi, %r8
  354508: 41 0f 9c c6                  	setl	%r14b
  35450c: 41 08 d4                     	orb	%dl, %r12b
  35450f: 45 08 cc                     	orb	%r9b, %r12b
  354512: 4c 8d 41 01                  	leaq	0x1(%rcx), %r8
  354516: 49 d3 e6                     	shlq	%cl, %r14
  354519: 4d 09 f7                     	orq	%r14, %r15
  35451c: 44 89 e2                     	movl	%r12d, %edx
  35451f: 4d 39 c3                     	cmpq	%r8, %r11
  354522: 4c 89 c1                     	movq	%r8, %rcx
  354525: 75 c9                        	jne	0x3544f0 <vortex_buffer::bit::pack::collect_bool_words_avx512::<vortex_array::scalar_fn::unstable::row::execute::packed_bool::execute_bool_dense_attempt<(i64, i64), (), bool, true, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#1}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}>+0xb70>
  354527: e9 88 04 00 00               	jmp	0x3549b4 <vortex_buffer::bit::pack::collect_bool_words_avx512::<vortex_array::scalar_fn::unstable::row::execute::packed_bool::execute_bool_dense_attempt<(i64, i64), (), bool, true, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#1}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}>+0x1034>
  35452c: 89 d1                        	movl	%edx, %ecx
  35452e: 83 e1 30                     	andl	$0x30, %ecx
  354531: 4c 39 d3                     	cmpq	%r10, %rbx
  354534: c4 c1 7b 92 c5               	kmovd	%r13d, %k0
  354539: c4 e3 79 32 c0 07            	kshiftlb	$0x7, %k0, %k0
  35453f: c4 e3 79 30 c0 07            	kshiftrb	$0x7, %k0, %k0
  354545: 62 f2 fd 48 7c c3            	vpbroadcastq	%rbx, %zmm0
  35454b: 41 b9 ff 00 00 00            	movl	$0xff, %r9d
  354551: 45 0f 45 c8                  	cmovnel	%r8d, %r9d
  354555: c4 c1 7b 92 c9               	kmovd	%r9d, %k1
  35455a: 4e 8d 0c f7                  	leaq	(%rdi,%r14,8), %r9
  35455e: 62 f1 fd 48 6f 15 d8 d9 d4 ff	vmovdqa64	-0x2b2628(%rip), %zmm2 # 0xa1f40 <anon.96e61e178d3ba3a8159b8fb9481c24f2.0.llvm.3084270292863525209+0x1c0>
  354568: c5 f1 ef c9                  	vpxor	%xmm1, %xmm1, %xmm1
  35456c: c5 fd 47 d0                  	kxorb	%k0, %k0, %k2
  354570: 62 f2 fd 48 59 1d 4e 76 d4 ff	vpbroadcastq	-0x2b89b2(%rip), %zmm3 # 0x9bbc8 <anon.4d90d9de9367fb21f144d13924f001fb.147.llvm.1315488742485634063>
  35457a: 62 f2 fd 48 59 25 9c 7b d4 ff	vpbroadcastq	-0x2b8464(%rip), %zmm4 # 0x9c120 <anon.4e66a2221d34052adc0f362485be06ac.633.llvm.27319815200592487+0x50>
  354584: 62 f2 fd 48 59 2d 4a 82 d4 ff	vpbroadcastq	-0x2b7db6(%rip), %zmm5 # 0x9c7d8 <anon.4e66a2221d34052adc0f362485be06ac.728.llvm.27319815200592487+0x48>
  35458e: 49 89 cf                     	movq	%rcx, %r15
  354591: c5 c9 ef f6                  	vpxor	%xmm6, %xmm6, %xmm6
  354595: 66 66 2e 0f 1f 84 00 00 00 00 00     	nopw	%cs:(%rax,%rax)
  3545a0: 62 f1 ed 48 d4 fb            	vpaddq	%zmm3, %zmm2, %zmm7
  3545a6: c4 c1 f9 7e d4               	vmovq	%xmm2, %r12
  3545ab: 62 11 fe 48 6f 04 e1         	vmovdqu64	(%r9,%r12,8), %zmm8
  3545b2: 62 11 fe 48 6f 4c e1 01      	vmovdqu64	0x40(%r9,%r12,8), %zmm9
  3545ba: 62 f2 bd 48 29 e4            	vpcmpeqq	%zmm4, %zmm8, %k4
  3545c0: 62 f2 b5 48 29 dc            	vpcmpeqq	%zmm4, %zmm9, %k3
  3545c6: c5 e5 45 da                  	korb	%k2, %k3, %k3
  3545ca: 62 f2 bd 48 37 e8            	vpcmpgtq	%zmm0, %zmm8, %k5
  3545d0: 62 f2 b5 48 37 f0            	vpcmpgtq	%zmm0, %zmm9, %k6
  3545d6: c5 fd 45 c1                  	korb	%k1, %k0, %k0
  3545da: c5 dd 45 c0                  	korb	%k0, %k4, %k0
  3545de: c5 e5 45 d1                  	korb	%k1, %k3, %k2
  3545e2: 62 72 fe 48 38 c5            	vpmovm2q	%k5, %zmm8
  3545e8: 62 d1 bd 48 73 d0 3f         	vpsrlq	$0x3f, %zmm8, %zmm8
  3545ef: 62 72 fe 48 38 ce            	vpmovm2q	%k6, %zmm9
  3545f5: 62 d1 b5 48 73 d1 3f         	vpsrlq	$0x3f, %zmm9, %zmm9
  3545fc: 62 f2 b5 48 47 ff            	vpsllvq	%zmm7, %zmm9, %zmm7
  354602: 62 f1 c5 48 eb f6            	vporq	%zmm6, %zmm7, %zmm6
  354608: 62 f2 bd 48 47 fa            	vpsllvq	%zmm2, %zmm8, %zmm7
  35460e: 62 f1 c5 48 eb c9            	vporq	%zmm1, %zmm7, %zmm1
  354614: 62 f1 ed 48 d4 d5            	vpaddq	%zmm5, %zmm2, %zmm2
  35461a: 49 83 c7 f0                  	addq	$-0x10, %r15
  35461e: 75 80                        	jne	0x3545a0 <vortex_buffer::bit::pack::collect_bool_words_avx512::<vortex_array::scalar_fn::unstable::row::execute::packed_bool::execute_bool_dense_attempt<(i64, i64), (), bool, true, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#1}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}>+0xc20>
  354620: c5 f9 98 d8                  	kortestb	%k0, %k3
  354624: 41 0f 95 c4                  	setne	%r12b
  354628: 62 f1 cd 48 eb c1            	vporq	%zmm1, %zmm6, %zmm0
  35462e: 62 f3 fd 48 3b c1 01         	vextracti64x4	$0x1, %zmm0, %ymm1
  354635: 62 f1 fd 48 eb c1            	vporq	%zmm1, %zmm0, %zmm0
  35463b: c4 e3 7d 39 c1 01            	vextracti128	$0x1, %ymm0, %xmm1
  354641: c5 f9 eb c1                  	vpor	%xmm1, %xmm0, %xmm0
  354645: c5 f9 70 c8 ee               	vpshufd	$0xee, %xmm0, %xmm1     # xmm1 = xmm0[2,3,2,3]
  35464a: c5 f9 eb c1                  	vpor	%xmm1, %xmm0, %xmm0
  35464e: c4 c1 f9 7e c7               	vmovq	%xmm0, %r15
  354653: 41 39 cb                     	cmpl	%ecx, %r11d
  354656: 4c 8b 6d d0                  	movq	-0x30(%rbp), %r13
  35465a: 0f 84 54 03 00 00            	je	0x3549b4 <vortex_buffer::bit::pack::collect_bool_words_avx512::<vortex_array::scalar_fn::unstable::row::execute::packed_bool::execute_bool_dense_attempt<(i64, i64), (), bool, true, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#1}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}>+0x1034>
  354660: f6 c2 0c                     	testb	$0xc, %dl
  354663: 0f 84 c9 00 00 00            	je	0x354732 <vortex_buffer::bit::pack::collect_bool_words_avx512::<vortex_array::scalar_fn::unstable::row::execute::packed_bool::execute_bool_dense_attempt<(i64, i64), (), bool, true, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#1}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}>+0xdb2>
  354669: 49 89 c9                     	movq	%rcx, %r9
  35466c: 89 d1                        	movl	%edx, %ecx
  35466e: 83 e1 3c                     	andl	$0x3c, %ecx
  354671: 4c 39 d3                     	cmpq	%r10, %rbx
  354674: c4 c1 7b 92 c4               	kmovd	%r12d, %k0
  354679: c4 e3 79 32 c0 07            	kshiftlb	$0x7, %k0, %k0
  35467f: c4 e3 79 30 c0 07            	kshiftrb	$0x7, %k0, %k0
  354685: c4 c1 f9 6e c7               	vmovq	%r15, %xmm0
  35468a: 62 f2 fd 28 7c cb            	vpbroadcastq	%rbx, %ymm1
  354690: ba ff 00 00 00               	movl	$0xff, %edx
  354695: 44 0f 44 c2                  	cmovel	%edx, %r8d
  354699: c4 c1 7b 92 c8               	kmovd	%r8d, %k1
  35469e: 62 d2 fd 28 7c d1            	vpbroadcastq	%r9, %ymm2
  3546a4: c5 ed eb 15 b4 c8 d4 ff      	vpor	-0x2b374c(%rip), %ymm2, %ymm2 # 0xa0f60 <anon.e6fb86fb181e0aaaed06fff69d344575.10.llvm.18170190781415527360+0xa0>
  3546ac: 4a 8d 14 f7                  	leaq	(%rdi,%r14,8), %rdx
  3546b0: 49 29 c9                     	subq	%rcx, %r9
  3546b3: c4 e2 7d 59 1d 64 7a d4 ff   	vpbroadcastq	-0x2b859c(%rip), %ymm3 # 0x9c120 <anon.4e66a2221d34052adc0f362485be06ac.633.llvm.27319815200592487+0x50>
  3546bc: c4 e2 7d 59 25 a3 80 d4 ff   	vpbroadcastq	-0x2b7f5d(%rip), %ymm4 # 0x9c768 <anon.4d90d9de9367fb21f144d13924f001fb.148.llvm.1315488742485634063>
  3546c5: 66 66 2e 0f 1f 84 00 00 00 00 00     	nopw	%cs:(%rax,%rax)
  3546d0: c4 c1 f9 7e d0               	vmovq	%xmm2, %r8
  3546d5: c4 a1 7e 6f 2c c2            	vmovdqu	(%rdx,%r8,8), %ymm5
  3546db: 62 f2 d5 28 29 d3            	vpcmpeqq	%ymm3, %ymm5, %k2
  3546e1: c5 ec 45 c0                  	korw	%k0, %k2, %k0
  3546e5: c5 fc 45 c1                  	korw	%k1, %k0, %k0
  3546e9: c4 e2 55 37 e9               	vpcmpgtq	%ymm1, %ymm5, %ymm5
  3546ee: c5 d5 73 d5 3f               	vpsrlq	$0x3f, %ymm5, %ymm5
  3546f3: c4 e2 d5 47 ea               	vpsllvq	%ymm2, %ymm5, %ymm5
  3546f8: c5 d5 eb c0                  	vpor	%ymm0, %ymm5, %ymm0
  3546fc: c5 ed d4 d4                  	vpaddq	%ymm4, %ymm2, %ymm2
  354700: 49 83 c1 04                  	addq	$0x4, %r9
  354704: 75 ca                        	jne	0x3546d0 <vortex_buffer::bit::pack::collect_bool_words_avx512::<vortex_array::scalar_fn::unstable::row::execute::packed_bool::execute_bool_dense_attempt<(i64, i64), (), bool, true, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#1}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}>+0xd50>
  354706: c5 fb 93 d0                  	kmovd	%k0, %edx
  35470a: f6 c2 0f                     	testb	$0xf, %dl
  35470d: 41 0f 95 c4                  	setne	%r12b
  354711: c4 e3 7d 39 c1 01            	vextracti128	$0x1, %ymm0, %xmm1
  354717: c5 f9 eb c1                  	vpor	%xmm1, %xmm0, %xmm0
  35471b: c5 f9 70 c8 ee               	vpshufd	$0xee, %xmm0, %xmm1     # xmm1 = xmm0[2,3,2,3]
  354720: c5 f9 eb c1                  	vpor	%xmm1, %xmm0, %xmm0
  354724: c4 c1 f9 7e c7               	vmovq	%xmm0, %r15
  354729: 41 39 cb                     	cmpl	%ecx, %r11d
  35472c: 0f 84 82 02 00 00            	je	0x3549b4 <vortex_buffer::bit::pack::collect_bool_words_avx512::<vortex_array::scalar_fn::unstable::row::execute::packed_bool::execute_bool_dense_attempt<(i64, i64), (), bool, true, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#1}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}>+0x1034>
  354732: 48 89 f2                     	movq	%rsi, %rdx
  354735: 48 c1 e2 09                  	shlq	$0x9, %rdx
  354739: 48 01 d7                     	addq	%rdx, %rdi
  35473c: 44 89 e2                     	movl	%r12d, %edx
  35473f: 90                           	nop
  354740: 4c 39 d3                     	cmpq	%r10, %rbx
  354743: 41 0f 94 c4                  	sete	%r12b
  354747: 4c 8b 04 cf                  	movq	(%rdi,%rcx,8), %r8
  35474b: 4d 39 d0                     	cmpq	%r10, %r8
  35474e: 41 0f 94 c1                  	sete	%r9b
  354752: 45 31 f6                     	xorl	%r14d, %r14d
  354755: 4c 39 c3                     	cmpq	%r8, %rbx
  354758: 41 0f 9c c6                  	setl	%r14b
  35475c: 41 08 d4                     	orb	%dl, %r12b
  35475f: 45 08 cc                     	orb	%r9b, %r12b
  354762: 4c 8d 41 01                  	leaq	0x1(%rcx), %r8
  354766: 49 d3 e6                     	shlq	%cl, %r14
  354769: 4d 09 f7                     	orq	%r14, %r15
  35476c: 44 89 e2                     	movl	%r12d, %edx
  35476f: 4d 39 c3                     	cmpq	%r8, %r11
  354772: 4c 89 c1                     	movq	%r8, %rcx
  354775: 75 c9                        	jne	0x354740 <vortex_buffer::bit::pack::collect_bool_words_avx512::<vortex_array::scalar_fn::unstable::row::execute::packed_bool::execute_bool_dense_attempt<(i64, i64), (), bool, true, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#1}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}>+0xdc0>
  354777: e9 38 02 00 00               	jmp	0x3549b4 <vortex_buffer::bit::pack::collect_bool_words_avx512::<vortex_array::scalar_fn::unstable::row::execute::packed_bool::execute_bool_dense_attempt<(i64, i64), (), bool, true, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#1}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}>+0x1034>
  35477c: 89 d1                        	movl	%edx, %ecx
  35477e: 83 e1 30                     	andl	$0x30, %ecx
  354781: c4 c1 7b 92 c5               	kmovd	%r13d, %k0
  354786: c4 e3 79 32 c0 07            	kshiftlb	$0x7, %k0, %k0
  35478c: c4 e3 79 30 c0 07            	kshiftrb	$0x7, %k0, %k0
  354792: 62 f1 fd 48 6f 0d a4 d7 d4 ff	vmovdqa64	-0x2b285c(%rip), %zmm1 # 0xa1f40 <anon.96e61e178d3ba3a8159b8fb9481c24f2.0.llvm.3084270292863525209+0x1c0>
  35479c: c5 f9 ef c0                  	vpxor	%xmm0, %xmm0, %xmm0
  3547a0: c5 fd 47 c8                  	kxorb	%k0, %k0, %k1
  3547a4: 62 f2 fd 48 59 15 1a 74 d4 ff	vpbroadcastq	-0x2b8be6(%rip), %zmm2 # 0x9bbc8 <anon.4d90d9de9367fb21f144d13924f001fb.147.llvm.1315488742485634063>
  3547ae: 62 f2 fd 48 59 1d 68 79 d4 ff	vpbroadcastq	-0x2b8698(%rip), %zmm3 # 0x9c120 <anon.4e66a2221d34052adc0f362485be06ac.633.llvm.27319815200592487+0x50>
  3547b8: 62 f2 fd 48 59 25 16 80 d4 ff	vpbroadcastq	-0x2b7fea(%rip), %zmm4 # 0x9c7d8 <anon.4e66a2221d34052adc0f362485be06ac.728.llvm.27319815200592487+0x48>
  3547c2: 49 89 c8                     	movq	%rcx, %r8
  3547c5: c5 d1 ef ed                  	vpxor	%xmm5, %xmm5, %xmm5
  3547c9: 0f 1f 80 00 00 00 00         	nopl	(%rax)
  3547d0: 62 f1 f5 48 d4 f2            	vpaddq	%zmm2, %zmm1, %zmm6
  3547d6: c4 c1 f9 7e c9               	vmovq	%xmm1, %r9
  3547db: 4d 01 f1                     	addq	%r14, %r9
  3547de: 62 b1 fe 48 6f 3c cb         	vmovdqu64	(%rbx,%r9,8), %zmm7
  3547e5: 62 31 fe 48 6f 44 cb 01      	vmovdqu64	0x40(%rbx,%r9,8), %zmm8
  3547ed: 62 31 fe 48 6f 0c cf         	vmovdqu64	(%rdi,%r9,8), %zmm9
  3547f4: 62 31 fe 48 6f 54 cf 01      	vmovdqu64	0x40(%rdi,%r9,8), %zmm10
  3547fc: 62 f2 c5 48 29 d3            	vpcmpeqq	%zmm3, %zmm7, %k2
  354802: 62 f2 bd 48 29 db            	vpcmpeqq	%zmm3, %zmm8, %k3
  354808: 62 f2 b5 48 29 e3            	vpcmpeqq	%zmm3, %zmm9, %k4
  35480e: c5 ed 45 d4                  	korb	%k4, %k2, %k2
  354812: c5 fd 45 c2                  	korb	%k2, %k0, %k0
  354816: 62 f2 ad 48 29 d3            	vpcmpeqq	%zmm3, %zmm10, %k2
  35481c: c5 e5 45 d2                  	korb	%k2, %k3, %k2
  354820: c5 f5 45 ca                  	korb	%k2, %k1, %k1
  354824: 62 f2 b5 48 37 d7            	vpcmpgtq	%zmm7, %zmm9, %k2
  35482a: 62 d2 ad 48 37 d8            	vpcmpgtq	%zmm8, %zmm10, %k3
  354830: 62 f2 fe 48 38 fa            	vpmovm2q	%k2, %zmm7
  354836: 62 f1 c5 48 73 d7 3f         	vpsrlq	$0x3f, %zmm7, %zmm7
  35483d: 62 72 fe 48 38 c3            	vpmovm2q	%k3, %zmm8
  354843: 62 d1 bd 48 73 d0 3f         	vpsrlq	$0x3f, %zmm8, %zmm8
  35484a: 62 f2 bd 48 47 f6            	vpsllvq	%zmm6, %zmm8, %zmm6
  354850: 62 f1 cd 48 eb ed            	vporq	%zmm5, %zmm6, %zmm5
  354856: 62 f2 c5 48 47 f1            	vpsllvq	%zmm1, %zmm7, %zmm6
  35485c: 62 f1 cd 48 eb c0            	vporq	%zmm0, %zmm6, %zmm0
  354862: 62 f1 f5 48 d4 cc            	vpaddq	%zmm4, %zmm1, %zmm1
  354868: 49 83 c0 f0                  	addq	$-0x10, %r8
  35486c: 0f 85 5e ff ff ff            	jne	0x3547d0 <vortex_buffer::bit::pack::collect_bool_words_avx512::<vortex_array::scalar_fn::unstable::row::execute::packed_bool::execute_bool_dense_attempt<(i64, i64), (), bool, true, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#1}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}>+0xe50>
  354872: c5 f9 98 c8                  	kortestb	%k0, %k1
  354876: 41 0f 95 c4                  	setne	%r12b
  35487a: 62 f1 d5 48 eb c0            	vporq	%zmm0, %zmm5, %zmm0
  354880: 62 f3 fd 48 3b c1 01         	vextracti64x4	$0x1, %zmm0, %ymm1
  354887: 62 f1 fd 48 eb c1            	vporq	%zmm1, %zmm0, %zmm0
  35488d: c4 e3 7d 39 c1 01            	vextracti128	$0x1, %ymm0, %xmm1
  354893: c5 f9 eb c1                  	vpor	%xmm1, %xmm0, %xmm0
  354897: c5 f9 70 c8 ee               	vpshufd	$0xee, %xmm0, %xmm1     # xmm1 = xmm0[2,3,2,3]
  35489c: c5 f9 eb c1                  	vpor	%xmm1, %xmm0, %xmm0
  3548a0: c4 c1 f9 7e c7               	vmovq	%xmm0, %r15
  3548a5: 41 39 cb                     	cmpl	%ecx, %r11d
  3548a8: 4c 8b 6d d0                  	movq	-0x30(%rbp), %r13
  3548ac: 0f 84 02 01 00 00            	je	0x3549b4 <vortex_buffer::bit::pack::collect_bool_words_avx512::<vortex_array::scalar_fn::unstable::row::execute::packed_bool::execute_bool_dense_attempt<(i64, i64), (), bool, true, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#1}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}>+0x1034>
  3548b2: f6 c2 0c                     	testb	$0xc, %dl
  3548b5: 0f 84 b0 00 00 00            	je	0x35496b <vortex_buffer::bit::pack::collect_bool_words_avx512::<vortex_array::scalar_fn::unstable::row::execute::packed_bool::execute_bool_dense_attempt<(i64, i64), (), bool, true, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#1}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}>+0xfeb>
  3548bb: 49 89 c8                     	movq	%rcx, %r8
  3548be: 89 d1                        	movl	%edx, %ecx
  3548c0: 83 e1 3c                     	andl	$0x3c, %ecx
  3548c3: c4 c1 7b 92 c4               	kmovd	%r12d, %k0
  3548c8: c4 e3 79 32 c0 07            	kshiftlb	$0x7, %k0, %k0
  3548ce: c4 e3 79 30 c0 07            	kshiftrb	$0x7, %k0, %k0
  3548d4: c4 c1 f9 6e c7               	vmovq	%r15, %xmm0
  3548d9: 62 d2 fd 28 7c c8            	vpbroadcastq	%r8, %ymm1
  3548df: c5 f5 eb 0d 79 c6 d4 ff      	vpor	-0x2b3987(%rip), %ymm1, %ymm1 # 0xa0f60 <anon.e6fb86fb181e0aaaed06fff69d344575.10.llvm.18170190781415527360+0xa0>
  3548e7: 49 29 c8                     	subq	%rcx, %r8
  3548ea: c4 e2 7d 59 15 2d 78 d4 ff   	vpbroadcastq	-0x2b87d3(%rip), %ymm2 # 0x9c120 <anon.4e66a2221d34052adc0f362485be06ac.633.llvm.27319815200592487+0x50>
  3548f3: c4 e2 7d 59 1d 6c 7e d4 ff   	vpbroadcastq	-0x2b8194(%rip), %ymm3 # 0x9c768 <anon.4d90d9de9367fb21f144d13924f001fb.148.llvm.1315488742485634063>
  3548fc: 0f 1f 40 00                  	nopl	(%rax)
  354900: c4 e1 f9 7e ca               	vmovq	%xmm1, %rdx
  354905: 4c 01 f2                     	addq	%r14, %rdx
  354908: c5 fe 6f 24 d3               	vmovdqu	(%rbx,%rdx,8), %ymm4
  35490d: c5 fe 6f 2c d7               	vmovdqu	(%rdi,%rdx,8), %ymm5
  354912: 62 f2 dd 28 29 ca            	vpcmpeqq	%ymm2, %ymm4, %k1
  354918: 62 f2 d5 28 29 d2            	vpcmpeqq	%ymm2, %ymm5, %k2
  35491e: c5 f4 45 ca                  	korw	%k2, %k1, %k1
  354922: c5 fc 45 c1                  	korw	%k1, %k0, %k0
  354926: c4 e2 55 37 e4               	vpcmpgtq	%ymm4, %ymm5, %ymm4
  35492b: c5 dd 73 d4 3f               	vpsrlq	$0x3f, %ymm4, %ymm4
  354930: c4 e2 dd 47 e1               	vpsllvq	%ymm1, %ymm4, %ymm4
  354935: c5 dd eb c0                  	vpor	%ymm0, %ymm4, %ymm0
  354939: c5 f5 d4 cb                  	vpaddq	%ymm3, %ymm1, %ymm1
  35493d: 49 83 c0 04                  	addq	$0x4, %r8
  354941: 75 bd                        	jne	0x354900 <vortex_buffer::bit::pack::collect_bool_words_avx512::<vortex_array::scalar_fn::unstable::row::execute::packed_bool::execute_bool_dense_attempt<(i64, i64), (), bool, true, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#1}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}>+0xf80>
  354943: c5 fb 93 d0                  	kmovd	%k0, %edx
  354947: f6 c2 0f                     	testb	$0xf, %dl
  35494a: 41 0f 95 c4                  	setne	%r12b
  35494e: c4 e3 7d 39 c1 01            	vextracti128	$0x1, %ymm0, %xmm1
  354954: c5 f9 eb c1                  	vpor	%xmm1, %xmm0, %xmm0
  354958: c5 f9 70 c8 ee               	vpshufd	$0xee, %xmm0, %xmm1     # xmm1 = xmm0[2,3,2,3]
  35495d: c5 f9 eb c1                  	vpor	%xmm1, %xmm0, %xmm0
  354961: c4 c1 f9 7e c7               	vmovq	%xmm0, %r15
  354966: 41 39 cb                     	cmpl	%ecx, %r11d
  354969: 74 49                        	je	0x3549b4 <vortex_buffer::bit::pack::collect_bool_words_avx512::<vortex_array::scalar_fn::unstable::row::execute::packed_bool::execute_bool_dense_attempt<(i64, i64), (), bool, true, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#1}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}>+0x1034>
  35496b: 48 89 f2                     	movq	%rsi, %rdx
  35496e: 48 c1 e2 09                  	shlq	$0x9, %rdx
  354972: 48 01 d7                     	addq	%rdx, %rdi
  354975: 48 01 d3                     	addq	%rdx, %rbx
  354978: 0f 1f 84 00 00 00 00 00      	nopl	(%rax,%rax)
  354980: 48 8b 14 cb                  	movq	(%rbx,%rcx,8), %rdx
  354984: 4c 8b 04 cf                  	movq	(%rdi,%rcx,8), %r8
  354988: 4c 39 d2                     	cmpq	%r10, %rdx
  35498b: 41 0f 94 c1                  	sete	%r9b
  35498f: 4d 39 d0                     	cmpq	%r10, %r8
  354992: 41 0f 94 c6                  	sete	%r14b
  354996: 45 08 ce                     	orb	%r9b, %r14b
  354999: 45 31 c9                     	xorl	%r9d, %r9d
  35499c: 4c 39 c2                     	cmpq	%r8, %rdx
  35499f: 41 0f 9c c1                  	setl	%r9b
  3549a3: 45 08 f4                     	orb	%r14b, %r12b
  3549a6: 49 d3 e1                     	shlq	%cl, %r9
  3549a9: 48 ff c1                     	incq	%rcx
  3549ac: 4d 09 cf                     	orq	%r9, %r15
  3549af: 49 39 cb                     	cmpq	%rcx, %r11
  3549b2: 75 cc                        	jne	0x354980 <vortex_buffer::bit::pack::collect_bool_words_avx512::<vortex_array::scalar_fn::unstable::row::execute::packed_bool::execute_bool_dense_attempt<(i64, i64), (), bool, true, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#1}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}>+0x1000>
  3549b4: 41 80 e4 01                  	andb	$0x1, %r12b
  3549b8: 44 88 60 38                  	movb	%r12b, 0x38(%rax)
  3549bc: 48 8b 45 c8                  	movq	-0x38(%rbp), %rax
  3549c0: 48 39 c6                     	cmpq	%rax, %rsi
  3549c3: 73 28                        	jae	0x3549ed <vortex_buffer::bit::pack::collect_bool_words_avx512::<vortex_array::scalar_fn::unstable::row::execute::packed_bool::execute_bool_dense_attempt<(i64, i64), (), bool, true, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}, <vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>> as vortex_array::scalar_fn::unstable::row::visitor::row_visitor::RowVisitor>::visit_deferred_bool<(i64, i64), bool, true, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#0}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#1}, <row_fn_bool_retry::Predicate<true> as vortex_array::scalar_fn::unstable::row::row_fn::RowFn>::dispatch<vortex_array::scalar_fn::unstable::row::visitor::execute::ExecuteRows<row_fn_bool_retry::Predicate<true>>>::{closure#1}>::{closure#0}>+0x106d>
  3549c5: 4d 89 7d 00                  	movq	%r15, (%r13)
  3549c9: 48 83 c4 18                  	addq	$0x18, %rsp
  3549cd: 5b                           	popq	%rbx
  3549ce: 41 5c                        	popq	%r12
  3549d0: 41 5d                        	popq	%r13
  3549d2: 41 5e                        	popq	%r14
  3549d4: 41 5f                        	popq	%r15
  3549d6: 5d                           	popq	%rbp
  3549d7: c5 f8 77                     	vzeroupper
  3549da: c3                           	retq
  3549db: 48 8d 0d 5e 8d f8 00         	leaq	0xf88d5e(%rip), %rcx    # 0x12dd740 <anon.d1268d7de3bb3fe7cb1b4d883edef8ba.6.llvm.7493331256730262183>
  3549e2: 31 ff                        	xorl	%edi, %edi
  3549e4: 4c 89 c2                     	movq	%r8, %rdx
  3549e7: ff 15 e3 54 fe 00            	callq	*0xfe54e3(%rip)         # 0x1339ed0 <writev+0x1339ed0>
  3549ed: 48 8d 15 34 8d f8 00         	leaq	0xf88d34(%rip), %rdx    # 0x12dd728 <anon.d1268d7de3bb3fe7cb1b4d883edef8ba.5.llvm.7493331256730262183>
  3549f4: 48 89 f7                     	movq	%rsi, %rdi
  3549f7: 48 89 c6                     	movq	%rax, %rsi
  3549fa: c5 f8 77                     	vzeroupper
  3549fd: ff 15 05 51 fe 00            	callq	*0xfe5105(%rip)         # 0x1339b08 <writev+0x1339b08>
