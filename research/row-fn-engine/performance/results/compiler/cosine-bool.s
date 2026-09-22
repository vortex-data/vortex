; SPDX-License-Identifier: Apache-2.0
; SPDX-FileCopyrightText: Copyright the Vortex contributors
__RINvNtNtNtNtNtCs5mvLrSLkDP1_12vortex_array9scalar_fn8unstable3row7execute5retry27execute_owned_dense_attemptTxxEbubNCINvYINtNtNtB6_7visitor5retry21ExecuteDenseWithRetryINtCsaHo96IcALPt_24row_fn_performance_probe9PredicateKb0_EENtNtB20_11row_visitor10RowVisitor19visit_deferred_boolB1I_bKB3y_NCINvXs_B2J_B2G_NtNtB6_6row_fn5RowFn8dispatchB1V_E0NCB4H_s_0E0NCB1R_s_0B5u_EB2J_:
Lfunc_begin6:
	.cfi_startproc
	.cfi_personality 155, _rust_eh_personality
	.cfi_lsda 16, Lexception6
	sub	sp, sp, #304
	.cfi_def_cfa_offset 304
	stp	x28, x27, [sp, #240]
	stp	x22, x21, [sp, #256]
	stp	x20, x19, [sp, #272]
	stp	x29, x30, [sp, #288]
	add	x29, sp, #288
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_offset w19, -24
	.cfi_offset w20, -32
	.cfi_offset w21, -40
	.cfi_offset w22, -48
	.cfi_offset w27, -56
	.cfi_offset w28, -64
	.cfi_remember_state
	mov	x21, x1
	mov	x20, x0
	mov	x19, x8
	sub	x22, x29, #128
	sub	x8, x29, #128
	bl	__RNvXs4_NtNtNtNtNtNtNtCs5mvLrSLkDP1_12vortex_array9scalar_fn8unstable3row5types7element5tuple13element_tupleTxxENtB5_12ElementTuple6decodeCsaHo96IcALPt_24row_fn_performance_probe
	ldur	x8, [x29, #-128]
	ldur	q0, [x22, #8]
	ldur	q1, [x22, #24]
	stp	q0, q1, [sp, #64]
	ldur	q0, [x22, #40]
	ldur	q1, [x22, #56]
	stp	q0, q1, [sp, #96]
	cmn	x8, #1
	b.eq	LBB12_3
	ldur	x9, [x29, #-56]
	ldp	q0, q1, [sp, #64]
	stp	q0, q1, [x19, #16]
	ldp	q0, q1, [sp, #96]
	stp	q0, q1, [x19, #48]
	str	x9, [x19, #80]
	mov	w9, #1
	stp	x9, x8, [x19]
LBB12_2:
	.cfi_def_cfa wsp, 304
	ldp	x29, x30, [sp, #288]
	ldp	x20, x19, [sp, #272]
	ldp	x22, x21, [sp, #256]
	ldp	x28, x27, [sp, #240]
	add	sp, sp, #304
	.cfi_def_cfa_offset 0
	.cfi_restore w30
	.cfi_restore w29
	.cfi_restore w19
	.cfi_restore w20
	.cfi_restore w21
	.cfi_restore w22
	.cfi_restore w27
	.cfi_restore w28
	ret
LBB12_3:
	.cfi_restore_state
	ldp	q0, q1, [sp, #64]
	stp	q0, q1, [sp]
	ldp	q0, q1, [sp, #96]
	stp	q0, q1, [sp, #32]
	ldr	x8, [x21, #40]
Ltmp94:
	mov	x0, x20
	blr	x8
Ltmp95:
	mov	x20, x0
	str	x0, [sp, #136]
	tbz	x0, #63, LBB12_7
	mov	x22, #0
LBB12_6:
Ltmp116:
	mov	x0, x22
	mov	x1, x20
	bl	__RNvNtCs6KVRSXc8uZF_5alloc7raw_vec12handle_error
Ltmp117:
	b	LBB12_65
LBB12_7:
	cbz	x20, LBB12_19
	bl	__RNvCs6rREvFdRhLb_7___rustc35___rust_no_alloc_shim_is_unstable_v2
	mov	w22, #1
	mov	x0, x20
	mov	w1, #1
	bl	_mi_malloc_aligned
	cbz	x0, LBB12_6
	mov	x21, x0
	ldr	x8, [sp]
	cbz	x8, LBB12_20
LBB12_10:
	ldr	x9, [sp, #32]
	ldr	x10, [sp, #8]
	cmp	x10, x20
	cbz	x9, LBB12_23
	b.ne	LBB12_60
	ldr	x10, [sp, #40]
	cmp	x10, x20
	b.ne	LBB12_60
	cbz	x20, LBB12_38
	cmp	x20, #3
	b.hi	LBB12_49
	mov	x10, #0
	mov	w12, #0
LBB12_16:
	mov	x11, #9223372036854775807
LBB12_17:
	ldr	x13, [x8, x10, lsl #3]
	ldr	x14, [x9, x10, lsl #3]
	cmp	x13, x11
	ccmp	x14, x11, #4, ne
	cset	w15, eq
	cmp	x13, x14
	cset	w13, lt
	strb	w13, [x21, x10]
	add	x10, x10, #1
	orr	w12, w12, w15
	cmp	x20, x10
	b.ne	LBB12_17
LBB12_18:
	ldr	x22, [sp, #136]
	tbz	w12, #0, LBB12_39
	b	LBB12_34
LBB12_19:
	mov	w21, #1
	ldr	x8, [sp]
	cbnz	x8, LBB12_10
LBB12_20:
	ldr	x8, [sp, #32]
	cbz	x8, LBB12_24
	ldr	x8, [sp, #40]
	cmp	x8, x20
	b.eq	LBB12_24
LBB12_22:
	add	x8, sp, #136
Lloh52:
	adrp	x9, __RNvXsi_NtNtNtCslWxY2MhVcag_4core3fmt3num3impjNtB9_7Display3fmt@GOTPAGE
Lloh53:
	ldr	x9, [x9, __RNvXsi_NtNtNtCslWxY2MhVcag_4core3fmt3num3impjNtB9_7Display3fmt@GOTPAGEOFF]
	stp	x8, x9, [sp, #64]
Lloh54:
	adrp	x8, l_alloc_8cb1b9e1c7b448ff8aa56230a8661784@PAGE
Lloh55:
	add	x8, x8, l_alloc_8cb1b9e1c7b448ff8aa56230a8661784@PAGEOFF
	add	x9, sp, #64
	stp	x8, x9, [sp, #144]
Ltmp99:
Lloh56:
	adrp	x0, __RNcNtNtCs4jPh2r5lbWM_12vortex_error11VortexError5Other0@PAGE
Lloh57:
	add	x0, x0, __RNcNtNtCs4jPh2r5lbWM_12vortex_error11VortexError5Other0@PAGEOFF
	sub	x8, x29, #128
	add	x1, sp, #144
	bl	__RNvNtCs4jPh2r5lbWM_12vortex_error9___private7fmt_err
Ltmp100:
	b	LBB12_61
LBB12_23:
	b.ne	LBB12_22
LBB12_24:
	cbz	x20, LBB12_38
	mov	w8, #0
	mov	x0, #0
	mov	x9, #9223372036854775807
	b	LBB12_27
LBB12_26:
	cmp	x10, x9
	ccmp	x1, x9, #4, ne
	cset	w11, eq
	cmp	x10, x1
	cset	w10, lt
	strb	w10, [x21, x0]
	add	x0, x0, #1
	orr	w8, w8, w11
	cmp	x20, x0
	b.eq	LBB12_33
LBB12_27:
	ldp	x10, x1, [sp]
	cbz	x10, LBB12_30
	cmp	x0, x1
	b.hs	LBB12_64
	ldr	x10, [x10, x0, lsl #3]
	ldp	x11, x1, [sp, #32]
	cbnz	x11, LBB12_31
	b	LBB12_26
LBB12_30:
	mov	x10, x1
	ldp	x11, x1, [sp, #32]
	cbz	x11, LBB12_26
LBB12_31:
	cmp	x0, x1
	b.hs	LBB12_64
	ldr	x1, [x11, x0, lsl #3]
	b	LBB12_26
LBB12_33:
	ldr	x22, [sp, #136]
	tbz	w8, #0, LBB12_39
LBB12_34:
Lloh58:
	adrp	x8, l_alloc_78c2b751c4d3e2d7197d387cac5cfac2@PAGE
Lloh59:
	add	x8, x8, l_alloc_78c2b751c4d3e2d7197d387cac5cfac2@PAGEOFF
	mov	w9, #55
	stp	x8, x9, [sp, #64]
Ltmp104:
Lloh60:
	adrp	x0, __RNcNtNtCs4jPh2r5lbWM_12vortex_error11VortexError5Other0@PAGE
Lloh61:
	add	x0, x0, __RNcNtNtCs4jPh2r5lbWM_12vortex_error11VortexError5Other0@PAGEOFF
	sub	x8, x29, #128
	add	x1, sp, #64
	bl	__RNvNtCs4jPh2r5lbWM_12vortex_error9___private7fmt_err
Ltmp105:
	ldur	x8, [x29, #-128]
	cmn	x8, #1
	b.eq	LBB12_39
	ldp	q1, q0, [x29, #-128]
	stur	q0, [x19, #24]
	ldp	q0, q2, [x29, #-96]
	stur	q0, [x19, #40]
	stur	q2, [x19, #56]
	ldur	q0, [x29, #-64]
	stur	q0, [x19, #72]
	stur	q1, [x19, #8]
	str	xzr, [x19]
	cbz	x20, LBB12_41
	mov	x0, x21
	bl	_mi_free
	b	LBB12_41
LBB12_38:
	mov	x22, #0
LBB12_39:
	stp	x20, x21, [sp, #64]
	str	x22, [sp, #80]
Ltmp107:
	add	x0, sp, #64
	bl	__RNvXs_NtNtNtNtNtNtCs5mvLrSLkDP1_12vortex_array9scalar_fn8unstable3row5types7element4boolbNtNtB6_6output13OutputElement5build
Ltmp108:
	stp	x0, x1, [x19, #16]
Lloh62:
	adrp	x8, lCPI12_0@PAGE
Lloh63:
	ldr	q0, [x8, lCPI12_0@PAGEOFF]
	str	q0, [x19]
LBB12_41:
	ldr	x8, [sp]
	cbz	x8, LBB12_45
	ldr	x8, [sp, #16]
	cbz	x8, LBB12_45
	mov	x9, #-1
	ldaddl	x9, x8, [x8]
	cmp	x8, #1
	b.ne	LBB12_45
	mov	x8, sp
	dmb	ishld
Ltmp110:
	add	x0, x8, #16
	bl	__RNvMsn_NtCs6KVRSXc8uZF_5alloc4syncINtB5_3ArcNtNtCsEOnbuuTlDO_13vortex_buffer10allocation13BufferBackingE9drop_slowBK_
Ltmp111:
LBB12_45:
	ldr	x8, [sp, #32]
	cbz	x8, LBB12_2
	ldr	x8, [sp, #48]
	cbz	x8, LBB12_2
	mov	x9, #-1
	ldaddl	x9, x8, [x8]
	cmp	x8, #1
	b.ne	LBB12_2
	mov	x8, sp
	dmb	ishld
	add	x0, x8, #48
	bl	__RNvMsn_NtCs6KVRSXc8uZF_5alloc4syncINtB5_3ArcNtNtCsEOnbuuTlDO_13vortex_buffer10allocation13BufferBackingE9drop_slowBK_
	b	LBB12_2
LBB12_49:
	mov	w12, #0
	mov	x10, #0
	add	x13, x21, x20
	lsl	x11, x20, #3
	add	x14, x8, x11
	add	x11, x9, x11
	cmp	x11, x21
	ccmp	x9, x13, #2, hi
	cset	w11, lo
	cmp	x8, x13
	ccmp	x14, x21, #0, lo
	b.hi	LBB12_16
	tbnz	w11, #0, LBB12_16
	cmp	x20, #16
	b.hs	LBB12_53
	mov	x10, #0
	mov	w12, #0
	b	LBB12_57
LBB12_53:
	and	x11, x20, #0xc
	and	x10, x20, #0x7ffffffffffffff0
	movi.2d	v0, #0000000000000000
	movi.2d	v1, #0xffffffffffffffff
	fneg.2d	v1, v1
	movi.16b	v2, #1
	mov	x12, x8
	mov	x13, x9
	mov	x14, x21
	and	x15, x20, #0x7ffffffffffffff0
LBB12_54:
	ldp	q3, q4, [x12]
	ldp	q5, q6, [x12, #32]
	ldp	q7, q16, [x12, #64]
	ldp	q17, q18, [x12, #96]
	ldp	q19, q20, [x13]
	ldp	q21, q22, [x13, #32]
	ldp	q23, q24, [x13, #64]
	cmeq.2d	v25, v18, v1
	cmeq.2d	v26, v17, v1
	uzp1.4s	v25, v26, v25
	ldp	q26, q27, [x13, #96]
	cmeq.2d	v28, v16, v1
	cmeq.2d	v29, v7, v1
	uzp1.4s	v28, v29, v28
	uzp1.8h	v25, v28, v25
	cmeq.2d	v28, v6, v1
	cmeq.2d	v29, v5, v1
	uzp1.4s	v28, v29, v28
	cmeq.2d	v29, v4, v1
	cmeq.2d	v30, v3, v1
	uzp1.4s	v29, v30, v29
	uzp1.8h	v28, v29, v28
	uzp1.16b	v25, v28, v25
	cmeq.2d	v28, v27, v1
	cmeq.2d	v29, v26, v1
	uzp1.4s	v28, v29, v28
	cmeq.2d	v29, v24, v1
	cmeq.2d	v30, v23, v1
	uzp1.4s	v29, v30, v29
	uzp1.8h	v28, v29, v28
	cmeq.2d	v29, v22, v1
	cmeq.2d	v30, v21, v1
	uzp1.4s	v29, v30, v29
	cmeq.2d	v30, v20, v1
	cmeq.2d	v31, v19, v1
	uzp1.4s	v30, v31, v30
	uzp1.8h	v29, v30, v29
	uzp1.16b	v28, v29, v28
	orr.16b	v25, v25, v28
	orr.16b	v0, v0, v25
	cmgt.2d	v18, v27, v18
	cmgt.2d	v17, v26, v17
	uzp1.4s	v17, v17, v18
	cmgt.2d	v16, v24, v16
	cmgt.2d	v7, v23, v7
	uzp1.4s	v7, v7, v16
	uzp1.8h	v7, v7, v17
	cmgt.2d	v6, v22, v6
	cmgt.2d	v5, v21, v5
	uzp1.4s	v5, v5, v6
	cmgt.2d	v4, v20, v4
	cmgt.2d	v3, v19, v3
	uzp1.4s	v3, v3, v4
	uzp1.8h	v3, v3, v5
	uzp1.16b	v3, v3, v7
	and.16b	v3, v3, v2
	str	q3, [x14], #16
	add	x13, x13, #128
	add	x12, x12, #128
	subs	x15, x15, #16
	b.ne	LBB12_54
	shl.16b	v0, v0, #7
	cmlt.16b	v0, v0, #0
	umaxv.16b	b0, v0
	fmov	w12, s0
	and	w12, w12, #0x1
	cmp	x20, x10
	b.eq	LBB12_18
	cbz	x11, LBB12_16
LBB12_57:
	mov	x13, x10
	and	x10, x20, #0x7ffffffffffffffc
	movi.2d	v0, #0000000000000000
	mov.h	v0[0], w12
	sub	x11, x13, x10
	add	x12, x21, x13
	lsl	x14, x13, #3
	add	x13, x9, x14
	add	x14, x8, x14
	movi.2d	v1, #0xffffffffffffffff
	fneg.2d	v1, v1
	movi.4h	v2, #1
LBB12_58:
	ldp	q3, q4, [x14], #32
	ldp	q5, q6, [x13], #32
	cmeq.2d	v7, v4, v1
	cmeq.2d	v16, v3, v1
	uzp1.4s	v7, v16, v7
	cmeq.2d	v16, v6, v1
	cmeq.2d	v17, v5, v1
	uzp1.4s	v16, v17, v16
	orr.16b	v7, v7, v16
	xtn.4h	v7, v7
	orr.8b	v0, v0, v7
	cmgt.2d	v4, v6, v4
	cmgt.2d	v3, v5, v3
	uzp1.4s	v3, v3, v4
	xtn.4h	v3, v3
	and.8b	v3, v3, v2
	uzp1.8b	v3, v3, v0
	st1.s	{ v3 }[0], [x12], #4
	adds	x11, x11, #4
	b.ne	LBB12_58
	shl.4h	v0, v0, #15
	cmlt.4h	v0, v0, #0
	umaxv.4h	h0, v0
	fmov	w11, s0
	and	w12, w11, #0x1
	cmp	x20, x10
	b.ne	LBB12_16
	b	LBB12_18
LBB12_60:
	add	x8, sp, #136
Lloh64:
	adrp	x9, __RNvXsi_NtNtNtCslWxY2MhVcag_4core3fmt3num3impjNtB9_7Display3fmt@GOTPAGE
Lloh65:
	ldr	x9, [x9, __RNvXsi_NtNtNtCslWxY2MhVcag_4core3fmt3num3impjNtB9_7Display3fmt@GOTPAGEOFF]
	stp	x8, x9, [sp, #64]
Lloh66:
	adrp	x8, l_alloc_8cb1b9e1c7b448ff8aa56230a8661784@PAGE
Lloh67:
	add	x8, x8, l_alloc_8cb1b9e1c7b448ff8aa56230a8661784@PAGEOFF
	add	x9, sp, #64
	stp	x8, x9, [sp, #144]
Ltmp96:
Lloh68:
	adrp	x0, __RNcNtNtCs4jPh2r5lbWM_12vortex_error11VortexError5Other0@PAGE
Lloh69:
	add	x0, x0, __RNcNtNtCs4jPh2r5lbWM_12vortex_error11VortexError5Other0@PAGEOFF
	sub	x8, x29, #128
	add	x1, sp, #144
	bl	__RNvNtCs4jPh2r5lbWM_12vortex_error9___private7fmt_err
Ltmp97:
LBB12_61:
	ldp	q1, q0, [x29, #-128]
	stur	q0, [x19, #24]
	ldp	q0, q2, [x29, #-96]
	stur	q0, [x19, #40]
	stur	q2, [x19, #56]
	ldur	q0, [x29, #-64]
	stur	q0, [x19, #72]
	stur	q1, [x19, #8]
	mov	w8, #1
	str	x8, [x19]
	cbz	x20, LBB12_63
	mov	x0, x21
	bl	_mi_free
LBB12_63:
	mov	x0, sp
	bl	__RINvNtCslWxY2MhVcag_4core3ptr9drop_glueTINtNtNtNtNtNtNtNtCs5mvLrSLkDP1_12vortex_array9scalar_fn8unstable3row5types7element5tuple13element_tuple9ArgColumnINtCsaHo96IcALPt_24row_fn_performance_probe11SelectedI64Kb0_EEBC_EEB2u_
	b	LBB12_2
LBB12_64:
Ltmp101:
Lloh70:
	adrp	x2, _alloc_ef228a7a1610e5e546147c56beb2165c.llvm.3148866541286181846@PAGE
Lloh71:
	add	x2, x2, _alloc_ef228a7a1610e5e546147c56beb2165c.llvm.3148866541286181846@PAGEOFF
	bl	__RNvNtCslWxY2MhVcag_4core9panicking18panic_bounds_check
Ltmp102:
LBB12_65:
	brk	#0x1
LBB12_66:
Ltmp98:
	b	LBB12_72
LBB12_67:
Ltmp106:
	b	LBB12_72
LBB12_68:
Ltmp112:
	mov	x19, x0
	mov	x8, sp
Ltmp113:
	add	x0, x8, #32
	bl	__RINvNtCslWxY2MhVcag_4core3ptr9drop_glueINtNtNtNtNtNtNtNtCs5mvLrSLkDP1_12vortex_array9scalar_fn8unstable3row5types7element5tuple13element_tuple9ArgColumnINtCsaHo96IcALPt_24row_fn_performance_probe11SelectedI64Kb0_EEEB2t_
Ltmp114:
	b	LBB12_77
LBB12_69:
Ltmp115:
	bl	__RNvNtCslWxY2MhVcag_4core9panicking16panic_in_cleanup
LBB12_70:
Ltmp109:
	b	LBB12_75
LBB12_71:
Ltmp103:
LBB12_72:
	mov	x19, x0
	cbz	x20, LBB12_76
	mov	x0, x21
	bl	_mi_free
	b	LBB12_76
LBB12_74:
Ltmp118:
LBB12_75:
	mov	x19, x0
LBB12_76:
Ltmp119:
	mov	x0, sp
	bl	__RINvNtCslWxY2MhVcag_4core3ptr9drop_glueTINtNtNtNtNtNtNtNtCs5mvLrSLkDP1_12vortex_array9scalar_fn8unstable3row5types7element5tuple13element_tuple9ArgColumnINtCsaHo96IcALPt_24row_fn_performance_probe11SelectedI64Kb0_EEBC_EEB2u_
Ltmp120:
LBB12_77:
	mov	x0, x19
	bl	__Unwind_Resume
LBB12_78:
Ltmp121:
	bl	__RNvNtCslWxY2MhVcag_4core9panicking16panic_in_cleanup
	.loh AdrpAdd	Lloh56, Lloh57
	.loh AdrpAdd	Lloh54, Lloh55
	.loh AdrpLdrGot	Lloh52, Lloh53
	.loh AdrpAdd	Lloh60, Lloh61
	.loh AdrpAdd	Lloh58, Lloh59
	.loh AdrpLdr	Lloh62, Lloh63
	.loh AdrpAdd	Lloh68, Lloh69
	.loh AdrpAdd	Lloh66, Lloh67
	.loh AdrpLdrGot	Lloh64, Lloh65
	.loh AdrpAdd	Lloh70, Lloh71
Lfunc_end6:
	.cfi_endproc
