__RINvNtNtNtNtNtCscIinMv7ig43_12vortex_array9scalar_fn8unstable3row7execute5retry27execute_owned_dense_attemptTxxEbubNCINvYINtNtNtB6_7visitor5retry21ExecuteDenseWithRetryINtCsls3KbOWy6Nu_17row_fn_bool_probe9PredicateKb0_EENtNtB20_11row_visitor10RowVisitor19visit_deferred_boolB1I_bKB3r_NCINvXB2J_B2G_NtNtB6_6row_fn5RowFn8dispatchB1V_E0NCB4A_s_0E0NCB1R_s_0B5l_EB2J_:
Lfunc_begin1:
	.cfi_startproc
	.cfi_personality 155, _rust_eh_personality
	.cfi_lsda 16, Lexception1
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
	bl	__RNvXs4_NtNtNtNtNtNtNtCscIinMv7ig43_12vortex_array9scalar_fn8unstable3row5types7element5tuple13element_tupleTxxENtB5_12ElementTuple6decodeCsls3KbOWy6Nu_17row_fn_bool_probe
	ldur	x8, [x29, #-128]
	ldur	q0, [x22, #8]
	ldur	q1, [x22, #24]
	stp	q0, q1, [sp, #64]
	ldur	q0, [x22, #40]
	ldur	q1, [x22, #56]
	stp	q0, q1, [sp, #96]
	cmn	x8, #1
	b.eq	LBB2_3
	ldur	x9, [x29, #-56]
	ldp	q0, q1, [sp, #64]
	stp	q0, q1, [x19, #16]
	ldp	q0, q1, [sp, #96]
	stp	q0, q1, [x19, #48]
	str	x9, [x19, #80]
	mov	w9, #1
	stp	x9, x8, [x19]
LBB2_2:
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
LBB2_3:
	.cfi_restore_state
	ldp	q0, q1, [sp, #64]
	stp	q0, q1, [sp]
	ldp	q0, q1, [sp, #96]
	stp	q0, q1, [sp, #32]
	ldr	x8, [x21, #40]
Ltmp6:
	mov	x0, x20
	blr	x8
Ltmp7:
	mov	x20, x0
	str	x0, [sp, #136]
	tbz	x0, #63, LBB2_7
	mov	x22, #0
LBB2_6:
Ltmp28:
	mov	x0, x22
	mov	x1, x20
	bl	__RNvNtCs6KVRSXc8uZF_5alloc7raw_vec12handle_error
Ltmp29:
	b	LBB2_65
LBB2_7:
	cbz	x20, LBB2_19
	bl	__RNvCs6rREvFdRhLb_7___rustc35___rust_no_alloc_shim_is_unstable_v2
	mov	w22, #1
	mov	x0, x20
	mov	w1, #1
	bl	_mi_malloc_aligned
	cbz	x0, LBB2_6
	mov	x21, x0
	ldr	x8, [sp]
	cbz	x8, LBB2_20
LBB2_10:
	ldr	x9, [sp, #32]
	ldr	x10, [sp, #8]
	cmp	x10, x20
	cbz	x9, LBB2_23
	b.ne	LBB2_60
	ldr	x10, [sp, #40]
	cmp	x10, x20
	b.ne	LBB2_60
	cbz	x20, LBB2_38
	cmp	x20, #3
	b.hi	LBB2_49
	mov	x10, #0
	mov	w12, #0
LBB2_16:
	mov	x11, #9223372036854775807
LBB2_17:
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
	b.ne	LBB2_17
LBB2_18:
	ldr	x22, [sp, #136]
	tbz	w12, #0, LBB2_39
	b	LBB2_34
LBB2_19:
	mov	w21, #1
	ldr	x8, [sp]
	cbnz	x8, LBB2_10
LBB2_20:
	ldr	x8, [sp, #32]
	cbz	x8, LBB2_24
	ldr	x8, [sp, #40]
	cmp	x8, x20
	b.eq	LBB2_24
LBB2_22:
	add	x8, sp, #136
Lloh0:
	adrp	x9, __RNvXsi_NtNtNtCslWxY2MhVcag_4core3fmt3num3impjNtB9_7Display3fmt@GOTPAGE
Lloh1:
	ldr	x9, [x9, __RNvXsi_NtNtNtCslWxY2MhVcag_4core3fmt3num3impjNtB9_7Display3fmt@GOTPAGEOFF]
	stp	x8, x9, [sp, #64]
Lloh2:
	adrp	x8, l_alloc_8cb1b9e1c7b448ff8aa56230a8661784@PAGE
Lloh3:
	add	x8, x8, l_alloc_8cb1b9e1c7b448ff8aa56230a8661784@PAGEOFF
	add	x9, sp, #64
	stp	x8, x9, [sp, #144]
Ltmp11:
Lloh4:
	adrp	x0, __RNcNtNtCsfoiiTnO3OQ6_12vortex_error11VortexError5Other0.llvm.8602018377278662986@PAGE
Lloh5:
	add	x0, x0, __RNcNtNtCsfoiiTnO3OQ6_12vortex_error11VortexError5Other0.llvm.8602018377278662986@PAGEOFF
	sub	x8, x29, #128
	add	x1, sp, #144
	bl	__RNvNtCsfoiiTnO3OQ6_12vortex_error9___private7fmt_err
Ltmp12:
	b	LBB2_61
LBB2_23:
	b.ne	LBB2_22
LBB2_24:
	cbz	x20, LBB2_38
	mov	w8, #0
	mov	x0, #0
	mov	x9, #9223372036854775807
	b	LBB2_27
LBB2_26:
	cmp	x10, x9
	ccmp	x1, x9, #4, ne
	cset	w11, eq
	cmp	x10, x1
	cset	w10, lt
	strb	w10, [x21, x0]
	add	x0, x0, #1
	orr	w8, w8, w11
	cmp	x20, x0
	b.eq	LBB2_33
LBB2_27:
	ldp	x10, x1, [sp]
	cbz	x10, LBB2_30
	cmp	x0, x1
	b.hs	LBB2_64
	ldr	x10, [x10, x0, lsl #3]
	ldp	x11, x1, [sp, #32]
	cbnz	x11, LBB2_31
	b	LBB2_26
LBB2_30:
	mov	x10, x1
	ldp	x11, x1, [sp, #32]
	cbz	x11, LBB2_26
LBB2_31:
	cmp	x0, x1
	b.hs	LBB2_64
	ldr	x1, [x11, x0, lsl #3]
	b	LBB2_26
LBB2_33:
	ldr	x22, [sp, #136]
	tbz	w8, #0, LBB2_39
LBB2_34:
Lloh6:
	adrp	x8, l_alloc_78c2b751c4d3e2d7197d387cac5cfac2@PAGE
Lloh7:
	add	x8, x8, l_alloc_78c2b751c4d3e2d7197d387cac5cfac2@PAGEOFF
	mov	w9, #55
	stp	x8, x9, [sp, #64]
Ltmp16:
Lloh8:
	adrp	x0, __RNcNtNtCsfoiiTnO3OQ6_12vortex_error11VortexError5Other0.llvm.8602018377278662986@PAGE
Lloh9:
	add	x0, x0, __RNcNtNtCsfoiiTnO3OQ6_12vortex_error11VortexError5Other0.llvm.8602018377278662986@PAGEOFF
	sub	x8, x29, #128
	add	x1, sp, #64
	bl	__RNvNtCsfoiiTnO3OQ6_12vortex_error9___private7fmt_err
Ltmp17:
	ldur	x8, [x29, #-128]
	cmn	x8, #1
	b.eq	LBB2_39
	ldp	q1, q0, [x29, #-128]
	stur	q0, [x19, #24]
	ldp	q0, q2, [x29, #-96]
	stur	q0, [x19, #40]
	stur	q2, [x19, #56]
	ldur	q0, [x29, #-64]
	stur	q0, [x19, #72]
	stur	q1, [x19, #8]
	str	xzr, [x19]
	cbz	x20, LBB2_41
	mov	x0, x21
	bl	_mi_free
	b	LBB2_41
LBB2_38:
	mov	x22, #0
LBB2_39:
	stp	x20, x21, [sp, #64]
	str	x22, [sp, #80]
Ltmp19:
	add	x0, sp, #64
	bl	__RNvXs_NtNtNtNtNtNtCscIinMv7ig43_12vortex_array9scalar_fn8unstable3row5types7element4boolbNtNtB6_6output13OutputElement5build
Ltmp20:
	stp	x0, x1, [x19, #16]
Lloh10:
	adrp	x8, lCPI2_0@PAGE
Lloh11:
	ldr	q0, [x8, lCPI2_0@PAGEOFF]
	str	q0, [x19]
LBB2_41:
	ldr	x8, [sp]
	cbz	x8, LBB2_45
	ldr	x8, [sp, #16]
	cbz	x8, LBB2_45
	mov	x9, #-1
	ldaddl	x9, x8, [x8]
	cmp	x8, #1
	b.ne	LBB2_45
	mov	x8, sp
	dmb	ishld
Ltmp22:
	add	x0, x8, #16
	bl	__RNvMsn_NtCs6KVRSXc8uZF_5alloc4syncINtB5_3ArcNtNtCs2glbd03i89i_13vortex_buffer10allocation13BufferBackingE9drop_slowBK_
Ltmp23:
LBB2_45:
	ldr	x8, [sp, #32]
	cbz	x8, LBB2_2
	ldr	x8, [sp, #48]
	cbz	x8, LBB2_2
	mov	x9, #-1
	ldaddl	x9, x8, [x8]
	cmp	x8, #1
	b.ne	LBB2_2
	mov	x8, sp
	dmb	ishld
	add	x0, x8, #48
	bl	__RNvMsn_NtCs6KVRSXc8uZF_5alloc4syncINtB5_3ArcNtNtCs2glbd03i89i_13vortex_buffer10allocation13BufferBackingE9drop_slowBK_
	b	LBB2_2
LBB2_49:
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
	b.hi	LBB2_16
	tbnz	w11, #0, LBB2_16
	cmp	x20, #16
	b.hs	LBB2_53
	mov	x10, #0
	mov	w12, #0
	b	LBB2_57
LBB2_53:
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
LBB2_54:
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
	b.ne	LBB2_54
	shl.16b	v0, v0, #7
	cmlt.16b	v0, v0, #0
	umaxv.16b	b0, v0
	fmov	w12, s0
	and	w12, w12, #0x1
	cmp	x20, x10
	b.eq	LBB2_18
	cbz	x11, LBB2_16
LBB2_57:
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
LBB2_58:
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
	b.ne	LBB2_58
	shl.4h	v0, v0, #15
	cmlt.4h	v0, v0, #0
	umaxv.4h	h0, v0
	fmov	w11, s0
	and	w12, w11, #0x1
	cmp	x20, x10
	b.ne	LBB2_16
	b	LBB2_18
LBB2_60:
	add	x8, sp, #136
Lloh12:
	adrp	x9, __RNvXsi_NtNtNtCslWxY2MhVcag_4core3fmt3num3impjNtB9_7Display3fmt@GOTPAGE
Lloh13:
	ldr	x9, [x9, __RNvXsi_NtNtNtCslWxY2MhVcag_4core3fmt3num3impjNtB9_7Display3fmt@GOTPAGEOFF]
	stp	x8, x9, [sp, #64]
Lloh14:
	adrp	x8, l_alloc_8cb1b9e1c7b448ff8aa56230a8661784@PAGE
Lloh15:
	add	x8, x8, l_alloc_8cb1b9e1c7b448ff8aa56230a8661784@PAGEOFF
	add	x9, sp, #64
	stp	x8, x9, [sp, #144]
Ltmp8:
Lloh16:
	adrp	x0, __RNcNtNtCsfoiiTnO3OQ6_12vortex_error11VortexError5Other0.llvm.8602018377278662986@PAGE
Lloh17:
	add	x0, x0, __RNcNtNtCsfoiiTnO3OQ6_12vortex_error11VortexError5Other0.llvm.8602018377278662986@PAGEOFF
	sub	x8, x29, #128
	add	x1, sp, #144
	bl	__RNvNtCsfoiiTnO3OQ6_12vortex_error9___private7fmt_err
Ltmp9:
LBB2_61:
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
	cbz	x20, LBB2_63
	mov	x0, x21
	bl	_mi_free
LBB2_63:
	mov	x0, sp
	bl	__RINvNtCslWxY2MhVcag_4core3ptr9drop_glueTINtNtNtNtNtNtNtNtCscIinMv7ig43_12vortex_array9scalar_fn8unstable3row5types7element5tuple13element_tuple9ArgColumnxEBC_EECsls3KbOWy6Nu_17row_fn_bool_probe
	b	LBB2_2
LBB2_64:
Ltmp13:
Lloh18:
	adrp	x2, _alloc_ef228a7a1610e5e546147c56beb2165c.llvm.1308957373746945908@PAGE
Lloh19:
	add	x2, x2, _alloc_ef228a7a1610e5e546147c56beb2165c.llvm.1308957373746945908@PAGEOFF
	bl	__RNvNtCslWxY2MhVcag_4core9panicking18panic_bounds_check
Ltmp14:
LBB2_65:
	brk	#0x1
LBB2_66:
Ltmp10:
	b	LBB2_72
LBB2_67:
Ltmp18:
	b	LBB2_72
LBB2_68:
Ltmp24:
	mov	x19, x0
	mov	x8, sp
Ltmp25:
	add	x0, x8, #32
	bl	__RINvNtCslWxY2MhVcag_4core3ptr9drop_glueINtNtNtNtNtNtNtNtCscIinMv7ig43_12vortex_array9scalar_fn8unstable3row5types7element5tuple13element_tuple9ArgColumnxEECsls3KbOWy6Nu_17row_fn_bool_probe
Ltmp26:
	b	LBB2_77
LBB2_69:
Ltmp27:
	bl	__RNvNtCslWxY2MhVcag_4core9panicking16panic_in_cleanup
LBB2_70:
Ltmp21:
	b	LBB2_75
LBB2_71:
Ltmp15:
LBB2_72:
	mov	x19, x0
	cbz	x20, LBB2_76
	mov	x0, x21
	bl	_mi_free
	b	LBB2_76
LBB2_74:
Ltmp30:
LBB2_75:
	mov	x19, x0
LBB2_76:
Ltmp31:
	mov	x0, sp
	bl	__RINvNtCslWxY2MhVcag_4core3ptr9drop_glueTINtNtNtNtNtNtNtNtCscIinMv7ig43_12vortex_array9scalar_fn8unstable3row5types7element5tuple13element_tuple9ArgColumnxEBC_EECsls3KbOWy6Nu_17row_fn_bool_probe
Ltmp32:
LBB2_77:
	mov	x0, x19
	bl	__Unwind_Resume
LBB2_78:
Ltmp33:
	bl	__RNvNtCslWxY2MhVcag_4core9panicking16panic_in_cleanup
	.loh AdrpAdd	Lloh4, Lloh5
	.loh AdrpAdd	Lloh2, Lloh3
	.loh AdrpLdrGot	Lloh0, Lloh1
	.loh AdrpAdd	Lloh8, Lloh9
	.loh AdrpAdd	Lloh6, Lloh7
	.loh AdrpLdr	Lloh10, Lloh11
	.loh AdrpAdd	Lloh16, Lloh17
	.loh AdrpAdd	Lloh14, Lloh15
	.loh AdrpLdrGot	Lloh12, Lloh13
	.loh AdrpAdd	Lloh18, Lloh19
Lfunc_end1:
	.cfi_endproc
