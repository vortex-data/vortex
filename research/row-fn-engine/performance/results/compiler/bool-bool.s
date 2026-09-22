; SPDX-License-Identifier: Apache-2.0
; SPDX-FileCopyrightText: Copyright the Vortex contributors
__RINvNtNtNtNtNtCs5mvLrSLkDP1_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb0_NCINvYINtNtNtB6_7visitor5retry21ExecuteDenseWithRetryINtCsaHo96IcALPt_24row_fn_performance_probe9PredicateKB1V_EENtNtB29_11row_visitor10RowVisitor19visit_deferred_boolB1O_bKB1V_NCINvXs_B2S_B2P_NtNtB6_6row_fn5RowFn8dispatchB24_E0NCB4R_s_0E0NCB20_s_0B5E_EB2S_:
Lfunc_begin11:
	.cfi_startproc
	.cfi_personality 155, _rust_eh_personality
	.cfi_lsda 16, Lexception11
	sub	sp, sp, #464
	.cfi_def_cfa_offset 464
	stp	d11, d10, [sp, #336]
	stp	d9, d8, [sp, #352]
	stp	x28, x27, [sp, #368]
	stp	x26, x25, [sp, #384]
	stp	x24, x23, [sp, #400]
	stp	x22, x21, [sp, #416]
	stp	x20, x19, [sp, #432]
	stp	x29, x30, [sp, #448]
	add	x29, sp, #448
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_offset w19, -24
	.cfi_offset w20, -32
	.cfi_offset w21, -40
	.cfi_offset w22, -48
	.cfi_offset w23, -56
	.cfi_offset w24, -64
	.cfi_offset w25, -72
	.cfi_offset w26, -80
	.cfi_offset w27, -88
	.cfi_offset w28, -96
	.cfi_offset b8, -104
	.cfi_offset b9, -112
	.cfi_offset b10, -120
	.cfi_offset b11, -128
	.cfi_remember_state
	mov	x21, x1
	mov	x20, x0
	mov	x19, x8
	add	x22, sp, #208
	add	x8, sp, #208
	bl	__RNvXs4_NtNtNtNtNtNtNtCs5mvLrSLkDP1_12vortex_array9scalar_fn8unstable3row5types7element5tuple13element_tupleTxxENtB5_12ElementTuple6decodeCsaHo96IcALPt_24row_fn_performance_probe
	ldr	x8, [sp, #208]
	ldur	q0, [x22, #8]
	ldur	q1, [x22, #24]
	stp	q0, q1, [sp, #80]
	ldur	q0, [x22, #40]
	ldur	q1, [x22, #56]
	stp	q0, q1, [sp, #112]
	cmn	x8, #1
	b.eq	LBB12_2
	ldr	x9, [sp, #280]
	ldp	q0, q1, [sp, #80]
	stp	q0, q1, [x19, #16]
	ldp	q0, q1, [sp, #112]
	stp	q0, q1, [x19, #48]
	str	x9, [x19, #80]
	mov	w9, #1
	stp	x9, x8, [x19]
	b	LBB12_57
LBB12_2:
	ldp	q0, q1, [sp, #80]
	stp	q0, q1, [sp, #16]
	ldp	q0, q1, [sp, #112]
	stp	q0, q1, [sp, #48]
	ldr	x8, [x21, #40]
Ltmp416:
	mov	x0, x20
	blr	x8
Ltmp417:
	str	x0, [sp, #152]
	ldr	x23, [sp, #16]
	cbz	x23, LBB12_9
	ldr	x8, [sp, #24]
	mov	x28, x23
	mov	x20, x0
	cmp	x8, x0
	b.ne	LBB12_7
	add	x8, sp, #16
	ldr	x24, [sp, #48]
	cbz	x24, LBB12_10
LBB12_6:
	ldr	x8, [sp, #56]
	mov	x26, x24
	mov	x25, x0
	cmp	x8, x0
	b.eq	LBB12_11
LBB12_7:
	add	x8, sp, #152
Lloh368:
	adrp	x9, __RNvXsi_NtNtNtCslWxY2MhVcag_4core3fmt3num3impjNtB9_7Display3fmt@GOTPAGE
Lloh369:
	ldr	x9, [x9, __RNvXsi_NtNtNtCslWxY2MhVcag_4core3fmt3num3impjNtB9_7Display3fmt@GOTPAGEOFF]
	stp	x8, x9, [sp, #80]
Lloh370:
	adrp	x8, l_alloc_8cb1b9e1c7b448ff8aa56230a8661784@PAGE
Lloh371:
	add	x8, x8, l_alloc_8cb1b9e1c7b448ff8aa56230a8661784@PAGEOFF
	add	x9, sp, #80
	stp	x8, x9, [sp, #160]
Ltmp418:
Lloh372:
	adrp	x0, __RNcNtNtCs4jPh2r5lbWM_12vortex_error11VortexError5Other0@PAGE
Lloh373:
	add	x0, x0, __RNcNtNtCs4jPh2r5lbWM_12vortex_error11VortexError5Other0@PAGEOFF
	add	x8, sp, #208
	add	x1, sp, #160
	bl	__RNvNtCs4jPh2r5lbWM_12vortex_error9___private7fmt_err
Ltmp419:
	ldp	q1, q0, [sp, #208]
	stur	q0, [x19, #24]
	ldp	q0, q2, [sp, #240]
	stur	q0, [x19, #40]
	stur	q2, [x19, #56]
	ldr	q0, [sp, #272]
	stur	q0, [x19, #72]
	stur	q1, [x19, #8]
	mov	w8, #1
	str	x8, [x19]
	add	x0, sp, #16
	bl	__RINvNtCslWxY2MhVcag_4core3ptr9drop_glueTINtNtNtNtNtNtNtNtCs5mvLrSLkDP1_12vortex_array9scalar_fn8unstable3row5types7element5tuple13element_tuple9ArgColumnINtCsaHo96IcALPt_24row_fn_performance_probe11SelectedI64Kb0_EEBC_EEB2u_
	b	LBB12_57
LBB12_9:
	add	x8, sp, #16
	orr	x20, x8, #0x8
	mov	x28, x0
	add	x8, sp, #16
	ldr	x24, [sp, #48]
	cbnz	x24, LBB12_6
LBB12_10:
	add	x25, x8, #40
	mov	x26, x0
LBB12_11:
	lsr	x22, x0, #6
	ands	x27, x0, #0x3f
	cinc	x8, x22, ne
	lsl	x21, x8, #3
	add	x8, x21, #256
	mov	w9, #256
	str	x0, [sp, #8]
	cmp	x0, #0
	csel	x1, xzr, x8, eq
	csinc	x0, x9, xzr, eq
Ltmp421:
	add	x8, sp, #208
	mov	x2, #0
	bl	__RNvMs6_NtCsEOnbuuTlDO_13vortex_buffer10allocationNtB5_10Allocation8allocate
Ltmp422:
	ldr	x8, [sp, #224]
	add	x9, x8, #255
	and	x0, x9, #0xffffffffffffff00
	ldr	q0, [sp, #208]
	str	q0, [sp, #80]
	add	x9, sp, #208
	ldur	q0, [x9, #24]
	str	q0, [sp, #160]
	lsl	x9, x22, #3
	mov	x4, x22
	mov	x6, x23
	mov	x7, x20
	cbz	x22, LBB12_17
	add	x5, sp, #208
	mov	x1, x21
	cbz	x6, LBB12_18
	adrp	x12, lCPI12_0@PAGE
	mov	w21, #0
	add	x10, x28, #256
	cbz	x24, LBB12_23
	add	x11, x26, #256
	movi.2d	v0, #0xffffffffffffffff
	fneg.2d	v0, v0
	ldr	q1, [x12, lCPI12_0@PAGEOFF]
	mov	x12, x9
	mov	x13, x0
LBB12_16:
	movi.2d	v3, #0000000000000000
	mov.b	v3[0], w21
	ldp	q2, q4, [x10, #-160]
	cmeq.2d	v5, v4, v0
	cmeq.2d	v6, v2, v0
	uzp1.4s	v5, v6, v5
	ldp	q6, q7, [x10, #-192]
	cmeq.2d	v16, v7, v0
	cmeq.2d	v17, v6, v0
	uzp1.4s	v16, v17, v16
	uzp1.8h	v5, v16, v5
	ldp	q16, q17, [x10, #-224]
	cmeq.2d	v18, v17, v0
	cmeq.2d	v19, v16, v0
	uzp1.4s	v18, v19, v18
	ldp	q19, q20, [x10, #-256]
	cmeq.2d	v21, v20, v0
	cmeq.2d	v22, v19, v0
	uzp1.4s	v21, v22, v21
	uzp1.8h	v18, v21, v18
	ldp	q21, q22, [x11, #-160]
	uzp1.16b	v5, v18, v5
	cmeq.2d	v18, v22, v0
	cmeq.2d	v23, v21, v0
	ldp	q24, q25, [x11, #-192]
	uzp1.4s	v18, v23, v18
	cmeq.2d	v23, v25, v0
	cmeq.2d	v26, v24, v0
	ldp	q27, q28, [x11, #-224]
	uzp1.4s	v23, v26, v23
	uzp1.8h	v18, v23, v18
	cmeq.2d	v23, v28, v0
	ldp	q26, q29, [x11, #-256]
	cmeq.2d	v30, v27, v0
	uzp1.4s	v23, v30, v23
	cmeq.2d	v30, v29, v0
	cmeq.2d	v31, v26, v0
	uzp1.4s	v30, v31, v30
	uzp1.8h	v23, v30, v23
	uzp1.16b	v18, v23, v18
	orr.16b	v5, v5, v18
	cmgt.2d	v4, v22, v4
	cmgt.2d	v2, v21, v2
	uzp1.4s	v2, v2, v4
	cmgt.2d	v4, v25, v7
	cmgt.2d	v6, v24, v6
	uzp1.4s	v4, v6, v4
	uzp1.8h	v2, v4, v2
	cmgt.2d	v4, v28, v17
	cmgt.2d	v6, v27, v16
	uzp1.4s	v4, v6, v4
	cmgt.2d	v6, v29, v20
	cmgt.2d	v7, v26, v19
	uzp1.4s	v6, v7, v6
	uzp1.8h	v4, v6, v4
	uzp1.16b	v2, v4, v2
	orr.16b	v4, v3, v5
	ldp	q3, q6, [x10, #-128]
	ldp	q7, q16, [x10, #-96]
	ldp	q17, q18, [x10, #-64]
	ldp	q19, q20, [x10, #-32]
	ldp	q21, q22, [x11, #-128]
	ldp	q23, q24, [x11, #-96]
	ldp	q25, q26, [x11, #-64]
	ldp	q27, q28, [x11, #-32]
	cmeq.2d	v5, v20, v0
	cmeq.2d	v29, v19, v0
	uzp1.4s	v5, v29, v5
	cmeq.2d	v29, v18, v0
	cmeq.2d	v30, v17, v0
	uzp1.4s	v29, v30, v29
	uzp1.8h	v5, v29, v5
	cmeq.2d	v29, v16, v0
	cmeq.2d	v30, v7, v0
	uzp1.4s	v29, v30, v29
	cmeq.2d	v30, v6, v0
	cmeq.2d	v31, v3, v0
	uzp1.4s	v30, v31, v30
	uzp1.8h	v29, v30, v29
	uzp1.16b	v5, v29, v5
	cmeq.2d	v29, v28, v0
	cmeq.2d	v30, v27, v0
	uzp1.4s	v29, v30, v29
	cmeq.2d	v30, v26, v0
	cmeq.2d	v31, v25, v0
	uzp1.4s	v30, v31, v30
	uzp1.8h	v29, v30, v29
	cmeq.2d	v30, v24, v0
	cmeq.2d	v31, v23, v0
	uzp1.4s	v30, v31, v30
	cmeq.2d	v31, v22, v0
	cmeq.2d	v8, v21, v0
	uzp1.4s	v31, v8, v31
	uzp1.8h	v30, v31, v30
	uzp1.16b	v29, v30, v29
	orr.16b	v5, v5, v29
	cmgt.2d	v20, v28, v20
	cmgt.2d	v19, v27, v19
	uzp1.4s	v19, v19, v20
	cmgt.2d	v18, v26, v18
	cmgt.2d	v17, v25, v17
	uzp1.4s	v17, v17, v18
	uzp1.8h	v17, v17, v19
	cmgt.2d	v16, v24, v16
	cmgt.2d	v7, v23, v7
	uzp1.4s	v7, v7, v16
	cmgt.2d	v6, v22, v6
	cmgt.2d	v3, v21, v3
	uzp1.4s	v3, v3, v6
	uzp1.8h	v3, v3, v7
	uzp1.16b	v3, v3, v17
	ldp	q6, q7, [x10]
	ldp	q16, q17, [x10, #32]
	ldp	q18, q19, [x10, #64]
	ldp	q20, q21, [x10, #96]
	ldp	q22, q23, [x11]
	ldp	q24, q25, [x11, #32]
	ldp	q26, q27, [x11, #64]
	ldp	q28, q29, [x11, #96]
	cmeq.2d	v30, v21, v0
	cmeq.2d	v31, v20, v0
	uzp1.4s	v30, v31, v30
	cmeq.2d	v31, v19, v0
	cmeq.2d	v8, v18, v0
	uzp1.4s	v31, v8, v31
	uzp1.8h	v30, v31, v30
	cmeq.2d	v31, v17, v0
	cmeq.2d	v8, v16, v0
	uzp1.4s	v31, v8, v31
	cmeq.2d	v8, v7, v0
	cmeq.2d	v9, v6, v0
	uzp1.4s	v8, v9, v8
	uzp1.8h	v31, v8, v31
	uzp1.16b	v30, v31, v30
	cmeq.2d	v31, v29, v0
	cmeq.2d	v8, v28, v0
	uzp1.4s	v31, v8, v31
	cmeq.2d	v8, v27, v0
	cmeq.2d	v9, v26, v0
	uzp1.4s	v8, v9, v8
	uzp1.8h	v31, v8, v31
	cmeq.2d	v8, v25, v0
	cmeq.2d	v9, v24, v0
	uzp1.4s	v8, v9, v8
	cmeq.2d	v9, v23, v0
	cmeq.2d	v10, v22, v0
	uzp1.4s	v9, v10, v9
	uzp1.8h	v8, v9, v8
	uzp1.16b	v31, v8, v31
	orr.16b	v30, v30, v31
	cmgt.2d	v21, v29, v21
	cmgt.2d	v20, v28, v20
	uzp1.4s	v20, v20, v21
	cmgt.2d	v19, v27, v19
	cmgt.2d	v18, v26, v18
	uzp1.4s	v18, v18, v19
	uzp1.8h	v18, v18, v20
	cmgt.2d	v17, v25, v17
	cmgt.2d	v16, v24, v16
	uzp1.4s	v16, v16, v17
	cmgt.2d	v7, v23, v7
	cmgt.2d	v6, v22, v6
	uzp1.4s	v6, v6, v7
	uzp1.8h	v6, v6, v16
	uzp1.16b	v6, v6, v18
	orr.16b	v5, v5, v30
	orr.16b	v4, v4, v5
	ldp	q5, q7, [x10, #128]
	ldp	q16, q17, [x10, #160]
	ldp	q18, q19, [x10, #192]
	ldp	q20, q21, [x10, #224]
	ldp	q22, q23, [x11, #128]
	ldp	q24, q25, [x11, #160]
	ldp	q26, q27, [x11, #192]
	ldp	q28, q29, [x11, #224]
	cmeq.2d	v30, v21, v0
	cmeq.2d	v31, v20, v0
	uzp1.4s	v30, v31, v30
	cmeq.2d	v31, v19, v0
	cmeq.2d	v8, v18, v0
	uzp1.4s	v31, v8, v31
	uzp1.8h	v30, v31, v30
	cmeq.2d	v31, v17, v0
	cmeq.2d	v8, v16, v0
	uzp1.4s	v31, v8, v31
	cmeq.2d	v8, v7, v0
	cmeq.2d	v9, v5, v0
	uzp1.4s	v8, v9, v8
	uzp1.8h	v31, v8, v31
	uzp1.16b	v30, v31, v30
	cmeq.2d	v31, v29, v0
	cmeq.2d	v8, v28, v0
	uzp1.4s	v31, v8, v31
	cmeq.2d	v8, v27, v0
	cmeq.2d	v9, v26, v0
	uzp1.4s	v8, v9, v8
	uzp1.8h	v31, v8, v31
	cmeq.2d	v8, v25, v0
	cmeq.2d	v9, v24, v0
	uzp1.4s	v8, v9, v8
	cmeq.2d	v9, v23, v0
	cmeq.2d	v10, v22, v0
	uzp1.4s	v9, v10, v9
	uzp1.8h	v8, v9, v8
	uzp1.16b	v31, v8, v31
	orr.16b	v30, v30, v31
	cmgt.2d	v21, v29, v21
	cmgt.2d	v20, v28, v20
	uzp1.4s	v20, v20, v21
	cmgt.2d	v19, v27, v19
	cmgt.2d	v18, v26, v18
	uzp1.4s	v18, v18, v19
	uzp1.8h	v18, v18, v20
	cmgt.2d	v17, v25, v17
	cmgt.2d	v16, v24, v16
	uzp1.4s	v16, v16, v17
	cmgt.2d	v7, v23, v7
	cmgt.2d	v5, v22, v5
	uzp1.4s	v5, v5, v7
	uzp1.8h	v5, v5, v16
	uzp1.16b	v5, v5, v18
	orr.16b	v4, v4, v30
	shl.16b	v4, v4, #7
	cmlt.16b	v4, v4, #0
	umaxv.16b	b4, v4
	fmov	w14, s4
	and	w21, w14, #0x1
	and.16b	v2, v2, v1
	and.16b	v3, v3, v1
	and.16b	v4, v6, v1
	and.16b	v5, v5, v1
	uzp1.16b	v6, v2, v3
	uzp2.16b	v2, v2, v3
	orr.16b	v2, v6, v2
	uzp1.16b	v3, v4, v5
	uzp2.16b	v4, v4, v5
	orr.16b	v3, v3, v4
	uzp1.16b	v4, v2, v3
	uzp2.16b	v2, v2, v3
	orr.16b	v2, v4, v2
	xtn.8b	v3, v2
	add	x10, x10, #512
	uzp2.16b	v2, v2, v0
	add	x11, x11, #512
	add.16b	v2, v3, v2
	st1.d	{ v2 }[0], [x13], #8
	subs	x12, x12, #8
	b.ne	LBB12_16
	b	LBB12_25
LBB12_17:
	mov	x1, x21
	mov	w21, #0
	add	x5, sp, #208
	b	LBB12_25
LBB12_18:
	mov	w21, #0
	add	x10, x26, #256
	mov	x11, #9223372036854775807
	mov	w12, #257
Lloh374:
	adrp	x13, lCPI12_1@PAGE
Lloh375:
	ldr	q0, [x13, lCPI12_1@PAGEOFF]
	movi.2d	v1, #0xffffffffffffffff
	fneg.2d	v1, v1
	movi.16b	v2, #1
	mov	x13, x9
	mov	x14, x0
	b	LBB12_21
LBB12_19:
	cset	w16, eq
	dup.16b	v4, w16
	dup.2d	v3, x15
	movi.2d	v5, #0000000000000000
	mov.b	v5[0], w21
	ldp	q6, q16, [x10, #-160]
	cmeq.2d	v7, v16, v1
	cmeq.2d	v17, v6, v1
	uzp1.4s	v7, v17, v7
	ldp	q17, q18, [x10, #-192]
	cmeq.2d	v19, v18, v1
	cmeq.2d	v20, v17, v1
	uzp1.4s	v19, v20, v19
	uzp1.8h	v7, v19, v7
	ldp	q19, q20, [x10, #-224]
	cmeq.2d	v21, v20, v1
	cmeq.2d	v22, v19, v1
	uzp1.4s	v21, v22, v21
	ldp	q22, q23, [x10, #-256]
	cmeq.2d	v24, v23, v1
	cmeq.2d	v25, v22, v1
	uzp1.4s	v24, v25, v24
	uzp1.8h	v21, v24, v21
	uzp1.16b	v7, v21, v7
	orr.16b	v7, v7, v5
	cmgt.2d	v5, v16, v3
	cmgt.2d	v6, v6, v3
	uzp1.4s	v5, v6, v5
	cmgt.2d	v6, v18, v3
	cmgt.2d	v16, v17, v3
	uzp1.4s	v6, v16, v6
	uzp1.8h	v5, v6, v5
	cmgt.2d	v6, v20, v3
	cmgt.2d	v16, v19, v3
	uzp1.4s	v6, v16, v6
	cmgt.2d	v16, v23, v3
	cmgt.2d	v17, v22, v3
	uzp1.4s	v16, v17, v16
	uzp1.8h	v6, v16, v6
	uzp1.16b	v5, v6, v5
	and.16b	v5, v5, v2
	ldp	q6, q16, [x10, #-128]
	ldp	q17, q18, [x10, #-96]
	ldp	q19, q20, [x10, #-64]
	ldp	q21, q22, [x10, #-32]
	cmeq.2d	v23, v22, v1
	cmeq.2d	v24, v21, v1
	uzp1.4s	v23, v24, v23
	cmeq.2d	v24, v20, v1
	cmeq.2d	v25, v19, v1
	uzp1.4s	v24, v25, v24
	uzp1.8h	v23, v24, v23
	cmeq.2d	v24, v18, v1
	cmeq.2d	v25, v17, v1
	uzp1.4s	v24, v25, v24
	cmeq.2d	v25, v16, v1
	cmeq.2d	v26, v6, v1
	uzp1.4s	v25, v26, v25
	uzp1.8h	v24, v25, v24
	uzp1.16b	v23, v24, v23
	cmgt.2d	v22, v22, v3
	cmgt.2d	v21, v21, v3
	uzp1.4s	v21, v21, v22
	cmgt.2d	v20, v20, v3
	cmgt.2d	v19, v19, v3
	uzp1.4s	v19, v19, v20
	uzp1.8h	v19, v19, v21
	cmgt.2d	v18, v18, v3
	cmgt.2d	v17, v17, v3
	uzp1.4s	v17, v17, v18
	cmgt.2d	v16, v16, v3
	cmgt.2d	v6, v6, v3
	uzp1.4s	v6, v6, v16
	uzp1.8h	v6, v6, v17
	uzp1.16b	v6, v6, v19
	and.16b	v6, v6, v2
	ldp	q16, q17, [x10]
	ldp	q18, q19, [x10, #32]
	ldp	q20, q21, [x10, #64]
	ldp	q22, q24, [x10, #96]
	cmeq.2d	v25, v24, v1
	cmeq.2d	v26, v22, v1
	uzp1.4s	v25, v26, v25
	cmeq.2d	v26, v21, v1
	cmeq.2d	v27, v20, v1
	uzp1.4s	v26, v27, v26
	uzp1.8h	v25, v26, v25
	cmeq.2d	v26, v19, v1
	cmeq.2d	v27, v18, v1
	uzp1.4s	v26, v27, v26
	cmeq.2d	v27, v17, v1
	cmeq.2d	v28, v16, v1
	uzp1.4s	v27, v28, v27
	uzp1.8h	v26, v27, v26
	uzp1.16b	v25, v26, v25
	orr.16b	v23, v23, v25
	orr.16b	v23, v7, v23
	cmgt.2d	v7, v24, v3
	cmgt.2d	v22, v22, v3
	uzp1.4s	v7, v22, v7
	cmgt.2d	v21, v21, v3
	cmgt.2d	v20, v20, v3
	uzp1.4s	v20, v20, v21
	uzp1.8h	v7, v20, v7
	cmgt.2d	v19, v19, v3
	cmgt.2d	v18, v18, v3
	uzp1.4s	v18, v18, v19
	cmgt.2d	v17, v17, v3
	cmgt.2d	v16, v16, v3
	uzp1.4s	v16, v16, v17
	uzp1.8h	v16, v16, v18
	uzp1.16b	v7, v16, v7
	and.16b	v7, v7, v2
	ldp	q16, q17, [x10, #128]
	ldp	q18, q19, [x10, #160]
	ldp	q20, q21, [x10, #192]
	ldp	q22, q24, [x10, #224]
	cmeq.2d	v25, v24, v1
	cmeq.2d	v26, v22, v1
	uzp1.4s	v25, v26, v25
	cmeq.2d	v26, v21, v1
	cmeq.2d	v27, v20, v1
	uzp1.4s	v26, v27, v26
	uzp1.8h	v25, v26, v25
	cmeq.2d	v26, v19, v1
	cmeq.2d	v27, v18, v1
	uzp1.4s	v26, v27, v26
	cmeq.2d	v27, v17, v1
	cmeq.2d	v28, v16, v1
	uzp1.4s	v27, v28, v27
	uzp1.8h	v26, v27, v26
	uzp1.16b	v25, v26, v25
	orr.16b	v4, v25, v4
	orr.16b	v4, v23, v4
	cmgt.2d	v23, v24, v3
	cmgt.2d	v22, v22, v3
	uzp1.4s	v22, v22, v23
	cmgt.2d	v21, v21, v3
	cmgt.2d	v20, v20, v3
	uzp1.4s	v20, v20, v21
	uzp1.8h	v20, v20, v22
	cmgt.2d	v19, v19, v3
	cmgt.2d	v18, v18, v3
	uzp1.4s	v18, v18, v19
	cmgt.2d	v17, v17, v3
	cmgt.2d	v3, v16, v3
	uzp1.4s	v3, v3, v17
	uzp1.8h	v3, v3, v18
	uzp1.16b	v3, v3, v20
	shl.16b	v4, v4, #7
	cmlt.16b	v4, v4, #0
	umaxv.16b	b4, v4
	fmov	w15, s4
	and	w21, w15, #0x1
	and.16b	v3, v3, v2
LBB12_20:
	ushl.16b	v4, v5, v0
	ushl.16b	v5, v6, v0
	ushl.16b	v6, v7, v0
	ushl.16b	v3, v3, v0
	addp.16b	v4, v4, v5
	addp.16b	v3, v6, v3
	addp.16b	v3, v4, v3
	xtn.8b	v4, v3
	uzp2.16b	v3, v3, v0
	add.16b	v3, v4, v3
	st1.d	{ v3 }[0], [x14], #8
	add	x10, x10, #512
	subs	x13, x13, #8
	b.eq	LBB12_25
LBB12_21:
	ldr	x15, [x7]
	cmp	x15, x11
	cbnz	x24, LBB12_19
	ldr	x16, [x25]
	ccmp	x16, x11, #4, ne
	cset	w17, eq
	orr	w21, w21, w17
	cmp	x15, x16
	csel	x15, x12, xzr, lt
	orr	x15, x15, x15, lsl #16
	orr	x15, x15, x15, lsl #32
	dup.2d	v3, x15
	mov.16b	v7, v3
	mov.16b	v6, v3
	mov.16b	v5, v3
	b	LBB12_20
LBB12_23:
	mov	x11, #9223372036854775807
	movi.2d	v0, #0xffffffffffffffff
	fneg.2d	v0, v0
	ldr	q1, [x12, lCPI12_0@PAGEOFF]
	mov	x12, x9
	mov	x13, x0
LBB12_24:
	ldr	x14, [x25]
	dup.2d	v2, x14
	movi.2d	v5, #0000000000000000
	ldp	q3, q4, [x10, #-160]
	mov.b	v5[0], w21
	cmeq.2d	v7, v4, v0
	cmp	x14, x11
	cset	w14, eq
	dup.16b	v6, w14
	cmeq.2d	v16, v3, v0
	ldp	q17, q18, [x10, #-192]
	cmeq.2d	v19, v18, v0
	cmeq.2d	v20, v17, v0
	uzp1.4s	v7, v16, v7
	uzp1.4s	v16, v20, v19
	ldp	q19, q20, [x10, #-224]
	cmeq.2d	v21, v20, v0
	cmeq.2d	v22, v19, v0
	uzp1.4s	v21, v22, v21
	ldp	q22, q23, [x10, #-256]
	cmeq.2d	v24, v23, v0
	cmeq.2d	v25, v22, v0
	uzp1.4s	v24, v25, v24
	uzp1.8h	v7, v16, v7
	uzp1.8h	v16, v24, v21
	uzp1.16b	v7, v16, v7
	cmgt.2d	v16, v2, v4
	cmgt.2d	v21, v2, v3
	cmgt.2d	v18, v2, v18
	cmgt.2d	v17, v2, v17
	cmgt.2d	v20, v2, v20
	cmgt.2d	v19, v2, v19
	ldp	q3, q4, [x10, #-128]
	ldp	q24, q25, [x10, #-32]
	cmgt.2d	v23, v2, v23
	cmeq.2d	v26, v25, v0
	ldp	q27, q28, [x10, #-96]
	cmeq.2d	v29, v24, v0
	cmgt.2d	v22, v2, v22
	uzp1.4s	v26, v29, v26
	ldp	q29, q30, [x10, #-64]
	cmeq.2d	v31, v30, v0
	cmeq.2d	v8, v29, v0
	orr.16b	v7, v7, v5
	uzp1.4s	v5, v8, v31
	uzp1.8h	v26, v5, v26
	cmeq.2d	v5, v28, v0
	cmeq.2d	v31, v27, v0
	uzp1.4s	v31, v31, v5
	uzp1.4s	v5, v21, v16
	cmeq.2d	v16, v4, v0
	cmeq.2d	v21, v3, v0
	uzp1.4s	v16, v21, v16
	uzp1.8h	v16, v16, v31
	uzp1.16b	v16, v16, v26
	uzp1.4s	v17, v17, v18
	orr.16b	v6, v16, v6
	cmgt.2d	v16, v2, v25
	cmgt.2d	v18, v2, v24
	cmgt.2d	v21, v2, v30
	cmgt.2d	v24, v2, v29
	uzp1.4s	v19, v19, v20
	cmgt.2d	v20, v2, v28
	cmgt.2d	v25, v2, v27
	ldp	q26, q27, [x10]
	uzp1.4s	v22, v22, v23
	ldp	q28, q23, [x10, #32]
	ldp	q29, q30, [x10, #64]
	ldp	q31, q8, [x10, #96]
	uzp1.4s	v16, v18, v16
	cmeq.2d	v18, v8, v0
	cmeq.2d	v9, v31, v0
	uzp1.4s	v18, v9, v18
	cmeq.2d	v9, v30, v0
	cmeq.2d	v10, v29, v0
	uzp1.4s	v21, v24, v21
	uzp1.4s	v24, v10, v9
	uzp1.8h	v18, v24, v18
	cmeq.2d	v24, v23, v0
	cmeq.2d	v9, v28, v0
	uzp1.4s	v24, v9, v24
	uzp1.4s	v20, v25, v20
	cmeq.2d	v25, v27, v0
	cmeq.2d	v9, v26, v0
	uzp1.4s	v25, v9, v25
	cmgt.2d	v4, v2, v4
	cmgt.2d	v3, v2, v3
	uzp1.4s	v3, v3, v4
	uzp1.8h	v4, v25, v24
	uzp1.16b	v4, v4, v18
	cmgt.2d	v18, v2, v8
	cmgt.2d	v24, v2, v31
	uzp1.4s	v18, v24, v18
	orr.16b	v6, v7, v6
	cmgt.2d	v7, v2, v30
	cmgt.2d	v24, v2, v29
	uzp1.4s	v7, v24, v7
	cmgt.2d	v23, v2, v23
	cmgt.2d	v24, v2, v28
	uzp1.8h	v5, v17, v5
	uzp1.4s	v17, v24, v23
	cmgt.2d	v23, v2, v27
	cmgt.2d	v24, v2, v26
	uzp1.4s	v23, v24, v23
	uzp1.8h	v19, v22, v19
	ldp	q24, q22, [x10, #128]
	ldp	q25, q26, [x10, #160]
	ldp	q27, q28, [x10, #192]
	uzp1.8h	v16, v21, v16
	ldp	q21, q29, [x10, #224]
	cmeq.2d	v30, v29, v0
	cmeq.2d	v31, v21, v0
	uzp1.4s	v30, v31, v30
	uzp1.8h	v3, v3, v20
	cmeq.2d	v20, v28, v0
	cmeq.2d	v31, v27, v0
	uzp1.4s	v20, v31, v20
	uzp1.8h	v20, v20, v30
	cmeq.2d	v30, v26, v0
	uzp1.8h	v7, v7, v18
	cmeq.2d	v18, v25, v0
	uzp1.4s	v18, v18, v30
	cmeq.2d	v30, v22, v0
	cmeq.2d	v31, v24, v0
	uzp1.4s	v30, v31, v30
	uzp1.8h	v17, v23, v17
	uzp1.8h	v18, v30, v18
	uzp1.16b	v18, v18, v20
	cmgt.2d	v20, v2, v29
	cmgt.2d	v21, v2, v21
	uzp1.4s	v20, v21, v20
	uzp1.16b	v5, v19, v5
	cmgt.2d	v19, v2, v28
	cmgt.2d	v21, v2, v27
	uzp1.4s	v19, v21, v19
	uzp1.8h	v19, v19, v20
	cmgt.2d	v20, v2, v26
	uzp1.16b	v3, v3, v16
	cmgt.2d	v16, v2, v25
	uzp1.4s	v16, v16, v20
	cmgt.2d	v20, v2, v22
	cmgt.2d	v2, v2, v24
	uzp1.4s	v2, v2, v20
	uzp1.16b	v7, v17, v7
	uzp1.8h	v2, v2, v16
	uzp1.16b	v2, v2, v19
	orr.16b	v4, v18, v4
	orr.16b	v4, v4, v6
	shl.16b	v4, v4, #7
	cmlt.16b	v4, v4, #0
	and.16b	v5, v5, v1
	and.16b	v3, v3, v1
	and.16b	v6, v7, v1
	and.16b	v2, v2, v1
	uzp1.16b	v7, v5, v3
	umaxv.16b	b4, v4
	uzp2.16b	v3, v5, v3
	orr.16b	v3, v7, v3
	uzp1.16b	v5, v6, v2
	uzp2.16b	v2, v6, v2
	orr.16b	v2, v5, v2
	fmov	w14, s4
	uzp1.16b	v4, v3, v2
	uzp2.16b	v2, v3, v2
	orr.16b	v2, v4, v2
	xtn.8b	v3, v2
	uzp2.16b	v2, v2, v0
	and	w21, w14, #0x1
	add	x10, x10, #512
	add.16b	v2, v3, v2
	st1.d	{ v2 }[0], [x13], #8
	subs	x12, x12, #8
	b.ne	LBB12_24
LBB12_25:
	cbz	x27, LBB12_30
	cbz	x6, LBB12_31
	cbz	x24, LBB12_34
	cmp	x27, #8
	b.hs	LBB12_36
	mov	x11, #0
	mov	x10, #0
	b	LBB12_39
LBB12_30:
	mov	x23, x0
	b	LBB12_63
LBB12_31:
	ldr	x10, [x7]
	cbz	x24, LBB12_41
	add	x12, x26, x4, lsl #9
	cmp	x27, #8
	b.hs	LBB12_43
	mov	x11, #0
	mov	x13, #0
	b	LBB12_46
LBB12_34:
	ldr	x10, [x25]
	add	x12, x28, x4, lsl #9
	cmp	x27, #8
	b.hs	LBB12_48
	mov	x11, #0
	mov	x13, #0
	b	LBB12_51
LBB12_36:
	ldr	x14, [sp, #8]
	and	x10, x14, #0x38
	movi.2d	v0, #0000000000000000
	movi.2d	v1, #0000000000000000
	mov.h	v1[0], w21
	mov	w11, #32
	orr	x12, x11, x4, lsl #9
	add	x11, x26, x12
	add	x12, x28, x12
Lloh376:
	adrp	x13, lCPI12_2@PAGE
Lloh377:
	ldr	q2, [x13, lCPI12_2@PAGEOFF]
Lloh378:
	adrp	x13, lCPI12_3@PAGE
Lloh379:
	ldr	q3, [x13, lCPI12_3@PAGEOFF]
	mov	w13, #4
	dup.2d	v4, x13
	movi.2d	v6, #0xffffffffffffffff
	mov	w13, #1
	dup.2d	v5, x13
	mov	w13, #8
	dup.2d	v17, x13
	fneg.2d	v18, v6
	and	x13, x14, #0x38
	movi.2d	v19, #0000000000000000
	movi.2d	v6, #0000000000000000
	movi.2d	v7, #0000000000000000
	movi.2d	v16, #0000000000000000
LBB12_37:
	add.2d	v20, v2, v4
	add.2d	v21, v3, v4
	ldp	q22, q23, [x12, #-32]
	ldp	q24, q25, [x12], #64
	ldp	q26, q27, [x11, #-32]
	cmeq.2d	v28, v23, v18
	cmeq.2d	v29, v22, v18
	uzp1.4s	v28, v29, v28
	ldp	q29, q30, [x11], #64
	cmeq.2d	v31, v25, v18
	cmeq.2d	v8, v24, v18
	uzp1.4s	v31, v8, v31
	cmeq.2d	v8, v27, v18
	cmeq.2d	v9, v26, v18
	uzp1.4s	v8, v9, v8
	cmeq.2d	v9, v30, v18
	cmeq.2d	v10, v29, v18
	uzp1.4s	v9, v10, v9
	orr.16b	v28, v28, v8
	xtn.4h	v28, v28
	orr.16b	v31, v31, v9
	xtn.4h	v31, v31
	orr.8b	v1, v1, v28
	orr.8b	v19, v19, v31
	cmgt.2d	v23, v27, v23
	and.16b	v23, v23, v5
	cmgt.2d	v22, v26, v22
	and.16b	v22, v22, v5
	cmgt.2d	v25, v30, v25
	and.16b	v25, v25, v5
	cmgt.2d	v24, v29, v24
	and.16b	v24, v24, v5
	ushl.2d	v22, v22, v3
	ushl.2d	v23, v23, v2
	ushl.2d	v21, v24, v21
	ushl.2d	v20, v25, v20
	orr.16b	v7, v23, v7
	orr.16b	v6, v22, v6
	orr.16b	v0, v20, v0
	orr.16b	v16, v21, v16
	add.2d	v2, v2, v17
	add.2d	v3, v3, v17
	subs	x13, x13, #8
	b.ne	LBB12_37
	orr.8b	v1, v19, v1
	shl.4h	v1, v1, #15
	cmlt.4h	v1, v1, #0
	umaxv.4h	h1, v1
	fmov	w11, s1
	and	w21, w11, #0x1
	orr.16b	v1, v16, v6
	orr.16b	v0, v0, v7
	orr.16b	v0, v1, v0
	ext.16b	v1, v0, v0, #8
	orr.8b	v0, v0, v1
	fmov	x11, d0
	cmp	x27, x10
	b.eq	LBB12_62
LBB12_39:
	lsl	x13, x4, #9
	add	x12, x26, x13
	add	x13, x28, x13
	mov	x14, #9223372036854775807
LBB12_40:
	ldr	x15, [x13, x10, lsl #3]
	ldr	x16, [x12, x10, lsl #3]
	cmp	x15, x14
	ccmp	x16, x14, #4, ne
	cset	w17, eq
	cmp	x15, x16
	cset	w15, lt
	orr	w21, w21, w17
	lsl	x15, x15, x10
	add	x10, x10, #1
	orr	x11, x15, x11
	cmp	x27, x10
	b.ne	LBB12_40
	b	LBB12_62
LBB12_41:
	ldr	x12, [x25]
	cmp	x10, x12
	cset	w13, lt
	cmp	x27, #8
	b.hs	LBB12_53
	mov	x11, #0
	mov	x14, #0
	b	LBB12_59
LBB12_43:
	mov	x11, #9223372036854775807
	cmp	x10, x11
	cset	w11, eq
	ldr	x15, [sp, #8]
	and	x13, x15, #0x38
	movi.2d	v1, #0000000000000000
	mov.h	v1[0], w21
	dup.2d	v2, x10
	movi.2d	v0, #0000000000000000
	dup.4h	v3, w11
	add	x11, x12, #32
Lloh380:
	adrp	x14, lCPI12_2@PAGE
Lloh381:
	ldr	q4, [x14, lCPI12_2@PAGEOFF]
Lloh382:
	adrp	x14, lCPI12_3@PAGE
Lloh383:
	ldr	q5, [x14, lCPI12_3@PAGEOFF]
	mov	w14, #4
	dup.2d	v6, x14
	movi.2d	v7, #0xffffffffffffffff
	mov	w14, #1
	dup.2d	v17, x14
	mov	w14, #8
	dup.2d	v19, x14
	fneg.2d	v20, v7
	and	x14, x15, #0x38
	movi.2d	v21, #0000000000000000
	movi.2d	v7, #0000000000000000
	movi.2d	v16, #0000000000000000
	movi.2d	v18, #0000000000000000
LBB12_44:
	add.2d	v23, v4, v6
	add.2d	v24, v5, v6
	ldp	q25, q26, [x11, #-32]
	ldp	q27, q28, [x11], #64
	cmeq.2d	v22, v26, v20
	cmeq.2d	v29, v25, v20
	uzp1.4s	v22, v29, v22
	xtn.4h	v29, v22
	cmeq.2d	v22, v28, v20
	cmeq.2d	v30, v27, v20
	uzp1.4s	v22, v30, v22
	xtn.4h	v22, v22
	orr.8b	v22, v22, v21
	orr.8b	v1, v1, v3
	orr.8b	v1, v29, v1
	orr.8b	v21, v22, v3
	cmgt.2d	v26, v26, v2
	and.16b	v26, v26, v17
	cmgt.2d	v25, v25, v2
	and.16b	v25, v25, v17
	cmgt.2d	v28, v28, v2
	and.16b	v28, v28, v17
	cmgt.2d	v27, v27, v2
	and.16b	v27, v27, v17
	ushl.2d	v25, v25, v5
	ushl.2d	v26, v26, v4
	ushl.2d	v24, v27, v24
	ushl.2d	v23, v28, v23
	orr.16b	v16, v26, v16
	orr.16b	v7, v25, v7
	orr.16b	v0, v23, v0
	orr.16b	v18, v24, v18
	add.2d	v4, v4, v19
	add.2d	v5, v5, v19
	subs	x14, x14, #8
	b.ne	LBB12_44
	orr.8b	v1, v22, v1
	shl.4h	v1, v1, #15
	cmlt.4h	v1, v1, #0
	umaxv.4h	h1, v1
	fmov	w11, s1
	and	w21, w11, #0x1
	orr.16b	v1, v18, v7
	orr.16b	v0, v0, v16
	orr.16b	v0, v1, v0
	ext.16b	v1, v0, v0, #8
	orr.8b	v0, v0, v1
	fmov	x11, d0
	cmp	x27, x13
	b.eq	LBB12_62
LBB12_46:
	mov	x14, #9223372036854775807
LBB12_47:
	cmp	x10, x14
	cset	w15, eq
	ldr	x16, [x12, x13, lsl #3]
	cmp	x16, x14
	cset	w17, eq
	cmp	x10, x16
	cset	w16, lt
	orr	w15, w21, w15
	orr	w21, w17, w15
	lsl	x15, x16, x13
	add	x13, x13, #1
	orr	x11, x15, x11
	cmp	x27, x13
	b.ne	LBB12_47
	b	LBB12_62
LBB12_48:
	mov	x11, #9223372036854775807
	cmp	x10, x11
	cset	w11, eq
	ldr	x15, [sp, #8]
	and	x13, x15, #0x38
	movi.2d	v1, #0000000000000000
	mov.h	v1[0], w21
	dup.2d	v2, x10
	movi.2d	v0, #0000000000000000
	dup.4h	v3, w11
	add	x11, x12, #32
Lloh384:
	adrp	x14, lCPI12_2@PAGE
Lloh385:
	ldr	q4, [x14, lCPI12_2@PAGEOFF]
Lloh386:
	adrp	x14, lCPI12_3@PAGE
Lloh387:
	ldr	q5, [x14, lCPI12_3@PAGEOFF]
	mov	w14, #4
	dup.2d	v6, x14
	movi.2d	v7, #0xffffffffffffffff
	mov	w14, #1
	dup.2d	v17, x14
	mov	w14, #8
	dup.2d	v19, x14
	fneg.2d	v20, v7
	and	x14, x15, #0x38
	movi.2d	v21, #0000000000000000
	movi.2d	v7, #0000000000000000
	movi.2d	v16, #0000000000000000
	movi.2d	v18, #0000000000000000
LBB12_49:
	add.2d	v23, v4, v6
	add.2d	v24, v5, v6
	ldp	q25, q26, [x11, #-32]
	ldp	q27, q28, [x11], #64
	cmeq.2d	v22, v26, v20
	cmeq.2d	v29, v25, v20
	uzp1.4s	v22, v29, v22
	xtn.4h	v22, v22
	cmeq.2d	v29, v28, v20
	cmeq.2d	v30, v27, v20
	uzp1.4s	v29, v30, v29
	xtn.4h	v29, v29
	orr.8b	v1, v1, v3
	orr.8b	v1, v22, v1
	orr.8b	v22, v29, v21
	orr.8b	v21, v22, v3
	cmgt.2d	v26, v2, v26
	and.16b	v26, v26, v17
	cmgt.2d	v25, v2, v25
	and.16b	v25, v25, v17
	cmgt.2d	v28, v2, v28
	and.16b	v28, v28, v17
	cmgt.2d	v27, v2, v27
	and.16b	v27, v27, v17
	ushl.2d	v25, v25, v5
	ushl.2d	v26, v26, v4
	ushl.2d	v24, v27, v24
	ushl.2d	v23, v28, v23
	orr.16b	v16, v26, v16
	orr.16b	v7, v25, v7
	orr.16b	v0, v23, v0
	orr.16b	v18, v24, v18
	add.2d	v4, v4, v19
	add.2d	v5, v5, v19
	subs	x14, x14, #8
	b.ne	LBB12_49
	orr.8b	v1, v22, v1
	shl.4h	v1, v1, #15
	cmlt.4h	v1, v1, #0
	umaxv.4h	h1, v1
	fmov	w11, s1
	and	w21, w11, #0x1
	orr.16b	v1, v18, v7
	orr.16b	v0, v0, v16
	orr.16b	v0, v1, v0
	ext.16b	v1, v0, v0, #8
	orr.8b	v0, v0, v1
	fmov	x11, d0
	cmp	x27, x13
	b.eq	LBB12_62
LBB12_51:
	mov	x14, #9223372036854775807
LBB12_52:
	cmp	x10, x14
	cset	w15, eq
	ldr	x16, [x12, x13, lsl #3]
	cmp	x16, x14
	cset	w17, eq
	cmp	x16, x10
	cset	w16, lt
	orr	w15, w21, w15
	orr	w21, w17, w15
	lsl	x15, x16, x13
	add	x13, x13, #1
	orr	x11, x15, x11
	cmp	x27, x13
	b.ne	LBB12_52
	b	LBB12_62
LBB12_53:
	ldr	x15, [sp, #8]
	and	x14, x15, #0x38
	dup.2d	v0, x13
Lloh388:
	adrp	x11, lCPI12_3@PAGE
Lloh389:
	ldr	q2, [x11, lCPI12_3@PAGEOFF]
	movi.2d	v1, #0000000000000000
	mov	w11, #2
	dup.2d	v3, x11
	mov	w11, #4
	dup.2d	v4, x11
	mov	w11, #6
	dup.2d	v5, x11
	mov	w11, #8
	dup.2d	v6, x11
	and	x11, x15, #0x38
	movi.2d	v7, #0000000000000000
	movi.2d	v16, #0000000000000000
	movi.2d	v17, #0000000000000000
LBB12_54:
	add.2d	v18, v2, v3
	add.2d	v19, v2, v4
	add.2d	v20, v2, v5
	ushl.2d	v21, v0, v2
	ushl.2d	v18, v0, v18
	ushl.2d	v19, v0, v19
	ushl.2d	v20, v0, v20
	orr.16b	v1, v21, v1
	orr.16b	v7, v18, v7
	orr.16b	v16, v19, v16
	orr.16b	v17, v20, v17
	add.2d	v2, v2, v6
	subs	x11, x11, #8
	b.ne	LBB12_54
	orr.16b	v0, v7, v1
	orr.16b	v0, v16, v0
	orr.16b	v0, v17, v0
	ext.16b	v1, v0, v0, #8
	orr.8b	v0, v0, v1
	fmov	x11, d0
	b	LBB12_60
LBB12_56:
Ltmp420:
	b	LBB12_92
LBB12_57:
	.cfi_def_cfa wsp, 464
	ldp	x29, x30, [sp, #448]
	ldp	x20, x19, [sp, #432]
	ldp	x22, x21, [sp, #416]
	ldp	x24, x23, [sp, #400]
	ldp	x26, x25, [sp, #384]
	ldp	x28, x27, [sp, #368]
	ldp	d9, d8, [sp, #352]
	ldp	d11, d10, [sp, #336]
	add	sp, sp, #464
	.cfi_def_cfa_offset 0
	.cfi_restore w30
	.cfi_restore w29
	.cfi_restore w19
	.cfi_restore w20
	.cfi_restore w21
	.cfi_restore w22
	.cfi_restore w23
	.cfi_restore w24
	.cfi_restore w25
	.cfi_restore w26
	.cfi_restore w27
	.cfi_restore w28
	.cfi_restore b8
	.cfi_restore b9
	.cfi_restore b10
	.cfi_restore b11
	ret
LBB12_58:
	.cfi_restore_state
Ltmp432:
	b	LBB12_92
LBB12_59:
	lsl	x15, x13, x14
	add	x14, x14, #1
	orr	x11, x15, x11
LBB12_60:
	cmp	x27, x14
	b.ne	LBB12_59
	mov	x13, #9223372036854775807
	cmp	x10, x13
	ccmp	x12, x13, #4, ne
	cset	w10, eq
	orr	w21, w21, w10
LBB12_62:
	mov	x23, x0
	str	x11, [x0, x9]
LBB12_63:
	ldr	q0, [sp, #80]
	ldr	q1, [sp, #160]
	stur	q1, [x5, #40]
	ldr	x20, [sp, #8]
	lsr	x9, x20, #3
	tst	x20, #0x7
	cinc	x9, x9, ne
	cmp	x9, x1
	csel	x22, x9, x1, lo
	mov	w9, #1
	dup.2d	v1, x9
	stp	q1, q0, [sp, #208]
	str	x8, [sp, #240]
	bl	__RNvCs6rREvFdRhLb_7___rustc35___rust_no_alloc_shim_is_unstable_v2
	mov	w0, #56
	mov	w1, #8
	bl	_mi_malloc_aligned
	cbz	x0, LBB12_78
	ldp	q0, q1, [sp, #208]
	stp	q0, q1, [x0]
	ldr	q0, [sp, #240]
	str	q0, [x0, #32]
	ldr	x8, [sp, #256]
	str	x8, [x0, #48]
	stp	x23, x22, [sp, #80]
	mov	w8, #3
	strb	w8, [sp, #104]
	str	x0, [sp, #96]
	stur	x20, [x29, #-136]
	lsr	x8, x22, #61
	stur	xzr, [x29, #-160]
	cbnz	x8, LBB12_66
	lsl	x8, x22, #3
	cmp	x20, x8
	b.hi	LBB12_79
LBB12_66:
	ldp	q0, q1, [sp, #80]
	stp	q0, q1, [sp, #208]
Ltmp423:
	add	x8, sp, #160
	add	x0, sp, #208
	mov	w1, #0
	bl	__RNvMs6_NtCsEOnbuuTlDO_13vortex_buffer6bufferINtB5_6BufferhE7alignedCsaHo96IcALPt_24row_fn_performance_probe
Ltmp424:
	ldr	x8, [sp, #8]
	stp	xzr, x8, [sp, #192]
	tbnz	w21, #0, LBB12_81
LBB12_68:
	ldp	q0, q1, [sp, #160]
	stp	q0, q1, [sp, #80]
	ldr	q0, [sp, #192]
	str	q0, [sp, #112]
	stur	xzr, [x29, #-160]
Ltmp433:
	add	x0, sp, #80
	sub	x1, x29, #160
	bl	__RNvMs1_NtNtNtCs5mvLrSLkDP1_12vortex_array6arrays4bool5arrayINtNtNtBb_5array5typed5ArrayNtNtB7_6vtable4BoolE3new
Ltmp434:
	stp	x0, x1, [x19, #16]
Lloh390:
	adrp	x8, lCPI12_4@PAGE
Lloh391:
	ldr	q0, [x8, lCPI12_4@PAGEOFF]
	str	q0, [x19]
LBB12_70:
	ldr	x8, [sp, #16]
	cbz	x8, LBB12_74
	ldr	x8, [sp, #32]
	cbz	x8, LBB12_74
	mov	x9, #-1
	ldaddl	x9, x8, [x8]
	cmp	x8, #1
	b.ne	LBB12_74
	add	x8, sp, #16
	dmb	ishld
Ltmp436:
	add	x0, x8, #16
	bl	__RNvMsn_NtCs6KVRSXc8uZF_5alloc4syncINtB5_3ArcNtNtCsEOnbuuTlDO_13vortex_buffer10allocation13BufferBackingE9drop_slowBK_
Ltmp437:
LBB12_74:
	ldr	x8, [sp, #48]
	cbz	x8, LBB12_57
	ldr	x8, [sp, #64]
	cbz	x8, LBB12_57
	mov	x9, #-1
	ldaddl	x9, x8, [x8]
	cmp	x8, #1
	b.ne	LBB12_57
	add	x8, sp, #16
	dmb	ishld
	add	x0, x8, #48
	bl	__RNvMsn_NtCs6KVRSXc8uZF_5alloc4syncINtB5_3ArcNtNtCsEOnbuuTlDO_13vortex_buffer10allocation13BufferBackingE9drop_slowBK_
	b	LBB12_57
LBB12_78:
Ltmp448:
	mov	w0, #8
	mov	w1, #56
	bl	__RNvNtCs6KVRSXc8uZF_5alloc5alloc18handle_alloc_error
Ltmp449:
	b	LBB12_80
LBB12_79:
	mov	x21, x0
	add	x20, sp, #80
	str	x22, [sp, #160]
	add	x8, sp, #160
Lloh392:
	adrp	x9, __RNvXsi_NtNtNtCslWxY2MhVcag_4core3fmt3num3impjNtB9_7Display3fmt@GOTPAGE
Lloh393:
	ldr	x9, [x9, __RNvXsi_NtNtNtCslWxY2MhVcag_4core3fmt3num3impjNtB9_7Display3fmt@GOTPAGEOFF]
	stp	x8, x9, [sp, #208]
	sub	x8, x29, #160
	stp	x8, x9, [sp, #224]
	sub	x8, x29, #136
	stp	x8, x9, [sp, #240]
Ltmp442:
Lloh394:
	adrp	x0, l_alloc_b3d41c32cfb279917b2adf8d6b28aeea@PAGE
Lloh395:
	add	x0, x0, l_alloc_b3d41c32cfb279917b2adf8d6b28aeea@PAGEOFF
Lloh396:
	adrp	x2, l_alloc_aa1b3af58bcc7e8fd0ac15ecdd00b41e@PAGE
Lloh397:
	add	x2, x2, l_alloc_aa1b3af58bcc7e8fd0ac15ecdd00b41e@PAGEOFF
	add	x1, sp, #208
	bl	__RNvNtCslWxY2MhVcag_4core9panicking9panic_fmt
Ltmp443:
LBB12_80:
	brk	#0x1
LBB12_81:
Lloh398:
	adrp	x8, l_alloc_78c2b751c4d3e2d7197d387cac5cfac2@PAGE
Lloh399:
	add	x8, x8, l_alloc_78c2b751c4d3e2d7197d387cac5cfac2@PAGEOFF
	mov	w9, #55
	stp	x8, x9, [sp, #80]
Ltmp425:
Lloh400:
	adrp	x0, __RNcNtNtCs4jPh2r5lbWM_12vortex_error11VortexError5Other0@PAGE
Lloh401:
	add	x0, x0, __RNcNtNtCs4jPh2r5lbWM_12vortex_error11VortexError5Other0@PAGEOFF
	add	x8, sp, #208
	add	x1, sp, #80
	bl	__RNvNtCs4jPh2r5lbWM_12vortex_error9___private7fmt_err
Ltmp426:
	ldr	x8, [sp, #208]
	cmn	x8, #1
	b.eq	LBB12_68
	ldp	q1, q0, [sp, #208]
	stur	q0, [x19, #24]
	ldp	q0, q2, [sp, #240]
	stur	q0, [x19, #40]
	stur	q2, [x19, #56]
	ldr	q0, [sp, #272]
	stur	q0, [x19, #72]
	stur	q1, [x19, #8]
	str	xzr, [x19]
	ldr	x8, [sp, #176]
	cbz	x8, LBB12_70
	mov	x9, #-1
	ldaddl	x9, x8, [x8]
	cmp	x8, #1
	b.ne	LBB12_70
	add	x8, sp, #160
	dmb	ishld
Ltmp430:
	add	x0, x8, #16
	bl	__RNvMsn_NtCs6KVRSXc8uZF_5alloc4syncINtB5_3ArcNtNtCsEOnbuuTlDO_13vortex_buffer10allocation13BufferBackingE9drop_slowBK_
Ltmp431:
	b	LBB12_70
LBB12_86:
Ltmp427:
	mov	x19, x0
	ldr	x8, [sp, #176]
	cbz	x8, LBB12_97
	mov	x9, #-1
	ldaddl	x9, x8, [x8]
	cmp	x8, #1
	b.ne	LBB12_97
	add	x8, sp, #160
	dmb	ishld
Ltmp428:
	add	x0, x8, #16
	bl	__RNvMsn_NtCs6KVRSXc8uZF_5alloc4syncINtB5_3ArcNtNtCsEOnbuuTlDO_13vortex_buffer10allocation13BufferBackingE9drop_slowBK_
Ltmp429:
	b	LBB12_97
LBB12_89:
Ltmp438:
	mov	x19, x0
Ltmp439:
	add	x8, sp, #16
	add	x0, x8, #32
	bl	__RINvNtCslWxY2MhVcag_4core3ptr9drop_glueINtNtNtNtNtNtNtNtCs5mvLrSLkDP1_12vortex_array9scalar_fn8unstable3row5types7element5tuple13element_tuple9ArgColumnINtCsaHo96IcALPt_24row_fn_performance_probe11SelectedI64Kb0_EEEB2t_
Ltmp440:
	b	LBB12_98
LBB12_90:
Ltmp441:
	bl	__RNvNtCslWxY2MhVcag_4core9panicking16panic_in_cleanup
LBB12_91:
Ltmp435:
LBB12_92:
	mov	x19, x0
	b	LBB12_97
LBB12_93:
Ltmp444:
	mov	x19, x0
	mov	x8, #-1
	ldaddl	x8, x8, [x21]
	cmp	x8, #1
	b.ne	LBB12_97
	dmb	ishld
Ltmp445:
	add	x0, x20, #16
	bl	__RNvMsn_NtCs6KVRSXc8uZF_5alloc4syncINtB5_3ArcNtNtCsEOnbuuTlDO_13vortex_buffer10allocation13BufferBackingE9drop_slowBK_
Ltmp446:
	b	LBB12_97
LBB12_95:
Ltmp447:
	bl	__RNvNtCslWxY2MhVcag_4core9panicking16panic_in_cleanup
LBB12_96:
Ltmp450:
	mov	x19, x0
Ltmp451:
	add	x0, sp, #208
	bl	__RINvNtCslWxY2MhVcag_4core3ptr9drop_glueINtNtCs6KVRSXc8uZF_5alloc4sync8ArcInnerNtNtCsEOnbuuTlDO_13vortex_buffer10allocation13BufferBackingEECsaHo96IcALPt_24row_fn_performance_probe.llvm.17826475516497301309
Ltmp452:
LBB12_97:
Ltmp454:
	add	x0, sp, #16
	bl	__RINvNtCslWxY2MhVcag_4core3ptr9drop_glueTINtNtNtNtNtNtNtNtCs5mvLrSLkDP1_12vortex_array9scalar_fn8unstable3row5types7element5tuple13element_tuple9ArgColumnINtCsaHo96IcALPt_24row_fn_performance_probe11SelectedI64Kb0_EEBC_EEB2u_
Ltmp455:
LBB12_98:
	mov	x0, x19
	bl	__Unwind_Resume
LBB12_99:
Ltmp453:
	bl	__RNvNtCslWxY2MhVcag_4core9panicking16panic_in_cleanup
LBB12_100:
Ltmp456:
	bl	__RNvNtCslWxY2MhVcag_4core9panicking16panic_in_cleanup
	.loh AdrpAdd	Lloh372, Lloh373
	.loh AdrpAdd	Lloh370, Lloh371
	.loh AdrpLdrGot	Lloh368, Lloh369
	.loh AdrpLdr	Lloh374, Lloh375
	.loh AdrpLdr	Lloh378, Lloh379
	.loh AdrpAdrp	Lloh376, Lloh378
	.loh AdrpLdr	Lloh376, Lloh377
	.loh AdrpLdr	Lloh382, Lloh383
	.loh AdrpAdrp	Lloh380, Lloh382
	.loh AdrpLdr	Lloh380, Lloh381
	.loh AdrpLdr	Lloh386, Lloh387
	.loh AdrpAdrp	Lloh384, Lloh386
	.loh AdrpLdr	Lloh384, Lloh385
	.loh AdrpLdr	Lloh388, Lloh389
	.loh AdrpLdr	Lloh390, Lloh391
	.loh AdrpAdd	Lloh396, Lloh397
	.loh AdrpAdd	Lloh394, Lloh395
	.loh AdrpLdrGot	Lloh392, Lloh393
	.loh AdrpAdd	Lloh400, Lloh401
	.loh AdrpAdd	Lloh398, Lloh399
Lfunc_end11:
	.cfi_endproc
