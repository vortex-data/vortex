; SPDX-License-Identifier: Apache-2.0
; SPDX-FileCopyrightText: Copyright the Vortex contributors
__RINvNtNtNtNtNtCs5mvLrSLkDP1_12vortex_array9scalar_fn8unstable3row7execute4sink12execute_sinkTINtNtNtCs2rdgasfnGJ9_13vortex_tensor10scalar_fns3row9TensorRowdEB1t_EuINtNtNtNtB6_5types4sink14uninit_element17UninitElementSinkdENtB2E_18InitializedElementNCINvYINtNtNtB6_7visitor5retry21ExecuteDenseWithRetryNtNtB1y_17cosine_similarity16CosineSimilarityENtNtB4a_11row_visitor10RowVisitor10visit_intoB1s_B2B_B3z_NCINvXs_B4S_B4Q_NtNtB6_6row_fn5RowFn8dispatchB45_Es0_0E0NCB41_s_0ECsaHo96IcALPt_24row_fn_performance_probe:
Lfunc_begin2:
	.cfi_startproc
	.cfi_personality 155, _rust_eh_personality
	.cfi_lsda 16, Lexception2
	sub	sp, sp, #320
	.cfi_def_cfa_offset 320
	stp	x24, x23, [sp, #256]
	stp	x22, x21, [sp, #272]
	stp	x20, x19, [sp, #288]
	stp	x29, x30, [sp, #304]
	add	x29, sp, #304
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_offset w19, -24
	.cfi_offset w20, -32
	.cfi_offset w21, -40
	.cfi_offset w22, -48
	.cfi_offset w23, -56
	.cfi_offset w24, -64
	.cfi_remember_state
	mov	x2, x3
	mov	x21, x1
	mov	x20, x0
	mov	x19, x8
	add	x22, sp, #112
	add	x8, sp, #112
	bl	__RNvXs4_NtNtNtNtNtNtNtCs5mvLrSLkDP1_12vortex_array9scalar_fn8unstable3row5types7element5tuple13element_tupleTINtNtNtCs2rdgasfnGJ9_13vortex_tensor10scalar_fns3row9TensorRowdEB1I_ENtB5_12ElementTuple6decodeCsaHo96IcALPt_24row_fn_performance_probe
	ldr	x8, [sp, #112]
	cmp	x8, #1
	b.ne	LBB3_2
	ldur	q0, [x22, #24]
	ldur	q1, [x22, #40]
	ldur	q2, [x22, #56]
	stp	q1, q2, [sp, #32]
	ldur	q3, [x22, #72]
	str	q3, [sp, #64]
	ldur	q4, [x22, #8]
	stp	q4, q0, [sp]
	stp	q1, q2, [x19, #32]
	str	q3, [x19, #64]
	stp	q4, q0, [x19]
	b	LBB3_95
LBB3_2:
	ldur	q0, [x22, #72]
	ldur	q1, [x22, #88]
	stp	q0, q1, [sp, #64]
	ldur	q2, [x22, #104]
	ldur	q3, [x22, #8]
	ldur	q4, [x22, #24]
	stp	q3, q4, [sp]
	ldur	q5, [x22, #56]
	ldur	q6, [x22, #40]
	stp	q2, q3, [sp, #96]
	stp	q4, q6, [sp, #128]
	stp	q5, q0, [sp, #160]
	stp	q1, q2, [sp, #192]
	ldr	x8, [x21, #40]
Ltmp33:
	mov	x0, x20
	blr	x8
Ltmp34:
	mov	x22, #0
	lsl	x21, x0, #3
	lsr	x8, x0, #61
	cbnz	x8, LBB3_8
	mov	x8, #9223372036854775800
	cmp	x21, x8
	b.hi	LBB3_8
	cbz	x21, LBB3_9
	mov	x23, x0
	bl	__RNvCs6rREvFdRhLb_7___rustc35___rust_no_alloc_shim_is_unstable_v2
	mov	w22, #8
	mov	x0, x21
	mov	w1, #8
	bl	_mi_malloc_aligned
	mov	x20, x0
	cbz	x0, LBB3_8
	mov	x0, x23
	mov	x21, x23
	ldr	x8, [sp, #168]
	ldr	x10, [sp, #112]
	ldr	x9, [sp, #144]
	cbnz	x10, LBB3_10
	b	LBB3_42
LBB3_8:
Ltmp48:
	mov	x0, x22
	mov	x1, x21
	bl	__RNvNtCs6KVRSXc8uZF_5alloc7raw_vec12handle_error
Ltmp49:
	b	LBB3_83
LBB3_9:
	mov	w20, #8
	ldr	x8, [sp, #168]
	ldr	x10, [sp, #112]
	ldr	x9, [sp, #144]
	cbz	x10, LBB3_42
LBB3_10:
	cbz	x8, LBB3_42
	cmp	x9, x0
	b.ne	LBB3_87
	ldr	x8, [sp, #200]
	cmp	x8, x0
	b.ne	LBB3_87
	cbz	x0, LBB3_92
LBB3_14:
	mov	x8, #0
	mov	x9, #0
	movi.2d	v0, #0000000000000000
	b	LBB3_17
LBB3_15:
	fsqrt	d2, d2
LBB3_16:
	fmul	d2, d3, d2
	fdiv	d1, d1, d2
	fcmp	d2, #0.0
	fcsel	d1, d0, d1, eq
	str	d1, [x20, x9, lsl #3]
	add	x9, x9, #1
	add	x8, x8, #8
	cmp	x9, x0
	b.eq	LBB3_92
LBB3_17:
	ldr	x13, [sp, #112]
	ldp	x15, x14, [sp, #152]
	ldr	x10, [sp, #168]
	ldp	x12, x11, [sp, #208]
	cmp	x12, x15
	csel	x16, x12, x15, lo
	movi.2d	v1, #0000000000000000
	cbz	x16, LBB3_25
	cmp	x16, #8
	b.hs	LBB3_20
	mov	x17, #0
	b	LBB3_23
LBB3_20:
	and	x17, x16, #0xfffffffffffffff8
	madd	x1, x14, x8, x13
	add	x1, x1, #32
	madd	x2, x11, x8, x10
	add	x2, x2, #32
	and	x3, x16, #0xfffffffffffffff8
LBB3_21:
	ldp	q2, q3, [x1, #-32]
	ldp	q4, q5, [x1], #64
	ldp	q6, q7, [x2, #-32]
	ldp	q16, q17, [x2], #64
	fmul.2d	v2, v2, v6
	mov	d6, v2[1]
	fmul.2d	v3, v3, v7
	mov	d7, v3[1]
	fmul.2d	v4, v4, v16
	mov	d16, v4[1]
	fmul.2d	v5, v5, v17
	mov	d17, v5[1]
	fadd	d1, d1, d2
	fadd	d1, d1, d6
	fadd	d1, d1, d3
	fadd	d1, d1, d7
	fadd	d1, d1, d4
	fadd	d1, d1, d16
	fadd	d1, d1, d5
	fadd	d1, d1, d17
	subs	x3, x3, #8
	b.ne	LBB3_21
	cmp	x16, x17
	b.eq	LBB3_25
LBB3_23:
	lsl	x2, x17, #3
	madd	x1, x11, x8, x2
	add	x1, x10, x1
	madd	x2, x14, x8, x2
	add	x2, x13, x2
	sub	x16, x16, x17
LBB3_24:
	ldr	d2, [x2], #8
	ldr	d3, [x1], #8
	fmul	d2, d2, d3
	fadd	d1, d1, d2
	subs	x16, x16, #1
	b.ne	LBB3_24
LBB3_25:
	movi.2d	v2, #0000000000000000
	movi.2d	v3, #0000000000000000
	cbz	x15, LBB3_34
	mul	x16, x14, x9
	add	x16, x13, x16, lsl #3
	lsl	x17, x15, #3
	sub	x15, x17, #8
	cmp	x15, #56
	b.hs	LBB3_28
	mov	x15, x16
	b	LBB3_31
LBB3_28:
	lsr	x15, x15, #3
	add	x1, x15, #1
	and	x2, x1, #0x3ffffffffffffff8
	add	x15, x16, x2, lsl #3
	madd	x13, x14, x8, x13
	add	x13, x13, #32
	and	x14, x1, #0x3ffffffffffffff8
LBB3_29:
	ldp	q4, q5, [x13, #-32]
	ldp	q6, q7, [x13], #64
	fmul.2d	v4, v4, v4
	mov	d16, v4[1]
	fmul.2d	v5, v5, v5
	mov	d17, v5[1]
	fmul.2d	v6, v6, v6
	mov	d18, v6[1]
	fmul.2d	v7, v7, v7
	mov	d19, v7[1]
	fadd	d3, d3, d4
	fadd	d3, d3, d16
	fadd	d3, d3, d5
	fadd	d3, d3, d17
	fadd	d3, d3, d6
	fadd	d3, d3, d18
	fadd	d3, d3, d7
	fadd	d3, d3, d19
	subs	x14, x14, #8
	b.ne	LBB3_29
	cmp	x1, x2
	b.eq	LBB3_33
LBB3_31:
	add	x13, x16, x17
LBB3_32:
	ldr	d4, [x15], #8
	fmul	d4, d4, d4
	fadd	d3, d3, d4
	cmp	x15, x13
	b.ne	LBB3_32
LBB3_33:
	fsqrt	d3, d3
LBB3_34:
	cbz	x12, LBB3_16
	mul	x13, x11, x9
	add	x13, x10, x13, lsl #3
	lsl	x14, x12, #3
	sub	x12, x14, #8
	cmp	x12, #56
	b.hs	LBB3_37
	movi.2d	v2, #0000000000000000
	mov	x12, x13
	b	LBB3_40
LBB3_37:
	lsr	x12, x12, #3
	add	x15, x12, #1
	and	x16, x15, #0x3ffffffffffffff8
	add	x12, x13, x16, lsl #3
	madd	x10, x11, x8, x10
	add	x10, x10, #32
	movi.2d	v2, #0000000000000000
	and	x11, x15, #0x3ffffffffffffff8
LBB3_38:
	ldp	q4, q5, [x10, #-32]
	ldp	q6, q7, [x10], #64
	fmul.2d	v4, v4, v4
	mov	d16, v4[1]
	fmul.2d	v5, v5, v5
	mov	d17, v5[1]
	fmul.2d	v6, v6, v6
	mov	d18, v6[1]
	fmul.2d	v7, v7, v7
	mov	d19, v7[1]
	fadd	d2, d2, d4
	fadd	d2, d2, d16
	fadd	d2, d2, d5
	fadd	d2, d2, d17
	fadd	d2, d2, d6
	fadd	d2, d2, d18
	fadd	d2, d2, d7
	fadd	d2, d2, d19
	subs	x11, x11, #8
	b.ne	LBB3_38
	cmp	x15, x16
	b.eq	LBB3_15
LBB3_40:
	add	x10, x13, x14
LBB3_41:
	ldr	d4, [x12], #8
	fmul	d4, d4, d4
	fadd	d2, d2, d4
	cmp	x12, x10
	b.ne	LBB3_41
	b	LBB3_15
LBB3_42:
	cbz	x10, LBB3_44
	cmp	x9, x0
	b.ne	LBB3_84
LBB3_44:
	cbz	x8, LBB3_46
	ldr	x8, [sp, #200]
	cmp	x8, x0
	b.ne	LBB3_84
LBB3_46:
	cbz	x0, LBB3_92
LBB3_47:
	mov	x9, #0
	movi.2d	v0, #0000000000000000
	b	LBB3_50
LBB3_48:
	movi.2d	v3, #0000000000000000
LBB3_49:
	fmul	d2, d2, d3
	fdiv	d1, d1, d2
	fcmp	d2, #0.0
	fcsel	d1, d0, d1, eq
	str	d1, [x20, x9, lsl #3]
	add	x9, x9, #1
	cmp	x9, x0
	b.eq	LBB3_92
LBB3_50:
	ldr	x10, [sp, #112]
	cbz	x10, LBB3_57
	ldp	x12, x8, [sp, #152]
	mul	x8, x8, x9
	ldr	x2, [sp, #120]
	adds	x1, x12, x8
	b.hs	LBB3_82
	cmp	x1, x2
	b.hi	LBB3_82
	add	x10, x10, x8, lsl #3
	ldr	x13, [sp, #168]
	cbz	x13, LBB3_58
LBB3_54:
	ldp	x11, x8, [sp, #208]
	mul	x8, x8, x9
	ldr	x2, [sp, #176]
	adds	x1, x11, x8
	b.hs	LBB3_82
	cmp	x1, x2
	b.hi	LBB3_82
	add	x8, x13, x8, lsl #3
	cmp	x11, x12
	csel	x13, x11, x12, lo
	movi.2d	v2, #0000000000000000
	movi.2d	v1, #0000000000000000
	cbnz	x13, LBB3_59
	b	LBB3_66
LBB3_57:
	ldp	x10, x12, [sp, #120]
	ldr	x13, [sp, #168]
	cbnz	x13, LBB3_54
LBB3_58:
	ldp	x8, x11, [sp, #176]
	cmp	x11, x12
	csel	x13, x11, x12, lo
	movi.2d	v2, #0000000000000000
	movi.2d	v1, #0000000000000000
	cbz	x13, LBB3_66
LBB3_59:
	cmp	x13, #8
	b.hs	LBB3_61
	mov	x14, #0
	b	LBB3_64
LBB3_61:
	and	x14, x13, #0xfffffffffffffff8
	add	x15, x10, #32
	add	x16, x8, #32
	and	x17, x13, #0xfffffffffffffff8
LBB3_62:
	ldp	q3, q4, [x15, #-32]
	ldp	q5, q6, [x15], #64
	ldp	q7, q16, [x16, #-32]
	ldp	q17, q18, [x16], #64
	fmul.2d	v3, v3, v7
	mov	d7, v3[1]
	fmul.2d	v4, v4, v16
	mov	d16, v4[1]
	fmul.2d	v5, v5, v17
	mov	d17, v5[1]
	fmul.2d	v6, v6, v18
	mov	d18, v6[1]
	fadd	d1, d1, d3
	fadd	d1, d1, d7
	fadd	d1, d1, d4
	fadd	d1, d1, d16
	fadd	d1, d1, d5
	fadd	d1, d1, d17
	fadd	d1, d1, d6
	fadd	d1, d1, d18
	subs	x17, x17, #8
	b.ne	LBB3_62
	cmp	x13, x14
	b.eq	LBB3_66
LBB3_64:
	lsl	x16, x14, #3
	add	x15, x8, x16
	add	x16, x10, x16
	sub	x13, x13, x14
LBB3_65:
	ldr	d3, [x16], #8
	ldr	d4, [x15], #8
	fmul	d3, d3, d4
	fadd	d1, d1, d3
	subs	x13, x13, #1
	b.ne	LBB3_65
LBB3_66:
	cbz	x12, LBB3_74
	lsl	x13, x12, #3
	sub	x14, x13, #8
	movi.2d	v2, #0000000000000000
	mov	x12, x10
	cmp	x14, #56
	b.lo	LBB3_71
	lsr	x12, x14, #3
	add	x14, x12, #1
	and	x15, x14, #0x3ffffffffffffff8
	add	x12, x10, x15, lsl #3
	add	x16, x10, #32
	and	x17, x14, #0x3ffffffffffffff8
LBB3_69:
	ldp	q3, q4, [x16, #-32]
	ldp	q5, q6, [x16], #64
	fmul.2d	v3, v3, v3
	mov	d7, v3[1]
	fmul.2d	v4, v4, v4
	mov	d16, v4[1]
	fmul.2d	v5, v5, v5
	mov	d17, v5[1]
	fmul.2d	v6, v6, v6
	mov	d18, v6[1]
	fadd	d2, d2, d3
	fadd	d2, d2, d7
	fadd	d2, d2, d4
	fadd	d2, d2, d16
	fadd	d2, d2, d5
	fadd	d2, d2, d17
	fadd	d2, d2, d6
	fadd	d2, d2, d18
	subs	x17, x17, #8
	b.ne	LBB3_69
	cmp	x14, x15
	b.eq	LBB3_73
LBB3_71:
	add	x10, x10, x13
LBB3_72:
	ldr	d3, [x12], #8
	fmul	d3, d3, d3
	fadd	d2, d2, d3
	cmp	x12, x10
	b.ne	LBB3_72
LBB3_73:
	fsqrt	d2, d2
LBB3_74:
	cbz	x11, LBB3_48
	lsl	x11, x11, #3
	sub	x12, x11, #8
	movi.2d	v3, #0000000000000000
	mov	x10, x8
	cmp	x12, #56
	b.lo	LBB3_79
	lsr	x10, x12, #3
	add	x12, x10, #1
	and	x13, x12, #0x3ffffffffffffff8
	add	x10, x8, x13, lsl #3
	add	x14, x8, #32
	and	x15, x12, #0x3ffffffffffffff8
LBB3_77:
	ldp	q4, q5, [x14, #-32]
	ldp	q6, q7, [x14], #64
	fmul.2d	v4, v4, v4
	mov	d16, v4[1]
	fmul.2d	v5, v5, v5
	mov	d17, v5[1]
	fmul.2d	v6, v6, v6
	mov	d18, v6[1]
	fmul.2d	v7, v7, v7
	mov	d19, v7[1]
	fadd	d3, d3, d4
	fadd	d3, d3, d16
	fadd	d3, d3, d5
	fadd	d3, d3, d17
	fadd	d3, d3, d6
	fadd	d3, d3, d18
	fadd	d3, d3, d7
	fadd	d3, d3, d19
	subs	x15, x15, #8
	b.ne	LBB3_77
	cmp	x12, x13
	b.eq	LBB3_81
LBB3_79:
	add	x8, x8, x11
LBB3_80:
	ldr	d4, [x10], #8
	fmul	d4, d4, d4
	fadd	d3, d3, d4
	cmp	x10, x8
	b.ne	LBB3_80
LBB3_81:
	fsqrt	d3, d3
	b	LBB3_49
LBB3_82:
Ltmp42:
Lloh2:
	adrp	x3, _alloc_cd44ca30c046dd08226918654a8231a3.llvm.12919063887536347969@PAGE
Lloh3:
	add	x3, x3, _alloc_cd44ca30c046dd08226918654a8231a3.llvm.12919063887536347969@PAGEOFF
	mov	x0, x8
	bl	__RNvNtNtCslWxY2MhVcag_4core5slice5index16slice_index_fail
Ltmp43:
LBB3_83:
	brk	#0x1
LBB3_84:
Ltmp39:
	mov	x8, sp
	mov	x22, x0
	bl	__RNvNtNtNtNtNtCs5mvLrSLkDP1_12vortex_array9scalar_fn8unstable3row7execute4sink20decoded_length_error
Ltmp40:
	ldr	x8, [sp]
	cmn	x8, #1
	b.ne	LBB3_89
	mov	x0, x22
	cbnz	x22, LBB3_47
	b	LBB3_92
LBB3_87:
Ltmp36:
	mov	x8, sp
	mov	x22, x0
	bl	__RNvNtNtNtNtNtCs5mvLrSLkDP1_12vortex_array9scalar_fn8unstable3row7execute4sink20decoded_length_error
Ltmp37:
	ldr	x8, [sp]
	cmn	x8, #1
	b.eq	LBB3_91
LBB3_89:
	ldp	q0, q1, [sp, #32]
	stp	q0, q1, [x19, #32]
	ldr	q0, [sp, #64]
	str	q0, [x19, #64]
	ldp	q1, q0, [sp]
	stp	q1, q0, [x19]
	cbz	x21, LBB3_94
	mov	x0, x20
	bl	_mi_free
	b	LBB3_94
LBB3_91:
	mov	x0, x22
	cbnz	x22, LBB3_14
LBB3_92:
	stp	x21, x20, [x29, #-72]
	stur	x0, [x29, #-56]
	str	xzr, [sp]
Ltmp45:
	sub	x0, x29, #72
	mov	x1, sp
	bl	__RINvMs1_NtNtNtCs5mvLrSLkDP1_12vortex_array6arrays9primitive5arrayINtNtNtBc_5array5typed5ArrayNtNtB8_6vtable9PrimitiveE3newdINtNtCs6KVRSXc8uZF_5alloc3vec3VecdEECsaHo96IcALPt_24row_fn_performance_probe
Ltmp46:
	stp	x0, x1, [x19, #8]
	mov	x8, #-1
	str	x8, [x19]
LBB3_94:
	add	x0, sp, #112
	bl	__RINvNtCslWxY2MhVcag_4core3ptr9drop_glueTINtNtNtNtNtNtNtNtCs5mvLrSLkDP1_12vortex_array9scalar_fn8unstable3row5types7element5tuple13element_tuple9ArgColumnINtNtNtCs2rdgasfnGJ9_13vortex_tensor10scalar_fns3row9TensorRowNtNtCsajgNrFuHJYM_4half8binary163f16EEBC_EECsaHo96IcALPt_24row_fn_performance_probe
LBB3_95:
	.cfi_def_cfa wsp, 320
	ldp	x29, x30, [sp, #304]
	ldp	x20, x19, [sp, #288]
	ldp	x22, x21, [sp, #272]
	ldp	x24, x23, [sp, #256]
	add	sp, sp, #320
	.cfi_def_cfa_offset 0
	.cfi_restore w30
	.cfi_restore w29
	.cfi_restore w19
	.cfi_restore w20
	.cfi_restore w21
	.cfi_restore w22
	.cfi_restore w23
	.cfi_restore w24
	ret
LBB3_96:
	.cfi_restore_state
Ltmp38:
	b	LBB3_101
LBB3_97:
Ltmp41:
	b	LBB3_101
LBB3_98:
Ltmp47:
	b	LBB3_104
LBB3_99:
Ltmp35:
	b	LBB3_104
LBB3_100:
Ltmp44:
LBB3_101:
	mov	x19, x0
	cbz	x21, LBB3_105
	mov	x0, x20
	bl	_mi_free
	b	LBB3_105
LBB3_103:
Ltmp50:
LBB3_104:
	mov	x19, x0
LBB3_105:
Ltmp51:
	add	x0, sp, #112
	bl	__RINvNtCslWxY2MhVcag_4core3ptr9drop_glueTINtNtNtNtNtNtNtNtCs5mvLrSLkDP1_12vortex_array9scalar_fn8unstable3row5types7element5tuple13element_tuple9ArgColumnINtNtNtCs2rdgasfnGJ9_13vortex_tensor10scalar_fns3row9TensorRowNtNtCsajgNrFuHJYM_4half8binary163f16EEBC_EECsaHo96IcALPt_24row_fn_performance_probe
Ltmp52:
	mov	x0, x19
	bl	__Unwind_Resume
LBB3_107:
Ltmp53:
	bl	__RNvNtCslWxY2MhVcag_4core9panicking16panic_in_cleanup
	.loh AdrpAdd	Lloh2, Lloh3
Lfunc_end2:
	.cfi_endproc
