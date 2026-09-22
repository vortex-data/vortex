define hidden void @_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack23collect_bool_words_avx2NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1g_7visitor5retry21ExecuteDenseWithRetryINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB35_EENtNtB3j_11row_visitor10RowVisitor19visit_deferred_boolB2Y_bKB35_NCINvXB43_B40_NtNtB1g_6row_fn5RowFn8dispatchB3e_E0NCB5V_s_0E0NCB3a_s_0B6H_E0EB43_(ptr noalias nofree noundef nonnull writeonly align 8 captures(address) %words.0, i64 noundef range(i64 0, 1152921504606846976) %words.1, i64 noundef %len, ptr dead_on_return noalias nofree noundef readonly align 8 captures(none) dereferenceable(32) %f) unnamed_addr #1 personality ptr @rust_eh_personality !dbg !260 {
start:
  %bools.i = alloca [64 x i8], align 1
  tail call void @llvm.experimental.noalias.scope.decl(metadata !265), !dbg !268
  %full5.i = lshr i64 %len, 6, !dbg !269
  %remainder.i = and i64 %len, 63, !dbg !272
  %_42.not.i = icmp samesign ugt i64 %full5.i, %words.1
  br i1 %_42.not.i, label %bb22.i, label %bb21.i, !dbg !274, !prof !293

bb22.i:                                           ; preds = %start
; call core::slice::index::slice_index_fail
  tail call void @_RNvNtNtCsc36rpYXAlPq_4core5slice5index16slice_index_fail(i64 noundef 0, i64 noundef %full5.i, i64 noundef range(i64 0, 1152921504606846976) %words.1, ptr noalias nofree noundef readonly align 8 captures(address, read_provenance) dereferenceable(24) @alloc_ebbd31bf42d91f579e6aa57e93569175.llvm.523936492659896562) #22, !dbg !294, !noalias !295
  unreachable

bb21.i:                                           ; preds = %start
  %_52.i.idx = shl nuw nsw i64 %full5.i, 3, !dbg !297
  %_52.i = getelementptr inbounds nuw i8, ptr %words.0, i64 %_52.i.idx, !dbg !297
  %_7.i.i31 = icmp eq i64 %full5.i, 0, !dbg !314
  br i1 %_7.i.i31, label %bb5.i, label %bb4.i.lr.ph, !dbg !332

bb4.i.lr.ph:                                      ; preds = %bb21.i
  %_8.i = load ptr, ptr %f, align 8, !alias.scope !333, !nonnull !23, !align !336, !noundef !23
  %0 = getelementptr inbounds nuw i8, ptr %f, i64 24
  %_11.i = load ptr, ptr %0, align 8, !nonnull !23
  %1 = getelementptr inbounds nuw i8, ptr %bools.i, i64 32
  %_6.i12 = getelementptr inbounds nuw i8, ptr %_8.i, i64 24
  %view.i.i4 = getelementptr inbounds nuw i8, ptr %_8.i, i64 8
  %2 = getelementptr inbounds nuw i8, ptr %_8.i, i64 16
  %view.i3.i15 = getelementptr inbounds nuw i8, ptr %_8.i, i64 32
  %3 = getelementptr inbounds nuw i8, ptr %_8.i, i64 40
  br label %bb4.i, !dbg !332

bb4.i:                                            ; preds = %bb9.i, %bb4.i.lr.ph
  %iter.i.sroa.0.033 = phi ptr [ %words.0, %bb4.i.lr.ph ], [ %_17.i.i, %bb9.i ]
  %iter.i.sroa.7.032 = phi i64 [ 0, %bb4.i.lr.ph ], [ %_9.0.i, %bb9.i ]
  %offset2.i = shl i64 %iter.i.sroa.7.032, 6, !dbg !337
  call void @llvm.lifetime.start.p0(ptr nonnull %bools.i), !dbg !339, !noalias !295
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 1 dereferenceable(64) %bools.i, i8 0, i64 64, i1 false), !dbg !341, !noalias !295
  br label %bb8.i, !dbg !342

bb5.i:                                            ; preds = %bb9.i, %bb21.i
  %4 = icmp eq i64 %remainder.i, 0, !dbg !351
  br i1 %4, label %_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack23collect_bool_words_withNCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1g_7visitor5retry21ExecuteDenseWithRetryINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB35_EENtNtB3j_11row_visitor10RowVisitor19visit_deferred_boolB2Y_bKB35_NCINvXB43_B40_NtNtB1g_6row_fn5RowFn8dispatchB3e_E0NCB5V_s_0E0NCB3a_s_0B6H_E0NCINvB2_23collect_bool_words_avx2B17_E0EB43_.exit, label %bb12.i, !dbg !351

bb12.i:                                           ; preds = %bb5.i
  %5 = and i64 %len, -64, !dbg !352
  tail call void @llvm.experimental.noalias.scope.decl(metadata !353), !dbg !356
  %_8.i.i.i = load ptr, ptr %f, align 8, !alias.scope !358, !noalias !361, !nonnull !23, !align !336, !noundef !23
  %6 = getelementptr inbounds nuw i8, ptr %f, i64 24
  %_11.i.i.i = load ptr, ptr %6, align 8, !alias.scope !353, !noalias !361, !nonnull !23
  %_6.i = getelementptr inbounds nuw i8, ptr %_8.i.i.i, i64 24
  %view.i.i = getelementptr inbounds nuw i8, ptr %_8.i.i.i, i64 8
  %7 = getelementptr inbounds nuw i8, ptr %_8.i.i.i, i64 16
  %view.i3.i = getelementptr inbounds nuw i8, ptr %_8.i.i.i, i64 32
  %8 = getelementptr inbounds nuw i8, ptr %_8.i.i.i, i64 40
  br label %bb8.i4, !dbg !363

bb8.i4:                                           ; preds = %_RNvXsc_NtNtNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row5types7element5tuple7indexedINtB5_14ArgTupleSourceTINtB5_15ArgColumnSourcexEB1X_EENtNtNtCslhTiPzLr6QZ_14vortex_compute12lane_kernels6source13IndexedSource13get_uncheckedCsfoTid9TleBG_17row_fn_bool_retry.exit, %bb12.i
  %packed.sroa.0.05.i = phi i64 [ 0, %bb12.i ], [ %18, %_RNvXsc_NtNtNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row5types7element5tuple7indexedINtB5_14ArgTupleSourceTINtB5_15ArgColumnSourcexEB1X_EENtNtNtCslhTiPzLr6QZ_14vortex_compute12lane_kernels6source13IndexedSource13get_uncheckedCsfoTid9TleBG_17row_fn_bool_retry.exit ]
  %iter.sroa.0.04.i = phi i64 [ 0, %bb12.i ], [ %17, %_RNvXsc_NtNtNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row5types7element5tuple7indexedINtB5_14ArgTupleSourceTINtB5_15ArgColumnSourcexEB1X_EENtNtNtCslhTiPzLr6QZ_14vortex_compute12lane_kernels6source13IndexedSource13get_uncheckedCsfoTid9TleBG_17row_fn_bool_retry.exit ]
  %_4.i.i = add nuw nsw i64 %iter.sroa.0.04.i, %5, !dbg !377
  tail call void @llvm.experimental.noalias.scope.decl(metadata !382), !dbg !383
  tail call void @llvm.experimental.noalias.scope.decl(metadata !384), !dbg !387
  tail call void @llvm.experimental.noalias.scope.decl(metadata !398), !dbg !401, !noalias !410
  %_3.i.i = load i64, ptr %_8.i.i.i, align 8, !dbg !411, !range !415, !alias.scope !416, !noalias !410, !noundef !23
  %9 = trunc nuw i64 %_3.i.i to i1, !dbg !417
  br i1 %9, label %bb2.i.i, label %bb3.i.i, !dbg !417

bb2.i.i:                                          ; preds = %bb8.i4
  %_6.i.i = load ptr, ptr %7, align 8, !dbg !418, !alias.scope !416, !noalias !410, !nonnull !23, !align !336, !noundef !23
  br label %_RNvXs_NtNtNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row5types7element5tuple7indexedINtB4_15ArgColumnSourcexENtNtNtCslhTiPzLr6QZ_14vortex_compute12lane_kernels6source13IndexedSource13get_uncheckedCsfoTid9TleBG_17row_fn_bool_retry.exit.i, !dbg !420

bb3.i.i:                                          ; preds = %bb8.i4
  %view.val.i.i = load ptr, ptr %view.i.i, align 8, !dbg !421, !alias.scope !416, !noalias !410, !nonnull !23, !align !336, !noundef !23
  %view.val1.i.i = load i64, ptr %7, align 8, !dbg !421, !alias.scope !416, !noalias !410, !noundef !23
  %_5.i.i.i1 = icmp ult i64 %_4.i.i, %view.val1.i.i, !dbg !423
  tail call void @llvm.assume(i1 %_5.i.i.i1), !dbg !434, !noalias !410
  %_4.i.i.i = getelementptr inbounds nuw i64, ptr %view.val.i.i, i64 %_4.i.i, !dbg !435
  br label %_RNvXs_NtNtNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row5types7element5tuple7indexedINtB4_15ArgColumnSourcexENtNtNtCslhTiPzLr6QZ_14vortex_compute12lane_kernels6source13IndexedSource13get_uncheckedCsfoTid9TleBG_17row_fn_bool_retry.exit.i, !dbg !436

_RNvXs_NtNtNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row5types7element5tuple7indexedINtB4_15ArgColumnSourcexENtNtNtCslhTiPzLr6QZ_14vortex_compute12lane_kernels6source13IndexedSource13get_uncheckedCsfoTid9TleBG_17row_fn_bool_retry.exit.i: ; preds = %bb3.i.i, %bb2.i.i
  %_0.sroa.0.0.in.i.i = phi ptr [ %_6.i.i, %bb2.i.i ], [ %_4.i.i.i, %bb3.i.i ]
  %_0.sroa.0.0.i.i = load i64, ptr %_0.sroa.0.0.in.i.i, align 8, !dbg !437, !noalias !438, !noundef !23
  tail call void @llvm.experimental.noalias.scope.decl(metadata !439), !dbg !401, !noalias !410
  %_3.i1.i = load i64, ptr %_6.i, align 8, !dbg !442, !range !415, !alias.scope !444, !noalias !410, !noundef !23
  %10 = trunc nuw i64 %_3.i1.i to i1, !dbg !445
  br i1 %10, label %bb2.i10.i, label %bb3.i2.i, !dbg !445

bb2.i10.i:                                        ; preds = %_RNvXs_NtNtNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row5types7element5tuple7indexedINtB4_15ArgColumnSourcexENtNtNtCslhTiPzLr6QZ_14vortex_compute12lane_kernels6source13IndexedSource13get_uncheckedCsfoTid9TleBG_17row_fn_bool_retry.exit.i
  %_6.i11.i = load ptr, ptr %8, align 8, !dbg !446, !alias.scope !444, !noalias !410, !nonnull !23, !align !336, !noundef !23
  br label %_RNvXsc_NtNtNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row5types7element5tuple7indexedINtB5_14ArgTupleSourceTINtB5_15ArgColumnSourcexEB1X_EENtNtNtCslhTiPzLr6QZ_14vortex_compute12lane_kernels6source13IndexedSource13get_uncheckedCsfoTid9TleBG_17row_fn_bool_retry.exit, !dbg !447

bb3.i2.i:                                         ; preds = %_RNvXs_NtNtNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row5types7element5tuple7indexedINtB4_15ArgColumnSourcexENtNtNtCslhTiPzLr6QZ_14vortex_compute12lane_kernels6source13IndexedSource13get_uncheckedCsfoTid9TleBG_17row_fn_bool_retry.exit.i
  %view.val.i4.i = load ptr, ptr %view.i3.i, align 8, !dbg !448, !alias.scope !444, !noalias !410, !nonnull !23, !align !336, !noundef !23
  %view.val1.i5.i = load i64, ptr %8, align 8, !dbg !448, !alias.scope !444, !noalias !410, !noundef !23
  %_5.i.i6.i = icmp ult i64 %_4.i.i, %view.val1.i5.i, !dbg !449
  tail call void @llvm.assume(i1 %_5.i.i6.i), !dbg !453, !noalias !410
  %_4.i.i7.i = getelementptr inbounds nuw i64, ptr %view.val.i4.i, i64 %_4.i.i, !dbg !454
  br label %_RNvXsc_NtNtNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row5types7element5tuple7indexedINtB5_14ArgTupleSourceTINtB5_15ArgColumnSourcexEB1X_EENtNtNtCslhTiPzLr6QZ_14vortex_compute12lane_kernels6source13IndexedSource13get_uncheckedCsfoTid9TleBG_17row_fn_bool_retry.exit, !dbg !455

_RNvXsc_NtNtNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row5types7element5tuple7indexedINtB5_14ArgTupleSourceTINtB5_15ArgColumnSourcexEB1X_EENtNtNtCslhTiPzLr6QZ_14vortex_compute12lane_kernels6source13IndexedSource13get_uncheckedCsfoTid9TleBG_17row_fn_bool_retry.exit: ; preds = %bb2.i10.i, %bb3.i2.i
  %_0.sroa.0.0.in.i8.i = phi ptr [ %_6.i11.i, %bb2.i10.i ], [ %_4.i.i7.i, %bb3.i2.i ]
  %_0.sroa.0.0.i9.i = load i64, ptr %_0.sroa.0.0.in.i8.i, align 8, !dbg !456, !noalias !457, !noundef !23
  %11 = icmp eq i64 %_0.sroa.0.0.i.i, 9223372036854775807, !dbg !458
  %12 = icmp eq i64 %_0.sroa.0.0.i9.i, 9223372036854775807, !dbg !458
  %_6.sroa.0.0.i.i.i.i.i = or i1 %11, %12, !dbg !458
  %_5.i.i.i.i.i = icmp slt i64 %_0.sroa.0.0.i.i, %_0.sroa.0.0.i9.i, !dbg !474
  %13 = load i8, ptr %_11.i.i.i, align 1, !dbg !475, !range !483, !alias.scope !484, !noalias !410, !noundef !23
  %14 = trunc nuw i8 %13 to i1, !dbg !475
  %15 = or i1 %_6.sroa.0.0.i.i.i.i.i, %14, !dbg !475
  %16 = zext i1 %15 to i8, !dbg !475
  store i8 %16, ptr %_11.i.i.i, align 1, !dbg !475, !alias.scope !484, !noalias !410
  %17 = add nuw nsw i64 %iter.sroa.0.04.i, 1, !dbg !487
  %_13.i = zext i1 %_5.i.i.i.i.i to i64, !dbg !494
  %_12.i = shl nuw i64 %_13.i, %iter.sroa.0.04.i, !dbg !494
  %18 = or i64 %_12.i, %packed.sroa.0.05.i, !dbg !495
  %exitcond.not.i = icmp eq i64 %17, %remainder.i, !dbg !496
  br i1 %exitcond.not.i, label %_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack24collect_bool_word_scalarNCINvB2_23collect_bool_words_withNCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1O_7visitor5retry21ExecuteDenseWithRetryINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB3D_EENtNtB3R_11row_visitor10RowVisitor19visit_deferred_boolB3w_bKB3D_NCINvXB4B_B4y_NtNtB1O_6row_fn5RowFn8dispatchB3M_E0NCB6t_s_0E0NCB3I_s_0B7f_E0NCINvB2_23collect_bool_words_avx2B1F_E0E0EB4B_.exit, label %bb8.i4, !dbg !363

_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack24collect_bool_word_scalarNCINvB2_23collect_bool_words_withNCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1O_7visitor5retry21ExecuteDenseWithRetryINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB3D_EENtNtB3R_11row_visitor10RowVisitor19visit_deferred_boolB3w_bKB3D_NCINvXB4B_B4y_NtNtB1O_6row_fn5RowFn8dispatchB3M_E0NCB6t_s_0E0NCB3I_s_0B7f_E0NCINvB2_23collect_bool_words_avx2B1F_E0E0EB4B_.exit: ; preds = %_RNvXsc_NtNtNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row5types7element5tuple7indexedINtB5_14ArgTupleSourceTINtB5_15ArgColumnSourcexEB1X_EENtNtNtCslhTiPzLr6QZ_14vortex_compute12lane_kernels6source13IndexedSource13get_uncheckedCsfoTid9TleBG_17row_fn_bool_retry.exit
  %_41.i = icmp samesign ult i64 %full5.i, %words.1, !dbg !503
  br i1 %_41.i, label %bb14.i, label %panic.i, !dbg !503

bb14.i:                                           ; preds = %_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack24collect_bool_word_scalarNCINvB2_23collect_bool_words_withNCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1O_7visitor5retry21ExecuteDenseWithRetryINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB3D_EENtNtB3R_11row_visitor10RowVisitor19visit_deferred_boolB3w_bKB3D_NCINvXB4B_B4y_NtNtB1O_6row_fn5RowFn8dispatchB3M_E0NCB6t_s_0E0NCB3I_s_0B7f_E0NCINvB2_23collect_bool_words_avx2B1F_E0E0EB4B_.exit
  store i64 %18, ptr %_52.i, align 8, !dbg !503, !alias.scope !265, !noalias !504
  br label %_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack23collect_bool_words_withNCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1g_7visitor5retry21ExecuteDenseWithRetryINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB35_EENtNtB3j_11row_visitor10RowVisitor19visit_deferred_boolB2Y_bKB35_NCINvXB43_B40_NtNtB1g_6row_fn5RowFn8dispatchB3e_E0NCB5V_s_0E0NCB3a_s_0B6H_E0NCINvB2_23collect_bool_words_avx2B17_E0EB43_.exit, !dbg !505

panic.i:                                          ; preds = %_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack24collect_bool_word_scalarNCINvB2_23collect_bool_words_withNCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1O_7visitor5retry21ExecuteDenseWithRetryINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB3D_EENtNtB3R_11row_visitor10RowVisitor19visit_deferred_boolB3w_bKB3D_NCINvXB4B_B4y_NtNtB1O_6row_fn5RowFn8dispatchB3M_E0NCB6t_s_0E0NCB3I_s_0B7f_E0NCINvB2_23collect_bool_words_avx2B1F_E0E0EB4B_.exit
; call core::panicking::panic_bounds_check
  tail call void @_RNvNtCsc36rpYXAlPq_4core9panicking18panic_bounds_check(i64 noundef %full5.i, i64 noundef range(i64 0, 1152921504606846976) %words.1, ptr noalias nofree noundef readonly align 8 captures(address, read_provenance) dereferenceable(24) @alloc_1b2922caae1b461da04857d3d02eae5b.llvm.523936492659896562) #22, !dbg !503
  unreachable

bb8.i:                                            ; preds = %_RNvXsc_NtNtNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row5types7element5tuple7indexedINtB5_14ArgTupleSourceTINtB5_15ArgColumnSourcexEB1X_EENtNtNtCslhTiPzLr6QZ_14vortex_compute12lane_kernels6source13IndexedSource13get_uncheckedCsfoTid9TleBG_17row_fn_bool_retry.exit26, %bb4.i
  %iter1.i.sroa.7.030 = phi i64 [ 0, %bb4.i ], [ %_9.0.i12, %_RNvXsc_NtNtNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row5types7element5tuple7indexedINtB5_14ArgTupleSourceTINtB5_15ArgColumnSourcexEB1X_EENtNtNtCslhTiPzLr6QZ_14vortex_compute12lane_kernels6source13IndexedSource13get_uncheckedCsfoTid9TleBG_17row_fn_bool_retry.exit26 ]
  %iter1.i.sroa.0.0.ptr = getelementptr inbounds nuw i8, ptr %bools.i, i64 %iter1.i.sroa.7.030, !dbg !506
  %_9.0.i12 = add nuw nsw i64 %iter1.i.sroa.7.030, 1, !dbg !509
  %_31.i = add nuw nsw i64 %iter1.i.sroa.7.030, %offset2.i, !dbg !512
  tail call void @llvm.experimental.noalias.scope.decl(metadata !333), !dbg !514
  tail call void @llvm.experimental.noalias.scope.decl(metadata !515), !dbg !518
  tail call void @llvm.experimental.noalias.scope.decl(metadata !520), !dbg !523, !noalias !333
  %_3.i.i2 = load i64, ptr %_8.i, align 8, !dbg !525, !range !415, !alias.scope !527, !noalias !333, !noundef !23
  %19 = trunc nuw i64 %_3.i.i2 to i1, !dbg !528
  br i1 %19, label %bb2.i.i24, label %bb3.i.i3, !dbg !528

bb2.i.i24:                                        ; preds = %bb8.i
  %_6.i.i25 = load ptr, ptr %2, align 8, !dbg !529, !alias.scope !527, !noalias !333, !nonnull !23, !align !336, !noundef !23
  br label %_RNvXs_NtNtNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row5types7element5tuple7indexedINtB4_15ArgColumnSourcexENtNtNtCslhTiPzLr6QZ_14vortex_compute12lane_kernels6source13IndexedSource13get_uncheckedCsfoTid9TleBG_17row_fn_bool_retry.exit.i9, !dbg !530

bb3.i.i3:                                         ; preds = %bb8.i
  %view.val.i.i5 = load ptr, ptr %view.i.i4, align 8, !dbg !531, !alias.scope !527, !noalias !333, !nonnull !23, !align !336, !noundef !23
  %view.val1.i.i6 = load i64, ptr %2, align 8, !dbg !531, !alias.scope !527, !noalias !333, !noundef !23
  %_5.i.i.i7 = icmp ult i64 %_31.i, %view.val1.i.i6, !dbg !532
  tail call void @llvm.assume(i1 %_5.i.i.i7), !dbg !536, !noalias !333
  %_4.i.i.i8 = getelementptr inbounds nuw i64, ptr %view.val.i.i5, i64 %_31.i, !dbg !537
  br label %_RNvXs_NtNtNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row5types7element5tuple7indexedINtB4_15ArgColumnSourcexENtNtNtCslhTiPzLr6QZ_14vortex_compute12lane_kernels6source13IndexedSource13get_uncheckedCsfoTid9TleBG_17row_fn_bool_retry.exit.i9, !dbg !538

_RNvXs_NtNtNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row5types7element5tuple7indexedINtB4_15ArgColumnSourcexENtNtNtCslhTiPzLr6QZ_14vortex_compute12lane_kernels6source13IndexedSource13get_uncheckedCsfoTid9TleBG_17row_fn_bool_retry.exit.i9: ; preds = %bb3.i.i3, %bb2.i.i24
  %_0.sroa.0.0.in.i.i10 = phi ptr [ %_6.i.i25, %bb2.i.i24 ], [ %_4.i.i.i8, %bb3.i.i3 ]
  %_0.sroa.0.0.i.i11 = load i64, ptr %_0.sroa.0.0.in.i.i10, align 8, !dbg !539, !noalias !540, !noundef !23
  tail call void @llvm.experimental.noalias.scope.decl(metadata !541), !dbg !523, !noalias !333
  %_3.i1.i13 = load i64, ptr %_6.i12, align 8, !dbg !544, !range !415, !alias.scope !546, !noalias !333, !noundef !23
  %20 = trunc nuw i64 %_3.i1.i13 to i1, !dbg !547
  br i1 %20, label %bb2.i10.i22, label %bb3.i2.i14, !dbg !547

bb2.i10.i22:                                      ; preds = %_RNvXs_NtNtNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row5types7element5tuple7indexedINtB4_15ArgColumnSourcexENtNtNtCslhTiPzLr6QZ_14vortex_compute12lane_kernels6source13IndexedSource13get_uncheckedCsfoTid9TleBG_17row_fn_bool_retry.exit.i9
  %_6.i11.i23 = load ptr, ptr %3, align 8, !dbg !548, !alias.scope !546, !noalias !333, !nonnull !23, !align !336, !noundef !23
  br label %_RNvXsc_NtNtNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row5types7element5tuple7indexedINtB5_14ArgTupleSourceTINtB5_15ArgColumnSourcexEB1X_EENtNtNtCslhTiPzLr6QZ_14vortex_compute12lane_kernels6source13IndexedSource13get_uncheckedCsfoTid9TleBG_17row_fn_bool_retry.exit26, !dbg !549

bb3.i2.i14:                                       ; preds = %_RNvXs_NtNtNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row5types7element5tuple7indexedINtB4_15ArgColumnSourcexENtNtNtCslhTiPzLr6QZ_14vortex_compute12lane_kernels6source13IndexedSource13get_uncheckedCsfoTid9TleBG_17row_fn_bool_retry.exit.i9
  %view.val.i4.i16 = load ptr, ptr %view.i3.i15, align 8, !dbg !550, !alias.scope !546, !noalias !333, !nonnull !23, !align !336, !noundef !23
  %view.val1.i5.i17 = load i64, ptr %3, align 8, !dbg !550, !alias.scope !546, !noalias !333, !noundef !23
  %_5.i.i6.i18 = icmp ult i64 %_31.i, %view.val1.i5.i17, !dbg !551
  tail call void @llvm.assume(i1 %_5.i.i6.i18), !dbg !555, !noalias !333
  %_4.i.i7.i19 = getelementptr inbounds nuw i64, ptr %view.val.i4.i16, i64 %_31.i, !dbg !556
  br label %_RNvXsc_NtNtNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row5types7element5tuple7indexedINtB5_14ArgTupleSourceTINtB5_15ArgColumnSourcexEB1X_EENtNtNtCslhTiPzLr6QZ_14vortex_compute12lane_kernels6source13IndexedSource13get_uncheckedCsfoTid9TleBG_17row_fn_bool_retry.exit26, !dbg !557

_RNvXsc_NtNtNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row5types7element5tuple7indexedINtB5_14ArgTupleSourceTINtB5_15ArgColumnSourcexEB1X_EENtNtNtCslhTiPzLr6QZ_14vortex_compute12lane_kernels6source13IndexedSource13get_uncheckedCsfoTid9TleBG_17row_fn_bool_retry.exit26: ; preds = %bb2.i10.i22, %bb3.i2.i14
  %_0.sroa.0.0.in.i8.i20 = phi ptr [ %_6.i11.i23, %bb2.i10.i22 ], [ %_4.i.i7.i19, %bb3.i2.i14 ]
  %_0.sroa.0.0.i9.i21 = load i64, ptr %_0.sroa.0.0.in.i8.i20, align 8, !dbg !558, !noalias !559, !noundef !23
  %21 = icmp eq i64 %_0.sroa.0.0.i.i11, 9223372036854775807, !dbg !560
  %22 = icmp eq i64 %_0.sroa.0.0.i9.i21, 9223372036854775807, !dbg !560
  %_6.sroa.0.0.i.i.i = or i1 %21, %22, !dbg !560
  %_5.i.i.i = icmp slt i64 %_0.sroa.0.0.i.i11, %_0.sroa.0.0.i9.i21, !dbg !563
  %23 = load i8, ptr %_11.i, align 1, !dbg !564, !range !483, !alias.scope !566, !noalias !333, !noundef !23
  %24 = trunc nuw i8 %23 to i1, !dbg !564
  %25 = or i1 %_6.sroa.0.0.i.i.i, %24, !dbg !564
  %26 = zext i1 %25 to i8, !dbg !564
  store i8 %26, ptr %_11.i, align 1, !dbg !564, !alias.scope !566, !noalias !333
  %27 = zext i1 %_5.i.i.i to i8, !dbg !569
  store i8 %27, ptr %iter1.i.sroa.0.0.ptr, align 1, !dbg !569
  %_7.i.i8 = icmp eq i64 %_9.0.i12, 64, !dbg !506
  br i1 %_7.i.i8, label %bb9.i, label %bb8.i, !dbg !342

bb9.i:                                            ; preds = %_RNvXsc_NtNtNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row5types7element5tuple7indexedINtB5_14ArgTupleSourceTINtB5_15ArgColumnSourcexEB1X_EENtNtNtCslhTiPzLr6QZ_14vortex_compute12lane_kernels6source13IndexedSource13get_uncheckedCsfoTid9TleBG_17row_fn_bool_retry.exit26
  %_17.i.i = getelementptr inbounds nuw i8, ptr %iter.i.sroa.0.033, i64 8, !dbg !570
  %_9.0.i = add nuw nsw i64 %iter.i.sroa.7.032, 1, !dbg !573
  %bools.i.val27 = load <32 x i8>, ptr %bools.i, align 1, !dbg !576, !noalias !577
  %bools.i.val128 = load <32 x i8>, ptr %1, align 1, !dbg !576, !noalias !580
  %28 = shufflevector <32 x i8> %bools.i.val27, <32 x i8> %bools.i.val128, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>, !dbg !583
  %29 = icmp ne <64 x i8> %28, zeroinitializer, !dbg !583
  store <64 x i1> %29, ptr %iter.i.sroa.0.033, align 8, !dbg !595
  call void @llvm.lifetime.end.p0(ptr nonnull %bools.i), !dbg !596, !noalias !295
  %_7.i.i = icmp eq ptr %_17.i.i, %_52.i, !dbg !314
  br i1 %_7.i.i, label %bb5.i, label %bb4.i, !dbg !332

_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack23collect_bool_words_withNCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1g_7visitor5retry21ExecuteDenseWithRetryINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB35_EENtNtB3j_11row_visitor10RowVisitor19visit_deferred_boolB2Y_bKB35_NCINvXB43_B40_NtNtB1g_6row_fn5RowFn8dispatchB3e_E0NCB5V_s_0E0NCB3a_s_0B6H_E0NCINvB2_23collect_bool_words_avx2B17_E0EB43_.exit: ; preds = %bb14.i, %bb5.i
  ret void, !dbg !597
}
