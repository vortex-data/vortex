define hidden void @_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor5retry21ExecuteDenseWithRetryINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB45_B42_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5X_s_0E0NCB3c_s_0B6J_E0EB45_(ptr noalias nofree noundef nonnull writeonly align 8 captures(address) %words.0, i64 noundef range(i64 0, 1152921504606846976) %words.1, i64 noundef %len, ptr dead_on_return noalias nofree noundef readonly align 8 captures(none) dereferenceable(32) %f) unnamed_addr #3 personality ptr @rust_eh_personality !dbg !680 {
start:
  %bools.i = alloca [64 x i8], align 1
  tail call void @llvm.experimental.noalias.scope.decl(metadata !681), !dbg !684
  %full5.i = lshr i64 %len, 6, !dbg !685
  %remainder.i = and i64 %len, 63, !dbg !688
  %_42.not.i = icmp samesign ugt i64 %full5.i, %words.1
  br i1 %_42.not.i, label %bb22.i, label %bb21.i, !dbg !690, !prof !293

bb22.i:                                           ; preds = %start
; call core::slice::index::slice_index_fail
  tail call void @_RNvNtNtCsc36rpYXAlPq_4core5slice5index16slice_index_fail(i64 noundef 0, i64 noundef %full5.i, i64 noundef range(i64 0, 1152921504606846976) %words.1, ptr noalias nofree noundef readonly align 8 captures(address, read_provenance) dereferenceable(24) @alloc_ebbd31bf42d91f579e6aa57e93569175.llvm.523936492659896562) #22, !dbg !700, !noalias !701
  unreachable

bb21.i:                                           ; preds = %start
  %_52.i.idx = shl nuw nsw i64 %full5.i, 3, !dbg !703
  %_52.i = getelementptr inbounds nuw i8, ptr %words.0, i64 %_52.i.idx, !dbg !703
  %_7.i.i25 = icmp eq i64 %full5.i, 0, !dbg !712
  br i1 %_7.i.i25, label %bb5.i, label %bb4.i.preheader, !dbg !717

bb4.i.preheader:                                  ; preds = %bb21.i
  %_8.i = load ptr, ptr %f, align 8, !alias.scope !718, !nonnull !23, !align !336, !noundef !23
  %_6.i.i13 = getelementptr inbounds nuw i8, ptr %_8.i, i64 24
  %0 = getelementptr inbounds nuw i8, ptr %f, i64 24
  %_11.i = load ptr, ptr %0, align 8, !alias.scope !718, !nonnull !23, !noundef !23
  %view.i.i.i5 = getelementptr inbounds nuw i8, ptr %_8.i, i64 8
  %1 = getelementptr inbounds nuw i8, ptr %_8.i, i64 16
  %view.i3.i.i16 = getelementptr inbounds nuw i8, ptr %_8.i, i64 32
  %2 = getelementptr inbounds nuw i8, ptr %_8.i, i64 40
  br label %bb4.i, !dbg !721

bb4.i:                                            ; preds = %bb4.i.preheader, %bb9.i
  %iter.i.sroa.0.027 = phi ptr [ %_17.i.i, %bb9.i ], [ %words.0, %bb4.i.preheader ]
  %iter.i.sroa.7.026 = phi i64 [ %_9.0.i, %bb9.i ], [ 0, %bb4.i.preheader ]
  %offset2.i = shl i64 %iter.i.sroa.7.026, 6, !dbg !728
  call void @llvm.lifetime.start.p0(ptr nonnull %bools.i), !dbg !729, !noalias !701
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 1 dereferenceable(64) %bools.i, i8 0, i64 64, i1 false), !dbg !730, !noalias !701
  br label %bb8.i, !dbg !721

bb5.i:                                            ; preds = %bb9.i, %bb21.i
  %3 = icmp eq i64 %remainder.i, 0, !dbg !731
  br i1 %3, label %_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack23collect_bool_words_withNCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1g_7visitor5retry21ExecuteDenseWithRetryINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB35_EENtNtB3j_11row_visitor10RowVisitor19visit_deferred_boolB2Y_bKB35_NCINvXB43_B40_NtNtB1g_6row_fn5RowFn8dispatchB3e_E0NCB5V_s_0E0NCB3a_s_0B6H_E0NCINvB2_25collect_bool_words_avx512B17_E0EB43_.exit, label %bb12.i, !dbg !731

bb12.i:                                           ; preds = %bb5.i
  %4 = and i64 %len, -64, !dbg !732
  tail call void @llvm.experimental.noalias.scope.decl(metadata !733), !dbg !736
  %_8.i.i.i = load ptr, ptr %f, align 8, !alias.scope !738, !noalias !741, !nonnull !23, !align !336, !noundef !23
  %5 = getelementptr inbounds nuw i8, ptr %f, i64 24
  %_11.i.i.i = load ptr, ptr %5, align 8, !alias.scope !733, !noalias !741, !nonnull !23
  %_6.i.i = getelementptr inbounds nuw i8, ptr %_8.i.i.i, i64 24
  %view.i.i.i = getelementptr inbounds nuw i8, ptr %_8.i.i.i, i64 8
  %6 = getelementptr inbounds nuw i8, ptr %_8.i.i.i, i64 16
  %view.i3.i.i = getelementptr inbounds nuw i8, ptr %_8.i.i.i, i64 32
  %7 = getelementptr inbounds nuw i8, ptr %_8.i.i.i, i64 40
  br label %bb8.i1, !dbg !743

bb8.i1:                                           ; preds = %_RNvXsc_NtNtNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row5types7element5tuple7indexedINtB5_14ArgTupleSourceTINtB5_15ArgColumnSourcexEB1X_EENtNtNtCslhTiPzLr6QZ_14vortex_compute12lane_kernels6source13IndexedSource13get_uncheckedCsfoTid9TleBG_17row_fn_bool_retry.exit.i, %bb12.i
  %packed.sroa.0.05.i = phi i64 [ 0, %bb12.i ], [ %17, %_RNvXsc_NtNtNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row5types7element5tuple7indexedINtB5_14ArgTupleSourceTINtB5_15ArgColumnSourcexEB1X_EENtNtNtCslhTiPzLr6QZ_14vortex_compute12lane_kernels6source13IndexedSource13get_uncheckedCsfoTid9TleBG_17row_fn_bool_retry.exit.i ]
  %iter.sroa.0.04.i = phi i64 [ 0, %bb12.i ], [ %16, %_RNvXsc_NtNtNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row5types7element5tuple7indexedINtB5_14ArgTupleSourceTINtB5_15ArgColumnSourcexEB1X_EENtNtNtCslhTiPzLr6QZ_14vortex_compute12lane_kernels6source13IndexedSource13get_uncheckedCsfoTid9TleBG_17row_fn_bool_retry.exit.i ]
  %_4.i.i = add nuw nsw i64 %iter.sroa.0.04.i, %4, !dbg !747
  tail call void @llvm.experimental.noalias.scope.decl(metadata !749), !dbg !750
  tail call void @llvm.experimental.noalias.scope.decl(metadata !751), !dbg !754
  tail call void @llvm.experimental.noalias.scope.decl(metadata !756), !dbg !759, !noalias !749
  %_3.i.i.i = load i64, ptr %_8.i.i.i, align 8, !dbg !761, !range !415, !alias.scope !763, !noalias !764, !noundef !23
  %8 = trunc nuw i64 %_3.i.i.i to i1, !dbg !765
  br i1 %8, label %bb2.i.i.i, label %bb3.i.i.i, !dbg !765

bb2.i.i.i:                                        ; preds = %bb8.i1
  %_6.i.i.i = load ptr, ptr %6, align 8, !dbg !766, !alias.scope !763, !noalias !764, !nonnull !23, !align !336, !noundef !23
  br label %_RNvXs_NtNtNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row5types7element5tuple7indexedINtB4_15ArgColumnSourcexENtNtNtCslhTiPzLr6QZ_14vortex_compute12lane_kernels6source13IndexedSource13get_uncheckedCsfoTid9TleBG_17row_fn_bool_retry.exit.i.i, !dbg !767

bb3.i.i.i:                                        ; preds = %bb8.i1
  %view.val.i.i.i = load ptr, ptr %view.i.i.i, align 8, !dbg !768, !alias.scope !763, !noalias !764, !nonnull !23, !align !336, !noundef !23
  %view.val1.i.i.i = load i64, ptr %6, align 8, !dbg !768, !alias.scope !763, !noalias !764, !noundef !23
  %_5.i.i.i.i = icmp ult i64 %_4.i.i, %view.val1.i.i.i, !dbg !769
  tail call void @llvm.assume(i1 %_5.i.i.i.i), !dbg !773, !noalias !749
  %_4.i.i.i.i = getelementptr inbounds nuw i64, ptr %view.val.i.i.i, i64 %_4.i.i, !dbg !774
  br label %_RNvXs_NtNtNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row5types7element5tuple7indexedINtB4_15ArgColumnSourcexENtNtNtCslhTiPzLr6QZ_14vortex_compute12lane_kernels6source13IndexedSource13get_uncheckedCsfoTid9TleBG_17row_fn_bool_retry.exit.i.i, !dbg !775

_RNvXs_NtNtNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row5types7element5tuple7indexedINtB4_15ArgColumnSourcexENtNtNtCslhTiPzLr6QZ_14vortex_compute12lane_kernels6source13IndexedSource13get_uncheckedCsfoTid9TleBG_17row_fn_bool_retry.exit.i.i: ; preds = %bb3.i.i.i, %bb2.i.i.i
  %_0.sroa.0.0.in.i.i.i = phi ptr [ %_6.i.i.i, %bb2.i.i.i ], [ %_4.i.i.i.i, %bb3.i.i.i ]
  %_0.sroa.0.0.i.i.i = load i64, ptr %_0.sroa.0.0.in.i.i.i, align 8, !dbg !776, !noalias !777, !noundef !23
  tail call void @llvm.experimental.noalias.scope.decl(metadata !778), !dbg !759, !noalias !749
  %_3.i1.i.i = load i64, ptr %_6.i.i, align 8, !dbg !781, !range !415, !alias.scope !783, !noalias !764, !noundef !23
  %9 = trunc nuw i64 %_3.i1.i.i to i1, !dbg !784
  br i1 %9, label %bb2.i10.i.i, label %bb3.i2.i.i, !dbg !784

bb2.i10.i.i:                                      ; preds = %_RNvXs_NtNtNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row5types7element5tuple7indexedINtB4_15ArgColumnSourcexENtNtNtCslhTiPzLr6QZ_14vortex_compute12lane_kernels6source13IndexedSource13get_uncheckedCsfoTid9TleBG_17row_fn_bool_retry.exit.i.i
  %_6.i11.i.i = load ptr, ptr %7, align 8, !dbg !785, !alias.scope !783, !noalias !764, !nonnull !23, !align !336, !noundef !23
  br label %_RNvXsc_NtNtNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row5types7element5tuple7indexedINtB5_14ArgTupleSourceTINtB5_15ArgColumnSourcexEB1X_EENtNtNtCslhTiPzLr6QZ_14vortex_compute12lane_kernels6source13IndexedSource13get_uncheckedCsfoTid9TleBG_17row_fn_bool_retry.exit.i, !dbg !786

bb3.i2.i.i:                                       ; preds = %_RNvXs_NtNtNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row5types7element5tuple7indexedINtB4_15ArgColumnSourcexENtNtNtCslhTiPzLr6QZ_14vortex_compute12lane_kernels6source13IndexedSource13get_uncheckedCsfoTid9TleBG_17row_fn_bool_retry.exit.i.i
  %view.val.i4.i.i = load ptr, ptr %view.i3.i.i, align 8, !dbg !787, !alias.scope !783, !noalias !764, !nonnull !23, !align !336, !noundef !23
  %view.val1.i5.i.i = load i64, ptr %7, align 8, !dbg !787, !alias.scope !783, !noalias !764, !noundef !23
  %_5.i.i6.i.i = icmp ult i64 %_4.i.i, %view.val1.i5.i.i, !dbg !788
  tail call void @llvm.assume(i1 %_5.i.i6.i.i), !dbg !792, !noalias !749
  %_4.i.i7.i.i = getelementptr inbounds nuw i64, ptr %view.val.i4.i.i, i64 %_4.i.i, !dbg !793
  br label %_RNvXsc_NtNtNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row5types7element5tuple7indexedINtB5_14ArgTupleSourceTINtB5_15ArgColumnSourcexEB1X_EENtNtNtCslhTiPzLr6QZ_14vortex_compute12lane_kernels6source13IndexedSource13get_uncheckedCsfoTid9TleBG_17row_fn_bool_retry.exit.i, !dbg !794

_RNvXsc_NtNtNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row5types7element5tuple7indexedINtB5_14ArgTupleSourceTINtB5_15ArgColumnSourcexEB1X_EENtNtNtCslhTiPzLr6QZ_14vortex_compute12lane_kernels6source13IndexedSource13get_uncheckedCsfoTid9TleBG_17row_fn_bool_retry.exit.i: ; preds = %bb3.i2.i.i, %bb2.i10.i.i
  %_0.sroa.0.0.in.i8.i.i = phi ptr [ %_6.i11.i.i, %bb2.i10.i.i ], [ %_4.i.i7.i.i, %bb3.i2.i.i ]
  %_0.sroa.0.0.i9.i.i = load i64, ptr %_0.sroa.0.0.in.i8.i.i, align 8, !dbg !795, !noalias !796, !noundef !23
  %10 = icmp eq i64 %_0.sroa.0.0.i.i.i, 9223372036854775807, !dbg !797
  %11 = icmp eq i64 %_0.sroa.0.0.i9.i.i, 9223372036854775807, !dbg !797
  %_6.sroa.0.0.i.i.i.i.i = or i1 %10, %11, !dbg !797
  %_5.i.i.i.i.i = icmp slt i64 %_0.sroa.0.0.i.i.i, %_0.sroa.0.0.i9.i.i, !dbg !800
  %12 = load i8, ptr %_11.i.i.i, align 1, !dbg !801, !range !483, !alias.scope !803, !noalias !764, !noundef !23
  %13 = trunc nuw i8 %12 to i1, !dbg !801
  %14 = or i1 %_6.sroa.0.0.i.i.i.i.i, %13, !dbg !801
  %15 = zext i1 %14 to i8, !dbg !801
  store i8 %15, ptr %_11.i.i.i, align 1, !dbg !801, !alias.scope !803, !noalias !764
  %16 = add nuw nsw i64 %iter.sroa.0.04.i, 1, !dbg !806
  %_13.i = zext i1 %_5.i.i.i.i.i to i64, !dbg !809
  %_12.i = shl nuw i64 %_13.i, %iter.sroa.0.04.i, !dbg !809
  %17 = or i64 %_12.i, %packed.sroa.0.05.i, !dbg !810
  %exitcond.not.i = icmp eq i64 %16, %remainder.i, !dbg !811
  br i1 %exitcond.not.i, label %_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack24collect_bool_word_scalarNCINvB2_23collect_bool_words_withNCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1O_7visitor5retry21ExecuteDenseWithRetryINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB3D_EENtNtB3R_11row_visitor10RowVisitor19visit_deferred_boolB3w_bKB3D_NCINvXB4B_B4y_NtNtB1O_6row_fn5RowFn8dispatchB3M_E0NCB6t_s_0E0NCB3I_s_0B7f_E0NCINvB2_25collect_bool_words_avx512B1F_E0E0EB4B_.llvm.523936492659896562.exit, label %bb8.i1, !dbg !743

_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack24collect_bool_word_scalarNCINvB2_23collect_bool_words_withNCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1O_7visitor5retry21ExecuteDenseWithRetryINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB3D_EENtNtB3R_11row_visitor10RowVisitor19visit_deferred_boolB3w_bKB3D_NCINvXB4B_B4y_NtNtB1O_6row_fn5RowFn8dispatchB3M_E0NCB6t_s_0E0NCB3I_s_0B7f_E0NCINvB2_25collect_bool_words_avx512B1F_E0E0EB4B_.llvm.523936492659896562.exit: ; preds = %_RNvXsc_NtNtNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row5types7element5tuple7indexedINtB5_14ArgTupleSourceTINtB5_15ArgColumnSourcexEB1X_EENtNtNtCslhTiPzLr6QZ_14vortex_compute12lane_kernels6source13IndexedSource13get_uncheckedCsfoTid9TleBG_17row_fn_bool_retry.exit.i
  %_41.i = icmp samesign ult i64 %full5.i, %words.1, !dbg !813
  br i1 %_41.i, label %bb14.i, label %panic.i, !dbg !813

bb14.i:                                           ; preds = %_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack24collect_bool_word_scalarNCINvB2_23collect_bool_words_withNCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1O_7visitor5retry21ExecuteDenseWithRetryINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB3D_EENtNtB3R_11row_visitor10RowVisitor19visit_deferred_boolB3w_bKB3D_NCINvXB4B_B4y_NtNtB1O_6row_fn5RowFn8dispatchB3M_E0NCB6t_s_0E0NCB3I_s_0B7f_E0NCINvB2_25collect_bool_words_avx512B1F_E0E0EB4B_.llvm.523936492659896562.exit
  store i64 %17, ptr %_52.i, align 8, !dbg !813, !alias.scope !681, !noalias !814
  br label %_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack23collect_bool_words_withNCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1g_7visitor5retry21ExecuteDenseWithRetryINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB35_EENtNtB3j_11row_visitor10RowVisitor19visit_deferred_boolB2Y_bKB35_NCINvXB43_B40_NtNtB1g_6row_fn5RowFn8dispatchB3e_E0NCB5V_s_0E0NCB3a_s_0B6H_E0NCINvB2_25collect_bool_words_avx512B17_E0EB43_.exit, !dbg !815

panic.i:                                          ; preds = %_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack24collect_bool_word_scalarNCINvB2_23collect_bool_words_withNCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1O_7visitor5retry21ExecuteDenseWithRetryINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB3D_EENtNtB3R_11row_visitor10RowVisitor19visit_deferred_boolB3w_bKB3D_NCINvXB4B_B4y_NtNtB1O_6row_fn5RowFn8dispatchB3M_E0NCB6t_s_0E0NCB3I_s_0B7f_E0NCINvB2_25collect_bool_words_avx512B1F_E0E0EB4B_.llvm.523936492659896562.exit
; call core::panicking::panic_bounds_check
  tail call void @_RNvNtCsc36rpYXAlPq_4core9panicking18panic_bounds_check(i64 noundef %full5.i, i64 noundef range(i64 0, 1152921504606846976) %words.1, ptr noalias nofree noundef readonly align 8 captures(address, read_provenance) dereferenceable(24) @alloc_1b2922caae1b461da04857d3d02eae5b.llvm.523936492659896562) #22, !dbg !813
  unreachable

bb8.i:                                            ; preds = %_RNCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB8_7visitor5retry21ExecuteDenseWithRetryINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB1X_EENtNtB2b_11row_visitor10RowVisitor19visit_deferred_boolB1Q_bKB1X_NCINvXB2U_B2R_NtNtB8_6row_fn5RowFn8dispatchB26_E0NCB4M_s_0E0NCB22_s_0B5x_E0B2U_.llvm.523936492659896562.exit, %bb4.i
  %iter1.i.sroa.7.024 = phi i64 [ 0, %bb4.i ], [ %_9.0.i9, %_RNCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB8_7visitor5retry21ExecuteDenseWithRetryINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB1X_EENtNtB2b_11row_visitor10RowVisitor19visit_deferred_boolB1Q_bKB1X_NCINvXB2U_B2R_NtNtB8_6row_fn5RowFn8dispatchB26_E0NCB4M_s_0E0NCB22_s_0B5x_E0B2U_.llvm.523936492659896562.exit ]
  %iter1.i.sroa.0.0.ptr = getelementptr inbounds nuw i8, ptr %bools.i, i64 %iter1.i.sroa.7.024, !dbg !816
  %_9.0.i9 = add nuw nsw i64 %iter1.i.sroa.7.024, 1, !dbg !818
  %_31.i = add nuw nsw i64 %iter1.i.sroa.7.024, %offset2.i, !dbg !819
  tail call void @llvm.experimental.noalias.scope.decl(metadata !718), !dbg !821
  tail call void @llvm.experimental.noalias.scope.decl(metadata !822), !dbg !825
  tail call void @llvm.experimental.noalias.scope.decl(metadata !827), !dbg !830
  %_3.i.i.i3 = load i64, ptr %_8.i, align 8, !dbg !832, !range !415, !alias.scope !834, !noalias !718, !noundef !23
  %18 = trunc nuw i64 %_3.i.i.i3 to i1, !dbg !835
  br i1 %18, label %bb2.i.i.i26, label %bb3.i.i.i4, !dbg !835

bb2.i.i.i26:                                      ; preds = %bb8.i
  %_6.i.i.i27 = load ptr, ptr %1, align 8, !dbg !836, !alias.scope !834, !noalias !718, !nonnull !23, !align !336, !noundef !23
  br label %_RNvXs_NtNtNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row5types7element5tuple7indexedINtB4_15ArgColumnSourcexENtNtNtCslhTiPzLr6QZ_14vortex_compute12lane_kernels6source13IndexedSource13get_uncheckedCsfoTid9TleBG_17row_fn_bool_retry.exit.i.i10, !dbg !837

bb3.i.i.i4:                                       ; preds = %bb8.i
  %view.val.i.i.i6 = load ptr, ptr %view.i.i.i5, align 8, !dbg !838, !alias.scope !834, !noalias !718, !nonnull !23, !align !336, !noundef !23
  %view.val1.i.i.i7 = load i64, ptr %1, align 8, !dbg !838, !alias.scope !834, !noalias !718, !noundef !23
  %_5.i.i.i.i8 = icmp ult i64 %_31.i, %view.val1.i.i.i7, !dbg !839
  tail call void @llvm.assume(i1 %_5.i.i.i.i8), !dbg !843
  %_4.i.i.i.i9 = getelementptr inbounds nuw i64, ptr %view.val.i.i.i6, i64 %_31.i, !dbg !844
  br label %_RNvXs_NtNtNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row5types7element5tuple7indexedINtB4_15ArgColumnSourcexENtNtNtCslhTiPzLr6QZ_14vortex_compute12lane_kernels6source13IndexedSource13get_uncheckedCsfoTid9TleBG_17row_fn_bool_retry.exit.i.i10, !dbg !845

_RNvXs_NtNtNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row5types7element5tuple7indexedINtB4_15ArgColumnSourcexENtNtNtCslhTiPzLr6QZ_14vortex_compute12lane_kernels6source13IndexedSource13get_uncheckedCsfoTid9TleBG_17row_fn_bool_retry.exit.i.i10: ; preds = %bb3.i.i.i4, %bb2.i.i.i26
  %_0.sroa.0.0.in.i.i.i11 = phi ptr [ %_6.i.i.i27, %bb2.i.i.i26 ], [ %_4.i.i.i.i9, %bb3.i.i.i4 ]
  %_0.sroa.0.0.i.i.i12 = load i64, ptr %_0.sroa.0.0.in.i.i.i11, align 8, !dbg !846, !noalias !847, !noundef !23
  tail call void @llvm.experimental.noalias.scope.decl(metadata !848), !dbg !830
  %_3.i1.i.i14 = load i64, ptr %_6.i.i13, align 8, !dbg !851, !range !415, !alias.scope !853, !noalias !718, !noundef !23
  %19 = trunc nuw i64 %_3.i1.i.i14 to i1, !dbg !854
  br i1 %19, label %bb2.i10.i.i24, label %bb3.i2.i.i15, !dbg !854

bb2.i10.i.i24:                                    ; preds = %_RNvXs_NtNtNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row5types7element5tuple7indexedINtB4_15ArgColumnSourcexENtNtNtCslhTiPzLr6QZ_14vortex_compute12lane_kernels6source13IndexedSource13get_uncheckedCsfoTid9TleBG_17row_fn_bool_retry.exit.i.i10
  %_6.i11.i.i25 = load ptr, ptr %2, align 8, !dbg !855, !alias.scope !853, !noalias !718, !nonnull !23, !align !336, !noundef !23
  br label %_RNCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB8_7visitor5retry21ExecuteDenseWithRetryINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB1X_EENtNtB2b_11row_visitor10RowVisitor19visit_deferred_boolB1Q_bKB1X_NCINvXB2U_B2R_NtNtB8_6row_fn5RowFn8dispatchB26_E0NCB4M_s_0E0NCB22_s_0B5x_E0B2U_.llvm.523936492659896562.exit, !dbg !856

bb3.i2.i.i15:                                     ; preds = %_RNvXs_NtNtNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row5types7element5tuple7indexedINtB4_15ArgColumnSourcexENtNtNtCslhTiPzLr6QZ_14vortex_compute12lane_kernels6source13IndexedSource13get_uncheckedCsfoTid9TleBG_17row_fn_bool_retry.exit.i.i10
  %view.val.i4.i.i17 = load ptr, ptr %view.i3.i.i16, align 8, !dbg !857, !alias.scope !853, !noalias !718, !nonnull !23, !align !336, !noundef !23
  %view.val1.i5.i.i18 = load i64, ptr %2, align 8, !dbg !857, !alias.scope !853, !noalias !718, !noundef !23
  %_5.i.i6.i.i19 = icmp ult i64 %_31.i, %view.val1.i5.i.i18, !dbg !858
  tail call void @llvm.assume(i1 %_5.i.i6.i.i19), !dbg !862
  %_4.i.i7.i.i20 = getelementptr inbounds nuw i64, ptr %view.val.i4.i.i17, i64 %_31.i, !dbg !863
  br label %_RNCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB8_7visitor5retry21ExecuteDenseWithRetryINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB1X_EENtNtB2b_11row_visitor10RowVisitor19visit_deferred_boolB1Q_bKB1X_NCINvXB2U_B2R_NtNtB8_6row_fn5RowFn8dispatchB26_E0NCB4M_s_0E0NCB22_s_0B5x_E0B2U_.llvm.523936492659896562.exit, !dbg !864

_RNCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB8_7visitor5retry21ExecuteDenseWithRetryINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB1X_EENtNtB2b_11row_visitor10RowVisitor19visit_deferred_boolB1Q_bKB1X_NCINvXB2U_B2R_NtNtB8_6row_fn5RowFn8dispatchB26_E0NCB4M_s_0E0NCB22_s_0B5x_E0B2U_.llvm.523936492659896562.exit: ; preds = %bb2.i10.i.i24, %bb3.i2.i.i15
  %_0.sroa.0.0.in.i8.i.i22 = phi ptr [ %_6.i11.i.i25, %bb2.i10.i.i24 ], [ %_4.i.i7.i.i20, %bb3.i2.i.i15 ]
  %_0.sroa.0.0.i9.i.i23 = load i64, ptr %_0.sroa.0.0.in.i8.i.i22, align 8, !dbg !865, !noalias !866, !noundef !23
  %20 = icmp eq i64 %_0.sroa.0.0.i.i.i12, 9223372036854775807, !dbg !867
  %21 = icmp eq i64 %_0.sroa.0.0.i9.i.i23, 9223372036854775807, !dbg !867
  %_6.sroa.0.0.i.i.i = or i1 %20, %21, !dbg !867
  %_5.i.i.i = icmp slt i64 %_0.sroa.0.0.i.i.i12, %_0.sroa.0.0.i9.i.i23, !dbg !870
  %22 = load i8, ptr %_11.i, align 1, !dbg !871, !range !483, !alias.scope !873, !noalias !718, !noundef !23
  %23 = trunc nuw i8 %22 to i1, !dbg !871
  %24 = or i1 %_6.sroa.0.0.i.i.i, %23, !dbg !871
  %25 = zext i1 %24 to i8, !dbg !871
  store i8 %25, ptr %_11.i, align 1, !dbg !871, !alias.scope !873, !noalias !718
  %26 = zext i1 %_5.i.i.i to i8, !dbg !876
  store i8 %26, ptr %iter1.i.sroa.0.0.ptr, align 1, !dbg !876
  %_7.i.i5 = icmp eq i64 %_9.0.i9, 64, !dbg !816
  br i1 %_7.i.i5, label %bb9.i, label %bb8.i, !dbg !721

bb9.i:                                            ; preds = %_RNCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB8_7visitor5retry21ExecuteDenseWithRetryINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB1X_EENtNtB2b_11row_visitor10RowVisitor19visit_deferred_boolB1Q_bKB1X_NCINvXB2U_B2R_NtNtB8_6row_fn5RowFn8dispatchB26_E0NCB4M_s_0E0NCB22_s_0B5x_E0B2U_.llvm.523936492659896562.exit
  %_17.i.i = getelementptr inbounds nuw i8, ptr %iter.i.sroa.0.027, i64 8, !dbg !877
  %_9.0.i = add nuw nsw i64 %iter.i.sroa.7.026, 1, !dbg !879
  %bools.i.val = load <64 x i8>, ptr %bools.i, align 1, !dbg !880, !alias.scope !881
  %.not.i.i.i = icmp ne <64 x i8> %bools.i.val, zeroinitializer, !dbg !884
  store <64 x i1> %.not.i.i.i, ptr %iter.i.sroa.0.027, align 8, !dbg !901
  call void @llvm.lifetime.end.p0(ptr nonnull %bools.i), !dbg !902, !noalias !701
  %_7.i.i = icmp eq ptr %_17.i.i, %_52.i, !dbg !712
  br i1 %_7.i.i, label %bb5.i, label %bb4.i, !dbg !717

_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack23collect_bool_words_withNCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1g_7visitor5retry21ExecuteDenseWithRetryINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB35_EENtNtB3j_11row_visitor10RowVisitor19visit_deferred_boolB2Y_bKB35_NCINvXB43_B40_NtNtB1g_6row_fn5RowFn8dispatchB3e_E0NCB5V_s_0E0NCB3a_s_0B6H_E0NCINvB2_25collect_bool_words_avx512B17_E0EB43_.exit: ; preds = %bb14.i, %bb5.i
  ret void, !dbg !903
}
