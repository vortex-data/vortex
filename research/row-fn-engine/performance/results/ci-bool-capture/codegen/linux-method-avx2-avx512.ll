define hidden void @_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor5retry21ExecuteDenseWithRetryINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB45_B42_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5X_s_0E0NCB3c_s_0B6J_E0EB45_(ptr noalias nofree noundef nonnull writeonly align 8 captures(address) %words.0, i64 noundef range(i64 0, 1152921504606846976) %words.1, i64 noundef %len, ptr noalias nofree noundef align 8 dereferenceable(64) %f) unnamed_addr #3 personality ptr @rust_eh_personality !dbg !695 {
start:
  %offset.i = alloca [8 x i8], align 8
  %bools.i = alloca [64 x i8], align 1
  %f.i = alloca [8 x i8], align 8
  tail call void @llvm.experimental.noalias.scope.decl(metadata !696), !dbg !699
  call void @llvm.lifetime.start.p0(ptr nonnull %f.i)
  store ptr %f, ptr %f.i, align 8, !noalias !700
  %full5.i = lshr i64 %len, 6, !dbg !702
  %remainder.i = and i64 %len, 63, !dbg !705
  %_42.not.i = icmp samesign ugt i64 %full5.i, %words.1
  br i1 %_42.not.i, label %bb22.i, label %bb21.i, !dbg !707, !prof !293

bb22.i:                                           ; preds = %start
; call core::slice::index::slice_index_fail
  tail call void @_RNvNtNtCsc36rpYXAlPq_4core5slice5index16slice_index_fail(i64 noundef 0, i64 noundef %full5.i, i64 noundef range(i64 0, 1152921504606846976) %words.1, ptr noalias nofree noundef readonly align 8 captures(address, read_provenance) dereferenceable(24) @alloc_ebbd31bf42d91f579e6aa57e93569175.llvm.2032604888034560029) #21, !dbg !717, !noalias !696
  unreachable

bb21.i:                                           ; preds = %start
  %_52.i.idx = shl nuw nsw i64 %full5.i, 3, !dbg !718
  %_52.i = getelementptr inbounds nuw i8, ptr %words.0, i64 %_52.i.idx, !dbg !718
  %_7.i.i25 = icmp eq i64 %full5.i, 0, !dbg !727
  br i1 %_7.i.i25, label %bb5.i, label %bb4.i.preheader, !dbg !732

bb4.i.preheader:                                  ; preds = %bb21.i
  %_3.i.i.i = load i64, ptr %f, align 8, !range !359, !alias.scope !733, !noalias !738, !noundef !23
  %0 = trunc nuw i64 %_3.i.i.i to i1
  %_6.i.i = getelementptr inbounds nuw i8, ptr %f, i64 24
  %_3.i1.i.i = load i64, ptr %_6.i.i, align 8, !range !359, !alias.scope !741, !noalias !738, !noundef !23
  %1 = trunc nuw i64 %_3.i1.i.i to i1
  %_11.i.i = getelementptr inbounds nuw i8, ptr %f, i64 56
  %view.i.i.i = getelementptr inbounds nuw i8, ptr %f, i64 8
  %view.val.i.i.i = load ptr, ptr %view.i.i.i, align 8, !nonnull !23, !align !371
  %2 = getelementptr inbounds nuw i8, ptr %f, i64 16
  %view.val1.i.i.i = load i64, ptr %2, align 8
  %_6.i.i.i.cast = inttoptr i64 %view.val1.i.i.i to ptr
  %view.i3.i.i = getelementptr inbounds nuw i8, ptr %f, i64 32
  %view.val.i4.i.i = load ptr, ptr %view.i3.i.i, align 8, !nonnull !23, !align !371
  %3 = getelementptr inbounds nuw i8, ptr %f, i64 40
  %view.val1.i5.i.i = load i64, ptr %3, align 8
  %_6.i11.i.i.cast = inttoptr i64 %view.val1.i5.i.i to ptr
  %_11.i.i.promoted.us.us.pre = load i8, ptr %_11.i.i, align 8, !alias.scope !744, !noalias !738
  %4 = trunc nuw i8 %_11.i.i.promoted.us.us.pre to i1, !dbg !747
  br i1 %0, label %bb4.i.preheader.split.us, label %bb4.i.preheader.split

bb4.i.preheader.split.us:                         ; preds = %bb4.i.preheader
  br i1 %1, label %bb4.i.us.us, label %bb4.i.us

bb4.i.us.us:                                      ; preds = %bb4.i.preheader.split.us, %bb9.i.split.us.us.split.us.us
  %_11.i.i.promoted.us.us = phi i1 [ %14, %bb9.i.split.us.us.split.us.us ], [ %4, %bb4.i.preheader.split.us ]
  %iter.i.sroa.0.027.us.us = phi ptr [ %_17.i.i.us.us, %bb9.i.split.us.us.split.us.us ], [ %words.0, %bb4.i.preheader.split.us ]
  call void @llvm.lifetime.start.p0(ptr nonnull %bools.i), !dbg !756, !noalias !700
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 1 dereferenceable(64) %bools.i, i8 0, i64 64, i1 false), !dbg !757, !noalias !700
  br label %bb8.i.us.us.us.us, !dbg !758

bb8.i.us.us.us.us:                                ; preds = %bb8.i.us.us.us.us, %bb4.i.us.us
  %5 = phi i1 [ %_11.i.i.promoted.us.us, %bb4.i.us.us ], [ %14, %bb8.i.us.us.us.us ]
  %iter1.i.sroa.7.024.us.us.us.us = phi i64 [ 0, %bb4.i.us.us ], [ %_9.0.i9.us.us.us.us.1, %bb8.i.us.us.us.us ]
  %iter1.i.sroa.0.0.ptr.us.us.us.us = getelementptr inbounds nuw i8, ptr %bools.i, i64 %iter1.i.sroa.7.024.us.us.us.us, !dbg !761
  tail call void @llvm.experimental.noalias.scope.decl(metadata !764), !dbg !765
  tail call void @llvm.experimental.noalias.scope.decl(metadata !766), !dbg !767
  %_0.sroa.0.0.i.i.i.us.us.us.us = load i64, ptr %_6.i.i.i.cast, align 8, !dbg !769, !noalias !771, !noundef !23
  tail call void @llvm.experimental.noalias.scope.decl(metadata !772), !dbg !767
  %_0.sroa.0.0.i9.i.i.us.us.us.us = load i64, ptr %_6.i11.i.i.cast, align 8, !dbg !773, !noalias !775, !noundef !23
  %6 = icmp eq i64 %_0.sroa.0.0.i.i.i.us.us.us.us, 9223372036854775807, !dbg !776
  %7 = icmp eq i64 %_0.sroa.0.0.i9.i.i.us.us.us.us, 9223372036854775807, !dbg !776
  %_6.sroa.0.0.i.i.i.us.us.us.us = or i1 %6, %7, !dbg !776
  %_5.i.i.i.us.us.us.us = icmp slt i64 %_0.sroa.0.0.i.i.i.us.us.us.us, %_0.sroa.0.0.i9.i.i.us.us.us.us, !dbg !779
  %8 = or i1 %_6.sroa.0.0.i.i.i.us.us.us.us, %5, !dbg !747
  %9 = zext i1 %8 to i8, !dbg !747
  store i8 %9, ptr %_11.i.i, align 8, !dbg !747, !alias.scope !744, !noalias !738
  %10 = zext i1 %_5.i.i.i.us.us.us.us to i8, !dbg !780
  store i8 %10, ptr %iter1.i.sroa.0.0.ptr.us.us.us.us, align 1, !dbg !780
  %11 = getelementptr inbounds nuw i8, ptr %bools.i, i64 %iter1.i.sroa.7.024.us.us.us.us, !dbg !761
  %iter1.i.sroa.0.0.ptr.us.us.us.us.1 = getelementptr inbounds nuw i8, ptr %11, i64 1, !dbg !761
  %_9.0.i9.us.us.us.us.1 = add nuw nsw i64 %iter1.i.sroa.7.024.us.us.us.us, 2, !dbg !781
  %_0.sroa.0.0.i.i.i.us.us.us.us.1 = load i64, ptr %_6.i.i.i.cast, align 8, !dbg !769, !noalias !784, !noundef !23
  %_0.sroa.0.0.i9.i.i.us.us.us.us.1 = load i64, ptr %_6.i11.i.i.cast, align 8, !dbg !773, !noalias !787, !noundef !23
  %12 = icmp eq i64 %_0.sroa.0.0.i.i.i.us.us.us.us.1, 9223372036854775807, !dbg !776
  %13 = icmp eq i64 %_0.sroa.0.0.i9.i.i.us.us.us.us.1, 9223372036854775807, !dbg !776
  %_6.sroa.0.0.i.i.i.us.us.us.us.1 = or i1 %12, %13, !dbg !776
  %_5.i.i.i.us.us.us.us.1 = icmp slt i64 %_0.sroa.0.0.i.i.i.us.us.us.us.1, %_0.sroa.0.0.i9.i.i.us.us.us.us.1, !dbg !779
  %14 = or i1 %_6.sroa.0.0.i.i.i.us.us.us.us.1, %8, !dbg !747
  %15 = zext i1 %14 to i8, !dbg !747
  store i8 %15, ptr %_11.i.i, align 8, !dbg !747, !alias.scope !744, !noalias !738
  %16 = zext i1 %_5.i.i.i.us.us.us.us.1 to i8, !dbg !780
  store i8 %16, ptr %iter1.i.sroa.0.0.ptr.us.us.us.us.1, align 1, !dbg !780
  %_7.i.i5.us.us.us.us.1 = icmp eq i64 %_9.0.i9.us.us.us.us.1, 64, !dbg !761
  br i1 %_7.i.i5.us.us.us.us.1, label %bb9.i.split.us.us.split.us.us, label %bb8.i.us.us.us.us, !dbg !758

bb9.i.split.us.us.split.us.us:                    ; preds = %bb8.i.us.us.us.us
  %_17.i.i.us.us = getelementptr inbounds nuw i8, ptr %iter.i.sroa.0.027.us.us, i64 8, !dbg !789
  %bools.i.val.us.us = load <64 x i8>, ptr %bools.i, align 1, !dbg !791, !alias.scope !792
  %.not.i.i.i.us.us = icmp ne <64 x i8> %bools.i.val.us.us, zeroinitializer, !dbg !795
  store <64 x i1> %.not.i.i.i.us.us, ptr %iter.i.sroa.0.027.us.us, align 8, !dbg !812
  call void @llvm.lifetime.end.p0(ptr nonnull %bools.i), !dbg !813, !noalias !700
  %_7.i.i.us.us = icmp eq ptr %_17.i.i.us.us, %_52.i, !dbg !727
  br i1 %_7.i.i.us.us, label %bb5.i, label %bb4.i.us.us, !dbg !732

bb4.i.us:                                         ; preds = %bb4.i.preheader.split.us, %bb9.i.split.us.us.split
  %_11.i.i.promoted.us = phi i1 [ %20, %bb9.i.split.us.us.split ], [ %4, %bb4.i.preheader.split.us ]
  %iter.i.sroa.0.027.us = phi ptr [ %_17.i.i.us, %bb9.i.split.us.us.split ], [ %words.0, %bb4.i.preheader.split.us ]
  %iter.i.sroa.7.026.us = phi i64 [ %_9.0.i.us, %bb9.i.split.us.us.split ], [ 0, %bb4.i.preheader.split.us ]
  %offset2.i.us = shl i64 %iter.i.sroa.7.026.us, 6, !dbg !814
  call void @llvm.lifetime.start.p0(ptr nonnull %bools.i), !dbg !756, !noalias !700
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 1 dereferenceable(64) %bools.i, i8 0, i64 64, i1 false), !dbg !757, !noalias !700
  br label %bb8.i.us.us, !dbg !758

bb8.i.us.us:                                      ; preds = %bb8.i.us.us, %bb4.i.us
  %17 = phi i1 [ %_11.i.i.promoted.us, %bb4.i.us ], [ %20, %bb8.i.us.us ]
  %iter1.i.sroa.7.024.us.us = phi i64 [ 0, %bb4.i.us ], [ %_9.0.i9.us.us, %bb8.i.us.us ]
  %iter1.i.sroa.0.0.ptr.us.us = getelementptr inbounds nuw i8, ptr %bools.i, i64 %iter1.i.sroa.7.024.us.us, !dbg !761
  %_9.0.i9.us.us = add nuw nsw i64 %iter1.i.sroa.7.024.us.us, 1, !dbg !781
  %_31.i.us.us = add nuw nsw i64 %iter1.i.sroa.7.024.us.us, %offset2.i.us, !dbg !815
  tail call void @llvm.experimental.noalias.scope.decl(metadata !764), !dbg !765
  tail call void @llvm.experimental.noalias.scope.decl(metadata !766), !dbg !767
  %_0.sroa.0.0.i.i.i.us.us = load i64, ptr %_6.i.i.i.cast, align 8, !dbg !769, !noalias !771, !noundef !23
  tail call void @llvm.experimental.noalias.scope.decl(metadata !772), !dbg !767
  %_5.i.i6.i.i.us.us = icmp ult i64 %_31.i.us.us, %view.val1.i5.i.i, !dbg !816
  tail call void @llvm.assume(i1 %_5.i.i6.i.i.us.us), !dbg !820
  %_4.i.i7.i.i.us.us = getelementptr inbounds nuw i64, ptr %view.val.i4.i.i, i64 %_31.i.us.us, !dbg !821
  %_0.sroa.0.0.i9.i.i.us.us = load i64, ptr %_4.i.i7.i.i.us.us, align 8, !dbg !773, !noalias !775, !noundef !23
  %18 = icmp eq i64 %_0.sroa.0.0.i.i.i.us.us, 9223372036854775807, !dbg !776
  %19 = icmp eq i64 %_0.sroa.0.0.i9.i.i.us.us, 9223372036854775807, !dbg !776
  %_6.sroa.0.0.i.i.i.us.us = or i1 %18, %19, !dbg !776
  %_5.i.i.i.us.us = icmp slt i64 %_0.sroa.0.0.i.i.i.us.us, %_0.sroa.0.0.i9.i.i.us.us, !dbg !779
  %20 = or i1 %_6.sroa.0.0.i.i.i.us.us, %17, !dbg !747
  %21 = zext i1 %20 to i8, !dbg !747
  store i8 %21, ptr %_11.i.i, align 8, !dbg !747, !alias.scope !744, !noalias !738
  %22 = zext i1 %_5.i.i.i.us.us to i8, !dbg !780
  store i8 %22, ptr %iter1.i.sroa.0.0.ptr.us.us, align 1, !dbg !780
  %_7.i.i5.us.us = icmp eq i64 %_9.0.i9.us.us, 64, !dbg !761
  br i1 %_7.i.i5.us.us, label %bb9.i.split.us.us.split, label %bb8.i.us.us, !dbg !758

bb9.i.split.us.us.split:                          ; preds = %bb8.i.us.us
  %_17.i.i.us = getelementptr inbounds nuw i8, ptr %iter.i.sroa.0.027.us, i64 8, !dbg !789
  %_9.0.i.us = add nuw nsw i64 %iter.i.sroa.7.026.us, 1, !dbg !822
  %bools.i.val.us = load <64 x i8>, ptr %bools.i, align 1, !dbg !791, !alias.scope !792
  %.not.i.i.i.us = icmp ne <64 x i8> %bools.i.val.us, zeroinitializer, !dbg !795
  store <64 x i1> %.not.i.i.i.us, ptr %iter.i.sroa.0.027.us, align 8, !dbg !812
  call void @llvm.lifetime.end.p0(ptr nonnull %bools.i), !dbg !813, !noalias !700
  %_7.i.i.us = icmp eq ptr %_17.i.i.us, %_52.i, !dbg !727
  br i1 %_7.i.i.us, label %bb5.i, label %bb4.i.us, !dbg !732

bb4.i.preheader.split:                            ; preds = %bb4.i.preheader
  br i1 %1, label %bb4.i.us14, label %bb4.i

bb4.i.us14:                                       ; preds = %bb4.i.preheader.split, %bb9.i.split.split.us.us
  %_11.i.i.promoted.us18 = phi i1 [ %26, %bb9.i.split.split.us.us ], [ %4, %bb4.i.preheader.split ]
  %iter.i.sroa.0.027.us15 = phi ptr [ %_17.i.i.us19, %bb9.i.split.split.us.us ], [ %words.0, %bb4.i.preheader.split ]
  %iter.i.sroa.7.026.us16 = phi i64 [ %_9.0.i.us20, %bb9.i.split.split.us.us ], [ 0, %bb4.i.preheader.split ]
  %offset2.i.us17 = shl i64 %iter.i.sroa.7.026.us16, 6, !dbg !814
  call void @llvm.lifetime.start.p0(ptr nonnull %bools.i), !dbg !756, !noalias !700
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 1 dereferenceable(64) %bools.i, i8 0, i64 64, i1 false), !dbg !757, !noalias !700
  br label %bb8.i.us1.us, !dbg !758

bb8.i.us1.us:                                     ; preds = %bb8.i.us1.us, %bb4.i.us14
  %23 = phi i1 [ %_11.i.i.promoted.us18, %bb4.i.us14 ], [ %26, %bb8.i.us1.us ]
  %iter1.i.sroa.7.024.us2.us = phi i64 [ 0, %bb4.i.us14 ], [ %_9.0.i9.us4.us, %bb8.i.us1.us ]
  %iter1.i.sroa.0.0.ptr.us3.us = getelementptr inbounds nuw i8, ptr %bools.i, i64 %iter1.i.sroa.7.024.us2.us, !dbg !761
  %_9.0.i9.us4.us = add nuw nsw i64 %iter1.i.sroa.7.024.us2.us, 1, !dbg !781
  %_31.i.us5.us = add nuw nsw i64 %iter1.i.sroa.7.024.us2.us, %offset2.i.us17, !dbg !815
  tail call void @llvm.experimental.noalias.scope.decl(metadata !764), !dbg !765
  tail call void @llvm.experimental.noalias.scope.decl(metadata !766), !dbg !767
  %_5.i.i.i.i.us.us = icmp ult i64 %_31.i.us5.us, %view.val1.i.i.i, !dbg !823
  tail call void @llvm.assume(i1 %_5.i.i.i.i.us.us), !dbg !827
  %_4.i.i.i.i.us.us = getelementptr inbounds nuw i64, ptr %view.val.i.i.i, i64 %_31.i.us5.us, !dbg !828
  %_0.sroa.0.0.i.i.i.us6.us = load i64, ptr %_4.i.i.i.i.us.us, align 8, !dbg !769, !noalias !771, !noundef !23
  tail call void @llvm.experimental.noalias.scope.decl(metadata !772), !dbg !767
  %_0.sroa.0.0.i9.i.i.us10.us = load i64, ptr %_6.i11.i.i.cast, align 8, !dbg !773, !noalias !775, !noundef !23
  %24 = icmp eq i64 %_0.sroa.0.0.i.i.i.us6.us, 9223372036854775807, !dbg !776
  %25 = icmp eq i64 %_0.sroa.0.0.i9.i.i.us10.us, 9223372036854775807, !dbg !776
  %_6.sroa.0.0.i.i.i.us11.us = or i1 %24, %25, !dbg !776
  %_5.i.i.i.us12.us = icmp slt i64 %_0.sroa.0.0.i.i.i.us6.us, %_0.sroa.0.0.i9.i.i.us10.us, !dbg !779
  %26 = or i1 %_6.sroa.0.0.i.i.i.us11.us, %23, !dbg !747
  %27 = zext i1 %26 to i8, !dbg !747
  store i8 %27, ptr %_11.i.i, align 8, !dbg !747, !alias.scope !744, !noalias !738
  %28 = zext i1 %_5.i.i.i.us12.us to i8, !dbg !780
  store i8 %28, ptr %iter1.i.sroa.0.0.ptr.us3.us, align 1, !dbg !780
  %_7.i.i5.us13.us = icmp eq i64 %_9.0.i9.us4.us, 64, !dbg !761
  br i1 %_7.i.i5.us13.us, label %bb9.i.split.split.us.us, label %bb8.i.us1.us, !dbg !758

bb9.i.split.split.us.us:                          ; preds = %bb8.i.us1.us
  %_17.i.i.us19 = getelementptr inbounds nuw i8, ptr %iter.i.sroa.0.027.us15, i64 8, !dbg !789
  %_9.0.i.us20 = add nuw nsw i64 %iter.i.sroa.7.026.us16, 1, !dbg !822
  %bools.i.val.us21 = load <64 x i8>, ptr %bools.i, align 1, !dbg !791, !alias.scope !792
  %.not.i.i.i.us22 = icmp ne <64 x i8> %bools.i.val.us21, zeroinitializer, !dbg !795
  store <64 x i1> %.not.i.i.i.us22, ptr %iter.i.sroa.0.027.us15, align 8, !dbg !812
  call void @llvm.lifetime.end.p0(ptr nonnull %bools.i), !dbg !813, !noalias !700
  %_7.i.i.us23 = icmp eq ptr %_17.i.i.us19, %_52.i, !dbg !727
  br i1 %_7.i.i.us23, label %bb5.i, label %bb4.i.us14, !dbg !732

bb4.i:                                            ; preds = %bb4.i.preheader.split, %bb9.i.split.split
  %_11.i.i.promoted25 = phi i1 [ %34, %bb9.i.split.split ], [ %4, %bb4.i.preheader.split ]
  %iter.i.sroa.0.027 = phi ptr [ %_17.i.i, %bb9.i.split.split ], [ %words.0, %bb4.i.preheader.split ]
  %iter.i.sroa.7.026 = phi i64 [ %_9.0.i, %bb9.i.split.split ], [ 0, %bb4.i.preheader.split ]
  %offset2.i = shl i64 %iter.i.sroa.7.026, 6, !dbg !814
  call void @llvm.lifetime.start.p0(ptr nonnull %bools.i), !dbg !756, !noalias !700
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 1 dereferenceable(64) %bools.i, i8 0, i64 64, i1 false), !dbg !757, !noalias !700
  br label %bb8.i, !dbg !758

bb5.i:                                            ; preds = %bb9.i.split.split, %bb9.i.split.split.us.us, %bb9.i.split.us.us.split, %bb9.i.split.us.us.split.us.us, %bb21.i
  %29 = icmp eq i64 %remainder.i, 0, !dbg !829
  br i1 %29, label %_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack23collect_bool_words_withNCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1g_7visitor5retry21ExecuteDenseWithRetryINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB35_EENtNtB3j_11row_visitor10RowVisitor19visit_deferred_boolB2Y_bKB35_NCINvXB43_B40_NtNtB1g_6row_fn5RowFn8dispatchB3e_E0NCB5V_s_0E0NCB3a_s_0B6H_E0NCINvB2_25collect_bool_words_avx512B17_E0EB43_.exit, label %bb12.i, !dbg !829

bb12.i:                                           ; preds = %bb5.i
  call void @llvm.lifetime.start.p0(ptr nonnull %offset.i), !dbg !830, !noalias !700
  %30 = and i64 %len, -64, !dbg !831
  store i64 %30, ptr %offset.i, align 8, !dbg !831, !noalias !700
  %_37.i = call fastcc noundef i64 @_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack24collect_bool_word_scalarNCINvB2_23collect_bool_words_withNCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1O_7visitor5retry21ExecuteDenseWithRetryINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB3D_EENtNtB3R_11row_visitor10RowVisitor19visit_deferred_boolB3w_bKB3D_NCINvXB4B_B4y_NtNtB1O_6row_fn5RowFn8dispatchB3M_E0NCB6t_s_0E0NCB3I_s_0B7f_E0NCINvB2_25collect_bool_words_avx512B1F_E0E0EB4B_.llvm.2032604888034560029(i64 noundef %remainder.i, ptr noalias nofree noundef align 8 dereferenceable(8) %f.i, ptr noalias nofree noundef readonly align 8 captures(address, read_provenance) dereferenceable(8) %offset.i), !dbg !832
  %_41.i = icmp samesign ult i64 %full5.i, %words.1, !dbg !834
  br i1 %_41.i, label %bb14.i, label %panic.i, !dbg !834

bb14.i:                                           ; preds = %bb12.i
  store i64 %_37.i, ptr %_52.i, align 8, !dbg !834, !alias.scope !696, !noalias !835
  call void @llvm.lifetime.end.p0(ptr nonnull %offset.i), !dbg !836, !noalias !700
  br label %_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack23collect_bool_words_withNCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1g_7visitor5retry21ExecuteDenseWithRetryINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB35_EENtNtB3j_11row_visitor10RowVisitor19visit_deferred_boolB2Y_bKB35_NCINvXB43_B40_NtNtB1g_6row_fn5RowFn8dispatchB3e_E0NCB5V_s_0E0NCB3a_s_0B6H_E0NCINvB2_25collect_bool_words_avx512B17_E0EB43_.exit, !dbg !837

panic.i:                                          ; preds = %bb12.i
; call core::panicking::panic_bounds_check
  tail call void @_RNvNtCsc36rpYXAlPq_4core9panicking18panic_bounds_check(i64 noundef %full5.i, i64 noundef range(i64 0, 1152921504606846976) %words.1, ptr noalias nofree noundef readonly align 8 captures(address, read_provenance) dereferenceable(24) @alloc_1b2922caae1b461da04857d3d02eae5b.llvm.2032604888034560029) #21, !dbg !834
  unreachable

bb8.i:                                            ; preds = %bb8.i, %bb4.i
  %31 = phi i1 [ %_11.i.i.promoted25, %bb4.i ], [ %34, %bb8.i ]
  %iter1.i.sroa.7.024 = phi i64 [ 0, %bb4.i ], [ %_9.0.i9, %bb8.i ]
  %iter1.i.sroa.0.0.ptr = getelementptr inbounds nuw i8, ptr %bools.i, i64 %iter1.i.sroa.7.024, !dbg !761
  %_9.0.i9 = add nuw nsw i64 %iter1.i.sroa.7.024, 1, !dbg !781
  %_31.i = add nuw nsw i64 %iter1.i.sroa.7.024, %offset2.i, !dbg !815
  tail call void @llvm.experimental.noalias.scope.decl(metadata !764), !dbg !765
  tail call void @llvm.experimental.noalias.scope.decl(metadata !766), !dbg !767
  %_5.i.i.i.i = icmp ult i64 %_31.i, %view.val1.i.i.i, !dbg !823
  tail call void @llvm.assume(i1 %_5.i.i.i.i), !dbg !827
  %_4.i.i.i.i = getelementptr inbounds nuw i64, ptr %view.val.i.i.i, i64 %_31.i, !dbg !828
  %_0.sroa.0.0.i.i.i = load i64, ptr %_4.i.i.i.i, align 8, !dbg !769, !noalias !771, !noundef !23
  tail call void @llvm.experimental.noalias.scope.decl(metadata !772), !dbg !767
  %_5.i.i6.i.i = icmp ult i64 %_31.i, %view.val1.i5.i.i, !dbg !816
  tail call void @llvm.assume(i1 %_5.i.i6.i.i), !dbg !820
  %_4.i.i7.i.i = getelementptr inbounds nuw i64, ptr %view.val.i4.i.i, i64 %_31.i, !dbg !821
  %_0.sroa.0.0.i9.i.i = load i64, ptr %_4.i.i7.i.i, align 8, !dbg !773, !noalias !775, !noundef !23
  %32 = icmp eq i64 %_0.sroa.0.0.i.i.i, 9223372036854775807, !dbg !776
  %33 = icmp eq i64 %_0.sroa.0.0.i9.i.i, 9223372036854775807, !dbg !776
  %_6.sroa.0.0.i.i.i = or i1 %32, %33, !dbg !776
  %_5.i.i.i = icmp slt i64 %_0.sroa.0.0.i.i.i, %_0.sroa.0.0.i9.i.i, !dbg !779
  %34 = or i1 %_6.sroa.0.0.i.i.i, %31, !dbg !747
  %35 = zext i1 %34 to i8, !dbg !747
  store i8 %35, ptr %_11.i.i, align 8, !dbg !747, !alias.scope !744, !noalias !738
  %36 = zext i1 %_5.i.i.i to i8, !dbg !780
  store i8 %36, ptr %iter1.i.sroa.0.0.ptr, align 1, !dbg !780
  %_7.i.i5 = icmp eq i64 %_9.0.i9, 64, !dbg !761
  br i1 %_7.i.i5, label %bb9.i.split.split, label %bb8.i, !dbg !758

bb9.i.split.split:                                ; preds = %bb8.i
  %_17.i.i = getelementptr inbounds nuw i8, ptr %iter.i.sroa.0.027, i64 8, !dbg !789
  %_9.0.i = add nuw nsw i64 %iter.i.sroa.7.026, 1, !dbg !822
  %bools.i.val = load <64 x i8>, ptr %bools.i, align 1, !dbg !791, !alias.scope !792
  %.not.i.i.i = icmp ne <64 x i8> %bools.i.val, zeroinitializer, !dbg !795
  store <64 x i1> %.not.i.i.i, ptr %iter.i.sroa.0.027, align 8, !dbg !812
  call void @llvm.lifetime.end.p0(ptr nonnull %bools.i), !dbg !813, !noalias !700
  %_7.i.i = icmp eq ptr %_17.i.i, %_52.i, !dbg !727
  br i1 %_7.i.i, label %bb5.i, label %bb4.i, !dbg !732

_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack23collect_bool_words_withNCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1g_7visitor5retry21ExecuteDenseWithRetryINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB35_EENtNtB3j_11row_visitor10RowVisitor19visit_deferred_boolB2Y_bKB35_NCINvXB43_B40_NtNtB1g_6row_fn5RowFn8dispatchB3e_E0NCB5V_s_0E0NCB3a_s_0B6H_E0NCINvB2_25collect_bool_words_avx512B17_E0EB43_.exit: ; preds = %bb14.i, %bb5.i
  call void @llvm.lifetime.end.p0(ptr nonnull %f.i), !dbg !838
  ret void, !dbg !839
}
