define hidden void @_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor5retry21ExecuteDenseWithRetryINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB45_B42_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5X_s_0E0NCB3c_s_0B6J_E0EB45_(ptr noalias nofree noundef nonnull writeonly align 8 captures(address) %words.0, i64 noundef range(i64 0, 1152921504606846976) %words.1, i64 noundef %len, ptr noalias nofree noundef align 8 captures(none) dereferenceable(64) %f) unnamed_addr #3 personality ptr @rust_eh_personality !dbg !634 {
start:
  tail call void @llvm.experimental.noalias.scope.decl(metadata !635), !dbg !638
  %full5.i = lshr i64 %len, 6, !dbg !639
  %remainder.i = and i64 %len, 63, !dbg !642
  %_42.not.i = icmp samesign ugt i64 %full5.i, %words.1
  br i1 %_42.not.i, label %bb22.i, label %bb21.i, !dbg !644, !prof !313

bb22.i:                                           ; preds = %start
; call core::slice::index::slice_index_fail
  tail call void @_RNvNtNtCsc36rpYXAlPq_4core5slice5index16slice_index_fail(i64 noundef 0, i64 noundef %full5.i, i64 noundef range(i64 0, 1152921504606846976) %words.1, ptr noalias nofree noundef readonly align 8 captures(address, read_provenance) dereferenceable(24) @alloc_ebbd31bf42d91f579e6aa57e93569175) #21, !dbg !654, !noalias !635
  unreachable

bb21.i:                                           ; preds = %start
  %_52.i.idx = shl nuw nsw i64 %full5.i, 3, !dbg !655
  %_52.i = getelementptr inbounds nuw i8, ptr %words.0, i64 %_52.i.idx, !dbg !655
  %_7.i.i54 = icmp eq i64 %full5.i, 0, !dbg !664
  br i1 %_7.i.i54, label %bb5.i, label %bb4.i.lr.ph, !dbg !669

bb4.i.lr.ph:                                      ; preds = %bb21.i
  %_3.i.i.i = load i64, ptr %f, align 8, !range !351, !alias.scope !670, !noalias !675, !noundef !23
  %0 = trunc nuw i64 %_3.i.i.i to i1
  %_6.i.i = getelementptr inbounds nuw i8, ptr %f, i64 24
  %_3.i1.i.i = load i64, ptr %_6.i.i, align 8, !range !351, !alias.scope !678, !noalias !675, !noundef !23
  %1 = trunc nuw i64 %_3.i1.i.i to i1
  %_11.i = getelementptr inbounds nuw i8, ptr %f, i64 56
  %view.i.i.i = getelementptr inbounds nuw i8, ptr %f, i64 8
  %view.val.i.i.i = load ptr, ptr %view.i.i.i, align 8, !nonnull !23, !align !363
  %view.i3.i.i = getelementptr inbounds nuw i8, ptr %f, i64 32
  %view.val.i4.i.i = load ptr, ptr %view.i3.i.i, align 8, !nonnull !23, !align !363
  %2 = getelementptr inbounds nuw i8, ptr %f, i64 40
  %view.val1.i5.i.i = load i64, ptr %2, align 8
  %_6.i11.i.i.cast = inttoptr i64 %view.val1.i5.i.i to ptr
  %_11.i.promoted57 = load i8, ptr %_11.i, align 8, !alias.scope !681, !noalias !675
  br i1 %0, label %bb4.i.lr.ph.split.us, label %bb4.i.lr.ph.split

bb4.i.lr.ph.split.us:                             ; preds = %bb4.i.lr.ph
  %3 = getelementptr inbounds nuw i8, ptr %f, i64 16
  %view.val1.i.i.i = load i64, ptr %3, align 8
  %4 = inttoptr i64 %view.val1.i.i.i to ptr
  %_0.sroa.0.0.i.i.i.us.us = load i64, ptr %4, align 8, !noalias !684, !noundef !23
  %5 = icmp eq i64 %_0.sroa.0.0.i.i.i.us.us, 9223372036854775807
  br i1 %1, label %bb4.i.lr.ph.split.us.split.us, label %bb4.i.us.preheader

bb4.i.us.preheader:                               ; preds = %bb4.i.lr.ph.split.us
  %6 = trunc nuw i8 %_11.i.promoted57 to i1, !dbg !685
  %broadcast.splatinsert113 = insertelement <8 x i1> poison, i1 %5, i64 0
  %broadcast.splat114 = shufflevector <8 x i1> %broadcast.splatinsert113, <8 x i1> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert111 = insertelement <8 x i64> poison, i64 %_0.sroa.0.0.i.i.i.us.us, i64 0
  %broadcast.splat112 = shufflevector <8 x i64> %broadcast.splatinsert111, <8 x i64> poison, <8 x i32> zeroinitializer
  %7 = getelementptr inbounds nuw i8, ptr %view.val.i4.i.i, i64 256
  br label %bb4.i.us, !dbg !693

bb4.i.lr.ph.split.us.split.us:                    ; preds = %bb4.i.lr.ph.split.us
  %_0.sroa.0.0.i9.i.i.us.us.us.us = load i64, ptr %_6.i11.i.i.cast, align 8, !noalias !696, !noundef !23
  %_5.i.i.i.us.us.us.us = icmp slt i64 %_0.sroa.0.0.i.i.i.us.us, %_0.sroa.0.0.i9.i.i.us.us.us.us
  %8 = select i1 %_5.i.i.i.us.us.us.us, i512 1334440654591915542993625911497130241, i512 0
  %9 = shl nuw nsw i512 %8, 128
  %10 = or disjoint i512 %9, %8
  %11 = shl nuw nsw i512 %10, 256
  %12 = or disjoint i512 %11, %10
  %13 = bitcast i512 %12 to <64 x i8>
  %.not.i.i.i.us.us = icmp ne <64 x i8> %13, zeroinitializer
  %14 = add nsw i64 %_52.i.idx, -8, !dbg !669
  %15 = lshr exact i64 %14, 3, !dbg !669
  %16 = add nuw nsw i64 %15, 1, !dbg !669
  %xtraiter = and i64 %16, 7, !dbg !669
  %17 = and i64 %14, 56, !dbg !669
  %lcmp.mod.not = icmp eq i64 %17, 56, !dbg !669
  br i1 %lcmp.mod.not, label %bb4.i.us.us.prol.loopexit, label %bb4.i.us.us.prol, !dbg !669

bb4.i.us.us.prol:                                 ; preds = %bb4.i.lr.ph.split.us.split.us, %bb4.i.us.us.prol
  %iter.i.sroa.0.056.us.us.prol = phi ptr [ %_17.i.i.us.us.prol, %bb4.i.us.us.prol ], [ %words.0, %bb4.i.lr.ph.split.us.split.us ]
  %prol.iter = phi i64 [ %prol.iter.next, %bb4.i.us.us.prol ], [ 0, %bb4.i.lr.ph.split.us.split.us ]
  %_17.i.i.us.us.prol = getelementptr inbounds nuw i8, ptr %iter.i.sroa.0.056.us.us.prol, i64 8, !dbg !697
  store <64 x i1> %.not.i.i.i.us.us, ptr %iter.i.sroa.0.056.us.us.prol, align 8, !dbg !699
  %prol.iter.next = add i64 %prol.iter, 1, !dbg !669
  %prol.iter.cmp.not = icmp eq i64 %prol.iter.next, %xtraiter, !dbg !669
  br i1 %prol.iter.cmp.not, label %bb4.i.us.us.prol.loopexit, label %bb4.i.us.us.prol, !dbg !669, !llvm.loop !700

bb4.i.us.us.prol.loopexit:                        ; preds = %bb4.i.us.us.prol, %bb4.i.lr.ph.split.us.split.us
  %iter.i.sroa.0.056.us.us.unr = phi ptr [ %words.0, %bb4.i.lr.ph.split.us.split.us ], [ %_17.i.i.us.us.prol, %bb4.i.us.us.prol ]
  %18 = icmp ult i64 %14, 56, !dbg !669
  br i1 %18, label %bb1.i.bb5.i_crit_edge.loopexit, label %bb4.i.us.us, !dbg !669

bb4.i.us.us:                                      ; preds = %bb4.i.us.us.prol.loopexit, %bb4.i.us.us
  %iter.i.sroa.0.056.us.us = phi ptr [ %_17.i.i.us.us.7, %bb4.i.us.us ], [ %iter.i.sroa.0.056.us.us.unr, %bb4.i.us.us.prol.loopexit ]
  %_17.i.i.us.us = getelementptr inbounds nuw i8, ptr %iter.i.sroa.0.056.us.us, i64 8, !dbg !697
  store <64 x i1> %.not.i.i.i.us.us, ptr %iter.i.sroa.0.056.us.us, align 8, !dbg !699
  %_17.i.i.us.us.1 = getelementptr inbounds nuw i8, ptr %iter.i.sroa.0.056.us.us, i64 16, !dbg !697
  store <64 x i1> %.not.i.i.i.us.us, ptr %_17.i.i.us.us, align 8, !dbg !699
  %_17.i.i.us.us.2 = getelementptr inbounds nuw i8, ptr %iter.i.sroa.0.056.us.us, i64 24, !dbg !697
  store <64 x i1> %.not.i.i.i.us.us, ptr %_17.i.i.us.us.1, align 8, !dbg !699
  %_17.i.i.us.us.3 = getelementptr inbounds nuw i8, ptr %iter.i.sroa.0.056.us.us, i64 32, !dbg !697
  store <64 x i1> %.not.i.i.i.us.us, ptr %_17.i.i.us.us.2, align 8, !dbg !699
  %_17.i.i.us.us.4 = getelementptr inbounds nuw i8, ptr %iter.i.sroa.0.056.us.us, i64 40, !dbg !697
  store <64 x i1> %.not.i.i.i.us.us, ptr %_17.i.i.us.us.3, align 8, !dbg !699
  %_17.i.i.us.us.5 = getelementptr inbounds nuw i8, ptr %iter.i.sroa.0.056.us.us, i64 48, !dbg !697
  store <64 x i1> %.not.i.i.i.us.us, ptr %_17.i.i.us.us.4, align 8, !dbg !699
  %_17.i.i.us.us.6 = getelementptr inbounds nuw i8, ptr %iter.i.sroa.0.056.us.us, i64 56, !dbg !697
  store <64 x i1> %.not.i.i.i.us.us, ptr %_17.i.i.us.us.5, align 8, !dbg !699
  %_17.i.i.us.us.7 = getelementptr inbounds nuw i8, ptr %iter.i.sroa.0.056.us.us, i64 64, !dbg !697
  store <64 x i1> %.not.i.i.i.us.us, ptr %_17.i.i.us.us.6, align 8, !dbg !699
  %_7.i.i.us.us.7 = icmp eq ptr %_17.i.i.us.us.7, %_52.i, !dbg !664
  br i1 %_7.i.i.us.us.7, label %bb1.i.bb5.i_crit_edge.loopexit, label %bb4.i.us.us, !dbg !669

bb4.i.us:                                         ; preds = %bb4.i.us, %bb4.i.us.preheader
  %.us-phi58.us = phi i1 [ %59, %bb4.i.us ], [ %6, %bb4.i.us.preheader ]
  %iter.i.sroa.0.056.us = phi ptr [ %_17.i.i.us, %bb4.i.us ], [ %words.0, %bb4.i.us.preheader ]
  %iter.i.sroa.7.055.us = phi i64 [ %_9.0.i.us, %bb4.i.us ], [ 0, %bb4.i.us.preheader ]
  %19 = insertelement <8 x i1> <i1 poison, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false>, i1 %.us-phi58.us, i64 0
  tail call void @llvm.experimental.noalias.scope.decl(metadata !702), !dbg !703
  tail call void @llvm.experimental.noalias.scope.decl(metadata !704), !dbg !705
  tail call void @llvm.experimental.noalias.scope.decl(metadata !707), !dbg !705
  %.idx358 = shl i64 %iter.i.sroa.7.055.us, 9, !dbg !708
  %20 = getelementptr inbounds nuw i8, ptr %view.val.i4.i.i, i64 %.idx358, !dbg !708
  %21 = getelementptr inbounds nuw i8, ptr %20, i64 64, !dbg !713
  %22 = getelementptr inbounds nuw i8, ptr %20, i64 128, !dbg !713
  %23 = getelementptr inbounds nuw i8, ptr %20, i64 192, !dbg !713
  %wide.load125 = load <8 x i64>, ptr %20, align 8, !dbg !713, !noalias !696
  %wide.load126 = load <8 x i64>, ptr %21, align 8, !dbg !713, !noalias !696
  %wide.load127 = load <8 x i64>, ptr %22, align 8, !dbg !713, !noalias !696
  %wide.load128 = load <8 x i64>, ptr %23, align 8, !dbg !713, !noalias !696
  %24 = icmp eq <8 x i64> %wide.load125, splat (i64 9223372036854775807), !dbg !714
  %25 = icmp eq <8 x i64> %wide.load126, splat (i64 9223372036854775807), !dbg !714
  %26 = icmp eq <8 x i64> %wide.load127, splat (i64 9223372036854775807), !dbg !714
  %27 = icmp eq <8 x i64> %wide.load128, splat (i64 9223372036854775807), !dbg !714
  %28 = icmp slt <8 x i64> %broadcast.splat112, %wide.load125, !dbg !717
  %29 = icmp slt <8 x i64> %broadcast.splat112, %wide.load126, !dbg !717
  %30 = icmp slt <8 x i64> %broadcast.splat112, %wide.load127, !dbg !717
  %31 = icmp slt <8 x i64> %broadcast.splat112, %wide.load128, !dbg !717
  %32 = or <8 x i1> %19, %24, !dbg !685
  %33 = zext <8 x i1> %28 to <8 x i8>, !dbg !718
  %34 = zext <8 x i1> %29 to <8 x i8>, !dbg !718
  %35 = zext <8 x i1> %30 to <8 x i8>, !dbg !718
  %36 = zext <8 x i1> %31 to <8 x i8>, !dbg !718
  %bools.i.sroa.0.0.vec.expand417 = shufflevector <8 x i8> %33, <8 x i8> poison, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !718
  %bools.i.sroa.0.8.vec.expand425 = shufflevector <8 x i8> %34, <8 x i8> poison, <64 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !718
  %bools.i.sroa.0.8.vecblend426 = shufflevector <64 x i8> %bools.i.sroa.0.0.vec.expand417, <64 x i8> %bools.i.sroa.0.8.vec.expand425, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 72, i32 73, i32 74, i32 75, i32 76, i32 77, i32 78, i32 79, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !718
  %bools.i.sroa.0.16.vec.expand431 = shufflevector <8 x i8> %35, <8 x i8> poison, <64 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !718
  %bools.i.sroa.0.16.vecblend432 = shufflevector <64 x i8> %bools.i.sroa.0.8.vecblend426, <64 x i8> %bools.i.sroa.0.16.vec.expand431, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 80, i32 81, i32 82, i32 83, i32 84, i32 85, i32 86, i32 87, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !718
  %bools.i.sroa.0.24.vec.expand437 = shufflevector <8 x i8> %36, <8 x i8> poison, <64 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !718
  %bools.i.sroa.0.24.vecblend438 = shufflevector <64 x i8> %bools.i.sroa.0.16.vecblend432, <64 x i8> %bools.i.sroa.0.24.vec.expand437, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 88, i32 89, i32 90, i32 91, i32 92, i32 93, i32 94, i32 95, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !718
  %.idx358.1 = shl i64 %iter.i.sroa.7.055.us, 9, !dbg !708
  %37 = getelementptr inbounds nuw i8, ptr %7, i64 %.idx358.1, !dbg !708
  %38 = getelementptr inbounds nuw i8, ptr %37, i64 64, !dbg !713
  %39 = getelementptr inbounds nuw i8, ptr %37, i64 128, !dbg !713
  %40 = getelementptr inbounds nuw i8, ptr %37, i64 192, !dbg !713
  %wide.load125.1 = load <8 x i64>, ptr %37, align 8, !dbg !713, !noalias !719
  %wide.load126.1 = load <8 x i64>, ptr %38, align 8, !dbg !713, !noalias !719
  %wide.load127.1 = load <8 x i64>, ptr %39, align 8, !dbg !713, !noalias !719
  %wide.load128.1 = load <8 x i64>, ptr %40, align 8, !dbg !713, !noalias !719
  %41 = icmp eq <8 x i64> %wide.load125.1, splat (i64 9223372036854775807), !dbg !714
  %42 = icmp eq <8 x i64> %wide.load126.1, splat (i64 9223372036854775807), !dbg !714
  %43 = icmp eq <8 x i64> %wide.load127.1, splat (i64 9223372036854775807), !dbg !714
  %44 = icmp eq <8 x i64> %wide.load128.1, splat (i64 9223372036854775807), !dbg !714
  %45 = icmp slt <8 x i64> %broadcast.splat112, %wide.load125.1, !dbg !717
  %46 = icmp slt <8 x i64> %broadcast.splat112, %wide.load126.1, !dbg !717
  %47 = icmp slt <8 x i64> %broadcast.splat112, %wide.load127.1, !dbg !717
  %48 = icmp slt <8 x i64> %broadcast.splat112, %wide.load128.1, !dbg !717
  %49 = or <8 x i1> %32, %41, !dbg !685
  %50 = or <8 x i1> %49, %broadcast.splat114, !dbg !685
  %51 = or <8 x i1> %25, %42, !dbg !685
  %52 = or <8 x i1> %26, %43, !dbg !685
  %53 = or <8 x i1> %27, %44, !dbg !685
  %54 = zext <8 x i1> %45 to <8 x i8>, !dbg !718
  %55 = zext <8 x i1> %46 to <8 x i8>, !dbg !718
  %56 = zext <8 x i1> %47 to <8 x i8>, !dbg !718
  %57 = zext <8 x i1> %48 to <8 x i8>, !dbg !718
  %bools.i.sroa.0.32.vec.expand443 = shufflevector <8 x i8> %54, <8 x i8> poison, <64 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !718
  %bools.i.sroa.0.32.vecblend444 = shufflevector <64 x i8> %bools.i.sroa.0.24.vecblend438, <64 x i8> %bools.i.sroa.0.32.vec.expand443, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 96, i32 97, i32 98, i32 99, i32 100, i32 101, i32 102, i32 103, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !718
  %bools.i.sroa.0.40.vec.expand449 = shufflevector <8 x i8> %55, <8 x i8> poison, <64 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !718
  %bools.i.sroa.0.40.vecblend450 = shufflevector <64 x i8> %bools.i.sroa.0.32.vecblend444, <64 x i8> %bools.i.sroa.0.40.vec.expand449, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 104, i32 105, i32 106, i32 107, i32 108, i32 109, i32 110, i32 111, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !718
  %bools.i.sroa.0.48.vec.expand455 = shufflevector <8 x i8> %56, <8 x i8> poison, <64 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !718
  %bools.i.sroa.0.48.vecblend456 = shufflevector <64 x i8> %bools.i.sroa.0.40.vecblend450, <64 x i8> %bools.i.sroa.0.48.vec.expand455, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 112, i32 113, i32 114, i32 115, i32 116, i32 117, i32 118, i32 119, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !718
  %bools.i.sroa.0.56.vec.expand461 = shufflevector <8 x i8> %57, <8 x i8> poison, <64 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !718
  %bools.i.sroa.0.56.vecblend462 = shufflevector <64 x i8> %bools.i.sroa.0.48.vecblend456, <64 x i8> %bools.i.sroa.0.56.vec.expand461, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 120, i32 121, i32 122, i32 123, i32 124, i32 125, i32 126, i32 127>, !dbg !718
  %bin.rdx132 = or <8 x i1> %51, %50, !dbg !693
  %bin.rdx133 = or <8 x i1> %52, %bin.rdx132, !dbg !693
  %bin.rdx134 = or <8 x i1> %53, %bin.rdx133, !dbg !693
  %58 = bitcast <8 x i1> %bin.rdx134 to i8, !dbg !693
  %59 = icmp ne i8 %58, 0, !dbg !693
  %_17.i.i.us = getelementptr inbounds nuw i8, ptr %iter.i.sroa.0.056.us, i64 8, !dbg !697
  %_9.0.i.us = add nuw nsw i64 %iter.i.sroa.7.055.us, 1, !dbg !722
  %.not.i.i.i.us = icmp ne <64 x i8> %bools.i.sroa.0.56.vecblend462, zeroinitializer, !dbg !723
  store <64 x i1> %.not.i.i.i.us, ptr %iter.i.sroa.0.056.us, align 8, !dbg !699
  %_7.i.i.us = icmp eq ptr %_17.i.i.us, %_52.i, !dbg !664
  br i1 %_7.i.i.us, label %bb1.i.bb5.i_crit_edge, label %bb4.i.us, !dbg !669

bb4.i.lr.ph.split:                                ; preds = %bb4.i.lr.ph
  br i1 %1, label %bb4.i.lr.ph.split.split.us, label %bb4.i.preheader

bb4.i.preheader:                                  ; preds = %bb4.i.lr.ph.split
  %60 = trunc nuw i8 %_11.i.promoted57 to i1, !dbg !685
  br label %bb4.i, !dbg !693

bb4.i.lr.ph.split.split.us:                       ; preds = %bb4.i.lr.ph.split
  %_0.sroa.0.0.i9.i.i.us46.us = load i64, ptr %_6.i11.i.i.cast, align 8, !noalias !696, !noundef !23
  %61 = icmp eq i64 %_0.sroa.0.0.i9.i.i.us46.us, 9223372036854775807
  %62 = trunc nuw i8 %_11.i.promoted57 to i1, !dbg !685
  %broadcast.splatinsert84 = insertelement <8 x i64> poison, i64 %_0.sroa.0.0.i9.i.i.us46.us, i64 0
  %broadcast.splat85 = shufflevector <8 x i64> %broadcast.splatinsert84, <8 x i64> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert82 = insertelement <8 x i1> poison, i1 %61, i64 0
  %broadcast.splat83 = shufflevector <8 x i1> %broadcast.splatinsert82, <8 x i1> poison, <8 x i32> zeroinitializer
  %63 = getelementptr inbounds nuw i8, ptr %view.val.i.i.i, i64 256
  br label %bb4.i.us74, !dbg !669

bb4.i.us74:                                       ; preds = %bb4.i.us74, %bb4.i.lr.ph.split.split.us
  %.us-phi58.us75 = phi i1 [ %62, %bb4.i.lr.ph.split.split.us ], [ %104, %bb4.i.us74 ]
  %iter.i.sroa.0.056.us76 = phi ptr [ %words.0, %bb4.i.lr.ph.split.split.us ], [ %_17.i.i.us78, %bb4.i.us74 ]
  %iter.i.sroa.7.055.us77 = phi i64 [ 0, %bb4.i.lr.ph.split.split.us ], [ %_9.0.i.us79, %bb4.i.us74 ]
  %64 = insertelement <8 x i1> <i1 poison, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false>, i1 %.us-phi58.us75, i64 0
  tail call void @llvm.experimental.noalias.scope.decl(metadata !702), !dbg !703
  tail call void @llvm.experimental.noalias.scope.decl(metadata !704), !dbg !705
  %.idx = shl i64 %iter.i.sroa.7.055.us77, 9, !dbg !740
  %65 = getelementptr inbounds nuw i8, ptr %view.val.i.i.i, i64 %.idx, !dbg !740
  %66 = getelementptr inbounds nuw i8, ptr %65, i64 64, !dbg !745
  %67 = getelementptr inbounds nuw i8, ptr %65, i64 128, !dbg !745
  %68 = getelementptr inbounds nuw i8, ptr %65, i64 192, !dbg !745
  %wide.load96 = load <8 x i64>, ptr %65, align 8, !dbg !745, !noalias !684
  %wide.load97 = load <8 x i64>, ptr %66, align 8, !dbg !745, !noalias !684
  %wide.load98 = load <8 x i64>, ptr %67, align 8, !dbg !745, !noalias !684
  %wide.load99 = load <8 x i64>, ptr %68, align 8, !dbg !745, !noalias !684
  tail call void @llvm.experimental.noalias.scope.decl(metadata !707), !dbg !705
  %69 = icmp eq <8 x i64> %wide.load96, splat (i64 9223372036854775807), !dbg !714
  %70 = icmp eq <8 x i64> %wide.load97, splat (i64 9223372036854775807), !dbg !714
  %71 = icmp eq <8 x i64> %wide.load98, splat (i64 9223372036854775807), !dbg !714
  %72 = icmp eq <8 x i64> %wide.load99, splat (i64 9223372036854775807), !dbg !714
  %73 = icmp slt <8 x i64> %wide.load96, %broadcast.splat85, !dbg !717
  %74 = icmp slt <8 x i64> %wide.load97, %broadcast.splat85, !dbg !717
  %75 = icmp slt <8 x i64> %wide.load98, %broadcast.splat85, !dbg !717
  %76 = icmp slt <8 x i64> %wide.load99, %broadcast.splat85, !dbg !717
  %77 = or <8 x i1> %69, %64, !dbg !685
  %78 = zext <8 x i1> %73 to <8 x i8>, !dbg !718
  %79 = zext <8 x i1> %74 to <8 x i8>, !dbg !718
  %80 = zext <8 x i1> %75 to <8 x i8>, !dbg !718
  %81 = zext <8 x i1> %76 to <8 x i8>, !dbg !718
  %bools.i.sroa.0.0.vec.expand414 = shufflevector <8 x i8> %78, <8 x i8> poison, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !718
  %bools.i.sroa.0.8.vec.expand422 = shufflevector <8 x i8> %79, <8 x i8> poison, <64 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !718
  %bools.i.sroa.0.8.vecblend423 = shufflevector <64 x i8> %bools.i.sroa.0.0.vec.expand414, <64 x i8> %bools.i.sroa.0.8.vec.expand422, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 72, i32 73, i32 74, i32 75, i32 76, i32 77, i32 78, i32 79, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !718
  %bools.i.sroa.0.16.vec.expand428 = shufflevector <8 x i8> %80, <8 x i8> poison, <64 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !718
  %bools.i.sroa.0.16.vecblend429 = shufflevector <64 x i8> %bools.i.sroa.0.8.vecblend423, <64 x i8> %bools.i.sroa.0.16.vec.expand428, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 80, i32 81, i32 82, i32 83, i32 84, i32 85, i32 86, i32 87, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !718
  %bools.i.sroa.0.24.vec.expand434 = shufflevector <8 x i8> %81, <8 x i8> poison, <64 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !718
  %bools.i.sroa.0.24.vecblend435 = shufflevector <64 x i8> %bools.i.sroa.0.16.vecblend429, <64 x i8> %bools.i.sroa.0.24.vec.expand434, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 88, i32 89, i32 90, i32 91, i32 92, i32 93, i32 94, i32 95, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !718
  %.idx.1 = shl i64 %iter.i.sroa.7.055.us77, 9, !dbg !740
  %82 = getelementptr inbounds nuw i8, ptr %63, i64 %.idx.1, !dbg !740
  %83 = getelementptr inbounds nuw i8, ptr %82, i64 64, !dbg !745
  %84 = getelementptr inbounds nuw i8, ptr %82, i64 128, !dbg !745
  %85 = getelementptr inbounds nuw i8, ptr %82, i64 192, !dbg !745
  %wide.load96.1 = load <8 x i64>, ptr %82, align 8, !dbg !745, !noalias !746
  %wide.load97.1 = load <8 x i64>, ptr %83, align 8, !dbg !745, !noalias !746
  %wide.load98.1 = load <8 x i64>, ptr %84, align 8, !dbg !745, !noalias !746
  %wide.load99.1 = load <8 x i64>, ptr %85, align 8, !dbg !745, !noalias !746
  %86 = icmp eq <8 x i64> %wide.load96.1, splat (i64 9223372036854775807), !dbg !714
  %87 = icmp eq <8 x i64> %wide.load97.1, splat (i64 9223372036854775807), !dbg !714
  %88 = icmp eq <8 x i64> %wide.load98.1, splat (i64 9223372036854775807), !dbg !714
  %89 = icmp eq <8 x i64> %wide.load99.1, splat (i64 9223372036854775807), !dbg !714
  %90 = icmp slt <8 x i64> %wide.load96.1, %broadcast.splat85, !dbg !717
  %91 = icmp slt <8 x i64> %wide.load97.1, %broadcast.splat85, !dbg !717
  %92 = icmp slt <8 x i64> %wide.load98.1, %broadcast.splat85, !dbg !717
  %93 = icmp slt <8 x i64> %wide.load99.1, %broadcast.splat85, !dbg !717
  %94 = or <8 x i1> %77, %86, !dbg !685
  %95 = or <8 x i1> %94, %broadcast.splat83, !dbg !685
  %96 = or <8 x i1> %70, %87, !dbg !685
  %97 = or <8 x i1> %71, %88, !dbg !685
  %98 = or <8 x i1> %72, %89, !dbg !685
  %99 = zext <8 x i1> %90 to <8 x i8>, !dbg !718
  %100 = zext <8 x i1> %91 to <8 x i8>, !dbg !718
  %101 = zext <8 x i1> %92 to <8 x i8>, !dbg !718
  %102 = zext <8 x i1> %93 to <8 x i8>, !dbg !718
  %bools.i.sroa.0.32.vec.expand440 = shufflevector <8 x i8> %99, <8 x i8> poison, <64 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !718
  %bools.i.sroa.0.32.vecblend441 = shufflevector <64 x i8> %bools.i.sroa.0.24.vecblend435, <64 x i8> %bools.i.sroa.0.32.vec.expand440, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 96, i32 97, i32 98, i32 99, i32 100, i32 101, i32 102, i32 103, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !718
  %bools.i.sroa.0.40.vec.expand446 = shufflevector <8 x i8> %100, <8 x i8> poison, <64 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !718
  %bools.i.sroa.0.40.vecblend447 = shufflevector <64 x i8> %bools.i.sroa.0.32.vecblend441, <64 x i8> %bools.i.sroa.0.40.vec.expand446, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 104, i32 105, i32 106, i32 107, i32 108, i32 109, i32 110, i32 111, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !718
  %bools.i.sroa.0.48.vec.expand452 = shufflevector <8 x i8> %101, <8 x i8> poison, <64 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !718
  %bools.i.sroa.0.48.vecblend453 = shufflevector <64 x i8> %bools.i.sroa.0.40.vecblend447, <64 x i8> %bools.i.sroa.0.48.vec.expand452, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 112, i32 113, i32 114, i32 115, i32 116, i32 117, i32 118, i32 119, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !718
  %bools.i.sroa.0.56.vec.expand458 = shufflevector <8 x i8> %102, <8 x i8> poison, <64 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !718
  %bools.i.sroa.0.56.vecblend459 = shufflevector <64 x i8> %bools.i.sroa.0.48.vecblend453, <64 x i8> %bools.i.sroa.0.56.vec.expand458, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 120, i32 121, i32 122, i32 123, i32 124, i32 125, i32 126, i32 127>, !dbg !718
  %bin.rdx103 = or <8 x i1> %96, %95, !dbg !693
  %bin.rdx104 = or <8 x i1> %97, %bin.rdx103, !dbg !693
  %bin.rdx105 = or <8 x i1> %98, %bin.rdx104, !dbg !693
  %103 = bitcast <8 x i1> %bin.rdx105 to i8, !dbg !693
  %104 = icmp ne i8 %103, 0, !dbg !693
  %_17.i.i.us78 = getelementptr inbounds nuw i8, ptr %iter.i.sroa.0.056.us76, i64 8, !dbg !697
  %_9.0.i.us79 = add nuw nsw i64 %iter.i.sroa.7.055.us77, 1, !dbg !722
  %.not.i.i.i.us82 = icmp ne <64 x i8> %bools.i.sroa.0.56.vecblend459, zeroinitializer, !dbg !723
  store <64 x i1> %.not.i.i.i.us82, ptr %iter.i.sroa.0.056.us76, align 8, !dbg !699
  %_7.i.i.us83 = icmp eq ptr %_17.i.i.us78, %_52.i, !dbg !664
  br i1 %_7.i.i.us83, label %bb1.i.bb5.i_crit_edge, label %bb4.i.us74, !dbg !669

bb4.i:                                            ; preds = %bb4.i, %bb4.i.preheader
  %.us-phi58 = phi i1 [ %169, %bb4.i ], [ %60, %bb4.i.preheader ]
  %iter.i.sroa.0.056 = phi ptr [ %_17.i.i, %bb4.i ], [ %words.0, %bb4.i.preheader ]
  %iter.i.sroa.7.055 = phi i64 [ %_9.0.i, %bb4.i ], [ 0, %bb4.i.preheader ]
  %105 = insertelement <8 x i1> <i1 poison, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false>, i1 %.us-phi58, i64 0
  %offset2.i = shl i64 %iter.i.sroa.7.055, 6, !dbg !749
  tail call void @llvm.experimental.noalias.scope.decl(metadata !702), !dbg !703
  tail call void @llvm.experimental.noalias.scope.decl(metadata !704), !dbg !705
  %106 = getelementptr inbounds nuw i64, ptr %view.val.i.i.i, i64 %offset2.i, !dbg !740
  %107 = getelementptr inbounds nuw i8, ptr %106, i64 64, !dbg !745
  %108 = getelementptr inbounds nuw i8, ptr %106, i64 128, !dbg !745
  %109 = getelementptr inbounds nuw i8, ptr %106, i64 192, !dbg !745
  %wide.load = load <8 x i64>, ptr %106, align 8, !dbg !745, !noalias !684
  %wide.load68 = load <8 x i64>, ptr %107, align 8, !dbg !745, !noalias !684
  %wide.load69 = load <8 x i64>, ptr %108, align 8, !dbg !745, !noalias !684
  %wide.load70 = load <8 x i64>, ptr %109, align 8, !dbg !745, !noalias !684
  tail call void @llvm.experimental.noalias.scope.decl(metadata !707), !dbg !705
  %110 = getelementptr inbounds nuw i64, ptr %view.val.i4.i.i, i64 %offset2.i, !dbg !708
  %111 = getelementptr inbounds nuw i8, ptr %110, i64 64, !dbg !713
  %112 = getelementptr inbounds nuw i8, ptr %110, i64 128, !dbg !713
  %113 = getelementptr inbounds nuw i8, ptr %110, i64 192, !dbg !713
  %wide.load71 = load <8 x i64>, ptr %110, align 8, !dbg !713, !noalias !696
  %wide.load72 = load <8 x i64>, ptr %111, align 8, !dbg !713, !noalias !696
  %wide.load73 = load <8 x i64>, ptr %112, align 8, !dbg !713, !noalias !696
  %wide.load74 = load <8 x i64>, ptr %113, align 8, !dbg !713, !noalias !696
  %114 = icmp eq <8 x i64> %wide.load, splat (i64 9223372036854775807), !dbg !714
  %115 = icmp eq <8 x i64> %wide.load68, splat (i64 9223372036854775807), !dbg !714
  %116 = icmp eq <8 x i64> %wide.load69, splat (i64 9223372036854775807), !dbg !714
  %117 = icmp eq <8 x i64> %wide.load70, splat (i64 9223372036854775807), !dbg !714
  %118 = icmp eq <8 x i64> %wide.load71, splat (i64 9223372036854775807), !dbg !714
  %119 = icmp eq <8 x i64> %wide.load72, splat (i64 9223372036854775807), !dbg !714
  %120 = icmp eq <8 x i64> %wide.load73, splat (i64 9223372036854775807), !dbg !714
  %121 = icmp eq <8 x i64> %wide.load74, splat (i64 9223372036854775807), !dbg !714
  %122 = or <8 x i1> %114, %118, !dbg !714
  %123 = or <8 x i1> %115, %119, !dbg !714
  %124 = or <8 x i1> %116, %120, !dbg !714
  %125 = or <8 x i1> %117, %121, !dbg !714
  %126 = icmp slt <8 x i64> %wide.load, %wide.load71, !dbg !717
  %127 = icmp slt <8 x i64> %wide.load68, %wide.load72, !dbg !717
  %128 = icmp slt <8 x i64> %wide.load69, %wide.load73, !dbg !717
  %129 = icmp slt <8 x i64> %wide.load70, %wide.load74, !dbg !717
  %130 = or <8 x i1> %105, %122, !dbg !685
  %131 = zext <8 x i1> %126 to <8 x i8>, !dbg !718
  %132 = zext <8 x i1> %127 to <8 x i8>, !dbg !718
  %133 = zext <8 x i1> %128 to <8 x i8>, !dbg !718
  %134 = zext <8 x i1> %129 to <8 x i8>, !dbg !718
  %bools.i.sroa.0.0.vec.expand = shufflevector <8 x i8> %131, <8 x i8> poison, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !718
  %bools.i.sroa.0.8.vec.expand = shufflevector <8 x i8> %132, <8 x i8> poison, <64 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !718
  %bools.i.sroa.0.8.vecblend = shufflevector <64 x i8> %bools.i.sroa.0.0.vec.expand, <64 x i8> %bools.i.sroa.0.8.vec.expand, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 72, i32 73, i32 74, i32 75, i32 76, i32 77, i32 78, i32 79, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !718
  %bools.i.sroa.0.16.vec.expand = shufflevector <8 x i8> %133, <8 x i8> poison, <64 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !718
  %bools.i.sroa.0.16.vecblend = shufflevector <64 x i8> %bools.i.sroa.0.8.vecblend, <64 x i8> %bools.i.sroa.0.16.vec.expand, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 80, i32 81, i32 82, i32 83, i32 84, i32 85, i32 86, i32 87, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !718
  %bools.i.sroa.0.24.vec.expand = shufflevector <8 x i8> %134, <8 x i8> poison, <64 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !718
  %bools.i.sroa.0.24.vecblend = shufflevector <64 x i8> %bools.i.sroa.0.16.vecblend, <64 x i8> %bools.i.sroa.0.24.vec.expand, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 88, i32 89, i32 90, i32 91, i32 92, i32 93, i32 94, i32 95, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !718
  %135 = or disjoint i64 %offset2.i, 32
  %136 = getelementptr inbounds nuw i64, ptr %view.val.i.i.i, i64 %135, !dbg !740
  %137 = getelementptr inbounds nuw i8, ptr %136, i64 64, !dbg !745
  %138 = getelementptr inbounds nuw i8, ptr %136, i64 128, !dbg !745
  %139 = getelementptr inbounds nuw i8, ptr %136, i64 192, !dbg !745
  %wide.load.1 = load <8 x i64>, ptr %136, align 8, !dbg !745, !noalias !750
  %wide.load68.1 = load <8 x i64>, ptr %137, align 8, !dbg !745, !noalias !750
  %wide.load69.1 = load <8 x i64>, ptr %138, align 8, !dbg !745, !noalias !750
  %wide.load70.1 = load <8 x i64>, ptr %139, align 8, !dbg !745, !noalias !750
  %140 = getelementptr inbounds nuw i64, ptr %view.val.i4.i.i, i64 %135, !dbg !708
  %141 = getelementptr inbounds nuw i8, ptr %140, i64 64, !dbg !713
  %142 = getelementptr inbounds nuw i8, ptr %140, i64 128, !dbg !713
  %143 = getelementptr inbounds nuw i8, ptr %140, i64 192, !dbg !713
  %wide.load71.1 = load <8 x i64>, ptr %140, align 8, !dbg !713, !noalias !753
  %wide.load72.1 = load <8 x i64>, ptr %141, align 8, !dbg !713, !noalias !753
  %wide.load73.1 = load <8 x i64>, ptr %142, align 8, !dbg !713, !noalias !753
  %wide.load74.1 = load <8 x i64>, ptr %143, align 8, !dbg !713, !noalias !753
  %144 = icmp eq <8 x i64> %wide.load.1, splat (i64 9223372036854775807), !dbg !714
  %145 = icmp eq <8 x i64> %wide.load68.1, splat (i64 9223372036854775807), !dbg !714
  %146 = icmp eq <8 x i64> %wide.load69.1, splat (i64 9223372036854775807), !dbg !714
  %147 = icmp eq <8 x i64> %wide.load70.1, splat (i64 9223372036854775807), !dbg !714
  %148 = icmp eq <8 x i64> %wide.load71.1, splat (i64 9223372036854775807), !dbg !714
  %149 = icmp eq <8 x i64> %wide.load72.1, splat (i64 9223372036854775807), !dbg !714
  %150 = icmp eq <8 x i64> %wide.load73.1, splat (i64 9223372036854775807), !dbg !714
  %151 = icmp eq <8 x i64> %wide.load74.1, splat (i64 9223372036854775807), !dbg !714
  %152 = or <8 x i1> %144, %148, !dbg !714
  %153 = or <8 x i1> %145, %149, !dbg !714
  %154 = or <8 x i1> %146, %150, !dbg !714
  %155 = or <8 x i1> %147, %151, !dbg !714
  %156 = icmp slt <8 x i64> %wide.load.1, %wide.load71.1, !dbg !717
  %157 = icmp slt <8 x i64> %wide.load68.1, %wide.load72.1, !dbg !717
  %158 = icmp slt <8 x i64> %wide.load69.1, %wide.load73.1, !dbg !717
  %159 = icmp slt <8 x i64> %wide.load70.1, %wide.load74.1, !dbg !717
  %160 = or <8 x i1> %130, %152, !dbg !685
  %161 = or <8 x i1> %123, %153, !dbg !685
  %162 = or <8 x i1> %124, %154, !dbg !685
  %163 = or <8 x i1> %125, %155, !dbg !685
  %164 = zext <8 x i1> %156 to <8 x i8>, !dbg !718
  %165 = zext <8 x i1> %157 to <8 x i8>, !dbg !718
  %166 = zext <8 x i1> %158 to <8 x i8>, !dbg !718
  %167 = zext <8 x i1> %159 to <8 x i8>, !dbg !718
  %bools.i.sroa.0.32.vec.expand = shufflevector <8 x i8> %164, <8 x i8> poison, <64 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !718
  %bools.i.sroa.0.32.vecblend = shufflevector <64 x i8> %bools.i.sroa.0.24.vecblend, <64 x i8> %bools.i.sroa.0.32.vec.expand, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 96, i32 97, i32 98, i32 99, i32 100, i32 101, i32 102, i32 103, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !718
  %bools.i.sroa.0.40.vec.expand = shufflevector <8 x i8> %165, <8 x i8> poison, <64 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !718
  %bools.i.sroa.0.40.vecblend = shufflevector <64 x i8> %bools.i.sroa.0.32.vecblend, <64 x i8> %bools.i.sroa.0.40.vec.expand, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 104, i32 105, i32 106, i32 107, i32 108, i32 109, i32 110, i32 111, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !718
  %bools.i.sroa.0.48.vec.expand = shufflevector <8 x i8> %166, <8 x i8> poison, <64 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !718
  %bools.i.sroa.0.48.vecblend = shufflevector <64 x i8> %bools.i.sroa.0.40.vecblend, <64 x i8> %bools.i.sroa.0.48.vec.expand, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 112, i32 113, i32 114, i32 115, i32 116, i32 117, i32 118, i32 119, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !718
  %bools.i.sroa.0.56.vec.expand = shufflevector <8 x i8> %167, <8 x i8> poison, <64 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !718
  %bools.i.sroa.0.56.vecblend = shufflevector <64 x i8> %bools.i.sroa.0.48.vecblend, <64 x i8> %bools.i.sroa.0.56.vec.expand, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 120, i32 121, i32 122, i32 123, i32 124, i32 125, i32 126, i32 127>, !dbg !718
  %bin.rdx = or <8 x i1> %161, %160, !dbg !693
  %bin.rdx75 = or <8 x i1> %162, %bin.rdx, !dbg !693
  %bin.rdx76 = or <8 x i1> %163, %bin.rdx75, !dbg !693
  %168 = bitcast <8 x i1> %bin.rdx76 to i8, !dbg !693
  %169 = icmp ne i8 %168, 0, !dbg !693
  %_17.i.i = getelementptr inbounds nuw i8, ptr %iter.i.sroa.0.056, i64 8, !dbg !697
  %_9.0.i = add nuw nsw i64 %iter.i.sroa.7.055, 1, !dbg !722
  %.not.i.i.i = icmp ne <64 x i8> %bools.i.sroa.0.56.vecblend, zeroinitializer, !dbg !723
  store <64 x i1> %.not.i.i.i, ptr %iter.i.sroa.0.056, align 8, !dbg !699
  %_7.i.i = icmp eq ptr %_17.i.i, %_52.i, !dbg !664
  br i1 %_7.i.i, label %bb1.i.bb5.i_crit_edge, label %bb4.i, !dbg !669

bb1.i.bb5.i_crit_edge.loopexit:                   ; preds = %bb4.i.us.us, %bb4.i.us.us.prol.loopexit
  %170 = icmp eq i64 %_0.sroa.0.0.i9.i.i.us.us.us.us, 9223372036854775807
  %171 = trunc nuw i8 %_11.i.promoted57 to i1, !dbg !685
  %172 = or i1 %170, %171
  %173 = or i1 %5, %172
  br label %bb1.i.bb5.i_crit_edge, !dbg !685

bb1.i.bb5.i_crit_edge:                            ; preds = %bb4.i, %bb4.i.us74, %bb4.i.us, %bb1.i.bb5.i_crit_edge.loopexit
  %.us-phi73.in = phi i1 [ %173, %bb1.i.bb5.i_crit_edge.loopexit ], [ %59, %bb4.i.us ], [ %104, %bb4.i.us74 ], [ %169, %bb4.i ]
  %.us-phi73 = zext i1 %.us-phi73.in to i8, !dbg !685
  store i8 %.us-phi73, ptr %_11.i, align 8, !alias.scope !681, !noalias !675
  br label %bb5.i, !dbg !669

bb5.i:                                            ; preds = %bb1.i.bb5.i_crit_edge, %bb21.i
  %174 = icmp eq i64 %remainder.i, 0, !dbg !755
  br i1 %174, label %_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack23collect_bool_words_withNCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1g_7visitor5retry21ExecuteDenseWithRetryINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB35_EENtNtB3j_11row_visitor10RowVisitor19visit_deferred_boolB2Y_bKB35_NCINvXB43_B40_NtNtB1g_6row_fn5RowFn8dispatchB3e_E0NCB5V_s_0E0NCB3a_s_0B6H_E0NCINvB2_25collect_bool_words_avx512B17_E0EB43_.exit, label %bb12.i, !dbg !755

bb12.i:                                           ; preds = %bb5.i
  %175 = and i64 %len, -64, !dbg !756
  %_3.i.i.i.i.i = load i64, ptr %f, align 8, !range !351, !alias.scope !757, !noalias !762, !noundef !23
  %176 = trunc nuw i64 %_3.i.i.i.i.i to i1
  %_6.i.i.i.i = getelementptr inbounds nuw i8, ptr %f, i64 24
  %_3.i1.i.i.i.i = load i64, ptr %_6.i.i.i.i, align 8, !range !351, !alias.scope !768, !noalias !762, !noundef !23
  %177 = trunc nuw i64 %_3.i1.i.i.i.i to i1
  %_11.i.i.i = getelementptr inbounds nuw i8, ptr %f, i64 56
  %_11.i.i.promoted.i = load i8, ptr %_11.i.i.i, align 8, !alias.scope !771, !noalias !762
  %view.i3.i.i.i.i = getelementptr inbounds nuw i8, ptr %f, i64 32
  %178 = getelementptr inbounds nuw i8, ptr %f, i64 40
  %179 = getelementptr inbounds nuw i8, ptr %f, i64 16
  br i1 %176, label %start.split.us.i, label %start.split.i

start.split.us.i:                                 ; preds = %bb12.i
  %_6.i.i.i.i.us.i = load ptr, ptr %179, align 8, !alias.scope !757, !noalias !762, !nonnull !23, !align !363, !noundef !23
  %min.iters.check314 = icmp samesign ult i64 %remainder.i, 4, !dbg !774
  br i1 %177, label %iter.check316, label %iter.check251

iter.check316:                                    ; preds = %start.split.us.i
  %_6.i11.i.i.i.us.us.i = load ptr, ptr %178, align 8, !alias.scope !768, !noalias !762, !nonnull !23, !align !363, !noundef !23
  %_0.sroa.0.0.i.i.i.i.us.us.i = load i64, ptr %_6.i.i.i.i.us.i, align 8, !noalias !785, !noundef !23
  %_0.sroa.0.0.i9.i.i.i.us.us.i = load i64, ptr %_6.i11.i.i.i.us.us.i, align 8, !noalias !786, !noundef !23
  %_5.i.i.i.i.us.us.i = icmp slt i64 %_0.sroa.0.0.i.i.i.i.us.us.i, %_0.sroa.0.0.i9.i.i.i.us.us.i
  %_13.us.us.i = zext i1 %_5.i.i.i.i.us.us.i to i64
  br i1 %min.iters.check314, label %bb8.us.us.i.preheader, label %vector.main.loop.iter.check318, !dbg !774

vector.main.loop.iter.check318:                   ; preds = %iter.check316
  %min.iters.check317 = icmp samesign ult i64 %remainder.i, 16, !dbg !774
  br i1 %min.iters.check317, label %vec.epilog.ph340, label %vector.ph319, !dbg !774

vector.ph319:                                     ; preds = %vector.main.loop.iter.check318
  %n.mod.vf320 = and i64 %len, 12
  %n.vec321 = and i64 %len, 48
  %broadcast.splatinsert322 = insertelement <8 x i64> poison, i64 %_13.us.us.i, i64 0
  %broadcast.splat323 = shufflevector <8 x i64> %broadcast.splatinsert322, <8 x i64> poison, <8 x i32> zeroinitializer
  br label %vector.body324, !dbg !774

vector.body324:                                   ; preds = %vector.body324, %vector.ph319
  %index325 = phi i64 [ 0, %vector.ph319 ], [ %index.next330, %vector.body324 ], !dbg !787
  %vec.phi326 = phi <8 x i64> [ zeroinitializer, %vector.ph319 ], [ %182, %vector.body324 ]
  %vec.phi327 = phi <8 x i64> [ zeroinitializer, %vector.ph319 ], [ %183, %vector.body324 ]
  %vec.ind328 = phi <8 x i64> [ <i64 0, i64 1, i64 2, i64 3, i64 4, i64 5, i64 6, i64 7>, %vector.ph319 ], [ %vec.ind.next331, %vector.body324 ]
  %step.add329 = add <8 x i64> %vec.ind328, splat (i64 8)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !793), !dbg !794
  tail call void @llvm.experimental.noalias.scope.decl(metadata !799), !dbg !800
  tail call void @llvm.experimental.noalias.scope.decl(metadata !802), !dbg !800
  %180 = shl nuw <8 x i64> %broadcast.splat323, %vec.ind328, !dbg !803
  %181 = shl nuw <8 x i64> %broadcast.splat323, %step.add329, !dbg !803
  %182 = or <8 x i64> %180, %vec.phi326, !dbg !804
  %183 = or <8 x i64> %181, %vec.phi327, !dbg !804
  %index.next330 = add nuw i64 %index325, 16, !dbg !787
  %vec.ind.next331 = add <8 x i64> %vec.ind328, splat (i64 16)
  %184 = icmp eq i64 %index.next330, %n.vec321, !dbg !774
  br i1 %184, label %middle.block332, label %vector.body324, !dbg !774, !llvm.loop !805

middle.block332:                                  ; preds = %vector.body324
  %bin.rdx333 = or <8 x i64> %183, %182, !dbg !774
  %185 = tail call i64 @llvm.vector.reduce.or.v8i64(<8 x i64> %bin.rdx333), !dbg !774
  %cmp.n334 = icmp eq i64 %remainder.i, %n.vec321, !dbg !774
  br i1 %cmp.n334, label %_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack24collect_bool_word_scalarNCINvB2_23collect_bool_words_withNCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1O_7visitor5retry21ExecuteDenseWithRetryINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB3D_EENtNtB3R_11row_visitor10RowVisitor19visit_deferred_boolB3w_bKB3D_NCINvXB4B_B4y_NtNtB1O_6row_fn5RowFn8dispatchB3M_E0NCB6t_s_0E0NCB3I_s_0B7f_E0NCINvB2_25collect_bool_words_avx512B1F_E0E0EB4B_.exit.loopexit, label %vec.epilog.iter.check338, !dbg !774

vec.epilog.iter.check338:                         ; preds = %middle.block332
  %min.epilog.iters.check339 = icmp eq i64 %n.mod.vf320, 0
  br i1 %min.epilog.iters.check339, label %bb8.us.us.i.preheader, label %vec.epilog.ph340, !prof !404

vec.epilog.ph340:                                 ; preds = %vector.main.loop.iter.check318, %vec.epilog.iter.check338
  %bc.resume.val335 = phi i64 [ %n.vec321, %vec.epilog.iter.check338 ], [ 0, %vector.main.loop.iter.check318 ]
  %bc.merge.rdx336 = phi i64 [ %185, %vec.epilog.iter.check338 ], [ 0, %vector.main.loop.iter.check318 ]
  %n.vec342 = and i64 %len, 60
  %186 = insertelement <4 x i64> <i64 poison, i64 0, i64 0, i64 0>, i64 %bc.merge.rdx336, i64 0
  %broadcast.splatinsert343 = insertelement <4 x i64> poison, i64 %_13.us.us.i, i64 0
  %broadcast.splat344 = shufflevector <4 x i64> %broadcast.splatinsert343, <4 x i64> poison, <4 x i32> zeroinitializer
  %broadcast.splatinsert345 = insertelement <4 x i64> poison, i64 %bc.resume.val335, i64 0
  %broadcast.splat346 = shufflevector <4 x i64> %broadcast.splatinsert345, <4 x i64> poison, <4 x i32> zeroinitializer
  %induction347 = or disjoint <4 x i64> %broadcast.splat346, <i64 0, i64 1, i64 2, i64 3>
  br label %vec.epilog.vector.body348

vec.epilog.vector.body348:                        ; preds = %vec.epilog.vector.body348, %vec.epilog.ph340
  %index349 = phi i64 [ %bc.resume.val335, %vec.epilog.ph340 ], [ %index.next352, %vec.epilog.vector.body348 ], !dbg !787
  %vec.phi350 = phi <4 x i64> [ %186, %vec.epilog.ph340 ], [ %188, %vec.epilog.vector.body348 ]
  %vec.ind351 = phi <4 x i64> [ %induction347, %vec.epilog.ph340 ], [ %vec.ind.next353, %vec.epilog.vector.body348 ]
  tail call void @llvm.experimental.noalias.scope.decl(metadata !793), !dbg !794
  tail call void @llvm.experimental.noalias.scope.decl(metadata !799), !dbg !800
  tail call void @llvm.experimental.noalias.scope.decl(metadata !802), !dbg !800
  %187 = shl nuw <4 x i64> %broadcast.splat344, %vec.ind351, !dbg !803
  %188 = or <4 x i64> %187, %vec.phi350, !dbg !804
  %index.next352 = add nuw i64 %index349, 4, !dbg !787
  %vec.ind.next353 = add nuw nsw <4 x i64> %vec.ind351, splat (i64 4)
  %189 = icmp eq i64 %index.next352, %n.vec342, !dbg !774
  br i1 %189, label %vec.epilog.middle.block354, label %vec.epilog.vector.body348, !dbg !774, !llvm.loop !806

vec.epilog.middle.block354:                       ; preds = %vec.epilog.vector.body348
  %190 = tail call i64 @llvm.vector.reduce.or.v4i64(<4 x i64> %188), !dbg !774
  %cmp.n355 = icmp eq i64 %remainder.i, %n.vec342, !dbg !774
  br i1 %cmp.n355, label %_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack24collect_bool_word_scalarNCINvB2_23collect_bool_words_withNCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1O_7visitor5retry21ExecuteDenseWithRetryINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB3D_EENtNtB3R_11row_visitor10RowVisitor19visit_deferred_boolB3w_bKB3D_NCINvXB4B_B4y_NtNtB1O_6row_fn5RowFn8dispatchB3M_E0NCB6t_s_0E0NCB3I_s_0B7f_E0NCINvB2_25collect_bool_words_avx512B1F_E0E0EB4B_.exit.loopexit, label %bb8.us.us.i.preheader, !dbg !774

bb8.us.us.i.preheader:                            ; preds = %iter.check316, %vec.epilog.iter.check338, %vec.epilog.middle.block354
  %packed.sroa.0.05.us.us.i.ph = phi i64 [ 0, %iter.check316 ], [ %185, %vec.epilog.iter.check338 ], [ %190, %vec.epilog.middle.block354 ]
  %iter.sroa.0.04.us.us.i.ph = phi i64 [ 0, %iter.check316 ], [ %n.vec321, %vec.epilog.iter.check338 ], [ %n.vec342, %vec.epilog.middle.block354 ]
  br label %bb8.us.us.i, !dbg !774

bb8.us.us.i:                                      ; preds = %bb8.us.us.i.preheader, %bb8.us.us.i
  %packed.sroa.0.05.us.us.i = phi i64 [ %192, %bb8.us.us.i ], [ %packed.sroa.0.05.us.us.i.ph, %bb8.us.us.i.preheader ]
  %iter.sroa.0.04.us.us.i = phi i64 [ %191, %bb8.us.us.i ], [ %iter.sroa.0.04.us.us.i.ph, %bb8.us.us.i.preheader ]
  tail call void @llvm.experimental.noalias.scope.decl(metadata !793), !dbg !794
  tail call void @llvm.experimental.noalias.scope.decl(metadata !799), !dbg !800
  tail call void @llvm.experimental.noalias.scope.decl(metadata !802), !dbg !800
  %191 = add nuw nsw i64 %iter.sroa.0.04.us.us.i, 1, !dbg !787
  %_12.us.us.i = shl nuw i64 %_13.us.us.i, %iter.sroa.0.04.us.us.i, !dbg !803
  %192 = or i64 %_12.us.us.i, %packed.sroa.0.05.us.us.i, !dbg !804
  %exitcond32.not.i = icmp eq i64 %191, %remainder.i, !dbg !807
  br i1 %exitcond32.not.i, label %_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack24collect_bool_word_scalarNCINvB2_23collect_bool_words_withNCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1O_7visitor5retry21ExecuteDenseWithRetryINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB3D_EENtNtB3R_11row_visitor10RowVisitor19visit_deferred_boolB3w_bKB3D_NCINvXB4B_B4y_NtNtB1O_6row_fn5RowFn8dispatchB3M_E0NCB6t_s_0E0NCB3I_s_0B7f_E0NCINvB2_25collect_bool_words_avx512B1F_E0E0EB4B_.exit.loopexit, label %bb8.us.us.i, !dbg !774, !llvm.loop !810

iter.check251:                                    ; preds = %start.split.us.i
  %view.val.i4.i.i.i.us.i = load ptr, ptr %view.i3.i.i.i.i, align 8, !alias.scope !768, !noalias !762, !nonnull !23, !align !363, !noundef !23
  %view.val1.i5.i.i.i.us.i = load i64, ptr %178, align 8, !alias.scope !768, !noalias !762, !noundef !23
  %193 = trunc nuw i8 %_11.i.i.promoted.i to i1, !dbg !811
  %_0.sroa.0.0.i.i.i.i.us.i = load i64, ptr %_6.i.i.i.i.us.i, align 8, !noalias !785, !noundef !23
  %194 = icmp eq i64 %_0.sroa.0.0.i.i.i.i.us.i, 9223372036854775807
  br i1 %min.iters.check314, label %bb8.us.i.preheader, label %vector.main.loop.iter.check253, !dbg !774

vector.main.loop.iter.check253:                   ; preds = %iter.check251
  %min.iters.check252 = icmp samesign ult i64 %remainder.i, 16, !dbg !774
  br i1 %min.iters.check252, label %vec.epilog.ph287, label %vector.ph254, !dbg !774

vector.ph254:                                     ; preds = %vector.main.loop.iter.check253
  %n.mod.vf255 = and i64 %len, 12
  %n.vec256 = and i64 %len, 48
  %195 = insertelement <8 x i1> <i1 poison, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false>, i1 %193, i64 0
  %broadcast.splatinsert259 = insertelement <8 x i64> poison, i64 %_0.sroa.0.0.i.i.i.i.us.i, i64 0
  %broadcast.splat260 = shufflevector <8 x i64> %broadcast.splatinsert259, <8 x i64> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert261 = insertelement <8 x i1> poison, i1 %194, i64 0
  %broadcast.splat262 = shufflevector <8 x i1> %broadcast.splatinsert261, <8 x i1> poison, <8 x i32> zeroinitializer
  %invariant.gep496 = getelementptr i64, ptr %view.val.i4.i.i.i.us.i, i64 %175, !dbg !774
  br label %vector.body265, !dbg !774

vector.body265:                                   ; preds = %vector.body265, %vector.ph254
  %index266 = phi i64 [ 0, %vector.ph254 ], [ %index.next275, %vector.body265 ], !dbg !787
  %vec.phi267 = phi <8 x i64> [ zeroinitializer, %vector.ph254 ], [ %210, %vector.body265 ]
  %vec.phi268 = phi <8 x i64> [ zeroinitializer, %vector.ph254 ], [ %211, %vector.body265 ]
  %vec.ind269 = phi <8 x i64> [ <i64 0, i64 1, i64 2, i64 3, i64 4, i64 5, i64 6, i64 7>, %vector.ph254 ], [ %vec.ind.next276, %vector.body265 ]
  %vec.phi270 = phi <8 x i1> [ %195, %vector.ph254 ], [ %203, %vector.body265 ]
  %vec.phi271 = phi <8 x i1> [ zeroinitializer, %vector.ph254 ], [ %205, %vector.body265 ]
  %step.add272 = add <8 x i64> %vec.ind269, splat (i64 8)
  %196 = extractelement <8 x i64> %vec.ind269, i64 0
  tail call void @llvm.experimental.noalias.scope.decl(metadata !793), !dbg !794
  tail call void @llvm.experimental.noalias.scope.decl(metadata !799), !dbg !800
  tail call void @llvm.experimental.noalias.scope.decl(metadata !802), !dbg !800
  %gep497 = getelementptr i64, ptr %invariant.gep496, i64 %196, !dbg !813
  %197 = getelementptr inbounds nuw i8, ptr %gep497, i64 64, !dbg !818
  %wide.load273 = load <8 x i64>, ptr %gep497, align 8, !dbg !818, !noalias !786
  %wide.load274 = load <8 x i64>, ptr %197, align 8, !dbg !818, !noalias !786
  %198 = icmp eq <8 x i64> %wide.load273, splat (i64 9223372036854775807), !dbg !819
  %199 = icmp eq <8 x i64> %wide.load274, splat (i64 9223372036854775807), !dbg !819
  %200 = icmp slt <8 x i64> %broadcast.splat260, %wide.load273, !dbg !822
  %201 = icmp slt <8 x i64> %broadcast.splat260, %wide.load274, !dbg !822
  %202 = or <8 x i1> %198, %vec.phi270, !dbg !811
  %203 = or <8 x i1> %202, %broadcast.splat262, !dbg !811
  %204 = or <8 x i1> %199, %vec.phi271, !dbg !811
  %205 = or <8 x i1> %204, %broadcast.splat262, !dbg !811
  %206 = zext <8 x i1> %200 to <8 x i64>, !dbg !803
  %207 = zext <8 x i1> %201 to <8 x i64>, !dbg !803
  %208 = shl nuw <8 x i64> %206, %vec.ind269, !dbg !803
  %209 = shl nuw <8 x i64> %207, %step.add272, !dbg !803
  %210 = or <8 x i64> %208, %vec.phi267, !dbg !804
  %211 = or <8 x i64> %209, %vec.phi268, !dbg !804
  %index.next275 = add nuw i64 %index266, 16, !dbg !787
  %vec.ind.next276 = add <8 x i64> %vec.ind269, splat (i64 16)
  %212 = icmp eq i64 %index.next275, %n.vec256, !dbg !774
  br i1 %212, label %middle.block277, label %vector.body265, !dbg !774, !llvm.loop !823

middle.block277:                                  ; preds = %vector.body265
  %bin.rdx278 = or <8 x i64> %211, %210, !dbg !774
  %213 = tail call i64 @llvm.vector.reduce.or.v8i64(<8 x i64> %bin.rdx278), !dbg !774
  %bin.rdx279 = or <8 x i1> %204, %203, !dbg !774
  %214 = bitcast <8 x i1> %bin.rdx279 to i8, !dbg !774
  %215 = icmp ne i8 %214, 0, !dbg !774
  %cmp.n280 = icmp eq i64 %remainder.i, %n.vec256, !dbg !774
  br i1 %cmp.n280, label %_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack24collect_bool_word_scalarNCINvB2_23collect_bool_words_withNCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1O_7visitor5retry21ExecuteDenseWithRetryINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB3D_EENtNtB3R_11row_visitor10RowVisitor19visit_deferred_boolB3w_bKB3D_NCINvXB4B_B4y_NtNtB1O_6row_fn5RowFn8dispatchB3M_E0NCB6t_s_0E0NCB3I_s_0B7f_E0NCINvB2_25collect_bool_words_avx512B1F_E0E0EB4B_.exit, label %vec.epilog.iter.check285, !dbg !774

vec.epilog.iter.check285:                         ; preds = %middle.block277
  %min.epilog.iters.check286 = icmp eq i64 %n.mod.vf255, 0
  br i1 %min.epilog.iters.check286, label %bb8.us.i.preheader, label %vec.epilog.ph287, !prof !404

vec.epilog.ph287:                                 ; preds = %vector.main.loop.iter.check253, %vec.epilog.iter.check285
  %bc.resume.val281 = phi i64 [ %n.vec256, %vec.epilog.iter.check285 ], [ 0, %vector.main.loop.iter.check253 ]
  %bc.merge.rdx282 = phi i64 [ %213, %vec.epilog.iter.check285 ], [ 0, %vector.main.loop.iter.check253 ]
  %bc.merge.rdx283 = phi i1 [ %215, %vec.epilog.iter.check285 ], [ %193, %vector.main.loop.iter.check253 ]
  %n.vec289 = and i64 %len, 60
  %216 = insertelement <4 x i64> <i64 poison, i64 0, i64 0, i64 0>, i64 %bc.merge.rdx282, i64 0
  %217 = insertelement <4 x i1> <i1 poison, i1 false, i1 false, i1 false>, i1 %bc.merge.rdx283, i64 0
  %broadcast.splatinsert292 = insertelement <4 x i64> poison, i64 %_0.sroa.0.0.i.i.i.i.us.i, i64 0
  %broadcast.splat293 = shufflevector <4 x i64> %broadcast.splatinsert292, <4 x i64> poison, <4 x i32> zeroinitializer
  %broadcast.splatinsert294 = insertelement <4 x i1> poison, i1 %194, i64 0
  %broadcast.splat295 = shufflevector <4 x i1> %broadcast.splatinsert294, <4 x i1> poison, <4 x i32> zeroinitializer
  %broadcast.splatinsert298 = insertelement <4 x i64> poison, i64 %bc.resume.val281, i64 0
  %broadcast.splat299 = shufflevector <4 x i64> %broadcast.splatinsert298, <4 x i64> poison, <4 x i32> zeroinitializer
  %induction300 = or disjoint <4 x i64> %broadcast.splat299, <i64 0, i64 1, i64 2, i64 3>
  %invariant.gep498 = getelementptr i64, ptr %view.val.i4.i.i.i.us.i, i64 %175
  br label %vec.epilog.vector.body301

vec.epilog.vector.body301:                        ; preds = %vec.epilog.vector.body301, %vec.epilog.ph287
  %index302 = phi i64 [ %bc.resume.val281, %vec.epilog.ph287 ], [ %index.next307, %vec.epilog.vector.body301 ], !dbg !787
  %vec.phi303 = phi <4 x i64> [ %216, %vec.epilog.ph287 ], [ %225, %vec.epilog.vector.body301 ]
  %vec.ind304 = phi <4 x i64> [ %induction300, %vec.epilog.ph287 ], [ %vec.ind.next308, %vec.epilog.vector.body301 ]
  %vec.phi305 = phi <4 x i1> [ %217, %vec.epilog.ph287 ], [ %222, %vec.epilog.vector.body301 ]
  %218 = extractelement <4 x i64> %vec.ind304, i64 0
  tail call void @llvm.experimental.noalias.scope.decl(metadata !793), !dbg !794
  tail call void @llvm.experimental.noalias.scope.decl(metadata !799), !dbg !800
  tail call void @llvm.experimental.noalias.scope.decl(metadata !802), !dbg !800
  %gep499 = getelementptr i64, ptr %invariant.gep498, i64 %218, !dbg !813
  %wide.load306 = load <4 x i64>, ptr %gep499, align 8, !dbg !818, !noalias !786
  %219 = icmp eq <4 x i64> %wide.load306, splat (i64 9223372036854775807), !dbg !819
  %220 = or <4 x i1> %broadcast.splat295, %219, !dbg !819
  %221 = icmp slt <4 x i64> %broadcast.splat293, %wide.load306, !dbg !822
  %222 = or <4 x i1> %vec.phi305, %220, !dbg !811
  %223 = zext <4 x i1> %221 to <4 x i64>, !dbg !803
  %224 = shl nuw <4 x i64> %223, %vec.ind304, !dbg !803
  %225 = or <4 x i64> %224, %vec.phi303, !dbg !804
  %index.next307 = add nuw i64 %index302, 4, !dbg !787
  %vec.ind.next308 = add nuw nsw <4 x i64> %vec.ind304, splat (i64 4)
  %226 = icmp eq i64 %index.next307, %n.vec289, !dbg !774
  br i1 %226, label %vec.epilog.middle.block309, label %vec.epilog.vector.body301, !dbg !774, !llvm.loop !824

vec.epilog.middle.block309:                       ; preds = %vec.epilog.vector.body301
  %227 = tail call i64 @llvm.vector.reduce.or.v4i64(<4 x i64> %225), !dbg !774
  %228 = bitcast <4 x i1> %222 to i4, !dbg !774
  %229 = icmp ne i4 %228, 0, !dbg !774
  %cmp.n310 = icmp eq i64 %remainder.i, %n.vec289, !dbg !774
  br i1 %cmp.n310, label %_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack24collect_bool_word_scalarNCINvB2_23collect_bool_words_withNCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1O_7visitor5retry21ExecuteDenseWithRetryINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB3D_EENtNtB3R_11row_visitor10RowVisitor19visit_deferred_boolB3w_bKB3D_NCINvXB4B_B4y_NtNtB1O_6row_fn5RowFn8dispatchB3M_E0NCB6t_s_0E0NCB3I_s_0B7f_E0NCINvB2_25collect_bool_words_avx512B1F_E0E0EB4B_.exit, label %bb8.us.i.preheader, !dbg !774

bb8.us.i.preheader:                               ; preds = %iter.check251, %vec.epilog.iter.check285, %vec.epilog.middle.block309
  %packed.sroa.0.05.us.i.ph = phi i64 [ 0, %iter.check251 ], [ %213, %vec.epilog.iter.check285 ], [ %227, %vec.epilog.middle.block309 ]
  %iter.sroa.0.04.us.i.ph = phi i64 [ 0, %iter.check251 ], [ %n.vec256, %vec.epilog.iter.check285 ], [ %n.vec289, %vec.epilog.middle.block309 ]
  %.ph = phi i1 [ %193, %iter.check251 ], [ %215, %vec.epilog.iter.check285 ], [ %229, %vec.epilog.middle.block309 ]
  br label %bb8.us.i, !dbg !774

bb8.us.i:                                         ; preds = %bb8.us.i.preheader, %bb8.us.i
  %packed.sroa.0.05.us.i = phi i64 [ %235, %bb8.us.i ], [ %packed.sroa.0.05.us.i.ph, %bb8.us.i.preheader ]
  %iter.sroa.0.04.us.i = phi i64 [ %234, %bb8.us.i ], [ %iter.sroa.0.04.us.i.ph, %bb8.us.i.preheader ]
  %230 = phi i1 [ %233, %bb8.us.i ], [ %.ph, %bb8.us.i.preheader ]
  %_4.i.us.i = add nuw nsw i64 %iter.sroa.0.04.us.i, %175, !dbg !825
  tail call void @llvm.experimental.noalias.scope.decl(metadata !793), !dbg !794
  tail call void @llvm.experimental.noalias.scope.decl(metadata !799), !dbg !800
  tail call void @llvm.experimental.noalias.scope.decl(metadata !802), !dbg !800
  %_5.i.i6.i.i.i.us.i = icmp ult i64 %_4.i.us.i, %view.val1.i5.i.i.i.us.i, !dbg !826
  tail call void @llvm.assume(i1 %_5.i.i6.i.i.i.us.i), !dbg !827
  %_4.i.i7.i.i.i.us.i = getelementptr inbounds nuw i64, ptr %view.val.i4.i.i.i.us.i, i64 %_4.i.us.i, !dbg !813
  %_0.sroa.0.0.i9.i.i.i.us.i = load i64, ptr %_4.i.i7.i.i.i.us.i, align 8, !dbg !818, !noalias !786, !noundef !23
  %231 = icmp eq i64 %_0.sroa.0.0.i9.i.i.i.us.i, 9223372036854775807, !dbg !819
  %_5.i.i.i.i.us.i = icmp slt i64 %_0.sroa.0.0.i.i.i.i.us.i, %_0.sroa.0.0.i9.i.i.i.us.i, !dbg !822
  %232 = or i1 %231, %230, !dbg !811
  %233 = or i1 %232, %194, !dbg !811
  %234 = add nuw nsw i64 %iter.sroa.0.04.us.i, 1, !dbg !787
  %_13.us.i = zext i1 %_5.i.i.i.i.us.i to i64, !dbg !803
  %_12.us.i = shl nuw i64 %_13.us.i, %iter.sroa.0.04.us.i, !dbg !803
  %235 = or i64 %_12.us.i, %packed.sroa.0.05.us.i, !dbg !804
  %exitcond31.not.i = icmp eq i64 %234, %remainder.i, !dbg !807
  br i1 %exitcond31.not.i, label %_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack24collect_bool_word_scalarNCINvB2_23collect_bool_words_withNCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1O_7visitor5retry21ExecuteDenseWithRetryINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB3D_EENtNtB3R_11row_visitor10RowVisitor19visit_deferred_boolB3w_bKB3D_NCINvXB4B_B4y_NtNtB1O_6row_fn5RowFn8dispatchB3M_E0NCB6t_s_0E0NCB3I_s_0B7f_E0NCINvB2_25collect_bool_words_avx512B1F_E0E0EB4B_.exit, label %bb8.us.i, !dbg !774, !llvm.loop !828

start.split.i:                                    ; preds = %bb12.i
  %view.i.i.i.i.i = getelementptr inbounds nuw i8, ptr %f, i64 8
  %view.val.i.i.i.i.i = load ptr, ptr %view.i.i.i.i.i, align 8, !alias.scope !757, !noalias !762, !nonnull !23, !align !363, !noundef !23
  %view.val1.i.i.i.i.i = load i64, ptr %179, align 8, !alias.scope !757, !noalias !762, !noundef !23
  br i1 %177, label %iter.check186, label %iter.check

iter.check186:                                    ; preds = %start.split.i
  %_6.i11.i.i.i.us12.i = load ptr, ptr %178, align 8, !alias.scope !768, !noalias !762, !nonnull !23, !align !363, !noundef !23
  %236 = trunc nuw i8 %_11.i.i.promoted.i to i1, !dbg !811
  %_0.sroa.0.0.i9.i.i.i.us15.i = load i64, ptr %_6.i11.i.i.i.us12.i, align 8, !noalias !786, !noundef !23
  %237 = icmp eq i64 %_0.sroa.0.0.i9.i.i.i.us15.i, 9223372036854775807
  %min.iters.check184 = icmp samesign ult i64 %remainder.i, 4, !dbg !774
  br i1 %min.iters.check184, label %bb8.us6.i.preheader, label %vector.main.loop.iter.check188, !dbg !774

vector.main.loop.iter.check188:                   ; preds = %iter.check186
  %min.iters.check187 = icmp samesign ult i64 %remainder.i, 16, !dbg !774
  br i1 %min.iters.check187, label %vec.epilog.ph222, label %vector.ph189, !dbg !774

vector.ph189:                                     ; preds = %vector.main.loop.iter.check188
  %n.mod.vf190 = and i64 %len, 12
  %n.vec191 = and i64 %len, 48
  %238 = insertelement <8 x i1> <i1 poison, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false>, i1 %236, i64 0
  %broadcast.splatinsert192 = insertelement <8 x i64> poison, i64 %_0.sroa.0.0.i9.i.i.i.us15.i, i64 0
  %broadcast.splat193 = shufflevector <8 x i64> %broadcast.splatinsert192, <8 x i64> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert194 = insertelement <8 x i1> poison, i1 %237, i64 0
  %broadcast.splat195 = shufflevector <8 x i1> %broadcast.splatinsert194, <8 x i1> poison, <8 x i32> zeroinitializer
  %invariant.gep = getelementptr i64, ptr %view.val.i.i.i.i.i, i64 %175, !dbg !774
  br label %vector.body200, !dbg !774

vector.body200:                                   ; preds = %vector.body200, %vector.ph189
  %index201 = phi i64 [ 0, %vector.ph189 ], [ %index.next210, %vector.body200 ], !dbg !787
  %vec.phi202 = phi <8 x i64> [ zeroinitializer, %vector.ph189 ], [ %253, %vector.body200 ]
  %vec.phi203 = phi <8 x i64> [ zeroinitializer, %vector.ph189 ], [ %254, %vector.body200 ]
  %vec.ind204 = phi <8 x i64> [ <i64 0, i64 1, i64 2, i64 3, i64 4, i64 5, i64 6, i64 7>, %vector.ph189 ], [ %vec.ind.next211, %vector.body200 ]
  %vec.phi205 = phi <8 x i1> [ %238, %vector.ph189 ], [ %246, %vector.body200 ]
  %vec.phi206 = phi <8 x i1> [ zeroinitializer, %vector.ph189 ], [ %248, %vector.body200 ]
  %step.add207 = add <8 x i64> %vec.ind204, splat (i64 8)
  %239 = extractelement <8 x i64> %vec.ind204, i64 0
  tail call void @llvm.experimental.noalias.scope.decl(metadata !793), !dbg !794
  tail call void @llvm.experimental.noalias.scope.decl(metadata !799), !dbg !800
  %gep = getelementptr i64, ptr %invariant.gep, i64 %239, !dbg !829
  %240 = getelementptr inbounds nuw i8, ptr %gep, i64 64, !dbg !834
  %wide.load208 = load <8 x i64>, ptr %gep, align 8, !dbg !834, !noalias !785
  %wide.load209 = load <8 x i64>, ptr %240, align 8, !dbg !834, !noalias !785
  tail call void @llvm.experimental.noalias.scope.decl(metadata !802), !dbg !800
  %241 = icmp eq <8 x i64> %wide.load208, splat (i64 9223372036854775807), !dbg !819
  %242 = icmp eq <8 x i64> %wide.load209, splat (i64 9223372036854775807), !dbg !819
  %243 = icmp slt <8 x i64> %wide.load208, %broadcast.splat193, !dbg !822
  %244 = icmp slt <8 x i64> %wide.load209, %broadcast.splat193, !dbg !822
  %245 = or <8 x i1> %241, %vec.phi205, !dbg !811
  %246 = or <8 x i1> %245, %broadcast.splat195, !dbg !811
  %247 = or <8 x i1> %242, %vec.phi206, !dbg !811
  %248 = or <8 x i1> %247, %broadcast.splat195, !dbg !811
  %249 = zext <8 x i1> %243 to <8 x i64>, !dbg !803
  %250 = zext <8 x i1> %244 to <8 x i64>, !dbg !803
  %251 = shl nuw <8 x i64> %249, %vec.ind204, !dbg !803
  %252 = shl nuw <8 x i64> %250, %step.add207, !dbg !803
  %253 = or <8 x i64> %251, %vec.phi202, !dbg !804
  %254 = or <8 x i64> %252, %vec.phi203, !dbg !804
  %index.next210 = add nuw i64 %index201, 16, !dbg !787
  %vec.ind.next211 = add <8 x i64> %vec.ind204, splat (i64 16)
  %255 = icmp eq i64 %index.next210, %n.vec191, !dbg !774
  br i1 %255, label %middle.block212, label %vector.body200, !dbg !774, !llvm.loop !835

middle.block212:                                  ; preds = %vector.body200
  %bin.rdx213 = or <8 x i64> %254, %253, !dbg !774
  %256 = tail call i64 @llvm.vector.reduce.or.v8i64(<8 x i64> %bin.rdx213), !dbg !774
  %bin.rdx214 = or <8 x i1> %247, %246, !dbg !774
  %257 = bitcast <8 x i1> %bin.rdx214 to i8, !dbg !774
  %258 = icmp ne i8 %257, 0, !dbg !774
  %cmp.n215 = icmp eq i64 %remainder.i, %n.vec191, !dbg !774
  br i1 %cmp.n215, label %_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack24collect_bool_word_scalarNCINvB2_23collect_bool_words_withNCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1O_7visitor5retry21ExecuteDenseWithRetryINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB3D_EENtNtB3R_11row_visitor10RowVisitor19visit_deferred_boolB3w_bKB3D_NCINvXB4B_B4y_NtNtB1O_6row_fn5RowFn8dispatchB3M_E0NCB6t_s_0E0NCB3I_s_0B7f_E0NCINvB2_25collect_bool_words_avx512B1F_E0E0EB4B_.exit, label %vec.epilog.iter.check220, !dbg !774

vec.epilog.iter.check220:                         ; preds = %middle.block212
  %min.epilog.iters.check221 = icmp eq i64 %n.mod.vf190, 0
  br i1 %min.epilog.iters.check221, label %bb8.us6.i.preheader, label %vec.epilog.ph222, !prof !404

vec.epilog.ph222:                                 ; preds = %vector.main.loop.iter.check188, %vec.epilog.iter.check220
  %bc.resume.val216 = phi i64 [ %n.vec191, %vec.epilog.iter.check220 ], [ 0, %vector.main.loop.iter.check188 ]
  %bc.merge.rdx217 = phi i64 [ %256, %vec.epilog.iter.check220 ], [ 0, %vector.main.loop.iter.check188 ]
  %bc.merge.rdx218 = phi i1 [ %258, %vec.epilog.iter.check220 ], [ %236, %vector.main.loop.iter.check188 ]
  %n.vec224 = and i64 %len, 60
  %259 = insertelement <4 x i64> <i64 poison, i64 0, i64 0, i64 0>, i64 %bc.merge.rdx217, i64 0
  %260 = insertelement <4 x i1> <i1 poison, i1 false, i1 false, i1 false>, i1 %bc.merge.rdx218, i64 0
  %broadcast.splatinsert225 = insertelement <4 x i64> poison, i64 %_0.sroa.0.0.i9.i.i.i.us15.i, i64 0
  %broadcast.splat226 = shufflevector <4 x i64> %broadcast.splatinsert225, <4 x i64> poison, <4 x i32> zeroinitializer
  %broadcast.splatinsert227 = insertelement <4 x i1> poison, i1 %237, i64 0
  %broadcast.splat228 = shufflevector <4 x i1> %broadcast.splatinsert227, <4 x i1> poison, <4 x i32> zeroinitializer
  %broadcast.splatinsert233 = insertelement <4 x i64> poison, i64 %bc.resume.val216, i64 0
  %broadcast.splat234 = shufflevector <4 x i64> %broadcast.splatinsert233, <4 x i64> poison, <4 x i32> zeroinitializer
  %induction235 = or disjoint <4 x i64> %broadcast.splat234, <i64 0, i64 1, i64 2, i64 3>
  %invariant.gep494 = getelementptr i64, ptr %view.val.i.i.i.i.i, i64 %175
  br label %vec.epilog.vector.body236

vec.epilog.vector.body236:                        ; preds = %vec.epilog.vector.body236, %vec.epilog.ph222
  %index237 = phi i64 [ %bc.resume.val216, %vec.epilog.ph222 ], [ %index.next242, %vec.epilog.vector.body236 ], !dbg !787
  %vec.phi238 = phi <4 x i64> [ %259, %vec.epilog.ph222 ], [ %268, %vec.epilog.vector.body236 ]
  %vec.ind239 = phi <4 x i64> [ %induction235, %vec.epilog.ph222 ], [ %vec.ind.next243, %vec.epilog.vector.body236 ]
  %vec.phi240 = phi <4 x i1> [ %260, %vec.epilog.ph222 ], [ %265, %vec.epilog.vector.body236 ]
  %261 = extractelement <4 x i64> %vec.ind239, i64 0
  tail call void @llvm.experimental.noalias.scope.decl(metadata !793), !dbg !794
  tail call void @llvm.experimental.noalias.scope.decl(metadata !799), !dbg !800
  %gep495 = getelementptr i64, ptr %invariant.gep494, i64 %261, !dbg !829
  %wide.load241 = load <4 x i64>, ptr %gep495, align 8, !dbg !834, !noalias !785
  tail call void @llvm.experimental.noalias.scope.decl(metadata !802), !dbg !800
  %262 = icmp eq <4 x i64> %wide.load241, splat (i64 9223372036854775807), !dbg !819
  %263 = or <4 x i1> %broadcast.splat228, %262, !dbg !819
  %264 = icmp slt <4 x i64> %wide.load241, %broadcast.splat226, !dbg !822
  %265 = or <4 x i1> %vec.phi240, %263, !dbg !811
  %266 = zext <4 x i1> %264 to <4 x i64>, !dbg !803
  %267 = shl nuw <4 x i64> %266, %vec.ind239, !dbg !803
  %268 = or <4 x i64> %267, %vec.phi238, !dbg !804
  %index.next242 = add nuw i64 %index237, 4, !dbg !787
  %vec.ind.next243 = add nuw nsw <4 x i64> %vec.ind239, splat (i64 4)
  %269 = icmp eq i64 %index.next242, %n.vec224, !dbg !774
  br i1 %269, label %vec.epilog.middle.block244, label %vec.epilog.vector.body236, !dbg !774, !llvm.loop !836

vec.epilog.middle.block244:                       ; preds = %vec.epilog.vector.body236
  %270 = tail call i64 @llvm.vector.reduce.or.v4i64(<4 x i64> %268), !dbg !774
  %271 = bitcast <4 x i1> %265 to i4, !dbg !774
  %272 = icmp ne i4 %271, 0, !dbg !774
  %cmp.n245 = icmp eq i64 %remainder.i, %n.vec224, !dbg !774
  br i1 %cmp.n245, label %_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack24collect_bool_word_scalarNCINvB2_23collect_bool_words_withNCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1O_7visitor5retry21ExecuteDenseWithRetryINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB3D_EENtNtB3R_11row_visitor10RowVisitor19visit_deferred_boolB3w_bKB3D_NCINvXB4B_B4y_NtNtB1O_6row_fn5RowFn8dispatchB3M_E0NCB6t_s_0E0NCB3I_s_0B7f_E0NCINvB2_25collect_bool_words_avx512B1F_E0E0EB4B_.exit, label %bb8.us6.i.preheader, !dbg !774

bb8.us6.i.preheader:                              ; preds = %iter.check186, %vec.epilog.iter.check220, %vec.epilog.middle.block244
  %packed.sroa.0.05.us7.i.ph = phi i64 [ 0, %iter.check186 ], [ %256, %vec.epilog.iter.check220 ], [ %270, %vec.epilog.middle.block244 ]
  %iter.sroa.0.04.us8.i.ph = phi i64 [ 0, %iter.check186 ], [ %n.vec191, %vec.epilog.iter.check220 ], [ %n.vec224, %vec.epilog.middle.block244 ]
  %.ph372 = phi i1 [ %236, %iter.check186 ], [ %258, %vec.epilog.iter.check220 ], [ %272, %vec.epilog.middle.block244 ]
  br label %bb8.us6.i, !dbg !774

bb8.us6.i:                                        ; preds = %bb8.us6.i.preheader, %bb8.us6.i
  %packed.sroa.0.05.us7.i = phi i64 [ %278, %bb8.us6.i ], [ %packed.sroa.0.05.us7.i.ph, %bb8.us6.i.preheader ]
  %iter.sroa.0.04.us8.i = phi i64 [ %277, %bb8.us6.i ], [ %iter.sroa.0.04.us8.i.ph, %bb8.us6.i.preheader ]
  %273 = phi i1 [ %276, %bb8.us6.i ], [ %.ph372, %bb8.us6.i.preheader ]
  %_4.i.us9.i = add nuw nsw i64 %iter.sroa.0.04.us8.i, %175, !dbg !825
  tail call void @llvm.experimental.noalias.scope.decl(metadata !793), !dbg !794
  tail call void @llvm.experimental.noalias.scope.decl(metadata !799), !dbg !800
  %_5.i.i.i.i.i.us.i = icmp ult i64 %_4.i.us9.i, %view.val1.i.i.i.i.i, !dbg !837
  tail call void @llvm.assume(i1 %_5.i.i.i.i.i.us.i), !dbg !838
  %_4.i.i.i.i.i.us.i = getelementptr inbounds nuw i64, ptr %view.val.i.i.i.i.i, i64 %_4.i.us9.i, !dbg !829
  %_0.sroa.0.0.i.i.i.i.us10.i = load i64, ptr %_4.i.i.i.i.i.us.i, align 8, !dbg !834, !noalias !785, !noundef !23
  tail call void @llvm.experimental.noalias.scope.decl(metadata !802), !dbg !800
  %274 = icmp eq i64 %_0.sroa.0.0.i.i.i.i.us10.i, 9223372036854775807, !dbg !819
  %_5.i.i.i.i.us17.i = icmp slt i64 %_0.sroa.0.0.i.i.i.i.us10.i, %_0.sroa.0.0.i9.i.i.i.us15.i, !dbg !822
  %275 = or i1 %274, %273, !dbg !811
  %276 = or i1 %275, %237, !dbg !811
  %277 = add nuw nsw i64 %iter.sroa.0.04.us8.i, 1, !dbg !787
  %_13.us18.i = zext i1 %_5.i.i.i.i.us17.i to i64, !dbg !803
  %_12.us19.i = shl nuw i64 %_13.us18.i, %iter.sroa.0.04.us8.i, !dbg !803
  %278 = or i64 %_12.us19.i, %packed.sroa.0.05.us7.i, !dbg !804
  %exitcond30.not.i = icmp eq i64 %277, %remainder.i, !dbg !807
  br i1 %exitcond30.not.i, label %_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack24collect_bool_word_scalarNCINvB2_23collect_bool_words_withNCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1O_7visitor5retry21ExecuteDenseWithRetryINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB3D_EENtNtB3R_11row_visitor10RowVisitor19visit_deferred_boolB3w_bKB3D_NCINvXB4B_B4y_NtNtB1O_6row_fn5RowFn8dispatchB3M_E0NCB6t_s_0E0NCB3I_s_0B7f_E0NCINvB2_25collect_bool_words_avx512B1F_E0E0EB4B_.exit, label %bb8.us6.i, !dbg !774, !llvm.loop !839

iter.check:                                       ; preds = %start.split.i
  %view.val.i4.i.i.i.i = load ptr, ptr %view.i3.i.i.i.i, align 8, !alias.scope !768, !noalias !762, !nonnull !23, !align !363, !noundef !23
  %view.val1.i5.i.i.i.i = load i64, ptr %178, align 8, !alias.scope !768, !noalias !762, !noundef !23
  %279 = trunc nuw i8 %_11.i.i.promoted.i to i1, !dbg !811
  %min.iters.check = icmp samesign ult i64 %remainder.i, 4, !dbg !774
  br i1 %min.iters.check, label %bb8.i3.preheader, label %vector.main.loop.iter.check, !dbg !774

vector.main.loop.iter.check:                      ; preds = %iter.check
  %min.iters.check136 = icmp samesign ult i64 %remainder.i, 16, !dbg !774
  br i1 %min.iters.check136, label %vec.epilog.ph, label %vector.ph137, !dbg !774

vector.ph137:                                     ; preds = %vector.main.loop.iter.check
  %n.mod.vf = and i64 %len, 12
  %n.vec = and i64 %len, 48
  %280 = insertelement <8 x i1> <i1 poison, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false>, i1 %279, i64 0
  br label %vector.body144, !dbg !774

vector.body144:                                   ; preds = %vector.body144, %vector.ph137
  %index145 = phi i64 [ 0, %vector.ph137 ], [ %index.next156, %vector.body144 ], !dbg !787
  %vec.phi146 = phi <8 x i64> [ zeroinitializer, %vector.ph137 ], [ %301, %vector.body144 ]
  %vec.phi147 = phi <8 x i64> [ zeroinitializer, %vector.ph137 ], [ %302, %vector.body144 ]
  %vec.ind148 = phi <8 x i64> [ <i64 0, i64 1, i64 2, i64 3, i64 4, i64 5, i64 6, i64 7>, %vector.ph137 ], [ %vec.ind.next157, %vector.body144 ]
  %vec.phi149 = phi <8 x i1> [ %280, %vector.ph137 ], [ %295, %vector.body144 ]
  %vec.phi150 = phi <8 x i1> [ zeroinitializer, %vector.ph137 ], [ %296, %vector.body144 ]
  %step.add151 = add <8 x i64> %vec.ind148, splat (i64 8)
  %281 = extractelement <8 x i64> %vec.ind148, i64 0
  %282 = add nuw nsw i64 %281, %175
  tail call void @llvm.experimental.noalias.scope.decl(metadata !793), !dbg !794
  tail call void @llvm.experimental.noalias.scope.decl(metadata !799), !dbg !800
  %283 = getelementptr inbounds nuw i64, ptr %view.val.i.i.i.i.i, i64 %282, !dbg !829
  %284 = getelementptr inbounds nuw i8, ptr %283, i64 64, !dbg !834
  %wide.load152 = load <8 x i64>, ptr %283, align 8, !dbg !834, !noalias !785
  %wide.load153 = load <8 x i64>, ptr %284, align 8, !dbg !834, !noalias !785
  tail call void @llvm.experimental.noalias.scope.decl(metadata !802), !dbg !800
  %285 = getelementptr inbounds nuw i64, ptr %view.val.i4.i.i.i.i, i64 %282, !dbg !813
  %286 = getelementptr inbounds nuw i8, ptr %285, i64 64, !dbg !818
  %wide.load154 = load <8 x i64>, ptr %285, align 8, !dbg !818, !noalias !786
  %wide.load155 = load <8 x i64>, ptr %286, align 8, !dbg !818, !noalias !786
  %287 = icmp eq <8 x i64> %wide.load152, splat (i64 9223372036854775807), !dbg !819
  %288 = icmp eq <8 x i64> %wide.load153, splat (i64 9223372036854775807), !dbg !819
  %289 = icmp eq <8 x i64> %wide.load154, splat (i64 9223372036854775807), !dbg !819
  %290 = icmp eq <8 x i64> %wide.load155, splat (i64 9223372036854775807), !dbg !819
  %291 = or <8 x i1> %287, %289, !dbg !819
  %292 = or <8 x i1> %288, %290, !dbg !819
  %293 = icmp slt <8 x i64> %wide.load152, %wide.load154, !dbg !822
  %294 = icmp slt <8 x i64> %wide.load153, %wide.load155, !dbg !822
  %295 = or <8 x i1> %vec.phi149, %291, !dbg !811
  %296 = or <8 x i1> %vec.phi150, %292, !dbg !811
  %297 = zext <8 x i1> %293 to <8 x i64>, !dbg !803
  %298 = zext <8 x i1> %294 to <8 x i64>, !dbg !803
  %299 = shl nuw <8 x i64> %297, %vec.ind148, !dbg !803
  %300 = shl nuw <8 x i64> %298, %step.add151, !dbg !803
  %301 = or <8 x i64> %299, %vec.phi146, !dbg !804
  %302 = or <8 x i64> %300, %vec.phi147, !dbg !804
  %index.next156 = add nuw i64 %index145, 16, !dbg !787
  %vec.ind.next157 = add <8 x i64> %vec.ind148, splat (i64 16)
  %303 = icmp eq i64 %index.next156, %n.vec, !dbg !774
  br i1 %303, label %middle.block158, label %vector.body144, !dbg !774, !llvm.loop !840

middle.block158:                                  ; preds = %vector.body144
  %bin.rdx159 = or <8 x i64> %302, %301, !dbg !774
  %304 = tail call i64 @llvm.vector.reduce.or.v8i64(<8 x i64> %bin.rdx159), !dbg !774
  %bin.rdx160 = or <8 x i1> %296, %295, !dbg !774
  %305 = bitcast <8 x i1> %bin.rdx160 to i8, !dbg !774
  %306 = icmp ne i8 %305, 0, !dbg !774
  %cmp.n = icmp eq i64 %remainder.i, %n.vec, !dbg !774
  br i1 %cmp.n, label %_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack24collect_bool_word_scalarNCINvB2_23collect_bool_words_withNCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1O_7visitor5retry21ExecuteDenseWithRetryINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB3D_EENtNtB3R_11row_visitor10RowVisitor19visit_deferred_boolB3w_bKB3D_NCINvXB4B_B4y_NtNtB1O_6row_fn5RowFn8dispatchB3M_E0NCB6t_s_0E0NCB3I_s_0B7f_E0NCINvB2_25collect_bool_words_avx512B1F_E0E0EB4B_.exit, label %vec.epilog.iter.check, !dbg !774

vec.epilog.iter.check:                            ; preds = %middle.block158
  %min.epilog.iters.check = icmp eq i64 %n.mod.vf, 0
  br i1 %min.epilog.iters.check, label %bb8.i3.preheader, label %vec.epilog.ph, !prof !404

vec.epilog.ph:                                    ; preds = %vector.main.loop.iter.check, %vec.epilog.iter.check
  %bc.resume.val = phi i64 [ %n.vec, %vec.epilog.iter.check ], [ 0, %vector.main.loop.iter.check ]
  %bc.merge.rdx = phi i64 [ %304, %vec.epilog.iter.check ], [ 0, %vector.main.loop.iter.check ]
  %bc.merge.rdx161 = phi i1 [ %306, %vec.epilog.iter.check ], [ %279, %vector.main.loop.iter.check ]
  %n.vec163 = and i64 %len, 60
  %307 = insertelement <4 x i64> <i64 poison, i64 0, i64 0, i64 0>, i64 %bc.merge.rdx, i64 0
  %308 = insertelement <4 x i1> <i1 poison, i1 false, i1 false, i1 false>, i1 %bc.merge.rdx161, i64 0
  %broadcast.splatinsert170 = insertelement <4 x i64> poison, i64 %bc.resume.val, i64 0
  %broadcast.splat171 = shufflevector <4 x i64> %broadcast.splatinsert170, <4 x i64> poison, <4 x i32> zeroinitializer
  %induction = or disjoint <4 x i64> %broadcast.splat171, <i64 0, i64 1, i64 2, i64 3>
  br label %vec.epilog.vector.body

vec.epilog.vector.body:                           ; preds = %vec.epilog.vector.body, %vec.epilog.ph
  %index172 = phi i64 [ %bc.resume.val, %vec.epilog.ph ], [ %index.next178, %vec.epilog.vector.body ], !dbg !787
  %vec.phi173 = phi <4 x i64> [ %307, %vec.epilog.ph ], [ %320, %vec.epilog.vector.body ]
  %vec.ind174 = phi <4 x i64> [ %induction, %vec.epilog.ph ], [ %vec.ind.next179, %vec.epilog.vector.body ]
  %vec.phi175 = phi <4 x i1> [ %308, %vec.epilog.ph ], [ %317, %vec.epilog.vector.body ]
  %309 = extractelement <4 x i64> %vec.ind174, i64 0
  %310 = add nuw nsw i64 %309, %175
  tail call void @llvm.experimental.noalias.scope.decl(metadata !793), !dbg !794
  tail call void @llvm.experimental.noalias.scope.decl(metadata !799), !dbg !800
  %311 = getelementptr inbounds nuw i64, ptr %view.val.i.i.i.i.i, i64 %310, !dbg !829
  %wide.load176 = load <4 x i64>, ptr %311, align 8, !dbg !834, !noalias !785
  tail call void @llvm.experimental.noalias.scope.decl(metadata !802), !dbg !800
  %312 = getelementptr inbounds nuw i64, ptr %view.val.i4.i.i.i.i, i64 %310, !dbg !813
  %wide.load177 = load <4 x i64>, ptr %312, align 8, !dbg !818, !noalias !786
  %313 = icmp eq <4 x i64> %wide.load176, splat (i64 9223372036854775807), !dbg !819
  %314 = icmp eq <4 x i64> %wide.load177, splat (i64 9223372036854775807), !dbg !819
  %315 = or <4 x i1> %313, %314, !dbg !819
  %316 = icmp slt <4 x i64> %wide.load176, %wide.load177, !dbg !822
  %317 = or <4 x i1> %vec.phi175, %315, !dbg !811
  %318 = zext <4 x i1> %316 to <4 x i64>, !dbg !803
  %319 = shl nuw <4 x i64> %318, %vec.ind174, !dbg !803
  %320 = or <4 x i64> %319, %vec.phi173, !dbg !804
  %index.next178 = add nuw i64 %index172, 4, !dbg !787
  %vec.ind.next179 = add nuw nsw <4 x i64> %vec.ind174, splat (i64 4)
  %321 = icmp eq i64 %index.next178, %n.vec163, !dbg !774
  br i1 %321, label %vec.epilog.middle.block, label %vec.epilog.vector.body, !dbg !774, !llvm.loop !841

vec.epilog.middle.block:                          ; preds = %vec.epilog.vector.body
  %322 = tail call i64 @llvm.vector.reduce.or.v4i64(<4 x i64> %320), !dbg !774
  %323 = bitcast <4 x i1> %317 to i4, !dbg !774
  %324 = icmp ne i4 %323, 0, !dbg !774
  %cmp.n180 = icmp eq i64 %remainder.i, %n.vec163, !dbg !774
  br i1 %cmp.n180, label %_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack24collect_bool_word_scalarNCINvB2_23collect_bool_words_withNCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1O_7visitor5retry21ExecuteDenseWithRetryINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB3D_EENtNtB3R_11row_visitor10RowVisitor19visit_deferred_boolB3w_bKB3D_NCINvXB4B_B4y_NtNtB1O_6row_fn5RowFn8dispatchB3M_E0NCB6t_s_0E0NCB3I_s_0B7f_E0NCINvB2_25collect_bool_words_avx512B1F_E0E0EB4B_.exit, label %bb8.i3.preheader, !dbg !774

bb8.i3.preheader:                                 ; preds = %iter.check, %vec.epilog.iter.check, %vec.epilog.middle.block
  %packed.sroa.0.05.i.ph = phi i64 [ 0, %iter.check ], [ %304, %vec.epilog.iter.check ], [ %322, %vec.epilog.middle.block ]
  %iter.sroa.0.04.i.ph = phi i64 [ 0, %iter.check ], [ %n.vec, %vec.epilog.iter.check ], [ %n.vec163, %vec.epilog.middle.block ]
  %.ph382 = phi i1 [ %279, %iter.check ], [ %306, %vec.epilog.iter.check ], [ %324, %vec.epilog.middle.block ]
  br label %bb8.i3, !dbg !774

bb8.i3:                                           ; preds = %bb8.i3.preheader, %bb8.i3
  %packed.sroa.0.05.i = phi i64 [ %330, %bb8.i3 ], [ %packed.sroa.0.05.i.ph, %bb8.i3.preheader ]
  %iter.sroa.0.04.i = phi i64 [ %329, %bb8.i3 ], [ %iter.sroa.0.04.i.ph, %bb8.i3.preheader ]
  %325 = phi i1 [ %328, %bb8.i3 ], [ %.ph382, %bb8.i3.preheader ]
  %_4.i.i = add nuw nsw i64 %iter.sroa.0.04.i, %175, !dbg !825
  tail call void @llvm.experimental.noalias.scope.decl(metadata !793), !dbg !794
  tail call void @llvm.experimental.noalias.scope.decl(metadata !799), !dbg !800
  %_5.i.i.i.i.i.i = icmp ult i64 %_4.i.i, %view.val1.i.i.i.i.i, !dbg !837
  tail call void @llvm.assume(i1 %_5.i.i.i.i.i.i), !dbg !838
  %_4.i.i.i.i.i.i = getelementptr inbounds nuw i64, ptr %view.val.i.i.i.i.i, i64 %_4.i.i, !dbg !829
  %_0.sroa.0.0.i.i.i.i.i = load i64, ptr %_4.i.i.i.i.i.i, align 8, !dbg !834, !noalias !785, !noundef !23
  tail call void @llvm.experimental.noalias.scope.decl(metadata !802), !dbg !800
  %_5.i.i6.i.i.i.i = icmp ult i64 %_4.i.i, %view.val1.i5.i.i.i.i, !dbg !826
  tail call void @llvm.assume(i1 %_5.i.i6.i.i.i.i), !dbg !827
  %_4.i.i7.i.i.i.i = getelementptr inbounds nuw i64, ptr %view.val.i4.i.i.i.i, i64 %_4.i.i, !dbg !813
  %_0.sroa.0.0.i9.i.i.i.i = load i64, ptr %_4.i.i7.i.i.i.i, align 8, !dbg !818, !noalias !786, !noundef !23
  %326 = icmp eq i64 %_0.sroa.0.0.i.i.i.i.i, 9223372036854775807, !dbg !819
  %327 = icmp eq i64 %_0.sroa.0.0.i9.i.i.i.i, 9223372036854775807, !dbg !819
  %_6.sroa.0.0.i.i.i.i.i = or i1 %326, %327, !dbg !819
  %_5.i.i.i.i.i = icmp slt i64 %_0.sroa.0.0.i.i.i.i.i, %_0.sroa.0.0.i9.i.i.i.i, !dbg !822
  %328 = or i1 %325, %_6.sroa.0.0.i.i.i.i.i, !dbg !811
  %329 = add nuw nsw i64 %iter.sroa.0.04.i, 1, !dbg !787
  %_13.i = zext i1 %_5.i.i.i.i.i to i64, !dbg !803
  %_12.i = shl nuw i64 %_13.i, %iter.sroa.0.04.i, !dbg !803
  %330 = or i64 %_12.i, %packed.sroa.0.05.i, !dbg !804
  %exitcond.not.i = icmp eq i64 %329, %remainder.i, !dbg !807
  br i1 %exitcond.not.i, label %_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack24collect_bool_word_scalarNCINvB2_23collect_bool_words_withNCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1O_7visitor5retry21ExecuteDenseWithRetryINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB3D_EENtNtB3R_11row_visitor10RowVisitor19visit_deferred_boolB3w_bKB3D_NCINvXB4B_B4y_NtNtB1O_6row_fn5RowFn8dispatchB3M_E0NCB6t_s_0E0NCB3I_s_0B7f_E0NCINvB2_25collect_bool_words_avx512B1F_E0E0EB4B_.exit, label %bb8.i3, !dbg !774, !llvm.loop !842

_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack24collect_bool_word_scalarNCINvB2_23collect_bool_words_withNCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1O_7visitor5retry21ExecuteDenseWithRetryINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB3D_EENtNtB3R_11row_visitor10RowVisitor19visit_deferred_boolB3w_bKB3D_NCINvXB4B_B4y_NtNtB1O_6row_fn5RowFn8dispatchB3M_E0NCB6t_s_0E0NCB3I_s_0B7f_E0NCINvB2_25collect_bool_words_avx512B1F_E0E0EB4B_.exit.loopexit: ; preds = %bb8.us.us.i, %vec.epilog.middle.block354, %middle.block332
  %.lcssa = phi i64 [ %190, %vec.epilog.middle.block354 ], [ %185, %middle.block332 ], [ %192, %bb8.us.us.i ], !dbg !804
  %331 = trunc nuw i8 %_11.i.i.promoted.i to i1, !dbg !811
  %332 = icmp eq i64 %_0.sroa.0.0.i.i.i.i.us.us.i, 9223372036854775807
  %333 = icmp eq i64 %_0.sroa.0.0.i9.i.i.i.us.us.i, 9223372036854775807
  %_6.sroa.0.0.i.i.i.i.us.us.i = or i1 %332, %333
  %334 = or i1 %_6.sroa.0.0.i.i.i.i.us.us.i, %331, !dbg !811
  br label %_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack24collect_bool_word_scalarNCINvB2_23collect_bool_words_withNCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1O_7visitor5retry21ExecuteDenseWithRetryINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB3D_EENtNtB3R_11row_visitor10RowVisitor19visit_deferred_boolB3w_bKB3D_NCINvXB4B_B4y_NtNtB1O_6row_fn5RowFn8dispatchB3M_E0NCB6t_s_0E0NCB3I_s_0B7f_E0NCINvB2_25collect_bool_words_avx512B1F_E0E0EB4B_.exit, !dbg !843

_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack24collect_bool_word_scalarNCINvB2_23collect_bool_words_withNCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1O_7visitor5retry21ExecuteDenseWithRetryINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB3D_EENtNtB3R_11row_visitor10RowVisitor19visit_deferred_boolB3w_bKB3D_NCINvXB4B_B4y_NtNtB1O_6row_fn5RowFn8dispatchB3M_E0NCB6t_s_0E0NCB3I_s_0B7f_E0NCINvB2_25collect_bool_words_avx512B1F_E0E0EB4B_.exit: ; preds = %bb8.i3, %bb8.us6.i, %bb8.us.i, %middle.block158, %vec.epilog.middle.block, %middle.block212, %vec.epilog.middle.block244, %middle.block277, %vec.epilog.middle.block309, %_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack24collect_bool_word_scalarNCINvB2_23collect_bool_words_withNCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1O_7visitor5retry21ExecuteDenseWithRetryINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB3D_EENtNtB3R_11row_visitor10RowVisitor19visit_deferred_boolB3w_bKB3D_NCINvXB4B_B4y_NtNtB1O_6row_fn5RowFn8dispatchB3M_E0NCB6t_s_0E0NCB3I_s_0B7f_E0NCINvB2_25collect_bool_words_avx512B1F_E0E0EB4B_.exit.loopexit
  %.lcssa118.sink = phi i1 [ %276, %bb8.us6.i ], [ %233, %bb8.us.i ], [ %334, %_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack24collect_bool_word_scalarNCINvB2_23collect_bool_words_withNCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1O_7visitor5retry21ExecuteDenseWithRetryINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB3D_EENtNtB3R_11row_visitor10RowVisitor19visit_deferred_boolB3w_bKB3D_NCINvXB4B_B4y_NtNtB1O_6row_fn5RowFn8dispatchB3M_E0NCB6t_s_0E0NCB3I_s_0B7f_E0NCINvB2_25collect_bool_words_avx512B1F_E0E0EB4B_.exit.loopexit ], [ %229, %vec.epilog.middle.block309 ], [ %215, %middle.block277 ], [ %272, %vec.epilog.middle.block244 ], [ %258, %middle.block212 ], [ %324, %vec.epilog.middle.block ], [ %306, %middle.block158 ], [ %328, %bb8.i3 ]
  %.us-phi.i = phi i64 [ %278, %bb8.us6.i ], [ %235, %bb8.us.i ], [ %.lcssa, %_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack24collect_bool_word_scalarNCINvB2_23collect_bool_words_withNCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1O_7visitor5retry21ExecuteDenseWithRetryINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB3D_EENtNtB3R_11row_visitor10RowVisitor19visit_deferred_boolB3w_bKB3D_NCINvXB4B_B4y_NtNtB1O_6row_fn5RowFn8dispatchB3M_E0NCB6t_s_0E0NCB3I_s_0B7f_E0NCINvB2_25collect_bool_words_avx512B1F_E0E0EB4B_.exit.loopexit ], [ %227, %vec.epilog.middle.block309 ], [ %213, %middle.block277 ], [ %270, %vec.epilog.middle.block244 ], [ %256, %middle.block212 ], [ %322, %vec.epilog.middle.block ], [ %304, %middle.block158 ], [ %330, %bb8.i3 ], !dbg !844
  %335 = zext i1 %.lcssa118.sink to i8
  store i8 %335, ptr %_11.i.i.i, align 8, !dbg !811, !alias.scope !771, !noalias !762
  %_41.i = icmp samesign ult i64 %full5.i, %words.1, !dbg !843
  br i1 %_41.i, label %bb14.i, label %panic.i, !dbg !843

bb14.i:                                           ; preds = %_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack24collect_bool_word_scalarNCINvB2_23collect_bool_words_withNCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1O_7visitor5retry21ExecuteDenseWithRetryINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB3D_EENtNtB3R_11row_visitor10RowVisitor19visit_deferred_boolB3w_bKB3D_NCINvXB4B_B4y_NtNtB1O_6row_fn5RowFn8dispatchB3M_E0NCB6t_s_0E0NCB3I_s_0B7f_E0NCINvB2_25collect_bool_words_avx512B1F_E0E0EB4B_.exit
  store i64 %.us-phi.i, ptr %_52.i, align 8, !dbg !843, !alias.scope !635, !noalias !845
  br label %_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack23collect_bool_words_withNCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1g_7visitor5retry21ExecuteDenseWithRetryINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB35_EENtNtB3j_11row_visitor10RowVisitor19visit_deferred_boolB2Y_bKB35_NCINvXB43_B40_NtNtB1g_6row_fn5RowFn8dispatchB3e_E0NCB5V_s_0E0NCB3a_s_0B6H_E0NCINvB2_25collect_bool_words_avx512B17_E0EB43_.exit, !dbg !847

panic.i:                                          ; preds = %_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack24collect_bool_word_scalarNCINvB2_23collect_bool_words_withNCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1O_7visitor5retry21ExecuteDenseWithRetryINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB3D_EENtNtB3R_11row_visitor10RowVisitor19visit_deferred_boolB3w_bKB3D_NCINvXB4B_B4y_NtNtB1O_6row_fn5RowFn8dispatchB3M_E0NCB6t_s_0E0NCB3I_s_0B7f_E0NCINvB2_25collect_bool_words_avx512B1F_E0E0EB4B_.exit
; call core::panicking::panic_bounds_check
  tail call void @_RNvNtCsc36rpYXAlPq_4core9panicking18panic_bounds_check(i64 noundef %full5.i, i64 noundef range(i64 0, 1152921504606846976) %words.1, ptr noalias nofree noundef readonly align 8 captures(address, read_provenance) dereferenceable(24) @alloc_1b2922caae1b461da04857d3d02eae5b) #21, !dbg !843
  unreachable

_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack23collect_bool_words_withNCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1g_7visitor5retry21ExecuteDenseWithRetryINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB35_EENtNtB3j_11row_visitor10RowVisitor19visit_deferred_boolB2Y_bKB35_NCINvXB43_B40_NtNtB1g_6row_fn5RowFn8dispatchB3e_E0NCB5V_s_0E0NCB3a_s_0B6H_E0NCINvB2_25collect_bool_words_avx512B17_E0EB43_.exit: ; preds = %bb14.i, %bb5.i
  ret void, !dbg !848
}
