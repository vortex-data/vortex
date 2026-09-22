define hidden void @_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack23collect_bool_words_avx2NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1g_7visitor5retry21ExecuteDenseWithRetryINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB35_EENtNtB3j_11row_visitor10RowVisitor19visit_deferred_boolB2Y_bKB35_NCINvXB43_B40_NtNtB1g_6row_fn5RowFn8dispatchB3e_E0NCB5V_s_0E0NCB3a_s_0B6H_E0EB43_(ptr noalias nofree noundef nonnull writeonly align 8 captures(address) %words.0, i64 noundef range(i64 0, 1152921504606846976) %words.1, i64 noundef %len, ptr noalias nofree noundef align 8 captures(none) dereferenceable(64) %f) unnamed_addr #2 personality ptr @rust_eh_personality !dbg !280 {
start:
  tail call void @llvm.experimental.noalias.scope.decl(metadata !285), !dbg !288
  %full5.i = lshr i64 %len, 6, !dbg !289
  %remainder.i = and i64 %len, 63, !dbg !292
  %_42.not.i = icmp samesign ugt i64 %full5.i, %words.1
  br i1 %_42.not.i, label %bb22.i, label %bb21.i, !dbg !294, !prof !313

bb22.i:                                           ; preds = %start
; call core::slice::index::slice_index_fail
  tail call void @_RNvNtNtCsc36rpYXAlPq_4core5slice5index16slice_index_fail(i64 noundef 0, i64 noundef %full5.i, i64 noundef range(i64 0, 1152921504606846976) %words.1, ptr noalias nofree noundef readonly align 8 captures(address, read_provenance) dereferenceable(24) @alloc_ebbd31bf42d91f579e6aa57e93569175) #21, !dbg !314, !noalias !285
  unreachable

bb21.i:                                           ; preds = %start
  %_52.i.idx = shl nuw nsw i64 %full5.i, 3, !dbg !315
  %_52.i = getelementptr inbounds nuw i8, ptr %words.0, i64 %_52.i.idx, !dbg !315
  %_7.i.i58 = icmp eq i64 %full5.i, 0, !dbg !332
  br i1 %_7.i.i58, label %bb5.i, label %bb4.i.lr.ph, !dbg !350

bb4.i.lr.ph:                                      ; preds = %bb21.i
  %_3.i.i.i = load i64, ptr %f, align 8, !range !351, !alias.scope !352, !noalias !357, !noundef !23
  %0 = trunc nuw i64 %_3.i.i.i to i1
  %_6.i.i = getelementptr inbounds nuw i8, ptr %f, i64 24
  %_3.i1.i.i = load i64, ptr %_6.i.i, align 8, !range !351, !alias.scope !360, !noalias !357, !noundef !23
  %1 = trunc nuw i64 %_3.i1.i.i to i1
  %_11.i = getelementptr inbounds nuw i8, ptr %f, i64 56
  %view.i.i.i = getelementptr inbounds nuw i8, ptr %f, i64 8
  %view.val.i.i.i = load ptr, ptr %view.i.i.i, align 8, !nonnull !23, !align !363
  %view.i3.i.i = getelementptr inbounds nuw i8, ptr %f, i64 32
  %view.val.i4.i.i = load ptr, ptr %view.i3.i.i, align 8, !nonnull !23, !align !363
  %2 = getelementptr inbounds nuw i8, ptr %f, i64 40
  %view.val1.i5.i.i = load i64, ptr %2, align 8
  %_6.i11.i.i.cast = inttoptr i64 %view.val1.i5.i.i to ptr
  %_11.i.promoted61 = load i8, ptr %_11.i, align 8, !alias.scope !364, !noalias !357
  br i1 %0, label %bb4.i.lr.ph.split.us, label %bb4.i.preheader

bb4.i.preheader:                                  ; preds = %bb4.i.lr.ph
  %3 = trunc nuw i8 %_11.i.promoted61 to i1, !dbg !367
  %4 = getelementptr inbounds nuw i8, ptr %view.val.i.i.i, i64 128
  %5 = getelementptr inbounds nuw i8, ptr %view.val.i.i.i, i64 256
  %6 = getelementptr inbounds nuw i8, ptr %view.val.i.i.i, i64 384
  br label %bb4.i

bb4.i.lr.ph.split.us:                             ; preds = %bb4.i.lr.ph
  %7 = getelementptr inbounds nuw i8, ptr %f, i64 16
  %view.val1.i.i.i = load i64, ptr %7, align 8
  %8 = inttoptr i64 %view.val1.i.i.i to ptr
  %_0.sroa.0.0.i.i.i.us.us = load i64, ptr %8, align 8, !noalias !391, !noundef !23
  %9 = icmp eq i64 %_0.sroa.0.0.i.i.i.us.us, 9223372036854775807
  %10 = trunc nuw i8 %_11.i.promoted61 to i1, !dbg !367
  br i1 %1, label %iter.check, label %bb4.i.us.preheader

bb4.i.us.preheader:                               ; preds = %bb4.i.lr.ph.split.us
  %broadcast.splatinsert91 = insertelement <4 x i1> poison, i1 %9, i64 0
  %broadcast.splat92 = shufflevector <4 x i1> %broadcast.splatinsert91, <4 x i1> poison, <4 x i32> zeroinitializer
  %broadcast.splatinsert89 = insertelement <4 x i64> poison, i64 %_0.sroa.0.0.i.i.i.us.us, i64 0
  %broadcast.splat90 = shufflevector <4 x i64> %broadcast.splatinsert89, <4 x i64> poison, <4 x i32> zeroinitializer
  %11 = getelementptr inbounds nuw i8, ptr %view.val.i4.i.i, i64 128
  %12 = getelementptr inbounds nuw i8, ptr %view.val.i4.i.i, i64 256
  %13 = getelementptr inbounds nuw i8, ptr %view.val.i4.i.i, i64 384
  br label %bb4.i.us, !dbg !392

iter.check:                                       ; preds = %bb4.i.lr.ph.split.us
  %_0.sroa.0.0.i9.i.i.us.us.us.us = load i64, ptr %_6.i11.i.i.cast, align 8, !noalias !399, !noundef !23
  %_5.i.i.i.us.us.us.us = icmp slt i64 %_0.sroa.0.0.i.i.i.us.us, %_0.sroa.0.0.i9.i.i.us.us.us.us
  %14 = select i1 %_5.i.i.i.us.us.us.us, i256 1334440654591915542993625911497130241, i256 0
  %15 = shl nuw nsw i256 %14, 128
  %16 = or disjoint i256 %15, %14
  %17 = bitcast i256 %16 to <32 x i8>
  %18 = icmp ne <32 x i8> %17, zeroinitializer
  %_12.i.i.us.us = bitcast <32 x i1> %18 to i32
  %lo_mask.i.i.us.us = zext i32 %_12.i.i.us.us to i64
  %_21.i.i.us.us = shl nuw i64 %lo_mask.i.i.us.us, 32
  %_0.i.i.us.us = or disjoint i64 %_21.i.i.us.us, %lo_mask.i.i.us.us
  %19 = add nsw i64 %_52.i.idx, -8, !dbg !350
  %20 = lshr exact i64 %19, 3, !dbg !350
  %21 = add nuw nsw i64 %20, 1, !dbg !350
  %min.iters.check = icmp ult i64 %19, 24, !dbg !350
  br i1 %min.iters.check, label %bb4.i.us.us.preheader, label %vector.main.loop.iter.check, !dbg !350

vector.main.loop.iter.check:                      ; preds = %iter.check
  %min.iters.check109 = icmp ult i64 %19, 120, !dbg !350
  br i1 %min.iters.check109, label %vec.epilog.ph, label %vector.ph110, !dbg !350

vector.ph110:                                     ; preds = %vector.main.loop.iter.check
  %n.mod.vf = and i64 %21, 12
  %n.vec = and i64 %21, 4611686018427387888
  %broadcast.splatinsert111 = insertelement <4 x i64> poison, i64 %_0.i.i.us.us, i64 0
  %broadcast.splat112 = shufflevector <4 x i64> %broadcast.splatinsert111, <4 x i64> poison, <4 x i32> zeroinitializer
  br label %vector.body113, !dbg !350

vector.body113:                                   ; preds = %vector.body113, %vector.ph110
  %index114 = phi i64 [ 0, %vector.ph110 ], [ %index.next115, %vector.body113 ]
  %22 = shl i64 %index114, 3
  %next.gep = getelementptr i8, ptr %words.0, i64 %22
  %23 = getelementptr i8, ptr %next.gep, i64 32, !dbg !400
  %24 = getelementptr i8, ptr %next.gep, i64 64, !dbg !400
  %25 = getelementptr i8, ptr %next.gep, i64 96, !dbg !400
  store <4 x i64> %broadcast.splat112, ptr %next.gep, align 8, !dbg !400
  store <4 x i64> %broadcast.splat112, ptr %23, align 8, !dbg !400
  store <4 x i64> %broadcast.splat112, ptr %24, align 8, !dbg !400
  store <4 x i64> %broadcast.splat112, ptr %25, align 8, !dbg !400
  %index.next115 = add nuw i64 %index114, 16
  %26 = icmp eq i64 %index.next115, %n.vec, !dbg !350
  br i1 %26, label %middle.block116, label %vector.body113, !dbg !350, !llvm.loop !401

middle.block116:                                  ; preds = %vector.body113
  %cmp.n = icmp eq i64 %21, %n.vec, !dbg !350
  br i1 %cmp.n, label %bb1.i.bb5.i_crit_edge.loopexit.loopexit, label %vec.epilog.iter.check, !dbg !350

vec.epilog.iter.check:                            ; preds = %middle.block116
  %27 = shl i64 %n.vec, 3
  %ind.end = getelementptr i8, ptr %words.0, i64 %27
  %min.epilog.iters.check = icmp eq i64 %n.mod.vf, 0
  br i1 %min.epilog.iters.check, label %bb4.i.us.us.preheader, label %vec.epilog.ph, !prof !404

vec.epilog.ph:                                    ; preds = %vector.main.loop.iter.check, %vec.epilog.iter.check
  %vec.epilog.resume.val = phi i64 [ %n.vec, %vec.epilog.iter.check ], [ 0, %vector.main.loop.iter.check ]
  %n.vec118 = and i64 %21, 4611686018427387900
  %28 = shl i64 %n.vec118, 3
  %29 = getelementptr i8, ptr %words.0, i64 %28
  %broadcast.splatinsert119 = insertelement <4 x i64> poison, i64 %_0.i.i.us.us, i64 0
  %broadcast.splat120 = shufflevector <4 x i64> %broadcast.splatinsert119, <4 x i64> poison, <4 x i32> zeroinitializer
  br label %vec.epilog.vector.body

vec.epilog.vector.body:                           ; preds = %vec.epilog.vector.body, %vec.epilog.ph
  %index121 = phi i64 [ %vec.epilog.resume.val, %vec.epilog.ph ], [ %index.next123, %vec.epilog.vector.body ]
  %offset.idx = shl i64 %index121, 3
  %next.gep122 = getelementptr i8, ptr %words.0, i64 %offset.idx
  store <4 x i64> %broadcast.splat120, ptr %next.gep122, align 8, !dbg !400
  %index.next123 = add nuw i64 %index121, 4
  %30 = icmp eq i64 %index.next123, %n.vec118, !dbg !350
  br i1 %30, label %vec.epilog.middle.block, label %vec.epilog.vector.body, !dbg !350, !llvm.loop !405

vec.epilog.middle.block:                          ; preds = %vec.epilog.vector.body
  %cmp.n124 = icmp eq i64 %21, %n.vec118, !dbg !350
  br i1 %cmp.n124, label %bb1.i.bb5.i_crit_edge.loopexit.loopexit, label %bb4.i.us.us.preheader, !dbg !350

bb4.i.us.us.preheader:                            ; preds = %iter.check, %vec.epilog.iter.check, %vec.epilog.middle.block
  %iter.i.sroa.0.060.us.us.ph = phi ptr [ %words.0, %iter.check ], [ %ind.end, %vec.epilog.iter.check ], [ %29, %vec.epilog.middle.block ]
  br label %bb4.i.us.us, !dbg !350

bb4.i.us.us:                                      ; preds = %bb4.i.us.us.preheader, %bb4.i.us.us
  %iter.i.sroa.0.060.us.us = phi ptr [ %_17.i.i.us.us, %bb4.i.us.us ], [ %iter.i.sroa.0.060.us.us.ph, %bb4.i.us.us.preheader ]
  %_17.i.i.us.us = getelementptr inbounds nuw i8, ptr %iter.i.sroa.0.060.us.us, i64 8, !dbg !406
  store i64 %_0.i.i.us.us, ptr %iter.i.sroa.0.060.us.us, align 8, !dbg !400
  %_7.i.i.us.us = icmp eq ptr %_17.i.i.us.us, %_52.i, !dbg !332
  br i1 %_7.i.i.us.us, label %bb1.i.bb5.i_crit_edge.loopexit.loopexit, label %bb4.i.us.us, !dbg !350, !llvm.loop !409

bb4.i.us:                                         ; preds = %bb4.i.us.preheader, %bb4.i.us
  %.us-phi62.us = phi i1 [ %111, %bb4.i.us ], [ %10, %bb4.i.us.preheader ]
  %iter.i.sroa.0.060.us = phi ptr [ %_17.i.i.us, %bb4.i.us ], [ %words.0, %bb4.i.us.preheader ]
  %iter.i.sroa.7.059.us = phi i64 [ %_9.0.i.us, %bb4.i.us ], [ 0, %bb4.i.us.preheader ]
  %31 = insertelement <4 x i1> <i1 poison, i1 false, i1 false, i1 false>, i1 %.us-phi62.us, i64 0
  tail call void @llvm.experimental.noalias.scope.decl(metadata !410), !dbg !411
  tail call void @llvm.experimental.noalias.scope.decl(metadata !412), !dbg !413
  tail call void @llvm.experimental.noalias.scope.decl(metadata !422), !dbg !413
  %.idx = shl i64 %iter.i.sroa.7.059.us, 9, !dbg !423
  %32 = getelementptr inbounds nuw i8, ptr %view.val.i4.i.i, i64 %.idx, !dbg !423
  %33 = getelementptr inbounds nuw i8, ptr %32, i64 32, !dbg !438
  %34 = getelementptr inbounds nuw i8, ptr %32, i64 64, !dbg !438
  %35 = getelementptr inbounds nuw i8, ptr %32, i64 96, !dbg !438
  %wide.load99 = load <4 x i64>, ptr %32, align 8, !dbg !438, !noalias !399
  %wide.load100 = load <4 x i64>, ptr %33, align 8, !dbg !438, !noalias !399
  %wide.load101 = load <4 x i64>, ptr %34, align 8, !dbg !438, !noalias !399
  %wide.load102 = load <4 x i64>, ptr %35, align 8, !dbg !438, !noalias !399
  %36 = icmp eq <4 x i64> %wide.load99, splat (i64 9223372036854775807), !dbg !439
  %37 = icmp eq <4 x i64> %wide.load100, splat (i64 9223372036854775807), !dbg !439
  %38 = icmp eq <4 x i64> %wide.load101, splat (i64 9223372036854775807), !dbg !439
  %39 = icmp eq <4 x i64> %wide.load102, splat (i64 9223372036854775807), !dbg !439
  %40 = icmp slt <4 x i64> %broadcast.splat90, %wide.load99, !dbg !454
  %41 = icmp slt <4 x i64> %broadcast.splat90, %wide.load100, !dbg !454
  %42 = icmp slt <4 x i64> %broadcast.splat90, %wide.load101, !dbg !454
  %43 = icmp slt <4 x i64> %broadcast.splat90, %wide.load102, !dbg !454
  %44 = or <4 x i1> %31, %36, !dbg !367
  %45 = zext <4 x i1> %40 to <4 x i8>, !dbg !455
  %46 = zext <4 x i1> %41 to <4 x i8>, !dbg !455
  %47 = zext <4 x i1> %42 to <4 x i8>, !dbg !455
  %48 = zext <4 x i1> %43 to <4 x i8>, !dbg !455
  %bools.i.sroa.0.0.vec.expand305 = shufflevector <4 x i8> %45, <4 x i8> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !455
  %bools.i.sroa.0.4.vec.expand312 = shufflevector <4 x i8> %46, <4 x i8> poison, <32 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !455
  %bools.i.sroa.0.4.vecblend313 = shufflevector <32 x i8> %bools.i.sroa.0.0.vec.expand305, <32 x i8> %bools.i.sroa.0.4.vec.expand312, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 36, i32 37, i32 38, i32 39, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !455
  %bools.i.sroa.0.8.vec.expand318 = shufflevector <4 x i8> %47, <4 x i8> poison, <32 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !455
  %bools.i.sroa.0.8.vecblend319 = shufflevector <32 x i8> %bools.i.sroa.0.4.vecblend313, <32 x i8> %bools.i.sroa.0.8.vec.expand318, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 40, i32 41, i32 42, i32 43, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !455
  %bools.i.sroa.0.12.vec.expand324 = shufflevector <4 x i8> %48, <4 x i8> poison, <32 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !455
  %bools.i.sroa.0.12.vecblend325 = shufflevector <32 x i8> %bools.i.sroa.0.8.vecblend319, <32 x i8> %bools.i.sroa.0.12.vec.expand324, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 44, i32 45, i32 46, i32 47, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !455
  %.idx.1 = shl i64 %iter.i.sroa.7.059.us, 9, !dbg !423
  %49 = getelementptr inbounds nuw i8, ptr %11, i64 %.idx.1, !dbg !423
  %50 = getelementptr inbounds nuw i8, ptr %49, i64 32, !dbg !438
  %51 = getelementptr inbounds nuw i8, ptr %49, i64 64, !dbg !438
  %52 = getelementptr inbounds nuw i8, ptr %49, i64 96, !dbg !438
  %wide.load99.1 = load <4 x i64>, ptr %49, align 8, !dbg !438, !noalias !456
  %wide.load100.1 = load <4 x i64>, ptr %50, align 8, !dbg !438, !noalias !456
  %wide.load101.1 = load <4 x i64>, ptr %51, align 8, !dbg !438, !noalias !456
  %wide.load102.1 = load <4 x i64>, ptr %52, align 8, !dbg !438, !noalias !456
  %53 = icmp eq <4 x i64> %wide.load99.1, splat (i64 9223372036854775807), !dbg !439
  %54 = icmp eq <4 x i64> %wide.load100.1, splat (i64 9223372036854775807), !dbg !439
  %55 = icmp eq <4 x i64> %wide.load101.1, splat (i64 9223372036854775807), !dbg !439
  %56 = icmp eq <4 x i64> %wide.load102.1, splat (i64 9223372036854775807), !dbg !439
  %57 = icmp slt <4 x i64> %broadcast.splat90, %wide.load99.1, !dbg !454
  %58 = icmp slt <4 x i64> %broadcast.splat90, %wide.load100.1, !dbg !454
  %59 = icmp slt <4 x i64> %broadcast.splat90, %wide.load101.1, !dbg !454
  %60 = icmp slt <4 x i64> %broadcast.splat90, %wide.load102.1, !dbg !454
  %61 = or <4 x i1> %44, %53, !dbg !367
  %62 = or <4 x i1> %37, %54, !dbg !367
  %63 = or <4 x i1> %38, %55, !dbg !367
  %64 = or <4 x i1> %39, %56, !dbg !367
  %65 = zext <4 x i1> %57 to <4 x i8>, !dbg !455
  %66 = zext <4 x i1> %58 to <4 x i8>, !dbg !455
  %67 = zext <4 x i1> %59 to <4 x i8>, !dbg !455
  %68 = zext <4 x i1> %60 to <4 x i8>, !dbg !455
  %bools.i.sroa.0.16.vec.expand330 = shufflevector <4 x i8> %65, <4 x i8> poison, <32 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !455
  %bools.i.sroa.0.16.vecblend331 = shufflevector <32 x i8> %bools.i.sroa.0.12.vecblend325, <32 x i8> %bools.i.sroa.0.16.vec.expand330, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 48, i32 49, i32 50, i32 51, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !455
  %bools.i.sroa.0.20.vec.expand336 = shufflevector <4 x i8> %66, <4 x i8> poison, <32 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !455
  %bools.i.sroa.0.20.vecblend337 = shufflevector <32 x i8> %bools.i.sroa.0.16.vecblend331, <32 x i8> %bools.i.sroa.0.20.vec.expand336, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 52, i32 53, i32 54, i32 55, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !455
  %bools.i.sroa.0.24.vec.expand342 = shufflevector <4 x i8> %67, <4 x i8> poison, <32 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !455
  %bools.i.sroa.0.24.vecblend343 = shufflevector <32 x i8> %bools.i.sroa.0.20.vecblend337, <32 x i8> %bools.i.sroa.0.24.vec.expand342, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 56, i32 57, i32 58, i32 59, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !455
  %bools.i.sroa.0.28.vec.expand348 = shufflevector <4 x i8> %68, <4 x i8> poison, <32 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3>, !dbg !455
  %bools.i.sroa.0.28.vecblend349 = shufflevector <32 x i8> %bools.i.sroa.0.24.vecblend343, <32 x i8> %bools.i.sroa.0.28.vec.expand348, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 60, i32 61, i32 62, i32 63>, !dbg !455
  %.idx.2 = shl i64 %iter.i.sroa.7.059.us, 9, !dbg !423
  %69 = getelementptr inbounds nuw i8, ptr %12, i64 %.idx.2, !dbg !423
  %70 = getelementptr inbounds nuw i8, ptr %69, i64 32, !dbg !438
  %71 = getelementptr inbounds nuw i8, ptr %69, i64 64, !dbg !438
  %72 = getelementptr inbounds nuw i8, ptr %69, i64 96, !dbg !438
  %wide.load99.2 = load <4 x i64>, ptr %69, align 8, !dbg !438, !noalias !459
  %wide.load100.2 = load <4 x i64>, ptr %70, align 8, !dbg !438, !noalias !459
  %wide.load101.2 = load <4 x i64>, ptr %71, align 8, !dbg !438, !noalias !459
  %wide.load102.2 = load <4 x i64>, ptr %72, align 8, !dbg !438, !noalias !459
  %73 = icmp eq <4 x i64> %wide.load99.2, splat (i64 9223372036854775807), !dbg !439
  %74 = icmp eq <4 x i64> %wide.load100.2, splat (i64 9223372036854775807), !dbg !439
  %75 = icmp eq <4 x i64> %wide.load101.2, splat (i64 9223372036854775807), !dbg !439
  %76 = icmp eq <4 x i64> %wide.load102.2, splat (i64 9223372036854775807), !dbg !439
  %77 = icmp slt <4 x i64> %broadcast.splat90, %wide.load99.2, !dbg !454
  %78 = icmp slt <4 x i64> %broadcast.splat90, %wide.load100.2, !dbg !454
  %79 = icmp slt <4 x i64> %broadcast.splat90, %wide.load101.2, !dbg !454
  %80 = icmp slt <4 x i64> %broadcast.splat90, %wide.load102.2, !dbg !454
  %81 = or <4 x i1> %61, %73, !dbg !367
  %82 = or <4 x i1> %62, %74, !dbg !367
  %83 = or <4 x i1> %63, %75, !dbg !367
  %84 = or <4 x i1> %64, %76, !dbg !367
  %85 = zext <4 x i1> %77 to <4 x i8>, !dbg !455
  %86 = zext <4 x i1> %78 to <4 x i8>, !dbg !455
  %87 = zext <4 x i1> %79 to <4 x i8>, !dbg !455
  %88 = zext <4 x i1> %80 to <4 x i8>, !dbg !455
  %bools.i.sroa.44.32.vec.expand357 = shufflevector <4 x i8> %85, <4 x i8> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !455
  %bools.i.sroa.44.36.vec.expand363 = shufflevector <4 x i8> %86, <4 x i8> poison, <32 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !455
  %bools.i.sroa.44.36.vecblend364 = shufflevector <32 x i8> %bools.i.sroa.44.32.vec.expand357, <32 x i8> %bools.i.sroa.44.36.vec.expand363, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 36, i32 37, i32 38, i32 39, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !455
  %bools.i.sroa.44.40.vec.expand369 = shufflevector <4 x i8> %87, <4 x i8> poison, <32 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !455
  %bools.i.sroa.44.40.vecblend370 = shufflevector <32 x i8> %bools.i.sroa.44.36.vecblend364, <32 x i8> %bools.i.sroa.44.40.vec.expand369, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 40, i32 41, i32 42, i32 43, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !455
  %bools.i.sroa.44.44.vec.expand375 = shufflevector <4 x i8> %88, <4 x i8> poison, <32 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !455
  %bools.i.sroa.44.44.vecblend376 = shufflevector <32 x i8> %bools.i.sroa.44.40.vecblend370, <32 x i8> %bools.i.sroa.44.44.vec.expand375, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 44, i32 45, i32 46, i32 47, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !455
  %.idx.3 = shl i64 %iter.i.sroa.7.059.us, 9, !dbg !423
  %89 = getelementptr inbounds nuw i8, ptr %13, i64 %.idx.3, !dbg !423
  %90 = getelementptr inbounds nuw i8, ptr %89, i64 32, !dbg !438
  %91 = getelementptr inbounds nuw i8, ptr %89, i64 64, !dbg !438
  %92 = getelementptr inbounds nuw i8, ptr %89, i64 96, !dbg !438
  %wide.load99.3 = load <4 x i64>, ptr %89, align 8, !dbg !438, !noalias !462
  %wide.load100.3 = load <4 x i64>, ptr %90, align 8, !dbg !438, !noalias !462
  %wide.load101.3 = load <4 x i64>, ptr %91, align 8, !dbg !438, !noalias !462
  %wide.load102.3 = load <4 x i64>, ptr %92, align 8, !dbg !438, !noalias !462
  %93 = icmp eq <4 x i64> %wide.load99.3, splat (i64 9223372036854775807), !dbg !439
  %94 = icmp eq <4 x i64> %wide.load100.3, splat (i64 9223372036854775807), !dbg !439
  %95 = icmp eq <4 x i64> %wide.load101.3, splat (i64 9223372036854775807), !dbg !439
  %96 = icmp eq <4 x i64> %wide.load102.3, splat (i64 9223372036854775807), !dbg !439
  %97 = icmp slt <4 x i64> %broadcast.splat90, %wide.load99.3, !dbg !454
  %98 = icmp slt <4 x i64> %broadcast.splat90, %wide.load100.3, !dbg !454
  %99 = icmp slt <4 x i64> %broadcast.splat90, %wide.load101.3, !dbg !454
  %100 = icmp slt <4 x i64> %broadcast.splat90, %wide.load102.3, !dbg !454
  %101 = or <4 x i1> %81, %93, !dbg !367
  %102 = or <4 x i1> %101, %broadcast.splat92, !dbg !367
  %103 = or <4 x i1> %82, %94, !dbg !367
  %104 = or <4 x i1> %83, %95, !dbg !367
  %105 = or <4 x i1> %84, %96, !dbg !367
  %106 = zext <4 x i1> %97 to <4 x i8>, !dbg !455
  %107 = zext <4 x i1> %98 to <4 x i8>, !dbg !455
  %108 = zext <4 x i1> %99 to <4 x i8>, !dbg !455
  %109 = zext <4 x i1> %100 to <4 x i8>, !dbg !455
  %bools.i.sroa.44.48.vec.expand381 = shufflevector <4 x i8> %106, <4 x i8> poison, <32 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !455
  %bools.i.sroa.44.48.vecblend382 = shufflevector <32 x i8> %bools.i.sroa.44.44.vecblend376, <32 x i8> %bools.i.sroa.44.48.vec.expand381, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 48, i32 49, i32 50, i32 51, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !455
  %bools.i.sroa.44.52.vec.expand387 = shufflevector <4 x i8> %107, <4 x i8> poison, <32 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !455
  %bools.i.sroa.44.52.vecblend388 = shufflevector <32 x i8> %bools.i.sroa.44.48.vecblend382, <32 x i8> %bools.i.sroa.44.52.vec.expand387, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 52, i32 53, i32 54, i32 55, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !455
  %bools.i.sroa.44.56.vec.expand393 = shufflevector <4 x i8> %108, <4 x i8> poison, <32 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !455
  %bools.i.sroa.44.56.vecblend394 = shufflevector <32 x i8> %bools.i.sroa.44.52.vecblend388, <32 x i8> %bools.i.sroa.44.56.vec.expand393, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 56, i32 57, i32 58, i32 59, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !455
  %bools.i.sroa.44.60.vec.expand399 = shufflevector <4 x i8> %109, <4 x i8> poison, <32 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3>, !dbg !455
  %bools.i.sroa.44.60.vecblend400 = shufflevector <32 x i8> %bools.i.sroa.44.56.vecblend394, <32 x i8> %bools.i.sroa.44.60.vec.expand399, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 60, i32 61, i32 62, i32 63>, !dbg !455
  %bin.rdx105 = or <4 x i1> %103, %102, !dbg !392
  %bin.rdx106 = or <4 x i1> %104, %bin.rdx105, !dbg !392
  %bin.rdx107 = or <4 x i1> %105, %bin.rdx106, !dbg !392
  %110 = bitcast <4 x i1> %bin.rdx107 to i4, !dbg !392
  %111 = icmp ne i4 %110, 0, !dbg !392
  %_17.i.i.us = getelementptr inbounds nuw i8, ptr %iter.i.sroa.0.060.us, i64 8, !dbg !406
  %_9.0.i.us = add nuw nsw i64 %iter.i.sroa.7.059.us, 1, !dbg !465
  %112 = shufflevector <32 x i8> %bools.i.sroa.0.28.vecblend349, <32 x i8> %bools.i.sroa.44.60.vecblend400, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>, !dbg !468
  %113 = icmp ne <64 x i8> %112, zeroinitializer, !dbg !468
  store <64 x i1> %113, ptr %iter.i.sroa.0.060.us, align 8, !dbg !400
  %_7.i.i.us = icmp eq ptr %_17.i.i.us, %_52.i, !dbg !332
  br i1 %_7.i.i.us, label %bb1.i.bb5.i_crit_edge, label %bb4.i.us, !dbg !350

bb4.i:                                            ; preds = %bb4.i.preheader, %bb9.i.split
  %.us-phi62 = phi i1 [ %.us-phi56.in, %bb9.i.split ], [ %3, %bb4.i.preheader ]
  %iter.i.sroa.0.060 = phi ptr [ %_17.i.i, %bb9.i.split ], [ %words.0, %bb4.i.preheader ]
  %iter.i.sroa.7.059 = phi i64 [ %_9.0.i, %bb9.i.split ], [ 0, %bb4.i.preheader ]
  %_17.i.i = getelementptr inbounds nuw i8, ptr %iter.i.sroa.0.060, i64 8, !dbg !406
  %_9.0.i = add nuw nsw i64 %iter.i.sroa.7.059, 1, !dbg !465
  %offset2.i = shl i64 %iter.i.sroa.7.059, 6, !dbg !480
  br i1 %1, label %vector.body, label %vector.body75

vector.body75:                                    ; preds = %bb4.i
  %114 = insertelement <4 x i1> <i1 poison, i1 false, i1 false, i1 false>, i1 %.us-phi62, i64 0
  tail call void @llvm.experimental.noalias.scope.decl(metadata !410), !dbg !411
  tail call void @llvm.experimental.noalias.scope.decl(metadata !412), !dbg !413
  %115 = getelementptr inbounds nuw i64, ptr %view.val.i.i.i, i64 %offset2.i, !dbg !481
  %116 = getelementptr inbounds nuw i8, ptr %115, i64 32, !dbg !486
  %wide.load79 = load <4 x i64>, ptr %115, align 8, !dbg !486, !noalias !391
  %wide.load80 = load <4 x i64>, ptr %116, align 8, !dbg !486, !noalias !391
  tail call void @llvm.experimental.noalias.scope.decl(metadata !422), !dbg !413
  %117 = getelementptr inbounds nuw i64, ptr %view.val.i4.i.i, i64 %offset2.i, !dbg !423
  %118 = getelementptr inbounds nuw i8, ptr %117, i64 32, !dbg !438
  %wide.load81 = load <4 x i64>, ptr %117, align 8, !dbg !438, !noalias !399
  %wide.load82 = load <4 x i64>, ptr %118, align 8, !dbg !438, !noalias !399
  %119 = icmp eq <4 x i64> %wide.load79, splat (i64 9223372036854775807), !dbg !439
  %120 = icmp eq <4 x i64> %wide.load80, splat (i64 9223372036854775807), !dbg !439
  %121 = icmp eq <4 x i64> %wide.load81, splat (i64 9223372036854775807), !dbg !439
  %122 = icmp eq <4 x i64> %wide.load82, splat (i64 9223372036854775807), !dbg !439
  %123 = or <4 x i1> %119, %121, !dbg !439
  %124 = or <4 x i1> %120, %122, !dbg !439
  %125 = icmp slt <4 x i64> %wide.load79, %wide.load81, !dbg !454
  %126 = icmp slt <4 x i64> %wide.load80, %wide.load82, !dbg !454
  %127 = or <4 x i1> %114, %123, !dbg !367
  %128 = zext <4 x i1> %125 to <4 x i8>, !dbg !455
  %129 = zext <4 x i1> %126 to <4 x i8>, !dbg !455
  %bools.i.sroa.0.0.vec.expand = shufflevector <4 x i8> %128, <4 x i8> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !455
  %bools.i.sroa.0.4.vec.expand = shufflevector <4 x i8> %129, <4 x i8> poison, <32 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !455
  %bools.i.sroa.0.4.vecblend = shufflevector <32 x i8> %bools.i.sroa.0.0.vec.expand, <32 x i8> %bools.i.sroa.0.4.vec.expand, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 36, i32 37, i32 38, i32 39, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !455
  %130 = or disjoint i64 %offset2.i, 8, !dbg !487
  %131 = getelementptr inbounds nuw i64, ptr %view.val.i.i.i, i64 %130, !dbg !481
  %132 = getelementptr inbounds nuw i8, ptr %131, i64 32, !dbg !486
  %wide.load79.1 = load <4 x i64>, ptr %131, align 8, !dbg !486, !noalias !488
  %wide.load80.1 = load <4 x i64>, ptr %132, align 8, !dbg !486, !noalias !488
  %133 = getelementptr inbounds nuw i64, ptr %view.val.i4.i.i, i64 %130, !dbg !423
  %134 = getelementptr inbounds nuw i8, ptr %133, i64 32, !dbg !438
  %wide.load81.1 = load <4 x i64>, ptr %133, align 8, !dbg !438, !noalias !491
  %wide.load82.1 = load <4 x i64>, ptr %134, align 8, !dbg !438, !noalias !491
  %135 = icmp eq <4 x i64> %wide.load79.1, splat (i64 9223372036854775807), !dbg !439
  %136 = icmp eq <4 x i64> %wide.load80.1, splat (i64 9223372036854775807), !dbg !439
  %137 = icmp eq <4 x i64> %wide.load81.1, splat (i64 9223372036854775807), !dbg !439
  %138 = icmp eq <4 x i64> %wide.load82.1, splat (i64 9223372036854775807), !dbg !439
  %139 = or <4 x i1> %135, %137, !dbg !439
  %140 = or <4 x i1> %136, %138, !dbg !439
  %141 = icmp slt <4 x i64> %wide.load79.1, %wide.load81.1, !dbg !454
  %142 = icmp slt <4 x i64> %wide.load80.1, %wide.load82.1, !dbg !454
  %143 = or <4 x i1> %127, %139, !dbg !367
  %144 = or <4 x i1> %124, %140, !dbg !367
  %145 = zext <4 x i1> %141 to <4 x i8>, !dbg !455
  %146 = zext <4 x i1> %142 to <4 x i8>, !dbg !455
  %bools.i.sroa.0.8.vec.expand = shufflevector <4 x i8> %145, <4 x i8> poison, <32 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !455
  %bools.i.sroa.0.8.vecblend = shufflevector <32 x i8> %bools.i.sroa.0.4.vecblend, <32 x i8> %bools.i.sroa.0.8.vec.expand, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 40, i32 41, i32 42, i32 43, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !455
  %bools.i.sroa.0.12.vec.expand = shufflevector <4 x i8> %146, <4 x i8> poison, <32 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !455
  %bools.i.sroa.0.12.vecblend = shufflevector <32 x i8> %bools.i.sroa.0.8.vecblend, <32 x i8> %bools.i.sroa.0.12.vec.expand, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 44, i32 45, i32 46, i32 47, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !455
  %147 = or disjoint i64 %offset2.i, 16, !dbg !487
  %148 = getelementptr inbounds nuw i64, ptr %view.val.i.i.i, i64 %147, !dbg !481
  %149 = getelementptr inbounds nuw i8, ptr %148, i64 32, !dbg !486
  %wide.load79.2 = load <4 x i64>, ptr %148, align 8, !dbg !486, !noalias !493
  %wide.load80.2 = load <4 x i64>, ptr %149, align 8, !dbg !486, !noalias !493
  %150 = getelementptr inbounds nuw i64, ptr %view.val.i4.i.i, i64 %147, !dbg !423
  %151 = getelementptr inbounds nuw i8, ptr %150, i64 32, !dbg !438
  %wide.load81.2 = load <4 x i64>, ptr %150, align 8, !dbg !438, !noalias !496
  %wide.load82.2 = load <4 x i64>, ptr %151, align 8, !dbg !438, !noalias !496
  %152 = icmp eq <4 x i64> %wide.load79.2, splat (i64 9223372036854775807), !dbg !439
  %153 = icmp eq <4 x i64> %wide.load80.2, splat (i64 9223372036854775807), !dbg !439
  %154 = icmp eq <4 x i64> %wide.load81.2, splat (i64 9223372036854775807), !dbg !439
  %155 = icmp eq <4 x i64> %wide.load82.2, splat (i64 9223372036854775807), !dbg !439
  %156 = or <4 x i1> %152, %154, !dbg !439
  %157 = or <4 x i1> %153, %155, !dbg !439
  %158 = icmp slt <4 x i64> %wide.load79.2, %wide.load81.2, !dbg !454
  %159 = icmp slt <4 x i64> %wide.load80.2, %wide.load82.2, !dbg !454
  %160 = or <4 x i1> %143, %156, !dbg !367
  %161 = or <4 x i1> %144, %157, !dbg !367
  %162 = zext <4 x i1> %158 to <4 x i8>, !dbg !455
  %163 = zext <4 x i1> %159 to <4 x i8>, !dbg !455
  %bools.i.sroa.0.16.vec.expand = shufflevector <4 x i8> %162, <4 x i8> poison, <32 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !455
  %bools.i.sroa.0.16.vecblend = shufflevector <32 x i8> %bools.i.sroa.0.12.vecblend, <32 x i8> %bools.i.sroa.0.16.vec.expand, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 48, i32 49, i32 50, i32 51, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !455
  %bools.i.sroa.0.20.vec.expand = shufflevector <4 x i8> %163, <4 x i8> poison, <32 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !455
  %bools.i.sroa.0.20.vecblend = shufflevector <32 x i8> %bools.i.sroa.0.16.vecblend, <32 x i8> %bools.i.sroa.0.20.vec.expand, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 52, i32 53, i32 54, i32 55, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !455
  %164 = or disjoint i64 %offset2.i, 24, !dbg !487
  %165 = getelementptr inbounds nuw i64, ptr %view.val.i.i.i, i64 %164, !dbg !481
  %166 = getelementptr inbounds nuw i8, ptr %165, i64 32, !dbg !486
  %wide.load79.3 = load <4 x i64>, ptr %165, align 8, !dbg !486, !noalias !498
  %wide.load80.3 = load <4 x i64>, ptr %166, align 8, !dbg !486, !noalias !498
  %167 = getelementptr inbounds nuw i64, ptr %view.val.i4.i.i, i64 %164, !dbg !423
  %168 = getelementptr inbounds nuw i8, ptr %167, i64 32, !dbg !438
  %wide.load81.3 = load <4 x i64>, ptr %167, align 8, !dbg !438, !noalias !501
  %wide.load82.3 = load <4 x i64>, ptr %168, align 8, !dbg !438, !noalias !501
  %169 = icmp eq <4 x i64> %wide.load79.3, splat (i64 9223372036854775807), !dbg !439
  %170 = icmp eq <4 x i64> %wide.load80.3, splat (i64 9223372036854775807), !dbg !439
  %171 = icmp eq <4 x i64> %wide.load81.3, splat (i64 9223372036854775807), !dbg !439
  %172 = icmp eq <4 x i64> %wide.load82.3, splat (i64 9223372036854775807), !dbg !439
  %173 = or <4 x i1> %169, %171, !dbg !439
  %174 = or <4 x i1> %170, %172, !dbg !439
  %175 = icmp slt <4 x i64> %wide.load79.3, %wide.load81.3, !dbg !454
  %176 = icmp slt <4 x i64> %wide.load80.3, %wide.load82.3, !dbg !454
  %177 = or <4 x i1> %160, %173, !dbg !367
  %178 = or <4 x i1> %161, %174, !dbg !367
  %179 = zext <4 x i1> %175 to <4 x i8>, !dbg !455
  %180 = zext <4 x i1> %176 to <4 x i8>, !dbg !455
  %bools.i.sroa.0.24.vec.expand = shufflevector <4 x i8> %179, <4 x i8> poison, <32 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !455
  %bools.i.sroa.0.24.vecblend = shufflevector <32 x i8> %bools.i.sroa.0.20.vecblend, <32 x i8> %bools.i.sroa.0.24.vec.expand, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 56, i32 57, i32 58, i32 59, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !455
  %bools.i.sroa.0.28.vec.expand = shufflevector <4 x i8> %180, <4 x i8> poison, <32 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3>, !dbg !455
  %bools.i.sroa.0.28.vecblend = shufflevector <32 x i8> %bools.i.sroa.0.24.vecblend, <32 x i8> %bools.i.sroa.0.28.vec.expand, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 60, i32 61, i32 62, i32 63>, !dbg !455
  %181 = or disjoint i64 %offset2.i, 32, !dbg !487
  %182 = getelementptr inbounds nuw i64, ptr %view.val.i.i.i, i64 %181, !dbg !481
  %183 = getelementptr inbounds nuw i8, ptr %182, i64 32, !dbg !486
  %wide.load79.4 = load <4 x i64>, ptr %182, align 8, !dbg !486, !noalias !503
  %wide.load80.4 = load <4 x i64>, ptr %183, align 8, !dbg !486, !noalias !503
  %184 = getelementptr inbounds nuw i64, ptr %view.val.i4.i.i, i64 %181, !dbg !423
  %185 = getelementptr inbounds nuw i8, ptr %184, i64 32, !dbg !438
  %wide.load81.4 = load <4 x i64>, ptr %184, align 8, !dbg !438, !noalias !506
  %wide.load82.4 = load <4 x i64>, ptr %185, align 8, !dbg !438, !noalias !506
  %186 = icmp eq <4 x i64> %wide.load79.4, splat (i64 9223372036854775807), !dbg !439
  %187 = icmp eq <4 x i64> %wide.load80.4, splat (i64 9223372036854775807), !dbg !439
  %188 = icmp eq <4 x i64> %wide.load81.4, splat (i64 9223372036854775807), !dbg !439
  %189 = icmp eq <4 x i64> %wide.load82.4, splat (i64 9223372036854775807), !dbg !439
  %190 = or <4 x i1> %186, %188, !dbg !439
  %191 = or <4 x i1> %187, %189, !dbg !439
  %192 = icmp slt <4 x i64> %wide.load79.4, %wide.load81.4, !dbg !454
  %193 = icmp slt <4 x i64> %wide.load80.4, %wide.load82.4, !dbg !454
  %194 = or <4 x i1> %177, %190, !dbg !367
  %195 = or <4 x i1> %178, %191, !dbg !367
  %196 = zext <4 x i1> %192 to <4 x i8>, !dbg !455
  %197 = zext <4 x i1> %193 to <4 x i8>, !dbg !455
  %bools.i.sroa.44.32.vec.expand = shufflevector <4 x i8> %196, <4 x i8> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !455
  %bools.i.sroa.44.36.vec.expand = shufflevector <4 x i8> %197, <4 x i8> poison, <32 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !455
  %bools.i.sroa.44.36.vecblend = shufflevector <32 x i8> %bools.i.sroa.44.32.vec.expand, <32 x i8> %bools.i.sroa.44.36.vec.expand, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 36, i32 37, i32 38, i32 39, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !455
  %198 = or disjoint i64 %offset2.i, 40, !dbg !487
  %199 = getelementptr inbounds nuw i64, ptr %view.val.i.i.i, i64 %198, !dbg !481
  %200 = getelementptr inbounds nuw i8, ptr %199, i64 32, !dbg !486
  %wide.load79.5 = load <4 x i64>, ptr %199, align 8, !dbg !486, !noalias !508
  %wide.load80.5 = load <4 x i64>, ptr %200, align 8, !dbg !486, !noalias !508
  %201 = getelementptr inbounds nuw i64, ptr %view.val.i4.i.i, i64 %198, !dbg !423
  %202 = getelementptr inbounds nuw i8, ptr %201, i64 32, !dbg !438
  %wide.load81.5 = load <4 x i64>, ptr %201, align 8, !dbg !438, !noalias !511
  %wide.load82.5 = load <4 x i64>, ptr %202, align 8, !dbg !438, !noalias !511
  %203 = icmp eq <4 x i64> %wide.load79.5, splat (i64 9223372036854775807), !dbg !439
  %204 = icmp eq <4 x i64> %wide.load80.5, splat (i64 9223372036854775807), !dbg !439
  %205 = icmp eq <4 x i64> %wide.load81.5, splat (i64 9223372036854775807), !dbg !439
  %206 = icmp eq <4 x i64> %wide.load82.5, splat (i64 9223372036854775807), !dbg !439
  %207 = or <4 x i1> %203, %205, !dbg !439
  %208 = or <4 x i1> %204, %206, !dbg !439
  %209 = icmp slt <4 x i64> %wide.load79.5, %wide.load81.5, !dbg !454
  %210 = icmp slt <4 x i64> %wide.load80.5, %wide.load82.5, !dbg !454
  %211 = or <4 x i1> %194, %207, !dbg !367
  %212 = or <4 x i1> %195, %208, !dbg !367
  %213 = zext <4 x i1> %209 to <4 x i8>, !dbg !455
  %214 = zext <4 x i1> %210 to <4 x i8>, !dbg !455
  %bools.i.sroa.44.40.vec.expand = shufflevector <4 x i8> %213, <4 x i8> poison, <32 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !455
  %bools.i.sroa.44.40.vecblend = shufflevector <32 x i8> %bools.i.sroa.44.36.vecblend, <32 x i8> %bools.i.sroa.44.40.vec.expand, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 40, i32 41, i32 42, i32 43, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !455
  %bools.i.sroa.44.44.vec.expand = shufflevector <4 x i8> %214, <4 x i8> poison, <32 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !455
  %bools.i.sroa.44.44.vecblend = shufflevector <32 x i8> %bools.i.sroa.44.40.vecblend, <32 x i8> %bools.i.sroa.44.44.vec.expand, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 44, i32 45, i32 46, i32 47, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !455
  %215 = or disjoint i64 %offset2.i, 48, !dbg !487
  %216 = getelementptr inbounds nuw i64, ptr %view.val.i.i.i, i64 %215, !dbg !481
  %217 = getelementptr inbounds nuw i8, ptr %216, i64 32, !dbg !486
  %wide.load79.6 = load <4 x i64>, ptr %216, align 8, !dbg !486, !noalias !513
  %wide.load80.6 = load <4 x i64>, ptr %217, align 8, !dbg !486, !noalias !513
  %218 = getelementptr inbounds nuw i64, ptr %view.val.i4.i.i, i64 %215, !dbg !423
  %219 = getelementptr inbounds nuw i8, ptr %218, i64 32, !dbg !438
  %wide.load81.6 = load <4 x i64>, ptr %218, align 8, !dbg !438, !noalias !516
  %wide.load82.6 = load <4 x i64>, ptr %219, align 8, !dbg !438, !noalias !516
  %220 = icmp eq <4 x i64> %wide.load79.6, splat (i64 9223372036854775807), !dbg !439
  %221 = icmp eq <4 x i64> %wide.load80.6, splat (i64 9223372036854775807), !dbg !439
  %222 = icmp eq <4 x i64> %wide.load81.6, splat (i64 9223372036854775807), !dbg !439
  %223 = icmp eq <4 x i64> %wide.load82.6, splat (i64 9223372036854775807), !dbg !439
  %224 = or <4 x i1> %220, %222, !dbg !439
  %225 = or <4 x i1> %221, %223, !dbg !439
  %226 = icmp slt <4 x i64> %wide.load79.6, %wide.load81.6, !dbg !454
  %227 = icmp slt <4 x i64> %wide.load80.6, %wide.load82.6, !dbg !454
  %228 = or <4 x i1> %211, %224, !dbg !367
  %229 = or <4 x i1> %212, %225, !dbg !367
  %230 = zext <4 x i1> %226 to <4 x i8>, !dbg !455
  %231 = zext <4 x i1> %227 to <4 x i8>, !dbg !455
  %bools.i.sroa.44.48.vec.expand = shufflevector <4 x i8> %230, <4 x i8> poison, <32 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !455
  %bools.i.sroa.44.48.vecblend = shufflevector <32 x i8> %bools.i.sroa.44.44.vecblend, <32 x i8> %bools.i.sroa.44.48.vec.expand, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 48, i32 49, i32 50, i32 51, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !455
  %bools.i.sroa.44.52.vec.expand = shufflevector <4 x i8> %231, <4 x i8> poison, <32 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !455
  %bools.i.sroa.44.52.vecblend = shufflevector <32 x i8> %bools.i.sroa.44.48.vecblend, <32 x i8> %bools.i.sroa.44.52.vec.expand, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 52, i32 53, i32 54, i32 55, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !455
  %232 = or disjoint i64 %offset2.i, 56, !dbg !487
  %233 = getelementptr inbounds nuw i64, ptr %view.val.i.i.i, i64 %232, !dbg !481
  %234 = getelementptr inbounds nuw i8, ptr %233, i64 32, !dbg !486
  %wide.load79.7 = load <4 x i64>, ptr %233, align 8, !dbg !486, !noalias !518
  %wide.load80.7 = load <4 x i64>, ptr %234, align 8, !dbg !486, !noalias !518
  %235 = getelementptr inbounds nuw i64, ptr %view.val.i4.i.i, i64 %232, !dbg !423
  %236 = getelementptr inbounds nuw i8, ptr %235, i64 32, !dbg !438
  %wide.load81.7 = load <4 x i64>, ptr %235, align 8, !dbg !438, !noalias !521
  %wide.load82.7 = load <4 x i64>, ptr %236, align 8, !dbg !438, !noalias !521
  %237 = icmp eq <4 x i64> %wide.load79.7, splat (i64 9223372036854775807), !dbg !439
  %238 = icmp eq <4 x i64> %wide.load80.7, splat (i64 9223372036854775807), !dbg !439
  %239 = icmp eq <4 x i64> %wide.load81.7, splat (i64 9223372036854775807), !dbg !439
  %240 = icmp eq <4 x i64> %wide.load82.7, splat (i64 9223372036854775807), !dbg !439
  %241 = or <4 x i1> %237, %239, !dbg !439
  %242 = or <4 x i1> %238, %240, !dbg !439
  %243 = icmp slt <4 x i64> %wide.load79.7, %wide.load81.7, !dbg !454
  %244 = icmp slt <4 x i64> %wide.load80.7, %wide.load82.7, !dbg !454
  %245 = or <4 x i1> %228, %241, !dbg !367
  %246 = or <4 x i1> %229, %242, !dbg !367
  %247 = zext <4 x i1> %243 to <4 x i8>, !dbg !455
  %248 = zext <4 x i1> %244 to <4 x i8>, !dbg !455
  %bools.i.sroa.44.56.vec.expand = shufflevector <4 x i8> %247, <4 x i8> poison, <32 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !455
  %bools.i.sroa.44.56.vecblend = shufflevector <32 x i8> %bools.i.sroa.44.52.vecblend, <32 x i8> %bools.i.sroa.44.56.vec.expand, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 56, i32 57, i32 58, i32 59, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !455
  %bools.i.sroa.44.60.vec.expand = shufflevector <4 x i8> %248, <4 x i8> poison, <32 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3>, !dbg !455
  %bools.i.sroa.44.60.vecblend = shufflevector <32 x i8> %bools.i.sroa.44.56.vecblend, <32 x i8> %bools.i.sroa.44.60.vec.expand, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 60, i32 61, i32 62, i32 63>, !dbg !455
  %bin.rdx85 = or <4 x i1> %246, %245, !dbg !392
  br label %bb9.i.split, !dbg !523

vector.body:                                      ; preds = %bb4.i
  %_0.sroa.0.0.i9.i.i.us50 = load i64, ptr %_6.i11.i.i.cast, align 8, !noalias !399, !noundef !23
  %249 = icmp eq i64 %_0.sroa.0.0.i9.i.i.us50, 9223372036854775807
  %broadcast.splatinsert58 = insertelement <4 x i1> poison, i1 %249, i64 0
  %broadcast.splat59 = shufflevector <4 x i1> %broadcast.splatinsert58, <4 x i1> poison, <4 x i32> zeroinitializer
  %broadcast.splatinsert = insertelement <4 x i64> poison, i64 %_0.sroa.0.0.i9.i.i.us50, i64 0
  %broadcast.splat = shufflevector <4 x i64> %broadcast.splatinsert, <4 x i64> poison, <4 x i32> zeroinitializer
  %250 = insertelement <4 x i1> <i1 poison, i1 false, i1 false, i1 false>, i1 %.us-phi62, i64 0
  tail call void @llvm.experimental.noalias.scope.decl(metadata !410), !dbg !411
  tail call void @llvm.experimental.noalias.scope.decl(metadata !412), !dbg !413
  %251 = getelementptr inbounds nuw i64, ptr %view.val.i.i.i, i64 %offset2.i, !dbg !481
  %252 = getelementptr inbounds nuw i8, ptr %251, i64 32, !dbg !486
  %253 = getelementptr inbounds nuw i8, ptr %251, i64 64, !dbg !486
  %254 = getelementptr inbounds nuw i8, ptr %251, i64 96, !dbg !486
  %wide.load = load <4 x i64>, ptr %251, align 8, !dbg !486, !noalias !391
  %wide.load65 = load <4 x i64>, ptr %252, align 8, !dbg !486, !noalias !391
  %wide.load66 = load <4 x i64>, ptr %253, align 8, !dbg !486, !noalias !391
  %wide.load67 = load <4 x i64>, ptr %254, align 8, !dbg !486, !noalias !391
  tail call void @llvm.experimental.noalias.scope.decl(metadata !422), !dbg !413
  %255 = icmp eq <4 x i64> %wide.load, splat (i64 9223372036854775807), !dbg !439
  %256 = icmp eq <4 x i64> %wide.load65, splat (i64 9223372036854775807), !dbg !439
  %257 = icmp eq <4 x i64> %wide.load66, splat (i64 9223372036854775807), !dbg !439
  %258 = icmp eq <4 x i64> %wide.load67, splat (i64 9223372036854775807), !dbg !439
  %259 = icmp slt <4 x i64> %wide.load, %broadcast.splat, !dbg !454
  %260 = icmp slt <4 x i64> %wide.load65, %broadcast.splat, !dbg !454
  %261 = icmp slt <4 x i64> %wide.load66, %broadcast.splat, !dbg !454
  %262 = icmp slt <4 x i64> %wide.load67, %broadcast.splat, !dbg !454
  %263 = or <4 x i1> %255, %250, !dbg !367
  %264 = zext <4 x i1> %259 to <4 x i8>, !dbg !455
  %265 = zext <4 x i1> %260 to <4 x i8>, !dbg !455
  %266 = zext <4 x i1> %261 to <4 x i8>, !dbg !455
  %267 = zext <4 x i1> %262 to <4 x i8>, !dbg !455
  %bools.i.sroa.0.0.vec.expand302 = shufflevector <4 x i8> %264, <4 x i8> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !455
  %bools.i.sroa.0.4.vec.expand309 = shufflevector <4 x i8> %265, <4 x i8> poison, <32 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !455
  %bools.i.sroa.0.4.vecblend310 = shufflevector <32 x i8> %bools.i.sroa.0.0.vec.expand302, <32 x i8> %bools.i.sroa.0.4.vec.expand309, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 36, i32 37, i32 38, i32 39, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !455
  %bools.i.sroa.0.8.vec.expand315 = shufflevector <4 x i8> %266, <4 x i8> poison, <32 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !455
  %bools.i.sroa.0.8.vecblend316 = shufflevector <32 x i8> %bools.i.sroa.0.4.vecblend310, <32 x i8> %bools.i.sroa.0.8.vec.expand315, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 40, i32 41, i32 42, i32 43, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !455
  %bools.i.sroa.0.12.vec.expand321 = shufflevector <4 x i8> %267, <4 x i8> poison, <32 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !455
  %bools.i.sroa.0.12.vecblend322 = shufflevector <32 x i8> %bools.i.sroa.0.8.vecblend316, <32 x i8> %bools.i.sroa.0.12.vec.expand321, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 44, i32 45, i32 46, i32 47, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !455
  %268 = getelementptr inbounds nuw i64, ptr %4, i64 %offset2.i, !dbg !481
  %269 = getelementptr inbounds nuw i8, ptr %268, i64 32, !dbg !486
  %270 = getelementptr inbounds nuw i8, ptr %268, i64 64, !dbg !486
  %271 = getelementptr inbounds nuw i8, ptr %268, i64 96, !dbg !486
  %wide.load.1 = load <4 x i64>, ptr %268, align 8, !dbg !486, !noalias !524
  %wide.load65.1 = load <4 x i64>, ptr %269, align 8, !dbg !486, !noalias !524
  %wide.load66.1 = load <4 x i64>, ptr %270, align 8, !dbg !486, !noalias !524
  %wide.load67.1 = load <4 x i64>, ptr %271, align 8, !dbg !486, !noalias !524
  %272 = icmp eq <4 x i64> %wide.load.1, splat (i64 9223372036854775807), !dbg !439
  %273 = icmp eq <4 x i64> %wide.load65.1, splat (i64 9223372036854775807), !dbg !439
  %274 = icmp eq <4 x i64> %wide.load66.1, splat (i64 9223372036854775807), !dbg !439
  %275 = icmp eq <4 x i64> %wide.load67.1, splat (i64 9223372036854775807), !dbg !439
  %276 = icmp slt <4 x i64> %wide.load.1, %broadcast.splat, !dbg !454
  %277 = icmp slt <4 x i64> %wide.load65.1, %broadcast.splat, !dbg !454
  %278 = icmp slt <4 x i64> %wide.load66.1, %broadcast.splat, !dbg !454
  %279 = icmp slt <4 x i64> %wide.load67.1, %broadcast.splat, !dbg !454
  %280 = or <4 x i1> %263, %272, !dbg !367
  %281 = or <4 x i1> %256, %273, !dbg !367
  %282 = or <4 x i1> %257, %274, !dbg !367
  %283 = or <4 x i1> %258, %275, !dbg !367
  %284 = zext <4 x i1> %276 to <4 x i8>, !dbg !455
  %285 = zext <4 x i1> %277 to <4 x i8>, !dbg !455
  %286 = zext <4 x i1> %278 to <4 x i8>, !dbg !455
  %287 = zext <4 x i1> %279 to <4 x i8>, !dbg !455
  %bools.i.sroa.0.16.vec.expand327 = shufflevector <4 x i8> %284, <4 x i8> poison, <32 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !455
  %bools.i.sroa.0.16.vecblend328 = shufflevector <32 x i8> %bools.i.sroa.0.12.vecblend322, <32 x i8> %bools.i.sroa.0.16.vec.expand327, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 48, i32 49, i32 50, i32 51, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !455
  %bools.i.sroa.0.20.vec.expand333 = shufflevector <4 x i8> %285, <4 x i8> poison, <32 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !455
  %bools.i.sroa.0.20.vecblend334 = shufflevector <32 x i8> %bools.i.sroa.0.16.vecblend328, <32 x i8> %bools.i.sroa.0.20.vec.expand333, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 52, i32 53, i32 54, i32 55, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !455
  %bools.i.sroa.0.24.vec.expand339 = shufflevector <4 x i8> %286, <4 x i8> poison, <32 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !455
  %bools.i.sroa.0.24.vecblend340 = shufflevector <32 x i8> %bools.i.sroa.0.20.vecblend334, <32 x i8> %bools.i.sroa.0.24.vec.expand339, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 56, i32 57, i32 58, i32 59, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !455
  %bools.i.sroa.0.28.vec.expand345 = shufflevector <4 x i8> %287, <4 x i8> poison, <32 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3>, !dbg !455
  %bools.i.sroa.0.28.vecblend346 = shufflevector <32 x i8> %bools.i.sroa.0.24.vecblend340, <32 x i8> %bools.i.sroa.0.28.vec.expand345, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 60, i32 61, i32 62, i32 63>, !dbg !455
  %288 = getelementptr inbounds nuw i64, ptr %5, i64 %offset2.i, !dbg !481
  %289 = getelementptr inbounds nuw i8, ptr %288, i64 32, !dbg !486
  %290 = getelementptr inbounds nuw i8, ptr %288, i64 64, !dbg !486
  %291 = getelementptr inbounds nuw i8, ptr %288, i64 96, !dbg !486
  %wide.load.2 = load <4 x i64>, ptr %288, align 8, !dbg !486, !noalias !527
  %wide.load65.2 = load <4 x i64>, ptr %289, align 8, !dbg !486, !noalias !527
  %wide.load66.2 = load <4 x i64>, ptr %290, align 8, !dbg !486, !noalias !527
  %wide.load67.2 = load <4 x i64>, ptr %291, align 8, !dbg !486, !noalias !527
  %292 = icmp eq <4 x i64> %wide.load.2, splat (i64 9223372036854775807), !dbg !439
  %293 = icmp eq <4 x i64> %wide.load65.2, splat (i64 9223372036854775807), !dbg !439
  %294 = icmp eq <4 x i64> %wide.load66.2, splat (i64 9223372036854775807), !dbg !439
  %295 = icmp eq <4 x i64> %wide.load67.2, splat (i64 9223372036854775807), !dbg !439
  %296 = icmp slt <4 x i64> %wide.load.2, %broadcast.splat, !dbg !454
  %297 = icmp slt <4 x i64> %wide.load65.2, %broadcast.splat, !dbg !454
  %298 = icmp slt <4 x i64> %wide.load66.2, %broadcast.splat, !dbg !454
  %299 = icmp slt <4 x i64> %wide.load67.2, %broadcast.splat, !dbg !454
  %300 = or <4 x i1> %280, %292, !dbg !367
  %301 = or <4 x i1> %281, %293, !dbg !367
  %302 = or <4 x i1> %282, %294, !dbg !367
  %303 = or <4 x i1> %283, %295, !dbg !367
  %304 = zext <4 x i1> %296 to <4 x i8>, !dbg !455
  %305 = zext <4 x i1> %297 to <4 x i8>, !dbg !455
  %306 = zext <4 x i1> %298 to <4 x i8>, !dbg !455
  %307 = zext <4 x i1> %299 to <4 x i8>, !dbg !455
  %bools.i.sroa.44.32.vec.expand354 = shufflevector <4 x i8> %304, <4 x i8> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !455
  %bools.i.sroa.44.36.vec.expand360 = shufflevector <4 x i8> %305, <4 x i8> poison, <32 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !455
  %bools.i.sroa.44.36.vecblend361 = shufflevector <32 x i8> %bools.i.sroa.44.32.vec.expand354, <32 x i8> %bools.i.sroa.44.36.vec.expand360, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 36, i32 37, i32 38, i32 39, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !455
  %bools.i.sroa.44.40.vec.expand366 = shufflevector <4 x i8> %306, <4 x i8> poison, <32 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !455
  %bools.i.sroa.44.40.vecblend367 = shufflevector <32 x i8> %bools.i.sroa.44.36.vecblend361, <32 x i8> %bools.i.sroa.44.40.vec.expand366, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 40, i32 41, i32 42, i32 43, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !455
  %bools.i.sroa.44.44.vec.expand372 = shufflevector <4 x i8> %307, <4 x i8> poison, <32 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !455
  %bools.i.sroa.44.44.vecblend373 = shufflevector <32 x i8> %bools.i.sroa.44.40.vecblend367, <32 x i8> %bools.i.sroa.44.44.vec.expand372, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 44, i32 45, i32 46, i32 47, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !455
  %308 = getelementptr inbounds nuw i64, ptr %6, i64 %offset2.i, !dbg !481
  %309 = getelementptr inbounds nuw i8, ptr %308, i64 32, !dbg !486
  %310 = getelementptr inbounds nuw i8, ptr %308, i64 64, !dbg !486
  %311 = getelementptr inbounds nuw i8, ptr %308, i64 96, !dbg !486
  %wide.load.3 = load <4 x i64>, ptr %308, align 8, !dbg !486, !noalias !530
  %wide.load65.3 = load <4 x i64>, ptr %309, align 8, !dbg !486, !noalias !530
  %wide.load66.3 = load <4 x i64>, ptr %310, align 8, !dbg !486, !noalias !530
  %wide.load67.3 = load <4 x i64>, ptr %311, align 8, !dbg !486, !noalias !530
  %312 = icmp eq <4 x i64> %wide.load.3, splat (i64 9223372036854775807), !dbg !439
  %313 = icmp eq <4 x i64> %wide.load65.3, splat (i64 9223372036854775807), !dbg !439
  %314 = icmp eq <4 x i64> %wide.load66.3, splat (i64 9223372036854775807), !dbg !439
  %315 = icmp eq <4 x i64> %wide.load67.3, splat (i64 9223372036854775807), !dbg !439
  %316 = icmp slt <4 x i64> %wide.load.3, %broadcast.splat, !dbg !454
  %317 = icmp slt <4 x i64> %wide.load65.3, %broadcast.splat, !dbg !454
  %318 = icmp slt <4 x i64> %wide.load66.3, %broadcast.splat, !dbg !454
  %319 = icmp slt <4 x i64> %wide.load67.3, %broadcast.splat, !dbg !454
  %320 = or <4 x i1> %300, %312, !dbg !367
  %321 = or <4 x i1> %320, %broadcast.splat59, !dbg !367
  %322 = or <4 x i1> %301, %313, !dbg !367
  %323 = or <4 x i1> %302, %314, !dbg !367
  %324 = or <4 x i1> %303, %315, !dbg !367
  %325 = zext <4 x i1> %316 to <4 x i8>, !dbg !455
  %326 = zext <4 x i1> %317 to <4 x i8>, !dbg !455
  %327 = zext <4 x i1> %318 to <4 x i8>, !dbg !455
  %328 = zext <4 x i1> %319 to <4 x i8>, !dbg !455
  %bools.i.sroa.44.48.vec.expand378 = shufflevector <4 x i8> %325, <4 x i8> poison, <32 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !455
  %bools.i.sroa.44.48.vecblend379 = shufflevector <32 x i8> %bools.i.sroa.44.44.vecblend373, <32 x i8> %bools.i.sroa.44.48.vec.expand378, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 48, i32 49, i32 50, i32 51, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !455
  %bools.i.sroa.44.52.vec.expand384 = shufflevector <4 x i8> %326, <4 x i8> poison, <32 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !455
  %bools.i.sroa.44.52.vecblend385 = shufflevector <32 x i8> %bools.i.sroa.44.48.vecblend379, <32 x i8> %bools.i.sroa.44.52.vec.expand384, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 52, i32 53, i32 54, i32 55, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !455
  %bools.i.sroa.44.56.vec.expand390 = shufflevector <4 x i8> %327, <4 x i8> poison, <32 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !455
  %bools.i.sroa.44.56.vecblend391 = shufflevector <32 x i8> %bools.i.sroa.44.52.vecblend385, <32 x i8> %bools.i.sroa.44.56.vec.expand390, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 56, i32 57, i32 58, i32 59, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !455
  %bools.i.sroa.44.60.vec.expand396 = shufflevector <4 x i8> %328, <4 x i8> poison, <32 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3>, !dbg !455
  %bools.i.sroa.44.60.vecblend397 = shufflevector <32 x i8> %bools.i.sroa.44.56.vecblend391, <32 x i8> %bools.i.sroa.44.60.vec.expand396, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 60, i32 61, i32 62, i32 63>, !dbg !455
  %bin.rdx = or <4 x i1> %322, %321, !dbg !392
  %bin.rdx68 = or <4 x i1> %323, %bin.rdx, !dbg !392
  %bin.rdx69 = or <4 x i1> %324, %bin.rdx68, !dbg !392
  br label %bb9.i.split, !dbg !523

bb1.i.bb5.i_crit_edge.loopexit.loopexit:          ; preds = %bb4.i.us.us, %vec.epilog.middle.block, %middle.block116
  %329 = icmp eq i64 %_0.sroa.0.0.i9.i.i.us.us.us.us, 9223372036854775807
  %invariant.op = or i1 %329, %9, !dbg !350
  %330 = or i1 %invariant.op, %10
  br label %bb1.i.bb5.i_crit_edge

bb1.i.bb5.i_crit_edge:                            ; preds = %bb9.i.split, %bb4.i.us, %bb1.i.bb5.i_crit_edge.loopexit.loopexit
  %.us-phi77.in = phi i1 [ %111, %bb4.i.us ], [ %330, %bb1.i.bb5.i_crit_edge.loopexit.loopexit ], [ %.us-phi56.in, %bb9.i.split ]
  %.us-phi77 = zext i1 %.us-phi77.in to i8
  store i8 %.us-phi77, ptr %_11.i, align 8, !alias.scope !364, !noalias !357
  br label %bb5.i, !dbg !350

bb5.i:                                            ; preds = %bb1.i.bb5.i_crit_edge, %bb21.i
  %331 = icmp eq i64 %remainder.i, 0, !dbg !533
  br i1 %331, label %_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack23collect_bool_words_withNCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1g_7visitor5retry21ExecuteDenseWithRetryINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB35_EENtNtB3j_11row_visitor10RowVisitor19visit_deferred_boolB2Y_bKB35_NCINvXB43_B40_NtNtB1g_6row_fn5RowFn8dispatchB3e_E0NCB5V_s_0E0NCB3a_s_0B6H_E0NCINvB2_23collect_bool_words_avx2B17_E0EB43_.exit, label %bb12.i, !dbg !533

bb12.i:                                           ; preds = %bb5.i
  %332 = and i64 %len, -64, !dbg !534
  %_3.i.i.i.i.i = load i64, ptr %f, align 8, !range !351, !alias.scope !535, !noalias !540, !noundef !23
  %333 = trunc nuw i64 %_3.i.i.i.i.i to i1
  %_6.i.i.i.i = getelementptr inbounds nuw i8, ptr %f, i64 24
  %_3.i1.i.i.i.i = load i64, ptr %_6.i.i.i.i, align 8, !range !351, !alias.scope !546, !noalias !540, !noundef !23
  %334 = trunc nuw i64 %_3.i1.i.i.i.i to i1
  %_11.i.i.i = getelementptr inbounds nuw i8, ptr %f, i64 56
  %_11.i.i.promoted.i = load i8, ptr %_11.i.i.i, align 8, !alias.scope !549, !noalias !540
  %view.i3.i.i.i.i = getelementptr inbounds nuw i8, ptr %f, i64 32
  %335 = getelementptr inbounds nuw i8, ptr %f, i64 40
  %336 = getelementptr inbounds nuw i8, ptr %f, i64 16
  br i1 %333, label %start.split.us.i, label %start.split.i

start.split.us.i:                                 ; preds = %bb12.i
  %_6.i.i.i.i.us.i = load ptr, ptr %336, align 8, !alias.scope !535, !noalias !540, !nonnull !23, !align !363, !noundef !23
  br i1 %334, label %iter.check213, label %start.split.us.split.i

iter.check213:                                    ; preds = %start.split.us.i
  %_6.i11.i.i.i.us.us.i = load ptr, ptr %335, align 8, !alias.scope !546, !noalias !540, !nonnull !23, !align !363, !noundef !23
  %_0.sroa.0.0.i.i.i.i.us.us.i = load i64, ptr %_6.i.i.i.i.us.i, align 8, !noalias !552, !noundef !23
  %_0.sroa.0.0.i9.i.i.i.us.us.i = load i64, ptr %_6.i11.i.i.i.us.us.i, align 8, !noalias !553, !noundef !23
  %_5.i.i.i.i.us.us.i = icmp slt i64 %_0.sroa.0.0.i.i.i.i.us.us.i, %_0.sroa.0.0.i9.i.i.i.us.us.i
  %_13.us.us.i = zext i1 %_5.i.i.i.i.us.us.i to i64
  %min.iters.check211 = icmp samesign ult i64 %remainder.i, 4, !dbg !554
  br i1 %min.iters.check211, label %bb8.us.us.i.preheader, label %vector.main.loop.iter.check215, !dbg !554

vector.main.loop.iter.check215:                   ; preds = %iter.check213
  %min.iters.check214 = icmp samesign ult i64 %remainder.i, 16, !dbg !554
  br i1 %min.iters.check214, label %vec.epilog.ph241, label %vector.ph216, !dbg !554

vector.ph216:                                     ; preds = %vector.main.loop.iter.check215
  %n.mod.vf217 = and i64 %len, 12
  %n.vec218 = and i64 %len, 48
  %broadcast.splatinsert219 = insertelement <4 x i64> poison, i64 %_13.us.us.i, i64 0
  %broadcast.splat220 = shufflevector <4 x i64> %broadcast.splatinsert219, <4 x i64> poison, <4 x i32> zeroinitializer
  br label %vector.body221, !dbg !554

vector.body221:                                   ; preds = %vector.body221, %vector.ph216
  %index222 = phi i64 [ 0, %vector.ph216 ], [ %index.next229, %vector.body221 ], !dbg !569
  %vec.phi223 = phi <4 x i64> [ zeroinitializer, %vector.ph216 ], [ %341, %vector.body221 ]
  %vec.phi224 = phi <4 x i64> [ zeroinitializer, %vector.ph216 ], [ %342, %vector.body221 ]
  %vec.phi225 = phi <4 x i64> [ zeroinitializer, %vector.ph216 ], [ %343, %vector.body221 ]
  %vec.phi226 = phi <4 x i64> [ zeroinitializer, %vector.ph216 ], [ %344, %vector.body221 ]
  %vec.ind227 = phi <4 x i64> [ <i64 0, i64 1, i64 2, i64 3>, %vector.ph216 ], [ %vec.ind.next230, %vector.body221 ]
  %step.add228 = add <4 x i64> %vec.ind227, splat (i64 4)
  %step.add.2 = add <4 x i64> %vec.ind227, splat (i64 8)
  %step.add.3 = add <4 x i64> %vec.ind227, splat (i64 12)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !576), !dbg !577
  tail call void @llvm.experimental.noalias.scope.decl(metadata !583), !dbg !584
  tail call void @llvm.experimental.noalias.scope.decl(metadata !586), !dbg !584
  %337 = shl nuw <4 x i64> %broadcast.splat220, %vec.ind227, !dbg !587
  %338 = shl nuw <4 x i64> %broadcast.splat220, %step.add228, !dbg !587
  %339 = shl nuw <4 x i64> %broadcast.splat220, %step.add.2, !dbg !587
  %340 = shl nuw <4 x i64> %broadcast.splat220, %step.add.3, !dbg !587
  %341 = or <4 x i64> %337, %vec.phi223, !dbg !588
  %342 = or <4 x i64> %338, %vec.phi224, !dbg !588
  %343 = or <4 x i64> %339, %vec.phi225, !dbg !588
  %344 = or <4 x i64> %340, %vec.phi226, !dbg !588
  %index.next229 = add nuw i64 %index222, 16, !dbg !569
  %vec.ind.next230 = add <4 x i64> %vec.ind227, splat (i64 16)
  %345 = icmp eq i64 %index.next229, %n.vec218, !dbg !554
  br i1 %345, label %middle.block231, label %vector.body221, !dbg !554, !llvm.loop !589

middle.block231:                                  ; preds = %vector.body221
  %bin.rdx232 = or <4 x i64> %342, %341, !dbg !554
  %bin.rdx233 = or <4 x i64> %343, %bin.rdx232, !dbg !554
  %bin.rdx234 = or <4 x i64> %344, %bin.rdx233, !dbg !554
  %346 = tail call i64 @llvm.vector.reduce.or.v4i64(<4 x i64> %bin.rdx234), !dbg !554
  %cmp.n235 = icmp eq i64 %remainder.i, %n.vec218, !dbg !554
  br i1 %cmp.n235, label %_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack24collect_bool_word_scalarNCINvB2_23collect_bool_words_withNCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1O_7visitor5retry21ExecuteDenseWithRetryINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB3D_EENtNtB3R_11row_visitor10RowVisitor19visit_deferred_boolB3w_bKB3D_NCINvXB4B_B4y_NtNtB1O_6row_fn5RowFn8dispatchB3M_E0NCB6t_s_0E0NCB3I_s_0B7f_E0NCINvB2_23collect_bool_words_avx2B1F_E0E0EB4B_.exit.loopexit, label %vec.epilog.iter.check239, !dbg !554

vec.epilog.iter.check239:                         ; preds = %middle.block231
  %min.epilog.iters.check240 = icmp eq i64 %n.mod.vf217, 0
  br i1 %min.epilog.iters.check240, label %bb8.us.us.i.preheader, label %vec.epilog.ph241, !prof !404

vec.epilog.ph241:                                 ; preds = %vector.main.loop.iter.check215, %vec.epilog.iter.check239
  %bc.resume.val236 = phi i64 [ %n.vec218, %vec.epilog.iter.check239 ], [ 0, %vector.main.loop.iter.check215 ]
  %bc.merge.rdx237 = phi i64 [ %346, %vec.epilog.iter.check239 ], [ 0, %vector.main.loop.iter.check215 ]
  %n.vec243 = and i64 %len, 60
  %347 = insertelement <4 x i64> <i64 poison, i64 0, i64 0, i64 0>, i64 %bc.merge.rdx237, i64 0
  %broadcast.splatinsert244 = insertelement <4 x i64> poison, i64 %_13.us.us.i, i64 0
  %broadcast.splat245 = shufflevector <4 x i64> %broadcast.splatinsert244, <4 x i64> poison, <4 x i32> zeroinitializer
  %broadcast.splatinsert246 = insertelement <4 x i64> poison, i64 %bc.resume.val236, i64 0
  %broadcast.splat247 = shufflevector <4 x i64> %broadcast.splatinsert246, <4 x i64> poison, <4 x i32> zeroinitializer
  %induction = or disjoint <4 x i64> %broadcast.splat247, <i64 0, i64 1, i64 2, i64 3>
  br label %vec.epilog.vector.body248

vec.epilog.vector.body248:                        ; preds = %vec.epilog.vector.body248, %vec.epilog.ph241
  %index249 = phi i64 [ %bc.resume.val236, %vec.epilog.ph241 ], [ %index.next252, %vec.epilog.vector.body248 ], !dbg !569
  %vec.phi250 = phi <4 x i64> [ %347, %vec.epilog.ph241 ], [ %349, %vec.epilog.vector.body248 ]
  %vec.ind251 = phi <4 x i64> [ %induction, %vec.epilog.ph241 ], [ %vec.ind.next253, %vec.epilog.vector.body248 ]
  tail call void @llvm.experimental.noalias.scope.decl(metadata !576), !dbg !577
  tail call void @llvm.experimental.noalias.scope.decl(metadata !583), !dbg !584
  tail call void @llvm.experimental.noalias.scope.decl(metadata !586), !dbg !584
  %348 = shl nuw <4 x i64> %broadcast.splat245, %vec.ind251, !dbg !587
  %349 = or <4 x i64> %348, %vec.phi250, !dbg !588
  %index.next252 = add nuw i64 %index249, 4, !dbg !569
  %vec.ind.next253 = add nuw nsw <4 x i64> %vec.ind251, splat (i64 4)
  %350 = icmp eq i64 %index.next252, %n.vec243, !dbg !554
  br i1 %350, label %vec.epilog.middle.block254, label %vec.epilog.vector.body248, !dbg !554, !llvm.loop !590

vec.epilog.middle.block254:                       ; preds = %vec.epilog.vector.body248
  %351 = tail call i64 @llvm.vector.reduce.or.v4i64(<4 x i64> %349), !dbg !554
  %cmp.n255 = icmp eq i64 %remainder.i, %n.vec243, !dbg !554
  br i1 %cmp.n255, label %_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack24collect_bool_word_scalarNCINvB2_23collect_bool_words_withNCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1O_7visitor5retry21ExecuteDenseWithRetryINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB3D_EENtNtB3R_11row_visitor10RowVisitor19visit_deferred_boolB3w_bKB3D_NCINvXB4B_B4y_NtNtB1O_6row_fn5RowFn8dispatchB3M_E0NCB6t_s_0E0NCB3I_s_0B7f_E0NCINvB2_23collect_bool_words_avx2B1F_E0E0EB4B_.exit.loopexit, label %bb8.us.us.i.preheader, !dbg !554

bb8.us.us.i.preheader:                            ; preds = %iter.check213, %vec.epilog.iter.check239, %vec.epilog.middle.block254
  %packed.sroa.0.05.us.us.i.ph = phi i64 [ 0, %iter.check213 ], [ %346, %vec.epilog.iter.check239 ], [ %351, %vec.epilog.middle.block254 ]
  %iter.sroa.0.04.us.us.i.ph = phi i64 [ 0, %iter.check213 ], [ %n.vec218, %vec.epilog.iter.check239 ], [ %n.vec243, %vec.epilog.middle.block254 ]
  br label %bb8.us.us.i, !dbg !554

bb8.us.us.i:                                      ; preds = %bb8.us.us.i.preheader, %bb8.us.us.i
  %packed.sroa.0.05.us.us.i = phi i64 [ %353, %bb8.us.us.i ], [ %packed.sroa.0.05.us.us.i.ph, %bb8.us.us.i.preheader ]
  %iter.sroa.0.04.us.us.i = phi i64 [ %352, %bb8.us.us.i ], [ %iter.sroa.0.04.us.us.i.ph, %bb8.us.us.i.preheader ]
  tail call void @llvm.experimental.noalias.scope.decl(metadata !576), !dbg !577
  tail call void @llvm.experimental.noalias.scope.decl(metadata !583), !dbg !584
  tail call void @llvm.experimental.noalias.scope.decl(metadata !586), !dbg !584
  %352 = add nuw nsw i64 %iter.sroa.0.04.us.us.i, 1, !dbg !569
  %_12.us.us.i = shl nuw i64 %_13.us.us.i, %iter.sroa.0.04.us.us.i, !dbg !587
  %353 = or i64 %_12.us.us.i, %packed.sroa.0.05.us.us.i, !dbg !588
  %exitcond32.not.i = icmp eq i64 %352, %remainder.i, !dbg !591
  br i1 %exitcond32.not.i, label %_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack24collect_bool_word_scalarNCINvB2_23collect_bool_words_withNCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1O_7visitor5retry21ExecuteDenseWithRetryINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB3D_EENtNtB3R_11row_visitor10RowVisitor19visit_deferred_boolB3w_bKB3D_NCINvXB4B_B4y_NtNtB1O_6row_fn5RowFn8dispatchB3M_E0NCB6t_s_0E0NCB3I_s_0B7f_E0NCINvB2_23collect_bool_words_avx2B1F_E0E0EB4B_.exit.loopexit, label %bb8.us.us.i, !dbg !554, !llvm.loop !598

start.split.us.split.i:                           ; preds = %start.split.us.i
  %view.val.i4.i.i.i.us.i = load ptr, ptr %view.i3.i.i.i.i, align 8, !alias.scope !546, !noalias !540, !nonnull !23, !align !363, !noundef !23
  %view.val1.i5.i.i.i.us.i = load i64, ptr %335, align 8, !alias.scope !546, !noalias !540, !noundef !23
  %354 = trunc nuw i8 %_11.i.i.promoted.i to i1, !dbg !599
  %_0.sroa.0.0.i.i.i.i.us.i = load i64, ptr %_6.i.i.i.i.us.i, align 8, !noalias !552, !noundef !23
  %355 = icmp eq i64 %_0.sroa.0.0.i.i.i.i.us.i, 9223372036854775807
  %min.iters.check181 = icmp samesign ult i64 %remainder.i, 8, !dbg !554
  br i1 %min.iters.check181, label %bb8.us.i.preheader, label %vector.ph182, !dbg !554

vector.ph182:                                     ; preds = %start.split.us.split.i
  %n.vec184 = and i64 %len, 56
  %356 = insertelement <4 x i1> <i1 poison, i1 false, i1 false, i1 false>, i1 %354, i64 0
  %broadcast.splatinsert187 = insertelement <4 x i64> poison, i64 %_0.sroa.0.0.i.i.i.i.us.i, i64 0
  %broadcast.splat188 = shufflevector <4 x i64> %broadcast.splatinsert187, <4 x i64> poison, <4 x i32> zeroinitializer
  %broadcast.splatinsert189 = insertelement <4 x i1> poison, i1 %355, i64 0
  %broadcast.splat190 = shufflevector <4 x i1> %broadcast.splatinsert189, <4 x i1> poison, <4 x i32> zeroinitializer
  %invariant.gep426 = getelementptr i64, ptr %view.val.i4.i.i.i.us.i, i64 %332, !dbg !554
  br label %vector.body191, !dbg !554

vector.body191:                                   ; preds = %vector.body191, %vector.ph182
  %index192 = phi i64 [ 0, %vector.ph182 ], [ %index.next201, %vector.body191 ], !dbg !569
  %vec.phi193 = phi <4 x i64> [ zeroinitializer, %vector.ph182 ], [ %370, %vector.body191 ]
  %vec.phi194 = phi <4 x i64> [ zeroinitializer, %vector.ph182 ], [ %371, %vector.body191 ]
  %vec.ind195 = phi <4 x i64> [ <i64 0, i64 1, i64 2, i64 3>, %vector.ph182 ], [ %vec.ind.next202, %vector.body191 ]
  %vec.phi196 = phi <4 x i1> [ %356, %vector.ph182 ], [ %363, %vector.body191 ]
  %vec.phi197 = phi <4 x i1> [ zeroinitializer, %vector.ph182 ], [ %365, %vector.body191 ]
  %step.add198 = add <4 x i64> %vec.ind195, splat (i64 4)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !576), !dbg !577
  tail call void @llvm.experimental.noalias.scope.decl(metadata !583), !dbg !584
  tail call void @llvm.experimental.noalias.scope.decl(metadata !586), !dbg !584
  %gep427 = getelementptr i64, ptr %invariant.gep426, i64 %index192, !dbg !601
  %357 = getelementptr inbounds nuw i8, ptr %gep427, i64 32, !dbg !606
  %wide.load199 = load <4 x i64>, ptr %gep427, align 8, !dbg !606, !noalias !553
  %wide.load200 = load <4 x i64>, ptr %357, align 8, !dbg !606, !noalias !553
  %358 = icmp eq <4 x i64> %wide.load199, splat (i64 9223372036854775807), !dbg !607
  %359 = icmp eq <4 x i64> %wide.load200, splat (i64 9223372036854775807), !dbg !607
  %360 = icmp slt <4 x i64> %broadcast.splat188, %wide.load199, !dbg !610
  %361 = icmp slt <4 x i64> %broadcast.splat188, %wide.load200, !dbg !610
  %362 = or <4 x i1> %358, %vec.phi196, !dbg !599
  %363 = or <4 x i1> %362, %broadcast.splat190, !dbg !599
  %364 = or <4 x i1> %359, %vec.phi197, !dbg !599
  %365 = or <4 x i1> %364, %broadcast.splat190, !dbg !599
  %366 = zext <4 x i1> %360 to <4 x i64>, !dbg !587
  %367 = zext <4 x i1> %361 to <4 x i64>, !dbg !587
  %368 = shl nuw <4 x i64> %366, %vec.ind195, !dbg !587
  %369 = shl nuw <4 x i64> %367, %step.add198, !dbg !587
  %370 = or <4 x i64> %368, %vec.phi193, !dbg !588
  %371 = or <4 x i64> %369, %vec.phi194, !dbg !588
  %index.next201 = add nuw i64 %index192, 8, !dbg !569
  %vec.ind.next202 = add <4 x i64> %vec.ind195, splat (i64 8)
  %372 = icmp eq i64 %index.next201, %n.vec184, !dbg !554
  br i1 %372, label %middle.block203, label %vector.body191, !dbg !554, !llvm.loop !611

middle.block203:                                  ; preds = %vector.body191
  %bin.rdx204 = or <4 x i64> %371, %370, !dbg !554
  %373 = tail call i64 @llvm.vector.reduce.or.v4i64(<4 x i64> %bin.rdx204), !dbg !554
  %bin.rdx205 = or <4 x i1> %364, %363, !dbg !554
  %374 = bitcast <4 x i1> %bin.rdx205 to i4, !dbg !554
  %375 = icmp ne i4 %374, 0, !dbg !554
  %cmp.n206 = icmp eq i64 %remainder.i, %n.vec184, !dbg !554
  br i1 %cmp.n206, label %_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack24collect_bool_word_scalarNCINvB2_23collect_bool_words_withNCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1O_7visitor5retry21ExecuteDenseWithRetryINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB3D_EENtNtB3R_11row_visitor10RowVisitor19visit_deferred_boolB3w_bKB3D_NCINvXB4B_B4y_NtNtB1O_6row_fn5RowFn8dispatchB3M_E0NCB6t_s_0E0NCB3I_s_0B7f_E0NCINvB2_23collect_bool_words_avx2B1F_E0E0EB4B_.exit, label %bb8.us.i.preheader, !dbg !554

bb8.us.i.preheader:                               ; preds = %start.split.us.split.i, %middle.block203
  %packed.sroa.0.05.us.i.ph = phi i64 [ 0, %start.split.us.split.i ], [ %373, %middle.block203 ]
  %iter.sroa.0.04.us.i.ph = phi i64 [ 0, %start.split.us.split.i ], [ %n.vec184, %middle.block203 ]
  %.ph = phi i1 [ %354, %start.split.us.split.i ], [ %375, %middle.block203 ]
  br label %bb8.us.i, !dbg !554

bb8.us.i:                                         ; preds = %bb8.us.i.preheader, %bb8.us.i
  %packed.sroa.0.05.us.i = phi i64 [ %381, %bb8.us.i ], [ %packed.sroa.0.05.us.i.ph, %bb8.us.i.preheader ]
  %iter.sroa.0.04.us.i = phi i64 [ %380, %bb8.us.i ], [ %iter.sroa.0.04.us.i.ph, %bb8.us.i.preheader ]
  %376 = phi i1 [ %379, %bb8.us.i ], [ %.ph, %bb8.us.i.preheader ]
  %_4.i.us.i = add nuw nsw i64 %iter.sroa.0.04.us.i, %332, !dbg !612
  tail call void @llvm.experimental.noalias.scope.decl(metadata !576), !dbg !577
  tail call void @llvm.experimental.noalias.scope.decl(metadata !583), !dbg !584
  tail call void @llvm.experimental.noalias.scope.decl(metadata !586), !dbg !584
  %_5.i.i6.i.i.i.us.i = icmp ult i64 %_4.i.us.i, %view.val1.i5.i.i.i.us.i, !dbg !613
  tail call void @llvm.assume(i1 %_5.i.i6.i.i.i.us.i), !dbg !614
  %_4.i.i7.i.i.i.us.i = getelementptr inbounds nuw i64, ptr %view.val.i4.i.i.i.us.i, i64 %_4.i.us.i, !dbg !601
  %_0.sroa.0.0.i9.i.i.i.us.i = load i64, ptr %_4.i.i7.i.i.i.us.i, align 8, !dbg !606, !noalias !553, !noundef !23
  %377 = icmp eq i64 %_0.sroa.0.0.i9.i.i.i.us.i, 9223372036854775807, !dbg !607
  %_5.i.i.i.i.us.i = icmp slt i64 %_0.sroa.0.0.i.i.i.i.us.i, %_0.sroa.0.0.i9.i.i.i.us.i, !dbg !610
  %378 = or i1 %377, %376, !dbg !599
  %379 = or i1 %378, %355, !dbg !599
  %380 = add nuw nsw i64 %iter.sroa.0.04.us.i, 1, !dbg !569
  %_13.us.i = zext i1 %_5.i.i.i.i.us.i to i64, !dbg !587
  %_12.us.i = shl nuw i64 %_13.us.i, %iter.sroa.0.04.us.i, !dbg !587
  %381 = or i64 %_12.us.i, %packed.sroa.0.05.us.i, !dbg !588
  %exitcond31.not.i = icmp eq i64 %380, %remainder.i, !dbg !591
  br i1 %exitcond31.not.i, label %_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack24collect_bool_word_scalarNCINvB2_23collect_bool_words_withNCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1O_7visitor5retry21ExecuteDenseWithRetryINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB3D_EENtNtB3R_11row_visitor10RowVisitor19visit_deferred_boolB3w_bKB3D_NCINvXB4B_B4y_NtNtB1O_6row_fn5RowFn8dispatchB3M_E0NCB6t_s_0E0NCB3I_s_0B7f_E0NCINvB2_23collect_bool_words_avx2B1F_E0E0EB4B_.exit, label %bb8.us.i, !dbg !554, !llvm.loop !615

start.split.i:                                    ; preds = %bb12.i
  %view.i.i.i.i.i = getelementptr inbounds nuw i8, ptr %f, i64 8
  %view.val.i.i.i.i.i = load ptr, ptr %view.i.i.i.i.i, align 8, !alias.scope !535, !noalias !540, !nonnull !23, !align !363, !noundef !23
  %view.val1.i.i.i.i.i = load i64, ptr %336, align 8, !alias.scope !535, !noalias !540, !noundef !23
  br i1 %334, label %start.split.split.us.i, label %start.split.split.i

start.split.split.us.i:                           ; preds = %start.split.i
  %_6.i11.i.i.i.us12.i = load ptr, ptr %335, align 8, !alias.scope !546, !noalias !540, !nonnull !23, !align !363, !noundef !23
  %382 = trunc nuw i8 %_11.i.i.promoted.i to i1, !dbg !599
  %_0.sroa.0.0.i9.i.i.i.us15.i = load i64, ptr %_6.i11.i.i.i.us12.i, align 8, !noalias !553, !noundef !23
  %383 = icmp eq i64 %_0.sroa.0.0.i9.i.i.i.us15.i, 9223372036854775807
  %min.iters.check151 = icmp samesign ult i64 %remainder.i, 8, !dbg !554
  br i1 %min.iters.check151, label %bb8.us6.i.preheader, label %vector.ph152, !dbg !554

vector.ph152:                                     ; preds = %start.split.split.us.i
  %n.vec154 = and i64 %len, 56
  %384 = insertelement <4 x i1> <i1 poison, i1 false, i1 false, i1 false>, i1 %382, i64 0
  %broadcast.splatinsert155 = insertelement <4 x i64> poison, i64 %_0.sroa.0.0.i9.i.i.i.us15.i, i64 0
  %broadcast.splat156 = shufflevector <4 x i64> %broadcast.splatinsert155, <4 x i64> poison, <4 x i32> zeroinitializer
  %broadcast.splatinsert157 = insertelement <4 x i1> poison, i1 %383, i64 0
  %broadcast.splat158 = shufflevector <4 x i1> %broadcast.splatinsert157, <4 x i1> poison, <4 x i32> zeroinitializer
  %invariant.gep = getelementptr i64, ptr %view.val.i.i.i.i.i, i64 %332, !dbg !554
  br label %vector.body161, !dbg !554

vector.body161:                                   ; preds = %vector.body161, %vector.ph152
  %index162 = phi i64 [ 0, %vector.ph152 ], [ %index.next171, %vector.body161 ], !dbg !569
  %vec.phi163 = phi <4 x i64> [ zeroinitializer, %vector.ph152 ], [ %398, %vector.body161 ]
  %vec.phi164 = phi <4 x i64> [ zeroinitializer, %vector.ph152 ], [ %399, %vector.body161 ]
  %vec.ind165 = phi <4 x i64> [ <i64 0, i64 1, i64 2, i64 3>, %vector.ph152 ], [ %vec.ind.next172, %vector.body161 ]
  %vec.phi166 = phi <4 x i1> [ %384, %vector.ph152 ], [ %391, %vector.body161 ]
  %vec.phi167 = phi <4 x i1> [ zeroinitializer, %vector.ph152 ], [ %393, %vector.body161 ]
  %step.add168 = add <4 x i64> %vec.ind165, splat (i64 4)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !576), !dbg !577
  tail call void @llvm.experimental.noalias.scope.decl(metadata !583), !dbg !584
  %gep = getelementptr i64, ptr %invariant.gep, i64 %index162, !dbg !616
  %385 = getelementptr inbounds nuw i8, ptr %gep, i64 32, !dbg !621
  %wide.load169 = load <4 x i64>, ptr %gep, align 8, !dbg !621, !noalias !552
  %wide.load170 = load <4 x i64>, ptr %385, align 8, !dbg !621, !noalias !552
  tail call void @llvm.experimental.noalias.scope.decl(metadata !586), !dbg !584
  %386 = icmp eq <4 x i64> %wide.load169, splat (i64 9223372036854775807), !dbg !607
  %387 = icmp eq <4 x i64> %wide.load170, splat (i64 9223372036854775807), !dbg !607
  %388 = icmp slt <4 x i64> %wide.load169, %broadcast.splat156, !dbg !610
  %389 = icmp slt <4 x i64> %wide.load170, %broadcast.splat156, !dbg !610
  %390 = or <4 x i1> %386, %vec.phi166, !dbg !599
  %391 = or <4 x i1> %390, %broadcast.splat158, !dbg !599
  %392 = or <4 x i1> %387, %vec.phi167, !dbg !599
  %393 = or <4 x i1> %392, %broadcast.splat158, !dbg !599
  %394 = zext <4 x i1> %388 to <4 x i64>, !dbg !587
  %395 = zext <4 x i1> %389 to <4 x i64>, !dbg !587
  %396 = shl nuw <4 x i64> %394, %vec.ind165, !dbg !587
  %397 = shl nuw <4 x i64> %395, %step.add168, !dbg !587
  %398 = or <4 x i64> %396, %vec.phi163, !dbg !588
  %399 = or <4 x i64> %397, %vec.phi164, !dbg !588
  %index.next171 = add nuw i64 %index162, 8, !dbg !569
  %vec.ind.next172 = add <4 x i64> %vec.ind165, splat (i64 8)
  %400 = icmp eq i64 %index.next171, %n.vec154, !dbg !554
  br i1 %400, label %middle.block173, label %vector.body161, !dbg !554, !llvm.loop !622

middle.block173:                                  ; preds = %vector.body161
  %bin.rdx174 = or <4 x i64> %399, %398, !dbg !554
  %401 = tail call i64 @llvm.vector.reduce.or.v4i64(<4 x i64> %bin.rdx174), !dbg !554
  %bin.rdx175 = or <4 x i1> %392, %391, !dbg !554
  %402 = bitcast <4 x i1> %bin.rdx175 to i4, !dbg !554
  %403 = icmp ne i4 %402, 0, !dbg !554
  %cmp.n176 = icmp eq i64 %remainder.i, %n.vec154, !dbg !554
  br i1 %cmp.n176, label %_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack24collect_bool_word_scalarNCINvB2_23collect_bool_words_withNCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1O_7visitor5retry21ExecuteDenseWithRetryINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB3D_EENtNtB3R_11row_visitor10RowVisitor19visit_deferred_boolB3w_bKB3D_NCINvXB4B_B4y_NtNtB1O_6row_fn5RowFn8dispatchB3M_E0NCB6t_s_0E0NCB3I_s_0B7f_E0NCINvB2_23collect_bool_words_avx2B1F_E0E0EB4B_.exit, label %bb8.us6.i.preheader, !dbg !554

bb8.us6.i.preheader:                              ; preds = %start.split.split.us.i, %middle.block173
  %packed.sroa.0.05.us7.i.ph = phi i64 [ 0, %start.split.split.us.i ], [ %401, %middle.block173 ]
  %iter.sroa.0.04.us8.i.ph = phi i64 [ 0, %start.split.split.us.i ], [ %n.vec154, %middle.block173 ]
  %.ph271 = phi i1 [ %382, %start.split.split.us.i ], [ %403, %middle.block173 ]
  br label %bb8.us6.i, !dbg !554

bb8.us6.i:                                        ; preds = %bb8.us6.i.preheader, %bb8.us6.i
  %packed.sroa.0.05.us7.i = phi i64 [ %409, %bb8.us6.i ], [ %packed.sroa.0.05.us7.i.ph, %bb8.us6.i.preheader ]
  %iter.sroa.0.04.us8.i = phi i64 [ %408, %bb8.us6.i ], [ %iter.sroa.0.04.us8.i.ph, %bb8.us6.i.preheader ]
  %404 = phi i1 [ %407, %bb8.us6.i ], [ %.ph271, %bb8.us6.i.preheader ]
  %_4.i.us9.i = add nuw nsw i64 %iter.sroa.0.04.us8.i, %332, !dbg !612
  tail call void @llvm.experimental.noalias.scope.decl(metadata !576), !dbg !577
  tail call void @llvm.experimental.noalias.scope.decl(metadata !583), !dbg !584
  %_5.i.i.i.i.i.us.i = icmp ult i64 %_4.i.us9.i, %view.val1.i.i.i.i.i, !dbg !623
  tail call void @llvm.assume(i1 %_5.i.i.i.i.i.us.i), !dbg !624
  %_4.i.i.i.i.i.us.i = getelementptr inbounds nuw i64, ptr %view.val.i.i.i.i.i, i64 %_4.i.us9.i, !dbg !616
  %_0.sroa.0.0.i.i.i.i.us10.i = load i64, ptr %_4.i.i.i.i.i.us.i, align 8, !dbg !621, !noalias !552, !noundef !23
  tail call void @llvm.experimental.noalias.scope.decl(metadata !586), !dbg !584
  %405 = icmp eq i64 %_0.sroa.0.0.i.i.i.i.us10.i, 9223372036854775807, !dbg !607
  %_5.i.i.i.i.us17.i = icmp slt i64 %_0.sroa.0.0.i.i.i.i.us10.i, %_0.sroa.0.0.i9.i.i.i.us15.i, !dbg !610
  %406 = or i1 %405, %404, !dbg !599
  %407 = or i1 %406, %383, !dbg !599
  %408 = add nuw nsw i64 %iter.sroa.0.04.us8.i, 1, !dbg !569
  %_13.us18.i = zext i1 %_5.i.i.i.i.us17.i to i64, !dbg !587
  %_12.us19.i = shl nuw i64 %_13.us18.i, %iter.sroa.0.04.us8.i, !dbg !587
  %409 = or i64 %_12.us19.i, %packed.sroa.0.05.us7.i, !dbg !588
  %exitcond30.not.i = icmp eq i64 %408, %remainder.i, !dbg !591
  br i1 %exitcond30.not.i, label %_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack24collect_bool_word_scalarNCINvB2_23collect_bool_words_withNCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1O_7visitor5retry21ExecuteDenseWithRetryINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB3D_EENtNtB3R_11row_visitor10RowVisitor19visit_deferred_boolB3w_bKB3D_NCINvXB4B_B4y_NtNtB1O_6row_fn5RowFn8dispatchB3M_E0NCB6t_s_0E0NCB3I_s_0B7f_E0NCINvB2_23collect_bool_words_avx2B1F_E0E0EB4B_.exit, label %bb8.us6.i, !dbg !554, !llvm.loop !625

start.split.split.i:                              ; preds = %start.split.i
  %view.val.i4.i.i.i.i = load ptr, ptr %view.i3.i.i.i.i, align 8, !alias.scope !546, !noalias !540, !nonnull !23, !align !363, !noundef !23
  %view.val1.i5.i.i.i.i = load i64, ptr %335, align 8, !alias.scope !546, !noalias !540, !noundef !23
  %410 = trunc nuw i8 %_11.i.i.promoted.i to i1, !dbg !599
  %min.iters.check125 = icmp samesign ult i64 %remainder.i, 8, !dbg !554
  br i1 %min.iters.check125, label %bb8.i4.preheader, label %vector.ph126, !dbg !554

vector.ph126:                                     ; preds = %start.split.split.i
  %n.vec128 = and i64 %len, 56
  %411 = insertelement <4 x i1> <i1 poison, i1 false, i1 false, i1 false>, i1 %410, i64 0
  br label %vector.body133, !dbg !554

vector.body133:                                   ; preds = %vector.body133, %vector.ph126
  %index134 = phi i64 [ 0, %vector.ph126 ], [ %index.next143, %vector.body133 ], !dbg !569
  %vec.phi135 = phi <4 x i64> [ zeroinitializer, %vector.ph126 ], [ %431, %vector.body133 ]
  %vec.phi136 = phi <4 x i64> [ zeroinitializer, %vector.ph126 ], [ %432, %vector.body133 ]
  %vec.ind = phi <4 x i64> [ <i64 0, i64 1, i64 2, i64 3>, %vector.ph126 ], [ %vec.ind.next, %vector.body133 ]
  %vec.phi137 = phi <4 x i1> [ %411, %vector.ph126 ], [ %425, %vector.body133 ]
  %vec.phi138 = phi <4 x i1> [ zeroinitializer, %vector.ph126 ], [ %426, %vector.body133 ]
  %step.add = add <4 x i64> %vec.ind, splat (i64 4)
  %412 = add nuw nsw i64 %index134, %332, !dbg !612
  tail call void @llvm.experimental.noalias.scope.decl(metadata !576), !dbg !577
  tail call void @llvm.experimental.noalias.scope.decl(metadata !583), !dbg !584
  %413 = getelementptr inbounds nuw i64, ptr %view.val.i.i.i.i.i, i64 %412, !dbg !616
  %414 = getelementptr inbounds nuw i8, ptr %413, i64 32, !dbg !621
  %wide.load139 = load <4 x i64>, ptr %413, align 8, !dbg !621, !noalias !552
  %wide.load140 = load <4 x i64>, ptr %414, align 8, !dbg !621, !noalias !552
  tail call void @llvm.experimental.noalias.scope.decl(metadata !586), !dbg !584
  %415 = getelementptr inbounds nuw i64, ptr %view.val.i4.i.i.i.i, i64 %412, !dbg !601
  %416 = getelementptr inbounds nuw i8, ptr %415, i64 32, !dbg !606
  %wide.load141 = load <4 x i64>, ptr %415, align 8, !dbg !606, !noalias !553
  %wide.load142 = load <4 x i64>, ptr %416, align 8, !dbg !606, !noalias !553
  %417 = icmp eq <4 x i64> %wide.load139, splat (i64 9223372036854775807), !dbg !607
  %418 = icmp eq <4 x i64> %wide.load140, splat (i64 9223372036854775807), !dbg !607
  %419 = icmp eq <4 x i64> %wide.load141, splat (i64 9223372036854775807), !dbg !607
  %420 = icmp eq <4 x i64> %wide.load142, splat (i64 9223372036854775807), !dbg !607
  %421 = or <4 x i1> %417, %419, !dbg !607
  %422 = or <4 x i1> %418, %420, !dbg !607
  %423 = icmp slt <4 x i64> %wide.load139, %wide.load141, !dbg !610
  %424 = icmp slt <4 x i64> %wide.load140, %wide.load142, !dbg !610
  %425 = or <4 x i1> %vec.phi137, %421, !dbg !599
  %426 = or <4 x i1> %vec.phi138, %422, !dbg !599
  %427 = zext <4 x i1> %423 to <4 x i64>, !dbg !587
  %428 = zext <4 x i1> %424 to <4 x i64>, !dbg !587
  %429 = shl nuw <4 x i64> %427, %vec.ind, !dbg !587
  %430 = shl nuw <4 x i64> %428, %step.add, !dbg !587
  %431 = or <4 x i64> %429, %vec.phi135, !dbg !588
  %432 = or <4 x i64> %430, %vec.phi136, !dbg !588
  %index.next143 = add nuw i64 %index134, 8, !dbg !569
  %vec.ind.next = add <4 x i64> %vec.ind, splat (i64 8)
  %433 = icmp eq i64 %index.next143, %n.vec128, !dbg !554
  br i1 %433, label %middle.block144, label %vector.body133, !dbg !554, !llvm.loop !626

middle.block144:                                  ; preds = %vector.body133
  %bin.rdx145 = or <4 x i64> %432, %431, !dbg !554
  %434 = tail call i64 @llvm.vector.reduce.or.v4i64(<4 x i64> %bin.rdx145), !dbg !554
  %bin.rdx146 = or <4 x i1> %426, %425, !dbg !554
  %435 = bitcast <4 x i1> %bin.rdx146 to i4, !dbg !554
  %436 = icmp ne i4 %435, 0, !dbg !554
  %cmp.n147 = icmp eq i64 %remainder.i, %n.vec128, !dbg !554
  br i1 %cmp.n147, label %_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack24collect_bool_word_scalarNCINvB2_23collect_bool_words_withNCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1O_7visitor5retry21ExecuteDenseWithRetryINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB3D_EENtNtB3R_11row_visitor10RowVisitor19visit_deferred_boolB3w_bKB3D_NCINvXB4B_B4y_NtNtB1O_6row_fn5RowFn8dispatchB3M_E0NCB6t_s_0E0NCB3I_s_0B7f_E0NCINvB2_23collect_bool_words_avx2B1F_E0E0EB4B_.exit, label %bb8.i4.preheader, !dbg !554

bb8.i4.preheader:                                 ; preds = %start.split.split.i, %middle.block144
  %packed.sroa.0.05.i.ph = phi i64 [ 0, %start.split.split.i ], [ %434, %middle.block144 ]
  %iter.sroa.0.04.i.ph = phi i64 [ 0, %start.split.split.i ], [ %n.vec128, %middle.block144 ]
  %.ph279 = phi i1 [ %410, %start.split.split.i ], [ %436, %middle.block144 ]
  br label %bb8.i4, !dbg !554

bb8.i4:                                           ; preds = %bb8.i4.preheader, %bb8.i4
  %packed.sroa.0.05.i = phi i64 [ %442, %bb8.i4 ], [ %packed.sroa.0.05.i.ph, %bb8.i4.preheader ]
  %iter.sroa.0.04.i = phi i64 [ %441, %bb8.i4 ], [ %iter.sroa.0.04.i.ph, %bb8.i4.preheader ]
  %437 = phi i1 [ %440, %bb8.i4 ], [ %.ph279, %bb8.i4.preheader ]
  %_4.i.i = add nuw nsw i64 %iter.sroa.0.04.i, %332, !dbg !612
  tail call void @llvm.experimental.noalias.scope.decl(metadata !576), !dbg !577
  tail call void @llvm.experimental.noalias.scope.decl(metadata !583), !dbg !584
  %_5.i.i.i.i.i.i = icmp ult i64 %_4.i.i, %view.val1.i.i.i.i.i, !dbg !623
  tail call void @llvm.assume(i1 %_5.i.i.i.i.i.i), !dbg !624
  %_4.i.i.i.i.i.i = getelementptr inbounds nuw i64, ptr %view.val.i.i.i.i.i, i64 %_4.i.i, !dbg !616
  %_0.sroa.0.0.i.i.i.i.i = load i64, ptr %_4.i.i.i.i.i.i, align 8, !dbg !621, !noalias !552, !noundef !23
  tail call void @llvm.experimental.noalias.scope.decl(metadata !586), !dbg !584
  %_5.i.i6.i.i.i.i = icmp ult i64 %_4.i.i, %view.val1.i5.i.i.i.i, !dbg !613
  tail call void @llvm.assume(i1 %_5.i.i6.i.i.i.i), !dbg !614
  %_4.i.i7.i.i.i.i = getelementptr inbounds nuw i64, ptr %view.val.i4.i.i.i.i, i64 %_4.i.i, !dbg !601
  %_0.sroa.0.0.i9.i.i.i.i = load i64, ptr %_4.i.i7.i.i.i.i, align 8, !dbg !606, !noalias !553, !noundef !23
  %438 = icmp eq i64 %_0.sroa.0.0.i.i.i.i.i, 9223372036854775807, !dbg !607
  %439 = icmp eq i64 %_0.sroa.0.0.i9.i.i.i.i, 9223372036854775807, !dbg !607
  %_6.sroa.0.0.i.i.i.i.i = or i1 %438, %439, !dbg !607
  %_5.i.i.i.i.i = icmp slt i64 %_0.sroa.0.0.i.i.i.i.i, %_0.sroa.0.0.i9.i.i.i.i, !dbg !610
  %440 = or i1 %437, %_6.sroa.0.0.i.i.i.i.i, !dbg !599
  %441 = add nuw nsw i64 %iter.sroa.0.04.i, 1, !dbg !569
  %_13.i = zext i1 %_5.i.i.i.i.i to i64, !dbg !587
  %_12.i = shl nuw i64 %_13.i, %iter.sroa.0.04.i, !dbg !587
  %442 = or i64 %_12.i, %packed.sroa.0.05.i, !dbg !588
  %exitcond.not.i = icmp eq i64 %441, %remainder.i, !dbg !591
  br i1 %exitcond.not.i, label %_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack24collect_bool_word_scalarNCINvB2_23collect_bool_words_withNCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1O_7visitor5retry21ExecuteDenseWithRetryINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB3D_EENtNtB3R_11row_visitor10RowVisitor19visit_deferred_boolB3w_bKB3D_NCINvXB4B_B4y_NtNtB1O_6row_fn5RowFn8dispatchB3M_E0NCB6t_s_0E0NCB3I_s_0B7f_E0NCINvB2_23collect_bool_words_avx2B1F_E0E0EB4B_.exit, label %bb8.i4, !dbg !554, !llvm.loop !627

_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack24collect_bool_word_scalarNCINvB2_23collect_bool_words_withNCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1O_7visitor5retry21ExecuteDenseWithRetryINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB3D_EENtNtB3R_11row_visitor10RowVisitor19visit_deferred_boolB3w_bKB3D_NCINvXB4B_B4y_NtNtB1O_6row_fn5RowFn8dispatchB3M_E0NCB6t_s_0E0NCB3I_s_0B7f_E0NCINvB2_23collect_bool_words_avx2B1F_E0E0EB4B_.exit.loopexit: ; preds = %bb8.us.us.i, %vec.epilog.middle.block254, %middle.block231
  %.lcssa = phi i64 [ %351, %vec.epilog.middle.block254 ], [ %346, %middle.block231 ], [ %353, %bb8.us.us.i ], !dbg !588
  %443 = trunc nuw i8 %_11.i.i.promoted.i to i1, !dbg !599
  %444 = icmp eq i64 %_0.sroa.0.0.i.i.i.i.us.us.i, 9223372036854775807
  %445 = icmp eq i64 %_0.sroa.0.0.i9.i.i.i.us.us.i, 9223372036854775807
  %_6.sroa.0.0.i.i.i.i.us.us.i = or i1 %444, %445
  %446 = or i1 %_6.sroa.0.0.i.i.i.i.us.us.i, %443, !dbg !599
  br label %_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack24collect_bool_word_scalarNCINvB2_23collect_bool_words_withNCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1O_7visitor5retry21ExecuteDenseWithRetryINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB3D_EENtNtB3R_11row_visitor10RowVisitor19visit_deferred_boolB3w_bKB3D_NCINvXB4B_B4y_NtNtB1O_6row_fn5RowFn8dispatchB3M_E0NCB6t_s_0E0NCB3I_s_0B7f_E0NCINvB2_23collect_bool_words_avx2B1F_E0E0EB4B_.exit, !dbg !628

_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack24collect_bool_word_scalarNCINvB2_23collect_bool_words_withNCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1O_7visitor5retry21ExecuteDenseWithRetryINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB3D_EENtNtB3R_11row_visitor10RowVisitor19visit_deferred_boolB3w_bKB3D_NCINvXB4B_B4y_NtNtB1O_6row_fn5RowFn8dispatchB3M_E0NCB6t_s_0E0NCB3I_s_0B7f_E0NCINvB2_23collect_bool_words_avx2B1F_E0E0EB4B_.exit: ; preds = %bb8.i4, %bb8.us6.i, %bb8.us.i, %middle.block144, %middle.block173, %middle.block203, %_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack24collect_bool_word_scalarNCINvB2_23collect_bool_words_withNCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1O_7visitor5retry21ExecuteDenseWithRetryINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB3D_EENtNtB3R_11row_visitor10RowVisitor19visit_deferred_boolB3w_bKB3D_NCINvXB4B_B4y_NtNtB1O_6row_fn5RowFn8dispatchB3M_E0NCB6t_s_0E0NCB3I_s_0B7f_E0NCINvB2_23collect_bool_words_avx2B1F_E0E0EB4B_.exit.loopexit
  %.lcssa108.sink = phi i1 [ %407, %bb8.us6.i ], [ %379, %bb8.us.i ], [ %446, %_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack24collect_bool_word_scalarNCINvB2_23collect_bool_words_withNCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1O_7visitor5retry21ExecuteDenseWithRetryINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB3D_EENtNtB3R_11row_visitor10RowVisitor19visit_deferred_boolB3w_bKB3D_NCINvXB4B_B4y_NtNtB1O_6row_fn5RowFn8dispatchB3M_E0NCB6t_s_0E0NCB3I_s_0B7f_E0NCINvB2_23collect_bool_words_avx2B1F_E0E0EB4B_.exit.loopexit ], [ %375, %middle.block203 ], [ %403, %middle.block173 ], [ %436, %middle.block144 ], [ %440, %bb8.i4 ]
  %.us-phi.i = phi i64 [ %409, %bb8.us6.i ], [ %381, %bb8.us.i ], [ %.lcssa, %_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack24collect_bool_word_scalarNCINvB2_23collect_bool_words_withNCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1O_7visitor5retry21ExecuteDenseWithRetryINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB3D_EENtNtB3R_11row_visitor10RowVisitor19visit_deferred_boolB3w_bKB3D_NCINvXB4B_B4y_NtNtB1O_6row_fn5RowFn8dispatchB3M_E0NCB6t_s_0E0NCB3I_s_0B7f_E0NCINvB2_23collect_bool_words_avx2B1F_E0E0EB4B_.exit.loopexit ], [ %373, %middle.block203 ], [ %401, %middle.block173 ], [ %434, %middle.block144 ], [ %442, %bb8.i4 ], !dbg !629
  %447 = zext i1 %.lcssa108.sink to i8
  store i8 %447, ptr %_11.i.i.i, align 8, !dbg !599, !alias.scope !549, !noalias !540
  %_41.i = icmp samesign ult i64 %full5.i, %words.1, !dbg !628
  br i1 %_41.i, label %bb14.i, label %panic.i, !dbg !628

bb14.i:                                           ; preds = %_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack24collect_bool_word_scalarNCINvB2_23collect_bool_words_withNCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1O_7visitor5retry21ExecuteDenseWithRetryINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB3D_EENtNtB3R_11row_visitor10RowVisitor19visit_deferred_boolB3w_bKB3D_NCINvXB4B_B4y_NtNtB1O_6row_fn5RowFn8dispatchB3M_E0NCB6t_s_0E0NCB3I_s_0B7f_E0NCINvB2_23collect_bool_words_avx2B1F_E0E0EB4B_.exit
  store i64 %.us-phi.i, ptr %_52.i, align 8, !dbg !628, !alias.scope !285, !noalias !630
  br label %_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack23collect_bool_words_withNCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1g_7visitor5retry21ExecuteDenseWithRetryINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB35_EENtNtB3j_11row_visitor10RowVisitor19visit_deferred_boolB2Y_bKB35_NCINvXB43_B40_NtNtB1g_6row_fn5RowFn8dispatchB3e_E0NCB5V_s_0E0NCB3a_s_0B6H_E0NCINvB2_23collect_bool_words_avx2B17_E0EB43_.exit, !dbg !632

panic.i:                                          ; preds = %_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack24collect_bool_word_scalarNCINvB2_23collect_bool_words_withNCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1O_7visitor5retry21ExecuteDenseWithRetryINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB3D_EENtNtB3R_11row_visitor10RowVisitor19visit_deferred_boolB3w_bKB3D_NCINvXB4B_B4y_NtNtB1O_6row_fn5RowFn8dispatchB3M_E0NCB6t_s_0E0NCB3I_s_0B7f_E0NCINvB2_23collect_bool_words_avx2B1F_E0E0EB4B_.exit
; call core::panicking::panic_bounds_check
  tail call void @_RNvNtCsc36rpYXAlPq_4core9panicking18panic_bounds_check(i64 noundef %full5.i, i64 noundef range(i64 0, 1152921504606846976) %words.1, ptr noalias nofree noundef readonly align 8 captures(address, read_provenance) dereferenceable(24) @alloc_1b2922caae1b461da04857d3d02eae5b) #21, !dbg !628
  unreachable

bb9.i.split:                                      ; preds = %vector.body75, %vector.body
  %bools.i.sroa.44.2 = phi <32 x i8> [ %bools.i.sroa.44.60.vecblend397, %vector.body ], [ %bools.i.sroa.44.60.vecblend, %vector.body75 ], !dbg !455
  %bools.i.sroa.0.2 = phi <32 x i8> [ %bools.i.sroa.0.28.vecblend346, %vector.body ], [ %bools.i.sroa.0.28.vecblend, %vector.body75 ], !dbg !455
  %.us-phi56.in.in.in = phi <4 x i1> [ %bin.rdx69, %vector.body ], [ %bin.rdx85, %vector.body75 ]
  %.us-phi56.in.in = bitcast <4 x i1> %.us-phi56.in.in.in to i4, !dbg !392
  %.us-phi56.in = icmp ne i4 %.us-phi56.in.in, 0, !dbg !392
  %448 = shufflevector <32 x i8> %bools.i.sroa.0.2, <32 x i8> %bools.i.sroa.44.2, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>, !dbg !468
  %449 = icmp ne <64 x i8> %448, zeroinitializer, !dbg !468
  store <64 x i1> %449, ptr %iter.i.sroa.0.060, align 8, !dbg !400
  %_7.i.i = icmp eq ptr %_17.i.i, %_52.i, !dbg !332
  br i1 %_7.i.i, label %bb1.i.bb5.i_crit_edge, label %bb4.i, !dbg !350

_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack23collect_bool_words_withNCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1g_7visitor5retry21ExecuteDenseWithRetryINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB35_EENtNtB3j_11row_visitor10RowVisitor19visit_deferred_boolB2Y_bKB35_NCINvXB43_B40_NtNtB1g_6row_fn5RowFn8dispatchB3e_E0NCB5V_s_0E0NCB3a_s_0B6H_E0NCINvB2_23collect_bool_words_avx2B17_E0EB43_.exit: ; preds = %bb14.i, %bb5.i
  ret void, !dbg !633
}
