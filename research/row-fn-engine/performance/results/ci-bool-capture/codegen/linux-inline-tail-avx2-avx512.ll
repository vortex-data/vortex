define hidden void @_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor5retry21ExecuteDenseWithRetryINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB45_B42_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5X_s_0E0NCB3c_s_0B6J_E0EB45_(ptr noalias nofree noundef nonnull writeonly align 8 captures(address) %words.0, i64 noundef range(i64 0, 1152921504606846976) %words.1, i64 noundef %len, ptr noalias nofree noundef align 8 captures(none) dereferenceable(64) %f) unnamed_addr #2 personality ptr @rust_eh_personality !dbg !612 {
start:
  tail call void @llvm.experimental.noalias.scope.decl(metadata !613), !dbg !616
  %full5.i = lshr i64 %len, 6, !dbg !617
  %remainder.i = and i64 %len, 63, !dbg !620
  %_42.not.i = icmp samesign ugt i64 %full5.i, %words.1
  br i1 %_42.not.i, label %bb22.i, label %bb21.i, !dbg !622, !prof !293

bb22.i:                                           ; preds = %start
; call core::slice::index::slice_index_fail
  tail call void @_RNvNtNtCsc36rpYXAlPq_4core5slice5index16slice_index_fail(i64 noundef 0, i64 noundef %full5.i, i64 noundef range(i64 0, 1152921504606846976) %words.1, ptr noalias nofree noundef readonly align 8 captures(address, read_provenance) dereferenceable(24) @alloc_44f32c0500caa70dfe7fcfa87250b31f.llvm.5180523300074625071) #19, !dbg !632, !noalias !613
  unreachable

bb21.i:                                           ; preds = %start
  %_52.i.idx = shl nuw nsw i64 %full5.i, 3, !dbg !633
  %_52.i = getelementptr inbounds nuw i8, ptr %words.0, i64 %_52.i.idx, !dbg !633
  %_7.i.i28 = icmp eq i64 %full5.i, 0, !dbg !642
  br i1 %_7.i.i28, label %bb5.i, label %bb4.i.preheader, !dbg !647

bb4.i.preheader:                                  ; preds = %bb21.i
  %_3.i.i.i1 = load i64, ptr %f, align 8, !range !356, !alias.scope !648, !noalias !653, !noundef !23
  %0 = trunc nuw i64 %_3.i.i.i1 to i1
  %_6.i.i11 = getelementptr inbounds nuw i8, ptr %f, i64 24
  %_3.i1.i.i12 = load i64, ptr %_6.i.i11, align 8, !range !356, !alias.scope !656, !noalias !653, !noundef !23
  %1 = trunc nuw i64 %_3.i1.i.i12 to i1
  %_11.i = getelementptr inbounds nuw i8, ptr %f, i64 56
  %view.i.i.i3 = getelementptr inbounds nuw i8, ptr %f, i64 8
  %view.val.i.i.i4 = load ptr, ptr %view.i.i.i3, align 8, !nonnull !23, !align !368
  %view.i3.i.i14 = getelementptr inbounds nuw i8, ptr %f, i64 32
  %view.val.i4.i.i15 = load ptr, ptr %view.i3.i.i14, align 8, !nonnull !23, !align !368
  %2 = getelementptr inbounds nuw i8, ptr %f, i64 40
  %view.val1.i5.i.i16 = load i64, ptr %2, align 8
  %_6.i11.i.i22.cast = inttoptr i64 %view.val1.i5.i.i16 to ptr
  %_11.i.promoted42 = load i8, ptr %_11.i, align 8, !alias.scope !659, !noalias !653
  br i1 %0, label %bb4.i.preheader.split.us, label %bb4.i.preheader.split

bb4.i.preheader.split.us:                         ; preds = %bb4.i.preheader
  %3 = getelementptr inbounds nuw i8, ptr %f, i64 16
  %view.val1.i.i.i5 = load i64, ptr %3, align 8
  %4 = inttoptr i64 %view.val1.i.i.i5 to ptr
  %_0.sroa.0.0.i.i.i10.us.us = load i64, ptr %4, align 8, !noalias !662, !noundef !23
  %5 = icmp eq i64 %_0.sroa.0.0.i.i.i10.us.us, 9223372036854775807
  br i1 %1, label %bb4.i.preheader.split.us.split.us, label %bb4.i.us.preheader

bb4.i.us.preheader:                               ; preds = %bb4.i.preheader.split.us
  %6 = trunc nuw i8 %_11.i.promoted42 to i1, !dbg !663
  %broadcast.splatinsert195 = insertelement <8 x i1> poison, i1 %5, i64 0
  %broadcast.splat196 = shufflevector <8 x i1> %broadcast.splatinsert195, <8 x i1> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert193 = insertelement <8 x i64> poison, i64 %_0.sroa.0.0.i.i.i10.us.us, i64 0
  %broadcast.splat194 = shufflevector <8 x i64> %broadcast.splatinsert193, <8 x i64> poison, <8 x i32> zeroinitializer
  %7 = getelementptr inbounds nuw i8, ptr %view.val.i4.i.i15, i64 256
  br label %bb4.i.us, !dbg !671

bb4.i.preheader.split.us.split.us:                ; preds = %bb4.i.preheader.split.us
  %_0.sroa.0.0.i9.i.i20.us.us.us.us = load i64, ptr %_6.i11.i.i22.cast, align 8, !noalias !674, !noundef !23
  %8 = icmp eq i64 %_0.sroa.0.0.i9.i.i20.us.us.us.us, 9223372036854775807
  %_5.i.i.i.us.us.us.us = icmp slt i64 %_0.sroa.0.0.i.i.i10.us.us, %_0.sroa.0.0.i9.i.i20.us.us.us.us
  %9 = trunc nuw i8 %_11.i.promoted42 to i1, !dbg !663
  %10 = select i1 %_5.i.i.i.us.us.us.us, i512 257, i512 0
  %11 = shl nuw nsw i512 %10, 16
  %12 = or disjoint i512 %10, %11
  %13 = shl nuw nsw i512 %12, 32
  %14 = or disjoint i512 %12, %13
  %15 = shl nuw nsw i512 %14, 64
  %16 = or disjoint i512 %14, %15
  %17 = shl nuw nsw i512 %16, 128
  %18 = or i512 %16, %17
  %19 = shl nuw nsw i512 %18, 256
  %20 = or i512 %18, %19
  %21 = bitcast i512 %20 to <64 x i8>
  %.not.i.i.i.us.us = icmp ne <64 x i8> %21, zeroinitializer
  %22 = or i1 %8, %9
  %23 = or i1 %22, %5
  %24 = add nsw i64 %_52.i.idx, -8, !dbg !671
  %25 = lshr exact i64 %24, 3, !dbg !671
  %26 = add nuw nsw i64 %25, 1, !dbg !671
  %xtraiter = and i64 %26, 7, !dbg !671
  %27 = and i64 %24, 56, !dbg !671
  %lcmp.mod.not = icmp eq i64 %27, 56, !dbg !671
  br i1 %lcmp.mod.not, label %bb4.i.us.us.prol.loopexit, label %bb4.i.us.us.prol, !dbg !671

bb4.i.us.us.prol:                                 ; preds = %bb4.i.preheader.split.us.split.us, %bb4.i.us.us.prol
  %iter.i.sroa.0.030.us.us.prol = phi ptr [ %_17.i.i.us.us.prol, %bb4.i.us.us.prol ], [ %words.0, %bb4.i.preheader.split.us.split.us ]
  %prol.iter = phi i64 [ %prol.iter.next, %bb4.i.us.us.prol ], [ 0, %bb4.i.preheader.split.us.split.us ]
  %_17.i.i.us.us.prol = getelementptr inbounds nuw i8, ptr %iter.i.sroa.0.030.us.us.prol, i64 8, !dbg !675
  store <64 x i1> %.not.i.i.i.us.us, ptr %iter.i.sroa.0.030.us.us.prol, align 8, !dbg !677
  %prol.iter.next = add i64 %prol.iter, 1, !dbg !647
  %prol.iter.cmp.not = icmp eq i64 %prol.iter.next, %xtraiter, !dbg !647
  br i1 %prol.iter.cmp.not, label %bb4.i.us.us.prol.loopexit, label %bb4.i.us.us.prol, !dbg !647, !llvm.loop !678

bb4.i.us.us.prol.loopexit:                        ; preds = %bb4.i.us.us.prol, %bb4.i.preheader.split.us.split.us
  %iter.i.sroa.0.030.us.us.unr = phi ptr [ %words.0, %bb4.i.preheader.split.us.split.us ], [ %_17.i.i.us.us.prol, %bb4.i.us.us.prol ]
  %28 = icmp ult i64 %24, 56, !dbg !671
  br i1 %28, label %bb5.i.loopexit, label %bb4.i.us.us, !dbg !671

bb4.i.us.us:                                      ; preds = %bb4.i.us.us.prol.loopexit, %bb4.i.us.us
  %iter.i.sroa.0.030.us.us = phi ptr [ %_17.i.i.us.us.7, %bb4.i.us.us ], [ %iter.i.sroa.0.030.us.us.unr, %bb4.i.us.us.prol.loopexit ]
  %_17.i.i.us.us = getelementptr inbounds nuw i8, ptr %iter.i.sroa.0.030.us.us, i64 8, !dbg !675
  store <64 x i1> %.not.i.i.i.us.us, ptr %iter.i.sroa.0.030.us.us, align 8, !dbg !677
  %_17.i.i.us.us.1 = getelementptr inbounds nuw i8, ptr %iter.i.sroa.0.030.us.us, i64 16, !dbg !675
  store <64 x i1> %.not.i.i.i.us.us, ptr %_17.i.i.us.us, align 8, !dbg !677
  %_17.i.i.us.us.2 = getelementptr inbounds nuw i8, ptr %iter.i.sroa.0.030.us.us, i64 24, !dbg !675
  store <64 x i1> %.not.i.i.i.us.us, ptr %_17.i.i.us.us.1, align 8, !dbg !677
  %_17.i.i.us.us.3 = getelementptr inbounds nuw i8, ptr %iter.i.sroa.0.030.us.us, i64 32, !dbg !675
  store <64 x i1> %.not.i.i.i.us.us, ptr %_17.i.i.us.us.2, align 8, !dbg !677
  %_17.i.i.us.us.4 = getelementptr inbounds nuw i8, ptr %iter.i.sroa.0.030.us.us, i64 40, !dbg !675
  store <64 x i1> %.not.i.i.i.us.us, ptr %_17.i.i.us.us.3, align 8, !dbg !677
  %_17.i.i.us.us.5 = getelementptr inbounds nuw i8, ptr %iter.i.sroa.0.030.us.us, i64 48, !dbg !675
  store <64 x i1> %.not.i.i.i.us.us, ptr %_17.i.i.us.us.4, align 8, !dbg !677
  %_17.i.i.us.us.6 = getelementptr inbounds nuw i8, ptr %iter.i.sroa.0.030.us.us, i64 56, !dbg !675
  store <64 x i1> %.not.i.i.i.us.us, ptr %_17.i.i.us.us.5, align 8, !dbg !677
  %_17.i.i.us.us.7 = getelementptr inbounds nuw i8, ptr %iter.i.sroa.0.030.us.us, i64 64, !dbg !675
  store <64 x i1> %.not.i.i.i.us.us, ptr %_17.i.i.us.us.6, align 8, !dbg !677
  %_7.i.i.us.us.7 = icmp eq ptr %_17.i.i.us.us.7, %_52.i, !dbg !642
  br i1 %_7.i.i.us.us.7, label %bb5.i.loopexit, label %bb4.i.us.us, !dbg !647

bb4.i.us:                                         ; preds = %bb4.i.us.preheader, %bb4.i.us
  %.us-phi43.us = phi i1 [ %69, %bb4.i.us ], [ %6, %bb4.i.us.preheader ]
  %iter.i.sroa.0.030.us = phi ptr [ %_17.i.i.us, %bb4.i.us ], [ %words.0, %bb4.i.us.preheader ]
  %iter.i.sroa.7.029.us = phi i64 [ %_9.0.i.us, %bb4.i.us ], [ 0, %bb4.i.us.preheader ]
  %29 = insertelement <8 x i1> <i1 poison, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false>, i1 %.us-phi43.us, i64 0
  tail call void @llvm.experimental.noalias.scope.decl(metadata !680), !dbg !681
  tail call void @llvm.experimental.noalias.scope.decl(metadata !682), !dbg !683
  tail call void @llvm.experimental.noalias.scope.decl(metadata !685), !dbg !683
  %.idx440 = shl i64 %iter.i.sroa.7.029.us, 9, !dbg !686
  %30 = getelementptr inbounds nuw i8, ptr %view.val.i4.i.i15, i64 %.idx440, !dbg !686
  %31 = getelementptr inbounds nuw i8, ptr %30, i64 64, !dbg !691
  %32 = getelementptr inbounds nuw i8, ptr %30, i64 128, !dbg !691
  %33 = getelementptr inbounds nuw i8, ptr %30, i64 192, !dbg !691
  %wide.load207 = load <8 x i64>, ptr %30, align 8, !dbg !691, !noalias !674
  %wide.load208 = load <8 x i64>, ptr %31, align 8, !dbg !691, !noalias !674
  %wide.load209 = load <8 x i64>, ptr %32, align 8, !dbg !691, !noalias !674
  %wide.load210 = load <8 x i64>, ptr %33, align 8, !dbg !691, !noalias !674
  %34 = icmp eq <8 x i64> %wide.load207, splat (i64 9223372036854775807), !dbg !692
  %35 = icmp eq <8 x i64> %wide.load208, splat (i64 9223372036854775807), !dbg !692
  %36 = icmp eq <8 x i64> %wide.load209, splat (i64 9223372036854775807), !dbg !692
  %37 = icmp eq <8 x i64> %wide.load210, splat (i64 9223372036854775807), !dbg !692
  %38 = icmp slt <8 x i64> %broadcast.splat194, %wide.load207, !dbg !695
  %39 = icmp slt <8 x i64> %broadcast.splat194, %wide.load208, !dbg !695
  %40 = icmp slt <8 x i64> %broadcast.splat194, %wide.load209, !dbg !695
  %41 = icmp slt <8 x i64> %broadcast.splat194, %wide.load210, !dbg !695
  %42 = or <8 x i1> %34, %29, !dbg !663
  %43 = zext <8 x i1> %38 to <8 x i8>, !dbg !696
  %44 = zext <8 x i1> %39 to <8 x i8>, !dbg !696
  %45 = zext <8 x i1> %40 to <8 x i8>, !dbg !696
  %46 = zext <8 x i1> %41 to <8 x i8>, !dbg !696
  %bools.i.sroa.0.0.vec.expand498 = shufflevector <8 x i8> %43, <8 x i8> poison, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !696
  %bools.i.sroa.0.8.vec.expand506 = shufflevector <8 x i8> %44, <8 x i8> poison, <64 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !696
  %bools.i.sroa.0.8.vecblend507 = shufflevector <64 x i8> %bools.i.sroa.0.0.vec.expand498, <64 x i8> %bools.i.sroa.0.8.vec.expand506, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 72, i32 73, i32 74, i32 75, i32 76, i32 77, i32 78, i32 79, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !696
  %bools.i.sroa.0.16.vec.expand512 = shufflevector <8 x i8> %45, <8 x i8> poison, <64 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !696
  %bools.i.sroa.0.16.vecblend513 = shufflevector <64 x i8> %bools.i.sroa.0.8.vecblend507, <64 x i8> %bools.i.sroa.0.16.vec.expand512, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 80, i32 81, i32 82, i32 83, i32 84, i32 85, i32 86, i32 87, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !696
  %bools.i.sroa.0.24.vec.expand518 = shufflevector <8 x i8> %46, <8 x i8> poison, <64 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !696
  %bools.i.sroa.0.24.vecblend519 = shufflevector <64 x i8> %bools.i.sroa.0.16.vecblend513, <64 x i8> %bools.i.sroa.0.24.vec.expand518, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 88, i32 89, i32 90, i32 91, i32 92, i32 93, i32 94, i32 95, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !696
  %.idx440.1 = shl i64 %iter.i.sroa.7.029.us, 9, !dbg !686
  %47 = getelementptr inbounds nuw i8, ptr %7, i64 %.idx440.1, !dbg !686
  %48 = getelementptr inbounds nuw i8, ptr %47, i64 64, !dbg !691
  %49 = getelementptr inbounds nuw i8, ptr %47, i64 128, !dbg !691
  %50 = getelementptr inbounds nuw i8, ptr %47, i64 192, !dbg !691
  %wide.load207.1 = load <8 x i64>, ptr %47, align 8, !dbg !691, !noalias !697
  %wide.load208.1 = load <8 x i64>, ptr %48, align 8, !dbg !691, !noalias !697
  %wide.load209.1 = load <8 x i64>, ptr %49, align 8, !dbg !691, !noalias !697
  %wide.load210.1 = load <8 x i64>, ptr %50, align 8, !dbg !691, !noalias !697
  %51 = icmp eq <8 x i64> %wide.load207.1, splat (i64 9223372036854775807), !dbg !692
  %52 = icmp eq <8 x i64> %wide.load208.1, splat (i64 9223372036854775807), !dbg !692
  %53 = icmp eq <8 x i64> %wide.load209.1, splat (i64 9223372036854775807), !dbg !692
  %54 = icmp eq <8 x i64> %wide.load210.1, splat (i64 9223372036854775807), !dbg !692
  %55 = icmp slt <8 x i64> %broadcast.splat194, %wide.load207.1, !dbg !695
  %56 = icmp slt <8 x i64> %broadcast.splat194, %wide.load208.1, !dbg !695
  %57 = icmp slt <8 x i64> %broadcast.splat194, %wide.load209.1, !dbg !695
  %58 = icmp slt <8 x i64> %broadcast.splat194, %wide.load210.1, !dbg !695
  %59 = or <8 x i1> %42, %51, !dbg !663
  %60 = or <8 x i1> %59, %broadcast.splat196, !dbg !663
  %61 = or <8 x i1> %35, %52, !dbg !663
  %62 = or <8 x i1> %36, %53, !dbg !663
  %63 = or <8 x i1> %37, %54, !dbg !663
  %64 = zext <8 x i1> %55 to <8 x i8>, !dbg !696
  %65 = zext <8 x i1> %56 to <8 x i8>, !dbg !696
  %66 = zext <8 x i1> %57 to <8 x i8>, !dbg !696
  %67 = zext <8 x i1> %58 to <8 x i8>, !dbg !696
  %bools.i.sroa.0.32.vec.expand524 = shufflevector <8 x i8> %64, <8 x i8> poison, <64 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !696
  %bools.i.sroa.0.32.vecblend525 = shufflevector <64 x i8> %bools.i.sroa.0.24.vecblend519, <64 x i8> %bools.i.sroa.0.32.vec.expand524, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 96, i32 97, i32 98, i32 99, i32 100, i32 101, i32 102, i32 103, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !696
  %bools.i.sroa.0.40.vec.expand530 = shufflevector <8 x i8> %65, <8 x i8> poison, <64 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !696
  %bools.i.sroa.0.40.vecblend531 = shufflevector <64 x i8> %bools.i.sroa.0.32.vecblend525, <64 x i8> %bools.i.sroa.0.40.vec.expand530, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 104, i32 105, i32 106, i32 107, i32 108, i32 109, i32 110, i32 111, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !696
  %bools.i.sroa.0.48.vec.expand536 = shufflevector <8 x i8> %66, <8 x i8> poison, <64 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !696
  %bools.i.sroa.0.48.vecblend537 = shufflevector <64 x i8> %bools.i.sroa.0.40.vecblend531, <64 x i8> %bools.i.sroa.0.48.vec.expand536, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 112, i32 113, i32 114, i32 115, i32 116, i32 117, i32 118, i32 119, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !696
  %bools.i.sroa.0.56.vec.expand542 = shufflevector <8 x i8> %67, <8 x i8> poison, <64 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !696
  %bools.i.sroa.0.56.vecblend543 = shufflevector <64 x i8> %bools.i.sroa.0.48.vecblend537, <64 x i8> %bools.i.sroa.0.56.vec.expand542, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 120, i32 121, i32 122, i32 123, i32 124, i32 125, i32 126, i32 127>, !dbg !696
  %bin.rdx214 = or <8 x i1> %61, %60, !dbg !671
  %bin.rdx215 = or <8 x i1> %62, %bin.rdx214, !dbg !671
  %bin.rdx216 = or <8 x i1> %63, %bin.rdx215, !dbg !671
  %68 = bitcast <8 x i1> %bin.rdx216 to i8, !dbg !671
  %69 = icmp ne i8 %68, 0, !dbg !671
  %_17.i.i.us = getelementptr inbounds nuw i8, ptr %iter.i.sroa.0.030.us, i64 8, !dbg !675
  %_9.0.i.us = add nuw nsw i64 %iter.i.sroa.7.029.us, 1, !dbg !700
  %.not.i.i.i.us = icmp ne <64 x i8> %bools.i.sroa.0.56.vecblend543, zeroinitializer, !dbg !701
  store <64 x i1> %.not.i.i.i.us, ptr %iter.i.sroa.0.030.us, align 8, !dbg !677
  %_7.i.i.us = icmp eq ptr %_17.i.i.us, %_52.i, !dbg !642
  br i1 %_7.i.i.us, label %bb5.i.loopexit, label %bb4.i.us, !dbg !647

bb4.i.preheader.split:                            ; preds = %bb4.i.preheader
  br i1 %1, label %bb4.i.preheader.split.split.us, label %bb4.i.preheader96

bb4.i.preheader96:                                ; preds = %bb4.i.preheader.split
  %70 = trunc nuw i8 %_11.i.promoted42 to i1, !dbg !663
  br label %bb4.i, !dbg !671

bb4.i.preheader.split.split.us:                   ; preds = %bb4.i.preheader.split
  %_0.sroa.0.0.i9.i.i20.us35.us = load i64, ptr %_6.i11.i.i22.cast, align 8, !noalias !674, !noundef !23
  %71 = icmp eq i64 %_0.sroa.0.0.i9.i.i20.us35.us, 9223372036854775807
  %72 = trunc nuw i8 %_11.i.promoted42 to i1, !dbg !663
  %broadcast.splatinsert166 = insertelement <8 x i64> poison, i64 %_0.sroa.0.0.i9.i.i20.us35.us, i64 0
  %broadcast.splat167 = shufflevector <8 x i64> %broadcast.splatinsert166, <8 x i64> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert164 = insertelement <8 x i1> poison, i1 %71, i64 0
  %broadcast.splat165 = shufflevector <8 x i1> %broadcast.splatinsert164, <8 x i1> poison, <8 x i32> zeroinitializer
  %73 = getelementptr inbounds nuw i8, ptr %view.val.i.i.i4, i64 256
  br label %bb4.i.us56, !dbg !671

bb4.i.us56:                                       ; preds = %bb4.i.us56, %bb4.i.preheader.split.split.us
  %.us-phi43.us57 = phi i1 [ %114, %bb4.i.us56 ], [ %72, %bb4.i.preheader.split.split.us ]
  %iter.i.sroa.0.030.us58 = phi ptr [ %_17.i.i.us61, %bb4.i.us56 ], [ %words.0, %bb4.i.preheader.split.split.us ]
  %iter.i.sroa.7.029.us59 = phi i64 [ %_9.0.i.us62, %bb4.i.us56 ], [ 0, %bb4.i.preheader.split.split.us ]
  %74 = insertelement <8 x i1> <i1 poison, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false>, i1 %.us-phi43.us57, i64 0
  tail call void @llvm.experimental.noalias.scope.decl(metadata !680), !dbg !681
  tail call void @llvm.experimental.noalias.scope.decl(metadata !682), !dbg !683
  %.idx = shl i64 %iter.i.sroa.7.029.us59, 9, !dbg !718
  %75 = getelementptr inbounds nuw i8, ptr %view.val.i.i.i4, i64 %.idx, !dbg !718
  %76 = getelementptr inbounds nuw i8, ptr %75, i64 64, !dbg !723
  %77 = getelementptr inbounds nuw i8, ptr %75, i64 128, !dbg !723
  %78 = getelementptr inbounds nuw i8, ptr %75, i64 192, !dbg !723
  %wide.load178 = load <8 x i64>, ptr %75, align 8, !dbg !723, !noalias !662
  %wide.load179 = load <8 x i64>, ptr %76, align 8, !dbg !723, !noalias !662
  %wide.load180 = load <8 x i64>, ptr %77, align 8, !dbg !723, !noalias !662
  %wide.load181 = load <8 x i64>, ptr %78, align 8, !dbg !723, !noalias !662
  tail call void @llvm.experimental.noalias.scope.decl(metadata !685), !dbg !683
  %79 = icmp eq <8 x i64> %wide.load178, splat (i64 9223372036854775807), !dbg !692
  %80 = icmp eq <8 x i64> %wide.load179, splat (i64 9223372036854775807), !dbg !692
  %81 = icmp eq <8 x i64> %wide.load180, splat (i64 9223372036854775807), !dbg !692
  %82 = icmp eq <8 x i64> %wide.load181, splat (i64 9223372036854775807), !dbg !692
  %83 = icmp slt <8 x i64> %wide.load178, %broadcast.splat167, !dbg !695
  %84 = icmp slt <8 x i64> %wide.load179, %broadcast.splat167, !dbg !695
  %85 = icmp slt <8 x i64> %wide.load180, %broadcast.splat167, !dbg !695
  %86 = icmp slt <8 x i64> %wide.load181, %broadcast.splat167, !dbg !695
  %87 = or <8 x i1> %79, %74, !dbg !663
  %88 = zext <8 x i1> %83 to <8 x i8>, !dbg !696
  %89 = zext <8 x i1> %84 to <8 x i8>, !dbg !696
  %90 = zext <8 x i1> %85 to <8 x i8>, !dbg !696
  %91 = zext <8 x i1> %86 to <8 x i8>, !dbg !696
  %bools.i.sroa.0.0.vec.expand495 = shufflevector <8 x i8> %88, <8 x i8> poison, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !696
  %bools.i.sroa.0.8.vec.expand503 = shufflevector <8 x i8> %89, <8 x i8> poison, <64 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !696
  %bools.i.sroa.0.8.vecblend504 = shufflevector <64 x i8> %bools.i.sroa.0.0.vec.expand495, <64 x i8> %bools.i.sroa.0.8.vec.expand503, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 72, i32 73, i32 74, i32 75, i32 76, i32 77, i32 78, i32 79, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !696
  %bools.i.sroa.0.16.vec.expand509 = shufflevector <8 x i8> %90, <8 x i8> poison, <64 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !696
  %bools.i.sroa.0.16.vecblend510 = shufflevector <64 x i8> %bools.i.sroa.0.8.vecblend504, <64 x i8> %bools.i.sroa.0.16.vec.expand509, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 80, i32 81, i32 82, i32 83, i32 84, i32 85, i32 86, i32 87, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !696
  %bools.i.sroa.0.24.vec.expand515 = shufflevector <8 x i8> %91, <8 x i8> poison, <64 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !696
  %bools.i.sroa.0.24.vecblend516 = shufflevector <64 x i8> %bools.i.sroa.0.16.vecblend510, <64 x i8> %bools.i.sroa.0.24.vec.expand515, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 88, i32 89, i32 90, i32 91, i32 92, i32 93, i32 94, i32 95, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !696
  %.idx.1 = shl i64 %iter.i.sroa.7.029.us59, 9, !dbg !718
  %92 = getelementptr inbounds nuw i8, ptr %73, i64 %.idx.1, !dbg !718
  %93 = getelementptr inbounds nuw i8, ptr %92, i64 64, !dbg !723
  %94 = getelementptr inbounds nuw i8, ptr %92, i64 128, !dbg !723
  %95 = getelementptr inbounds nuw i8, ptr %92, i64 192, !dbg !723
  %wide.load178.1 = load <8 x i64>, ptr %92, align 8, !dbg !723, !noalias !724
  %wide.load179.1 = load <8 x i64>, ptr %93, align 8, !dbg !723, !noalias !724
  %wide.load180.1 = load <8 x i64>, ptr %94, align 8, !dbg !723, !noalias !724
  %wide.load181.1 = load <8 x i64>, ptr %95, align 8, !dbg !723, !noalias !724
  %96 = icmp eq <8 x i64> %wide.load178.1, splat (i64 9223372036854775807), !dbg !692
  %97 = icmp eq <8 x i64> %wide.load179.1, splat (i64 9223372036854775807), !dbg !692
  %98 = icmp eq <8 x i64> %wide.load180.1, splat (i64 9223372036854775807), !dbg !692
  %99 = icmp eq <8 x i64> %wide.load181.1, splat (i64 9223372036854775807), !dbg !692
  %100 = icmp slt <8 x i64> %wide.load178.1, %broadcast.splat167, !dbg !695
  %101 = icmp slt <8 x i64> %wide.load179.1, %broadcast.splat167, !dbg !695
  %102 = icmp slt <8 x i64> %wide.load180.1, %broadcast.splat167, !dbg !695
  %103 = icmp slt <8 x i64> %wide.load181.1, %broadcast.splat167, !dbg !695
  %104 = or <8 x i1> %87, %96, !dbg !663
  %105 = or <8 x i1> %104, %broadcast.splat165, !dbg !663
  %106 = or <8 x i1> %80, %97, !dbg !663
  %107 = or <8 x i1> %81, %98, !dbg !663
  %108 = or <8 x i1> %82, %99, !dbg !663
  %109 = zext <8 x i1> %100 to <8 x i8>, !dbg !696
  %110 = zext <8 x i1> %101 to <8 x i8>, !dbg !696
  %111 = zext <8 x i1> %102 to <8 x i8>, !dbg !696
  %112 = zext <8 x i1> %103 to <8 x i8>, !dbg !696
  %bools.i.sroa.0.32.vec.expand521 = shufflevector <8 x i8> %109, <8 x i8> poison, <64 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !696
  %bools.i.sroa.0.32.vecblend522 = shufflevector <64 x i8> %bools.i.sroa.0.24.vecblend516, <64 x i8> %bools.i.sroa.0.32.vec.expand521, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 96, i32 97, i32 98, i32 99, i32 100, i32 101, i32 102, i32 103, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !696
  %bools.i.sroa.0.40.vec.expand527 = shufflevector <8 x i8> %110, <8 x i8> poison, <64 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !696
  %bools.i.sroa.0.40.vecblend528 = shufflevector <64 x i8> %bools.i.sroa.0.32.vecblend522, <64 x i8> %bools.i.sroa.0.40.vec.expand527, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 104, i32 105, i32 106, i32 107, i32 108, i32 109, i32 110, i32 111, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !696
  %bools.i.sroa.0.48.vec.expand533 = shufflevector <8 x i8> %111, <8 x i8> poison, <64 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !696
  %bools.i.sroa.0.48.vecblend534 = shufflevector <64 x i8> %bools.i.sroa.0.40.vecblend528, <64 x i8> %bools.i.sroa.0.48.vec.expand533, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 112, i32 113, i32 114, i32 115, i32 116, i32 117, i32 118, i32 119, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !696
  %bools.i.sroa.0.56.vec.expand539 = shufflevector <8 x i8> %112, <8 x i8> poison, <64 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !696
  %bools.i.sroa.0.56.vecblend540 = shufflevector <64 x i8> %bools.i.sroa.0.48.vecblend534, <64 x i8> %bools.i.sroa.0.56.vec.expand539, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 120, i32 121, i32 122, i32 123, i32 124, i32 125, i32 126, i32 127>, !dbg !696
  %bin.rdx185 = or <8 x i1> %106, %105, !dbg !671
  %bin.rdx186 = or <8 x i1> %107, %bin.rdx185, !dbg !671
  %bin.rdx187 = or <8 x i1> %108, %bin.rdx186, !dbg !671
  %113 = bitcast <8 x i1> %bin.rdx187 to i8, !dbg !671
  %114 = icmp ne i8 %113, 0, !dbg !671
  %_17.i.i.us61 = getelementptr inbounds nuw i8, ptr %iter.i.sroa.0.030.us58, i64 8, !dbg !675
  %_9.0.i.us62 = add nuw nsw i64 %iter.i.sroa.7.029.us59, 1, !dbg !700
  %.not.i.i.i.us64 = icmp ne <64 x i8> %bools.i.sroa.0.56.vecblend540, zeroinitializer, !dbg !701
  store <64 x i1> %.not.i.i.i.us64, ptr %iter.i.sroa.0.030.us58, align 8, !dbg !677
  %_7.i.i.us65 = icmp eq ptr %_17.i.i.us61, %_52.i, !dbg !642
  br i1 %_7.i.i.us65, label %bb5.i.loopexit, label %bb4.i.us56, !dbg !647

bb4.i:                                            ; preds = %bb4.i.preheader96, %bb4.i
  %.us-phi43 = phi i1 [ %179, %bb4.i ], [ %70, %bb4.i.preheader96 ]
  %iter.i.sroa.0.030 = phi ptr [ %_17.i.i, %bb4.i ], [ %words.0, %bb4.i.preheader96 ]
  %iter.i.sroa.7.029 = phi i64 [ %_9.0.i, %bb4.i ], [ 0, %bb4.i.preheader96 ]
  %115 = insertelement <8 x i1> <i1 poison, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false>, i1 %.us-phi43, i64 0
  %offset2.i = shl i64 %iter.i.sroa.7.029, 6, !dbg !727
  tail call void @llvm.experimental.noalias.scope.decl(metadata !680), !dbg !681
  tail call void @llvm.experimental.noalias.scope.decl(metadata !682), !dbg !683
  %116 = getelementptr inbounds nuw i64, ptr %view.val.i.i.i4, i64 %offset2.i, !dbg !718
  %117 = getelementptr inbounds nuw i8, ptr %116, i64 64, !dbg !723
  %118 = getelementptr inbounds nuw i8, ptr %116, i64 128, !dbg !723
  %119 = getelementptr inbounds nuw i8, ptr %116, i64 192, !dbg !723
  %wide.load = load <8 x i64>, ptr %116, align 8, !dbg !723, !noalias !662
  %wide.load150 = load <8 x i64>, ptr %117, align 8, !dbg !723, !noalias !662
  %wide.load151 = load <8 x i64>, ptr %118, align 8, !dbg !723, !noalias !662
  %wide.load152 = load <8 x i64>, ptr %119, align 8, !dbg !723, !noalias !662
  tail call void @llvm.experimental.noalias.scope.decl(metadata !685), !dbg !683
  %120 = getelementptr inbounds nuw i64, ptr %view.val.i4.i.i15, i64 %offset2.i, !dbg !686
  %121 = getelementptr inbounds nuw i8, ptr %120, i64 64, !dbg !691
  %122 = getelementptr inbounds nuw i8, ptr %120, i64 128, !dbg !691
  %123 = getelementptr inbounds nuw i8, ptr %120, i64 192, !dbg !691
  %wide.load153 = load <8 x i64>, ptr %120, align 8, !dbg !691, !noalias !674
  %wide.load154 = load <8 x i64>, ptr %121, align 8, !dbg !691, !noalias !674
  %wide.load155 = load <8 x i64>, ptr %122, align 8, !dbg !691, !noalias !674
  %wide.load156 = load <8 x i64>, ptr %123, align 8, !dbg !691, !noalias !674
  %124 = icmp eq <8 x i64> %wide.load, splat (i64 9223372036854775807), !dbg !692
  %125 = icmp eq <8 x i64> %wide.load150, splat (i64 9223372036854775807), !dbg !692
  %126 = icmp eq <8 x i64> %wide.load151, splat (i64 9223372036854775807), !dbg !692
  %127 = icmp eq <8 x i64> %wide.load152, splat (i64 9223372036854775807), !dbg !692
  %128 = icmp eq <8 x i64> %wide.load153, splat (i64 9223372036854775807), !dbg !692
  %129 = icmp eq <8 x i64> %wide.load154, splat (i64 9223372036854775807), !dbg !692
  %130 = icmp eq <8 x i64> %wide.load155, splat (i64 9223372036854775807), !dbg !692
  %131 = icmp eq <8 x i64> %wide.load156, splat (i64 9223372036854775807), !dbg !692
  %132 = or <8 x i1> %124, %128, !dbg !692
  %133 = or <8 x i1> %125, %129, !dbg !692
  %134 = or <8 x i1> %126, %130, !dbg !692
  %135 = or <8 x i1> %127, %131, !dbg !692
  %136 = icmp slt <8 x i64> %wide.load, %wide.load153, !dbg !695
  %137 = icmp slt <8 x i64> %wide.load150, %wide.load154, !dbg !695
  %138 = icmp slt <8 x i64> %wide.load151, %wide.load155, !dbg !695
  %139 = icmp slt <8 x i64> %wide.load152, %wide.load156, !dbg !695
  %140 = or <8 x i1> %132, %115, !dbg !663
  %141 = zext <8 x i1> %136 to <8 x i8>, !dbg !696
  %142 = zext <8 x i1> %137 to <8 x i8>, !dbg !696
  %143 = zext <8 x i1> %138 to <8 x i8>, !dbg !696
  %144 = zext <8 x i1> %139 to <8 x i8>, !dbg !696
  %bools.i.sroa.0.0.vec.expand = shufflevector <8 x i8> %141, <8 x i8> poison, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !696
  %bools.i.sroa.0.8.vec.expand = shufflevector <8 x i8> %142, <8 x i8> poison, <64 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !696
  %bools.i.sroa.0.8.vecblend = shufflevector <64 x i8> %bools.i.sroa.0.0.vec.expand, <64 x i8> %bools.i.sroa.0.8.vec.expand, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 72, i32 73, i32 74, i32 75, i32 76, i32 77, i32 78, i32 79, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !696
  %bools.i.sroa.0.16.vec.expand = shufflevector <8 x i8> %143, <8 x i8> poison, <64 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !696
  %bools.i.sroa.0.16.vecblend = shufflevector <64 x i8> %bools.i.sroa.0.8.vecblend, <64 x i8> %bools.i.sroa.0.16.vec.expand, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 80, i32 81, i32 82, i32 83, i32 84, i32 85, i32 86, i32 87, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !696
  %bools.i.sroa.0.24.vec.expand = shufflevector <8 x i8> %144, <8 x i8> poison, <64 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !696
  %bools.i.sroa.0.24.vecblend = shufflevector <64 x i8> %bools.i.sroa.0.16.vecblend, <64 x i8> %bools.i.sroa.0.24.vec.expand, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 88, i32 89, i32 90, i32 91, i32 92, i32 93, i32 94, i32 95, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !696
  %145 = or disjoint i64 %offset2.i, 32
  %146 = getelementptr inbounds nuw i64, ptr %view.val.i.i.i4, i64 %145, !dbg !718
  %147 = getelementptr inbounds nuw i8, ptr %146, i64 64, !dbg !723
  %148 = getelementptr inbounds nuw i8, ptr %146, i64 128, !dbg !723
  %149 = getelementptr inbounds nuw i8, ptr %146, i64 192, !dbg !723
  %wide.load.1 = load <8 x i64>, ptr %146, align 8, !dbg !723, !noalias !728
  %wide.load150.1 = load <8 x i64>, ptr %147, align 8, !dbg !723, !noalias !728
  %wide.load151.1 = load <8 x i64>, ptr %148, align 8, !dbg !723, !noalias !728
  %wide.load152.1 = load <8 x i64>, ptr %149, align 8, !dbg !723, !noalias !728
  %150 = getelementptr inbounds nuw i64, ptr %view.val.i4.i.i15, i64 %145, !dbg !686
  %151 = getelementptr inbounds nuw i8, ptr %150, i64 64, !dbg !691
  %152 = getelementptr inbounds nuw i8, ptr %150, i64 128, !dbg !691
  %153 = getelementptr inbounds nuw i8, ptr %150, i64 192, !dbg !691
  %wide.load153.1 = load <8 x i64>, ptr %150, align 8, !dbg !691, !noalias !731
  %wide.load154.1 = load <8 x i64>, ptr %151, align 8, !dbg !691, !noalias !731
  %wide.load155.1 = load <8 x i64>, ptr %152, align 8, !dbg !691, !noalias !731
  %wide.load156.1 = load <8 x i64>, ptr %153, align 8, !dbg !691, !noalias !731
  %154 = icmp eq <8 x i64> %wide.load.1, splat (i64 9223372036854775807), !dbg !692
  %155 = icmp eq <8 x i64> %wide.load150.1, splat (i64 9223372036854775807), !dbg !692
  %156 = icmp eq <8 x i64> %wide.load151.1, splat (i64 9223372036854775807), !dbg !692
  %157 = icmp eq <8 x i64> %wide.load152.1, splat (i64 9223372036854775807), !dbg !692
  %158 = icmp eq <8 x i64> %wide.load153.1, splat (i64 9223372036854775807), !dbg !692
  %159 = icmp eq <8 x i64> %wide.load154.1, splat (i64 9223372036854775807), !dbg !692
  %160 = icmp eq <8 x i64> %wide.load155.1, splat (i64 9223372036854775807), !dbg !692
  %161 = icmp eq <8 x i64> %wide.load156.1, splat (i64 9223372036854775807), !dbg !692
  %162 = or <8 x i1> %154, %158, !dbg !692
  %163 = or <8 x i1> %155, %159, !dbg !692
  %164 = or <8 x i1> %156, %160, !dbg !692
  %165 = or <8 x i1> %157, %161, !dbg !692
  %166 = icmp slt <8 x i64> %wide.load.1, %wide.load153.1, !dbg !695
  %167 = icmp slt <8 x i64> %wide.load150.1, %wide.load154.1, !dbg !695
  %168 = icmp slt <8 x i64> %wide.load151.1, %wide.load155.1, !dbg !695
  %169 = icmp slt <8 x i64> %wide.load152.1, %wide.load156.1, !dbg !695
  %170 = or <8 x i1> %162, %140, !dbg !663
  %171 = or <8 x i1> %163, %133, !dbg !663
  %172 = or <8 x i1> %164, %134, !dbg !663
  %173 = or <8 x i1> %165, %135, !dbg !663
  %174 = zext <8 x i1> %166 to <8 x i8>, !dbg !696
  %175 = zext <8 x i1> %167 to <8 x i8>, !dbg !696
  %176 = zext <8 x i1> %168 to <8 x i8>, !dbg !696
  %177 = zext <8 x i1> %169 to <8 x i8>, !dbg !696
  %bools.i.sroa.0.32.vec.expand = shufflevector <8 x i8> %174, <8 x i8> poison, <64 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !696
  %bools.i.sroa.0.32.vecblend = shufflevector <64 x i8> %bools.i.sroa.0.24.vecblend, <64 x i8> %bools.i.sroa.0.32.vec.expand, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 96, i32 97, i32 98, i32 99, i32 100, i32 101, i32 102, i32 103, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !696
  %bools.i.sroa.0.40.vec.expand = shufflevector <8 x i8> %175, <8 x i8> poison, <64 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !696
  %bools.i.sroa.0.40.vecblend = shufflevector <64 x i8> %bools.i.sroa.0.32.vecblend, <64 x i8> %bools.i.sroa.0.40.vec.expand, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 104, i32 105, i32 106, i32 107, i32 108, i32 109, i32 110, i32 111, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !696
  %bools.i.sroa.0.48.vec.expand = shufflevector <8 x i8> %176, <8 x i8> poison, <64 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !696
  %bools.i.sroa.0.48.vecblend = shufflevector <64 x i8> %bools.i.sroa.0.40.vecblend, <64 x i8> %bools.i.sroa.0.48.vec.expand, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 112, i32 113, i32 114, i32 115, i32 116, i32 117, i32 118, i32 119, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !696
  %bools.i.sroa.0.56.vec.expand = shufflevector <8 x i8> %177, <8 x i8> poison, <64 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !696
  %bools.i.sroa.0.56.vecblend = shufflevector <64 x i8> %bools.i.sroa.0.48.vecblend, <64 x i8> %bools.i.sroa.0.56.vec.expand, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 120, i32 121, i32 122, i32 123, i32 124, i32 125, i32 126, i32 127>, !dbg !696
  %bin.rdx = or <8 x i1> %171, %170, !dbg !671
  %bin.rdx157 = or <8 x i1> %172, %bin.rdx, !dbg !671
  %bin.rdx158 = or <8 x i1> %173, %bin.rdx157, !dbg !671
  %178 = bitcast <8 x i1> %bin.rdx158 to i8, !dbg !671
  %179 = icmp ne i8 %178, 0, !dbg !671
  %_17.i.i = getelementptr inbounds nuw i8, ptr %iter.i.sroa.0.030, i64 8, !dbg !675
  %_9.0.i = add nuw nsw i64 %iter.i.sroa.7.029, 1, !dbg !700
  %.not.i.i.i = icmp ne <64 x i8> %bools.i.sroa.0.56.vecblend, zeroinitializer, !dbg !701
  store <64 x i1> %.not.i.i.i, ptr %iter.i.sroa.0.030, align 8, !dbg !677
  %_7.i.i = icmp eq ptr %_17.i.i, %_52.i, !dbg !642
  br i1 %_7.i.i, label %bb5.i.loopexit, label %bb4.i, !dbg !647

bb5.i.loopexit:                                   ; preds = %bb4.i, %bb4.i.us56, %bb4.i.us, %bb4.i.us.us.prol.loopexit, %bb4.i.us.us
  %.us-phi55.in = phi i1 [ %114, %bb4.i.us56 ], [ %69, %bb4.i.us ], [ %23, %bb4.i.us.us.prol.loopexit ], [ %23, %bb4.i.us.us ], [ %179, %bb4.i ]
  %.us-phi55 = zext i1 %.us-phi55.in to i8, !dbg !663
  store i8 %.us-phi55, ptr %_11.i, align 8, !alias.scope !659, !noalias !653
  br label %bb5.i, !dbg !733

bb5.i:                                            ; preds = %bb5.i.loopexit, %bb21.i
  %180 = icmp eq i64 %remainder.i, 0, !dbg !733
  br i1 %180, label %_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack23collect_bool_words_withNCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1g_7visitor5retry21ExecuteDenseWithRetryINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB35_EENtNtB3j_11row_visitor10RowVisitor19visit_deferred_boolB2Y_bKB35_NCINvXB43_B40_NtNtB1g_6row_fn5RowFn8dispatchB3e_E0NCB5V_s_0E0NCB3a_s_0B6H_E0NCINvB2_25collect_bool_words_avx512B17_E0EB43_.exit, label %bb12.i, !dbg !733

bb12.i:                                           ; preds = %bb5.i
  %181 = and i64 %len, -64, !dbg !734
  %_3.i.i.i = load i64, ptr %f, align 8, !range !356, !alias.scope !735, !noalias !740, !noundef !23
  %182 = trunc nuw i64 %_3.i.i.i to i1
  %_6.i.i = getelementptr inbounds nuw i8, ptr %f, i64 24
  %_3.i1.i.i = load i64, ptr %_6.i.i, align 8, !range !356, !alias.scope !745, !noalias !740, !noundef !23
  %183 = trunc nuw i64 %_3.i1.i.i to i1
  %_11.i.i = getelementptr inbounds nuw i8, ptr %f, i64 56
  %_11.i.i.promoted = load i8, ptr %_11.i.i, align 8, !alias.scope !748, !noalias !740
  %view.i.i.i = getelementptr inbounds nuw i8, ptr %f, i64 8
  %view.val.i.i.i = load ptr, ptr %view.i.i.i, align 8, !nonnull !23, !align !368
  %184 = getelementptr inbounds nuw i8, ptr %f, i64 16
  %view.val1.i.i.i = load i64, ptr %184, align 8
  %view.i3.i.i = getelementptr inbounds nuw i8, ptr %f, i64 32
  %view.val.i4.i.i = load ptr, ptr %view.i3.i.i, align 8, !nonnull !23, !align !368
  %185 = getelementptr inbounds nuw i8, ptr %f, i64 40
  %view.val1.i5.i.i = load i64, ptr %185, align 8
  %_6.i11.i.i.cast = inttoptr i64 %view.val1.i5.i.i to ptr
  br i1 %182, label %bb12.i.split.us, label %bb12.i.split

bb12.i.split.us:                                  ; preds = %bb12.i
  %186 = inttoptr i64 %view.val1.i.i.i to ptr
  %_0.sroa.0.0.i.i.i.us = load i64, ptr %186, align 8, !noalias !751, !noundef !23
  %187 = icmp eq i64 %_0.sroa.0.0.i.i.i.us, 9223372036854775807
  br i1 %183, label %iter.check398, label %iter.check333

iter.check333:                                    ; preds = %bb12.i.split.us
  %188 = trunc nuw i8 %_11.i.i.promoted to i1, !dbg !752
  %min.iters.check331 = icmp samesign ult i64 %remainder.i, 4, !dbg !763
  br i1 %min.iters.check331, label %bb8.i3.us.preheader, label %vector.main.loop.iter.check335, !dbg !763

vector.main.loop.iter.check335:                   ; preds = %iter.check333
  %min.iters.check334 = icmp samesign ult i64 %remainder.i, 16, !dbg !763
  br i1 %min.iters.check334, label %vec.epilog.ph369, label %vector.ph336, !dbg !763

vector.ph336:                                     ; preds = %vector.main.loop.iter.check335
  %n.mod.vf337 = and i64 %len, 12
  %n.vec338 = and i64 %len, 48
  %189 = insertelement <8 x i1> <i1 poison, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false>, i1 %188, i64 0
  %broadcast.splatinsert343 = insertelement <8 x i64> poison, i64 %_0.sroa.0.0.i.i.i.us, i64 0
  %broadcast.splat344 = shufflevector <8 x i64> %broadcast.splatinsert343, <8 x i64> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert345 = insertelement <8 x i1> poison, i1 %187, i64 0
  %broadcast.splat346 = shufflevector <8 x i1> %broadcast.splatinsert345, <8 x i1> poison, <8 x i32> zeroinitializer
  %invariant.gep576 = getelementptr i64, ptr %view.val.i4.i.i, i64 %181, !dbg !763
  br label %vector.body347, !dbg !763

vector.body347:                                   ; preds = %vector.body347, %vector.ph336
  %index348 = phi i64 [ 0, %vector.ph336 ], [ %index.next357, %vector.body347 ], !dbg !769
  %vec.phi349 = phi <8 x i1> [ %189, %vector.ph336 ], [ %198, %vector.body347 ]
  %vec.phi350 = phi <8 x i1> [ zeroinitializer, %vector.ph336 ], [ %199, %vector.body347 ]
  %vec.phi351 = phi <8 x i64> [ zeroinitializer, %vector.ph336 ], [ %204, %vector.body347 ]
  %vec.phi352 = phi <8 x i64> [ zeroinitializer, %vector.ph336 ], [ %205, %vector.body347 ]
  %vec.ind353 = phi <8 x i64> [ <i64 0, i64 1, i64 2, i64 3, i64 4, i64 5, i64 6, i64 7>, %vector.ph336 ], [ %vec.ind.next358, %vector.body347 ]
  %step.add354 = add <8 x i64> %vec.ind353, splat (i64 8)
  %190 = extractelement <8 x i64> %vec.ind353, i64 0
  tail call void @llvm.experimental.noalias.scope.decl(metadata !775), !dbg !776
  tail call void @llvm.experimental.noalias.scope.decl(metadata !777), !dbg !778, !noalias !780
  tail call void @llvm.experimental.noalias.scope.decl(metadata !781), !dbg !778, !noalias !780
  %gep577 = getelementptr i64, ptr %invariant.gep576, i64 %190, !dbg !782
  %191 = getelementptr inbounds nuw i8, ptr %gep577, i64 64, !dbg !787
  %wide.load355 = load <8 x i64>, ptr %gep577, align 8, !dbg !787, !noalias !788
  %wide.load356 = load <8 x i64>, ptr %191, align 8, !dbg !787, !noalias !788
  %192 = icmp eq <8 x i64> %wide.load355, splat (i64 9223372036854775807), !dbg !789
  %193 = icmp eq <8 x i64> %wide.load356, splat (i64 9223372036854775807), !dbg !789
  %194 = icmp slt <8 x i64> %broadcast.splat344, %wide.load355, !dbg !792
  %195 = icmp slt <8 x i64> %broadcast.splat344, %wide.load356, !dbg !792
  %196 = or <8 x i1> %192, %vec.phi349, !dbg !752
  %197 = or <8 x i1> %193, %vec.phi350, !dbg !752
  %198 = or <8 x i1> %196, %broadcast.splat346, !dbg !752
  %199 = or <8 x i1> %197, %broadcast.splat346, !dbg !752
  %200 = zext <8 x i1> %194 to <8 x i64>, !dbg !793
  %201 = zext <8 x i1> %195 to <8 x i64>, !dbg !793
  %202 = shl nuw <8 x i64> %200, %vec.ind353, !dbg !793
  %203 = shl nuw <8 x i64> %201, %step.add354, !dbg !793
  %204 = or <8 x i64> %202, %vec.phi351, !dbg !794
  %205 = or <8 x i64> %203, %vec.phi352, !dbg !794
  %index.next357 = add nuw i64 %index348, 16, !dbg !769
  %vec.ind.next358 = add <8 x i64> %vec.ind353, splat (i64 16)
  %206 = icmp eq i64 %index.next357, %n.vec338, !dbg !763
  br i1 %206, label %middle.block359, label %vector.body347, !dbg !763, !llvm.loop !795

middle.block359:                                  ; preds = %vector.body347
  %bin.rdx360 = or <8 x i1> %197, %198, !dbg !763
  %207 = bitcast <8 x i1> %bin.rdx360 to i8, !dbg !763
  %208 = icmp ne i8 %207, 0, !dbg !763
  %bin.rdx361 = or <8 x i64> %205, %204, !dbg !763
  %209 = tail call i64 @llvm.vector.reduce.or.v8i64(<8 x i64> %bin.rdx361), !dbg !763
  %cmp.n362 = icmp eq i64 %remainder.i, %n.vec338, !dbg !763
  br i1 %cmp.n362, label %_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack24collect_bool_word_scalarNCINvB2_23collect_bool_words_withNCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1O_7visitor5retry21ExecuteDenseWithRetryINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB3D_EENtNtB3R_11row_visitor10RowVisitor19visit_deferred_boolB3w_bKB3D_NCINvXB4B_B4y_NtNtB1O_6row_fn5RowFn8dispatchB3M_E0NCB6t_s_0E0NCB3I_s_0B7f_E0NCINvB2_25collect_bool_words_avx512B1F_E0E0EB4B_.exit, label %vec.epilog.iter.check367, !dbg !763

vec.epilog.iter.check367:                         ; preds = %middle.block359
  %min.epilog.iters.check368 = icmp eq i64 %n.mod.vf337, 0
  br i1 %min.epilog.iters.check368, label %bb8.i3.us.preheader, label %vec.epilog.ph369, !prof !382

vec.epilog.ph369:                                 ; preds = %vector.main.loop.iter.check335, %vec.epilog.iter.check367
  %bc.resume.val363 = phi i64 [ %n.vec338, %vec.epilog.iter.check367 ], [ 0, %vector.main.loop.iter.check335 ]
  %bc.merge.rdx364 = phi i1 [ %208, %vec.epilog.iter.check367 ], [ %188, %vector.main.loop.iter.check335 ]
  %bc.merge.rdx365 = phi i64 [ %209, %vec.epilog.iter.check367 ], [ 0, %vector.main.loop.iter.check335 ]
  %n.vec371 = and i64 %len, 60
  %210 = insertelement <4 x i1> <i1 poison, i1 false, i1 false, i1 false>, i1 %bc.merge.rdx364, i64 0
  %211 = insertelement <4 x i64> <i64 poison, i64 0, i64 0, i64 0>, i64 %bc.merge.rdx365, i64 0
  %broadcast.splatinsert376 = insertelement <4 x i64> poison, i64 %_0.sroa.0.0.i.i.i.us, i64 0
  %broadcast.splat377 = shufflevector <4 x i64> %broadcast.splatinsert376, <4 x i64> poison, <4 x i32> zeroinitializer
  %broadcast.splatinsert378 = insertelement <4 x i1> poison, i1 %187, i64 0
  %broadcast.splat379 = shufflevector <4 x i1> %broadcast.splatinsert378, <4 x i1> poison, <4 x i32> zeroinitializer
  %broadcast.splatinsert380 = insertelement <4 x i64> poison, i64 %bc.resume.val363, i64 0
  %broadcast.splat381 = shufflevector <4 x i64> %broadcast.splatinsert380, <4 x i64> poison, <4 x i32> zeroinitializer
  %induction382 = or disjoint <4 x i64> %broadcast.splat381, <i64 0, i64 1, i64 2, i64 3>
  %invariant.gep578 = getelementptr i64, ptr %view.val.i4.i.i, i64 %181
  br label %vec.epilog.vector.body383

vec.epilog.vector.body383:                        ; preds = %vec.epilog.vector.body383, %vec.epilog.ph369
  %index384 = phi i64 [ %bc.resume.val363, %vec.epilog.ph369 ], [ %index.next389, %vec.epilog.vector.body383 ], !dbg !769
  %vec.phi385 = phi <4 x i1> [ %210, %vec.epilog.ph369 ], [ %216, %vec.epilog.vector.body383 ]
  %vec.phi386 = phi <4 x i64> [ %211, %vec.epilog.ph369 ], [ %219, %vec.epilog.vector.body383 ]
  %vec.ind387 = phi <4 x i64> [ %induction382, %vec.epilog.ph369 ], [ %vec.ind.next390, %vec.epilog.vector.body383 ]
  %212 = extractelement <4 x i64> %vec.ind387, i64 0
  tail call void @llvm.experimental.noalias.scope.decl(metadata !775), !dbg !776
  tail call void @llvm.experimental.noalias.scope.decl(metadata !777), !dbg !778, !noalias !780
  tail call void @llvm.experimental.noalias.scope.decl(metadata !781), !dbg !778, !noalias !780
  %gep579 = getelementptr i64, ptr %invariant.gep578, i64 %212, !dbg !782
  %wide.load388 = load <4 x i64>, ptr %gep579, align 8, !dbg !787, !noalias !788
  %213 = icmp eq <4 x i64> %wide.load388, splat (i64 9223372036854775807), !dbg !789
  %214 = icmp slt <4 x i64> %broadcast.splat377, %wide.load388, !dbg !792
  %215 = or <4 x i1> %213, %vec.phi385, !dbg !752
  %216 = or <4 x i1> %215, %broadcast.splat379, !dbg !752
  %217 = zext <4 x i1> %214 to <4 x i64>, !dbg !793
  %218 = shl nuw <4 x i64> %217, %vec.ind387, !dbg !793
  %219 = or <4 x i64> %218, %vec.phi386, !dbg !794
  %index.next389 = add nuw i64 %index384, 4, !dbg !769
  %vec.ind.next390 = add nuw nsw <4 x i64> %vec.ind387, splat (i64 4)
  %220 = icmp eq i64 %index.next389, %n.vec371, !dbg !763
  br i1 %220, label %vec.epilog.middle.block391, label %vec.epilog.vector.body383, !dbg !763, !llvm.loop !796

vec.epilog.middle.block391:                       ; preds = %vec.epilog.vector.body383
  %221 = bitcast <4 x i1> %216 to i4, !dbg !763
  %222 = icmp ne i4 %221, 0, !dbg !763
  %223 = tail call i64 @llvm.vector.reduce.or.v4i64(<4 x i64> %219), !dbg !763
  %cmp.n392 = icmp eq i64 %remainder.i, %n.vec371, !dbg !763
  br i1 %cmp.n392, label %_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack24collect_bool_word_scalarNCINvB2_23collect_bool_words_withNCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1O_7visitor5retry21ExecuteDenseWithRetryINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB3D_EENtNtB3R_11row_visitor10RowVisitor19visit_deferred_boolB3w_bKB3D_NCINvXB4B_B4y_NtNtB1O_6row_fn5RowFn8dispatchB3M_E0NCB6t_s_0E0NCB3I_s_0B7f_E0NCINvB2_25collect_bool_words_avx512B1F_E0E0EB4B_.exit, label %bb8.i3.us.preheader, !dbg !763

bb8.i3.us.preheader:                              ; preds = %iter.check333, %vec.epilog.iter.check367, %vec.epilog.middle.block391
  %.ph = phi i1 [ %188, %iter.check333 ], [ %208, %vec.epilog.iter.check367 ], [ %222, %vec.epilog.middle.block391 ]
  %packed.sroa.0.0.i32.us.ph = phi i64 [ 0, %iter.check333 ], [ %209, %vec.epilog.iter.check367 ], [ %223, %vec.epilog.middle.block391 ]
  %iter.sroa.0.0.i31.us.ph = phi i64 [ 0, %iter.check333 ], [ %n.vec338, %vec.epilog.iter.check367 ], [ %n.vec371, %vec.epilog.middle.block391 ]
  br label %bb8.i3.us, !dbg !763

iter.check398:                                    ; preds = %bb12.i.split.us
  %_0.sroa.0.0.i9.i.i.us.us = load i64, ptr %_6.i11.i.i.cast, align 8, !noalias !788, !noundef !23
  %224 = icmp eq i64 %_0.sroa.0.0.i9.i.i.us.us, 9223372036854775807
  %_5.i.i.i.i.us.us = icmp slt i64 %_0.sroa.0.0.i.i.i.us, %_0.sroa.0.0.i9.i.i.us.us
  %_13.i.us.us = zext i1 %_5.i.i.i.i.us.us to i64
  %225 = trunc nuw i8 %_11.i.i.promoted to i1, !dbg !752
  %226 = or i1 %224, %225, !dbg !752
  %227 = or i1 %226, %187, !dbg !752
  %min.iters.check396 = icmp samesign ult i64 %remainder.i, 4, !dbg !763
  br i1 %min.iters.check396, label %bb8.i3.us.us.preheader, label %vector.main.loop.iter.check400, !dbg !763

vector.main.loop.iter.check400:                   ; preds = %iter.check398
  %min.iters.check399 = icmp samesign ult i64 %remainder.i, 16, !dbg !763
  br i1 %min.iters.check399, label %vec.epilog.ph422, label %vector.ph401, !dbg !763

vector.ph401:                                     ; preds = %vector.main.loop.iter.check400
  %n.mod.vf402 = and i64 %len, 12
  %n.vec403 = and i64 %len, 48
  %broadcast.splatinsert404 = insertelement <8 x i64> poison, i64 %_13.i.us.us, i64 0
  %broadcast.splat405 = shufflevector <8 x i64> %broadcast.splatinsert404, <8 x i64> poison, <8 x i32> zeroinitializer
  br label %vector.body406, !dbg !763

vector.body406:                                   ; preds = %vector.body406, %vector.ph401
  %index407 = phi i64 [ 0, %vector.ph401 ], [ %index.next412, %vector.body406 ], !dbg !769
  %vec.phi408 = phi <8 x i64> [ zeroinitializer, %vector.ph401 ], [ %230, %vector.body406 ]
  %vec.phi409 = phi <8 x i64> [ zeroinitializer, %vector.ph401 ], [ %231, %vector.body406 ]
  %vec.ind410 = phi <8 x i64> [ <i64 0, i64 1, i64 2, i64 3, i64 4, i64 5, i64 6, i64 7>, %vector.ph401 ], [ %vec.ind.next413, %vector.body406 ]
  %step.add411 = add <8 x i64> %vec.ind410, splat (i64 8)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !775), !dbg !776
  tail call void @llvm.experimental.noalias.scope.decl(metadata !777), !dbg !778, !noalias !780
  tail call void @llvm.experimental.noalias.scope.decl(metadata !781), !dbg !778, !noalias !780
  %228 = shl nuw <8 x i64> %broadcast.splat405, %vec.ind410, !dbg !793
  %229 = shl nuw <8 x i64> %broadcast.splat405, %step.add411, !dbg !793
  %230 = or <8 x i64> %228, %vec.phi408, !dbg !794
  %231 = or <8 x i64> %229, %vec.phi409, !dbg !794
  %index.next412 = add nuw i64 %index407, 16, !dbg !769
  %vec.ind.next413 = add <8 x i64> %vec.ind410, splat (i64 16)
  %232 = icmp eq i64 %index.next412, %n.vec403, !dbg !763
  br i1 %232, label %middle.block414, label %vector.body406, !dbg !763, !llvm.loop !797

middle.block414:                                  ; preds = %vector.body406
  %bin.rdx415 = or <8 x i64> %231, %230, !dbg !763
  %233 = tail call i64 @llvm.vector.reduce.or.v8i64(<8 x i64> %bin.rdx415), !dbg !763
  %cmp.n416 = icmp eq i64 %remainder.i, %n.vec403, !dbg !763
  br i1 %cmp.n416, label %_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack24collect_bool_word_scalarNCINvB2_23collect_bool_words_withNCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1O_7visitor5retry21ExecuteDenseWithRetryINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB3D_EENtNtB3R_11row_visitor10RowVisitor19visit_deferred_boolB3w_bKB3D_NCINvXB4B_B4y_NtNtB1O_6row_fn5RowFn8dispatchB3M_E0NCB6t_s_0E0NCB3I_s_0B7f_E0NCINvB2_25collect_bool_words_avx512B1F_E0E0EB4B_.exit, label %vec.epilog.iter.check420, !dbg !763

vec.epilog.iter.check420:                         ; preds = %middle.block414
  %min.epilog.iters.check421 = icmp eq i64 %n.mod.vf402, 0
  br i1 %min.epilog.iters.check421, label %bb8.i3.us.us.preheader, label %vec.epilog.ph422, !prof !382

vec.epilog.ph422:                                 ; preds = %vector.main.loop.iter.check400, %vec.epilog.iter.check420
  %bc.resume.val417 = phi i64 [ %n.vec403, %vec.epilog.iter.check420 ], [ 0, %vector.main.loop.iter.check400 ]
  %bc.merge.rdx418 = phi i64 [ %233, %vec.epilog.iter.check420 ], [ 0, %vector.main.loop.iter.check400 ]
  %n.vec424 = and i64 %len, 60
  %234 = insertelement <4 x i64> <i64 poison, i64 0, i64 0, i64 0>, i64 %bc.merge.rdx418, i64 0
  %broadcast.splatinsert425 = insertelement <4 x i64> poison, i64 %_13.i.us.us, i64 0
  %broadcast.splat426 = shufflevector <4 x i64> %broadcast.splatinsert425, <4 x i64> poison, <4 x i32> zeroinitializer
  %broadcast.splatinsert427 = insertelement <4 x i64> poison, i64 %bc.resume.val417, i64 0
  %broadcast.splat428 = shufflevector <4 x i64> %broadcast.splatinsert427, <4 x i64> poison, <4 x i32> zeroinitializer
  %induction429 = or disjoint <4 x i64> %broadcast.splat428, <i64 0, i64 1, i64 2, i64 3>
  br label %vec.epilog.vector.body430

vec.epilog.vector.body430:                        ; preds = %vec.epilog.vector.body430, %vec.epilog.ph422
  %index431 = phi i64 [ %bc.resume.val417, %vec.epilog.ph422 ], [ %index.next434, %vec.epilog.vector.body430 ], !dbg !769
  %vec.phi432 = phi <4 x i64> [ %234, %vec.epilog.ph422 ], [ %236, %vec.epilog.vector.body430 ]
  %vec.ind433 = phi <4 x i64> [ %induction429, %vec.epilog.ph422 ], [ %vec.ind.next435, %vec.epilog.vector.body430 ]
  tail call void @llvm.experimental.noalias.scope.decl(metadata !775), !dbg !776
  tail call void @llvm.experimental.noalias.scope.decl(metadata !777), !dbg !778, !noalias !780
  tail call void @llvm.experimental.noalias.scope.decl(metadata !781), !dbg !778, !noalias !780
  %235 = shl nuw <4 x i64> %broadcast.splat426, %vec.ind433, !dbg !793
  %236 = or <4 x i64> %235, %vec.phi432, !dbg !794
  %index.next434 = add nuw i64 %index431, 4, !dbg !769
  %vec.ind.next435 = add nuw nsw <4 x i64> %vec.ind433, splat (i64 4)
  %237 = icmp eq i64 %index.next434, %n.vec424, !dbg !763
  br i1 %237, label %vec.epilog.middle.block436, label %vec.epilog.vector.body430, !dbg !763, !llvm.loop !798

vec.epilog.middle.block436:                       ; preds = %vec.epilog.vector.body430
  %238 = tail call i64 @llvm.vector.reduce.or.v4i64(<4 x i64> %236), !dbg !763
  %cmp.n437 = icmp eq i64 %remainder.i, %n.vec424, !dbg !763
  br i1 %cmp.n437, label %_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack24collect_bool_word_scalarNCINvB2_23collect_bool_words_withNCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1O_7visitor5retry21ExecuteDenseWithRetryINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB3D_EENtNtB3R_11row_visitor10RowVisitor19visit_deferred_boolB3w_bKB3D_NCINvXB4B_B4y_NtNtB1O_6row_fn5RowFn8dispatchB3M_E0NCB6t_s_0E0NCB3I_s_0B7f_E0NCINvB2_25collect_bool_words_avx512B1F_E0E0EB4B_.exit, label %bb8.i3.us.us.preheader, !dbg !763

bb8.i3.us.us.preheader:                           ; preds = %iter.check398, %vec.epilog.iter.check420, %vec.epilog.middle.block436
  %packed.sroa.0.0.i32.us.us.ph = phi i64 [ 0, %iter.check398 ], [ %233, %vec.epilog.iter.check420 ], [ %238, %vec.epilog.middle.block436 ]
  %iter.sroa.0.0.i31.us.us.ph = phi i64 [ 0, %iter.check398 ], [ %n.vec403, %vec.epilog.iter.check420 ], [ %n.vec424, %vec.epilog.middle.block436 ]
  br label %bb8.i3.us.us, !dbg !763

bb8.i3.us.us:                                     ; preds = %bb8.i3.us.us.preheader, %bb8.i3.us.us
  %packed.sroa.0.0.i32.us.us = phi i64 [ %240, %bb8.i3.us.us ], [ %packed.sroa.0.0.i32.us.us.ph, %bb8.i3.us.us.preheader ]
  %iter.sroa.0.0.i31.us.us = phi i64 [ %239, %bb8.i3.us.us ], [ %iter.sroa.0.0.i31.us.us.ph, %bb8.i3.us.us.preheader ]
  tail call void @llvm.experimental.noalias.scope.decl(metadata !775), !dbg !776
  tail call void @llvm.experimental.noalias.scope.decl(metadata !777), !dbg !778, !noalias !780
  tail call void @llvm.experimental.noalias.scope.decl(metadata !781), !dbg !778, !noalias !780
  %239 = add nuw nsw i64 %iter.sroa.0.0.i31.us.us, 1, !dbg !769
  %_12.i.us.us = shl nuw i64 %_13.i.us.us, %iter.sroa.0.0.i31.us.us, !dbg !793
  %240 = or i64 %_12.i.us.us, %packed.sroa.0.0.i32.us.us, !dbg !794
  %exitcond.not.us.us = icmp eq i64 %239, %remainder.i, !dbg !799
  br i1 %exitcond.not.us.us, label %_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack24collect_bool_word_scalarNCINvB2_23collect_bool_words_withNCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1O_7visitor5retry21ExecuteDenseWithRetryINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB3D_EENtNtB3R_11row_visitor10RowVisitor19visit_deferred_boolB3w_bKB3D_NCINvXB4B_B4y_NtNtB1O_6row_fn5RowFn8dispatchB3M_E0NCB6t_s_0E0NCB3I_s_0B7f_E0NCINvB2_25collect_bool_words_avx512B1F_E0E0EB4B_.exit, label %bb8.i3.us.us, !dbg !763, !llvm.loop !802

bb8.i3.us:                                        ; preds = %bb8.i3.us.preheader, %bb8.i3.us
  %241 = phi i1 [ %244, %bb8.i3.us ], [ %.ph, %bb8.i3.us.preheader ]
  %packed.sroa.0.0.i32.us = phi i64 [ %246, %bb8.i3.us ], [ %packed.sroa.0.0.i32.us.ph, %bb8.i3.us.preheader ]
  %iter.sroa.0.0.i31.us = phi i64 [ %245, %bb8.i3.us ], [ %iter.sroa.0.0.i31.us.ph, %bb8.i3.us.preheader ]
  %_4.i.us = add nuw nsw i64 %iter.sroa.0.0.i31.us, %181, !dbg !803
  tail call void @llvm.experimental.noalias.scope.decl(metadata !775), !dbg !776
  tail call void @llvm.experimental.noalias.scope.decl(metadata !777), !dbg !778, !noalias !780
  tail call void @llvm.experimental.noalias.scope.decl(metadata !781), !dbg !778, !noalias !780
  %_5.i.i6.i.i.us = icmp ult i64 %_4.i.us, %view.val1.i5.i.i, !dbg !804
  tail call void @llvm.assume(i1 %_5.i.i6.i.i.us), !dbg !805, !noalias !780
  %_4.i.i7.i.i.us = getelementptr inbounds nuw i64, ptr %view.val.i4.i.i, i64 %_4.i.us, !dbg !782
  %_0.sroa.0.0.i9.i.i.us = load i64, ptr %_4.i.i7.i.i.us, align 8, !dbg !787, !noalias !788, !noundef !23
  %242 = icmp eq i64 %_0.sroa.0.0.i9.i.i.us, 9223372036854775807, !dbg !789
  %_5.i.i.i.i.us = icmp slt i64 %_0.sroa.0.0.i.i.i.us, %_0.sroa.0.0.i9.i.i.us, !dbg !792
  %243 = or i1 %242, %241, !dbg !752
  %244 = or i1 %243, %187, !dbg !752
  %245 = add nuw nsw i64 %iter.sroa.0.0.i31.us, 1, !dbg !769
  %_13.i.us = zext i1 %_5.i.i.i.i.us to i64, !dbg !793
  %_12.i.us = shl nuw i64 %_13.i.us, %iter.sroa.0.0.i31.us, !dbg !793
  %246 = or i64 %_12.i.us, %packed.sroa.0.0.i32.us, !dbg !794
  %exitcond.not.us = icmp eq i64 %245, %remainder.i, !dbg !799
  br i1 %exitcond.not.us, label %_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack24collect_bool_word_scalarNCINvB2_23collect_bool_words_withNCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1O_7visitor5retry21ExecuteDenseWithRetryINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB3D_EENtNtB3R_11row_visitor10RowVisitor19visit_deferred_boolB3w_bKB3D_NCINvXB4B_B4y_NtNtB1O_6row_fn5RowFn8dispatchB3M_E0NCB6t_s_0E0NCB3I_s_0B7f_E0NCINvB2_25collect_bool_words_avx512B1F_E0E0EB4B_.exit, label %bb8.i3.us, !dbg !763, !llvm.loop !806

bb12.i.split:                                     ; preds = %bb12.i
  br i1 %183, label %iter.check268, label %iter.check

iter.check:                                       ; preds = %bb12.i.split
  %247 = trunc nuw i8 %_11.i.i.promoted to i1, !dbg !752
  %min.iters.check = icmp samesign ult i64 %remainder.i, 4, !dbg !763
  br i1 %min.iters.check, label %bb8.i3.preheader, label %vector.main.loop.iter.check, !dbg !763

vector.main.loop.iter.check:                      ; preds = %iter.check
  %min.iters.check218 = icmp samesign ult i64 %remainder.i, 16, !dbg !763
  br i1 %min.iters.check218, label %vec.epilog.ph, label %vector.ph219, !dbg !763

vector.ph219:                                     ; preds = %vector.main.loop.iter.check
  %n.mod.vf = and i64 %len, 12
  %n.vec = and i64 %len, 48
  %248 = insertelement <8 x i1> <i1 poison, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false>, i1 %247, i64 0
  br label %vector.body226, !dbg !763

vector.body226:                                   ; preds = %vector.body226, %vector.ph219
  %index227 = phi i64 [ 0, %vector.ph219 ], [ %index.next238, %vector.body226 ], !dbg !769
  %vec.phi228 = phi <8 x i1> [ %248, %vector.ph219 ], [ %263, %vector.body226 ]
  %vec.phi229 = phi <8 x i1> [ zeroinitializer, %vector.ph219 ], [ %264, %vector.body226 ]
  %vec.phi230 = phi <8 x i64> [ zeroinitializer, %vector.ph219 ], [ %269, %vector.body226 ]
  %vec.phi231 = phi <8 x i64> [ zeroinitializer, %vector.ph219 ], [ %270, %vector.body226 ]
  %vec.ind232 = phi <8 x i64> [ <i64 0, i64 1, i64 2, i64 3, i64 4, i64 5, i64 6, i64 7>, %vector.ph219 ], [ %vec.ind.next239, %vector.body226 ]
  %step.add233 = add <8 x i64> %vec.ind232, splat (i64 8)
  %249 = extractelement <8 x i64> %vec.ind232, i64 0
  %250 = add nuw nsw i64 %249, %181
  tail call void @llvm.experimental.noalias.scope.decl(metadata !775), !dbg !776
  tail call void @llvm.experimental.noalias.scope.decl(metadata !777), !dbg !778, !noalias !780
  %251 = getelementptr inbounds nuw i64, ptr %view.val.i.i.i, i64 %250, !dbg !807
  %252 = getelementptr inbounds nuw i8, ptr %251, i64 64, !dbg !812
  %wide.load234 = load <8 x i64>, ptr %251, align 8, !dbg !812, !noalias !751
  %wide.load235 = load <8 x i64>, ptr %252, align 8, !dbg !812, !noalias !751
  tail call void @llvm.experimental.noalias.scope.decl(metadata !781), !dbg !778, !noalias !780
  %253 = getelementptr inbounds nuw i64, ptr %view.val.i4.i.i, i64 %250, !dbg !782
  %254 = getelementptr inbounds nuw i8, ptr %253, i64 64, !dbg !787
  %wide.load236 = load <8 x i64>, ptr %253, align 8, !dbg !787, !noalias !788
  %wide.load237 = load <8 x i64>, ptr %254, align 8, !dbg !787, !noalias !788
  %255 = icmp eq <8 x i64> %wide.load234, splat (i64 9223372036854775807), !dbg !789
  %256 = icmp eq <8 x i64> %wide.load235, splat (i64 9223372036854775807), !dbg !789
  %257 = icmp eq <8 x i64> %wide.load236, splat (i64 9223372036854775807), !dbg !789
  %258 = icmp eq <8 x i64> %wide.load237, splat (i64 9223372036854775807), !dbg !789
  %259 = or <8 x i1> %255, %257, !dbg !789
  %260 = or <8 x i1> %256, %258, !dbg !789
  %261 = icmp slt <8 x i64> %wide.load234, %wide.load236, !dbg !792
  %262 = icmp slt <8 x i64> %wide.load235, %wide.load237, !dbg !792
  %263 = or <8 x i1> %259, %vec.phi228, !dbg !752
  %264 = or <8 x i1> %260, %vec.phi229, !dbg !752
  %265 = zext <8 x i1> %261 to <8 x i64>, !dbg !793
  %266 = zext <8 x i1> %262 to <8 x i64>, !dbg !793
  %267 = shl nuw <8 x i64> %265, %vec.ind232, !dbg !793
  %268 = shl nuw <8 x i64> %266, %step.add233, !dbg !793
  %269 = or <8 x i64> %267, %vec.phi230, !dbg !794
  %270 = or <8 x i64> %268, %vec.phi231, !dbg !794
  %index.next238 = add nuw i64 %index227, 16, !dbg !769
  %vec.ind.next239 = add <8 x i64> %vec.ind232, splat (i64 16)
  %271 = icmp eq i64 %index.next238, %n.vec, !dbg !763
  br i1 %271, label %middle.block240, label %vector.body226, !dbg !763, !llvm.loop !813

middle.block240:                                  ; preds = %vector.body226
  %bin.rdx241 = or <8 x i1> %264, %263, !dbg !763
  %272 = bitcast <8 x i1> %bin.rdx241 to i8, !dbg !763
  %273 = icmp ne i8 %272, 0, !dbg !763
  %bin.rdx242 = or <8 x i64> %270, %269, !dbg !763
  %274 = tail call i64 @llvm.vector.reduce.or.v8i64(<8 x i64> %bin.rdx242), !dbg !763
  %cmp.n = icmp eq i64 %remainder.i, %n.vec, !dbg !763
  br i1 %cmp.n, label %_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack24collect_bool_word_scalarNCINvB2_23collect_bool_words_withNCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1O_7visitor5retry21ExecuteDenseWithRetryINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB3D_EENtNtB3R_11row_visitor10RowVisitor19visit_deferred_boolB3w_bKB3D_NCINvXB4B_B4y_NtNtB1O_6row_fn5RowFn8dispatchB3M_E0NCB6t_s_0E0NCB3I_s_0B7f_E0NCINvB2_25collect_bool_words_avx512B1F_E0E0EB4B_.exit, label %vec.epilog.iter.check, !dbg !763

vec.epilog.iter.check:                            ; preds = %middle.block240
  %min.epilog.iters.check = icmp eq i64 %n.mod.vf, 0
  br i1 %min.epilog.iters.check, label %bb8.i3.preheader, label %vec.epilog.ph, !prof !382

vec.epilog.ph:                                    ; preds = %vector.main.loop.iter.check, %vec.epilog.iter.check
  %bc.resume.val = phi i64 [ %n.vec, %vec.epilog.iter.check ], [ 0, %vector.main.loop.iter.check ]
  %bc.merge.rdx = phi i1 [ %273, %vec.epilog.iter.check ], [ %247, %vector.main.loop.iter.check ]
  %bc.merge.rdx243 = phi i64 [ %274, %vec.epilog.iter.check ], [ 0, %vector.main.loop.iter.check ]
  %n.vec245 = and i64 %len, 60
  %275 = insertelement <4 x i1> <i1 poison, i1 false, i1 false, i1 false>, i1 %bc.merge.rdx, i64 0
  %276 = insertelement <4 x i64> <i64 poison, i64 0, i64 0, i64 0>, i64 %bc.merge.rdx243, i64 0
  %broadcast.splatinsert252 = insertelement <4 x i64> poison, i64 %bc.resume.val, i64 0
  %broadcast.splat253 = shufflevector <4 x i64> %broadcast.splatinsert252, <4 x i64> poison, <4 x i32> zeroinitializer
  %induction = or disjoint <4 x i64> %broadcast.splat253, <i64 0, i64 1, i64 2, i64 3>
  br label %vec.epilog.vector.body

vec.epilog.vector.body:                           ; preds = %vec.epilog.vector.body, %vec.epilog.ph
  %index254 = phi i64 [ %bc.resume.val, %vec.epilog.ph ], [ %index.next260, %vec.epilog.vector.body ], !dbg !769
  %vec.phi255 = phi <4 x i1> [ %275, %vec.epilog.ph ], [ %285, %vec.epilog.vector.body ]
  %vec.phi256 = phi <4 x i64> [ %276, %vec.epilog.ph ], [ %288, %vec.epilog.vector.body ]
  %vec.ind257 = phi <4 x i64> [ %induction, %vec.epilog.ph ], [ %vec.ind.next261, %vec.epilog.vector.body ]
  %277 = extractelement <4 x i64> %vec.ind257, i64 0
  %278 = add nuw nsw i64 %277, %181
  tail call void @llvm.experimental.noalias.scope.decl(metadata !775), !dbg !776
  tail call void @llvm.experimental.noalias.scope.decl(metadata !777), !dbg !778, !noalias !780
  %279 = getelementptr inbounds nuw i64, ptr %view.val.i.i.i, i64 %278, !dbg !807
  %wide.load258 = load <4 x i64>, ptr %279, align 8, !dbg !812, !noalias !751
  tail call void @llvm.experimental.noalias.scope.decl(metadata !781), !dbg !778, !noalias !780
  %280 = getelementptr inbounds nuw i64, ptr %view.val.i4.i.i, i64 %278, !dbg !782
  %wide.load259 = load <4 x i64>, ptr %280, align 8, !dbg !787, !noalias !788
  %281 = icmp eq <4 x i64> %wide.load258, splat (i64 9223372036854775807), !dbg !789
  %282 = icmp eq <4 x i64> %wide.load259, splat (i64 9223372036854775807), !dbg !789
  %283 = or <4 x i1> %281, %282, !dbg !789
  %284 = icmp slt <4 x i64> %wide.load258, %wide.load259, !dbg !792
  %285 = or <4 x i1> %283, %vec.phi255, !dbg !752
  %286 = zext <4 x i1> %284 to <4 x i64>, !dbg !793
  %287 = shl nuw <4 x i64> %286, %vec.ind257, !dbg !793
  %288 = or <4 x i64> %287, %vec.phi256, !dbg !794
  %index.next260 = add nuw i64 %index254, 4, !dbg !769
  %vec.ind.next261 = add nuw nsw <4 x i64> %vec.ind257, splat (i64 4)
  %289 = icmp eq i64 %index.next260, %n.vec245, !dbg !763
  br i1 %289, label %vec.epilog.middle.block, label %vec.epilog.vector.body, !dbg !763, !llvm.loop !814

vec.epilog.middle.block:                          ; preds = %vec.epilog.vector.body
  %290 = bitcast <4 x i1> %285 to i4, !dbg !763
  %291 = icmp ne i4 %290, 0, !dbg !763
  %292 = tail call i64 @llvm.vector.reduce.or.v4i64(<4 x i64> %288), !dbg !763
  %cmp.n262 = icmp eq i64 %remainder.i, %n.vec245, !dbg !763
  br i1 %cmp.n262, label %_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack24collect_bool_word_scalarNCINvB2_23collect_bool_words_withNCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1O_7visitor5retry21ExecuteDenseWithRetryINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB3D_EENtNtB3R_11row_visitor10RowVisitor19visit_deferred_boolB3w_bKB3D_NCINvXB4B_B4y_NtNtB1O_6row_fn5RowFn8dispatchB3M_E0NCB6t_s_0E0NCB3I_s_0B7f_E0NCINvB2_25collect_bool_words_avx512B1F_E0E0EB4B_.exit, label %bb8.i3.preheader, !dbg !763

bb8.i3.preheader:                                 ; preds = %iter.check, %vec.epilog.iter.check, %vec.epilog.middle.block
  %.ph463 = phi i1 [ %247, %iter.check ], [ %273, %vec.epilog.iter.check ], [ %291, %vec.epilog.middle.block ]
  %packed.sroa.0.0.i32.ph = phi i64 [ 0, %iter.check ], [ %274, %vec.epilog.iter.check ], [ %292, %vec.epilog.middle.block ]
  %iter.sroa.0.0.i31.ph = phi i64 [ 0, %iter.check ], [ %n.vec, %vec.epilog.iter.check ], [ %n.vec245, %vec.epilog.middle.block ]
  br label %bb8.i3, !dbg !763

iter.check268:                                    ; preds = %bb12.i.split
  %_0.sroa.0.0.i9.i.i.us79 = load i64, ptr %_6.i11.i.i.cast, align 8, !noalias !788, !noundef !23
  %293 = icmp eq i64 %_0.sroa.0.0.i9.i.i.us79, 9223372036854775807
  %294 = trunc nuw i8 %_11.i.i.promoted to i1, !dbg !752
  %min.iters.check266 = icmp samesign ult i64 %remainder.i, 4, !dbg !763
  br i1 %min.iters.check266, label %bb8.i3.us71.preheader, label %vector.main.loop.iter.check270, !dbg !763

vector.main.loop.iter.check270:                   ; preds = %iter.check268
  %min.iters.check269 = icmp samesign ult i64 %remainder.i, 16, !dbg !763
  br i1 %min.iters.check269, label %vec.epilog.ph304, label %vector.ph271, !dbg !763

vector.ph271:                                     ; preds = %vector.main.loop.iter.check270
  %n.mod.vf272 = and i64 %len, 12
  %n.vec273 = and i64 %len, 48
  %295 = insertelement <8 x i1> <i1 poison, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false>, i1 %294, i64 0
  %broadcast.splatinsert274 = insertelement <8 x i64> poison, i64 %_0.sroa.0.0.i9.i.i.us79, i64 0
  %broadcast.splat275 = shufflevector <8 x i64> %broadcast.splatinsert274, <8 x i64> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert276 = insertelement <8 x i1> poison, i1 %293, i64 0
  %broadcast.splat277 = shufflevector <8 x i1> %broadcast.splatinsert276, <8 x i1> poison, <8 x i32> zeroinitializer
  %invariant.gep = getelementptr i64, ptr %view.val.i.i.i, i64 %181, !dbg !763
  br label %vector.body282, !dbg !763

vector.body282:                                   ; preds = %vector.body282, %vector.ph271
  %index283 = phi i64 [ 0, %vector.ph271 ], [ %index.next292, %vector.body282 ], !dbg !769
  %vec.phi284 = phi <8 x i1> [ %295, %vector.ph271 ], [ %303, %vector.body282 ]
  %vec.phi285 = phi <8 x i1> [ zeroinitializer, %vector.ph271 ], [ %305, %vector.body282 ]
  %vec.phi286 = phi <8 x i64> [ zeroinitializer, %vector.ph271 ], [ %310, %vector.body282 ]
  %vec.phi287 = phi <8 x i64> [ zeroinitializer, %vector.ph271 ], [ %311, %vector.body282 ]
  %vec.ind288 = phi <8 x i64> [ <i64 0, i64 1, i64 2, i64 3, i64 4, i64 5, i64 6, i64 7>, %vector.ph271 ], [ %vec.ind.next293, %vector.body282 ]
  %step.add289 = add <8 x i64> %vec.ind288, splat (i64 8)
  %296 = extractelement <8 x i64> %vec.ind288, i64 0
  tail call void @llvm.experimental.noalias.scope.decl(metadata !775), !dbg !776
  tail call void @llvm.experimental.noalias.scope.decl(metadata !777), !dbg !778, !noalias !780
  %gep = getelementptr i64, ptr %invariant.gep, i64 %296, !dbg !807
  %297 = getelementptr inbounds nuw i8, ptr %gep, i64 64, !dbg !812
  %wide.load290 = load <8 x i64>, ptr %gep, align 8, !dbg !812, !noalias !751
  %wide.load291 = load <8 x i64>, ptr %297, align 8, !dbg !812, !noalias !751
  tail call void @llvm.experimental.noalias.scope.decl(metadata !781), !dbg !778, !noalias !780
  %298 = icmp eq <8 x i64> %wide.load290, splat (i64 9223372036854775807), !dbg !789
  %299 = icmp eq <8 x i64> %wide.load291, splat (i64 9223372036854775807), !dbg !789
  %300 = icmp slt <8 x i64> %wide.load290, %broadcast.splat275, !dbg !792
  %301 = icmp slt <8 x i64> %wide.load291, %broadcast.splat275, !dbg !792
  %302 = or <8 x i1> %298, %vec.phi284, !dbg !752
  %303 = or <8 x i1> %302, %broadcast.splat277, !dbg !752
  %304 = or <8 x i1> %299, %vec.phi285, !dbg !752
  %305 = or <8 x i1> %304, %broadcast.splat277, !dbg !752
  %306 = zext <8 x i1> %300 to <8 x i64>, !dbg !793
  %307 = zext <8 x i1> %301 to <8 x i64>, !dbg !793
  %308 = shl nuw <8 x i64> %306, %vec.ind288, !dbg !793
  %309 = shl nuw <8 x i64> %307, %step.add289, !dbg !793
  %310 = or <8 x i64> %308, %vec.phi286, !dbg !794
  %311 = or <8 x i64> %309, %vec.phi287, !dbg !794
  %index.next292 = add nuw i64 %index283, 16, !dbg !769
  %vec.ind.next293 = add <8 x i64> %vec.ind288, splat (i64 16)
  %312 = icmp eq i64 %index.next292, %n.vec273, !dbg !763
  br i1 %312, label %middle.block294, label %vector.body282, !dbg !763, !llvm.loop !815

middle.block294:                                  ; preds = %vector.body282
  %bin.rdx295 = or <8 x i1> %304, %303, !dbg !763
  %313 = bitcast <8 x i1> %bin.rdx295 to i8, !dbg !763
  %314 = icmp ne i8 %313, 0, !dbg !763
  %bin.rdx296 = or <8 x i64> %311, %310, !dbg !763
  %315 = tail call i64 @llvm.vector.reduce.or.v8i64(<8 x i64> %bin.rdx296), !dbg !763
  %cmp.n297 = icmp eq i64 %remainder.i, %n.vec273, !dbg !763
  br i1 %cmp.n297, label %_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack24collect_bool_word_scalarNCINvB2_23collect_bool_words_withNCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1O_7visitor5retry21ExecuteDenseWithRetryINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB3D_EENtNtB3R_11row_visitor10RowVisitor19visit_deferred_boolB3w_bKB3D_NCINvXB4B_B4y_NtNtB1O_6row_fn5RowFn8dispatchB3M_E0NCB6t_s_0E0NCB3I_s_0B7f_E0NCINvB2_25collect_bool_words_avx512B1F_E0E0EB4B_.exit, label %vec.epilog.iter.check302, !dbg !763

vec.epilog.iter.check302:                         ; preds = %middle.block294
  %min.epilog.iters.check303 = icmp eq i64 %n.mod.vf272, 0
  br i1 %min.epilog.iters.check303, label %bb8.i3.us71.preheader, label %vec.epilog.ph304, !prof !382

vec.epilog.ph304:                                 ; preds = %vector.main.loop.iter.check270, %vec.epilog.iter.check302
  %bc.resume.val298 = phi i64 [ %n.vec273, %vec.epilog.iter.check302 ], [ 0, %vector.main.loop.iter.check270 ]
  %bc.merge.rdx299 = phi i1 [ %314, %vec.epilog.iter.check302 ], [ %294, %vector.main.loop.iter.check270 ]
  %bc.merge.rdx300 = phi i64 [ %315, %vec.epilog.iter.check302 ], [ 0, %vector.main.loop.iter.check270 ]
  %n.vec306 = and i64 %len, 60
  %316 = insertelement <4 x i1> <i1 poison, i1 false, i1 false, i1 false>, i1 %bc.merge.rdx299, i64 0
  %317 = insertelement <4 x i64> <i64 poison, i64 0, i64 0, i64 0>, i64 %bc.merge.rdx300, i64 0
  %broadcast.splatinsert307 = insertelement <4 x i64> poison, i64 %_0.sroa.0.0.i9.i.i.us79, i64 0
  %broadcast.splat308 = shufflevector <4 x i64> %broadcast.splatinsert307, <4 x i64> poison, <4 x i32> zeroinitializer
  %broadcast.splatinsert309 = insertelement <4 x i1> poison, i1 %293, i64 0
  %broadcast.splat310 = shufflevector <4 x i1> %broadcast.splatinsert309, <4 x i1> poison, <4 x i32> zeroinitializer
  %broadcast.splatinsert315 = insertelement <4 x i64> poison, i64 %bc.resume.val298, i64 0
  %broadcast.splat316 = shufflevector <4 x i64> %broadcast.splatinsert315, <4 x i64> poison, <4 x i32> zeroinitializer
  %induction317 = or disjoint <4 x i64> %broadcast.splat316, <i64 0, i64 1, i64 2, i64 3>
  %invariant.gep574 = getelementptr i64, ptr %view.val.i.i.i, i64 %181
  br label %vec.epilog.vector.body318

vec.epilog.vector.body318:                        ; preds = %vec.epilog.vector.body318, %vec.epilog.ph304
  %index319 = phi i64 [ %bc.resume.val298, %vec.epilog.ph304 ], [ %index.next324, %vec.epilog.vector.body318 ], !dbg !769
  %vec.phi320 = phi <4 x i1> [ %316, %vec.epilog.ph304 ], [ %322, %vec.epilog.vector.body318 ]
  %vec.phi321 = phi <4 x i64> [ %317, %vec.epilog.ph304 ], [ %325, %vec.epilog.vector.body318 ]
  %vec.ind322 = phi <4 x i64> [ %induction317, %vec.epilog.ph304 ], [ %vec.ind.next325, %vec.epilog.vector.body318 ]
  %318 = extractelement <4 x i64> %vec.ind322, i64 0
  tail call void @llvm.experimental.noalias.scope.decl(metadata !775), !dbg !776
  tail call void @llvm.experimental.noalias.scope.decl(metadata !777), !dbg !778, !noalias !780
  %gep575 = getelementptr i64, ptr %invariant.gep574, i64 %318, !dbg !807
  %wide.load323 = load <4 x i64>, ptr %gep575, align 8, !dbg !812, !noalias !751
  tail call void @llvm.experimental.noalias.scope.decl(metadata !781), !dbg !778, !noalias !780
  %319 = icmp eq <4 x i64> %wide.load323, splat (i64 9223372036854775807), !dbg !789
  %320 = or <4 x i1> %319, %broadcast.splat310, !dbg !789
  %321 = icmp slt <4 x i64> %wide.load323, %broadcast.splat308, !dbg !792
  %322 = or <4 x i1> %320, %vec.phi320, !dbg !752
  %323 = zext <4 x i1> %321 to <4 x i64>, !dbg !793
  %324 = shl nuw <4 x i64> %323, %vec.ind322, !dbg !793
  %325 = or <4 x i64> %324, %vec.phi321, !dbg !794
  %index.next324 = add nuw i64 %index319, 4, !dbg !769
  %vec.ind.next325 = add nuw nsw <4 x i64> %vec.ind322, splat (i64 4)
  %326 = icmp eq i64 %index.next324, %n.vec306, !dbg !763
  br i1 %326, label %vec.epilog.middle.block326, label %vec.epilog.vector.body318, !dbg !763, !llvm.loop !816

vec.epilog.middle.block326:                       ; preds = %vec.epilog.vector.body318
  %327 = bitcast <4 x i1> %322 to i4, !dbg !763
  %328 = icmp ne i4 %327, 0, !dbg !763
  %329 = tail call i64 @llvm.vector.reduce.or.v4i64(<4 x i64> %325), !dbg !763
  %cmp.n327 = icmp eq i64 %remainder.i, %n.vec306, !dbg !763
  br i1 %cmp.n327, label %_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack24collect_bool_word_scalarNCINvB2_23collect_bool_words_withNCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1O_7visitor5retry21ExecuteDenseWithRetryINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB3D_EENtNtB3R_11row_visitor10RowVisitor19visit_deferred_boolB3w_bKB3D_NCINvXB4B_B4y_NtNtB1O_6row_fn5RowFn8dispatchB3M_E0NCB6t_s_0E0NCB3I_s_0B7f_E0NCINvB2_25collect_bool_words_avx512B1F_E0E0EB4B_.exit, label %bb8.i3.us71.preheader, !dbg !763

bb8.i3.us71.preheader:                            ; preds = %iter.check268, %vec.epilog.iter.check302, %vec.epilog.middle.block326
  %.ph453 = phi i1 [ %294, %iter.check268 ], [ %314, %vec.epilog.iter.check302 ], [ %328, %vec.epilog.middle.block326 ]
  %packed.sroa.0.0.i32.us72.ph = phi i64 [ 0, %iter.check268 ], [ %315, %vec.epilog.iter.check302 ], [ %329, %vec.epilog.middle.block326 ]
  %iter.sroa.0.0.i31.us73.ph = phi i64 [ 0, %iter.check268 ], [ %n.vec273, %vec.epilog.iter.check302 ], [ %n.vec306, %vec.epilog.middle.block326 ]
  br label %bb8.i3.us71, !dbg !763

bb8.i3.us71:                                      ; preds = %bb8.i3.us71.preheader, %bb8.i3.us71
  %330 = phi i1 [ %333, %bb8.i3.us71 ], [ %.ph453, %bb8.i3.us71.preheader ]
  %packed.sroa.0.0.i32.us72 = phi i64 [ %335, %bb8.i3.us71 ], [ %packed.sroa.0.0.i32.us72.ph, %bb8.i3.us71.preheader ]
  %iter.sroa.0.0.i31.us73 = phi i64 [ %334, %bb8.i3.us71 ], [ %iter.sroa.0.0.i31.us73.ph, %bb8.i3.us71.preheader ]
  %_4.i.us74 = add nuw nsw i64 %iter.sroa.0.0.i31.us73, %181, !dbg !803
  tail call void @llvm.experimental.noalias.scope.decl(metadata !775), !dbg !776
  tail call void @llvm.experimental.noalias.scope.decl(metadata !777), !dbg !778, !noalias !780
  %_5.i.i.i1.i.us = icmp ult i64 %_4.i.us74, %view.val1.i.i.i, !dbg !817
  tail call void @llvm.assume(i1 %_5.i.i.i1.i.us), !dbg !818, !noalias !780
  %_4.i.i.i.i.us = getelementptr inbounds nuw i64, ptr %view.val.i.i.i, i64 %_4.i.us74, !dbg !807
  %_0.sroa.0.0.i.i.i.us75 = load i64, ptr %_4.i.i.i.i.us, align 8, !dbg !812, !noalias !751, !noundef !23
  tail call void @llvm.experimental.noalias.scope.decl(metadata !781), !dbg !778, !noalias !780
  %331 = icmp eq i64 %_0.sroa.0.0.i.i.i.us75, 9223372036854775807, !dbg !789
  %_5.i.i.i.i.us81 = icmp slt i64 %_0.sroa.0.0.i.i.i.us75, %_0.sroa.0.0.i9.i.i.us79, !dbg !792
  %332 = or i1 %331, %330, !dbg !752
  %333 = or i1 %332, %293, !dbg !752
  %334 = add nuw nsw i64 %iter.sroa.0.0.i31.us73, 1, !dbg !769
  %_13.i.us82 = zext i1 %_5.i.i.i.i.us81 to i64, !dbg !793
  %_12.i.us83 = shl nuw i64 %_13.i.us82, %iter.sroa.0.0.i31.us73, !dbg !793
  %335 = or i64 %_12.i.us83, %packed.sroa.0.0.i32.us72, !dbg !794
  %exitcond.not.us84 = icmp eq i64 %334, %remainder.i, !dbg !799
  br i1 %exitcond.not.us84, label %_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack24collect_bool_word_scalarNCINvB2_23collect_bool_words_withNCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1O_7visitor5retry21ExecuteDenseWithRetryINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB3D_EENtNtB3R_11row_visitor10RowVisitor19visit_deferred_boolB3w_bKB3D_NCINvXB4B_B4y_NtNtB1O_6row_fn5RowFn8dispatchB3M_E0NCB6t_s_0E0NCB3I_s_0B7f_E0NCINvB2_25collect_bool_words_avx512B1F_E0E0EB4B_.exit, label %bb8.i3.us71, !dbg !763, !llvm.loop !819

bb8.i3:                                           ; preds = %bb8.i3.preheader, %bb8.i3
  %336 = phi i1 [ %339, %bb8.i3 ], [ %.ph463, %bb8.i3.preheader ]
  %packed.sroa.0.0.i32 = phi i64 [ %341, %bb8.i3 ], [ %packed.sroa.0.0.i32.ph, %bb8.i3.preheader ]
  %iter.sroa.0.0.i31 = phi i64 [ %340, %bb8.i3 ], [ %iter.sroa.0.0.i31.ph, %bb8.i3.preheader ]
  %_4.i = add nuw nsw i64 %iter.sroa.0.0.i31, %181, !dbg !803
  tail call void @llvm.experimental.noalias.scope.decl(metadata !775), !dbg !776
  tail call void @llvm.experimental.noalias.scope.decl(metadata !777), !dbg !778, !noalias !780
  %_5.i.i.i1.i = icmp ult i64 %_4.i, %view.val1.i.i.i, !dbg !817
  tail call void @llvm.assume(i1 %_5.i.i.i1.i), !dbg !818, !noalias !780
  %_4.i.i.i.i = getelementptr inbounds nuw i64, ptr %view.val.i.i.i, i64 %_4.i, !dbg !807
  %_0.sroa.0.0.i.i.i = load i64, ptr %_4.i.i.i.i, align 8, !dbg !812, !noalias !751, !noundef !23
  tail call void @llvm.experimental.noalias.scope.decl(metadata !781), !dbg !778, !noalias !780
  %_5.i.i6.i.i = icmp ult i64 %_4.i, %view.val1.i5.i.i, !dbg !804
  tail call void @llvm.assume(i1 %_5.i.i6.i.i), !dbg !805, !noalias !780
  %_4.i.i7.i.i = getelementptr inbounds nuw i64, ptr %view.val.i4.i.i, i64 %_4.i, !dbg !782
  %_0.sroa.0.0.i9.i.i = load i64, ptr %_4.i.i7.i.i, align 8, !dbg !787, !noalias !788, !noundef !23
  %337 = icmp eq i64 %_0.sroa.0.0.i.i.i, 9223372036854775807, !dbg !789
  %338 = icmp eq i64 %_0.sroa.0.0.i9.i.i, 9223372036854775807, !dbg !789
  %_6.sroa.0.0.i.i.i.i = or i1 %337, %338, !dbg !789
  %_5.i.i.i.i = icmp slt i64 %_0.sroa.0.0.i.i.i, %_0.sroa.0.0.i9.i.i, !dbg !792
  %339 = or i1 %_6.sroa.0.0.i.i.i.i, %336, !dbg !752
  %340 = add nuw nsw i64 %iter.sroa.0.0.i31, 1, !dbg !769
  %_13.i = zext i1 %_5.i.i.i.i to i64, !dbg !793
  %_12.i = shl nuw i64 %_13.i, %iter.sroa.0.0.i31, !dbg !793
  %341 = or i64 %_12.i, %packed.sroa.0.0.i32, !dbg !794
  %exitcond.not = icmp eq i64 %340, %remainder.i, !dbg !799
  br i1 %exitcond.not, label %_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack24collect_bool_word_scalarNCINvB2_23collect_bool_words_withNCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1O_7visitor5retry21ExecuteDenseWithRetryINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB3D_EENtNtB3R_11row_visitor10RowVisitor19visit_deferred_boolB3w_bKB3D_NCINvXB4B_B4y_NtNtB1O_6row_fn5RowFn8dispatchB3M_E0NCB6t_s_0E0NCB3I_s_0B7f_E0NCINvB2_25collect_bool_words_avx512B1F_E0E0EB4B_.exit, label %bb8.i3, !dbg !763, !llvm.loop !820

_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack24collect_bool_word_scalarNCINvB2_23collect_bool_words_withNCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1O_7visitor5retry21ExecuteDenseWithRetryINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB3D_EENtNtB3R_11row_visitor10RowVisitor19visit_deferred_boolB3w_bKB3D_NCINvXB4B_B4y_NtNtB1O_6row_fn5RowFn8dispatchB3M_E0NCB6t_s_0E0NCB3I_s_0B7f_E0NCINvB2_25collect_bool_words_avx512B1F_E0E0EB4B_.exit: ; preds = %bb8.i3, %bb8.i3.us71, %bb8.i3.us, %bb8.i3.us.us, %middle.block240, %vec.epilog.middle.block, %middle.block294, %vec.epilog.middle.block326, %middle.block359, %vec.epilog.middle.block391, %middle.block414, %vec.epilog.middle.block436
  %.us-phi69.in = phi i1 [ %333, %bb8.i3.us71 ], [ %227, %middle.block414 ], [ %227, %bb8.i3.us.us ], [ %244, %bb8.i3.us ], [ %227, %vec.epilog.middle.block436 ], [ %222, %vec.epilog.middle.block391 ], [ %208, %middle.block359 ], [ %328, %vec.epilog.middle.block326 ], [ %314, %middle.block294 ], [ %291, %vec.epilog.middle.block ], [ %273, %middle.block240 ], [ %339, %bb8.i3 ]
  %.us-phi70 = phi i64 [ %335, %bb8.i3.us71 ], [ %233, %middle.block414 ], [ %240, %bb8.i3.us.us ], [ %246, %bb8.i3.us ], [ %238, %vec.epilog.middle.block436 ], [ %223, %vec.epilog.middle.block391 ], [ %209, %middle.block359 ], [ %329, %vec.epilog.middle.block326 ], [ %315, %middle.block294 ], [ %292, %vec.epilog.middle.block ], [ %274, %middle.block240 ], [ %341, %bb8.i3 ], !dbg !752
  %.us-phi69 = zext i1 %.us-phi69.in to i8, !dbg !752
  store i8 %.us-phi69, ptr %_11.i.i, align 8, !dbg !752, !alias.scope !748, !noalias !740
  %_41.i = icmp samesign ult i64 %full5.i, %words.1, !dbg !821
  br i1 %_41.i, label %bb14.i, label %panic.i, !dbg !821

bb14.i:                                           ; preds = %_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack24collect_bool_word_scalarNCINvB2_23collect_bool_words_withNCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1O_7visitor5retry21ExecuteDenseWithRetryINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB3D_EENtNtB3R_11row_visitor10RowVisitor19visit_deferred_boolB3w_bKB3D_NCINvXB4B_B4y_NtNtB1O_6row_fn5RowFn8dispatchB3M_E0NCB6t_s_0E0NCB3I_s_0B7f_E0NCINvB2_25collect_bool_words_avx512B1F_E0E0EB4B_.exit
  store i64 %.us-phi70, ptr %_52.i, align 8, !dbg !821, !alias.scope !613, !noalias !822
  br label %_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack23collect_bool_words_withNCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1g_7visitor5retry21ExecuteDenseWithRetryINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB35_EENtNtB3j_11row_visitor10RowVisitor19visit_deferred_boolB2Y_bKB35_NCINvXB43_B40_NtNtB1g_6row_fn5RowFn8dispatchB3e_E0NCB5V_s_0E0NCB3a_s_0B6H_E0NCINvB2_25collect_bool_words_avx512B17_E0EB43_.exit, !dbg !824

panic.i:                                          ; preds = %_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack24collect_bool_word_scalarNCINvB2_23collect_bool_words_withNCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1O_7visitor5retry21ExecuteDenseWithRetryINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB3D_EENtNtB3R_11row_visitor10RowVisitor19visit_deferred_boolB3w_bKB3D_NCINvXB4B_B4y_NtNtB1O_6row_fn5RowFn8dispatchB3M_E0NCB6t_s_0E0NCB3I_s_0B7f_E0NCINvB2_25collect_bool_words_avx512B1F_E0E0EB4B_.exit
; call core::panicking::panic_bounds_check
  tail call void @_RNvNtCsc36rpYXAlPq_4core9panicking18panic_bounds_check(i64 noundef %full5.i, i64 noundef range(i64 0, 1152921504606846976) %words.1, ptr noalias nofree noundef readonly align 8 captures(address, read_provenance) dereferenceable(24) @alloc_1fbb1658c3e837d8e13e0cd32c0c5d29.llvm.5180523300074625071) #19, !dbg !821
  unreachable

_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack23collect_bool_words_withNCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1g_7visitor5retry21ExecuteDenseWithRetryINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB35_EENtNtB3j_11row_visitor10RowVisitor19visit_deferred_boolB2Y_bKB35_NCINvXB43_B40_NtNtB1g_6row_fn5RowFn8dispatchB3e_E0NCB5V_s_0E0NCB3a_s_0B6H_E0NCINvB2_25collect_bool_words_avx512B17_E0EB43_.exit: ; preds = %bb14.i, %bb5.i
  ret void, !dbg !825
}
