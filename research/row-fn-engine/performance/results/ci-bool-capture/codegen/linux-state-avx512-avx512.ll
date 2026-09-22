define hidden void @_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack25collect_bool_words_avx512NCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1i_7visitor5retry21ExecuteDenseWithRetryINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB37_EENtNtB3l_11row_visitor10RowVisitor19visit_deferred_boolB30_bKB37_NCINvXB45_B42_NtNtB1i_6row_fn5RowFn8dispatchB3g_E0NCB5X_s_0E0NCB3c_s_0B6J_E0EB45_(ptr noalias nofree noundef nonnull writeonly align 8 captures(address) %words.0, i64 noundef range(i64 0, 1152921504606846976) %words.1, i64 noundef %len, ptr noalias nofree noundef align 8 captures(none) dereferenceable(64) %f) unnamed_addr #2 personality ptr @rust_eh_personality !dbg !1764 {
start:
  tail call void @llvm.experimental.noalias.scope.decl(metadata !1769), !dbg !1772
  %full5.i = lshr i64 %len, 6, !dbg !1773
  %remainder.i = and i64 %len, 63, !dbg !1776
  %_42.not.i = icmp samesign ugt i64 %full5.i, %words.1
  br i1 %_42.not.i, label %bb22.i, label %bb21.i, !dbg !1778, !prof !1797

bb22.i:                                           ; preds = %start
; call core::slice::index::slice_index_fail
  tail call void @_RNvNtNtCsc36rpYXAlPq_4core5slice5index16slice_index_fail(i64 noundef 0, i64 noundef %full5.i, i64 noundef range(i64 0, 1152921504606846976) %words.1, ptr noalias nofree noundef readonly align 8 captures(address, read_provenance) dereferenceable(24) @alloc_ebbd31bf42d91f579e6aa57e93569175.llvm.7192105927927146806) #21, !dbg !1798, !noalias !1769
  unreachable

bb21.i:                                           ; preds = %start
  %_52.i.idx = shl nuw nsw i64 %full5.i, 3, !dbg !1799
  %_52.i = getelementptr inbounds nuw i8, ptr %words.0, i64 %_52.i.idx, !dbg !1799
  %_7.i.i27 = icmp eq i64 %full5.i, 0, !dbg !1813
  br i1 %_7.i.i27, label %bb5.i, label %bb4.i.lr.ph, !dbg !1831

bb4.i.lr.ph:                                      ; preds = %bb21.i
  %_11.i = getelementptr inbounds nuw i8, ptr %f, i64 56
  %_11.i.promoted30 = load i8, ptr %_11.i, align 8
  %0 = trunc nuw i8 %_11.i.promoted30 to i1, !dbg !1832
  %_3.i.i2 = load i64, ptr %f, align 8, !range !1857, !alias.scope !1858, !noalias !1863, !noundef !31
  %1 = trunc nuw i64 %_3.i.i2 to i1
  %_6.i12 = getelementptr inbounds nuw i8, ptr %f, i64 24
  %_3.i1.i13 = load i64, ptr %_6.i12, align 8, !range !1857, !alias.scope !1866, !noalias !1863, !noundef !31
  %2 = trunc nuw i64 %_3.i1.i13 to i1
  %view.i.i4 = getelementptr inbounds nuw i8, ptr %f, i64 8
  %view.val.i.i5 = load ptr, ptr %view.i.i4, align 8, !nonnull !31, !align !89
  %view.i3.i15 = getelementptr inbounds nuw i8, ptr %f, i64 32
  %view.val.i4.i16 = load ptr, ptr %view.i3.i15, align 8, !nonnull !31, !align !89
  %3 = getelementptr inbounds nuw i8, ptr %f, i64 40
  %view.val1.i5.i17 = load i64, ptr %3, align 8
  %_6.i11.i23.cast = inttoptr i64 %view.val1.i5.i17 to ptr
  br i1 %1, label %bb4.i.lr.ph.split.us, label %bb4.i.lr.ph.split

bb4.i.lr.ph.split.us:                             ; preds = %bb4.i.lr.ph
  %4 = getelementptr inbounds nuw i8, ptr %f, i64 16
  %view.val1.i.i6 = load i64, ptr %4, align 8
  %5 = inttoptr i64 %view.val1.i.i6 to ptr
  %_0.sroa.0.0.i.i11.us.us = load i64, ptr %5, align 8, !noalias !1869, !noundef !31
  %6 = icmp eq i64 %_0.sroa.0.0.i.i11.us.us, 9223372036854775807
  br i1 %2, label %bb4.i.lr.ph.split.us.split.us, label %bb4.i.us.preheader

bb4.i.us.preheader:                               ; preds = %bb4.i.lr.ph.split.us
  %broadcast.splatinsert197 = insertelement <8 x i1> poison, i1 %6, i64 0
  %broadcast.splat198 = shufflevector <8 x i1> %broadcast.splatinsert197, <8 x i1> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert195 = insertelement <8 x i64> poison, i64 %_0.sroa.0.0.i.i11.us.us, i64 0
  %broadcast.splat196 = shufflevector <8 x i64> %broadcast.splatinsert195, <8 x i64> poison, <8 x i32> zeroinitializer
  %7 = getelementptr inbounds nuw i8, ptr %view.val.i4.i16, i64 256
  br label %bb4.i.us, !dbg !1870

bb4.i.lr.ph.split.us.split.us:                    ; preds = %bb4.i.lr.ph.split.us
  %_0.sroa.0.0.i9.i21.us.us.us.us = load i64, ptr %_6.i11.i23.cast, align 8, !noalias !1877, !noundef !31
  %_5.i.i.i.us.us.us.us = icmp slt i64 %_0.sroa.0.0.i.i11.us.us, %_0.sroa.0.0.i9.i21.us.us.us.us
  %8 = select i1 %_5.i.i.i.us.us.us.us, i512 257, i512 0
  %9 = shl nuw nsw i512 %8, 16
  %10 = or disjoint i512 %8, %9
  %11 = shl nuw nsw i512 %10, 32
  %12 = or disjoint i512 %10, %11
  %13 = shl nuw nsw i512 %12, 64
  %14 = or disjoint i512 %12, %13
  %15 = shl nuw nsw i512 %14, 128
  %16 = or i512 %14, %15
  %17 = shl nuw nsw i512 %16, 256
  %18 = or i512 %16, %17
  %19 = bitcast i512 %18 to <64 x i8>
  %.not.i.i.i.us.us = icmp ne <64 x i8> %19, zeroinitializer
  %20 = add nsw i64 %_52.i.idx, -8, !dbg !1831
  %21 = lshr exact i64 %20, 3, !dbg !1831
  %22 = add nuw nsw i64 %21, 1, !dbg !1831
  %xtraiter = and i64 %22, 7, !dbg !1831
  %23 = and i64 %20, 56, !dbg !1831
  %lcmp.mod.not = icmp eq i64 %23, 56, !dbg !1831
  br i1 %lcmp.mod.not, label %bb4.i.us.us.prol.loopexit, label %bb4.i.us.us.prol, !dbg !1831

bb4.i.us.us.prol:                                 ; preds = %bb4.i.lr.ph.split.us.split.us, %bb4.i.us.us.prol
  %iter.i.sroa.0.029.us.us.prol = phi ptr [ %_17.i.i.us.us.prol, %bb4.i.us.us.prol ], [ %words.0, %bb4.i.lr.ph.split.us.split.us ]
  %prol.iter = phi i64 [ %prol.iter.next, %bb4.i.us.us.prol ], [ 0, %bb4.i.lr.ph.split.us.split.us ]
  %_17.i.i.us.us.prol = getelementptr inbounds nuw i8, ptr %iter.i.sroa.0.029.us.us.prol, i64 8, !dbg !1878
  store <64 x i1> %.not.i.i.i.us.us, ptr %iter.i.sroa.0.029.us.us.prol, align 8, !dbg !1881
  %prol.iter.next = add i64 %prol.iter, 1, !dbg !1831
  %prol.iter.cmp.not = icmp eq i64 %prol.iter.next, %xtraiter, !dbg !1831
  br i1 %prol.iter.cmp.not, label %bb4.i.us.us.prol.loopexit, label %bb4.i.us.us.prol, !dbg !1831, !llvm.loop !1882

bb4.i.us.us.prol.loopexit:                        ; preds = %bb4.i.us.us.prol, %bb4.i.lr.ph.split.us.split.us
  %iter.i.sroa.0.029.us.us.unr = phi ptr [ %words.0, %bb4.i.lr.ph.split.us.split.us ], [ %_17.i.i.us.us.prol, %bb4.i.us.us.prol ]
  %24 = icmp ult i64 %20, 56, !dbg !1831
  br i1 %24, label %bb5.i.loopexit.loopexit, label %bb4.i.us.us, !dbg !1831

bb4.i.us.us:                                      ; preds = %bb4.i.us.us.prol.loopexit, %bb4.i.us.us
  %iter.i.sroa.0.029.us.us = phi ptr [ %_17.i.i.us.us.7, %bb4.i.us.us ], [ %iter.i.sroa.0.029.us.us.unr, %bb4.i.us.us.prol.loopexit ]
  %_17.i.i.us.us = getelementptr inbounds nuw i8, ptr %iter.i.sroa.0.029.us.us, i64 8, !dbg !1878
  store <64 x i1> %.not.i.i.i.us.us, ptr %iter.i.sroa.0.029.us.us, align 8, !dbg !1881
  %_17.i.i.us.us.1 = getelementptr inbounds nuw i8, ptr %iter.i.sroa.0.029.us.us, i64 16, !dbg !1878
  store <64 x i1> %.not.i.i.i.us.us, ptr %_17.i.i.us.us, align 8, !dbg !1881
  %_17.i.i.us.us.2 = getelementptr inbounds nuw i8, ptr %iter.i.sroa.0.029.us.us, i64 24, !dbg !1878
  store <64 x i1> %.not.i.i.i.us.us, ptr %_17.i.i.us.us.1, align 8, !dbg !1881
  %_17.i.i.us.us.3 = getelementptr inbounds nuw i8, ptr %iter.i.sroa.0.029.us.us, i64 32, !dbg !1878
  store <64 x i1> %.not.i.i.i.us.us, ptr %_17.i.i.us.us.2, align 8, !dbg !1881
  %_17.i.i.us.us.4 = getelementptr inbounds nuw i8, ptr %iter.i.sroa.0.029.us.us, i64 40, !dbg !1878
  store <64 x i1> %.not.i.i.i.us.us, ptr %_17.i.i.us.us.3, align 8, !dbg !1881
  %_17.i.i.us.us.5 = getelementptr inbounds nuw i8, ptr %iter.i.sroa.0.029.us.us, i64 48, !dbg !1878
  store <64 x i1> %.not.i.i.i.us.us, ptr %_17.i.i.us.us.4, align 8, !dbg !1881
  %_17.i.i.us.us.6 = getelementptr inbounds nuw i8, ptr %iter.i.sroa.0.029.us.us, i64 56, !dbg !1878
  store <64 x i1> %.not.i.i.i.us.us, ptr %_17.i.i.us.us.5, align 8, !dbg !1881
  %_17.i.i.us.us.7 = getelementptr inbounds nuw i8, ptr %iter.i.sroa.0.029.us.us, i64 64, !dbg !1878
  store <64 x i1> %.not.i.i.i.us.us, ptr %_17.i.i.us.us.6, align 8, !dbg !1881
  %_7.i.i.us.us.7 = icmp eq ptr %_17.i.i.us.us.7, %_52.i, !dbg !1813
  br i1 %_7.i.i.us.us.7, label %bb5.i.loopexit.loopexit, label %bb4.i.us.us, !dbg !1831

bb4.i.us:                                         ; preds = %bb4.i.us.preheader, %bb4.i.us
  %_11.i.promoted31.us = phi i1 [ %65, %bb4.i.us ], [ %0, %bb4.i.us.preheader ]
  %iter.i.sroa.0.029.us = phi ptr [ %_17.i.i.us, %bb4.i.us ], [ %words.0, %bb4.i.us.preheader ]
  %iter.i.sroa.7.028.us = phi i64 [ %_9.0.i.us, %bb4.i.us ], [ 0, %bb4.i.us.preheader ]
  %25 = insertelement <8 x i1> <i1 poison, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false>, i1 %_11.i.promoted31.us, i64 0
  tail call void @llvm.experimental.noalias.scope.decl(metadata !1884), !dbg !1885
  tail call void @llvm.experimental.noalias.scope.decl(metadata !1886), !dbg !1887, !noalias !1863
  tail call void @llvm.experimental.noalias.scope.decl(metadata !1896), !dbg !1887, !noalias !1863
  %.idx442 = shl i64 %iter.i.sroa.7.028.us, 9, !dbg !1897
  %26 = getelementptr inbounds nuw i8, ptr %view.val.i4.i16, i64 %.idx442, !dbg !1897
  %27 = getelementptr inbounds nuw i8, ptr %26, i64 64, !dbg !1912
  %28 = getelementptr inbounds nuw i8, ptr %26, i64 128, !dbg !1912
  %29 = getelementptr inbounds nuw i8, ptr %26, i64 192, !dbg !1912
  %wide.load209 = load <8 x i64>, ptr %26, align 8, !dbg !1912, !noalias !1877
  %wide.load210 = load <8 x i64>, ptr %27, align 8, !dbg !1912, !noalias !1877
  %wide.load211 = load <8 x i64>, ptr %28, align 8, !dbg !1912, !noalias !1877
  %wide.load212 = load <8 x i64>, ptr %29, align 8, !dbg !1912, !noalias !1877
  %30 = icmp eq <8 x i64> %wide.load209, splat (i64 9223372036854775807), !dbg !1913
  %31 = icmp eq <8 x i64> %wide.load210, splat (i64 9223372036854775807), !dbg !1913
  %32 = icmp eq <8 x i64> %wide.load211, splat (i64 9223372036854775807), !dbg !1913
  %33 = icmp eq <8 x i64> %wide.load212, splat (i64 9223372036854775807), !dbg !1913
  %34 = icmp slt <8 x i64> %broadcast.splat196, %wide.load209, !dbg !1926
  %35 = icmp slt <8 x i64> %broadcast.splat196, %wide.load210, !dbg !1926
  %36 = icmp slt <8 x i64> %broadcast.splat196, %wide.load211, !dbg !1926
  %37 = icmp slt <8 x i64> %broadcast.splat196, %wide.load212, !dbg !1926
  %38 = or <8 x i1> %30, %25, !dbg !1832
  %39 = zext <8 x i1> %34 to <8 x i8>, !dbg !1927
  %40 = zext <8 x i1> %35 to <8 x i8>, !dbg !1927
  %41 = zext <8 x i1> %36 to <8 x i8>, !dbg !1927
  %42 = zext <8 x i1> %37 to <8 x i8>, !dbg !1927
  %bools.i.sroa.0.0.vec.expand501 = shufflevector <8 x i8> %39, <8 x i8> poison, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !1927
  %bools.i.sroa.0.8.vec.expand509 = shufflevector <8 x i8> %40, <8 x i8> poison, <64 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !1927
  %bools.i.sroa.0.8.vecblend510 = shufflevector <64 x i8> %bools.i.sroa.0.0.vec.expand501, <64 x i8> %bools.i.sroa.0.8.vec.expand509, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 72, i32 73, i32 74, i32 75, i32 76, i32 77, i32 78, i32 79, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !1927
  %bools.i.sroa.0.16.vec.expand515 = shufflevector <8 x i8> %41, <8 x i8> poison, <64 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !1927
  %bools.i.sroa.0.16.vecblend516 = shufflevector <64 x i8> %bools.i.sroa.0.8.vecblend510, <64 x i8> %bools.i.sroa.0.16.vec.expand515, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 80, i32 81, i32 82, i32 83, i32 84, i32 85, i32 86, i32 87, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !1927
  %bools.i.sroa.0.24.vec.expand521 = shufflevector <8 x i8> %42, <8 x i8> poison, <64 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !1927
  %bools.i.sroa.0.24.vecblend522 = shufflevector <64 x i8> %bools.i.sroa.0.16.vecblend516, <64 x i8> %bools.i.sroa.0.24.vec.expand521, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 88, i32 89, i32 90, i32 91, i32 92, i32 93, i32 94, i32 95, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !1927
  %.idx442.1 = shl i64 %iter.i.sroa.7.028.us, 9, !dbg !1897
  %43 = getelementptr inbounds nuw i8, ptr %7, i64 %.idx442.1, !dbg !1897
  %44 = getelementptr inbounds nuw i8, ptr %43, i64 64, !dbg !1912
  %45 = getelementptr inbounds nuw i8, ptr %43, i64 128, !dbg !1912
  %46 = getelementptr inbounds nuw i8, ptr %43, i64 192, !dbg !1912
  %wide.load209.1 = load <8 x i64>, ptr %43, align 8, !dbg !1912, !noalias !1928
  %wide.load210.1 = load <8 x i64>, ptr %44, align 8, !dbg !1912, !noalias !1928
  %wide.load211.1 = load <8 x i64>, ptr %45, align 8, !dbg !1912, !noalias !1928
  %wide.load212.1 = load <8 x i64>, ptr %46, align 8, !dbg !1912, !noalias !1928
  %47 = icmp eq <8 x i64> %wide.load209.1, splat (i64 9223372036854775807), !dbg !1913
  %48 = icmp eq <8 x i64> %wide.load210.1, splat (i64 9223372036854775807), !dbg !1913
  %49 = icmp eq <8 x i64> %wide.load211.1, splat (i64 9223372036854775807), !dbg !1913
  %50 = icmp eq <8 x i64> %wide.load212.1, splat (i64 9223372036854775807), !dbg !1913
  %51 = icmp slt <8 x i64> %broadcast.splat196, %wide.load209.1, !dbg !1926
  %52 = icmp slt <8 x i64> %broadcast.splat196, %wide.load210.1, !dbg !1926
  %53 = icmp slt <8 x i64> %broadcast.splat196, %wide.load211.1, !dbg !1926
  %54 = icmp slt <8 x i64> %broadcast.splat196, %wide.load212.1, !dbg !1926
  %55 = or <8 x i1> %38, %47, !dbg !1832
  %56 = or <8 x i1> %55, %broadcast.splat198, !dbg !1832
  %57 = or <8 x i1> %31, %48, !dbg !1832
  %58 = or <8 x i1> %32, %49, !dbg !1832
  %59 = or <8 x i1> %33, %50, !dbg !1832
  %60 = zext <8 x i1> %51 to <8 x i8>, !dbg !1927
  %61 = zext <8 x i1> %52 to <8 x i8>, !dbg !1927
  %62 = zext <8 x i1> %53 to <8 x i8>, !dbg !1927
  %63 = zext <8 x i1> %54 to <8 x i8>, !dbg !1927
  %bools.i.sroa.0.32.vec.expand527 = shufflevector <8 x i8> %60, <8 x i8> poison, <64 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !1927
  %bools.i.sroa.0.32.vecblend528 = shufflevector <64 x i8> %bools.i.sroa.0.24.vecblend522, <64 x i8> %bools.i.sroa.0.32.vec.expand527, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 96, i32 97, i32 98, i32 99, i32 100, i32 101, i32 102, i32 103, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !1927
  %bools.i.sroa.0.40.vec.expand533 = shufflevector <8 x i8> %61, <8 x i8> poison, <64 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !1927
  %bools.i.sroa.0.40.vecblend534 = shufflevector <64 x i8> %bools.i.sroa.0.32.vecblend528, <64 x i8> %bools.i.sroa.0.40.vec.expand533, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 104, i32 105, i32 106, i32 107, i32 108, i32 109, i32 110, i32 111, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !1927
  %bools.i.sroa.0.48.vec.expand539 = shufflevector <8 x i8> %62, <8 x i8> poison, <64 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !1927
  %bools.i.sroa.0.48.vecblend540 = shufflevector <64 x i8> %bools.i.sroa.0.40.vecblend534, <64 x i8> %bools.i.sroa.0.48.vec.expand539, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 112, i32 113, i32 114, i32 115, i32 116, i32 117, i32 118, i32 119, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !1927
  %bools.i.sroa.0.56.vec.expand545 = shufflevector <8 x i8> %63, <8 x i8> poison, <64 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !1927
  %bools.i.sroa.0.56.vecblend546 = shufflevector <64 x i8> %bools.i.sroa.0.48.vecblend540, <64 x i8> %bools.i.sroa.0.56.vec.expand545, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 120, i32 121, i32 122, i32 123, i32 124, i32 125, i32 126, i32 127>, !dbg !1927
  %bin.rdx216 = or <8 x i1> %57, %56, !dbg !1870
  %bin.rdx217 = or <8 x i1> %58, %bin.rdx216, !dbg !1870
  %bin.rdx218 = or <8 x i1> %59, %bin.rdx217, !dbg !1870
  %64 = bitcast <8 x i1> %bin.rdx218 to i8, !dbg !1870
  %65 = icmp ne i8 %64, 0, !dbg !1870
  %_17.i.i.us = getelementptr inbounds nuw i8, ptr %iter.i.sroa.0.029.us, i64 8, !dbg !1878
  %_9.0.i.us = add nuw nsw i64 %iter.i.sroa.7.028.us, 1, !dbg !1931
  %.not.i.i.i.us = icmp ne <64 x i8> %bools.i.sroa.0.56.vecblend546, zeroinitializer, !dbg !1934
  store <64 x i1> %.not.i.i.i.us, ptr %iter.i.sroa.0.029.us, align 8, !dbg !1881
  %_7.i.i.us = icmp eq ptr %_17.i.i.us, %_52.i, !dbg !1813
  br i1 %_7.i.i.us, label %bb5.i.loopexit, label %bb4.i.us, !dbg !1831

bb4.i.lr.ph.split:                                ; preds = %bb4.i.lr.ph
  br i1 %2, label %bb4.i.lr.ph.split.split.us, label %bb4.i

bb4.i.lr.ph.split.split.us:                       ; preds = %bb4.i.lr.ph.split
  %_0.sroa.0.0.i9.i21.us40.us = load i64, ptr %_6.i11.i23.cast, align 8, !noalias !1877, !noundef !31
  %66 = icmp eq i64 %_0.sroa.0.0.i9.i21.us40.us, 9223372036854775807
  %broadcast.splatinsert168 = insertelement <8 x i64> poison, i64 %_0.sroa.0.0.i9.i21.us40.us, i64 0
  %broadcast.splat169 = shufflevector <8 x i64> %broadcast.splatinsert168, <8 x i64> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert166 = insertelement <8 x i1> poison, i1 %66, i64 0
  %broadcast.splat167 = shufflevector <8 x i1> %broadcast.splatinsert166, <8 x i1> poison, <8 x i32> zeroinitializer
  %67 = getelementptr inbounds nuw i8, ptr %view.val.i.i5, i64 256
  br label %bb4.i.us64, !dbg !1831

bb4.i.us64:                                       ; preds = %bb4.i.us64, %bb4.i.lr.ph.split.split.us
  %_11.i.promoted31.us65 = phi i1 [ %0, %bb4.i.lr.ph.split.split.us ], [ %108, %bb4.i.us64 ]
  %iter.i.sroa.0.029.us66 = phi ptr [ %words.0, %bb4.i.lr.ph.split.split.us ], [ %_17.i.i.us69, %bb4.i.us64 ]
  %iter.i.sroa.7.028.us67 = phi i64 [ 0, %bb4.i.lr.ph.split.split.us ], [ %_9.0.i.us70, %bb4.i.us64 ]
  %68 = insertelement <8 x i1> <i1 poison, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false>, i1 %_11.i.promoted31.us65, i64 0
  tail call void @llvm.experimental.noalias.scope.decl(metadata !1884), !dbg !1885
  tail call void @llvm.experimental.noalias.scope.decl(metadata !1886), !dbg !1887, !noalias !1863
  %.idx = shl i64 %iter.i.sroa.7.028.us67, 9, !dbg !1951
  %69 = getelementptr inbounds nuw i8, ptr %view.val.i.i5, i64 %.idx, !dbg !1951
  %70 = getelementptr inbounds nuw i8, ptr %69, i64 64, !dbg !1956
  %71 = getelementptr inbounds nuw i8, ptr %69, i64 128, !dbg !1956
  %72 = getelementptr inbounds nuw i8, ptr %69, i64 192, !dbg !1956
  %wide.load180 = load <8 x i64>, ptr %69, align 8, !dbg !1956, !noalias !1869
  %wide.load181 = load <8 x i64>, ptr %70, align 8, !dbg !1956, !noalias !1869
  %wide.load182 = load <8 x i64>, ptr %71, align 8, !dbg !1956, !noalias !1869
  %wide.load183 = load <8 x i64>, ptr %72, align 8, !dbg !1956, !noalias !1869
  tail call void @llvm.experimental.noalias.scope.decl(metadata !1896), !dbg !1887, !noalias !1863
  %73 = icmp eq <8 x i64> %wide.load180, splat (i64 9223372036854775807), !dbg !1913
  %74 = icmp eq <8 x i64> %wide.load181, splat (i64 9223372036854775807), !dbg !1913
  %75 = icmp eq <8 x i64> %wide.load182, splat (i64 9223372036854775807), !dbg !1913
  %76 = icmp eq <8 x i64> %wide.load183, splat (i64 9223372036854775807), !dbg !1913
  %77 = icmp slt <8 x i64> %wide.load180, %broadcast.splat169, !dbg !1926
  %78 = icmp slt <8 x i64> %wide.load181, %broadcast.splat169, !dbg !1926
  %79 = icmp slt <8 x i64> %wide.load182, %broadcast.splat169, !dbg !1926
  %80 = icmp slt <8 x i64> %wide.load183, %broadcast.splat169, !dbg !1926
  %81 = or <8 x i1> %73, %68, !dbg !1832
  %82 = zext <8 x i1> %77 to <8 x i8>, !dbg !1927
  %83 = zext <8 x i1> %78 to <8 x i8>, !dbg !1927
  %84 = zext <8 x i1> %79 to <8 x i8>, !dbg !1927
  %85 = zext <8 x i1> %80 to <8 x i8>, !dbg !1927
  %bools.i.sroa.0.0.vec.expand498 = shufflevector <8 x i8> %82, <8 x i8> poison, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !1927
  %bools.i.sroa.0.8.vec.expand506 = shufflevector <8 x i8> %83, <8 x i8> poison, <64 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !1927
  %bools.i.sroa.0.8.vecblend507 = shufflevector <64 x i8> %bools.i.sroa.0.0.vec.expand498, <64 x i8> %bools.i.sroa.0.8.vec.expand506, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 72, i32 73, i32 74, i32 75, i32 76, i32 77, i32 78, i32 79, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !1927
  %bools.i.sroa.0.16.vec.expand512 = shufflevector <8 x i8> %84, <8 x i8> poison, <64 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !1927
  %bools.i.sroa.0.16.vecblend513 = shufflevector <64 x i8> %bools.i.sroa.0.8.vecblend507, <64 x i8> %bools.i.sroa.0.16.vec.expand512, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 80, i32 81, i32 82, i32 83, i32 84, i32 85, i32 86, i32 87, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !1927
  %bools.i.sroa.0.24.vec.expand518 = shufflevector <8 x i8> %85, <8 x i8> poison, <64 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !1927
  %bools.i.sroa.0.24.vecblend519 = shufflevector <64 x i8> %bools.i.sroa.0.16.vecblend513, <64 x i8> %bools.i.sroa.0.24.vec.expand518, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 88, i32 89, i32 90, i32 91, i32 92, i32 93, i32 94, i32 95, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !1927
  %.idx.1 = shl i64 %iter.i.sroa.7.028.us67, 9, !dbg !1951
  %86 = getelementptr inbounds nuw i8, ptr %67, i64 %.idx.1, !dbg !1951
  %87 = getelementptr inbounds nuw i8, ptr %86, i64 64, !dbg !1956
  %88 = getelementptr inbounds nuw i8, ptr %86, i64 128, !dbg !1956
  %89 = getelementptr inbounds nuw i8, ptr %86, i64 192, !dbg !1956
  %wide.load180.1 = load <8 x i64>, ptr %86, align 8, !dbg !1956, !noalias !1957
  %wide.load181.1 = load <8 x i64>, ptr %87, align 8, !dbg !1956, !noalias !1957
  %wide.load182.1 = load <8 x i64>, ptr %88, align 8, !dbg !1956, !noalias !1957
  %wide.load183.1 = load <8 x i64>, ptr %89, align 8, !dbg !1956, !noalias !1957
  %90 = icmp eq <8 x i64> %wide.load180.1, splat (i64 9223372036854775807), !dbg !1913
  %91 = icmp eq <8 x i64> %wide.load181.1, splat (i64 9223372036854775807), !dbg !1913
  %92 = icmp eq <8 x i64> %wide.load182.1, splat (i64 9223372036854775807), !dbg !1913
  %93 = icmp eq <8 x i64> %wide.load183.1, splat (i64 9223372036854775807), !dbg !1913
  %94 = icmp slt <8 x i64> %wide.load180.1, %broadcast.splat169, !dbg !1926
  %95 = icmp slt <8 x i64> %wide.load181.1, %broadcast.splat169, !dbg !1926
  %96 = icmp slt <8 x i64> %wide.load182.1, %broadcast.splat169, !dbg !1926
  %97 = icmp slt <8 x i64> %wide.load183.1, %broadcast.splat169, !dbg !1926
  %98 = or <8 x i1> %81, %90, !dbg !1832
  %99 = or <8 x i1> %98, %broadcast.splat167, !dbg !1832
  %100 = or <8 x i1> %74, %91, !dbg !1832
  %101 = or <8 x i1> %75, %92, !dbg !1832
  %102 = or <8 x i1> %76, %93, !dbg !1832
  %103 = zext <8 x i1> %94 to <8 x i8>, !dbg !1927
  %104 = zext <8 x i1> %95 to <8 x i8>, !dbg !1927
  %105 = zext <8 x i1> %96 to <8 x i8>, !dbg !1927
  %106 = zext <8 x i1> %97 to <8 x i8>, !dbg !1927
  %bools.i.sroa.0.32.vec.expand524 = shufflevector <8 x i8> %103, <8 x i8> poison, <64 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !1927
  %bools.i.sroa.0.32.vecblend525 = shufflevector <64 x i8> %bools.i.sroa.0.24.vecblend519, <64 x i8> %bools.i.sroa.0.32.vec.expand524, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 96, i32 97, i32 98, i32 99, i32 100, i32 101, i32 102, i32 103, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !1927
  %bools.i.sroa.0.40.vec.expand530 = shufflevector <8 x i8> %104, <8 x i8> poison, <64 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !1927
  %bools.i.sroa.0.40.vecblend531 = shufflevector <64 x i8> %bools.i.sroa.0.32.vecblend525, <64 x i8> %bools.i.sroa.0.40.vec.expand530, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 104, i32 105, i32 106, i32 107, i32 108, i32 109, i32 110, i32 111, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !1927
  %bools.i.sroa.0.48.vec.expand536 = shufflevector <8 x i8> %105, <8 x i8> poison, <64 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !1927
  %bools.i.sroa.0.48.vecblend537 = shufflevector <64 x i8> %bools.i.sroa.0.40.vecblend531, <64 x i8> %bools.i.sroa.0.48.vec.expand536, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 112, i32 113, i32 114, i32 115, i32 116, i32 117, i32 118, i32 119, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !1927
  %bools.i.sroa.0.56.vec.expand542 = shufflevector <8 x i8> %106, <8 x i8> poison, <64 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !1927
  %bools.i.sroa.0.56.vecblend543 = shufflevector <64 x i8> %bools.i.sroa.0.48.vecblend537, <64 x i8> %bools.i.sroa.0.56.vec.expand542, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 120, i32 121, i32 122, i32 123, i32 124, i32 125, i32 126, i32 127>, !dbg !1927
  %bin.rdx187 = or <8 x i1> %100, %99, !dbg !1870
  %bin.rdx188 = or <8 x i1> %101, %bin.rdx187, !dbg !1870
  %bin.rdx189 = or <8 x i1> %102, %bin.rdx188, !dbg !1870
  %107 = bitcast <8 x i1> %bin.rdx189 to i8, !dbg !1870
  %108 = icmp ne i8 %107, 0, !dbg !1870
  %_17.i.i.us69 = getelementptr inbounds nuw i8, ptr %iter.i.sroa.0.029.us66, i64 8, !dbg !1878
  %_9.0.i.us70 = add nuw nsw i64 %iter.i.sroa.7.028.us67, 1, !dbg !1931
  %.not.i.i.i.us72 = icmp ne <64 x i8> %bools.i.sroa.0.56.vecblend543, zeroinitializer, !dbg !1934
  store <64 x i1> %.not.i.i.i.us72, ptr %iter.i.sroa.0.029.us66, align 8, !dbg !1881
  %_7.i.i.us73 = icmp eq ptr %_17.i.i.us69, %_52.i, !dbg !1813
  br i1 %_7.i.i.us73, label %bb5.i.loopexit, label %bb4.i.us64, !dbg !1831

bb4.i:                                            ; preds = %bb4.i.lr.ph.split, %bb4.i
  %_11.i.promoted31 = phi i1 [ %173, %bb4.i ], [ %0, %bb4.i.lr.ph.split ]
  %iter.i.sroa.0.029 = phi ptr [ %_17.i.i, %bb4.i ], [ %words.0, %bb4.i.lr.ph.split ]
  %iter.i.sroa.7.028 = phi i64 [ %_9.0.i, %bb4.i ], [ 0, %bb4.i.lr.ph.split ]
  %109 = insertelement <8 x i1> <i1 poison, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false>, i1 %_11.i.promoted31, i64 0
  %offset2.i = shl i64 %iter.i.sroa.7.028, 6, !dbg !1960
  tail call void @llvm.experimental.noalias.scope.decl(metadata !1884), !dbg !1885
  tail call void @llvm.experimental.noalias.scope.decl(metadata !1886), !dbg !1887, !noalias !1863
  %110 = getelementptr inbounds nuw i64, ptr %view.val.i.i5, i64 %offset2.i, !dbg !1951
  %111 = getelementptr inbounds nuw i8, ptr %110, i64 64, !dbg !1956
  %112 = getelementptr inbounds nuw i8, ptr %110, i64 128, !dbg !1956
  %113 = getelementptr inbounds nuw i8, ptr %110, i64 192, !dbg !1956
  %wide.load = load <8 x i64>, ptr %110, align 8, !dbg !1956, !noalias !1869
  %wide.load152 = load <8 x i64>, ptr %111, align 8, !dbg !1956, !noalias !1869
  %wide.load153 = load <8 x i64>, ptr %112, align 8, !dbg !1956, !noalias !1869
  %wide.load154 = load <8 x i64>, ptr %113, align 8, !dbg !1956, !noalias !1869
  tail call void @llvm.experimental.noalias.scope.decl(metadata !1896), !dbg !1887, !noalias !1863
  %114 = getelementptr inbounds nuw i64, ptr %view.val.i4.i16, i64 %offset2.i, !dbg !1897
  %115 = getelementptr inbounds nuw i8, ptr %114, i64 64, !dbg !1912
  %116 = getelementptr inbounds nuw i8, ptr %114, i64 128, !dbg !1912
  %117 = getelementptr inbounds nuw i8, ptr %114, i64 192, !dbg !1912
  %wide.load155 = load <8 x i64>, ptr %114, align 8, !dbg !1912, !noalias !1877
  %wide.load156 = load <8 x i64>, ptr %115, align 8, !dbg !1912, !noalias !1877
  %wide.load157 = load <8 x i64>, ptr %116, align 8, !dbg !1912, !noalias !1877
  %wide.load158 = load <8 x i64>, ptr %117, align 8, !dbg !1912, !noalias !1877
  %118 = icmp eq <8 x i64> %wide.load, splat (i64 9223372036854775807), !dbg !1913
  %119 = icmp eq <8 x i64> %wide.load152, splat (i64 9223372036854775807), !dbg !1913
  %120 = icmp eq <8 x i64> %wide.load153, splat (i64 9223372036854775807), !dbg !1913
  %121 = icmp eq <8 x i64> %wide.load154, splat (i64 9223372036854775807), !dbg !1913
  %122 = icmp eq <8 x i64> %wide.load155, splat (i64 9223372036854775807), !dbg !1913
  %123 = icmp eq <8 x i64> %wide.load156, splat (i64 9223372036854775807), !dbg !1913
  %124 = icmp eq <8 x i64> %wide.load157, splat (i64 9223372036854775807), !dbg !1913
  %125 = icmp eq <8 x i64> %wide.load158, splat (i64 9223372036854775807), !dbg !1913
  %126 = or <8 x i1> %118, %122, !dbg !1913
  %127 = or <8 x i1> %119, %123, !dbg !1913
  %128 = or <8 x i1> %120, %124, !dbg !1913
  %129 = or <8 x i1> %121, %125, !dbg !1913
  %130 = icmp slt <8 x i64> %wide.load, %wide.load155, !dbg !1926
  %131 = icmp slt <8 x i64> %wide.load152, %wide.load156, !dbg !1926
  %132 = icmp slt <8 x i64> %wide.load153, %wide.load157, !dbg !1926
  %133 = icmp slt <8 x i64> %wide.load154, %wide.load158, !dbg !1926
  %134 = or <8 x i1> %109, %126, !dbg !1832
  %135 = zext <8 x i1> %130 to <8 x i8>, !dbg !1927
  %136 = zext <8 x i1> %131 to <8 x i8>, !dbg !1927
  %137 = zext <8 x i1> %132 to <8 x i8>, !dbg !1927
  %138 = zext <8 x i1> %133 to <8 x i8>, !dbg !1927
  %bools.i.sroa.0.0.vec.expand = shufflevector <8 x i8> %135, <8 x i8> poison, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !1927
  %bools.i.sroa.0.8.vec.expand = shufflevector <8 x i8> %136, <8 x i8> poison, <64 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !1927
  %bools.i.sroa.0.8.vecblend = shufflevector <64 x i8> %bools.i.sroa.0.0.vec.expand, <64 x i8> %bools.i.sroa.0.8.vec.expand, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 72, i32 73, i32 74, i32 75, i32 76, i32 77, i32 78, i32 79, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !1927
  %bools.i.sroa.0.16.vec.expand = shufflevector <8 x i8> %137, <8 x i8> poison, <64 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !1927
  %bools.i.sroa.0.16.vecblend = shufflevector <64 x i8> %bools.i.sroa.0.8.vecblend, <64 x i8> %bools.i.sroa.0.16.vec.expand, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 80, i32 81, i32 82, i32 83, i32 84, i32 85, i32 86, i32 87, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !1927
  %bools.i.sroa.0.24.vec.expand = shufflevector <8 x i8> %138, <8 x i8> poison, <64 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !1927
  %bools.i.sroa.0.24.vecblend = shufflevector <64 x i8> %bools.i.sroa.0.16.vecblend, <64 x i8> %bools.i.sroa.0.24.vec.expand, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 88, i32 89, i32 90, i32 91, i32 92, i32 93, i32 94, i32 95, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !1927
  %139 = or disjoint i64 %offset2.i, 32
  %140 = getelementptr inbounds nuw i64, ptr %view.val.i.i5, i64 %139, !dbg !1951
  %141 = getelementptr inbounds nuw i8, ptr %140, i64 64, !dbg !1956
  %142 = getelementptr inbounds nuw i8, ptr %140, i64 128, !dbg !1956
  %143 = getelementptr inbounds nuw i8, ptr %140, i64 192, !dbg !1956
  %wide.load.1 = load <8 x i64>, ptr %140, align 8, !dbg !1956, !noalias !1961
  %wide.load152.1 = load <8 x i64>, ptr %141, align 8, !dbg !1956, !noalias !1961
  %wide.load153.1 = load <8 x i64>, ptr %142, align 8, !dbg !1956, !noalias !1961
  %wide.load154.1 = load <8 x i64>, ptr %143, align 8, !dbg !1956, !noalias !1961
  %144 = getelementptr inbounds nuw i64, ptr %view.val.i4.i16, i64 %139, !dbg !1897
  %145 = getelementptr inbounds nuw i8, ptr %144, i64 64, !dbg !1912
  %146 = getelementptr inbounds nuw i8, ptr %144, i64 128, !dbg !1912
  %147 = getelementptr inbounds nuw i8, ptr %144, i64 192, !dbg !1912
  %wide.load155.1 = load <8 x i64>, ptr %144, align 8, !dbg !1912, !noalias !1964
  %wide.load156.1 = load <8 x i64>, ptr %145, align 8, !dbg !1912, !noalias !1964
  %wide.load157.1 = load <8 x i64>, ptr %146, align 8, !dbg !1912, !noalias !1964
  %wide.load158.1 = load <8 x i64>, ptr %147, align 8, !dbg !1912, !noalias !1964
  %148 = icmp eq <8 x i64> %wide.load.1, splat (i64 9223372036854775807), !dbg !1913
  %149 = icmp eq <8 x i64> %wide.load152.1, splat (i64 9223372036854775807), !dbg !1913
  %150 = icmp eq <8 x i64> %wide.load153.1, splat (i64 9223372036854775807), !dbg !1913
  %151 = icmp eq <8 x i64> %wide.load154.1, splat (i64 9223372036854775807), !dbg !1913
  %152 = icmp eq <8 x i64> %wide.load155.1, splat (i64 9223372036854775807), !dbg !1913
  %153 = icmp eq <8 x i64> %wide.load156.1, splat (i64 9223372036854775807), !dbg !1913
  %154 = icmp eq <8 x i64> %wide.load157.1, splat (i64 9223372036854775807), !dbg !1913
  %155 = icmp eq <8 x i64> %wide.load158.1, splat (i64 9223372036854775807), !dbg !1913
  %156 = or <8 x i1> %148, %152, !dbg !1913
  %157 = or <8 x i1> %149, %153, !dbg !1913
  %158 = or <8 x i1> %150, %154, !dbg !1913
  %159 = or <8 x i1> %151, %155, !dbg !1913
  %160 = icmp slt <8 x i64> %wide.load.1, %wide.load155.1, !dbg !1926
  %161 = icmp slt <8 x i64> %wide.load152.1, %wide.load156.1, !dbg !1926
  %162 = icmp slt <8 x i64> %wide.load153.1, %wide.load157.1, !dbg !1926
  %163 = icmp slt <8 x i64> %wide.load154.1, %wide.load158.1, !dbg !1926
  %164 = or <8 x i1> %134, %156, !dbg !1832
  %165 = or <8 x i1> %127, %157, !dbg !1832
  %166 = or <8 x i1> %128, %158, !dbg !1832
  %167 = or <8 x i1> %129, %159, !dbg !1832
  %168 = zext <8 x i1> %160 to <8 x i8>, !dbg !1927
  %169 = zext <8 x i1> %161 to <8 x i8>, !dbg !1927
  %170 = zext <8 x i1> %162 to <8 x i8>, !dbg !1927
  %171 = zext <8 x i1> %163 to <8 x i8>, !dbg !1927
  %bools.i.sroa.0.32.vec.expand = shufflevector <8 x i8> %168, <8 x i8> poison, <64 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !1927
  %bools.i.sroa.0.32.vecblend = shufflevector <64 x i8> %bools.i.sroa.0.24.vecblend, <64 x i8> %bools.i.sroa.0.32.vec.expand, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 96, i32 97, i32 98, i32 99, i32 100, i32 101, i32 102, i32 103, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !1927
  %bools.i.sroa.0.40.vec.expand = shufflevector <8 x i8> %169, <8 x i8> poison, <64 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !1927
  %bools.i.sroa.0.40.vecblend = shufflevector <64 x i8> %bools.i.sroa.0.32.vecblend, <64 x i8> %bools.i.sroa.0.40.vec.expand, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 104, i32 105, i32 106, i32 107, i32 108, i32 109, i32 110, i32 111, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !1927
  %bools.i.sroa.0.48.vec.expand = shufflevector <8 x i8> %170, <8 x i8> poison, <64 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !1927
  %bools.i.sroa.0.48.vecblend = shufflevector <64 x i8> %bools.i.sroa.0.40.vecblend, <64 x i8> %bools.i.sroa.0.48.vec.expand, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 112, i32 113, i32 114, i32 115, i32 116, i32 117, i32 118, i32 119, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !1927
  %bools.i.sroa.0.56.vec.expand = shufflevector <8 x i8> %171, <8 x i8> poison, <64 x i32> <i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !1927
  %bools.i.sroa.0.56.vecblend = shufflevector <64 x i8> %bools.i.sroa.0.48.vecblend, <64 x i8> %bools.i.sroa.0.56.vec.expand, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 120, i32 121, i32 122, i32 123, i32 124, i32 125, i32 126, i32 127>, !dbg !1927
  %bin.rdx = or <8 x i1> %165, %164, !dbg !1870
  %bin.rdx159 = or <8 x i1> %166, %bin.rdx, !dbg !1870
  %bin.rdx160 = or <8 x i1> %167, %bin.rdx159, !dbg !1870
  %172 = bitcast <8 x i1> %bin.rdx160 to i8, !dbg !1870
  %173 = icmp ne i8 %172, 0, !dbg !1870
  %_17.i.i = getelementptr inbounds nuw i8, ptr %iter.i.sroa.0.029, i64 8, !dbg !1878
  %_9.0.i = add nuw nsw i64 %iter.i.sroa.7.028, 1, !dbg !1931
  %.not.i.i.i = icmp ne <64 x i8> %bools.i.sroa.0.56.vecblend, zeroinitializer, !dbg !1934
  store <64 x i1> %.not.i.i.i, ptr %iter.i.sroa.0.029, align 8, !dbg !1881
  %_7.i.i = icmp eq ptr %_17.i.i, %_52.i, !dbg !1813
  br i1 %_7.i.i, label %bb5.i.loopexit, label %bb4.i, !dbg !1831

bb5.i.loopexit.loopexit:                          ; preds = %bb4.i.us.us, %bb4.i.us.us.prol.loopexit
  %174 = icmp eq i64 %_0.sroa.0.0.i9.i21.us.us.us.us, 9223372036854775807
  %_6.sroa.0.0.i.i.i.us.us.us.us = or i1 %6, %174
  %175 = or i1 %_6.sroa.0.0.i.i.i.us.us.us.us, %0, !dbg !1832
  br label %bb5.i.loopexit

bb5.i.loopexit:                                   ; preds = %bb4.i, %bb4.i.us64, %bb4.i.us, %bb5.i.loopexit.loopexit
  %.us-phi63 = phi i1 [ %175, %bb5.i.loopexit.loopexit ], [ %65, %bb4.i.us ], [ %108, %bb4.i.us64 ], [ %173, %bb4.i ]
  %176 = zext i1 %.us-phi63 to i8
  store i8 %176, ptr %_11.i, align 8, !dbg !1832, !alias.scope !1966, !noalias !1863
  br label %bb5.i, !dbg !1969

bb5.i:                                            ; preds = %bb5.i.loopexit, %bb21.i
  %177 = icmp eq i64 %remainder.i, 0, !dbg !1969
  br i1 %177, label %_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack23collect_bool_words_withNCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1g_7visitor5retry21ExecuteDenseWithRetryINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB35_EENtNtB3j_11row_visitor10RowVisitor19visit_deferred_boolB2Y_bKB35_NCINvXB43_B40_NtNtB1g_6row_fn5RowFn8dispatchB3e_E0NCB5V_s_0E0NCB3a_s_0B6H_E0NCINvB2_25collect_bool_words_avx512B17_E0EB43_.exit, label %bb12.i, !dbg !1969

bb12.i:                                           ; preds = %bb5.i
  %178 = and i64 %len, -64, !dbg !1970
  %_11.i.i.i = getelementptr inbounds nuw i8, ptr %f, i64 56
  %_11.i.i.i.promoted = load i8, ptr %_11.i.i.i, align 8
  %179 = trunc nuw i8 %_11.i.i.i.promoted to i1, !dbg !1971
  %_3.i.i = load i64, ptr %f, align 8, !range !1857, !alias.scope !1983, !noalias !1988, !noundef !31
  %180 = trunc nuw i64 %_3.i.i to i1
  %_6.i = getelementptr inbounds nuw i8, ptr %f, i64 24
  %_3.i1.i = load i64, ptr %_6.i, align 8, !range !1857, !alias.scope !1994, !noalias !1988, !noundef !31
  %181 = trunc nuw i64 %_3.i1.i to i1
  %view.i.i = getelementptr inbounds nuw i8, ptr %f, i64 8
  %view.val.i.i = load ptr, ptr %view.i.i, align 8, !nonnull !31, !align !89
  %182 = getelementptr inbounds nuw i8, ptr %f, i64 16
  %view.val1.i.i = load i64, ptr %182, align 8
  %view.i3.i = getelementptr inbounds nuw i8, ptr %f, i64 32
  %view.val.i4.i = load ptr, ptr %view.i3.i, align 8, !nonnull !31, !align !89
  %183 = getelementptr inbounds nuw i8, ptr %f, i64 40
  %view.val1.i5.i = load i64, ptr %183, align 8
  %_6.i11.i.cast = inttoptr i64 %view.val1.i5.i to ptr
  br i1 %180, label %bb12.i.split.us, label %bb12.i.split

bb12.i.split.us:                                  ; preds = %bb12.i
  %184 = inttoptr i64 %view.val1.i.i to ptr
  %_0.sroa.0.0.i.i.us = load i64, ptr %184, align 8, !noalias !1997, !noundef !31
  %185 = icmp eq i64 %_0.sroa.0.0.i.i.us, 9223372036854775807
  br i1 %181, label %iter.check400, label %iter.check335

iter.check335:                                    ; preds = %bb12.i.split.us
  %min.iters.check333 = icmp samesign ult i64 %remainder.i, 4, !dbg !1998
  br i1 %min.iters.check333, label %bb8.i3.us.preheader, label %vector.main.loop.iter.check337, !dbg !1998

vector.main.loop.iter.check337:                   ; preds = %iter.check335
  %min.iters.check336 = icmp samesign ult i64 %remainder.i, 16, !dbg !1998
  br i1 %min.iters.check336, label %vec.epilog.ph371, label %vector.ph338, !dbg !1998

vector.ph338:                                     ; preds = %vector.main.loop.iter.check337
  %n.mod.vf339 = and i64 %len, 12
  %n.vec340 = and i64 %len, 48
  %186 = insertelement <8 x i1> <i1 poison, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false>, i1 %179, i64 0
  %broadcast.splatinsert345 = insertelement <8 x i64> poison, i64 %_0.sroa.0.0.i.i.us, i64 0
  %broadcast.splat346 = shufflevector <8 x i64> %broadcast.splatinsert345, <8 x i64> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert347 = insertelement <8 x i1> poison, i1 %185, i64 0
  %broadcast.splat348 = shufflevector <8 x i1> %broadcast.splatinsert347, <8 x i1> poison, <8 x i32> zeroinitializer
  %invariant.gep580 = getelementptr i64, ptr %view.val.i4.i, i64 %178, !dbg !1998
  br label %vector.body349, !dbg !1998

vector.body349:                                   ; preds = %vector.body349, %vector.ph338
  %index350 = phi i64 [ 0, %vector.ph338 ], [ %index.next359, %vector.body349 ], !dbg !2008
  %vec.phi351 = phi <8 x i1> [ %186, %vector.ph338 ], [ %195, %vector.body349 ]
  %vec.phi352 = phi <8 x i1> [ zeroinitializer, %vector.ph338 ], [ %196, %vector.body349 ]
  %vec.phi353 = phi <8 x i64> [ zeroinitializer, %vector.ph338 ], [ %201, %vector.body349 ]
  %vec.phi354 = phi <8 x i64> [ zeroinitializer, %vector.ph338 ], [ %202, %vector.body349 ]
  %vec.ind355 = phi <8 x i64> [ <i64 0, i64 1, i64 2, i64 3, i64 4, i64 5, i64 6, i64 7>, %vector.ph338 ], [ %vec.ind.next360, %vector.body349 ]
  %step.add356 = add <8 x i64> %vec.ind355, splat (i64 8)
  %187 = extractelement <8 x i64> %vec.ind355, i64 0
  tail call void @llvm.experimental.noalias.scope.decl(metadata !2015), !dbg !2016
  tail call void @llvm.experimental.noalias.scope.decl(metadata !2017), !dbg !2018, !noalias !1988
  tail call void @llvm.experimental.noalias.scope.decl(metadata !2020), !dbg !2018, !noalias !1988
  %gep581 = getelementptr i64, ptr %invariant.gep580, i64 %187, !dbg !2021
  %188 = getelementptr inbounds nuw i8, ptr %gep581, i64 64, !dbg !2026
  %wide.load357 = load <8 x i64>, ptr %gep581, align 8, !dbg !2026, !noalias !2027
  %wide.load358 = load <8 x i64>, ptr %188, align 8, !dbg !2026, !noalias !2027
  %189 = icmp eq <8 x i64> %wide.load357, splat (i64 9223372036854775807), !dbg !2028
  %190 = icmp eq <8 x i64> %wide.load358, splat (i64 9223372036854775807), !dbg !2028
  %191 = icmp slt <8 x i64> %broadcast.splat346, %wide.load357, !dbg !2031
  %192 = icmp slt <8 x i64> %broadcast.splat346, %wide.load358, !dbg !2031
  %193 = or <8 x i1> %189, %vec.phi351, !dbg !1971
  %194 = or <8 x i1> %190, %vec.phi352, !dbg !1971
  %195 = or <8 x i1> %193, %broadcast.splat348, !dbg !1971
  %196 = or <8 x i1> %194, %broadcast.splat348, !dbg !1971
  %197 = zext <8 x i1> %191 to <8 x i64>, !dbg !2032
  %198 = zext <8 x i1> %192 to <8 x i64>, !dbg !2032
  %199 = shl nuw <8 x i64> %197, %vec.ind355, !dbg !2032
  %200 = shl nuw <8 x i64> %198, %step.add356, !dbg !2032
  %201 = or <8 x i64> %199, %vec.phi353, !dbg !2033
  %202 = or <8 x i64> %200, %vec.phi354, !dbg !2033
  %index.next359 = add nuw i64 %index350, 16, !dbg !2008
  %vec.ind.next360 = add <8 x i64> %vec.ind355, splat (i64 16)
  %203 = icmp eq i64 %index.next359, %n.vec340, !dbg !1998
  br i1 %203, label %middle.block361, label %vector.body349, !dbg !1998, !llvm.loop !2034

middle.block361:                                  ; preds = %vector.body349
  %bin.rdx362 = or <8 x i1> %194, %195, !dbg !1998
  %204 = bitcast <8 x i1> %bin.rdx362 to i8, !dbg !1998
  %205 = icmp ne i8 %204, 0, !dbg !1998
  %bin.rdx363 = or <8 x i64> %202, %201, !dbg !1998
  %206 = tail call i64 @llvm.vector.reduce.or.v8i64(<8 x i64> %bin.rdx363), !dbg !1998
  %cmp.n364 = icmp eq i64 %remainder.i, %n.vec340, !dbg !1998
  br i1 %cmp.n364, label %_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack24collect_bool_word_scalarNCINvB2_23collect_bool_words_withNCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1O_7visitor5retry21ExecuteDenseWithRetryINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB3D_EENtNtB3R_11row_visitor10RowVisitor19visit_deferred_boolB3w_bKB3D_NCINvXB4B_B4y_NtNtB1O_6row_fn5RowFn8dispatchB3M_E0NCB6t_s_0E0NCB3I_s_0B7f_E0NCINvB2_25collect_bool_words_avx512B1F_E0E0EB4B_.exit, label %vec.epilog.iter.check369, !dbg !1998

vec.epilog.iter.check369:                         ; preds = %middle.block361
  %min.epilog.iters.check370 = icmp eq i64 %n.mod.vf339, 0
  br i1 %min.epilog.iters.check370, label %bb8.i3.us.preheader, label %vec.epilog.ph371, !prof !2037

vec.epilog.ph371:                                 ; preds = %vector.main.loop.iter.check337, %vec.epilog.iter.check369
  %bc.resume.val365 = phi i64 [ %n.vec340, %vec.epilog.iter.check369 ], [ 0, %vector.main.loop.iter.check337 ]
  %bc.merge.rdx366 = phi i1 [ %205, %vec.epilog.iter.check369 ], [ %179, %vector.main.loop.iter.check337 ], !dbg !2038
  %bc.merge.rdx367 = phi i64 [ %206, %vec.epilog.iter.check369 ], [ 0, %vector.main.loop.iter.check337 ]
  %n.vec373 = and i64 %len, 60
  %207 = insertelement <4 x i1> <i1 poison, i1 false, i1 false, i1 false>, i1 %bc.merge.rdx366, i64 0
  %208 = insertelement <4 x i64> <i64 poison, i64 0, i64 0, i64 0>, i64 %bc.merge.rdx367, i64 0
  %broadcast.splatinsert378 = insertelement <4 x i64> poison, i64 %_0.sroa.0.0.i.i.us, i64 0
  %broadcast.splat379 = shufflevector <4 x i64> %broadcast.splatinsert378, <4 x i64> poison, <4 x i32> zeroinitializer
  %broadcast.splatinsert380 = insertelement <4 x i1> poison, i1 %185, i64 0
  %broadcast.splat381 = shufflevector <4 x i1> %broadcast.splatinsert380, <4 x i1> poison, <4 x i32> zeroinitializer
  %broadcast.splatinsert382 = insertelement <4 x i64> poison, i64 %bc.resume.val365, i64 0
  %broadcast.splat383 = shufflevector <4 x i64> %broadcast.splatinsert382, <4 x i64> poison, <4 x i32> zeroinitializer
  %induction384 = or disjoint <4 x i64> %broadcast.splat383, <i64 0, i64 1, i64 2, i64 3>
  %invariant.gep582 = getelementptr i64, ptr %view.val.i4.i, i64 %178
  br label %vec.epilog.vector.body385

vec.epilog.vector.body385:                        ; preds = %vec.epilog.vector.body385, %vec.epilog.ph371
  %index386 = phi i64 [ %bc.resume.val365, %vec.epilog.ph371 ], [ %index.next391, %vec.epilog.vector.body385 ], !dbg !2008
  %vec.phi387 = phi <4 x i1> [ %207, %vec.epilog.ph371 ], [ %213, %vec.epilog.vector.body385 ]
  %vec.phi388 = phi <4 x i64> [ %208, %vec.epilog.ph371 ], [ %216, %vec.epilog.vector.body385 ]
  %vec.ind389 = phi <4 x i64> [ %induction384, %vec.epilog.ph371 ], [ %vec.ind.next392, %vec.epilog.vector.body385 ]
  %209 = extractelement <4 x i64> %vec.ind389, i64 0
  tail call void @llvm.experimental.noalias.scope.decl(metadata !2015), !dbg !2016
  tail call void @llvm.experimental.noalias.scope.decl(metadata !2017), !dbg !2018, !noalias !1988
  tail call void @llvm.experimental.noalias.scope.decl(metadata !2020), !dbg !2018, !noalias !1988
  %gep583 = getelementptr i64, ptr %invariant.gep582, i64 %209, !dbg !2021
  %wide.load390 = load <4 x i64>, ptr %gep583, align 8, !dbg !2026, !noalias !2027
  %210 = icmp eq <4 x i64> %wide.load390, splat (i64 9223372036854775807), !dbg !2028
  %211 = icmp slt <4 x i64> %broadcast.splat379, %wide.load390, !dbg !2031
  %212 = or <4 x i1> %210, %vec.phi387, !dbg !1971
  %213 = or <4 x i1> %212, %broadcast.splat381, !dbg !1971
  %214 = zext <4 x i1> %211 to <4 x i64>, !dbg !2032
  %215 = shl nuw <4 x i64> %214, %vec.ind389, !dbg !2032
  %216 = or <4 x i64> %215, %vec.phi388, !dbg !2033
  %index.next391 = add nuw i64 %index386, 4, !dbg !2008
  %vec.ind.next392 = add nuw nsw <4 x i64> %vec.ind389, splat (i64 4)
  %217 = icmp eq i64 %index.next391, %n.vec373, !dbg !1998
  br i1 %217, label %vec.epilog.middle.block393, label %vec.epilog.vector.body385, !dbg !1998, !llvm.loop !2039

vec.epilog.middle.block393:                       ; preds = %vec.epilog.vector.body385
  %218 = bitcast <4 x i1> %213 to i4, !dbg !1998
  %219 = icmp ne i4 %218, 0, !dbg !1998
  %220 = tail call i64 @llvm.vector.reduce.or.v4i64(<4 x i64> %216), !dbg !1998
  %cmp.n394 = icmp eq i64 %remainder.i, %n.vec373, !dbg !1998
  br i1 %cmp.n394, label %_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack24collect_bool_word_scalarNCINvB2_23collect_bool_words_withNCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1O_7visitor5retry21ExecuteDenseWithRetryINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB3D_EENtNtB3R_11row_visitor10RowVisitor19visit_deferred_boolB3w_bKB3D_NCINvXB4B_B4y_NtNtB1O_6row_fn5RowFn8dispatchB3M_E0NCB6t_s_0E0NCB3I_s_0B7f_E0NCINvB2_25collect_bool_words_avx512B1F_E0E0EB4B_.exit, label %bb8.i3.us.preheader, !dbg !1998

bb8.i3.us.preheader:                              ; preds = %iter.check335, %vec.epilog.iter.check369, %vec.epilog.middle.block393
  %.ph = phi i1 [ %179, %iter.check335 ], [ %205, %vec.epilog.iter.check369 ], [ %219, %vec.epilog.middle.block393 ]
  %packed.sroa.0.05.i.us.ph = phi i64 [ 0, %iter.check335 ], [ %206, %vec.epilog.iter.check369 ], [ %220, %vec.epilog.middle.block393 ]
  %iter.sroa.0.04.i.us.ph = phi i64 [ 0, %iter.check335 ], [ %n.vec340, %vec.epilog.iter.check369 ], [ %n.vec373, %vec.epilog.middle.block393 ]
  br label %bb8.i3.us, !dbg !1998

iter.check400:                                    ; preds = %bb12.i.split.us
  %_0.sroa.0.0.i9.i.us.us = load i64, ptr %_6.i11.i.cast, align 8, !noalias !2027, !noundef !31
  %_5.i.i.i.i.i.us.us = icmp slt i64 %_0.sroa.0.0.i.i.us, %_0.sroa.0.0.i9.i.us.us
  %_13.i.us.us = zext i1 %_5.i.i.i.i.i.us.us to i64
  %min.iters.check398 = icmp samesign ult i64 %remainder.i, 4, !dbg !1998
  br i1 %min.iters.check398, label %bb8.i3.us.us.preheader, label %vector.main.loop.iter.check402, !dbg !1998

vector.main.loop.iter.check402:                   ; preds = %iter.check400
  %min.iters.check401 = icmp samesign ult i64 %remainder.i, 16, !dbg !1998
  br i1 %min.iters.check401, label %vec.epilog.ph424, label %vector.ph403, !dbg !1998

vector.ph403:                                     ; preds = %vector.main.loop.iter.check402
  %n.mod.vf404 = and i64 %len, 12
  %n.vec405 = and i64 %len, 48
  %broadcast.splatinsert406 = insertelement <8 x i64> poison, i64 %_13.i.us.us, i64 0
  %broadcast.splat407 = shufflevector <8 x i64> %broadcast.splatinsert406, <8 x i64> poison, <8 x i32> zeroinitializer
  br label %vector.body408, !dbg !1998

vector.body408:                                   ; preds = %vector.body408, %vector.ph403
  %index409 = phi i64 [ 0, %vector.ph403 ], [ %index.next414, %vector.body408 ], !dbg !2008
  %vec.phi410 = phi <8 x i64> [ zeroinitializer, %vector.ph403 ], [ %223, %vector.body408 ]
  %vec.phi411 = phi <8 x i64> [ zeroinitializer, %vector.ph403 ], [ %224, %vector.body408 ]
  %vec.ind412 = phi <8 x i64> [ <i64 0, i64 1, i64 2, i64 3, i64 4, i64 5, i64 6, i64 7>, %vector.ph403 ], [ %vec.ind.next415, %vector.body408 ]
  %step.add413 = add <8 x i64> %vec.ind412, splat (i64 8)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !2015), !dbg !2016
  tail call void @llvm.experimental.noalias.scope.decl(metadata !2017), !dbg !2018, !noalias !1988
  tail call void @llvm.experimental.noalias.scope.decl(metadata !2020), !dbg !2018, !noalias !1988
  %221 = shl nuw <8 x i64> %broadcast.splat407, %vec.ind412, !dbg !2032
  %222 = shl nuw <8 x i64> %broadcast.splat407, %step.add413, !dbg !2032
  %223 = or <8 x i64> %221, %vec.phi410, !dbg !2033
  %224 = or <8 x i64> %222, %vec.phi411, !dbg !2033
  %index.next414 = add nuw i64 %index409, 16, !dbg !2008
  %vec.ind.next415 = add <8 x i64> %vec.ind412, splat (i64 16)
  %225 = icmp eq i64 %index.next414, %n.vec405, !dbg !1998
  br i1 %225, label %middle.block416, label %vector.body408, !dbg !1998, !llvm.loop !2040

middle.block416:                                  ; preds = %vector.body408
  %bin.rdx417 = or <8 x i64> %224, %223, !dbg !1998
  %226 = tail call i64 @llvm.vector.reduce.or.v8i64(<8 x i64> %bin.rdx417), !dbg !1998
  %cmp.n418 = icmp eq i64 %remainder.i, %n.vec405, !dbg !1998
  br i1 %cmp.n418, label %_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack24collect_bool_word_scalarNCINvB2_23collect_bool_words_withNCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1O_7visitor5retry21ExecuteDenseWithRetryINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB3D_EENtNtB3R_11row_visitor10RowVisitor19visit_deferred_boolB3w_bKB3D_NCINvXB4B_B4y_NtNtB1O_6row_fn5RowFn8dispatchB3M_E0NCB6t_s_0E0NCB3I_s_0B7f_E0NCINvB2_25collect_bool_words_avx512B1F_E0E0EB4B_.exit.loopexit, label %vec.epilog.iter.check422, !dbg !1998

vec.epilog.iter.check422:                         ; preds = %middle.block416
  %min.epilog.iters.check423 = icmp eq i64 %n.mod.vf404, 0
  br i1 %min.epilog.iters.check423, label %bb8.i3.us.us.preheader, label %vec.epilog.ph424, !prof !2037

vec.epilog.ph424:                                 ; preds = %vector.main.loop.iter.check402, %vec.epilog.iter.check422
  %bc.resume.val419 = phi i64 [ %n.vec405, %vec.epilog.iter.check422 ], [ 0, %vector.main.loop.iter.check402 ]
  %bc.merge.rdx420 = phi i64 [ %226, %vec.epilog.iter.check422 ], [ 0, %vector.main.loop.iter.check402 ]
  %n.vec426 = and i64 %len, 60
  %227 = insertelement <4 x i64> <i64 poison, i64 0, i64 0, i64 0>, i64 %bc.merge.rdx420, i64 0
  %broadcast.splatinsert427 = insertelement <4 x i64> poison, i64 %_13.i.us.us, i64 0
  %broadcast.splat428 = shufflevector <4 x i64> %broadcast.splatinsert427, <4 x i64> poison, <4 x i32> zeroinitializer
  %broadcast.splatinsert429 = insertelement <4 x i64> poison, i64 %bc.resume.val419, i64 0
  %broadcast.splat430 = shufflevector <4 x i64> %broadcast.splatinsert429, <4 x i64> poison, <4 x i32> zeroinitializer
  %induction431 = or disjoint <4 x i64> %broadcast.splat430, <i64 0, i64 1, i64 2, i64 3>
  br label %vec.epilog.vector.body432

vec.epilog.vector.body432:                        ; preds = %vec.epilog.vector.body432, %vec.epilog.ph424
  %index433 = phi i64 [ %bc.resume.val419, %vec.epilog.ph424 ], [ %index.next436, %vec.epilog.vector.body432 ], !dbg !2008
  %vec.phi434 = phi <4 x i64> [ %227, %vec.epilog.ph424 ], [ %229, %vec.epilog.vector.body432 ]
  %vec.ind435 = phi <4 x i64> [ %induction431, %vec.epilog.ph424 ], [ %vec.ind.next437, %vec.epilog.vector.body432 ]
  tail call void @llvm.experimental.noalias.scope.decl(metadata !2015), !dbg !2016
  tail call void @llvm.experimental.noalias.scope.decl(metadata !2017), !dbg !2018, !noalias !1988
  tail call void @llvm.experimental.noalias.scope.decl(metadata !2020), !dbg !2018, !noalias !1988
  %228 = shl nuw <4 x i64> %broadcast.splat428, %vec.ind435, !dbg !2032
  %229 = or <4 x i64> %228, %vec.phi434, !dbg !2033
  %index.next436 = add nuw i64 %index433, 4, !dbg !2008
  %vec.ind.next437 = add nuw nsw <4 x i64> %vec.ind435, splat (i64 4)
  %230 = icmp eq i64 %index.next436, %n.vec426, !dbg !1998
  br i1 %230, label %vec.epilog.middle.block438, label %vec.epilog.vector.body432, !dbg !1998, !llvm.loop !2041

vec.epilog.middle.block438:                       ; preds = %vec.epilog.vector.body432
  %231 = tail call i64 @llvm.vector.reduce.or.v4i64(<4 x i64> %229), !dbg !1998
  %cmp.n439 = icmp eq i64 %remainder.i, %n.vec426, !dbg !1998
  br i1 %cmp.n439, label %_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack24collect_bool_word_scalarNCINvB2_23collect_bool_words_withNCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1O_7visitor5retry21ExecuteDenseWithRetryINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB3D_EENtNtB3R_11row_visitor10RowVisitor19visit_deferred_boolB3w_bKB3D_NCINvXB4B_B4y_NtNtB1O_6row_fn5RowFn8dispatchB3M_E0NCB6t_s_0E0NCB3I_s_0B7f_E0NCINvB2_25collect_bool_words_avx512B1F_E0E0EB4B_.exit.loopexit, label %bb8.i3.us.us.preheader, !dbg !1998

bb8.i3.us.us.preheader:                           ; preds = %iter.check400, %vec.epilog.iter.check422, %vec.epilog.middle.block438
  %packed.sroa.0.05.i.us.us.ph = phi i64 [ 0, %iter.check400 ], [ %226, %vec.epilog.iter.check422 ], [ %231, %vec.epilog.middle.block438 ]
  %iter.sroa.0.04.i.us.us.ph = phi i64 [ 0, %iter.check400 ], [ %n.vec405, %vec.epilog.iter.check422 ], [ %n.vec426, %vec.epilog.middle.block438 ]
  br label %bb8.i3.us.us, !dbg !1998

bb8.i3.us.us:                                     ; preds = %bb8.i3.us.us.preheader, %bb8.i3.us.us
  %packed.sroa.0.05.i.us.us = phi i64 [ %233, %bb8.i3.us.us ], [ %packed.sroa.0.05.i.us.us.ph, %bb8.i3.us.us.preheader ]
  %iter.sroa.0.04.i.us.us = phi i64 [ %232, %bb8.i3.us.us ], [ %iter.sroa.0.04.i.us.us.ph, %bb8.i3.us.us.preheader ]
  tail call void @llvm.experimental.noalias.scope.decl(metadata !2015), !dbg !2016
  tail call void @llvm.experimental.noalias.scope.decl(metadata !2017), !dbg !2018, !noalias !1988
  tail call void @llvm.experimental.noalias.scope.decl(metadata !2020), !dbg !2018, !noalias !1988
  %232 = add nuw nsw i64 %iter.sroa.0.04.i.us.us, 1, !dbg !2008
  %_12.i.us.us = shl nuw i64 %_13.i.us.us, %iter.sroa.0.04.i.us.us, !dbg !2032
  %233 = or i64 %_12.i.us.us, %packed.sroa.0.05.i.us.us, !dbg !2033
  %exitcond.not.i.us.us = icmp eq i64 %232, %remainder.i, !dbg !2042
  br i1 %exitcond.not.i.us.us, label %_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack24collect_bool_word_scalarNCINvB2_23collect_bool_words_withNCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1O_7visitor5retry21ExecuteDenseWithRetryINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB3D_EENtNtB3R_11row_visitor10RowVisitor19visit_deferred_boolB3w_bKB3D_NCINvXB4B_B4y_NtNtB1O_6row_fn5RowFn8dispatchB3M_E0NCB6t_s_0E0NCB3I_s_0B7f_E0NCINvB2_25collect_bool_words_avx512B1F_E0E0EB4B_.exit.loopexit, label %bb8.i3.us.us, !dbg !1998, !llvm.loop !2049

bb8.i3.us:                                        ; preds = %bb8.i3.us.preheader, %bb8.i3.us
  %234 = phi i1 [ %237, %bb8.i3.us ], [ %.ph, %bb8.i3.us.preheader ], !dbg !2038
  %packed.sroa.0.05.i.us = phi i64 [ %239, %bb8.i3.us ], [ %packed.sroa.0.05.i.us.ph, %bb8.i3.us.preheader ]
  %iter.sroa.0.04.i.us = phi i64 [ %238, %bb8.i3.us ], [ %iter.sroa.0.04.i.us.ph, %bb8.i3.us.preheader ]
  %_4.i.i.us = add nuw nsw i64 %iter.sroa.0.04.i.us, %178, !dbg !2038
  tail call void @llvm.experimental.noalias.scope.decl(metadata !2015), !dbg !2016
  tail call void @llvm.experimental.noalias.scope.decl(metadata !2017), !dbg !2018, !noalias !1988
  tail call void @llvm.experimental.noalias.scope.decl(metadata !2020), !dbg !2018, !noalias !1988
  %_5.i.i6.i.us = icmp ult i64 %_4.i.i.us, %view.val1.i5.i, !dbg !2050
  tail call void @llvm.assume(i1 %_5.i.i6.i.us), !dbg !2051, !noalias !1988
  %_4.i.i7.i.us = getelementptr inbounds nuw i64, ptr %view.val.i4.i, i64 %_4.i.i.us, !dbg !2021
  %_0.sroa.0.0.i9.i.us = load i64, ptr %_4.i.i7.i.us, align 8, !dbg !2026, !noalias !2027, !noundef !31
  %235 = icmp eq i64 %_0.sroa.0.0.i9.i.us, 9223372036854775807, !dbg !2028
  %_5.i.i.i.i.i.us = icmp slt i64 %_0.sroa.0.0.i.i.us, %_0.sroa.0.0.i9.i.us, !dbg !2031
  %236 = or i1 %235, %234, !dbg !1971
  %237 = or i1 %236, %185, !dbg !1971
  %238 = add nuw nsw i64 %iter.sroa.0.04.i.us, 1, !dbg !2008
  %_13.i.us = zext i1 %_5.i.i.i.i.i.us to i64, !dbg !2032
  %_12.i.us = shl nuw i64 %_13.i.us, %iter.sroa.0.04.i.us, !dbg !2032
  %239 = or i64 %_12.i.us, %packed.sroa.0.05.i.us, !dbg !2033
  %exitcond.not.i.us = icmp eq i64 %238, %remainder.i, !dbg !2042
  br i1 %exitcond.not.i.us, label %_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack24collect_bool_word_scalarNCINvB2_23collect_bool_words_withNCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1O_7visitor5retry21ExecuteDenseWithRetryINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB3D_EENtNtB3R_11row_visitor10RowVisitor19visit_deferred_boolB3w_bKB3D_NCINvXB4B_B4y_NtNtB1O_6row_fn5RowFn8dispatchB3M_E0NCB6t_s_0E0NCB3I_s_0B7f_E0NCINvB2_25collect_bool_words_avx512B1F_E0E0EB4B_.exit, label %bb8.i3.us, !dbg !1998, !llvm.loop !2052

bb12.i.split:                                     ; preds = %bb12.i
  br i1 %181, label %iter.check270, label %iter.check

iter.check:                                       ; preds = %bb12.i.split
  %min.iters.check = icmp samesign ult i64 %remainder.i, 4, !dbg !1998
  br i1 %min.iters.check, label %bb8.i3.preheader, label %vector.main.loop.iter.check, !dbg !1998

vector.main.loop.iter.check:                      ; preds = %iter.check
  %min.iters.check220 = icmp samesign ult i64 %remainder.i, 16, !dbg !1998
  br i1 %min.iters.check220, label %vec.epilog.ph, label %vector.ph221, !dbg !1998

vector.ph221:                                     ; preds = %vector.main.loop.iter.check
  %n.mod.vf = and i64 %len, 12
  %n.vec = and i64 %len, 48
  %240 = insertelement <8 x i1> <i1 poison, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false>, i1 %179, i64 0
  br label %vector.body228, !dbg !1998

vector.body228:                                   ; preds = %vector.body228, %vector.ph221
  %index229 = phi i64 [ 0, %vector.ph221 ], [ %index.next240, %vector.body228 ], !dbg !2008
  %vec.phi230 = phi <8 x i1> [ %240, %vector.ph221 ], [ %255, %vector.body228 ]
  %vec.phi231 = phi <8 x i1> [ zeroinitializer, %vector.ph221 ], [ %256, %vector.body228 ]
  %vec.phi232 = phi <8 x i64> [ zeroinitializer, %vector.ph221 ], [ %261, %vector.body228 ]
  %vec.phi233 = phi <8 x i64> [ zeroinitializer, %vector.ph221 ], [ %262, %vector.body228 ]
  %vec.ind234 = phi <8 x i64> [ <i64 0, i64 1, i64 2, i64 3, i64 4, i64 5, i64 6, i64 7>, %vector.ph221 ], [ %vec.ind.next241, %vector.body228 ]
  %step.add235 = add <8 x i64> %vec.ind234, splat (i64 8)
  %241 = extractelement <8 x i64> %vec.ind234, i64 0
  %242 = add nuw nsw i64 %241, %178
  tail call void @llvm.experimental.noalias.scope.decl(metadata !2015), !dbg !2016
  tail call void @llvm.experimental.noalias.scope.decl(metadata !2017), !dbg !2018, !noalias !1988
  %243 = getelementptr inbounds nuw i64, ptr %view.val.i.i, i64 %242, !dbg !2053
  %244 = getelementptr inbounds nuw i8, ptr %243, i64 64, !dbg !2058
  %wide.load236 = load <8 x i64>, ptr %243, align 8, !dbg !2058, !noalias !1997
  %wide.load237 = load <8 x i64>, ptr %244, align 8, !dbg !2058, !noalias !1997
  tail call void @llvm.experimental.noalias.scope.decl(metadata !2020), !dbg !2018, !noalias !1988
  %245 = getelementptr inbounds nuw i64, ptr %view.val.i4.i, i64 %242, !dbg !2021
  %246 = getelementptr inbounds nuw i8, ptr %245, i64 64, !dbg !2026
  %wide.load238 = load <8 x i64>, ptr %245, align 8, !dbg !2026, !noalias !2027
  %wide.load239 = load <8 x i64>, ptr %246, align 8, !dbg !2026, !noalias !2027
  %247 = icmp eq <8 x i64> %wide.load236, splat (i64 9223372036854775807), !dbg !2028
  %248 = icmp eq <8 x i64> %wide.load237, splat (i64 9223372036854775807), !dbg !2028
  %249 = icmp eq <8 x i64> %wide.load238, splat (i64 9223372036854775807), !dbg !2028
  %250 = icmp eq <8 x i64> %wide.load239, splat (i64 9223372036854775807), !dbg !2028
  %251 = or <8 x i1> %247, %249, !dbg !2028
  %252 = or <8 x i1> %248, %250, !dbg !2028
  %253 = icmp slt <8 x i64> %wide.load236, %wide.load238, !dbg !2031
  %254 = icmp slt <8 x i64> %wide.load237, %wide.load239, !dbg !2031
  %255 = or <8 x i1> %vec.phi230, %251, !dbg !1971
  %256 = or <8 x i1> %vec.phi231, %252, !dbg !1971
  %257 = zext <8 x i1> %253 to <8 x i64>, !dbg !2032
  %258 = zext <8 x i1> %254 to <8 x i64>, !dbg !2032
  %259 = shl nuw <8 x i64> %257, %vec.ind234, !dbg !2032
  %260 = shl nuw <8 x i64> %258, %step.add235, !dbg !2032
  %261 = or <8 x i64> %259, %vec.phi232, !dbg !2033
  %262 = or <8 x i64> %260, %vec.phi233, !dbg !2033
  %index.next240 = add nuw i64 %index229, 16, !dbg !2008
  %vec.ind.next241 = add <8 x i64> %vec.ind234, splat (i64 16)
  %263 = icmp eq i64 %index.next240, %n.vec, !dbg !1998
  br i1 %263, label %middle.block242, label %vector.body228, !dbg !1998, !llvm.loop !2059

middle.block242:                                  ; preds = %vector.body228
  %bin.rdx243 = or <8 x i1> %256, %255, !dbg !1998
  %264 = bitcast <8 x i1> %bin.rdx243 to i8, !dbg !1998
  %265 = icmp ne i8 %264, 0, !dbg !1998
  %bin.rdx244 = or <8 x i64> %262, %261, !dbg !1998
  %266 = tail call i64 @llvm.vector.reduce.or.v8i64(<8 x i64> %bin.rdx244), !dbg !1998
  %cmp.n = icmp eq i64 %remainder.i, %n.vec, !dbg !1998
  br i1 %cmp.n, label %_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack24collect_bool_word_scalarNCINvB2_23collect_bool_words_withNCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1O_7visitor5retry21ExecuteDenseWithRetryINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB3D_EENtNtB3R_11row_visitor10RowVisitor19visit_deferred_boolB3w_bKB3D_NCINvXB4B_B4y_NtNtB1O_6row_fn5RowFn8dispatchB3M_E0NCB6t_s_0E0NCB3I_s_0B7f_E0NCINvB2_25collect_bool_words_avx512B1F_E0E0EB4B_.exit, label %vec.epilog.iter.check, !dbg !1998

vec.epilog.iter.check:                            ; preds = %middle.block242
  %min.epilog.iters.check = icmp eq i64 %n.mod.vf, 0
  br i1 %min.epilog.iters.check, label %bb8.i3.preheader, label %vec.epilog.ph, !prof !2037

vec.epilog.ph:                                    ; preds = %vector.main.loop.iter.check, %vec.epilog.iter.check
  %bc.resume.val = phi i64 [ %n.vec, %vec.epilog.iter.check ], [ 0, %vector.main.loop.iter.check ]
  %bc.merge.rdx = phi i1 [ %265, %vec.epilog.iter.check ], [ %179, %vector.main.loop.iter.check ], !dbg !2038
  %bc.merge.rdx245 = phi i64 [ %266, %vec.epilog.iter.check ], [ 0, %vector.main.loop.iter.check ]
  %n.vec247 = and i64 %len, 60
  %267 = insertelement <4 x i1> <i1 poison, i1 false, i1 false, i1 false>, i1 %bc.merge.rdx, i64 0
  %268 = insertelement <4 x i64> <i64 poison, i64 0, i64 0, i64 0>, i64 %bc.merge.rdx245, i64 0
  %broadcast.splatinsert254 = insertelement <4 x i64> poison, i64 %bc.resume.val, i64 0
  %broadcast.splat255 = shufflevector <4 x i64> %broadcast.splatinsert254, <4 x i64> poison, <4 x i32> zeroinitializer
  %induction = or disjoint <4 x i64> %broadcast.splat255, <i64 0, i64 1, i64 2, i64 3>
  br label %vec.epilog.vector.body

vec.epilog.vector.body:                           ; preds = %vec.epilog.vector.body, %vec.epilog.ph
  %index256 = phi i64 [ %bc.resume.val, %vec.epilog.ph ], [ %index.next262, %vec.epilog.vector.body ], !dbg !2008
  %vec.phi257 = phi <4 x i1> [ %267, %vec.epilog.ph ], [ %277, %vec.epilog.vector.body ]
  %vec.phi258 = phi <4 x i64> [ %268, %vec.epilog.ph ], [ %280, %vec.epilog.vector.body ]
  %vec.ind259 = phi <4 x i64> [ %induction, %vec.epilog.ph ], [ %vec.ind.next263, %vec.epilog.vector.body ]
  %269 = extractelement <4 x i64> %vec.ind259, i64 0
  %270 = add nuw nsw i64 %269, %178
  tail call void @llvm.experimental.noalias.scope.decl(metadata !2015), !dbg !2016
  tail call void @llvm.experimental.noalias.scope.decl(metadata !2017), !dbg !2018, !noalias !1988
  %271 = getelementptr inbounds nuw i64, ptr %view.val.i.i, i64 %270, !dbg !2053
  %wide.load260 = load <4 x i64>, ptr %271, align 8, !dbg !2058, !noalias !1997
  tail call void @llvm.experimental.noalias.scope.decl(metadata !2020), !dbg !2018, !noalias !1988
  %272 = getelementptr inbounds nuw i64, ptr %view.val.i4.i, i64 %270, !dbg !2021
  %wide.load261 = load <4 x i64>, ptr %272, align 8, !dbg !2026, !noalias !2027
  %273 = icmp eq <4 x i64> %wide.load260, splat (i64 9223372036854775807), !dbg !2028
  %274 = icmp eq <4 x i64> %wide.load261, splat (i64 9223372036854775807), !dbg !2028
  %275 = or <4 x i1> %273, %274, !dbg !2028
  %276 = icmp slt <4 x i64> %wide.load260, %wide.load261, !dbg !2031
  %277 = or <4 x i1> %vec.phi257, %275, !dbg !1971
  %278 = zext <4 x i1> %276 to <4 x i64>, !dbg !2032
  %279 = shl nuw <4 x i64> %278, %vec.ind259, !dbg !2032
  %280 = or <4 x i64> %279, %vec.phi258, !dbg !2033
  %index.next262 = add nuw i64 %index256, 4, !dbg !2008
  %vec.ind.next263 = add nuw nsw <4 x i64> %vec.ind259, splat (i64 4)
  %281 = icmp eq i64 %index.next262, %n.vec247, !dbg !1998
  br i1 %281, label %vec.epilog.middle.block, label %vec.epilog.vector.body, !dbg !1998, !llvm.loop !2060

vec.epilog.middle.block:                          ; preds = %vec.epilog.vector.body
  %282 = bitcast <4 x i1> %277 to i4, !dbg !1998
  %283 = icmp ne i4 %282, 0, !dbg !1998
  %284 = tail call i64 @llvm.vector.reduce.or.v4i64(<4 x i64> %280), !dbg !1998
  %cmp.n264 = icmp eq i64 %remainder.i, %n.vec247, !dbg !1998
  br i1 %cmp.n264, label %_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack24collect_bool_word_scalarNCINvB2_23collect_bool_words_withNCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1O_7visitor5retry21ExecuteDenseWithRetryINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB3D_EENtNtB3R_11row_visitor10RowVisitor19visit_deferred_boolB3w_bKB3D_NCINvXB4B_B4y_NtNtB1O_6row_fn5RowFn8dispatchB3M_E0NCB6t_s_0E0NCB3I_s_0B7f_E0NCINvB2_25collect_bool_words_avx512B1F_E0E0EB4B_.exit, label %bb8.i3.preheader, !dbg !1998

bb8.i3.preheader:                                 ; preds = %iter.check, %vec.epilog.iter.check, %vec.epilog.middle.block
  %.ph466 = phi i1 [ %179, %iter.check ], [ %265, %vec.epilog.iter.check ], [ %283, %vec.epilog.middle.block ]
  %packed.sroa.0.05.i.ph = phi i64 [ 0, %iter.check ], [ %266, %vec.epilog.iter.check ], [ %284, %vec.epilog.middle.block ]
  %iter.sroa.0.04.i.ph = phi i64 [ 0, %iter.check ], [ %n.vec, %vec.epilog.iter.check ], [ %n.vec247, %vec.epilog.middle.block ]
  br label %bb8.i3, !dbg !1998

iter.check270:                                    ; preds = %bb12.i.split
  %_0.sroa.0.0.i9.i.us88 = load i64, ptr %_6.i11.i.cast, align 8, !noalias !2027, !noundef !31
  %285 = icmp eq i64 %_0.sroa.0.0.i9.i.us88, 9223372036854775807
  %min.iters.check268 = icmp samesign ult i64 %remainder.i, 4, !dbg !1998
  br i1 %min.iters.check268, label %bb8.i3.us80.preheader, label %vector.main.loop.iter.check272, !dbg !1998

vector.main.loop.iter.check272:                   ; preds = %iter.check270
  %min.iters.check271 = icmp samesign ult i64 %remainder.i, 16, !dbg !1998
  br i1 %min.iters.check271, label %vec.epilog.ph306, label %vector.ph273, !dbg !1998

vector.ph273:                                     ; preds = %vector.main.loop.iter.check272
  %n.mod.vf274 = and i64 %len, 12
  %n.vec275 = and i64 %len, 48
  %286 = insertelement <8 x i1> <i1 poison, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false>, i1 %179, i64 0
  %broadcast.splatinsert276 = insertelement <8 x i64> poison, i64 %_0.sroa.0.0.i9.i.us88, i64 0
  %broadcast.splat277 = shufflevector <8 x i64> %broadcast.splatinsert276, <8 x i64> poison, <8 x i32> zeroinitializer
  %broadcast.splatinsert278 = insertelement <8 x i1> poison, i1 %285, i64 0
  %broadcast.splat279 = shufflevector <8 x i1> %broadcast.splatinsert278, <8 x i1> poison, <8 x i32> zeroinitializer
  %invariant.gep = getelementptr i64, ptr %view.val.i.i, i64 %178, !dbg !1998
  br label %vector.body284, !dbg !1998

vector.body284:                                   ; preds = %vector.body284, %vector.ph273
  %index285 = phi i64 [ 0, %vector.ph273 ], [ %index.next294, %vector.body284 ], !dbg !2008
  %vec.phi286 = phi <8 x i1> [ %286, %vector.ph273 ], [ %294, %vector.body284 ]
  %vec.phi287 = phi <8 x i1> [ zeroinitializer, %vector.ph273 ], [ %296, %vector.body284 ]
  %vec.phi288 = phi <8 x i64> [ zeroinitializer, %vector.ph273 ], [ %301, %vector.body284 ]
  %vec.phi289 = phi <8 x i64> [ zeroinitializer, %vector.ph273 ], [ %302, %vector.body284 ]
  %vec.ind290 = phi <8 x i64> [ <i64 0, i64 1, i64 2, i64 3, i64 4, i64 5, i64 6, i64 7>, %vector.ph273 ], [ %vec.ind.next295, %vector.body284 ]
  %step.add291 = add <8 x i64> %vec.ind290, splat (i64 8)
  %287 = extractelement <8 x i64> %vec.ind290, i64 0
  tail call void @llvm.experimental.noalias.scope.decl(metadata !2015), !dbg !2016
  tail call void @llvm.experimental.noalias.scope.decl(metadata !2017), !dbg !2018, !noalias !1988
  %gep = getelementptr i64, ptr %invariant.gep, i64 %287, !dbg !2053
  %288 = getelementptr inbounds nuw i8, ptr %gep, i64 64, !dbg !2058
  %wide.load292 = load <8 x i64>, ptr %gep, align 8, !dbg !2058, !noalias !1997
  %wide.load293 = load <8 x i64>, ptr %288, align 8, !dbg !2058, !noalias !1997
  tail call void @llvm.experimental.noalias.scope.decl(metadata !2020), !dbg !2018, !noalias !1988
  %289 = icmp eq <8 x i64> %wide.load292, splat (i64 9223372036854775807), !dbg !2028
  %290 = icmp eq <8 x i64> %wide.load293, splat (i64 9223372036854775807), !dbg !2028
  %291 = icmp slt <8 x i64> %wide.load292, %broadcast.splat277, !dbg !2031
  %292 = icmp slt <8 x i64> %wide.load293, %broadcast.splat277, !dbg !2031
  %293 = or <8 x i1> %289, %vec.phi286, !dbg !1971
  %294 = or <8 x i1> %293, %broadcast.splat279, !dbg !1971
  %295 = or <8 x i1> %290, %vec.phi287, !dbg !1971
  %296 = or <8 x i1> %295, %broadcast.splat279, !dbg !1971
  %297 = zext <8 x i1> %291 to <8 x i64>, !dbg !2032
  %298 = zext <8 x i1> %292 to <8 x i64>, !dbg !2032
  %299 = shl nuw <8 x i64> %297, %vec.ind290, !dbg !2032
  %300 = shl nuw <8 x i64> %298, %step.add291, !dbg !2032
  %301 = or <8 x i64> %299, %vec.phi288, !dbg !2033
  %302 = or <8 x i64> %300, %vec.phi289, !dbg !2033
  %index.next294 = add nuw i64 %index285, 16, !dbg !2008
  %vec.ind.next295 = add <8 x i64> %vec.ind290, splat (i64 16)
  %303 = icmp eq i64 %index.next294, %n.vec275, !dbg !1998
  br i1 %303, label %middle.block296, label %vector.body284, !dbg !1998, !llvm.loop !2061

middle.block296:                                  ; preds = %vector.body284
  %bin.rdx297 = or <8 x i1> %295, %294, !dbg !1998
  %304 = bitcast <8 x i1> %bin.rdx297 to i8, !dbg !1998
  %305 = icmp ne i8 %304, 0, !dbg !1998
  %bin.rdx298 = or <8 x i64> %302, %301, !dbg !1998
  %306 = tail call i64 @llvm.vector.reduce.or.v8i64(<8 x i64> %bin.rdx298), !dbg !1998
  %cmp.n299 = icmp eq i64 %remainder.i, %n.vec275, !dbg !1998
  br i1 %cmp.n299, label %_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack24collect_bool_word_scalarNCINvB2_23collect_bool_words_withNCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1O_7visitor5retry21ExecuteDenseWithRetryINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB3D_EENtNtB3R_11row_visitor10RowVisitor19visit_deferred_boolB3w_bKB3D_NCINvXB4B_B4y_NtNtB1O_6row_fn5RowFn8dispatchB3M_E0NCB6t_s_0E0NCB3I_s_0B7f_E0NCINvB2_25collect_bool_words_avx512B1F_E0E0EB4B_.exit, label %vec.epilog.iter.check304, !dbg !1998

vec.epilog.iter.check304:                         ; preds = %middle.block296
  %min.epilog.iters.check305 = icmp eq i64 %n.mod.vf274, 0
  br i1 %min.epilog.iters.check305, label %bb8.i3.us80.preheader, label %vec.epilog.ph306, !prof !2037

vec.epilog.ph306:                                 ; preds = %vector.main.loop.iter.check272, %vec.epilog.iter.check304
  %bc.resume.val300 = phi i64 [ %n.vec275, %vec.epilog.iter.check304 ], [ 0, %vector.main.loop.iter.check272 ]
  %bc.merge.rdx301 = phi i1 [ %305, %vec.epilog.iter.check304 ], [ %179, %vector.main.loop.iter.check272 ], !dbg !2038
  %bc.merge.rdx302 = phi i64 [ %306, %vec.epilog.iter.check304 ], [ 0, %vector.main.loop.iter.check272 ]
  %n.vec308 = and i64 %len, 60
  %307 = insertelement <4 x i1> <i1 poison, i1 false, i1 false, i1 false>, i1 %bc.merge.rdx301, i64 0
  %308 = insertelement <4 x i64> <i64 poison, i64 0, i64 0, i64 0>, i64 %bc.merge.rdx302, i64 0
  %broadcast.splatinsert309 = insertelement <4 x i64> poison, i64 %_0.sroa.0.0.i9.i.us88, i64 0
  %broadcast.splat310 = shufflevector <4 x i64> %broadcast.splatinsert309, <4 x i64> poison, <4 x i32> zeroinitializer
  %broadcast.splatinsert311 = insertelement <4 x i1> poison, i1 %285, i64 0
  %broadcast.splat312 = shufflevector <4 x i1> %broadcast.splatinsert311, <4 x i1> poison, <4 x i32> zeroinitializer
  %broadcast.splatinsert317 = insertelement <4 x i64> poison, i64 %bc.resume.val300, i64 0
  %broadcast.splat318 = shufflevector <4 x i64> %broadcast.splatinsert317, <4 x i64> poison, <4 x i32> zeroinitializer
  %induction319 = or disjoint <4 x i64> %broadcast.splat318, <i64 0, i64 1, i64 2, i64 3>
  %invariant.gep578 = getelementptr i64, ptr %view.val.i.i, i64 %178
  br label %vec.epilog.vector.body320

vec.epilog.vector.body320:                        ; preds = %vec.epilog.vector.body320, %vec.epilog.ph306
  %index321 = phi i64 [ %bc.resume.val300, %vec.epilog.ph306 ], [ %index.next326, %vec.epilog.vector.body320 ], !dbg !2008
  %vec.phi322 = phi <4 x i1> [ %307, %vec.epilog.ph306 ], [ %313, %vec.epilog.vector.body320 ]
  %vec.phi323 = phi <4 x i64> [ %308, %vec.epilog.ph306 ], [ %316, %vec.epilog.vector.body320 ]
  %vec.ind324 = phi <4 x i64> [ %induction319, %vec.epilog.ph306 ], [ %vec.ind.next327, %vec.epilog.vector.body320 ]
  %309 = extractelement <4 x i64> %vec.ind324, i64 0
  tail call void @llvm.experimental.noalias.scope.decl(metadata !2015), !dbg !2016
  tail call void @llvm.experimental.noalias.scope.decl(metadata !2017), !dbg !2018, !noalias !1988
  %gep579 = getelementptr i64, ptr %invariant.gep578, i64 %309, !dbg !2053
  %wide.load325 = load <4 x i64>, ptr %gep579, align 8, !dbg !2058, !noalias !1997
  tail call void @llvm.experimental.noalias.scope.decl(metadata !2020), !dbg !2018, !noalias !1988
  %310 = icmp eq <4 x i64> %wide.load325, splat (i64 9223372036854775807), !dbg !2028
  %311 = or <4 x i1> %310, %broadcast.splat312, !dbg !2028
  %312 = icmp slt <4 x i64> %wide.load325, %broadcast.splat310, !dbg !2031
  %313 = or <4 x i1> %vec.phi322, %311, !dbg !1971
  %314 = zext <4 x i1> %312 to <4 x i64>, !dbg !2032
  %315 = shl nuw <4 x i64> %314, %vec.ind324, !dbg !2032
  %316 = or <4 x i64> %315, %vec.phi323, !dbg !2033
  %index.next326 = add nuw i64 %index321, 4, !dbg !2008
  %vec.ind.next327 = add nuw nsw <4 x i64> %vec.ind324, splat (i64 4)
  %317 = icmp eq i64 %index.next326, %n.vec308, !dbg !1998
  br i1 %317, label %vec.epilog.middle.block328, label %vec.epilog.vector.body320, !dbg !1998, !llvm.loop !2062

vec.epilog.middle.block328:                       ; preds = %vec.epilog.vector.body320
  %318 = bitcast <4 x i1> %313 to i4, !dbg !1998
  %319 = icmp ne i4 %318, 0, !dbg !1998
  %320 = tail call i64 @llvm.vector.reduce.or.v4i64(<4 x i64> %316), !dbg !1998
  %cmp.n329 = icmp eq i64 %remainder.i, %n.vec308, !dbg !1998
  br i1 %cmp.n329, label %_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack24collect_bool_word_scalarNCINvB2_23collect_bool_words_withNCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1O_7visitor5retry21ExecuteDenseWithRetryINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB3D_EENtNtB3R_11row_visitor10RowVisitor19visit_deferred_boolB3w_bKB3D_NCINvXB4B_B4y_NtNtB1O_6row_fn5RowFn8dispatchB3M_E0NCB6t_s_0E0NCB3I_s_0B7f_E0NCINvB2_25collect_bool_words_avx512B1F_E0E0EB4B_.exit, label %bb8.i3.us80.preheader, !dbg !1998

bb8.i3.us80.preheader:                            ; preds = %iter.check270, %vec.epilog.iter.check304, %vec.epilog.middle.block328
  %.ph456 = phi i1 [ %179, %iter.check270 ], [ %305, %vec.epilog.iter.check304 ], [ %319, %vec.epilog.middle.block328 ]
  %packed.sroa.0.05.i.us81.ph = phi i64 [ 0, %iter.check270 ], [ %306, %vec.epilog.iter.check304 ], [ %320, %vec.epilog.middle.block328 ]
  %iter.sroa.0.04.i.us82.ph = phi i64 [ 0, %iter.check270 ], [ %n.vec275, %vec.epilog.iter.check304 ], [ %n.vec308, %vec.epilog.middle.block328 ]
  br label %bb8.i3.us80, !dbg !1998

bb8.i3.us80:                                      ; preds = %bb8.i3.us80.preheader, %bb8.i3.us80
  %321 = phi i1 [ %324, %bb8.i3.us80 ], [ %.ph456, %bb8.i3.us80.preheader ], !dbg !2038
  %packed.sroa.0.05.i.us81 = phi i64 [ %326, %bb8.i3.us80 ], [ %packed.sroa.0.05.i.us81.ph, %bb8.i3.us80.preheader ]
  %iter.sroa.0.04.i.us82 = phi i64 [ %325, %bb8.i3.us80 ], [ %iter.sroa.0.04.i.us82.ph, %bb8.i3.us80.preheader ]
  %_4.i.i.us83 = add nuw nsw i64 %iter.sroa.0.04.i.us82, %178, !dbg !2038
  tail call void @llvm.experimental.noalias.scope.decl(metadata !2015), !dbg !2016
  tail call void @llvm.experimental.noalias.scope.decl(metadata !2017), !dbg !2018, !noalias !1988
  %_5.i.i.i1.us = icmp ult i64 %_4.i.i.us83, %view.val1.i.i, !dbg !2063
  tail call void @llvm.assume(i1 %_5.i.i.i1.us), !dbg !2064, !noalias !1988
  %_4.i.i.i.us = getelementptr inbounds nuw i64, ptr %view.val.i.i, i64 %_4.i.i.us83, !dbg !2053
  %_0.sroa.0.0.i.i.us84 = load i64, ptr %_4.i.i.i.us, align 8, !dbg !2058, !noalias !1997, !noundef !31
  tail call void @llvm.experimental.noalias.scope.decl(metadata !2020), !dbg !2018, !noalias !1988
  %322 = icmp eq i64 %_0.sroa.0.0.i.i.us84, 9223372036854775807, !dbg !2028
  %_5.i.i.i.i.i.us90 = icmp slt i64 %_0.sroa.0.0.i.i.us84, %_0.sroa.0.0.i9.i.us88, !dbg !2031
  %323 = or i1 %322, %321, !dbg !1971
  %324 = or i1 %323, %285, !dbg !1971
  %325 = add nuw nsw i64 %iter.sroa.0.04.i.us82, 1, !dbg !2008
  %_13.i.us91 = zext i1 %_5.i.i.i.i.i.us90 to i64, !dbg !2032
  %_12.i.us92 = shl nuw i64 %_13.i.us91, %iter.sroa.0.04.i.us82, !dbg !2032
  %326 = or i64 %_12.i.us92, %packed.sroa.0.05.i.us81, !dbg !2033
  %exitcond.not.i.us93 = icmp eq i64 %325, %remainder.i, !dbg !2042
  br i1 %exitcond.not.i.us93, label %_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack24collect_bool_word_scalarNCINvB2_23collect_bool_words_withNCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1O_7visitor5retry21ExecuteDenseWithRetryINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB3D_EENtNtB3R_11row_visitor10RowVisitor19visit_deferred_boolB3w_bKB3D_NCINvXB4B_B4y_NtNtB1O_6row_fn5RowFn8dispatchB3M_E0NCB6t_s_0E0NCB3I_s_0B7f_E0NCINvB2_25collect_bool_words_avx512B1F_E0E0EB4B_.exit, label %bb8.i3.us80, !dbg !1998, !llvm.loop !2065

bb8.i3:                                           ; preds = %bb8.i3.preheader, %bb8.i3
  %327 = phi i1 [ %330, %bb8.i3 ], [ %.ph466, %bb8.i3.preheader ], !dbg !2038
  %packed.sroa.0.05.i = phi i64 [ %332, %bb8.i3 ], [ %packed.sroa.0.05.i.ph, %bb8.i3.preheader ]
  %iter.sroa.0.04.i = phi i64 [ %331, %bb8.i3 ], [ %iter.sroa.0.04.i.ph, %bb8.i3.preheader ]
  %_4.i.i = add nuw nsw i64 %iter.sroa.0.04.i, %178, !dbg !2038
  tail call void @llvm.experimental.noalias.scope.decl(metadata !2015), !dbg !2016
  tail call void @llvm.experimental.noalias.scope.decl(metadata !2017), !dbg !2018, !noalias !1988
  %_5.i.i.i1 = icmp ult i64 %_4.i.i, %view.val1.i.i, !dbg !2063
  tail call void @llvm.assume(i1 %_5.i.i.i1), !dbg !2064, !noalias !1988
  %_4.i.i.i = getelementptr inbounds nuw i64, ptr %view.val.i.i, i64 %_4.i.i, !dbg !2053
  %_0.sroa.0.0.i.i = load i64, ptr %_4.i.i.i, align 8, !dbg !2058, !noalias !1997, !noundef !31
  tail call void @llvm.experimental.noalias.scope.decl(metadata !2020), !dbg !2018, !noalias !1988
  %_5.i.i6.i = icmp ult i64 %_4.i.i, %view.val1.i5.i, !dbg !2050
  tail call void @llvm.assume(i1 %_5.i.i6.i), !dbg !2051, !noalias !1988
  %_4.i.i7.i = getelementptr inbounds nuw i64, ptr %view.val.i4.i, i64 %_4.i.i, !dbg !2021
  %_0.sroa.0.0.i9.i = load i64, ptr %_4.i.i7.i, align 8, !dbg !2026, !noalias !2027, !noundef !31
  %328 = icmp eq i64 %_0.sroa.0.0.i.i, 9223372036854775807, !dbg !2028
  %329 = icmp eq i64 %_0.sroa.0.0.i9.i, 9223372036854775807, !dbg !2028
  %_6.sroa.0.0.i.i.i.i.i = or i1 %328, %329, !dbg !2028
  %_5.i.i.i.i.i = icmp slt i64 %_0.sroa.0.0.i.i, %_0.sroa.0.0.i9.i, !dbg !2031
  %330 = or i1 %327, %_6.sroa.0.0.i.i.i.i.i, !dbg !1971
  %331 = add nuw nsw i64 %iter.sroa.0.04.i, 1, !dbg !2008
  %_13.i = zext i1 %_5.i.i.i.i.i to i64, !dbg !2032
  %_12.i = shl nuw i64 %_13.i, %iter.sroa.0.04.i, !dbg !2032
  %332 = or i64 %_12.i, %packed.sroa.0.05.i, !dbg !2033
  %exitcond.not.i = icmp eq i64 %331, %remainder.i, !dbg !2042
  br i1 %exitcond.not.i, label %_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack24collect_bool_word_scalarNCINvB2_23collect_bool_words_withNCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1O_7visitor5retry21ExecuteDenseWithRetryINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB3D_EENtNtB3R_11row_visitor10RowVisitor19visit_deferred_boolB3w_bKB3D_NCINvXB4B_B4y_NtNtB1O_6row_fn5RowFn8dispatchB3M_E0NCB6t_s_0E0NCB3I_s_0B7f_E0NCINvB2_25collect_bool_words_avx512B1F_E0E0EB4B_.exit, label %bb8.i3, !dbg !1998, !llvm.loop !2066

_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack24collect_bool_word_scalarNCINvB2_23collect_bool_words_withNCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1O_7visitor5retry21ExecuteDenseWithRetryINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB3D_EENtNtB3R_11row_visitor10RowVisitor19visit_deferred_boolB3w_bKB3D_NCINvXB4B_B4y_NtNtB1O_6row_fn5RowFn8dispatchB3M_E0NCB6t_s_0E0NCB3I_s_0B7f_E0NCINvB2_25collect_bool_words_avx512B1F_E0E0EB4B_.exit.loopexit: ; preds = %bb8.i3.us.us, %vec.epilog.middle.block438, %middle.block416
  %.lcssa = phi i64 [ %231, %vec.epilog.middle.block438 ], [ %226, %middle.block416 ], [ %233, %bb8.i3.us.us ], !dbg !2033
  %333 = icmp eq i64 %_0.sroa.0.0.i9.i.us.us, 9223372036854775807
  %_6.sroa.0.0.i.i.i.i.i.us.us = or i1 %185, %333
  %334 = or i1 %_6.sroa.0.0.i.i.i.i.i.us.us, %179, !dbg !1971
  br label %_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack24collect_bool_word_scalarNCINvB2_23collect_bool_words_withNCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1O_7visitor5retry21ExecuteDenseWithRetryINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB3D_EENtNtB3R_11row_visitor10RowVisitor19visit_deferred_boolB3w_bKB3D_NCINvXB4B_B4y_NtNtB1O_6row_fn5RowFn8dispatchB3M_E0NCB6t_s_0E0NCB3I_s_0B7f_E0NCINvB2_25collect_bool_words_avx512B1F_E0E0EB4B_.exit

_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack24collect_bool_word_scalarNCINvB2_23collect_bool_words_withNCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1O_7visitor5retry21ExecuteDenseWithRetryINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB3D_EENtNtB3R_11row_visitor10RowVisitor19visit_deferred_boolB3w_bKB3D_NCINvXB4B_B4y_NtNtB1O_6row_fn5RowFn8dispatchB3M_E0NCB6t_s_0E0NCB3I_s_0B7f_E0NCINvB2_25collect_bool_words_avx512B1F_E0E0EB4B_.exit: ; preds = %bb8.i3, %bb8.i3.us80, %bb8.i3.us, %middle.block242, %vec.epilog.middle.block, %middle.block296, %vec.epilog.middle.block328, %middle.block361, %vec.epilog.middle.block393, %_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack24collect_bool_word_scalarNCINvB2_23collect_bool_words_withNCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1O_7visitor5retry21ExecuteDenseWithRetryINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB3D_EENtNtB3R_11row_visitor10RowVisitor19visit_deferred_boolB3w_bKB3D_NCINvXB4B_B4y_NtNtB1O_6row_fn5RowFn8dispatchB3M_E0NCB6t_s_0E0NCB3I_s_0B7f_E0NCINvB2_25collect_bool_words_avx512B1F_E0E0EB4B_.exit.loopexit
  %.us-phi78 = phi i1 [ %334, %_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack24collect_bool_word_scalarNCINvB2_23collect_bool_words_withNCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1O_7visitor5retry21ExecuteDenseWithRetryINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB3D_EENtNtB3R_11row_visitor10RowVisitor19visit_deferred_boolB3w_bKB3D_NCINvXB4B_B4y_NtNtB1O_6row_fn5RowFn8dispatchB3M_E0NCB6t_s_0E0NCB3I_s_0B7f_E0NCINvB2_25collect_bool_words_avx512B1F_E0E0EB4B_.exit.loopexit ], [ %237, %bb8.i3.us ], [ %324, %bb8.i3.us80 ], [ %219, %vec.epilog.middle.block393 ], [ %205, %middle.block361 ], [ %319, %vec.epilog.middle.block328 ], [ %305, %middle.block296 ], [ %283, %vec.epilog.middle.block ], [ %265, %middle.block242 ], [ %330, %bb8.i3 ]
  %.us-phi79 = phi i64 [ %.lcssa, %_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack24collect_bool_word_scalarNCINvB2_23collect_bool_words_withNCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1O_7visitor5retry21ExecuteDenseWithRetryINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB3D_EENtNtB3R_11row_visitor10RowVisitor19visit_deferred_boolB3w_bKB3D_NCINvXB4B_B4y_NtNtB1O_6row_fn5RowFn8dispatchB3M_E0NCB6t_s_0E0NCB3I_s_0B7f_E0NCINvB2_25collect_bool_words_avx512B1F_E0E0EB4B_.exit.loopexit ], [ %239, %bb8.i3.us ], [ %326, %bb8.i3.us80 ], [ %220, %vec.epilog.middle.block393 ], [ %206, %middle.block361 ], [ %320, %vec.epilog.middle.block328 ], [ %306, %middle.block296 ], [ %284, %vec.epilog.middle.block ], [ %266, %middle.block242 ], [ %332, %bb8.i3 ]
  %335 = zext i1 %.us-phi78 to i8
  store i8 %335, ptr %_11.i.i.i, align 8, !dbg !1971, !alias.scope !2067, !noalias !1988
  %_41.i = icmp samesign ult i64 %full5.i, %words.1, !dbg !2070
  br i1 %_41.i, label %bb14.i, label %panic.i, !dbg !2070

bb14.i:                                           ; preds = %_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack24collect_bool_word_scalarNCINvB2_23collect_bool_words_withNCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1O_7visitor5retry21ExecuteDenseWithRetryINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB3D_EENtNtB3R_11row_visitor10RowVisitor19visit_deferred_boolB3w_bKB3D_NCINvXB4B_B4y_NtNtB1O_6row_fn5RowFn8dispatchB3M_E0NCB6t_s_0E0NCB3I_s_0B7f_E0NCINvB2_25collect_bool_words_avx512B1F_E0E0EB4B_.exit
  store i64 %.us-phi79, ptr %_52.i, align 8, !dbg !2070, !alias.scope !1769, !noalias !2071
  br label %_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack23collect_bool_words_withNCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1g_7visitor5retry21ExecuteDenseWithRetryINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB35_EENtNtB3j_11row_visitor10RowVisitor19visit_deferred_boolB2Y_bKB35_NCINvXB43_B40_NtNtB1g_6row_fn5RowFn8dispatchB3e_E0NCB5V_s_0E0NCB3a_s_0B6H_E0NCINvB2_25collect_bool_words_avx512B17_E0EB43_.exit, !dbg !2073

panic.i:                                          ; preds = %_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack24collect_bool_word_scalarNCINvB2_23collect_bool_words_withNCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1O_7visitor5retry21ExecuteDenseWithRetryINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB3D_EENtNtB3R_11row_visitor10RowVisitor19visit_deferred_boolB3w_bKB3D_NCINvXB4B_B4y_NtNtB1O_6row_fn5RowFn8dispatchB3M_E0NCB6t_s_0E0NCB3I_s_0B7f_E0NCINvB2_25collect_bool_words_avx512B1F_E0E0EB4B_.exit
; call core::panicking::panic_bounds_check
  tail call void @_RNvNtCsc36rpYXAlPq_4core9panicking18panic_bounds_check(i64 noundef %full5.i, i64 noundef range(i64 0, 1152921504606846976) %words.1, ptr noalias nofree noundef readonly align 8 captures(address, read_provenance) dereferenceable(24) @alloc_1b2922caae1b461da04857d3d02eae5b.llvm.7192105927927146806) #21, !dbg !2070
  unreachable

_RINvNtNtCs8GGzvTvD8jM_13vortex_buffer3bit4pack23collect_bool_words_withNCINvNtNtNtNtNtCsh4ToCk7pbZD_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb1_NCINvYINtNtNtB1g_7visitor5retry21ExecuteDenseWithRetryINtCsfoTid9TleBG_17row_fn_bool_retry9PredicateKB35_EENtNtB3j_11row_visitor10RowVisitor19visit_deferred_boolB2Y_bKB35_NCINvXB43_B40_NtNtB1g_6row_fn5RowFn8dispatchB3e_E0NCB5V_s_0E0NCB3a_s_0B6H_E0NCINvB2_25collect_bool_words_avx512B17_E0EB43_.exit: ; preds = %bb14.i, %bb5.i
  ret void, !dbg !2074
}
