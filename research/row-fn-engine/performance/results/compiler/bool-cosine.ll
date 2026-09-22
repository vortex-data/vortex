; SPDX-License-Identifier: Apache-2.0
; SPDX-FileCopyrightText: Copyright the Vortex contributors
; Function excerpt. See provenance.json for the complete artifact.
define hidden void @_RINvNtNtNtNtNtCs5mvLrSLkDP1_12vortex_array9scalar_fn8unstable3row7execute4sink12execute_sinkTINtNtNtCs2rdgasfnGJ9_13vortex_tensor10scalar_fns3row9TensorRowdEB1t_EINtNtB1y_17cosine_similarity13ConstantNormsdEINtNtNtNtB6_5types4sink14uninit_element17UninitElementSinkdENtB3m_18InitializedElementNCINvXs_B2D_NtB2D_16CosineSimilarityNtNtB6_6row_fn5RowFn8dispatchINtNtNtB6_7visitor5retry21ExecuteDenseWithRetryB4T_EEs2_0NCB4J_s3_0ECsaHo96IcALPt_24row_fn_performance_probe(ptr dead_on_unwind noalias nofree noundef writable writeonly sret([80 x i8]) align 8 captures(none) dereferenceable(80) %_0, ptr noundef nonnull %args.0, ptr noalias nofree noundef readonly align 8 captures(address, read_provenance) dereferenceable(48) %args.1, ptr noalias nofree noundef nonnull readonly captures(none) %params, ptr noalias nofree noundef align 8 dereferenceable(40) %ctx) unnamed_addr #0 personality ptr @rust_eh_personality {
start:
  %_3.i.i = alloca [24 x i8], align 8
  %_4.i39 = alloca [24 x i8], align 8
  %_63 = alloca [80 x i8], align 8
  %_44 = alloca [80 x i8], align 8
  %_8 = alloca [120 x i8], align 8
  %_7.sroa.5 = alloca [112 x i8], align 8
  %columns = alloca [112 x i8], align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %columns)
  call void @llvm.lifetime.start.p0(ptr nonnull %_7.sroa.5)
  call void @llvm.lifetime.start.p0(ptr nonnull %_8)
; call <(vortex_tensor::scalar_fns::row::TensorRow<f64>, vortex_tensor::scalar_fns::row::TensorRow<f64>) as vortex_array::scalar_fn::unstable::row::types::element::tuple::element_tuple::ElementTuple>::decode
  call void @_RNvXs4_NtNtNtNtNtNtNtCs5mvLrSLkDP1_12vortex_array9scalar_fn8unstable3row5types7element5tuple13element_tupleTINtNtNtCs2rdgasfnGJ9_13vortex_tensor10scalar_fns3row9TensorRowdEB1I_ENtB5_12ElementTuple6decodeCsaHo96IcALPt_24row_fn_performance_probe(ptr noalias nofree noundef nonnull sret([120 x i8]) align 8 captures(none) dereferenceable(120) %_8, ptr noundef nonnull %args.0, ptr noalias nofree noundef nonnull readonly align 8 captures(address, read_provenance) dereferenceable(48) %args.1, ptr noalias nofree noundef nonnull align 8 dereferenceable(40) %ctx)
  %_84 = load i64, ptr %_8, align 8, !range !104, !noundef !8
  %0 = trunc nuw i64 %_84 to i1
  %1 = getelementptr inbounds nuw i8, ptr %_8, i64 8
  br i1 %0, label %bb66, label %bb67

bb66:                                             ; preds = %start
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(80) %_7.sroa.5, ptr noundef nonnull align 8 dereferenceable(80) %1, i64 80, i1 false)
  call void @llvm.lifetime.end.p0(ptr nonnull %_8)
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(80) %_0, ptr noundef nonnull align 8 dereferenceable(80) %_7.sroa.5, i64 80, i1 false)
  call void @llvm.lifetime.end.p0(ptr nonnull %_7.sroa.5)
  br label %bb47

bb67:                                             ; preds = %start
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(112) %_7.sroa.5, ptr noundef nonnull align 8 dereferenceable(112) %1, i64 112, i1 false)
  call void @llvm.lifetime.end.p0(ptr nonnull %_8)
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(112) %columns, ptr noundef nonnull align 8 dereferenceable(112) %_7.sroa.5, i64 112, i1 false)
  call void @llvm.lifetime.end.p0(ptr nonnull %_7.sroa.5)
  %2 = getelementptr inbounds nuw i8, ptr %args.1, i64 40
  %3 = load ptr, ptr %2, align 8, !invariant.load !8, !nonnull !8
  %4 = invoke noundef i64 %3(ptr noundef nonnull %args.0)
          to label %bb3 unwind label %cleanup9

bb51:                                             ; preds = %bb2.i.i.i98, %bb62, %bb63, %cleanup10, %cleanup9
  %.pn29 = phi { ptr, i32 } [ %5, %cleanup9 ], [ %50, %cleanup10 ], [ %51, %bb63 ], [ %.pn47, %bb62 ], [ %.pn47, %bb2.i.i.i98 ]
; invoke core::ptr::drop_glue::<(vortex_array::scalar_fn::unstable::row::types::element::tuple::element_tuple::ArgColumn<vortex_tensor::scalar_fns::row::TensorRow<half::binary16::f16>>, vortex_array::scalar_fn::unstable::row::types::element::tuple::element_tuple::ArgColumn<vortex_tensor::scalar_fns::row::TensorRow<half::binary16::f16>>)>
  invoke fastcc void @_RINvNtCslWxY2MhVcag_4core3ptr9drop_glueTINtNtNtNtNtNtNtNtCs5mvLrSLkDP1_12vortex_array9scalar_fn8unstable3row5types7element5tuple13element_tuple9ArgColumnINtNtNtCs2rdgasfnGJ9_13vortex_tensor10scalar_fns3row9TensorRowNtNtCsajgNrFuHJYM_4half8binary163f16EEBC_EECsaHo96IcALPt_24row_fn_performance_probe(ptr noalias nofree noundef align 8 dereferenceable(112) %columns) #16
          to label %bb53 unwind label %terminate

cleanup9:                                         ; preds = %bb67
  %5 = landingpad { ptr, i32 }
          cleanup
  br label %bb51

bb3:                                              ; preds = %bb67
  %6 = load ptr, ptr %columns, align 8, !alias.scope !183, !noalias !186, !noundef !8
  %.not = icmp eq ptr %6, null
  %_5.i = getelementptr inbounds nuw i8, ptr %columns, i64 8
  %7 = getelementptr inbounds nuw i8, ptr %columns, i64 16
  %8 = getelementptr inbounds nuw i8, ptr %columns, i64 56
  %9 = load ptr, ptr %8, align 8, !alias.scope !183, !noalias !186, !noundef !8
  %.not120 = icmp eq ptr %9, null
  %_8.i = getelementptr inbounds nuw i8, ptr %columns, i64 64
  %_8.val.i = load ptr, ptr %_8.i, align 8, !alias.scope !183, !noalias !186, !nonnull !8
  %10 = getelementptr inbounds nuw i8, ptr %columns, i64 72
  %_8.val1.i = load i64, ptr %10, align 8, !alias.scope !183, !noalias !186
  br i1 %.not, label %bb4.i, label %bb1.i

bb4.i:                                            ; preds = %bb3
  %_5.val.i = load ptr, ptr %_5.i, align 8, !alias.scope !183, !noalias !186, !nonnull !8
  %_5.val2.i = load i64, ptr %7, align 8, !alias.scope !183, !noalias !186
  %_11.idx.i = shl i64 %_5.val2.i, 3
  %_11.i = getelementptr inbounds nuw i8, ptr %_5.val.i, i64 %_11.idx.i
  %_204.i = icmp eq i64 %_5.val2.i, 0
  br i1 %_204.i, label %bb1.i, label %bb13.i.preheader

bb13.i.preheader:                                 ; preds = %bb4.i
  %11 = add i64 %_11.idx.i, -8
  %12 = lshr exact i64 %11, 3
  %13 = add nuw nsw i64 %12, 1
  %min.iters.check = icmp ult i64 %11, 56
  br i1 %min.iters.check, label %bb13.i.preheader289, label %vector.ph

vector.ph:                                        ; preds = %bb13.i.preheader
  %n.vec = and i64 %13, 4611686018427387896
  %14 = shl i64 %n.vec, 3
  %15 = getelementptr i8, ptr %_5.val.i, i64 %14
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %vec.phi = phi double [ 0.000000e+00, %vector.ph ], [ %26, %vector.body ]
  %offset.idx = shl i64 %index, 3
  %next.gep = getelementptr i8, ptr %_5.val.i, i64 %offset.idx
  %16 = getelementptr i8, ptr %next.gep, i64 16
  %17 = getelementptr i8, ptr %next.gep, i64 32
  %18 = getelementptr i8, ptr %next.gep, i64 48
  %wide.load = load <2 x double>, ptr %next.gep, align 8, !alias.scope !188
  %wide.load130 = load <2 x double>, ptr %16, align 8, !alias.scope !188
  %wide.load131 = load <2 x double>, ptr %17, align 8, !alias.scope !188
  %wide.load132 = load <2 x double>, ptr %18, align 8, !alias.scope !188
  %19 = fmul <2 x double> %wide.load, %wide.load
  %20 = fmul <2 x double> %wide.load130, %wide.load130
  %21 = fmul <2 x double> %wide.load131, %wide.load131
  %22 = fmul <2 x double> %wide.load132, %wide.load132
  %23 = tail call double @llvm.vector.reduce.fadd.v2f64(double %vec.phi, <2 x double> %19)
  %24 = tail call double @llvm.vector.reduce.fadd.v2f64(double %23, <2 x double> %20)
  %25 = tail call double @llvm.vector.reduce.fadd.v2f64(double %24, <2 x double> %21)
  %26 = tail call double @llvm.vector.reduce.fadd.v2f64(double %25, <2 x double> %22)
  %index.next = add nuw i64 %index, 8
  %27 = icmp eq i64 %index.next, %n.vec
  br i1 %27, label %middle.block, label %vector.body, !llvm.loop !191

middle.block:                                     ; preds = %vector.body
  %cmp.n = icmp eq i64 %13, %n.vec
  br i1 %cmp.n, label %bb15.loopexit.i, label %bb13.i.preheader289

bb13.i.preheader289:                              ; preds = %bb13.i.preheader, %middle.block
  %iter1.sroa.0.06.i.ph = phi ptr [ %_5.val.i, %bb13.i.preheader ], [ %15, %middle.block ]
  %sum_squared.sroa.0.05.i.ph = phi double [ 0.000000e+00, %bb13.i.preheader ], [ %26, %middle.block ]
  br label %bb13.i

bb13.i:                                           ; preds = %bb13.i.preheader289, %bb13.i
  %iter1.sroa.0.06.i = phi ptr [ %_26.i, %bb13.i ], [ %iter1.sroa.0.06.i.ph, %bb13.i.preheader289 ]
  %sum_squared.sroa.0.05.i = phi double [ %_0.i2.i, %bb13.i ], [ %sum_squared.sroa.0.05.i.ph, %bb13.i.preheader289 ]
  %_26.i = getelementptr inbounds nuw i8, ptr %iter1.sroa.0.06.i, i64 8
  %value.i = load double, ptr %iter1.sroa.0.06.i, align 8, !alias.scope !188, !noundef !8
  %_0.i.i = fmul double %value.i, %value.i
  %_0.i2.i = fadd double %sum_squared.sroa.0.05.i, %_0.i.i
  %_20.i = icmp eq ptr %_26.i, %_11.i
  br i1 %_20.i, label %bb15.loopexit.i, label %bb13.i, !llvm.loop !194

bb15.loopexit.i:                                  ; preds = %bb13.i, %middle.block
  %_0.i2.i.lcssa = phi double [ %26, %middle.block ], [ %_0.i2.i, %bb13.i ]
  %28 = tail call double @llvm.sqrt.f64(double %_0.i2.i.lcssa)
  br label %bb1.i

bb1.i:                                            ; preds = %bb15.loopexit.i, %bb4.i, %bb3
  %_5.sroa.5.0.i = phi double [ undef, %bb3 ], [ 0.000000e+00, %bb4.i ], [ %28, %bb15.loopexit.i ]
  br i1 %.not120, label %bb8.i, label %bb5

bb8.i:                                            ; preds = %bb1.i
  %_11.idx.i6 = shl i64 %_8.val1.i, 3
  %_11.i7 = getelementptr inbounds nuw i8, ptr %_8.val.i, i64 %_11.idx.i6
  %_204.i8 = icmp eq i64 %_8.val1.i, 0
  br i1 %_204.i8, label %bb5, label %bb13.i9.preheader

bb13.i9.preheader:                                ; preds = %bb8.i
  %29 = add i64 %_11.idx.i6, -8
  %30 = lshr exact i64 %29, 3
  %31 = add nuw nsw i64 %30, 1
  %min.iters.check134 = icmp ult i64 %29, 56
  br i1 %min.iters.check134, label %bb13.i9.preheader286, label %vector.ph135

vector.ph135:                                     ; preds = %bb13.i9.preheader
  %n.vec137 = and i64 %31, 4611686018427387896
  %32 = shl i64 %n.vec137, 3
  %33 = getelementptr i8, ptr %_8.val.i, i64 %32
  br label %vector.body138

vector.body138:                                   ; preds = %vector.body138, %vector.ph135
  %index139 = phi i64 [ 0, %vector.ph135 ], [ %index.next147, %vector.body138 ]
  %vec.phi140 = phi double [ 0.000000e+00, %vector.ph135 ], [ %44, %vector.body138 ]
  %offset.idx141 = shl i64 %index139, 3
  %next.gep142 = getelementptr i8, ptr %_8.val.i, i64 %offset.idx141
  %34 = getelementptr i8, ptr %next.gep142, i64 16
  %35 = getelementptr i8, ptr %next.gep142, i64 32
  %36 = getelementptr i8, ptr %next.gep142, i64 48
  %wide.load143 = load <2 x double>, ptr %next.gep142, align 8, !alias.scope !195
  %wide.load144 = load <2 x double>, ptr %34, align 8, !alias.scope !195
  %wide.load145 = load <2 x double>, ptr %35, align 8, !alias.scope !195
  %wide.load146 = load <2 x double>, ptr %36, align 8, !alias.scope !195
  %37 = fmul <2 x double> %wide.load143, %wide.load143
  %38 = fmul <2 x double> %wide.load144, %wide.load144
  %39 = fmul <2 x double> %wide.load145, %wide.load145
  %40 = fmul <2 x double> %wide.load146, %wide.load146
  %41 = tail call double @llvm.vector.reduce.fadd.v2f64(double %vec.phi140, <2 x double> %37)
  %42 = tail call double @llvm.vector.reduce.fadd.v2f64(double %41, <2 x double> %38)
  %43 = tail call double @llvm.vector.reduce.fadd.v2f64(double %42, <2 x double> %39)
  %44 = tail call double @llvm.vector.reduce.fadd.v2f64(double %43, <2 x double> %40)
  %index.next147 = add nuw i64 %index139, 8
  %45 = icmp eq i64 %index.next147, %n.vec137
  br i1 %45, label %middle.block148, label %vector.body138, !llvm.loop !198

middle.block148:                                  ; preds = %vector.body138
  %cmp.n149 = icmp eq i64 %31, %n.vec137
  br i1 %cmp.n149, label %bb15.loopexit.i17, label %bb13.i9.preheader286

bb13.i9.preheader286:                             ; preds = %bb13.i9.preheader, %middle.block148
  %iter1.sroa.0.06.i10.ph = phi ptr [ %_8.val.i, %bb13.i9.preheader ], [ %33, %middle.block148 ]
  %sum_squared.sroa.0.05.i11.ph = phi double [ 0.000000e+00, %bb13.i9.preheader ], [ %44, %middle.block148 ]
  br label %bb13.i9

bb13.i9:                                          ; preds = %bb13.i9.preheader286, %bb13.i9
  %iter1.sroa.0.06.i10 = phi ptr [ %_26.i12, %bb13.i9 ], [ %iter1.sroa.0.06.i10.ph, %bb13.i9.preheader286 ]
  %sum_squared.sroa.0.05.i11 = phi double [ %_0.i2.i15, %bb13.i9 ], [ %sum_squared.sroa.0.05.i11.ph, %bb13.i9.preheader286 ]
  %_26.i12 = getelementptr inbounds nuw i8, ptr %iter1.sroa.0.06.i10, i64 8
  %value.i13 = load double, ptr %iter1.sroa.0.06.i10, align 8, !alias.scope !195, !noundef !8
  %_0.i.i14 = fmul double %value.i13, %value.i13
  %_0.i2.i15 = fadd double %sum_squared.sroa.0.05.i11, %_0.i.i14
  %_20.i16 = icmp eq ptr %_26.i12, %_11.i7
  br i1 %_20.i16, label %bb15.loopexit.i17, label %bb13.i9, !llvm.loop !199

bb15.loopexit.i17:                                ; preds = %bb13.i9, %middle.block148
  %_0.i2.i15.lcssa = phi double [ %44, %middle.block148 ], [ %_0.i2.i15, %bb13.i9 ]
  %46 = tail call double @llvm.sqrt.f64(double %_0.i2.i15.lcssa)
  br label %bb5

bb5:                                              ; preds = %bb15.loopexit.i17, %bb8.i, %bb1.i
  %_6.sroa.5.0.i = phi double [ undef, %bb1.i ], [ 0.000000e+00, %bb8.i ], [ %46, %bb15.loopexit.i17 ]
  %_34.0.i.i = shl i64 %4, 3
  %_34.1.i.i = icmp ugt i64 %4, 2305843009213693951
  %_39.not.i.i = icmp ugt i64 %_34.0.i.i, 9223372036854775800
  %or.cond.i.i = or i1 %_34.1.i.i, %_39.not.i.i
  br i1 %or.cond.i.i, label %bb3.i, label %bb18.i.i, !prof !120

bb18.i.i:                                         ; preds = %bb5
  %47 = icmp eq i64 %_34.0.i.i, 0
  br i1 %47, label %bb11, label %bb3.i.i

bb3.i.i:                                          ; preds = %bb18.i.i
; call __rustc::__rust_no_alloc_shim_is_unstable_v2
  tail call void @_RNvCs6rREvFdRhLb_7___rustc35___rust_no_alloc_shim_is_unstable_v2() #19, !noalias !200
  %_3.i1.i.i = tail call noalias noundef ptr @mi_malloc_aligned(i64 noundef range(i64 0, -9223372036854775808) %_34.0.i.i, i64 noundef range(i64 1, -9223372036854775807) 8) #19, !noalias !200
  %48 = icmp eq ptr %_3.i1.i.i, null
  br i1 %48, label %bb3.i, label %bb10.i.i

bb10.i.i:                                         ; preds = %bb3.i.i
  %49 = ptrtoint ptr %_3.i1.i.i to i64
  br label %bb11

bb3.i:                                            ; preds = %bb3.i.i, %bb5
  %_8.sroa.4.0.ph.i = phi i64 [ 8, %bb3.i.i ], [ 0, %bb5 ]
; invoke alloc::raw_vec::handle_error
  invoke void @_RNvNtCs6KVRSXc8uZF_5alloc7raw_vec12handle_error(i64 noundef %_8.sroa.4.0.ph.i, i64 %_34.0.i.i) #20
          to label %.noexc unwind label %cleanup10

.noexc:                                           ; preds = %bb3.i
  unreachable

cleanup10:                                        ; preds = %bb3.i
  %50 = landingpad { ptr, i32 }
          cleanup
  br label %bb51

bb63:                                             ; preds = %bb54
  %51 = landingpad { ptr, i32 }
          cleanup
  br label %bb51

cleanup12:                                        ; preds = %bb28
  %52 = landingpad { ptr, i32 }
          cleanup
  br label %bb62

bb11:                                             ; preds = %bb10.i.i, %bb18.i.i
  %_8.sroa.4.0.i = phi i64 [ %4, %bb10.i.i ], [ 0, %bb18.i.i ]
  %_8.sroa.9.0.i = phi i64 [ %49, %bb10.i.i ], [ 8, %bb18.i.i ]
  %53 = inttoptr i64 %_8.sroa.9.0.i to ptr
  %_11.i21 = icmp samesign ule i64 %4, %_8.sroa.4.0.i
  tail call void @llvm.assume(i1 %_11.i21)
  %54 = or i1 %.not, %.not120
  %55 = getelementptr inbounds nuw i8, ptr %columns, i64 32
  %_2.val.i.i.i = load i64, ptr %55, align 8
  %56 = icmp eq i64 %_2.val.i.i.i, %4
  br i1 %54, label %bb24, label %bb13

bb13:                                             ; preds = %bb11
  %57 = getelementptr inbounds nuw i8, ptr %columns, i64 88
  %_2.val.i2.i = load i64, ptr %57, align 8
  %_6.i = icmp eq i64 %_2.val.i2.i, %4
  %or.cond = select i1 %56, i1 %_6.i, i1 false, !prof !126
  br i1 %or.cond, label %bb18, label %bb16, !prof !126

bb24:                                             ; preds = %bb11
  %_0.sroa.0.0.off0.i.i = select i1 %.not, i1 true, i1 %56
  br i1 %_0.sroa.0.0.off0.i.i, label %bb26, label %bb28, !prof !127

cleanup13.loopexit.split-lp:                      ; preds = %bb16
  %lpad.loopexit.split-lp = landingpad { ptr, i32 }
          cleanup
  br label %bb62

bb16:                                             ; preds = %bb13
  call void @llvm.lifetime.start.p0(ptr nonnull %_44)
; invoke vortex_array::scalar_fn::unstable::row::execute::sink::decoded_length_error
  invoke void @_RNvNtNtNtNtNtCs5mvLrSLkDP1_12vortex_array9scalar_fn8unstable3row7execute4sink20decoded_length_error(ptr noalias nofree noundef nonnull sret([80 x i8]) align 8 captures(none) dereferenceable(80) %_44, i64 noundef %4)
          to label %bb17 unwind label %cleanup13.loopexit.split-lp

bb17:                                             ; preds = %bb16
  %58 = load i64, ptr %_44, align 8, !range !128, !noundef !8
  %.not26 = icmp eq i64 %58, -1
  br i1 %.not26, label %bb71, label %bb41

bb71:                                             ; preds = %bb17
  call void @llvm.lifetime.end.p0(ptr nonnull %_44)
  br label %bb18

bb18:                                             ; preds = %bb13, %bb71
  %_11650.not = icmp eq i64 %4, 0
  br i1 %_11650.not, label %bb54, label %bb72.preheader

bb72.preheader:                                   ; preds = %bb18
  %59 = getelementptr inbounds nuw i8, ptr %columns, i64 48
  %60 = getelementptr inbounds nuw i8, ptr %columns, i64 40
  %61 = getelementptr inbounds nuw i8, ptr %columns, i64 104
  %62 = getelementptr inbounds nuw i8, ptr %columns, i64 96
  br label %bb72

bb72:                                             ; preds = %bb72.preheader, %bb75
  %iter.sroa.0.051 = phi i64 [ %63, %bb75 ], [ 0, %bb72.preheader ]
  %63 = add nuw i64 %iter.sroa.0.051, 1
  %_4.i.i = load i64, ptr %59, align 8, !noalias !205, !noundef !8
  %start1.i.i = mul i64 %_4.i.i, %iter.sroa.0.051
  %_12.i.i = load ptr, ptr %columns, align 8, !noalias !205, !nonnull !8, !noundef !8
  %_5.i.i = getelementptr inbounds nuw double, ptr %_12.i.i, i64 %start1.i.i
  %_7.i.i = load i64, ptr %60, align 8, !noalias !205, !noundef !8
  %_4.i1.i = load i64, ptr %61, align 8, !noalias !205, !noundef !8
  %start1.i2.i = mul i64 %_4.i1.i, %iter.sroa.0.051
  %_12.i3.i = load ptr, ptr %8, align 8, !noalias !205, !nonnull !8, !noundef !8
  %_5.i4.i = getelementptr inbounds nuw double, ptr %_12.i3.i, i64 %start1.i2.i
  %_7.i5.i = load i64, ptr %62, align 8, !noalias !205, !noundef !8
  tail call void @llvm.experimental.noalias.scope.decl(metadata !209)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !212)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !214)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !217)
  %_0.sroa.0.0.i.i.i.i = tail call noundef i64 @llvm.umin.i64(i64 range(i64 0, 1152921504606846976) %_7.i5.i, i64 range(i64 0, 1152921504606846976) %_7.i.i)
  %_205.not.i.i.i = icmp eq i64 %_0.sroa.0.0.i.i.i.i, 0
  br i1 %_205.not.i.i.i, label %bb11.i, label %bb14.i.i.i.preheader

bb14.i.i.i.preheader:                             ; preds = %bb72
  %min.iters.check191 = icmp ult i64 %_0.sroa.0.0.i.i.i.i, 8
  br i1 %min.iters.check191, label %bb14.i.i.i.preheader279, label %vector.ph192

vector.ph192:                                     ; preds = %bb14.i.i.i.preheader
  %n.vec194 = and i64 %_0.sroa.0.0.i.i.i.i, -8
  br label %vector.body195

vector.body195:                                   ; preds = %vector.body195, %vector.ph192
  %index196 = phi i64 [ 0, %vector.ph192 ], [ %index.next206, %vector.body195 ]
  %vec.phi197 = phi double [ 0.000000e+00, %vector.ph192 ], [ %79, %vector.body195 ]
  %64 = getelementptr inbounds nuw double, ptr %_5.i.i, i64 %index196
  %65 = getelementptr inbounds nuw double, ptr %_5.i4.i, i64 %index196
  %66 = getelementptr inbounds nuw i8, ptr %64, i64 16
  %67 = getelementptr inbounds nuw i8, ptr %64, i64 32
  %68 = getelementptr inbounds nuw i8, ptr %64, i64 48
  %wide.load198 = load <2 x double>, ptr %64, align 8, !alias.scope !219, !noalias !220
  %wide.load199 = load <2 x double>, ptr %66, align 8, !alias.scope !219, !noalias !220
  %wide.load200 = load <2 x double>, ptr %67, align 8, !alias.scope !219, !noalias !220
  %wide.load201 = load <2 x double>, ptr %68, align 8, !alias.scope !219, !noalias !220
  %69 = getelementptr inbounds nuw i8, ptr %65, i64 16
  %70 = getelementptr inbounds nuw i8, ptr %65, i64 32
  %71 = getelementptr inbounds nuw i8, ptr %65, i64 48
  %wide.load202 = load <2 x double>, ptr %65, align 8, !alias.scope !224, !noalias !225
  %wide.load203 = load <2 x double>, ptr %69, align 8, !alias.scope !224, !noalias !225
  %wide.load204 = load <2 x double>, ptr %70, align 8, !alias.scope !224, !noalias !225
  %wide.load205 = load <2 x double>, ptr %71, align 8, !alias.scope !224, !noalias !225
  %72 = fmul <2 x double> %wide.load198, %wide.load202
  %73 = fmul <2 x double> %wide.load199, %wide.load203
  %74 = fmul <2 x double> %wide.load200, %wide.load204
  %75 = fmul <2 x double> %wide.load201, %wide.load205
  %76 = tail call double @llvm.vector.reduce.fadd.v2f64(double %vec.phi197, <2 x double> %72)
  %77 = tail call double @llvm.vector.reduce.fadd.v2f64(double %76, <2 x double> %73)
  %78 = tail call double @llvm.vector.reduce.fadd.v2f64(double %77, <2 x double> %74)
  %79 = tail call double @llvm.vector.reduce.fadd.v2f64(double %78, <2 x double> %75)
  %index.next206 = add nuw i64 %index196, 8
  %80 = icmp eq i64 %index.next206, %n.vec194
  br i1 %80, label %middle.block207, label %vector.body195, !llvm.loop !226

middle.block207:                                  ; preds = %vector.body195
  %cmp.n208 = icmp eq i64 %_0.sroa.0.0.i.i.i.i, %n.vec194
  br i1 %cmp.n208, label %bb11.i, label %bb14.i.i.i.preheader279

bb14.i.i.i.preheader279:                          ; preds = %bb14.i.i.i.preheader, %middle.block207
  %accum.sroa.0.07.i.i.i.ph = phi double [ 0.000000e+00, %bb14.i.i.i.preheader ], [ %79, %middle.block207 ]
  %iter.sroa.0.06.i.i.i.ph = phi i64 [ 0, %bb14.i.i.i.preheader ], [ %n.vec194, %middle.block207 ]
  br label %bb14.i.i.i

bb14.i.i.i:                                       ; preds = %bb14.i.i.i.preheader279, %bb14.i.i.i
  %accum.sroa.0.07.i.i.i = phi double [ %_0.i.i1.i.i.i.i, %bb14.i.i.i ], [ %accum.sroa.0.07.i.i.i.ph, %bb14.i.i.i.preheader279 ]
  %iter.sroa.0.06.i.i.i = phi i64 [ %_24.i.i.i, %bb14.i.i.i ], [ %iter.sroa.0.06.i.i.i.ph, %bb14.i.i.i.preheader279 ]
  %_24.i.i.i = add nuw nsw i64 %iter.sroa.0.06.i.i.i, 1
  %_3.i.i.i.i.i = getelementptr inbounds nuw double, ptr %_5.i.i, i64 %iter.sroa.0.06.i.i.i
  %_3.i2.i.i.i.i = getelementptr inbounds nuw double, ptr %_5.i4.i, i64 %iter.sroa.0.06.i.i.i
  %_16.0.val.i.i.i = load double, ptr %_3.i.i.i.i.i, align 8, !alias.scope !219, !noalias !220, !noundef !8
  %_16.1.val.i.i.i = load double, ptr %_3.i2.i.i.i.i, align 8, !alias.scope !224, !noalias !225, !noundef !8
  %_0.i.i.i.i.i.i = fmul double %_16.0.val.i.i.i, %_16.1.val.i.i.i
  %_0.i.i1.i.i.i.i = fadd double %accum.sroa.0.07.i.i.i, %_0.i.i.i.i.i.i
  %exitcond.not.i.i.i = icmp eq i64 %_24.i.i.i, %_0.sroa.0.0.i.i.i.i
  br i1 %exitcond.not.i.i.i, label %bb11.i, label %bb14.i.i.i, !llvm.loop !227

bb11.i:                                           ; preds = %bb14.i.i.i, %middle.block207, %bb72
  %accum.sroa.0.0.lcssa.i.i.i = phi double [ 0.000000e+00, %bb72 ], [ %79, %middle.block207 ], [ %_0.i.i1.i.i.i.i, %bb14.i.i.i ]
  %_11.idx.i.i = shl i64 %_7.i.i, 3
  %_11.i.i = getelementptr inbounds nuw i8, ptr %_5.i.i, i64 %_11.idx.i.i
  %_204.i.i = icmp eq i64 %_7.i.i, 0
  br i1 %_204.i.i, label %bb14.i, label %bb13.i.i.preheader

bb13.i.i.preheader:                               ; preds = %bb11.i
  %81 = add i64 %_11.idx.i.i, -8
  %82 = lshr exact i64 %81, 3
  %83 = add nuw nsw i64 %82, 1
  %min.iters.check172 = icmp ult i64 %81, 56
  br i1 %min.iters.check172, label %bb13.i.i.preheader278, label %vector.ph173

vector.ph173:                                     ; preds = %bb13.i.i.preheader
  %n.vec175 = and i64 %83, 4611686018427387896
  %84 = shl i64 %n.vec175, 3
  %85 = getelementptr i8, ptr %_5.i.i, i64 %84
  br label %vector.body176

vector.body176:                                   ; preds = %vector.body176, %vector.ph173
  %index177 = phi i64 [ 0, %vector.ph173 ], [ %index.next185, %vector.body176 ]
  %vec.phi178 = phi double [ 0.000000e+00, %vector.ph173 ], [ %96, %vector.body176 ]
  %offset.idx179 = shl i64 %index177, 3
  %next.gep180 = getelementptr i8, ptr %_5.i.i, i64 %offset.idx179
  %86 = getelementptr i8, ptr %next.gep180, i64 16
  %87 = getelementptr i8, ptr %next.gep180, i64 32
  %88 = getelementptr i8, ptr %next.gep180, i64 48
  %wide.load181 = load <2 x double>, ptr %next.gep180, align 8, !alias.scope !228, !noalias !231
  %wide.load182 = load <2 x double>, ptr %86, align 8, !alias.scope !228, !noalias !231
  %wide.load183 = load <2 x double>, ptr %87, align 8, !alias.scope !228, !noalias !231
  %wide.load184 = load <2 x double>, ptr %88, align 8, !alias.scope !228, !noalias !231
  %89 = fmul <2 x double> %wide.load181, %wide.load181
  %90 = fmul <2 x double> %wide.load182, %wide.load182
  %91 = fmul <2 x double> %wide.load183, %wide.load183
  %92 = fmul <2 x double> %wide.load184, %wide.load184
  %93 = tail call double @llvm.vector.reduce.fadd.v2f64(double %vec.phi178, <2 x double> %89)
  %94 = tail call double @llvm.vector.reduce.fadd.v2f64(double %93, <2 x double> %90)
  %95 = tail call double @llvm.vector.reduce.fadd.v2f64(double %94, <2 x double> %91)
  %96 = tail call double @llvm.vector.reduce.fadd.v2f64(double %95, <2 x double> %92)
  %index.next185 = add nuw i64 %index177, 8
  %97 = icmp eq i64 %index.next185, %n.vec175
  br i1 %97, label %middle.block186, label %vector.body176, !llvm.loop !232

middle.block186:                                  ; preds = %vector.body176
  %cmp.n187 = icmp eq i64 %83, %n.vec175
  br i1 %cmp.n187, label %bb15.loopexit.i.i, label %bb13.i.i.preheader278

bb13.i.i.preheader278:                            ; preds = %bb13.i.i.preheader, %middle.block186
  %iter1.sroa.0.06.i.i.ph = phi ptr [ %_5.i.i, %bb13.i.i.preheader ], [ %85, %middle.block186 ]
  %sum_squared.sroa.0.05.i.i.ph = phi double [ 0.000000e+00, %bb13.i.i.preheader ], [ %96, %middle.block186 ]
  br label %bb13.i.i

bb13.i.i:                                         ; preds = %bb13.i.i.preheader278, %bb13.i.i
  %iter1.sroa.0.06.i.i = phi ptr [ %_26.i.i, %bb13.i.i ], [ %iter1.sroa.0.06.i.i.ph, %bb13.i.i.preheader278 ]
  %sum_squared.sroa.0.05.i.i = phi double [ %_0.i2.i.i, %bb13.i.i ], [ %sum_squared.sroa.0.05.i.i.ph, %bb13.i.i.preheader278 ]
  %_26.i.i = getelementptr inbounds nuw i8, ptr %iter1.sroa.0.06.i.i, i64 8
  %value.i.i = load double, ptr %iter1.sroa.0.06.i.i, align 8, !alias.scope !228, !noalias !231, !noundef !8
  %_0.i.i.i = fmul double %value.i.i, %value.i.i
  %_0.i2.i.i = fadd double %sum_squared.sroa.0.05.i.i, %_0.i.i.i
  %_20.i.i = icmp eq ptr %_26.i.i, %_11.i.i
  br i1 %_20.i.i, label %bb15.loopexit.i.i, label %bb13.i.i, !llvm.loop !233

bb15.loopexit.i.i:                                ; preds = %bb13.i.i, %middle.block186
  %_0.i2.i.i.lcssa = phi double [ %96, %middle.block186 ], [ %_0.i2.i.i, %bb13.i.i ]
  %98 = tail call double @llvm.sqrt.f64(double %_0.i2.i.i.lcssa)
  br label %bb14.i

bb14.i:                                           ; preds = %bb11.i, %bb15.loopexit.i.i
  %lhs_norm.sroa.0.0.i = phi double [ %98, %bb15.loopexit.i.i ], [ 0.000000e+00, %bb11.i ]
  %_11.idx.i1.i = shl i64 %_7.i5.i, 3
  %_11.i2.i = getelementptr inbounds nuw i8, ptr %_5.i4.i, i64 %_11.idx.i1.i
  %_204.i3.i = icmp eq i64 %_7.i5.i, 0
  br i1 %_204.i3.i, label %bb75, label %bb13.i4.i.preheader

bb13.i4.i.preheader:                              ; preds = %bb14.i
  %99 = add i64 %_11.idx.i1.i, -8
  %100 = lshr exact i64 %99, 3
  %101 = add nuw nsw i64 %100, 1
  %min.iters.check153 = icmp ult i64 %99, 56
  br i1 %min.iters.check153, label %bb13.i4.i.preheader277, label %vector.ph154

vector.ph154:                                     ; preds = %bb13.i4.i.preheader
  %n.vec156 = and i64 %101, 4611686018427387896
  %102 = shl i64 %n.vec156, 3
  %103 = getelementptr i8, ptr %_5.i4.i, i64 %102
  br label %vector.body157

vector.body157:                                   ; preds = %vector.body157, %vector.ph154
  %index158 = phi i64 [ 0, %vector.ph154 ], [ %index.next166, %vector.body157 ]
  %vec.phi159 = phi double [ 0.000000e+00, %vector.ph154 ], [ %114, %vector.body157 ]
  %offset.idx160 = shl i64 %index158, 3
  %next.gep161 = getelementptr i8, ptr %_5.i4.i, i64 %offset.idx160
  %104 = getelementptr i8, ptr %next.gep161, i64 16
  %105 = getelementptr i8, ptr %next.gep161, i64 32
  %106 = getelementptr i8, ptr %next.gep161, i64 48
  %wide.load162 = load <2 x double>, ptr %next.gep161, align 8, !alias.scope !234, !noalias !237
  %wide.load163 = load <2 x double>, ptr %104, align 8, !alias.scope !234, !noalias !237
  %wide.load164 = load <2 x double>, ptr %105, align 8, !alias.scope !234, !noalias !237
  %wide.load165 = load <2 x double>, ptr %106, align 8, !alias.scope !234, !noalias !237
  %107 = fmul <2 x double> %wide.load162, %wide.load162
  %108 = fmul <2 x double> %wide.load163, %wide.load163
  %109 = fmul <2 x double> %wide.load164, %wide.load164
  %110 = fmul <2 x double> %wide.load165, %wide.load165
  %111 = tail call double @llvm.vector.reduce.fadd.v2f64(double %vec.phi159, <2 x double> %107)
  %112 = tail call double @llvm.vector.reduce.fadd.v2f64(double %111, <2 x double> %108)
  %113 = tail call double @llvm.vector.reduce.fadd.v2f64(double %112, <2 x double> %109)
  %114 = tail call double @llvm.vector.reduce.fadd.v2f64(double %113, <2 x double> %110)
  %index.next166 = add nuw i64 %index158, 8
  %115 = icmp eq i64 %index.next166, %n.vec156
  br i1 %115, label %middle.block167, label %vector.body157, !llvm.loop !238

middle.block167:                                  ; preds = %vector.body157
  %cmp.n168 = icmp eq i64 %101, %n.vec156
  br i1 %cmp.n168, label %bb15.loopexit.i12.i, label %bb13.i4.i.preheader277

bb13.i4.i.preheader277:                           ; preds = %bb13.i4.i.preheader, %middle.block167
  %iter1.sroa.0.06.i5.i.ph = phi ptr [ %_5.i4.i, %bb13.i4.i.preheader ], [ %103, %middle.block167 ]
  %sum_squared.sroa.0.05.i6.i.ph = phi double [ 0.000000e+00, %bb13.i4.i.preheader ], [ %114, %middle.block167 ]
  br label %bb13.i4.i

bb13.i4.i:                                        ; preds = %bb13.i4.i.preheader277, %bb13.i4.i
  %iter1.sroa.0.06.i5.i = phi ptr [ %_26.i7.i, %bb13.i4.i ], [ %iter1.sroa.0.06.i5.i.ph, %bb13.i4.i.preheader277 ]
  %sum_squared.sroa.0.05.i6.i = phi double [ %_0.i2.i10.i, %bb13.i4.i ], [ %sum_squared.sroa.0.05.i6.i.ph, %bb13.i4.i.preheader277 ]
  %_26.i7.i = getelementptr inbounds nuw i8, ptr %iter1.sroa.0.06.i5.i, i64 8
  %value.i8.i = load double, ptr %iter1.sroa.0.06.i5.i, align 8, !alias.scope !234, !noalias !237, !noundef !8
  %_0.i.i9.i = fmul double %value.i8.i, %value.i8.i
  %_0.i2.i10.i = fadd double %sum_squared.sroa.0.05.i6.i, %_0.i.i9.i
  %_20.i11.i = icmp eq ptr %_26.i7.i, %_11.i2.i
  br i1 %_20.i11.i, label %bb15.loopexit.i12.i, label %bb13.i4.i, !llvm.loop !239

bb15.loopexit.i12.i:                              ; preds = %bb13.i4.i, %middle.block167
  %_0.i2.i10.i.lcssa = phi double [ %114, %middle.block167 ], [ %_0.i2.i10.i, %bb13.i4.i ]
  %116 = tail call double @llvm.sqrt.f64(double %_0.i2.i10.i.lcssa)
  br label %bb75

bb54:                                             ; preds = %bb75, %bb81, %bb18, %bb30
  tail call void @llvm.experimental.noalias.scope.decl(metadata !240)
  call void @llvm.lifetime.start.p0(ptr nonnull %_4.i39), !noalias !243
  store i64 %_8.sroa.4.0.i, ptr %_4.i39, align 8, !noalias !240
  %_79.sroa.4.0._4.i39.sroa_idx = getelementptr inbounds nuw i8, ptr %_4.i39, i64 8
  store ptr %53, ptr %_79.sroa.4.0._4.i39.sroa_idx, align 8, !noalias !240
  %_79.sroa.5.0._4.i39.sroa_idx = getelementptr inbounds nuw i8, ptr %_4.i39, i64 16
  store i64 %4, ptr %_79.sroa.5.0._4.i39.sroa_idx, align 8, !noalias !240
  call void @llvm.lifetime.start.p0(ptr nonnull %_3.i.i), !noalias !245
  store i64 0, ptr %_3.i.i, align 8, !noalias !245
; invoke <vortex_array::array::typed::Array<vortex_array::arrays::primitive::vtable::Primitive>>::new::<f64, alloc::vec::Vec<f64>>
  %117 = invoke { ptr, ptr } @_RINvMs1_NtNtNtCs5mvLrSLkDP1_12vortex_array6arrays9primitive5arrayINtNtNtBc_5array5typed5ArrayNtNtB8_6vtable9PrimitiveE3newdINtNtCs6KVRSXc8uZF_5alloc3vec3VecdEECsaHo96IcALPt_24row_fn_performance_probe(ptr noalias nofree noundef nonnull align 8 captures(address) dereferenceable(24) %_4.i39, ptr noalias nofree noundef nonnull align 8 captures(address) dereferenceable(24) %_3.i.i)
          to label %bb37 unwind label %bb63

bb75:                                             ; preds = %bb15.loopexit.i12.i, %bb14.i
  %rhs_norm.sroa.0.0.i = phi double [ %116, %bb15.loopexit.i12.i ], [ 0.000000e+00, %bb14.i ]
  %_0.i.i36 = fmul double %lhs_norm.sroa.0.0.i, %rhs_norm.sroa.0.0.i
  %_0.i3.i = fcmp oeq double %_0.i.i36, 0.000000e+00
  %_0.i4.i = fdiv double %accum.sroa.0.0.lcssa.i.i.i, %_0.i.i36
  %_0.sroa.0.0.i37 = select i1 %_0.i3.i, double 0.000000e+00, double %_0.i4.i
  %_4.i = getelementptr inbounds nuw double, ptr %53, i64 %iter.sroa.0.051
  store double %_0.sroa.0.0.i37, ptr %_4.i, align 8, !alias.scope !248, !noalias !251
  %exitcond.not = icmp eq i64 %63, %4
  br i1 %exitcond.not, label %bb54, label %bb72

bb41:                                             ; preds = %bb17
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(80) %_0, ptr noundef nonnull align 8 dereferenceable(80) %_44, i64 80, i1 false)
  call void @llvm.lifetime.end.p0(ptr nonnull %_44)
  br label %bb42

bb26:                                             ; preds = %bb24
  %118 = getelementptr inbounds nuw i8, ptr %columns, i64 88
  %_2.val.i.i1.i = load i64, ptr %118, align 8, !alias.scope !254
  %119 = icmp eq i64 %_2.val.i.i1.i, %4
  %_0.sroa.0.0.off0.i2.i = select i1 %.not120, i1 true, i1 %119
  br i1 %_0.sroa.0.0.off0.i2.i, label %bb30, label %bb28, !prof !152

bb28:                                             ; preds = %bb24, %bb26
  call void @llvm.lifetime.start.p0(ptr nonnull %_63)
; invoke vortex_array::scalar_fn::unstable::row::execute::sink::decoded_length_error
  invoke void @_RNvNtNtNtNtNtCs5mvLrSLkDP1_12vortex_array9scalar_fn8unstable3row7execute4sink20decoded_length_error(ptr noalias nofree noundef nonnull sret([80 x i8]) align 8 captures(none) dereferenceable(80) %_63, i64 noundef %4)
          to label %bb29 unwind label %cleanup12

bb29:                                             ; preds = %bb28
  %120 = load i64, ptr %_63, align 8, !range !128, !noundef !8
  %.not24 = icmp eq i64 %120, -1
  br i1 %.not24, label %bb77, label %bb76

bb76:                                             ; preds = %bb29
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(80) %_0, ptr noundef nonnull align 8 dereferenceable(80) %_63, i64 80, i1 false)
  call void @llvm.lifetime.end.p0(ptr nonnull %_63)
  br label %bb42

bb77:                                             ; preds = %bb29
  call void @llvm.lifetime.end.p0(ptr nonnull %_63)
  br label %bb30

bb30:                                             ; preds = %bb77, %bb26
  %_13252.not = icmp eq i64 %4, 0
  br i1 %_13252.not, label %bb54, label %bb32.preheader

bb32.preheader:                                   ; preds = %bb30
  %121 = getelementptr inbounds nuw i8, ptr %columns, i64 48
  %122 = getelementptr inbounds nuw i8, ptr %columns, i64 40
  %123 = getelementptr inbounds nuw i8, ptr %columns, i64 104
  %124 = getelementptr inbounds nuw i8, ptr %columns, i64 96
  br label %bb32

bb37:                                             ; preds = %bb54
  call void @llvm.lifetime.end.p0(ptr nonnull %_3.i.i), !noalias !245
  %_3.0.i = extractvalue { ptr, ptr } %117, 0
  %_3.1.i = extractvalue { ptr, ptr } %117, 1
  call void @llvm.lifetime.end.p0(ptr nonnull %_4.i39), !noalias !243
  %125 = getelementptr inbounds nuw i8, ptr %_0, i64 8
  store ptr %_3.0.i, ptr %125, align 8, !alias.scope !240, !noalias !259
  %126 = getelementptr inbounds nuw i8, ptr %_0, i64 16
  store ptr %_3.1.i, ptr %126, align 8, !alias.scope !240, !noalias !259
  store i64 -1, ptr %_0, align 8, !alias.scope !240, !noalias !259
; call core::ptr::drop_glue::<(vortex_array::scalar_fn::unstable::row::types::element::tuple::element_tuple::ArgColumn<vortex_tensor::scalar_fns::row::TensorRow<half::binary16::f16>>, vortex_array::scalar_fn::unstable::row::types::element::tuple::element_tuple::ArgColumn<vortex_tensor::scalar_fns::row::TensorRow<half::binary16::f16>>)>
  call fastcc void @_RINvNtCslWxY2MhVcag_4core3ptr9drop_glueTINtNtNtNtNtNtNtNtCs5mvLrSLkDP1_12vortex_array9scalar_fn8unstable3row5types7element5tuple13element_tuple9ArgColumnINtNtNtCs2rdgasfnGJ9_13vortex_tensor10scalar_fns3row9TensorRowNtNtCsajgNrFuHJYM_4half8binary163f16EEBC_EECsaHo96IcALPt_24row_fn_performance_probe(ptr noalias nofree noundef align 8 dereferenceable(112) %columns)
  br label %bb47

bb47:                                             ; preds = %bb44, %bb37, %bb66
  call void @llvm.lifetime.end.p0(ptr nonnull %columns)
  ret void

bb32:                                             ; preds = %bb32.preheader, %bb81
  %iter8.sroa.0.053 = phi i64 [ %127, %bb81 ], [ 0, %bb32.preheader ]
  %127 = add nuw i64 %iter8.sroa.0.053, 1
  %_4.i38 = getelementptr inbounds nuw double, ptr %53, i64 %iter8.sroa.0.053
  %128 = load ptr, ptr %columns, align 8, !alias.scope !260, !noalias !265, !noundef !8
  %129 = icmp eq ptr %128, null
  br i1 %129, label %bb2.i.i, label %bb3.i.i41

bb2.i.i:                                          ; preds = %bb32
  %constant.val.i.i = load ptr, ptr %_5.i, align 8, !alias.scope !260, !noalias !265, !nonnull !8, !noundef !8
  %constant.val2.i.i = load i64, ptr %7, align 8, !alias.scope !260, !noalias !265, !noundef !8
  br label %_RNvMNtNtNtNtNtNtNtCs5mvLrSLkDP1_12vortex_array9scalar_fn8unstable3row5types7element5tuple13element_tupleINtB2_9ArgColumnINtNtNtCs2rdgasfnGJ9_13vortex_tensor10scalar_fns3row9TensorRowdEE3getCsaHo96IcALPt_24row_fn_performance_probe.exit.i

bb3.i.i41:                                        ; preds = %bb32
  %_4.i.i.i = load i64, ptr %121, align 8, !alias.scope !267, !noalias !265, !noundef !8
  %start1.i.i.i = mul i64 %_4.i.i.i, %iter8.sroa.0.053
  %_9.i.i.i = load i64, ptr %_5.i, align 8, !alias.scope !267, !noalias !265, !noundef !8
  %_6.i.i.i = load i64, ptr %122, align 8, !alias.scope !267, !noalias !265, !noundef !8
  %_5.i.i.i = add i64 %_6.i.i.i, %start1.i.i.i
  %_15.i.i.i = icmp ult i64 %_5.i.i.i, %start1.i.i.i
  %_11.not.i.i.i = icmp ugt i64 %_5.i.i.i, %_9.i.i.i
  %or.cond.i.i.i = or i1 %_15.i.i.i, %_11.not.i.i.i
  br i1 %or.cond.i.i.i, label %bb2.i.i16.i.invoke, label %_RNvXs_NtNtCs2rdgasfnGJ9_13vortex_tensor10scalar_fns3rowINtB4_9TensorRowdENtNtNtNtNtNtNtCs5mvLrSLkDP1_12vortex_array9scalar_fn8unstable3row5types7element5input12InputElement3getCsaHo96IcALPt_24row_fn_performance_probe.exit.i.i, !prof !164

_RNvXs_NtNtCs2rdgasfnGJ9_13vortex_tensor10scalar_fns3rowINtB4_9TensorRowdENtNtNtNtNtNtNtCs5mvLrSLkDP1_12vortex_array9scalar_fn8unstable3row5types7element5input12InputElement3getCsaHo96IcALPt_24row_fn_performance_probe.exit.i.i: ; preds = %bb3.i.i41
  %_18.i.i.i = getelementptr inbounds nuw double, ptr %128, i64 %start1.i.i.i
  br label %_RNvMNtNtNtNtNtNtNtCs5mvLrSLkDP1_12vortex_array9scalar_fn8unstable3row5types7element5tuple13element_tupleINtB2_9ArgColumnINtNtNtCs2rdgasfnGJ9_13vortex_tensor10scalar_fns3row9TensorRowdEE3getCsaHo96IcALPt_24row_fn_performance_probe.exit.i

_RNvMNtNtNtNtNtNtNtCs5mvLrSLkDP1_12vortex_array9scalar_fn8unstable3row5types7element5tuple13element_tupleINtB2_9ArgColumnINtNtNtCs2rdgasfnGJ9_13vortex_tensor10scalar_fns3row9TensorRowdEE3getCsaHo96IcALPt_24row_fn_performance_probe.exit.i: ; preds = %_RNvXs_NtNtCs2rdgasfnGJ9_13vortex_tensor10scalar_fns3rowINtB4_9TensorRowdENtNtNtNtNtNtNtCs5mvLrSLkDP1_12vortex_array9scalar_fn8unstable3row5types7element5input12InputElement3getCsaHo96IcALPt_24row_fn_performance_probe.exit.i.i, %bb2.i.i
  %constant.val.pn.i.i = phi ptr [ %constant.val.i.i, %bb2.i.i ], [ %_18.i.i.i, %_RNvXs_NtNtCs2rdgasfnGJ9_13vortex_tensor10scalar_fns3rowINtB4_9TensorRowdENtNtNtNtNtNtNtCs5mvLrSLkDP1_12vortex_array9scalar_fn8unstable3row5types7element5input12InputElement3getCsaHo96IcALPt_24row_fn_performance_probe.exit.i.i ]
  %constant.val2.pn.i.i = phi i64 [ %constant.val2.i.i, %bb2.i.i ], [ %_6.i.i.i, %_RNvXs_NtNtCs2rdgasfnGJ9_13vortex_tensor10scalar_fns3rowINtB4_9TensorRowdENtNtNtNtNtNtNtCs5mvLrSLkDP1_12vortex_array9scalar_fn8unstable3row5types7element5input12InputElement3getCsaHo96IcALPt_24row_fn_performance_probe.exit.i.i ]
  %130 = load ptr, ptr %8, align 8, !alias.scope !270, !noalias !265, !noundef !8
  %131 = icmp eq ptr %130, null
  br i1 %131, label %bb2.i17.i, label %bb3.i1.i

bb2.i17.i:                                        ; preds = %_RNvMNtNtNtNtNtNtNtCs5mvLrSLkDP1_12vortex_array9scalar_fn8unstable3row5types7element5tuple13element_tupleINtB2_9ArgColumnINtNtNtCs2rdgasfnGJ9_13vortex_tensor10scalar_fns3row9TensorRowdEE3getCsaHo96IcALPt_24row_fn_performance_probe.exit.i
  %constant.val.i19.i = load ptr, ptr %_8.i, align 8, !alias.scope !270, !noalias !265, !nonnull !8, !noundef !8
  %constant.val2.i20.i = load i64, ptr %10, align 8, !alias.scope !270, !noalias !265, !noundef !8
  br label %bb33

bb3.i1.i:                                         ; preds = %_RNvMNtNtNtNtNtNtNtCs5mvLrSLkDP1_12vortex_array9scalar_fn8unstable3row5types7element5tuple13element_tupleINtB2_9ArgColumnINtNtNtCs2rdgasfnGJ9_13vortex_tensor10scalar_fns3row9TensorRowdEE3getCsaHo96IcALPt_24row_fn_performance_probe.exit.i
  %_4.i.i2.i = load i64, ptr %123, align 8, !alias.scope !273, !noalias !265, !noundef !8
  %start1.i.i3.i = mul i64 %_4.i.i2.i, %iter8.sroa.0.053
  %_9.i.i4.i = load i64, ptr %_8.i, align 8, !alias.scope !273, !noalias !265, !noundef !8
  %_6.i.i5.i = load i64, ptr %124, align 8, !alias.scope !273, !noalias !265, !noundef !8
  %_5.i.i6.i = add i64 %_6.i.i5.i, %start1.i.i3.i
  %_15.i.i7.i = icmp ult i64 %_5.i.i6.i, %start1.i.i3.i
  %_11.not.i.i8.i = icmp ugt i64 %_5.i.i6.i, %_9.i.i4.i
  %or.cond.i.i9.i = or i1 %_15.i.i7.i, %_11.not.i.i8.i
  br i1 %or.cond.i.i9.i, label %bb2.i.i16.i.invoke, label %_RNvXs_NtNtCs2rdgasfnGJ9_13vortex_tensor10scalar_fns3rowINtB4_9TensorRowdENtNtNtNtNtNtNtCs5mvLrSLkDP1_12vortex_array9scalar_fn8unstable3row5types7element5input12InputElement3getCsaHo96IcALPt_24row_fn_performance_probe.exit.i10.i, !prof !164

bb2.i.i16.i.invoke:                               ; preds = %bb3.i1.i, %bb3.i.i41
  %132 = phi i64 [ %start1.i.i.i, %bb3.i.i41 ], [ %start1.i.i3.i, %bb3.i1.i ]
  %133 = phi i64 [ %_5.i.i.i, %bb3.i.i41 ], [ %_5.i.i6.i, %bb3.i1.i ]
  %134 = phi i64 [ %_9.i.i.i, %bb3.i.i41 ], [ %_9.i.i4.i, %bb3.i1.i ]
; invoke core::slice::index::slice_index_fail
  invoke void @_RNvNtNtCslWxY2MhVcag_4core5slice5index16slice_index_fail(i64 noundef %132, i64 noundef %133, i64 noundef %134, ptr noalias nofree noundef readonly align 8 captures(address, read_provenance) dereferenceable(24) @alloc_cd44ca30c046dd08226918654a8231a3.llvm.4918409475768635466) #21
          to label %bb2.i.i16.i.cont unwind label %cleanup17

bb2.i.i16.i.cont:                                 ; preds = %bb2.i.i16.i.invoke
  unreachable

_RNvXs_NtNtCs2rdgasfnGJ9_13vortex_tensor10scalar_fns3rowINtB4_9TensorRowdENtNtNtNtNtNtNtCs5mvLrSLkDP1_12vortex_array9scalar_fn8unstable3row5types7element5input12InputElement3getCsaHo96IcALPt_24row_fn_performance_probe.exit.i10.i: ; preds = %bb3.i1.i
  %_18.i.i11.i = getelementptr inbounds nuw double, ptr %130, i64 %start1.i.i3.i
  br label %bb33

cleanup17:                                        ; preds = %bb2.i.i16.i.invoke
  %135 = landingpad { ptr, i32 }
          cleanup
  br label %bb62

bb33:                                             ; preds = %_RNvXs_NtNtCs2rdgasfnGJ9_13vortex_tensor10scalar_fns3rowINtB4_9TensorRowdENtNtNtNtNtNtNtCs5mvLrSLkDP1_12vortex_array9scalar_fn8unstable3row5types7element5input12InputElement3getCsaHo96IcALPt_24row_fn_performance_probe.exit.i10.i, %bb2.i17.i
  %constant.val.pn.i12.i = phi ptr [ %constant.val.i19.i, %bb2.i17.i ], [ %_18.i.i11.i, %_RNvXs_NtNtCs2rdgasfnGJ9_13vortex_tensor10scalar_fns3rowINtB4_9TensorRowdENtNtNtNtNtNtNtCs5mvLrSLkDP1_12vortex_array9scalar_fn8unstable3row5types7element5input12InputElement3getCsaHo96IcALPt_24row_fn_performance_probe.exit.i10.i ]
  %constant.val2.pn.i13.i = phi i64 [ %constant.val2.i20.i, %bb2.i17.i ], [ %_6.i.i5.i, %_RNvXs_NtNtCs2rdgasfnGJ9_13vortex_tensor10scalar_fns3rowINtB4_9TensorRowdENtNtNtNtNtNtNtCs5mvLrSLkDP1_12vortex_array9scalar_fn8unstable3row5types7element5input12InputElement3getCsaHo96IcALPt_24row_fn_performance_probe.exit.i10.i ]
  tail call void @llvm.experimental.noalias.scope.decl(metadata !276)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !279)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !281)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !284)
  %_0.sroa.0.0.i.i.i.i45 = tail call noundef i64 @llvm.umin.i64(i64 range(i64 0, 1152921504606846976) %constant.val2.pn.i13.i, i64 range(i64 0, 1152921504606846976) %constant.val2.pn.i.i)
  %_205.not.i.i.i46 = icmp eq i64 %_0.sroa.0.0.i.i.i.i45, 0
  br i1 %_205.not.i.i.i46, label %_RINvNtNtCs2rdgasfnGJ9_13vortex_tensor10scalar_fns10arithmetic17inner_product_rowdECsaHo96IcALPt_24row_fn_performance_probe.exit.i58, label %bb14.i.i.i47.preheader

bb14.i.i.i47.preheader:                           ; preds = %bb33
  %min.iters.check250 = icmp ult i64 %_0.sroa.0.0.i.i.i.i45, 8
  br i1 %min.iters.check250, label %bb14.i.i.i47.preheader272, label %vector.ph251

vector.ph251:                                     ; preds = %bb14.i.i.i47.preheader
  %n.vec253 = and i64 %_0.sroa.0.0.i.i.i.i45, -8
  br label %vector.body254

vector.body254:                                   ; preds = %vector.body254, %vector.ph251
  %index255 = phi i64 [ 0, %vector.ph251 ], [ %index.next265, %vector.body254 ]
  %vec.phi256 = phi double [ 0.000000e+00, %vector.ph251 ], [ %151, %vector.body254 ]
  %136 = getelementptr inbounds nuw double, ptr %constant.val.pn.i.i, i64 %index255
  %137 = getelementptr inbounds nuw double, ptr %constant.val.pn.i12.i, i64 %index255
  %138 = getelementptr inbounds nuw i8, ptr %136, i64 16
  %139 = getelementptr inbounds nuw i8, ptr %136, i64 32
  %140 = getelementptr inbounds nuw i8, ptr %136, i64 48
  %wide.load257 = load <2 x double>, ptr %136, align 8, !alias.scope !286, !noalias !287
  %wide.load258 = load <2 x double>, ptr %138, align 8, !alias.scope !286, !noalias !287
  %wide.load259 = load <2 x double>, ptr %139, align 8, !alias.scope !286, !noalias !287
  %wide.load260 = load <2 x double>, ptr %140, align 8, !alias.scope !286, !noalias !287
  %141 = getelementptr inbounds nuw i8, ptr %137, i64 16
  %142 = getelementptr inbounds nuw i8, ptr %137, i64 32
  %143 = getelementptr inbounds nuw i8, ptr %137, i64 48
  %wide.load261 = load <2 x double>, ptr %137, align 8, !alias.scope !291, !noalias !292
  %wide.load262 = load <2 x double>, ptr %141, align 8, !alias.scope !291, !noalias !292
  %wide.load263 = load <2 x double>, ptr %142, align 8, !alias.scope !291, !noalias !292
  %wide.load264 = load <2 x double>, ptr %143, align 8, !alias.scope !291, !noalias !292
  %144 = fmul <2 x double> %wide.load257, %wide.load261
  %145 = fmul <2 x double> %wide.load258, %wide.load262
  %146 = fmul <2 x double> %wide.load259, %wide.load263
  %147 = fmul <2 x double> %wide.load260, %wide.load264
  %148 = tail call double @llvm.vector.reduce.fadd.v2f64(double %vec.phi256, <2 x double> %144)
  %149 = tail call double @llvm.vector.reduce.fadd.v2f64(double %148, <2 x double> %145)
  %150 = tail call double @llvm.vector.reduce.fadd.v2f64(double %149, <2 x double> %146)
  %151 = tail call double @llvm.vector.reduce.fadd.v2f64(double %150, <2 x double> %147)
  %index.next265 = add nuw i64 %index255, 8
  %152 = icmp eq i64 %index.next265, %n.vec253
  br i1 %152, label %middle.block266, label %vector.body254, !llvm.loop !293

middle.block266:                                  ; preds = %vector.body254
  %cmp.n267 = icmp eq i64 %_0.sroa.0.0.i.i.i.i45, %n.vec253
  br i1 %cmp.n267, label %_RINvNtNtCs2rdgasfnGJ9_13vortex_tensor10scalar_fns10arithmetic17inner_product_rowdECsaHo96IcALPt_24row_fn_performance_probe.exit.i58, label %bb14.i.i.i47.preheader272

bb14.i.i.i47.preheader272:                        ; preds = %bb14.i.i.i47.preheader, %middle.block266
  %accum.sroa.0.07.i.i.i48.ph = phi double [ 0.000000e+00, %bb14.i.i.i47.preheader ], [ %151, %middle.block266 ]
  %iter.sroa.0.06.i.i.i49.ph = phi i64 [ 0, %bb14.i.i.i47.preheader ], [ %n.vec253, %middle.block266 ]
  br label %bb14.i.i.i47

bb14.i.i.i47:                                     ; preds = %bb14.i.i.i47.preheader272, %bb14.i.i.i47
  %accum.sroa.0.07.i.i.i48 = phi double [ %_0.i.i1.i.i.i.i56, %bb14.i.i.i47 ], [ %accum.sroa.0.07.i.i.i48.ph, %bb14.i.i.i47.preheader272 ]
  %iter.sroa.0.06.i.i.i49 = phi i64 [ %_24.i.i.i50, %bb14.i.i.i47 ], [ %iter.sroa.0.06.i.i.i49.ph, %bb14.i.i.i47.preheader272 ]
  %_24.i.i.i50 = add nuw nsw i64 %iter.sroa.0.06.i.i.i49, 1
  %_3.i.i.i.i.i51 = getelementptr inbounds nuw double, ptr %constant.val.pn.i.i, i64 %iter.sroa.0.06.i.i.i49
  %_3.i2.i.i.i.i52 = getelementptr inbounds nuw double, ptr %constant.val.pn.i12.i, i64 %iter.sroa.0.06.i.i.i49
  %_16.0.val.i.i.i53 = load double, ptr %_3.i.i.i.i.i51, align 8, !alias.scope !286, !noalias !287, !noundef !8
  %_16.1.val.i.i.i54 = load double, ptr %_3.i2.i.i.i.i52, align 8, !alias.scope !291, !noalias !292, !noundef !8
  %_0.i.i.i.i.i.i55 = fmul double %_16.0.val.i.i.i53, %_16.1.val.i.i.i54
  %_0.i.i1.i.i.i.i56 = fadd double %accum.sroa.0.07.i.i.i48, %_0.i.i.i.i.i.i55
  %exitcond.not.i.i.i57 = icmp eq i64 %_24.i.i.i50, %_0.sroa.0.0.i.i.i.i45
  br i1 %exitcond.not.i.i.i57, label %_RINvNtNtCs2rdgasfnGJ9_13vortex_tensor10scalar_fns10arithmetic17inner_product_rowdECsaHo96IcALPt_24row_fn_performance_probe.exit.i58, label %bb14.i.i.i47, !llvm.loop !294

_RINvNtNtCs2rdgasfnGJ9_13vortex_tensor10scalar_fns10arithmetic17inner_product_rowdECsaHo96IcALPt_24row_fn_performance_probe.exit.i58: ; preds = %bb14.i.i.i47, %middle.block266, %bb33
  %accum.sroa.0.0.lcssa.i.i.i59 = phi double [ 0.000000e+00, %bb33 ], [ %151, %middle.block266 ], [ %_0.i.i1.i.i.i.i56, %bb14.i.i.i47 ]
  br i1 %.not, label %bb9.i73, label %bb11.i60

bb11.i60:                                         ; preds = %_RINvNtNtCs2rdgasfnGJ9_13vortex_tensor10scalar_fns10arithmetic17inner_product_rowdECsaHo96IcALPt_24row_fn_performance_probe.exit.i58
  %_11.idx.i.i61 = shl i64 %constant.val2.pn.i.i, 3
  %_11.i.i62 = getelementptr inbounds nuw i8, ptr %constant.val.pn.i.i, i64 %_11.idx.i.i61
  %_204.i.i63 = icmp eq i64 %constant.val2.pn.i.i, 0
  br i1 %_204.i.i63, label %bb9.i73, label %bb13.i.i64.preheader

bb13.i.i64.preheader:                             ; preds = %bb11.i60
  %153 = add i64 %_11.idx.i.i61, -8
  %154 = lshr exact i64 %153, 3
  %155 = add nuw nsw i64 %154, 1
  %min.iters.check231 = icmp ult i64 %153, 56
  br i1 %min.iters.check231, label %bb13.i.i64.preheader271, label %vector.ph232

vector.ph232:                                     ; preds = %bb13.i.i64.preheader
  %n.vec234 = and i64 %155, 4611686018427387896
  %156 = shl i64 %n.vec234, 3
  %157 = getelementptr i8, ptr %constant.val.pn.i.i, i64 %156
  br label %vector.body235

vector.body235:                                   ; preds = %vector.body235, %vector.ph232
  %index236 = phi i64 [ 0, %vector.ph232 ], [ %index.next244, %vector.body235 ]
  %vec.phi237 = phi double [ 0.000000e+00, %vector.ph232 ], [ %168, %vector.body235 ]
  %offset.idx238 = shl i64 %index236, 3
  %next.gep239 = getelementptr i8, ptr %constant.val.pn.i.i, i64 %offset.idx238
  %158 = getelementptr i8, ptr %next.gep239, i64 16
  %159 = getelementptr i8, ptr %next.gep239, i64 32
  %160 = getelementptr i8, ptr %next.gep239, i64 48
  %wide.load240 = load <2 x double>, ptr %next.gep239, align 8, !alias.scope !295, !noalias !298
  %wide.load241 = load <2 x double>, ptr %158, align 8, !alias.scope !295, !noalias !298
  %wide.load242 = load <2 x double>, ptr %159, align 8, !alias.scope !295, !noalias !298
  %wide.load243 = load <2 x double>, ptr %160, align 8, !alias.scope !295, !noalias !298
  %161 = fmul <2 x double> %wide.load240, %wide.load240
  %162 = fmul <2 x double> %wide.load241, %wide.load241
  %163 = fmul <2 x double> %wide.load242, %wide.load242
  %164 = fmul <2 x double> %wide.load243, %wide.load243
  %165 = tail call double @llvm.vector.reduce.fadd.v2f64(double %vec.phi237, <2 x double> %161)
  %166 = tail call double @llvm.vector.reduce.fadd.v2f64(double %165, <2 x double> %162)
  %167 = tail call double @llvm.vector.reduce.fadd.v2f64(double %166, <2 x double> %163)
  %168 = tail call double @llvm.vector.reduce.fadd.v2f64(double %167, <2 x double> %164)
  %index.next244 = add nuw i64 %index236, 8
  %169 = icmp eq i64 %index.next244, %n.vec234
  br i1 %169, label %middle.block245, label %vector.body235, !llvm.loop !299

middle.block245:                                  ; preds = %vector.body235
  %cmp.n246 = icmp eq i64 %155, %n.vec234
  br i1 %cmp.n246, label %bb15.loopexit.i.i72, label %bb13.i.i64.preheader271

bb13.i.i64.preheader271:                          ; preds = %bb13.i.i64.preheader, %middle.block245
  %iter1.sroa.0.06.i.i65.ph = phi ptr [ %constant.val.pn.i.i, %bb13.i.i64.preheader ], [ %157, %middle.block245 ]
  %sum_squared.sroa.0.05.i.i66.ph = phi double [ 0.000000e+00, %bb13.i.i64.preheader ], [ %168, %middle.block245 ]
  br label %bb13.i.i64

bb13.i.i64:                                       ; preds = %bb13.i.i64.preheader271, %bb13.i.i64
  %iter1.sroa.0.06.i.i65 = phi ptr [ %_26.i.i67, %bb13.i.i64 ], [ %iter1.sroa.0.06.i.i65.ph, %bb13.i.i64.preheader271 ]
  %sum_squared.sroa.0.05.i.i66 = phi double [ %_0.i2.i.i70, %bb13.i.i64 ], [ %sum_squared.sroa.0.05.i.i66.ph, %bb13.i.i64.preheader271 ]
  %_26.i.i67 = getelementptr inbounds nuw i8, ptr %iter1.sroa.0.06.i.i65, i64 8
  %value.i.i68 = load double, ptr %iter1.sroa.0.06.i.i65, align 8, !alias.scope !295, !noalias !298, !noundef !8
  %_0.i.i.i69 = fmul double %value.i.i68, %value.i.i68
  %_0.i2.i.i70 = fadd double %sum_squared.sroa.0.05.i.i66, %_0.i.i.i69
  %_20.i.i71 = icmp eq ptr %_26.i.i67, %_11.i.i62
  br i1 %_20.i.i71, label %bb15.loopexit.i.i72, label %bb13.i.i64, !llvm.loop !300

bb15.loopexit.i.i72:                              ; preds = %bb13.i.i64, %middle.block245
  %_0.i2.i.i70.lcssa = phi double [ %168, %middle.block245 ], [ %_0.i2.i.i70, %bb13.i.i64 ]
  %170 = tail call double @llvm.sqrt.f64(double %_0.i2.i.i70.lcssa)
  br label %bb9.i73

bb9.i73:                                          ; preds = %_RINvNtNtCs2rdgasfnGJ9_13vortex_tensor10scalar_fns10arithmetic17inner_product_rowdECsaHo96IcALPt_24row_fn_performance_probe.exit.i58, %bb15.loopexit.i.i72, %bb11.i60
  %lhs_norm.sroa.0.0.i74 = phi double [ %170, %bb15.loopexit.i.i72 ], [ 0.000000e+00, %bb11.i60 ], [ %_5.sroa.5.0.i, %_RINvNtNtCs2rdgasfnGJ9_13vortex_tensor10scalar_fns10arithmetic17inner_product_rowdECsaHo96IcALPt_24row_fn_performance_probe.exit.i58 ]
  br i1 %.not120, label %bb81, label %bb14.i75

bb14.i75:                                         ; preds = %bb9.i73
  %_11.idx.i1.i76 = shl i64 %constant.val2.pn.i13.i, 3
  %_11.i2.i77 = getelementptr inbounds nuw i8, ptr %constant.val.pn.i12.i, i64 %_11.idx.i1.i76
  %_204.i3.i78 = icmp eq i64 %constant.val2.pn.i13.i, 0
  br i1 %_204.i3.i78, label %bb81, label %bb13.i4.i79.preheader

bb13.i4.i79.preheader:                            ; preds = %bb14.i75
  %171 = add i64 %_11.idx.i1.i76, -8
  %172 = lshr exact i64 %171, 3
  %173 = add nuw nsw i64 %172, 1
  %min.iters.check212 = icmp ult i64 %171, 56
  br i1 %min.iters.check212, label %bb13.i4.i79.preheader270, label %vector.ph213

vector.ph213:                                     ; preds = %bb13.i4.i79.preheader
  %n.vec215 = and i64 %173, 4611686018427387896
  %174 = shl i64 %n.vec215, 3
  %175 = getelementptr i8, ptr %constant.val.pn.i12.i, i64 %174
  br label %vector.body216

vector.body216:                                   ; preds = %vector.body216, %vector.ph213
  %index217 = phi i64 [ 0, %vector.ph213 ], [ %index.next225, %vector.body216 ]
  %vec.phi218 = phi double [ 0.000000e+00, %vector.ph213 ], [ %186, %vector.body216 ]
  %offset.idx219 = shl i64 %index217, 3
  %next.gep220 = getelementptr i8, ptr %constant.val.pn.i12.i, i64 %offset.idx219
  %176 = getelementptr i8, ptr %next.gep220, i64 16
  %177 = getelementptr i8, ptr %next.gep220, i64 32
  %178 = getelementptr i8, ptr %next.gep220, i64 48
  %wide.load221 = load <2 x double>, ptr %next.gep220, align 8, !alias.scope !301, !noalias !304
  %wide.load222 = load <2 x double>, ptr %176, align 8, !alias.scope !301, !noalias !304
  %wide.load223 = load <2 x double>, ptr %177, align 8, !alias.scope !301, !noalias !304
  %wide.load224 = load <2 x double>, ptr %178, align 8, !alias.scope !301, !noalias !304
  %179 = fmul <2 x double> %wide.load221, %wide.load221
  %180 = fmul <2 x double> %wide.load222, %wide.load222
  %181 = fmul <2 x double> %wide.load223, %wide.load223
  %182 = fmul <2 x double> %wide.load224, %wide.load224
  %183 = tail call double @llvm.vector.reduce.fadd.v2f64(double %vec.phi218, <2 x double> %179)
  %184 = tail call double @llvm.vector.reduce.fadd.v2f64(double %183, <2 x double> %180)
  %185 = tail call double @llvm.vector.reduce.fadd.v2f64(double %184, <2 x double> %181)
  %186 = tail call double @llvm.vector.reduce.fadd.v2f64(double %185, <2 x double> %182)
  %index.next225 = add nuw i64 %index217, 8
  %187 = icmp eq i64 %index.next225, %n.vec215
  br i1 %187, label %middle.block226, label %vector.body216, !llvm.loop !305

middle.block226:                                  ; preds = %vector.body216
  %cmp.n227 = icmp eq i64 %173, %n.vec215
  br i1 %cmp.n227, label %bb15.loopexit.i12.i87, label %bb13.i4.i79.preheader270

bb13.i4.i79.preheader270:                         ; preds = %bb13.i4.i79.preheader, %middle.block226
  %iter1.sroa.0.06.i5.i80.ph = phi ptr [ %constant.val.pn.i12.i, %bb13.i4.i79.preheader ], [ %175, %middle.block226 ]
  %sum_squared.sroa.0.05.i6.i81.ph = phi double [ 0.000000e+00, %bb13.i4.i79.preheader ], [ %186, %middle.block226 ]
  br label %bb13.i4.i79

bb13.i4.i79:                                      ; preds = %bb13.i4.i79.preheader270, %bb13.i4.i79
  %iter1.sroa.0.06.i5.i80 = phi ptr [ %_26.i7.i82, %bb13.i4.i79 ], [ %iter1.sroa.0.06.i5.i80.ph, %bb13.i4.i79.preheader270 ]
  %sum_squared.sroa.0.05.i6.i81 = phi double [ %_0.i2.i10.i85, %bb13.i4.i79 ], [ %sum_squared.sroa.0.05.i6.i81.ph, %bb13.i4.i79.preheader270 ]
  %_26.i7.i82 = getelementptr inbounds nuw i8, ptr %iter1.sroa.0.06.i5.i80, i64 8
  %value.i8.i83 = load double, ptr %iter1.sroa.0.06.i5.i80, align 8, !alias.scope !301, !noalias !304, !noundef !8
  %_0.i.i9.i84 = fmul double %value.i8.i83, %value.i8.i83
  %_0.i2.i10.i85 = fadd double %sum_squared.sroa.0.05.i6.i81, %_0.i.i9.i84
  %_20.i11.i86 = icmp eq ptr %_26.i7.i82, %_11.i2.i77
  br i1 %_20.i11.i86, label %bb15.loopexit.i12.i87, label %bb13.i4.i79, !llvm.loop !306

bb15.loopexit.i12.i87:                            ; preds = %bb13.i4.i79, %middle.block226
  %_0.i2.i10.i85.lcssa = phi double [ %186, %middle.block226 ], [ %_0.i2.i10.i85, %bb13.i4.i79 ]
  %188 = tail call double @llvm.sqrt.f64(double %_0.i2.i10.i85.lcssa)
  br label %bb81

bb81:                                             ; preds = %bb15.loopexit.i12.i87, %bb14.i75, %bb9.i73
  %rhs_norm.sroa.0.0.i89 = phi double [ %188, %bb15.loopexit.i12.i87 ], [ 0.000000e+00, %bb14.i75 ], [ %_6.sroa.5.0.i, %bb9.i73 ]
  %_0.i.i90 = fmul double %lhs_norm.sroa.0.0.i74, %rhs_norm.sroa.0.0.i89
  %_0.i3.i91 = fcmp oeq double %_0.i.i90, 0.000000e+00
  %_0.i4.i92 = fdiv double %accum.sroa.0.0.lcssa.i.i.i59, %_0.i.i90
  %_0.sroa.0.0.i93 = select i1 %_0.i3.i91, double 0.000000e+00, double %_0.i4.i92
  store double %_0.sroa.0.0.i93, ptr %_4.i38, align 8, !alias.scope !307, !noalias !310
  %exitcond123.not = icmp eq i64 %127, %4
  br i1 %exitcond123.not, label %bb54, label %bb32

bb42:                                             ; preds = %bb76, %bb41
  %189 = icmp eq i64 %_8.sroa.4.0.i, 0
  br i1 %189, label %bb44, label %bb2.i.i97

bb2.i.i97:                                        ; preds = %bb42
  tail call void @mi_free(ptr noundef nonnull %53) #19, !noalias !313
  br label %bb44

bb44:                                             ; preds = %bb2.i.i97, %bb42
; call core::ptr::drop_glue::<(vortex_array::scalar_fn::unstable::row::types::element::tuple::element_tuple::ArgColumn<vortex_tensor::scalar_fns::row::TensorRow<half::binary16::f16>>, vortex_array::scalar_fn::unstable::row::types::element::tuple::element_tuple::ArgColumn<vortex_tensor::scalar_fns::row::TensorRow<half::binary16::f16>>)>
  call fastcc void @_RINvNtCslWxY2MhVcag_4core3ptr9drop_glueTINtNtNtNtNtNtNtNtCs5mvLrSLkDP1_12vortex_array9scalar_fn8unstable3row5types7element5tuple13element_tuple9ArgColumnINtNtNtCs2rdgasfnGJ9_13vortex_tensor10scalar_fns3row9TensorRowNtNtCsajgNrFuHJYM_4half8binary163f16EEBC_EECsaHo96IcALPt_24row_fn_performance_probe(ptr noalias nofree noundef align 8 dereferenceable(112) %columns)
  br label %bb47

bb62:                                             ; preds = %cleanup17, %cleanup13.loopexit.split-lp, %cleanup12
  %.pn47 = phi { ptr, i32 } [ %lpad.loopexit.split-lp, %cleanup13.loopexit.split-lp ], [ %135, %cleanup17 ], [ %52, %cleanup12 ]
  %190 = icmp eq i64 %_8.sroa.4.0.i, 0
  br i1 %190, label %bb51, label %bb2.i.i.i98

bb2.i.i.i98:                                      ; preds = %bb62
  tail call void @mi_free(ptr noundef nonnull %53) #19, !noalias !316
  br label %bb51

terminate:                                        ; preds = %bb51
  %191 = landingpad { ptr, i32 }
          filter [0 x ptr] zeroinitializer
; call core::panicking::panic_in_cleanup
  call void @_RNvNtCslWxY2MhVcag_4core9panicking16panic_in_cleanup() #17
  unreachable

bb53:                                             ; preds = %bb51
  resume { ptr, i32 } %.pn29
}
