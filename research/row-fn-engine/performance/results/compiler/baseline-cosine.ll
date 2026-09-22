; SPDX-License-Identifier: Apache-2.0
; SPDX-FileCopyrightText: Copyright the Vortex contributors
; Function excerpt. See provenance.json for the complete artifact.
define hidden void @_RINvNtNtNtNtNtCs5mvLrSLkDP1_12vortex_array9scalar_fn8unstable3row7execute4sink12execute_sinkTINtNtNtCs2rdgasfnGJ9_13vortex_tensor10scalar_fns3row9TensorRowdEB1t_EuINtNtNtNtB6_5types4sink14uninit_element17UninitElementSinkdENtB2E_18InitializedElementNCINvYINtNtNtB6_7visitor5retry21ExecuteDenseWithRetryNtNtB1y_17cosine_similarity16CosineSimilarityENtNtB4a_11row_visitor10RowVisitor10visit_intoB1s_B2B_B3z_NCINvXs_B4S_B4Q_NtNtB6_6row_fn5RowFn8dispatchB45_Es0_0E0NCB41_s_0ECsaHo96IcALPt_24row_fn_performance_probe(ptr dead_on_unwind noalias nofree noundef writable writeonly sret([80 x i8]) align 8 captures(none) dereferenceable(80) %_0, ptr noundef nonnull %args.0, ptr noalias nofree noundef readonly align 8 captures(address, read_provenance) dereferenceable(48) %args.1, ptr noalias nofree noundef nonnull readonly captures(none) %params, ptr noalias nofree noundef align 8 dereferenceable(40) %ctx) unnamed_addr #0 personality ptr @rust_eh_personality {
start:
  %_3.i.i = alloca [24 x i8], align 8
  %_4.i15 = alloca [24 x i8], align 8
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

bb51:                                             ; preds = %bb2.i.i.i69, %bb62, %bb63, %cleanup10, %cleanup9
  %.pn29 = phi { ptr, i32 } [ %5, %cleanup9 ], [ %9, %cleanup10 ], [ %10, %bb63 ], [ %.pn48, %bb62 ], [ %.pn48, %bb2.i.i.i69 ]
; invoke core::ptr::drop_glue::<(vortex_array::scalar_fn::unstable::row::types::element::tuple::element_tuple::ArgColumn<vortex_tensor::scalar_fns::row::TensorRow<half::binary16::f16>>, vortex_array::scalar_fn::unstable::row::types::element::tuple::element_tuple::ArgColumn<vortex_tensor::scalar_fns::row::TensorRow<half::binary16::f16>>)>
  invoke fastcc void @_RINvNtCslWxY2MhVcag_4core3ptr9drop_glueTINtNtNtNtNtNtNtNtCs5mvLrSLkDP1_12vortex_array9scalar_fn8unstable3row5types7element5tuple13element_tuple9ArgColumnINtNtNtCs2rdgasfnGJ9_13vortex_tensor10scalar_fns3row9TensorRowNtNtCsajgNrFuHJYM_4half8binary163f16EEBC_EECsaHo96IcALPt_24row_fn_performance_probe(ptr noalias nofree noundef align 8 dereferenceable(112) %columns) #16
          to label %bb53 unwind label %terminate

cleanup9:                                         ; preds = %bb67
  %5 = landingpad { ptr, i32 }
          cleanup
  br label %bb51

bb3:                                              ; preds = %bb67
  %_34.0.i.i = shl i64 %4, 3
  %_34.1.i.i = icmp ugt i64 %4, 2305843009213693951
  %_39.not.i.i = icmp ugt i64 %_34.0.i.i, 9223372036854775800
  %or.cond.i.i = or i1 %_34.1.i.i, %_39.not.i.i
  br i1 %or.cond.i.i, label %bb3.i, label %bb18.i.i, !prof !105

bb18.i.i:                                         ; preds = %bb3
  %6 = icmp eq i64 %_34.0.i.i, 0
  br i1 %6, label %bb11, label %bb3.i.i

bb3.i.i:                                          ; preds = %bb18.i.i
; call __rustc::__rust_no_alloc_shim_is_unstable_v2
  tail call void @_RNvCs6rREvFdRhLb_7___rustc35___rust_no_alloc_shim_is_unstable_v2() #18, !noalias !175
  %_3.i1.i.i = tail call noalias noundef ptr @mi_malloc_aligned(i64 noundef range(i64 0, -9223372036854775808) %_34.0.i.i, i64 noundef range(i64 1, -9223372036854775807) 8) #18, !noalias !175
  %7 = icmp eq ptr %_3.i1.i.i, null
  br i1 %7, label %bb3.i, label %bb10.i.i

bb10.i.i:                                         ; preds = %bb3.i.i
  %8 = ptrtoint ptr %_3.i1.i.i to i64
  br label %bb11

bb3.i:                                            ; preds = %bb3.i.i, %bb3
  %_8.sroa.4.0.ph.i = phi i64 [ 8, %bb3.i.i ], [ 0, %bb3 ]
; invoke alloc::raw_vec::handle_error
  invoke void @_RNvNtCs6KVRSXc8uZF_5alloc7raw_vec12handle_error(i64 noundef %_8.sroa.4.0.ph.i, i64 %_34.0.i.i) #19
          to label %.noexc unwind label %cleanup10

.noexc:                                           ; preds = %bb3.i
  unreachable

cleanup10:                                        ; preds = %bb3.i
  %9 = landingpad { ptr, i32 }
          cleanup
  br label %bb51

bb63:                                             ; preds = %bb54
  %10 = landingpad { ptr, i32 }
          cleanup
  br label %bb51

cleanup12:                                        ; preds = %bb28
  %11 = landingpad { ptr, i32 }
          cleanup
  br label %bb62

bb11:                                             ; preds = %bb10.i.i, %bb18.i.i
  %_8.sroa.4.0.i = phi i64 [ %4, %bb10.i.i ], [ 0, %bb18.i.i ]
  %_8.sroa.9.0.i = phi i64 [ %8, %bb10.i.i ], [ 8, %bb18.i.i ]
  %12 = inttoptr i64 %_8.sroa.9.0.i to ptr
  %_11.i = icmp samesign ule i64 %4, %_8.sroa.4.0.i
  tail call void @llvm.assume(i1 %_11.i)
  %13 = load ptr, ptr %columns, align 8, !alias.scope !180, !noundef !8
  %14 = icmp eq ptr %13, null
  %15 = getelementptr inbounds nuw i8, ptr %columns, i64 56
  %16 = load ptr, ptr %15, align 8, !alias.scope !180
  %17 = icmp eq ptr %16, null
  %18 = select i1 %14, i1 true, i1 %17
  %19 = getelementptr inbounds nuw i8, ptr %columns, i64 32
  %_2.val.i.i.i = load i64, ptr %19, align 8
  %20 = icmp eq i64 %_2.val.i.i.i, %4
  br i1 %18, label %bb24, label %bb13

bb13:                                             ; preds = %bb11
  %21 = getelementptr inbounds nuw i8, ptr %columns, i64 88
  %_2.val.i2.i = load i64, ptr %21, align 8
  %_6.i = icmp eq i64 %_2.val.i2.i, %4
  %or.cond = select i1 %20, i1 %_6.i, i1 false, !prof !114
  br i1 %or.cond, label %bb18, label %bb16, !prof !114

bb24:                                             ; preds = %bb11
  %_0.sroa.0.0.off0.i.i = select i1 %14, i1 true, i1 %20
  br i1 %_0.sroa.0.0.off0.i.i, label %bb26, label %bb28, !prof !115

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
  %22 = load i64, ptr %_44, align 8, !range !116, !noundef !8
  %.not26 = icmp eq i64 %22, -1
  br i1 %.not26, label %bb71, label %bb41

bb71:                                             ; preds = %bb17
  call void @llvm.lifetime.end.p0(ptr nonnull %_44)
  br label %bb18

bb18:                                             ; preds = %bb13, %bb71
  %_11651.not = icmp eq i64 %4, 0
  br i1 %_11651.not, label %bb54, label %bb72.preheader

bb72.preheader:                                   ; preds = %bb18
  %23 = getelementptr inbounds nuw i8, ptr %columns, i64 48
  %24 = getelementptr inbounds nuw i8, ptr %columns, i64 40
  %25 = getelementptr inbounds nuw i8, ptr %columns, i64 104
  %26 = getelementptr inbounds nuw i8, ptr %columns, i64 96
  br label %bb72

bb72:                                             ; preds = %bb72.preheader, %bb75
  %iter.sroa.0.052 = phi i64 [ %27, %bb75 ], [ 0, %bb72.preheader ]
  %27 = add nuw i64 %iter.sroa.0.052, 1
  %_4.i.i = load i64, ptr %23, align 8, !noalias !183, !noundef !8
  %start1.i.i = mul i64 %_4.i.i, %iter.sroa.0.052
  %_12.i.i = load ptr, ptr %columns, align 8, !noalias !183, !nonnull !8, !noundef !8
  %_5.i.i = getelementptr inbounds nuw double, ptr %_12.i.i, i64 %start1.i.i
  %_7.i.i = load i64, ptr %24, align 8, !noalias !183, !noundef !8
  %_4.i1.i = load i64, ptr %25, align 8, !noalias !183, !noundef !8
  %start1.i2.i = mul i64 %_4.i1.i, %iter.sroa.0.052
  %_12.i3.i = load ptr, ptr %15, align 8, !noalias !183, !nonnull !8, !noundef !8
  %_5.i4.i = getelementptr inbounds nuw double, ptr %_12.i3.i, i64 %start1.i2.i
  %_7.i5.i = load i64, ptr %26, align 8, !noalias !183, !noundef !8
  tail call void @llvm.experimental.noalias.scope.decl(metadata !187)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !190)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !192)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !195)
  %_0.sroa.0.0.i.i.i.i = tail call noundef i64 @llvm.umin.i64(i64 range(i64 0, 1152921504606846976) %_7.i5.i, i64 range(i64 0, 1152921504606846976) %_7.i.i)
  %_205.not.i.i.i = icmp eq i64 %_0.sroa.0.0.i.i.i.i, 0
  br i1 %_205.not.i.i.i, label %_RINvNtNtCs2rdgasfnGJ9_13vortex_tensor10scalar_fns10arithmetic17inner_product_rowdECsaHo96IcALPt_24row_fn_performance_probe.exit.i, label %bb14.i.i.i.preheader

bb14.i.i.i.preheader:                             ; preds = %bb72
  %min.iters.check119 = icmp ult i64 %_0.sroa.0.0.i.i.i.i, 8
  br i1 %min.iters.check119, label %bb14.i.i.i.preheader207, label %vector.ph120

vector.ph120:                                     ; preds = %bb14.i.i.i.preheader
  %n.vec122 = and i64 %_0.sroa.0.0.i.i.i.i, -8
  br label %vector.body123

vector.body123:                                   ; preds = %vector.body123, %vector.ph120
  %index124 = phi i64 [ 0, %vector.ph120 ], [ %index.next134, %vector.body123 ]
  %vec.phi125 = phi double [ 0.000000e+00, %vector.ph120 ], [ %43, %vector.body123 ]
  %28 = getelementptr inbounds nuw double, ptr %_5.i.i, i64 %index124
  %29 = getelementptr inbounds nuw double, ptr %_5.i4.i, i64 %index124
  %30 = getelementptr inbounds nuw i8, ptr %28, i64 16
  %31 = getelementptr inbounds nuw i8, ptr %28, i64 32
  %32 = getelementptr inbounds nuw i8, ptr %28, i64 48
  %wide.load126 = load <2 x double>, ptr %28, align 8, !alias.scope !197, !noalias !198
  %wide.load127 = load <2 x double>, ptr %30, align 8, !alias.scope !197, !noalias !198
  %wide.load128 = load <2 x double>, ptr %31, align 8, !alias.scope !197, !noalias !198
  %wide.load129 = load <2 x double>, ptr %32, align 8, !alias.scope !197, !noalias !198
  %33 = getelementptr inbounds nuw i8, ptr %29, i64 16
  %34 = getelementptr inbounds nuw i8, ptr %29, i64 32
  %35 = getelementptr inbounds nuw i8, ptr %29, i64 48
  %wide.load130 = load <2 x double>, ptr %29, align 8, !alias.scope !201, !noalias !202
  %wide.load131 = load <2 x double>, ptr %33, align 8, !alias.scope !201, !noalias !202
  %wide.load132 = load <2 x double>, ptr %34, align 8, !alias.scope !201, !noalias !202
  %wide.load133 = load <2 x double>, ptr %35, align 8, !alias.scope !201, !noalias !202
  %36 = fmul <2 x double> %wide.load126, %wide.load130
  %37 = fmul <2 x double> %wide.load127, %wide.load131
  %38 = fmul <2 x double> %wide.load128, %wide.load132
  %39 = fmul <2 x double> %wide.load129, %wide.load133
  %40 = tail call double @llvm.vector.reduce.fadd.v2f64(double %vec.phi125, <2 x double> %36)
  %41 = tail call double @llvm.vector.reduce.fadd.v2f64(double %40, <2 x double> %37)
  %42 = tail call double @llvm.vector.reduce.fadd.v2f64(double %41, <2 x double> %38)
  %43 = tail call double @llvm.vector.reduce.fadd.v2f64(double %42, <2 x double> %39)
  %index.next134 = add nuw i64 %index124, 8
  %44 = icmp eq i64 %index.next134, %n.vec122
  br i1 %44, label %middle.block135, label %vector.body123, !llvm.loop !203

middle.block135:                                  ; preds = %vector.body123
  %cmp.n136 = icmp eq i64 %_0.sroa.0.0.i.i.i.i, %n.vec122
  br i1 %cmp.n136, label %_RINvNtNtCs2rdgasfnGJ9_13vortex_tensor10scalar_fns10arithmetic17inner_product_rowdECsaHo96IcALPt_24row_fn_performance_probe.exit.i, label %bb14.i.i.i.preheader207

bb14.i.i.i.preheader207:                          ; preds = %bb14.i.i.i.preheader, %middle.block135
  %accum.sroa.0.07.i.i.i.ph = phi double [ 0.000000e+00, %bb14.i.i.i.preheader ], [ %43, %middle.block135 ]
  %iter.sroa.0.06.i.i.i.ph = phi i64 [ 0, %bb14.i.i.i.preheader ], [ %n.vec122, %middle.block135 ]
  br label %bb14.i.i.i

bb14.i.i.i:                                       ; preds = %bb14.i.i.i.preheader207, %bb14.i.i.i
  %accum.sroa.0.07.i.i.i = phi double [ %_0.i.i1.i.i.i.i, %bb14.i.i.i ], [ %accum.sroa.0.07.i.i.i.ph, %bb14.i.i.i.preheader207 ]
  %iter.sroa.0.06.i.i.i = phi i64 [ %_24.i.i.i, %bb14.i.i.i ], [ %iter.sroa.0.06.i.i.i.ph, %bb14.i.i.i.preheader207 ]
  %_24.i.i.i = add nuw nsw i64 %iter.sroa.0.06.i.i.i, 1
  %_3.i.i.i.i.i = getelementptr inbounds nuw double, ptr %_5.i.i, i64 %iter.sroa.0.06.i.i.i
  %_3.i2.i.i.i.i = getelementptr inbounds nuw double, ptr %_5.i4.i, i64 %iter.sroa.0.06.i.i.i
  %_16.0.val.i.i.i = load double, ptr %_3.i.i.i.i.i, align 8, !alias.scope !197, !noalias !198, !noundef !8
  %_16.1.val.i.i.i = load double, ptr %_3.i2.i.i.i.i, align 8, !alias.scope !201, !noalias !202, !noundef !8
  %_0.i.i.i.i.i.i = fmul double %_16.0.val.i.i.i, %_16.1.val.i.i.i
  %_0.i.i1.i.i.i.i = fadd double %accum.sroa.0.07.i.i.i, %_0.i.i.i.i.i.i
  %exitcond.not.i.i.i = icmp eq i64 %_24.i.i.i, %_0.sroa.0.0.i.i.i.i
  br i1 %exitcond.not.i.i.i, label %_RINvNtNtCs2rdgasfnGJ9_13vortex_tensor10scalar_fns10arithmetic17inner_product_rowdECsaHo96IcALPt_24row_fn_performance_probe.exit.i, label %bb14.i.i.i, !llvm.loop !206

_RINvNtNtCs2rdgasfnGJ9_13vortex_tensor10scalar_fns10arithmetic17inner_product_rowdECsaHo96IcALPt_24row_fn_performance_probe.exit.i: ; preds = %bb14.i.i.i, %middle.block135, %bb72
  %accum.sroa.0.0.lcssa.i.i.i = phi double [ 0.000000e+00, %bb72 ], [ %43, %middle.block135 ], [ %_0.i.i1.i.i.i.i, %bb14.i.i.i ]
  %_11.idx.i.i = shl i64 %_7.i.i, 3
  %_11.i.i = getelementptr inbounds nuw i8, ptr %_5.i.i, i64 %_11.idx.i.i
  %_204.i.i = icmp eq i64 %_7.i.i, 0
  br i1 %_204.i.i, label %_RINvNtNtCs2rdgasfnGJ9_13vortex_tensor10scalar_fns10arithmetic11l2_norm_rowdECsaHo96IcALPt_24row_fn_performance_probe.exit.i, label %bb13.i.i.preheader

bb13.i.i.preheader:                               ; preds = %_RINvNtNtCs2rdgasfnGJ9_13vortex_tensor10scalar_fns10arithmetic17inner_product_rowdECsaHo96IcALPt_24row_fn_performance_probe.exit.i
  %45 = add i64 %_11.idx.i.i, -8
  %46 = lshr exact i64 %45, 3
  %47 = add nuw nsw i64 %46, 1
  %min.iters.check100 = icmp ult i64 %45, 56
  br i1 %min.iters.check100, label %bb13.i.i.preheader206, label %vector.ph101

vector.ph101:                                     ; preds = %bb13.i.i.preheader
  %n.vec103 = and i64 %47, 4611686018427387896
  %48 = shl i64 %n.vec103, 3
  %49 = getelementptr i8, ptr %_5.i.i, i64 %48
  br label %vector.body104

vector.body104:                                   ; preds = %vector.body104, %vector.ph101
  %index105 = phi i64 [ 0, %vector.ph101 ], [ %index.next113, %vector.body104 ]
  %vec.phi106 = phi double [ 0.000000e+00, %vector.ph101 ], [ %60, %vector.body104 ]
  %offset.idx107 = shl i64 %index105, 3
  %next.gep108 = getelementptr i8, ptr %_5.i.i, i64 %offset.idx107
  %50 = getelementptr i8, ptr %next.gep108, i64 16
  %51 = getelementptr i8, ptr %next.gep108, i64 32
  %52 = getelementptr i8, ptr %next.gep108, i64 48
  %wide.load109 = load <2 x double>, ptr %next.gep108, align 8, !alias.scope !207, !noalias !190
  %wide.load110 = load <2 x double>, ptr %50, align 8, !alias.scope !207, !noalias !190
  %wide.load111 = load <2 x double>, ptr %51, align 8, !alias.scope !207, !noalias !190
  %wide.load112 = load <2 x double>, ptr %52, align 8, !alias.scope !207, !noalias !190
  %53 = fmul <2 x double> %wide.load109, %wide.load109
  %54 = fmul <2 x double> %wide.load110, %wide.load110
  %55 = fmul <2 x double> %wide.load111, %wide.load111
  %56 = fmul <2 x double> %wide.load112, %wide.load112
  %57 = tail call double @llvm.vector.reduce.fadd.v2f64(double %vec.phi106, <2 x double> %53)
  %58 = tail call double @llvm.vector.reduce.fadd.v2f64(double %57, <2 x double> %54)
  %59 = tail call double @llvm.vector.reduce.fadd.v2f64(double %58, <2 x double> %55)
  %60 = tail call double @llvm.vector.reduce.fadd.v2f64(double %59, <2 x double> %56)
  %index.next113 = add nuw i64 %index105, 8
  %61 = icmp eq i64 %index.next113, %n.vec103
  br i1 %61, label %middle.block114, label %vector.body104, !llvm.loop !210

middle.block114:                                  ; preds = %vector.body104
  %cmp.n115 = icmp eq i64 %47, %n.vec103
  br i1 %cmp.n115, label %bb15.loopexit.i.i, label %bb13.i.i.preheader206

bb13.i.i.preheader206:                            ; preds = %bb13.i.i.preheader, %middle.block114
  %iter1.sroa.0.06.i.i.ph = phi ptr [ %_5.i.i, %bb13.i.i.preheader ], [ %49, %middle.block114 ]
  %sum_squared.sroa.0.05.i.i.ph = phi double [ 0.000000e+00, %bb13.i.i.preheader ], [ %60, %middle.block114 ]
  br label %bb13.i.i

bb13.i.i:                                         ; preds = %bb13.i.i.preheader206, %bb13.i.i
  %iter1.sroa.0.06.i.i = phi ptr [ %_26.i.i, %bb13.i.i ], [ %iter1.sroa.0.06.i.i.ph, %bb13.i.i.preheader206 ]
  %sum_squared.sroa.0.05.i.i = phi double [ %_0.i2.i.i, %bb13.i.i ], [ %sum_squared.sroa.0.05.i.i.ph, %bb13.i.i.preheader206 ]
  %_26.i.i = getelementptr inbounds nuw i8, ptr %iter1.sroa.0.06.i.i, i64 8
  %value.i.i = load double, ptr %iter1.sroa.0.06.i.i, align 8, !alias.scope !207, !noalias !190, !noundef !8
  %_0.i.i.i = fmul double %value.i.i, %value.i.i
  %_0.i2.i.i = fadd double %sum_squared.sroa.0.05.i.i, %_0.i.i.i
  %_20.i.i = icmp eq ptr %_26.i.i, %_11.i.i
  br i1 %_20.i.i, label %bb15.loopexit.i.i, label %bb13.i.i, !llvm.loop !211

bb15.loopexit.i.i:                                ; preds = %bb13.i.i, %middle.block114
  %_0.i2.i.i.lcssa = phi double [ %60, %middle.block114 ], [ %_0.i2.i.i, %bb13.i.i ]
  %62 = tail call double @llvm.sqrt.f64(double %_0.i2.i.i.lcssa)
  br label %_RINvNtNtCs2rdgasfnGJ9_13vortex_tensor10scalar_fns10arithmetic11l2_norm_rowdECsaHo96IcALPt_24row_fn_performance_probe.exit.i

_RINvNtNtCs2rdgasfnGJ9_13vortex_tensor10scalar_fns10arithmetic11l2_norm_rowdECsaHo96IcALPt_24row_fn_performance_probe.exit.i: ; preds = %bb15.loopexit.i.i, %_RINvNtNtCs2rdgasfnGJ9_13vortex_tensor10scalar_fns10arithmetic17inner_product_rowdECsaHo96IcALPt_24row_fn_performance_probe.exit.i
  %sum_squared.sroa.0.0.lcssa.i.i = phi double [ 0.000000e+00, %_RINvNtNtCs2rdgasfnGJ9_13vortex_tensor10scalar_fns10arithmetic17inner_product_rowdECsaHo96IcALPt_24row_fn_performance_probe.exit.i ], [ %62, %bb15.loopexit.i.i ]
  %_11.idx.i1.i = shl i64 %_7.i5.i, 3
  %_11.i2.i = getelementptr inbounds nuw i8, ptr %_5.i4.i, i64 %_11.idx.i1.i
  %_204.i3.i = icmp eq i64 %_7.i5.i, 0
  br i1 %_204.i3.i, label %bb75, label %bb13.i4.i.preheader

bb13.i4.i.preheader:                              ; preds = %_RINvNtNtCs2rdgasfnGJ9_13vortex_tensor10scalar_fns10arithmetic11l2_norm_rowdECsaHo96IcALPt_24row_fn_performance_probe.exit.i
  %63 = add i64 %_11.idx.i1.i, -8
  %64 = lshr exact i64 %63, 3
  %65 = add nuw nsw i64 %64, 1
  %min.iters.check = icmp ult i64 %63, 56
  br i1 %min.iters.check, label %bb13.i4.i.preheader205, label %vector.ph

vector.ph:                                        ; preds = %bb13.i4.i.preheader
  %n.vec = and i64 %65, 4611686018427387896
  %66 = shl i64 %n.vec, 3
  %67 = getelementptr i8, ptr %_5.i4.i, i64 %66
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %vec.phi = phi double [ 0.000000e+00, %vector.ph ], [ %78, %vector.body ]
  %offset.idx = shl i64 %index, 3
  %next.gep = getelementptr i8, ptr %_5.i4.i, i64 %offset.idx
  %68 = getelementptr i8, ptr %next.gep, i64 16
  %69 = getelementptr i8, ptr %next.gep, i64 32
  %70 = getelementptr i8, ptr %next.gep, i64 48
  %wide.load = load <2 x double>, ptr %next.gep, align 8, !alias.scope !212, !noalias !187
  %wide.load96 = load <2 x double>, ptr %68, align 8, !alias.scope !212, !noalias !187
  %wide.load97 = load <2 x double>, ptr %69, align 8, !alias.scope !212, !noalias !187
  %wide.load98 = load <2 x double>, ptr %70, align 8, !alias.scope !212, !noalias !187
  %71 = fmul <2 x double> %wide.load, %wide.load
  %72 = fmul <2 x double> %wide.load96, %wide.load96
  %73 = fmul <2 x double> %wide.load97, %wide.load97
  %74 = fmul <2 x double> %wide.load98, %wide.load98
  %75 = tail call double @llvm.vector.reduce.fadd.v2f64(double %vec.phi, <2 x double> %71)
  %76 = tail call double @llvm.vector.reduce.fadd.v2f64(double %75, <2 x double> %72)
  %77 = tail call double @llvm.vector.reduce.fadd.v2f64(double %76, <2 x double> %73)
  %78 = tail call double @llvm.vector.reduce.fadd.v2f64(double %77, <2 x double> %74)
  %index.next = add nuw i64 %index, 8
  %79 = icmp eq i64 %index.next, %n.vec
  br i1 %79, label %middle.block, label %vector.body, !llvm.loop !215

middle.block:                                     ; preds = %vector.body
  %cmp.n = icmp eq i64 %65, %n.vec
  br i1 %cmp.n, label %bb15.loopexit.i12.i, label %bb13.i4.i.preheader205

bb13.i4.i.preheader205:                           ; preds = %bb13.i4.i.preheader, %middle.block
  %iter1.sroa.0.06.i5.i.ph = phi ptr [ %_5.i4.i, %bb13.i4.i.preheader ], [ %67, %middle.block ]
  %sum_squared.sroa.0.05.i6.i.ph = phi double [ 0.000000e+00, %bb13.i4.i.preheader ], [ %78, %middle.block ]
  br label %bb13.i4.i

bb13.i4.i:                                        ; preds = %bb13.i4.i.preheader205, %bb13.i4.i
  %iter1.sroa.0.06.i5.i = phi ptr [ %_26.i7.i, %bb13.i4.i ], [ %iter1.sroa.0.06.i5.i.ph, %bb13.i4.i.preheader205 ]
  %sum_squared.sroa.0.05.i6.i = phi double [ %_0.i2.i10.i, %bb13.i4.i ], [ %sum_squared.sroa.0.05.i6.i.ph, %bb13.i4.i.preheader205 ]
  %_26.i7.i = getelementptr inbounds nuw i8, ptr %iter1.sroa.0.06.i5.i, i64 8
  %value.i8.i = load double, ptr %iter1.sroa.0.06.i5.i, align 8, !alias.scope !212, !noalias !187, !noundef !8
  %_0.i.i9.i = fmul double %value.i8.i, %value.i8.i
  %_0.i2.i10.i = fadd double %sum_squared.sroa.0.05.i6.i, %_0.i.i9.i
  %_20.i11.i = icmp eq ptr %_26.i7.i, %_11.i2.i
  br i1 %_20.i11.i, label %bb15.loopexit.i12.i, label %bb13.i4.i, !llvm.loop !216

bb15.loopexit.i12.i:                              ; preds = %bb13.i4.i, %middle.block
  %_0.i2.i10.i.lcssa = phi double [ %78, %middle.block ], [ %_0.i2.i10.i, %bb13.i4.i ]
  %80 = tail call double @llvm.sqrt.f64(double %_0.i2.i10.i.lcssa)
  br label %bb75

bb54:                                             ; preds = %bb75, %bb81, %bb18, %bb30
  tail call void @llvm.experimental.noalias.scope.decl(metadata !217)
  call void @llvm.lifetime.start.p0(ptr nonnull %_4.i15), !noalias !220
  store i64 %_8.sroa.4.0.i, ptr %_4.i15, align 8, !noalias !217
  %_79.sroa.4.0._4.i15.sroa_idx = getelementptr inbounds nuw i8, ptr %_4.i15, i64 8
  store ptr %12, ptr %_79.sroa.4.0._4.i15.sroa_idx, align 8, !noalias !217
  %_79.sroa.5.0._4.i15.sroa_idx = getelementptr inbounds nuw i8, ptr %_4.i15, i64 16
  store i64 %4, ptr %_79.sroa.5.0._4.i15.sroa_idx, align 8, !noalias !217
  call void @llvm.lifetime.start.p0(ptr nonnull %_3.i.i), !noalias !222
  store i64 0, ptr %_3.i.i, align 8, !noalias !222
; invoke <vortex_array::array::typed::Array<vortex_array::arrays::primitive::vtable::Primitive>>::new::<f64, alloc::vec::Vec<f64>>
  %81 = invoke { ptr, ptr } @_RINvMs1_NtNtNtCs5mvLrSLkDP1_12vortex_array6arrays9primitive5arrayINtNtNtBc_5array5typed5ArrayNtNtB8_6vtable9PrimitiveE3newdINtNtCs6KVRSXc8uZF_5alloc3vec3VecdEECsaHo96IcALPt_24row_fn_performance_probe(ptr noalias nofree noundef nonnull align 8 captures(address) dereferenceable(24) %_4.i15, ptr noalias nofree noundef nonnull align 8 captures(address) dereferenceable(24) %_3.i.i)
          to label %bb37 unwind label %bb63

bb75:                                             ; preds = %bb15.loopexit.i12.i, %_RINvNtNtCs2rdgasfnGJ9_13vortex_tensor10scalar_fns10arithmetic11l2_norm_rowdECsaHo96IcALPt_24row_fn_performance_probe.exit.i
  %sum_squared.sroa.0.0.lcssa.i13.i = phi double [ 0.000000e+00, %_RINvNtNtCs2rdgasfnGJ9_13vortex_tensor10scalar_fns10arithmetic11l2_norm_rowdECsaHo96IcALPt_24row_fn_performance_probe.exit.i ], [ %80, %bb15.loopexit.i12.i ]
  %_0.i.i = fmul double %sum_squared.sroa.0.0.lcssa.i.i, %sum_squared.sroa.0.0.lcssa.i13.i
  %_0.i1.i = fcmp oeq double %_0.i.i, 0.000000e+00
  %_0.i2.i = fdiv double %accum.sroa.0.0.lcssa.i.i.i, %_0.i.i
  %_0.sroa.0.0.i14 = select i1 %_0.i1.i, double 0.000000e+00, double %_0.i2.i
  %_4.i = getelementptr inbounds nuw double, ptr %12, i64 %iter.sroa.0.052
  store double %_0.sroa.0.0.i14, ptr %_4.i, align 8, !alias.scope !225, !noalias !230
  %exitcond.not = icmp eq i64 %27, %4
  br i1 %exitcond.not, label %bb54, label %bb72

bb41:                                             ; preds = %bb17
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(80) %_0, ptr noundef nonnull align 8 dereferenceable(80) %_44, i64 80, i1 false)
  call void @llvm.lifetime.end.p0(ptr nonnull %_44)
  br label %bb42

bb26:                                             ; preds = %bb24
  %82 = getelementptr inbounds nuw i8, ptr %columns, i64 88
  %_2.val.i.i1.i = load i64, ptr %82, align 8, !alias.scope !233
  %83 = icmp eq i64 %_2.val.i.i1.i, %4
  %_0.sroa.0.0.off0.i2.i = select i1 %17, i1 true, i1 %83
  br i1 %_0.sroa.0.0.off0.i2.i, label %bb30, label %bb28, !prof !142

bb28:                                             ; preds = %bb24, %bb26
  call void @llvm.lifetime.start.p0(ptr nonnull %_63)
; invoke vortex_array::scalar_fn::unstable::row::execute::sink::decoded_length_error
  invoke void @_RNvNtNtNtNtNtCs5mvLrSLkDP1_12vortex_array9scalar_fn8unstable3row7execute4sink20decoded_length_error(ptr noalias nofree noundef nonnull sret([80 x i8]) align 8 captures(none) dereferenceable(80) %_63, i64 noundef %4)
          to label %bb29 unwind label %cleanup12

bb29:                                             ; preds = %bb28
  %84 = load i64, ptr %_63, align 8, !range !116, !noundef !8
  %.not24 = icmp eq i64 %84, -1
  br i1 %.not24, label %bb77, label %bb76

bb76:                                             ; preds = %bb29
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(80) %_0, ptr noundef nonnull align 8 dereferenceable(80) %_63, i64 80, i1 false)
  call void @llvm.lifetime.end.p0(ptr nonnull %_63)
  br label %bb42

bb77:                                             ; preds = %bb29
  call void @llvm.lifetime.end.p0(ptr nonnull %_63)
  br label %bb30

bb30:                                             ; preds = %bb77, %bb26
  %_13253.not = icmp eq i64 %4, 0
  br i1 %_13253.not, label %bb54, label %bb32.preheader

bb32.preheader:                                   ; preds = %bb30
  %85 = getelementptr inbounds nuw i8, ptr %columns, i64 48
  %86 = getelementptr inbounds nuw i8, ptr %columns, i64 8
  %87 = getelementptr inbounds nuw i8, ptr %columns, i64 40
  %88 = getelementptr inbounds nuw i8, ptr %columns, i64 16
  %89 = getelementptr inbounds nuw i8, ptr %columns, i64 104
  %90 = getelementptr inbounds nuw i8, ptr %columns, i64 64
  %91 = getelementptr inbounds nuw i8, ptr %columns, i64 96
  %92 = getelementptr inbounds nuw i8, ptr %columns, i64 72
  br label %bb32

bb37:                                             ; preds = %bb54
  call void @llvm.lifetime.end.p0(ptr nonnull %_3.i.i), !noalias !222
  %_3.0.i = extractvalue { ptr, ptr } %81, 0
  %_3.1.i = extractvalue { ptr, ptr } %81, 1
  call void @llvm.lifetime.end.p0(ptr nonnull %_4.i15), !noalias !220
  %93 = getelementptr inbounds nuw i8, ptr %_0, i64 8
  store ptr %_3.0.i, ptr %93, align 8, !alias.scope !217, !noalias !238
  %94 = getelementptr inbounds nuw i8, ptr %_0, i64 16
  store ptr %_3.1.i, ptr %94, align 8, !alias.scope !217, !noalias !238
  store i64 -1, ptr %_0, align 8, !alias.scope !217, !noalias !238
; call core::ptr::drop_glue::<(vortex_array::scalar_fn::unstable::row::types::element::tuple::element_tuple::ArgColumn<vortex_tensor::scalar_fns::row::TensorRow<half::binary16::f16>>, vortex_array::scalar_fn::unstable::row::types::element::tuple::element_tuple::ArgColumn<vortex_tensor::scalar_fns::row::TensorRow<half::binary16::f16>>)>
  call fastcc void @_RINvNtCslWxY2MhVcag_4core3ptr9drop_glueTINtNtNtNtNtNtNtNtCs5mvLrSLkDP1_12vortex_array9scalar_fn8unstable3row5types7element5tuple13element_tuple9ArgColumnINtNtNtCs2rdgasfnGJ9_13vortex_tensor10scalar_fns3row9TensorRowNtNtCsajgNrFuHJYM_4half8binary163f16EEBC_EECsaHo96IcALPt_24row_fn_performance_probe(ptr noalias nofree noundef align 8 dereferenceable(112) %columns)
  br label %bb47

bb47:                                             ; preds = %bb44, %bb37, %bb66
  call void @llvm.lifetime.end.p0(ptr nonnull %columns)
  ret void

bb32:                                             ; preds = %bb32.preheader, %bb81
  %iter8.sroa.0.054 = phi i64 [ %95, %bb81 ], [ 0, %bb32.preheader ]
  %95 = add nuw i64 %iter8.sroa.0.054, 1
  %_4.i36 = getelementptr inbounds nuw double, ptr %12, i64 %iter8.sroa.0.054
  %96 = load ptr, ptr %columns, align 8, !alias.scope !239, !noalias !244, !noundef !8
  %97 = icmp eq ptr %96, null
  br i1 %97, label %bb2.i.i, label %bb3.i.i17

bb2.i.i:                                          ; preds = %bb32
  %constant.val.i.i = load ptr, ptr %86, align 8, !alias.scope !239, !noalias !244, !nonnull !8, !noundef !8
  %constant.val2.i.i = load i64, ptr %88, align 8, !alias.scope !239, !noalias !244, !noundef !8
  br label %_RNvMNtNtNtNtNtNtNtCs5mvLrSLkDP1_12vortex_array9scalar_fn8unstable3row5types7element5tuple13element_tupleINtB2_9ArgColumnINtNtNtCs2rdgasfnGJ9_13vortex_tensor10scalar_fns3row9TensorRowdEE3getCsaHo96IcALPt_24row_fn_performance_probe.exit.i

bb3.i.i17:                                        ; preds = %bb32
  %_4.i.i.i = load i64, ptr %85, align 8, !alias.scope !246, !noalias !244, !noundef !8
  %start1.i.i.i = mul i64 %_4.i.i.i, %iter8.sroa.0.054
  %_9.i.i.i = load i64, ptr %86, align 8, !alias.scope !246, !noalias !244, !noundef !8
  %_6.i.i.i = load i64, ptr %87, align 8, !alias.scope !246, !noalias !244, !noundef !8
  %_5.i.i.i = add i64 %_6.i.i.i, %start1.i.i.i
  %_15.i.i.i = icmp ult i64 %_5.i.i.i, %start1.i.i.i
  %_11.not.i.i.i = icmp ugt i64 %_5.i.i.i, %_9.i.i.i
  %or.cond.i.i.i = or i1 %_15.i.i.i, %_11.not.i.i.i
  br i1 %or.cond.i.i.i, label %bb2.i.i16.i.invoke, label %_RNvXs_NtNtCs2rdgasfnGJ9_13vortex_tensor10scalar_fns3rowINtB4_9TensorRowdENtNtNtNtNtNtNtCs5mvLrSLkDP1_12vortex_array9scalar_fn8unstable3row5types7element5input12InputElement3getCsaHo96IcALPt_24row_fn_performance_probe.exit.i.i, !prof !154

_RNvXs_NtNtCs2rdgasfnGJ9_13vortex_tensor10scalar_fns3rowINtB4_9TensorRowdENtNtNtNtNtNtNtCs5mvLrSLkDP1_12vortex_array9scalar_fn8unstable3row5types7element5input12InputElement3getCsaHo96IcALPt_24row_fn_performance_probe.exit.i.i: ; preds = %bb3.i.i17
  %_18.i.i.i = getelementptr inbounds nuw double, ptr %96, i64 %start1.i.i.i
  br label %_RNvMNtNtNtNtNtNtNtCs5mvLrSLkDP1_12vortex_array9scalar_fn8unstable3row5types7element5tuple13element_tupleINtB2_9ArgColumnINtNtNtCs2rdgasfnGJ9_13vortex_tensor10scalar_fns3row9TensorRowdEE3getCsaHo96IcALPt_24row_fn_performance_probe.exit.i

_RNvMNtNtNtNtNtNtNtCs5mvLrSLkDP1_12vortex_array9scalar_fn8unstable3row5types7element5tuple13element_tupleINtB2_9ArgColumnINtNtNtCs2rdgasfnGJ9_13vortex_tensor10scalar_fns3row9TensorRowdEE3getCsaHo96IcALPt_24row_fn_performance_probe.exit.i: ; preds = %_RNvXs_NtNtCs2rdgasfnGJ9_13vortex_tensor10scalar_fns3rowINtB4_9TensorRowdENtNtNtNtNtNtNtCs5mvLrSLkDP1_12vortex_array9scalar_fn8unstable3row5types7element5input12InputElement3getCsaHo96IcALPt_24row_fn_performance_probe.exit.i.i, %bb2.i.i
  %constant.val.pn.i.i = phi ptr [ %constant.val.i.i, %bb2.i.i ], [ %_18.i.i.i, %_RNvXs_NtNtCs2rdgasfnGJ9_13vortex_tensor10scalar_fns3rowINtB4_9TensorRowdENtNtNtNtNtNtNtCs5mvLrSLkDP1_12vortex_array9scalar_fn8unstable3row5types7element5input12InputElement3getCsaHo96IcALPt_24row_fn_performance_probe.exit.i.i ]
  %constant.val2.pn.i.i = phi i64 [ %constant.val2.i.i, %bb2.i.i ], [ %_6.i.i.i, %_RNvXs_NtNtCs2rdgasfnGJ9_13vortex_tensor10scalar_fns3rowINtB4_9TensorRowdENtNtNtNtNtNtNtCs5mvLrSLkDP1_12vortex_array9scalar_fn8unstable3row5types7element5input12InputElement3getCsaHo96IcALPt_24row_fn_performance_probe.exit.i.i ]
  %98 = load ptr, ptr %15, align 8, !alias.scope !249, !noalias !244, !noundef !8
  %99 = icmp eq ptr %98, null
  br i1 %99, label %bb2.i17.i, label %bb3.i1.i

bb2.i17.i:                                        ; preds = %_RNvMNtNtNtNtNtNtNtCs5mvLrSLkDP1_12vortex_array9scalar_fn8unstable3row5types7element5tuple13element_tupleINtB2_9ArgColumnINtNtNtCs2rdgasfnGJ9_13vortex_tensor10scalar_fns3row9TensorRowdEE3getCsaHo96IcALPt_24row_fn_performance_probe.exit.i
  %constant.val.i19.i = load ptr, ptr %90, align 8, !alias.scope !249, !noalias !244, !nonnull !8, !noundef !8
  %constant.val2.i20.i = load i64, ptr %92, align 8, !alias.scope !249, !noalias !244, !noundef !8
  br label %bb33

bb3.i1.i:                                         ; preds = %_RNvMNtNtNtNtNtNtNtCs5mvLrSLkDP1_12vortex_array9scalar_fn8unstable3row5types7element5tuple13element_tupleINtB2_9ArgColumnINtNtNtCs2rdgasfnGJ9_13vortex_tensor10scalar_fns3row9TensorRowdEE3getCsaHo96IcALPt_24row_fn_performance_probe.exit.i
  %_4.i.i2.i = load i64, ptr %89, align 8, !alias.scope !252, !noalias !244, !noundef !8
  %start1.i.i3.i = mul i64 %_4.i.i2.i, %iter8.sroa.0.054
  %_9.i.i4.i = load i64, ptr %90, align 8, !alias.scope !252, !noalias !244, !noundef !8
  %_6.i.i5.i = load i64, ptr %91, align 8, !alias.scope !252, !noalias !244, !noundef !8
  %_5.i.i6.i = add i64 %_6.i.i5.i, %start1.i.i3.i
  %_15.i.i7.i = icmp ult i64 %_5.i.i6.i, %start1.i.i3.i
  %_11.not.i.i8.i = icmp ugt i64 %_5.i.i6.i, %_9.i.i4.i
  %or.cond.i.i9.i = or i1 %_15.i.i7.i, %_11.not.i.i8.i
  br i1 %or.cond.i.i9.i, label %bb2.i.i16.i.invoke, label %_RNvXs_NtNtCs2rdgasfnGJ9_13vortex_tensor10scalar_fns3rowINtB4_9TensorRowdENtNtNtNtNtNtNtCs5mvLrSLkDP1_12vortex_array9scalar_fn8unstable3row5types7element5input12InputElement3getCsaHo96IcALPt_24row_fn_performance_probe.exit.i10.i, !prof !154

bb2.i.i16.i.invoke:                               ; preds = %bb3.i1.i, %bb3.i.i17
  %100 = phi i64 [ %start1.i.i.i, %bb3.i.i17 ], [ %start1.i.i3.i, %bb3.i1.i ]
  %101 = phi i64 [ %_5.i.i.i, %bb3.i.i17 ], [ %_5.i.i6.i, %bb3.i1.i ]
  %102 = phi i64 [ %_9.i.i.i, %bb3.i.i17 ], [ %_9.i.i4.i, %bb3.i1.i ]
; invoke core::slice::index::slice_index_fail
  invoke void @_RNvNtNtCslWxY2MhVcag_4core5slice5index16slice_index_fail(i64 noundef %100, i64 noundef %101, i64 noundef %102, ptr noalias nofree noundef readonly align 8 captures(address, read_provenance) dereferenceable(24) @alloc_cd44ca30c046dd08226918654a8231a3.llvm.12919063887536347969) #20
          to label %bb2.i.i16.i.cont unwind label %cleanup17

bb2.i.i16.i.cont:                                 ; preds = %bb2.i.i16.i.invoke
  unreachable

_RNvXs_NtNtCs2rdgasfnGJ9_13vortex_tensor10scalar_fns3rowINtB4_9TensorRowdENtNtNtNtNtNtNtCs5mvLrSLkDP1_12vortex_array9scalar_fn8unstable3row5types7element5input12InputElement3getCsaHo96IcALPt_24row_fn_performance_probe.exit.i10.i: ; preds = %bb3.i1.i
  %_18.i.i11.i = getelementptr inbounds nuw double, ptr %98, i64 %start1.i.i3.i
  br label %bb33

cleanup17:                                        ; preds = %bb2.i.i16.i.invoke
  %103 = landingpad { ptr, i32 }
          cleanup
  br label %bb62

bb33:                                             ; preds = %_RNvXs_NtNtCs2rdgasfnGJ9_13vortex_tensor10scalar_fns3rowINtB4_9TensorRowdENtNtNtNtNtNtNtCs5mvLrSLkDP1_12vortex_array9scalar_fn8unstable3row5types7element5input12InputElement3getCsaHo96IcALPt_24row_fn_performance_probe.exit.i10.i, %bb2.i17.i
  %constant.val.pn.i12.i = phi ptr [ %constant.val.i19.i, %bb2.i17.i ], [ %_18.i.i11.i, %_RNvXs_NtNtCs2rdgasfnGJ9_13vortex_tensor10scalar_fns3rowINtB4_9TensorRowdENtNtNtNtNtNtNtCs5mvLrSLkDP1_12vortex_array9scalar_fn8unstable3row5types7element5input12InputElement3getCsaHo96IcALPt_24row_fn_performance_probe.exit.i10.i ]
  %constant.val2.pn.i13.i = phi i64 [ %constant.val2.i20.i, %bb2.i17.i ], [ %_6.i.i5.i, %_RNvXs_NtNtCs2rdgasfnGJ9_13vortex_tensor10scalar_fns3rowINtB4_9TensorRowdENtNtNtNtNtNtNtCs5mvLrSLkDP1_12vortex_array9scalar_fn8unstable3row5types7element5input12InputElement3getCsaHo96IcALPt_24row_fn_performance_probe.exit.i10.i ]
  tail call void @llvm.experimental.noalias.scope.decl(metadata !255)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !258)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !260)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !263)
  %_0.sroa.0.0.i.i.i.i21 = tail call noundef i64 @llvm.umin.i64(i64 range(i64 0, 1152921504606846976) %constant.val2.pn.i13.i, i64 range(i64 0, 1152921504606846976) %constant.val2.pn.i.i)
  %_205.not.i.i.i22 = icmp eq i64 %_0.sroa.0.0.i.i.i.i21, 0
  br i1 %_205.not.i.i.i22, label %_RINvNtNtCs2rdgasfnGJ9_13vortex_tensor10scalar_fns10arithmetic17inner_product_rowdECsaHo96IcALPt_24row_fn_performance_probe.exit.i34, label %bb14.i.i.i23.preheader

bb14.i.i.i23.preheader:                           ; preds = %bb33
  %min.iters.check178 = icmp ult i64 %_0.sroa.0.0.i.i.i.i21, 8
  br i1 %min.iters.check178, label %bb14.i.i.i23.preheader200, label %vector.ph179

vector.ph179:                                     ; preds = %bb14.i.i.i23.preheader
  %n.vec181 = and i64 %_0.sroa.0.0.i.i.i.i21, -8
  br label %vector.body182

vector.body182:                                   ; preds = %vector.body182, %vector.ph179
  %index183 = phi i64 [ 0, %vector.ph179 ], [ %index.next193, %vector.body182 ]
  %vec.phi184 = phi double [ 0.000000e+00, %vector.ph179 ], [ %119, %vector.body182 ]
  %104 = getelementptr inbounds nuw double, ptr %constant.val.pn.i.i, i64 %index183
  %105 = getelementptr inbounds nuw double, ptr %constant.val.pn.i12.i, i64 %index183
  %106 = getelementptr inbounds nuw i8, ptr %104, i64 16
  %107 = getelementptr inbounds nuw i8, ptr %104, i64 32
  %108 = getelementptr inbounds nuw i8, ptr %104, i64 48
  %wide.load185 = load <2 x double>, ptr %104, align 8, !alias.scope !265, !noalias !266
  %wide.load186 = load <2 x double>, ptr %106, align 8, !alias.scope !265, !noalias !266
  %wide.load187 = load <2 x double>, ptr %107, align 8, !alias.scope !265, !noalias !266
  %wide.load188 = load <2 x double>, ptr %108, align 8, !alias.scope !265, !noalias !266
  %109 = getelementptr inbounds nuw i8, ptr %105, i64 16
  %110 = getelementptr inbounds nuw i8, ptr %105, i64 32
  %111 = getelementptr inbounds nuw i8, ptr %105, i64 48
  %wide.load189 = load <2 x double>, ptr %105, align 8, !alias.scope !269, !noalias !270
  %wide.load190 = load <2 x double>, ptr %109, align 8, !alias.scope !269, !noalias !270
  %wide.load191 = load <2 x double>, ptr %110, align 8, !alias.scope !269, !noalias !270
  %wide.load192 = load <2 x double>, ptr %111, align 8, !alias.scope !269, !noalias !270
  %112 = fmul <2 x double> %wide.load185, %wide.load189
  %113 = fmul <2 x double> %wide.load186, %wide.load190
  %114 = fmul <2 x double> %wide.load187, %wide.load191
  %115 = fmul <2 x double> %wide.load188, %wide.load192
  %116 = tail call double @llvm.vector.reduce.fadd.v2f64(double %vec.phi184, <2 x double> %112)
  %117 = tail call double @llvm.vector.reduce.fadd.v2f64(double %116, <2 x double> %113)
  %118 = tail call double @llvm.vector.reduce.fadd.v2f64(double %117, <2 x double> %114)
  %119 = tail call double @llvm.vector.reduce.fadd.v2f64(double %118, <2 x double> %115)
  %index.next193 = add nuw i64 %index183, 8
  %120 = icmp eq i64 %index.next193, %n.vec181
  br i1 %120, label %middle.block194, label %vector.body182, !llvm.loop !271

middle.block194:                                  ; preds = %vector.body182
  %cmp.n195 = icmp eq i64 %_0.sroa.0.0.i.i.i.i21, %n.vec181
  br i1 %cmp.n195, label %_RINvNtNtCs2rdgasfnGJ9_13vortex_tensor10scalar_fns10arithmetic17inner_product_rowdECsaHo96IcALPt_24row_fn_performance_probe.exit.i34, label %bb14.i.i.i23.preheader200

bb14.i.i.i23.preheader200:                        ; preds = %bb14.i.i.i23.preheader, %middle.block194
  %accum.sroa.0.07.i.i.i24.ph = phi double [ 0.000000e+00, %bb14.i.i.i23.preheader ], [ %119, %middle.block194 ]
  %iter.sroa.0.06.i.i.i25.ph = phi i64 [ 0, %bb14.i.i.i23.preheader ], [ %n.vec181, %middle.block194 ]
  br label %bb14.i.i.i23

bb14.i.i.i23:                                     ; preds = %bb14.i.i.i23.preheader200, %bb14.i.i.i23
  %accum.sroa.0.07.i.i.i24 = phi double [ %_0.i.i1.i.i.i.i32, %bb14.i.i.i23 ], [ %accum.sroa.0.07.i.i.i24.ph, %bb14.i.i.i23.preheader200 ]
  %iter.sroa.0.06.i.i.i25 = phi i64 [ %_24.i.i.i26, %bb14.i.i.i23 ], [ %iter.sroa.0.06.i.i.i25.ph, %bb14.i.i.i23.preheader200 ]
  %_24.i.i.i26 = add nuw nsw i64 %iter.sroa.0.06.i.i.i25, 1
  %_3.i.i.i.i.i27 = getelementptr inbounds nuw double, ptr %constant.val.pn.i.i, i64 %iter.sroa.0.06.i.i.i25
  %_3.i2.i.i.i.i28 = getelementptr inbounds nuw double, ptr %constant.val.pn.i12.i, i64 %iter.sroa.0.06.i.i.i25
  %_16.0.val.i.i.i29 = load double, ptr %_3.i.i.i.i.i27, align 8, !alias.scope !265, !noalias !266, !noundef !8
  %_16.1.val.i.i.i30 = load double, ptr %_3.i2.i.i.i.i28, align 8, !alias.scope !269, !noalias !270, !noundef !8
  %_0.i.i.i.i.i.i31 = fmul double %_16.0.val.i.i.i29, %_16.1.val.i.i.i30
  %_0.i.i1.i.i.i.i32 = fadd double %accum.sroa.0.07.i.i.i24, %_0.i.i.i.i.i.i31
  %exitcond.not.i.i.i33 = icmp eq i64 %_24.i.i.i26, %_0.sroa.0.0.i.i.i.i21
  br i1 %exitcond.not.i.i.i33, label %_RINvNtNtCs2rdgasfnGJ9_13vortex_tensor10scalar_fns10arithmetic17inner_product_rowdECsaHo96IcALPt_24row_fn_performance_probe.exit.i34, label %bb14.i.i.i23, !llvm.loop !272

_RINvNtNtCs2rdgasfnGJ9_13vortex_tensor10scalar_fns10arithmetic17inner_product_rowdECsaHo96IcALPt_24row_fn_performance_probe.exit.i34: ; preds = %bb14.i.i.i23, %middle.block194, %bb33
  %accum.sroa.0.0.lcssa.i.i.i35 = phi double [ 0.000000e+00, %bb33 ], [ %119, %middle.block194 ], [ %_0.i.i1.i.i.i.i32, %bb14.i.i.i23 ]
  %_11.idx.i.i36 = shl i64 %constant.val2.pn.i.i, 3
  %_11.i.i37 = getelementptr inbounds nuw i8, ptr %constant.val.pn.i.i, i64 %_11.idx.i.i36
  %_204.i.i38 = icmp eq i64 %constant.val2.pn.i.i, 0
  br i1 %_204.i.i38, label %_RINvNtNtCs2rdgasfnGJ9_13vortex_tensor10scalar_fns10arithmetic11l2_norm_rowdECsaHo96IcALPt_24row_fn_performance_probe.exit.i48, label %bb13.i.i39.preheader

bb13.i.i39.preheader:                             ; preds = %_RINvNtNtCs2rdgasfnGJ9_13vortex_tensor10scalar_fns10arithmetic17inner_product_rowdECsaHo96IcALPt_24row_fn_performance_probe.exit.i34
  %121 = add i64 %_11.idx.i.i36, -8
  %122 = lshr exact i64 %121, 3
  %123 = add nuw nsw i64 %122, 1
  %min.iters.check159 = icmp ult i64 %121, 56
  br i1 %min.iters.check159, label %bb13.i.i39.preheader199, label %vector.ph160

vector.ph160:                                     ; preds = %bb13.i.i39.preheader
  %n.vec162 = and i64 %123, 4611686018427387896
  %124 = shl i64 %n.vec162, 3
  %125 = getelementptr i8, ptr %constant.val.pn.i.i, i64 %124
  br label %vector.body163

vector.body163:                                   ; preds = %vector.body163, %vector.ph160
  %index164 = phi i64 [ 0, %vector.ph160 ], [ %index.next172, %vector.body163 ]
  %vec.phi165 = phi double [ 0.000000e+00, %vector.ph160 ], [ %136, %vector.body163 ]
  %offset.idx166 = shl i64 %index164, 3
  %next.gep167 = getelementptr i8, ptr %constant.val.pn.i.i, i64 %offset.idx166
  %126 = getelementptr i8, ptr %next.gep167, i64 16
  %127 = getelementptr i8, ptr %next.gep167, i64 32
  %128 = getelementptr i8, ptr %next.gep167, i64 48
  %wide.load168 = load <2 x double>, ptr %next.gep167, align 8, !alias.scope !273, !noalias !258
  %wide.load169 = load <2 x double>, ptr %126, align 8, !alias.scope !273, !noalias !258
  %wide.load170 = load <2 x double>, ptr %127, align 8, !alias.scope !273, !noalias !258
  %wide.load171 = load <2 x double>, ptr %128, align 8, !alias.scope !273, !noalias !258
  %129 = fmul <2 x double> %wide.load168, %wide.load168
  %130 = fmul <2 x double> %wide.load169, %wide.load169
  %131 = fmul <2 x double> %wide.load170, %wide.load170
  %132 = fmul <2 x double> %wide.load171, %wide.load171
  %133 = tail call double @llvm.vector.reduce.fadd.v2f64(double %vec.phi165, <2 x double> %129)
  %134 = tail call double @llvm.vector.reduce.fadd.v2f64(double %133, <2 x double> %130)
  %135 = tail call double @llvm.vector.reduce.fadd.v2f64(double %134, <2 x double> %131)
  %136 = tail call double @llvm.vector.reduce.fadd.v2f64(double %135, <2 x double> %132)
  %index.next172 = add nuw i64 %index164, 8
  %137 = icmp eq i64 %index.next172, %n.vec162
  br i1 %137, label %middle.block173, label %vector.body163, !llvm.loop !276

middle.block173:                                  ; preds = %vector.body163
  %cmp.n174 = icmp eq i64 %123, %n.vec162
  br i1 %cmp.n174, label %bb15.loopexit.i.i47, label %bb13.i.i39.preheader199

bb13.i.i39.preheader199:                          ; preds = %bb13.i.i39.preheader, %middle.block173
  %iter1.sroa.0.06.i.i40.ph = phi ptr [ %constant.val.pn.i.i, %bb13.i.i39.preheader ], [ %125, %middle.block173 ]
  %sum_squared.sroa.0.05.i.i41.ph = phi double [ 0.000000e+00, %bb13.i.i39.preheader ], [ %136, %middle.block173 ]
  br label %bb13.i.i39

bb13.i.i39:                                       ; preds = %bb13.i.i39.preheader199, %bb13.i.i39
  %iter1.sroa.0.06.i.i40 = phi ptr [ %_26.i.i42, %bb13.i.i39 ], [ %iter1.sroa.0.06.i.i40.ph, %bb13.i.i39.preheader199 ]
  %sum_squared.sroa.0.05.i.i41 = phi double [ %_0.i2.i.i45, %bb13.i.i39 ], [ %sum_squared.sroa.0.05.i.i41.ph, %bb13.i.i39.preheader199 ]
  %_26.i.i42 = getelementptr inbounds nuw i8, ptr %iter1.sroa.0.06.i.i40, i64 8
  %value.i.i43 = load double, ptr %iter1.sroa.0.06.i.i40, align 8, !alias.scope !273, !noalias !258, !noundef !8
  %_0.i.i.i44 = fmul double %value.i.i43, %value.i.i43
  %_0.i2.i.i45 = fadd double %sum_squared.sroa.0.05.i.i41, %_0.i.i.i44
  %_20.i.i46 = icmp eq ptr %_26.i.i42, %_11.i.i37
  br i1 %_20.i.i46, label %bb15.loopexit.i.i47, label %bb13.i.i39, !llvm.loop !277

bb15.loopexit.i.i47:                              ; preds = %bb13.i.i39, %middle.block173
  %_0.i2.i.i45.lcssa = phi double [ %136, %middle.block173 ], [ %_0.i2.i.i45, %bb13.i.i39 ]
  %138 = tail call double @llvm.sqrt.f64(double %_0.i2.i.i45.lcssa)
  br label %_RINvNtNtCs2rdgasfnGJ9_13vortex_tensor10scalar_fns10arithmetic11l2_norm_rowdECsaHo96IcALPt_24row_fn_performance_probe.exit.i48

_RINvNtNtCs2rdgasfnGJ9_13vortex_tensor10scalar_fns10arithmetic11l2_norm_rowdECsaHo96IcALPt_24row_fn_performance_probe.exit.i48: ; preds = %bb15.loopexit.i.i47, %_RINvNtNtCs2rdgasfnGJ9_13vortex_tensor10scalar_fns10arithmetic17inner_product_rowdECsaHo96IcALPt_24row_fn_performance_probe.exit.i34
  %sum_squared.sroa.0.0.lcssa.i.i49 = phi double [ 0.000000e+00, %_RINvNtNtCs2rdgasfnGJ9_13vortex_tensor10scalar_fns10arithmetic17inner_product_rowdECsaHo96IcALPt_24row_fn_performance_probe.exit.i34 ], [ %138, %bb15.loopexit.i.i47 ]
  %_11.idx.i1.i50 = shl i64 %constant.val2.pn.i13.i, 3
  %_11.i2.i51 = getelementptr inbounds nuw i8, ptr %constant.val.pn.i12.i, i64 %_11.idx.i1.i50
  %_204.i3.i52 = icmp eq i64 %constant.val2.pn.i13.i, 0
  br i1 %_204.i3.i52, label %bb81, label %bb13.i4.i53.preheader

bb13.i4.i53.preheader:                            ; preds = %_RINvNtNtCs2rdgasfnGJ9_13vortex_tensor10scalar_fns10arithmetic11l2_norm_rowdECsaHo96IcALPt_24row_fn_performance_probe.exit.i48
  %139 = add i64 %_11.idx.i1.i50, -8
  %140 = lshr exact i64 %139, 3
  %141 = add nuw nsw i64 %140, 1
  %min.iters.check140 = icmp ult i64 %139, 56
  br i1 %min.iters.check140, label %bb13.i4.i53.preheader198, label %vector.ph141

vector.ph141:                                     ; preds = %bb13.i4.i53.preheader
  %n.vec143 = and i64 %141, 4611686018427387896
  %142 = shl i64 %n.vec143, 3
  %143 = getelementptr i8, ptr %constant.val.pn.i12.i, i64 %142
  br label %vector.body144

vector.body144:                                   ; preds = %vector.body144, %vector.ph141
  %index145 = phi i64 [ 0, %vector.ph141 ], [ %index.next153, %vector.body144 ]
  %vec.phi146 = phi double [ 0.000000e+00, %vector.ph141 ], [ %154, %vector.body144 ]
  %offset.idx147 = shl i64 %index145, 3
  %next.gep148 = getelementptr i8, ptr %constant.val.pn.i12.i, i64 %offset.idx147
  %144 = getelementptr i8, ptr %next.gep148, i64 16
  %145 = getelementptr i8, ptr %next.gep148, i64 32
  %146 = getelementptr i8, ptr %next.gep148, i64 48
  %wide.load149 = load <2 x double>, ptr %next.gep148, align 8, !alias.scope !278, !noalias !255
  %wide.load150 = load <2 x double>, ptr %144, align 8, !alias.scope !278, !noalias !255
  %wide.load151 = load <2 x double>, ptr %145, align 8, !alias.scope !278, !noalias !255
  %wide.load152 = load <2 x double>, ptr %146, align 8, !alias.scope !278, !noalias !255
  %147 = fmul <2 x double> %wide.load149, %wide.load149
  %148 = fmul <2 x double> %wide.load150, %wide.load150
  %149 = fmul <2 x double> %wide.load151, %wide.load151
  %150 = fmul <2 x double> %wide.load152, %wide.load152
  %151 = tail call double @llvm.vector.reduce.fadd.v2f64(double %vec.phi146, <2 x double> %147)
  %152 = tail call double @llvm.vector.reduce.fadd.v2f64(double %151, <2 x double> %148)
  %153 = tail call double @llvm.vector.reduce.fadd.v2f64(double %152, <2 x double> %149)
  %154 = tail call double @llvm.vector.reduce.fadd.v2f64(double %153, <2 x double> %150)
  %index.next153 = add nuw i64 %index145, 8
  %155 = icmp eq i64 %index.next153, %n.vec143
  br i1 %155, label %middle.block154, label %vector.body144, !llvm.loop !281

middle.block154:                                  ; preds = %vector.body144
  %cmp.n155 = icmp eq i64 %141, %n.vec143
  br i1 %cmp.n155, label %bb15.loopexit.i12.i61, label %bb13.i4.i53.preheader198

bb13.i4.i53.preheader198:                         ; preds = %bb13.i4.i53.preheader, %middle.block154
  %iter1.sroa.0.06.i5.i54.ph = phi ptr [ %constant.val.pn.i12.i, %bb13.i4.i53.preheader ], [ %143, %middle.block154 ]
  %sum_squared.sroa.0.05.i6.i55.ph = phi double [ 0.000000e+00, %bb13.i4.i53.preheader ], [ %154, %middle.block154 ]
  br label %bb13.i4.i53

bb13.i4.i53:                                      ; preds = %bb13.i4.i53.preheader198, %bb13.i4.i53
  %iter1.sroa.0.06.i5.i54 = phi ptr [ %_26.i7.i56, %bb13.i4.i53 ], [ %iter1.sroa.0.06.i5.i54.ph, %bb13.i4.i53.preheader198 ]
  %sum_squared.sroa.0.05.i6.i55 = phi double [ %_0.i2.i10.i59, %bb13.i4.i53 ], [ %sum_squared.sroa.0.05.i6.i55.ph, %bb13.i4.i53.preheader198 ]
  %_26.i7.i56 = getelementptr inbounds nuw i8, ptr %iter1.sroa.0.06.i5.i54, i64 8
  %value.i8.i57 = load double, ptr %iter1.sroa.0.06.i5.i54, align 8, !alias.scope !278, !noalias !255, !noundef !8
  %_0.i.i9.i58 = fmul double %value.i8.i57, %value.i8.i57
  %_0.i2.i10.i59 = fadd double %sum_squared.sroa.0.05.i6.i55, %_0.i.i9.i58
  %_20.i11.i60 = icmp eq ptr %_26.i7.i56, %_11.i2.i51
  br i1 %_20.i11.i60, label %bb15.loopexit.i12.i61, label %bb13.i4.i53, !llvm.loop !282

bb15.loopexit.i12.i61:                            ; preds = %bb13.i4.i53, %middle.block154
  %_0.i2.i10.i59.lcssa = phi double [ %154, %middle.block154 ], [ %_0.i2.i10.i59, %bb13.i4.i53 ]
  %156 = tail call double @llvm.sqrt.f64(double %_0.i2.i10.i59.lcssa)
  br label %bb81

bb81:                                             ; preds = %bb15.loopexit.i12.i61, %_RINvNtNtCs2rdgasfnGJ9_13vortex_tensor10scalar_fns10arithmetic11l2_norm_rowdECsaHo96IcALPt_24row_fn_performance_probe.exit.i48
  %sum_squared.sroa.0.0.lcssa.i13.i62 = phi double [ 0.000000e+00, %_RINvNtNtCs2rdgasfnGJ9_13vortex_tensor10scalar_fns10arithmetic11l2_norm_rowdECsaHo96IcALPt_24row_fn_performance_probe.exit.i48 ], [ %156, %bb15.loopexit.i12.i61 ]
  %_0.i.i63 = fmul double %sum_squared.sroa.0.0.lcssa.i.i49, %sum_squared.sroa.0.0.lcssa.i13.i62
  %_0.i1.i64 = fcmp oeq double %_0.i.i63, 0.000000e+00
  %_0.i2.i65 = fdiv double %accum.sroa.0.0.lcssa.i.i.i35, %_0.i.i63
  %_0.sroa.0.0.i66 = select i1 %_0.i1.i64, double 0.000000e+00, double %_0.i2.i65
  store double %_0.sroa.0.0.i66, ptr %_4.i36, align 8, !alias.scope !283, !noalias !288
  %exitcond89.not = icmp eq i64 %95, %4
  br i1 %exitcond89.not, label %bb54, label %bb32

bb42:                                             ; preds = %bb76, %bb41
  %157 = icmp eq i64 %_8.sroa.4.0.i, 0
  br i1 %157, label %bb44, label %bb2.i.i68

bb2.i.i68:                                        ; preds = %bb42
  tail call void @mi_free(ptr noundef nonnull %12) #18, !noalias !291
  br label %bb44

bb44:                                             ; preds = %bb2.i.i68, %bb42
; call core::ptr::drop_glue::<(vortex_array::scalar_fn::unstable::row::types::element::tuple::element_tuple::ArgColumn<vortex_tensor::scalar_fns::row::TensorRow<half::binary16::f16>>, vortex_array::scalar_fn::unstable::row::types::element::tuple::element_tuple::ArgColumn<vortex_tensor::scalar_fns::row::TensorRow<half::binary16::f16>>)>
  call fastcc void @_RINvNtCslWxY2MhVcag_4core3ptr9drop_glueTINtNtNtNtNtNtNtNtCs5mvLrSLkDP1_12vortex_array9scalar_fn8unstable3row5types7element5tuple13element_tuple9ArgColumnINtNtNtCs2rdgasfnGJ9_13vortex_tensor10scalar_fns3row9TensorRowNtNtCsajgNrFuHJYM_4half8binary163f16EEBC_EECsaHo96IcALPt_24row_fn_performance_probe(ptr noalias nofree noundef align 8 dereferenceable(112) %columns)
  br label %bb47

bb62:                                             ; preds = %cleanup17, %cleanup13.loopexit.split-lp, %cleanup12
  %.pn48 = phi { ptr, i32 } [ %lpad.loopexit.split-lp, %cleanup13.loopexit.split-lp ], [ %103, %cleanup17 ], [ %11, %cleanup12 ]
  %158 = icmp eq i64 %_8.sroa.4.0.i, 0
  br i1 %158, label %bb51, label %bb2.i.i.i69

bb2.i.i.i69:                                      ; preds = %bb62
  tail call void @mi_free(ptr noundef nonnull %12) #18, !noalias !294
  br label %bb51

terminate:                                        ; preds = %bb51
  %159 = landingpad { ptr, i32 }
          filter [0 x ptr] zeroinitializer
; call core::panicking::panic_in_cleanup
  call void @_RNvNtCslWxY2MhVcag_4core9panicking16panic_in_cleanup() #17
  unreachable

bb53:                                             ; preds = %bb51
  resume { ptr, i32 } %.pn29
}
