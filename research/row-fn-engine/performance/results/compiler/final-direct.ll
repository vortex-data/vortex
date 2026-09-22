; SPDX-License-Identifier: Apache-2.0
; SPDX-FileCopyrightText: Copyright the Vortex contributors
define internal fastcc void @_RNvCsaHo96IcALPt_24row_fn_performance_probe6direct(ptr dead_on_unwind noalias nofree noundef nonnull writable writeonly align 8 captures(none) dereferenceable(80) %_0, ptr nonnull %array.0.val, ptr captures(address, read_provenance) %array.8.val, ptr noalias nofree noundef nonnull align 8 dereferenceable(40) %ctx, i1 noundef zeroext %collector) unnamed_addr #8 personality ptr @rust_eh_personality {
start:
  %_30 = alloca [40 x i8], align 8
  %_16 = alloca [24 x i8], align 8
  %_13 = alloca [24 x i8], align 8
  %_6 = alloca [80 x i8], align 8
  %input = alloca [32 x i8], align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %input)
  call void @llvm.lifetime.start.p0(ptr nonnull %_6)
  %_18 = atomicrmw add ptr %array.0.val, i64 1 monotonic, align 8
  %_19 = icmp slt i64 %_18, 0
  br i1 %_19, label %bb11, label %bb12

bb12:                                             ; preds = %start
  %0 = icmp ne ptr %array.8.val, null
  tail call void @llvm.assume(i1 %0)
; call <vortex_array::array::typed::Array<vortex_array::arrays::primitive::vtable::Primitive> as vortex_array::executor::Executable>::execute
  call void @_RNvXs8_NtCs5mvLrSLkDP1_12vortex_array9canonicalINtNtNtB7_5array5typed5ArrayNtNtNtNtB7_6arrays9primitive6vtable9PrimitiveENtNtB7_8executor10Executable7execute(ptr noalias nofree noundef nonnull sret([80 x i8]) align 8 captures(none) dereferenceable(80) %_6, ptr noundef nonnull %array.0.val, ptr noalias nofree noundef nonnull readonly align 8 captures(address, read_provenance) dereferenceable(208) %array.8.val, ptr noalias nofree noundef nonnull align 8 dereferenceable(40) %ctx)
  %1 = load i64, ptr %_6, align 8, !range !465, !noundef !12
  %.not = icmp eq i64 %1, -1
  %2 = getelementptr inbounds nuw i8, ptr %_6, i64 8
  %_27.0 = load ptr, ptr %2, align 8
  %3 = getelementptr inbounds nuw i8, ptr %_6, i64 16
  %_27.1 = load ptr, ptr %3, align 8
  br i1 %.not, label %bb16, label %bb15

bb11:                                             ; preds = %start
  tail call void @llvm.trap()
  unreachable

bb15:                                             ; preds = %bb12
  %_28.sroa.6.0._6.sroa_idx = getelementptr inbounds nuw i8, ptr %_6, i64 24
  %_32.sroa.6.0._0.sroa_idx = getelementptr inbounds nuw i8, ptr %_0, i64 24
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(56) %_32.sroa.6.0._0.sroa_idx, ptr noundef nonnull align 8 dereferenceable(56) %_28.sroa.6.0._6.sroa_idx, i64 56, i1 false)
  call void @llvm.lifetime.end.p0(ptr nonnull %_6)
  store i64 %1, ptr %_0, align 8
  %_32.sroa.4.0._0.sroa_idx = getelementptr inbounds nuw i8, ptr %_0, i64 8
  store ptr %_27.0, ptr %_32.sroa.4.0._0.sroa_idx, align 8
  %_32.sroa.5.0._0.sroa_idx = getelementptr inbounds nuw i8, ptr %_0, i64 16
  store ptr %_27.1, ptr %_32.sroa.5.0._0.sroa_idx, align 8
  br label %bb8

bb16:                                             ; preds = %bb12
  call void @llvm.lifetime.end.p0(ptr nonnull %_6)
  call void @llvm.lifetime.start.p0(ptr nonnull %_30)
; call <vortex_array::array::typed::Array<vortex_array::arrays::primitive::vtable::Primitive>>::into_data
  call void @_RNvMs6_NtNtCs5mvLrSLkDP1_12vortex_array5array5typedINtB5_5ArrayNtNtNtNtB9_6arrays9primitive6vtable9PrimitiveE9into_dataCsaHo96IcALPt_24row_fn_performance_probe(ptr noalias nofree noundef nonnull sret([40 x i8]) align 8 captures(address) dereferenceable(40) %_30, ptr noundef nonnull %_27.0, ptr noalias nofree noundef nonnull readonly align 8 captures(address, read_provenance) dereferenceable(208) %_27.1)
; call <vortex_array::arrays::primitive::array::PrimitiveData>::into_buffer::<i64>
  call void @_RINvMs2_NtNtNtCs5mvLrSLkDP1_12vortex_array6arrays9primitive5arrayNtB6_13PrimitiveData11into_bufferxECsaHo96IcALPt_24row_fn_performance_probe(ptr noalias nofree noundef nonnull sret([32 x i8]) align 8 captures(address) dereferenceable(32) %input, ptr noalias nofree noundef nonnull align 8 captures(address) dereferenceable(40) %_30)
  call void @llvm.lifetime.end.p0(ptr nonnull %_30)
  br i1 %collector, label %bb2, label %bb4

bb4:                                              ; preds = %bb16
  call void @llvm.lifetime.start.p0(ptr nonnull %_13)
  %_39 = load ptr, ptr %input, align 8, !nonnull !12, !noundef !12
  %_397 = ptrtoint ptr %_39 to i64
  %4 = getelementptr inbounds nuw i8, ptr %input, i64 8
  %_40 = load i64, ptr %4, align 8, !noundef !12
  call void @llvm.experimental.noalias.scope.decl(metadata !2105)
  %_45.idx = shl nuw nsw i64 %_40, 3
  %5 = icmp eq i64 %_40, 0
  br i1 %5, label %bb19, label %bb3.i1.i

bb3.i1.i:                                         ; preds = %bb4
; call __rustc::__rust_no_alloc_shim_is_unstable_v2
  call void @_RNvCs6rREvFdRhLb_7___rustc35___rust_no_alloc_shim_is_unstable_v2() #28, !noalias !2108
  %_3.i1.i.i = call noalias noundef ptr @mi_malloc_aligned(i64 noundef range(i64 0, -9223372036854775808) %_45.idx, i64 noundef range(i64 1, -9223372036854775807) 8) #28, !noalias !2108
  %6 = icmp eq ptr %_3.i1.i.i, null
  br i1 %6, label %bb3.i.i, label %bb27.i.i.i.preheader

bb27.i.i.i.preheader:                             ; preds = %bb3.i1.i
  %_3.i1.i.i6 = ptrtoint ptr %_3.i1.i.i to i64
  %min.iters.check = icmp ult i64 %_40, 8
  %7 = sub i64 %_3.i1.i.i6, %_397
  %diff.check = icmp ult i64 %7, 64
  %or.cond = or i1 %min.iters.check, %diff.check
  br i1 %or.cond, label %bb27.i.i.i.preheader11, label %vector.ph

vector.ph:                                        ; preds = %bb27.i.i.i.preheader
  %n.vec = and i64 %_40, -8
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %8 = getelementptr inbounds nuw i64, ptr %_39, i64 %index
  %9 = getelementptr inbounds nuw i8, ptr %8, i64 16
  %10 = getelementptr inbounds nuw i8, ptr %8, i64 32
  %11 = getelementptr inbounds nuw i8, ptr %8, i64 48
  %wide.load = load <2 x i64>, ptr %8, align 8, !noalias !2111
  %wide.load8 = load <2 x i64>, ptr %9, align 8, !noalias !2111
  %wide.load9 = load <2 x i64>, ptr %10, align 8, !noalias !2111
  %wide.load10 = load <2 x i64>, ptr %11, align 8, !noalias !2111
  %12 = add <2 x i64> %wide.load, splat (i64 1)
  %13 = add <2 x i64> %wide.load8, splat (i64 1)
  %14 = add <2 x i64> %wide.load9, splat (i64 1)
  %15 = add <2 x i64> %wide.load10, splat (i64 1)
  %16 = getelementptr inbounds nuw i64, ptr %_3.i1.i.i, i64 %index
  %17 = getelementptr inbounds nuw i8, ptr %16, i64 16
  %18 = getelementptr inbounds nuw i8, ptr %16, i64 32
  %19 = getelementptr inbounds nuw i8, ptr %16, i64 48
  store <2 x i64> %12, ptr %16, align 8, !noalias !2116
  store <2 x i64> %13, ptr %17, align 8, !noalias !2116
  store <2 x i64> %14, ptr %18, align 8, !noalias !2116
  store <2 x i64> %15, ptr %19, align 8, !noalias !2116
  %index.next = add nuw i64 %index, 8
  %20 = icmp eq i64 %index.next, %n.vec
  br i1 %20, label %middle.block, label %vector.body, !llvm.loop !2123

middle.block:                                     ; preds = %vector.body
  %cmp.n = icmp eq i64 %_40, %n.vec
  br i1 %cmp.n, label %bb19, label %bb27.i.i.i.preheader11

bb27.i.i.i.preheader11:                           ; preds = %bb27.i.i.i.preheader, %middle.block
  %.ph = phi i64 [ 0, %bb27.i.i.i.preheader ], [ %n.vec, %middle.block ]
  br label %bb27.i.i.i

bb3.i.i:                                          ; preds = %bb3.i1.i
; invoke alloc::raw_vec::handle_error
  invoke void @_RNvNtCs6KVRSXc8uZF_5alloc7raw_vec12handle_error(i64 noundef 8, i64 %_45.idx) #32
          to label %.noexc unwind label %cleanup

.noexc:                                           ; preds = %bb3.i.i
  unreachable

bb27.i.i.i:                                       ; preds = %bb27.i.i.i.preheader11, %bb27.i.i.i
  %21 = phi i64 [ %22, %bb27.i.i.i ], [ %.ph, %bb27.i.i.i.preheader11 ]
  %_45.i.i.i = getelementptr inbounds nuw i64, ptr %_39, i64 %21
  %_45.val.i.i.i = load i64, ptr %_45.i.i.i, align 8, !noalias !2111, !noundef !12
  %_0.i.i.i.i.i = add i64 %_45.val.i.i.i, 1
  %_3.i.i.i.i.i.i = getelementptr inbounds nuw i64, ptr %_3.i1.i.i, i64 %21
  store i64 %_0.i.i.i.i.i, ptr %_3.i.i.i.i.i.i, align 8, !noalias !2116
  %22 = add nuw i64 %21, 1
  %_28.i.i.i = icmp eq i64 %22, %_40
  br i1 %_28.i.i.i, label %bb19, label %bb27.i.i.i, !llvm.loop !2126

bb2:                                              ; preds = %bb16
  %_35 = load ptr, ptr %input, align 8, !nonnull !12, !noundef !12
  %23 = getelementptr inbounds nuw i8, ptr %input, i64 8
  %_36 = load i64, ptr %23, align 8, !noundef !12
; invoke <i64 as vortex_array::scalar_fn::unstable::row::types::element::output::OutputElement>::build_from::<&[i64], row_fn_performance_probe::direct::{closure#0}>
  %24 = invoke { ptr, ptr } @_RINvYxNtNtNtNtNtNtNtCs5mvLrSLkDP1_12vortex_array9scalar_fn8unstable3row5types7element6output13OutputElement10build_fromRSxNCNvCsaHo96IcALPt_24row_fn_performance_probe6direct0EB20_(ptr noalias nofree noundef nonnull readonly align 8 captures(address, read_provenance) %_35, i64 noundef %_36)
          to label %bb6 unwind label %cleanup

cleanup:                                          ; preds = %bb3.i.i, %bb19, %bb2
  %25 = landingpad { ptr, i32 }
          cleanup
  call void @llvm.experimental.noalias.scope.decl(metadata !2127)
  %26 = getelementptr inbounds nuw i8, ptr %input, i64 16
  call void @llvm.experimental.noalias.scope.decl(metadata !2130)
  %27 = load ptr, ptr %26, align 8, !alias.scope !2133, !noundef !12
  %28 = icmp eq ptr %27, null
  br i1 %28, label %bb10, label %bb2.i.i

bb2.i.i:                                          ; preds = %cleanup
  %_2.i.i.i.i = atomicrmw sub ptr %27, i64 1 release, align 8, !noalias !2134
  %29 = icmp eq i64 %_2.i.i.i.i, 1
  br i1 %29, label %bb2.i.i.i.i, label %bb10

bb2.i.i.i.i:                                      ; preds = %bb2.i.i
  fence acquire
; invoke <alloc::sync::Arc<vortex_buffer::allocation::BufferBacking>>::drop_slow
  invoke void @_RNvMsn_NtCs6KVRSXc8uZF_5alloc4syncINtB5_3ArcNtNtCsEOnbuuTlDO_13vortex_buffer10allocation13BufferBackingE9drop_slowBK_(ptr noalias nofree noundef nonnull align 8 dereferenceable(8) %26) #27
          to label %bb10 unwind label %terminate

bb19:                                             ; preds = %bb27.i.i.i, %middle.block, %bb4
  %30 = phi ptr [ inttoptr (i64 8 to ptr), %bb4 ], [ %_3.i1.i.i, %middle.block ], [ %_3.i1.i.i, %bb27.i.i.i ]
  store i64 %_40, ptr %_13, align 8, !alias.scope !2105
  %vector.sroa.4.0._0.sroa_idx.i = getelementptr inbounds nuw i8, ptr %_13, i64 8
  store ptr %30, ptr %vector.sroa.4.0._0.sroa_idx.i, align 8, !alias.scope !2105
  %vector.sroa.5.0._0.sroa_idx.i = getelementptr inbounds nuw i8, ptr %_13, i64 16
  store i64 %_40, ptr %vector.sroa.5.0._0.sroa_idx.i, align 8, !alias.scope !2105
  call void @llvm.lifetime.start.p0(ptr nonnull %_16)
  store i64 0, ptr %_16, align 8
; invoke <vortex_array::array::typed::Array<vortex_array::arrays::primitive::vtable::Primitive>>::new::<i64, alloc::vec::Vec<i64>>
  %31 = invoke { ptr, ptr } @_RINvMs1_NtNtNtCs5mvLrSLkDP1_12vortex_array6arrays9primitive5arrayINtNtNtBc_5array5typed5ArrayNtNtB8_6vtable9PrimitiveE3newxINtNtCs6KVRSXc8uZF_5alloc3vec3VecxEECsaHo96IcALPt_24row_fn_performance_probe(ptr noalias nofree noundef nonnull align 8 captures(address) dereferenceable(24) %_13, ptr noalias nofree noundef nonnull align 8 captures(address) dereferenceable(24) %_16)
          to label %bb5 unwind label %cleanup

bb5:                                              ; preds = %bb19
  call void @llvm.lifetime.end.p0(ptr nonnull %_16)
  call void @llvm.lifetime.end.p0(ptr nonnull %_13)
  br label %bb6

bb6:                                              ; preds = %bb5, %bb2
  %.pn = phi { ptr, ptr } [ %31, %bb5 ], [ %24, %bb2 ]
  %_10.sroa.5.0 = extractvalue { ptr, ptr } %.pn, 1
  %_10.sroa.0.0 = extractvalue { ptr, ptr } %.pn, 0
  %32 = icmp ne ptr %_10.sroa.0.0, null
  call void @llvm.assume(i1 %32)
  %33 = icmp ne ptr %_10.sroa.5.0, null
  call void @llvm.assume(i1 %33)
  %34 = getelementptr inbounds nuw i8, ptr %_0, i64 8
  store ptr %_10.sroa.0.0, ptr %34, align 8
  %35 = getelementptr inbounds nuw i8, ptr %_0, i64 16
  store ptr %_10.sroa.5.0, ptr %35, align 8
  store i64 -1, ptr %_0, align 8
  call void @llvm.experimental.noalias.scope.decl(metadata !2139)
  %36 = getelementptr inbounds nuw i8, ptr %input, i64 16
  call void @llvm.experimental.noalias.scope.decl(metadata !2142)
  %37 = load ptr, ptr %36, align 8, !alias.scope !2145, !noundef !12
  %38 = icmp eq ptr %37, null
  br i1 %38, label %bb8, label %bb2.i.i5

bb2.i.i5:                                         ; preds = %bb6
  %_2.i.i.i.i6 = atomicrmw sub ptr %37, i64 1 release, align 8, !noalias !2146
  %39 = icmp eq i64 %_2.i.i.i.i6, 1
  br i1 %39, label %bb2.i.i.i.i7, label %bb8

bb2.i.i.i.i7:                                     ; preds = %bb2.i.i5
  fence acquire
; call <alloc::sync::Arc<vortex_buffer::allocation::BufferBacking>>::drop_slow
  call void @_RNvMsn_NtCs6KVRSXc8uZF_5alloc4syncINtB5_3ArcNtNtCsEOnbuuTlDO_13vortex_buffer10allocation13BufferBackingE9drop_slowBK_(ptr noalias nofree noundef nonnull align 8 dereferenceable(8) %36) #27
  br label %bb8

bb8:                                              ; preds = %bb2.i.i.i.i7, %bb2.i.i5, %bb6, %bb15
  call void @llvm.lifetime.end.p0(ptr nonnull %input)
  ret void

terminate:                                        ; preds = %bb2.i.i.i.i
  %40 = landingpad { ptr, i32 }
          filter [0 x ptr] zeroinitializer
; call core::panicking::panic_in_cleanup
  call void @_RNvNtCslWxY2MhVcag_4core9panicking16panic_in_cleanup() #30
  unreachable

bb10:                                             ; preds = %bb2.i.i.i.i, %bb2.i.i, %cleanup
  resume { ptr, i32 } %25
}
