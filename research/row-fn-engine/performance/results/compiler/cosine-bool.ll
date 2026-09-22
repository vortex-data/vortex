; SPDX-License-Identifier: Apache-2.0
; SPDX-FileCopyrightText: Copyright the Vortex contributors
; Function excerpt. See provenance.json for the complete artifact.
define hidden void @_RINvNtNtNtNtNtCs5mvLrSLkDP1_12vortex_array9scalar_fn8unstable3row7execute5retry27execute_owned_dense_attemptTxxEbubNCINvYINtNtNtB6_7visitor5retry21ExecuteDenseWithRetryINtCsaHo96IcALPt_24row_fn_performance_probe9PredicateKb0_EENtNtB20_11row_visitor10RowVisitor19visit_deferred_boolB1I_bKB3y_NCINvXs_B2J_B2G_NtNtB6_6row_fn5RowFn8dispatchB1V_E0NCB4H_s_0E0NCB1R_s_0B5u_EB2J_(ptr dead_on_unwind noalias nofree noundef writable writeonly sret([88 x i8]) align 8 captures(none) dereferenceable(88) %_0, ptr noundef nonnull %args.0, ptr noalias nofree noundef readonly align 8 captures(address, read_provenance) dereferenceable(48) %args.1, ptr noalias nofree noundef align 8 dereferenceable(40) %ctx) unnamed_addr #0 personality ptr @rust_eh_personality {
start:
  %_6.i = alloca [16 x i8], align 8
  %_75 = alloca [24 x i8], align 8
  %_69 = alloca [80 x i8], align 8
  %args1 = alloca [16 x i8], align 8
  %_45 = alloca [16 x i8], align 8
  %_42 = alloca [80 x i8], align 8
  %args = alloca [16 x i8], align 8
  %_30 = alloca [16 x i8], align 8
  %_27 = alloca [80 x i8], align 8
  %row_count = alloca [8 x i8], align 8
  %_8 = alloca [80 x i8], align 8
  %_7.sroa.6 = alloca [64 x i8], align 8
  %columns = alloca [64 x i8], align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %columns)
  call void @llvm.lifetime.start.p0(ptr nonnull %_7.sroa.6)
  call void @llvm.lifetime.start.p0(ptr nonnull %_8)
; call <(i64, i64) as vortex_array::scalar_fn::unstable::row::types::element::tuple::element_tuple::ElementTuple>::decode
  call void @_RNvXs4_NtNtNtNtNtNtNtCs5mvLrSLkDP1_12vortex_array9scalar_fn8unstable3row5types7element5tuple13element_tupleTxxENtB5_12ElementTuple6decodeCsaHo96IcALPt_24row_fn_performance_probe(ptr noalias nofree noundef nonnull sret([80 x i8]) align 8 captures(none) dereferenceable(80) %_8, ptr noundef nonnull %args.0, ptr noalias nofree noundef nonnull readonly align 8 captures(address, read_provenance) dereferenceable(48) %args.1, ptr noalias nofree noundef nonnull align 8 dereferenceable(40) %ctx)
  %0 = load i64, ptr %_8, align 8, !range !197, !noundef !6
  %.not = icmp eq i64 %0, -1
  %1 = getelementptr inbounds nuw i8, ptr %_8, i64 8
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(64) %_7.sroa.6, ptr noundef nonnull align 8 dereferenceable(64) %1, i64 64, i1 false)
  br i1 %.not, label %bb60, label %bb59

common.resume:                                    ; preds = %bb42, %cleanup.i26
  %common.resume.op = phi { ptr, i32 } [ %.pn15, %bb42 ], [ %76, %cleanup.i26 ]
  resume { ptr, i32 } %common.resume.op

bb59:                                             ; preds = %start
  %_85.sroa.5.0._8.sroa_idx = getelementptr inbounds nuw i8, ptr %_8, i64 72
  %_85.sroa.5.0.copyload = load i64, ptr %_85.sroa.5.0._8.sroa_idx, align 8
  call void @llvm.lifetime.end.p0(ptr nonnull %_8)
  %_88.sroa.4.0..sroa_idx = getelementptr inbounds nuw i8, ptr %_0, i64 16
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(64) %_88.sroa.4.0..sroa_idx, ptr noundef nonnull align 8 dereferenceable(64) %_7.sroa.6, i64 64, i1 false)
  %2 = getelementptr inbounds nuw i8, ptr %_0, i64 8
  store i64 %0, ptr %2, align 8
  %_88.sroa.5.0..sroa_idx = getelementptr inbounds nuw i8, ptr %_0, i64 80
  store i64 %_85.sroa.5.0.copyload, ptr %_88.sroa.5.0..sroa_idx, align 8
  store i64 1, ptr %_0, align 8
  call void @llvm.lifetime.end.p0(ptr nonnull %_7.sroa.6)
  br label %bb40

bb60:                                             ; preds = %start
  call void @llvm.lifetime.end.p0(ptr nonnull %_8)
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(64) %columns, ptr noundef nonnull align 8 dereferenceable(64) %_7.sroa.6, i64 64, i1 false)
  call void @llvm.lifetime.end.p0(ptr nonnull %_7.sroa.6)
  call void @llvm.lifetime.start.p0(ptr nonnull %row_count)
  %3 = getelementptr inbounds nuw i8, ptr %args.1, i64 40
  %4 = load ptr, ptr %3, align 8, !invariant.load !6, !nonnull !6
  %5 = invoke noundef i64 %4(ptr noundef nonnull %args.0)
          to label %bb5 unwind label %cleanup3

cleanup3:                                         ; preds = %bb62, %bb60
  %6 = landingpad { ptr, i32 }
          cleanup
  br label %bb42

bb5:                                              ; preds = %bb60
  store i64 %5, ptr %row_count, align 8
  %_39.not.i = icmp slt i64 %5, 0
  br i1 %_39.not.i, label %bb62, label %bb18.i, !prof !198

bb18.i:                                           ; preds = %bb5
  %7 = icmp eq i64 %5, 0
  br i1 %7, label %bb70, label %bb3.i8

bb3.i8:                                           ; preds = %bb18.i
; call __rustc::__rust_no_alloc_shim_is_unstable_v2
  tail call void @_RNvCs6rREvFdRhLb_7___rustc35___rust_no_alloc_shim_is_unstable_v2() #18, !noalias !398
  %_3.i1.i = tail call noalias noundef ptr @mi_malloc_aligned(i64 noundef range(i64 0, -9223372036854775808) %5, i64 noundef range(i64 1, -9223372036854775807) 1) #18, !noalias !398
  %8 = icmp eq ptr %_3.i1.i, null
  br i1 %8, label %bb62, label %bb10.i

bb10.i:                                           ; preds = %bb3.i8
  %9 = ptrtoint ptr %_3.i1.i to i64
  br label %bb70

bb62:                                             ; preds = %bb5, %bb3.i8
  %_93.sroa.4.0.ph = phi i64 [ 1, %bb3.i8 ], [ 0, %bb5 ]
; invoke alloc::raw_vec::handle_error
  invoke void @_RNvNtCs6KVRSXc8uZF_5alloc7raw_vec12handle_error(i64 noundef %_93.sroa.4.0.ph, i64 %5) #15
          to label %unreachable unwind label %cleanup3

bb70:                                             ; preds = %bb10.i, %bb18.i
  %_93.sroa.9.0 = phi i64 [ %9, %bb10.i ], [ 1, %bb18.i ]
  %10 = inttoptr i64 %_93.sroa.9.0 to ptr
  %11 = load ptr, ptr %columns, align 8, !alias.scope !401, !noalias !404, !noundef !6
  %12 = icmp eq ptr %11, null
  br i1 %12, label %bb2.i13, label %bb10.i10

bb10.i10:                                         ; preds = %bb70
  %13 = getelementptr inbounds nuw i8, ptr %columns, i64 32
  %14 = load ptr, ptr %13, align 8, !alias.scope !401, !noalias !404, !noundef !6
  %15 = icmp eq ptr %14, null
  %16 = getelementptr inbounds nuw i8, ptr %columns, i64 8
  %columns.val2.i = load i64, ptr %16, align 8
  br i1 %15, label %bb45, label %bb8

bb52.thread33.loopexit:                           ; preds = %panic.i.i5.i.invoke
  %lpad.loopexit = landingpad { ptr, i32 }
          cleanup
  br label %bb51

bb52.thread33.loopexit.split-lp:                  ; preds = %bb16, %bb1.i
  %lpad.loopexit.split-lp = landingpad { ptr, i32 }
          cleanup
  br label %bb51

bb52:                                             ; preds = %bb29
  %lpad.thr_comm.split-lp = landingpad { ptr, i32 }
          cleanup
  br label %bb42

bb45:                                             ; preds = %bb10.i10
  %17 = icmp eq i64 %columns.val2.i, %5
  br i1 %17, label %bb2.i13.thread, label %bb16, !prof !207

bb2.i13.thread:                                   ; preds = %bb45
  %_6.i1475 = getelementptr inbounds nuw i8, ptr %columns, i64 32
  %18 = getelementptr inbounds nuw i8, ptr %columns, i64 40
  br label %bb18

bb2.i13:                                          ; preds = %bb70
  %19 = getelementptr inbounds nuw i8, ptr %columns, i64 8
  %_6.i14.phi.trans.insert = getelementptr inbounds nuw i8, ptr %columns, i64 32
  %_6.val.i.pre = load ptr, ptr %_6.i14.phi.trans.insert, align 8, !alias.scope !406
  %20 = icmp eq ptr %_6.val.i.pre, null
  %_6.i14 = getelementptr inbounds nuw i8, ptr %columns, i64 32
  %21 = getelementptr inbounds nuw i8, ptr %columns, i64 40
  %_6.val1.i = load i64, ptr %21, align 8
  %22 = icmp eq i64 %_6.val1.i, %5
  %or.cond = select i1 %20, i1 true, i1 %22, !prof !409
  br i1 %or.cond, label %bb18, label %bb16, !prof !410

bb52.thread.loopexit.split-lp:                    ; preds = %bb10
  %lpad.loopexit.split-lp44 = landingpad { ptr, i32 }
          cleanup
  br label %bb51

bb8:                                              ; preds = %bb10.i10
  %23 = getelementptr inbounds nuw i8, ptr %columns, i64 40
  %.val2.i = load i64, ptr %23, align 8, !alias.scope !401, !noalias !404, !noundef !6
  %_3.i = icmp eq i64 %columns.val2.i, %5
  %_6.i11 = icmp eq i64 %.val2.i, %5
  %_0.sroa.0.0.off0.i = and i1 %_3.i, %_6.i11
  br i1 %_0.sroa.0.0.off0.i, label %bb9, label %bb10, !prof !208

bb10:                                             ; preds = %bb8
  call void @llvm.lifetime.start.p0(ptr nonnull %_27)
  call void @llvm.lifetime.start.p0(ptr nonnull %_30)
  call void @llvm.lifetime.start.p0(ptr nonnull %args)
  store ptr %row_count, ptr %args, align 8
  %_33.sroa.4.0..sroa_idx = getelementptr inbounds nuw i8, ptr %args, i64 8
  store ptr @_RNvXsi_NtNtNtCslWxY2MhVcag_4core3fmt3num3impjNtB9_7Display3fmt, ptr %_33.sroa.4.0..sroa_idx, align 8
  store ptr @alloc_8cb1b9e1c7b448ff8aa56230a8661784, ptr %_30, align 8
  %24 = getelementptr inbounds nuw i8, ptr %_30, i64 8
  store ptr %args, ptr %24, align 8
; invoke vortex_error::__private::fmt_err
  invoke void @_RNvNtCs4jPh2r5lbWM_12vortex_error9___private7fmt_err(ptr noalias nofree noundef nonnull sret([80 x i8]) align 8 captures(address) dereferenceable(80) %_27, ptr noundef nonnull @_RNcNtNtCs4jPh2r5lbWM_12vortex_error11VortexError5Other0, ptr noalias nofree noundef nonnull readonly align 8 captures(address, read_provenance) dereferenceable(16) %_30)
          to label %bb11 unwind label %bb52.thread.loopexit.split-lp

bb9:                                              ; preds = %bb8
  %_365.not.i = icmp eq i64 %5, 0
  br i1 %_365.not.i, label %bb29.sink.split, label %iter.check

iter.check:                                       ; preds = %bb9
  %min.iters.check = icmp ult i64 %5, 4
  br i1 %min.iters.check, label %bb15.i.preheader, label %vector.memcheck

vector.memcheck:                                  ; preds = %iter.check
  %scevgep = getelementptr i8, ptr %10, i64 %5
  %25 = shl i64 %5, 3
  %scevgep87 = getelementptr i8, ptr %11, i64 %25
  %scevgep88 = getelementptr i8, ptr %14, i64 %25
  %bound0 = icmp ugt ptr %scevgep87, %10
  %bound1 = icmp ult ptr %11, %scevgep
  %found.conflict = and i1 %bound0, %bound1
  %bound089 = icmp ugt ptr %scevgep88, %10
  %bound190 = icmp ult ptr %14, %scevgep
  %found.conflict91 = and i1 %bound089, %bound190
  %conflict.rdx = or i1 %found.conflict, %found.conflict91
  br i1 %conflict.rdx, label %bb15.i.preheader, label %vector.main.loop.iter.check

vector.main.loop.iter.check:                      ; preds = %vector.memcheck
  %min.iters.check92 = icmp ult i64 %5, 16
  br i1 %min.iters.check92, label %vec.epilog.ph, label %vector.ph

vector.ph:                                        ; preds = %vector.main.loop.iter.check
  %n.mod.vf = and i64 %5, 12
  %n.vec = and i64 %5, 9223372036854775792
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %vec.phi = phi <16 x i1> [ zeroinitializer, %vector.ph ], [ %32, %vector.body ]
  %26 = getelementptr inbounds nuw i64, ptr %11, i64 %index
  %wide.load = load <16 x i64>, ptr %26, align 8, !alias.scope !411, !noalias !414
  %27 = getelementptr inbounds nuw i64, ptr %14, i64 %index
  %wide.load93 = load <16 x i64>, ptr %27, align 8, !alias.scope !417, !noalias !419
  %28 = icmp eq <16 x i64> %wide.load, splat (i64 9223372036854775807)
  %29 = icmp eq <16 x i64> %wide.load93, splat (i64 9223372036854775807)
  %30 = or <16 x i1> %28, %29
  %31 = icmp slt <16 x i64> %wide.load, %wide.load93
  %32 = or <16 x i1> %vec.phi, %30
  %33 = getelementptr inbounds nuw i8, ptr %10, i64 %index
  %34 = zext <16 x i1> %31 to <16 x i8>
  store <16 x i8> %34, ptr %33, align 1, !alias.scope !422, !noalias !426
  %index.next = add nuw i64 %index, 16
  %35 = icmp eq i64 %index.next, %n.vec
  br i1 %35, label %middle.block, label %vector.body, !llvm.loop !428

middle.block:                                     ; preds = %vector.body
  %36 = bitcast <16 x i1> %32 to i16
  %37 = icmp ne i16 %36, 0
  %cmp.n = icmp eq i64 %5, %n.vec
  br i1 %cmp.n, label %bb13, label %vec.epilog.iter.check

vec.epilog.iter.check:                            ; preds = %middle.block
  %min.epilog.iters.check = icmp eq i64 %n.mod.vf, 0
  br i1 %min.epilog.iters.check, label %bb15.i.preheader, label %vec.epilog.ph, !prof !229

vec.epilog.ph:                                    ; preds = %vector.main.loop.iter.check, %vec.epilog.iter.check
  %bc.resume.val = phi i64 [ %n.vec, %vec.epilog.iter.check ], [ 0, %vector.main.loop.iter.check ]
  %bc.merge.rdx = phi i1 [ %37, %vec.epilog.iter.check ], [ false, %vector.main.loop.iter.check ]
  %n.vec95 = and i64 %5, 9223372036854775804
  %38 = insertelement <4 x i1> <i1 poison, i1 false, i1 false, i1 false>, i1 %bc.merge.rdx, i64 0
  br label %vec.epilog.vector.body

vec.epilog.vector.body:                           ; preds = %vec.epilog.vector.body, %vec.epilog.ph
  %index100 = phi i64 [ %bc.resume.val, %vec.epilog.ph ], [ %index.next105, %vec.epilog.vector.body ]
  %vec.phi102 = phi <4 x i1> [ %38, %vec.epilog.ph ], [ %45, %vec.epilog.vector.body ]
  %39 = getelementptr inbounds nuw i64, ptr %11, i64 %index100
  %wide.load103 = load <4 x i64>, ptr %39, align 8, !alias.scope !411, !noalias !414
  %40 = getelementptr inbounds nuw i64, ptr %14, i64 %index100
  %wide.load104 = load <4 x i64>, ptr %40, align 8, !alias.scope !417, !noalias !419
  %41 = icmp eq <4 x i64> %wide.load103, splat (i64 9223372036854775807)
  %42 = icmp eq <4 x i64> %wide.load104, splat (i64 9223372036854775807)
  %43 = or <4 x i1> %41, %42
  %44 = icmp slt <4 x i64> %wide.load103, %wide.load104
  %45 = or <4 x i1> %vec.phi102, %43
  %46 = getelementptr inbounds nuw i8, ptr %10, i64 %index100
  %47 = zext <4 x i1> %44 to <4 x i8>
  store <4 x i8> %47, ptr %46, align 1, !alias.scope !422, !noalias !426
  %index.next105 = add nuw i64 %index100, 4
  %48 = icmp eq i64 %index.next105, %n.vec95
  br i1 %48, label %vec.epilog.middle.block, label %vec.epilog.vector.body, !llvm.loop !429

vec.epilog.middle.block:                          ; preds = %vec.epilog.vector.body
  %49 = bitcast <4 x i1> %45 to i4
  %50 = icmp ne i4 %49, 0
  %cmp.n107 = icmp eq i64 %5, %n.vec95
  br i1 %cmp.n107, label %bb13, label %bb15.i.preheader

bb15.i.preheader:                                 ; preds = %vector.memcheck, %iter.check, %vec.epilog.iter.check, %vec.epilog.middle.block
  %iter.sroa.0.07.i.ph = phi i64 [ 0, %iter.check ], [ 0, %vector.memcheck ], [ %n.vec, %vec.epilog.iter.check ], [ %n.vec95, %vec.epilog.middle.block ]
  %failed.sroa.0.0.off06.i.ph = phi i1 [ false, %iter.check ], [ false, %vector.memcheck ], [ %37, %vec.epilog.iter.check ], [ %50, %vec.epilog.middle.block ]
  br label %bb15.i

bb11:                                             ; preds = %bb10
  %51 = getelementptr inbounds nuw i8, ptr %_0, i64 8
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(80) %51, ptr noundef nonnull align 8 dereferenceable(80) %_27, i64 80, i1 false)
  store i64 1, ptr %_0, align 8
  call void @llvm.lifetime.end.p0(ptr nonnull %_27)
  call void @llvm.lifetime.end.p0(ptr nonnull %args)
  call void @llvm.lifetime.end.p0(ptr nonnull %_30)
  br label %bb35

bb35:                                             ; preds = %bb17, %bb11
  br i1 %7, label %bb36, label %bb2.i.i.i17

bb2.i.i.i17:                                      ; preds = %bb35
  call void @mi_free(ptr noundef nonnull %10) #18, !noalias !430
  br label %bb36

bb15.i:                                           ; preds = %bb15.i.preheader, %bb15.i
  %iter.sroa.0.07.i = phi i64 [ %52, %bb15.i ], [ %iter.sroa.0.07.i.ph, %bb15.i.preheader ]
  %failed.sroa.0.0.off06.i = phi i1 [ %55, %bb15.i ], [ %failed.sroa.0.0.off06.i.ph, %bb15.i.preheader ]
  %52 = add nuw i64 %iter.sroa.0.07.i, 1
  %_4.i.i20 = getelementptr inbounds nuw i64, ptr %11, i64 %iter.sroa.0.07.i
  %_0.i.i = load i64, ptr %_4.i.i20, align 8, !noalias !414, !noundef !6
  %_5.i.i23 = icmp ult i64 %iter.sroa.0.07.i, %5
  tail call void @llvm.assume(i1 %_5.i.i23)
  %_4.i.i24 = getelementptr inbounds nuw i64, ptr %14, i64 %iter.sroa.0.07.i
  %_0.i.i25 = load i64, ptr %_4.i.i24, align 8, !noalias !419, !noundef !6
  %53 = icmp eq i64 %_0.i.i, 9223372036854775807
  %54 = icmp eq i64 %_0.i.i25, 9223372036854775807
  %_6.sroa.0.0.off0.i.i.i.i = or i1 %53, %54
  %_5.i.i.i.i = icmp slt i64 %_0.i.i, %_0.i.i25
  %55 = or i1 %failed.sroa.0.0.off06.i, %_6.sroa.0.0.off0.i.i.i.i
  %_40.i = getelementptr inbounds nuw i8, ptr %10, i64 %iter.sroa.0.07.i
  %_41.i = zext i1 %_5.i.i.i.i to i8
  store i8 %_41.i, ptr %_40.i, align 1, !alias.scope !433, !noalias !434
  %exitcond.not.i = icmp eq i64 %52, %5
  br i1 %exitcond.not.i, label %bb13, label %bb15.i, !llvm.loop !435

bb13:                                             ; preds = %bb15.i, %vec.epilog.middle.block, %middle.block
  %.lcssa86 = phi i1 [ %50, %vec.epilog.middle.block ], [ %37, %middle.block ], [ %55, %bb15.i ]
  %56 = load i64, ptr %row_count, align 8, !noundef !6
  call void @llvm.lifetime.start.p0(ptr nonnull %_69)
  br i1 %.lcssa86, label %bb1.i, label %bb29, !prof !237

bb26:                                             ; preds = %bb24
  %57 = load i64, ptr %row_count, align 8, !noundef !6
  call void @llvm.lifetime.start.p0(ptr nonnull %_69)
  br i1 %87, label %bb1.i, label %bb29, !prof !7

bb1.i:                                            ; preds = %bb26, %bb13
  %values.sroa.12.0 = phi i64 [ %57, %bb26 ], [ %56, %bb13 ]
  call void @llvm.lifetime.start.p0(ptr nonnull %_6.i), !noalias !436
  store ptr @alloc_78c2b751c4d3e2d7197d387cac5cfac2, ptr %_6.i, align 8, !noalias !436
  %58 = getelementptr inbounds nuw i8, ptr %_6.i, i64 8
  store ptr inttoptr (i64 55 to ptr), ptr %58, align 8, !noalias !436
; invoke vortex_error::__private::fmt_err
  invoke void @_RNvNtCs4jPh2r5lbWM_12vortex_error9___private7fmt_err(ptr noalias nofree noundef nonnull sret([80 x i8]) align 8 captures(address) dereferenceable(80) %_69, ptr noundef nonnull @_RNcNtNtCs4jPh2r5lbWM_12vortex_error11VortexError5Other0, ptr noalias nofree noundef nonnull readonly align 8 captures(address, read_provenance) dereferenceable(16) %_6.i) #17
          to label %bb27 unwind label %bb52.thread33.loopexit.split-lp

bb16:                                             ; preds = %bb2.i13, %bb45
  call void @llvm.lifetime.start.p0(ptr nonnull %_42)
  call void @llvm.lifetime.start.p0(ptr nonnull %_45)
  call void @llvm.lifetime.start.p0(ptr nonnull %args1)
  store ptr %row_count, ptr %args1, align 8
  %_48.sroa.4.0..sroa_idx = getelementptr inbounds nuw i8, ptr %args1, i64 8
  store ptr @_RNvXsi_NtNtNtCslWxY2MhVcag_4core3fmt3num3impjNtB9_7Display3fmt, ptr %_48.sroa.4.0..sroa_idx, align 8
  store ptr @alloc_8cb1b9e1c7b448ff8aa56230a8661784, ptr %_45, align 8
  %59 = getelementptr inbounds nuw i8, ptr %_45, i64 8
  store ptr %args1, ptr %59, align 8
; invoke vortex_error::__private::fmt_err
  invoke void @_RNvNtCs4jPh2r5lbWM_12vortex_error9___private7fmt_err(ptr noalias nofree noundef nonnull sret([80 x i8]) align 8 captures(address) dereferenceable(80) %_42, ptr noundef nonnull @_RNcNtNtCs4jPh2r5lbWM_12vortex_error11VortexError5Other0, ptr noalias nofree noundef nonnull readonly align 8 captures(address, read_provenance) dereferenceable(16) %_45)
          to label %bb17 unwind label %bb52.thread33.loopexit.split-lp

bb17:                                             ; preds = %bb16
  %60 = getelementptr inbounds nuw i8, ptr %_0, i64 8
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(80) %60, ptr noundef nonnull align 8 dereferenceable(80) %_42, i64 80, i1 false)
  store i64 1, ptr %_0, align 8
  call void @llvm.lifetime.end.p0(ptr nonnull %_42)
  call void @llvm.lifetime.end.p0(ptr nonnull %args1)
  call void @llvm.lifetime.end.p0(ptr nonnull %_45)
  br label %bb35

bb36:                                             ; preds = %bb2.i.i.i17, %bb35
  call void @llvm.lifetime.end.p0(ptr nonnull %row_count)
; call core::ptr::drop_glue::<(vortex_array::scalar_fn::unstable::row::types::element::tuple::element_tuple::ArgColumn<row_fn_performance_probe::SelectedI64<false>>, vortex_array::scalar_fn::unstable::row::types::element::tuple::element_tuple::ArgColumn<row_fn_performance_probe::SelectedI64<false>>)>
  call fastcc void @_RINvNtCslWxY2MhVcag_4core3ptr9drop_glueTINtNtNtNtNtNtNtNtCs5mvLrSLkDP1_12vortex_array9scalar_fn8unstable3row5types7element5tuple13element_tuple9ArgColumnINtCsaHo96IcALPt_24row_fn_performance_probe11SelectedI64Kb0_EEBC_EEB2u_(ptr noalias nofree noundef align 8 dereferenceable(64) %columns)
  br label %bb40

bb18:                                             ; preds = %bb2.i13.thread, %bb2.i13
  %61 = phi ptr [ %18, %bb2.i13.thread ], [ %21, %bb2.i13 ]
  %_6.i1478 = phi ptr [ %_6.i1475, %bb2.i13.thread ], [ %_6.i14, %bb2.i13 ]
  %62 = phi ptr [ %16, %bb2.i13.thread ], [ %19, %bb2.i13 ]
  %63 = getelementptr inbounds nuw i8, ptr %10, i64 %5
  %_7.i.i47 = icmp samesign eq i64 %5, 0
  br i1 %_7.i.i47, label %bb29.sink.split, label %bb21

bb21:                                             ; preds = %bb18, %bb24
  %accumulated_failure.sroa.0.0.off050 = phi i1 [ %87, %bb24 ], [ false, %bb18 ]
  %iter.sroa.0.049 = phi ptr [ %_17.i.i, %bb24 ], [ %10, %bb18 ]
  %iter.sroa.7.048 = phi i64 [ %_9.0.i, %bb24 ], [ 0, %bb18 ]
  tail call void @llvm.experimental.noalias.scope.decl(metadata !439)
  %columns.val.i26 = load ptr, ptr %columns, align 8, !alias.scope !439, !noundef !6
  %columns.val2.i27 = load i64, ptr %62, align 8, !alias.scope !439
  %64 = icmp eq ptr %columns.val.i26, null
  br i1 %64, label %_RNvMNtNtNtNtNtNtNtCs5mvLrSLkDP1_12vortex_array9scalar_fn8unstable3row5types7element5tuple13element_tupleINtB2_9ArgColumnxE3getCsaHo96IcALPt_24row_fn_performance_probe.exit.i, label %bb3.i.i

bb3.i.i:                                          ; preds = %bb21
  %_3.i.i.i = icmp ult i64 %iter.sroa.7.048, %columns.val2.i27
  br i1 %_3.i.i.i, label %_RNvXNtNtNtNtNtNtCs5mvLrSLkDP1_12vortex_array9scalar_fn8unstable3row5types7element9primitivexNtNtB4_5input12InputElement3getCsaHo96IcALPt_24row_fn_performance_probe.exit.i.i, label %panic.i.i5.i.invoke

_RNvXNtNtNtNtNtNtCs5mvLrSLkDP1_12vortex_array9scalar_fn8unstable3row5types7element9primitivexNtNtB4_5input12InputElement3getCsaHo96IcALPt_24row_fn_performance_probe.exit.i.i: ; preds = %bb3.i.i
  %65 = getelementptr inbounds nuw i64, ptr %columns.val.i26, i64 %iter.sroa.7.048
  %_0.i.i.i = load i64, ptr %65, align 8, !noalias !439, !noundef !6
  br label %_RNvMNtNtNtNtNtNtNtCs5mvLrSLkDP1_12vortex_array9scalar_fn8unstable3row5types7element5tuple13element_tupleINtB2_9ArgColumnxE3getCsaHo96IcALPt_24row_fn_performance_probe.exit.i

_RNvMNtNtNtNtNtNtNtCs5mvLrSLkDP1_12vortex_array9scalar_fn8unstable3row5types7element5tuple13element_tupleINtB2_9ArgColumnxE3getCsaHo96IcALPt_24row_fn_performance_probe.exit.i: ; preds = %_RNvXNtNtNtNtNtNtCs5mvLrSLkDP1_12vortex_array9scalar_fn8unstable3row5types7element9primitivexNtNtB4_5input12InputElement3getCsaHo96IcALPt_24row_fn_performance_probe.exit.i.i, %bb21
  %_0.sroa.0.0.i.i = phi i64 [ %_0.i.i.i, %_RNvXNtNtNtNtNtNtCs5mvLrSLkDP1_12vortex_array9scalar_fn8unstable3row5types7element9primitivexNtNtB4_5input12InputElement3getCsaHo96IcALPt_24row_fn_performance_probe.exit.i.i ], [ %columns.val2.i27, %bb21 ]
  %_6.val.i29 = load ptr, ptr %_6.i1478, align 8, !alias.scope !439, !noundef !6
  %_6.val1.i30 = load i64, ptr %61, align 8, !alias.scope !439
  %66 = icmp eq ptr %_6.val.i29, null
  br i1 %66, label %bb24, label %bb3.i3.i

bb3.i3.i:                                         ; preds = %_RNvMNtNtNtNtNtNtNtCs5mvLrSLkDP1_12vortex_array9scalar_fn8unstable3row5types7element5tuple13element_tupleINtB2_9ArgColumnxE3getCsaHo96IcALPt_24row_fn_performance_probe.exit.i
  %_3.i.i4.i = icmp ult i64 %iter.sroa.7.048, %_6.val1.i30
  br i1 %_3.i.i4.i, label %_RNvXNtNtNtNtNtNtCs5mvLrSLkDP1_12vortex_array9scalar_fn8unstable3row5types7element9primitivexNtNtB4_5input12InputElement3getCsaHo96IcALPt_24row_fn_performance_probe.exit.i6.i, label %panic.i.i5.i.invoke

panic.i.i5.i.invoke:                              ; preds = %bb3.i3.i, %bb3.i.i
  %67 = phi i64 [ %columns.val2.i27, %bb3.i.i ], [ %_6.val1.i30, %bb3.i3.i ]
; invoke core::panicking::panic_bounds_check
  invoke void @_RNvNtCslWxY2MhVcag_4core9panicking18panic_bounds_check(i64 noundef %iter.sroa.7.048, i64 noundef %67, ptr noalias nofree noundef readonly align 8 captures(address, read_provenance) dereferenceable(24) @alloc_ef228a7a1610e5e546147c56beb2165c.llvm.3148866541286181846) #20
          to label %panic.i.i5.i.cont unwind label %bb52.thread33.loopexit

panic.i.i5.i.cont:                                ; preds = %panic.i.i5.i.invoke
  unreachable

_RNvXNtNtNtNtNtNtCs5mvLrSLkDP1_12vortex_array9scalar_fn8unstable3row5types7element9primitivexNtNtB4_5input12InputElement3getCsaHo96IcALPt_24row_fn_performance_probe.exit.i6.i: ; preds = %bb3.i3.i
  %68 = getelementptr inbounds nuw i64, ptr %_6.val.i29, i64 %iter.sroa.7.048
  %_0.i.i7.i = load i64, ptr %68, align 8, !noalias !439, !noundef !6
  br label %bb24

bb27:                                             ; preds = %bb1.i
  call void @llvm.lifetime.end.p0(ptr nonnull %_6.i), !noalias !436
  %.pr = load i64, ptr %_69, align 8
  %.not13 = icmp eq i64 %.pr, -1
  br i1 %.not13, label %bb29, label %bb28

bb28:                                             ; preds = %bb27
  %69 = getelementptr inbounds nuw i8, ptr %_0, i64 8
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(80) %69, ptr noundef nonnull align 8 dereferenceable(80) %_69, i64 80, i1 false)
  store i64 0, ptr %_0, align 8
  call void @llvm.lifetime.end.p0(ptr nonnull %_69)
  br i1 %7, label %bb31, label %bb2.i.i

bb2.i.i:                                          ; preds = %bb28
  call void @mi_free(ptr noundef nonnull %10) #18, !noalias !442
  br label %bb31

bb29.sink.split:                                  ; preds = %bb18, %bb9
  call void @llvm.lifetime.start.p0(ptr nonnull %_69)
  br label %bb29

bb29:                                             ; preds = %bb29.sink.split, %bb27, %bb26, %bb13
  %values.sroa.12.1 = phi i64 [ %56, %bb13 ], [ %values.sroa.12.0, %bb27 ], [ %57, %bb26 ], [ 0, %bb29.sink.split ]
  call void @llvm.lifetime.start.p0(ptr nonnull %_75)
  store i64 %5, ptr %_75, align 8
  %values.sroa.8.0._75.sroa_idx = getelementptr inbounds nuw i8, ptr %_75, i64 8
  store ptr %10, ptr %values.sroa.8.0._75.sroa_idx, align 8
  %values.sroa.12.0._75.sroa_idx = getelementptr inbounds nuw i8, ptr %_75, i64 16
  store i64 %values.sroa.12.1, ptr %values.sroa.12.0._75.sroa_idx, align 8
; invoke <bool as vortex_array::scalar_fn::unstable::row::types::element::output::OutputElement>::build
  %70 = invoke { ptr, ptr } @_RNvXs_NtNtNtNtNtNtCs5mvLrSLkDP1_12vortex_array9scalar_fn8unstable3row5types7element4boolbNtNtB6_6output13OutputElement5build(ptr noalias nofree noundef nonnull align 8 captures(address) dereferenceable(24) %_75)
          to label %bb30 unwind label %bb52

bb30:                                             ; preds = %bb29
  %_74.0 = extractvalue { ptr, ptr } %70, 0
  %_74.1 = extractvalue { ptr, ptr } %70, 1
  call void @llvm.lifetime.end.p0(ptr nonnull %_75)
  %_73.sroa.4.0..sroa_idx = getelementptr inbounds nuw i8, ptr %_0, i64 16
  store ptr %_74.0, ptr %_73.sroa.4.0..sroa_idx, align 8
  %_73.sroa.5.0..sroa_idx = getelementptr inbounds nuw i8, ptr %_0, i64 24
  store ptr %_74.1, ptr %_73.sroa.5.0..sroa_idx, align 8
  store <2 x i64> <i64 0, i64 -1>, ptr %_0, align 8
  call void @llvm.lifetime.end.p0(ptr nonnull %_69)
  br label %bb31

bb31:                                             ; preds = %bb2.i.i, %bb28, %bb30
  call void @llvm.lifetime.end.p0(ptr nonnull %row_count)
  call void @llvm.experimental.noalias.scope.decl(metadata !445)
  call void @llvm.experimental.noalias.scope.decl(metadata !448)
  call void @llvm.experimental.noalias.scope.decl(metadata !451)
  %71 = load ptr, ptr %columns, align 8, !alias.scope !454, !noundef !6
  %.not.i.i.i = icmp eq ptr %71, null
  br i1 %.not.i.i.i, label %bb4.i25, label %bb2.i.i.i

bb2.i.i.i:                                        ; preds = %bb31
  call void @llvm.experimental.noalias.scope.decl(metadata !455)
  %72 = getelementptr inbounds nuw i8, ptr %columns, i64 16
  call void @llvm.experimental.noalias.scope.decl(metadata !458)
  %73 = load ptr, ptr %72, align 8, !alias.scope !461, !noundef !6
  %74 = icmp eq ptr %73, null
  br i1 %74, label %bb4.i25, label %bb2.i.i.i.i.i

bb2.i.i.i.i.i:                                    ; preds = %bb2.i.i.i
  %_2.i.i.i.i.i.i.i = atomicrmw sub ptr %73, i64 1 release, align 8, !noalias !462
  %75 = icmp eq i64 %_2.i.i.i.i.i.i.i, 1
  br i1 %75, label %bb2.i.i.i.i.i.i.i, label %bb4.i25

bb2.i.i.i.i.i.i.i:                                ; preds = %bb2.i.i.i.i.i
  fence acquire
; invoke <alloc::sync::Arc<vortex_buffer::allocation::BufferBacking>>::drop_slow
  invoke void @_RNvMsn_NtCs6KVRSXc8uZF_5alloc4syncINtB5_3ArcNtNtCsEOnbuuTlDO_13vortex_buffer10allocation13BufferBackingE9drop_slowBK_(ptr noalias nofree noundef nonnull align 8 dereferenceable(8) %72) #17
          to label %bb4.i25 unwind label %cleanup.i26

cleanup.i26:                                      ; preds = %bb2.i.i.i.i.i.i.i
  %76 = landingpad { ptr, i32 }
          cleanup
  %77 = getelementptr inbounds nuw i8, ptr %columns, i64 32
; invoke core::ptr::drop_glue::<vortex_array::scalar_fn::unstable::row::types::element::tuple::element_tuple::ArgColumn<row_fn_performance_probe::SelectedI64<false>>>
  invoke fastcc void @_RINvNtCslWxY2MhVcag_4core3ptr9drop_glueINtNtNtNtNtNtNtNtCs5mvLrSLkDP1_12vortex_array9scalar_fn8unstable3row5types7element5tuple13element_tuple9ArgColumnINtCsaHo96IcALPt_24row_fn_performance_probe11SelectedI64Kb0_EEEB2t_(ptr noalias nofree noundef align 8 dereferenceable(32) %77) #19
          to label %common.resume unwind label %terminate.i27

bb4.i25:                                          ; preds = %bb2.i.i.i.i.i.i.i, %bb2.i.i.i.i.i, %bb2.i.i.i, %bb31
  %78 = getelementptr inbounds nuw i8, ptr %columns, i64 32
  call void @llvm.experimental.noalias.scope.decl(metadata !467)
  call void @llvm.experimental.noalias.scope.decl(metadata !470)
  %79 = load ptr, ptr %78, align 8, !alias.scope !473, !noundef !6
  %.not.i.i1.i = icmp eq ptr %79, null
  br i1 %.not.i.i1.i, label %bb40, label %bb2.i.i2.i

bb2.i.i2.i:                                       ; preds = %bb4.i25
  call void @llvm.experimental.noalias.scope.decl(metadata !474)
  %80 = getelementptr inbounds nuw i8, ptr %columns, i64 48
  call void @llvm.experimental.noalias.scope.decl(metadata !477)
  %81 = load ptr, ptr %80, align 8, !alias.scope !480, !noundef !6
  %82 = icmp eq ptr %81, null
  br i1 %82, label %bb40, label %bb2.i.i.i.i3.i

bb2.i.i.i.i3.i:                                   ; preds = %bb2.i.i2.i
  %_2.i.i.i.i.i.i4.i = atomicrmw sub ptr %81, i64 1 release, align 8, !noalias !481
  %83 = icmp eq i64 %_2.i.i.i.i.i.i4.i, 1
  br i1 %83, label %bb2.i.i.i.i.i.i5.i, label %bb40

bb2.i.i.i.i.i.i5.i:                               ; preds = %bb2.i.i.i.i3.i
  fence acquire
; call <alloc::sync::Arc<vortex_buffer::allocation::BufferBacking>>::drop_slow
  call void @_RNvMsn_NtCs6KVRSXc8uZF_5alloc4syncINtB5_3ArcNtNtCsEOnbuuTlDO_13vortex_buffer10allocation13BufferBackingE9drop_slowBK_(ptr noalias nofree noundef nonnull align 8 dereferenceable(8) %80) #17
  br label %bb40

terminate.i27:                                    ; preds = %cleanup.i26
  %84 = landingpad { ptr, i32 }
          filter [0 x ptr] zeroinitializer
; call core::panicking::panic_in_cleanup
  call void @_RNvNtCslWxY2MhVcag_4core9panicking16panic_in_cleanup() #16
  unreachable

bb40:                                             ; preds = %bb2.i.i.i.i.i.i5.i, %bb2.i.i.i.i3.i, %bb2.i.i2.i, %bb4.i25, %bb36, %bb59
  call void @llvm.lifetime.end.p0(ptr nonnull %columns)
  ret void

bb24:                                             ; preds = %_RNvXNtNtNtNtNtNtCs5mvLrSLkDP1_12vortex_array9scalar_fn8unstable3row5types7element9primitivexNtNtB4_5input12InputElement3getCsaHo96IcALPt_24row_fn_performance_probe.exit.i6.i, %_RNvMNtNtNtNtNtNtNtCs5mvLrSLkDP1_12vortex_array9scalar_fn8unstable3row5types7element5tuple13element_tupleINtB2_9ArgColumnxE3getCsaHo96IcALPt_24row_fn_performance_probe.exit.i
  %_0.sroa.0.0.i8.i = phi i64 [ %_0.i.i7.i, %_RNvXNtNtNtNtNtNtCs5mvLrSLkDP1_12vortex_array9scalar_fn8unstable3row5types7element9primitivexNtNtB4_5input12InputElement3getCsaHo96IcALPt_24row_fn_performance_probe.exit.i6.i ], [ %_6.val1.i30, %_RNvMNtNtNtNtNtNtNtCs5mvLrSLkDP1_12vortex_array9scalar_fn8unstable3row5types7element5tuple13element_tupleINtB2_9ArgColumnxE3getCsaHo96IcALPt_24row_fn_performance_probe.exit.i ]
  %_9.0.i = add nuw i64 %iter.sroa.7.048, 1
  %_17.i.i = getelementptr inbounds nuw i8, ptr %iter.sroa.0.049, i64 1
  %85 = icmp eq i64 %_0.sroa.0.0.i.i, 9223372036854775807
  %86 = icmp eq i64 %_0.sroa.0.0.i8.i, 9223372036854775807
  %_6.sroa.0.0.off0.i.i = or i1 %85, %86
  %_5.i.i = icmp slt i64 %_0.sroa.0.0.i.i, %_0.sroa.0.0.i8.i
  %_139 = zext i1 %_5.i.i to i8
  store i8 %_139, ptr %iter.sroa.0.049, align 1
  %87 = or i1 %accumulated_failure.sroa.0.0.off050, %_6.sroa.0.0.off0.i.i
  %_7.i.i = icmp eq ptr %_17.i.i, %63
  br i1 %_7.i.i, label %bb26, label %bb21

unreachable:                                      ; preds = %bb62
  unreachable

bb51:                                             ; preds = %bb52.thread.loopexit.split-lp, %bb52.thread33.loopexit.split-lp, %bb52.thread33.loopexit
  %.pn32 = phi { ptr, i32 } [ %lpad.loopexit.split-lp, %bb52.thread33.loopexit.split-lp ], [ %lpad.loopexit, %bb52.thread33.loopexit ], [ %lpad.loopexit.split-lp44, %bb52.thread.loopexit.split-lp ]
  br i1 %7, label %bb42, label %bb2.i.i.i35

bb2.i.i.i35:                                      ; preds = %bb51
  call void @mi_free(ptr noundef nonnull %10) #18, !noalias !486
  br label %bb42

terminate:                                        ; preds = %bb42
  %88 = landingpad { ptr, i32 }
          filter [0 x ptr] zeroinitializer
; call core::panicking::panic_in_cleanup
  call void @_RNvNtCslWxY2MhVcag_4core9panicking16panic_in_cleanup() #16
  unreachable

bb42:                                             ; preds = %bb2.i.i.i35, %bb51, %bb52, %cleanup3
  %.pn15 = phi { ptr, i32 } [ %6, %cleanup3 ], [ %lpad.thr_comm.split-lp, %bb52 ], [ %.pn32, %bb51 ], [ %.pn32, %bb2.i.i.i35 ]
; invoke core::ptr::drop_glue::<(vortex_array::scalar_fn::unstable::row::types::element::tuple::element_tuple::ArgColumn<row_fn_performance_probe::SelectedI64<false>>, vortex_array::scalar_fn::unstable::row::types::element::tuple::element_tuple::ArgColumn<row_fn_performance_probe::SelectedI64<false>>)>
  invoke fastcc void @_RINvNtCslWxY2MhVcag_4core3ptr9drop_glueTINtNtNtNtNtNtNtNtCs5mvLrSLkDP1_12vortex_array9scalar_fn8unstable3row5types7element5tuple13element_tuple9ArgColumnINtCsaHo96IcALPt_24row_fn_performance_probe11SelectedI64Kb0_EEBC_EEB2u_(ptr noalias nofree noundef align 8 dereferenceable(64) %columns) #19
          to label %common.resume unwind label %terminate
}
