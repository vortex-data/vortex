; SPDX-License-Identifier: Apache-2.0
; SPDX-FileCopyrightText: Copyright the Vortex contributors
define hidden { ptr, ptr } @_RINvYxNtNtNtNtNtNtNtCs5mvLrSLkDP1_12vortex_array9scalar_fn8unstable3row5types7element6output13OutputElement10build_fromRSxNCNvCsaHo96IcALPt_24row_fn_performance_probe6direct0EB20_(ptr noalias nofree noundef nonnull readonly align 8 captures(none) %0, i64 noundef range(i64 0, 1152921504606846976) %1) unnamed_addr #0 personality ptr @rust_eh_personality {
start:
  %_3.i = alloca [24 x i8], align 8
  %_11 = alloca [24 x i8], align 8
  %2 = shl nuw nsw i64 %1, 3
  %3 = icmp eq i64 %1, 0
  br i1 %3, label %bb2, label %bb3.i

bb3.i:                                            ; preds = %start
; call __rustc::__rust_no_alloc_shim_is_unstable_v2
  tail call void @_RNvCs6rREvFdRhLb_7___rustc35___rust_no_alloc_shim_is_unstable_v2() #18, !noalias !643
  %_3.i1.i = tail call noalias noundef ptr @mi_malloc_aligned(i64 noundef range(i64 0, -9223372036854775808) %2, i64 noundef range(i64 1, -9223372036854775807) 8) #18, !noalias !643
  %4 = icmp eq ptr %_3.i1.i, null
  br i1 %4, label %bb13, label %bb2.i

bb13:                                             ; preds = %bb3.i
; call alloc::raw_vec::handle_error
  tail call void @_RNvNtCs6KVRSXc8uZF_5alloc7raw_vec12handle_error(i64 noundef 8, i64 %2) #17
  unreachable

bb2.i:                                            ; preds = %bb3.i
  tail call void @llvm.experimental.noalias.scope.decl(metadata !646)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !649)
  %chunks_count2.i = lshr i64 %1, 6
  %remainder.i = and i64 %1, 63
  %_3524.not.i = icmp eq i64 %chunks_count2.i, 0
  br i1 %_3524.not.i, label %bb15.i, label %bb14.i

bb15.i:                                           ; preds = %bb14.i, %bb2.i
  %5 = icmp eq i64 %remainder.i, 0
  br i1 %5, label %bb2, label %bb6.i

bb14.i:                                           ; preds = %bb2.i, %bb14.i
  %iter.sroa.0.025.i = phi i64 [ %109, %bb14.i ], [ 0, %bb2.i ]
  %_21.i = shl nuw nsw i64 %iter.sroa.0.025.i, 6
  %6 = getelementptr inbounds nuw i64, ptr %0, i64 %_21.i
  %7 = getelementptr inbounds nuw i8, ptr %6, i64 16
  %8 = getelementptr inbounds nuw i8, ptr %6, i64 32
  %9 = getelementptr inbounds nuw i8, ptr %6, i64 48
  %wide.load = load <2 x i64>, ptr %6, align 8, !alias.scope !646, !noalias !649
  %wide.load19 = load <2 x i64>, ptr %7, align 8, !alias.scope !646, !noalias !649
  %wide.load20 = load <2 x i64>, ptr %8, align 8, !alias.scope !646, !noalias !649
  %wide.load21 = load <2 x i64>, ptr %9, align 8, !alias.scope !646, !noalias !649
  %10 = getelementptr inbounds nuw i64, ptr %_3.i1.i, i64 %_21.i
  %11 = add <2 x i64> %wide.load, splat (i64 1)
  %12 = add <2 x i64> %wide.load19, splat (i64 1)
  %13 = add <2 x i64> %wide.load20, splat (i64 1)
  %14 = add <2 x i64> %wide.load21, splat (i64 1)
  %15 = getelementptr inbounds nuw i8, ptr %10, i64 16
  %16 = getelementptr inbounds nuw i8, ptr %10, i64 32
  %17 = getelementptr inbounds nuw i8, ptr %10, i64 48
  store <2 x i64> %11, ptr %10, align 8, !alias.scope !651, !noalias !654
  store <2 x i64> %12, ptr %15, align 8, !alias.scope !651, !noalias !654
  store <2 x i64> %13, ptr %16, align 8, !alias.scope !651, !noalias !654
  store <2 x i64> %14, ptr %17, align 8, !alias.scope !651, !noalias !654
  %18 = or disjoint i64 %_21.i, 8
  %19 = getelementptr inbounds nuw i64, ptr %0, i64 %18
  %20 = getelementptr inbounds nuw i8, ptr %19, i64 16
  %21 = getelementptr inbounds nuw i8, ptr %19, i64 32
  %22 = getelementptr inbounds nuw i8, ptr %19, i64 48
  %wide.load.1 = load <2 x i64>, ptr %19, align 8, !alias.scope !646, !noalias !649
  %wide.load19.1 = load <2 x i64>, ptr %20, align 8, !alias.scope !646, !noalias !649
  %wide.load20.1 = load <2 x i64>, ptr %21, align 8, !alias.scope !646, !noalias !649
  %wide.load21.1 = load <2 x i64>, ptr %22, align 8, !alias.scope !646, !noalias !649
  %23 = getelementptr inbounds nuw i64, ptr %_3.i1.i, i64 %18
  %24 = add <2 x i64> %wide.load.1, splat (i64 1)
  %25 = add <2 x i64> %wide.load19.1, splat (i64 1)
  %26 = add <2 x i64> %wide.load20.1, splat (i64 1)
  %27 = add <2 x i64> %wide.load21.1, splat (i64 1)
  %28 = getelementptr inbounds nuw i8, ptr %23, i64 16
  %29 = getelementptr inbounds nuw i8, ptr %23, i64 32
  %30 = getelementptr inbounds nuw i8, ptr %23, i64 48
  store <2 x i64> %24, ptr %23, align 8, !alias.scope !651, !noalias !654
  store <2 x i64> %25, ptr %28, align 8, !alias.scope !651, !noalias !654
  store <2 x i64> %26, ptr %29, align 8, !alias.scope !651, !noalias !654
  store <2 x i64> %27, ptr %30, align 8, !alias.scope !651, !noalias !654
  %31 = or disjoint i64 %_21.i, 16
  %32 = getelementptr inbounds nuw i64, ptr %0, i64 %31
  %33 = getelementptr inbounds nuw i8, ptr %32, i64 16
  %34 = getelementptr inbounds nuw i8, ptr %32, i64 32
  %35 = getelementptr inbounds nuw i8, ptr %32, i64 48
  %wide.load.2 = load <2 x i64>, ptr %32, align 8, !alias.scope !646, !noalias !649
  %wide.load19.2 = load <2 x i64>, ptr %33, align 8, !alias.scope !646, !noalias !649
  %wide.load20.2 = load <2 x i64>, ptr %34, align 8, !alias.scope !646, !noalias !649
  %wide.load21.2 = load <2 x i64>, ptr %35, align 8, !alias.scope !646, !noalias !649
  %36 = getelementptr inbounds nuw i64, ptr %_3.i1.i, i64 %31
  %37 = add <2 x i64> %wide.load.2, splat (i64 1)
  %38 = add <2 x i64> %wide.load19.2, splat (i64 1)
  %39 = add <2 x i64> %wide.load20.2, splat (i64 1)
  %40 = add <2 x i64> %wide.load21.2, splat (i64 1)
  %41 = getelementptr inbounds nuw i8, ptr %36, i64 16
  %42 = getelementptr inbounds nuw i8, ptr %36, i64 32
  %43 = getelementptr inbounds nuw i8, ptr %36, i64 48
  store <2 x i64> %37, ptr %36, align 8, !alias.scope !651, !noalias !654
  store <2 x i64> %38, ptr %41, align 8, !alias.scope !651, !noalias !654
  store <2 x i64> %39, ptr %42, align 8, !alias.scope !651, !noalias !654
  store <2 x i64> %40, ptr %43, align 8, !alias.scope !651, !noalias !654
  %44 = or disjoint i64 %_21.i, 24
  %45 = getelementptr inbounds nuw i64, ptr %0, i64 %44
  %46 = getelementptr inbounds nuw i8, ptr %45, i64 16
  %47 = getelementptr inbounds nuw i8, ptr %45, i64 32
  %48 = getelementptr inbounds nuw i8, ptr %45, i64 48
  %wide.load.3 = load <2 x i64>, ptr %45, align 8, !alias.scope !646, !noalias !649
  %wide.load19.3 = load <2 x i64>, ptr %46, align 8, !alias.scope !646, !noalias !649
  %wide.load20.3 = load <2 x i64>, ptr %47, align 8, !alias.scope !646, !noalias !649
  %wide.load21.3 = load <2 x i64>, ptr %48, align 8, !alias.scope !646, !noalias !649
  %49 = getelementptr inbounds nuw i64, ptr %_3.i1.i, i64 %44
  %50 = add <2 x i64> %wide.load.3, splat (i64 1)
  %51 = add <2 x i64> %wide.load19.3, splat (i64 1)
  %52 = add <2 x i64> %wide.load20.3, splat (i64 1)
  %53 = add <2 x i64> %wide.load21.3, splat (i64 1)
  %54 = getelementptr inbounds nuw i8, ptr %49, i64 16
  %55 = getelementptr inbounds nuw i8, ptr %49, i64 32
  %56 = getelementptr inbounds nuw i8, ptr %49, i64 48
  store <2 x i64> %50, ptr %49, align 8, !alias.scope !651, !noalias !654
  store <2 x i64> %51, ptr %54, align 8, !alias.scope !651, !noalias !654
  store <2 x i64> %52, ptr %55, align 8, !alias.scope !651, !noalias !654
  store <2 x i64> %53, ptr %56, align 8, !alias.scope !651, !noalias !654
  %57 = or disjoint i64 %_21.i, 32
  %58 = getelementptr inbounds nuw i64, ptr %0, i64 %57
  %59 = getelementptr inbounds nuw i8, ptr %58, i64 16
  %60 = getelementptr inbounds nuw i8, ptr %58, i64 32
  %61 = getelementptr inbounds nuw i8, ptr %58, i64 48
  %wide.load.4 = load <2 x i64>, ptr %58, align 8, !alias.scope !646, !noalias !649
  %wide.load19.4 = load <2 x i64>, ptr %59, align 8, !alias.scope !646, !noalias !649
  %wide.load20.4 = load <2 x i64>, ptr %60, align 8, !alias.scope !646, !noalias !649
  %wide.load21.4 = load <2 x i64>, ptr %61, align 8, !alias.scope !646, !noalias !649
  %62 = getelementptr inbounds nuw i64, ptr %_3.i1.i, i64 %57
  %63 = add <2 x i64> %wide.load.4, splat (i64 1)
  %64 = add <2 x i64> %wide.load19.4, splat (i64 1)
  %65 = add <2 x i64> %wide.load20.4, splat (i64 1)
  %66 = add <2 x i64> %wide.load21.4, splat (i64 1)
  %67 = getelementptr inbounds nuw i8, ptr %62, i64 16
  %68 = getelementptr inbounds nuw i8, ptr %62, i64 32
  %69 = getelementptr inbounds nuw i8, ptr %62, i64 48
  store <2 x i64> %63, ptr %62, align 8, !alias.scope !651, !noalias !654
  store <2 x i64> %64, ptr %67, align 8, !alias.scope !651, !noalias !654
  store <2 x i64> %65, ptr %68, align 8, !alias.scope !651, !noalias !654
  store <2 x i64> %66, ptr %69, align 8, !alias.scope !651, !noalias !654
  %70 = or disjoint i64 %_21.i, 40
  %71 = getelementptr inbounds nuw i64, ptr %0, i64 %70
  %72 = getelementptr inbounds nuw i8, ptr %71, i64 16
  %73 = getelementptr inbounds nuw i8, ptr %71, i64 32
  %74 = getelementptr inbounds nuw i8, ptr %71, i64 48
  %wide.load.5 = load <2 x i64>, ptr %71, align 8, !alias.scope !646, !noalias !649
  %wide.load19.5 = load <2 x i64>, ptr %72, align 8, !alias.scope !646, !noalias !649
  %wide.load20.5 = load <2 x i64>, ptr %73, align 8, !alias.scope !646, !noalias !649
  %wide.load21.5 = load <2 x i64>, ptr %74, align 8, !alias.scope !646, !noalias !649
  %75 = getelementptr inbounds nuw i64, ptr %_3.i1.i, i64 %70
  %76 = add <2 x i64> %wide.load.5, splat (i64 1)
  %77 = add <2 x i64> %wide.load19.5, splat (i64 1)
  %78 = add <2 x i64> %wide.load20.5, splat (i64 1)
  %79 = add <2 x i64> %wide.load21.5, splat (i64 1)
  %80 = getelementptr inbounds nuw i8, ptr %75, i64 16
  %81 = getelementptr inbounds nuw i8, ptr %75, i64 32
  %82 = getelementptr inbounds nuw i8, ptr %75, i64 48
  store <2 x i64> %76, ptr %75, align 8, !alias.scope !651, !noalias !654
  store <2 x i64> %77, ptr %80, align 8, !alias.scope !651, !noalias !654
  store <2 x i64> %78, ptr %81, align 8, !alias.scope !651, !noalias !654
  store <2 x i64> %79, ptr %82, align 8, !alias.scope !651, !noalias !654
  %83 = or disjoint i64 %_21.i, 48
  %84 = getelementptr inbounds nuw i64, ptr %0, i64 %83
  %85 = getelementptr inbounds nuw i8, ptr %84, i64 16
  %86 = getelementptr inbounds nuw i8, ptr %84, i64 32
  %87 = getelementptr inbounds nuw i8, ptr %84, i64 48
  %wide.load.6 = load <2 x i64>, ptr %84, align 8, !alias.scope !646, !noalias !649
  %wide.load19.6 = load <2 x i64>, ptr %85, align 8, !alias.scope !646, !noalias !649
  %wide.load20.6 = load <2 x i64>, ptr %86, align 8, !alias.scope !646, !noalias !649
  %wide.load21.6 = load <2 x i64>, ptr %87, align 8, !alias.scope !646, !noalias !649
  %88 = getelementptr inbounds nuw i64, ptr %_3.i1.i, i64 %83
  %89 = add <2 x i64> %wide.load.6, splat (i64 1)
  %90 = add <2 x i64> %wide.load19.6, splat (i64 1)
  %91 = add <2 x i64> %wide.load20.6, splat (i64 1)
  %92 = add <2 x i64> %wide.load21.6, splat (i64 1)
  %93 = getelementptr inbounds nuw i8, ptr %88, i64 16
  %94 = getelementptr inbounds nuw i8, ptr %88, i64 32
  %95 = getelementptr inbounds nuw i8, ptr %88, i64 48
  store <2 x i64> %89, ptr %88, align 8, !alias.scope !651, !noalias !654
  store <2 x i64> %90, ptr %93, align 8, !alias.scope !651, !noalias !654
  store <2 x i64> %91, ptr %94, align 8, !alias.scope !651, !noalias !654
  store <2 x i64> %92, ptr %95, align 8, !alias.scope !651, !noalias !654
  %96 = or disjoint i64 %_21.i, 56
  %97 = getelementptr inbounds nuw i64, ptr %0, i64 %96
  %98 = getelementptr inbounds nuw i8, ptr %97, i64 16
  %99 = getelementptr inbounds nuw i8, ptr %97, i64 32
  %100 = getelementptr inbounds nuw i8, ptr %97, i64 48
  %wide.load.7 = load <2 x i64>, ptr %97, align 8, !alias.scope !646, !noalias !649
  %wide.load19.7 = load <2 x i64>, ptr %98, align 8, !alias.scope !646, !noalias !649
  %wide.load20.7 = load <2 x i64>, ptr %99, align 8, !alias.scope !646, !noalias !649
  %wide.load21.7 = load <2 x i64>, ptr %100, align 8, !alias.scope !646, !noalias !649
  %101 = getelementptr inbounds nuw i64, ptr %_3.i1.i, i64 %96
  %102 = add <2 x i64> %wide.load.7, splat (i64 1)
  %103 = add <2 x i64> %wide.load19.7, splat (i64 1)
  %104 = add <2 x i64> %wide.load20.7, splat (i64 1)
  %105 = add <2 x i64> %wide.load21.7, splat (i64 1)
  %106 = getelementptr inbounds nuw i8, ptr %101, i64 16
  %107 = getelementptr inbounds nuw i8, ptr %101, i64 32
  %108 = getelementptr inbounds nuw i8, ptr %101, i64 48
  store <2 x i64> %102, ptr %101, align 8, !alias.scope !651, !noalias !654
  store <2 x i64> %103, ptr %106, align 8, !alias.scope !651, !noalias !654
  store <2 x i64> %104, ptr %107, align 8, !alias.scope !651, !noalias !654
  store <2 x i64> %105, ptr %108, align 8, !alias.scope !651, !noalias !654
  %109 = add nuw nsw i64 %iter.sroa.0.025.i, 1
  %exitcond27.not.i = icmp eq i64 %109, %chunks_count2.i
  br i1 %exitcond27.not.i, label %bb15.i, label %bb14.i

bb6.i:                                            ; preds = %bb15.i
  %_25.i = and i64 %1, 1152921504606846912
  %min.iters.check = icmp samesign ult i64 %remainder.i, 8
  br i1 %min.iters.check, label %bb4.i.i.preheader, label %vector.ph22

vector.ph22:                                      ; preds = %bb6.i
  %n.vec = and i64 %1, 56
  br label %vector.body25

vector.body25:                                    ; preds = %vector.body25, %vector.ph22
  %index26 = phi i64 [ 0, %vector.ph22 ], [ %index.next31, %vector.body25 ]
  %110 = add nuw nsw i64 %index26, %_25.i
  %111 = getelementptr inbounds nuw i64, ptr %0, i64 %110
  %112 = getelementptr inbounds nuw i8, ptr %111, i64 16
  %113 = getelementptr inbounds nuw i8, ptr %111, i64 32
  %114 = getelementptr inbounds nuw i8, ptr %111, i64 48
  %wide.load27 = load <2 x i64>, ptr %111, align 8, !alias.scope !646, !noalias !649
  %wide.load28 = load <2 x i64>, ptr %112, align 8, !alias.scope !646, !noalias !649
  %wide.load29 = load <2 x i64>, ptr %113, align 8, !alias.scope !646, !noalias !649
  %wide.load30 = load <2 x i64>, ptr %114, align 8, !alias.scope !646, !noalias !649
  %115 = getelementptr inbounds nuw i64, ptr %_3.i1.i, i64 %110
  %116 = add <2 x i64> %wide.load27, splat (i64 1)
  %117 = add <2 x i64> %wide.load28, splat (i64 1)
  %118 = add <2 x i64> %wide.load29, splat (i64 1)
  %119 = add <2 x i64> %wide.load30, splat (i64 1)
  %120 = getelementptr inbounds nuw i8, ptr %115, i64 16
  %121 = getelementptr inbounds nuw i8, ptr %115, i64 32
  %122 = getelementptr inbounds nuw i8, ptr %115, i64 48
  store <2 x i64> %116, ptr %115, align 8, !alias.scope !657, !noalias !660
  store <2 x i64> %117, ptr %120, align 8, !alias.scope !657, !noalias !660
  store <2 x i64> %118, ptr %121, align 8, !alias.scope !657, !noalias !660
  store <2 x i64> %119, ptr %122, align 8, !alias.scope !657, !noalias !660
  %index.next31 = add nuw i64 %index26, 8
  %123 = icmp eq i64 %index.next31, %n.vec
  br i1 %123, label %middle.block32, label %vector.body25, !llvm.loop !663

middle.block32:                                   ; preds = %vector.body25
  %cmp.n = icmp eq i64 %remainder.i, %n.vec
  br i1 %cmp.n, label %bb2, label %bb4.i.i.preheader

bb4.i.i.preheader:                                ; preds = %bb6.i, %middle.block32
  %iter.sroa.0.0.i26.i.ph = phi i64 [ 0, %bb6.i ], [ %n.vec, %middle.block32 ]
  br label %bb4.i.i

bb4.i.i:                                          ; preds = %bb4.i.i.preheader, %bb4.i.i
  %iter.sroa.0.0.i26.i = phi i64 [ %124, %bb4.i.i ], [ %iter.sroa.0.0.i26.i.ph, %bb4.i.i.preheader ]
  %124 = add nuw nsw i64 %iter.sroa.0.0.i26.i, 1
  %idx.i.i = add nuw nsw i64 %iter.sroa.0.0.i26.i, %_25.i
  %_5.i17.i = icmp samesign ult i64 %idx.i.i, %1
  tail call void @llvm.assume(i1 %_5.i17.i)
  %_4.i18.i = getelementptr inbounds nuw i64, ptr %0, i64 %idx.i.i
  %_0.i19.i = load i64, ptr %_4.i18.i, align 8, !alias.scope !646, !noalias !649, !noundef !7
  %_16.i.i = getelementptr inbounds nuw i64, ptr %_3.i1.i, i64 %idx.i.i
  %_0.i20.i = add i64 %_0.i19.i, 1
  store i64 %_0.i20.i, ptr %_16.i.i, align 8, !alias.scope !657, !noalias !660
  %exitcond28.not.i = icmp eq i64 %124, %remainder.i
  br i1 %exitcond28.not.i, label %bb2, label %bb4.i.i, !llvm.loop !664

bb2:                                              ; preds = %bb4.i.i, %middle.block32, %start, %bb15.i
  %_18.sroa.4.01418 = phi i64 [ 0, %start ], [ %1, %bb15.i ], [ %1, %middle.block32 ], [ %1, %bb4.i.i ]
  %125 = phi ptr [ inttoptr (i64 8 to ptr), %start ], [ %_3.i1.i, %bb15.i ], [ %_3.i1.i, %middle.block32 ], [ %_3.i1.i, %bb4.i.i ]
  call void @llvm.lifetime.start.p0(ptr nonnull %_11)
  store i64 %_18.sroa.4.01418, ptr %_11, align 8
  %values.sroa.4.0._11.sroa_idx = getelementptr inbounds nuw i8, ptr %_11, i64 8
  store ptr %125, ptr %values.sroa.4.0._11.sroa_idx, align 8
  %values.sroa.5.0._11.sroa_idx = getelementptr inbounds nuw i8, ptr %_11, i64 16
  store i64 %1, ptr %values.sroa.5.0._11.sroa_idx, align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %_3.i), !noalias !665
  store i64 0, ptr %_3.i, align 8, !noalias !665
; call <vortex_array::array::typed::Array<vortex_array::arrays::primitive::vtable::Primitive>>::new::<i64, alloc::vec::Vec<i64>>
  %126 = call { ptr, ptr } @_RINvMs1_NtNtNtCs5mvLrSLkDP1_12vortex_array6arrays9primitive5arrayINtNtNtBc_5array5typed5ArrayNtNtB8_6vtable9PrimitiveE3newxINtNtCs6KVRSXc8uZF_5alloc3vec3VecxEECsaHo96IcALPt_24row_fn_performance_probe(ptr noalias nofree noundef nonnull align 8 captures(address) dereferenceable(24) %_11, ptr noalias nofree noundef nonnull align 8 captures(address) dereferenceable(24) %_3.i)
  call void @llvm.lifetime.end.p0(ptr nonnull %_3.i), !noalias !665
  call void @llvm.lifetime.end.p0(ptr nonnull %_11)
  ret { ptr, ptr } %126
}
