define hidden void @_RINvNtNtNtNtNtCsc1BqfENIlVH_12vortex_array9scalar_fn8unstable3row7execute11packed_bool26execute_bool_dense_attemptTxxEubKb0_NCINvYINtNtNtB6_7visitor5retry21ExecuteDenseWithRetryINtCs4CRfLIao8WA_17row_fn_bool_probe9PredicateKB1V_EENtNtB29_11row_visitor10RowVisitor19visit_deferred_boolB1O_bKB1V_NCINvXB2S_B2P_NtNtB6_6row_fn5RowFn8dispatchB24_E0NCB4K_s_0E0NCB20_s_0B5v_EB2S_(ptr dead_on_unwind noalias nofree noundef writable writeonly sret([88 x i8]) align 8 captures(none) dereferenceable(88) %_0, ptr noundef nonnull %args.0, ptr noalias nofree noundef readonly align 8 captures(address, read_provenance) dereferenceable(48) %args.1, ptr noalias nofree noundef align 8 dereferenceable(40) %ctx) unnamed_addr #0 personality ptr @rust_eh_personality {
start:
  %args.i = alloca [48 x i8], align 8
  %_10.i50 = alloca [8 x i8], align 8
  %offset.i51 = alloca [8 x i8], align 8
  %len.i = alloca [8 x i8], align 8
  %_13.i.i44 = alloca [56 x i8], align 16
  %allocation.i.i = alloca [40 x i8], align 8
  %_6.i = alloca [16 x i8], align 8
  %buffer.i.i.sroa.0.sroa.0 = alloca [16 x i8], align 8
  %buffer.i.i.sroa.0.sroa.5 = alloca [16 x i8], align 8
  %_7.i = alloca [32 x i8], align 8
  %_45 = alloca [24 x i8], align 8
  %_44 = alloca [48 x i8], align 8
  %_36 = alloca [80 x i8], align 8
  %values = alloca [48 x i8], align 8
  %args = alloca [16 x i8], align 8
  %_20 = alloca [16 x i8], align 8
  %_17 = alloca [80 x i8], align 8
  %row_count = alloca [8 x i8], align 8
  %_8 = alloca [80 x i8], align 8
  %_7.sroa.6 = alloca [64 x i8], align 8
  %columns = alloca [64 x i8], align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %columns)
  call void @llvm.lifetime.start.p0(ptr nonnull %_7.sroa.6)
  call void @llvm.lifetime.start.p0(ptr nonnull %_8)
; call <(i64, i64) as vortex_array::scalar_fn::unstable::row::types::element::tuple::element_tuple::ElementTuple>::decode
  call void @_RNvXs4_NtNtNtNtNtNtNtCsc1BqfENIlVH_12vortex_array9scalar_fn8unstable3row5types7element5tuple13element_tupleTxxENtB5_12ElementTuple6decodeCs4CRfLIao8WA_17row_fn_bool_probe(ptr noalias nofree noundef nonnull sret([80 x i8]) align 8 captures(none) dereferenceable(80) %_8, ptr noundef nonnull %args.0, ptr noalias nofree noundef nonnull readonly align 8 captures(address, read_provenance) dereferenceable(48) %args.1, ptr noalias nofree noundef nonnull align 8 dereferenceable(40) %ctx)
  %0 = load i64, ptr %_8, align 8, !range !66, !noundef !8
  %.not = icmp eq i64 %0, -1
  %1 = getelementptr inbounds nuw i8, ptr %_8, i64 8
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(64) %_7.sroa.6, ptr noundef nonnull align 8 dereferenceable(64) %1, i64 64, i1 false)
  br i1 %.not, label %bb44, label %bb43

common.resume:                                    ; preds = %bb29, %cleanup.i
  %common.resume.op = phi { ptr, i32 } [ %.pn, %bb29 ], [ %250, %cleanup.i ]
  resume { ptr, i32 } %common.resume.op

bb43:                                             ; preds = %start
  %_54.sroa.5.0._8.sroa_idx = getelementptr inbounds nuw i8, ptr %_8, i64 72
  %_54.sroa.5.0.copyload = load i64, ptr %_54.sroa.5.0._8.sroa_idx, align 8
  call void @llvm.lifetime.end.p0(ptr nonnull %_8)
  %_57.sroa.4.0..sroa_idx = getelementptr inbounds nuw i8, ptr %_0, i64 16
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(64) %_57.sroa.4.0..sroa_idx, ptr noundef nonnull align 8 dereferenceable(64) %_7.sroa.6, i64 64, i1 false)
  %2 = getelementptr inbounds nuw i8, ptr %_0, i64 8
  store i64 %0, ptr %2, align 8
  %_57.sroa.5.0..sroa_idx = getelementptr inbounds nuw i8, ptr %_0, i64 80
  store i64 %_54.sroa.5.0.copyload, ptr %_57.sroa.5.0..sroa_idx, align 8
  store i64 1, ptr %_0, align 8
  call void @llvm.lifetime.end.p0(ptr nonnull %_7.sroa.6)
  br label %bb26

bb44:                                             ; preds = %start
  call void @llvm.lifetime.end.p0(ptr nonnull %_8)
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(64) %columns, ptr noundef nonnull align 8 dereferenceable(64) %_7.sroa.6, i64 64, i1 false)
  call void @llvm.lifetime.end.p0(ptr nonnull %_7.sroa.6)
  call void @llvm.lifetime.start.p0(ptr nonnull %row_count)
  %3 = getelementptr inbounds nuw i8, ptr %args.1, i64 40
  %4 = load ptr, ptr %3, align 8, !invariant.load !8, !nonnull !8
  %5 = invoke noundef i64 %4(ptr noundef nonnull %args.0)
          to label %bb5 unwind label %cleanup2

cleanup2:                                         ; preds = %bb32, %bb44
  %6 = landingpad { ptr, i32 }
          cleanup
  br label %bb29

bb5:                                              ; preds = %bb44
  store i64 %5, ptr %row_count, align 8
  %7 = load ptr, ptr %columns, align 8, !alias.scope !445, !noalias !452, !noundef !8
  %8 = icmp eq ptr %7, null
  %constant.i.i.i = getelementptr inbounds nuw i8, ptr %columns, i64 8
  br i1 %8, label %_RNvMNtNtNtNtNtNtNtCsc1BqfENIlVH_12vortex_array9scalar_fn8unstable3row5types7element5tuple7indexedINtB2_15ArgColumnSourcexE7try_newCs4CRfLIao8WA_17row_fn_bool_probe.exit.thread.i.i, label %bb3.i.i.i

_RNvMNtNtNtNtNtNtNtCsc1BqfENIlVH_12vortex_array9scalar_fn8unstable3row5types7element5tuple7indexedINtB2_15ArgColumnSourcexE7try_newCs4CRfLIao8WA_17row_fn_bool_probe.exit.thread.i.i: ; preds = %bb5
  %9 = ptrtoint ptr %constant.i.i.i to i64
  %10 = inttoptr i64 %5 to ptr
  br label %bb10.i.i

bb3.i.i.i:                                        ; preds = %bb5
  %column.val1.i.i.i = load i64, ptr %constant.i.i.i, align 8, !alias.scope !445, !noalias !452, !noundef !8
  %_6.i.i.i = icmp eq i64 %column.val1.i.i.i, %5
  br i1 %_6.i.i.i, label %bb10.i.i, label %bb32

bb10.i.i:                                         ; preds = %bb3.i.i.i, %_RNvMNtNtNtNtNtNtNtCsc1BqfENIlVH_12vortex_array9scalar_fn8unstable3row5types7element5tuple7indexedINtB2_15ArgColumnSourcexE7try_newCs4CRfLIao8WA_17row_fn_bool_probe.exit.thread.i.i
  %_7.sroa.7.123.i.i = phi ptr [ %10, %_RNvMNtNtNtNtNtNtNtCsc1BqfENIlVH_12vortex_array9scalar_fn8unstable3row5types7element5tuple7indexedINtB2_15ArgColumnSourcexE7try_newCs4CRfLIao8WA_17row_fn_bool_probe.exit.thread.i.i ], [ %7, %bb3.i.i.i ]
  %_7.sroa.9.122.i.i = phi i64 [ %9, %_RNvMNtNtNtNtNtNtNtCsc1BqfENIlVH_12vortex_array9scalar_fn8unstable3row5types7element5tuple7indexedINtB2_15ArgColumnSourcexE7try_newCs4CRfLIao8WA_17row_fn_bool_probe.exit.thread.i.i ], [ %5, %bb3.i.i.i ]
  %_13.i.i6 = getelementptr inbounds nuw i8, ptr %columns, i64 32
  %11 = load ptr, ptr %_13.i.i6, align 8, !alias.scope !456, !noalias !459, !noundef !8
  %12 = icmp eq ptr %11, null
  %constant.i14.i.i = getelementptr inbounds nuw i8, ptr %columns, i64 40
  br i1 %12, label %_RNvMNtNtNtNtNtNtNtCsc1BqfENIlVH_12vortex_array9scalar_fn8unstable3row5types7element5tuple7indexedINtB2_15ArgColumnSourcexE7try_newCs4CRfLIao8WA_17row_fn_bool_probe.exit17.thread.i.i, label %bb3.i4.i.i

_RNvMNtNtNtNtNtNtNtCsc1BqfENIlVH_12vortex_array9scalar_fn8unstable3row5types7element5tuple7indexedINtB2_15ArgColumnSourcexE7try_newCs4CRfLIao8WA_17row_fn_bool_probe.exit17.thread.i.i: ; preds = %bb10.i.i
  %13 = ptrtoint ptr %constant.i14.i.i to i64
  %14 = inttoptr i64 %5 to ptr
  br label %bb9

bb3.i4.i.i:                                       ; preds = %bb10.i.i
  %column.val1.i5.i.i = load i64, ptr %constant.i14.i.i, align 8, !alias.scope !456, !noalias !459, !noundef !8
  %_6.i7.i.i = icmp eq i64 %column.val1.i5.i.i, %5
  br i1 %_6.i7.i.i, label %bb9, label %bb32

bb32:                                             ; preds = %bb3.i.i.i, %bb3.i4.i.i
  call void @llvm.lifetime.start.p0(ptr nonnull %_17)
  call void @llvm.lifetime.start.p0(ptr nonnull %_20)
  call void @llvm.lifetime.start.p0(ptr nonnull %args)
  store ptr %row_count, ptr %args, align 8
  %_23.sroa.4.0..sroa_idx = getelementptr inbounds nuw i8, ptr %args, i64 8
  store ptr @_RNvXsi_NtNtNtCslWxY2MhVcag_4core3fmt3num3impjNtB9_7Display3fmt, ptr %_23.sroa.4.0..sroa_idx, align 8
  store ptr @alloc_8cb1b9e1c7b448ff8aa56230a8661784, ptr %_20, align 8
  %15 = getelementptr inbounds nuw i8, ptr %_20, i64 8
  store ptr %args, ptr %15, align 8
; invoke vortex_error::__private::fmt_err
  invoke void @_RNvNtCslnFpLG3TNtn_12vortex_error9___private7fmt_err(ptr noalias nofree noundef nonnull sret([80 x i8]) align 8 captures(address) dereferenceable(80) %_17, ptr noundef nonnull @_RNcNtNtCslnFpLG3TNtn_12vortex_error11VortexError5Other0, ptr noalias nofree noundef nonnull readonly align 8 captures(address, read_provenance) dereferenceable(16) %_20)
          to label %bb6 unwind label %cleanup2

cleanup3:                                         ; preds = %bb9, %bb2.i.i.i.i.i
  %16 = landingpad { ptr, i32 }
          cleanup
  br label %bb29

bb9:                                              ; preds = %bb3.i4.i.i, %_RNvMNtNtNtNtNtNtNtCsc1BqfENIlVH_12vortex_array9scalar_fn8unstable3row5types7element5tuple7indexedINtB2_15ArgColumnSourcexE7try_newCs4CRfLIao8WA_17row_fn_bool_probe.exit17.thread.i.i
  %_26.sroa.14.0 = phi ptr [ %14, %_RNvMNtNtNtNtNtNtNtCsc1BqfENIlVH_12vortex_array9scalar_fn8unstable3row5types7element5tuple7indexedINtB2_15ArgColumnSourcexE7try_newCs4CRfLIao8WA_17row_fn_bool_probe.exit17.thread.i.i ], [ %11, %bb3.i4.i.i ]
  %_26.sroa.16.0 = phi i64 [ %13, %_RNvMNtNtNtNtNtNtNtCsc1BqfENIlVH_12vortex_array9scalar_fn8unstable3row5types7element5tuple7indexedINtB2_15ArgColumnSourcexE7try_newCs4CRfLIao8WA_17row_fn_bool_probe.exit17.thread.i.i ], [ %5, %bb3.i4.i.i ]
  call void @llvm.lifetime.start.p0(ptr nonnull %values)
  %_184.i.i = lshr i64 %5, 6
  %_19.i.i = and i64 %5, 63
  %_20.not.i.i = icmp ne i64 %_19.i.i, 0
  %17 = zext i1 %_20.not.i.i to i64
  %num_words.sroa.0.0.i.i = add nuw nsw i64 %_184.i.i, %17
  call void @llvm.lifetime.start.p0(ptr nonnull %buffer.i.i.sroa.0.sroa.0)
  call void @llvm.lifetime.start.p0(ptr nonnull %buffer.i.i.sroa.0.sroa.5)
  %_39.0.i.i = shl nuw nsw i64 %num_words.sroa.0.0.i.i, 3
  %18 = icmp eq i64 %5, 0
  %_64.0.i.i = add nuw nsw i64 %_39.0.i.i, 256
  %layout.sroa.0.0.i.i = select i1 %18, i64 256, i64 1
  %layout.sroa.3.0.i.i = select i1 %18, i64 0, i64 %_64.0.i.i
  call void @llvm.lifetime.start.p0(ptr nonnull %allocation.i.i), !noalias !461
; invoke <vortex_buffer::allocation::Allocation>::allocate
  invoke void @_RNvMs6_NtCs4EGy4ysETW6_13vortex_buffer10allocationNtB5_10Allocation8allocate(ptr noalias nofree noundef nonnull sret([40 x i8]) align 8 captures(address) dereferenceable(40) %allocation.i.i, i64 noundef %layout.sroa.0.0.i.i, i64 noundef %layout.sroa.3.0.i.i, ptr noundef null, ptr undef)
          to label %.noexc unwind label %cleanup3

.noexc:                                           ; preds = %bb9
  %19 = getelementptr inbounds nuw i8, ptr %allocation.i.i, i64 16
  %_24.i.i = load ptr, ptr %19, align 8, !noalias !461, !nonnull !8, !noundef !8
  %addr.i.i = ptrtoint ptr %_24.i.i to i64
  %_9.i.i = add i64 %addr.i.i, 255
  %aligned_address.i.i = and i64 %_9.i.i, -256
  %byte_offset.i.i = sub i64 %aligned_address.i.i, %addr.i.i
  %_12.i.i = icmp ult i64 %byte_offset.i.i, 256
  call void @llvm.assume(i1 %_12.i.i), !noalias !466
  %_83.i.i = getelementptr inbounds nuw i8, ptr %_24.i.i, i64 %byte_offset.i.i
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %buffer.i.i.sroa.0.sroa.0, ptr noundef nonnull align 8 dereferenceable(16) %allocation.i.i, i64 16, i1 false)
  %buffer.i.i.sroa.0.sroa.5.0.allocation.i.i.sroa_idx = getelementptr inbounds nuw i8, ptr %allocation.i.i, i64 24
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %buffer.i.i.sroa.0.sroa.5, ptr noundef nonnull align 8 dereferenceable(16) %buffer.i.i.sroa.0.sroa.5.0.allocation.i.i.sroa_idx, i64 16, i1 false)
  call void @llvm.lifetime.end.p0(ptr nonnull %allocation.i.i), !noalias !461
  %_52.i.idx.i.i.i.i = shl nuw nsw i64 %_184.i.i, 3
  %_52.i.i.i.i.i = getelementptr inbounds nuw i8, ptr %_83.i.i, i64 %_52.i.idx.i.i.i.i
  %_7.i.i30.i.i.i.i = icmp eq i64 %_184.i.i, 0
  br i1 %_7.i.i30.i.i.i.i, label %bb5.i.i.i.i.i, label %bb4.i.lr.ph.i.i.i.i

bb4.i.lr.ph.i.i.i.i:                              ; preds = %.noexc
  %20 = icmp ne ptr %_7.sroa.7.123.i.i, null
  %21 = inttoptr i64 %_7.sroa.9.122.i.i to ptr
  %22 = icmp ne ptr %_26.sroa.14.0, null
  %23 = inttoptr i64 %_26.sroa.16.0 to ptr
  br i1 %8, label %bb4.i.i.i.i.i.us.preheader, label %bb4.i.lr.ph.i.i.i.i.split

bb4.i.i.i.i.i.us.preheader:                       ; preds = %bb4.i.lr.ph.i.i.i.i
  %24 = getelementptr inbounds nuw i8, ptr %_26.sroa.14.0, i64 128
  %25 = getelementptr inbounds nuw i8, ptr %_26.sroa.14.0, i64 256
  %26 = getelementptr inbounds nuw i8, ptr %_26.sroa.14.0, i64 384
  br label %bb4.i.i.i.i.i.us

bb4.i.i.i.i.i.us:                                 ; preds = %bb4.i.i.i.i.i.us.preheader, %bb9.i.i.i.i.i.split.us.us
  %.lcssa2830.off0.us = phi i1 [ %.us-phi86.us, %bb9.i.i.i.i.i.split.us.us ], [ false, %bb4.i.i.i.i.i.us.preheader ]
  %iter.i.sroa.0.032.i.i.i.i.us = phi ptr [ %_17.i.i.i.i.i.i.us, %bb9.i.i.i.i.i.split.us.us ], [ %_83.i.i, %bb4.i.i.i.i.i.us.preheader ]
  %iter.i.sroa.7.031.i.i.i.i.us = phi i64 [ %_9.0.i.i.i.i.i.us, %bb9.i.i.i.i.i.split.us.us ], [ 0, %bb4.i.i.i.i.i.us.preheader ]
  %_0.sroa.0.0.i.i22.us.us = load i64, ptr %21, align 8, !noalias !467, !noundef !8
  %27 = icmp eq i64 %_0.sroa.0.0.i.i22.us.us, 9223372036854775807
  br i1 %12, label %bb4.i.i.i.i.i.split.us.split.us.us, label %bb4.i.i.i.i.i.split.us.split.us98

bb4.i.i.i.i.i.split.us.split.us98:                ; preds = %bb4.i.i.i.i.i.us
  call void @llvm.assume(i1 %22)
  %broadcast.splatinsert229 = insertelement <16 x i1> poison, i1 %27, i64 0
  %broadcast.splat230 = shufflevector <16 x i1> %broadcast.splatinsert229, <16 x i1> poison, <16 x i32> zeroinitializer
  %broadcast.splatinsert227 = insertelement <16 x i64> poison, i64 %_0.sroa.0.0.i.i22.us.us, i64 0
  %broadcast.splat228 = shufflevector <16 x i64> %broadcast.splatinsert227, <16 x i64> poison, <16 x i32> zeroinitializer
  %28 = insertelement <16 x i1> <i1 poison, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false>, i1 %.lcssa2830.off0.us, i64 0
  %.idx340 = shl i64 %iter.i.sroa.7.031.i.i.i.i.us, 9
  %29 = getelementptr inbounds nuw i8, ptr %_26.sroa.14.0, i64 %.idx340
  %wide.load234 = load <16 x i64>, ptr %29, align 8, !noalias !478
  %30 = icmp eq <16 x i64> %wide.load234, splat (i64 9223372036854775807)
  %31 = icmp slt <16 x i64> %broadcast.splat228, %wide.load234
  %32 = or <16 x i1> %30, %28
  %33 = zext <16 x i1> %31 to <16 x i8>
  %.idx340.1 = shl i64 %iter.i.sroa.7.031.i.i.i.i.us, 9
  %34 = getelementptr inbounds nuw i8, ptr %24, i64 %.idx340.1
  %wide.load234.1 = load <16 x i64>, ptr %34, align 8, !noalias !478
  %35 = icmp eq <16 x i64> %wide.load234.1, splat (i64 9223372036854775807)
  %36 = icmp slt <16 x i64> %broadcast.splat228, %wide.load234.1
  %37 = or <16 x i1> %32, %35
  %38 = zext <16 x i1> %36 to <16 x i8>
  %.idx340.2 = shl i64 %iter.i.sroa.7.031.i.i.i.i.us, 9
  %39 = getelementptr inbounds nuw i8, ptr %25, i64 %.idx340.2
  %wide.load234.2 = load <16 x i64>, ptr %39, align 8, !noalias !478
  %40 = icmp eq <16 x i64> %wide.load234.2, splat (i64 9223372036854775807)
  %41 = icmp slt <16 x i64> %broadcast.splat228, %wide.load234.2
  %42 = or <16 x i1> %37, %40
  %43 = zext <16 x i1> %41 to <16 x i8>
  %.idx340.3 = shl i64 %iter.i.sroa.7.031.i.i.i.i.us, 9
  %44 = getelementptr inbounds nuw i8, ptr %26, i64 %.idx340.3
  %wide.load234.3 = load <16 x i64>, ptr %44, align 8, !noalias !478
  %45 = icmp eq <16 x i64> %wide.load234.3, splat (i64 9223372036854775807)
  %46 = icmp slt <16 x i64> %broadcast.splat228, %wide.load234.3
  %47 = or <16 x i1> %42, %45
  %48 = or <16 x i1> %47, %broadcast.splat230
  %49 = zext <16 x i1> %46 to <16 x i8>
  %50 = bitcast <16 x i1> %48 to i16
  %51 = icmp ne i16 %50, 0
  br label %bb9.i.i.i.i.i.split.us.us

bb9.i.i.i.i.i.split.us.us:                        ; preds = %bb4.i.i.i.i.i.split.us.split.us98, %bb4.i.i.i.i.i.split.us.split.us.us
  %.sroa.07.0.copyload.i.i.i.i.i.i.us = phi <16 x i8> [ %61, %bb4.i.i.i.i.i.split.us.split.us.us ], [ %49, %bb4.i.i.i.i.i.split.us.split.us98 ]
  %.sroa.06.0.copyload.i.i.i.i.i.i.us = phi <16 x i8> [ %61, %bb4.i.i.i.i.i.split.us.split.us.us ], [ %43, %bb4.i.i.i.i.i.split.us.split.us98 ]
  %.sroa.05.0.copyload.i.i.i.i.i.i.us = phi <16 x i8> [ %61, %bb4.i.i.i.i.i.split.us.split.us.us ], [ %38, %bb4.i.i.i.i.i.split.us.split.us98 ]
  %.sroa.04.0.copyload.i.i.i.i.i.i.us = phi <16 x i8> [ %61, %bb4.i.i.i.i.i.split.us.split.us.us ], [ %33, %bb4.i.i.i.i.i.split.us.split.us98 ]
  %.us-phi86.us = phi i1 [ %53, %bb4.i.i.i.i.i.split.us.split.us.us ], [ %51, %bb4.i.i.i.i.i.split.us.split.us98 ]
  %_17.i.i.i.i.i.i.us = getelementptr inbounds nuw i8, ptr %iter.i.sroa.0.032.i.i.i.i.us, i64 8
  %_9.0.i.i.i.i.i.us = add nuw nsw i64 %iter.i.sroa.7.031.i.i.i.i.us, 1
  %m0.i.i.i.i.i.i.us = shl <16 x i8> %.sroa.04.0.copyload.i.i.i.i.i.i.us, <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7>
  %m1.i.i.i.i.i.i.us = shl <16 x i8> %.sroa.05.0.copyload.i.i.i.i.i.i.us, <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7>
  %m2.i.i.i.i.i.i.us = shl <16 x i8> %.sroa.06.0.copyload.i.i.i.i.i.i.us, <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7>
  %m3.i.i.i.i.i.i.us = shl <16 x i8> %.sroa.07.0.copyload.i.i.i.i.i.i.us, <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7>
  %_29.i.i.i.i.i.i.us = shufflevector <16 x i8> %m0.i.i.i.i.i.i.us, <16 x i8> %m1.i.i.i.i.i.i.us, <16 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14, i32 16, i32 18, i32 20, i32 22, i32 24, i32 26, i32 28, i32 30>
  %_30.i.i.i.i.i.i.us = shufflevector <16 x i8> %m0.i.i.i.i.i.i.us, <16 x i8> %m1.i.i.i.i.i.i.us, <16 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15, i32 17, i32 19, i32 21, i32 23, i32 25, i32 27, i32 29, i32 31>
  %sum01.i.i.i.i.i.i.us = add <16 x i8> %_29.i.i.i.i.i.i.us, %_30.i.i.i.i.i.i.us
  %_31.i.i.i.i.i.i.us = shufflevector <16 x i8> %m2.i.i.i.i.i.i.us, <16 x i8> %m3.i.i.i.i.i.i.us, <16 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14, i32 16, i32 18, i32 20, i32 22, i32 24, i32 26, i32 28, i32 30>
  %_32.i.i.i.i.i.i.us = shufflevector <16 x i8> %m2.i.i.i.i.i.i.us, <16 x i8> %m3.i.i.i.i.i.i.us, <16 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15, i32 17, i32 19, i32 21, i32 23, i32 25, i32 27, i32 29, i32 31>
  %sum23.i.i.i.i.i.i.us = add <16 x i8> %_31.i.i.i.i.i.i.us, %_32.i.i.i.i.i.i.us
  %_33.i.i.i.i.i.i.us = shufflevector <16 x i8> %sum01.i.i.i.i.i.i.us, <16 x i8> %sum23.i.i.i.i.i.i.us, <16 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14, i32 16, i32 18, i32 20, i32 22, i32 24, i32 26, i32 28, i32 30>
  %_34.i.i.i.i.i.i.us = shufflevector <16 x i8> %sum01.i.i.i.i.i.i.us, <16 x i8> %sum23.i.i.i.i.i.i.us, <16 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15, i32 17, i32 19, i32 21, i32 23, i32 25, i32 27, i32 29, i32 31>
  %sum.i.i.i.i.i.i.us = add <16 x i8> %_33.i.i.i.i.i.i.us, %_34.i.i.i.i.i.i.us
  %_35.i.i.i.i.i.i.us = shufflevector <16 x i8> %sum.i.i.i.i.i.i.us, <16 x i8> poison, <16 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %_36.i.i.i.i.i.i.us = shufflevector <16 x i8> %sum.i.i.i.i.i.i.us, <16 x i8> poison, <16 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %sum1.i.i.i.i.i.i.us = add <16 x i8> %_35.i.i.i.i.i.i.us, %_36.i.i.i.i.i.i.us
  %_23.i.i.i.i.i.i.us = bitcast <16 x i8> %sum1.i.i.i.i.i.i.us to <2 x i64>
  %_0.i.i.i.i.i.i.us = extractelement <2 x i64> %_23.i.i.i.i.i.i.us, i64 0
  store i64 %_0.i.i.i.i.i.i.us, ptr %iter.i.sroa.0.032.i.i.i.i.us, align 8, !alias.scope !481, !noalias !486
  %_7.i.i.i.i.i.i.us = icmp eq ptr %_17.i.i.i.i.i.i.us, %_52.i.i.i.i.i
  br i1 %_7.i.i.i.i.i.i.us, label %bb5.i.i.i.i.i, label %bb4.i.i.i.i.i.us

bb4.i.i.i.i.i.split.us.split.us.us:               ; preds = %bb4.i.i.i.i.i.us
  %_0.sroa.0.0.i9.i32.us.us.us = load i64, ptr %23, align 8, !noalias !478, !noundef !8
  %_5.i.i.i.i.i.i.i.us.us.us = icmp slt i64 %_0.sroa.0.0.i.i22.us.us, %_0.sroa.0.0.i9.i32.us.us.us
  %52 = icmp eq i64 %_0.sroa.0.0.i9.i32.us.us.us, 9223372036854775807
  %_6.sroa.0.0.off0.i.i.i.i.i.i.i.us.us.us = or i1 %27, %52
  %53 = or i1 %.lcssa2830.off0.us, %_6.sroa.0.0.off0.i.i.i.i.i.i.i.us.us.us
  %54 = select i1 %_5.i.i.i.i.i.i.i.us.us.us, i128 257, i128 0
  %55 = shl nuw nsw i128 %54, 16
  %56 = or disjoint i128 %54, %55
  %57 = shl nuw nsw i128 %56, 32
  %58 = or disjoint i128 %56, %57
  %59 = shl nuw nsw i128 %58, 64
  %60 = or disjoint i128 %58, %59
  %61 = bitcast i128 %60 to <16 x i8>
  br label %bb9.i.i.i.i.i.split.us.us

bb4.i.lr.ph.i.i.i.i.split:                        ; preds = %bb4.i.lr.ph.i.i.i.i
  call void @llvm.assume(i1 %20)
  br i1 %12, label %bb4.i.i.i.i.i.us100.preheader, label %bb4.i.lr.ph.i.i.i.i.split.split

bb4.i.i.i.i.i.us100.preheader:                    ; preds = %bb4.i.lr.ph.i.i.i.i.split
  %62 = getelementptr inbounds nuw i8, ptr %_7.sroa.7.123.i.i, i64 128
  %63 = getelementptr inbounds nuw i8, ptr %_7.sroa.7.123.i.i, i64 256
  %64 = getelementptr inbounds nuw i8, ptr %_7.sroa.7.123.i.i, i64 384
  br label %bb4.i.i.i.i.i.us100

bb4.i.i.i.i.i.us100:                              ; preds = %bb4.i.i.i.i.i.us100.preheader, %bb4.i.i.i.i.i.us100
  %.lcssa2830.off0.us101 = phi i1 [ %85, %bb4.i.i.i.i.i.us100 ], [ false, %bb4.i.i.i.i.i.us100.preheader ]
  %iter.i.sroa.0.032.i.i.i.i.us102 = phi ptr [ %_17.i.i.i.i.i.i.us105, %bb4.i.i.i.i.i.us100 ], [ %_83.i.i, %bb4.i.i.i.i.i.us100.preheader ]
  %iter.i.sroa.7.031.i.i.i.i.us103 = phi i64 [ %_9.0.i.i.i.i.i.us106, %bb4.i.i.i.i.i.us100 ], [ 0, %bb4.i.i.i.i.i.us100.preheader ]
  %_0.sroa.0.0.i9.i32.us78.us = load i64, ptr %23, align 8, !noalias !478, !noundef !8
  %65 = icmp eq i64 %_0.sroa.0.0.i9.i32.us78.us, 9223372036854775807
  %broadcast.splatinsert214 = insertelement <16 x i1> poison, i1 %65, i64 0
  %broadcast.splat215 = shufflevector <16 x i1> %broadcast.splatinsert214, <16 x i1> poison, <16 x i32> zeroinitializer
  %broadcast.splatinsert212 = insertelement <16 x i64> poison, i64 %_0.sroa.0.0.i9.i32.us78.us, i64 0
  %broadcast.splat213 = shufflevector <16 x i64> %broadcast.splatinsert212, <16 x i64> poison, <16 x i32> zeroinitializer
  %66 = insertelement <16 x i1> <i1 poison, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false>, i1 %.lcssa2830.off0.us101, i64 0
  %.idx = shl i64 %iter.i.sroa.7.031.i.i.i.i.us103, 9
  %67 = getelementptr inbounds nuw i8, ptr %_7.sroa.7.123.i.i, i64 %.idx
  %wide.load221 = load <16 x i64>, ptr %67, align 8, !noalias !467
  %68 = icmp eq <16 x i64> %wide.load221, splat (i64 9223372036854775807)
  %69 = icmp slt <16 x i64> %wide.load221, %broadcast.splat213
  %70 = or <16 x i1> %68, %66
  %.idx.1 = shl i64 %iter.i.sroa.7.031.i.i.i.i.us103, 9
  %71 = getelementptr inbounds nuw i8, ptr %62, i64 %.idx.1
  %wide.load221.1 = load <16 x i64>, ptr %71, align 8, !noalias !467
  %72 = icmp eq <16 x i64> %wide.load221.1, splat (i64 9223372036854775807)
  %73 = or <16 x i1> %72, %broadcast.splat215
  %74 = icmp slt <16 x i64> %wide.load221.1, %broadcast.splat213
  %75 = or <16 x i1> %70, %73
  %.idx.2 = shl i64 %iter.i.sroa.7.031.i.i.i.i.us103, 9
  %76 = getelementptr inbounds nuw i8, ptr %63, i64 %.idx.2
  %wide.load221.2 = load <16 x i64>, ptr %76, align 8, !noalias !467
  %77 = icmp eq <16 x i64> %wide.load221.2, splat (i64 9223372036854775807)
  %78 = icmp slt <16 x i64> %wide.load221.2, %broadcast.splat213
  %79 = or <16 x i1> %77, %75
  %.idx.3 = shl i64 %iter.i.sroa.7.031.i.i.i.i.us103, 9
  %80 = getelementptr inbounds nuw i8, ptr %64, i64 %.idx.3
  %wide.load221.3 = load <16 x i64>, ptr %80, align 8, !noalias !467
  %81 = icmp eq <16 x i64> %wide.load221.3, splat (i64 9223372036854775807)
  %82 = icmp slt <16 x i64> %wide.load221.3, %broadcast.splat213
  %83 = or <16 x i1> %81, %79
  %84 = bitcast <16 x i1> %83 to i16
  %85 = icmp ne i16 %84, 0
  %_17.i.i.i.i.i.i.us105 = getelementptr inbounds nuw i8, ptr %iter.i.sroa.0.032.i.i.i.i.us102, i64 8
  %_9.0.i.i.i.i.i.us106 = add nuw nsw i64 %iter.i.sroa.7.031.i.i.i.i.us103, 1
  %m0.i.i.i.i.i.i.us108 = select <16 x i1> %69, <16 x i8> <i8 1, i8 2, i8 4, i8 8, i8 16, i8 32, i8 64, i8 -128, i8 1, i8 2, i8 4, i8 8, i8 16, i8 32, i8 64, i8 -128>, <16 x i8> zeroinitializer
  %m1.i.i.i.i.i.i.us110 = select <16 x i1> %74, <16 x i8> <i8 1, i8 2, i8 4, i8 8, i8 16, i8 32, i8 64, i8 -128, i8 1, i8 2, i8 4, i8 8, i8 16, i8 32, i8 64, i8 -128>, <16 x i8> zeroinitializer
  %m2.i.i.i.i.i.i.us112 = select <16 x i1> %78, <16 x i8> <i8 1, i8 2, i8 4, i8 8, i8 16, i8 32, i8 64, i8 -128, i8 1, i8 2, i8 4, i8 8, i8 16, i8 32, i8 64, i8 -128>, <16 x i8> zeroinitializer
  %m3.i.i.i.i.i.i.us114 = select <16 x i1> %82, <16 x i8> <i8 1, i8 2, i8 4, i8 8, i8 16, i8 32, i8 64, i8 -128, i8 1, i8 2, i8 4, i8 8, i8 16, i8 32, i8 64, i8 -128>, <16 x i8> zeroinitializer
  %_29.i.i.i.i.i.i.us115 = shufflevector <16 x i8> %m0.i.i.i.i.i.i.us108, <16 x i8> %m1.i.i.i.i.i.i.us110, <16 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14, i32 16, i32 18, i32 20, i32 22, i32 24, i32 26, i32 28, i32 30>
  %_30.i.i.i.i.i.i.us116 = shufflevector <16 x i8> %m0.i.i.i.i.i.i.us108, <16 x i8> %m1.i.i.i.i.i.i.us110, <16 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15, i32 17, i32 19, i32 21, i32 23, i32 25, i32 27, i32 29, i32 31>
  %sum01.i.i.i.i.i.i.us117 = or disjoint <16 x i8> %_29.i.i.i.i.i.i.us115, %_30.i.i.i.i.i.i.us116
  %_31.i.i.i.i.i.i.us118 = shufflevector <16 x i8> %m2.i.i.i.i.i.i.us112, <16 x i8> %m3.i.i.i.i.i.i.us114, <16 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14, i32 16, i32 18, i32 20, i32 22, i32 24, i32 26, i32 28, i32 30>
  %_32.i.i.i.i.i.i.us119 = shufflevector <16 x i8> %m2.i.i.i.i.i.i.us112, <16 x i8> %m3.i.i.i.i.i.i.us114, <16 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15, i32 17, i32 19, i32 21, i32 23, i32 25, i32 27, i32 29, i32 31>
  %sum23.i.i.i.i.i.i.us120 = or disjoint <16 x i8> %_31.i.i.i.i.i.i.us118, %_32.i.i.i.i.i.i.us119
  %_33.i.i.i.i.i.i.us121 = shufflevector <16 x i8> %sum01.i.i.i.i.i.i.us117, <16 x i8> %sum23.i.i.i.i.i.i.us120, <16 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14, i32 16, i32 18, i32 20, i32 22, i32 24, i32 26, i32 28, i32 30>
  %_34.i.i.i.i.i.i.us122 = shufflevector <16 x i8> %sum01.i.i.i.i.i.i.us117, <16 x i8> %sum23.i.i.i.i.i.i.us120, <16 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15, i32 17, i32 19, i32 21, i32 23, i32 25, i32 27, i32 29, i32 31>
  %sum.i.i.i.i.i.i.us123 = or disjoint <16 x i8> %_33.i.i.i.i.i.i.us121, %_34.i.i.i.i.i.i.us122
  %_35.i.i.i.i.i.i.us124 = shufflevector <16 x i8> %sum.i.i.i.i.i.i.us123, <16 x i8> poison, <16 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %_36.i.i.i.i.i.i.us125 = shufflevector <16 x i8> %sum.i.i.i.i.i.i.us123, <16 x i8> poison, <16 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %sum1.i.i.i.i.i.i.us126 = add <16 x i8> %_35.i.i.i.i.i.i.us124, %_36.i.i.i.i.i.i.us125
  %_23.i.i.i.i.i.i.us127 = bitcast <16 x i8> %sum1.i.i.i.i.i.i.us126 to <2 x i64>
  %_0.i.i.i.i.i.i.us128 = extractelement <2 x i64> %_23.i.i.i.i.i.i.us127, i64 0
  store i64 %_0.i.i.i.i.i.i.us128, ptr %iter.i.sroa.0.032.i.i.i.i.us102, align 8, !alias.scope !481, !noalias !486
  %_7.i.i.i.i.i.i.us129 = icmp eq ptr %_17.i.i.i.i.i.i.us105, %_52.i.i.i.i.i
  br i1 %_7.i.i.i.i.i.i.us129, label %bb5.i.i.i.i.i, label %bb4.i.i.i.i.i.us100

bb4.i.lr.ph.i.i.i.i.split.split:                  ; preds = %bb4.i.lr.ph.i.i.i.i.split
  call void @llvm.assume(i1 %22)
  br label %bb4.i.i.i.i.i

bb4.i.i.i.i.i:                                    ; preds = %bb4.i.i.i.i.i, %bb4.i.lr.ph.i.i.i.i.split.split
  %.lcssa2830.off0 = phi i1 [ false, %bb4.i.lr.ph.i.i.i.i.split.split ], [ %119, %bb4.i.i.i.i.i ]
  %iter.i.sroa.0.032.i.i.i.i = phi ptr [ %_83.i.i, %bb4.i.lr.ph.i.i.i.i.split.split ], [ %_17.i.i.i.i.i.i, %bb4.i.i.i.i.i ]
  %iter.i.sroa.7.031.i.i.i.i = phi i64 [ 0, %bb4.i.lr.ph.i.i.i.i.split.split ], [ %_9.0.i.i.i.i.i, %bb4.i.i.i.i.i ]
  %86 = insertelement <16 x i1> <i1 poison, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false>, i1 %.lcssa2830.off0, i64 0
  %offset2.i.i.i.i.i = shl i64 %iter.i.sroa.7.031.i.i.i.i, 6
  %87 = getelementptr inbounds nuw i64, ptr %_7.sroa.7.123.i.i, i64 %offset2.i.i.i.i.i
  %wide.load = load <16 x i64>, ptr %87, align 8, !noalias !467
  %88 = getelementptr inbounds nuw i64, ptr %_26.sroa.14.0, i64 %offset2.i.i.i.i.i
  %wide.load210 = load <16 x i64>, ptr %88, align 8, !noalias !478
  %89 = icmp eq <16 x i64> %wide.load, splat (i64 9223372036854775807)
  %90 = icmp eq <16 x i64> %wide.load210, splat (i64 9223372036854775807)
  %91 = or <16 x i1> %89, %90
  %92 = icmp slt <16 x i64> %wide.load, %wide.load210
  %93 = or <16 x i1> %86, %91
  %94 = or disjoint i64 %offset2.i.i.i.i.i, 16
  %95 = getelementptr inbounds nuw i64, ptr %_7.sroa.7.123.i.i, i64 %94
  %wide.load.1 = load <16 x i64>, ptr %95, align 8, !noalias !467
  %96 = getelementptr inbounds nuw i64, ptr %_26.sroa.14.0, i64 %94
  %wide.load210.1 = load <16 x i64>, ptr %96, align 8, !noalias !478
  %97 = icmp eq <16 x i64> %wide.load.1, splat (i64 9223372036854775807)
  %98 = icmp eq <16 x i64> %wide.load210.1, splat (i64 9223372036854775807)
  %99 = or <16 x i1> %97, %98
  %100 = icmp slt <16 x i64> %wide.load.1, %wide.load210.1
  %101 = or <16 x i1> %93, %99
  %102 = or disjoint i64 %offset2.i.i.i.i.i, 32
  %103 = getelementptr inbounds nuw i64, ptr %_7.sroa.7.123.i.i, i64 %102
  %wide.load.2 = load <16 x i64>, ptr %103, align 8, !noalias !467
  %104 = getelementptr inbounds nuw i64, ptr %_26.sroa.14.0, i64 %102
  %wide.load210.2 = load <16 x i64>, ptr %104, align 8, !noalias !478
  %105 = icmp eq <16 x i64> %wide.load.2, splat (i64 9223372036854775807)
  %106 = icmp eq <16 x i64> %wide.load210.2, splat (i64 9223372036854775807)
  %107 = or <16 x i1> %105, %106
  %108 = icmp slt <16 x i64> %wide.load.2, %wide.load210.2
  %109 = or <16 x i1> %101, %107
  %110 = or disjoint i64 %offset2.i.i.i.i.i, 48
  %111 = getelementptr inbounds nuw i64, ptr %_7.sroa.7.123.i.i, i64 %110
  %wide.load.3 = load <16 x i64>, ptr %111, align 8, !noalias !467
  %112 = getelementptr inbounds nuw i64, ptr %_26.sroa.14.0, i64 %110
  %wide.load210.3 = load <16 x i64>, ptr %112, align 8, !noalias !478
  %113 = icmp eq <16 x i64> %wide.load.3, splat (i64 9223372036854775807)
  %114 = icmp eq <16 x i64> %wide.load210.3, splat (i64 9223372036854775807)
  %115 = or <16 x i1> %113, %114
  %116 = icmp slt <16 x i64> %wide.load.3, %wide.load210.3
  %117 = or <16 x i1> %109, %115
  %118 = bitcast <16 x i1> %117 to i16
  %119 = icmp ne i16 %118, 0
  %_17.i.i.i.i.i.i = getelementptr inbounds nuw i8, ptr %iter.i.sroa.0.032.i.i.i.i, i64 8
  %_9.0.i.i.i.i.i = add nuw nsw i64 %iter.i.sroa.7.031.i.i.i.i, 1
  %m0.i.i.i.i.i.i = select <16 x i1> %92, <16 x i8> <i8 1, i8 2, i8 4, i8 8, i8 16, i8 32, i8 64, i8 -128, i8 1, i8 2, i8 4, i8 8, i8 16, i8 32, i8 64, i8 -128>, <16 x i8> zeroinitializer
  %m1.i.i.i.i.i.i = select <16 x i1> %100, <16 x i8> <i8 1, i8 2, i8 4, i8 8, i8 16, i8 32, i8 64, i8 -128, i8 1, i8 2, i8 4, i8 8, i8 16, i8 32, i8 64, i8 -128>, <16 x i8> zeroinitializer
  %m2.i.i.i.i.i.i = select <16 x i1> %108, <16 x i8> <i8 1, i8 2, i8 4, i8 8, i8 16, i8 32, i8 64, i8 -128, i8 1, i8 2, i8 4, i8 8, i8 16, i8 32, i8 64, i8 -128>, <16 x i8> zeroinitializer
  %m3.i.i.i.i.i.i = select <16 x i1> %116, <16 x i8> <i8 1, i8 2, i8 4, i8 8, i8 16, i8 32, i8 64, i8 -128, i8 1, i8 2, i8 4, i8 8, i8 16, i8 32, i8 64, i8 -128>, <16 x i8> zeroinitializer
  %_29.i.i.i.i.i.i = shufflevector <16 x i8> %m0.i.i.i.i.i.i, <16 x i8> %m1.i.i.i.i.i.i, <16 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14, i32 16, i32 18, i32 20, i32 22, i32 24, i32 26, i32 28, i32 30>
  %_30.i.i.i.i.i.i = shufflevector <16 x i8> %m0.i.i.i.i.i.i, <16 x i8> %m1.i.i.i.i.i.i, <16 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15, i32 17, i32 19, i32 21, i32 23, i32 25, i32 27, i32 29, i32 31>
  %sum01.i.i.i.i.i.i = or disjoint <16 x i8> %_29.i.i.i.i.i.i, %_30.i.i.i.i.i.i
  %_31.i.i.i.i.i.i = shufflevector <16 x i8> %m2.i.i.i.i.i.i, <16 x i8> %m3.i.i.i.i.i.i, <16 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14, i32 16, i32 18, i32 20, i32 22, i32 24, i32 26, i32 28, i32 30>
  %_32.i.i.i.i.i.i = shufflevector <16 x i8> %m2.i.i.i.i.i.i, <16 x i8> %m3.i.i.i.i.i.i, <16 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15, i32 17, i32 19, i32 21, i32 23, i32 25, i32 27, i32 29, i32 31>
  %sum23.i.i.i.i.i.i = or disjoint <16 x i8> %_31.i.i.i.i.i.i, %_32.i.i.i.i.i.i
  %_33.i.i.i.i.i.i = shufflevector <16 x i8> %sum01.i.i.i.i.i.i, <16 x i8> %sum23.i.i.i.i.i.i, <16 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14, i32 16, i32 18, i32 20, i32 22, i32 24, i32 26, i32 28, i32 30>
  %_34.i.i.i.i.i.i = shufflevector <16 x i8> %sum01.i.i.i.i.i.i, <16 x i8> %sum23.i.i.i.i.i.i, <16 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15, i32 17, i32 19, i32 21, i32 23, i32 25, i32 27, i32 29, i32 31>
  %sum.i.i.i.i.i.i = or disjoint <16 x i8> %_33.i.i.i.i.i.i, %_34.i.i.i.i.i.i
  %_35.i.i.i.i.i.i = shufflevector <16 x i8> %sum.i.i.i.i.i.i, <16 x i8> poison, <16 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %_36.i.i.i.i.i.i = shufflevector <16 x i8> %sum.i.i.i.i.i.i, <16 x i8> poison, <16 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %sum1.i.i.i.i.i.i = add <16 x i8> %_35.i.i.i.i.i.i, %_36.i.i.i.i.i.i
  %_23.i.i.i.i.i.i = bitcast <16 x i8> %sum1.i.i.i.i.i.i to <2 x i64>
  %_0.i.i.i.i.i.i = extractelement <2 x i64> %_23.i.i.i.i.i.i, i64 0
  store i64 %_0.i.i.i.i.i.i, ptr %iter.i.sroa.0.032.i.i.i.i, align 8, !alias.scope !481, !noalias !486
  %_7.i.i.i.i.i.i = icmp eq ptr %_17.i.i.i.i.i.i, %_52.i.i.i.i.i
  br i1 %_7.i.i.i.i.i.i, label %bb5.i.i.i.i.i, label %bb4.i.i.i.i.i

bb5.i.i.i.i.i:                                    ; preds = %bb4.i.i.i.i.i, %bb4.i.i.i.i.i.us100, %bb9.i.i.i.i.i.split.us.us, %.noexc
  %failure.sroa.0.0.off0 = phi i1 [ false, %.noexc ], [ %.us-phi86.us, %bb9.i.i.i.i.i.split.us.us ], [ %85, %bb4.i.i.i.i.i.us100 ], [ %119, %bb4.i.i.i.i.i ]
  br i1 %_20.not.i.i, label %bb12.i.i.i.i.i, label %.noexc12

bb12.i.i.i.i.i:                                   ; preds = %bb5.i.i.i.i.i
  %120 = and i64 %5, -64
  %121 = icmp ne ptr %_7.sroa.7.123.i.i, null
  %122 = icmp ne ptr %_26.sroa.14.0, null
  %123 = inttoptr i64 %_26.sroa.16.0 to ptr
  br i1 %8, label %bb12.i.i.i.i.i.split.us, label %bb12.i.i.i.i.i.split

bb12.i.i.i.i.i.split.us:                          ; preds = %bb12.i.i.i.i.i
  %124 = inttoptr i64 %_7.sroa.9.122.i.i to ptr
  %_0.sroa.0.0.i.i.us = load i64, ptr %124, align 8, !noalias !489, !noundef !8
  %125 = icmp eq i64 %_0.sroa.0.0.i.i.us, 9223372036854775807
  br i1 %12, label %bb12.i.i.i.i.i.split.us.split.us, label %bb12.i.i.i.i.i.split.us.split

bb12.i.i.i.i.i.split.us.split.us:                 ; preds = %bb12.i.i.i.i.i.split.us
  %_0.sroa.0.0.i9.i.us.us = load i64, ptr %123, align 8, !noalias !494, !noundef !8
  %_5.i.i.i.i.i.i.i.i.i.us.us = icmp slt i64 %_0.sroa.0.0.i.i.us, %_0.sroa.0.0.i9.i.us.us
  %_13.i.i.i.i.i.us.us = zext i1 %_5.i.i.i.i.i.i.i.i.i.us.us to i64
  %min.iters.check317 = icmp samesign ult i64 %_19.i.i, 8
  br i1 %min.iters.check317, label %bb8.i5.i.i.i.i.us.us.preheader, label %vector.ph318

vector.ph318:                                     ; preds = %bb12.i.i.i.i.i.split.us.split.us
  %n.vec320 = and i64 %5, 56
  %broadcast.splatinsert321 = insertelement <2 x i64> poison, i64 %_13.i.i.i.i.i.us.us, i64 0
  %broadcast.splat322 = shufflevector <2 x i64> %broadcast.splatinsert321, <2 x i64> poison, <2 x i32> zeroinitializer
  br label %vector.body323

vector.body323:                                   ; preds = %vector.body323, %vector.ph318
  %index324 = phi i64 [ 0, %vector.ph318 ], [ %index.next331, %vector.body323 ]
  %vec.phi325 = phi <2 x i64> [ zeroinitializer, %vector.ph318 ], [ %130, %vector.body323 ]
  %vec.phi326 = phi <2 x i64> [ zeroinitializer, %vector.ph318 ], [ %131, %vector.body323 ]
  %vec.phi327 = phi <2 x i64> [ zeroinitializer, %vector.ph318 ], [ %132, %vector.body323 ]
  %vec.phi328 = phi <2 x i64> [ zeroinitializer, %vector.ph318 ], [ %133, %vector.body323 ]
  %vec.ind329 = phi <2 x i64> [ <i64 0, i64 1>, %vector.ph318 ], [ %vec.ind.next332, %vector.body323 ]
  %step.add330 = add <2 x i64> %vec.ind329, splat (i64 2)
  %step.add.2 = add <2 x i64> %vec.ind329, splat (i64 4)
  %step.add.3 = add <2 x i64> %vec.ind329, splat (i64 6)
  %126 = shl nuw <2 x i64> %broadcast.splat322, %vec.ind329
  %127 = shl nuw <2 x i64> %broadcast.splat322, %step.add330
  %128 = shl nuw <2 x i64> %broadcast.splat322, %step.add.2
  %129 = shl nuw <2 x i64> %broadcast.splat322, %step.add.3
  %130 = or <2 x i64> %126, %vec.phi325
  %131 = or <2 x i64> %127, %vec.phi326
  %132 = or <2 x i64> %128, %vec.phi327
  %133 = or <2 x i64> %129, %vec.phi328
  %index.next331 = add nuw i64 %index324, 8
  %vec.ind.next332 = add <2 x i64> %vec.ind329, splat (i64 8)
  %134 = icmp eq i64 %index.next331, %n.vec320
  br i1 %134, label %middle.block333, label %vector.body323, !llvm.loop !497

middle.block333:                                  ; preds = %vector.body323
  %bin.rdx334 = or <2 x i64> %131, %130
  %bin.rdx335 = or <2 x i64> %132, %bin.rdx334
  %bin.rdx336 = or <2 x i64> %133, %bin.rdx335
  %135 = call i64 @llvm.vector.reduce.or.v2i64(<2 x i64> %bin.rdx336)
  %cmp.n337 = icmp eq i64 %_19.i.i, %n.vec320
  br i1 %cmp.n337, label %bb14.i.i.i.i.i.loopexit, label %bb8.i5.i.i.i.i.us.us.preheader

bb8.i5.i.i.i.i.us.us.preheader:                   ; preds = %bb12.i.i.i.i.i.split.us.split.us, %middle.block333
  %packed.sroa.0.05.i.i.i.i.i.us.us.ph = phi i64 [ 0, %bb12.i.i.i.i.i.split.us.split.us ], [ %135, %middle.block333 ]
  %iter.sroa.0.04.i.i.i.i.i.us.us.ph = phi i64 [ 0, %bb12.i.i.i.i.i.split.us.split.us ], [ %n.vec320, %middle.block333 ]
  br label %bb8.i5.i.i.i.i.us.us

bb8.i5.i.i.i.i.us.us:                             ; preds = %bb8.i5.i.i.i.i.us.us.preheader, %bb8.i5.i.i.i.i.us.us
  %packed.sroa.0.05.i.i.i.i.i.us.us = phi i64 [ %137, %bb8.i5.i.i.i.i.us.us ], [ %packed.sroa.0.05.i.i.i.i.i.us.us.ph, %bb8.i5.i.i.i.i.us.us.preheader ]
  %iter.sroa.0.04.i.i.i.i.i.us.us = phi i64 [ %136, %bb8.i5.i.i.i.i.us.us ], [ %iter.sroa.0.04.i.i.i.i.i.us.us.ph, %bb8.i5.i.i.i.i.us.us.preheader ]
  %136 = add nuw nsw i64 %iter.sroa.0.04.i.i.i.i.i.us.us, 1
  %_12.i.i.i.i.i.us.us = shl nuw i64 %_13.i.i.i.i.i.us.us, %iter.sroa.0.04.i.i.i.i.i.us.us
  %137 = or i64 %_12.i.i.i.i.i.us.us, %packed.sroa.0.05.i.i.i.i.i.us.us
  %exitcond.not.i.i.i.i.i.us.us = icmp eq i64 %136, %_19.i.i
  br i1 %exitcond.not.i.i.i.i.i.us.us, label %bb14.i.i.i.i.i.loopexit, label %bb8.i5.i.i.i.i.us.us, !llvm.loop !498

bb12.i.i.i.i.i.split.us.split:                    ; preds = %bb12.i.i.i.i.i.split.us
  call void @llvm.assume(i1 %122)
  %min.iters.check287 = icmp samesign ult i64 %_19.i.i, 8
  br i1 %min.iters.check287, label %bb8.i5.i.i.i.i.us.preheader, label %vector.ph288

vector.ph288:                                     ; preds = %bb12.i.i.i.i.i.split.us.split
  %n.vec290 = and i64 %5, 56
  %138 = insertelement <4 x i1> <i1 poison, i1 false, i1 false, i1 false>, i1 %failure.sroa.0.0.off0, i64 0
  %broadcast.splatinsert293 = insertelement <4 x i64> poison, i64 %_0.sroa.0.0.i.i.us, i64 0
  %broadcast.splat294 = shufflevector <4 x i64> %broadcast.splatinsert293, <4 x i64> poison, <4 x i32> zeroinitializer
  %broadcast.splatinsert295 = insertelement <4 x i1> poison, i1 %125, i64 0
  %broadcast.splat296 = shufflevector <4 x i1> %broadcast.splatinsert295, <4 x i1> poison, <4 x i32> zeroinitializer
  %invariant.gep415 = getelementptr i64, ptr %_26.sroa.14.0, i64 %120
  br label %vector.body297

vector.body297:                                   ; preds = %vector.body297, %vector.ph288
  %index298 = phi i64 [ 0, %vector.ph288 ], [ %index.next307, %vector.body297 ]
  %vec.phi299 = phi <4 x i1> [ %138, %vector.ph288 ], [ %146, %vector.body297 ]
  %vec.phi300 = phi <4 x i1> [ zeroinitializer, %vector.ph288 ], [ %147, %vector.body297 ]
  %vec.phi301 = phi <4 x i64> [ zeroinitializer, %vector.ph288 ], [ %152, %vector.body297 ]
  %vec.phi302 = phi <4 x i64> [ zeroinitializer, %vector.ph288 ], [ %153, %vector.body297 ]
  %vec.ind303 = phi <4 x i64> [ <i64 0, i64 1, i64 2, i64 3>, %vector.ph288 ], [ %vec.ind.next308, %vector.body297 ]
  %step.add304 = add <4 x i64> %vec.ind303, splat (i64 4)
  %gep416 = getelementptr i64, ptr %invariant.gep415, i64 %index298
  %139 = getelementptr inbounds nuw i8, ptr %gep416, i64 32
  %wide.load305 = load <4 x i64>, ptr %gep416, align 8, !noalias !494
  %wide.load306 = load <4 x i64>, ptr %139, align 8, !noalias !494
  %140 = icmp eq <4 x i64> %wide.load305, splat (i64 9223372036854775807)
  %141 = icmp eq <4 x i64> %wide.load306, splat (i64 9223372036854775807)
  %142 = icmp slt <4 x i64> %broadcast.splat294, %wide.load305
  %143 = icmp slt <4 x i64> %broadcast.splat294, %wide.load306
  %144 = or <4 x i1> %140, %vec.phi299
  %145 = or <4 x i1> %141, %vec.phi300
  %146 = or <4 x i1> %144, %broadcast.splat296
  %147 = or <4 x i1> %145, %broadcast.splat296
  %148 = zext <4 x i1> %142 to <4 x i64>
  %149 = zext <4 x i1> %143 to <4 x i64>
  %150 = shl nuw <4 x i64> %148, %vec.ind303
  %151 = shl nuw <4 x i64> %149, %step.add304
  %152 = or <4 x i64> %150, %vec.phi301
  %153 = or <4 x i64> %151, %vec.phi302
  %index.next307 = add nuw i64 %index298, 8
  %vec.ind.next308 = add <4 x i64> %vec.ind303, splat (i64 8)
  %154 = icmp eq i64 %index.next307, %n.vec290
  br i1 %154, label %middle.block309, label %vector.body297, !llvm.loop !499

middle.block309:                                  ; preds = %vector.body297
  %bin.rdx310 = or <4 x i1> %145, %146
  %155 = bitcast <4 x i1> %bin.rdx310 to i4
  %156 = icmp ne i4 %155, 0
  %bin.rdx311 = or <4 x i64> %153, %152
  %157 = call i64 @llvm.vector.reduce.or.v4i64(<4 x i64> %bin.rdx311)
  %cmp.n312 = icmp eq i64 %_19.i.i, %n.vec290
  br i1 %cmp.n312, label %bb14.i.i.i.i.i, label %bb8.i5.i.i.i.i.us.preheader

bb8.i5.i.i.i.i.us.preheader:                      ; preds = %bb12.i.i.i.i.i.split.us.split, %middle.block309
  %.off049.us.ph = phi i1 [ %failure.sroa.0.0.off0, %bb12.i.i.i.i.i.split.us.split ], [ %156, %middle.block309 ]
  %packed.sroa.0.05.i.i.i.i.i.us.ph = phi i64 [ 0, %bb12.i.i.i.i.i.split.us.split ], [ %157, %middle.block309 ]
  %iter.sroa.0.04.i.i.i.i.i.us.ph = phi i64 [ 0, %bb12.i.i.i.i.i.split.us.split ], [ %n.vec290, %middle.block309 ]
  br label %bb8.i5.i.i.i.i.us

bb8.i5.i.i.i.i.us:                                ; preds = %bb8.i5.i.i.i.i.us.preheader, %bb8.i5.i.i.i.i.us
  %.off049.us = phi i1 [ %160, %bb8.i5.i.i.i.i.us ], [ %.off049.us.ph, %bb8.i5.i.i.i.i.us.preheader ]
  %packed.sroa.0.05.i.i.i.i.i.us = phi i64 [ %162, %bb8.i5.i.i.i.i.us ], [ %packed.sroa.0.05.i.i.i.i.i.us.ph, %bb8.i5.i.i.i.i.us.preheader ]
  %iter.sroa.0.04.i.i.i.i.i.us = phi i64 [ %161, %bb8.i5.i.i.i.i.us ], [ %iter.sroa.0.04.i.i.i.i.i.us.ph, %bb8.i5.i.i.i.i.us.preheader ]
  %_4.i.i.i.i.i.i.us = add nuw nsw i64 %iter.sroa.0.04.i.i.i.i.i.us, %120
  %_5.i.i6.i.us = icmp ult i64 %_4.i.i.i.i.i.i.us, %_26.sroa.16.0
  call void @llvm.assume(i1 %_5.i.i6.i.us), !noalias !500
  %_4.i.i7.i.us = getelementptr inbounds nuw i64, ptr %_26.sroa.14.0, i64 %_4.i.i.i.i.i.i.us
  %_0.sroa.0.0.i9.i.us = load i64, ptr %_4.i.i7.i.us, align 8, !noalias !494, !noundef !8
  %158 = icmp eq i64 %_0.sroa.0.0.i9.i.us, 9223372036854775807
  %_5.i.i.i.i.i.i.i.i.i.us = icmp slt i64 %_0.sroa.0.0.i.i.us, %_0.sroa.0.0.i9.i.us
  %159 = or i1 %158, %.off049.us
  %160 = or i1 %159, %125
  %161 = add nuw nsw i64 %iter.sroa.0.04.i.i.i.i.i.us, 1
  %_13.i.i.i.i.i.us = zext i1 %_5.i.i.i.i.i.i.i.i.i.us to i64
  %_12.i.i.i.i.i.us = shl nuw i64 %_13.i.i.i.i.i.us, %iter.sroa.0.04.i.i.i.i.i.us
  %162 = or i64 %_12.i.i.i.i.i.us, %packed.sroa.0.05.i.i.i.i.i.us
  %exitcond.not.i.i.i.i.i.us = icmp eq i64 %161, %_19.i.i
  br i1 %exitcond.not.i.i.i.i.i.us, label %bb14.i.i.i.i.i, label %bb8.i5.i.i.i.i.us, !llvm.loop !501

bb12.i.i.i.i.i.split:                             ; preds = %bb12.i.i.i.i.i
  call void @llvm.assume(i1 %121)
  br i1 %12, label %bb12.i.i.i.i.i.split.split.us, label %bb12.i.i.i.i.i.split.split

bb12.i.i.i.i.i.split.split.us:                    ; preds = %bb12.i.i.i.i.i.split
  %_0.sroa.0.0.i9.i.us142 = load i64, ptr %123, align 8, !noalias !494, !noundef !8
  %163 = icmp eq i64 %_0.sroa.0.0.i9.i.us142, 9223372036854775807
  %min.iters.check257 = icmp samesign ult i64 %_19.i.i, 8
  br i1 %min.iters.check257, label %bb8.i5.i.i.i.i.us133.preheader, label %vector.ph258

vector.ph258:                                     ; preds = %bb12.i.i.i.i.i.split.split.us
  %n.vec260 = and i64 %5, 56
  %164 = insertelement <4 x i1> <i1 poison, i1 false, i1 false, i1 false>, i1 %failure.sroa.0.0.off0, i64 0
  %broadcast.splatinsert261 = insertelement <4 x i64> poison, i64 %_0.sroa.0.0.i9.i.us142, i64 0
  %broadcast.splat262 = shufflevector <4 x i64> %broadcast.splatinsert261, <4 x i64> poison, <4 x i32> zeroinitializer
  %broadcast.splatinsert263 = insertelement <4 x i1> poison, i1 %163, i64 0
  %broadcast.splat264 = shufflevector <4 x i1> %broadcast.splatinsert263, <4 x i1> poison, <4 x i32> zeroinitializer
  %invariant.gep = getelementptr i64, ptr %_7.sroa.7.123.i.i, i64 %120
  br label %vector.body267

vector.body267:                                   ; preds = %vector.body267, %vector.ph258
  %index268 = phi i64 [ 0, %vector.ph258 ], [ %index.next277, %vector.body267 ]
  %vec.phi269 = phi <4 x i1> [ %164, %vector.ph258 ], [ %171, %vector.body267 ]
  %vec.phi270 = phi <4 x i1> [ zeroinitializer, %vector.ph258 ], [ %173, %vector.body267 ]
  %vec.phi271 = phi <4 x i64> [ zeroinitializer, %vector.ph258 ], [ %178, %vector.body267 ]
  %vec.phi272 = phi <4 x i64> [ zeroinitializer, %vector.ph258 ], [ %179, %vector.body267 ]
  %vec.ind273 = phi <4 x i64> [ <i64 0, i64 1, i64 2, i64 3>, %vector.ph258 ], [ %vec.ind.next278, %vector.body267 ]
  %step.add274 = add <4 x i64> %vec.ind273, splat (i64 4)
  %gep = getelementptr i64, ptr %invariant.gep, i64 %index268
  %165 = getelementptr inbounds nuw i8, ptr %gep, i64 32
  %wide.load275 = load <4 x i64>, ptr %gep, align 8, !noalias !489
  %wide.load276 = load <4 x i64>, ptr %165, align 8, !noalias !489
  %166 = icmp eq <4 x i64> %wide.load275, splat (i64 9223372036854775807)
  %167 = icmp eq <4 x i64> %wide.load276, splat (i64 9223372036854775807)
  %168 = icmp slt <4 x i64> %wide.load275, %broadcast.splat262
  %169 = icmp slt <4 x i64> %wide.load276, %broadcast.splat262
  %170 = or <4 x i1> %166, %vec.phi269
  %171 = or <4 x i1> %170, %broadcast.splat264
  %172 = or <4 x i1> %167, %vec.phi270
  %173 = or <4 x i1> %172, %broadcast.splat264
  %174 = zext <4 x i1> %168 to <4 x i64>
  %175 = zext <4 x i1> %169 to <4 x i64>
  %176 = shl nuw <4 x i64> %174, %vec.ind273
  %177 = shl nuw <4 x i64> %175, %step.add274
  %178 = or <4 x i64> %176, %vec.phi271
  %179 = or <4 x i64> %177, %vec.phi272
  %index.next277 = add nuw i64 %index268, 8
  %vec.ind.next278 = add <4 x i64> %vec.ind273, splat (i64 8)
  %180 = icmp eq i64 %index.next277, %n.vec260
  br i1 %180, label %middle.block279, label %vector.body267, !llvm.loop !502

middle.block279:                                  ; preds = %vector.body267
  %bin.rdx280 = or <4 x i1> %172, %171
  %181 = bitcast <4 x i1> %bin.rdx280 to i4
  %182 = icmp ne i4 %181, 0
  %bin.rdx281 = or <4 x i64> %179, %178
  %183 = call i64 @llvm.vector.reduce.or.v4i64(<4 x i64> %bin.rdx281)
  %cmp.n282 = icmp eq i64 %_19.i.i, %n.vec260
  br i1 %cmp.n282, label %bb14.i.i.i.i.i, label %bb8.i5.i.i.i.i.us133.preheader

bb8.i5.i.i.i.i.us133.preheader:                   ; preds = %bb12.i.i.i.i.i.split.split.us, %middle.block279
  %.off049.us134.ph = phi i1 [ %failure.sroa.0.0.off0, %bb12.i.i.i.i.i.split.split.us ], [ %182, %middle.block279 ]
  %packed.sroa.0.05.i.i.i.i.i.us135.ph = phi i64 [ 0, %bb12.i.i.i.i.i.split.split.us ], [ %183, %middle.block279 ]
  %iter.sroa.0.04.i.i.i.i.i.us136.ph = phi i64 [ 0, %bb12.i.i.i.i.i.split.split.us ], [ %n.vec260, %middle.block279 ]
  br label %bb8.i5.i.i.i.i.us133

bb8.i5.i.i.i.i.us133:                             ; preds = %bb8.i5.i.i.i.i.us133.preheader, %bb8.i5.i.i.i.i.us133
  %.off049.us134 = phi i1 [ %186, %bb8.i5.i.i.i.i.us133 ], [ %.off049.us134.ph, %bb8.i5.i.i.i.i.us133.preheader ]
  %packed.sroa.0.05.i.i.i.i.i.us135 = phi i64 [ %188, %bb8.i5.i.i.i.i.us133 ], [ %packed.sroa.0.05.i.i.i.i.i.us135.ph, %bb8.i5.i.i.i.i.us133.preheader ]
  %iter.sroa.0.04.i.i.i.i.i.us136 = phi i64 [ %187, %bb8.i5.i.i.i.i.us133 ], [ %iter.sroa.0.04.i.i.i.i.i.us136.ph, %bb8.i5.i.i.i.i.us133.preheader ]
  %_4.i.i.i.i.i.i.us137 = add nuw nsw i64 %iter.sroa.0.04.i.i.i.i.i.us136, %120
  %_5.i.i.i.us = icmp ult i64 %_4.i.i.i.i.i.i.us137, %_7.sroa.9.122.i.i
  call void @llvm.assume(i1 %_5.i.i.i.us), !noalias !500
  %_4.i.i.i.us = getelementptr inbounds nuw i64, ptr %_7.sroa.7.123.i.i, i64 %_4.i.i.i.i.i.i.us137
  %_0.sroa.0.0.i.i.us138 = load i64, ptr %_4.i.i.i.us, align 8, !noalias !489, !noundef !8
  %184 = icmp eq i64 %_0.sroa.0.0.i.i.us138, 9223372036854775807
  %_5.i.i.i.i.i.i.i.i.i.us144 = icmp slt i64 %_0.sroa.0.0.i.i.us138, %_0.sroa.0.0.i9.i.us142
  %185 = or i1 %184, %.off049.us134
  %186 = or i1 %185, %163
  %187 = add nuw nsw i64 %iter.sroa.0.04.i.i.i.i.i.us136, 1
  %_13.i.i.i.i.i.us145 = zext i1 %_5.i.i.i.i.i.i.i.i.i.us144 to i64
  %_12.i.i.i.i.i.us146 = shl nuw i64 %_13.i.i.i.i.i.us145, %iter.sroa.0.04.i.i.i.i.i.us136
  %188 = or i64 %_12.i.i.i.i.i.us146, %packed.sroa.0.05.i.i.i.i.i.us135
  %exitcond.not.i.i.i.i.i.us147 = icmp eq i64 %187, %_19.i.i
  br i1 %exitcond.not.i.i.i.i.i.us147, label %bb14.i.i.i.i.i, label %bb8.i5.i.i.i.i.us133, !llvm.loop !503

bb12.i.i.i.i.i.split.split:                       ; preds = %bb12.i.i.i.i.i.split
  call void @llvm.assume(i1 %122)
  %min.iters.check = icmp samesign ult i64 %_19.i.i, 8
  br i1 %min.iters.check, label %bb8.i5.i.i.i.i.preheader, label %vector.ph237

vector.ph237:                                     ; preds = %bb12.i.i.i.i.i.split.split
  %n.vec = and i64 %5, 56
  %189 = insertelement <4 x i1> <i1 poison, i1 false, i1 false, i1 false>, i1 %failure.sroa.0.0.off0, i64 0
  br label %vector.body242

vector.body242:                                   ; preds = %vector.body242, %vector.ph237
  %index243 = phi i64 [ 0, %vector.ph237 ], [ %index.next252, %vector.body242 ]
  %vec.phi244 = phi <4 x i1> [ %189, %vector.ph237 ], [ %203, %vector.body242 ]
  %vec.phi245 = phi <4 x i1> [ zeroinitializer, %vector.ph237 ], [ %204, %vector.body242 ]
  %vec.phi246 = phi <4 x i64> [ zeroinitializer, %vector.ph237 ], [ %209, %vector.body242 ]
  %vec.phi247 = phi <4 x i64> [ zeroinitializer, %vector.ph237 ], [ %210, %vector.body242 ]
  %vec.ind = phi <4 x i64> [ <i64 0, i64 1, i64 2, i64 3>, %vector.ph237 ], [ %vec.ind.next, %vector.body242 ]
  %step.add = add <4 x i64> %vec.ind, splat (i64 4)
  %190 = add nuw nsw i64 %index243, %120
  %191 = getelementptr inbounds nuw i64, ptr %_7.sroa.7.123.i.i, i64 %190
  %192 = getelementptr inbounds nuw i8, ptr %191, i64 32
  %wide.load248 = load <4 x i64>, ptr %191, align 8, !noalias !489
  %wide.load249 = load <4 x i64>, ptr %192, align 8, !noalias !489
  %193 = getelementptr inbounds nuw i64, ptr %_26.sroa.14.0, i64 %190
  %194 = getelementptr inbounds nuw i8, ptr %193, i64 32
  %wide.load250 = load <4 x i64>, ptr %193, align 8, !noalias !494
  %wide.load251 = load <4 x i64>, ptr %194, align 8, !noalias !494
  %195 = icmp eq <4 x i64> %wide.load248, splat (i64 9223372036854775807)
  %196 = icmp eq <4 x i64> %wide.load249, splat (i64 9223372036854775807)
  %197 = icmp eq <4 x i64> %wide.load250, splat (i64 9223372036854775807)
  %198 = icmp eq <4 x i64> %wide.load251, splat (i64 9223372036854775807)
  %199 = or <4 x i1> %195, %197
  %200 = or <4 x i1> %196, %198
  %201 = icmp slt <4 x i64> %wide.load248, %wide.load250
  %202 = icmp slt <4 x i64> %wide.load249, %wide.load251
  %203 = or <4 x i1> %vec.phi244, %199
  %204 = or <4 x i1> %vec.phi245, %200
  %205 = zext <4 x i1> %201 to <4 x i64>
  %206 = zext <4 x i1> %202 to <4 x i64>
  %207 = shl nuw <4 x i64> %205, %vec.ind
  %208 = shl nuw <4 x i64> %206, %step.add
  %209 = or <4 x i64> %207, %vec.phi246
  %210 = or <4 x i64> %208, %vec.phi247
  %index.next252 = add nuw i64 %index243, 8
  %vec.ind.next = add <4 x i64> %vec.ind, splat (i64 8)
  %211 = icmp eq i64 %index.next252, %n.vec
  br i1 %211, label %middle.block253, label %vector.body242, !llvm.loop !504

middle.block253:                                  ; preds = %vector.body242
  %bin.rdx = or <4 x i1> %204, %203
  %212 = bitcast <4 x i1> %bin.rdx to i4
  %213 = icmp ne i4 %212, 0
  %bin.rdx254 = or <4 x i64> %210, %209
  %214 = call i64 @llvm.vector.reduce.or.v4i64(<4 x i64> %bin.rdx254)
  %cmp.n = icmp eq i64 %_19.i.i, %n.vec
  br i1 %cmp.n, label %bb14.i.i.i.i.i, label %bb8.i5.i.i.i.i.preheader

bb8.i5.i.i.i.i.preheader:                         ; preds = %bb12.i.i.i.i.i.split.split, %middle.block253
  %.off049.ph = phi i1 [ %failure.sroa.0.0.off0, %bb12.i.i.i.i.i.split.split ], [ %213, %middle.block253 ]
  %packed.sroa.0.05.i.i.i.i.i.ph = phi i64 [ 0, %bb12.i.i.i.i.i.split.split ], [ %214, %middle.block253 ]
  %iter.sroa.0.04.i.i.i.i.i.ph = phi i64 [ 0, %bb12.i.i.i.i.i.split.split ], [ %n.vec, %middle.block253 ]
  br label %bb8.i5.i.i.i.i

bb8.i5.i.i.i.i:                                   ; preds = %bb8.i5.i.i.i.i.preheader, %bb8.i5.i.i.i.i
  %.off049 = phi i1 [ %217, %bb8.i5.i.i.i.i ], [ %.off049.ph, %bb8.i5.i.i.i.i.preheader ]
  %packed.sroa.0.05.i.i.i.i.i = phi i64 [ %219, %bb8.i5.i.i.i.i ], [ %packed.sroa.0.05.i.i.i.i.i.ph, %bb8.i5.i.i.i.i.preheader ]
  %iter.sroa.0.04.i.i.i.i.i = phi i64 [ %218, %bb8.i5.i.i.i.i ], [ %iter.sroa.0.04.i.i.i.i.i.ph, %bb8.i5.i.i.i.i.preheader ]
  %_4.i.i.i.i.i.i = add nuw nsw i64 %iter.sroa.0.04.i.i.i.i.i, %120
  %_5.i.i.i = icmp ult i64 %_4.i.i.i.i.i.i, %_7.sroa.9.122.i.i
  call void @llvm.assume(i1 %_5.i.i.i), !noalias !500
  %_4.i.i.i = getelementptr inbounds nuw i64, ptr %_7.sroa.7.123.i.i, i64 %_4.i.i.i.i.i.i
  %_0.sroa.0.0.i.i = load i64, ptr %_4.i.i.i, align 8, !noalias !489, !noundef !8
  %_5.i.i6.i = icmp ult i64 %_4.i.i.i.i.i.i, %_26.sroa.16.0
  call void @llvm.assume(i1 %_5.i.i6.i), !noalias !500
  %_4.i.i7.i = getelementptr inbounds nuw i64, ptr %_26.sroa.14.0, i64 %_4.i.i.i.i.i.i
  %_0.sroa.0.0.i9.i = load i64, ptr %_4.i.i7.i, align 8, !noalias !494, !noundef !8
  %215 = icmp eq i64 %_0.sroa.0.0.i.i, 9223372036854775807
  %216 = icmp eq i64 %_0.sroa.0.0.i9.i, 9223372036854775807
  %_6.sroa.0.0.off0.i.i.i.i.i.i.i.i.i = or i1 %215, %216
  %_5.i.i.i.i.i.i.i.i.i = icmp slt i64 %_0.sroa.0.0.i.i, %_0.sroa.0.0.i9.i
  %217 = or i1 %.off049, %_6.sroa.0.0.off0.i.i.i.i.i.i.i.i.i
  %218 = add nuw nsw i64 %iter.sroa.0.04.i.i.i.i.i, 1
  %_13.i.i.i.i.i = zext i1 %_5.i.i.i.i.i.i.i.i.i to i64
  %_12.i.i.i.i.i = shl nuw i64 %_13.i.i.i.i.i, %iter.sroa.0.04.i.i.i.i.i
  %219 = or i64 %_12.i.i.i.i.i, %packed.sroa.0.05.i.i.i.i.i
  %exitcond.not.i.i.i.i.i = icmp eq i64 %218, %_19.i.i
  br i1 %exitcond.not.i.i.i.i.i, label %bb14.i.i.i.i.i, label %bb8.i5.i.i.i.i, !llvm.loop !505

bb14.i.i.i.i.i.loopexit:                          ; preds = %bb8.i5.i.i.i.i.us.us, %middle.block333
  %.lcssa = phi i64 [ %135, %middle.block333 ], [ %137, %bb8.i5.i.i.i.i.us.us ]
  %220 = icmp eq i64 %_0.sroa.0.0.i9.i.us.us, 9223372036854775807
  %_6.sroa.0.0.off0.i.i.i.i.i.i.i.i.i.us.us = or i1 %125, %220
  %221 = or i1 %failure.sroa.0.0.off0, %_6.sroa.0.0.off0.i.i.i.i.i.i.i.i.i.us.us
  br label %bb14.i.i.i.i.i

bb14.i.i.i.i.i:                                   ; preds = %bb8.i5.i.i.i.i, %bb8.i5.i.i.i.i.us133, %bb8.i5.i.i.i.i.us, %middle.block253, %middle.block279, %middle.block309, %bb14.i.i.i.i.i.loopexit
  %.us-phi131 = phi i1 [ %221, %bb14.i.i.i.i.i.loopexit ], [ %160, %bb8.i5.i.i.i.i.us ], [ %186, %bb8.i5.i.i.i.i.us133 ], [ %156, %middle.block309 ], [ %182, %middle.block279 ], [ %213, %middle.block253 ], [ %217, %bb8.i5.i.i.i.i ]
  %.us-phi132 = phi i64 [ %.lcssa, %bb14.i.i.i.i.i.loopexit ], [ %162, %bb8.i5.i.i.i.i.us ], [ %188, %bb8.i5.i.i.i.i.us133 ], [ %157, %middle.block309 ], [ %183, %middle.block279 ], [ %214, %middle.block253 ], [ %219, %bb8.i5.i.i.i.i ]
  store i64 %.us-phi132, ptr %_52.i.i.i.i.i, align 8, !alias.scope !506, !noalias !509
  br label %.noexc12

.noexc12:                                         ; preds = %bb14.i.i.i.i.i, %bb5.i.i.i.i.i
  %failure.sroa.0.1.off0 = phi i1 [ %.us-phi131, %bb14.i.i.i.i.i ], [ %failure.sroa.0.0.off0, %bb5.i.i.i.i.i ]
  %222 = getelementptr inbounds nuw i8, ptr %_13.i.i44, i64 16
  call void @llvm.lifetime.start.p0(ptr nonnull %_13.i.i44), !noalias !511
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 16 dereferenceable(16) %222, ptr noundef nonnull align 8 dereferenceable(16) %buffer.i.i.sroa.0.sroa.0, i64 16, i1 false)
  %bytes.i.i.sroa.5.0..sroa_idx = getelementptr inbounds nuw i8, ptr %_13.i.i44, i64 40
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %bytes.i.i.sroa.5.0..sroa_idx, ptr noundef nonnull align 8 dereferenceable(16) %buffer.i.i.sroa.0.sroa.5, i64 16, i1 false)
  %_256.i.i = lshr i64 %5, 3
  %_26.i.i = and i64 %5, 7
  %_27.not.i.i = icmp ne i64 %_26.i.i, 0
  %223 = zext i1 %_27.not.i.i to i64
  %_14.sroa.0.0.i.i = add nuw nsw i64 %_256.i.i, %223
  %spec.store.select.i.i = call i64 @llvm.umin.i64(i64 %_14.sroa.0.0.i.i, i64 %_39.0.i.i)
  call void @llvm.lifetime.end.p0(ptr nonnull %buffer.i.i.sroa.0.sroa.0)
  call void @llvm.lifetime.end.p0(ptr nonnull %buffer.i.i.sroa.0.sroa.5)
  call void @llvm.lifetime.start.p0(ptr nonnull %_7.i), !noalias !518
  call void @llvm.experimental.noalias.scope.decl(metadata !519)
  call void @llvm.experimental.noalias.scope.decl(metadata !520)
  store <2 x i64> splat (i64 1), ptr %_13.i.i44, align 16, !noalias !511
  %bytes.i.i.sroa.4.0..sroa_idx = getelementptr inbounds nuw i8, ptr %_13.i.i44, i64 32
  store ptr %_24.i.i, ptr %bytes.i.i.sroa.4.0..sroa_idx, align 16, !noalias !521
; call __rustc::__rust_no_alloc_shim_is_unstable_v2
  call void @_RNvCs6rREvFdRhLb_7___rustc35___rust_no_alloc_shim_is_unstable_v2() #17, !noalias !522
  %_3.i.i.i = call noalias noundef align 8 dereferenceable_or_null(56) ptr @mi_malloc_aligned(i64 noundef 56, i64 noundef 8) #17, !noalias !522
  %224 = icmp eq ptr %_3.i.i.i, null
  br i1 %224, label %bb2.i2.i.i, label %.noexc13, !prof !151

bb2.i2.i.i:                                       ; preds = %.noexc12
; invoke alloc::alloc::handle_alloc_error
  invoke void @_RNvNtCs6KVRSXc8uZF_5alloc5alloc18handle_alloc_error(i64 noundef 8, i64 noundef 56) #18
          to label %.noexc.i.i unwind label %cleanup.i.i.i, !noalias !511

.noexc.i.i:                                       ; preds = %bb2.i2.i.i
  unreachable

cleanup.i.i.i:                                    ; preds = %bb2.i2.i.i
  %225 = landingpad { ptr, i32 }
          cleanup
; invoke core::ptr::drop_glue::<alloc::sync::ArcInner<vortex_buffer::allocation::BufferBacking>>
  invoke fastcc void @_RINvNtCslWxY2MhVcag_4core3ptr9drop_glueINtNtCs6KVRSXc8uZF_5alloc4sync8ArcInnerNtNtCs4EGy4ysETW6_13vortex_buffer10allocation13BufferBackingEECs4CRfLIao8WA_17row_fn_bool_probe.llvm.362247623077402552(ptr noalias nofree noundef nonnull align 8 dereferenceable(56) %_13.i.i44) #15
          to label %bb29 unwind label %terminate.i.i.i, !noalias !511

terminate.i.i.i:                                  ; preds = %cleanup.i.i.i
  %226 = landingpad { ptr, i32 }
          filter [0 x ptr] zeroinitializer
; call core::panicking::panic_in_cleanup
  call void @_RNvNtCslWxY2MhVcag_4core9panicking16panic_in_cleanup() #16, !noalias !511
  unreachable

.noexc13:                                         ; preds = %.noexc12
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(56) %_3.i.i.i, ptr noundef nonnull align 16 dereferenceable(56) %_13.i.i44, i64 56, i1 false), !noalias !511
  call void @llvm.lifetime.end.p0(ptr nonnull %_13.i.i44), !noalias !511
  store ptr %_83.i.i, ptr %_7.i, align 8, !alias.scope !521, !noalias !525
  %227 = getelementptr inbounds nuw i8, ptr %_7.i, i64 8
  store i64 %spec.store.select.i.i, ptr %227, align 8, !alias.scope !521, !noalias !525
  %228 = getelementptr inbounds nuw i8, ptr %_7.i, i64 24
  store i8 3, ptr %228, align 8, !alias.scope !521, !noalias !525
  %229 = getelementptr inbounds nuw i8, ptr %_7.i, i64 16
  store ptr %_3.i.i.i, ptr %229, align 8, !alias.scope !521, !noalias !525
  call void @llvm.experimental.noalias.scope.decl(metadata !526)
  call void @llvm.experimental.noalias.scope.decl(metadata !529)
  call void @llvm.lifetime.start.p0(ptr nonnull %offset.i51)
  call void @llvm.lifetime.start.p0(ptr nonnull %len.i)
  store i64 %5, ptr %len.i, align 8, !noalias !531
  store i64 0, ptr %offset.i51, align 8, !noalias !531
  %_30.0.i = shl nuw i64 %spec.store.select.i.i, 3
  %_30.1.i = icmp samesign ult i64 %spec.store.select.i.i, 2305843009213693952
  %_4.not.i = icmp ugt i64 %5, %_30.0.i
  %or.cond.i = select i1 %_30.1.i, i1 %_4.not.i, i1 false, !prof !159
  br i1 %or.cond.i, label %bb2.i, label %bb12, !prof !159

bb2.i:                                            ; preds = %.noexc13
  call void @llvm.lifetime.start.p0(ptr nonnull %_10.i50), !noalias !531
  store i64 %spec.store.select.i.i, ptr %_10.i50, align 8, !noalias !531
  call void @llvm.lifetime.start.p0(ptr nonnull %args.i), !noalias !531
  store ptr %_10.i50, ptr %args.i, align 8, !noalias !531
  %_14.sroa.4.0..sroa_idx.i = getelementptr inbounds nuw i8, ptr %args.i, i64 8
  store ptr @_RNvXsi_NtNtNtCslWxY2MhVcag_4core3fmt3num3impjNtB9_7Display3fmt, ptr %_14.sroa.4.0..sroa_idx.i, align 8, !noalias !531
  %230 = getelementptr inbounds nuw i8, ptr %args.i, i64 16
  store ptr %offset.i51, ptr %230, align 8, !noalias !531
  %_15.sroa.4.0..sroa_idx.i = getelementptr inbounds nuw i8, ptr %args.i, i64 24
  store ptr @_RNvXsi_NtNtNtCslWxY2MhVcag_4core3fmt3num3impjNtB9_7Display3fmt, ptr %_15.sroa.4.0..sroa_idx.i, align 8, !noalias !531
  %231 = getelementptr inbounds nuw i8, ptr %args.i, i64 32
  store ptr %len.i, ptr %231, align 8, !noalias !531
  %_16.sroa.4.0..sroa_idx.i = getelementptr inbounds nuw i8, ptr %args.i, i64 40
  store ptr @_RNvXsi_NtNtNtCslWxY2MhVcag_4core3fmt3num3impjNtB9_7Display3fmt, ptr %_16.sroa.4.0..sroa_idx.i, align 8, !noalias !531
; invoke core::panicking::panic_fmt
  invoke void @_RNvNtCslWxY2MhVcag_4core9panicking9panic_fmt(ptr noundef nonnull @alloc_b3d41c32cfb279917b2adf8d6b28aeea, ptr noundef nonnull %args.i, ptr noalias nofree noundef readonly align 8 captures(address, read_provenance) dereferenceable(24) @alloc_aa1b3af58bcc7e8fd0ac15ecdd00b41e) #18
          to label %unreachable.i54 unwind label %bb2.i.i18.i, !noalias !531

unreachable.i54:                                  ; preds = %bb2.i
  unreachable

terminate.i53:                                    ; preds = %bb2.i.i.i.i20.i
  %232 = landingpad { ptr, i32 }
          filter [0 x ptr] zeroinitializer
; call core::panicking::panic_in_cleanup
  call void @_RNvNtCslWxY2MhVcag_4core9panicking16panic_in_cleanup() #16, !noalias !526
  unreachable

bb2.i.i18.i:                                      ; preds = %bb2.i
  %233 = landingpad { ptr, i32 }
          cleanup
  %_2.i.i.i.i19.i = atomicrmw sub ptr %_3.i.i.i, i64 1 release, align 8, !noalias !532
  %234 = icmp eq i64 %_2.i.i.i.i19.i, 1
  br i1 %234, label %bb2.i.i.i.i20.i, label %bb29

bb2.i.i.i.i20.i:                                  ; preds = %bb2.i.i18.i
  fence acquire
; invoke <alloc::sync::Arc<vortex_buffer::allocation::BufferBacking>>::drop_slow
  invoke void @_RNvMsn_NtCs6KVRSXc8uZF_5alloc4syncINtB5_3ArcNtNtCs4EGy4ysETW6_13vortex_buffer10allocation13BufferBackingE9drop_slowBK_(ptr noalias nofree noundef nonnull align 8 dereferenceable(8) %229) #14
          to label %bb29 unwind label %terminate.i53, !noalias !526

bb12:                                             ; preds = %.noexc13
  %buffer1.sroa.4.0._0.sroa_idx.i = getelementptr inbounds nuw i8, ptr %values, i64 8
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %buffer1.sroa.4.0._0.sroa_idx.i, ptr noundef nonnull align 8 dereferenceable(16) %227, i64 16, i1 false), !alias.scope !531
  store ptr %_83.i.i, ptr %values, align 8, !alias.scope !526, !noalias !529
  %buffer1.sroa.5.0._0.sroa_idx.i = getelementptr inbounds nuw i8, ptr %values, i64 24
  store i8 0, ptr %buffer1.sroa.5.0._0.sroa_idx.i, align 8, !alias.scope !526, !noalias !529
  %235 = getelementptr inbounds nuw i8, ptr %values, i64 32
  store i64 0, ptr %235, align 8, !alias.scope !526, !noalias !529
  %236 = getelementptr inbounds nuw i8, ptr %values, i64 40
  store i64 %5, ptr %236, align 8, !alias.scope !526, !noalias !529
  call void @llvm.lifetime.end.p0(ptr nonnull %offset.i51)
  call void @llvm.lifetime.end.p0(ptr nonnull %len.i)
  call void @llvm.lifetime.end.p0(ptr nonnull %_7.i), !noalias !518
  call void @llvm.lifetime.start.p0(ptr nonnull %_36)
  br i1 %failure.sroa.0.1.off0, label %bb1.i, label %bb16, !prof !151

bb1.i:                                            ; preds = %bb12
  call void @llvm.lifetime.start.p0(ptr nonnull %_6.i), !noalias !541
  store ptr @alloc_78c2b751c4d3e2d7197d387cac5cfac2, ptr %_6.i, align 8, !noalias !541
  %237 = getelementptr inbounds nuw i8, ptr %_6.i, i64 8
  store ptr inttoptr (i64 55 to ptr), ptr %237, align 8, !noalias !541
; invoke vortex_error::__private::fmt_err
  invoke void @_RNvNtCslnFpLG3TNtn_12vortex_error9___private7fmt_err(ptr noalias nofree noundef nonnull sret([80 x i8]) align 8 captures(address) dereferenceable(80) %_36, ptr noundef nonnull @_RNcNtNtCslnFpLG3TNtn_12vortex_error11VortexError5Other0, ptr noalias nofree noundef nonnull readonly align 8 captures(address, read_provenance) dereferenceable(16) %_6.i) #14
          to label %bb14 unwind label %bb35

cleanup4:                                         ; preds = %bb16
  %238 = landingpad { ptr, i32 }
          cleanup
  br label %bb29

bb14:                                             ; preds = %bb1.i
  call void @llvm.lifetime.end.p0(ptr nonnull %_6.i), !noalias !541
  %.pr = load i64, ptr %_36, align 8
  %.not8 = icmp eq i64 %.pr, -1
  br i1 %.not8, label %bb16, label %bb15

bb15:                                             ; preds = %bb14
  %239 = getelementptr inbounds nuw i8, ptr %_0, i64 8
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(80) %239, ptr noundef nonnull align 8 dereferenceable(80) %_36, i64 80, i1 false)
  store i64 0, ptr %_0, align 8
  call void @llvm.lifetime.end.p0(ptr nonnull %_36)
  call void @llvm.experimental.noalias.scope.decl(metadata !544)
  call void @llvm.experimental.noalias.scope.decl(metadata !547)
  %240 = getelementptr inbounds nuw i8, ptr %values, i64 16
  call void @llvm.experimental.noalias.scope.decl(metadata !550)
  %241 = load ptr, ptr %240, align 8, !alias.scope !553, !noundef !8
  %242 = icmp eq ptr %241, null
  br i1 %242, label %bb19, label %bb2.i.i.i

bb2.i.i.i:                                        ; preds = %bb15
  %_2.i.i.i.i.i = atomicrmw sub ptr %241, i64 1 release, align 8, !noalias !554
  %243 = icmp eq i64 %_2.i.i.i.i.i, 1
  br i1 %243, label %bb2.i.i.i.i.i, label %bb19

bb2.i.i.i.i.i:                                    ; preds = %bb2.i.i.i
  fence acquire
; invoke <alloc::sync::Arc<vortex_buffer::allocation::BufferBacking>>::drop_slow
  invoke void @_RNvMsn_NtCs6KVRSXc8uZF_5alloc4syncINtB5_3ArcNtNtCs4EGy4ysETW6_13vortex_buffer10allocation13BufferBackingE9drop_slowBK_(ptr noalias nofree noundef nonnull align 8 dereferenceable(8) %240) #14
          to label %bb19 unwind label %cleanup3

bb16:                                             ; preds = %bb14, %bb12
  call void @llvm.lifetime.start.p0(ptr nonnull %_44)
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(48) %_44, ptr noundef nonnull align 8 dereferenceable(48) %values, i64 48, i1 false)
  call void @llvm.lifetime.start.p0(ptr nonnull %_45)
  store i64 0, ptr %_45, align 8
; invoke <vortex_array::array::typed::Array<vortex_array::arrays::bool::vtable::Bool>>::new
  %244 = invoke { ptr, ptr } @_RNvMs1_NtNtNtCsc1BqfENIlVH_12vortex_array6arrays4bool5arrayINtNtNtBb_5array5typed5ArrayNtNtB7_6vtable4BoolE3new(ptr noalias nofree noundef nonnull align 8 captures(address) dereferenceable(48) %_44, ptr noalias nofree noundef nonnull align 8 captures(address) dereferenceable(24) %_45)
          to label %bb18 unwind label %cleanup4

bb18:                                             ; preds = %bb16
  call void @llvm.lifetime.end.p0(ptr nonnull %_45)
  call void @llvm.lifetime.end.p0(ptr nonnull %_44)
  %_42.0 = extractvalue { ptr, ptr } %244, 0
  %_42.1 = extractvalue { ptr, ptr } %244, 1
  %_41.sroa.4.0..sroa_idx = getelementptr inbounds nuw i8, ptr %_0, i64 16
  store ptr %_42.0, ptr %_41.sroa.4.0..sroa_idx, align 8
  %_41.sroa.5.0..sroa_idx = getelementptr inbounds nuw i8, ptr %_0, i64 24
  store ptr %_42.1, ptr %_41.sroa.5.0..sroa_idx, align 8
  store <2 x i64> <i64 0, i64 -1>, ptr %_0, align 8
  call void @llvm.lifetime.end.p0(ptr nonnull %_36)
  br label %bb19

bb19:                                             ; preds = %bb18, %bb2.i.i.i.i.i, %bb2.i.i.i, %bb15
  call void @llvm.lifetime.end.p0(ptr nonnull %values)
  call void @llvm.lifetime.end.p0(ptr nonnull %row_count)
  call void @llvm.experimental.noalias.scope.decl(metadata !559)
  call void @llvm.experimental.noalias.scope.decl(metadata !562)
  call void @llvm.experimental.noalias.scope.decl(metadata !565)
  %245 = load ptr, ptr %columns, align 8, !alias.scope !568, !noundef !8
  %.not.i.i.i = icmp eq ptr %245, null
  br i1 %.not.i.i.i, label %bb4.i, label %bb2.i.i.i17

bb2.i.i.i17:                                      ; preds = %bb19
  call void @llvm.experimental.noalias.scope.decl(metadata !569)
  %246 = getelementptr inbounds nuw i8, ptr %columns, i64 16
  call void @llvm.experimental.noalias.scope.decl(metadata !572)
  %247 = load ptr, ptr %246, align 8, !alias.scope !575, !noundef !8
  %248 = icmp eq ptr %247, null
  br i1 %248, label %bb4.i, label %bb2.i.i.i.i.i18

bb2.i.i.i.i.i18:                                  ; preds = %bb2.i.i.i17
  %_2.i.i.i.i.i.i.i = atomicrmw sub ptr %247, i64 1 release, align 8, !noalias !576
  %249 = icmp eq i64 %_2.i.i.i.i.i.i.i, 1
  br i1 %249, label %bb2.i.i.i.i.i.i.i, label %bb4.i

bb2.i.i.i.i.i.i.i:                                ; preds = %bb2.i.i.i.i.i18
  fence acquire
; invoke <alloc::sync::Arc<vortex_buffer::allocation::BufferBacking>>::drop_slow
  invoke void @_RNvMsn_NtCs6KVRSXc8uZF_5alloc4syncINtB5_3ArcNtNtCs4EGy4ysETW6_13vortex_buffer10allocation13BufferBackingE9drop_slowBK_(ptr noalias nofree noundef nonnull align 8 dereferenceable(8) %246) #14
          to label %bb4.i unwind label %cleanup.i

cleanup.i:                                        ; preds = %bb2.i.i.i.i.i.i.i
  %250 = landingpad { ptr, i32 }
          cleanup
; invoke core::ptr::drop_glue::<vortex_array::scalar_fn::unstable::row::types::element::tuple::element_tuple::ArgColumn<i64>>
  invoke fastcc void @_RINvNtCslWxY2MhVcag_4core3ptr9drop_glueINtNtNtNtNtNtNtNtCsc1BqfENIlVH_12vortex_array9scalar_fn8unstable3row5types7element5tuple13element_tuple9ArgColumnxEECs4CRfLIao8WA_17row_fn_bool_probe(ptr noalias nofree noundef align 8 dereferenceable(32) %_13.i.i6) #15
          to label %common.resume unwind label %terminate.i

bb4.i:                                            ; preds = %bb2.i.i.i.i.i.i.i, %bb2.i.i.i.i.i18, %bb2.i.i.i17, %bb19
  call void @llvm.experimental.noalias.scope.decl(metadata !581)
  call void @llvm.experimental.noalias.scope.decl(metadata !584)
  %251 = load ptr, ptr %_13.i.i6, align 8, !alias.scope !587, !noundef !8
  %.not.i.i1.i = icmp eq ptr %251, null
  br i1 %.not.i.i1.i, label %bb26, label %bb2.i.i2.i

bb2.i.i2.i:                                       ; preds = %bb4.i
  call void @llvm.experimental.noalias.scope.decl(metadata !588)
  %252 = getelementptr inbounds nuw i8, ptr %columns, i64 48
  call void @llvm.experimental.noalias.scope.decl(metadata !591)
  %253 = load ptr, ptr %252, align 8, !alias.scope !594, !noundef !8
  %254 = icmp eq ptr %253, null
  br i1 %254, label %bb26, label %bb2.i.i.i.i3.i

bb2.i.i.i.i3.i:                                   ; preds = %bb2.i.i2.i
  %_2.i.i.i.i.i.i4.i = atomicrmw sub ptr %253, i64 1 release, align 8, !noalias !595
  %255 = icmp eq i64 %_2.i.i.i.i.i.i4.i, 1
  br i1 %255, label %bb2.i.i.i.i.i.i5.i, label %bb26

bb2.i.i.i.i.i.i5.i:                               ; preds = %bb2.i.i.i.i3.i
  fence acquire
; call <alloc::sync::Arc<vortex_buffer::allocation::BufferBacking>>::drop_slow
  call void @_RNvMsn_NtCs6KVRSXc8uZF_5alloc4syncINtB5_3ArcNtNtCs4EGy4ysETW6_13vortex_buffer10allocation13BufferBackingE9drop_slowBK_(ptr noalias nofree noundef nonnull align 8 dereferenceable(8) %252) #14
  br label %bb26

terminate.i:                                      ; preds = %cleanup.i
  %256 = landingpad { ptr, i32 }
          filter [0 x ptr] zeroinitializer
; call core::panicking::panic_in_cleanup
  call void @_RNvNtCslWxY2MhVcag_4core9panicking16panic_in_cleanup() #16
  unreachable

bb26:                                             ; preds = %bb6, %bb2.i.i.i.i.i.i5.i, %bb2.i.i.i.i3.i, %bb2.i.i2.i, %bb4.i, %bb43
  call void @llvm.lifetime.end.p0(ptr nonnull %columns)
  ret void

bb35:                                             ; preds = %bb1.i
  %257 = landingpad { ptr, i32 }
          cleanup
  call void @llvm.experimental.noalias.scope.decl(metadata !600)
  call void @llvm.experimental.noalias.scope.decl(metadata !603)
  %258 = getelementptr inbounds nuw i8, ptr %values, i64 16
  call void @llvm.experimental.noalias.scope.decl(metadata !606)
  %259 = load ptr, ptr %258, align 8, !alias.scope !609, !noundef !8
  %260 = icmp eq ptr %259, null
  br i1 %260, label %bb29, label %bb2.i.i.i20

bb2.i.i.i20:                                      ; preds = %bb35
  %_2.i.i.i.i.i21 = atomicrmw sub ptr %259, i64 1 release, align 8, !noalias !610
  %261 = icmp eq i64 %_2.i.i.i.i.i21, 1
  br i1 %261, label %bb2.i.i.i.i.i22, label %bb29

bb2.i.i.i.i.i22:                                  ; preds = %bb2.i.i.i20
  fence acquire
; invoke <alloc::sync::Arc<vortex_buffer::allocation::BufferBacking>>::drop_slow
  invoke void @_RNvMsn_NtCs6KVRSXc8uZF_5alloc4syncINtB5_3ArcNtNtCs4EGy4ysETW6_13vortex_buffer10allocation13BufferBackingE9drop_slowBK_(ptr noalias nofree noundef nonnull align 8 dereferenceable(8) %258) #14
          to label %bb29 unwind label %terminate

terminate:                                        ; preds = %bb29, %bb2.i.i.i.i.i22
  %262 = landingpad { ptr, i32 }
          filter [0 x ptr] zeroinitializer
; call core::panicking::panic_in_cleanup
  call void @_RNvNtCslWxY2MhVcag_4core9panicking16panic_in_cleanup() #16
  unreachable

bb6:                                              ; preds = %bb32
  %263 = getelementptr inbounds nuw i8, ptr %_0, i64 8
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(80) %263, ptr noundef nonnull align 8 dereferenceable(80) %_17, i64 80, i1 false)
  store i64 1, ptr %_0, align 8
  call void @llvm.lifetime.end.p0(ptr nonnull %_17)
  call void @llvm.lifetime.end.p0(ptr nonnull %args)
  call void @llvm.lifetime.end.p0(ptr nonnull %_20)
  call void @llvm.lifetime.end.p0(ptr nonnull %row_count)
; call core::ptr::drop_glue::<(vortex_array::scalar_fn::unstable::row::types::element::tuple::element_tuple::ArgColumn<i64>, vortex_array::scalar_fn::unstable::row::types::element::tuple::element_tuple::ArgColumn<i64>)>
  call fastcc void @_RINvNtCslWxY2MhVcag_4core3ptr9drop_glueTINtNtNtNtNtNtNtNtCsc1BqfENIlVH_12vortex_array9scalar_fn8unstable3row5types7element5tuple13element_tuple9ArgColumnxEBC_EECs4CRfLIao8WA_17row_fn_bool_probe(ptr noalias nofree noundef align 8 dereferenceable(64) %columns)
  br label %bb26

bb29:                                             ; preds = %bb2.i.i18.i, %bb2.i.i.i.i20.i, %cleanup.i.i.i, %cleanup3, %bb2.i.i.i.i.i22, %bb2.i.i.i20, %bb35, %cleanup4, %cleanup2
  %.pn = phi { ptr, i32 } [ %257, %bb2.i.i.i20 ], [ %257, %bb35 ], [ %238, %cleanup4 ], [ %6, %cleanup2 ], [ %257, %bb2.i.i.i.i.i22 ], [ %233, %bb2.i.i18.i ], [ %225, %cleanup.i.i.i ], [ %16, %cleanup3 ], [ %233, %bb2.i.i.i.i20.i ]
; invoke core::ptr::drop_glue::<(vortex_array::scalar_fn::unstable::row::types::element::tuple::element_tuple::ArgColumn<i64>, vortex_array::scalar_fn::unstable::row::types::element::tuple::element_tuple::ArgColumn<i64>)>
  invoke fastcc void @_RINvNtCslWxY2MhVcag_4core3ptr9drop_glueTINtNtNtNtNtNtNtNtCsc1BqfENIlVH_12vortex_array9scalar_fn8unstable3row5types7element5tuple13element_tuple9ArgColumnxEBC_EECs4CRfLIao8WA_17row_fn_bool_probe(ptr noalias nofree noundef align 8 dereferenceable(64) %columns) #15
          to label %common.resume unwind label %terminate
}
