# Vortex Public API Audit — Items to Remove or Restrict

> Generated 2026-02-05 on branch `ji/remove-pub`
>
> **Legend:**
> - **REMOVE** — No callers anywhere in the workspace (dead code). Safe to delete entirely.
> - **`pub(crate)`** — Only used within the defining crate. Downgrade visibility.
> - **KEEP** — Intentional, do not change.

---

## Table of Contents

1. [vortex-buffer](#1-vortex-buffer)
2. [vortex-error](#2-vortex-error)
3. [vortex-mask](#3-vortex-mask)
4. [vortex-utils](#4-vortex-utils)
5. [vortex-dtype](#5-vortex-dtype)
6. [vortex-scalar](#6-vortex-scalar)
7. [vortex-vector](#7-vortex-vector)
8. [vortex-array](#8-vortex-array)
9. [vortex-compute](#9-vortex-compute)
10. [vortex-io](#10-vortex-io)
11. [vortex-session](#11-vortex-session)
12. [vortex-layout](#12-vortex-layout)
13. [vortex-scan](#13-vortex-scan)
14. [vortex-file](#14-vortex-file)
15. [vortex-ipc](#15-vortex-ipc)
16. [vortex-datafusion](#16-vortex-datafusion)
17. [vortex-bench](#17-vortex-bench)
18. [vortex-duckdb](#18-vortex-duckdb)
19. [vortex-ffi](#19-vortex-ffi)
20. [vortex-metrics](#20-vortex-metrics)
21. [vortex-btrblocks](#21-vortex-btrblocks)
22. [Encoding Crates](#22-encoding-crates)
23. [`#[allow(dead_code)]` Annotations](#23-allowdead_code-annotations)

---

## 1. vortex-buffer

### REMOVE (13 items)

| Item | File | Line |
|------|------|------|
| `Buffer::from_byte_buffer_aligned` | `vortex-buffer/src/buffer.rs` | 160 |
| `Buffer::from_bytes_aligned` | `vortex-buffer/src/buffer.rs` | 170 |
| `Buffer::slice_with_alignment` | `vortex-buffer/src/buffer.rs` | 284 |
| `Buffer::slice_ref_with_alignment` | `vortex-buffer/src/buffer.rs` | 361 |
| `BufferMut::empty_aligned` | `vortex-buffer/src/buffer_mut.rs` | 85 |
| `BitView::try_iter_ones` | `vortex-buffer/src/bit/view.rs` | 184 |
| `BitView::iter_zeros` | `vortex-buffer/src/bit/view.rs` | 212 |
| `BitView::iter_slices` | `vortex-buffer/src/bit/view.rs` | 239 |
| `BitView::as_raw` | `vortex-buffer/src/bit/view.rs` | 323 |
| `BitView::from_slice` | `vortex-buffer/src/bit/view.rs` | 108 |
| `BitView::new_owned` | `vortex-buffer/src/bit/view.rs` | 80 |
| `Buffer::iter_bit_views` | `vortex-buffer/src/bit/view.rs` | 350 |
| `BitBuffer::iter_bit_views` | `vortex-buffer/src/bit/view.rs` | 372 |

### `pub(crate)` (3 items)

| Item | File | Line | Notes |
|------|------|------|-------|
| `BitView::N_WORDS` | `vortex-buffer/src/bit/view.rs` | 60 | Only used internally |
| `BitBuffer::unaligned_chunks` | `vortex-buffer/src/bit/buf.rs` | 286 | Only called by `true_count`/`false_count` |
| `BitBufferMut::unaligned_chunks` | `vortex-buffer/src/bit/buf_mut.rs` | 590 | Only used internally |

---

## 2. vortex-error

No changes needed. All public items are used externally.

---

## 3. vortex-mask

### REMOVE (3 items)

| Item | File | Line |
|------|------|------|
| `Mask::from_intersection_indices` | `vortex-mask/src/lib.rs` | 297 |
| `Mask::concat` | `vortex-mask/src/lib.rs` | 607 |
| `AllOr::cloned` | `vortex-mask/src/lib.rs` | 63 |

### `pub(crate)` (1 item)

| Item | File | Line | Notes |
|------|------|------|-------|
| `MaskMut::set_to_unchecked` | `vortex-mask/src/mask_mut.rs` | 442 | Only called by `set_unchecked`/`unset_unchecked` |

---

## 4. vortex-utils

No changes needed. All items are used externally. The `#[allow(dead_code)]` on
`EnvVarGuard::lock_guard` at `vortex-utils/src/env.rs:62` is intentional (RAII guard pattern).

---

## 5. vortex-dtype

### REMOVE (20 items)

| Item | File | Line | Notes |
|------|------|------|-------|
| `FieldName::inner` | `vortex-dtype/src/field.rs` | (varies) | No external callers |
| `Field::is_named` | `vortex-dtype/src/field.rs` | (varies) | No external callers |
| `TimeUnit::to_jiff_span` | `vortex-dtype/src/dtype/temporal.rs` | (varies) | No external callers |
| `DType::eq_with_nullability_subset` | `vortex-dtype/src/dtype/mod.rs` | (varies) | No external callers |
| `DType::is_decimal` | `vortex-dtype/src/dtype/mod.rs` | (varies) | No external callers |
| `DType::is_list` | `vortex-dtype/src/dtype/mod.rs` | (varies) | No external callers |
| `DType::is_fixed_size_list` | `vortex-dtype/src/dtype/mod.rs` | (varies) | No external callers |
| `DType::into_decimal_opt` | `vortex-dtype/src/dtype/mod.rs` | (varies) | No external callers |
| `DType::into_list_opt` | `vortex-dtype/src/dtype/mod.rs` | (varies) | No external callers |
| `DType::into_fixed_size_list_opt` | `vortex-dtype/src/dtype/mod.rs` | (varies) | No external callers |
| `DType::into_extension_opt` | `vortex-dtype/src/dtype/mod.rs` | (varies) | No external callers |
| `DType::BYTES` | `vortex-dtype/src/dtype/mod.rs` | (varies) | No external callers |
| `PType::max_signed_ptype` | `vortex-dtype/src/ptype.rs` | (varies) | No external callers |
| `PType::min_signed_ptype_for_value` | `vortex-dtype/src/ptype.rs` | (varies) | No external callers |
| `i256::MIN` | `vortex-dtype/src/i256.rs` | (varies) | No external callers |
| `i256::maybe_i128` | `vortex-dtype/src/i256.rs` | (varies) | No external callers |
| `i256::to_parts` | `vortex-dtype/src/i256.rs` | (varies) | No external callers |
| `i256::to_be_bytes` | `vortex-dtype/src/i256.rs` | (varies) | No external callers |
| `DecimalDType::required_bit_width` | `vortex-dtype/src/dtype/decimal.rs` | (varies) | No external callers |
| `MAX_SCALE` | `vortex-dtype/src/dtype/decimal.rs` | (varies) | No external callers |
| `PrecisionScale::is_valid` | `vortex-dtype/src/dtype/decimal.rs` | (varies) | No external callers |

### `pub(crate)` (9 items)

| Item | File | Line | Notes |
|------|------|------|-------|
| `Field::as_name` | `vortex-dtype/src/field.rs` | (varies) | Only used within vortex-dtype |
| `FieldPath::resolve` | `vortex-dtype/src/field.rs` | (varies) | Only used within vortex-dtype |
| `FieldPath::exists_in` | `vortex-dtype/src/field.rs` | (varies) | Only used within vortex-dtype |
| `FieldMask::matches_root` | `vortex-dtype/src/field.rs` | (varies) | Only used within vortex-dtype |
| `StructFields::from_fields` | `vortex-dtype/src/dtype/struct_.rs` | (varies) | Only used within vortex-dtype |
| `FieldDType::value` | `vortex-dtype/src/dtype/struct_.rs` | (varies) | Only used within vortex-dtype |
| `i256::into_parts` | `vortex-dtype/src/i256.rs` | (varies) | Only used within vortex-dtype |
| `i256::wrapping_add` | `vortex-dtype/src/i256.rs` | (varies) | Only used within vortex-dtype |
| `i256::ONE` | `vortex-dtype/src/i256.rs` | (varies) | Only used within vortex-dtype |

---

## 6. vortex-scalar

### REMOVE (4 items)

| Item | File | Line | Notes |
|------|------|------|-------|
| `BoolScalar::into_scalar` | `vortex-scalar/src/bool.rs` | (varies) | No external callers |
| `Scalar::as_utf8_opt` | `vortex-scalar/src/lib.rs` | (varies) | No external callers |
| `Scalar::as_struct_opt` | `vortex-scalar/src/lib.rs` | (varies) | No external callers |
| `Scalar::as_extension_opt` | `vortex-scalar/src/lib.rs` | (varies) | No external callers |

### `pub(crate)` (7 items)

| Item | File | Line | Notes |
|------|------|------|-------|
| `BinaryScalar::from_scalar_value` | `vortex-scalar/src/binary.rs` | (varies) | Only used within vortex-scalar |
| `BinaryScalar::is_empty` | `vortex-scalar/src/binary.rs` | (varies) | Only used within vortex-scalar |
| `Scalar::as_binary_opt` | `vortex-scalar/src/lib.rs` | (varies) | Only used within vortex-scalar |
| `Utf8Scalar::from_scalar_value` | `vortex-scalar/src/utf8.rs` | (varies) | Only used within vortex-scalar |
| `Utf8Scalar::is_empty` | `vortex-scalar/src/utf8.rs` | (varies) | Only used within vortex-scalar |
| `PValue::is_instance_of` | `vortex-scalar/src/pvalue.rs` | (varies) | Only used within vortex-scalar |
| `PValue::cast_opt` | `vortex-scalar/src/pvalue.rs` | (varies) | Only used within vortex-scalar |

---

## 7. vortex-vector

### REMOVE (24 items)

| Item | File | Line | Notes |
|------|------|------|-------|
| `datum_matches_dtype` | `vortex-vector/src/datum.rs` | (varies) | No external callers |
| `scalar_matches_dtype` | `vortex-vector/src/datum.rs` | (varies) | No external callers |
| `Datum::unwrap_into_vector` | `vortex-vector/src/datum.rs` | (varies) | No external callers |
| `Datum::as_scalar` | `vortex-vector/src/datum.rs` | (varies) | No external callers |
| `Datum::as_vector` | `vortex-vector/src/datum.rs` | (varies) | No external callers |
| `Datum::is_nested` | `vortex-vector/src/datum.rs` | (varies) | No external callers |
| `Vector::into_null_opt` | `vortex-vector/src/vector.rs` | (varies) | No external callers |
| `Vector::into_bool_opt` | `vortex-vector/src/vector.rs` | (varies) | No external callers |
| `Vector::into_primitive_opt` | `vortex-vector/src/vector.rs` | (varies) | No external callers |
| `Vector::into_decimal_opt` | `vortex-vector/src/vector.rs` | (varies) | No external callers |
| `Vector::into_string_opt` | `vortex-vector/src/vector.rs` | (varies) | No external callers |
| `Vector::into_binary_opt` | `vortex-vector/src/vector.rs` | (varies) | No external callers |
| `Vector::into_list_opt` | `vortex-vector/src/vector.rs` | (varies) | No external callers |
| `Vector::into_fixed_size_list_opt` | `vortex-vector/src/vector.rs` | (varies) | No external callers |
| `Vector::into_struct_opt` | `vortex-vector/src/vector.rs` | (varies) | No external callers |
| `VectorMut::as_null_mut` | `vortex-vector/src/vector_mut.rs` | (varies) | No external callers (entire `as_*_mut` family) |
| `VectorMut::as_bool_mut` | `vortex-vector/src/vector_mut.rs` | (varies) | |
| `VectorMut::as_primitive_mut` | `vortex-vector/src/vector_mut.rs` | (varies) | |
| `VectorMut::as_decimal_mut` | `vortex-vector/src/vector_mut.rs` | (varies) | |
| `VectorMut::as_string_mut` | `vortex-vector/src/vector_mut.rs` | (varies) | |
| `VectorMut::as_binary_mut` | `vortex-vector/src/vector_mut.rs` | (varies) | |
| `VectorMut::as_list_mut` | `vortex-vector/src/vector_mut.rs` | (varies) | |
| `VectorMut::as_fixed_size_list_mut` | `vortex-vector/src/vector_mut.rs` | (varies) | |
| `GenericPrimitiveVec::into_nonnull_buffer` | `vortex-vector/src/primitive.rs` | (varies) | No external callers |

### `pub(crate)` (8 items)

| Item | File | Line | Notes |
|------|------|------|-------|
| `PrimitiveVectorMut::upcast` | `vortex-vector/src/primitive.rs` | (varies) | Only used within vortex-vector |
| `PrimitiveVectorMut::extend_from_vector_with_upcast` | `vortex-vector/src/primitive.rs` | (varies) | Only used within vortex-vector |
| `PrimitiveVectorMut::resize` | `vortex-vector/src/primitive.rs` | (varies) | Only used within vortex-vector |
| `PrimitiveVectorMut::spare_capacity_mut` | `vortex-vector/src/primitive.rs` | (varies) | Only used within vortex-vector |
| `StructVectorMut::minimum_capacity` | `vortex-vector/src/struct_.rs` | (varies) | Only used within vortex-vector |
| `BinaryView::as_view_mut` | `vortex-vector/src/binary_view.rs` | (varies) | Only used within vortex-vector |
| `BinaryViewVectorMut::append_owned_values` | `vortex-vector/src/binary_view.rs` | (varies) | Only used within vortex-vector |
| `PrimitiveDatum::ptype` | `vortex-vector/src/primitive.rs` | (varies) | Only used within vortex-vector |

---

## 8. vortex-array

### REMOVE (~17 items — highest priority dead code)

| Item | File | Line | Notes |
|------|------|------|-------|
| `add_scalar` | `vortex-array/src/compute/numeric.rs` | 52 | Zero callers anywhere |
| `mul_scalar` | `vortex-array/src/compute/numeric.rs` | 80 | Zero callers anywhere |
| `div_scalar` | `vortex-array/src/compute/numeric.rs` | 94 | Zero callers anywhere |
| `with_column` | `vortex-array/src/arrays/struct_/array.rs` | 464 | Zero callers anywhere |
| `to_offsets_index` | `vortex-array/src/search_sorted.rs` | 66 | Zero callers anywhere |
| `deserialize_expr_proto` | `vortex-array/src/expr/proto.rs` | 62 | Already `#[deprecated]`, zero callers |
| `apply_to_primitive_vector` | `vortex-array/src/patches.rs` | 854 | Zero callers (only calls `apply_to_pvector` below) |
| `apply_to_pvector` | `vortex-array/src/patches.rs` | 864 | Only called by `apply_to_primitive_vector` above |
| `unwrap_device` | `vortex-array/src/buffer.rs` | 244 | Zero callers anywhere |
| `into_record_batch_with_schema` | `vortex-array/src/arrow/record_batch.rs` | 43 | Only used in same-file test |
| `canonical_bool_to_arrow` | `vortex-array/src/arrow/executor/bool.rs` | 16 | Only called by `to_arrow_bool` in same file (make private) |
| `canonical_null_to_arrow` | `vortex-array/src/arrow/executor/null.rs` | 15 | Only called by `to_arrow_null` in same file (make private) |
| `canonical_primitive_to_arrow` | `vortex-array/src/arrow/executor/primitive.rs` | 21 | Only called by `to_arrow_primitive` in same file (make private) |
| `canonical_varbinview_to_arrow` | `vortex-array/src/arrow/executor/byte_view.rs` | 24 | Only called by `to_arrow_byte_view` in same file (make private) |
| `has_same_dtype_as_array` | `vortex-array/src/expr/stats/mod.rs` | 157 | Only used in same-file tests |
| `opt_bool_vec` | `vortex-array/src/arrays/bool/test_harness.rs` | 10 | Behind `_test-harness` feature, zero callers |
| `bool_vec` | `vortex-array/src/arrays/bool/test_harness.rs` | 20 | Behind `_test-harness` feature, zero callers |

### `pub(crate)` (~50 items)

| Item | File | Notes |
|------|------|-------|
| `to_int_indices` | `vortex-array/src/test_harness.rs` | Only used in vortex-array tests |
| `partial_min` | `vortex-array/src/partial_ord.rs` | Only used in expr/stats |
| `partial_max` | `vortex-array/src/partial_ord.rs` | Only used in expr/stats |
| `to_null_buffer` | `vortex-array/src/arrow/null_buffer.rs` | Only used in arrow/executor |
| `from_arrow_array_with_len` | `vortex-array/src/arrow/datum.rs` | Only used in compute modules |
| `execute_varbinview_to_arrow` | `vortex-array/src/arrow/executor/byte_view.rs` | Only used by arrow/executor/byte.rs |
| `to_arrow_opts` | `vortex-array/src/arrow/compute/to_arrow/mod.rs` | Only used by `to_arrow_preferred` |
| `try_new_with_target_datatype` | `vortex-array/src/arrow/datum.rs` | Only used in compute modules |
| `dict_encode_with_constraints` | `vortex-array/src/builders/dict/mod.rs` | Only used by `dict_encode` |
| `bytes_dict_builder` | `vortex-array/src/builders/dict/bytes.rs` | Only used by `dict_encoder` |
| `primitive_dict_builder` | `vortex-array/src/builders/dict/primitive.rs` | Only used by `dict_encoder` |
| `with_buffer_deduplication` | `vortex-array/src/builders/varbinview.rs` | Only used in own test |
| `with_compaction` | `vortex-array/src/builders/varbinview.rs` | Only used in varbinview/compact |
| `label_is_fallible` | `vortex-array/src/expr/analysis/fallible.rs` | Only used in expr/analysis |
| `label_tree` | `vortex-array/src/expr/analysis/labeling.rs` | Only used in expr/analysis |
| `label_null_sensitive` | `vortex-array/src/expr/analysis/null_sensitive.rs` | Only used in expr/analysis |
| `immediate_scope_accesses` | `vortex-array/src/expr/analysis/immediate_access.rs` | Only used by `immediate_scope_access` |
| `descendent_annotations` | `vortex-array/src/expr/analysis/annotation.rs` | Only used in expr/transform |
| `normalize_to_included_fields` | `vortex-array/src/expr/exprs/select.rs` | Only used in expr/analysis |
| `and_collect_right` | `vortex-array/src/expr/exprs/binary.rs` | Only used in own tests |
| `try_optimize` | `vortex-array/src/expr/optimize.rs` | Only used within optimizer |
| `try_optimize_recursive` | `vortex-array/src/expr/optimize.rs` | Only used within optimizer |
| `simplify_untyped` | `vortex-array/src/expr/optimize.rs` | Only used within vortex-array |
| `stat_falsification` | `vortex-array/src/expr/expression.rs` | Only used in expr modules |
| `stat_expression` | `vortex-array/src/expr/expression.rs` | Only used in expr modules |
| `find_partition` | `vortex-array/src/expr/transform/partition.rs` | Only used in own tests |
| `find_between` | `vortex-array/src/expr/transform/match_between.rs` | Only used in expr/optimize |
| `pre_order_visit_up` | `vortex-array/src/expr/traversal/visitor.rs` | Only used within expr traversal |
| `pre_order_visit_down` | `vortex-array/src/expr/traversal/visitor.rs` | Only used within expr traversal |
| `is_commutative` | `vortex-array/src/expr/stats/mod.rs` | Only used in stats_set |
| `upcast_decimal_values` | `vortex-array/src/arrays/decimal/compute/cast.rs` | Only used in own cast module |
| `search_index` | `vortex-array/src/patches.rs` | Only used within patches.rs |
| `take_search` | `vortex-array/src/patches.rs` | Only used in vortex-array benches |
| `take_map` | `vortex-array/src/patches.rs` | Only used in vortex-array benches |
| `merge_ordered` | `vortex-array/src/stats/stats_set.rs` | Only used within stats_set.rs |
| `merge_unordered` | `vortex-array/src/stats/stats_set.rs` | Only used within stats_set.rs |
| `combine_sets` | `vortex-array/src/stats/stats_set.rs` | Only used in compute/take |
| `display_table` | `vortex-array/src/display/mod.rs` | Only used in struct_/compute/zip |
| `compact_with_threshold` | `vortex-array/src/arrays/varbinview/compact.rs` | Only used by `compact_buffers` |
| `overall_utilization` | `vortex-array/src/arrays/varbinview/compact.rs` | Only used in builders/varbinview |
| `range_utilization` | `vortex-array/src/arrays/varbinview/compact.rs` | Only used in builders/varbinview |
| `offsets_to_lengths` | `vortex-array/src/arrays/varbinview/build_views.rs` | Only used in varbin/vtable |
| `list_view_from_list` | `vortex-array/src/arrays/listview/conversion.rs` | Only used in list/vtable |
| `verify_is_zero_copy_to_list` | `vortex-array/src/arrays/listview/array.rs` | Only used in tests |
| `element_mask_from_offsets` | `vortex-array/src/arrays/list/compute/filter.rs` | Only used within same file |
| `from_iter_opt_slow` | `vortex-array/src/arrays/list/test_harness.rs` | Only used in builders |
| `validate_all_values_referenced` | `vortex-array/src/arrays/dict/array.rs` | Only used by `validate` |
| `compute_referenced_values_mask` | `vortex-array/src/arrays/dict/array.rs` | Only used in dict/compute |
| `remove_column` | `vortex-array/src/arrays/struct_/array.rs` | Only used in own tests |
| `new_fieldless_with_len` | `vortex-array/src/arrays/struct_/array.rs` | Only used in struct_/compute |
| `pack_nested_structs` | `vortex-array/src/arrays/chunked/vtable/canonical.rs` | Only used in chunked tests |
| `pack_nested_lists` | `vortex-array/src/arrays/chunked/vtable/canonical.rs` | Only used in chunked tests |
| `non_empty_chunks` | `vortex-array/src/arrays/chunked/array.rs` | Only used in chunked compute |
| `rechunk` | `vortex-array/src/arrays/chunked/array.rs` | Only used in own tests |
| `mask_validity_canonical` | `vortex-array/src/arrays/masked/mod.rs` | Only used in masked module |
| `zero_offsets` | `vortex-array/src/arrays/varbin/array.rs` | Only used within vortex-array |
| `try_to_mask_fill_null_false` | `vortex-array/src/compute/filter.rs` | Only used in array/mod.rs |
| `arrow_filter_fn` | `vortex-array/src/compute/filter.rs` | Only used in filter/execute |
| `sum_impl` | `vortex-array/src/compute/sum.rs` | Only used within sum.rs |
| `is_arrow` | `vortex-array/src/array/mod.rs` | Only used in compute |
| `to_found` | `vortex-array/src/search_sorted.rs` | Only used in patches.rs |
| `to_index` | `vortex-array/src/search_sorted.rs` | Only used in patches.rs |
| `from_bit_buffer` (Validity) | `vortex-array/src/validity.rs` | Only used within vortex-array |
| `filter_indices` | `vortex-array/src/arrays/bool/compute/filter.rs` | Only used in chunked compute |
| `filter_slices` | `vortex-array/src/arrays/bool/compute/filter.rs` | Only used in chunked compute |
| `format_indices` | `vortex-array/src/arrays/assertions.rs` | Only used within display |
| `random_validity` | `vortex-array/src/arrays/arbitrary.rs` | Only used in dict/arbitrary |
| `gen_primitive_for_dict` | `vortex-array/src/arrays/dict_test.rs` | Only used in vortex-array benches |

---

## 9. vortex-compute

### `pub(crate)` (1 item)

| Item | File | Line | Notes |
|------|------|------|-------|
| `take_portable` | `vortex-compute/src/take/slice/portable.rs` | 33 | Only used within vortex-compute |

---

## 10. vortex-io

### REMOVE (9 items)

| Item | File | Line | Notes |
|------|------|------|-------|
| `SizeLimitedStream::new` | `vortex-io/src/size_limited.rs` | (varies) | Entire type unused outside crate |
| `SizeLimitedStream::inner_ref` | `vortex-io/src/size_limited.rs` | (varies) | |
| `SizeLimitedStream::inner_mut` | `vortex-io/src/size_limited.rs` | (varies) | |
| `SizeLimitedStream::into_inner` | `vortex-io/src/size_limited.rs` | (varies) | |
| `ObjectStoreSource::with_coalesce_config` | `vortex-io/src/object_store/mod.rs` | (varies) | No external callers |
| `ObjectStoreSource::with_some_coalesce_config` | `vortex-io/src/object_store/mod.rs` | (varies) | No external callers |
| `CurrentThreadWorkerPool::worker_count` | `vortex-io/src/compio/mod.rs` | (varies) | No external callers |
| `ObjectStoreWriter::put_result` | `vortex-io/src/object_store/mod.rs` | (varies) | No external callers |
| `OwnedSlice::into_inner` | `vortex-io/src/owned_slice.rs` | (varies) | No external callers |
| `AsyncWriteAdapter` | `vortex-io/src/` | (varies) | No external callers |
| `single::block_on_stream` | `vortex-io/src/single.rs` | (varies) | No external callers |

### `pub(crate)` (6 items)

| Item | File | Line | Notes |
|------|------|------|-------|
| `Handle::find` | `vortex-io/src/compio/mod.rs` | (varies) | Only used within vortex-io |
| `TokioRuntime::current` | `vortex-io/src/tokio/mod.rs` | (varies) | Only used within vortex-io |
| `CurrentThreadWorkerPool::set_workers` | `vortex-io/src/compio/mod.rs` | (varies) | Only used within vortex-io |
| `CoalesceConfig::new` | `vortex-io/src/object_store/coalesce.rs` | (varies) | Only used within vortex-io |
| `CoalesceConfig::local` | `vortex-io/src/object_store/coalesce.rs` | (varies) | Only used within vortex-io |
| `CoalesceConfig::object_storage` | `vortex-io/src/object_store/coalesce.rs` | (varies) | Only used within vortex-io |

---

## 11. vortex-session

### REMOVE (5 items)

| Item | File | Line | Notes |
|------|------|------|-------|
| `Ref::map` | `vortex-session/src/lib.rs` | (varies) | No external callers |
| `RefMut::map` | `vortex-session/src/lib.rs` | (varies) | No external callers |
| `Registry::empty` | `vortex-session/src/lib.rs` | (varies) | No external callers |
| `Registry::items` | `vortex-session/src/lib.rs` | (varies) | No external callers |
| `Registry::find_many` | `vortex-session/src/lib.rs` | (varies) | No external callers |

---

## 12. vortex-layout

### REMOVE (10 items)

| Item | File | Line | Notes |
|------|------|------|-------|
| `dict_layout_supported` | `vortex-layout/src/layouts/dict/writer.rs` | 514 | No external callers |
| `display_tree_verbose` | `vortex-layout/src/layout.rs` | 207 | No external callers |
| `CompactOptions::with_pco_level` | `vortex-layout/src/layouts/compact.rs` | 53 | No external callers |
| `CompactOptions::with_zstd_level` | `vortex-layout/src/layouts/compact.rs` | 58 | No external callers |
| `CompactOptions::with_zstd_use_dicts` | `vortex-layout/src/layouts/compact.rs` | 63 | No external callers |
| `FlatLayoutStrategy::with_include_padding` | `vortex-layout/src/layouts/flat/writer.rs` | 54 | No external callers |
| `FlatLayoutStrategy::with_max_variable_length_statistics_size` | `vortex-layout/src/layouts/flat/writer.rs` | 60 | No external callers |
| `TableWriterOptions::with_default_strategy` | `vortex-layout/src/layouts/table.rs` | 136 | No external callers |
| `TableWriterOptions::with_validity_strategy` | `vortex-layout/src/layouts/table.rs` | 142 | No external callers |
| `FlatLayout::new_with_metadata` | `vortex-layout/src/layouts/flat/mod.rs` | 152 | No external callers |

### `pub(crate)` (38+ items)

| Item | File | Notes |
|------|------|-------|
| `OwnedLayoutChildren::layout_children` | `children.rs` | Only used within vortex-layout |
| `MAX_IS_TRUNCATED` | `layouts/zoned/builder.rs` | Only used within vortex-layout |
| `MIN_IS_TRUNCATED` | `layouts/zoned/builder.rs` | Only used within vortex-layout |
| `stats_builder_with_capacity` | `layouts/zoned/builder.rs` | Only used within vortex-layout |
| `ZoneMapBuilder::lower_bound` | `layouts/zoned/builder.rs` | Only used within vortex-layout |
| `ZoneMapBuilder::upper_bound` | `layouts/zoned/builder.rs` | Only used within vortex-layout |
| `ZoneMapBuilder::all_invalid` | `layouts/zoned/builder.rs` | Only used within vortex-layout |
| `ZoneMapAccumulator::push_chunk_without_compute` | `layouts/zoned/zone_map.rs` | Only used within vortex-layout |
| `ZoneMapAccumulator::push_chunk` | `layouts/zoned/zone_map.rs` | Only used within vortex-layout |
| `ZoneMapAccumulator::as_stats_table` | `layouts/zoned/zone_map.rs` | Only used within vortex-layout |
| `ZoneMapAccumulator::new` | `layouts/zoned/zone_map.rs` | Only used within vortex-layout |
| `ZoneMap::to_stats_set` | `layouts/zoned/zone_map.rs` | Only used within vortex-layout |
| `ZoneMap::get_stat` | `layouts/zoned/zone_map.rs` | Only used within vortex-layout |
| `ZoneMap::try_new` | `layouts/zoned/zone_map.rs` | Only used within vortex-layout |
| `ZoneMap::new_unchecked` | `layouts/zoned/zone_map.rs` | Only used within vortex-layout |
| `ZoneMap::dtype_for_stats_table` | `layouts/zoned/zone_map.rs` | Only used within vortex-layout |
| `ZoneMap::array` | `layouts/zoned/zone_map.rs` | Only used within vortex-layout |
| `ZoneMap::present_stats` | `layouts/zoned/zone_map.rs` | Only used within vortex-layout |
| `ZoneMap::prune` | `layouts/zoned/zone_map.rs` | Only used within vortex-layout |
| `ZonedLayout::nzones` | `layouts/zoned/mod.rs` | Only used within vortex-layout |
| `ZonedLayout::present_stats` | `layouts/zoned/mod.rs` | Only used within vortex-layout |
| `StructLayout::matching_fields` | `layouts/struct_/mod.rs` | Only used within vortex-layout |
| `SequenceReader::new` | `sequence.rs` | Only used within vortex-layout |
| `SequencePointer::root` | `sequence.rs` | Only used within vortex-layout |
| `SequencePointer::descend` | `sequence.rs` | Only used within vortex-layout |
| `SequencePointer::collapse` | `sequence.rs` | Only used within vortex-layout |
| `SequencePointer::split` | `sequence.rs` | Only used within vortex-layout |
| `SequencePointer::split_off` | `sequence.rs` | Only used within vortex-layout |
| `SequencePointer::advance` | `sequence.rs` | Only used within vortex-layout |
| `SequencePointer::downgrade` | `sequence.rs` | Only used within vortex-layout |
| `LazyReaderChildren::new` | `reader.rs` | Only used within vortex-layout |
| `LazyReaderChildren::get` | `reader.rs` | Only used within vortex-layout |
| `LayoutRegistry::register` | `session.rs` | Only used within vortex-layout |
| `LayoutRegistry::register_many` | `session.rs` | Only used within vortex-layout |
| `LayoutRegistry::registry` | `session.rs` | Only used within vortex-layout |
| `DictCodesPType::new` | `layouts/dict/mod.rs` | Only used within vortex-layout |
| `DictLayoutEncoding::new` | `layouts/dict/writer.rs` | Only used within vortex-layout |
| `FileStatsAccumulator::stats_sets` | `layouts/file_stats.rs` | Only used within vortex-layout |
| `DisplayLayoutTree::new` | `display.rs` | Only used within vortex-layout |
| `CompressedStrategy::new_compact` | `layouts/compressed.rs` | Only used within vortex-layout |

---

## 13. vortex-scan

### REMOVE (2 items)

| Item | File | Line | Notes |
|------|------|------|-------|
| `RepeatedScan::execute_stream` | `vortex-scan/src/repeated_scan.rs` | 185 | No external callers |
| `RepeatedScan::execute_array_stream` | `vortex-scan/src/repeated_scan.rs` | 74 | No external callers |

### `pub(crate)` (13 items)

| Item | File | Notes |
|------|------|-------|
| `VortexFilter::new` | `filter.rs` | Only used within vortex-scan |
| `VortexFilter::conjuncts` | `filter.rs` | Only used within vortex-scan |
| `VortexFilter::dynamic_updates` | `filter.rs` | Only used within vortex-scan |
| `VortexFilter::next_conjunct` | `filter.rs` | Only used within vortex-scan |
| `VortexFilter::report_selectivity` | `filter.rs` | Only used within vortex-scan |
| `RowMask::new` | `row_mask.rs` | Only used within vortex-scan |
| `RowMask::row_range` | `row_mask.rs` | Only used within vortex-scan |
| `RowMask::mask` | `row_mask.rs` | Only used within vortex-scan |
| `LayoutScanContext::new` | `layout.rs` | Only used within vortex-scan |
| `SplitBy::splits` | `split_by.rs` | Only used within vortex-scan |
| `RecordBatchIterator::new` | `arrow.rs` | Only used within vortex-scan |
| `RepeatedScan::execute` | `repeated_scan.rs` | Only used within vortex-scan |
| `ScanBuilder::build` (after prepare) | `scan_builder.rs` | Only used within vortex-scan |

---

## 14. vortex-file

### REMOVE (6 items)

| Item | File | Line | Notes |
|------|------|------|-------|
| `VERSION` | `vortex-file/src/lib.rs` | 130 | No external callers |
| `V1_FOOTER_FBS_SIZE` | `vortex-file/src/lib.rs` | 132 | No external callers |
| `StrategyBuilder::with_row_block_size` | `vortex-file/src/strategy.rs` | 145 | No external callers |
| `FooterDeserializer::with_some_dtype` | `vortex-file/src/footer/deserializer.rs` | 59 | No external callers |
| `FooterDeserializer::with_some_size` | `vortex-file/src/footer/deserializer.rs` | 69 | No external callers |
| `FooterSerializer::with_exclude_dtype` | `vortex-file/src/footer/serializer.rs` | 57 | No external callers |

### `pub(crate)` (24 items)

| Item | File | Notes |
|------|------|-------|
| `CountingWrite::new` | `counting.rs` | Only used within vortex-file |
| `CountingWrite::counter` | `counting.rs` | Only used within vortex-file |
| `extract_relevant_file_stats_as_struct_row` | `pruning.rs` | Only used within vortex-file |
| `SegmentWriter::new` | `segments/writer.rs` | Only used within vortex-file |
| `SegmentWriter::segment_specs` | `segments/writer.rs` | Only used within vortex-file |
| `FileSegmentSource::open` | `segments/source.rs` | Only used within vortex-file |
| `ReadRequest::offset` | `read/request.rs` | Only used within vortex-file |
| `ReadRequest::len` | `read/request.rs` | Only used within vortex-file |
| `ReadRequest::alignment` | `read/request.rs` | Only used within vortex-file |
| `ReadRequest::resolve` | `read/request.rs` | Only used within vortex-file |
| `Footer::into_serializer` | `footer/mod.rs` | Only used within vortex-file |
| `Footer::deserializer` | `footer/mod.rs` | Only used within vortex-file |
| `VortexWriteOptions::exclude_dtype` | `writer.rs` | Only used within vortex-file |
| `VortexWriteOptions::with_file_statistics` | `writer.rs` | Only used within vortex-file |
| `Writer::push_stream` | `writer.rs` | Only used within vortex-file |
| `FooterSerializer::with_offset` | `footer/serializer.rs` | Only used within vortex-file |
| `FooterSerializer::exclude_dtype` | `footer/serializer.rs` | Only used within vortex-file |
| `FooterSerializer::serialize` | `footer/serializer.rs` | Only used within vortex-file |
| `FooterDeserializer::with_dtype` | `footer/deserializer.rs` | Only used within vortex-file |
| `FooterDeserializer::with_size` | `footer/deserializer.rs` | Only used within vortex-file |
| `FooterDeserializer::prefix_data` | `footer/deserializer.rs` | Only used within vortex-file |
| `FooterDeserializer::deserialize` | `footer/deserializer.rs` | Only used within vortex-file |
| `DeserializeStep::buffer` | `footer/deserializer.rs` | Only used within vortex-file |
| `SegmentSpec::byte_range` | `footer/segment.rs` | Only used within vortex-file |

---

## 15. vortex-ipc

### REMOVE (13 items — entire reader/writer layer is unused)

| Item | File | Notes |
|------|------|-------|
| `SyncIPCReader::try_new` | `iterator.rs` | No external callers |
| `ArrayIteratorIPCBytes::collect_to_buffer` | `iterator.rs` | No external callers |
| `AsyncIPCReader::try_new` | `stream.rs` | No external callers |
| `AsyncIPCReader::collect_to_buffer` | `stream.rs` | No external callers |
| `AsyncMessageWriter::new` | `messages/writer_async.rs` | No external callers |
| `AsyncMessageWriter::write_message` | `messages/writer_async.rs` | No external callers |
| `AsyncMessageWriter::inner` | `messages/writer_async.rs` | No external callers |
| `AsyncMessageWriter::into_inner` | `messages/writer_async.rs` | No external callers |
| `SyncMessageWriter::new` | `messages/writer_sync.rs` | No external callers |
| `SyncMessageWriter::write_message` | `messages/writer_sync.rs` | No external callers |
| `AsyncMessageReader::new` | `messages/reader_async.rs` | No external callers |
| `SyncMessageReader::new` | `messages/reader_sync.rs` | No external callers |
| `BufferedMessageReader::new` | `messages/reader_buf.rs` | No external callers |

### `pub(crate)` (1 item)

| Item | File | Notes |
|------|------|-------|
| `MessageDecoder::read_next` | `messages/decoder.rs` | Only used within vortex-ipc |

---

## 16. vortex-datafusion

### REMOVE (1 item)

| Item | File | Notes |
|------|------|-------|
| `VortexFormatFactory::with_options` | `persistent/format.rs:148` | Never called anywhere (only in doc comment example) |

### `pub(crate)` (8 items)

| Item | File | Notes |
|------|------|-------|
| `calculate_physical_schema` | `convert/schema.rs:22` | Only used in `opener.rs` |
| `CachedVortexMetadata::new` | `persistent/cache.rs:23` | Only used in `format.rs`, `opener.rs` |
| `CachedVortexMetadata::footer` | `persistent/cache.rs:30` | Only used in `opener.rs` |
| `VortexSource::with_expression_convertor` | `persistent/source.rs:92` | Only used internally |
| `VortexSource::with_vortex_reader_factory` | `persistent/source.rs:103` | Only used internally |
| `VortexSource::vx_metrics` | `persistent/source.rs:112` | Only used in `metrics.rs` |
| `VortexSource::with_file_metadata_cache` | `persistent/source.rs:117` | Only used in `format.rs` |
| `DefaultVortexReaderFactory::new` | `persistent/reader.rs:30` | Only used in `opener.rs` |

---

## 17. vortex-bench

### `pub(crate)` (40+ items)

Nearly all public items in vortex-bench are only used within the crate itself. Key modules:

**`statpopgen/`** — All items: `schema_from_vcf_header`, `VcfRecordBatchBuilder::new`, `consume_record`,
`consume_info`, `finish`, `InfoArrayBuilder` methods, `builder_from_info`, `data_type_from_info`,
all `value_*`/`parse_*` functions, `StatPopGenBenchmark::new`/`parquet_path`/`vortex_path`/`vortex_compact_path`,
`FILE_NAME`

**`tpch/`** — `TPC_H_ROW_COUNT_ARRAY_LENGTH`, `EXPECTED_ROW_COUNTS_SF1`, `EXPECTED_ROW_COUNTS_SF10`, `tpch_queries`

**`tpcds/`** — `tpcds_queries`, `generate_tpcds`

**`memory.rs`** — `MemoryStats::new`/`diff`, `MemoryTracker::new`/`current_memory`/`peak_memory`/`reset_peak`,
`MemoryMeasurement::new`/`start_query`/`end_query`/`peak_memory`, `MemoryMeasurementResult::start`/`end`

**`output.rs`** — `default_output_path`, `vortex_bench_dir`

**`measurements.rs`** — `mean_time`, `median_time`, `median_run`

**`runner.rs`** — `export_results`

**`utils/`** — `temp_download_filepath`, `default_env_filter`, `STORAGE_S3`, `STORAGE_GCS`

**`conversions.rs`** — `parquet_to_vortex_stream`

**`public_bi.rs`** — `fetch_schemas_and_queries`

**`datasets/data_downloads.rs`** — `decompress_bz2`

---

## 18. vortex-duckdb

### `pub(crate)` (10+ items)

Most are already in private modules (`convert`, `utils`) so they're effectively crate-private.
Items in public modules that should be restricted:

| Item | File | Notes |
|------|------|-------|
| `ArrayExporter::try_new` | `exporter/mod.rs` | Only used in `scan.rs` |
| `ArrayExporter::export` | `exporter/mod.rs` | Only used in `scan.rs` |
| `precision_to_duckdb_storage_size` | `exporter/decimal.rs` | Only used in `exporter/mod.rs` and `convert/vector.rs` |
| `ConversionCache::new` | `exporter/cache.rs` | Only used in `scan.rs` |
| `ConversionCache::instance_id` | `exporter/cache.rs` | Only used internally |
| `i128_from_parts` | `duckdb/value.rs` | Only used within `value.rs` |
| `s3_store` | `utils/object_store.rs` | Already private module |
| `flat_vector_to_vortex` | `convert/vector.rs` | Already private module |
| `data_chunk_to_vortex` | `convert/vector.rs` | Already private module |
| `from_duckdb_table` | `convert/dtype.rs` | Already private module |
| `try_from_table_filter` | `convert/table_filter.rs` | Already private module |
| `try_from_bound_expression` | `convert/expr.rs` | Already private module |

---

## 19. vortex-ffi

### `pub(crate)` (1 item)

| Item | File | Notes |
|------|------|-------|
| `try_or_default` | `error.rs:23` | Only used in `array.rs`, `file.rs`, `array_iterator.rs`, `sink.rs` |

---

## 20. vortex-metrics

### REMOVE (3 items)

| Item | File | Notes |
|------|------|-------|
| `pub use Tags` | `lib.rs` | Re-exported but never imported externally |
| `pub use MetricId` | `lib.rs` | Re-exported but never imported externally |
| `VortexMetricsIter` | `lib.rs` | Never used externally |

### `pub(crate)` (3 items)

| Item | File | Notes |
|------|------|-------|
| `DefaultTags` | `lib.rs` | Only used within vortex-metrics |
| `VortexMetrics::new` | `lib.rs` | Only used within vortex-metrics |
| `VortexMetrics::new_with_tags` | `lib.rs` | Only used within vortex-metrics |

---

## 21. vortex-btrblocks

### `pub(crate)` (27+ items)

**Re-exports that should be restricted:**
- `FloatCode`, `StringCode`, `CanonicalCompressor` — re-exported from lib.rs but not imported by any other crate

**Internal items across:**
- `ctx.rs` — All compressor context items
- `sample.rs` — All sampling items
- `rle.rs` — All RLE items
- `patches.rs` — All patch items
- `stats/` — All statistics items
- `dictionary/` — All dictionary items

### Removable `#[allow(dead_code)]`

| File | Line | Notes |
|------|------|-------|
| `vortex-btrblocks/src/compressor/float/stats.rs` | 62 | On `average_run_length` field — field IS used, annotation is unnecessary |

---

## 22. Encoding Crates

### REMOVE (11 items)

| Item | Crate | File | Notes |
|------|-------|------|-------|
| `RDEncoder::from_parts` | roaring | `encodings/roaring/src/` | No external callers |
| `DateTimePartsArray::get_days_ptype` | datetime-parts | `encodings/datetime-parts/src/` | No external callers |
| `DateTimePartsArray::get_seconds_ptype` | datetime-parts | `encodings/datetime-parts/src/` | No external callers |
| `DateTimePartsArray::get_subseconds_ptype` | datetime-parts | `encodings/datetime-parts/src/` | No external callers |
| `BitPackedArray::max_packed_value` | fastlanes | `encodings/fastlanes/src/` | No external callers |
| `DeltaArray::is_empty` | fastlanes | `encodings/fastlanes/src/` | No external callers |
| `RLEArray::is_empty` | runend | `encodings/runend/src/` | No external callers |
| `PcoArray::from_array` | pco | `encodings/pco/src/` | No external callers |
| `runend_decode_typed_primitive` | runend | `encodings/runend/src/` | No external callers |
| `runend_decode_typed_bool` | runend | `encodings/runend/src/` | No external callers |
| `RunEndArray::try_new_offset_length` | runend | `encodings/runend/src/` | No external callers |

### `pub(crate)` (91 items across all encoding crates)

**fastlanes** (bitpacking + delta internals):
- `BitPackedArray` internal accessors, packing/unpacking helpers, `DeltaArray` internal accessors

**alp**: Internal encoding/decoding helpers

**bytebool**: Internal array accessors

**datetime-parts**: Internal accessors for days/seconds/subseconds arrays

**fsst**: Internal symbol table and encoding helpers

**pco**: Internal compression/decompression helpers

**runend**: Internal run-end array accessors, encoding/decoding helpers

**sequence**: Internal array accessors

**sparse**: Internal sparse array accessors, fill value helpers

**zigzag**: Internal encoding/decoding helpers

**zstd**: Internal compression/decompression helpers

---

## 23. `#[allow(dead_code)]` Annotations

| Location | Action | Reason |
|----------|--------|--------|
| `duckdb-rs/` (vendored) | KEEP | Not vortex code |
| `vortex-utils/src/env.rs:62` — `EnvVarGuard::lock_guard` | KEEP | Intentional RAII pattern |
| `vortex-flatbuffers/` generated modules | KEEP | Generated code |
| `vortex-array/src/variants.rs` — `NullTyped`, `Utf8Typed`, `BinaryTyped`, `DecimalTyped`, `ListTyped` | KEEP | Wrapper structs holding `&dyn Array` |
| **`vortex-btrblocks/src/compressor/float/stats.rs:62`** — `average_run_length` | **REMOVE** | Field is actually used; annotation is unnecessary |