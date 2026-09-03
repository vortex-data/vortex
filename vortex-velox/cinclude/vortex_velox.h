// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// THIS FILE IS AUTO-GENERATED. DO NOT EDIT IT DIRECTLY.

typedef struct ArrowSchema ArrowSchema;
typedef struct ArrowArray ArrowArray;

typedef struct vx_velox_error vx_velox_error;
typedef struct vx_velox_session vx_velox_session;
typedef struct vx_velox_dtype vx_velox_dtype;
typedef struct vx_velox_scalar vx_velox_scalar;
typedef struct vx_velox_expression vx_velox_expression;
typedef struct vx_velox_data_source vx_velox_data_source;
typedef struct vx_velox_scan vx_velox_scan;
typedef struct vx_velox_partition vx_velox_partition;
typedef struct vx_velox_array vx_velox_array;
typedef struct vx_velox_export_cursor vx_velox_export_cursor;


#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

/**
 * The current major version of the Vortex and Velox adapter ABI.
 */
#define VX_VELOX_ABI_VERSION 6

/**
 * The adapter supports batched host range reads.
 */
#define VX_VELOX_CAPABILITY_BATCH_READ (1 << 0)

/**
 * The adapter can open callback-backed Vortex sources.
 */
#define VX_VELOX_CAPABILITY_CALLBACK_SOURCE (1 << 1)

/**
 * The adapter reports stable natural row splits.
 */
#define VX_VELOX_CAPABILITY_NATURAL_SPLITS (1 << 2)

/**
 * The adapter can visit canonical primitive values in retained blocks.
 */
#define VX_VELOX_CAPABILITY_PRIMITIVE_VISITOR (1 << 3)

/**
 * The adapter can export source schemas through the Arrow C Data Interface.
 */
#define VX_VELOX_CAPABILITY_ARROW_SCHEMA (1 << 4)

/**
 * The adapter can export one Vortex array through the Arrow C Data Interface.
 */
#define VX_VELOX_CAPABILITY_ARRAY_ARROW_EXPORT (1 << 5)

/**
 * The adapter can project absolute file-row indexes with scan fields.
 */
#define VX_VELOX_CAPABILITY_ROW_INDEX_PROJECTION (1 << 6)

/**
 * The adapter can prove that natural splits cannot match an expression.
 */
#define VX_VELOX_CAPABILITY_NATURAL_SPLIT_PRUNING (1 << 7)

/**
 * The callback reader observes host cancellation before each host read callback.
 *
 * This capability does not claim cancellation during cached scans or CPU execution.
 */
#define VX_VELOX_CAPABILITY_READ_CANCELLATION (1 << 8)

/**
 * The adapter retains one prepared array across several Velox output windows.
 */
#define VX_VELOX_CAPABILITY_EXPORT_CURSOR (1 << 9)

/**
 * The adapter can omit row-index projection from scans with contiguous rows.
 */
#define VX_VELOX_CAPABILITY_PLAIN_PROJECTION (1 << 10)

/**
 * The adapter can visit canonical UTF-8 and binary values in retained blocks.
 */
#define VX_VELOX_CAPABILITY_VARBIN_VISITOR (1 << 11)

/**
 * The adapter can preserve dictionary arrays during native export.
 */
#define VX_VELOX_CAPABILITY_DICTIONARY_VISITOR (1 << 12)

/**
 * The adapter can preserve constant arrays during native export.
 */
#define VX_VELOX_CAPABILITY_CONSTANT_VISITOR (1 << 13)

/**
 * The adapter can visit canonical packed Boolean values in retained blocks.
 */
#define VX_VELOX_CAPABILITY_BOOL_VISITOR (1 << 14)

/**
 * The adapter can export Vortex day-based dates through the primitive visitor.
 */
#define VX_VELOX_CAPABILITY_DATE_VISITOR (1 << 15)

/**
 * The adapter can normalize Vortex decimals for the primitive visitor.
 */
#define VX_VELOX_CAPABILITY_DECIMAL_VISITOR (1 << 16)

/**
 * The adapter can preserve canonical struct children during native export.
 */
#define VX_VELOX_CAPABILITY_STRUCT_VISITOR (1 << 17)

/**
 * The adapter can preserve canonical list children during native export.
 */
#define VX_VELOX_CAPABILITY_LIST_VISITOR (1 << 18)

/**
 * The adapter can preserve canonical map children during native export.
 */
#define VX_VELOX_CAPABILITY_MAP_VISITOR (1 << 19)

/**
 * Natural splits include a stable byte-range assignment token.
 */
#define VX_VELOX_CAPABILITY_SPLIT_ASSIGNMENT (1 << 20)

/**
 * An opaque Vortex positional reader backed by Velox callbacks.
 */
typedef struct vx_velox_read_at vx_velox_read_at;

/**
 * An opened Vortex file that uses Velox callbacks for all reads.
 */
typedef struct vx_velox_source vx_velox_source;

typedef struct {
  const char *ptr;
  size_t len;
} vx_velox_view;

/**
 * A fixed-width primitive type identifier for Velox scalar construction.
 */
typedef uint32_t vx_velox_ptype;

/**
 * A fixed-width binary expression operator identifier.
 */
typedef uint32_t vx_velox_binary_operator;

/**
 * A fixed-width row-selection mode identifier.
 */
typedef uint32_t vx_velox_scan_selection_include;

/**
 * A stable row selection for one scan request.
 */
typedef struct {
  /**
   * The selected row indexes.
   */
  const uint64_t *indices;
  /**
   * The number of selected row indexes.
   */
  size_t length;
  /**
   * The selection mode.
   */
  vx_velox_scan_selection_include include;
} vx_velox_scan_selection;

/**
 * Stable options for one Vortex scan.
 */
typedef struct {
  /**
   * Set this field to `sizeof(vx_velox_scan_options)`.
   */
  size_t struct_size;
  /**
   * Set this field to [`crate::VX_VELOX_ABI_VERSION`].
   */
  uint32_t abi_version;
  /**
   * The projected expression, or null for every field.
   */
  const vx_velox_expression *projection;
  /**
   * The exact filter expression, or null for no filter.
   */
  const vx_velox_expression *filter;
  /**
   * The first row in the scan range.
   */
  uint64_t row_range_begin;
  /**
   * One past the final row in the scan range.
   */
  uint64_t row_range_end;
  /**
   * An optional row-index selection.
   */
  vx_velox_scan_selection selection;
  /**
   * The maximum output row count, or zero for no limit.
   */
  uint64_t limit;
  /**
   * Return rows in storage order.
   */
  bool ordered;
} vx_velox_scan_options;

/**
 * Host memory callbacks for one Arrow C Data export.
 */
typedef struct {
  /**
   * Set this field to `sizeof(vx_velox_arrow_memory_callbacks)`.
   */
  size_t struct_size;
  /**
   * Set this field to [`crate::VX_VELOX_ABI_VERSION`].
   */
  uint32_t abi_version;
  /**
   * An opaque host context.
   */
  void *context;
  /**
   * Retain the host context until the Arrow array release callback runs.
   */
  void (*retain_context)(void *context);
  /**
   * Release one host context reference.
   */
  void (*release_context)(void *context);
  /**
   * Reserve Arrow payload bytes before conversion. Zero means success.
   */
  int32_t (*report_allocation)(void *context, size_t retained_bytes);
  /**
   * Free retained Arrow payload bytes.
   */
  void (*report_free)(void *context, size_t retained_bytes);
  /**
   * Return the last callback error as a null-terminated string.
   */
  const char *(*last_error)(void *context);
} vx_velox_arrow_memory_callbacks;

/**
 * A positional read request passed to the Velox callback.
 */
typedef struct {
  /**
   * Set this field to `sizeof(vx_velox_read_request)`.
   */
  size_t struct_size;
  /**
   * The file offset in bytes.
   */
  uint64_t offset;
  /**
   * The exact requested length in bytes.
   */
  size_t length;
  /**
   * The required buffer alignment in bytes.
   */
  size_t alignment;
} vx_velox_read_request;

/**
 * A retained host buffer returned by the Velox callback.
 */
typedef struct {
  /**
   * Set this field to `sizeof(vx_velox_buffer)`.
   */
  size_t struct_size;
  /**
   * The first byte of the returned range.
   */
  const uint8_t *data;
  /**
   * The number of returned bytes.
   */
  size_t length;
  /**
   * An opaque owner passed to `release`.
   */
  void *owner;
  /**
   * Release the owner after Vortex no longer needs the bytes.
   */
  void (*release)(void *owner);
} vx_velox_buffer;

/**
 * Velox callbacks that provide a Vortex positional reader.
 *
 * Vortex can call these functions concurrently. The context and every callback must be
 * thread-safe. `concurrency` limits one callback batch and gives Vortex a scheduling hint. It does
 * not provide synchronization. `last_error` must return the calling thread's most recent callback
 * error. Its string must remain valid until the next callback on that thread. Every callback must
 * catch foreign exceptions and must not unwind across this ABI.
 */
typedef struct {
  /**
   * Set this field to `sizeof(vx_velox_read_at_callbacks)`.
   */
  size_t struct_size;
  /**
   * Set this field to [`crate::VX_VELOX_ABI_VERSION`].
   */
  uint32_t abi_version;
  /**
   * An opaque callback context.
   */
  void *context;
  /**
   * Return the file size through `size_out`. Zero means success.
   */
  int32_t (*size)(void *context, uint64_t *size_out);
  /**
   * Read every request and populate the matching output. Zero means success.
   */
  int32_t (*read_ranges)(void *context,
                         const vx_velox_read_request *requests,
                         size_t request_count,
                         vx_velox_buffer *outputs);
  /**
   * Return the last callback error as a null-terminated string.
   */
  const char *(*last_error)(void *context);
  /**
   * Release the callback context.
   */
  void (*release_context)(void *context);
  /**
   * Return a non-zero value after the host cancels the scan.
   */
  int32_t (*is_cancelled)(void *context);
  /**
   * Limit one callback batch and give Vortex a preferred concurrency value.
   */
  size_t concurrency;
} vx_velox_read_at_callbacks;

/**
 * A stable natural row range reported by a Vortex file.
 */
typedef struct {
  /**
   * Set this field to `sizeof(vx_velox_natural_split)`.
   */
  size_t struct_size;
  /**
   * The first row in the split.
   */
  uint64_t row_begin;
  /**
   * One past the final row in the split.
   */
  uint64_t row_end;
  /**
   * The file byte that assigns this split to one external byte range.
   */
  uint64_t assignment_byte;
} vx_velox_natural_split;

/**
 * A fixed-width primitive value identifier in a semantic visitor block.
 */
typedef uint32_t vx_velox_primitive_type;

/**
 * A fixed-width validity representation identifier for one visitor block.
 */
typedef uint32_t vx_velox_validity_kind;

/**
 * A retained owner for buffers in a visitor block.
 */
typedef struct {
  /**
   * Set this field to `sizeof(vx_velox_buffer_owner)`.
   */
  size_t struct_size;
  /**
   * An opaque retained object.
   */
  const void *owner;
  /**
   * Add one owner reference before the callback returns.
   */
  void (*retain)(const void *owner);
  /**
   * Release one retained owner reference.
   */
  void (*release)(const void *owner);
  /**
   * The exact sum of the value and validity allocation sizes retained by this owner.
   */
  size_t retained_bytes;
} vx_velox_buffer_owner;

/**
 * A canonical primitive block delivered to Velox.
 */
typedef struct {
  /**
   * Set this field to `sizeof(vx_velox_primitive_view)`.
   */
  size_t struct_size;
  /**
   * The physical type of each value.
   */
  vx_velox_primitive_type primitive_type;
  /**
   * The logical decimal precision, or zero for a non-decimal block.
   */
  uint32_t decimal_precision;
  /**
   * The logical decimal scale, or zero for a non-decimal block.
   */
  int32_t decimal_scale;
  /**
   * The number of logical values in the block.
   */
  size_t length;
  /**
   * The first value byte.
   */
  const uint8_t *values;
  /**
   * The number of value bytes.
   */
  size_t values_length;
  /**
   * The validity representation.
   */
  vx_velox_validity_kind validity_kind;
  /**
   * The first validity byte when `validity_kind` is `Bitmap`.
   */
  const uint8_t *validity;
  /**
   * The number of validity bytes.
   */
  size_t validity_length;
  /**
   * The first logical validity bit within `validity`.
   */
  size_t validity_bit_offset;
  /**
   * Retains all pointers in this view.
   */
  vx_velox_buffer_owner buffers;
  /**
   * The guaranteed byte alignment of a non-empty values buffer.
   */
  size_t values_alignment;
  /**
   * The guaranteed byte alignment of a non-empty validity buffer.
   */
  size_t validity_alignment;
} vx_velox_primitive_view;

/**
 * Identifies the logical type of a variable-width binary block.
 */
typedef uint32_t vx_velox_varbin_kind;

/**
 * Defines the stable 16-byte variable-width binary view contract.
 */
typedef struct {
  /**
   * Stores the logical byte length.
   */
  uint32_t length;
  /**
   * Stores inline bytes, or prefix, buffer index, and offset for outlined values.
   */
  uint8_t data[12];
} vx_velox_binary_view;

/**
 * Describes one retained payload buffer.
 */
typedef struct {
  /**
   * The first payload byte.
   */
  const uint8_t *data;
  /**
   * The number of visible payload bytes.
   */
  size_t length;
} vx_velox_byte_buffer_view;

/**
 * A canonical variable-width binary block delivered to Velox.
 */
typedef struct {
  /**
   * Set this field to `sizeof(vx_velox_varbin_view)`.
   */
  size_t struct_size;
  /**
   * Identifies UTF-8 or binary values.
   */
  vx_velox_varbin_kind kind;
  /**
   * The number of logical values in the block.
   */
  size_t length;
  /**
   * The first 16-byte binary view.
   */
  const vx_velox_binary_view *views;
  /**
   * The number of readable bytes in `views`.
   */
  size_t views_length;
  /**
   * The retained payload buffer descriptors.
   */
  const vx_velox_byte_buffer_view *data_buffers;
  /**
   * The number of payload buffer descriptors.
   */
  size_t data_buffer_count;
  /**
   * The validity representation.
   */
  vx_velox_validity_kind validity_kind;
  /**
   * The first validity byte when `validity_kind` is `Bitmap`.
   */
  const uint8_t *validity;
  /**
   * The number of readable validity bytes.
   */
  size_t validity_length;
  /**
   * The first logical validity bit within `validity`.
   */
  size_t validity_bit_offset;
  /**
   * Retains all pointers in this view.
   */
  vx_velox_buffer_owner buffers;
  /**
   * The guaranteed byte alignment of a non-empty view buffer.
   */
  size_t views_alignment;
  /**
   * The guaranteed byte alignment of a non-empty validity buffer.
   */
  size_t validity_alignment;
} vx_velox_varbin_view;

/**
 * A dictionary block delivered to Velox.
 */
typedef struct {
  /**
   * Set this field to `sizeof(vx_velox_dictionary_view)`.
   */
  size_t struct_size;
  /**
   * The number of logical dictionary codes.
   */
  size_t length;
  /**
   * The canonical integer codes for this output window.
   */
  vx_velox_primitive_view codes;
  /**
   * A borrowed prepared cursor for the dictionary values.
   */
  const vx_velox_export_cursor *values;
  /**
   * The number of dictionary values.
   */
  size_t values_length;
} vx_velox_dictionary_view;

/**
 * A constant block delivered to Velox.
 */
typedef struct {
  /**
   * Set this field to `sizeof(vx_velox_constant_view)`.
   */
  size_t struct_size;
  /**
   * The number of repeated logical values.
   */
  size_t length;
  /**
   * A borrowed prepared cursor with one canonical value.
   */
  const vx_velox_export_cursor *value;
} vx_velox_constant_view;

/**
 * A canonical packed Boolean block delivered to Velox.
 */
typedef struct {
  /**
   * Set this field to `sizeof(vx_velox_bool_view)`.
   */
  size_t struct_size;
  /**
   * The number of logical Boolean values.
   */
  size_t length;
  /**
   * The first packed value byte.
   */
  const uint8_t *values;
  /**
   * The number of readable value bytes.
   */
  size_t values_length;
  /**
   * The first logical value bit within `values`.
   */
  size_t values_bit_offset;
  /**
   * The validity representation.
   */
  vx_velox_validity_kind validity_kind;
  /**
   * The first validity byte when `validity_kind` is `Bitmap`.
   */
  const uint8_t *validity;
  /**
   * The number of readable validity bytes.
   */
  size_t validity_length;
  /**
   * The first logical validity bit within `validity`.
   */
  size_t validity_bit_offset;
  /**
   * Retains all pointers in this view.
   */
  vx_velox_buffer_owner buffers;
  /**
   * The guaranteed byte alignment of a non-empty value buffer.
   */
  size_t values_alignment;
  /**
   * The guaranteed byte alignment of a non-empty validity buffer.
   */
  size_t validity_alignment;
} vx_velox_bool_view;

/**
 * A canonical struct block delivered to Velox.
 */
typedef struct {
  /**
   * Set this field to `sizeof(vx_velox_struct_view)`.
   */
  size_t struct_size;
  /**
   * The number of logical struct values in this window.
   */
  size_t length;
  /**
   * The first logical row in each field cursor.
   */
  size_t offset;
  /**
   * Borrowed prepared cursors in field order.
   */
  const vx_velox_export_cursor *const *fields;
  /**
   * The number of field cursors.
   */
  size_t field_count;
  /**
   * The validity representation.
   */
  vx_velox_validity_kind validity_kind;
  /**
   * The first validity byte when `validity_kind` is `Bitmap`.
   */
  const uint8_t *validity;
  /**
   * The number of readable validity bytes.
   */
  size_t validity_length;
  /**
   * The first logical validity bit within `validity`.
   */
  size_t validity_bit_offset;
  /**
   * Retains the parent validity buffer.
   */
  vx_velox_buffer_owner buffers;
  /**
   * The guaranteed byte alignment of a non-empty validity buffer.
   */
  size_t validity_alignment;
} vx_velox_struct_view;

/**
 * A canonical list block delivered to Velox.
 */
typedef struct {
  /**
   * Set this field to `sizeof(vx_velox_list_view)`.
   */
  size_t struct_size;
  /**
   * The number of logical lists in this window.
   */
  size_t length;
  /**
   * One non-negative element offset per list. Values remain absolute against `elements`.
   */
  const int32_t *offsets;
  /**
   * One non-negative element count per list.
   */
  const int32_t *sizes;
  /**
   * A borrowed prepared cursor for all referenced elements.
   */
  const vx_velox_export_cursor *elements;
  /**
   * The number of values in the element cursor.
   */
  size_t elements_length;
  /**
   * The validity representation.
   */
  vx_velox_validity_kind validity_kind;
  /**
   * The first validity byte when `validity_kind` is `Bitmap`.
   */
  const uint8_t *validity;
  /**
   * The number of readable validity bytes.
   */
  size_t validity_length;
  /**
   * The first logical validity bit within `validity`.
   */
  size_t validity_bit_offset;
  /**
   * Retains the complete offsets, sizes, and parent validity allocations.
   */
  vx_velox_buffer_owner buffers;
  /**
   * The guaranteed byte alignment of a non-empty offsets buffer.
   */
  size_t offsets_alignment;
  /**
   * The guaranteed byte alignment of a non-empty sizes buffer.
   */
  size_t sizes_alignment;
  /**
   * The guaranteed byte alignment of a non-empty validity buffer.
   */
  size_t validity_alignment;
} vx_velox_list_view;

/**
 * A canonical map block delivered to Velox.
 */
typedef struct {
  /**
   * Set this field to `sizeof(vx_velox_map_view)`.
   */
  size_t struct_size;
  /**
   * The number of logical maps in this window.
   */
  size_t length;
  /**
   * One non-negative entry offset per map. Values remain absolute against the child cursors.
   */
  const int32_t *offsets;
  /**
   * One non-negative entry count per map.
   */
  const int32_t *sizes;
  /**
   * A borrowed prepared cursor for all map keys.
   */
  const vx_velox_export_cursor *keys;
  /**
   * A borrowed prepared cursor for all map values.
   */
  const vx_velox_export_cursor *values;
  /**
   * The number of entries in each child cursor.
   */
  size_t entries_length;
  /**
   * True when each map asserts sorted keys.
   */
  bool keys_sorted;
  /**
   * The validity representation.
   */
  vx_velox_validity_kind validity_kind;
  /**
   * The first validity byte when `validity_kind` is `Bitmap`.
   */
  const uint8_t *validity;
  /**
   * The number of readable validity bytes.
   */
  size_t validity_length;
  /**
   * The first logical validity bit within `validity`.
   */
  size_t validity_bit_offset;
  /**
   * Retains the complete offsets, sizes, and parent validity allocations.
   */
  vx_velox_buffer_owner buffers;
  /**
   * The guaranteed byte alignment of a non-empty offsets buffer.
   */
  size_t offsets_alignment;
  /**
   * The guaranteed byte alignment of a non-empty sizes buffer.
   */
  size_t sizes_alignment;
  /**
   * The guaranteed byte alignment of a non-empty validity buffer.
   */
  size_t validity_alignment;
} vx_velox_map_view;

/**
 * Host callbacks for Vortex array traversal.
 *
 * One array visit calls the matching callback synchronously. Shared tables can receive concurrent
 * callbacks from simultaneous visits. `last_error` must return the calling thread's most recent
 * error. The string must remain valid until the next callback on that thread. Callbacks must catch
 * foreign exceptions and must not unwind across this ABI. The host owns the context.
 */
typedef struct {
  /**
   * Set this field to `sizeof(vx_velox_visitor)`.
   */
  size_t struct_size;
  /**
   * Set this field to [`crate::VX_VELOX_ABI_VERSION`].
   */
  uint32_t abi_version;
  /**
   * An opaque callback context.
   */
  void *context;
  /**
   * Consume one canonical primitive block. Zero means success.
   */
  int32_t (*visit_primitive)(void *context, const vx_velox_primitive_view *view);
  /**
   * Return the last callback error as a null-terminated string.
   */
  const char *(*last_error)(void *context);
  /**
   * Consume one canonical variable-width binary block. Zero means success.
   */
  int32_t (*visit_varbin)(void *context, const vx_velox_varbin_view *view);
  /**
   * Consume one dictionary block. Zero means success.
   */
  int32_t (*visit_dictionary)(void *context, const vx_velox_dictionary_view *view);
  /**
   * Consume one constant block. Zero means success.
   */
  int32_t (*visit_constant)(void *context, const vx_velox_constant_view *view);
  /**
   * Consume one canonical packed Boolean block. Zero means success.
   */
  int32_t (*visit_bool)(void *context, const vx_velox_bool_view *view);
  /**
   * Consume one canonical struct block. Zero means success.
   */
  int32_t (*visit_struct)(void *context, const vx_velox_struct_view *view);
  /**
   * Consume one canonical list block. Zero means success.
   */
  int32_t (*visit_list)(void *context, const vx_velox_list_view *view);
  /**
   * Consume one canonical map block. Zero means success.
   */
  int32_t (*visit_map)(void *context, const vx_velox_map_view *view);
} vx_velox_visitor;

/**
 * A single-shot subset request for the semantic visitor.
 */
typedef struct {
  /**
   * Set this field to `sizeof(vx_velox_visit_request)`.
   */
  size_t struct_size;
  /**
   * Unique, increasing source positions. Null selects every row.
   */
  const uint64_t *rows;
  /**
   * The number of source positions.
   */
  size_t row_count;
} vx_velox_visit_request;

/**
 * Unsigned 8-bit integer type identifier.
 */
#define VX_VELOX_PTYPE_U8 0

/**
 * Unsigned 16-bit integer type identifier.
 */
#define VX_VELOX_PTYPE_U16 1

/**
 * Unsigned 32-bit integer type identifier.
 */
#define VX_VELOX_PTYPE_U32 2

/**
 * Unsigned 64-bit integer type identifier.
 */
#define VX_VELOX_PTYPE_U64 3

/**
 * Signed 8-bit integer type identifier.
 */
#define VX_VELOX_PTYPE_I8 4

/**
 * Signed 16-bit integer type identifier.
 */
#define VX_VELOX_PTYPE_I16 5

/**
 * Signed 32-bit integer type identifier.
 */
#define VX_VELOX_PTYPE_I32 6

/**
 * Signed 64-bit integer type identifier.
 */
#define VX_VELOX_PTYPE_I64 7

/**
 * 16-bit floating-point type identifier.
 */
#define VX_VELOX_PTYPE_F16 8

/**
 * 32-bit floating-point type identifier.
 */
#define VX_VELOX_PTYPE_F32 9

/**
 * 64-bit floating-point type identifier.
 */
#define VX_VELOX_PTYPE_F64 10

/**
 * Equality operator identifier.
 */
#define VX_VELOX_OPERATOR_EQ 0

/**
 * Inequality operator identifier.
 */
#define VX_VELOX_OPERATOR_NOT_EQ 1

/**
 * Greater-than operator identifier.
 */
#define VX_VELOX_OPERATOR_GT 2

/**
 * Greater-than-or-equal operator identifier.
 */
#define VX_VELOX_OPERATOR_GTE 3

/**
 * Less-than operator identifier.
 */
#define VX_VELOX_OPERATOR_LT 4

/**
 * Less-than-or-equal operator identifier.
 */
#define VX_VELOX_OPERATOR_LTE 5

/**
 * Kleene logical AND operator identifier.
 */
#define VX_VELOX_OPERATOR_KLEENE_AND 6

/**
 * Kleene logical OR operator identifier.
 */
#define VX_VELOX_OPERATOR_KLEENE_OR 7

/**
 * Include every row.
 */
#define VX_VELOX_SELECTION_ALL 0

/**
 * Include the supplied row indexes.
 */
#define VX_VELOX_SELECTION_INCLUDE 1

/**
 * Exclude the supplied row indexes.
 */
#define VX_VELOX_SELECTION_EXCLUDE 2

/**
 * Unsigned 8-bit primitive identifier.
 */
#define VX_VELOX_PRIMITIVE_U8 0

/**
 * Unsigned 16-bit primitive identifier.
 */
#define VX_VELOX_PRIMITIVE_U16 1

/**
 * Unsigned 32-bit primitive identifier.
 */
#define VX_VELOX_PRIMITIVE_U32 2

/**
 * Unsigned 64-bit primitive identifier.
 */
#define VX_VELOX_PRIMITIVE_U64 3

/**
 * Signed 8-bit primitive identifier.
 */
#define VX_VELOX_PRIMITIVE_I8 4

/**
 * Signed 16-bit primitive identifier.
 */
#define VX_VELOX_PRIMITIVE_I16 5

/**
 * Signed 32-bit primitive identifier.
 */
#define VX_VELOX_PRIMITIVE_I32 6

/**
 * Signed 64-bit primitive identifier.
 */
#define VX_VELOX_PRIMITIVE_I64 7

/**
 * IEEE 754 binary16 primitive identifier.
 */
#define VX_VELOX_PRIMITIVE_F16 8

/**
 * IEEE 754 binary32 primitive identifier.
 */
#define VX_VELOX_PRIMITIVE_F32 9

/**
 * IEEE 754 binary64 primitive identifier.
 */
#define VX_VELOX_PRIMITIVE_F64 10

/**
 * Signed 128-bit primitive identifier.
 */
#define VX_VELOX_PRIMITIVE_I128 11

/**
 * The type is not nullable.
 */
#define VX_VELOX_VALIDITY_NON_NULLABLE 0

/**
 * Every value is valid.
 */
#define VX_VELOX_VALIDITY_ALL_VALID 1

/**
 * Every value is null.
 */
#define VX_VELOX_VALIDITY_ALL_INVALID 2

/**
 * A packed bitmap contains one valid bit per value.
 */
#define VX_VELOX_VALIDITY_BITMAP 3

/**
 * Identifies UTF-8 values.
 */
#define VX_VELOX_VARBIN_UTF8 0

/**
 * Identifies arbitrary binary values.
 */
#define VX_VELOX_VARBIN_BINARY 1

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/**
 * Return the adapter ABI version.
 */
uint32_t vx_velox_abi_version(void);

/**
 * Return the capabilities implemented by this adapter build.
 */
uint64_t vx_velox_capabilities(void);

/**
 * Return the message stored in an adapter error.
 *
 * # Safety
 *
 * `error` must point to a live error handle.
 */
vx_velox_view vx_velox_error_message(const vx_velox_error *error);

/**
 * Free an adapter error.
 *
 * # Safety
 *
 * `error` must be null or an owned error handle.
 */
void vx_velox_error_free(const vx_velox_error *error);

/**
 * Create a default Vortex session for Velox.
 */
vx_velox_session *vx_velox_session_new(void);

/**
 * Clone a Vortex session.
 *
 * # Safety
 *
 * `session` must point to a live Vortex session.
 */
vx_velox_session *vx_velox_session_clone(const vx_velox_session *session);

/**
 * Free a Vortex session.
 *
 * # Safety
 *
 * `session` must be null or an owned session handle.
 */
void vx_velox_session_free(const vx_velox_session *session);

/**
 * Create a primitive dtype for a list literal.
 *
 * # Safety
 *
 * `error_out` must be null or valid for one error pointer.
 */
const vx_velox_dtype *vx_velox_dtype_new_primitive(vx_velox_ptype ptype,
                                                   bool nullable,
                                                   vx_velox_error **error_out);

/**
 * Free a dtype.
 *
 * # Safety
 *
 * `dtype` must be null or an owned dtype handle.
 */
void vx_velox_dtype_free(const vx_velox_dtype *dtype);

/**
 * Create a Boolean scalar.
 */
vx_velox_scalar *vx_velox_scalar_new_bool(bool value, bool nullable);

/**
 *Create an i8 scalar.
 */
vx_velox_scalar *vx_velox_scalar_new_i8(int8_t value, bool nullable);

/**
 *Create an i16 scalar.
 */
vx_velox_scalar *vx_velox_scalar_new_i16(int16_t value, bool nullable);

/**
 *Create an i32 scalar.
 */
vx_velox_scalar *vx_velox_scalar_new_i32(int32_t value, bool nullable);

/**
 * Create a date scalar that stores days since the Unix epoch.
 *
 * # Safety
 *
 * `error_out` must be null or valid for one error pointer.
 */
vx_velox_scalar *vx_velox_scalar_new_date_days(int32_t value,
                                               bool nullable,
                                               vx_velox_error **error_out);

/**
 *Create an i64 scalar.
 */
vx_velox_scalar *vx_velox_scalar_new_i64(int64_t value, bool nullable);

/**
 *Create an f32 scalar.
 */
vx_velox_scalar *vx_velox_scalar_new_f32(float value, bool nullable);

/**
 *Create an f64 scalar.
 */
vx_velox_scalar *vx_velox_scalar_new_f64(double value, bool nullable);

/**
 * Create a UTF-8 scalar.
 *
 * # Safety
 *
 * `value` and `error_out` must satisfy the adapter header contract.
 */
vx_velox_scalar *vx_velox_scalar_new_utf8(vx_velox_view value,
                                          bool nullable,
                                          vx_velox_error **error_out);

/**
 * Create a binary scalar.
 *
 * # Safety
 *
 * `data` must identify `length` bytes. `error_out` must be null or valid.
 */
vx_velox_scalar *vx_velox_scalar_new_binary(const uint8_t *data,
                                            size_t length,
                                            bool nullable,
                                            vx_velox_error **error_out);

/**
 * Create a list scalar.
 *
 * # Safety
 *
 * Every pointer must satisfy the adapter header contract.
 */
vx_velox_scalar *vx_velox_scalar_new_list(const vx_velox_dtype *element_dtype,
                                          const vx_velox_scalar *const *elements,
                                          size_t length,
                                          bool nullable,
                                          vx_velox_error **error_out);

/**
 * Free a scalar.
 *
 * # Safety
 *
 * `scalar` must be null or an owned scalar handle.
 */
void vx_velox_scalar_free(const vx_velox_scalar *scalar);

/**
 * Create a literal expression.
 *
 * # Safety
 *
 * `scalar` must point to a live scalar. `error_out` must be null or valid.
 */
vx_velox_expression *vx_velox_expression_literal(const vx_velox_scalar *scalar,
                                                 vx_velox_error **error_out);

/**
 * Create a root expression.
 */
vx_velox_expression *vx_velox_expression_root(void);

/**
 * Create a field expression.
 *
 * # Safety
 *
 * `child` must point to a live expression. `name` must identify valid UTF-8.
 */
vx_velox_expression *vx_velox_expression_get_item(vx_velox_view name,
                                                  const vx_velox_expression *child);

/**
 * Create a binary expression.
 *
 * # Safety
 *
 * Both operands must point to live expressions. `error_out` must be null or valid.
 */
vx_velox_expression *vx_velox_expression_binary(vx_velox_binary_operator operator_,
                                                const vx_velox_expression *left,
                                                const vx_velox_expression *right,
                                                vx_velox_error **error_out);

/**
 * Create a conjunction from expressions.
 *
 * # Safety
 *
 * `expressions` must identify `length` live expression pointers.
 */
vx_velox_expression *vx_velox_expression_and(const vx_velox_expression *const *expressions,
                                             size_t length);

/**
 * Create a disjunction from expressions.
 *
 * # Safety
 *
 * `expressions` must identify `length` live expression pointers.
 */
vx_velox_expression *vx_velox_expression_or(const vx_velox_expression *const *expressions,
                                            size_t length);

/**
 * Create a logical negation.
 *
 * # Safety
 *
 * `child` must point to a live expression.
 */
vx_velox_expression *vx_velox_expression_not(const vx_velox_expression *child);

/**
 * Create a null test.
 *
 * # Safety
 *
 * `child` must point to a live expression.
 */
vx_velox_expression *vx_velox_expression_is_null(const vx_velox_expression *child);

/**
 * Create a list membership test.
 *
 * # Safety
 *
 * Both operands must point to live expressions.
 */
vx_velox_expression *vx_velox_expression_list_contains(const vx_velox_expression *list,
                                                       const vx_velox_expression *value);

/**
 * Return whether Vortex can push down an integer value set without generic expansion.
 */
bool vx_velox_can_push_down_integer_values(size_t value_count);

/**
 * Free an expression.
 *
 * # Safety
 *
 * `expression` must be null or an owned expression handle.
 */
void vx_velox_expression_free(const vx_velox_expression *expression);

/**
 * Free a data source.
 *
 * # Safety
 *
 * `data_source` must be null or an owned data-source handle.
 */
void vx_velox_data_source_free(const vx_velox_data_source *data_source);

/**
 * Start a scan through the stable adapter options.
 *
 * # Safety
 *
 * Every pointer must satisfy the adapter header contract.
 */
vx_velox_scan *vx_velox_data_source_scan(const vx_velox_data_source *data_source,
                                         const vx_velox_scan_options *options,
                                         vx_velox_error **error_out);

/**
 * Free a scan.
 *
 * # Safety
 *
 * `scan` must be null or an owned scan handle.
 */
void vx_velox_scan_free(const vx_velox_scan *scan);

/**
 * Return the next partition from a scan.
 *
 * # Safety
 *
 * `scan` must point to a live scan. `error_out` must be null or valid.
 */
vx_velox_partition *vx_velox_scan_next_partition(vx_velox_scan *scan, vx_velox_error **error_out);

/**
 * Free a partition.
 *
 * # Safety
 *
 * `partition` must be null or an owned partition handle.
 */
void vx_velox_partition_free(const vx_velox_partition *partition);

/**
 * Return the next array from a partition.
 *
 * # Safety
 *
 * `partition` must point to a live partition. `error_out` must be null or valid.
 */
const vx_velox_array *vx_velox_partition_next(vx_velox_partition *partition,
                                              vx_velox_error **error_out);

/**
 * Free an array.
 *
 * # Safety
 *
 * `array` must be null or an owned array handle.
 */
void vx_velox_array_free(const vx_velox_array *array);

/**
 * Return an array length.
 *
 * # Safety
 *
 * `array` must point to a live array.
 */
size_t vx_velox_array_len(const vx_velox_array *array);

/**
 * Slice an array.
 *
 * # Safety
 *
 * `array` must point to a live array. `error_out` must be null or valid.
 */
const vx_velox_array *vx_velox_array_slice(const vx_velox_array *array,
                                           size_t begin,
                                           size_t end,
                                           vx_velox_error **error_out);

/**
 * Return one struct field with the supplied session.
 *
 * # Safety
 *
 * The session and array pointers must identify live handles. `error_out` must be null or valid.
 */
const vx_velox_array *vx_velox_array_get_field(const vx_velox_session *session,
                                               const vx_velox_array *array,
                                               size_t index,
                                               vx_velox_error **error_out);

/**
 * Return the invalid value count with the supplied session.
 *
 * # Safety
 *
 * The session and array pointers must identify live handles. `error_out` must be null or valid.
 */
size_t vx_velox_array_invalid_count(const vx_velox_session *session,
                                    const vx_velox_array *array,
                                    vx_velox_error **error_out);

/**
 * Export one Vortex array through the Arrow C Data Interface.
 *
 * The caller owns both outputs and must call their release callbacks. The memory callbacks reserve
 * a conservative payload charge before Arrow conversion. The adapter refunds the difference after
 * it knows the retained payload capacities. It requests a deficit before it returns the outputs.
 * The charge excludes schema and small FFI metadata.
 *
 * # Safety
 *
 * The session and array pointers must identify live handles. `memory_callbacks` must identify its
 * declared `struct_size` bytes for this call. Its callback context must remain valid through every
 * retained reference. Its callbacks and returned error strings must satisfy the header contract
 * and must not unwind. Both output pointers must identify uninitialized writable structures.
 * `error_out` must be null or identify writable storage for one error pointer.
 */
int32_t vx_velox_array_export_arrow(const vx_velox_session *session,
                                    const vx_velox_array *array,
                                    const vx_velox_arrow_memory_callbacks *memory_callbacks,
                                    ArrowSchema *schema_out,
                                    ArrowArray *array_out,
                                    vx_velox_error **error_out);

/**
 * Create a struct projection from the supplied field names.
 *
 * The returned expression stays owned by the caller.
 *
 * # Safety
 *
 * `names` must be null when `len` is zero or point to `len` valid views.
 * Every view must remain valid for this call.
 * `error_out` must be null or point to writable storage. No input operation can unwind.
 */
vx_velox_expression *vx_velox_expression_select(const vx_velox_view *names,
                                                size_t len,
                                                vx_velox_error **error_out);

/**
 * Create a struct projection with an absolute file-row index as its first field.
 *
 * The remaining fields select the supplied names from the scan root. The
 * returned expression stays owned by the caller.
 *
 * # Safety
 *
 * `names` must be null when `len` is zero or point to `len` valid views.
 * Every view and `row_index_name` must remain valid for this call.
 * `error_out` must be null or point to writable storage. No input operation can unwind.
 */
vx_velox_expression *vx_velox_expression_select_with_row_index(const vx_velox_view *names,
                                                               size_t len,
                                                               vx_velox_view row_index_name,
                                                               vx_velox_error **error_out);

/**
 * Create a Vortex positional reader from Velox callbacks.
 *
 * # Safety
 *
 * `callbacks` must point to a valid callback structure. Every callback and its context must be
 * thread-safe and must not unwind. `error_out` must be null or valid for one error pointer.
 */
vx_velox_read_at *vx_velox_read_at_new(const vx_velox_read_at_callbacks *callbacks,
                                       vx_velox_error **error_out);

/**
 * Free a Vortex positional reader.
 *
 * # Safety
 *
 * `reader` must be null or a pointer returned by [`vx_velox_read_at_new`].
 */
void vx_velox_read_at_free(vx_velox_read_at *reader);

/**
 * Return the size of a callback-backed source.
 *
 * This entry point validates the host callback contract before file-reader code consumes the
 * source.
 *
 * # Safety
 *
 * `reader` must point to a live reader. `error_out` must be null or valid for one error pointer.
 */
uint64_t vx_velox_read_at_size(const vx_velox_read_at *reader, vx_velox_error **error_out);

/**
 * Export an opened source schema through the Arrow C Data Interface.
 *
 * The caller owns the output and must invoke its release callback.
 *
 * # Safety
 *
 * `source` must point to a live source. `schema_out` must identify uninitialized writable
 * storage. `error_out` must be null or valid for one error pointer.
 */
int32_t vx_velox_source_export_schema(const vx_velox_source *source,
                                      ArrowSchema *schema_out,
                                      vx_velox_error **error_out);

/**
 * Open a Vortex source through a callback reader.
 *
 * The source retains the session and reader state. The caller can free both input handles after
 * this function returns.
 *
 * # Safety
 *
 * `session` and `reader` must point to live handles. `error_out` must be null or valid for one
 * error pointer.
 */
vx_velox_source *vx_velox_source_new(const vx_velox_session *session,
                                     const vx_velox_read_at *reader,
                                     vx_velox_error **error_out);

/**
 * Free a callback-backed Vortex source and release its callback-owned input buffers.
 *
 * # Safety
 *
 * `source` must be null or a pointer returned by [`vx_velox_source_new`].
 */
void vx_velox_source_free(vx_velox_source *source);

/**
 * Return the file row count.
 *
 * # Safety
 *
 * `source` must point to a live source.
 */
uint64_t vx_velox_source_row_count(const vx_velox_source *source);

/**
 * Return the file size in bytes.
 *
 * # Safety
 *
 * `source` must point to a live source.
 */
uint64_t vx_velox_source_file_size(const vx_velox_source *source);

/**
 * Return the number of natural row splits.
 *
 * # Safety
 *
 * `source` must point to a live source.
 */
size_t vx_velox_source_natural_split_count(const vx_velox_source *source);

/**
 * Write one natural row split.
 *
 * # Safety
 *
 * `source` must point to a live source. `split_out` must point to a structure with a valid size.
 * `error_out` must be null or valid for one error pointer.
 */
int32_t vx_velox_source_natural_split_at(const vx_velox_source *source,
                                         size_t index,
                                         vx_velox_natural_split *split_out,
                                         vx_velox_error **error_out);

/**
 * Evaluate whether natural splits cannot match an expression.
 *
 * Each output byte is one when the matching split cannot produce a true expression result. Zero
 * means that the split can match or that available statistics cannot prove exclusion.
 *
 * # Safety
 *
 * `source` and `expression` must point to live handles. `pruned_out` must identify `split_count`
 * writable bytes unless `split_count` is zero. `error_out` must be null or valid for one error
 * pointer.
 */
int32_t vx_velox_source_prune_natural_splits(const vx_velox_source *source,
                                             const vx_velox_expression *expression,
                                             size_t first_split,
                                             size_t split_count,
                                             uint8_t *pruned_out,
                                             vx_velox_error **error_out);

/**
 * Create a standard Vortex data source for this file.
 *
 * The caller owns the returned handle and must free it through `vx_data_source_free`.
 *
 * # Safety
 *
 * `source` must point to a live source. `error_out` must be null or valid for one error pointer.
 */
const vx_velox_data_source *vx_velox_source_data_source(const vx_velox_source *source,
                                                        vx_velox_error **error_out);

/**
 * Create one export cursor for several Velox output windows.
 *
 * # Safety
 *
 * The session and array pointers must identify live handles.
 * The memory callbacks must identify a complete, thread-safe callback table.
 * `error_out` must be null or valid.
 */
vx_velox_export_cursor *vx_velox_export_cursor_new(const vx_velox_session *session,
                                                   const vx_velox_array *array,
                                                   const vx_velox_arrow_memory_callbacks *memory_callbacks,
                                                   vx_velox_error **error_out);

/**
 * Free one export cursor.
 *
 * # Safety
 *
 * The pointer must be null or come from [`vx_velox_export_cursor_new`].
 */
void vx_velox_export_cursor_free(vx_velox_export_cursor *cursor);

/**
 * Visit one contiguous range from a retained export cursor.
 *
 * # Safety
 *
 * The cursor and visitor pointers must remain live until this call returns.
 * Concurrent calls are valid. The caller must not free the cursor before all calls return.
 */
int32_t vx_velox_export_cursor_visit(const vx_velox_export_cursor *cursor,
                                     size_t offset,
                                     size_t length,
                                     const vx_velox_visitor *visitor,
                                     vx_velox_error **error_out);

/**
 * Visit one Vortex array through host semantic callbacks.
 *
 * The request selects source positions once. Callback block positions are compact and follow the
 * request order.
 *
 * # Safety
 *
 * Every pointer must be null or valid for the documented access. The array and session handles
 * must remain live until this call returns.
 */
int32_t vx_velox_array_visit(const vx_velox_session *session,
                             const vx_velox_array *array,
                             const vx_velox_visit_request *request,
                             const vx_velox_visitor *visitor,
                             vx_velox_error **error_out);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
