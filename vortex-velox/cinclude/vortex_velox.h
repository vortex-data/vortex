// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "vortex.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Velox must call only vx_velox_* adapter symbols. The static archive can also
 * contain general vx_* symbols from linked Vortex FFI objects.
 */

#define VX_VELOX_ABI_VERSION                      1u
#define VX_VELOX_CAPABILITY_BATCH_READ            (UINT64_C(1) << 0)
#define VX_VELOX_CAPABILITY_CALLBACK_SOURCE       (UINT64_C(1) << 1)
#define VX_VELOX_CAPABILITY_NATURAL_SPLITS        (UINT64_C(1) << 2)
#define VX_VELOX_CAPABILITY_PRIMITIVE_VISITOR     (UINT64_C(1) << 3)
#define VX_VELOX_CAPABILITY_ARROW_SCHEMA          (UINT64_C(1) << 4)
#define VX_VELOX_CAPABILITY_ARRAY_ARROW_EXPORT    (UINT64_C(1) << 5)
#define VX_VELOX_CAPABILITY_ROW_INDEX_PROJECTION  (UINT64_C(1) << 6)
#define VX_VELOX_CAPABILITY_NATURAL_SPLIT_PRUNING (UINT64_C(1) << 7)
/* Vortex checks cancellation before each host read callback. */
#define VX_VELOX_CAPABILITY_READ_CANCELLATION     (UINT64_C(1) << 8)

typedef struct vx_velox_read_at vx_velox_read_at;
typedef struct vx_velox_source vx_velox_source;

typedef uint32_t vx_velox_ptype;
#define VX_VELOX_PTYPE_U8  UINT32_C(0)
#define VX_VELOX_PTYPE_U16 UINT32_C(1)
#define VX_VELOX_PTYPE_U32 UINT32_C(2)
#define VX_VELOX_PTYPE_U64 UINT32_C(3)
#define VX_VELOX_PTYPE_I8  UINT32_C(4)
#define VX_VELOX_PTYPE_I16 UINT32_C(5)
#define VX_VELOX_PTYPE_I32 UINT32_C(6)
#define VX_VELOX_PTYPE_I64 UINT32_C(7)
#define VX_VELOX_PTYPE_F16 UINT32_C(8)
#define VX_VELOX_PTYPE_F32 UINT32_C(9)
#define VX_VELOX_PTYPE_F64 UINT32_C(10)

typedef uint32_t vx_velox_binary_operator;
#define VX_VELOX_OPERATOR_EQ          UINT32_C(0)
#define VX_VELOX_OPERATOR_NOT_EQ      UINT32_C(1)
#define VX_VELOX_OPERATOR_GT          UINT32_C(2)
#define VX_VELOX_OPERATOR_GTE         UINT32_C(3)
#define VX_VELOX_OPERATOR_LT          UINT32_C(4)
#define VX_VELOX_OPERATOR_LTE         UINT32_C(5)
#define VX_VELOX_OPERATOR_KLEENE_AND  UINT32_C(6)
#define VX_VELOX_OPERATOR_KLEENE_OR   UINT32_C(7)

typedef uint32_t vx_velox_scan_selection_include;
#define VX_VELOX_SELECTION_ALL     UINT32_C(0)
#define VX_VELOX_SELECTION_INCLUDE UINT32_C(1)
#define VX_VELOX_SELECTION_EXCLUDE UINT32_C(2)

typedef struct vx_velox_scan_selection {
    const uint64_t *indices;
    size_t length;
    vx_velox_scan_selection_include include;
} vx_velox_scan_selection;

typedef struct vx_velox_scan_options {
    size_t struct_size;
    uint32_t abi_version;
    const vx_expression *projection;
    const vx_expression *filter;
    uint64_t row_range_begin;
    uint64_t row_range_end;
    vx_velox_scan_selection selection;
    uint64_t limit;
    bool ordered;
} vx_velox_scan_options;

typedef struct vx_velox_read_request {
    size_t struct_size;
    uint64_t offset;
    size_t length;
    size_t alignment;
} vx_velox_read_request;

typedef struct vx_velox_buffer {
    size_t struct_size;
    const uint8_t *data;
    size_t length;
    void *owner;
    void (*release)(void *owner);
} vx_velox_buffer;

/**
 * Callbacks for positional reads through the host engine.
 *
 * Vortex can call size, read_ranges, is_cancelled, and last_error concurrently.
 * The context and callbacks must be thread-safe. A non-zero concurrency value
 * limits requests in one callback and gives Vortex a scheduling hint. It is not
 * a synchronization guarantee.
 *
 * last_error must return the calling thread's most recent callback error. The
 * returned string must remain valid until the next callback on that thread.
 * Every callback must catch C++ exceptions. No callback can unwind across this
 * C ABI. release_context runs after the final callback and can run on any
 * thread.
 *
 * Vortex checks is_cancelled before each read_ranges call. The check does not
 * interrupt an active callback or CPU work.
 */
typedef struct vx_velox_read_at_callbacks {
    size_t struct_size;
    uint32_t abi_version;
    void *context;
    int32_t (*size)(void *context, uint64_t *size_out);
    int32_t (*read_ranges)(void *context,
                           const vx_velox_read_request *requests,
                           size_t request_count,
                           vx_velox_buffer *outputs);
    const char *(*last_error)(void *context);
    void (*release_context)(void *context);
    int32_t (*is_cancelled)(void *context);
    size_t concurrency;
} vx_velox_read_at_callbacks;

typedef struct vx_velox_natural_split {
    size_t struct_size;
    uint64_t row_begin;
    uint64_t row_end;
} vx_velox_natural_split;

typedef uint32_t vx_velox_primitive_type;
#define VX_VELOX_PRIMITIVE_U8  UINT32_C(0)
#define VX_VELOX_PRIMITIVE_U16 UINT32_C(1)
#define VX_VELOX_PRIMITIVE_U32 UINT32_C(2)
#define VX_VELOX_PRIMITIVE_U64 UINT32_C(3)
#define VX_VELOX_PRIMITIVE_I8  UINT32_C(4)
#define VX_VELOX_PRIMITIVE_I16 UINT32_C(5)
#define VX_VELOX_PRIMITIVE_I32 UINT32_C(6)
#define VX_VELOX_PRIMITIVE_I64 UINT32_C(7)
#define VX_VELOX_PRIMITIVE_F16 UINT32_C(8)
#define VX_VELOX_PRIMITIVE_F32 UINT32_C(9)
#define VX_VELOX_PRIMITIVE_F64 UINT32_C(10)

typedef uint32_t vx_velox_validity_kind;
#define VX_VELOX_VALIDITY_NON_NULLABLE UINT32_C(0)
#define VX_VELOX_VALIDITY_ALL_VALID    UINT32_C(1)
#define VX_VELOX_VALIDITY_ALL_INVALID  UINT32_C(2)
#define VX_VELOX_VALIDITY_BITMAP       UINT32_C(3)

typedef struct vx_velox_buffer_owner {
    size_t struct_size;
    const void *owner;
    void (*retain)(const void *owner);
    void (*release)(const void *owner);
    size_t retained_bytes;
} vx_velox_buffer_owner;

/**
 * A compact primitive payload and its owner.
 *
 * Vortex copies values into an allocation with uint64_t alignment. The values
 * allocation rounds values_length up to that alignment. Vortex copies a bitmap
 * into a compact byte allocation with validity_bit_offset set to zero.
 * buffers.retained_bytes is the exact sum of these allocation sizes.
 *
 * The pointers remain valid through visit_primitive. The host must call retain
 * before it stores a pointer beyond that callback.
 */
typedef struct vx_velox_primitive_view {
    size_t struct_size;
    vx_velox_primitive_type primitive_type;
    size_t length;
    const uint8_t *values;
    size_t values_length;
    vx_velox_validity_kind validity_kind;
    const uint8_t *validity;
    size_t validity_length;
    size_t validity_bit_offset;
    vx_velox_buffer_owner buffers;
    size_t values_alignment;
    size_t validity_alignment;
} vx_velox_primitive_view;

typedef struct vx_velox_visit_request {
    size_t struct_size;
    const uint64_t *rows;
    size_t row_count;
} vx_velox_visit_request;

/**
 * Host callbacks for one Vortex array visit.
 *
 * One array visit calls visit_primitive synchronously. If the host shares this
 * table between simultaneous visits, callbacks can occur concurrently.
 * last_error returns the calling thread's most recent visitor error. Its string
 * remains valid until the next callback on that thread.
 *
 * Every callback must catch C++ exceptions. No callback can unwind across this
 * C ABI. The host owns context and must keep it live until each visit returns.
 */
typedef struct vx_velox_visitor {
    size_t struct_size;
    uint32_t abi_version;
    void *context;
    int32_t (*visit_primitive)(void *context, const vx_velox_primitive_view *view);
    const char *(*last_error)(void *context);
} vx_velox_visitor;

/**
 * Host memory callbacks for one Arrow C Data export.
 *
 * Before Arrow conversion, Vortex requests a conservative reservation through
 * report_allocation. A rejection stops the export before Arrow allocations.
 * Vortex calls report_free after conversion to refund unused reservation bytes.
 * The remaining charge equals retained Arrow payload capacities. It excludes
 * the schema and small Arrow C Data metadata allocations.
 * If the actual charge exceeds the reservation, Vortex requests the difference
 * before it returns outputs. A rejection aborts the export and frees the data.
 *
 * A final Arrow release calls report_free for the remaining charge. It also
 * calls release_context. These calls can occur on any thread. All callbacks
 * must be thread-safe and must not unwind across the C ABI.
 *
 * last_error returns the calling thread's most recent allocation error. The
 * string remains valid until the next callback on that thread.
 */
typedef struct vx_velox_arrow_memory_callbacks {
    size_t struct_size;
    uint32_t abi_version;
    void *context;
    void (*retain_context)(void *context);
    void (*release_context)(void *context);
    int32_t (*report_allocation)(void *context, size_t retained_bytes);
    void (*report_free)(void *context, size_t retained_bytes);
    const char *(*last_error)(void *context);
} vx_velox_arrow_memory_callbacks;

uint32_t vx_velox_abi_version(void);
uint64_t vx_velox_capabilities(void);

vx_view vx_velox_error_message(const vx_error *error);
void vx_velox_error_free(const vx_error *error);
vx_session *vx_velox_session_new(void);
void vx_velox_session_free(const vx_session *session);

const vx_dtype *vx_velox_dtype_new_primitive(vx_velox_ptype ptype,
                                             bool nullable,
                                             vx_error **error_out);
void vx_velox_dtype_free(const vx_dtype *dtype);
vx_scalar *vx_velox_scalar_new_bool(bool value, bool nullable);
vx_scalar *vx_velox_scalar_new_i8(int8_t value, bool nullable);
vx_scalar *vx_velox_scalar_new_i16(int16_t value, bool nullable);
vx_scalar *vx_velox_scalar_new_i32(int32_t value, bool nullable);
vx_scalar *vx_velox_scalar_new_i64(int64_t value, bool nullable);
vx_scalar *vx_velox_scalar_new_f32(float value, bool nullable);
vx_scalar *vx_velox_scalar_new_f64(double value, bool nullable);
vx_scalar *vx_velox_scalar_new_utf8(vx_view value, bool nullable, vx_error **error_out);
vx_scalar *
vx_velox_scalar_new_binary(const uint8_t *data, size_t length, bool nullable, vx_error **error_out);
vx_scalar *vx_velox_scalar_new_list(const vx_dtype *element_dtype,
                                    const vx_scalar *const *elements,
                                    size_t length,
                                    bool nullable,
                                    vx_error **error_out);
void vx_velox_scalar_free(const vx_scalar *scalar);

vx_expression *vx_velox_expression_root(void);
vx_expression *vx_velox_expression_literal(const vx_scalar *scalar, vx_error **error_out);
vx_expression *vx_velox_expression_get_item(vx_view name, const vx_expression *child);
vx_expression *vx_velox_expression_binary(vx_velox_binary_operator operation,
                                          const vx_expression *left,
                                          const vx_expression *right,
                                          vx_error **error_out);
vx_expression *vx_velox_expression_and(const vx_expression *const *expressions, size_t length);
vx_expression *vx_velox_expression_or(const vx_expression *const *expressions, size_t length);
vx_expression *vx_velox_expression_not(const vx_expression *child);
vx_expression *vx_velox_expression_is_null(const vx_expression *child);
vx_expression *vx_velox_expression_list_contains(const vx_expression *list, const vx_expression *value);
void vx_velox_expression_free(const vx_expression *expression);
vx_expression *vx_velox_expression_select_with_row_index(const vx_view *names,
                                                         size_t length,
                                                         vx_view row_index_name,
                                                         vx_error **error_out);

/*
 * On success, the reader owns context and calls release_context once. On
 * failure, the caller still owns context.
 */
vx_velox_read_at *vx_velox_read_at_new(const vx_velox_read_at_callbacks *callbacks, vx_error **error_out);
void vx_velox_read_at_free(vx_velox_read_at *reader);
uint64_t vx_velox_read_at_size(const vx_velox_read_at *reader, vx_error **error_out);

vx_velox_source *
vx_velox_source_new(const vx_session *session, const vx_velox_read_at *reader, vx_error **error_out);
void vx_velox_source_free(vx_velox_source *source);
uint64_t vx_velox_source_row_count(const vx_velox_source *source);
uint64_t vx_velox_source_file_size(const vx_velox_source *source);
int32_t vx_velox_source_export_schema(const vx_velox_source *source,
                                      FFI_ArrowSchema *schema_out,
                                      vx_error **error_out);
size_t vx_velox_source_natural_split_count(const vx_velox_source *source);
int32_t vx_velox_source_natural_split_at(const vx_velox_source *source,
                                         size_t index,
                                         vx_velox_natural_split *split_out,
                                         vx_error **error_out);
int32_t vx_velox_source_prune_natural_splits(const vx_velox_source *source,
                                             const vx_expression *expression,
                                             size_t first_split,
                                             size_t split_count,
                                             uint8_t *pruned_out,
                                             vx_error **error_out);
const vx_data_source *vx_velox_source_data_source(const vx_velox_source *source, vx_error **error_out);

void vx_velox_data_source_free(const vx_data_source *data_source);
vx_scan *vx_velox_data_source_scan(const vx_data_source *data_source,
                                   const vx_velox_scan_options *options,
                                   vx_error **error_out);
void vx_velox_scan_free(const vx_scan *scan);
vx_partition *vx_velox_scan_next_partition(vx_scan *scan, vx_error **error_out);
void vx_velox_partition_free(const vx_partition *partition);
const vx_array *vx_velox_partition_next(vx_partition *partition, vx_error **error_out);
void vx_velox_array_free(const vx_array *array);
size_t vx_velox_array_len(const vx_array *array);
const vx_array *vx_velox_array_slice(const vx_array *array, size_t begin, size_t end, vx_error **error_out);
const vx_array *vx_velox_array_get_field(const vx_session *session,
                                         const vx_array *array,
                                         size_t index,
                                         vx_error **error_out);
size_t vx_velox_array_invalid_count(const vx_session *session, const vx_array *array, vx_error **error_out);
int32_t vx_velox_array_visit(const vx_session *session,
                             const vx_array *array,
                             const vx_velox_visit_request *request,
                             const vx_velox_visitor *visitor,
                             vx_error **error_out);
int32_t vx_velox_array_export_arrow(const vx_session *session,
                                    const vx_array *array,
                                    const vx_velox_arrow_memory_callbacks *memory_callbacks,
                                    FFI_ArrowSchema *schema_out,
                                    FFI_ArrowArray *array_out,
                                    vx_error **error_out);

#ifdef __cplusplus
}
#endif
