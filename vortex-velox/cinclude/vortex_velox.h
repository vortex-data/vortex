// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

struct ArrowSchema;
struct ArrowArray;

#ifdef __cplusplus
extern "C" {
#endif

/* Velox calls only vx_velox_* adapter symbols. */

#define VX_VELOX_ABI_VERSION                      5u
#define VX_VELOX_CAPABILITY_BATCH_READ            (UINT64_C(1) << 0)
#define VX_VELOX_CAPABILITY_CALLBACK_SOURCE       (UINT64_C(1) << 1)
#define VX_VELOX_CAPABILITY_NATURAL_SPLITS        (UINT64_C(1) << 2)
#define VX_VELOX_CAPABILITY_PRIMITIVE_VISITOR     (UINT64_C(1) << 3)
#define VX_VELOX_CAPABILITY_ARROW_SCHEMA          (UINT64_C(1) << 4)
#define VX_VELOX_CAPABILITY_ARRAY_ARROW_EXPORT    (UINT64_C(1) << 5)
#define VX_VELOX_CAPABILITY_ROW_INDEX_PROJECTION  (UINT64_C(1) << 6)
#define VX_VELOX_CAPABILITY_NATURAL_SPLIT_PRUNING (UINT64_C(1) << 7)
/* Vortex checks cancellation before each host read callback. */
#define VX_VELOX_CAPABILITY_READ_CANCELLATION  (UINT64_C(1) << 8)
#define VX_VELOX_CAPABILITY_EXPORT_CURSOR      (UINT64_C(1) << 9)
#define VX_VELOX_CAPABILITY_PLAIN_PROJECTION   (UINT64_C(1) << 10)
#define VX_VELOX_CAPABILITY_VARBIN_VISITOR     (UINT64_C(1) << 11)
#define VX_VELOX_CAPABILITY_DICTIONARY_VISITOR (UINT64_C(1) << 12)
#define VX_VELOX_CAPABILITY_CONSTANT_VISITOR   (UINT64_C(1) << 13)
#define VX_VELOX_CAPABILITY_BOOL_VISITOR       (UINT64_C(1) << 14)
#define VX_VELOX_CAPABILITY_DATE_VISITOR       (UINT64_C(1) << 15)
#define VX_VELOX_CAPABILITY_DECIMAL_VISITOR    (UINT64_C(1) << 16)
#define VX_VELOX_CAPABILITY_STRUCT_VISITOR     (UINT64_C(1) << 17)
#define VX_VELOX_CAPABILITY_LIST_VISITOR       (UINT64_C(1) << 18)
#define VX_VELOX_CAPABILITY_MAP_VISITOR        (UINT64_C(1) << 19)

typedef struct vx_velox_read_at vx_velox_read_at;
typedef struct vx_velox_source vx_velox_source;
typedef struct vx_velox_export_cursor vx_velox_export_cursor;
typedef struct vx_velox_error vx_velox_error;
typedef struct vx_velox_session vx_velox_session;
typedef struct vx_velox_dtype vx_velox_dtype;
typedef struct vx_velox_scalar vx_velox_scalar;
typedef struct vx_velox_expression vx_velox_expression;
typedef struct vx_velox_data_source vx_velox_data_source;
typedef struct vx_velox_scan vx_velox_scan;
typedef struct vx_velox_partition vx_velox_partition;
typedef struct vx_velox_array vx_velox_array;

typedef struct vx_velox_view {
    const char *ptr;
    size_t len;
} vx_velox_view;

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
#define VX_VELOX_OPERATOR_EQ         UINT32_C(0)
#define VX_VELOX_OPERATOR_NOT_EQ     UINT32_C(1)
#define VX_VELOX_OPERATOR_GT         UINT32_C(2)
#define VX_VELOX_OPERATOR_GTE        UINT32_C(3)
#define VX_VELOX_OPERATOR_LT         UINT32_C(4)
#define VX_VELOX_OPERATOR_LTE        UINT32_C(5)
#define VX_VELOX_OPERATOR_KLEENE_AND UINT32_C(6)
#define VX_VELOX_OPERATOR_KLEENE_OR  UINT32_C(7)

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
    const vx_velox_expression *projection;
    const vx_velox_expression *filter;
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
#define VX_VELOX_PRIMITIVE_U8   UINT32_C(0)
#define VX_VELOX_PRIMITIVE_U16  UINT32_C(1)
#define VX_VELOX_PRIMITIVE_U32  UINT32_C(2)
#define VX_VELOX_PRIMITIVE_U64  UINT32_C(3)
#define VX_VELOX_PRIMITIVE_I8   UINT32_C(4)
#define VX_VELOX_PRIMITIVE_I16  UINT32_C(5)
#define VX_VELOX_PRIMITIVE_I32  UINT32_C(6)
#define VX_VELOX_PRIMITIVE_I64  UINT32_C(7)
#define VX_VELOX_PRIMITIVE_F16  UINT32_C(8)
#define VX_VELOX_PRIMITIVE_F32  UINT32_C(9)
#define VX_VELOX_PRIMITIVE_F64  UINT32_C(10)
#define VX_VELOX_PRIMITIVE_I128 UINT32_C(11)

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
 * into a uint64_t-aligned, word-padded allocation. A window rebases the pointer
 * and reports its remaining bit offset.
 * buffers.retained_bytes is the exact sum of these allocation sizes.
 *
 * The pointers remain valid through visit_primitive. The host must call retain
 * before it stores a pointer beyond that callback.
 */
typedef struct vx_velox_primitive_view {
    size_t struct_size;
    vx_velox_primitive_type primitive_type;
    uint32_t decimal_precision;
    int32_t decimal_scale;
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

typedef uint32_t vx_velox_varbin_kind;
#define VX_VELOX_VARBIN_UTF8   UINT32_C(0)
#define VX_VELOX_VARBIN_BINARY UINT32_C(1)

typedef struct vx_velox_byte_buffer_view {
    const uint8_t *data;
    size_t length;
} vx_velox_byte_buffer_view;

/**
 * A stable 16-byte variable-width binary view.
 *
 * Values of 12 bytes or fewer occupy data directly. Longer values store four
 * prefix bytes, a uint32_t buffer index, and a uint32_t byte offset in data.
 * The host must reject lengths above INT32_MAX. Each outlined range fits its
 * payload buffer.
 */
typedef struct vx_velox_binary_view {
    uint32_t length;
    uint8_t data[12];
} vx_velox_binary_view;

/**
 * A canonical UTF-8 or binary payload and its owner.
 *
 * The view buffer and each payload buffer stay valid through visit_varbin.
 * The host must retain buffers before it stores pointers after the callback.
 * buffers.retained_bytes includes all retained allocation capacities.
 */
typedef struct vx_velox_varbin_view {
    size_t struct_size;
    vx_velox_varbin_kind kind;
    size_t length;
    const vx_velox_binary_view *views;
    size_t views_length;
    const vx_velox_byte_buffer_view *data_buffers;
    size_t data_buffer_count;
    vx_velox_validity_kind validity_kind;
    const uint8_t *validity;
    size_t validity_length;
    size_t validity_bit_offset;
    vx_velox_buffer_owner buffers;
    size_t views_alignment;
    size_t validity_alignment;
} vx_velox_varbin_view;

/**
 * A canonical packed Boolean payload and its owner.
 *
 * The value and validity buffers use least-significant-bit-first order.
 * The host must retain buffers before it stores pointers after the callback.
 */
typedef struct vx_velox_bool_view {
    size_t struct_size;
    size_t length;
    const uint8_t *values;
    size_t values_length;
    size_t values_bit_offset;
    vx_velox_validity_kind validity_kind;
    const uint8_t *validity;
    size_t validity_length;
    size_t validity_bit_offset;
    vx_velox_buffer_owner buffers;
    size_t values_alignment;
    size_t validity_alignment;
} vx_velox_bool_view;

/**
 * A dictionary payload for one output window.
 *
 * codes owns the integer code buffers. values remains valid only during the
 * callback. The host can visit the prepared cursor during that callback.
 */
typedef struct vx_velox_dictionary_view {
    size_t struct_size;
    size_t length;
    vx_velox_primitive_view codes;
    const vx_velox_export_cursor *values;
    size_t values_length;
} vx_velox_dictionary_view;

/**
 * A constant payload for one output window.
 *
 * value contains one canonical value. It remains valid only during the
 * callback. The host can visit the prepared cursor during that callback.
 */
typedef struct vx_velox_constant_view {
    size_t struct_size;
    size_t length;
    const vx_velox_export_cursor *value;
} vx_velox_constant_view;

/**
 * A canonical struct payload for one output window.
 *
 * fields contains borrowed prepared cursors in declaration order. The host
 * visits each field cursor at offset for length rows during this callback.
 * buffers owns only the parent validity buffer.
 */
typedef struct vx_velox_struct_view {
    size_t struct_size;
    size_t length;
    size_t offset;
    const vx_velox_export_cursor *const *fields;
    size_t field_count;
    vx_velox_validity_kind validity_kind;
    const uint8_t *validity;
    size_t validity_length;
    size_t validity_bit_offset;
    vx_velox_buffer_owner buffers;
    size_t validity_alignment;
} vx_velox_struct_view;

/**
 * A canonical list window.
 *
 * offsets and sizes start at the requested parent window. Offset values remain
 * absolute against the complete elements cursor. buffers retains the complete
 * prepared metadata allocation. The host must retain buffers before it stores
 * a metadata pointer. elements is borrowed during the callback. Vectors
 * imported from elements retain their own owners.
 */
typedef struct vx_velox_list_view {
    size_t struct_size;
    size_t length;
    const int32_t *offsets;
    const int32_t *sizes;
    const vx_velox_export_cursor *elements;
    size_t elements_length;
    vx_velox_validity_kind validity_kind;
    const uint8_t *validity;
    size_t validity_length;
    size_t validity_bit_offset;
    vx_velox_buffer_owner buffers;
    size_t offsets_alignment;
    size_t sizes_alignment;
    size_t validity_alignment;
} vx_velox_list_view;

/**
 * A canonical map window.
 *
 * offsets and sizes start at the requested parent window. Offset values remain
 * absolute against the complete key and value cursors. buffers retains the
 * complete prepared metadata allocation. The child cursors are borrowed during
 * the callback. Vectors imported from them retain their own owners.
 */
typedef struct vx_velox_map_view {
    size_t struct_size;
    size_t length;
    const int32_t *offsets;
    const int32_t *sizes;
    const vx_velox_export_cursor *keys;
    const vx_velox_export_cursor *values;
    size_t entries_length;
    bool keys_sorted;
    vx_velox_validity_kind validity_kind;
    const uint8_t *validity;
    size_t validity_length;
    size_t validity_bit_offset;
    vx_velox_buffer_owner buffers;
    size_t offsets_alignment;
    size_t sizes_alignment;
    size_t validity_alignment;
} vx_velox_map_view;

typedef struct vx_velox_visit_request {
    size_t struct_size;
    const uint64_t *rows;
    size_t row_count;
} vx_velox_visit_request;

/**
 * Host callbacks for one Vortex array visit.
 *
 * One array visit calls the matching callback synchronously. If the host shares this
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
    int32_t (*visit_varbin)(void *context, const vx_velox_varbin_view *view);
    int32_t (*visit_dictionary)(void *context, const vx_velox_dictionary_view *view);
    int32_t (*visit_constant)(void *context, const vx_velox_constant_view *view);
    int32_t (*visit_bool)(void *context, const vx_velox_bool_view *view);
    int32_t (*visit_struct)(void *context, const vx_velox_struct_view *view);
    int32_t (*visit_list)(void *context, const vx_velox_list_view *view);
    int32_t (*visit_map)(void *context, const vx_velox_map_view *view);
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

vx_velox_view vx_velox_error_message(const vx_velox_error *error);
void vx_velox_error_free(const vx_velox_error *error);
vx_velox_session *vx_velox_session_new(void);
vx_velox_session *vx_velox_session_clone(const vx_velox_session *session);
void vx_velox_session_free(const vx_velox_session *session);

const vx_velox_dtype *
vx_velox_dtype_new_primitive(vx_velox_ptype ptype, bool nullable, vx_velox_error **error_out);
void vx_velox_dtype_free(const vx_velox_dtype *dtype);
vx_velox_scalar *vx_velox_scalar_new_bool(bool value, bool nullable);
vx_velox_scalar *vx_velox_scalar_new_i8(int8_t value, bool nullable);
vx_velox_scalar *vx_velox_scalar_new_i16(int16_t value, bool nullable);
vx_velox_scalar *vx_velox_scalar_new_i32(int32_t value, bool nullable);
vx_velox_scalar *vx_velox_scalar_new_date_days(int32_t value, bool nullable, vx_velox_error **error_out);
vx_velox_scalar *vx_velox_scalar_new_i64(int64_t value, bool nullable);
vx_velox_scalar *vx_velox_scalar_new_f32(float value, bool nullable);
vx_velox_scalar *vx_velox_scalar_new_f64(double value, bool nullable);
vx_velox_scalar *vx_velox_scalar_new_utf8(vx_velox_view value, bool nullable, vx_velox_error **error_out);
vx_velox_scalar *
vx_velox_scalar_new_binary(const uint8_t *data, size_t length, bool nullable, vx_velox_error **error_out);
vx_velox_scalar *vx_velox_scalar_new_list(const vx_velox_dtype *element_dtype,
                                          const vx_velox_scalar *const *elements,
                                          size_t length,
                                          bool nullable,
                                          vx_velox_error **error_out);
void vx_velox_scalar_free(const vx_velox_scalar *scalar);

vx_velox_expression *vx_velox_expression_root(void);
vx_velox_expression *vx_velox_expression_literal(const vx_velox_scalar *scalar, vx_velox_error **error_out);
vx_velox_expression *vx_velox_expression_get_item(vx_velox_view name, const vx_velox_expression *child);
vx_velox_expression *vx_velox_expression_binary(vx_velox_binary_operator operation,
                                                const vx_velox_expression *left,
                                                const vx_velox_expression *right,
                                                vx_velox_error **error_out);
vx_velox_expression *vx_velox_expression_and(const vx_velox_expression *const *expressions, size_t length);
vx_velox_expression *vx_velox_expression_or(const vx_velox_expression *const *expressions, size_t length);
vx_velox_expression *vx_velox_expression_not(const vx_velox_expression *child);
vx_velox_expression *vx_velox_expression_is_null(const vx_velox_expression *child);
vx_velox_expression *vx_velox_expression_list_contains(const vx_velox_expression *list,
                                                       const vx_velox_expression *value);
bool vx_velox_can_push_down_integer_values(size_t value_count);
void vx_velox_expression_free(const vx_velox_expression *expression);
vx_velox_expression *
vx_velox_expression_select(const vx_velox_view *names, size_t length, vx_velox_error **error_out);
vx_velox_expression *vx_velox_expression_select_with_row_index(const vx_velox_view *names,
                                                               size_t length,
                                                               vx_velox_view row_index_name,
                                                               vx_velox_error **error_out);

/*
 * On success, the reader owns context and calls release_context once. On
 * failure, the caller still owns context.
 */
vx_velox_read_at *vx_velox_read_at_new(const vx_velox_read_at_callbacks *callbacks,
                                       vx_velox_error **error_out);
void vx_velox_read_at_free(vx_velox_read_at *reader);
uint64_t vx_velox_read_at_size(const vx_velox_read_at *reader, vx_velox_error **error_out);

vx_velox_source *vx_velox_source_new(const vx_velox_session *session,
                                     const vx_velox_read_at *reader,
                                     vx_velox_error **error_out);
void vx_velox_source_free(vx_velox_source *source);
uint64_t vx_velox_source_row_count(const vx_velox_source *source);
uint64_t vx_velox_source_file_size(const vx_velox_source *source);
int32_t vx_velox_source_export_schema(const vx_velox_source *source,
                                      struct ArrowSchema *schema_out,
                                      vx_velox_error **error_out);
size_t vx_velox_source_natural_split_count(const vx_velox_source *source);
int32_t vx_velox_source_natural_split_at(const vx_velox_source *source,
                                         size_t index,
                                         vx_velox_natural_split *split_out,
                                         vx_velox_error **error_out);
int32_t vx_velox_source_prune_natural_splits(const vx_velox_source *source,
                                             const vx_velox_expression *expression,
                                             size_t first_split,
                                             size_t split_count,
                                             uint8_t *pruned_out,
                                             vx_velox_error **error_out);
const vx_velox_data_source *vx_velox_source_data_source(const vx_velox_source *source,
                                                        vx_velox_error **error_out);

void vx_velox_data_source_free(const vx_velox_data_source *data_source);
vx_velox_scan *vx_velox_data_source_scan(const vx_velox_data_source *data_source,
                                         const vx_velox_scan_options *options,
                                         vx_velox_error **error_out);
void vx_velox_scan_free(const vx_velox_scan *scan);
vx_velox_partition *vx_velox_scan_next_partition(vx_velox_scan *scan, vx_velox_error **error_out);
void vx_velox_partition_free(const vx_velox_partition *partition);
const vx_velox_array *vx_velox_partition_next(vx_velox_partition *partition, vx_velox_error **error_out);
void vx_velox_array_free(const vx_velox_array *array);
size_t vx_velox_array_len(const vx_velox_array *array);
const vx_velox_array *
vx_velox_array_slice(const vx_velox_array *array, size_t begin, size_t end, vx_velox_error **error_out);
const vx_velox_array *vx_velox_array_get_field(const vx_velox_session *session,
                                               const vx_velox_array *array,
                                               size_t index,
                                               vx_velox_error **error_out);
size_t vx_velox_array_invalid_count(const vx_velox_session *session,
                                    const vx_velox_array *array,
                                    vx_velox_error **error_out);
int32_t vx_velox_array_visit(const vx_velox_session *session,
                             const vx_velox_array *array,
                             const vx_velox_visit_request *request,
                             const vx_velox_visitor *visitor,
                             vx_velox_error **error_out);

/**
 * Create one prepared exporter for several engine-sized output windows.
 *
 * memory_callbacks must identify a complete, thread-safe callback table.
 * The exporter retains its callback context until the last buffer owner releases it.
 */
vx_velox_export_cursor *vx_velox_export_cursor_new(const vx_velox_session *session,
                                                   const vx_velox_array *array,
                                                   const vx_velox_arrow_memory_callbacks *memory_callbacks,
                                                   vx_velox_error **error_out);

/** Free one prepared exporter. */
void vx_velox_export_cursor_free(vx_velox_export_cursor *cursor);

/**
 * Visit one contiguous output window from a prepared exporter.
 *
 * Concurrent visit calls are valid. Do not free the cursor before all visits return.
 */
int32_t vx_velox_export_cursor_visit(const vx_velox_export_cursor *cursor,
                                     size_t offset,
                                     size_t length,
                                     const vx_velox_visitor *visitor,
                                     vx_velox_error **error_out);

int32_t vx_velox_array_export_arrow(const vx_velox_session *session,
                                    const vx_velox_array *array,
                                    const vx_velox_arrow_memory_callbacks *memory_callbacks,
                                    struct ArrowSchema *schema_out,
                                    struct ArrowArray *array_out,
                                    vx_velox_error **error_out);

#ifdef __cplusplus
}
#endif
