// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#pragma once
#include <stdint.h>
#include <string.h>

// THIS FILE IS AUTO-GENERATED, DO NOT MAKE EDITS DIRECTLY

// All operations return owned types which need to be freed by calling a
// matching _free() function. This includes all arrays, data sources, scans,
// errors, error messages, and other allocated objects.
//
// Unless stated explicitly, all function arguments are required. Passing
// NULL to a function expecting a pointer is undefined behaviour.

// https://arrow.apache.org/docs/format/CDataInterface.html#structure-definitions
// If you want to use your own Arrow library like nanoarrow, define this macro
// and typedef your types:
//
// #include "nanoarrow/common/inline_types.h"
// #define USE_OWN_ARROW
// typedef struct ArrowSchema FFI_ArrowSchema;
// typedef struct ArrowArray FFI_ArrowArray;
// typedef struct ArrowArrayStream FFI_ArrowArrayStream;
// #include "vortex.h"
//
#ifndef USE_OWN_ARROW
struct ArrowSchema {
    const char *format;
    const char *name;
    const char *metadata;
    int64_t flags;
    int64_t n_children;
    struct ArrowSchema **children;
    struct ArrowSchema *dictionary;
    void (*release)(struct ArrowSchema *);
    void *private_data;
};
struct ArrowArray {
    int64_t length;
    int64_t null_count;
    int64_t offset;
    int64_t n_buffers;
    int64_t n_children;
    const void **buffers;
    struct ArrowArray **children;
    struct ArrowArray *dictionary;
    void (*release)(struct ArrowArray *);
    void *private_data;
};
struct ArrowArrayStream {
    int (*get_schema)(struct ArrowArrayStream *, struct ArrowSchema *out);
    int (*get_next)(struct ArrowArrayStream *, struct ArrowArray *out);
    const char *(*get_last_error)(struct ArrowArrayStream *);
    void (*release)(struct ArrowArrayStream *);
    void *private_data;
};
typedef struct ArrowSchema FFI_ArrowSchema;
typedef struct ArrowArray FFI_ArrowArray;
typedef struct ArrowArrayStream FFI_ArrowArrayStream;
#endif

#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

/**
 * The variant tag for a Vortex data type.
 */
typedef enum {
    /**
     * Null type.
     */
    DTYPE_NULL = 0,
    /**
     * Boolean type.
     */
    DTYPE_BOOL = 1,
    /**
     * Primitive types (e.g., u8, i16, f32, etc.).
     */
    DTYPE_PRIMITIVE = 2,
    /**
     * Variable-length UTF-8 string type.
     */
    DTYPE_UTF8 = 3,
    /**
     * Variable-length binary data type.
     */
    DTYPE_BINARY = 4,
    /**
     * Nested struct type.
     */
    DTYPE_STRUCT = 5,
    /**
     * Nested list type.
     */
    DTYPE_LIST = 6,
    /**
     * User-defined extension type.
     */
    DTYPE_EXTENSION = 7,
    /**
     * Decimal type with fixed precision and scale.
     */
    DTYPE_DECIMAL = 8,
    /**
     * Nested fixed-size list type.
     */
    DTYPE_FIXED_SIZE_LIST = 9,
} vx_dtype_variant;

/**
 * Variant enum for Vortex primitive types.
 */
typedef enum {
    /**
     * Unsigned 8-bit integer
     */
    PTYPE_U8 = 0,
    /**
     * Unsigned 16-bit integer
     */
    PTYPE_U16 = 1,
    /**
     * Unsigned 32-bit integer
     */
    PTYPE_U32 = 2,
    /**
     * Unsigned 64-bit integer
     */
    PTYPE_U64 = 3,
    /**
     * Signed 8-bit integer
     */
    PTYPE_I8 = 4,
    /**
     * Signed 16-bit integer
     */
    PTYPE_I16 = 5,
    /**
     * Signed 32-bit integer
     */
    PTYPE_I32 = 6,
    /**
     * Signed 64-bit integer
     */
    PTYPE_I64 = 7,
    /**
     * 16-bit floating point number
     */
    PTYPE_F16 = 8,
    /**
     * 32-bit floating point number
     */
    PTYPE_F32 = 9,
    /**
     * 64-bit floating point number
     */
    PTYPE_F64 = 10,
} vx_ptype;

/**
 * Validity representation for arrays constructed through the C FFI.
 */
typedef enum {
    /**
     * Items can't be null
     */
    VX_VALIDITY_NON_NULLABLE = 0,
    /**
     * All items are valid
     */
    VX_VALIDITY_ALL_VALID = 1,
    /**
     * All items are invalid
     */
    VX_VALIDITY_ALL_INVALID = 2,
    /**
     * Items validity is determined by a boolean array. True values in boolean
     * array are valid, false values are invalid (null)
     */
    VX_VALIDITY_ARRAY = 3,
} vx_validity_type;

typedef enum {
    /**
     * No estimate is available.
     */
    VX_ESTIMATE_UNKNOWN = 0,
    /**
     * The value in vx_estimate.estimate is exact.
     */
    VX_ESTIMATE_EXACT = 1,
    /**
     * The value in vx_estimate.estimate is an upper bound.
     */
    VX_ESTIMATE_INEXACT = 2,
} vx_estimate_type;

/**
 * Error category for vx_error.
 */
typedef enum {
    /**
     * All other errors
     */
    VX_ERROR_CODE_OTHER = 0,
    /**
     * Index out of bounds
     */
    VX_ERROR_CODE_OUT_OF_BOUNDS = 1,
    /**
     * Compute kernel execute error
     */
    VX_ERROR_CODE_COMPUTE = 2,
    /**
     * An invalid argument was provided.
     */
    VX_ERROR_CODE_INVALID_ARGUMENT = 3,
    /**
     * Serialization/deserialization error
     */
    VX_ERROR_CODE_SERIALIZATION = 4,
    /**
     * Unimplemented function
     */
    VX_ERROR_CODE_NOT_IMPLEMENTED = 5,
    /**
     * Type mismatch
     */
    VX_ERROR_CODE_MISMATCHED_TYPES = 6,
    /**
     * Assertion failed
     */
    VX_ERROR_CODE_ASSERTION_FAILED = 7,
    /**
     * IO error
     */
    VX_ERROR_CODE_IO = 8,
    /**
     * Panic inside FFI
     */
    VX_ERROR_CODE_PANIC = 9,
} vx_error_code;

/**
 * Equalities, inequalities, and boolean operations over possibly null values.
 * For most operations, if either side is null, the result is null.
 * VX_OPERATOR_KLEENE_AND, VX_OPERATOR_KLEENE_OR obey Kleene (three-valued)
 * logic
 */
typedef enum {
    /**
     * Expressions are equal.
     */
    VX_OPERATOR_EQ = 0,
    /**
     * Expressions are not equal.
     */
    VX_OPERATOR_NOT_EQ = 1,
    /**
     * Expression is greater than another
     */
    VX_OPERATOR_GT = 2,
    /**
     * Expression is greater or equal to another
     */
    VX_OPERATOR_GTE = 3,
    /**
     * Expression is less than another
     */
    VX_OPERATOR_LT = 4,
    /**
     * Expression is less or equal to another
     */
    VX_OPERATOR_LTE = 5,
    /**
     * Boolean AND /\.
     */
    VX_OPERATOR_KLEENE_AND = 6,
    /**
     * Boolean OR \/.
     */
    VX_OPERATOR_KLEENE_OR = 7,
    /**
     * The sum of the arguments.
     * Errors at runtime if the sum would overflow or underflow.
     */
    VX_OPERATOR_ADD = 8,
    /**
     * The difference between the arguments.
     * Errors at runtime if the sum would overflow or underflow.
     * The result is null at any index where either input is null.
     */
    VX_OPERATOR_SUB = 9,
    /**
     * Multiply two numbers
     */
    VX_OPERATOR_MUL = 10,
    /**
     * Divide the left side by the right side
     */
    VX_OPERATOR_DIV = 11,
} vx_binary_operator;

/**
 * Log levels for the Vortex library.
 */
typedef enum {
    /**
     * No logging will be performed.
     */
    LOG_LEVEL_OFF = 0,
    /**
     * Only error messages will be logged.
     */
    LOG_LEVEL_ERROR = 1,
    /**
     * Warnings and error messages will be logged.
     */
    LOG_LEVEL_WARN = 2,
    /**
     * Informational messages, warnings, and error messages will be logged.
     */
    LOG_LEVEL_INFO = 3,
    /**
     * Debug messages, informational messages, warnings, and error messages will be logged.
     */
    LOG_LEVEL_DEBUG = 4,
    /**
     * All messages, including trace messages, will be logged.
     */
    LOG_LEVEL_TRACE = 5,
} vx_log_level;

typedef enum {
    VX_SELECTION_INCLUDE_ALL = 0,
    /**
     * Include rows at the indices in vx_scan_selection.idx.
     */
    VX_SELECTION_INCLUDE_RANGE = 1,
    /**
     * Exclude rows at the indices in vx_scan_selection.idx.
     */
    VX_SELECTION_EXCLUDE_RANGE = 2,
} vx_scan_selection_include;

/**
 * Arrays are reference-counted handles to owned memory buffers that hold
 * scalars. These buffers can be held in a number of physical encodings to
 * perform lightweight compression that exploits the particular data
 * distribution of the array's values.
 *
 * Every data type recognized by Vortex also has a canonical physical
 * encoding format, which arrays can be canonicalized into for ease of
 * access in compute functions.
 *
 * Cloning an array is a cheap operation.
 *
 * Unless stated explicitly, all operations with vx_array don't take
 * ownership of it, and thus the array must be freed by the caller.
 */
typedef struct vx_array vx_array;

/**
 * The `sink` interface is used to collect array chunks and place them into a resource
 * (e.g. an array stream or file (`vx_array_sink_open_file`)).
 *
 * ## Thread Safety
 *
 * This struct is **not** thread-safe for concurrent operations. While the underlying
 * `Sender` is thread-safe, the FFI wrapper should only be accessed from a single thread
 * to avoid race conditions between `push` and `close` operations. The `close` operation
 * consumes the sink, making any subsequent operations undefined behavior.
 *
 * Multiple threads may safely hold pointers to the same sink, but only one thread should
 * perform operations on it at a time, and coordination is required to ensure `close` is
 * called exactly once after all `push` operations are complete.
 */
typedef struct vx_array_sink vx_array_sink;

/**
 * A reference to one or more possibly remote paths.
 *
 * Creating vx_data_source opens the first matched path to read the schema.
 * All other I/O is deferred until a scan is requested. Multiple vx_scan's
 * may be requested from a single vx_data_source.
 *
 * Copying a vx_data_source via vx_data_source_clone is a cheap operation.
 */
typedef struct vx_data_source vx_data_source;

/**
 * A reference-counted Vortex data type.
 *
 * Dtypes in Vortex are purely logical meaning they tell you what the data
 * is but but not the encoding in which it's stored physically.
 *
 * Copying a dtype with vx_dtype_clone is a cheap operation.
 */
typedef struct vx_dtype vx_dtype;

/**
 * The error structure populated by fallible Vortex C functions.
 */
typedef struct vx_error vx_error;

/**
 * A node in a Vortex expression tree.
 *
 * Expressions represent scalar computations that can be performed on
 * data. Each expression consists of an encoding (vtable), heap-allocated
 * metadata, and child expressions.
 *
 * Operations on expressions don't take ownership of input values, and so
 * input values must be freed by the caller.
 */
typedef struct vx_expression vx_expression;

/**
 * A vx_partition is an independent unit of work. Call vx_partition_next
 * repeatedly to retrieve arrays, then free the partition with
 * vx_partition_free.
 */
typedef struct vx_partition vx_partition;

/**
 * A vx_scalar is a single value with an associated vx_dtype.
 *
 * Scalar value may be Null is vx_dtype is nullable.
 * One example where you can get a Null scalar is vx_array_get_scalar
 * where the element at some index is invalid/null.
 */
typedef struct vx_scalar vx_scalar;

/**
 * A vx_scan is a single traversal of a vx_data_source with projections and
 * filters. A vx_scan can be consumed only once.
 */
typedef struct vx_scan vx_scan;

/**
 * A handle to a Vortex session.
 */
typedef struct vx_session vx_session;

typedef struct vx_struct_column_builder vx_struct_column_builder;

/**
 * Represents a Vortex struct data type, without top-level nullability.
 */
typedef struct vx_struct_fields vx_struct_fields;

/**
 * Builder for creating a [`vx_struct_fields`].
 */
typedef struct vx_struct_fields_builder vx_struct_fields_builder;

/**
 * Array validity descriptor used by C FFI constructors.
 */
typedef struct {
    /**
     * The kind of validity represented by this descriptor.
     */
    vx_validity_type type;
    /**
     * If type is not VX_VALIDITY_ARRAY, this is NULL.
     * If type is VX_VALIDITY_ARRAY, this is set to an owned boolean validity
     * array which must be freed by the caller.
     */
    const vx_array *array;
} vx_validity;

/**
 * A non owning view over a byte range.
 */
typedef struct {
    /**
     * NULL "ptr" requires len == 0
     */
    const char *ptr;
    /**
     * Length in bytes.
     */
    size_t len;
} vx_view;

/**
 * Options for creating a data source.
 */
typedef struct {
    /**
     * Required: paths to files, tables, or layout trees. Each entry may be a
     * glob pattern like "*.vortex". Must point to an array of size
     * "paths_len". "paths" bytes are copied.
     */
    const vx_view *paths;
    /**
     * Number of entries in "paths".
     */
    size_t paths_len;
} vx_data_source_options;

/**
 * Used for estimating number of partitions in a data source or number of rows
 * in a partition.
 */
typedef struct {
    vx_estimate_type type;
    /**
     * Set only when "type" is not VX_ESTIMATE_UNKNOWN.
     */
    uint64_t estimate;
} vx_estimate;

/**
 * Scan row selection.
 * "idx" is copied while calling vx_data_source_scan and can be freed after.
 */
typedef struct {
    /**
     * Used only when "include" is not VX_SELECTION_INCLUDE_ALL.
     * If set, must point to an array of len "idx_len" row_indices.
     */
    const uint64_t *idx;
    /**
     * Used only when "include" is not VX_SELECTION_INCLUDE_ALL
     */
    size_t idx_len;
    vx_scan_selection_include include;
} vx_scan_selection;

/**
 * Scan options. All fields are optional. To return everything,
 * zero-initialize this struct.
 */
typedef struct {
    /**
     * What columns to return. NULL means all columns.
     */
    const vx_expression *projection;
    /**
     * Predicate expression. NULL means no filter.
     */
    const vx_expression *filter;
    /**
     * Row range [begin, end). Setting row_range_begin and row_range_end to 0
     * means no limit.
     */
    uint64_t row_range_begin;
    uint64_t row_range_end;
    /**
     * Row-index filter applied after row_range.
     */
    vx_scan_selection selection;
    /**
     * Maximum number of rows to return. 0 means no limit.
     */
    uint64_t limit;
    /**
     * If true, return in storage order.
     */
    bool ordered;
} vx_scan_options;

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/**
 * Set the number of background worker threads driving the shared FFI runtime.
 *
 * Calling this with a non-zero count opts the process into a Vortex-owned thread pool. These
 * background threads drive the same executor as host threads currently inside FFI calls. If this
 * function is never called, Vortex creates no runtime worker threads and execution remains
 * entirely host-thread-driven.
 *
 * This setting is process-global and affects all FFI sessions. Passing zero restores the
 * host-thread-only configuration by signalling all background workers to stop. Increasing the
 * count starts workers immediately; decreasing it signals excess workers to stop.
 */
void vx_runtime_set_worker_threads(size_t worker_threads);

/**
 * Return the configured number of Vortex-owned background worker threads.
 *
 * Zero means the runtime is entirely driven by host threads entering FFI calls.
 */
size_t vx_runtime_worker_count(void);

/**
 * Free a vx_array
 */
void vx_array_free(const vx_array *ptr);

/**
 * Check if array's dtype is nullable.
 * As a particular example, a Null array is nullable.
 */
bool vx_array_is_nullable(const vx_array *array);

/**
 * Check array's dtype against a variant.
 * Equivalent to vx_get_dtype_variant(vx_array_dtype(array)).
 *
 * Example:
 *
 * const vx_array* array = vx_array_new_null(1);
 * assert(vx_array_has_dtype(array, DTYPE_NULL));
 * vx_array_free(array);
 */
bool vx_array_has_dtype(const vx_array *array, vx_dtype_variant variant);

/**
 * Check whether array has a Primitive dtype with a specific ptype.
 *
 * const vx_array* array = vx_array_new_null(1);
 * assert(!vx_array_is_primitive(array, PTYPE_U32));
 * vx_array_free(array);
 */
bool vx_array_is_primitive(const vx_array *array, vx_ptype ptype);

/**
 * Return array's validity as a type and a boolean array.
 */
void vx_array_get_validity(const vx_array *array, vx_validity *validity, vx_error **error);

/**
 * Get the length of the array.
 */
size_t vx_array_len(const vx_array *array);

/**
 * Get array's dtype
 */
const vx_dtype *vx_array_dtype(const vx_array *array);

const vx_array *vx_array_get_field(const vx_array *array, size_t index, vx_error **error_out);

const vx_array *vx_array_slice(const vx_array *array, size_t start, size_t stop, vx_error **error_out);

/**
 * Check whether array's element at index is invalid (null) according to the
 * validity array. Sets error if index is out of bounds or underlying validity
 * array is corrupted.
 */
bool vx_array_element_is_invalid(const vx_session *session,
                                 const vx_array *array,
                                 size_t index,
                                 vx_error **error);

/**
 * Check how many items in the array are invalid (null).
 */
size_t vx_array_invalid_count(const vx_array *array, vx_error **error_out);

/**
 * Increase reference count on vx_array
 */
const vx_array *vx_array_clone(const vx_array *ptr);

/**
 * Create a new array with DTYPE_NULL dtype.
 */
const vx_array *vx_array_new_null(size_t len);

/**
 * Create a new primitive array from an existing buffer.
 * It is caller's responsibility to ensure ptr points to a buffer of correct
 * type. ptr buffer contents are copied.
 * validity can't be NULL.
 *
 * Example:
 *
 * const vx_error* error = NULL;
 * vx_validity validity = {};
 * validity.type = VX_VALIDITY_NON_NULLABLE;
 * uint32_t buffer[] = {1, 2, 3};
 * const vx_array* array = vx_array_new_primitive(PTYPE_U32, buffer, 3,
 *     &validity, &error);
 * vx_array_free(array);
 */
const vx_array *vx_array_new_primitive(vx_ptype ptype,
                                       const void *ptr,
                                       size_t len,
                                       const vx_validity *validity,
                                       vx_error **error);

/**
 * Create a Vortex array by importing an Arrow array via the Arrow C Data Interface.
 *
 * `array` and `schema` together describe a single Arrow array (the standard Arrow C Data
 * Interface pair, e.g. as produced by exporting a record batch). Both are *consumed*: their
 * `release` callbacks are invoked by this function and the caller must not use or release them
 * afterwards.
 *
 * `nullable` controls the top-level nullability of the resulting array's dtype. For an Arrow
 * record batch (which has no top-level validity) pass `false`.
 *
 * On error, returns NULL and sets `error_out`.
 *
 * Example:
 *
 * // export an Arrow record batch into (array, schema), then:
 * vx_error* error = NULL;
 * const vx_array* vx = vx_array_from_arrow(&array, &schema, false, &error);
 * // ... push it to a sink or write it ...
 * vx_array_free(vx);
 */
const vx_array *
vx_array_from_arrow(FFI_ArrowArray *array, FFI_ArrowSchema *schema, bool nullable, vx_error **error_out);

/**
 * Return UTF-8 string at "index" in a canonical Utf8 array.
 *
 * For invalid elements the returned value is unspecified, check validity via
 * vx_array_get_validity.
 * Returned view is valid as long as "array" is valid.
 * Errors if index is out of bounds or array is not a canonical Utf8 array.
 */
vx_view vx_array_utf8_at(const vx_array *array, size_t index, vx_error **error_out);

/**
 * Return a binary string at "index" in a canonical Binary array.
 *
 * For invalid elements the returned value is unspecified, check validity via
 * vx_array_get_validity.
 * Returned view is valid as long as "array" is valid.
 * Errors if index is out of bounds or array is not a canonical Binary array.
 */
vx_view vx_array_binary_at(const vx_array *array, size_t index, vx_error **error_out);

/**
 * For a canonical Bool array, return bool at "index".
 * For invalid elements returned value is unspecified, check validity via
 * vx_array_get_validity.
 *
 * Panics if "array" is not canonical - call vx_array_canonicalize first.
 * Panics if "array" is not a Bool array.
 * Panics if "index" is out of bounds.
 */
bool vx_array_get_bool(const vx_array *array, size_t index);

/**
 * Get array's element at position "index".
 *
 * If element at index is invalid, returns a Null vx_scalar.
 *
 * This operation executes the array to extract a scalar and thus is
 * expensive. If you need bulk access, use
 * vx_array_data_ptr_primitive or vx_data_ptr_bool.
 *
 * Errors if "index" is out of bounds.
 */
const vx_scalar *
vx_array_get_scalar(const vx_session *session, const vx_array *array, size_t index, vx_error **error_out);

/**
 * Decode array into its canonical form.
 *
 * On error returns NULL and "sets error_out".
 */
const vx_array *vx_array_canonicalize(const vx_session *session, const vx_array *array, vx_error **error_out);

/**
 * Return a pointer to the values buffer of a canonical Primitive array.
 * Pointer is valid as long as "array" is valid.
 *
 * Errors if array is not a canonical Primitive.
 */
const void *vx_array_data_ptr_primitive(const vx_array *array, vx_error **error_out);

/**
 * Return a pointer to the bitpacked buffer of a canonical Bool array.
 * Pointer is valid as long as "array" is valid.
 *
 * Writes bit offset of the first element into "bit_offset_out".
 * "bit_offset_out" must not be NULL.
 *
 * Errors if array is not a canonical Bool.
 */
const void *vx_array_data_ptr_bool(const vx_array *array, size_t *bit_offset_out, vx_error **error_out);

/**
 * Apply the expression to the array, wrapping it with a ScalarFnArray.
 * This operation takes constant time as it doesn't execute the underlying
 * array. Executing the underlying array still takes O(n) time.
 */
const vx_array *vx_array_apply(const vx_array *array, const vx_expression *expression, vx_error **error);

/**
 * Free a vx_data_source
 */
void vx_data_source_free(const vx_data_source *ptr);

/**
 * Create a data source.
 * The first matched file is opened eagerly. to read the schema. All other I/O
 * is deferred until a scan is requested.
 *
 * On error, returns NULL and sets "err".
 */
const vx_data_source *
vx_data_source_new(const vx_session *session, const vx_data_source_options *options, vx_error **err);

/**
 * Create a data source from a single in-memory Vortex file.
 *
 * "buffer_len" is the length of "buffer" in bytes.
 * The bytes are borrowed, not copied: the caller must keep "buffer" alive and
 * unmodified until the data source is freed.
 *
 * On error, returns NULL and sets "err".
 */
const vx_data_source *
vx_data_source_new_buffer(const vx_session *session, const void *buffer, size_t buffer_len, vx_error **err);

/**
 * Increase reference count on vx_data_source
 */
const vx_data_source *vx_data_source_clone(const vx_data_source *ptr);

/**
 * Return data source's dtype
 */
const vx_dtype *vx_data_source_dtype(const vx_data_source *ds);

/**
 * Write data source's row count estimate into "row_count".
 */
void vx_data_source_get_row_count(const vx_data_source *ds, vx_estimate *row_count);

/**
 * Free a vx_dtype
 */
void vx_dtype_free(const vx_dtype *ptr);

/**
 * Increase reference count on vx_dtype
 */
const vx_dtype *vx_dtype_clone(const vx_dtype *ptr);

/**
 * Create a new null data type.
 */
const vx_dtype *vx_dtype_new_null(void);

/**
 * Create a new boolean data type.
 */
const vx_dtype *vx_dtype_new_bool(bool is_nullable);

/**
 * Create a new primitive data type.
 */
const vx_dtype *vx_dtype_new_primitive(vx_ptype ptype, bool is_nullable);

/**
 * Create a new variable length UTF-8 data type.
 */
const vx_dtype *vx_dtype_new_utf8(bool is_nullable);

/**
 * Create a new variable length binary data type.
 */
const vx_dtype *vx_dtype_new_binary(bool is_nullable);

/**
 * Create a new list data type.
 *
 * Takes ownership of "element".
 */
const vx_dtype *vx_dtype_new_list(const vx_dtype *element, bool is_nullable);

/**
 * Create a new fixed-size list data type.
 *
 * Takes ownership of the `element` pointer.
 */
const vx_dtype *vx_dtype_new_fixed_size_list(const vx_dtype *element, uint32_t size, bool is_nullable);

/**
 * Create a new struct data type.
 *
 * Takes ownership of the `struct_dtype` pointer.
 */
const vx_dtype *vx_dtype_new_struct(vx_struct_fields *struct_dtype, bool is_nullable);

/**
 * Create a new decimal data type.
 */
const vx_dtype *vx_dtype_new_decimal(uint8_t precision, int8_t scale, bool is_nullable);

/**
 * Get the variant of a [`vx_dtype`].
 */
vx_dtype_variant vx_dtype_get_variant(const vx_dtype *dtype);

/**
 * Return whether the given [`vx_dtype`] is nullable.
 */
bool vx_dtype_is_nullable(const vx_dtype *dtype);

/**
 * Returns the [`vx_ptype`] of a primitive.
 */
vx_ptype vx_dtype_primitive_ptype(const vx_dtype *dtype);

/**
 * Returns the precision of a decimal.
 */
uint8_t vx_dtype_decimal_precision(const vx_dtype *dtype);

/**
 * Returns the scale of a decimal.
 */
int8_t vx_dtype_decimal_scale(const vx_dtype *dtype);

/**
 * If "dtype" is DTYPE_STRUCT, return owned vx_struct_fields for this struct,
 * return NULL otherwise. Returned vx_struct_fields must be released with
 * vx_struct_fields_free.
 */
const vx_struct_fields *vx_dtype_struct_dtype(const vx_dtype *dtype);

/**
 * If "dtype" is DTYPE_LIST, return its element dtype, return NULL otherwise.
 */
const vx_dtype *vx_dtype_list_element(const vx_dtype *dtype);

/**
 * If "dtype" is DTYPE_FIXED_SIZE_LIST, return its element dtype, return NULL
 * otherwise.
 */
const vx_dtype *vx_dtype_fixed_size_list_element(const vx_dtype *dtype);

/**
 * Returns the size of a fixed-size list.
 */
uint32_t vx_dtype_fixed_size_list_size(const vx_dtype *dtype);

/**
 * Convert a dtype to ArrowSchema.
 * You can use the dtype after conversion
 * On success, returns 0. On error, sets err and returns 1.
 */
int vx_dtype_to_arrow_schema(const vx_dtype *dtype, FFI_ArrowSchema *schema, vx_error **err);

/**
 * Create a Vortex dtype from an Arrow C Data Interface schema.
 *
 * `schema` must point to a valid `ArrowSchema` describing a struct (record-batch) schema. It is
 * *consumed*: its `release` callback is invoked by this function and the caller must not use or
 * release it afterwards. The returned dtype is a non-nullable struct, mirroring how Arrow record
 * batches map to Vortex arrays.
 *
 * On error, returns NULL and sets `err`.
 */
const vx_dtype *vx_dtype_from_arrow_schema(FFI_ArrowSchema *schema, vx_error **err);

/**
 * Free a vx_error
 */
void vx_error_free(const vx_error *ptr);

/**
 * Return error message for this error.
 * Returned view is valid while "error" is valid.
 */
vx_view vx_error_message(const vx_error *error);

/**
 * Return category code for "error".
 */
vx_error_code vx_error_get_code(const vx_error *error);

/**
 * Free a vx_expression
 */
void vx_expression_free(const vx_expression *ptr);

/**
 * Create a root expression. A root expression, applied to an array in
 * vx_array_apply, takes the array itself as opposed to functions like
 * vx_expression_column or vx_expression_select which take the array's parts.
 *
 * Example:
 *
 * const vx_array* array = ...;
 * vx_expression* root = vx_expression_root();
 * const vx_error* error = NULL;
 * vx_array* applied_array = vx_array_apply(array, root, &error);
 * // array and applied_array are identical
 * vx_array_free(applied_array);
 * vx_expression_free(root);
 * vx_array_free(array);
 */
vx_expression *vx_expression_root(void);

/**
 * Increase reference count on vx_expression
 */
vx_expression *vx_expression_clone(const vx_expression *ptr);

/**
 * Create a literal expression from a scalar.
 *
 * Literal expressions are useful for constants in expression trees, especially scan
 * predicates. For example, a caller can compare a column expression to a scalar
 * threshold and pass the resulting predicate to `vx_data_source_scan`.
 *
 * Example:
 *
 * vx_error* error = NULL;
 * const vx_data_source* data_source = ...;
 *
 * vx_expression* root = vx_expression_root();
 * vx_expression* age = vx_expression_get_item("age", root);
 *
 * vx_scalar* threshold_scalar = vx_scalar_new_u8(50, false);
 * vx_expression* threshold = vx_expression_literal(threshold_scalar, &error);
 * vx_scalar_free(threshold_scalar);
 *
 * vx_expression* predicate = vx_expression_binary(VX_OPERATOR_GTE, age, threshold);
 * vx_scan_options options = {};
 * options.filter = predicate;
 *
 * vx_scan* scan = vx_data_source_scan(data_source, &options, NULL, &error);
 *
 * vx_scan_free(scan);
 * vx_expression_free(predicate);
 * vx_expression_free(threshold);
 * vx_expression_free(age);
 * vx_expression_free(root);
 */
vx_expression *vx_expression_literal(const vx_scalar *scalar, vx_error **err);

/**
 * Create an expression that selects (includes) specific fields from a child
 * expression. Child expression must have a DTYPE_STRUCT dtype. Errors in
 * vx_array_apply if the child expression doesn't have a specified field.
 *
 * Returns a DTYPE_STRUCT array with selected fields.
 *
 * Example:
 *
 * vx_expression* root = vx_expression_root();
 * const char* names[] = {"name", "age"};
 * vx_expression* select = vx_expression_select(names, 2, root);
 * vx_expression_free(select);
 * vx_expression_free(root);
 */
vx_expression *vx_expression_select(const vx_view *names, size_t len, const vx_expression *child);

/**
 * Create an AND expression for multiple child expressions.
 * If len == 0, returns NULL
 */
vx_expression *vx_expression_and(const vx_expression *const *expressions, size_t len);

/**
 * Create an OR disjunction expression for multiple child expressions.
 * If len == 0, returns NULL
 */
vx_expression *vx_expression_or(const vx_expression *const *expressions, size_t len);

/**
 * Create a binary expression for two expressions of form lhs OP rhs.
 *
 * Example for a binary sum:
 *
 * vx_expression* age = vx_expression_column("age");
 * vx_expression* height = vx_expression_column("height");
 * vx_expression* sum = vx_expression_binary(VX_OPERATOR_ADD, age, height);
 * vx_expression_free(sum);
 * vx_expression_free(height);
 * vx_expression_free(age);
 *
 * Example for a binary equality function:
 *
 * vx_expression* vx_expression_eq(
 *     const vx_expression* lhs,
 *     const vx_expression* rhs
 * ) {
 *     return vx_expression_binary(VX_OPERATOR_EQ, lhs, rhs);
 * }
 */
vx_expression *
vx_expression_binary(vx_binary_operator operator_, const vx_expression *lhs, const vx_expression *rhs);

/**
 * Create a logical NOT of the child expression.
 *
 * Returns the logical negation of the input boolean expression.
 */
vx_expression *vx_expression_not(const vx_expression *child);

/**
 * Create an expression that checks for null values.
 *
 * Returns a boolean array indicating which positions contain null values.
 */
vx_expression *vx_expression_is_null(const vx_expression *child);

/**
 * Create an expression that extracts a named field from a struct expression.
 * Child expression must have a DTYPE_STRUCT dtype.
 * Errors in vx_array_apply if the root array doesn't have a specified field.
 *
 * Accesses the specified field from the result of the child expression.
 *
 * Example: if child is Struct { name=u8, age=u16 } and we do
 * vx_expression_get_item("name", child), output type will be DTYPE_U8
 *
 * "item" is copied. Returns NULL if "item" is not valid UTF-8.
 */
vx_expression *vx_expression_get_item(vx_view item, const vx_expression *child);

/**
 * Create an expression that checks if a value is contained in a list.
 *
 * Returns a boolean array indicating whether the value appears in each list.
 */
vx_expression *vx_expression_list_contains(const vx_expression *list, const vx_expression *value);

/**
 * Set the stderr logger to output at the specified level.
 *
 * The logger will only be installed on the first call.
 */
void vx_set_log_level(vx_log_level level);

/**
 * Free a vx_scalar
 */
void vx_scalar_free(const vx_scalar *ptr);

/**
 * Clone a vx_scalar
 */
vx_scalar *vx_scalar_clone(const vx_scalar *scalar);

/**
 * Return scalar's dtype.
 */
const vx_dtype *vx_scalar_dtype(const vx_scalar *scalar);

/**
 * Return whether scalar is a typed Null value.
 */
bool vx_scalar_is_null(const vx_scalar *scalar);

/**
 * Create a boolean scalar.
 */
vx_scalar *vx_scalar_new_bool(bool value, bool is_nullable);

/**
 * Return the boolean value stored in the scalar.
 *
 * Panics if the scalar is not a Bool scalar, or is null.
 */
bool vx_scalar_get_bool(const vx_scalar *scalar);

/**
 * Create a u8 scalar.
 */
vx_scalar *vx_scalar_new_u8(uint8_t value, bool is_nullable);

/**
 * Return u8 value stored in scalar.
 *
 * Panics if scalar is not a primitive scalar of this type or is null.
 */
uint8_t vx_scalar_get_u8(const vx_scalar *scalar);

/**
 * Create a u16 scalar.
 */
vx_scalar *vx_scalar_new_u16(uint16_t value, bool is_nullable);

/**
 * Return u16 value stored in scalar.
 *
 * Panics if scalar is not a primitive scalar of this type or is null.
 */
uint16_t vx_scalar_get_u16(const vx_scalar *scalar);

/**
 * Create a u32 scalar.
 */
vx_scalar *vx_scalar_new_u32(uint32_t value, bool is_nullable);

/**
 * Return u32 value stored in scalar.
 *
 * Panics if scalar is not a primitive scalar of this type or is null.
 */
uint32_t vx_scalar_get_u32(const vx_scalar *scalar);

/**
 * Create a u64 scalar.
 */
vx_scalar *vx_scalar_new_u64(uint64_t value, bool is_nullable);

/**
 * Return u64 value stored in scalar.
 *
 * Panics if scalar is not a primitive scalar of this type or is null.
 */
uint64_t vx_scalar_get_u64(const vx_scalar *scalar);

/**
 * Create a i8 scalar.
 */
vx_scalar *vx_scalar_new_i8(int8_t value, bool is_nullable);

/**
 * Return i8 value stored in scalar.
 *
 * Panics if scalar is not a primitive scalar of this type or is null.
 */
int8_t vx_scalar_get_i8(const vx_scalar *scalar);

/**
 * Create a i16 scalar.
 */
vx_scalar *vx_scalar_new_i16(int16_t value, bool is_nullable);

/**
 * Return i16 value stored in scalar.
 *
 * Panics if scalar is not a primitive scalar of this type or is null.
 */
int16_t vx_scalar_get_i16(const vx_scalar *scalar);

/**
 * Create a i32 scalar.
 */
vx_scalar *vx_scalar_new_i32(int32_t value, bool is_nullable);

/**
 * Return i32 value stored in scalar.
 *
 * Panics if scalar is not a primitive scalar of this type or is null.
 */
int32_t vx_scalar_get_i32(const vx_scalar *scalar);

/**
 * Create a i64 scalar.
 */
vx_scalar *vx_scalar_new_i64(int64_t value, bool is_nullable);

/**
 * Return i64 value stored in scalar.
 *
 * Panics if scalar is not a primitive scalar of this type or is null.
 */
int64_t vx_scalar_get_i64(const vx_scalar *scalar);

/**
 * Create a f32 scalar.
 */
vx_scalar *vx_scalar_new_f32(float value, bool is_nullable);

/**
 * Return f32 value stored in scalar.
 *
 * Panics if scalar is not a primitive scalar of this type or is null.
 */
float vx_scalar_get_f32(const vx_scalar *scalar);

/**
 * Create a f64 scalar.
 */
vx_scalar *vx_scalar_new_f64(double value, bool is_nullable);

/**
 * Return f64 value stored in scalar.
 *
 * Panics if scalar is not a primitive scalar of this type or is null.
 */
double vx_scalar_get_f64(const vx_scalar *scalar);

/**
 * Create a 16-bit floating point scalar.
 * The value is read from raw uint16_t.
 */
vx_scalar *vx_scalar_new_f16_bits(uint16_t bits, bool is_nullable);

/**
 * Return 16-bit floating point value stored in scalar.
 * The value is read into raw uint16_t.
 *
 * Panics if scalar is not a primitive scalar of this type or is null.
 */
uint16_t vx_scalar_get_f16_bits(const vx_scalar *scalar);

/**
 * Create a UTF-8 scalar.
 *
 * "value" bytes are copied into scalar.
 * Errors on invalid UTF-8.
 */
vx_scalar *vx_scalar_new_utf8(vx_view value, bool is_nullable, vx_error **err);

/**
 * Create a binary scalar.
 *
 * Byte range is copied into the scalar.
 *
 * NULL "ptr" is allowed only when len == 0.
 *
 * Returns NULL and sets "err" on error.
 */
vx_scalar *vx_scalar_new_binary(const uint8_t *ptr, size_t len, bool is_nullable, vx_error **err);

/**
 * Return UTF-8 string stored in scalar.
 *
 * Returned view borrows the scalar and is valid as long as "scalar" is valid.
 *
 * Panics if scalar is not a Utf8 scalar, or is null.
 */
vx_view vx_scalar_get_utf8(const vx_scalar *scalar);

/**
 * Return binary bytes stored in the scalar.
 *
 * Returned view borrows scalar and is valid as long as "scalar" is valid.
 *
 * Panics if scalar is not a Binary scalar, or is null.
 */
vx_view vx_scalar_get_binary(const vx_scalar *scalar);

/**
 * Create a typed null scalar.
 *
 * Returned scalar uses a nullable copy of that logical type, regardless of
 * the input type's top-level nullability.
 *
 * Returns NULL and sets "err" on error.
 */
vx_scalar *vx_scalar_new_null(const vx_dtype *dtype, vx_error **err);

/**
 * Create a decimal scalar from a signed i8 unscaled value.
 *
 * Returns NULL and sets "err" on error.
 */
vx_scalar *
vx_scalar_new_decimal_i8(int8_t value, uint8_t precision, int8_t scale, bool is_nullable, vx_error **err);

/**
 * Return the unscaled i8 value of a decimal scalar.
 *
 * Panics if the scalar is not a decimal scalar, is null, or the
 * unscaled value does not fit in i8.
 */
int8_t vx_scalar_get_decimal_i8(const vx_scalar *scalar);

/**
 * Create a decimal scalar from a signed i16 unscaled value.
 *
 * Returns NULL and sets "err" on error.
 */
vx_scalar *
vx_scalar_new_decimal_i16(int16_t value, uint8_t precision, int8_t scale, bool is_nullable, vx_error **err);

/**
 * Return the unscaled i16 value of a decimal scalar.
 *
 * Panics if the scalar is not a decimal scalar, is null, or the
 * unscaled value does not fit in i16.
 */
int16_t vx_scalar_get_decimal_i16(const vx_scalar *scalar);

/**
 * Create a decimal scalar from a signed i32 unscaled value.
 *
 * Returns NULL and sets "err" on error.
 */
vx_scalar *
vx_scalar_new_decimal_i32(int32_t value, uint8_t precision, int8_t scale, bool is_nullable, vx_error **err);

/**
 * Return the unscaled i32 value of a decimal scalar.
 *
 * Panics if the scalar is not a decimal scalar, is null, or the
 * unscaled value does not fit in i32.
 */
int32_t vx_scalar_get_decimal_i32(const vx_scalar *scalar);

/**
 * Create a decimal scalar from a signed i64 unscaled value.
 *
 * Returns NULL and sets "err" on error.
 */
vx_scalar *
vx_scalar_new_decimal_i64(int64_t value, uint8_t precision, int8_t scale, bool is_nullable, vx_error **err);

/**
 * Return the unscaled i64 value of a decimal scalar.
 *
 * Panics if the scalar is not a decimal scalar, is null, or the
 * unscaled value does not fit in i64.
 */
int64_t vx_scalar_get_decimal_i64(const vx_scalar *scalar);

/**
 * Create a decimal scalar.
 *
 * The unscaled value is read from a 16-byte little-endian signed integer
 * buffer.
 *
 * Returns NULL and sets "err" on error.
 */
vx_scalar *vx_scalar_new_decimal_i128_le(const uint8_t *bytes16,
                                         uint8_t precision,
                                         int8_t scale,
                                         bool is_nullable,
                                         vx_error **err);

/**
 * Create a decimal scalar.
 *
 * The unscaled value is read from a 32-byte little-endian signed integer
 * buffer.
 *
 * Returns NULL and sets "err" on error.
 */
vx_scalar *vx_scalar_new_decimal_i256_le(const uint8_t *bytes32,
                                         uint8_t precision,
                                         int8_t scale,
                                         bool is_nullable,
                                         vx_error **err);

/**
 * Create a list scalar.
 *
 * NULL "elements" are allowed only if len == 0.
 */
vx_scalar *vx_scalar_new_list(const vx_dtype *element_dtype,
                              const vx_scalar *const *elements,
                              size_t len,
                              bool is_nullable,
                              vx_error **err);

/**
 * Create a fixed-size list scalar.
 *
 * NULL "elements" are allowed only if len == 0.
 */
vx_scalar *vx_scalar_new_fixed_size_list(const vx_dtype *element_dtype,
                                         const vx_scalar *const *elements,
                                         uint32_t len,
                                         bool is_nullable,
                                         vx_error **err);

/**
 * Create a struct scalar.
 *
 * NULL "fields" are allowed only if len == 0.
 */
vx_scalar *vx_scalar_new_struct(const vx_dtype *struct_dtype,
                                const vx_scalar *const *fields,
                                size_t len,
                                vx_error **err);

/**
 * Free a vx_scan
 */
void vx_scan_free(const vx_scan *ptr);

/**
 * Free a vx_partition
 */
void vx_partition_free(const vx_partition *ptr);

/**
 * Scan a data source.
 *
 * A scan may be consumed only once.
 * "options" and "estimate" may be NULL.
 *
 * If "options" is NULL, all rows and columns are returned.
 * If "estimate" is not NULL, the estimated partition count is written to
 * *estimate before returning.
 *
 * Returns NULL and writes an error to "*err" on failure.
 */
vx_scan *vx_data_source_scan(const vx_data_source *data_source,
                             const vx_scan_options *options,
                             vx_estimate *estimate,
                             vx_error **err);

/**
 * Return scan's dtype.
 * This function will fail if called after vx_scan_next_partition.
 * On error returns NULL and sets "err".
 */
const vx_dtype *vx_scan_dtype(const vx_scan *scan, vx_error **err);

/**
 * Return an partition from a scan.
 *
 * On success returns a partition.
 * On exhaustion (no more partitions in scan) returns NULL but doesn't set
 * "err".
 * On error returns NULL and sets "err".
 *
 * This function is thread-unsafe. Callers running a multi-threaded pipeline
 * should synchronise on calls to this function and dispatch each produced
 * partition to a dedicated worker thread.
 */
vx_partition *vx_scan_next_partition(vx_scan *scan, vx_error **err);

/**
 * Get partition's estimated row count.
 * Must be called before the first call to vx_partition_next.
 *
 * On success, returns 0.
 * On error, return 1 and sets "error".
 */
int vx_partition_row_count(const vx_partition *partition, vx_estimate *count, vx_error **err);

/**
 * Scan partition to ArrowArrayStream.
 * Consumes partition fully: subsequent calls to vx_partition_scan_arrow or
 * vx_partition_next are undefined behaviour.
 * This call blocks current thread until underlying stream is fully consumed.
 *
 * Caller must not free partition after calling this function.
 *
 * On success, sets "stream" and returns 0.
 * On error, sets "err" and returns 1, freeing the partition.
 */
int vx_partition_scan_arrow(const vx_session *session,
                            vx_partition *partition,
                            FFI_ArrowArrayStream *stream,
                            vx_error **err);

/**
 * Return an array from a partition.
 *
 * On success returns an array.
 * On exhaustion (no more arrays in partition) returns NULL but doesn't set
 * "err".
 * On error return NULL and sets "err".
 *
 * This function is not thread-safe: call from one thread per partition.
 */
const vx_array *vx_partition_next(vx_partition *partition, vx_error **err);

/**
 * Free a vx_session
 */
void vx_session_free(const vx_session *ptr);

/**
 * Create a new Vortex session.
 *
 * The caller is responsible for freeing the session with [`vx_session_free`].
 */
vx_session *vx_session_new(void);

/**
 * Clone a vx_session
 */
vx_session *vx_session_clone(const vx_session *session);

/**
 * Opens a writable array stream, where sink is used to push values into the stream.
 * To close the stream close the sink with `vx_array_sink_close`.
 * "path" is copied.
 */
vx_array_sink *
vx_array_sink_open_file(const vx_session *session, vx_view path, const vx_dtype *dtype, vx_error **error_out);

/**
 * Push an array into a file sink.
 * Does not take ownership of array.
 *
 * Errors if array's DType doesn't match sink's DType.
 */
void vx_array_sink_push(vx_array_sink *sink, const vx_array *array, vx_error **error_out);

/**
 * Closes an array sink, must be called to ensure all the values pushed to the sink are written
 * to the external resource.
 */
void vx_array_sink_close(vx_array_sink *sink, vx_error **error_out);

/**
 * Abort an array sink. File footer is not written, and file is left invalid.
 * Don't use sink after this call.
 */
void vx_array_sink_abort(vx_array_sink *sink);

/**
 * Free a vx_struct_column_builder
 */
void vx_struct_column_builder_free(const vx_struct_column_builder *ptr);

/**
 * Create a new column-wise struct array builder with given validity and a
 * capacity hint. validity can't be NULL.
 * Capacity hint is for the number of columns.
 * If you don't know capacity, pass 0.
 * if validity is NULL, returns NULL.
 */
vx_struct_column_builder *vx_struct_column_builder_new(const vx_validity *validity, size_t capacity);

/**
 * Add a named field to a struct array builder.
 * All arguments must be non-NULL.
 * If field's length doesn't match lengths of previous fields, sets error.
 * If an error is returned, the builder is still valid, and caller must
 * deallocate it using vx_struct_column_builder_free.
 */
void vx_struct_column_builder_add_field(vx_struct_column_builder *builder,
                                        vx_view name,
                                        const vx_array *field,
                                        vx_error **error);

/**
 * Finalize a struct array builder, returning a struct array.
 * Consumes the builder. Caller doesn't need to free the builder after calling
 * this function.
 *
 * Example:
 *
 * vx_error* error = NULL;
 *
 * vx_validity validity = {};
 * validity.type = VX_VALIDITY_NON_NULLABLE;
 *
 * const vx_array* field_array = vx_array_new_null(5);
 * const vx_struct_column_builder* builder =
 *     vx_struct_column_builder_new(&validity, 1);
 *
 * vx_struct_column_builder_add_field(builder, "age", field_array, &error);
 *
 * vx_array* struct_array = vx_struct_column_builder_finalize(builder, &error);
 *
 * vx_array_free(struct_array);
 * vx_array_free(field_array);
 */
const vx_array *vx_struct_column_builder_finalize(vx_struct_column_builder *builder, vx_error **error);

/**
 * Free a vx_struct_fields
 */
void vx_struct_fields_free(const vx_struct_fields *ptr);

/**
 * Return the number of fields in the struct dtype.
 */
uint64_t vx_struct_fields_nfields(const vx_struct_fields *fields);

/**
 * Return field name at a given index.
 * If index is out of bounds, returns {NULL, 0}.
 *
 * Returned view is valid as long as "dtype" is valid.
 */
vx_view vx_struct_fields_field_name(const vx_struct_fields *fields, size_t idx);

/**
 * Return an owned dtype of the field at a given index.
 * Returns NULL if index is out of bounds or if dtype cannot be parsed.
 */
const vx_dtype *vx_struct_fields_field_dtype(const vx_struct_fields *fields, size_t idx);

/**
 * Free a vx_struct_fields_builder
 */
void vx_struct_fields_builder_free(const vx_struct_fields_builder *ptr);

/**
 * Create a new struct dtype builder.
 */
vx_struct_fields_builder *vx_struct_fields_builder_new(void);

/**
 * Add a field to the struct dtype builder.
 *
 * "name" is copied. Takes ownership of "dtype".
 * Caller must free or finalize the builder.
 */
void vx_struct_fields_builder_add_field(vx_struct_fields_builder *builder,
                                        vx_view name,
                                        const vx_dtype *dtype,
                                        vx_error **error_out);

/**
 * Finalize the struct dtype builder, returning vx_struct_fields.
 *
 * Takes ownership of "builder".
 */
vx_struct_fields *vx_struct_fields_builder_finalize(vx_struct_fields_builder *builder);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

/**
 * Create a view over a null-terminated C string.
 * View is valid as long as "str" is valid
 */
static inline vx_view vx_view_from_cstr(const char *str) {
    vx_view s;
    s.ptr = str;
    s.len = strlen(str);
    return s;
}
