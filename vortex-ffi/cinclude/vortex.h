// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#pragma once
#include <stdint.h>

//
// THIS FILE IS AUTO-GENERATED, DO NOT MAKE EDITS DIRECTLY
//

// https://arrow.apache.org/docs/format/CDataInterface.html#structure-definitions
// We don't want to bundle nanoarrow or similar just for these two definitions.
// If you use your own Arrow library, define this macro and
// typedef FFI_ArrowSchema ArrowSchema;
// typedef FFI_ArrowArrayStream ArrowArrayStream;
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
typedef struct ArrowArrayStream FFI_ArrowArrayStream;
#endif

#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

/**
 * Maximum size of an inlined binary value.
 */
#define BinaryView_MAX_INLINED_SIZE 12

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
    VX_CARD_UNKNOWN = 0,
    VX_CARD_ESTIMATE = 1,
    VX_CARD_MAXIMUM = 2,
} vx_cardinality;

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
 * Physical type enum, represents the in-memory physical layout but might represent a different logical type.
 */
enum PType
#ifdef __cplusplus
    : uint8_t
#endif // __cplusplus
{
    /**
     * An 8-bit unsigned integer
     */
    U8 = 0,
    /**
     * A 16-bit unsigned integer
     */
    U16 = 1,
    /**
     * A 32-bit unsigned integer
     */
    U32 = 2,
    /**
     * A 64-bit unsigned integer
     */
    U64 = 3,
    /**
     * An 8-bit signed integer
     */
    I8 = 4,
    /**
     * A 16-bit signed integer
     */
    I16 = 5,
    /**
     * A 32-bit signed integer
     */
    I32 = 6,
    /**
     * A 64-bit signed integer
     */
    I64 = 7,
    /**
     * A 16-bit floating point number
     */
    F16 = 8,
    /**
     * A 32-bit floating point number
     */
    F32 = 9,
    /**
     * A 64-bit floating point number
     */
    F64 = 10,
};
#ifndef __cplusplus
typedef uint8_t PType;
#endif // __cplusplus

/**
 * The logical types of elements in Vortex arrays.
 *
 * `DType` represents the different logical data types that can be represented in a Vortex array.
 *
 * This is different from physical types, which represent the actual layout of data (compressed or
 * uncompressed). The set of physical types/formats (or data layout) is surjective into the set of
 * logical types (or in other words, all physical types map to a single logical type).
 *
 * Note that a `DType` represents the logical type of the elements in the `Array`s, **not** the
 * logical type of the `Array` itself.
 *
 * For example, an array with [`DType::Primitive`]([`I32`], [`NonNullable`]) could be physically
 * encoded as any of the following:
 *
 * - A flat array of `i32` values.
 * - A run-length encoded sequence.
 * - Dictionary encoded values with bitpacked codes.
 *
 * All of these physical encodings preserve the same logical [`I32`] type, even if the physical
 * data is different.
 *
 * [`I32`]: PType::I32
 * [`NonNullable`]: Nullability::NonNullable
 */
typedef struct DType DType;

/**
 * Whether an instance of a DType can be `null or not
 */
typedef struct Nullability Nullability;

typedef struct Primitive Primitive;

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
 * As an implementation detail, vx_array Arc'ed inside, so cloning an
 * array is a cheap operation.
 *
 * Unless stated explicitly, all operations with vx_array don't take
 * ownership of it, and thus it must be freed by the caller.
 */
typedef struct vx_array vx_array;

/**
 * A Vortex array iterator.
 *
 * Once the iterator is finished (returns `null` from [`vx_array_iterator_next`]), it may panic
 * on subsequent calls to [`vx_array_iterator_next`].
 *
 * Even after the iterator is finished, an owned iterator must be released by calling
 * [`vx_array_iter_free`].
 *
 * Iterators may be passed between threads, but calls to [`vx_array_iterator_next`] should be
 * serialized and not invoked concurrently.
 */
typedef struct vx_array_iterator vx_array_iterator;

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
 * Strings for use within Vortex.
 */
typedef struct vx_binary vx_binary;

/**
 * A reference to one or more (possibly remote) paths.
 * Creating vx_data_source opens the first matched path to read the schema.
 * All other I/O is deferred until a scan is requested. Multiple scans may
 * be requested from a single data source.
 */
typedef struct vx_data_source vx_data_source;

/**
 * A Vortex data type.
 *
 * Data types in Vortex are purely logical, meaning they confer no information about how the data
 * is physically stored.
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
 * Unless stated explicitly, all expressions returned are owned and must
 * be freed by the caller.
 * Unless stated explicitly, if an operation on const vx_expression* is
 * passed NULL, NULL is returned.
 * Operations on expressions don't take ownership of input values, and so
 * input values must be freed by the caller.
 */
typedef struct vx_expression vx_expression;

/**
 * A handle to a Vortex file encapsulating the footer and logic for instantiating a reader.
 */
typedef struct vx_file vx_file;

/**
 * A partition is an independent unit of work. Call vx_partition_next repeatedly to
 * retrieve arrays, then free the partition with vx_partition_free.
 */
typedef struct vx_partition vx_partition;

typedef struct vx_scan vx_scan;

/**
 * A handle to a Vortex session.
 */
typedef struct vx_session vx_session;

/**
 * Strings for use within Vortex.
 */
typedef struct vx_string vx_string;

typedef struct vx_struct_column_builder vx_struct_column_builder;

/**
 * Represents a Vortex struct data type, without top-level nullability.
 */
typedef struct vx_struct_fields vx_struct_fields;

/**
 * Builder for creating a [`vx_struct_fields`].
 */
typedef struct vx_struct_fields_builder vx_struct_fields_builder;

typedef struct {
    vx_validity_type type;
    /**
     * If type is not VX_VALIDITY_ARRAY, this is NULL.
     * If type is VX_VALIDITY_ARRAY, this is set to an owned boolean validity
     * array which must be freed by the caller.
     */
    const vx_array *array;
} vx_validity;

/**
 * Options for creating a data source.
 */
typedef struct {
    /**
     * Required: paths to files, tables, or layout trees.
     * May be a glob pattern like "*.vortex".
     * If you want to include multiple paths, concat them with a comma:
     * "file1.vortex,../file2.vortex".
     */
    const char *paths;
} vx_data_source_options;

typedef struct {
    vx_cardinality cardinality;
    /**
     * Set only when "cardinality" is not VX_CARD_UNKNOWN
     */
    uint64_t rows;
} vx_data_source_row_count;

/**
 * Options supplied for opening a file.
 */
typedef struct {
    /**
     * URI for opening the file.
     * This must be a valid URI, even for files (file:///path/to/file)
     */
    const char *uri;
    /**
     * Additional configuration for the file source (e.g. "s3.accessKey").
     * This may be null, in which case it is treated as empty.
     */
    const char *const *property_keys;
    /**
     * Additional configuration values for the file source (e.g. S3 credentials).
     */
    const char *const *property_vals;
    /**
     * Number of properties in `property_keys` and `property_vals`.
     */
    int property_len;
} vx_file_open_options;

/**
 * Scan options provided by an FFI client calling the `vx_file_scan` function.
 */
typedef struct {
    /**
     * Column names to project out in the scan. These must be null-terminated C strings.
     */
    const char *projection_expression;
    /**
     * Number of columns in `projection`.
     */
    unsigned int projection_expr_len;
    /**
     * Serialized expressions for pushdown
     */
    const char *filter_expression;
    /**
     * The len in bytes of the filter expression
     */
    unsigned int filter_expression_len;
    /**
     * Splits the file into chunks of this size, if zero then we use the write layout.
     */
    int split_by_row_count;
    /**
     * First row of a range to scan.
     */
    unsigned long row_range_start;
    /**
     * Last row of a range to scan.
     */
    unsigned long row_range_end;
    /**
     * The row offset of the file in a multi-file scan.
     */
    unsigned long row_offset;
} vx_file_scan_options;

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
     * Upper limit for parallelism. 0 means no limit.
     * Scan will return at most "max_threads" partitions.
     */
    uint64_t max_threads;
    /**
     * If true, return in storage order.
     */
    bool ordered;
} vx_scan_options;

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

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/**
 * Clone a borrowed [`vx_array`], returning an owned [`vx_array`].
 *
 *
 * Must be released with [`vx_array_free`].
 */
const vx_array *vx_array_clone(const vx_array *ptr);

/**
 * Free an owned [`vx_array`] object.
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
 *
 */
bool vx_array_has_dtype(const vx_array *array, vx_dtype_variant variant);

/**
 * Check whether array has a Primitive dtype with a specific ptype.
 *
 * const vx_array* array = vx_array_new_null(1);
 * assert(!vx_array_is_primitive(array, PTYPE_U32));
 * vx_array_free(array);
 *
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
 * Get the [`crate::vx_dtype`] of the array.
 *
 * The returned pointer is valid as long as the array is valid.
 * Do NOT free the returned dtype pointer - it shares the lifetime of the array.
 */
const vx_dtype *vx_array_dtype(const vx_array *array);

const vx_array *vx_array_get_field(const vx_array *array, size_t index, vx_error **error_out);

const vx_array *vx_array_slice(const vx_array *array, size_t start, size_t stop, vx_error **error_out);

/**
 * Check whether array's element at index is invalid (null) according to the
 * validity array. Sets error if index is out of bounds or underlying validity
 * array is corrupted.
 */
bool vx_array_element_is_invalid(const vx_array *array, size_t index, vx_error **error);

/**
 * Check how many items in the array are invalid (null).
 */
size_t vx_array_invalid_count(const vx_array *array, vx_error **error_out);

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
 *
 */
const vx_array *vx_array_new_primitive(vx_ptype ptype,
                                       const void *ptr,
                                       size_t len,
                                       const vx_validity *validity,
                                       vx_error **error);

uint8_t vx_array_get_u8(const vx_array *array, size_t index);

uint8_t vx_array_get_storage_u8(const vx_array *array, size_t index);

uint16_t vx_array_get_u16(const vx_array *array, size_t index);

uint16_t vx_array_get_storage_u16(const vx_array *array, size_t index);

uint32_t vx_array_get_u32(const vx_array *array, size_t index);

uint32_t vx_array_get_storage_u32(const vx_array *array, size_t index);

uint64_t vx_array_get_u64(const vx_array *array, size_t index);

uint64_t vx_array_get_storage_u64(const vx_array *array, size_t index);

int8_t vx_array_get_i8(const vx_array *array, size_t index);

int8_t vx_array_get_storage_i8(const vx_array *array, size_t index);

int16_t vx_array_get_i16(const vx_array *array, size_t index);

int16_t vx_array_get_storage_i16(const vx_array *array, size_t index);

int32_t vx_array_get_i32(const vx_array *array, size_t index);

int32_t vx_array_get_storage_i32(const vx_array *array, size_t index);

int64_t vx_array_get_i64(const vx_array *array, size_t index);

int64_t vx_array_get_storage_i64(const vx_array *array, size_t index);

uint16_t vx_array_get_f16(const vx_array *array, size_t index);

uint16_t vx_array_get_storage_f16(const vx_array *array, size_t index);

float vx_array_get_f32(const vx_array *array, size_t index);

float vx_array_get_storage_f32(const vx_array *array, size_t index);

double vx_array_get_f64(const vx_array *array, size_t index);

double vx_array_get_storage_f64(const vx_array *array, size_t index);

/**
 * Return the utf-8 string at `index` in the array. The pointer will be null if the value at `index` is null.
 * The caller must free the returned pointer.
 */
const vx_string *vx_array_get_utf8(const vx_array *array, uint32_t index);

/**
 * Return the binary at `index` in the array. The pointer will be null if the value at `index` is null.
 * The caller must free the returned pointer.
 */
const vx_binary *vx_array_get_binary(const vx_array *array, uint32_t index);

/**
 * Apply the expression to the array, wrapping it with a ScalarFnArray.
 * This operation takes constant time as it doesn't execute the underlying
 * array. Executing the underlying array still takes O(n) time.
 */
const vx_array *vx_array_apply(const vx_array *array, const vx_expression *expression, vx_error **error);

/**
 * Free an owned [`vx_array_iterator`] object.
 */
void vx_array_iterator_free(vx_array_iterator *ptr);

/**
 * Attempt to advance the `current` pointer of the iterator.
 *
 * A return value of `true` indicates that another element was pulled from the iterator, and a return
 * of `false` indicates that the iterator is finished.
 *
 * It is an error to call this function again after the iterator is finished.
 */
const vx_array *vx_array_iterator_next(vx_array_iterator *iter, vx_error **error_out);

/**
 * Clone a borrowed [`vx_binary`], returning an owned [`vx_binary`].
 *
 *
 * Must be released with [`vx_binary_free`].
 */
const vx_binary *vx_binary_clone(const vx_binary *ptr);

/**
 * Free an owned [`vx_binary`] object.
 */
void vx_binary_free(const vx_binary *ptr);

/**
 * Create a new Vortex UTF-8 string by copying from a pointer and length.
 */
const vx_binary *vx_binary_new(const char *ptr, size_t len);

/**
 * Return the length of the string in bytes.
 */
size_t vx_binary_len(const vx_binary *ptr);

/**
 * Return the pointer to the string data.
 */
const char *vx_binary_ptr(const vx_binary *ptr);

/**
 * Clone a borrowed [`vx_data_source`], returning an owned [`vx_data_source`].
 *
 *
 * Must be released with [`vx_data_source_free`].
 */
const vx_data_source *vx_data_source_clone(const vx_data_source *ptr);

/**
 * Free an owned [`vx_data_source`] object.
 */
void vx_data_source_free(const vx_data_source *ptr);

/**
 * Create a data source.
 * The first matched file is opened eagerly. to read the schema. All other I/O
 * is deferred until a scan is requested. The returned pointer is owned by the
 * caller and must be freed with vx_data_source_free.
 *
 * On error, returns NULL and sets "err".
 */
const vx_data_source *
vx_data_source_new(const vx_session *session, const vx_data_source_options *options, vx_error **err);

/**
 * Return the schema of the data source as a non-owned dtype.
 * The returned pointer is valid as long as "ds" is alive. Do not free it.
 */
const vx_dtype *vx_data_source_dtype(const vx_data_source *ds);

/**
 * Write data source's row count estimate into "row_count".
 */
void vx_data_source_get_row_count(const vx_data_source *ds, vx_data_source_row_count *row_count);

/**
 * Clone a borrowed [`vx_dtype`], returning an owned [`vx_dtype`].
 *
 *
 * Must be released with [`vx_dtype_free`].
 */
const vx_dtype *vx_dtype_clone(const vx_dtype *ptr);

/**
 * Free an owned [`vx_dtype`] object.
 */
void vx_dtype_free(const vx_dtype *ptr);

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
 * Takes ownership of the `element` pointer.
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
 * Return a borrowed reference to the [`vx_struct_fields`] of a struct.
 *
 * The returned pointer is valid as long as the struct dtype is valid.
 * Do NOT free the returned pointer - it shares the lifetime of the struct dtype.
 */
const vx_struct_fields *vx_dtype_struct_dtype(const vx_dtype *dtype);

/**
 * Returns the element type of a list.
 *
 * The returned pointer is valid as long as the list dtype is valid.
 * Do NOT free the returned dtype pointer - it shares the lifetime of the list dtype.
 */
const vx_dtype *vx_dtype_list_element(const vx_dtype *dtype);

/**
 * Returns the element type of a fixed-size list.
 *
 * The returned pointer is valid as long as the fixed-size list dtype is valid.
 * Do NOT free the returned dtype pointer - it shares the lifetime of the fixed-size list dtype.
 */
const vx_dtype *vx_dtype_fixed_size_list_element(const vx_dtype *dtype);

/**
 * Returns the size of a fixed-size list.
 */
uint32_t vx_dtype_fixed_size_list_size(const vx_dtype *dtype);

/**
 * Checks if the type is time.
 */
bool vx_dtype_is_time(const DType *dtype);

/**
 * Checks if the type is a date.
 */
bool vx_dtype_is_date(const DType *dtype);

/**
 * Checks if the type is a timestamp.
 */
bool vx_dtype_is_timestamp(const DType *dtype);

/**
 * Returns the time unit, assuming the type is time.
 */
uint8_t vx_dtype_time_unit(const DType *dtype);

/**
 * Returns the time zone, assuming the type is time. Caller is responsible for freeing the returned pointer.
 */
const vx_string *vx_dtype_time_zone(const DType *dtype);

/**
 * Convert a dtype to ArrowSchema.
 * You can use the dtype after conversion
 * On success, returns 0. On error, sets err and returns 1.
 */
int vx_dtype_to_arrow_schema(const vx_dtype *dtype, FFI_ArrowSchema *schema, vx_error **err);

/**
 * Free an owned [`vx_error`] object.
 */
void vx_error_free(vx_error *ptr);

/**
 * Returns the error message from the given Vortex error.
 *
 * The returned pointer is valid as long as the error is valid.
 * Do NOT free the returned string pointer - it shares the lifetime of the error.
 */
const vx_string *vx_error_get_message(const vx_error *error);

/**
 * Free an owned [`vx_expression`] object.
 */
void vx_expression_free(vx_expression *ptr);

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
 *
 */
vx_expression *vx_expression_root(void);

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
 *
 */
vx_expression *vx_expression_select(const char *const *names, size_t len, const vx_expression *child);

/**
 * Create an AND expression for multiple child expressions.
 * If there are no input expressions, returns NULL
 */
vx_expression *vx_expression_and(const vx_expression *const *expressions, size_t len);

/**
 * Create an OR disjunction expression for multiple child expressions.
 * If there are no input expressions, returns NULL;
 */
vx_expression *vx_expression_or(const vx_expression *const *expressions, size_t len);

/**
 * Create a binary expression for two expressions of form lhs OP rhs.
 * If either input is NULL, returns NULL.
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
 *
 */
vx_expression *
vx_expression_binary(vx_binary_operator operator_, const vx_expression *lhs, const vx_expression *rhs);

/**
 * Create a logical NOT of the child expression.
 *
 * Returns the logical negation of the input boolean expression.
 */
const vx_expression *vx_expression_not(const vx_expression *child);

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
 */
vx_expression *vx_expression_get_item(const char *item, const vx_expression *child);

/**
 * Create an expression that checks if a value is contained in a list.
 *
 * Returns a boolean array indicating whether the value appears in each list.
 */
vx_expression *vx_expression_list_contains(const vx_expression *list, const vx_expression *value);

/**
 * Clone a borrowed [`vx_file`], returning an owned [`vx_file`].
 *
 *
 * Must be released with [`vx_file_free`].
 */
const vx_file *vx_file_clone(const vx_file *ptr);

/**
 * Free an owned [`vx_file`] object.
 */
void vx_file_free(const vx_file *ptr);

/**
 * Open a file at the given path on the file system.
 */
const vx_file *
vx_file_open_reader(const vx_session *session, const vx_file_open_options *options, vx_error **error_out);

void vx_file_write_array(const vx_session *session,
                         const char *path,
                         const vx_array *array,
                         vx_error **error_out);

uint64_t vx_file_row_count(const vx_file *file);

/**
 * Return the DType of the file.
 *
 * The returned pointer is valid as long as the file is valid.
 * Do NOT free the returned dtype pointer - it shares the lifetime of the file.
 */
const vx_dtype *vx_file_dtype(const vx_file *file);

/**
 * Can we prune the whole file using file stats and an expression
 */
bool vx_file_can_prune(const vx_session *session,
                       const vx_file *file,
                       const char *filter_expression,
                       unsigned int filter_expression_len,
                       vx_error **error_out);

/**
 * Build a new `vx_array_iterator` that returns a series of `vx_array`s from a scan over a `vx_layout_reader`.
 */
vx_array_iterator *vx_file_scan(const vx_session *session,
                                const vx_file *file,
                                const vx_file_scan_options *opts,
                                vx_error **error_out);

/**
 * Set the stderr logger to output at the specified level.
 *
 * The logger will only be installed on the first call.
 */
void vx_set_log_level(vx_log_level level);

/**
 * Free an owned [`vx_scan`] object.
 */
void vx_scan_free(vx_scan *ptr);

/**
 * Free an owned [`vx_partition`] object.
 */
void vx_partition_free(vx_partition *ptr);

/**
 * Scan a data source.
 *
 * Return an owned scan that must be freed with vx_scan_free. A scan may be
 * consumed only once.
 *
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
 * Return borrowed vx_scan's dtype.
 * This function will fail if called after vx_scan_next_partition.
 * Called must not free the returned pointer as its lifetime is bound to the
 * lifetime of the scan.
 * On error returns NULL and sets "err".
 */
const vx_dtype *vx_scan_dtype(const vx_scan *scan, vx_error **err);

/**
 * Return an owned partition from a scan.
 * The returned partition must be freed with vx_partition_free.
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

int vx_partition_scan_arrow(const vx_session *session,
                            vx_partition *partition,
                            FFI_ArrowArrayStream *stream,
                            vx_error **err);

/**
 * Return an owned owned array from a partition.
 * The returned array must be freed with vx_array_free.
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
 * Free an owned [`vx_session`] object.
 */
void vx_session_free(vx_session *ptr);

/**
 * Create a new Vortex session.
 *
 * The caller is responsible for freeing the session with [`vx_session_free`].
 */
vx_session *vx_session_new(void);

/**
 * Clone a Vortex session, returning an owned copy.
 *
 * The caller is responsible for freeing the session with [`vx_session_free`].
 */
vx_session *vx_session_clone(const vx_session *session);

/**
 * Opens a writable array stream, where sink is used to push values into the stream.
 * To close the stream close the sink with `vx_array_sink_close`.
 */
vx_array_sink *vx_array_sink_open_file(const vx_session *session,
                                       const char *path,
                                       const vx_dtype *dtype,
                                       vx_error **error_out);

/**
 * Push an array into a file sink.
 * Does not take ownership of array
 */
void vx_array_sink_push(vx_array_sink *sink, const vx_array *array, vx_error **error_out);

/**
 * Closes an array sink, must be called to ensure all the values pushed to the sink are written
 * to the external resource.
 */
void vx_array_sink_close(vx_array_sink *sink, vx_error **error_out);

/**
 * Clone a borrowed [`vx_string`], returning an owned [`vx_string`].
 *
 *
 * Must be released with [`vx_string_free`].
 */
const vx_string *vx_string_clone(const vx_string *ptr);

/**
 * Free an owned [`vx_string`] object.
 */
void vx_string_free(const vx_string *ptr);

/**
 * Create a new Vortex UTF-8 string by copying from a pointer and length.
 */
const vx_string *vx_string_new(const char *ptr, size_t len);

/**
 * Create a new Vortex UTF-8 string by copying from a null-terminated C-style string.
 */
const vx_string *vx_string_new_from_cstr(const char *ptr);

/**
 * Return the length of the string in bytes.
 */
size_t vx_string_len(const vx_string *ptr);

/**
 * Return the pointer to the string data.
 */
const char *vx_string_ptr(const vx_string *ptr);

/**
 * Free an owned [`vx_struct_column_builder`] object.
 */
void vx_struct_column_builder_free(vx_struct_column_builder *ptr);

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
                                        const char *name,
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
 *
 */
const vx_array *vx_struct_column_builder_finalize(vx_struct_column_builder *builder, vx_error **error);

/**
 * Free an owned [`vx_struct_fields`] object.
 */
void vx_struct_fields_free(vx_struct_fields *ptr);

/**
 * Return the number of fields in the struct dtype.
 */
uint64_t vx_struct_fields_nfields(const vx_struct_fields *dtype);

/**
 * Return a borrowed reference to the name of the field at the given index.
 *
 * The returned pointer is valid as long as the struct fields is valid.
 * Do NOT free the returned string pointer - it shares the lifetime of the struct fields.
 * Returns null if the index is out of bounds.
 */
const vx_string *vx_struct_fields_field_name(const vx_struct_fields *dtype, size_t idx);

/**
 * Returns an *owned* reference to the dtype of the field at the given index.
 *
 * The return type is owned since struct dtypes can be lazily parsed from a binary format, in
 * which case it's not possible to return a borrowed reference to the field dtype.
 *
 * Returns null if the index is out of bounds or if the field dtype cannot be parsed.
 */
const vx_dtype *vx_struct_fields_field_dtype(const vx_struct_fields *dtype, size_t idx);

/**
 * Free an owned [`vx_struct_fields_builder`] object.
 */
void vx_struct_fields_builder_free(vx_struct_fields_builder *ptr);

/**
 * Create a new struct dtype builder.
 */
vx_struct_fields_builder *vx_struct_fields_builder_new(void);

/**
 * Add a field to the struct dtype builder.
 *
 * Takes ownership of both the `name` and `dtype` pointers.
 * Must either free or finalize the builder.
 */
void vx_struct_fields_builder_add_field(vx_struct_fields_builder *builder,
                                        const vx_string *name,
                                        const vx_dtype *dtype);

/**
 * Finalize the struct dtype builder, returning a new `vx_struct_fields`.
 *
 * Takes ownership of the `builder`.
 */
vx_struct_fields *vx_struct_fields_builder_finalize(vx_struct_fields_builder *builder);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus
