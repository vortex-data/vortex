// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//
// THIS FILE IS AUTO-GENERATED, DO NOT MAKE EDITS DIRECTLY
//


#pragma once

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
   * Null type
   */
  DTYPE_NULL = 0,
  /**
   * Boolean type
   */
  DTYPE_BOOL = 1,
  /**
   * Primitive types (e.g., u8, i16, f32, etc.)
   */
  DTYPE_PRIMITIVE = 2,
  /**
   * Variable-length UTF-8 string type
   */
  DTYPE_UTF8 = 3,
  /**
   * Variable-length binary data type
   */
  DTYPE_BINARY = 4,
  /**
   * Nested struct type
   */
  DTYPE_STRUCT = 5,
  /**
   * Nested list type
   */
  DTYPE_LIST = 6,
  /**
   * User-defined extension type
   */
  DTYPE_EXTENSION = 7,
  /**
   * Decimal type with fixed precision and scale
   */
  DTYPE_DECIMAL = 8,
} vx_dtype_variant;

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



#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/**
 * Attempt to shutdown the shared tokio runtime if no sessions are active.
 * May block indefinitely if the runtime is still running tasks.
 */
void vx_try_shutdown_runtime(void);

/**
 * Get the length of the array.
 */
size_t vx_array_len(const vx_array *array);

/**
 * Get the [`crate::vx_dtype`] of the array.
 *
 * The returned pointer is valid as long as the array is valid.
 */
const vx_dtype *vx_array_dtype(const vx_array *array);

const vx_array *vx_array_get_field(const vx_array *array, uint32_t index, vx_error **error_out);

const vx_array *vx_array_slice(const vx_array *array,
                               uint32_t start,
                               uint32_t stop,
                               vx_error **_error_out);

bool vx_array_is_null(const vx_array *array, uint32_t index, vx_error **error_out);

uint32_t vx_array_null_count(const vx_array *array, vx_error **error_out);

/**
 * Write the UTF-8 string at `index` in the array into the provided destination buffer, recording
 * the length in `len`.
 */
void vx_array_get_utf8(const vx_array *array, uint32_t index, void *dst, int *len);

/**
 * Write the UTF-8 string at `index` in the array into the provided destination buffer, recording
 * the length in `len`.
 */
void vx_array_get_binary(const vx_array *array, uint32_t index, void *dst, int *len);

/**
 * Attempt to advance the `current` pointer of the iterator.
 *
 * A return value of `true` indicates that another element was pulled from the iterator, and a return
 * of `false` indicates that the iterator is finished.
 *
 * It is an error to call this function again after the iterator is finished.
 */
const vx_array *vx_array_iterator_next(vx_array_iterator *iter,
                                       vx_error **error_out);

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
 * Create a new struct data type.
 *
 * Takes ownership of the `struct_dtype` pointer.
 */
const vx_dtype *vx_dtype_new_struct(const vx_struct_fields *struct_dtype, bool is_nullable);

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
 * Return the [`vx_ptype`] of a primitive data type.
 */
vx_ptype vx_dtype_primitive_ptype(const vx_dtype *dtype);

/**
 * Return the precision of a decimal data type.
 */
uint8_t vx_dtype_decimal_precision(const vx_dtype *dtype);

/**
 * Return the scale of a decimal data type.
 */
int8_t vx_dtype_decimal_scale(const vx_dtype *dtype);

/**
 * Return a borrowed reference to the [`vx_struct_fields`] of a struct data type.
 */
const vx_struct_fields *vx_dtype_struct_dtype(const vx_dtype *dtype);

/**
 * Return a borrowed reference to the `element` typee of a list data type.
 */
const vx_dtype *vx_dtype_list_element(const vx_dtype *dtype);

bool vx_dtype_is_time(const DType *dtype);

bool vx_dtype_is_date(const DType *dtype);

bool vx_dtype_is_timestamp(const DType *dtype);

uint8_t vx_dtype_time_unit(const DType *dtype);

void vx_dtype_time_zone(const DType *dtype, void *dst, int *len);

/**
 * Returns a borrowed reference to the error message from the given Vortex error.
 */
const vx_string *vx_error_get_message(const vx_error *error);

/**
 * Open a file at the given path on the file system.
 */
const vx_file *vx_file_open_reader(const vx_file_open_options *options,
                                   const vx_session *session,
                                   vx_error **error_out);

void vx_file_write_array(const char *path, const vx_array *array, vx_error **error_out);

uint64_t vx_file_row_count(const vx_file *file);

/**
 * Return a borrowed reference to the DType of the file.
 */
const vx_dtype *vx_file_dtype(const vx_file *file);

/**
 * Can we prune the whole file using file stats and an expression
 */
bool vx_file_can_prune(const vx_file *file,
                       const char *filter_expression,
                       unsigned int filter_expression_len,
                       vx_error **error_out);

/**
 * Build a new `vx_array_iterator` that returns a series of `vx_array`s from a scan over a `vx_layout_reader`.
 */
vx_array_iterator *vx_file_scan(const vx_file *file,
                                const vx_file_scan_options *opts,
                                vx_error **error_out);

/**
 * Set the stderr logger to output at the specified level.
 *
 * This function is optional, if it is not called then no logger will be installed.
 */
void vx_set_log_level(vx_log_level level);

/**
 * Create a new Vortex session.
 *
 * The caller is responsible for freeing the session with [`vx_session_free`].
 */
vx_session *vx_session_new(void);

/**
 * Opens a writable array stream, where sink is used to push values into the stream.
 * To close the stream close the sink with `vx_array_sink_close`.
 */
vx_array_sink *vx_array_sink_open_file(const char *path,
                                       const vx_dtype *dtype,
                                       vx_error **error_out);

/**
 * Pushed a single array chunk into a file sink.
 */
void vx_array_sink_push(vx_array_sink *sink, const vx_array *array, vx_error **error_out);

/**
 * Closes an array sink, must be called to ensure all the values pushed to the sink are written
 * to the external resource.
 */
void vx_array_sink_close(vx_array_sink *sink, vx_error **error_out);

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
 * Return the number of fields in the struct dtype.
 */
uint64_t vx_struct_fields_nfields(const vx_struct_fields *dtype);

/**
 * Return a borrowed reference to the name of the field at the given index.
 *
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
const vx_dtype *vx_struct_fields_field_dtype(const vx_struct_fields *dtype, uint64_t idx);

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
const vx_struct_fields *vx_struct_fields_builder_finalize(vx_struct_fields_builder *builder);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
