//
// THIS FILE IS AUTO-GENERATED, DO NOT MAKE EDITS DIRECTLY
//


// (c) Copyright 2025 SpiralDB Inc. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


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
 * Vortex arrays preserve a single logical type, while the encodings allow for multiple
 * physical ways to encode that type.
 */
typedef struct DType DType;

/**
 * Base type for all Vortex arrays.
 *
 * All built-in Vortex array types can be safely cast to this type to pass into functions that
 * expect a generic array type. e.g.
 *
 * ```cpp
 * auto primitive_array = vx_array_primitive_new(...);
 * vx_array_len((*vx_array) primitive_array));
 * ```
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
 */
typedef struct vx_array_sink vx_array_sink;

/**
 * A Vortex data type.
 *
 * Data types in Vortex are purely logical, meaning they confer no information about how the data
 * is physically stored.
 */
typedef struct vx_dtype vx_dtype;

#if defined(ENABLE_DUCKDB_FFI)
/**
 * A type for exporting Vortex arrays to a stream of mutable DuckDB vectors.
 */
typedef struct vx_duckdb_exporter vx_duckdb_exporter;
#endif

/**
 * The error structure populated by fallible Vortex C functions.
 */
typedef struct vx_error vx_error;

/**
 * A handle to a Vortex file encapsulating ther footer and logic for instantiating a reader.
 */
typedef struct vx_file vx_file;

/**
 * A Vortex session stores registries of extensible types, various caches, and other
 * top-level configuration.
 *
 * Extensible types include array encodings, layouts, extension dtypes, compute functions, etc.
 *
 * Multiple sessions may be created in a single process, and individual arrays are not tied to a
 * specific session.
 */
typedef struct vx_session vx_session;

/**
 * Strings for use within Vortex.
 */
typedef struct vx_string vx_string;

/**
 * Represents a Vortex struct data type, without top-level nullability.
 */
typedef struct vx_struct_fields vx_struct_fields;

/**
 * Builder for creating a [`vx_struct_fields`].
 */
typedef struct vx_struct_fields_builder vx_struct_fields_builder;

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
  const char *const *projection;
  /**
   * Number of columns in `projection`.
   */
  unsigned int projection_len;
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
} vx_file_scan_options;



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
 * Get the length of the array.
 */
size_t vx_array_len(const vx_array *array);

/**
 * Get the [`crate::vx_dtype`] of the array.
 *
 * The returned pointer is valid as long as the array is valid.
 */
const vx_dtype *vx_array_dtype(const vx_array *array);

const vx_array *vx_array_get_field(const vx_array *array, uint32_t index, vx_error **error);

const vx_array *vx_array_slice(const vx_array *array,
                               uint32_t start,
                               uint32_t stop,
                               vx_error **error);

bool vx_array_is_null(const vx_array *array, uint32_t index, vx_error **error);

uint32_t vx_array_null_count(const vx_array *array, vx_error **error);

uint8_t vx_array_get_u8(const vx_array *array, uint32_t index);

uint8_t vx_array_get_storage_u8(const vx_array *array, uint32_t index);

uint16_t vx_array_get_u16(const vx_array *array, uint32_t index);

uint16_t vx_array_get_storage_u16(const vx_array *array, uint32_t index);

uint32_t vx_array_get_u32(const vx_array *array, uint32_t index);

uint32_t vx_array_get_storage_u32(const vx_array *array, uint32_t index);

uint64_t vx_array_get_u64(const vx_array *array, uint32_t index);

uint64_t vx_array_get_storage_u64(const vx_array *array, uint32_t index);

int8_t vx_array_get_i8(const vx_array *array, uint32_t index);

int8_t vx_array_get_storage_i8(const vx_array *array, uint32_t index);

int16_t vx_array_get_i16(const vx_array *array, uint32_t index);

int16_t vx_array_get_storage_i16(const vx_array *array, uint32_t index);

int32_t vx_array_get_i32(const vx_array *array, uint32_t index);

int32_t vx_array_get_storage_i32(const vx_array *array, uint32_t index);

int64_t vx_array_get_i64(const vx_array *array, uint32_t index);

int64_t vx_array_get_storage_i64(const vx_array *array, uint32_t index);

uint16_t vx_array_get_f16(const vx_array *array, uint32_t index);

uint16_t vx_array_get_storage_f16(const vx_array *array, uint32_t index);

float vx_array_get_f32(const vx_array *array, uint32_t index);

float vx_array_get_storage_f32(const vx_array *array, uint32_t index);

double vx_array_get_f64(const vx_array *array, uint32_t index);

double vx_array_get_storage_f64(const vx_array *array, uint32_t index);

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
const vx_array *vx_array_iterator_next(vx_array_iterator *iter,
                                       vx_error **error);

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

bool vx_dype_is_date(const DType *dtype);

bool vx_dtype_is_timestamp(const DType *dtype);

uint8_t vx_dtype_time_unit(const DType *dtype);

void vx_dtype_time_zone(const DType *dtype, void *dst, int *len);

/**
 * Clone a borrowed [`vx_struct_fields`], returning an owned [`vx_struct_fields`].
 *
 *
 * Must be released with [`vx_struct_fields_free`].
 */
const vx_struct_fields *vx_struct_fields_clone(const vx_struct_fields *ptr);

/**
 * Free an owned [`vx_struct_fields`] object.
 */
void vx_struct_fields_free(const vx_struct_fields *ptr);

/**
 * Return the number of fields in the struct dtype.
 */
size_t vx_struct_fields_nfields(const vx_struct_fields *dtype);

/**
 * Return a borrowed reference to the name of the field at the given index.
 */
const vx_string *vx_struct_fields_field_name(const vx_struct_fields *dtype, size_t idx);

/**
 * Returns an *owned* reference to the dtype of the field at the given index.
 *
 * The return type is owned since struct dtypes can be lazily parsed from a binary format, in
 * which case it's not possible to return a borrowed reference to the field dtype.
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
const vx_struct_fields *vx_struct_fields_builder_finalize(vx_struct_fields_builder *builder);

#if defined(ENABLE_DUCKDB_FFI)
/**
 * Converts a DType into a duckdb
 */
duckdb_logical_type vx_dtype_to_duckdb_logical_type(const vx_dtype *dtype, vx_error **error);
#endif

#if defined(ENABLE_DUCKDB_FFI)
/**
 * Converts a DuckDB type into a vortex type
 */
const vx_dtype *vx_duckdb_logical_type_to_dtype(const duckdb_logical_type *column_types,
                                                const unsigned char *column_nullable,
                                                const char *const *column_names,
                                                int column_count,
                                                vx_error **error);
#endif

#if defined(ENABLE_DUCKDB_FFI)
/**
 * Pushed a single duckdb chunk into a file sink.
 */
const vx_array *vx_duckdb_chunk_to_array(duckdb_data_chunk chunk,
                                         const vx_dtype *dtype,
                                         vx_error **error);
#endif

#if defined(ENABLE_DUCKDB_FFI)
/**
 * Free an owned [`vx_duckdb_exporter`] object.
 */
void vx_duckdb_exporter_free(vx_duckdb_exporter *ptr);
#endif

#if defined(ENABLE_DUCKDB_FFI)
vx_duckdb_exporter *vx_duckdb_exporter_new(vx_array_iterator *iter);
#endif

#if defined(ENABLE_DUCKDB_FFI)
bool vx_duckdb_exporter_next(vx_duckdb_exporter *exporter,
                             duckdb_data_chunk data_chunk_ptr,
                             vx_error **error);
#endif

/**
 * Free an owned [`vx_error`] object.
 */
void vx_error_free(vx_error *ptr);

/**
 * Returns a borrowed reference to the error message from the given Vortex error.
 */
const vx_string *vx_error_get_message(const vx_error *error);

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
const vx_file *vx_file_open_reader(const vx_file_open_options *options,
                                   const vx_session *session,
                                   vx_error **error);

void vx_file_write_array(const char *path, const vx_array *array, vx_error **error);

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
                       vx_error **error);

/**
 * Build a new `vx_array_iterator` that returns a series of `vx_array`s from a scan over a `vx_layout_reader`.
 */
vx_array_iterator *vx_file_scan(const vx_file *file,
                                const vx_file_scan_options *opts,
                                vx_error **error);

/**
 * Set the stderr logger to output at the specified level.
 *
 * This function is optional, if it is not called then no logger will be installed.
 */
void vx_set_log_level(vx_log_level level);

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
 * Opens a writable array stream, where sink is used to push values into the stream.
 * To close the stream close the sink with `vx_array_sink_close`.
 */
vx_array_sink *vx_array_sink_open_file(const char *path, const vx_dtype *dtype, vx_error **error);

/**
 * Pushed a single array chunk into a file sink.
 */
void vx_array_sink_push(vx_array_sink *sink, const vx_array *array, vx_error **error);

/**
 * Closes an array sink, must be called to ensure all the values pushed to the sink are written
 * to the external resource.
 */
void vx_array_sink_close(vx_array_sink *sink, vx_error **error);

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

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
