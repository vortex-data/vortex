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

#ifdef __cplusplus
extern "C" {
#endif


#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

#define DTYPE_NULL 0

#define DTYPE_BOOL 1

#define DTYPE_PRIMITIVE_U8 2

#define DTYPE_PRIMITIVE_U16 3

#define DTYPE_PRIMITIVE_U32 4

#define DTYPE_PRIMITIVE_U64 5

#define DTYPE_PRIMITIVE_I8 6

#define DTYPE_PRIMITIVE_I16 7

#define DTYPE_PRIMITIVE_I32 8

#define DTYPE_PRIMITIVE_I64 9

#define DTYPE_PRIMITIVE_F16 10

#define DTYPE_PRIMITIVE_F32 11

#define DTYPE_PRIMITIVE_F64 12

#define DTYPE_UTF8 13

#define DTYPE_BINARY 14

#define DTYPE_STRUCT 15

#define DTYPE_LIST 16

#define DTYPE_EXTENSION 17

typedef enum vx_log_level {
  LOG_LEVEL_OFF = 0,
  LOG_LEVEL_ERROR = 1,
  LOG_LEVEL_WARN = 2,
  LOG_LEVEL_INFO = 3,
  LOG_LEVEL_DEBUG = 4,
  LOG_LEVEL_TRACE = 5,
} vx_log_level;

/**
 * The logical types of elements in Vortex arrays.
 *
 * Vortex arrays preserve a single logical type, while the encodings allow for multiple
 * physical ways to encode that type.
 */
typedef struct vx_dtype vx_dtype;

/**
 * The FFI interface for an [`Array`].
 *
 * Because dyn Trait pointers cannot be shared across FFI, we create a new struct to hold
 * the wide pointer. The C FFI only seems a pointer to this structure, and can pass it into
 * one of the various `vx_array_*` functions.
 */
typedef struct vx_array vx_array;

/**
 * FFI-exposed stream interface.
 */
typedef struct vx_array_stream vx_array_stream;

#if defined(ENABLE_DUCKDB_FFI)
typedef struct vx_conversion_cache vx_conversion_cache;
#endif

typedef struct vx_file_reader vx_file_reader;

typedef struct vx_file_writer vx_file_writer;

typedef struct vx_error {
  int code;
  const char *message;
} vx_error;

/**
 * Options supplied for opening a file.
 */
typedef struct vx_file_open_options {
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
 * Options supplied for opening a file.
 */
typedef struct vx_file_create_options {
  /**
   * path of the file to be created.
   */
  const char *path;
} vx_file_create_options;

/**
 * Whole file statistics.
 */
typedef struct vx_file_statistics {
  /**
   * The exact number of rows in the file.
   */
  uint64_t num_rows;
} vx_file_statistics;

/**
 * Scan options provided by an FFI client calling the `vx_file_scan` function.
 */
typedef struct vx_file_scan_options {
  /**
   * Column names to project out in the scan. These must be null-terminated C strings.
   */
  const char *const *projection;
  /**
   * Number of columns in `projection`.
   */
  int projection_len;
  const char *filter_expression;
  int filter_expression_len;
  /**
   * Splits the file into chunks of this size, if zero then we use the write layout.
   */
  int split_by_row_count;
} vx_file_scan_options;



/**
 * Get the length of the array.
 */
uint64_t vx_array_len(const struct vx_array *array);

/**
 * Get a pointer to the data type for an array.
 *
 * Note that this pointer is tied to the lifetime of the array, and the caller is responsible
 * for ensuring that it is never dereferenced after the array has been freed.
 */
const struct vx_dtype *vx_array_dtype(const struct vx_array *array);

const struct vx_array *vx_array_get_field(const struct vx_array *array,
                                          uint32_t index,
                                          struct vx_error **error);

/**
 * Free the array and all associated resources.
 */
void vx_array_free(struct vx_array *array);

const struct vx_array *vx_array_slice(const struct vx_array *array,
                                      uint32_t start,
                                      uint32_t stop,
                                      struct vx_error **error);

bool vx_array_is_null(const struct vx_array *array, uint32_t index, struct vx_error **error);

uint32_t vx_array_null_count(const struct vx_array *array, struct vx_error **error);

/**
 * Write the UTF-8 string at `index` in the array into the provided destination buffer, recording
 * the length in `len`.
 */
void vx_array_get_utf8(const struct vx_array *array, uint32_t index, void *dst, int *len);

/**
 * Write the UTF-8 string at `index` in the array into the provided destination buffer, recording
 * the length in `len`.
 */
void vx_array_get_binary(const struct vx_array *array, uint32_t index, void *dst, int *len);

/**
 * Pointer to a `DType` value that has been heap-allocated.
 * Create a new simple dtype.
 */
struct vx_dtype *vx_dtype_new(uint8_t variant, bool nullable);

/**
 * Create a new List type with the provided element type.
 *
 * Upon successful return, this function moves the value out of the provided element pointer,
 * so it is not safe to reference afterward.
 */
struct vx_dtype *vx_dtype_new_list(struct vx_dtype *element, bool nullable);

struct vx_dtype *vx_dtype_new_struct(const char *const *names,
                                     struct vx_dtype *const *dtypes,
                                     uint32_t len,
                                     bool nullable);

/**
 * Free an [`DType`] and all associated resources.
 */
void vx_dtype_free(struct vx_dtype *dtype);

/**
 * Get the dtype variant tag for an [`DType`].
 */
uint8_t vx_dtype_get(const struct vx_dtype *dtype);

bool vx_dtype_is_nullable(const struct vx_dtype *dtype);

/**
 * For `DTYPE_STRUCT` variant DTypes, get the number of fields.
 */
uint32_t vx_dtype_field_count(const struct vx_dtype *dtype);

void vx_dtype_field_name(const struct vx_dtype *dtype, uint32_t index, void *dst, int *len);

/**
 * Get the dtype of a field in a `DTYPE_STRUCT` variant DType.
 *
 * This returns a new owned, allocated copy of the DType that must be freed subsequently
 * by the caller.
 */
struct vx_dtype *vx_dtype_field_dtype(const struct vx_dtype *dtype, uint32_t index);

/**
 * For a list DType, get the inner element type.
 *
 * The pointee's lifetime is tied to the lifetime of the list DType. It should not be
 * accessed after the list DType has been freed.
 */
const struct vx_dtype *vx_dtype_element_type(const struct vx_dtype *dtype, struct vx_error **error);

bool vx_dtype_is_time(const struct vx_dtype *dtype);

bool vx_dype_is_date(const struct vx_dtype *dtype);

bool vx_dtype_is_timestamp(const struct vx_dtype *dtype);

uint8_t vx_dtype_time_unit(const struct vx_dtype *dtype);

void vx_dtype_time_zone(const struct vx_dtype *dtype, void *dst, int *len);

#if defined(ENABLE_DUCKDB_FFI)
/**
 * Converts a DType into a duckdb
 */
duckdb_logical_type vx_dtype_to_duckdb_logical_type(struct vx_dtype *dtype,
                                                    struct vx_error **error);
#endif

#if defined(ENABLE_DUCKDB_FFI)
/**
 * Back a single chunk of the array as a duckdb data chunk.
 * The initial call should pass offset = 0.
 * The offset is returned to the caller, which can be used to request the next chunk.
 * 0 is returned when the stream is finished.
 */
unsigned int vx_array_to_duckdb_chunk(struct vx_array *stream,
                                      unsigned int offset,
                                      duckdb_data_chunk data_chunk_ptr,
                                      struct vx_conversion_cache *cache,
                                      struct vx_error **error);
#endif

#if defined(ENABLE_DUCKDB_FFI)
struct vx_array *vx_array_create_empty_from_duckdb_table(const duckdb_logical_type *type_array,
                                                         const char *const *names,
                                                         int len,
                                                         struct vx_error **error);
#endif

#if defined(ENABLE_DUCKDB_FFI)
struct vx_array *vx_array_append_duckdb_chunk(struct vx_array *array, duckdb_data_chunk chunk);
#endif

#if defined(ENABLE_DUCKDB_FFI)
struct vx_conversion_cache *vx_conversion_cache_create(unsigned int id);
#endif

#if defined(ENABLE_DUCKDB_FFI)
void vx_conversion_cache_free(struct vx_conversion_cache *buffer);
#endif

void vx_error_free(struct vx_error *error);

/**
 * Open a file at the given path on the file system.
 */
struct vx_file_reader *vx_file_open_reader(const struct vx_file_open_options *options,
                                           struct vx_error **error);

struct vx_file_writer *vx_file_create(const struct vx_file_create_options *options,
                                      struct vx_error **error);

void vx_file_write_array(struct vx_file_writer *file,
                         struct vx_array *ffi_array,
                         struct vx_error **error);

struct vx_file_statistics *vx_file_extract_statistics(struct vx_file_reader *file);

void vx_file_statistics_free(struct vx_file_statistics *stat);

/**
 * Get a readonly pointer to the DType of the data inside of the file.
 *
 * The pointer's lifetime is tied to the lifetime of the underlying file, so it should not be
 * dereferenced after the file has been freed.
 */
const struct vx_dtype *vx_file_dtype(const struct vx_file_reader *file);

/**
 * Build a new `vx_array_stream` that return a series of `vx_array`s scan over a `vx_file`.
 */
struct vx_array_stream *vx_file_scan(const struct vx_file_reader *file,
                                     const struct vx_file_scan_options *opts,
                                     struct vx_error **error);

/**
 * Free the file and all associated resources.
 *
 * This function will not automatically free any `vx_array_stream`s that were built from this
 * file.
 */
void vx_file_reader_free(struct vx_file_reader *file);

void vx_file_writer_free(struct vx_file_writer *file);

/**
 * Initialize native logging with the specified level.
 *
 * This function is optional, if it is not called then no runtime
 * logger will be installed.
 */
void vx_init_logging(enum vx_log_level level);

/**
 * Gets the dtype from an array `stream`, if the stream is finished the `DType` is null
 */
const struct vx_dtype *vx_array_stream_dtype(const struct vx_array_stream *stream);

/**
 * Attempt to advance the `current` pointer of the stream.
 *
 * A return value of `true` indicates that another element was pulled from the stream, and a return
 * of `false` indicates that the stream is finished.
 *
 * It is an error to call this function again after the stream is finished.
 */
struct vx_array *vx_array_stream_next(struct vx_array_stream *stream, struct vx_error **error);

/**
 * Predicate function to check if the array stream is finished.
 */
bool vx_array_stream_finished(const struct vx_array_stream *stream);

/**
 * Free the array stream and all associated resources.
 */
void vx_array_stream_free(struct vx_array_stream *stream);

#ifdef __cplusplus
}
#endif
