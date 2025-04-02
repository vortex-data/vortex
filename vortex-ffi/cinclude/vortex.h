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

#define LOG_LEVEL_OFF 0

#define LOG_LEVEL_ERROR 1

#define LOG_LEVEL_WARN 2

#define LOG_LEVEL_INFO 3

#define LOG_LEVEL_DEBUG 4

#define LOG_LEVEL_TRACE 5

/**
 * The logical types of elements in Vortex arrays.
 *
 * Vortex arrays preserve a single logical type, while the encodings allow for multiple
 * physical ways to encode that type.
 */
typedef struct DType DType;

/**
 * The FFI interface for an [`Array`].
 *
 * Because dyn Trait pointers cannot be shared across FFI, we create a new struct to hold
 * the wide pointer. The C FFI only seems a pointer to this structure, and can pass it into
 * one of the various `FFIArray_*` functions.
 */
typedef struct Array Array;

/**
 * FFI-exposed stream interface.
 */
typedef struct ArrayStream ArrayStream;

#if defined(ENABLE_DUCKDB_FFI)
typedef struct FFIConversionCache FFIConversionCache;
#endif

typedef struct File File;

/**
 * Options supplied for opening a file.
 */
typedef struct FileOpenOptions {
  /**
   * URI for opening the file.
   * This must be a valid URI, even the files (file:///path/to/file)
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
} FileOpenOptions;

/**
 * Whole file statistics.
 */
typedef struct FileStatistics {
  /**
   * The exact number of rows in the file.
   */
  uint64_t num_rows;
} FileStatistics;

/**
 * Scan options provided by an FFI client calling the `File_scan` function.
 */
typedef struct FileScanOptions {
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
} FileScanOptions;



/**
 * Get the length of the array.
 */
uint64_t FFIArray_len(const struct Array *ffi_array);

/**
 * Get a pointer to the data type for an array.
 *
 * Note that this pointer is tied to the lifetime of the array, and the caller is responsible
 * for ensuring that it is never dereferenced after the array has been freed.
 */
const struct DType *FFIArray_dtype(const struct Array *ffi_array);

const struct Array *FFIArray_get_field(const struct Array *ffi_array, uint32_t index);

/**
 * Free the array and all associated resources.
 */
int32_t FFIArray_free(struct Array *ffi_array);

struct Array *FFIArray_slice(const struct Array *array, uint32_t start, uint32_t stop);

bool FFIArray_is_null(const struct Array *array, uint32_t index);

uint32_t FFIArray_null_count(const struct Array *array);

/**
 * Write the UTF-8 string at `index` in the array into the provided destination buffer, recording
 * the length in `len`.
 */
void FFIArray_get_utf8(const struct Array *array, uint32_t index, void *dst, int *len);

/**
 * Write the UTF-8 string at `index` in the array into the provided destination buffer, recording
 * the length in `len`.
 */
void FFIArray_get_binary(const struct Array *array, uint32_t index, void *dst, int *len);

/**
 * Pointer to a `DType` value that has been heap-allocated.
 * Create a new simple dtype.
 */
struct DType *DType_new(uint8_t variant, bool nullable);

/**
 * Create a new List type with the provided element type.
 *
 * Upon successful return, this function moves the value out of the provided element pointer,
 * so it is not safe to reference afterward.
 */
struct DType *DType_new_list(struct DType *element, bool nullable);

struct DType *DType_new_struct(const char *const *names,
                               struct DType *const *dtypes,
                               uint32_t len,
                               bool nullable);

/**
 * Free an [`DType`] and all associated resources.
 */
void DType_free(struct DType *dtype);

/**
 * Get the dtype variant tag for an [`DType`].
 */
uint8_t DType_get(const struct DType *dtype);

bool DType_nullable(const struct DType *dtype);

/**
 * For `DTYPE_STRUCT` variant DTypes, get the number of fields.
 */
uint32_t DType_field_count(const struct DType *dtype);

void DType_field_name(const struct DType *dtype, uint32_t index, void *dst, int *len);

/**
 * Get the dtype of a field in a `DTYPE_STRUCT` variant DType.
 *
 * This returns a new owned, allocated copy of the DType that must be freed subsequently
 * by the caller.
 */
struct DType *DType_field_dtype(const struct DType *dtype, uint32_t index);

/**
 * For a list DType, get the inner element type.
 *
 * The pointee's lifetime is tied to the lifetime of the list DType. It should not be
 * accessed after the list DType has been freed.
 */
const struct DType *DType_element_type(const struct DType *dtype);

bool DType_is_time(const struct DType *dtype);

bool DType_is_date(const struct DType *dtype);

bool DType_is_timestamp(const struct DType *dtype);

uint8_t DType_time_unit(const struct DType *dtype);

void DType_time_zone(const struct DType *dtype, void *dst, int *len);

#if defined(ENABLE_DUCKDB_FFI)
duckdb_logical_type DType_to_duckdb_logical_type(struct DType *dtype);
#endif

#if defined(ENABLE_DUCKDB_FFI)
/**
 * Back a single chunk of the array as a duckdb data chunk.
 * The initial call should pass offset = 0.
 * The offset is returned to the caller, which can be used to request the next chunk.
 * 0 is returned when the stream is finished.
 */
unsigned int FFIArray_to_duckdb_chunk(struct Array *stream,
                                      unsigned int offset,
                                      duckdb_data_chunk data_chunk_ptr,
                                      struct FFIConversionCache *cache);
#endif

#if defined(ENABLE_DUCKDB_FFI)
struct FFIConversionCache *ConversionCache_create(unsigned int id);
#endif

#if defined(ENABLE_DUCKDB_FFI)
void ConversionCache_free(struct FFIConversionCache *buffer);
#endif

/**
 * Open a file at the given path on the file system.
 */
struct File *File_open(const struct FileOpenOptions *options);

struct FileStatistics *File_statistics(struct File *file);

void FileStatistics_free(struct FileStatistics *stat);

/**
 * Get a readonly pointer to the DType of the data inside of the file.
 *
 * The pointer's lifetime is tied to the lifetime of the underlying file, so it should not be
 * dereferenced after the file has been freed.
 */
const struct DType *File_dtype(const struct File *file);

/**
 * Build a new `FFIArrayStream` that return a series of `FFIArray`s scan over a `FFIFile`.
 */
struct ArrayStream *File_scan(const struct File *file, const struct FileScanOptions *opts);

/**
 * Free the file and all associated resources.
 *
 * This function will not automatically free any `FFIArrayStream`s that were built from this
 * file.
 */
void File_free(struct File *file);

/**
 * Initialize native logging with the specified level.
 *
 * This function is optional, if it is not called then no runtime
 * logger will be installed.
 */
void vortex_init_logging(uint8_t level);

const struct DType *FFIArrayStream_dtype(const struct ArrayStream *stream);

/**
 * Attempt to advance the `current` pointer of the stream.
 *
 * A return value of `true` indicates that another element was pulled from the stream, and a return
 * of `false` indicates that the stream is finished.
 *
 * It is an error to call this function again after the stream is finished.
 */
bool FFIArrayStream_next(struct ArrayStream *stream);

/**
 * Predicate function to check if the array stream is finished.
 */
bool FFIArrayStream_finished(const struct ArrayStream *stream);

/**
 * Get the current array batch from the stream. Returns a unique pointer.
 *
 * It is an error to call this function if the stream is already finished.
 *
 * # Safety
 *
 * This function is unsafe because it dereferences the `stream` pointer.
 */
struct Array *FFIArrayStream_current(struct ArrayStream *stream);

/**
 * Free the array stream and all associated resources.
 */
int32_t FFIArrayStream_free(struct ArrayStream *stream);

#ifdef __cplusplus
}
#endif
