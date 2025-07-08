#pragma once

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "arrow_c_abi.hpp"
#include "rust/cxx.h"
#include "vortex-cxx/src/lib.rs.h"

namespace vortex {

class VortexException : public std::runtime_error {
 public:
  explicit VortexException(const std::string &message)
      : std::runtime_error(message) {}
};

class Array {
 public:
  // TODO: Remove this once we have a real array constructor
  static Array create_dummy() { return Array(ffi::create_dummy()); }

  explicit Array(rust::Box<ffi::VortexArray> impl) : impl_(std::move(impl)) {}

  size_t len() const { return ffi::array_len(*impl_); }

  rust::Box<ffi::VortexDType> dtype() const { return ffi::array_dtype(*impl_); }

  bool is_null(size_t index) const {
    try {
      return ffi::array_is_null(*impl_, index);
    } catch (const rust::cxxbridge1::Error &e) {
      throw VortexException(e.what());
    }
  }

  rust::Box<ffi::VortexScalar> scalar_at(size_t index) const {
    try {
      return ffi::array_scalar_at(*impl_, index);
    } catch (const rust::cxxbridge1::Error &e) {
      throw VortexException(e.what());
    }
  }

  Array slice(size_t start, size_t stop) const {
    try {
      return Array(ffi::array_slice(*impl_, start, stop));
    } catch (const rust::cxxbridge1::Error &e) {
      throw VortexException(e.what());
    }
  }

  // Convert to native Arrow C ABI structures
  std::pair<ArrowArray, ArrowSchema> to_arrow_c_abi() const {
    try {
      auto arrow_c_structs = ffi::array_to_arrow_with_schema(*impl_);

      // Convert from our C-compatible structs to native Arrow C ABI
      ArrowArray array;
      array.length = arrow_c_structs.array.length;
      array.null_count = arrow_c_structs.array.null_count;
      array.offset = arrow_c_structs.array.offset;
      array.n_buffers = arrow_c_structs.array.n_buffers;
      array.n_children = arrow_c_structs.array.n_children;
      array.buffers =
          reinterpret_cast<const void **>(arrow_c_structs.array.buffers);
      array.children = reinterpret_cast<struct ArrowArray **>(
          arrow_c_structs.array.children);
      array.dictionary = reinterpret_cast<struct ArrowArray *>(
          arrow_c_structs.array.dictionary);
      array.release = reinterpret_cast<void (*)(struct ArrowArray *)>(
          arrow_c_structs.array.release);
      array.private_data =
          reinterpret_cast<void *>(arrow_c_structs.array.private_data);

      ArrowSchema schema;
      schema.format =
          reinterpret_cast<const char *>(arrow_c_structs.schema.format);
      schema.name = reinterpret_cast<const char *>(arrow_c_structs.schema.name);
      schema.metadata =
          reinterpret_cast<const char *>(arrow_c_structs.schema.metadata);
      schema.flags = arrow_c_structs.schema.flags;
      schema.n_children = arrow_c_structs.schema.n_children;
      schema.children = reinterpret_cast<struct ArrowSchema **>(
          arrow_c_structs.schema.children);
      schema.dictionary = reinterpret_cast<struct ArrowSchema *>(
          arrow_c_structs.schema.dictionary);
      schema.release = reinterpret_cast<void (*)(struct ArrowSchema *)>(
          arrow_c_structs.schema.release);
      schema.private_data =
          reinterpret_cast<void *>(arrow_c_structs.schema.private_data);

      return {array, schema};
    } catch (const rust::cxxbridge1::Error &e) {
      throw VortexException(e.what());
    }
  }

 private:
  rust::Box<ffi::VortexArray> impl_;
};

class DType {
 public:
  explicit DType(rust::Box<ffi::VortexDType> impl) : impl_(std::move(impl)) {}

  ffi::DType dtype_variant() const { return ffi::dtype_variant(*impl_); }

  ffi::PType ptype_variant() const { return ffi::ptype_variant(*impl_); }

  bool is_nullable() const { return ffi::dtype_is_nullable(*impl_); }

 private:
  rust::Box<ffi::VortexDType> impl_;
};

class Scalar {
 public:
  explicit Scalar(rust::Box<ffi::VortexScalar> impl) : impl_(std::move(impl)) {}

  bool is_null() const { return ffi::scalar_is_null(*impl_); }

  bool as_bool() const {
    try {
      return ffi::scalar_as_bool(*impl_);
    } catch (const rust::cxxbridge1::Error &e) {
      throw VortexException(e.what());
    }
  }

  uint8_t as_u8() const {
    try {
      return ffi::scalar_as_u8(*impl_);
    } catch (const rust::cxxbridge1::Error &e) {
      throw VortexException(e.what());
    }
  }

  uint16_t as_u16() const {
    try {
      return ffi::scalar_as_u16(*impl_);
    } catch (const rust::cxxbridge1::Error &e) {
      throw VortexException(e.what());
    }
  }

  uint32_t as_u32() const {
    try {
      return ffi::scalar_as_u32(*impl_);
    } catch (const rust::cxxbridge1::Error &e) {
      throw VortexException(e.what());
    }
  }

  uint64_t as_u64() const {
    try {
      return ffi::scalar_as_u64(*impl_);
    } catch (const rust::cxxbridge1::Error &e) {
      throw VortexException(e.what());
    }
  }

  int8_t as_i8() const {
    try {
      return ffi::scalar_as_i8(*impl_);
    } catch (const rust::cxxbridge1::Error &e) {
      throw VortexException(e.what());
    }
  }

  int16_t as_i16() const {
    try {
      return ffi::scalar_as_i16(*impl_);
    } catch (const rust::cxxbridge1::Error &e) {
      throw VortexException(e.what());
    }
  }

  int32_t as_i32() const {
    try {
      return ffi::scalar_as_i32(*impl_);
    } catch (const rust::cxxbridge1::Error &e) {
      throw VortexException(e.what());
    }
  }

  int64_t as_i64() const {
    try {
      return ffi::scalar_as_i64(*impl_);
    } catch (const rust::cxxbridge1::Error &e) {
      throw VortexException(e.what());
    }
  }

  float as_f32() const {
    try {
      return ffi::scalar_as_f32(*impl_);
    } catch (const rust::cxxbridge1::Error &e) {
      throw VortexException(e.what());
    }
  }

  double as_f64() const {
    try {
      return ffi::scalar_as_f64(*impl_);
    } catch (const rust::cxxbridge1::Error &e) {
      throw VortexException(e.what());
    }
  }

  std::string as_string() const {
    try {
      return std::string(ffi::scalar_as_string(*impl_));
    } catch (const rust::cxxbridge1::Error &e) {
      throw VortexException(e.what());
    }
  }

 private:
  rust::Box<ffi::VortexScalar> impl_;
};

class File {
 public:
  static File open(const std::string &path) {
    try {
      return File(ffi::open_file(path));
    } catch (const rust::cxxbridge1::Error &e) {
      throw VortexException(e.what());
    }
  }

  explicit File(rust::Box<ffi::VortexFile> impl) : impl_(std::move(impl)) {}

  uint64_t row_count() const { return ffi::file_row_count(*impl_); }

  Array read_all() const {
    try {
      return Array(ffi::file_read_all(*impl_));
    } catch (const rust::cxxbridge1::Error &e) {
      throw VortexException(e.what());
    }
  }

 private:
  rust::Box<ffi::VortexFile> impl_;
};

}  // namespace vortex