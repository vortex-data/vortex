#pragma once

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

#include "arrow_c_abi.hpp"
#include "rust/cxx.h"
#include "vortex-cxx/src/lib.rs.h"

namespace vortex {

class VortexException : public std::runtime_error {
 public:
  explicit VortexException(const std::string &message)
      : std::runtime_error(message) {}
};

class VortexFile {
 public:
  static VortexFile open(const std::string &path) {
    try {
      return VortexFile(ffi::open_file(path));
    } catch (const rust::cxxbridge1::Error &e) {
      throw VortexException(e.what());
    }
  }

  explicit VortexFile(rust::Box<ffi::VortexFile> impl)
      : impl_(std::move(impl)) {}

  uint64_t row_count() const { return ffi::file_row_count(*impl_); }

  std::pair<ArrowArray, ArrowSchema> scan_to_arrow() const;

  // class StreamIterator {
  //  public:
  //   explicit StreamIterator(rust::Box<ffi::VortexArrayStream> stream)
  //       : stream_(std::move(stream)) {}

  //   std::pair<ArrowArray, ArrowSchema> get_schema() const {
  //     try {
  //       ffi::CArrowSchema c_schema;
  //       int result = ffi::stream_get_schema(*stream_, &c_schema);
  //       if (result != 0) {
  //         throw VortexException("Failed to get schema from stream");
  //       }

  //       ArrowSchema schema;
  //       schema.format = reinterpret_cast<const char *>(c_schema.format);
  //       schema.name = reinterpret_cast<const char *>(c_schema.name);
  //       schema.metadata = reinterpret_cast<const char *>(c_schema.metadata);
  //       schema.flags = c_schema.flags;
  //       schema.n_children = c_schema.n_children;
  //       schema.children = reinterpret_cast<struct ArrowSchema
  //       **>(c_schema.children); schema.dictionary = reinterpret_cast<struct
  //       ArrowSchema *>(c_schema.dictionary); schema.release =
  //       reinterpret_cast<void (*)(struct ArrowSchema *)>(c_schema.release);
  //       schema.private_data = reinterpret_cast<void
  //       *>(c_schema.private_data);

  //       // Return empty array with schema for now
  //       ArrowArray array = {};
  //       return {array, schema};
  //     } catch (const rust::cxxbridge1::Error &e) {
  //       throw VortexException(e.what());
  //     }
  //   }

  //   bool next(ArrowArray& array) {
  //     try {
  //       ffi::CArrowArray c_array;
  //       int result = ffi::stream_get_next(*stream_, &c_array);
  //       if (result != 0) {
  //         return false;
  //       }

  //       if (c_array.release == 0) {
  //         return false; // End of stream
  //       }

  //       array.length = c_array.length;
  //       array.null_count = c_array.null_count;
  //       array.offset = c_array.offset;
  //       array.n_buffers = c_array.n_buffers;
  //       array.n_children = c_array.n_children;
  //       array.buffers = reinterpret_cast<const void **>(c_array.buffers);
  //       array.children = reinterpret_cast<struct ArrowArray
  //       **>(c_array.children); array.dictionary = reinterpret_cast<struct
  //       ArrowArray *>(c_array.dictionary); array.release =
  //       reinterpret_cast<void (*)(struct ArrowArray *)>(c_array.release);
  //       array.private_data = reinterpret_cast<void *>(c_array.private_data);

  //       return true;
  //     } catch (const rust::cxxbridge1::Error &e) {
  //       throw VortexException(e.what());
  //     }
  //   }

  //  private:
  //   rust::Box<ffi::VortexArrayStream> stream_;
  // };

  // StreamIterator scan_to_stream() const {
  //   try {
  //     return StreamIterator(ffi::file_scan_to_stream(*impl_));
  //   } catch (const rust::cxxbridge1::Error &e) {
  //     throw VortexException(e.what());
  //   }
  // }

 private:
  rust::Box<ffi::VortexFile> impl_;
};

}  // namespace vortex