#include <cstdarg>
#include <cstdint>
#include <cstdlib>
#include <ostream>
#include <new>
#include "duckdb.h"

constexpr static const uintptr_t DUCKDB_STANDARD_VECTOR_SIZE = 2048;

extern "C" {

/// The DuckDB extension ABI initialization function.
void vortex_init(duckdb_database db);

/// The DuckDB extension ABI version function.
/// This function returns the version of the DuckDB library the extension is built against.
const char *vortex_version();

/// An additional function we export to expose the version of the extension itself to C++ code.
const char *vortex_extension_version();

}  // extern "C"
