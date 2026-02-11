cmake_minimum_required(VERSION 3.20)

# This CMakeLists.txt is used when building the C++ components
# as part of ClickHouse's build system.

# Source files
set(VORTEX_CLICKHOUSE_SOURCES
    VortexBlockInputFormat.cpp
    VortexBlockOutputFormat.cpp
    registerFormats.cpp
)

# Headers
set(VORTEX_CLICKHOUSE_HEADERS
    include/clickhouse_vx.h
    include/clickhouse_vx/common.h
    include/clickhouse_vx/format.h
    include/clickhouse_vx/column.h
)

# Include directories
include_directories(include)

# Note: When integrated into ClickHouse build system, this will be built
# as part of the main ClickHouse library. The vortex-clickhouse Rust crate
# will be linked as a static library.
