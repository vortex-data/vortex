# Replaces boost::unordered_flat_{map,set} with std::unordered_{map,set}
# in the fetched onpair_cpp source tree. Idempotent.
#
# Invoked by FetchContent_Declare(PATCH_COMMAND ...).
#
# We rewrite `#include <boost/unordered/...>` to `#include <unordered_{map,set}>`
# and substitute the qualified types. OnPair only uses the public, std-compatible
# subset of boost::unordered_flat_map (operator[], find, emplace, size, iterators),
# so this is a sound substitution.

if(NOT DEFINED SRC_DIR)
    message(FATAL_ERROR "strip_boost.cmake: SRC_DIR not set")
endif()

file(GLOB_RECURSE ONPAIR_SOURCES
    "${SRC_DIR}/include/onpair/*.h"
    "${SRC_DIR}/include/onpair/*.hpp"
    "${SRC_DIR}/src/onpair/*.cpp"
    "${SRC_DIR}/src/onpair/*.h"
    "${SRC_DIR}/src/onpair/*.hpp"
)

set(_PAIR_HASH_BLOCK
"// strip_boost.cmake: std::hash<std::pair<uint64_t, uint8_t>> for unordered_map keys\n#include <cstdint>\n#include <functional>\n#include <utility>\nnamespace std {\ntemplate<> struct hash<std::pair<uint64_t, uint8_t>> {\n    size_t operator()(const std::pair<uint64_t, uint8_t>& p) const noexcept {\n        return std::hash<uint64_t>{}(p.first) ^ (std::hash<uint8_t>{}(p.second) << 1);\n    }\n};\n} // namespace std\n")

foreach(F ${ONPAIR_SOURCES})
    file(READ "${F}" CONTENT)
    string(REGEX REPLACE
        "#include[ \t]+<boost/unordered/unordered_flat_map\\.hpp>"
        "#include <unordered_map>" CONTENT "${CONTENT}")
    string(REGEX REPLACE
        "#include[ \t]+<boost/unordered/unordered_flat_set\\.hpp>"
        "#include <unordered_set>" CONTENT "${CONTENT}")
    string(REGEX REPLACE
        "#include[ \t]+<boost/unordered\\.hpp>"
        "#include <unordered_map>\n#include <unordered_set>" CONTENT "${CONTENT}")
    string(REPLACE "boost::unordered_flat_map" "std::unordered_map" CONTENT "${CONTENT}")
    string(REPLACE "boost::unordered_flat_set" "std::unordered_set" CONTENT "${CONTENT}")
    string(REPLACE "boost::unordered::unordered_flat_map" "std::unordered_map" CONTENT "${CONTENT}")
    string(REPLACE "boost::unordered::unordered_flat_set" "std::unordered_set" CONTENT "${CONTENT}")
    # Inject the pair-hash specialization once, at the top of any file that
    # keys an unordered_map by std::pair. std::hash<std::pair<...>> does not
    # exist by default; boost::unordered_flat_map shipped its own.
    string(FIND "${CONTENT}" "unordered_map<std::pair" _has_pair_key)
    if(NOT _has_pair_key EQUAL -1)
        string(FIND "${CONTENT}" "strip_boost.cmake: std::hash<std::pair" _has_block)
        if(_has_block EQUAL -1)
            set(CONTENT "${_PAIR_HASH_BLOCK}${CONTENT}")
        endif()
    endif()
    file(WRITE "${F}" "${CONTENT}")
endforeach()

# Drop find_package(Boost) and Boost link lines from onpair_cpp's CMake files
# so the build doesn't error out looking for Boost on the host.
file(GLOB_RECURSE ONPAIR_CMAKE
    "${SRC_DIR}/CMakeLists.txt"
    "${SRC_DIR}/cmake/*.cmake"
)
foreach(F ${ONPAIR_CMAKE})
    file(READ "${F}" CONTENT)
    string(REGEX REPLACE "find_package\\([ \t]*Boost[^)]*\\)" "" CONTENT "${CONTENT}")
    string(REGEX REPLACE "FetchContent_Declare\\([ \t\r\n]*Boost[^)]*\\)" "" CONTENT "${CONTENT}")
    string(REGEX REPLACE "FetchContent_MakeAvailable\\([ \t]*Boost[ \t]*\\)" "" CONTENT "${CONTENT}")
    string(REGEX REPLACE "Boost::[A-Za-z_]+" "" CONTENT "${CONTENT}")
    file(WRITE "${F}" "${CONTENT}")
endforeach()
