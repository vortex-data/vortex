# Doxygen configuration for Vortex C++ API documentation.
# XML output is consumed by Sphinx via the Breathe extension.

PROJECT_NAME           = "Vortex C++"
OUTPUT_DIRECTORY       = _build/doxygen-cpp

# Input sources
INPUT                  = ../vortex-cxx/cpp/include/vortex
FILE_PATTERNS          = *.hpp
RECURSIVE              = NO

# We only care about XML output for Breathe
GENERATE_XML           = YES
GENERATE_HTML          = NO
GENERATE_LATEX         = NO
XML_PROGRAMLISTING     = YES

# Extract everything, even if not fully documented yet
EXTRACT_ALL            = YES
EXTRACT_PRIVATE        = NO
EXTRACT_STATIC         = YES

# Preprocessing — resolve includes but don't expand macros
ENABLE_PREPROCESSING   = YES
MACRO_EXPANSION        = NO

# Suppress warnings about undocumented members (WIP API)
WARN_IF_UNDOCUMENTED   = NO

# Exclude cxx bridge internals from documentation
EXCLUDE_SYMBOLS        = ffi::*
