// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include <vortex_cudf_dictionary.hpp>

// A second, ordinary C++ translation unit checks header-only/ODR use and defaults.
std::unique_ptr<cudf::column> concatenate_with_defaults(std::vector<cudf::column_view> const &batches) {
    return vortex::cuda::concatenate_string_dictionaries(batches);
}
