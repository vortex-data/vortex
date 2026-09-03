// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Dictionary encoding schemes for binary, integer, float, and string arrays.

mod binary;
mod float;
mod integer;
mod string;

pub use binary::BinaryDictScheme;
pub use float::FloatDictScheme;
pub use float::dictionary_encode as float_dictionary_encode;
pub use integer::IntDictScheme;
pub use integer::dictionary_encode as integer_dictionary_encode;
pub use string::StringDictScheme;

use vortex_array::arrays::varbinview::BinaryView;
use vortex_error::VortexExpect;

use crate::stats::StringStats;

/// Estimates the ratio for canonical variable-width values encoded as a dictionary.
fn varbinview_dict_compression_ratio(stats: &StringStats) -> f64 {
    let distinct_count = stats
        .distinct_count()
        .vortex_expect("distinct value count must be available");
    let distinct_value_bytes = stats
        .distinct_value_bytes()
        .vortex_expect("distinct value bytes must be available");
    let row_count = u64::from(stats.value_count()) + u64::from(stats.null_count());

    let view_size = size_of::<BinaryView>() as u64;
    let canonical_bytes = row_count * view_size + stats.value_bytes();
    let dictionary_values_bytes = u64::from(distinct_count) * view_size + distinct_value_bytes;
    let code_bits = u64::from(u32::BITS - distinct_count.leading_zeros());
    let code_bytes = (row_count * code_bits).div_ceil(8);

    // Child compression can only improve this conservative estimate.
    canonical_bytes as f64 / (dictionary_values_bytes + code_bytes) as f64
}
