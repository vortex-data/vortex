// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compatibility with fixed statistics fields in historical Vortex files.
//!
//! These identifiers describe wire fields. Runtime summaries use aggregate functions and finalized
//! scalar results. Array-node hints are no longer read or written.

mod legacy_stat;
pub use legacy_stat::LegacyStat;

mod footer;
use arrow_buffer::BooleanBufferBuilder;
use arrow_buffer::MutableBuffer;
use arrow_buffer::bit_iterator::BitIterator;
use enum_iterator::last;
pub use footer::read_summary;
pub use footer::validate_selection;
pub use footer::write_summary;
use vortex_error::VortexExpect;

/// Encode the field-presence bitset used by legacy zone-map metadata.
pub fn as_stat_bitset_bytes(stats: &[LegacyStat]) -> Vec<u8> {
    let max_stat = u8::from(last::<LegacyStat>().vortex_expect("last stat")) as usize + 1;
    // TODO(connor): Use vortex-buffer::BitBuffer for the legacy field bitset.
    let mut stat_bitset = BooleanBufferBuilder::new_from_buffer(
        MutableBuffer::from_len_zeroed(max_stat.div_ceil(8)),
        max_stat,
    );
    for stat in stats {
        stat_bitset.set_bit(u8::from(*stat) as usize, true);
    }

    stat_bitset
        .finish()
        .into_inner()
        .into_vec()
        .unwrap_or_else(|b| b.to_vec())
}

/// Decode known legacy zone-map fields, ignoring unknown field identifiers.
pub fn stats_from_bitset_bytes(bytes: &[u8]) -> Vec<LegacyStat> {
    BitIterator::new(bytes, 0, bytes.len() * 8)
        .enumerate()
        .filter_map(|(i, b)| b.then_some(i))
        // Ignore unknown fields written by newer versions.
        .filter_map(|i| {
            let Ok(stat) = u8::try_from(i) else {
                tracing::debug!("invalid stat encountered: {i}");
                return None;
            };
            LegacyStat::try_from(stat).ok()
        })
        .collect::<Vec<_>>()
}

#[cfg(test)]
mod tests;
