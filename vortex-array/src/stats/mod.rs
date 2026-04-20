// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Traits and utilities to compute and access array statistics.

use enum_iterator::last;
pub use stats_set::*;

mod array;
pub mod flatbuffers;
mod stats_set;

pub use array::*;
use vortex_buffer::BitBufferMut;
use vortex_buffer::BitIterator;
use vortex_error::VortexExpect;

use crate::expr::stats::Stat;

/// Statistics that are used for pruning files (i.e., we want to ensure they are computed when compressing/writing).
/// Sum is included for boolean arrays.
pub const PRUNING_STATS: &[Stat] = &[
    Stat::Min,
    Stat::Max,
    Stat::Sum,
    Stat::NullCount,
    Stat::NaNCount,
];

pub fn as_stat_bitset_bytes(stats: &[Stat]) -> Vec<u8> {
    let max_stat = u8::from(last::<Stat>().vortex_expect("last stat")) as usize + 1;
    let mut stat_bitset = BitBufferMut::with_capacity(max_stat);
    stat_bitset.append_n(false, max_stat);
    for stat in stats {
        stat_bitset.set(u8::from(*stat) as usize);
    }

    stat_bitset.freeze().inner().as_slice().to_vec()
}

pub fn stats_from_bitset_bytes(bytes: &[u8]) -> Vec<Stat> {
    BitIterator::new(bytes, 0, bytes.len() * 8)
        .enumerate()
        .filter_map(|(i, b)| b.then_some(i))
        // Filter out indices failing conversion, these are stats written by newer version of library
        .filter_map(|i| {
            let Ok(stat) = u8::try_from(i) else {
                tracing::debug!("invalid stat encountered: {i}");
                return None;
            };
            Stat::try_from(stat).ok()
        })
        .collect::<Vec<_>>()
}
