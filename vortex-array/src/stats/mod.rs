// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Array statistics: the statistics an array can carry ([`Stat`]), how exact a value is
//! ([`Precision`]) and its bounds, the [`StatsProvider`] traits that read them, and the cache each
//! array keeps ([`ArrayStats`], [`StatsSet`]).

use arrow_buffer::BooleanBufferBuilder;
use arrow_buffer::MutableBuffer;
use arrow_buffer::bit_iterator::BitIterator;
use enum_iterator::last;
pub use stats_set::*;

mod array;
mod bound;
pub mod flatbuffers;
mod precision;
mod provider;
mod stat;
mod stat_bound;
mod stats_set;

pub use array::*;
pub use bound::*;
pub use precision::*;
pub use provider::*;
pub use stat::*;
pub use stat_bound::*;
use vortex_error::VortexExpect;

pub fn as_stat_bitset_bytes(stats: &[Stat]) -> Vec<u8> {
    let max_stat = u8::from(last::<Stat>().vortex_expect("last stat")) as usize + 1;
    // TODO(ngates): use vortex-buffer::BitBuffer
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
