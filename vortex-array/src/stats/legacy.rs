// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compatibility helpers for stats still stored under the legacy [`Stat`] enum.

use crate::aggregate_fn::AggregateFnRef;
use crate::aggregate_fn::fns::nan_count::NanCount;
use crate::aggregate_fn::fns::sum::Sum;
use crate::aggregate_fn::fns::uncompressed_size_in_bytes::UncompressedSizeInBytes;
use crate::expr::stats::Stat;

pub(crate) fn legacy_stat_for_aggregate(aggregate_fn: &AggregateFnRef) -> Option<Stat> {
    if aggregate_fn.is::<Sum>() {
        return Some(Stat::Sum);
    }
    if aggregate_fn.is::<NanCount>() {
        return Some(Stat::NaNCount);
    }
    if aggregate_fn.is::<UncompressedSizeInBytes>() {
        return Some(Stat::UncompressedSizeInBytes);
    }

    None
}
