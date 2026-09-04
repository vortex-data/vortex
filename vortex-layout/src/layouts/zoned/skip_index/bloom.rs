// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Skip index implementation for the Bloom filter.

use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::aggregate_fn::AggregateFnVTable;
use vortex_array::aggregate_fn::AggregateFnVTableExt;
use vortex_array::aggregate_fn::session::AggregateFnSessionExt;
use vortex_array::dtype::DType;
use vortex_array::scalar_fn::session::ScalarFnSessionExt;
use vortex_array::stats::StatsSessionExt;
use vortex_session::VortexSession;

use crate::layouts::zoned::aggregates::bloom_filter::BloomFilter;
use crate::layouts::zoned::aggregates::bloom_filter::BloomOptions;
use crate::layouts::zoned::aggregates::bloom_filter::scalar_fn::BloomContains;
use crate::layouts::zoned::aggregates::bloom_filter::scalar_fn::BloomEqRewrite;
use crate::layouts::zoned::skip_index::SkipIndex;

/// An implementation of a skip index for the [`BloomFilter`] aggregate.
///
/// Instances carry the options used when writing the index. Register Bloom
/// support with a session using
/// `session.register_skip_index::<BloomSkipIndex>()`.
///
/// # Writing
///
/// Bloom skip indexes are not currently included in any Vortex edition. When
/// writing a file with this index, disable edition checks using
/// `WriteOptions::disable_editions`.
///
/// For more information about how the index works, see [`BloomFilter`].
#[derive(Clone, Debug, Default)]
pub struct BloomSkipIndex {
    options: BloomOptions,
}

impl BloomSkipIndex {
    /// Create an index with explicit Bloom tuning.
    pub fn new(options: BloomOptions) -> Self {
        Self { options }
    }

    /// The persisted Bloom options.
    pub fn options(&self) -> &BloomOptions {
        &self.options
    }
}

impl SkipIndex for BloomSkipIndex {
    fn aggregate_fn(&self, input_dtype: &DType) -> Option<AggregateFnRef> {
        BloomFilter
            .return_dtype(&self.options, input_dtype)
            .map(|_| BloomFilter.bind(self.options.clone()))
    }

    fn register(session: &VortexSession)
    where
        Self: Sized,
    {
        session.aggregate_fns().register(BloomFilter);
        session.scalar_fns().register(BloomContains);
        session.stats().register_rewrite(BloomEqRewrite);
    }
}
