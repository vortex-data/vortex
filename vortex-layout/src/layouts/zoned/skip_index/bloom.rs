// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Skip index implementation for the Bloom filter.

use std::sync::Arc;

use vortex_array::aggregate_fn::AggregateFnVTable;
use vortex_array::stats::rewrite::StatsRewriteRuleRef;

use crate::layouts::zoned::aggregates::bloom_filter::BloomFilter;
use crate::layouts::zoned::aggregates::bloom_filter::BloomOptions;
use crate::layouts::zoned::aggregates::bloom_filter::scalar_fn::BloomContains;
use crate::layouts::zoned::aggregates::bloom_filter::scalar_fn::BloomEqRewrite;
use crate::layouts::zoned::skip_index::SkipIndex;

/// An implementation of a skip index for the [`BloomFilter`] aggregate.
///
/// Instances carry the options used when writing the index. Register Bloom
/// support with a session using
/// `session.register_skip_index(&index)`, where `index` is a [`BloomSkipIndex`].
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
}

impl SkipIndex for BloomSkipIndex {
    type Aggregate = BloomFilter;
    type Scalar = BloomContains;

    fn aggregate_vtable(&self) -> Self::Aggregate {
        BloomFilter
    }

    fn scalar_vtable(&self) -> Self::Scalar {
        BloomContains
    }

    fn options(&self) -> <Self::Aggregate as AggregateFnVTable>::Options {
        self.options.clone()
    }

    fn rewrite_rules(&self) -> Vec<StatsRewriteRuleRef> {
        vec![Arc::new(BloomEqRewrite)]
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::aggregate_fn::AggregateFnVTableExt;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::expr::bound::eq;
    use vortex_array::expr::bound::lit;
    use vortex_array::expr::bound::root;
    use vortex_array::stats::rewrite::StatsRewriteCtx;
    use vortex_error::VortexResult;
    use vortex_error::vortex_err;

    use super::*;
    use crate::layouts::zoned::aggregates::bloom_filter::scalar_fn::BloomContains;
    use crate::layouts::zoned::skip_index::SkipIndexSessionExt;

    #[test]
    fn repeated_registration_installs_one_bloom_rewrite() -> VortexResult<()> {
        let session = vortex_array::array_session();
        let aggregate_fns = [BloomFilter.bind(BloomOptions::default())];
        let predicate = eq(
            root(DType::Primitive(PType::I64, Nullability::NonNullable)),
            lit(42i64),
        );

        for registration_session in [&session, &session.clone()] {
            let skip_index = BloomSkipIndex::default();
            registration_session.register_skip_index(&skip_index);
            let proof = StatsRewriteCtx::new(&session)
                .with_aggregate_fns(&aggregate_fns)
                .falsify(&predicate)?
                .ok_or_else(|| vortex_err!("expected an equality falsifier"))?;

            let mut pending = vec![&proof];
            let mut bloom_probes = 0;
            while let Some(expr) = pending.pop() {
                bloom_probes += usize::from(expr.is::<BloomContains>());
                pending.extend(expr.children());
            }
            assert_eq!(bloom_probes, 1);
        }
        Ok(())
    }
}
