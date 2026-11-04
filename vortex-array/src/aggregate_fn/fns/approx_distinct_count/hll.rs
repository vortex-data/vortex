// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! An approximate distinct count backed by an Apache DataSketches HyperLogLog sketch.

use datasketches::hll::HllType;
use datasketches::hll::HllUnion;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use super::DistinctSink;
use super::accumulate_columnar;
use super::key_kind;
use crate::ArrayRef;
use crate::Columnar;
use crate::ExecutionCtx;
use crate::aggregate_fn::AggregateArgs;
use crate::aggregate_fn::AggregateFnId;
use crate::aggregate_fn::AggregateFnVTable;
use crate::aggregate_fn::EmptyOptions;
use crate::dtype::DType;
use crate::dtype::Nullability::NonNullable;
use crate::dtype::PType;
use crate::scalar::Scalar;

/// Log2 of the number of HLL registers: 4096 registers, about 1.6% relative standard error.
const LG_K: u8 = 12;

/// Approximately count the distinct non-null values of an input with a HyperLogLog sketch.
///
/// Like [`ApproxDistinctCount`](super::ApproxDistinctCount), the serialized partial, which the
/// file footer stores, is only the count. In memory the partial is a sketch of fixed size, so
/// merging two sketches never double-counts a value they share, at the cost of the sketch's
/// estimation error. Merging when either side is only a count adds the counts.
///
/// Applies to the same dtypes as [`ApproxDistinctCount`](super::ApproxDistinctCount), and returns
/// a non-null `u64`.
#[derive(Clone, Debug)]
pub struct ApproxDistinctCountHll;

/// The partial state of [`ApproxDistinctCountHll`].
#[derive(Clone, Debug)]
pub enum ApproxDistinctCountHllPartial {
    /// A sketch of every value seen.
    Sketch(Box<HllUnion>),
    /// An estimate of the number of distinct values seen, read back from a serialized partial.
    Inexact(u64),
}

impl ApproxDistinctCountHllPartial {
    fn empty() -> VortexResult<Self> {
        Ok(Self::Sketch(Box::new(new_union()?)))
    }

    /// The estimated number of distinct values.
    pub fn count(&self) -> u64 {
        match self {
            Self::Sketch(union) => estimate(union),
            Self::Inexact(count) => *count,
        }
    }
}

fn new_union() -> VortexResult<HllUnion> {
    HllUnion::new(LG_K).map_err(|e| vortex_err!("Failed to create HLL union: {e}"))
}

fn estimate(union: &HllUnion) -> u64 {
    // The estimate is non-negative and far below `u64::MAX`.
    #[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let count = union.estimate().round() as u64;
    count
}

impl DistinctSink for ApproxDistinctCountHllPartial {
    fn add_fixed(&mut self, keys: impl Iterator<Item = u128>) -> VortexResult<()> {
        match self {
            Self::Sketch(union) => keys.for_each(|key| union.update_value(key)),
            Self::Inexact(count) => {
                let mut union = new_union()?;
                keys.for_each(|key| union.update_value(key));
                *count += estimate(&union);
            }
        }
        Ok(())
    }

    fn add_bytes<'a>(&mut self, values: impl Iterator<Item = &'a [u8]>) -> VortexResult<()> {
        match self {
            Self::Sketch(union) => values.for_each(|value| union.update_value(value)),
            Self::Inexact(count) => {
                let mut union = new_union()?;
                values.for_each(|value| union.update_value(value));
                *count += estimate(&union);
            }
        }
        Ok(())
    }
}

impl AggregateFnVTable for ApproxDistinctCountHll {
    type Options = EmptyOptions;
    type Partial = ApproxDistinctCountHllPartial;

    fn id(&self) -> AggregateFnId {
        static ID: CachedId = CachedId::new("vortex.approx_distinct_count_hll");
        *ID
    }

    fn serialize(&self, _options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(vec![]))
    }

    fn deserialize(
        &self,
        _metadata: &[u8],
        _session: &VortexSession,
    ) -> VortexResult<Self::Options> {
        Ok(EmptyOptions)
    }

    fn return_dtype(&self, _options: &Self::Options, input_dtype: &DType) -> Option<DType> {
        key_kind(input_dtype).map(|_| DType::Primitive(PType::U64, NonNullable))
    }

    fn partial_dtype(&self, options: &Self::Options, input_dtype: &DType) -> Option<DType> {
        self.return_dtype(options, input_dtype)
    }

    fn empty_partial(&self, args: AggregateArgs<'_, Self::Options>) -> VortexResult<Self::Partial> {
        if key_kind(args.dtype).is_none() {
            vortex_bail!(
                "ApproxDistinctCountHll does not support dtype {}",
                args.dtype
            );
        }
        ApproxDistinctCountHllPartial::empty()
    }

    fn partial_from_scalar(
        &self,
        args: AggregateArgs<'_, Self::Options>,
        scalar: &Scalar,
    ) -> VortexResult<Self::Partial> {
        let count = scalar
            .as_primitive()
            .typed_value::<u64>()
            .vortex_expect("approx_distinct_count_hll partial should not be null");
        // A count of zero saw no values, which a sketch represents exactly.
        if count == 0 {
            return self.empty_partial(args);
        }
        Ok(ApproxDistinctCountHllPartial::Inexact(count))
    }

    fn merge_partials(
        &self,
        _args: AggregateArgs<'_, Self::Options>,
        first: Self::Partial,
        second: Self::Partial,
    ) -> VortexResult<Self::Partial> {
        use ApproxDistinctCountHllPartial::Inexact;
        use ApproxDistinctCountHllPartial::Sketch;

        Ok(match (first, second) {
            (Sketch(mut a), Sketch(b)) => {
                a.update(&b.to_sketch(HllType::Hll8));
                Sketch(a)
            }
            (a, b) => Inexact(a.count() + b.count()),
        })
    }

    /// Only the estimated count is serialized: the sketch is dropped.
    fn to_scalar(
        &self,
        _args: AggregateArgs<'_, Self::Options>,
        partial: &Self::Partial,
    ) -> VortexResult<Scalar> {
        Ok(Scalar::primitive(partial.count(), NonNullable))
    }

    fn is_saturated(
        &self,
        _args: AggregateArgs<'_, Self::Options>,
        _partial: &Self::Partial,
    ) -> bool {
        false
    }

    fn accumulate(
        &self,
        _args: AggregateArgs<'_, Self::Options>,
        partial: &mut Self::Partial,
        batch: &Columnar,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        accumulate_columnar(partial, batch, ctx)
    }

    fn finalize(
        &self,
        _args: AggregateArgs<'_, Self::Options>,
        partials: ArrayRef,
    ) -> VortexResult<ArrayRef> {
        Ok(partials)
    }

    fn finalize_scalar(
        &self,
        args: AggregateArgs<'_, Self::Options>,
        partial: &Self::Partial,
    ) -> VortexResult<Scalar> {
        self.to_scalar(args, partial)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use vortex_buffer::Buffer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use super::*;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::aggregate_fn::Accumulator;
    use crate::aggregate_fn::DynAccumulator;
    use crate::arrays::VarBinViewArray;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    fn count(acc: &mut Accumulator<ApproxDistinctCountHll>) -> VortexResult<u64> {
        Ok(acc
            .finish()?
            .as_primitive()
            .typed_value::<u64>()
            .vortex_expect("non-null"))
    }

    /// Small sets are counted exactly: the sketch keeps a list of coupons until it is large.
    #[test]
    fn small_counts_are_exact() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let mut acc = Accumulator::try_new(
            ApproxDistinctCountHll,
            EmptyOptions,
            DType::Utf8(NonNullable),
        )?;
        acc.accumulate(
            &VarBinViewArray::from_iter_str(["a", "b", "a", "c"]).into_array(),
            &mut ctx,
        )?;
        assert_eq!(count(&mut acc)?, 3);
        Ok(())
    }

    /// Merged sketches count shared values once, within the sketch's error.
    #[test]
    fn merged_sketches_do_not_double_count() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let dtype = DType::Primitive(PType::U64, NonNullable);
        let n = 100_000u64;
        let mut file = Accumulator::try_new(ApproxDistinctCountHll, EmptyOptions, dtype.clone())?;
        for start in (0..n).step_by(1024) {
            let mut chunk =
                Accumulator::try_new(ApproxDistinctCountHll, EmptyOptions, dtype.clone())?;
            chunk.accumulate(&Buffer::from_iter(0..1024u64).into_array(), &mut ctx)?;
            chunk.accumulate(
                &Buffer::from_iter(start..(start + 1024).min(n)).into_array(),
                &mut ctx,
            )?;
            file.merge_from(&mut chunk)?;
        }
        let estimate = count(&mut file)?;
        assert!(estimate.abs_diff(n) < n / 20, "estimate {estimate} for {n}");
        Ok(())
    }
}
