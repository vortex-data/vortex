// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;
use std::num::NonZeroUsize;

use vortex_buffer::BufferString;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::Columnar;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::aggregate_fn::AggregateFnId;
use crate::aggregate_fn::AggregateFnRef;
use crate::aggregate_fn::AggregateFnSatisfaction;
use crate::aggregate_fn::AggregateFnVTable;
use crate::aggregate_fn::NumericalAggregateOpts;
use crate::aggregate_fn::fns::min::Min;
use crate::aggregate_fn::fns::min_max::MinMax;
use crate::aggregate_fn::fns::min_max::min_max;
use crate::dtype::DType;
use crate::partial_ord::partial_min;
use crate::scalar::Scalar;
use crate::scalar::ScalarTruncation;
use crate::scalar::lower_bound;

/// Options for [`BoundedMin`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BoundedMinOptions {
    /// Maximum byte length for UTF8/Binary bounds.
    pub max_bytes: NonZeroUsize,
}

impl Display for BoundedMinOptions {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.max_bytes.get())
    }
}

/// Compute a byte-bounded lower bound for the minimum non-null value of a UTF8/Binary array.
#[derive(Clone, Debug)]
pub struct BoundedMin;

enum BoundedMinState {
    Empty,
    Value(Scalar),
}

/// Partial accumulator state for the bounded minimum aggregate.
pub struct BoundedMinPartial {
    state: BoundedMinState,
    element_dtype: DType,
    max_bytes: NonZeroUsize,
}

impl BoundedMinPartial {
    fn merge(&mut self, min: Scalar) {
        if min.is_null() {
            return;
        }

        self.state = match std::mem::replace(&mut self.state, BoundedMinState::Empty) {
            BoundedMinState::Empty => BoundedMinState::Value(min),
            BoundedMinState::Value(current) => BoundedMinState::Value(
                partial_min(min, current).vortex_expect("incomparable bounded min scalars"),
            ),
        };
    }
}

impl AggregateFnVTable for BoundedMin {
    type Options = BoundedMinOptions;
    type Partial = BoundedMinPartial;

    fn id(&self) -> AggregateFnId {
        AggregateFnId::new("vortex.bounded_min")
    }

    fn serialize(&self, options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        let max_bytes = u64::try_from(options.max_bytes.get())?;
        Ok(Some(max_bytes.to_le_bytes().to_vec()))
    }

    fn deserialize(
        &self,
        metadata: &[u8],
        _session: &VortexSession,
    ) -> VortexResult<Self::Options> {
        vortex_ensure!(
            metadata.len() == size_of::<u64>(),
            "BoundedMin options expected {} bytes, got {}",
            size_of::<u64>(),
            metadata.len()
        );
        let mut bytes = [0u8; size_of::<u64>()];
        bytes.copy_from_slice(metadata);
        let max_bytes = usize::try_from(u64::from_le_bytes(bytes))?;
        vortex_ensure!(max_bytes > 0, "BoundedMin requires max_bytes > 0");
        Ok(BoundedMinOptions {
            max_bytes: NonZeroUsize::new(max_bytes).vortex_expect("checked non-zero max_bytes"),
        })
    }

    fn return_dtype(&self, options: &Self::Options, input_dtype: &DType) -> Option<DType> {
        supported_dtype(options, input_dtype).map(DType::as_nullable)
    }

    fn can_satisfy(
        &self,
        options: &Self::Options,
        requested: &AggregateFnRef,
    ) -> AggregateFnSatisfaction {
        if let Some(other) = requested.as_opt::<Self>() {
            return if other == options {
                AggregateFnSatisfaction::Exact
            } else if options.max_bytes >= other.max_bytes {
                AggregateFnSatisfaction::Approximate
            } else {
                AggregateFnSatisfaction::No
            };
        }

        // The stored bound skips NaNs, so it cannot stand in for a NaN-including minimum.
        if requested
            .as_opt::<Min>()
            .is_some_and(|options| options.skip_nans)
        {
            AggregateFnSatisfaction::Approximate
        } else {
            AggregateFnSatisfaction::No
        }
    }

    fn partial_dtype(&self, options: &Self::Options, input_dtype: &DType) -> Option<DType> {
        self.return_dtype(options, input_dtype)
    }

    fn empty_partial(
        &self,
        options: &Self::Options,
        input_dtype: &DType,
    ) -> VortexResult<Self::Partial> {
        Ok(BoundedMinPartial {
            state: BoundedMinState::Empty,
            element_dtype: input_dtype.clone(),
            max_bytes: options.max_bytes,
        })
    }

    fn combine_partials(&self, partial: &mut Self::Partial, other: Scalar) -> VortexResult<()> {
        partial.merge(other);
        Ok(())
    }

    fn to_scalar(&self, partial: &Self::Partial) -> VortexResult<Scalar> {
        let dtype = partial.element_dtype.as_nullable();
        match &partial.state {
            BoundedMinState::Empty => Ok(Scalar::null(dtype)),
            BoundedMinState::Value(min) => min.cast(&dtype),
        }
    }

    fn reset(&self, partial: &mut Self::Partial) {
        partial.state = BoundedMinState::Empty;
    }

    fn is_saturated(&self, _partial: &Self::Partial) -> bool {
        false
    }

    fn accumulate(
        &self,
        partial: &mut Self::Partial,
        batch: &Columnar,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        // Delegate to the existing min_max implementation for now. A dedicated bounded-min
        // aggregate would avoid computing max when only min is needed.
        let array = match batch {
            Columnar::Canonical(canonical) => canonical.clone().into_array(),
            Columnar::Constant(constant) => constant.clone().into_array(),
        };
        let Some(result) = min_max(&array, ctx, NumericalAggregateOpts::default())? else {
            return Ok(());
        };
        if let Some(bound) = truncate_min(result.min, partial.max_bytes.get())? {
            partial.merge(bound);
        }
        Ok(())
    }

    fn finalize(&self, partials: ArrayRef) -> VortexResult<ArrayRef> {
        Ok(partials)
    }

    fn finalize_scalar(&self, partial: &Self::Partial) -> VortexResult<Scalar> {
        self.to_scalar(partial)
    }
}

fn supported_dtype<'a>(_options: &BoundedMinOptions, input_dtype: &'a DType) -> Option<&'a DType> {
    MinMax
        .return_dtype(&NumericalAggregateOpts::default(), input_dtype)
        .map(|_| input_dtype)
}

fn truncate_min(value: Scalar, max_bytes: usize) -> VortexResult<Option<Scalar>> {
    let nullability = value.dtype().nullability();
    match value.dtype() {
        DType::Utf8(_) => {
            Ok(
                lower_bound(BufferString::from_scalar(value)?, max_bytes, nullability)
                    .map(|(bound, _)| bound),
            )
        }
        DType::Binary(_) => {
            Ok(
                lower_bound(ByteBuffer::from_scalar(value)?, max_bytes, nullability)
                    .map(|(bound, _)| bound),
            )
        }
        _ => Ok(Some(value)),
    }
}
#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use vortex_buffer::buffer;
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::aggregate_fn::Accumulator;
    use crate::aggregate_fn::AggregateFnSatisfaction;
    use crate::aggregate_fn::AggregateFnVTable;
    use crate::aggregate_fn::AggregateFnVTableExt;
    use crate::aggregate_fn::DynAccumulator;
    use crate::aggregate_fn::NumericalAggregateOpts;
    use crate::aggregate_fn::fns::bounded_min::BoundedMin;
    use crate::aggregate_fn::fns::bounded_min::BoundedMinOptions;
    use crate::aggregate_fn::fns::max::Max;
    use crate::aggregate_fn::fns::min::Min;
    use crate::array_session;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::VarBinViewArray;
    use crate::dtype::Nullability;
    use crate::scalar::Scalar;
    use crate::validity::Validity;

    fn max_bytes(value: usize) -> NonZeroUsize {
        NonZeroUsize::new(value).vortex_expect("non-zero max_bytes")
    }

    fn fresh_session() -> VortexSession {
        array_session()
    }

    #[test]
    fn bounded_min_truncates_utf8_to_lower_bound() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let array =
            VarBinViewArray::from_iter_str(["snowman⛄️snowman", "untruncated"]).into_array();
        let mut acc = Accumulator::try_new(
            BoundedMin,
            BoundedMinOptions {
                max_bytes: max_bytes(9),
            },
            array.dtype().clone(),
        )?;

        acc.accumulate(&array, &mut ctx)?;

        assert_eq!(
            acc.finish()?,
            Scalar::utf8("snowman", Nullability::Nullable)
        );
        Ok(())
    }

    #[test]
    fn bounded_min_keeps_fixed_width_values_exact() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let array = PrimitiveArray::new(buffer![10i32, 20, 5], Validity::NonNullable).into_array();
        let mut acc = Accumulator::try_new(
            BoundedMin,
            BoundedMinOptions {
                max_bytes: max_bytes(9),
            },
            array.dtype().clone(),
        )?;

        acc.accumulate(&array, &mut ctx)?;

        assert_eq!(
            acc.finish()?,
            Scalar::primitive(5i32, Nullability::Nullable)
        );
        Ok(())
    }

    #[test]
    fn bounded_min_null_partial_does_not_poison_existing_bound() -> VortexResult<()> {
        let mut ctx = fresh_session().create_execution_ctx();
        let values = VarBinViewArray::from_iter_bin([&[1u8][..]]).into_array();
        let mut acc = Accumulator::try_new(
            BoundedMin,
            BoundedMinOptions {
                max_bytes: max_bytes(2),
            },
            values.dtype().clone(),
        )?;

        acc.accumulate(&values, &mut ctx)?;
        acc.combine_partials(Scalar::null(values.dtype().as_nullable()))?;

        assert_eq!(
            acc.finish()?,
            Scalar::binary(buffer![1u8], Nullability::Nullable)
        );
        Ok(())
    }

    #[test]
    fn bounded_min_satisfies_min_bounds() {
        let stored = BoundedMin.bind(BoundedMinOptions {
            max_bytes: max_bytes(5),
        });
        let same = BoundedMin.bind(BoundedMinOptions {
            max_bytes: max_bytes(5),
        });
        let looser_bounded = BoundedMin.bind(BoundedMinOptions {
            max_bytes: max_bytes(4),
        });
        let tighter_bounded = BoundedMin.bind(BoundedMinOptions {
            max_bytes: max_bytes(6),
        });

        assert_eq!(stored.can_satisfy(&same), AggregateFnSatisfaction::Exact);
        assert_eq!(
            stored.can_satisfy(&looser_bounded),
            AggregateFnSatisfaction::Approximate
        );
        assert_eq!(
            stored.can_satisfy(&tighter_bounded),
            AggregateFnSatisfaction::No
        );
        assert_eq!(
            stored.can_satisfy(&Min.bind(NumericalAggregateOpts::default())),
            AggregateFnSatisfaction::Approximate
        );
        assert_eq!(
            stored.can_satisfy(&Min.bind(NumericalAggregateOpts::include_nans())),
            AggregateFnSatisfaction::No
        );
        assert_eq!(
            Min.bind(NumericalAggregateOpts::include_nans())
                .can_satisfy(&stored),
            AggregateFnSatisfaction::No
        );
        assert_eq!(
            Min.bind(NumericalAggregateOpts::default())
                .can_satisfy(&stored),
            AggregateFnSatisfaction::Approximate
        );
        assert_eq!(
            stored.can_satisfy(&Max.bind(NumericalAggregateOpts::default())),
            AggregateFnSatisfaction::No
        );
    }

    #[test]
    fn bounded_min_options_round_trip() -> VortexResult<()> {
        let options = BoundedMinOptions {
            max_bytes: max_bytes(64),
        };
        let metadata = BoundedMin
            .serialize(&options)?
            .expect("serializable options");
        let roundtrip = BoundedMin.deserialize(&metadata, &VortexSession::empty())?;

        assert_eq!(roundtrip, options);
        Ok(())
    }
}
