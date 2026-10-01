// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_buffer::Buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_mask::AllOr;
use vortex_mask::Mask;

use super::AggregateCacheMode;
use super::ArrayInput;
use super::VerifiedIntegerBounds;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::aggregate_fn::AggregateFnRef;
use crate::aggregate_fn::AggregateFnVTableExt;
use crate::aggregate_fn::NumericalAggregateOpts;
use crate::aggregate_fn::fns::is_sorted::IsSorted;
use crate::aggregate_fn::fns::is_sorted::IsSortedOptions;
use crate::aggregate_fn::fns::max::Max;
use crate::aggregate_fn::fns::min::Min;
use crate::aggregate_fn::fns::min_max::MinMax;
use crate::aggregate_fn::fns::min_max::MinMaxResult;
use crate::aggregate_fn::fns::min_max::make_minmax_dtype;
use crate::arrays::PrimitiveArray;
use crate::dtype::Nullability;
use crate::expr::stats::Precision;
use crate::scalar::Scalar;
use crate::validity::Validity;

impl ArrayInput {
    /// Construct nonnullable unsigned indices from a mask's selected positions.
    ///
    /// The mask producer establishes sortedness, uniqueness, and value bounds while constructing
    /// these values. No aggregate scan is needed. All-true masks produce `0..mask.len()`, and
    /// all-false masks produce an empty input. Integer conversion is checked.
    pub fn from_mask_indices(mask: &Mask) -> VortexResult<Self> {
        Self::from_mask_indices_with_cache_mode(mask, AggregateCacheMode::Input)
    }

    /// Construct mask indices and publish producer facts to the selected store.
    pub fn from_mask_indices_with_cache_mode(
        mask: &Mask,
        mode: AggregateCacheMode,
    ) -> VortexResult<Self> {
        let positions: Buffer<u64> = match mask.indices() {
            AllOr::All => (0..mask.len())
                .map(u64::try_from)
                .collect::<Result<_, _>>()?,
            AllOr::None => Buffer::empty(),
            AllOr::Some(indices) => indices
                .iter()
                .copied()
                .map(u64::try_from)
                .collect::<Result<_, _>>()?,
        };
        let bounds = positions
            .first()
            .zip(positions.last())
            .map(|(&min, &max)| MinMaxResult {
                min: Scalar::primitive(min, Nullability::NonNullable),
                max: Scalar::primitive(max, Nullability::NonNullable),
            });
        let input = Self::new(PrimitiveArray::new(positions, Validity::NonNullable).into_array())
            .with_cache_mode(mode);
        input.record_bounds(bounds.as_ref(), true)?;
        if mode != AggregateCacheMode::Disabled {
            input
                .inner
                .verified_bounds
                .set(VerifiedIntegerBounds::from_producer(bounds))
                .ok();
        }
        input.record_sorted(true)?;
        Ok(input)
    }

    /// Slice this input, retaining integer guarantees that hold for the subset.
    ///
    /// For integer inputs, exact extrema become bounds and positive sortedness keeps its original
    /// strictness. Negative sortedness and other aggregate results are omitted. Ordering places nulls
    /// first, and strict ordering excludes duplicate values and repeated nulls. A complete slice
    /// shares its owner's store. No values are read to propagate guarantees. Nonempty subsets
    /// drop the private cast proof because external transformation kernels do not establish native
    /// bounds.
    pub fn slice(&self, range: Range<usize>) -> VortexResult<Self> {
        let array = match self.cache_mode {
            AggregateCacheMode::Array => self.array().slice(range)?,
            AggregateCacheMode::Input | AggregateCacheMode::Disabled => {
                self.array().slice_without_aggregate_results(range)?
            }
        };
        if ArrayRef::ptr_eq(&array, self.array()) {
            return Ok(self.clone());
        }
        self.subset(array)
    }

    /// Apply a stable filter, retaining integer bounds and positive integer sortedness.
    ///
    /// The mask must match the input length. Relative order and null placement of retained rows
    /// are unchanged. An all-true mask shares the owner. Other selections get fresh stores. Only
    /// integer inputs propagate bounds and positive ordering, including strict ordering. Nonempty
    /// selections retain no private cast proof.
    pub fn filter(&self, mask: Mask) -> VortexResult<Self> {
        let array = self.array().filter(mask)?;
        if ArrayRef::ptr_eq(&array, self.array()) {
            return Ok(self.clone());
        }
        self.subset(array)
    }

    /// Validate canonical integer values once for checked casts that require value-range proof.
    ///
    /// This scans native values without registered aggregate kernels. Unsupported encodings and
    /// noninteger inputs return an error. The private proof includes null payloads so it remains
    /// valid across execution contexts. Logical extrema use the current validity mask. Enabled
    /// modes retain both. Disabled mode validates on each call and retains no facts.
    pub fn validate_integer_bounds(&self, ctx: &mut ExecutionCtx) -> VortexResult<()> {
        if self.cache_mode == AggregateCacheMode::Disabled {
            VerifiedIntegerBounds::validate(self.array(), ctx)?;
            return Ok(());
        }
        if self.inner.verified_bounds.get().is_none() {
            let (proof, logical_bounds) = VerifiedIntegerBounds::validate(self.array(), ctx)?;
            self.record_bounds(logical_bounds.as_ref(), true)?;
            self.inner.verified_bounds.set(proof).ok();
        }
        Ok(())
    }

    fn record_result(
        &self,
        aggregate: AggregateFnRef,
        result: Precision<Scalar>,
    ) -> VortexResult<()> {
        match self.cache_mode {
            AggregateCacheMode::Array => {
                self.array().aggregations().insert_result(aggregate, result)
            }
            AggregateCacheMode::Input => self.aggregations().insert_result(aggregate, result),
            AggregateCacheMode::Disabled => Ok(()),
        }
    }

    fn record_bounds(&self, bounds: Option<&MinMaxResult>, exact: bool) -> VortexResult<()> {
        for options in [
            NumericalAggregateOpts::skip_nans(),
            NumericalAggregateOpts::include_nans(),
        ] {
            let min_dtype = Min
                .bind(options)
                .return_dtype(self.array().dtype())
                .vortex_expect("integer extrema support the input dtype");
            let max_dtype = Max
                .bind(options)
                .return_dtype(self.array().dtype())
                .vortex_expect("integer extrema support the input dtype");
            let min = bounds.map_or_else(
                || Scalar::null(min_dtype),
                |b| b.min.clone().into_nullable(),
            );
            let max = bounds.map_or_else(
                || Scalar::null(max_dtype),
                |b| b.max.clone().into_nullable(),
            );
            let precision = |value| {
                if exact {
                    Precision::Exact(value)
                } else {
                    Precision::Inexact(value)
                }
            };
            self.record_result(Min.bind(options), precision(min))?;
            self.record_result(Max.bind(options), precision(max))?;
            if exact {
                let dtype = make_minmax_dtype(self.array().dtype());
                let result = bounds.map_or_else(
                    || Scalar::null(dtype.clone()),
                    |b| Scalar::struct_(dtype.clone(), vec![b.min.clone(), b.max.clone()]),
                );
                self.record_result(MinMax.bind(options), Precision::Exact(result))?;
            }
        }
        Ok(())
    }

    fn record_sorted(&self, strict: bool) -> VortexResult<()> {
        for strict in if strict {
            &[false, true][..]
        } else {
            &[false][..]
        } {
            self.record_result(
                IsSorted.bind(IsSortedOptions { strict: *strict }),
                Precision::Exact(Scalar::bool(true, Nullability::NonNullable)),
            )?;
        }
        Ok(())
    }

    fn subset(&self, array: ArrayRef) -> VortexResult<Self> {
        let output = Self::new(array).with_cache_mode(self.cache_mode);
        if self.cache_mode == AggregateCacheMode::Disabled || !self.array().dtype().is_int() {
            return Ok(output);
        }
        if output.array().is_empty() {
            output.record_bounds(None, true)?;
            output.record_sorted(true)?;
            output
                .inner
                .verified_bounds
                .set(VerifiedIntegerBounds::from_producer(None))
                .ok();
            return Ok(output);
        }
        for (aggregate, result) in self.snapshot_results().iter() {
            if aggregate.is::<Min>() || aggregate.is::<Max>() {
                if let Some(value) = result.as_ref().into_inner() {
                    output.record_result(
                        aggregate.clone(),
                        if value.is_null() {
                            Precision::Exact(value.clone())
                        } else {
                            Precision::Inexact(value.clone())
                        },
                    )?;
                }
            } else if aggregate.is::<IsSorted>()
                && result
                    .clone()
                    .as_exact()
                    .is_some_and(|v| v.as_bool().value() == Some(true))
            {
                output.record_result(aggregate.clone(), result.clone())?;
            } else if aggregate.is::<MinMax>()
                && let Precision::Exact(value) = result
                && let Some(bounds) = MinMaxResult::from_scalar(value.clone())?
            {
                let options = *aggregate.as_::<MinMax>();
                output.record_result(
                    Min.bind(options),
                    Precision::Inexact(bounds.min.into_nullable()),
                )?;
                output.record_result(
                    Max.bind(options),
                    Precision::Inexact(bounds.max.into_nullable()),
                )?;
            }
        }
        Ok(output)
    }
}
