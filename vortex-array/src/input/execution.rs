// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_error::VortexError;
use vortex_error::VortexResult;

use super::AggregateCacheMode;
use super::ArrayInput;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::aggregate_fn::AggregateFnRef;
use crate::dtype::DType;
use crate::expr::stats::Precision;
use crate::scalar::Scalar;

#[derive(Debug)]
pub(crate) struct AggregateInputFrame {
    input: ArrayInput,
    parent: Option<Arc<AggregateInputFrame>>,
}

impl ExecutionCtx {
    /// Create a child context that retains this input and resolves its aggregate lookups by identity.
    ///
    /// The parent context is unchanged, including when child execution errors or panics. Nested
    /// scopes retain previous owners. Each bound input keeps its own cache mode. The innermost
    /// mode applies to unbound arrays, and the nearest matching handle wins. Lazy arrays carry no
    /// scope: execute them with the returned context, or use the input's eager execution methods
    /// before dropping it.
    pub fn with_aggregate_input(&self, input: &ArrayInput) -> Self {
        let mut child = self.clone();
        child.aggregate_cache_mode = input.cache_mode();
        child.aggregate_inputs = Some(Arc::new(AggregateInputFrame {
            input: input.clone(),
            parent: self.aggregate_inputs.clone(),
        }));
        child
    }

    /// Return the aggregate ownership policy for this execution scope.
    pub fn aggregate_cache_mode(&self) -> AggregateCacheMode {
        self.aggregate_cache_mode
    }

    /// Look up a result without computing a miss or reading input values.
    ///
    /// Input-owned results require the exact array handle retained by a scope. A decoded or
    /// otherwise rewritten handle conservatively misses, even when it represents equal values.
    pub fn aggregate_result(
        &self,
        array: &ArrayRef,
        aggregate: &AggregateFnRef,
    ) -> Precision<Scalar> {
        match self.mode_for(array) {
            AggregateCacheMode::Array => array.aggregations().get_result(aggregate),
            AggregateCacheMode::Input => self
                .aggregate_input(array)
                .map_or(Precision::Absent, |input| {
                    input.aggregations().get_result(aggregate)
                }),
            AggregateCacheMode::Disabled => Precision::Absent,
        }
    }

    /// Convert a retained result without computing a miss.
    pub fn aggregate_result_as<T>(
        &self,
        array: &ArrayRef,
        aggregate: &AggregateFnRef,
    ) -> VortexResult<Precision<T>>
    where
        T: for<'a> TryFrom<&'a Scalar, Error = VortexError>,
    {
        self.aggregate_result(array, aggregate)
            .map(|value| T::try_from(&value))
            .transpose()
    }

    pub(crate) fn insert_aggregate_result(
        &self,
        array: &ArrayRef,
        aggregate: AggregateFnRef,
        result: Precision<Scalar>,
    ) -> VortexResult<()> {
        match self.mode_for(array) {
            AggregateCacheMode::Array => array.aggregations().insert_result(aggregate, result),
            AggregateCacheMode::Input => match self.aggregate_input(array) {
                Some(input) => input.aggregations().insert_result(aggregate, result),
                None => Ok(()),
            },
            AggregateCacheMode::Disabled => Ok(()),
        }
    }

    /// Compute a finalized result using this scope's selected result owner.
    pub fn compute_aggregate_result(
        &mut self,
        array: &ArrayRef,
        aggregate: &AggregateFnRef,
    ) -> VortexResult<Scalar> {
        match self.mode_for(array) {
            AggregateCacheMode::Array => {
                if let Precision::Exact(result) = array.aggregations().get_result(aggregate) {
                    return Ok(result);
                }
                if let Some(input) = self.aggregate_input(array).cloned()
                    && let Some(result) =
                        input.aggregations().finalize_retained_result(aggregate)?
                {
                    array
                        .aggregations()
                        .insert_result(aggregate.clone(), Precision::Exact(result.clone()))?;
                    return Ok(result);
                }
                array.aggregations().compute_result(aggregate, self)
            }
            AggregateCacheMode::Input => {
                if let Some(input) = self.aggregate_input(array).cloned() {
                    return input.aggregations().compute_result(aggregate, self);
                }
                compute_uncached(array, aggregate, self)
            }
            AggregateCacheMode::Disabled => compute_uncached(array, aggregate, self),
        }
    }

    pub(crate) fn verified_integer_bounds_fit(
        &self,
        array: &ArrayRef,
        dtype: &DType,
    ) -> Option<bool> {
        if self.mode_for(array) == AggregateCacheMode::Disabled {
            return None;
        }
        self.aggregate_input(array)?
            .verified_bounds()
            .map(|proof| proof.fits(dtype))
    }

    fn mode_for(&self, array: &ArrayRef) -> AggregateCacheMode {
        self.aggregate_input(array)
            .map_or(self.aggregate_cache_mode, ArrayInput::cache_mode)
    }

    fn aggregate_input(&self, array: &ArrayRef) -> Option<&ArrayInput> {
        let mut frame = self.aggregate_inputs.as_deref();
        while let Some(current) = frame {
            if ArrayRef::ptr_eq(current.input.array(), array) {
                return Some(&current.input);
            }
            frame = current.parent.as_deref();
        }
        None
    }
}

fn compute_uncached(
    array: &ArrayRef,
    aggregate: &AggregateFnRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Scalar> {
    let mut accumulator = aggregate.accumulator(array.dtype())?;
    accumulator.accumulate_uncached(array, ctx)?;
    accumulator.finish()
}
