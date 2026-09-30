// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Immutable finalized aggregate results used by file summaries.
//!
//! A summary describes one input and preserves missing values, exact results, and bounds. These
//! values are exposed separately from accumulator states so consumers cannot confuse their roles.

use std::sync::Arc;

use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;

use crate::aggregate_fn::AggregateFnRef;
use crate::dtype::DType;
use crate::expr::stats::Precision;
use crate::scalar::Scalar;

/// Finalized results keyed by aggregate function and options.
///
/// These values describe one input. They are not mergeable accumulator states. A missing entry means
/// the result is unknown. An exact null is a known result, such as an overflowing sum.
#[derive(Clone, Debug, Default)]
pub struct AggregateResults {
    // File summaries contain few entries. A linear lookup avoids a separate index and preserves
    // the caller's iteration order.
    entries: Arc<[(AggregateFnRef, Precision<Scalar>)]>,
}

impl AggregateResults {
    /// Construct a summary, checking each result against the aggregate's return type.
    ///
    /// Rejects duplicate functions, unsupported input types, and incompatible scalar types.
    /// Absent results are omitted from the stored entries.
    pub fn try_new(
        input_dtype: &DType,
        entries: impl IntoIterator<Item = (AggregateFnRef, Precision<Scalar>)>,
    ) -> VortexResult<Self> {
        let entries = entries.into_iter().collect::<Vec<_>>();

        for (index, (aggregate, value)) in entries.iter().enumerate() {
            vortex_ensure!(
                !entries[..index].iter().any(|(other, _)| other == aggregate),
                "Duplicate aggregate result: {aggregate}"
            );
            let dtype = aggregate.return_dtype(input_dtype).ok_or_else(|| {
                vortex_err!("Aggregate {aggregate} does not support {input_dtype}")
            })?;
            if let Some(value) = value.as_ref().into_inner() {
                vortex_ensure!(
                    value.dtype() == &dtype,
                    "Aggregate {aggregate} requires result dtype {dtype}, got {}",
                    value.dtype()
                );
            }
        }

        Ok(Self::from_validated(entries))
    }

    /// Construct results after validating unique keys and scalar types.
    ///
    /// The compatibility decoder uses historical field types because some stored summaries predate
    /// the current kernels. Callers must validate against either those types or aggregate return
    /// types. Violating this contract can expose incorrect metadata to pruning consumers.
    pub(crate) fn from_validated(mut entries: Vec<(AggregateFnRef, Precision<Scalar>)>) -> Self {
        entries.retain(|(_, value)| !value.is_absent());

        Self {
            entries: entries.into(),
        }
    }

    /// Look up a finalized result. Options are part of the key.
    pub fn get(&self, aggregate: &AggregateFnRef) -> Precision<Scalar> {
        self.entries
            .iter()
            .find(|(key, _)| key == aggregate)
            .map(|(_, value)| value.clone())
            .unwrap_or_default()
    }

    /// Iterate over finalized results in insertion order.
    pub fn iter(&self) -> impl Iterator<Item = (&AggregateFnRef, &Precision<Scalar>)> {
        self.entries.iter().map(|(key, value)| (key, value))
    }
}

#[cfg(test)]
mod tests;
