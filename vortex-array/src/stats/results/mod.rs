// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Immutable finalized aggregate results.
//!
//! Results and bounds describe one input. They remain separate from partial accumulator states,
//! which contain the information needed to merge batches.

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
/// Missing results are unknown. Exact nulls are known results, including overflowing sums.
#[derive(Clone, Debug, Default)]
pub struct AggregateResults {
    // File summaries contain few entries. A linear lookup avoids a separate index and preserves
    // the caller's iteration order.
    entries: Arc<[(AggregateFnRef, Precision<Scalar>)]>,
}

impl AggregateResults {
    /// Construct results, checking each scalar against the aggregate's return type.
    ///
    /// Rejects duplicate keys, unsupported input types, and incompatible scalar types.
    /// Absent results are omitted. This validates types, not whether the values describe the input.
    pub fn try_new(
        input_dtype: &DType,
        entries: impl IntoIterator<Item = (AggregateFnRef, Precision<Scalar>)>,
    ) -> VortexResult<Self> {
        let entries = entries.into_iter().collect::<Vec<_>>();

        validate_entries(input_dtype, &entries)?;

        Ok(Self::from_validated(entries))
    }

    /// Check keys and result dtypes against the input at a private publication boundary.
    pub(crate) fn validate(&self, input_dtype: &DType) -> VortexResult<()> {
        validate_entries(input_dtype, &self.entries)
    }

    /// Construct results after validating unique keys and scalar types.
    ///
    /// Callers must validate against aggregate return types or historical field types when decoding
    /// older summaries. This constructor preserves those types without checking them again.
    pub(crate) fn from_validated(mut entries: Vec<(AggregateFnRef, Precision<Scalar>)>) -> Self {
        entries.retain(|(_, value)| !value.is_absent());

        Self {
            entries: entries.into(),
        }
    }

    /// Look up a finalized result. Options are part of the key.
    pub fn get_result(&self, aggregate: &AggregateFnRef) -> Precision<Scalar> {
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

fn validate_entries(
    input_dtype: &DType,
    entries: &[(AggregateFnRef, Precision<Scalar>)],
) -> VortexResult<()> {
    for (index, (aggregate, value)) in entries.iter().enumerate() {
        vortex_ensure!(
            !entries[..index].iter().any(|(other, _)| other == aggregate),
            "Duplicate aggregate result: {aggregate}"
        );
        let dtype = aggregate
            .return_dtype(input_dtype)
            .ok_or_else(|| vortex_err!("Aggregate {aggregate} does not support {input_dtype}"))?;
        if let Some(value) = value.as_ref().into_inner() {
            vortex_ensure!(
                value.dtype() == &dtype,
                "Aggregate {aggregate} requires result dtype {dtype}, got {}",
                value.dtype()
            );
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests;
