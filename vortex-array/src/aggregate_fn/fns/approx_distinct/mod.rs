// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Approximate count of distinct values (NDV) using a HyperLogLog sketch.
//!
//! Query engines use the number of distinct values to estimate join and group-by cardinalities.
//! [`ApproxDistinct`] keeps a mergeable sketch as its partial state, so per-chunk sketches combine
//! into a file-level sketch, and file-level sketches combine across files, without rescanning data.

mod accumulate;
mod sketch;
#[cfg(test)]
mod tests;

use std::fmt;
use std::fmt::Display;
use std::fmt::Formatter;

pub use sketch::HyperLogLog;
pub use sketch::MAX_PRECISION;
pub use sketch::MIN_PRECISION;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::ArrayRef;
use crate::Columnar;
use crate::ExecutionCtx;
use crate::aggregate_fn::Accumulator;
use crate::aggregate_fn::AggregateArgs;
use crate::aggregate_fn::AggregateFnId;
use crate::aggregate_fn::AggregateFnVTable;
use crate::aggregate_fn::DynAccumulator;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::scalar::Scalar;

/// The default precision: 4,096 registers, about 1.6% standard error, and at most 4 KiB per sketch.
pub const DEFAULT_PRECISION: u8 = 12;

const OPTIONS_BYTES_LEN: usize = 1;

/// Returns the approximate number of distinct non-null values in `array`.
///
/// Uses [`ApproxDistinctOptions::default`]. Fails if the dtype is not supported; see
/// [`ApproxDistinct`].
pub fn approx_distinct(array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<u64> {
    let mut accumulator = Accumulator::try_new(
        ApproxDistinct,
        ApproxDistinctOptions::default(),
        array.dtype().clone(),
    )?;
    accumulator.accumulate(array, ctx)?;
    let result = accumulator.finish()?;
    Ok(result
        .as_primitive()
        .typed_value::<u64>()
        .vortex_expect("approx_distinct result is a non-null u64"))
}

/// Options for [`ApproxDistinct`].
///
/// **Serialization**: a single byte holding the precision.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ApproxDistinctOptions {
    /// Number of hash bits that select a register, so the sketch has `2^precision` registers.
    precision: u8,
}

impl ApproxDistinctOptions {
    /// Options with `2^precision` registers.
    ///
    /// The standard error is about `1.04 / sqrt(2^precision)`. Fails unless `precision` is within
    /// [`MIN_PRECISION`]`..=`[`MAX_PRECISION`].
    pub fn try_new(precision: u8) -> VortexResult<Self> {
        sketch::validate_precision(precision)?;
        Ok(Self { precision })
    }

    /// Returns the number of hash bits that select a register.
    pub fn precision(&self) -> u8 {
        self.precision
    }

    fn serialize(&self) -> Vec<u8> {
        vec![self.precision]
    }

    fn deserialize(bytes: &[u8]) -> VortexResult<Self> {
        vortex_ensure_eq!(bytes.len(), OPTIONS_BYTES_LEN);
        Self::try_new(bytes[0])
    }
}

impl Default for ApproxDistinctOptions {
    fn default() -> Self {
        Self {
            precision: DEFAULT_PRECISION,
        }
    }
}

impl Display for ApproxDistinctOptions {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "precision={}", self.precision)
    }
}

/// Approximate number of distinct non-null values, estimated with a HyperLogLog sketch.
///
/// Returns a non-nullable `u64`. Null values are not counted, so an all-null or empty input
/// returns 0. Two values are distinct when their bytes differ, which matches Vortex equality:
/// floats compare by their bits, so `-0.0` and `0.0` are different values, and so are NaNs with
/// different payloads.
///
/// Supports null, boolean, primitive, UTF-8, and binary dtypes, and extension dtypes whose storage
/// is one of these. Decimals, structs, lists, and other nested dtypes are not supported.
///
/// Grouped aggregation is not supported yet: finalizing an array of partial states fails.
///
/// # Stability
///
/// The sketch format, including the hash function, is part of this aggregate's ID. A change to
/// either needs a new ID, since files written with the old format must still merge correctly.
#[derive(Clone, Debug)]
pub struct ApproxDistinct;

impl AggregateFnVTable for ApproxDistinct {
    type Options = ApproxDistinctOptions;
    type Partial = HyperLogLog;

    fn id(&self) -> AggregateFnId {
        static ID: CachedId = CachedId::new("vortex.approx_distinct.hll");
        *ID
    }

    fn serialize(&self, options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(options.serialize()))
    }

    fn deserialize(
        &self,
        metadata: &[u8],
        _session: &VortexSession,
    ) -> VortexResult<Self::Options> {
        ApproxDistinctOptions::deserialize(metadata)
    }

    fn return_dtype(&self, _options: &Self::Options, input_dtype: &DType) -> Option<DType> {
        accumulate::is_supported_dtype(input_dtype)
            .then_some(DType::Primitive(PType::U64, Nullability::NonNullable))
    }

    fn partial_dtype(&self, _options: &Self::Options, input_dtype: &DType) -> Option<DType> {
        accumulate::is_supported_dtype(input_dtype)
            .then_some(DType::Binary(Nullability::NonNullable))
    }

    fn empty_partial(&self, args: AggregateArgs<'_, Self::Options>) -> VortexResult<Self::Partial> {
        HyperLogLog::try_new(args.options.precision())
    }

    /// A null scalar is an empty sketch.
    fn partial_from_scalar(
        &self,
        args: AggregateArgs<'_, Self::Options>,
        scalar: &Scalar,
    ) -> VortexResult<Self::Partial> {
        if scalar.is_null() {
            return self.empty_partial(args);
        }
        let bytes = scalar
            .as_binary()
            .value()
            .ok_or_else(|| vortex_err!("non-null approx_distinct partial has no bytes"))?;
        let sketch = HyperLogLog::deserialize(bytes.as_slice())?;
        // Merging sketches of different precisions would silently give a wrong estimate.
        vortex_ensure_eq!(sketch.precision(), args.options.precision());
        Ok(sketch)
    }

    fn merge_partials(
        &self,
        _args: AggregateArgs<'_, Self::Options>,
        mut first: Self::Partial,
        second: Self::Partial,
    ) -> VortexResult<Self::Partial> {
        first.merge(&second)?;
        Ok(first)
    }

    fn to_scalar(
        &self,
        _args: AggregateArgs<'_, Self::Options>,
        partial: &Self::Partial,
    ) -> VortexResult<Scalar> {
        Ok(Scalar::binary(
            partial.serialize(),
            Nullability::NonNullable,
        ))
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
        match batch {
            Columnar::Constant(constant) => accumulate::accumulate_constant(constant, partial),
            Columnar::Canonical(canonical) => {
                accumulate::accumulate_canonical(canonical, partial, ctx)
            }
        }
    }

    /// Decoding an array of sketches needs an execution context, which this method doesn't get.
    fn finalize(
        &self,
        _args: AggregateArgs<'_, Self::Options>,
        _partials: ArrayRef,
    ) -> VortexResult<ArrayRef> {
        vortex_bail!("approx_distinct does not support grouped aggregation yet")
    }

    fn finalize_scalar(
        &self,
        _args: AggregateArgs<'_, Self::Options>,
        partial: &Self::Partial,
    ) -> VortexResult<Scalar> {
        Ok(Scalar::primitive(
            partial.estimate(),
            Nullability::NonNullable,
        ))
    }
}
