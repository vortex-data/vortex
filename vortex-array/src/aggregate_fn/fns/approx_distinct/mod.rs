// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Approximate distinct counts with a HyperLogLog sketch.
//!
//! See [`ApproxDistinct`].

mod accumulate;
mod sketch;

use std::fmt;
use std::fmt::Display;
use std::fmt::Formatter;

pub use sketch::HllSketch;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::ArrayRef;
use crate::Columnar;
use crate::ExecutionCtx;
use crate::aggregate_fn::AggregateArgs;
use crate::aggregate_fn::AggregateFnId;
use crate::aggregate_fn::AggregateFnVTable;
use crate::arrays::Dict;
use crate::arrays::dict::DictArrayExt;
use crate::arrays::dict::DictArraySlotsExt;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::scalar::Scalar;

/// The default precision: 4,096 registers, a relative standard error of about 1.6%, and at most
/// 4 KiB per sketch.
pub const DEFAULT_APPROX_DISTINCT_PRECISION: u8 = 12;

/// Options of [`ApproxDistinct`].
///
/// **Serialization**: a single byte holding the precision.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ApproxDistinctOptions {
    precision: u8,
}

impl ApproxDistinctOptions {
    /// Options for a sketch with `2^precision` registers.
    ///
    /// # Errors
    ///
    /// Returns an error if `precision` is outside `4..=16`.
    pub fn try_new(precision: u8) -> VortexResult<Self> {
        // Validates the precision.
        HllSketch::try_new(precision)?;
        Ok(Self { precision })
    }

    /// The number of register index bits.
    pub fn precision(&self) -> u8 {
        self.precision
    }
}

impl Default for ApproxDistinctOptions {
    fn default() -> Self {
        Self {
            precision: DEFAULT_APPROX_DISTINCT_PRECISION,
        }
    }
}

impl Display for ApproxDistinctOptions {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "precision={}", self.precision)
    }
}

/// Estimates the number of distinct non-null values with a HyperLogLog sketch.
///
/// The partial state is a serialized [`HllSketch`], so partials of different chunks, or of
/// different files, merge into the sketch of their union. The result is a non-nullable `u64`
/// estimate: `0` exactly when no non-null value was seen, otherwise an approximation with a
/// relative standard error of about `1.04 / sqrt(2^precision)`.
///
/// Values are hashed by their logical value: integers, dates and timestamps with a 64-bit mix of
/// the value, decimals by their unscaled value regardless of the physical integer width, floats
/// by their bit pattern with `-0.0` folded into `0.0` and every NaN into one, and strings and
/// binaries with XXH3-64 over their bytes. Booleans contribute at most two values. A
/// dictionary-encoded array whose values are all referenced only hashes its values.
///
/// Nested types (structs, lists, maps, variants and unions) are not supported.
///
/// **Partial serialization**: one tag byte, then either every register as one byte (tag `0`), or
/// `(u16 LE index, u8 rank)` triples for the non-zero registers in increasing index order (tag
/// `1`), whichever is smaller.
///
/// Only scalar results are supported: [`AggregateFnVTable::finalize`] over an array of partials,
/// as grouped aggregation uses, fails.
#[derive(Clone, Debug)]
pub struct ApproxDistinct;

impl ApproxDistinct {
    /// Returns the sketch stored in a partial scalar of [`ApproxDistinct`] with `options`.
    ///
    /// # Errors
    ///
    /// Returns an error if `scalar` is not a valid partial.
    pub fn sketch_from_partial(
        options: &ApproxDistinctOptions,
        scalar: &Scalar,
    ) -> VortexResult<HllSketch> {
        if scalar.is_null() {
            return HllSketch::try_new(options.precision);
        }
        let bytes = scalar
            .as_binary_opt()
            .ok_or_else(|| vortex_err!("approx_distinct partial must be binary"))?
            .value()
            .ok_or_else(|| vortex_err!("non-null approx_distinct partial has no bytes"))?;
        HllSketch::deserialize(options.precision, bytes.as_slice())
    }
}

impl AggregateFnVTable for ApproxDistinct {
    type Options = ApproxDistinctOptions;
    type Partial = HllSketch;

    fn id(&self) -> AggregateFnId {
        static ID: CachedId = CachedId::new("vortex.approx_distinct.hll");
        *ID
    }

    fn serialize(&self, options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(vec![options.precision]))
    }

    fn deserialize(
        &self,
        metadata: &[u8],
        _session: &VortexSession,
    ) -> VortexResult<Self::Options> {
        vortex_ensure_eq!(metadata.len(), 1);
        ApproxDistinctOptions::try_new(metadata[0])
    }

    fn return_dtype(&self, _options: &Self::Options, input_dtype: &DType) -> Option<DType> {
        is_supported_dtype(input_dtype)
            .then_some(DType::Primitive(PType::U64, Nullability::NonNullable))
    }

    fn partial_dtype(&self, _options: &Self::Options, input_dtype: &DType) -> Option<DType> {
        is_supported_dtype(input_dtype).then_some(DType::Binary(Nullability::NonNullable))
    }

    fn empty_partial(&self, args: AggregateArgs<'_, Self::Options>) -> VortexResult<Self::Partial> {
        HllSketch::try_new(args.options.precision)
    }

    fn partial_from_scalar(
        &self,
        args: AggregateArgs<'_, Self::Options>,
        scalar: &Scalar,
    ) -> VortexResult<Self::Partial> {
        Self::sketch_from_partial(args.options, scalar)
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

    /// Hashes only the values of a dictionary that references all of them.
    fn try_accumulate(
        &self,
        args: AggregateArgs<'_, Self::Options>,
        state: &mut Self::Partial,
        batch: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<bool> {
        let Some(dict) = batch.as_opt::<Dict>() else {
            return Ok(false);
        };
        if !dict.has_all_values_referenced() {
            return Ok(false);
        }
        // A null code is a null row, which the count leaves out; a null value is left out too.
        let values = dict.values().clone().execute::<Columnar>(ctx)?;
        self.accumulate(args, state, &values, ctx)?;
        Ok(true)
    }

    fn accumulate(
        &self,
        _args: AggregateArgs<'_, Self::Options>,
        state: &mut Self::Partial,
        batch: &Columnar,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        accumulate::accumulate_columnar(batch, state, ctx)
    }

    fn finalize(
        &self,
        _args: AggregateArgs<'_, Self::Options>,
        _partials: ArrayRef,
    ) -> VortexResult<ArrayRef> {
        vortex_bail!("approx_distinct does not support finalizing an array of partials")
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

/// Whether [`ApproxDistinct`] can hash values of `dtype`.
fn is_supported_dtype(dtype: &DType) -> bool {
    match dtype {
        DType::Extension(ext) => is_supported_dtype(ext.storage_dtype()),
        DType::Bool(_)
        | DType::Primitive(..)
        | DType::Decimal(..)
        | DType::Utf8(_)
        | DType::Binary(_) => true,
        _ => false,
    }
}

#[cfg(test)]
mod tests;
