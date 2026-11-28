// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Temporal compression scheme using datetime-part decomposition.

use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::TemporalArray;
use vortex_array::arrays::extension::ExtensionArrayExt;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_array::dtype::extension::Matcher;
use vortex_array::extension::datetime::AnyTemporal;
use vortex_array::extension::datetime::TemporalMetadata;
use vortex_array::extension::datetime::TimeUnit;
use vortex_compressor::scheme::CompressionEstimate;
use vortex_compressor::scheme::EstimateVerdict;
use vortex_datetime_parts::DateTimeParts;
use vortex_datetime_parts::ScaledTemporalParts;
use vortex_datetime_parts::TemporalParts;
use vortex_datetime_parts::datetime_parts_v1_id;
use vortex_datetime_parts::datetime_parts_v2_id;
use vortex_datetime_parts::split_temporal;
use vortex_datetime_parts::split_temporal_v2;
use vortex_error::VortexResult;

use crate::ArrayAndStats;
use crate::CascadingCompressor;
use crate::Child;
use crate::CompressorContext;
use crate::Encoded;
use crate::Scheme;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum TemporalSchemeMode {
    V1,
    V2,
}

/// The temporal scheme that only produces v1-compatible `DateTimeParts` arrays
/// (`Timestamp` only, unscaled subseconds).
pub(crate) static TEMPORAL_V1: TemporalScheme = TemporalScheme::v1();

/// The temporal scheme that may produce v2 `DateTimeParts` arrays (`Timestamp` and sub-second
/// `Time`, with scaled subseconds when lossless).
pub(crate) static TEMPORAL_V2: TemporalScheme = TemporalScheme::v2();

/// Compression scheme for temporal arrays via datetime-part decomposition.
///
/// Splits timestamps (and in v2 mode, sub-second `Time` arrays) into days, seconds, and
/// subseconds components, compresses each independently, and wraps the result in a
/// `DateTimePartsArray`.
///
/// The v1 mode matches only `Timestamp` and keeps unscaled subseconds (`subseconds_divisor = 1`),
/// serializing as `vortex.datetimeparts`. The v2 mode also matches `Time` (`ms`, `µs`, `ns`) and
/// scales coarse-grained `µs` / `ns` subseconds by `1_000` or `1_000_000` when lossless,
/// serializing those arrays as `vortex.datetimeparts.v2` while still writing unscaled `Timestamp`
/// arrays as v1.
///
/// The default uses v1. [`refine`](Scheme::refine) picks v2 when `vortex.datetimeparts.v2` is
/// allowed and v1 otherwise.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct TemporalScheme {
    mode: TemporalSchemeMode,
}

impl TemporalScheme {
    /// Creates a temporal scheme configured for v1 (`Timestamp` only, unscaled subseconds).
    pub const fn v1() -> Self {
        Self {
            mode: TemporalSchemeMode::V1,
        }
    }

    /// Creates a temporal scheme configured for v2 (`Timestamp` and sub-second `Time`, with
    /// coarse-grained subseconds scaling).
    pub const fn v2() -> Self {
        Self {
            mode: TemporalSchemeMode::V2,
        }
    }
}

impl Default for TemporalScheme {
    fn default() -> Self {
        Self::v1()
    }
}

impl Scheme for TemporalScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.ext.temporal"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        let Canonical::Extension(ext) = canonical else {
            return false;
        };

        let ext_dtype = ext.ext_dtype();

        match self.mode {
            TemporalSchemeMode::V1 => matches!(
                AnyTemporal::try_match(ext_dtype),
                Some(TemporalMetadata::Timestamp(..))
            ),
            TemporalSchemeMode::V2 => matches!(
                AnyTemporal::try_match(ext_dtype),
                Some(TemporalMetadata::Timestamp(..))
                    | Some(TemporalMetadata::Time(
                        TimeUnit::Milliseconds | TimeUnit::Microseconds | TimeUnit::Nanoseconds
                    ))
            ),
        }
    }

    fn produced_encodings(&self) -> Vec<ArrayId> {
        match self.mode {
            TemporalSchemeMode::V1 => vec![datetime_parts_v1_id()],
            TemporalSchemeMode::V2 => vec![datetime_parts_v1_id(), datetime_parts_v2_id()],
        }
    }

    fn refine(&self, allowed: &dyn Fn(&ArrayId) -> bool) -> &dyn Scheme {
        if allowed(&datetime_parts_v2_id()) {
            &TEMPORAL_V2
        } else {
            &TEMPORAL_V1
        }
    }

    /// Children: days=0, seconds=1, subseconds=2.
    fn num_children(&self) -> usize {
        3
    }

    fn expected_compression_ratio(
        &self,
        _data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        // Temporal compression (splitting into parts) is almost always beneficial.
        CompressionEstimate::Verdict(EstimateVerdict::AlwaysUse)
    }

    /// Children: days=0, which carry the validity, seconds=1, subseconds=2, each narrowed.
    fn encode(
        &self,
        data: &ArrayAndStats,
        _compress_ctx: &CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<Encoded> {
        let array = data.array().clone();
        let ext_array = array.execute::<ExtensionArray>(exec_ctx)?;
        let temporal_array = TemporalArray::try_from(ext_array.into_array())?;

        let dtype = temporal_array.dtype().clone();
        let (days, seconds, subseconds, subseconds_divisor) = match self.mode {
            TemporalSchemeMode::V1 => {
                let TemporalParts {
                    days,
                    seconds,
                    subseconds,
                } = split_temporal(temporal_array, exec_ctx)?;
                (days, seconds, subseconds, 1)
            }
            TemporalSchemeMode::V2 => {
                let ScaledTemporalParts {
                    days,
                    seconds,
                    subseconds,
                    subseconds_divisor,
                } = split_temporal_v2(temporal_array, exec_ctx)?;
                (days, seconds, subseconds, subseconds_divisor)
            }
        };

        let children = vec![
            Child::new(self, 0, narrowed(days, exec_ctx)?),
            Child::new(self, 1, narrowed(seconds, exec_ctx)?),
            Child::new(self, 2, narrowed(subseconds, exec_ctx)?),
        ];
        Ok(Encoded::new(children, Vec::new(), move |children, _| {
            let [days, seconds, subseconds] = children.compressed()?;
            Ok(DateTimeParts::try_new_with_divisor(
                dtype,
                days,
                seconds,
                subseconds,
                subseconds_divisor,
            )?
            .into_array())
        }))
    }

    #[expect(clippy::unimplemented, reason = "replaced by `encode`")]
    fn compress(
        &self,
        _compressor: &CascadingCompressor,
        _data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        unimplemented!("{} compresses through Scheme::encode", self.scheme_name())
    }
}

/// Narrows a datetime part to the smallest integer type that holds its values.
fn narrowed(part: ArrayRef, exec_ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
    Ok(part
        .execute::<PrimitiveArray>(exec_ctx)?
        .narrow(exec_ctx)?
        .into_array())
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::ArrayId;
    use vortex_datetime_parts::datetime_parts_v1_id;
    use vortex_datetime_parts::datetime_parts_v2_id;

    use super::TEMPORAL_V1;
    use super::TEMPORAL_V2;
    use crate::Scheme;
    use crate::SchemeExt;

    #[rstest]
    #[case::neither(false, false, false)]
    #[case::v1(true, false, false)]
    #[case::v2_only(false, true, true)]
    #[case::both(true, true, true)]
    fn refine_picks_v2_when_v2_id_is_allowed(
        #[case] allow_v1: bool,
        #[case] allow_v2: bool,
        #[case] expect_v2: bool,
        #[values(&TEMPORAL_V1, &TEMPORAL_V2)] scheme: &'static dyn Scheme,
    ) {
        let allowed = |id: &ArrayId| {
            (allow_v1 && *id == datetime_parts_v1_id())
                || (allow_v2 && *id == datetime_parts_v2_id())
        };
        let refined = scheme.refine(&allowed);
        assert_eq!(refined.id(), scheme.id());
        let expected = if expect_v2 {
            vec![datetime_parts_v1_id(), datetime_parts_v2_id()]
        } else {
            vec![datetime_parts_v1_id()]
        };
        assert_eq!(refined.produced_encodings(), expected);
    }
}
