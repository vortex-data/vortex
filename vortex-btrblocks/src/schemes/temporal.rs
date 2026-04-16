// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Temporal compression scheme using datetime-part decomposition.

use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::LEGACY_SESSION;
use vortex_array::ToCanonical;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::fns::is_constant::is_constant;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::TemporalArray;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_array::dtype::extension::Matcher;
use vortex_array::extension::datetime::AnyTemporal;
use vortex_array::extension::datetime::TemporalMetadata;
use vortex_compressor::estimate::CompressionEstimate;
use vortex_compressor::estimate::EstimateVerdict;
use vortex_datetime_parts::DateTimeParts;
use vortex_datetime_parts::TemporalParts;
use vortex_datetime_parts::split_temporal;
use vortex_error::VortexResult;

use crate::ArrayAndStats;
use crate::CascadingCompressor;
use crate::CompressorContext;
use crate::Scheme;
use crate::SchemeExt;

/// Compression scheme for temporal timestamp arrays via datetime-part decomposition.
///
/// Splits timestamps into days, seconds, and subseconds components, compresses each
/// independently, and wraps the result in a `DateTimePartsArray`.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct TemporalScheme;

impl Scheme for TemporalScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.ext.temporal"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        let Canonical::Extension(ext) = canonical else {
            return false;
        };

        let ext_dtype = ext.ext_dtype();

        matches!(
            AnyTemporal::try_match(ext_dtype),
            Some(TemporalMetadata::Timestamp(..))
        )
    }

    /// Children: days=0, seconds=1, subseconds=2.
    fn num_children(&self) -> usize {
        3
    }

    fn expected_compression_ratio(
        &self,
        _data: &mut ArrayAndStats,
        _ctx: CompressorContext,
    ) -> CompressionEstimate {
        // Temporal compression (splitting into parts) is almost always beneficial.
        CompressionEstimate::Verdict(EstimateVerdict::AlwaysUse)
    }

    fn compress(
        &self,
        compressor: &CascadingCompressor,
        data: &mut ArrayAndStats,
        ctx: CompressorContext,
    ) -> VortexResult<ArrayRef> {
        let array = data.array().clone();
        let ext_array = array.to_extension();
        let temporal_array = TemporalArray::try_from(ext_array.clone().into_array())?;

        // Check for constant array and return early if so.
        let is_constant = is_constant(
            &ext_array.clone().into_array(),
            &mut compressor.execution_ctx(),
        )?;

        if is_constant {
            return Ok(ConstantArray::new(
                ext_array.execute_scalar(0, &mut LEGACY_SESSION.create_execution_ctx())?,
                ext_array.len(),
            )
            .into_array());
        }

        let dtype = temporal_array.dtype().clone();
        let TemporalParts {
            days,
            seconds,
            subseconds,
        } = split_temporal(temporal_array)?;

        let days = compressor.compress_child(
            &days.to_primitive().narrow()?.into_array(),
            &ctx,
            self.id(),
            0,
        )?;
        let seconds = compressor.compress_child(
            &seconds.to_primitive().narrow()?.into_array(),
            &ctx,
            self.id(),
            1,
        )?;
        let subseconds = compressor.compress_child(
            &subseconds.to_primitive().narrow()?.into_array(),
            &ctx,
            self.id(),
            2,
        )?;

        Ok(DateTimeParts::try_new(dtype, days, seconds, subseconds)?.into_array())
    }
}
