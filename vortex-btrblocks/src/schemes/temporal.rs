// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Temporal compression scheme using datetime-part decomposition.

use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VTable;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::TemporalArray;
use vortex_array::arrays::extension::ExtensionArrayExt;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_array::dtype::extension::Matcher;
use vortex_array::extension::datetime::AnyTemporal;
use vortex_array::extension::datetime::TemporalMetadata;
use vortex_compressor::scheme::CompressionEstimate;
use vortex_compressor::scheme::EstimateVerdict;
use vortex_datetime_parts::DateTimeParts;
use vortex_datetime_parts::TemporalParts;
use vortex_datetime_parts::split_temporal;
use vortex_error::VortexResult;

use crate::ArrayAndStats;
use crate::CascadingCompressor;
use crate::Child;
use crate::CompressorContext;
use crate::Encoded;
use crate::Scheme;

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

    fn produced_encodings(&self) -> Vec<ArrayId> {
        vec![DateTimeParts.id()]
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
        let TemporalParts {
            days,
            seconds,
            subseconds,
        } = split_temporal(temporal_array, exec_ctx)?;

        let children = vec![
            Child::new(self, 0, narrowed(days, exec_ctx)?),
            Child::new(self, 1, narrowed(seconds, exec_ctx)?),
            Child::new(self, 2, narrowed(subseconds, exec_ctx)?),
        ];
        Ok(Encoded::new(children, Vec::new(), move |children, _| {
            let [days, seconds, subseconds] = children.compressed()?;
            Ok(DateTimeParts::try_new(dtype, days, seconds, subseconds)?.into_array())
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
