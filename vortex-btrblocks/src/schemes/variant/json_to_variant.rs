// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::extension::ExtensionArrayExt;
use vortex_array::arrow::ArrowSessionExt;
use vortex_compressor::ctx::CompressorContext;
use vortex_compressor::estimate::CompressionEstimate;
use vortex_compressor::estimate::DeferredEstimate;
use vortex_compressor::scheme::Scheme;
use vortex_compressor::scheme::SchemeExt;
use vortex_compressor::stats::ArrayAndStats;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_json::Json;
use vortex_parquet_variant::ParquetVariant;
use vortex_parquet_variant::ParquetVariantArrayExt;
use vortex_parquet_variant::VariantToJson;

use crate::CascadingCompressor;

/// Compression scheme that converts JSON string extension arrays to Parquet Variant arrays.
///
/// When decompressed, the resulting JSON array might not be byte-to-byte identical, as this
/// compression doesn't maintain whitespaces.
#[derive(Debug)]
pub struct JsonToVariantScheme;

/// Child indices for recursively compressed Parquet Variant binary children.
mod parquet_variant_children {
    /// The Parquet Variant metadata child.
    pub const METADATA: usize = 0;
    /// The raw Parquet Variant value child.
    pub const VALUE: usize = 1;
}

impl Scheme for JsonToVariantScheme {
    fn scheme_name(&self) -> &'static str {
        "json_to_variant"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        let Canonical::Extension(ext_array) = canonical else {
            return false;
        };

        ext_array.ext_dtype().is::<Json>()
    }

    fn num_children(&self) -> usize {
        2
    }

    fn expected_compression_ratio(
        &self,
        _data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        CompressionEstimate::Deferred(DeferredEstimate::Sample)
    }

    fn compress(
        &self,
        compressor: &CascadingCompressor,
        data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let array = data.array().clone().execute::<ExtensionArray>(exec_ctx)?;
        let output_nullability = array.dtype().nullability();
        let storage = array.storage_array().clone();

        if !storage.dtype().is_utf8() {
            vortex_bail!("storage must be utf8");
        }

        let arrow_array = {
            let session = exec_ctx.session().clone();
            let arrow = session.arrow();
            arrow.execute_arrow(storage, None, exec_ctx)?
        };

        let array = parquet_variant_compute::json_to_variant(&arrow_array)?;
        let parquet_variant =
            ParquetVariant::from_arrow_variant(&array)?.downcast::<ParquetVariant>();

        let compressed_metadata = compressor.compress_child(
            parquet_variant.metadata_array(),
            &compress_ctx,
            self.id(),
            parquet_variant_children::METADATA,
            exec_ctx,
        )?;
        let compressed_value = parquet_variant
            .value_array()
            .map(|value| {
                compressor.compress_child(
                    value,
                    &compress_ctx,
                    self.id(),
                    parquet_variant_children::VALUE,
                    exec_ctx,
                )
            })
            .transpose()?;

        let variant_validity = parquet_variant
            .validity()?
            .union_nullability(output_nullability);
        let variant = ParquetVariant::try_new(
            variant_validity,
            compressed_metadata,
            compressed_value,
            parquet_variant.typed_value_array().cloned(),
        )?
        .into_array();

        Ok(VariantToJson::try_new(variant)?.into_array())
    }
}
