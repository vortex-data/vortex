// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compression scheme for JSON data into binary variant representation

use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::extension::ExtensionArrayExt;
use vortex_array::arrow::ArrowSessionExt;
use vortex_array::dtype::extension::ExtDType;
use vortex_array::dtype::extension::ExtId;
use vortex_array::dtype::extension::ExtVTable;
use vortex_array::extension::EmptyMetadata;
use vortex_array::scalar::ScalarValue;
use vortex_compressor::ctx::CompressorContext;
use vortex_compressor::estimate::CompressionEstimate;
use vortex_compressor::estimate::DeferredEstimate;
use vortex_compressor::scheme::Scheme;
use vortex_compressor::scheme::SchemeExt;
use vortex_compressor::stats::ArrayAndStats;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_parquet_variant::ParquetVariant;
use vortex_parquet_variant::ParquetVariantArrayExt;

use crate::CascadingCompressor;

/// Compression scheme that converts JSON string extension arrays to Parquet Variant arrays.
#[derive(Debug)]
pub struct JsonToVariantScheme;

/// Child indices for recursively compressed Parquet Variant binary children.
mod parquet_variant_children {
    /// The Parquet Variant metadata child.
    pub const METADATA: usize = 0;
    /// The raw Parquet Variant value child.
    pub const VALUE: usize = 1;
}

/// JSON logical type backed by UTF-8 string storage.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Json;

impl ExtVTable for Json {
    type Metadata = EmptyMetadata;
    type NativeValue<'a> = &'a str;

    fn id(&self) -> ExtId {
        ExtId::new("vortex.json")
    }

    fn serialize_metadata(&self, _metadata: &Self::Metadata) -> VortexResult<Vec<u8>> {
        Ok(Vec::new())
    }

    fn deserialize_metadata(&self, metadata: &[u8]) -> VortexResult<Self::Metadata> {
        vortex_ensure!(metadata.is_empty(), "JSON metadata must be empty");
        Ok(EmptyMetadata)
    }

    fn validate_dtype(ext_dtype: &ExtDType<Self>) -> VortexResult<()> {
        vortex_ensure!(
            ext_dtype.storage_dtype().is_utf8(),
            "JSON storage dtype must be utf8, got {}",
            ext_dtype.storage_dtype()
        );
        Ok(())
    }

    fn unpack_native<'a>(
        _ext_dtype: &'a ExtDType<Self>,
        storage_value: &'a ScalarValue,
    ) -> VortexResult<Self::NativeValue<'a>> {
        let ScalarValue::Utf8(value) = storage_value else {
            vortex_bail!("JSON storage scalar must be utf8, got {storage_value}");
        };
        Ok(value.as_str())
    }
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

        ParquetVariant::try_new(
            parquet_variant.validity()?,
            compressed_metadata,
            compressed_value,
            parquet_variant.typed_value_array().cloned(),
        )
        .map(IntoArray::into_array)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rand::RngExt;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::Extension;
    use vortex_array::arrays::ExtensionArray;
    use vortex_array::arrays::VarBinView;
    use vortex_array::arrays::VarBinViewArray;
    use vortex_array::arrays::extension::ExtensionArrayExt;
    use vortex_array::session::ArraySession;
    use vortex_compressor::builtins::BinaryDictScheme;
    use vortex_compressor::builtins::IntConstantScheme;
    use vortex_compressor::builtins::StringConstantScheme;
    use vortex_compressor::builtins::StringDictScheme;
    use vortex_session::VortexSession;
    use vortex_zstd::Zstd;

    use super::*;
    use crate::schemes::integer::BitPackingScheme;
    use crate::schemes::integer::FoRScheme;
    use crate::schemes::integer::RunEndScheme;
    use crate::schemes::integer::SequenceScheme;
    use crate::schemes::integer::SparseScheme;
    use crate::schemes::integer::ZigZagScheme;
    use crate::schemes::string::FSSTScheme;
    use crate::schemes::string::ZstdScheme;

    static SESSION: LazyLock<VortexSession> =
        LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

    fn json_data() -> Vec<String> {
        let mut rng = StdRng::seed_from_u64(0);
        const ACCOUNT_KEYS: &[&str] = &["account_id", "customer_id", "tenant_id", "buyer_id"];
        const REGION_KEYS: &[&str] = &["region", "market", "country"];
        const REGIONS: &[&str] = &["us-east", "us-west", "eu", "apac", "latam"];
        const STATUS_KEYS: &[&str] = &["status", "payment_state", "lifecycle_state"];
        const STATUSES: &[&str] = &["draft", "open", "paid", "void", "past_due"];
        const AMOUNT_KEYS: &[&str] = &["discount", "tax", "shipping", "credit"];
        const FLAG_KEYS: &[&str] = &["autopay", "fraud_review", "priority", "disputed"];
        const TAGS: &[&str] = &["renewal", "manual", "usage", "trial", "enterprise"];

        (0..1024)
            .map(|_| {
                let mut fields = vec![
                    format!(
                        r#""{}":"acct_{:04x}""#,
                        ACCOUNT_KEYS[rng.random_range(0..ACCOUNT_KEYS.len())],
                        rng.random::<u32>(),
                    ),
                    format!(
                        r#""invoice_total":{}.{:02}"#,
                        rng.random_range(10_u32..100_000),
                        rng.random_range(0_u32..100),
                    ),
                    format!(r#""line_items":{}"#, rng.random_range(1_u32..250)),
                ];

                if rng.random_bool(0.85) {
                    fields.push(format!(
                        r#""{}":"{}""#,
                        STATUS_KEYS[rng.random_range(0..STATUS_KEYS.len())],
                        STATUSES[rng.random_range(0..STATUSES.len())],
                    ));
                }
                if rng.random_bool(0.75) {
                    fields.push(format!(
                        r#""{}":"{}""#,
                        REGION_KEYS[rng.random_range(0..REGION_KEYS.len())],
                        REGIONS[rng.random_range(0..REGIONS.len())],
                    ));
                }
                if rng.random_bool(0.55) {
                    fields.push(format!(
                        r#""{}":{}.{:02}"#,
                        AMOUNT_KEYS[rng.random_range(0..AMOUNT_KEYS.len())],
                        rng.random_range(0_u32..2_500),
                        rng.random_range(0_u32..100),
                    ));
                }
                if rng.random_bool(0.40) {
                    fields.push(format!(
                        r#""{}":{}"#,
                        FLAG_KEYS[rng.random_range(0..FLAG_KEYS.len())],
                        rng.random_bool(0.5),
                    ));
                }
                if rng.random_bool(0.30) {
                    fields.push(format!(
                        r#""tags":["{}","{}"]"#,
                        TAGS[rng.random_range(0..TAGS.len())],
                        TAGS[rng.random_range(0..TAGS.len())],
                    ));
                }

                format!("{{{}}}", fields.join(","))
            })
            .collect()
    }

    fn json_array(values: &[String]) -> VortexResult<ArrayRef> {
        let storage =
            VarBinViewArray::from_iter_str(values.iter().map(String::as_str)).into_array();
        Ok(ExtensionArray::try_new_from_vtable(Json, EmptyMetadata, storage)?.into_array())
    }

    fn parquet_variant_child_compressor() -> CascadingCompressor {
        CascadingCompressor::new(vec![
            &JsonToVariantScheme,
            &BinaryDictScheme,
            &IntConstantScheme,
            &FoRScheme,
            &SparseScheme,
            &BitPackingScheme,
            &RunEndScheme,
            &SequenceScheme,
            &ZigZagScheme,
        ])
    }

    fn print_comparison_output(
        array: &ArrayRef,
        string_compressed: &ArrayRef,
        compressed: &ArrayRef,
    ) {
        let compressed_ratio = array.nbytes() as f64 / compressed.nbytes() as f64;
        let compressed_array_ratio = string_compressed.nbytes() as f64 / compressed.nbytes() as f64;
        println!(
            "Compression sizes: input={} bytes, compressed string={} bytes, compressed output={} bytes",
            array.nbytes(),
            string_compressed.nbytes(),
            compressed.nbytes(),
        );
        println!("Compressed output ratio: {compressed_ratio:.2}x\n");
        println!("Compressed string / compressed output ratio: {compressed_array_ratio:.2}x\n");
        println!("JSON input encoding tree:\n{}", array.tree_display());
        println!(
            "String-compressed encoding tree:\n{}",
            string_compressed.tree_display()
        );
        println!(
            "Compressed output encoding tree:\n{}",
            compressed.tree_display()
        );
    }

    #[test]
    fn parquet_variant_compresses_repeated_json_keys() -> VortexResult<()> {
        let array = json_array(&json_data())?;

        let string_compressor =
            CascadingCompressor::new(vec![&StringDictScheme, &StringConstantScheme]);
        let mut exec_ctx = SESSION.create_execution_ctx();
        let string_compressed = string_compressor.compress(&array, &mut exec_ctx)?;

        let variant_compressor = parquet_variant_child_compressor();
        let mut exec_ctx = SESSION.create_execution_ctx();
        let variant_data = ArrayAndStats::new(array.clone(), Default::default());
        let variant_compressed = JsonToVariantScheme.compress(
            &variant_compressor,
            &variant_data,
            CompressorContext::new(),
            &mut exec_ctx,
        )?;

        assert!(
            variant_compressed.is::<ParquetVariant>(),
            "expected ParquetVariant output, got encoding {} with dtype {} and {} bytes",
            variant_compressed.encoding_id(),
            variant_compressed.dtype(),
            variant_compressed.nbytes()
        );
        assert!(
            variant_compressed.nbytes() < string_compressed.nbytes(),
            "Parquet Variant conversion should compress repeated JSON keys: \
             variant={} bytes, input={} bytes",
            variant_compressed.nbytes(),
            string_compressed.nbytes(),
        );

        print_comparison_output(&array, &string_compressed, &variant_compressed);

        Ok(())
    }

    #[test]
    fn recursively_compresses_parquet_variant_binary_children() -> VortexResult<()> {
        let array: ArrayRef = json_array(&json_data())?;

        let variant_compressor = parquet_variant_child_compressor();
        let mut exec_ctx = SESSION.create_execution_ctx();
        let variant_data = ArrayAndStats::new(array.clone(), Default::default());
        let compressed = JsonToVariantScheme.compress(
            &variant_compressor,
            &variant_data,
            CompressorContext::new(),
            &mut exec_ctx,
        )?;
        let parquet_variant = compressed.clone().downcast::<ParquetVariant>();

        assert!(
            !parquet_variant.metadata_array().is::<VarBinView>(),
            "expected Parquet Variant metadata child to be compressed, got {}",
            parquet_variant.metadata_array().encoding_id(),
        );
        assert!(parquet_variant.value_array().is_some());
        assert!(parquet_variant.typed_value_array().is_none());

        Ok(())
    }

    #[test]
    fn prefers_smaller_extension_storage_over_variant_scheme() -> VortexResult<()> {
        let array: ArrayRef = json_array(&json_data())?;

        let string_compressor = CascadingCompressor::new(vec![
            &StringDictScheme,
            &FSSTScheme,
            &IntConstantScheme,
            &StringConstantScheme,
            &FoRScheme,
            &BitPackingScheme,
            &RunEndScheme,
            &SequenceScheme,
            &ZigZagScheme,
        ]);
        let mut exec_ctx = SESSION.create_execution_ctx();
        let string_compressed = string_compressor.compress(&array, &mut exec_ctx)?;

        let variant_compressor = CascadingCompressor::new(vec![
            &JsonToVariantScheme,
            &BinaryDictScheme,
            &FSSTScheme,
            &ZstdScheme,
            &IntConstantScheme,
            &StringConstantScheme,
            &FoRScheme,
            &SparseScheme,
            &BitPackingScheme,
            &RunEndScheme,
            &SequenceScheme,
            &ZigZagScheme,
        ]);
        let mut exec_ctx = SESSION.create_execution_ctx();
        let compressed = variant_compressor.compress(&array, &mut exec_ctx)?;
        let extension = compressed.clone().downcast::<Extension>();
        let storage = extension.storage_array();
        assert!(
            storage.is::<Zstd>(),
            "expected JSON extension storage fallback to use zstd, got {}",
            storage.encoding_id(),
        );

        print_comparison_output(&array, &string_compressed, &compressed);

        Ok(())
    }
}
