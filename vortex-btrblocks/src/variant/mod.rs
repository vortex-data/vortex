// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compression scheme for JSON data into binary variant representation

use std::sync::Arc;

use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::StructArray as ArrowStructArray;
use arrow_schema::DataType;
use arrow_schema::Field;
use vortex_array::Array;
use vortex_array::ArrayId;
use vortex_array::ArrayParts;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::Canonical;
use vortex_array::EmptyArrayData;
use vortex_array::EmptyMetadata;
use vortex_array::ExecutionCtx;
use vortex_array::ExecutionResult;
use vortex_array::IntoArray;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::VariantArray;
use vortex_array::arrays::extension::ExtensionArrayExt;
use vortex_array::arrays::variant::VariantArrayExt;
use vortex_array::arrow::ArrowSessionExt;
use vortex_array::arrow::FromArrowArray;
use vortex_array::arrow::to_arrow_null_buffer;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::dtype::extension::ExtDType;
use vortex_array::serde::ArrayChildren;
use vortex_array::validity::Validity;
use vortex_array::vtable::NotSupported;
use vortex_array::vtable::VTable;
use vortex_array::vtable::ValidityVTable;
use vortex_compressor::ctx::CompressorContext;
use vortex_compressor::estimate::CompressionEstimate;
use vortex_compressor::estimate::DeferredEstimate;
use vortex_compressor::scheme::Scheme;
use vortex_compressor::scheme::SchemeExt;
use vortex_compressor::stats::ArrayAndStats;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_json::Json;
use vortex_parquet_variant::ParquetVariant;
use vortex_parquet_variant::ParquetVariantArrayExt;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

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

mod variant_to_json_children {
    pub const VARIANT: usize = 0;
    pub const NUM_SLOTS: usize = 1;
    pub const SLOT_NAMES: [&str; NUM_SLOTS] = ["variant"];
}

/// Array that exposes a Variant array as JSON strings.
#[derive(Debug, Clone)]
pub struct VariantToJson;

/// A [`VariantToJson`]-encoded array.
pub type VariantToJsonArray = Array<VariantToJson>;

impl VariantToJson {
    /// Creates a JSON wrapper around a Variant-typed array.
    pub fn try_new(variant: ArrayRef) -> VortexResult<VariantToJsonArray> {
        vortex_ensure!(
            variant.dtype().is_variant(),
            "VariantToJson expects a Variant array, got {}",
            variant.dtype()
        );

        let storage_dtype = DType::Utf8(variant.dtype().nullability());
        let dtype =
            DType::Extension(ExtDType::<Json>::try_new(EmptyMetadata, storage_dtype)?.erased());
        let len = variant.len();

        Array::try_from_parts(
            ArrayParts::new(VariantToJson, dtype, len, EmptyArrayData)
                .with_slots(vec![Some(variant)].into()),
        )
    }
}

impl VTable for VariantToJson {
    type TypedArrayData = EmptyArrayData;
    type OperationsVTable = NotSupported;
    type ValidityVTable = Self;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.variant_to_json");
        *ID
    }

    fn validate(
        &self,
        _data: &Self::TypedArrayData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        vortex_ensure!(
            slots.len() == variant_to_json_children::NUM_SLOTS,
            "VariantToJsonArray expects {} slots, got {}",
            variant_to_json_children::NUM_SLOTS,
            slots.len()
        );
        let variant = slots[variant_to_json_children::VARIANT]
            .as_ref()
            .ok_or_else(|| vortex_err!("VariantToJsonArray variant slot must be present"))?;

        let DType::Extension(ext_dtype) = dtype else {
            vortex_bail!("VariantToJsonArray dtype must be a JSON extension, got {dtype}");
        };
        vortex_ensure!(
            ext_dtype.is::<Json>(),
            "VariantToJsonArray dtype must be a JSON extension, got {dtype}"
        );
        vortex_ensure!(
            variant.dtype() == &DType::Variant(dtype.nullability()),
            "VariantToJsonArray child dtype {} does not match JSON dtype nullability {}",
            variant.dtype(),
            dtype
        );
        vortex_ensure!(
            variant.len() == len,
            "VariantToJsonArray child length {} does not match outer length {}",
            variant.len(),
            len
        );

        Ok(())
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        0
    }

    fn buffer(_array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        vortex_panic!("VariantToJsonArray buffer index {idx} out of bounds")
    }

    fn buffer_name(_array: ArrayView<'_, Self>, _idx: usize) -> Option<String> {
        None
    }

    fn serialize(
        _array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(Vec::new()))
    }

    fn deserialize(
        &self,
        dtype: &DType,
        len: usize,
        metadata: &[u8],
        buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
        _session: &VortexSession,
    ) -> VortexResult<ArrayParts<Self>> {
        vortex_ensure!(
            metadata.is_empty(),
            "VariantToJsonArray metadata must be empty"
        );
        vortex_ensure!(
            buffers.is_empty(),
            "VariantToJsonArray expects 0 buffers, got {}",
            buffers.len()
        );
        vortex_ensure!(
            children.len() == variant_to_json_children::NUM_SLOTS,
            "VariantToJsonArray expects {} children, got {}",
            variant_to_json_children::NUM_SLOTS,
            children.len()
        );

        let variant_dtype = DType::Variant(dtype.nullability());
        let variant = children.get(variant_to_json_children::VARIANT, &variant_dtype, len)?;

        Ok(
            ArrayParts::new(self.clone(), dtype.clone(), len, EmptyArrayData)
                .with_slots(vec![Some(variant)].into()),
        )
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        match variant_to_json_children::SLOT_NAMES.get(idx) {
            Some(name) => (*name).to_string(),
            None => vortex_panic!("VariantToJsonArray slot_name index {idx} out of bounds"),
        }
    }

    fn execute(array: Array<Self>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        let variant = array.as_ref().slots()[variant_to_json_children::VARIANT]
            .as_ref()
            .ok_or_else(|| vortex_err!("VariantToJsonArray variant slot must be present"))?;
        let variant = variant.clone().execute::<VariantArray>(ctx)?;
        vortex_ensure!(
            variant.shredded().is_none(),
            "VariantToJsonArray can only export unshredded Parquet Variant storage to JSON"
        );

        let parquet_variant = variant
            .core_storage()
            .as_opt::<ParquetVariant>()
            .ok_or_else(|| {
                vortex_err!(
                    "VariantToJsonArray requires Parquet Variant core storage, got {}",
                    variant.core_storage().encoding_id()
                )
            })?;
        let arrow_variant = parquet_variant_to_json_arrow(parquet_variant, ctx)?;
        let arrow_json = parquet_variant_compute::variant_to_json(&arrow_variant)?;
        let storage = ArrayRef::from_arrow(&arrow_json, array.dtype().is_nullable())?;

        Ok(ExecutionResult::done(
            ExtensionArray::try_new_from_vtable(Json, EmptyMetadata, storage)?.into_array(),
        ))
    }
}

impl ValidityVTable<VariantToJson> for VariantToJson {
    fn validity(array: ArrayView<'_, VariantToJson>) -> VortexResult<Validity> {
        array.slots()[variant_to_json_children::VARIANT]
            .as_ref()
            .ok_or_else(|| vortex_err!("VariantToJsonArray variant slot must be present"))?
            .validity()
    }
}

fn parquet_variant_to_json_arrow(
    parquet_variant: ArrayView<'_, ParquetVariant>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrowArrayRef> {
    vortex_ensure!(
        parquet_variant.typed_value_array().is_none(),
        "VariantToJsonArray can only export unshredded Parquet Variant storage to JSON"
    );
    let value = parquet_variant
        .value_array()
        .ok_or_else(|| vortex_err!("VariantToJsonArray requires Parquet Variant value storage"))?;

    let metadata_arrow = {
        let target = Field::new("", DataType::Binary, false);
        let session = ctx.session().clone();
        session.arrow().execute_arrow(
            parquet_variant.metadata_array().clone(),
            Some(&target),
            ctx,
        )?
    };
    let value_arrow = {
        let target = Field::new("", DataType::Binary, value.dtype().is_nullable());
        let session = ctx.session().clone();
        session
            .arrow()
            .execute_arrow(value.clone(), Some(&target), ctx)?
    };
    let fields = vec![
        Arc::new(Field::new("metadata", DataType::Binary, false)),
        Arc::new(Field::new(
            "value",
            DataType::Binary,
            value.dtype().is_nullable(),
        )),
    ];
    let nulls = to_arrow_null_buffer(parquet_variant.validity()?, parquet_variant.len(), ctx)?;

    Ok(Arc::new(ArrowStructArray::try_new(
        fields.into(),
        vec![metadata_arrow, value_arrow],
        nulls,
    )?))
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

        let variant = ParquetVariant::try_new(
            parquet_variant.validity()?,
            compressed_metadata,
            compressed_value,
            parquet_variant.typed_value_array().cloned(),
        )?
        .into_array();

        Ok(VariantToJson::try_new(variant)?.into_array())
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
    use vortex_array::accessor::ArrayAccessor;
    use vortex_array::arrays::ExtensionArray;
    use vortex_array::arrays::VarBinViewArray;
    use vortex_array::arrays::extension::ExtensionArrayExt;
    use vortex_array::session::ArraySession;
    use vortex_compressor::builtins::BinaryDictScheme;
    use vortex_compressor::builtins::IntConstantScheme;
    use vortex_compressor::builtins::StringConstantScheme;
    use vortex_compressor::builtins::StringDictScheme;
    use vortex_session::VortexSession;

    use super::*;
    use crate::schemes::binary;
    use crate::schemes::binary::BinaryFSSTScheme;
    use crate::schemes::integer::BitPackingScheme;
    use crate::schemes::integer::FoRScheme;
    use crate::schemes::integer::RunEndScheme;
    use crate::schemes::integer::SequenceScheme;
    use crate::schemes::integer::SparseScheme;
    use crate::schemes::integer::ZigZagScheme;
    use crate::schemes::string::FSSTScheme;

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

    #[test]
    fn variant_to_json_canonicalizes_to_json_extension() -> VortexResult<()> {
        let values = [
            "0".to_string(),
            r#"{"a":32}"#.to_string(),
            r#""hello""#.to_string(),
            "null".to_string(),
        ];
        let storage =
            VarBinViewArray::from_iter_str(values.iter().map(String::as_str)).into_array();
        let source =
            ExtensionArray::try_new_from_vtable(Json, EmptyMetadata, storage.clone())?.into_array();

        let mut exec_ctx = SESSION.create_execution_ctx();
        let arrow_array = {
            let session = exec_ctx.session().clone();
            session
                .arrow()
                .execute_arrow(storage, None, &mut exec_ctx)?
        };
        let arrow_variant = parquet_variant_compute::json_to_variant(&arrow_array)?;
        let variant = ParquetVariant::from_arrow_variant(&arrow_variant)?;

        let wrapped = VariantToJson::try_new(variant)?;
        assert_eq!(wrapped.dtype(), source.dtype());

        let json = wrapped
            .into_array()
            .execute::<ExtensionArray>(&mut exec_ctx)?;
        assert_eq!(json.dtype(), source.dtype());
        assert!(json.storage_array().dtype().is_utf8());
        let json_storage = json
            .storage_array()
            .clone()
            .execute::<VarBinViewArray>(&mut exec_ctx)?;
        let actual = json_storage.with_iterator(|iter| {
            iter.map(|value| value.map(<[u8]>::to_vec))
                .collect::<Vec<_>>()
        });
        let expected = values
            .iter()
            .map(|value| Some(value.as_bytes().to_vec()))
            .collect::<Vec<_>>();

        assert_eq!(actual, expected);

        Ok(())
    }

    fn parquet_variant_child_compressor() -> CascadingCompressor {
        CascadingCompressor::new(vec![
            &JsonToVariantScheme,
            &BinaryDictScheme,
            &BinaryFSSTScheme,
            &IntConstantScheme,
            &FoRScheme,
            &SparseScheme,
            &BitPackingScheme,
            &RunEndScheme,
            &SequenceScheme,
            &ZigZagScheme,
        ])
    }

    #[test]
    fn json_to_variant_scheme_wraps_output_as_json() -> VortexResult<()> {
        let array = json_array(&json_data())?;

        let variant_compressor = parquet_variant_child_compressor();
        let mut exec_ctx = SESSION.create_execution_ctx();
        let compressed = variant_compressor.compress(&array, &mut exec_ctx)?;

        assert_eq!(compressed.dtype(), array.dtype());

        let json = compressed.execute::<ExtensionArray>(&mut exec_ctx)?;
        assert_eq!(json.dtype(), array.dtype());
        assert!(json.storage_array().dtype().is_utf8());

        Ok(())
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
        let variant_compressed = variant_compressor.compress(&array, &mut exec_ctx)?;

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

        let mut exec_ctx = SESSION.create_execution_ctx();
        let uncompressed_children =
            CascadingCompressor::new(vec![&JsonToVariantScheme]).compress(&array, &mut exec_ctx)?;

        let variant_compressor = parquet_variant_child_compressor();
        let mut exec_ctx = SESSION.create_execution_ctx();
        let compressed = variant_compressor.compress(&array, &mut exec_ctx)?;

        assert!(
            compressed.nbytes() < uncompressed_children.nbytes(),
            "recursive child compression should reduce Parquet Variant size: compressed={} bytes, uncompressed_children={} bytes",
            compressed.nbytes(),
            uncompressed_children.nbytes(),
        );
        assert_eq!(compressed.dtype(), array.dtype());
        Ok(())
    }

    #[test]
    fn binary_fsst_improves_parquet_variant_child_compression() -> VortexResult<()> {
        let array: ArrayRef = json_array(&json_data())?;
        let mut exec_ctx = SESSION.create_execution_ctx();
        let without_binary_fsst = CascadingCompressor::new(vec![
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
        .compress(&array, &mut exec_ctx)?;

        let mut exec_ctx = SESSION.create_execution_ctx();
        let with_binary_fsst =
            parquet_variant_child_compressor().compress(&array, &mut exec_ctx)?;

        assert!(
            with_binary_fsst.nbytes() < without_binary_fsst.nbytes(),
            "binary FSST should improve Parquet Variant child compression: with={} bytes, without={} bytes",
            with_binary_fsst.nbytes(),
            without_binary_fsst.nbytes(),
        );

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
            &BinaryFSSTScheme,
            &binary::ZstdScheme,
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

        print_comparison_output(&array, &string_compressed, &compressed);

        Ok(())
    }
}
