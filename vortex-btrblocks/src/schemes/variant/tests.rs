// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::LazyLock;

use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::extension::ExtensionArrayExt;
use vortex_array::session::ArraySession;
use vortex_array::validity::Validity;
use vortex_compressor::builtins::BinaryDictScheme;
use vortex_compressor::builtins::IntConstantScheme;
use vortex_compressor::builtins::StringConstantScheme;
use vortex_compressor::builtins::StringDictScheme;
use vortex_json::Json;
use vortex_parquet_variant::VariantToJson;
use vortex_session::VortexSession;

use super::*;
use crate::CascadingCompressor;
use crate::schemes::binary;
use crate::schemes::binary::BinaryFSSTScheme;
use crate::schemes::integer::BitPackingScheme;
use crate::schemes::integer::FoRScheme;
use crate::schemes::integer::RunEndScheme;
use crate::schemes::integer::SequenceScheme;
use crate::schemes::integer::SparseScheme;
use crate::schemes::integer::ZigZagScheme;
use crate::schemes::string::FSSTScheme;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::empty().with::<ArraySession>();
    vortex_parquet_variant::initialize(&session);
    session
});

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

fn json_array(values: &[String]) -> vortex_error::VortexResult<ArrayRef> {
    let storage = VarBinViewArray::from_iter_str(values.iter().map(String::as_str)).into_array();
    Ok(
        ExtensionArray::try_new_from_vtable(Json, vortex_array::EmptyMetadata, storage)?
            .into_array(),
    )
}

fn all_valid_nullable_json_array(
    values: impl IntoIterator<Item = &'static str>,
) -> vortex_error::VortexResult<ArrayRef> {
    let storage = VarBinViewArray::from_iter_str(values);
    let parts = storage.into_data_parts();
    let storage = VarBinViewArray::new_handle(
        parts.views,
        parts.buffers,
        parts.dtype.as_nullable(),
        Validity::AllValid,
    )
    .into_array();

    Ok(
        ExtensionArray::try_new_from_vtable(Json, vortex_array::EmptyMetadata, storage)?
            .into_array(),
    )
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
fn json_to_variant_scheme_wraps_output_as_json() -> vortex_error::VortexResult<()> {
    let array = json_array(&json_data())?;

    let variant_compressor = parquet_variant_child_compressor();
    let mut exec_ctx = SESSION.create_execution_ctx();
    let compressed = variant_compressor.compress(&array, &mut exec_ctx)?;

    assert_eq!(compressed.dtype(), array.dtype());
    assert!(compressed.is::<VariantToJson>());

    let json = compressed.execute::<ExtensionArray>(&mut exec_ctx)?;
    assert_eq!(json.dtype(), array.dtype());
    assert!(json.storage_array().dtype().is_utf8());

    Ok(())
}

#[test]
fn preserves_nullable_json_dtype_for_all_valid_storage() -> vortex_error::VortexResult<()> {
    let values = [r#"{"a":1}"#, r#"{"b":2}"#];
    let storage = VarBinViewArray::from_iter_str(values);
    let parts = storage.into_data_parts();
    let storage = VarBinViewArray::new_handle(
        parts.views,
        parts.buffers,
        parts.dtype.as_nullable(),
        Validity::AllValid,
    )
    .into_array();

    let array = ExtensionArray::try_new_from_vtable(Json, vortex_array::EmptyMetadata, storage)?
        .into_array();

    assert!(array.dtype().is_nullable());

    let variant_compressor = CascadingCompressor::new(vec![&JsonToVariantScheme]);
    let mut exec_ctx = SESSION.create_execution_ctx();
    let compressed = variant_compressor.compress(&array, &mut exec_ctx)?;

    assert_eq!(compressed.dtype(), array.dtype());

    let json = compressed.execute::<ExtensionArray>(&mut exec_ctx)?;
    assert_eq!(json.dtype(), array.dtype());

    Ok(())
}

fn print_comparison_output(array: &ArrayRef, string_compressed: &ArrayRef, compressed: &ArrayRef) {
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
fn parquet_variant_compresses_repeated_json_keys() -> vortex_error::VortexResult<()> {
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
fn recursively_compresses_parquet_variant_binary_children() -> vortex_error::VortexResult<()> {
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
fn binary_fsst_improves_parquet_variant_child_compression() -> vortex_error::VortexResult<()> {
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
    let with_binary_fsst = parquet_variant_child_compressor().compress(&array, &mut exec_ctx)?;

    assert!(
        with_binary_fsst.nbytes() < without_binary_fsst.nbytes(),
        "binary FSST should improve Parquet Variant child compression: with={} bytes, without={} bytes",
        with_binary_fsst.nbytes(),
        without_binary_fsst.nbytes(),
    );

    Ok(())
}

#[test]
fn prefers_smaller_extension_storage_over_variant_scheme() -> vortex_error::VortexResult<()> {
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
