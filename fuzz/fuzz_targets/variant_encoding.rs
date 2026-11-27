// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![no_main]

//! Compare the ParquetVariant and canonical Variant encodings with arrow-rs on the same rows.

use std::sync::Arc;
use std::sync::LazyLock;

use arrow_array::Array;
use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::StringArray;
use arrow_schema::DataType;
use arrow_schema::Field;
use libfuzzer_sys::fuzz_target;
use parquet_variant::VariantPath as ArrowPath;
use parquet_variant_compute::GetOptions;
use parquet_variant_compute::ShreddedSchemaBuilder;
use parquet_variant_compute::VariantArray as ArrowVariant;
use parquet_variant_compute::json_to_variant;
use parquet_variant_compute::shred_variant;
use parquet_variant_compute::unshred_variant;
use parquet_variant_compute::variant_get;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::VariantArray;
use vortex_arrow::ArrowSessionExt;
use vortex_arrow::ExecuteArrowExt;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_functions::variant_get::VariantGet;
use vortex_functions::variant_get::VariantGetOptions;
use vortex_functions::variant_get::VariantPath;
use vortex_functions::variant_get::VariantPathElement;
use vortex_parquet_variant::ParquetVariant;
use vortex_session::VortexSession;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::empty();
    vortex_array::initialize(&session);
    vortex_functions::initialize(&session);
    vortex_parquet_variant::initialize(&session);
    session
});

fn json_row(bytes: &[u8]) -> Option<String> {
    let number = i16::from_le_bytes([bytes[1], bytes[2]]);
    let value = match bytes[0] % 8 {
        0 => return None,
        1 => "null".to_owned(),
        2 => (number >= 0).to_string(),
        3 => format!("{number}.5"),
        4 => format!(r#""{number}""#),
        5 => "[]".to_owned(),
        6 => "{}".to_owned(),
        _ => number.to_string(),
    };
    Some(match bytes[0] / 8 % 5 {
        0 => value,
        1 => format!(r#"{{"a":{value},"extra":{number}}}"#),
        2 => format!(r#"{{"a":{{"b":{value},"extra":{number}}}}}"#),
        3 => format!(r#"{{"a":[{value},null,{number}]}}"#),
        _ => format!("[{value}]"),
    })
}

fn assert_variant_eq(actual: &ArrowVariant, expected: &ArrowVariant) -> VortexResult<()> {
    let actual = unshred_variant(actual)?;
    let expected = unshred_variant(expected)?;
    assert_eq!(actual.len(), expected.len());
    for i in 0..expected.len() {
        assert_eq!(actual.is_null(i), expected.is_null(i), "row {i}");
        if expected.is_valid(i) {
            assert_eq!(actual.try_value(i)?, expected.try_value(i)?, "row {i}");
        }
    }
    Ok(())
}

fn compare(data: &[u8]) -> VortexResult<()> {
    let Some((&[schema, path, target, offset, length], rows)) = data.split_first_chunk::<5>() else {
        return Ok(());
    };
    let strings: ArrowArrayRef = Arc::new(StringArray::from_iter(
        rows.chunks_exact(3).take(32).map(json_row),
    ));
    let source = json_to_variant(&strings)?;
    let source = match schema % 5 {
        0 => source,
        1 => shred_variant(&source, &DataType::Int64)?,
        n => {
            let (path, ty) = match n {
                2 => ("a", DataType::Utf8),
                3 => ("a.b", DataType::Boolean),
                _ => ("a[*]", DataType::Int32),
            };
            let schema = ShreddedSchemaBuilder::new().with_path(path, &ty)?.build();
            shred_variant(&source, &schema)?
        }
    };
    let start = usize::from(offset) % (source.len() + 1);
    let remaining = source.len() - start;
    let len = remaining - usize::from(length) % (remaining + 1);
    let arrow = SESSION.arrow();
    let encoded = ParquetVariant::from_arrow_variant(&source, &arrow)?.slice(start..start + len)?;
    let sliced = source.inner().slice(start, len);
    let reference = ArrowVariant::try_new(&sliced)?;
    let input: ArrowArrayRef = Arc::new(sliced);
    let (pq_path, vx_path) = match path % 5 {
        0 => (ArrowPath::default(), VariantPath::root()),
        1 => (ArrowPath::default().join("a"), VariantPath::field("a")),
        2 => (
            ArrowPath::default().join("a").join("b"),
            VariantPath::new(["a".into(), "b".into()]),
        ),
        3 => (
            ArrowPath::default().join("a").join(0usize),
            VariantPath::new(["a".into(), VariantPathElement::index(0)]),
        ),
        _ => (ArrowPath::default().join("missing"), VariantPath::field("missing")),
    };
    let ty = match target % 5 {
        0 => None,
        1 => Some(DataType::Utf8),
        2 => Some(DataType::Int64),
        3 => Some(DataType::Boolean),
        _ => Some(DataType::Float64),
    };
    let field = ty.map(|ty| Arc::new(Field::new("result", ty, true)));
    let get = GetOptions::new_with_path(pq_path).with_as_type(field.clone());
    let expected = variant_get(&input, get)?;
    let dtype = field.as_ref().map(|f| arrow.from_arrow_field(f)).transpose()?;
    let options = VariantGetOptions::new(vx_path, dtype);
    let mut ctx = SESSION.create_execution_ctx();
    let canonical = encoded.clone().execute::<VariantArray>(&mut ctx)?.into_array();
    let variant_field = arrow.to_arrow_field("variant", canonical.dtype())?;
    let exported = canonical.clone().execute_arrow(Some(&variant_field), &mut ctx)?;
    assert_variant_eq(&ArrowVariant::try_new(exported.as_ref())?, &reference)?;

    for array in [encoded, canonical] {
        let actual = VariantGet::try_new(array, options.clone())?.into_array();
        let result_field = arrow.to_arrow_field("result", actual.dtype())?;
        let export_field = field.as_deref().unwrap_or(&result_field);
        let actual = actual.execute_arrow(Some(export_field), &mut ctx)?;
        if field.is_some() {
            assert_eq!(actual.as_ref(), expected.as_ref());
        } else {
            assert_variant_eq(
                &ArrowVariant::try_new(actual.as_ref())?,
                &ArrowVariant::try_new(expected.as_ref())?,
            )?;
        }
    }
    Ok(())
}

fuzz_target!(|data: &[u8]| {
    compare(data).vortex_expect("Variant encoding should agree with arrow-rs");
});
