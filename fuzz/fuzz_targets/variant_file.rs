// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![no_main]

use std::sync::Arc;

use arbitrary::Arbitrary;
use arbitrary::Unstructured;
use arrow_array::Array;
use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::StringArray;
use arrow_schema::DataType;
use libfuzzer_sys::fuzz_target;
use parquet_variant_compute::ShreddedSchemaBuilder;
use parquet_variant_compute::VariantArray;
use parquet_variant_compute::json_to_variant;
use parquet_variant_compute::shred_variant;
use parquet_variant_compute::unshred_variant;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::fns::all_non_distinct::variant_scalars_equal;
use vortex_array::arrays::ChunkedArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::iter::ArrayIteratorExt;
use vortex_arrow::ArrowSessionExt;
use vortex_arrow::ExecuteArrowExt;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_file::OpenOptionsSessionExt;
use vortex_file::WriteOptionsSessionExt;
use vortex_file::WriteStrategyBuilder;
use vortex_fuzz::RUNTIME;
use vortex_fuzz::SESSION;

const ROWS: &[Option<&str>] = &[
    None,
    Some("null"),
    Some("{}"),
    Some("[]"),
    Some("true"),
    Some("3"),
    Some(r#""hello""#),
    Some(r#"{"a":42,"extra":0}"#),
    Some(r#"{"a":"x","b":null}"#),
    Some(r#"{"b":false}"#),
    Some(r#"{"a":{"x":7,"y":"z"}}"#),
    Some(r#"{"a":[{"x":2},{"y":4},null]}"#),
    Some(r#"[{"x":2},null,"x"]"#),
    Some(r#"{"a":[],"b":{}}"#),
];

#[derive(Debug)]
struct Chunk {
    schema: u8,
    rows: Vec<u8>,
}

#[derive(Debug)]
struct Case {
    block: u8,
    chunks: Vec<Chunk>,
    range: [u8; 2],
}

impl<'a> Arbitrary<'a> for Case {
    fn arbitrary(input: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let block = input.arbitrary()?;
        let range = input.arbitrary()?;
        let count = input.int_in_range(1..=4)?;
        let mut chunks = Vec::with_capacity(count);
        for _ in 0..count {
            let schema = input.arbitrary()?;
            let len = input.int_in_range(1..=5)?;
            let rows = input.bytes(len)?.to_vec();
            chunks.push(Chunk { schema, rows });
        }
        Ok(Self {
            block,
            chunks,
            range,
        })
    }
}

fn roundtrip(case: Case) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let mut chunks = Vec::new();
    let mut expected = Vec::new();
    let dtype = DType::Variant(Nullability::Nullable);
    let field = SESSION.arrow().to_arrow_field("v", &dtype)?;
    let import = |variant: &VariantArray| {
        let storage = variant.inner();
        let field = field.clone().with_data_type(storage.data_type().clone());
        SESSION
            .arrow()
            .from_arrow_array(Arc::new(storage.clone()), &field)
    };
    for chunk in &case.chunks {
        let strings: ArrowArrayRef = Arc::new(StringArray::from_iter(
            chunk
                .rows
                .iter()
                .map(|&i| ROWS[usize::from(i) % ROWS.len()]),
        ));
        let original = json_to_variant(&strings)?;
        let (path, ty) = match chunk.schema % 8 {
            0 => ("a", DataType::Int32),
            1 => ("a", DataType::Utf8),
            2 => ("b", DataType::Boolean),
            3 => ("a.x", DataType::Int64),
            4 => ("a[*].x", DataType::Int32),
            5 => ("", DataType::Int32),
            6 => ("", DataType::Utf8),
            _ => ("", DataType::Null),
        };
        let physical = if ty == DataType::Null {
            original
        } else if path.is_empty() {
            shred_variant(&original, &ty)?
        } else {
            let schema = ShreddedSchemaBuilder::new().with_path(path, &ty)?.build();
            shred_variant(&original, &schema)?
        };
        // Parquet shredding can change numeric widths. Compare with Arrow's decoding
        // of each input chunk before Vortex has seen it.
        let reference = if physical.typed_value_column().is_some() {
            unshred_variant(&physical)?
        } else {
            physical.clone()
        };
        let reference = import(&reference)?;
        for row in 0..reference.len() {
            expected.push(reference.execute_scalar(row, &mut ctx)?);
        }
        chunks.push(import(&physical)?);
    }

    let blocks = [1, 2, 3, 8, 8192];
    let strategy = WriteStrategyBuilder::from_session(&SESSION)
        .with_row_block_size(blocks[usize::from(case.block) % blocks.len()])
        .build();
    let input = ChunkedArray::try_new(chunks, dtype)?.into_array();
    let mut bytes = Vec::new();
    SESSION
        .write_options()
        .with_strategy(strategy)
        .blocking(&*RUNTIME)
        .write(&mut bytes, input.to_array_iterator())?;
    let file = SESSION.open_options().open_buffer(bytes)?;
    let full = file.scan()?.into_array_iter(&*RUNTIME)?.read_all()?;
    let start = usize::from(case.range[0]) % expected.len();
    let end = start + 1 + usize::from(case.range[1]) % (expected.len() - start);
    let partial = file
        .scan()?
        .with_row_range(start as u64..end as u64)
        .into_array_iter(&*RUNTIME)?
        .read_all()?;
    let mut check = |array: &ArrayRef, range: std::ops::Range<usize>| -> VortexResult<()> {
        assert_eq!(array.len(), range.len());
        for (row, index) in range.enumerate() {
            let got = array.execute_scalar(row, &mut ctx)?;
            assert!(
                variant_scalars_equal(&got, &expected[index]),
                "row {index}: {got:?} vs {:?}",
                expected[index]
            );
        }
        Ok(())
    };
    check(&full, 0..expected.len())?;
    check(&partial, start..end)?;
    let exported = full.execute_arrow(Some(&field), &mut SESSION.create_execution_ctx())?;
    let restored = SESSION.arrow().from_arrow_array(exported, &field)?;
    check(&restored, 0..expected.len())
}

fuzz_target!(|case: Case| {
    roundtrip(case).vortex_expect("valid Variant chunks should round-trip");
});
