// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Generated Variant values for Arrow transport checks.
//!
//! Rows mix null rows, Variant nulls, and primitive and string values, so containers see both
//! masked and stored Variant children. Lengths include zero, which covers empty Variants and empty
//! containers with Variant children.

use std::sync::Arc;

use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::Fields;
use hegel::TestCase;
use hegel::generators as gs;
use parquet_variant::Variant as PqVariant;
use parquet_variant::VariantBuilderExt;
use parquet_variant_compute::VariantArrayBuilder;
use parquet_variant_compute::shred_variant;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::VariantArray;
use vortex_array::compute::conformance::hegel::EncodingCheckOptions;
use vortex_array::compute::conformance::hegel::check_encoding_roundtrip_and_ops;
use vortex_arrow::ArrowSessionExt;
use vortex_arrow::test_harness::check_containers;
use vortex_arrow::test_harness::check_generated_containers;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::ParquetVariant;

#[derive(Clone, Copy, Debug)]
enum Row {
    Null,
    VariantNull,
    Int(i32),
    Bool(bool),
    Str(&'static str),
}

fn draw_row(tc: &TestCase, nullable: bool) -> Row {
    match tc.draw(gs::integers::<u8>().max_value(4)) {
        0 if nullable => Row::Null,
        0 | 1 => Row::VariantNull,
        2 => Row::Int(tc.draw(gs::integers::<i32>())),
        3 => Row::Bool(tc.draw(gs::booleans())),
        _ => Row::Str(tc.draw(gs::sampled_from(&["", "a", "variant", "ü"]))),
    }
}

fn build_arrow_variant(
    tc: &TestCase,
    len: usize,
    nullable: bool,
) -> parquet_variant_compute::VariantArray {
    let mut builder = VariantArrayBuilder::new(len);
    for _ in 0..len {
        match draw_row(tc, nullable) {
            Row::Null => builder.append_null(),
            Row::VariantNull => builder.append_variant(PqVariant::Null),
            Row::Int(v) => builder.append_variant(PqVariant::from(v)),
            Row::Bool(v) => builder.append_variant(PqVariant::from(v)),
            Row::Str(v) => builder.append_variant(PqVariant::from(v)),
        }
    }
    builder.build()
}

fn draw_array(tc: &TestCase, session: &VortexSession) -> VortexResult<ArrayRef> {
    let len = tc.draw(gs::integers::<usize>().max_value(32));
    let nullable = tc.draw(gs::booleans());
    let arrow_variant = build_arrow_variant(tc, len, nullable);
    ParquetVariant::from_arrow_variant_with_nullability(&arrow_variant, nullable, &session.arrow())
}

#[track_caller]
fn check_variant_round_trip(tc: &TestCase) -> VortexResult<()> {
    let session = VortexSession::empty();
    crate::initialize(&session);
    let mut ctx = session.create_execution_ctx();
    let array = draw_array(tc, &session)?;

    let empty = array.slice(0..0)?;
    check_containers(empty, &mut ctx)?;

    check_containers(array.clone(), &mut ctx)?;
    check_generated_containers(tc, array, &mut ctx)
}

#[hegel::test(test_cases = 100, print_blob = true)]
fn arrow_transport(tc: TestCase) {
    check_variant_round_trip(&tc)
        .vortex_expect("Arrow transport must preserve Variant dtype and values");
}

fn draw_shredded_variant(
    tc: &TestCase,
    session: &VortexSession,
) -> VortexResult<(ArrayRef, ArrayRef)> {
    let len = tc.draw(gs::integers::<usize>().max_value(32));
    let nullable = tc.draw(gs::booleans());

    let (arrow_variant, shred_schema) = if tc.draw(gs::booleans()) {
        // Shredded primitive (Int32) with mixed int/string/null rows so both `typed_value`
        // and residual `value` are populated, checked against the unshredded reference.
        (build_arrow_variant(tc, len, nullable), DataType::Int32)
    } else {
        // Shredded object `{a: Int32, b: Utf8}` mixing matching objects, partial/extra fields,
        // non-objects, and nulls, checked against the unshredded reference.
        let mut builder = VariantArrayBuilder::new(len);
        for _ in 0..len {
            match tc.draw(gs::integers::<u8>().max_value(4)) {
                0 if nullable => builder.append_null(),
                0 | 1 => {
                    let mut obj = builder.new_object();
                    if tc.draw(gs::booleans()) {
                        obj.insert("a", tc.draw(gs::integers::<i32>()));
                    }
                    if tc.draw(gs::booleans()) {
                        obj.insert("b", tc.draw(gs::sampled_from(&["", "x", "variant"])));
                    }
                    if tc.draw(gs::booleans()) {
                        obj.insert("extra", tc.draw(gs::booleans()));
                    }
                    obj.finish();
                }
                2 => builder.append_variant(PqVariant::Null),
                3 => builder.append_variant(PqVariant::from(tc.draw(gs::integers::<i32>()))),
                _ => builder.append_variant(PqVariant::from(
                    tc.draw(gs::sampled_from(&["", "fallback", "ü"])),
                )),
            }
        }
        let schema = DataType::Struct(Fields::from(vec![
            Arc::new(Field::new("a", DataType::Int32, true)),
            Arc::new(Field::new("b", DataType::Utf8, true)),
        ]));
        (builder.build(), schema)
    };

    let unshredded = ParquetVariant::from_arrow_variant_with_nullability(
        &arrow_variant,
        nullable,
        &session.arrow(),
    )?;
    let canonical = VariantArray::try_new(unshredded, None)?.into_array();
    let shredded = shred_variant(&arrow_variant, &shred_schema)?;
    let encoded =
        ParquetVariant::from_arrow_variant_with_nullability(&shredded, nullable, &session.arrow())?;
    Ok((encoded, canonical))
}

#[hegel::test(test_cases = 40, print_blob = true)]
fn variant_hegel_encoding_harness(tc: TestCase) {
    let session = VortexSession::empty();
    vortex_array::initialize(&session);
    crate::initialize(&session);
    let mut ctx = session.create_execution_ctx();
    let (encoded, canonical) = draw_shredded_variant(&tc, &session).vortex_expect("draw variant");

    let options = EncodingCheckOptions {
        compare: false,
        cast: false,
        aggregates: false,
        search_sorted: false,
        ..Default::default()
    };
    check_encoding_roundtrip_and_ops(&tc, &encoded, &canonical, &options, &mut ctx)
        .vortex_expect("ParquetVariant harness check");
}
