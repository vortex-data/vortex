// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Generated Variant values for Arrow transport checks.
//!
//! Rows mix null rows, Variant nulls, and primitive and string values, so containers see both
//! masked and stored Variant children. Lengths include zero, which covers empty Variants and empty
//! containers with Variant children.

use hegel::TestCase;
use hegel::generators as gs;
use parquet_variant::Variant as PqVariant;
use parquet_variant_compute::VariantArrayBuilder;
use vortex_array::ArrayRef;
use vortex_array::VortexSessionExecute;
use vortex_arrow::ArrowSessionExt;
use vortex_arrow::test_harness::check_containers;
use vortex_arrow::test_harness::check_generated;
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

fn draw_array(tc: &TestCase, session: &VortexSession) -> VortexResult<ArrayRef> {
    let len = tc.draw(gs::integers::<usize>().max_value(32));
    let nullable = tc.draw(gs::booleans());
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
    ParquetVariant::from_arrow_variant_with_nullability(
        &builder.build(),
        nullable,
        &session.arrow(),
    )
}

#[track_caller]
fn check_variant_round_trip(tc: &TestCase) -> VortexResult<()> {
    let session = VortexSession::empty();
    crate::initialize(&session);
    let mut ctx = session.create_execution_ctx();
    let array = draw_array(tc, &session)?;

    let empty = array.slice(0..0)?;
    check_containers(empty, &mut ctx)?;

    check_generated(tc, array, &mut ctx)
}

#[hegel::test(test_cases = 100, print_blob = true)]
fn arrow_transport(tc: TestCase) {
    check_variant_round_trip(&tc)
        .vortex_expect("Arrow transport must preserve Variant dtype and values");
}
