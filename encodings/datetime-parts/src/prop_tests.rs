// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;
use std::sync::LazyLock;

use hegel::TestCase;
use hegel::generators as gs;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::TemporalArray;
use vortex_array::compute::conformance::hegel::EncodingCheckOptions;
use vortex_array::compute::conformance::hegel::check_encoding_roundtrip_and_ops;
use vortex_array::compute::conformance::hegel::draw_validity_for;
use vortex_array::dtype::Nullability;
use vortex_array::extension::datetime::TimeUnit;
use vortex_buffer::Buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::DateTimeParts;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::empty();
    vortex_array::initialize(&session);
    crate::initialize(&session);
    session
});

const SUB_DAY_UNITS: [TimeUnit; 4] = [
    TimeUnit::Seconds,
    TimeUnit::Milliseconds,
    TimeUnit::Microseconds,
    TimeUnit::Nanoseconds,
];

fn draw_timestamp_values(tc: &TestCase, len: usize, unit: TimeUnit) -> Vec<i64> {
    // Mix narrow timestamps (fit in i32 days) with wide timestamps (force i64 days) and
    // negative pre-epoch values.
    let wide = tc.draw(gs::integers::<u8>().max_value(3)) == 0;
    let mut values: Vec<i64> = (0..len)
        .map(|_| {
            if wide && matches!(unit, TimeUnit::Seconds | TimeUnit::Milliseconds) {
                tc.draw(
                    gs::integers::<i64>()
                        .min_value(i64::MIN / 2)
                        .max_value(i64::MAX / 2),
                )
            } else {
                match tc.draw(gs::integers::<u8>().max_value(3)) {
                    0 => tc.draw(
                        gs::integers::<i64>()
                            .min_value(-1_000_000)
                            .max_value(1_000_000),
                    ),
                    1 => tc.draw(
                        gs::integers::<i64>()
                            .min_value(-1_000_000_000_000)
                            .max_value(1_000_000_000_000),
                    ),
                    2 => tc.draw(gs::sampled_from(&[0, -1, 1, 86_400, -86_400])),
                    _ => tc.draw(gs::integers::<i64>()),
                }
            }
        })
        .collect();

    if tc.draw(gs::integers::<u8>().max_value(3)) == 0 {
        values.sort_unstable();
    }
    values
}

fn check_datetime_parts(tc: &TestCase) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let unit = tc.draw(gs::sampled_from(&SUB_DAY_UNITS));
    let tz: Option<Arc<str>> = match tc.draw(gs::integers::<u8>().max_value(2)) {
        0 => None,
        1 => Some(Arc::from("UTC")),
        _ => Some(Arc::from("America/New_York")),
    };
    let len = tc.draw(gs::integers::<usize>().max_value(64));
    let nullability = if tc.draw(gs::booleans()) {
        Nullability::Nullable
    } else {
        Nullability::NonNullable
    };
    let validity = draw_validity_for(tc, nullability, len);
    let values = draw_timestamp_values(tc, len, unit);
    let prim = PrimitiveArray::new(Buffer::from(values), validity).into_array();
    let temporal = TemporalArray::new_timestamp(prim, unit, tz);
    let canonical = temporal.clone().into_array();
    let encoded = DateTimeParts::try_from_temporal(temporal, &mut ctx)?.into_array();

    check_encoding_roundtrip_and_ops(
        tc,
        &encoded,
        &canonical,
        &EncodingCheckOptions::default(),
        &mut ctx,
    )
}

#[hegel::test(test_cases = 50, print_blob = true)]
fn test_datetime_parts_hegel_encoding_harness(tc: TestCase) {
    check_datetime_parts(&tc).vortex_expect("DateTimeParts harness check");
}
