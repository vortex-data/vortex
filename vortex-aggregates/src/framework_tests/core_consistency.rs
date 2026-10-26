// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Aggregate consistency of the core encodings.
//!
//! `test_array_consistency` only checks aggregates when the session registers them, which the
//! core crate's own tests cannot do, so the core encodings are checked here.

use std::sync::LazyLock;

use rstest::rstest;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::compute::conformance::consistency::test_slice_aggregate_consistency;
use vortex_buffer::buffer;
use vortex_session::VortexSession;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::empty();
    crate::initialize(&session);
    session
});

#[rstest]
#[case::primitive(buffer![5i32, 1, 9, 3, 7, 2, 8].into_array())]
#[case::nullable(
    PrimitiveArray::from_option_iter([Some(4u64), None, Some(1), Some(9), None, Some(6), Some(2)])
        .into_array()
)]
#[case::float_with_nan(buffer![1.5f64, f64::NAN, -2.0, 8.25, f64::NAN, 0.0, 3.5].into_array())]
#[case::constant(ConstantArray::new(7i64, 9).into_array())]
#[case::dict(
    DictArray::new(
        buffer![0u8, 2, 1, 2, 0, 1, 2].into_array(),
        buffer![10i32, -3, 42].into_array(),
    )
    .into_array()
)]
#[case::chunked(
    ChunkedArray::from_iter([
        buffer![3i16, 1, 4].into_array(),
        buffer![1i16, 5, 9, 2].into_array(),
    ])
    .into_array()
)]
fn core_encodings_aggregate_consistency(#[case] array: ArrayRef) {
    test_slice_aggregate_consistency(&array, &mut SESSION.create_execution_ctx());
}
