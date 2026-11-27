// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![cfg(all(test, not(codspeed)))]

use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::DType;
use vortex_array::extension::datetime::TimeUnit;
use vortex_array::extension::datetime::Timestamp;
use vortex_array::scalar::Scalar;
use vortex_array::test_harness::fusion::fusion_trace;
use vortex_datetime_parts::DateTimeParts;
use vortex_error::VortexResult;
use vortex_session::VortexSession;
use vortex_sparse::Sparse;

#[test]
fn sliced_sparse_days() -> VortexResult<()> {
    let session = VortexSession::empty();
    vortex_datetime_parts::initialize(&session);
    vortex_sparse::initialize(&session);
    let mut ctx = session.create_execution_ctx();
    let days = Sparse::try_new(
        PrimitiveArray::from_iter([17u16, 1023, 1500, 2800]).into_array(),
        PrimitiveArray::from_option_iter([Some(-2i16), None, Some(3), Some(4)]).into_array(),
        3000,
        Scalar::null_native::<i16>(),
        &mut ctx,
    )?
    .into_array();
    let dtype =
        DType::Extension(Timestamp::new(TimeUnit::Seconds, days.dtype().nullability()).erased());
    let timestamps = DateTimeParts::try_new(
        dtype,
        days,
        ConstantArray::new(Scalar::from(3600i32), 3000).into_array(),
        ConstantArray::new(Scalar::from(0i32), 3000).into_array(),
    )?;
    let trace = fusion_trace(timestamps.into_array().slice(17..2801)?, &session)?;
    assert!(trace.contains("fuse start array=vortex.sparse"), "{trace}");
    Ok(())
}
