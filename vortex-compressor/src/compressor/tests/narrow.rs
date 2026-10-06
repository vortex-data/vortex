// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Integer narrowing precedes codec selection and preserves the input dtype.

use rstest::rstest;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::array_session;
use vortex_array::arrays::Constant;
use vortex_array::arrays::Narrow;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::narrow::NarrowArraySlotsExt;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::PType;
use vortex_buffer::buffer;
use vortex_error::VortexResult;

use crate::CascadingCompressor;

#[rstest]
#[case::disabled(false)]
#[case::enabled(true)]
fn integer_width(#[case] enabled: bool) -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let input = buffer![-128i64, 0, 127].into_array();
    let compressor = CascadingCompressor::new(vec![]).with_narrow_integers(enabled);
    let result = compressor.compress(&input, &mut ctx)?;

    assert_eq!(result.is::<Narrow>(), enabled);
    if enabled {
        assert_eq!(
            result.as_::<Narrow>().values().dtype().as_ptype(),
            PType::I8
        );
        assert!(result.nbytes() < input.nbytes());
    }
    assert_eq!(result.dtype(), input.dtype());
    assert_arrays_eq!(result, input, &mut ctx);

    Ok(())
}

#[rstest]
#[case::constant(PrimitiveArray::from_iter([127i64; 1024]))]
#[case::all_null(PrimitiveArray::from_option_iter([None::<i64>; 3]))]
fn constant_dtype(#[case] input: PrimitiveArray) -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let input = input.into_array();
    let compressor = CascadingCompressor::new(vec![]).with_narrow_integers(true);
    let result = compressor.compress(&input, &mut ctx)?;

    assert!(result.is::<Constant>());
    assert_eq!(result.dtype(), input.dtype());
    assert_arrays_eq!(result, input, &mut ctx);

    Ok(())
}

#[test]
fn full_width_values() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let input = buffer![i64::MIN, i64::MAX].into_array();
    let result = CascadingCompressor::new(vec![])
        .with_narrow_integers(true)
        .compress(&input, &mut ctx)?;
    assert!(!result.is::<Narrow>());
    assert_arrays_eq!(result, input, &mut ctx);

    Ok(())
}
