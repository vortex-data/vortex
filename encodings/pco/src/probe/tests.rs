// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use rstest::rstest;
use vortex_array::ArrayProbe;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::RepeatedArrayProbe;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::assert_arrays_eq;
use vortex_array::builders::builder_with_capacity_in;
use vortex_array::validity::Validity;
use vortex_error::VortexResult;

use crate::Pco;

/// A one-off or retained probe over `array`, the retained one living in `retained`.
fn probe_for<'a>(
    array: &'a ArrayRef,
    retained: &'a mut Option<RepeatedArrayProbe>,
    repeated: bool,
) -> ArrayProbe<'a> {
    if repeated {
        retained.insert(array.repeated_probe()).as_probe()
    } else {
        array.probe()
    }
}

#[rstest]
fn random_access(
    #[values(false, true)] repeated: bool,
    #[values(false, true)] nullable: bool,
    #[values(false, true)] sliced: bool,
) -> VortexResult<()> {
    let mut ctx = vortex_array::array_session().create_execution_ctx();
    let input = if nullable {
        PrimitiveArray::from_option_iter((0..4096i32).map(|i| (i % 7 != 0).then_some(i * 19)))
    } else {
        PrimitiveArray::from_iter((0..4096i32).map(|i| i * 19))
    };
    let encoded = Pco::from_primitive(input.as_view(), 3, 128, &mut ctx)?.into_array();
    let range = if sliced { 777..3333 } else { 0..4096 };
    let source = encoded.slice(range.clone())?;
    assert!(source.is::<Pco>());
    let input = input.slice(range)?;
    let indices = [0u32, 1, 127, 128, 512, 2048, 129, 2, 1024, 0];
    let mut actual = builder_with_capacity_in(source.dtype(), indices.len(), ctx.allocator());
    let mut retained = None;
    let mut probe = probe_for(&source, &mut retained, repeated);
    for index in indices {
        actual.append_scalar(&probe.execute_scalar(index as usize, &mut ctx)?)?;
    }
    assert_arrays_eq!(
        actual.finish(),
        input.take(PrimitiveArray::from_iter(indices).into_array())?,
        &mut ctx
    );
    assert!(probe.execute_scalar(source.len(), &mut ctx).is_err());
    Ok(())
}

#[rstest]
#[case(PrimitiveArray::from_iter([1u16, 9, 32768, 65535]))]
#[case(PrimitiveArray::from_iter([i64::MIN, -1, 0, i64::MAX]))]
#[case(PrimitiveArray::from_iter([1.25f64, -2.5, 0.0, f64::INFINITY]))]
fn preserves_physical_type(#[case] input: PrimitiveArray) -> VortexResult<()> {
    let mut ctx = vortex_array::array_session().create_execution_ctx();
    let encoded = Pco::from_primitive(input.as_view(), 3, 2, &mut ctx)?.into_array();
    let mut probe = encoded.repeated_probe();
    let mut actual = builder_with_capacity_in(input.dtype(), input.len(), ctx.allocator());
    for i in 0..input.len() {
        actual.append_scalar(&probe.execute_scalar(i, &mut ctx)?)?;
    }
    assert_arrays_eq!(actual.finish(), input, &mut ctx);
    Ok(())
}

#[test]
fn all_null_access_returns_null() -> VortexResult<()> {
    let mut ctx = vortex_array::array_session().create_execution_ctx();
    let input = PrimitiveArray::new(vec![0i32; 128], Validity::AllInvalid);
    let encoded = Pco::from_primitive(input.as_view(), 3, 128, &mut ctx)?;
    let encoded = encoded.into_array();
    let mut probe = encoded.repeated_probe();
    assert!(probe.execute_scalar(42, &mut ctx)?.is_null());
    Ok(())
}

#[test]
fn probe_outlives_source_and_moves() -> VortexResult<()> {
    let mut ctx = vortex_array::array_session().create_execution_ctx();
    let input = PrimitiveArray::from_iter(0..4096u32);
    let array = Pco::from_primitive(input.as_view(), 3, 512, &mut ctx)?.into_array();
    let probe = RepeatedArrayProbe::new(array.clone());
    drop(array);
    let mut moved = (probe, ());
    for index in [1, 5, 127, 255, 511, 1023, 0, 1, 4095] {
        assert_eq!(
            moved.0.execute_scalar(index, &mut ctx)?,
            u32::try_from(index)?.into()
        );
    }
    assert!(moved.0.execute_scalar(4096, &mut ctx).is_err());
    Ok(())
}
