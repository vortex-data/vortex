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
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_array::validity::Validity;
use vortex_error::VortexResult;

use crate::RLE;
use crate::RLEData;

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
fn random_access_across_chunks_and_nulls(
    #[values(false, true)] repeated: bool,
    #[values(false, true)] sliced: bool,
) -> VortexResult<()> {
    let mut ctx = crate::test::SESSION.create_execution_ctx();
    let input =
        PrimitiveArray::from_option_iter((0..8192u32).map(|i| (i % 11 != 0).then_some(i / 16)));
    let encoded = RLEData::encode(input.as_view(), &mut ctx)?.into_array();
    let range = if sliced { 1777..7333 } else { 0..8192 };
    let source = if sliced {
        encoded
            .slice(range.clone())?
            .execute::<ArrayRef>(&mut ctx)?
    } else {
        encoded
    };
    assert!(source.is::<RLE>());
    let input = input.slice(range)?;
    let indices = [0u32, 1, 1023, 1024, 2048, 2047, 4097, 11, 33, 17, 0];
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
fn lazy_validity_does_not_evaluate_unrequested_rows(
    #[values(false, true)] repeated: bool,
) -> VortexResult<()> {
    let mut ctx = crate::test::SESSION.create_execution_ctx();
    let numerators = PrimitiveArray::from_iter(vec![1u32; 1024]).into_array();
    let denominators =
        PrimitiveArray::from_iter((0..1024).map(|i| u32::from(i != 1023))).into_array();
    let validity = numerators
        .binary(denominators, Operator::Div)?
        .binary(numerators, Operator::Eq)?;
    let array = RLE::try_new(
        PrimitiveArray::from_iter([42u32]).into_array(),
        PrimitiveArray::new(vec![0u16; 1024], Validity::Array(validity)).into_array(),
        PrimitiveArray::from_iter([0u64]).into_array(),
        0,
        1024,
    )?
    .into_array();
    let expected = array.execute_scalar(0, &mut ctx)?;
    let mut retained = None;
    let mut probe = probe_for(&array, &mut retained, repeated);
    assert_eq!(probe.execute_scalar(0, &mut ctx)?, expected);
    assert!(probe.execute_scalar(1023, &mut ctx).is_err());
    Ok(())
}
