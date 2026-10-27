// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![cfg(test)]

use arbitrary::Arbitrary;
use arbitrary::Unstructured;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::scalar::Scalar;
use vortex_fuzz::Action;
use vortex_fuzz::ExpectedValue;
use vortex_fuzz::FuzzArrayAction;
use vortex_fuzz::SESSION;
use vortex_fuzz::run_fuzz_action;

#[test]
fn generated_casts_update_following_oracles() {
    // U8 [7, 9], then cast to U16, cast to U32, and read row 1.
    let seed = [1, 0, 0, 0, 2, 0, 7, 9, 2, 6, 1, 1, 0, 6, 1, 2, 0, 11, 0, 1];
    let fuzz = FuzzArrayAction::arbitrary(&mut Unstructured::new(&seed))
        .expect("infallible cast actions must be generated");
    let [
        (Action::Cast(first_dtype), ExpectedValue::Array(first)),
        (Action::Cast(second_dtype), ExpectedValue::Array(second)),
        (Action::ScalarAt(indices), ExpectedValue::ScalarVec(scalars)),
    ] = fuzz.actions.as_slice()
    else {
        panic!("seed must generate two casts followed by scalar lookup");
    };
    assert_eq!(
        *first_dtype,
        DType::Primitive(PType::U16, Nullability::NonNullable)
    );
    assert_eq!(
        *second_dtype,
        DType::Primitive(PType::U32, Nullability::NonNullable)
    );
    let mut ctx = SESSION.create_execution_ctx();
    assert_arrays_eq!(first, PrimitiveArray::from_iter([7u16, 9]), &mut ctx);
    assert_arrays_eq!(second, PrimitiveArray::from_iter([7u32, 9]), &mut ctx);
    assert_eq!(indices, &[1]);
    assert_eq!(scalars, &[Scalar::from(9u32)]);
    assert_eq!(scalars[0].dtype(), second_dtype);
    assert!(run_fuzz_action(fuzz).expect("generated cast sequence must execute"));
}

#[test]
fn nullable_cast_removal_is_rejected_by_generator() {
    // Nullable U8 [null, null], then cast to nonnullable U16.
    let seed = [1, 0, 1, 0, 2, 0, 7, 9, 1, 0, 6, 1, 1, 0];
    assert!(matches!(
        FuzzArrayAction::arbitrary(&mut Unstructured::new(&seed)),
        Err(arbitrary::Error::EmptyChoose)
    ));
}
