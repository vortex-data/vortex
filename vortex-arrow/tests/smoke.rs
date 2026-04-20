// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::IntoArray;
use vortex_array::LEGACY_SESSION;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_buffer::buffer;

#[test]
fn scalar_subtract_goes_through_arrow_hook() {
    vortex_arrow::init();

    let values = buffer![1u16, 2, 3].into_array();
    let one = Scalar::from(1u16);
    let constant = vortex_array::arrays::ConstantArray::new(one, values.len()).into_array();

    let result = values
        .binary(constant, Operator::Sub)
        .unwrap()
        .execute::<PrimitiveArray>(&mut LEGACY_SESSION.create_execution_ctx())
        .unwrap();

    assert_eq!(result.as_slice::<u16>(), &[0, 1, 2]);
}
