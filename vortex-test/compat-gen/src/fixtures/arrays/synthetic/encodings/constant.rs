// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex::array::ArrayRef;
use vortex::array::IntoArray;
use vortex::array::arrays::Constant;
use vortex::array::arrays::ConstantArray;
use vortex::array::arrays::StructArray;
use vortex::array::dtype::FieldNames;
use vortex::array::validity::Validity;
use vortex::array::vtable::ArrayId;
use vortex::error::VortexResult;

use super::N;
use crate::fixtures::FlatLayoutFixture;

pub struct ConstantFixture;

impl FlatLayoutFixture for ConstantFixture {
    fn name(&self) -> &str {
        "constant.vortex"
    }

    fn description(&self) -> &str {
        "Constant-value columns (int, float, string, bool, null) for Constant encoding"
    }

    fn expected_encodings(&self) -> Vec<ArrayId> {
        vec![Constant::ID]
    }

    fn build(&self) -> VortexResult<ArrayRef> {
        let const_i32 = ConstantArray::new(42i32, N);
        let const_f64 = ConstantArray::new(99.99f64, N);
        let const_bool = ConstantArray::new(true, N);
        let const_str = ConstantArray::new("constant_value", N);
        let const_zero = ConstantArray::new(0u64, N);
        let const_neg = ConstantArray::new(-1i64, N);

        let arr = StructArray::try_new(
            FieldNames::from([
                "const_i32",
                "const_f64",
                "const_bool",
                "const_str",
                "const_zero",
                "const_neg",
            ]),
            vec![
                const_i32.into_array(),
                const_f64.into_array(),
                const_bool.into_array(),
                const_str.into_array(),
                const_zero.into_array(),
                const_neg.into_array(),
            ],
            N,
            Validity::NonNullable,
        )?;
        Ok(arr.into_array())
    }
}
