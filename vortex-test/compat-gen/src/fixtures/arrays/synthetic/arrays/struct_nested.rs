// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::Struct;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinArray;
use vortex_array::dtype::FieldNames;
use vortex_array::validity::Validity;
use vortex_array::vtable::ArrayId;
use vortex_buffer::buffer;
use vortex_error::VortexResult;

use crate::fixtures::ArrayFixture;

pub struct StructNestedFixture;

impl ArrayFixture for StructNestedFixture {
    fn name(&self) -> &str {
        "struct_nested.vortex"
    }

    fn description(&self) -> &str {
        "Nested struct: outer struct containing an inner struct with primitive and string fields"
    }

    fn expected_encodings(&self) -> Vec<ArrayId> {
        vec![Struct::ID]
    }

    fn build(&self) -> VortexResult<ArrayRef> {
        let inner = StructArray::try_new(
            FieldNames::from(["a", "b"]),
            vec![
                PrimitiveArray::new(buffer![10i32, 20, 30], Validity::NonNullable).into_array(),
                VarBinArray::from(vec!["x", "y", "z"]).into_array(),
            ],
            3,
            Validity::NonNullable,
        )?;

        let arr = StructArray::try_new(
            FieldNames::from(["inner", "value"]),
            vec![
                inner.into_array(),
                PrimitiveArray::new(buffer![1.1f64, 2.2, 3.3], Validity::NonNullable).into_array(),
            ],
            3,
            Validity::NonNullable,
        )?;
        Ok(arr.into_array())
    }
}
