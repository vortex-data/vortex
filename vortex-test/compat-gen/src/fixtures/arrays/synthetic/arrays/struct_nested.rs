// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::Struct;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinArray;
use vortex_array::dtype::FieldNames;
use vortex_array::validity::Validity;
use vortex_buffer::buffer;
use vortex_error::VortexResult;

use crate::fixtures::FlatLayoutFixture;

pub struct StructNestedFixture;

impl FlatLayoutFixture for StructNestedFixture {
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
                VarBinArray::from_strs(vec!["x", "y", "z"]).into_array(),
            ],
            3,
            Validity::NonNullable,
        )?;

        // Nullable inner struct: second row is null at the struct level.
        let nullable_inner = StructArray::try_new(
            FieldNames::from(["c", "d"]),
            vec![
                PrimitiveArray::new(buffer![100i64, 0, 300], Validity::NonNullable).into_array(),
                PrimitiveArray::new(buffer![1.0f32, 0.0, 3.0], Validity::NonNullable).into_array(),
            ],
            3,
            Validity::from_iter([true, false, true]),
        )?;

        let arr = StructArray::try_new(
            FieldNames::from(["inner", "nullable_inner", "value"]),
            vec![
                inner.into_array(),
                nullable_inner.into_array(),
                PrimitiveArray::new(buffer![1.1f64, 2.2, 3.3], Validity::NonNullable).into_array(),
            ],
            3,
            Validity::NonNullable,
        )?;
        Ok(arr.into_array())
    }
}
