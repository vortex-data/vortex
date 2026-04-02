// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::List;
use vortex_array::arrays::ListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinArray;
use vortex_array::dtype::FieldNames;
use vortex_array::validity::Validity;
use vortex_buffer::buffer;
use vortex_error::VortexResult;

use crate::fixtures::FlatLayoutFixture;

pub struct ListFixture;

impl FlatLayoutFixture for ListFixture {
    fn name(&self) -> &str {
        "list.vortex"
    }

    fn description(&self) -> &str {
        "Variable-length list arrays with integer and string elements"
    }

    fn expected_encodings(&self) -> Vec<ArrayId> {
        vec![List::ID]
    }

    fn build(&self) -> VortexResult<ArrayRef> {
        // List of i32: [[1,2,3], [4,5], [6], [7,8,9,10]]
        let elements = PrimitiveArray::new(
            buffer![1i32, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            Validity::NonNullable,
        );
        let offsets = PrimitiveArray::new(buffer![0i64, 3, 5, 6, 10], Validity::NonNullable);
        let int_list = ListArray::try_new(
            elements.into_array(),
            offsets.into_array(),
            Validity::NonNullable,
        )?;

        // List of strings: [["a","b"], ["hello"], [], ["x","y","z"]]
        let str_elements = VarBinArray::from_strs(vec!["a", "b", "hello", "x", "y", "z"]);
        let str_offsets = PrimitiveArray::new(buffer![0i64, 2, 3, 3, 6], Validity::NonNullable);
        let str_list = ListArray::try_new(
            str_elements.into_array(),
            str_offsets.into_array(),
            Validity::NonNullable,
        )?;

        // Nullable list of i32: [[100,200], null, [], [300]]
        let nullable_elements =
            PrimitiveArray::new(buffer![100i32, 200, 300], Validity::NonNullable);
        let nullable_offsets =
            PrimitiveArray::new(buffer![0i64, 2, 2, 2, 3], Validity::NonNullable);
        let nullable_int_list = ListArray::try_new(
            nullable_elements.into_array(),
            nullable_offsets.into_array(),
            Validity::from_iter([true, false, true, true]),
        )?;

        let arr = StructArray::try_new(
            FieldNames::from(["int_list", "str_list", "nullable_int_list"]),
            vec![
                int_list.into_array(),
                str_list.into_array(),
                nullable_int_list.into_array(),
            ],
            4,
            Validity::NonNullable,
        )?;
        Ok(arr.into_array())
    }
}
