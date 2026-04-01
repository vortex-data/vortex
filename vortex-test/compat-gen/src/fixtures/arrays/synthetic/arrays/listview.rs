// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::ListView;
use vortex_array::arrays::ListViewArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinArray;
use vortex_array::dtype::FieldNames;
use vortex_array::validity::Validity;
use vortex_buffer::buffer;
use vortex_error::VortexResult;

use crate::fixtures::FlatLayoutFixture;

pub struct ListViewFixture;

impl FlatLayoutFixture for ListViewFixture {
    fn name(&self) -> &str {
        "listview.vortex"
    }

    fn description(&self) -> &str {
        "ListView arrays with integer and string elements"
    }

    fn expected_encodings(&self) -> Vec<ArrayId> {
        vec![ListView::ID]
    }

    fn build(&self) -> VortexResult<ArrayRef> {
        // ListView of i32: [[1,2,3], [4,5], [6], [7,8,9,10]]
        let int_elements = PrimitiveArray::new(
            buffer![1i32, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            Validity::NonNullable,
        );
        let int_offsets = PrimitiveArray::new(buffer![0u32, 3, 5, 6], Validity::NonNullable);
        let int_sizes = PrimitiveArray::new(buffer![3u32, 2, 1, 4], Validity::NonNullable);
        let int_listview = ListViewArray::try_new(
            int_elements.into_array(),
            int_offsets.into_array(),
            int_sizes.into_array(),
            Validity::NonNullable,
        )?;

        // ListView of strings: [["a","b"], ["hello"], [], ["x","y","z"]]
        let str_elements = VarBinArray::from_strs(vec!["a", "b", "hello", "x", "y", "z"]);
        let str_offsets = PrimitiveArray::new(buffer![0u32, 2, 3, 3], Validity::NonNullable);
        let str_sizes = PrimitiveArray::new(buffer![2u32, 1, 0, 3], Validity::NonNullable);
        let str_listview = ListViewArray::try_new(
            str_elements.into_array(),
            str_offsets.into_array(),
            str_sizes.into_array(),
            Validity::NonNullable,
        )?;

        // Nullable ListView of i32: [[10,20], null, [30], [40,50,60]]
        let nullable_elements =
            PrimitiveArray::new(buffer![10i32, 20, 30, 40, 50, 60], Validity::NonNullable);
        let nullable_offsets = PrimitiveArray::new(buffer![0u32, 2, 2, 3], Validity::NonNullable);
        let nullable_sizes = PrimitiveArray::new(buffer![2u32, 0, 1, 3], Validity::NonNullable);
        let nullable_listview = ListViewArray::try_new(
            nullable_elements.into_array(),
            nullable_offsets.into_array(),
            nullable_sizes.into_array(),
            Validity::from_iter([true, false, true, true]),
        )?;

        let arr = StructArray::try_new(
            FieldNames::from(["int_listview", "str_listview", "nullable_int_listview"]),
            vec![
                int_listview.into_array(),
                str_listview.into_array(),
                nullable_listview.into_array(),
            ],
            4,
            Validity::NonNullable,
        )?;
        Ok(arr.into_array())
    }
}
