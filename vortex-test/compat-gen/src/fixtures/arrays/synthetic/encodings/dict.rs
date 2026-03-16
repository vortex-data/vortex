// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex::array::ArrayRef;
use vortex::array::IntoArray;
use vortex::array::arrays::Dict;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::StructArray;
use vortex::array::arrays::VarBinArray;
use vortex::array::builders::dict::dict_encode;
use vortex::array::dtype::FieldNames;
use vortex::array::validity::Validity;
use vortex::array::vtable::ArrayId;
use vortex::buffer::Buffer;
use vortex::error::VortexResult;

use super::N;
use crate::fixtures::FlatLayoutFixture;

pub struct DictFixture;

impl FlatLayoutFixture for DictFixture {
    fn name(&self) -> &str {
        "dict.vortex"
    }

    fn description(&self) -> &str {
        "Low-cardinality repeated values (strings and integers) for Dict encoding"
    }

    fn expected_encodings(&self) -> Vec<ArrayId> {
        vec![Dict::ID]
    }

    fn build(&self) -> VortexResult<ArrayRef> {
        let categories = ["red", "green", "blue", "yellow", "purple"];
        let str_values: Vec<&str> = (0..N).map(|i| categories[i % categories.len()]).collect();
        let str_col = VarBinArray::from(str_values);

        let int_values: Vec<i32> = (0..N as i32).map(|i| (i % 10) * 100).collect();
        let int_col = PrimitiveArray::new(Buffer::from(int_values), Validity::NonNullable);

        let nullable_values: Vec<Option<&str>> = (0..N)
            .map(|i| (i % 7 != 0).then_some(categories[i % categories.len()]))
            .collect();
        let nullable_col = VarBinArray::from(nullable_values);

        let single_val: Vec<&str> = (0..N).map(|_| "only_value").collect();
        let single_col = VarBinArray::from(single_val);

        let bool_cat: Vec<&str> = (0..N)
            .map(|i| if i % 3 == 0 { "yes" } else { "no" })
            .collect();
        let bool_cat_col = VarBinArray::from(bool_cat);

        let arr = StructArray::try_new(
            FieldNames::from([
                "str_cat",
                "int_cat",
                "nullable_cat",
                "single_cat",
                "bool_cat",
            ]),
            vec![
                dict_encode(&str_col.into_array())?.into_array(),
                dict_encode(&int_col.into_array())?.into_array(),
                dict_encode(&nullable_col.into_array())?.into_array(),
                dict_encode(&single_col.into_array())?.into_array(),
                dict_encode(&bool_cat_col.into_array())?.into_array(),
            ],
            N,
            Validity::NonNullable,
        )?;
        Ok(arr.into_array())
    }
}
