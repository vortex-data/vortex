// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex::array::ArrayId;
use vortex::array::ArrayRef;
use vortex::array::IntoArray;
use vortex::array::arrays::Dict;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::StructArray;
use vortex::array::arrays::VarBinArray;
use vortex::array::builders::dict::dict_encode;
use vortex::array::dtype::FieldNames;
use vortex::array::validity::Validity;
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
        let str_col = VarBinArray::from_strs(str_values);

        let int_col: PrimitiveArray = (0..N as i32).map(|i| (i % 10) * 100).collect();

        let nullable_values: Vec<Option<&str>> = (0..N)
            .map(|i| (i % 7 != 0).then_some(categories[i % categories.len()]))
            .collect();
        let nullable_col = VarBinArray::from_nullable_strs(nullable_values);

        let single_val: Vec<&str> = (0..N).map(|_| "only_value").collect();
        let single_col = VarBinArray::from_strs(single_val);

        let bool_cat: Vec<&str> = (0..N)
            .map(|i| if i % 3 == 0 { "yes" } else { "no" })
            .collect();
        let bool_cat_col = VarBinArray::from_strs(bool_cat);
        let all_null_col =
            VarBinArray::from_nullable_strs((0..N).map(|_| None::<&str>).collect::<Vec<_>>());
        let single_non_null_col = VarBinArray::from_nullable_strs(
            (0..N)
                .map(|i| (i == N / 2).then_some("lonely"))
                .collect::<Vec<_>>(),
        );
        let threshold_255_values: Vec<String> =
            (0..N).map(|i| format!("u255-{}", i % 255)).collect();
        let threshold_255_refs: Vec<&str> =
            threshold_255_values.iter().map(String::as_str).collect();
        let threshold_255_col = VarBinArray::from_strs(threshold_255_refs);
        let threshold_256_values: Vec<String> =
            (0..N).map(|i| format!("u256-{}", i % 256)).collect();
        let threshold_256_refs: Vec<&str> =
            threshold_256_values.iter().map(String::as_str).collect();
        let threshold_256_col = VarBinArray::from_strs(threshold_256_refs);
        let threshold_257_values: Vec<String> =
            (0..N).map(|i| format!("u257-{}", i % 257)).collect();
        let threshold_257_refs: Vec<&str> =
            threshold_257_values.iter().map(String::as_str).collect();
        let threshold_257_col = VarBinArray::from_strs(threshold_257_refs);
        let long_values: Vec<String> = (0..N)
            .map(|i| format!("long-dict-value-{i:04}-{:08x}-suffix", i * 17))
            .collect();
        let long_refs: Vec<&str> = long_values.iter().map(String::as_str).collect();
        let long_col = VarBinArray::from_strs(long_refs);
        let insertion_values = ["late", "first", "middle", "early", "last"];
        let insertion_ordered: Vec<&str> = (0..N)
            .map(|i| insertion_values[(i * 7 + 3) % insertion_values.len()])
            .collect();
        let insertion_ordered_col = VarBinArray::from_strs(insertion_ordered);

        let arr = StructArray::try_new(
            FieldNames::from([
                "str_cat",
                "int_cat",
                "nullable_cat",
                "single_cat",
                "bool_cat",
                "all_null",
                "single_non_null",
                "threshold_255",
                "threshold_256",
                "threshold_257",
                "long_values",
                "insertion_ordered",
            ]),
            vec![
                dict_encode(&str_col.into_array())?.into_array(),
                dict_encode(&int_col.into_array())?.into_array(),
                dict_encode(&nullable_col.into_array())?.into_array(),
                dict_encode(&single_col.into_array())?.into_array(),
                dict_encode(&bool_cat_col.into_array())?.into_array(),
                dict_encode(&all_null_col.into_array())?.into_array(),
                dict_encode(&single_non_null_col.into_array())?.into_array(),
                dict_encode(&threshold_255_col.into_array())?.into_array(),
                dict_encode(&threshold_256_col.into_array())?.into_array(),
                dict_encode(&threshold_257_col.into_array())?.into_array(),
                dict_encode(&long_col.into_array())?.into_array(),
                dict_encode(&insertion_ordered_col.into_array())?.into_array(),
            ],
            N,
            Validity::NonNullable,
        )?;
        Ok(arr.into_array())
    }
}
