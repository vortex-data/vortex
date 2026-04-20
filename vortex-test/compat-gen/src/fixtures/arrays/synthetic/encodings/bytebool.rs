// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex::array::ArrayId;
use vortex::array::ArrayRef;
use vortex::array::ArrayVTable;
use vortex::array::IntoArray;
use vortex::array::arrays::BoolArray;
use vortex::array::arrays::StructArray;
use vortex::array::arrays::bool::BoolArrayExt;
use vortex::array::dtype::FieldNames;
use vortex::array::validity::Validity;
use vortex::encodings::bytebool::ByteBool;
use vortex::error::VortexResult;

use super::N;
use crate::fixtures::FlatLayoutFixture;

pub struct ByteBoolFixture;

impl FlatLayoutFixture for ByteBoolFixture {
    fn name(&self) -> &str {
        "bytebool.vortex"
    }

    fn description(&self) -> &str {
        "Boolean arrays for ByteBool encoding"
    }

    fn expected_encodings(&self) -> Vec<ArrayId> {
        vec![ByteBool.id()]
    }

    fn build(&self) -> VortexResult<ArrayRef> {
        let alternating: Vec<bool> = (0..N).map(|i| i % 2 == 0).collect();
        let mostly_true: Vec<bool> = (0..N).map(|i| i % 100 != 0).collect();
        let mixed: Vec<bool> = (0..N).map(|i| (i * 7 + 3) % 5 > 1).collect();
        let nullable_vals: Vec<bool> = (0..N).map(|i| i % 3 == 0).collect();
        let nullable_validity =
            Validity::from(BoolArray::from_iter((0..N).map(|i| i % 5 != 0)).to_bit_buffer());
        let all_false: Vec<bool> = vec![false; N];
        let all_true: Vec<bool> = vec![true; N];
        let all_null_vals: Vec<bool> = vec![false; N];
        let single_flip: Vec<bool> = (0..N).map(|i| i != N / 2).collect();
        let sparse_true: Vec<bool> = (0..N).map(|i| i % 127 == 0).collect();
        let edge_null_vals: Vec<bool> = (0..N).map(|i| i % 4 == 0).collect();
        let edge_null_validity = Validity::from(
            BoolArray::from_iter((0..N).map(|i| (8..N - 8).contains(&i))).to_bit_buffer(),
        );

        let arr = StructArray::try_new(
            FieldNames::from([
                "alternating",
                "mostly_true",
                "mixed",
                "nullable_bool",
                "all_false",
                "all_true",
                "all_null",
                "single_flip",
                "sparse_true",
                "edge_nulls",
            ]),
            vec![
                ByteBool::from_vec(alternating, Validity::NonNullable).into_array(),
                ByteBool::from_vec(mostly_true, Validity::NonNullable).into_array(),
                ByteBool::from_vec(mixed, Validity::NonNullable).into_array(),
                ByteBool::from_vec(nullable_vals, nullable_validity).into_array(),
                ByteBool::from_vec(all_false, Validity::NonNullable).into_array(),
                ByteBool::from_vec(all_true, Validity::NonNullable).into_array(),
                ByteBool::from_vec(all_null_vals, Validity::AllInvalid).into_array(),
                ByteBool::from_vec(single_flip, Validity::NonNullable).into_array(),
                ByteBool::from_vec(sparse_true, Validity::NonNullable).into_array(),
                ByteBool::from_vec(edge_null_vals, edge_null_validity).into_array(),
            ],
            N,
            Validity::NonNullable,
        )?;
        Ok(arr.into_array())
    }
}
