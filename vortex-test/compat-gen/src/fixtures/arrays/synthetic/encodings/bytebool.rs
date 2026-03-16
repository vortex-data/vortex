// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex::array::ArrayRef;
use vortex::array::IntoArray;
use vortex::array::arrays::BoolArray;
use vortex::array::arrays::StructArray;
use vortex::array::dtype::FieldNames;
use vortex::array::validity::Validity;
use vortex::array::vtable::ArrayId;
use vortex::encodings::bytebool::ByteBool;
use vortex::encodings::bytebool::ByteBoolArray;
use vortex::error::VortexResult;

use super::N;
use crate::fixtures::ArrayFixture;

pub struct ByteBoolFixture;

impl ArrayFixture for ByteBoolFixture {
    fn name(&self) -> &str {
        "bytebool.vortex"
    }

    fn description(&self) -> &str {
        "Boolean arrays for ByteBool encoding"
    }

    fn expected_encodings(&self) -> Vec<ArrayId> {
        vec![ByteBool::ID]
    }

    fn build(&self) -> VortexResult<ArrayRef> {
        let alternating: Vec<bool> = (0..N).map(|i| i % 2 == 0).collect();
        let mostly_true: Vec<bool> = (0..N).map(|i| i % 100 != 0).collect();
        let mixed: Vec<bool> = (0..N).map(|i| (i * 7 + 3) % 5 > 1).collect();
        let nullable_vals: Vec<bool> = (0..N).map(|i| i % 3 == 0).collect();
        let nullable_validity =
            Validity::from(BoolArray::from_iter((0..N).map(|i| i % 5 != 0)).to_bit_buffer());
        let all_false: Vec<bool> = vec![false; N];

        let arr = StructArray::try_new(
            FieldNames::from([
                "alternating",
                "mostly_true",
                "mixed",
                "nullable_bool",
                "all_false",
            ]),
            vec![
                ByteBoolArray::from_vec(alternating, Validity::NonNullable).into_array(),
                ByteBoolArray::from_vec(mostly_true, Validity::NonNullable).into_array(),
                ByteBoolArray::from_vec(mixed, Validity::NonNullable).into_array(),
                ByteBoolArray::from_vec(nullable_vals, nullable_validity).into_array(),
                ByteBoolArray::from_vec(all_false, Validity::NonNullable).into_array(),
            ],
            N,
            Validity::NonNullable,
        )?;
        Ok(arr.into_array())
    }
}
