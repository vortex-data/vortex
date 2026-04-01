// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::Bool;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::StructArray;
use vortex_array::dtype::FieldNames;
use vortex_array::validity::Validity;
use vortex_error::VortexResult;

use crate::fixtures::FlatLayoutFixture;

pub struct BooleansFixture;

impl FlatLayoutFixture for BooleansFixture {
    fn name(&self) -> &str {
        "booleans.vortex"
    }

    fn description(&self) -> &str {
        "Boolean arrays with mixed true/false values including a nullable column"
    }

    fn expected_encodings(&self) -> Vec<ArrayId> {
        vec![Bool::ID]
    }

    fn build(&self) -> VortexResult<ArrayRef> {
        let bools = BoolArray::from_iter([true, false, true, true, false]);
        let nullable_bools =
            BoolArray::from_iter([Some(true), None, Some(false), None, Some(true)]);
        let all_true = BoolArray::from_iter([true, true, true, true, true]);
        let all_false = BoolArray::from_iter([false, false, false, false, false]);
        let all_null: BoolArray = BoolArray::from_iter([None::<bool>, None, None, None, None]);
        let arr = StructArray::try_new(
            FieldNames::from(["flag", "nullable_flag", "all_true", "all_false", "all_null"]),
            vec![
                bools.into_array(),
                nullable_bools.into_array(),
                all_true.into_array(),
                all_false.into_array(),
                all_null.into_array(),
            ],
            5,
            Validity::NonNullable,
        )?;
        Ok(arr.into_array())
    }
}
