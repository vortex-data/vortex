// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex::array::ArrayId;
use vortex::array::ArrayRef;
use vortex::array::IntoArray;
use vortex::array::arrays::StructArray;
use vortex::array::dtype::FieldNames;
use vortex::array::dtype::Nullability;
use vortex::array::validity::Validity;
use vortex::encodings::sequence::Sequence;
use vortex::error::VortexResult;

use super::N;
use crate::fixtures::FlatLayoutFixture;

pub struct SequenceFixture;

impl FlatLayoutFixture for SequenceFixture {
    fn name(&self) -> &str {
        "sequence.vortex"
    }

    fn description(&self) -> &str {
        "Arithmetic sequences (0,1,2,... and stepped) for Sequence encoding, including nullable"
    }

    fn expected_encodings(&self) -> Vec<ArrayId> {
        vec![Sequence::ID]
    }

    fn build(&self) -> VortexResult<ArrayRef> {
        let row_ids = Sequence::try_new_typed::<u64>(0, 1, Nullability::NonNullable, N)?;
        let stepped = Sequence::try_new_typed::<i32>(0, 5, Nullability::NonNullable, N)?;
        let offset = Sequence::try_new_typed::<i64>(1000, 1, Nullability::NonNullable, N)?;
        let decreasing = Sequence::try_new_typed::<i64>(10000, -3, Nullability::NonNullable, N)?;
        let large_step = Sequence::try_new_typed::<u32>(0, 1000, Nullability::NonNullable, N)?;
        let zero_step = Sequence::try_new_typed::<i32>(7, 0, Nullability::NonNullable, N)?;
        let zero_crossing = Sequence::try_new_typed::<i32>(-512, 1, Nullability::NonNullable, N)?;
        let near_overflow =
            Sequence::try_new_typed::<u64>(u64::MAX - N as u64, 1, Nullability::NonNullable, N)?;
        let small_negative_i16 =
            Sequence::try_new_typed::<i16>(1200, -2, Nullability::NonNullable, N)?;
        let nullable_i64 = Sequence::try_new_typed::<i64>(0, 2, Nullability::Nullable, N)?;
        let nullable_u32 = Sequence::try_new_typed::<u32>(100, 7, Nullability::Nullable, N)?;

        let arr = StructArray::try_new(
            FieldNames::from([
                "row_ids",
                "stepped",
                "offset",
                "decreasing",
                "large_step",
                "zero_step",
                "zero_crossing",
                "near_overflow",
                "small_negative_i16",
                "nullable_i64",
                "nullable_u32",
            ]),
            vec![
                row_ids.into_array(),
                stepped.into_array(),
                offset.into_array(),
                decreasing.into_array(),
                large_step.into_array(),
                zero_step.into_array(),
                zero_crossing.into_array(),
                near_overflow.into_array(),
                small_negative_i16.into_array(),
                nullable_i64.into_array(),
                nullable_u32.into_array(),
            ],
            N,
            Validity::NonNullable,
        )?;
        Ok(arr.into_array())
    }
}
