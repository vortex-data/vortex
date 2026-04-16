// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex::array::ArrayId;
use vortex::array::ArrayRef;
use vortex::array::ArrayVTable;
use vortex::array::IntoArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::StructArray;
use vortex::array::dtype::FieldNames;
use vortex::array::validity::Validity;
use vortex::encodings::zigzag::ZigZag;
use vortex::encodings::zigzag::zigzag_encode;
use vortex::error::VortexResult;

use super::N;
use crate::fixtures::FlatLayoutFixture;

pub struct ZigZagFixture;

impl FlatLayoutFixture for ZigZagFixture {
    fn name(&self) -> &str {
        "zigzag.vortex"
    }

    fn description(&self) -> &str {
        "Signed integers with small absolute values for ZigZag encoding"
    }

    fn expected_encodings(&self) -> Vec<ArrayId> {
        vec![ZigZag.id()]
    }

    fn build(&self) -> VortexResult<ArrayRef> {
        let alternating_i32: PrimitiveArray = (0..N as i32)
            .map(|i| {
                let v = i / 2 + 1;
                if i % 2 == 0 { v } else { -v }
            })
            .collect();
        let small_i64: PrimitiveArray = (0..N as i64).map(|i| (i % 21) - 10).collect();
        let deltas_i32: PrimitiveArray = (0..N as i32).map(|i| -(i % 50)).collect();
        let small_i16: PrimitiveArray = (0..N as i16).map(|i| (i % 11) - 5).collect();
        let small_i8: PrimitiveArray = (0..N).map(|i| ((i % 9) as i8) - 4).collect();
        let nullable_zigzag = PrimitiveArray::from_option_iter(
            (0..N as i32).map(|i| (i % 6 != 0).then_some((i % 15) - 7)),
        );
        let extremes_i32: PrimitiveArray = (0..N)
            .map(|i| match i % 4 {
                0 => i32::MIN,
                1 => i32::MAX,
                2 => -1,
                _ => 1,
            })
            .collect();
        let zero_heavy_outliers: PrimitiveArray = (0..N as i64)
            .map(|i| if i % 257 == 0 { i64::MAX / 1024 - i } else { 0 })
            .collect();
        let repeated_negative: PrimitiveArray = std::iter::repeat_n(-42i32, N).collect();
        let zero_crossing: PrimitiveArray = (0..N as i32).map(|i| -512 + (i % 1024)).collect();
        let head_tail_nulls = PrimitiveArray::from_option_iter((0..N as i32).map(|i| {
            if i < 8 || i >= N as i32 - 8 {
                None
            } else {
                Some((i % 21) - 10)
            }
        }));

        let arr = StructArray::try_new(
            FieldNames::from([
                "alternating_i32",
                "small_i64",
                "deltas_i32",
                "small_i16",
                "small_i8",
                "nullable_zigzag",
                "extremes_i32",
                "zero_heavy_outliers",
                "repeated_negative",
                "zero_crossing",
                "head_tail_nulls",
            ]),
            vec![
                zigzag_encode(alternating_i32.as_view())?.into_array(),
                zigzag_encode(small_i64.as_view())?.into_array(),
                zigzag_encode(deltas_i32.as_view())?.into_array(),
                zigzag_encode(small_i16.as_view())?.into_array(),
                zigzag_encode(small_i8.as_view())?.into_array(),
                zigzag_encode(nullable_zigzag.as_view())?.into_array(),
                zigzag_encode(extremes_i32.as_view())?.into_array(),
                zigzag_encode(zero_heavy_outliers.as_view())?.into_array(),
                zigzag_encode(repeated_negative.as_view())?.into_array(),
                zigzag_encode(zero_crossing.as_view())?.into_array(),
                zigzag_encode(head_tail_nulls.as_view())?.into_array(),
            ],
            N,
            Validity::NonNullable,
        )?;
        Ok(arr.into_array())
    }
}
