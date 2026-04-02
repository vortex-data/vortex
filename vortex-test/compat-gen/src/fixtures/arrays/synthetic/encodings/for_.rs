// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex::array::ArrayId;
use vortex::array::ArrayRef;
use vortex::array::IntoArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::StructArray;
use vortex::array::dtype::FieldNames;
use vortex::array::validity::Validity;
use vortex::encodings::fastlanes::FoR;
use vortex::error::VortexResult;

use super::N;
use crate::fixtures::FlatLayoutFixture;

pub struct FoRFixture;

impl FlatLayoutFixture for FoRFixture {
    fn name(&self) -> &str {
        "for.vortex"
    }

    fn description(&self) -> &str {
        "Integers clustered around a base value for Frame-of-Reference encoding"
    }

    fn expected_encodings(&self) -> Vec<ArrayId> {
        vec![FoR::ID]
    }

    fn build(&self) -> VortexResult<ArrayRef> {
        let clustered_i32: PrimitiveArray = (0..N as i32).map(|i| 1_000_000 + (i % 100)).collect();
        let clustered_u64: PrimitiveArray =
            (0..N as u64).map(|i| 10_000_000_000 + (i % 256)).collect();
        let clustered_i64: PrimitiveArray =
            (0..N as i64).map(|i| 1_704_067_200 + (i % 3600)).collect();
        let negative_i32: PrimitiveArray = (0..N as i32).map(|i| -1_000_000 + (i % 50)).collect();
        let nullable_i32 = PrimitiveArray::from_option_iter(
            (0..N as i32).map(|i| (i % 11 != 0).then_some(500_000 + (i % 200))),
        );
        let clustered_i16: PrimitiveArray = (0..N as i16).map(|i| 10000 + (i % 30)).collect();
        let constant_offsets: PrimitiveArray = std::iter::repeat_n(123_456i32, N).collect();
        let zero_crossing_i32: PrimitiveArray = (0..N as i32).map(|i| -512 + (i % 1024)).collect();
        let far_outlier_i64: PrimitiveArray = (0..N as i64)
            .map(|i| {
                if i == 0 {
                    9_000_000_000
                } else {
                    1_000_000 + (i % 8)
                }
            })
            .collect();
        let near_max_u64: PrimitiveArray =
            (0..N as u64).map(|i| u64::MAX - 2048 + (i % 512)).collect();

        let arr = StructArray::try_new(
            FieldNames::from([
                "clustered_i32",
                "clustered_u64",
                "clustered_i64",
                "negative_i32",
                "nullable_i32",
                "clustered_i16",
                "constant_offsets",
                "zero_crossing_i32",
                "far_outlier_i64",
                "near_max_u64",
            ]),
            vec![
                FoR::encode(clustered_i32)?.into_array(),
                FoR::encode(clustered_u64)?.into_array(),
                FoR::encode(clustered_i64)?.into_array(),
                FoR::encode(negative_i32)?.into_array(),
                FoR::encode(nullable_i32)?.into_array(),
                FoR::encode(clustered_i16)?.into_array(),
                FoR::encode(constant_offsets)?.into_array(),
                FoR::encode(zero_crossing_i32)?.into_array(),
                FoR::encode(far_outlier_i64)?.into_array(),
                FoR::encode(near_max_u64)?.into_array(),
            ],
            N,
            Validity::NonNullable,
        )?;
        Ok(arr.into_array())
    }
}
