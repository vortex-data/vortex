// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex::array::ArrayRef;
use vortex::array::IntoArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::StructArray;
use vortex::array::dtype::FieldNames;
use vortex::array::validity::Validity;
use vortex::array::vtable::ArrayId;
use vortex::buffer::Buffer;
use vortex::encodings::fastlanes::FoR;
use vortex::encodings::fastlanes::FoRArray;
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
        let clustered_i32: Vec<i32> = (0..N as i32).map(|i| 1_000_000 + (i % 100)).collect();
        let clustered_u64: Vec<u64> = (0..N as u64).map(|i| 10_000_000_000 + (i % 256)).collect();
        let clustered_i64: Vec<i64> = (0..N as i64).map(|i| 1_704_067_200 + (i % 3600)).collect();
        let negative_i32: Vec<i32> = (0..N as i32).map(|i| -1_000_000 + (i % 50)).collect();
        let nullable_i32 = PrimitiveArray::from_option_iter(
            (0..N as i32).map(|i| (i % 11 != 0).then_some(500_000 + (i % 200))),
        );
        let clustered_i16: Vec<i16> = (0..N as i16).map(|i| 10000 + (i % 30)).collect();
        let constant_offsets: Vec<i32> = vec![123_456; N];
        let zero_crossing_i32: Vec<i32> = (0..N as i32).map(|i| -512 + (i % 1024)).collect();
        let far_outlier_i64: Vec<i64> = (0..N as i64)
            .map(|i| {
                if i == 0 {
                    9_000_000_000
                } else {
                    1_000_000 + (i % 8)
                }
            })
            .collect();
        let near_max_u64: Vec<u64> = (0..N as u64).map(|i| u64::MAX - 2048 + (i % 512)).collect();

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
                FoRArray::encode(PrimitiveArray::new(
                    Buffer::from(clustered_i32),
                    Validity::NonNullable,
                ))?
                .into_array(),
                FoRArray::encode(PrimitiveArray::new(
                    Buffer::from(clustered_u64),
                    Validity::NonNullable,
                ))?
                .into_array(),
                FoRArray::encode(PrimitiveArray::new(
                    Buffer::from(clustered_i64),
                    Validity::NonNullable,
                ))?
                .into_array(),
                FoRArray::encode(PrimitiveArray::new(
                    Buffer::from(negative_i32),
                    Validity::NonNullable,
                ))?
                .into_array(),
                FoRArray::encode(nullable_i32)?.into_array(),
                FoRArray::encode(PrimitiveArray::new(
                    Buffer::from(clustered_i16),
                    Validity::NonNullable,
                ))?
                .into_array(),
                FoRArray::encode(PrimitiveArray::new(
                    Buffer::from(constant_offsets),
                    Validity::NonNullable,
                ))?
                .into_array(),
                FoRArray::encode(PrimitiveArray::new(
                    Buffer::from(zero_crossing_i32),
                    Validity::NonNullable,
                ))?
                .into_array(),
                FoRArray::encode(PrimitiveArray::new(
                    Buffer::from(far_outlier_i64),
                    Validity::NonNullable,
                ))?
                .into_array(),
                FoRArray::encode(PrimitiveArray::new(
                    Buffer::from(near_max_u64),
                    Validity::NonNullable,
                ))?
                .into_array(),
            ],
            N,
            Validity::NonNullable,
        )?;
        Ok(arr.into_array())
    }
}
