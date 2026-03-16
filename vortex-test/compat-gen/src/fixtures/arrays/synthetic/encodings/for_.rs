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
use crate::fixtures::ArrayFixture;

pub struct FoRFixture;

impl ArrayFixture for FoRFixture {
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

        let arr = StructArray::try_new(
            FieldNames::from([
                "clustered_i32",
                "clustered_u64",
                "clustered_i64",
                "negative_i32",
                "nullable_i32",
                "clustered_i16",
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
            ],
            N,
            Validity::NonNullable,
        )?;
        Ok(arr.into_array())
    }
}
