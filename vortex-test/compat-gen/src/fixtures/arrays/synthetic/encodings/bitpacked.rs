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
use vortex::encodings::fastlanes::BitPacked;
use vortex::encodings::fastlanes::bitpack_compress::bitpack_encode;
use vortex::error::VortexResult;

use super::N;
use crate::fixtures::FlatLayoutFixture;

pub struct BitPackedFixture;

impl FlatLayoutFixture for BitPackedFixture {
    fn name(&self) -> &str {
        "bitpacked.vortex"
    }

    fn description(&self) -> &str {
        "Small unsigned integers that fit in fewer bits than their type width"
    }

    fn expected_encodings(&self) -> Vec<ArrayId> {
        vec![BitPacked::ID]
    }

    fn build(&self) -> VortexResult<ArrayRef> {
        let u32_8bit: Vec<u32> = (0..N as u32).map(|i| i % 256).collect();
        let u64_12bit: Vec<u64> = (0..N as u64).map(|i| i % 4096).collect();
        let u16_4bit: Vec<u16> = (0..N as u16).map(|i| i % 16).collect();
        let u16_1bit: Vec<u16> = (0..N as u16).map(|i| i % 2).collect();
        let u32_nullable = PrimitiveArray::from_option_iter(
            (0..N as u32).map(|i| (i % 8 != 0).then_some(i % 128)),
        );

        let arr = StructArray::try_new(
            FieldNames::from([
                "u32_8bit",
                "u64_12bit",
                "u16_4bit",
                "u16_1bit",
                "u32_nullable",
            ]),
            vec![
                bitpack_encode(
                    &PrimitiveArray::new(Buffer::from(u32_8bit), Validity::NonNullable),
                    8,
                    None,
                )?
                .into_array(),
                bitpack_encode(
                    &PrimitiveArray::new(Buffer::from(u64_12bit), Validity::NonNullable),
                    12,
                    None,
                )?
                .into_array(),
                bitpack_encode(
                    &PrimitiveArray::new(Buffer::from(u16_4bit), Validity::NonNullable),
                    4,
                    None,
                )?
                .into_array(),
                bitpack_encode(
                    &PrimitiveArray::new(Buffer::from(u16_1bit), Validity::NonNullable),
                    1,
                    None,
                )?
                .into_array(),
                bitpack_encode(&u32_nullable, 7, None)?.into_array(),
            ],
            N,
            Validity::NonNullable,
        )?;
        Ok(arr.into_array())
    }
}
