// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex::array::ArrayRef;
use vortex::array::IntoArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::StructArray;
use vortex::array::dtype::FieldNames;
use vortex::array::validity::Validity;
use vortex::array::vtable::ArrayId;
use vortex::encodings::fastlanes::BitPacked;
use vortex::encodings::fastlanes::bitpack_compress::BitPackEncoder;
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
        let u32_8bit: PrimitiveArray = (0..N as u32).map(|i| i % 256).collect();
        let u64_12bit: PrimitiveArray = (0..N as u64).map(|i| i % 4096).collect();
        let u16_4bit: PrimitiveArray = (0..N as u16).map(|i| i % 16).collect();
        let u16_1bit: PrimitiveArray = (0..N as u16).map(|i| i % 2).collect();
        let u32_nullable = PrimitiveArray::from_option_iter(
            (0..N as u32).map(|i| (i % 8 != 0).then_some(i % 128)),
        );
        let u32_all_zero: PrimitiveArray = std::iter::repeat_n(0u32, N).collect();
        let u16_all_equal: PrimitiveArray = std::iter::repeat_n(7u16, N).collect();
        let u16_15bit: PrimitiveArray =
            (0..N as u16).map(|i| i.wrapping_mul(97) & 0x7fff).collect();
        let u32_31bit: PrimitiveArray = (0..N as u32)
            .map(|i| i.wrapping_mul(65_537) & 0x7fff_ffff)
            .collect();
        let u64_63bit: PrimitiveArray = (0..N as u64)
            .map(|i| i.wrapping_mul(1_099_511_627_791) & 0x7fff_ffff_ffff_ffff)
            .collect();
        let u8_3bit: PrimitiveArray = (0..N).map(|i| (i % 8) as u8).collect();
        let u8_5bit: PrimitiveArray = (0..N).map(|i| (i % 32) as u8).collect();
        let u16_9bit: PrimitiveArray = (0..N as u16).map(|i| i % 512).collect();
        let u32_17bit: PrimitiveArray = (0..N as u32).map(|i| i % 131_072).collect();
        let u16_head_tail_nulls = PrimitiveArray::from_option_iter((0..N as u16).map(|i| {
            if i < 8 || i >= N as u16 - 8 {
                None
            } else {
                Some(i % 32)
            }
        }));

        let arr = StructArray::try_new(
            FieldNames::from([
                "u32_8bit",
                "u64_12bit",
                "u16_4bit",
                "u16_1bit",
                "u32_nullable",
                "u32_all_zero",
                "u16_all_equal",
                "u16_15bit",
                "u32_31bit",
                "u64_63bit",
                "u8_3bit",
                "u8_5bit",
                "u16_9bit",
                "u32_17bit",
                "u16_head_tail_nulls",
            ]),
            vec![
                BitPackEncoder::new(&u32_8bit)
                    .with_bit_width(8)
                    .pack()?
                    .into_array()?,
                BitPackEncoder::new(&u64_12bit)
                    .with_bit_width(12)
                    .pack()?
                    .into_array()?,
                BitPackEncoder::new(&u16_4bit)
                    .with_bit_width(4)
                    .pack()?
                    .into_array()?,
                BitPackEncoder::new(&u16_1bit)
                    .with_bit_width(1)
                    .pack()?
                    .into_array()?,
                BitPackEncoder::new(&u32_nullable)
                    .with_bit_width(7)
                    .pack()?
                    .into_array()?,
                BitPackEncoder::new(&u32_all_zero)
                    .with_bit_width(1)
                    .pack()?
                    .into_array()?,
                BitPackEncoder::new(&u16_all_equal)
                    .with_bit_width(3)
                    .pack()?
                    .into_array()?,
                BitPackEncoder::new(&u16_15bit)
                    .with_bit_width(15)
                    .pack()?
                    .into_array()?,
                BitPackEncoder::new(&u32_31bit)
                    .with_bit_width(31)
                    .pack()?
                    .into_array()?,
                BitPackEncoder::new(&u64_63bit)
                    .with_bit_width(63)
                    .pack()?
                    .into_array()?,
                BitPackEncoder::new(&u8_3bit)
                    .with_bit_width(3)
                    .pack()?
                    .into_array()?,
                BitPackEncoder::new(&u8_5bit)
                    .with_bit_width(5)
                    .pack()?
                    .into_array()?,
                BitPackEncoder::new(&u16_9bit)
                    .with_bit_width(9)
                    .pack()?
                    .into_array()?,
                BitPackEncoder::new(&u32_17bit)
                    .with_bit_width(17)
                    .pack()?
                    .into_array()?,
                BitPackEncoder::new(&u16_head_tail_nulls)
                    .with_bit_width(5)
                    .pack()?
                    .into_array()?,
            ],
            N,
            Validity::NonNullable,
        )?;
        Ok(arr.into_array())
    }
}
