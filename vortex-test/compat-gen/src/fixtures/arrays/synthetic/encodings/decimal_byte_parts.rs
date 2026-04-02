// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex::array::ArrayId;
use vortex::array::ArrayRef;
use vortex::array::IntoArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::StructArray;
use vortex::array::dtype::DecimalDType;
use vortex::array::dtype::FieldNames;
use vortex::array::validity::Validity;
use vortex::encodings::decimal_byte_parts::DecimalByteParts;
use vortex::error::VortexResult;

use super::N;
use crate::fixtures::FlatLayoutFixture;

pub struct DecimalBytePartsFixture;

impl FlatLayoutFixture for DecimalBytePartsFixture {
    fn name(&self) -> &str {
        "decimal_byte_parts.vortex"
    }

    fn description(&self) -> &str {
        "Fixed-precision decimal arrays for DecimalByteParts encoding"
    }

    fn expected_encodings(&self) -> Vec<ArrayId> {
        vec![DecimalByteParts::ID]
    }

    fn build(&self) -> VortexResult<ArrayRef> {
        let decimal_dtype = DecimalDType::new(10, 2);
        let values: PrimitiveArray = (0..N as i64).map(|i| i * 100 + (i % 100)).collect();
        let msp_arr = values.into_array();
        let decimal_arr = DecimalByteParts::try_new(msp_arr, decimal_dtype)?;

        let hi_prec_dtype = DecimalDType::new(18, 6);
        let hi_prec_values: PrimitiveArray = (0..N as i64)
            .map(|i| i * 1_000_000 + (i * 7 % 999_999))
            .collect();
        let hi_prec_msp = hi_prec_values.into_array();
        let hi_prec_arr = DecimalByteParts::try_new(hi_prec_msp, hi_prec_dtype)?;

        let neg_dtype = DecimalDType::new(10, 2);
        let neg_values: PrimitiveArray = (0..N as i64).map(|i| -5000 + (i * 3 % 10000)).collect();
        let neg_msp = neg_values.into_array();
        let neg_arr = DecimalByteParts::try_new(neg_msp, neg_dtype)?;
        let nullable_dtype = DecimalDType::new(12, 4);
        let nullable_values = PrimitiveArray::from_option_iter((0..N as i64).map(|i| {
            if i % 11 == 0 {
                None
            } else {
                Some((i - 500) * 10_000)
            }
        }))
        .into_array();
        let nullable_arr = DecimalByteParts::try_new(nullable_values, nullable_dtype)?;
        let zero_dtype = DecimalDType::new(10, 2);
        let zero_arr = DecimalByteParts::try_new(
            std::iter::repeat_n(0i64, N)
                .collect::<PrimitiveArray>()
                .into_array(),
            zero_dtype,
        )?;
        let crossing_dtype = DecimalDType::new(12, 3);
        let crossing_values: PrimitiveArray = (0..N as i64).map(|i| (i % 200) - 100).collect();
        let crossing_arr = DecimalByteParts::try_new(crossing_values.into_array(), crossing_dtype)?;
        let trailing_zero_dtype = DecimalDType::new(18, 4);
        let trailing_zero_values: PrimitiveArray =
            (0..N as i64).map(|i| (i % 1000) * 10_000).collect();
        let trailing_zero_arr =
            DecimalByteParts::try_new(trailing_zero_values.into_array(), trailing_zero_dtype)?;
        let near_limit_dtype = DecimalDType::new(18, 0);
        let near_limit_values: PrimitiveArray =
            (0..N as i64).map(|i| 900_000_000_000_000_000 - i).collect();
        let near_limit_arr =
            DecimalByteParts::try_new(near_limit_values.into_array(), near_limit_dtype)?;

        let arr = StructArray::try_new(
            FieldNames::from([
                "dec_10_2",
                "dec_18_6",
                "dec_negative",
                "dec_nullable",
                "dec_zero",
                "dec_crossing",
                "dec_trailing_zero",
                "dec_near_limit",
            ]),
            vec![
                decimal_arr.into_array(),
                hi_prec_arr.into_array(),
                neg_arr.into_array(),
                nullable_arr.into_array(),
                zero_arr.into_array(),
                crossing_arr.into_array(),
                trailing_zero_arr.into_array(),
                near_limit_arr.into_array(),
            ],
            N,
            Validity::NonNullable,
        )?;
        Ok(arr.into_array())
    }
}
