// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! `DecimalByteParts` fixture for wide decimal values that need lower parts.

use vortex::array::ArrayId;
use vortex::array::ArrayRef;
use vortex::array::ArrayVTable;
use vortex::array::IntoArray;
use vortex::array::arrays::DecimalArray;
use vortex::array::arrays::StructArray;
use vortex::array::dtype::DecimalDType;
use vortex::array::dtype::FieldNames;
use vortex::array::dtype::i256;
use vortex::array::validity::Validity;
use vortex::buffer::Buffer;
use vortex::encodings::decimal_byte_parts::DecimalByteParts;
use vortex::encodings::decimal_byte_parts::DecimalBytePartsArray;
use vortex::encodings::decimal_byte_parts::split_decimal;
use vortex::error::VortexResult;
use vortex_array::ExecutionCtx;

use super::N;
use crate::fixtures::FlatLayoutFixture;

/// Encode a canonical decimal as byte parts, splitting wide values into lower parts.
fn encode_byte_parts(
    decimal: &DecimalArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<DecimalBytePartsArray> {
    let parts = split_decimal(decimal, ctx)?;
    DecimalByteParts::try_new_with_lower_parts(
        parts.msp,
        parts.lower_parts,
        decimal.decimal_dtype(),
    )
}

pub struct DecimalBytePartsV2Fixture;

impl FlatLayoutFixture for DecimalBytePartsV2Fixture {
    fn name(&self) -> &str {
        "decimal_byte_parts_v2.vortex"
    }

    fn description(&self) -> &str {
        "Wide decimal arrays split into a most significant part plus 64-bit lower parts"
    }

    fn expected_encodings(&self) -> Vec<ArrayId> {
        vec![DecimalByteParts.id()]
    }

    fn build(&self, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
        // An `i128` magnitude above 2^64, so the encoding must carry one lower part.
        let wide_128_dtype = DecimalDType::new(38, 2);
        let wide_128 = DecimalArray::new(
            (0..N as i128)
                .map(|i| 10i128.pow(25) + i * 7)
                .collect::<Buffer<i128>>(),
            wide_128_dtype,
            Validity::NonNullable,
        );
        let wide_128_arr = encode_byte_parts(&wide_128, ctx)?;

        // Negative values, so the sign extension above the MSP is exercised on read back.
        let wide_128_negative = DecimalArray::new(
            (0..N as i128)
                .map(|i| -(10i128.pow(25)) - i * 7)
                .collect::<Buffer<i128>>(),
            wide_128_dtype,
            Validity::NonNullable,
        );
        let wide_128_negative_arr = encode_byte_parts(&wide_128_negative, ctx)?;

        // An `i256` magnitude beyond 128 bits, so all three lower parts are populated, with
        // nulls to pin that validity is carried by the MSP alone.
        let wide_256_dtype = DecimalDType::new(76, 2);
        let base = i256::from_i128(10).wrapping_pow(40);
        let wide_256 = DecimalArray::new(
            (0..N as i128)
                .map(|i| base + i256::from_i128(i * 7))
                .collect::<Buffer<i256>>(),
            wide_256_dtype,
            Validity::from_iter((0..N).map(|i| i % 7 != 0)),
        );
        let wide_256_arr = encode_byte_parts(&wide_256, ctx)?;

        let arr = StructArray::try_new(
            FieldNames::from([
                "dec_wide_128",
                "dec_wide_128_negative",
                "dec_wide_256_nullable",
            ]),
            vec![
                wide_128_arr.into_array(),
                wide_128_negative_arr.into_array(),
                wide_256_arr.into_array(),
            ],
            N,
            Validity::NonNullable,
        )?;
        Ok(arr.into_array())
    }
}
