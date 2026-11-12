// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex::array::ArrayId;
use vortex::array::ArrayRef;
use vortex::array::ArrayVTable;
use vortex::array::IntoArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::StructArray;
use vortex::array::dtype::DecimalDType;
use vortex::array::validity::Validity;
use vortex::buffer::Buffer;
use vortex::encodings::decimal_byte_parts::DecimalByteParts;
use vortex::error::VortexExpect;
use vortex::error::VortexResult;
use vortex_array::ExecutionCtx;

use super::N;
use crate::fixtures::FlatLayoutFixture;

/// DecimalByteParts columns whose MSP width differs from the canonical width of the decimal dtype.
///
/// The `decimal_byte_parts.vortex` fixture covers only `i64` MSPs.
pub struct DecimalBytePartsNarrowMspFixture;

impl FlatLayoutFixture for DecimalBytePartsNarrowMspFixture {
    fn name(&self) -> &str {
        "decimal_byte_parts_narrow_msp.vortex"
    }

    fn description(&self) -> &str {
        "DecimalByteParts arrays with an MSP narrower or wider than the canonical decimal width"
    }

    fn expected_encodings(&self) -> Vec<ArrayId> {
        vec![DecimalByteParts.id()]
    }

    fn build(&self, _ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
        let max_p18 = 10i64.pow(18) - 1;

        let msp_i8_p10_s2 = evenly_spaced(i8::MIN, i8::MAX).into_array();
        let msp_i16_p10_s2_gap = evenly_spaced(10_000i16, i16::MAX).into_array();
        let msp_i32_p18_s2 = evenly_spaced(-999_999_999i32, 999_999_999).into_array();
        let msp_i32_p18_s2_gap = evenly_spaced(1_000_000_000i32, i32::MAX).into_array();
        let msp_i64_p38_s2 = evenly_spaced(-max_p18, max_p18).into_array();
        let msp_i64_p4_s2_wide = evenly_spaced(-9_999i64, 9_999).into_array();

        // The null slots hold `i32::MAX`, a nonzero value. The remainder 3 keeps the first and the
        // last rows valid, so the valid values still reach both bounds.
        let is_null = |idx: usize| idx % 7 == 3;
        let nullable_values: Buffer<i32> = evenly_spaced(-999_999_999i32, 999_999_999)
            .into_iter()
            .enumerate()
            .map(|(idx, value)| if is_null(idx) { i32::MAX } else { value })
            .collect();
        let nullable_validity = Validity::from_iter((0..N).map(|idx| !is_null(idx)));
        let msp_i32_p18_s2_nullable =
            PrimitiveArray::new(nullable_values, nullable_validity).into_array();

        let arr = StructArray::try_from_iter([
            (
                "msp_i8_p10_s2",
                DecimalByteParts::try_new(msp_i8_p10_s2, DecimalDType::new(10, 2))?,
            ),
            (
                "msp_i16_p10_s2_gap",
                DecimalByteParts::try_new(msp_i16_p10_s2_gap, DecimalDType::new(10, 2))?,
            ),
            (
                "msp_i32_p18_s2",
                DecimalByteParts::try_new(msp_i32_p18_s2, DecimalDType::new(18, 2))?,
            ),
            (
                "msp_i32_p18_s2_gap",
                DecimalByteParts::try_new(msp_i32_p18_s2_gap, DecimalDType::new(18, 2))?,
            ),
            (
                "msp_i64_p38_s2",
                DecimalByteParts::try_new(msp_i64_p38_s2, DecimalDType::new(38, 2))?,
            ),
            (
                "msp_i32_p18_s2_nullable",
                DecimalByteParts::try_new(msp_i32_p18_s2_nullable, DecimalDType::new(18, 2))?,
            ),
            (
                "msp_i64_p4_s2_wide",
                DecimalByteParts::try_new(msp_i64_p4_s2_wide, DecimalDType::new(4, 2))?,
            ),
        ])?;

        Ok(arr.into_array())
    }
}

/// Returns `N` values from `min` to `max`, with both bounds included.
///
/// The steps are as even as integer division allows. When the range holds fewer than `N` values,
/// some values repeat, and every value of the range appears.
fn evenly_spaced<T>(min: T, max: T) -> Buffer<T>
where
    T: Copy + Into<i128> + TryFrom<i128>,
{
    let (min, max) = (min.into(), max.into());
    let last = (N - 1) as i128;

    (0..=last)
        .map(|idx| {
            T::try_from(min + (max - min) * idx / last)
                .ok()
                .vortex_expect("an evenly spaced value lies between min and max")
        })
        .collect()
}
