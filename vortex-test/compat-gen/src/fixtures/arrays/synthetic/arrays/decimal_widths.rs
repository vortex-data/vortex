// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::ArrayVTable;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::Decimal;
use vortex_array::arrays::DecimalArray;
use vortex_array::arrays::StructArray;
use vortex_array::dtype::DecimalDType;
use vortex_array::dtype::i256;
use vortex_array::validity::Validity;
use vortex_buffer::buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::fixtures::FlatLayoutFixture;

/// Decimal columns that are stored at the canonical width of their precision.
///
/// Each column holds the largest and the smallest value that its precision allows. The columns
/// cover the narrowest and the widest precision of the `i8`, `i16`, `i128` and `i256` widths. The
/// `i32` and `i64` widths are in `decimal.vortex`.
pub struct DecimalCanonicalWidthsFixture;

impl FlatLayoutFixture for DecimalCanonicalWidthsFixture {
    fn name(&self) -> &str {
        "decimal_canonical_widths.vortex"
    }

    fn description(&self) -> &str {
        "Decimal arrays at the canonical width of each precision, with the bounds of each precision"
    }

    fn expected_encodings(&self) -> Vec<ArrayId> {
        vec![Decimal.id()]
    }

    fn build(&self, _ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
        let max_p19 = 10i128.pow(19) - 1;
        let max_p38 = 10i128.pow(38) - 1;
        let max_p39 = i256_max_by_precision(39);
        let max_p76 = i256_max_by_precision(76);

        let i8_p1_s0 = DecimalArray::new(
            buffer![-9i8, -1, 0, 1, 9],
            DecimalDType::new(1, 0),
            Validity::NonNullable,
        );
        let i8_p2_s1 = DecimalArray::new(
            buffer![-99i8, -1, 0, 1, 99],
            DecimalDType::new(2, 1),
            Validity::NonNullable,
        );
        let i16_p3_s1 = DecimalArray::new(
            buffer![-999i16, -1, 0, 1, 999],
            DecimalDType::new(3, 1),
            Validity::NonNullable,
        );
        let i16_p4_s2 = DecimalArray::new(
            buffer![-9_999i16, -1, 0, 1, 9_999],
            DecimalDType::new(4, 2),
            Validity::NonNullable,
        );
        let i128_p19_s2 = DecimalArray::new(
            buffer![-max_p19, -1, 0, 1, max_p19],
            DecimalDType::new(19, 2),
            Validity::NonNullable,
        );
        let i128_p38_s2 = DecimalArray::new(
            buffer![-max_p38, -1, 0, 1, max_p38],
            DecimalDType::new(38, 2),
            Validity::NonNullable,
        );
        let i256_p39_s2 = DecimalArray::new(
            buffer![-max_p39, -i256::ONE, i256::ZERO, i256::ONE, max_p39],
            DecimalDType::new(39, 2),
            Validity::NonNullable,
        );
        let i256_p76_s0 = DecimalArray::new(
            buffer![-max_p76, -i256::ONE, i256::ZERO, i256::ONE, max_p76],
            DecimalDType::new(76, 0),
            Validity::NonNullable,
        );
        let i128_p38_s2_nullable = DecimalArray::from_option_iter(
            [Some(-max_p38), None, Some(0), None, Some(max_p38)],
            DecimalDType::new(38, 2),
        );

        let arr = StructArray::try_from_iter([
            ("i8_p1_s0", i8_p1_s0.into_array()),
            ("i8_p2_s1", i8_p2_s1.into_array()),
            ("i16_p3_s1", i16_p3_s1.into_array()),
            ("i16_p4_s2", i16_p4_s2.into_array()),
            ("i128_p19_s2", i128_p19_s2.into_array()),
            ("i128_p38_s2", i128_p38_s2.into_array()),
            ("i256_p39_s2", i256_p39_s2.into_array()),
            ("i256_p76_s0", i256_p76_s0.into_array()),
            ("i128_p38_s2_nullable", i128_p38_s2_nullable.into_array()),
        ])?;

        Ok(arr.into_array())
    }
}

/// Returns `10^precision - 1`, the largest value of a decimal with this precision.
fn i256_max_by_precision(precision: u32) -> i256 {
    i256::from_i128(10)
        .checked_pow(precision)
        .vortex_expect("10^precision fits in i256 for every decimal precision")
        - i256::ONE
}
