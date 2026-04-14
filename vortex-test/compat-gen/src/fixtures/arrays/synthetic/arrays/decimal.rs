// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::ArrayVTable;
use vortex_array::IntoArray;
use vortex_array::arrays::Decimal;
use vortex_array::arrays::DecimalArray;
use vortex_array::arrays::StructArray;
use vortex_array::dtype::DecimalDType;
use vortex_array::dtype::FieldNames;
use vortex_array::validity::Validity;
use vortex_buffer::buffer;
use vortex_error::VortexResult;

use crate::fixtures::FlatLayoutFixture;

pub struct DecimalFixture;

impl FlatLayoutFixture for DecimalFixture {
    fn name(&self) -> &str {
        "decimal.vortex"
    }

    fn description(&self) -> &str {
        "Decimal arrays with varying precisions and scales, including nullable variants"
    }

    fn expected_encodings(&self) -> Vec<ArrayId> {
        vec![Decimal.id()]
    }

    fn build(&self) -> VortexResult<ArrayRef> {
        // Decimal(5,2) stored as i32: represents 123.45, -678.90, 0.01
        let dec_5_2 = DecimalArray::new(
            buffer![12345i32, -67890, 1],
            DecimalDType::new(5, 2),
            Validity::NonNullable,
        );

        // Decimal(10,4) stored as i64: represents 123456.7890, -0.0001, 999999.9999
        let dec_10_4 = DecimalArray::new(
            buffer![1234567890i64, -1, 9999999999],
            DecimalDType::new(10, 4),
            Validity::NonNullable,
        );

        // Decimal(18,0) stored as i64: large integers
        let dec_18_0 = DecimalArray::new(
            buffer![0i64, 999_999_999_999_999_999, -999_999_999_999_999_999],
            DecimalDType::new(18, 0),
            Validity::NonNullable,
        );

        // Nullable Decimal(7,3) stored as i32
        let dec_nullable = DecimalArray::from_option_iter(
            [Some(1234567i32), None, Some(-9999999)],
            DecimalDType::new(7, 3),
        );

        let arr = StructArray::try_new(
            FieldNames::from(["dec_5_2", "dec_10_4", "dec_18_0", "dec_nullable_7_3"]),
            vec![
                dec_5_2.into_array(),
                dec_10_4.into_array(),
                dec_18_0.into_array(),
                dec_nullable.into_array(),
            ],
            3,
            Validity::NonNullable,
        )?;
        Ok(arr.into_array())
    }
}
