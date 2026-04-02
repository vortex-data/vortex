// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex::array::ArrayId;
use vortex::array::ArrayRef;
use vortex::array::IntoArray;
use vortex::array::arrays::Constant;
use vortex::array::arrays::ConstantArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::StructArray;
use vortex::array::arrays::TemporalArray;
use vortex::array::dtype::DType;
use vortex::array::dtype::DecimalDType;
use vortex::array::dtype::FieldNames;
use vortex::array::dtype::Nullability;
use vortex::array::extension::datetime::TimeUnit;
use vortex::array::scalar::DecimalValue;
use vortex::array::scalar::Scalar;
use vortex::array::validity::Validity;
use vortex::error::VortexResult;

use super::N;
use crate::fixtures::FlatLayoutFixture;

pub struct ConstantFixture;

impl FlatLayoutFixture for ConstantFixture {
    fn name(&self) -> &str {
        "constant.vortex"
    }

    fn description(&self) -> &str {
        "Constant-value columns (int, float, string, bool, null) for Constant encoding"
    }

    fn expected_encodings(&self) -> Vec<ArrayId> {
        vec![Constant::ID]
    }

    fn build(&self) -> VortexResult<ArrayRef> {
        let const_i32 = ConstantArray::new(42i32, N);
        let const_f64 = ConstantArray::new(99.99f64, N);
        let const_bool = ConstantArray::new(true, N);
        let const_str = ConstantArray::new("constant_value", N);
        let const_zero = ConstantArray::new(0u64, N);
        let const_neg = ConstantArray::new(-1i64, N);
        let const_null = ConstantArray::new(Scalar::null(DType::Null), N);
        let const_nullable_i32 =
            ConstantArray::new(Scalar::primitive(42i32, Nullability::Nullable), N);
        let const_long_utf8 = ConstantArray::new(
            Scalar::utf8(
                "constant-value-with-a-longer-inline-boundary-crossing-payload".to_string(),
                Nullability::NonNullable,
            ),
            N,
        );
        let const_binary = ConstantArray::new(
            Scalar::binary(
                vec![0u8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
                Nullability::NonNullable,
            ),
            N,
        );
        let const_decimal = ConstantArray::new(
            Scalar::decimal(
                DecimalValue::I64(123_456),
                DecimalDType::new(10, 2),
                Nullability::NonNullable,
            ),
            N,
        );
        let timestamp_scalar = TemporalArray::new_timestamp(
            PrimitiveArray::from_iter([1_704_067_200_000i64]).into_array(),
            TimeUnit::Milliseconds,
            Some("UTC".into()),
        )
        .into_array()
        .scalar_at(0)?;
        let const_timestamp = ConstantArray::new(timestamp_scalar, N);

        let arr = StructArray::try_new(
            FieldNames::from([
                "const_i32",
                "const_f64",
                "const_bool",
                "const_str",
                "const_zero",
                "const_neg",
                "const_null",
                "const_nullable_i32",
                "const_long_utf8",
                "const_binary",
                "const_decimal",
                "const_timestamp",
            ]),
            vec![
                const_i32.into_array(),
                const_f64.into_array(),
                const_bool.into_array(),
                const_str.into_array(),
                const_zero.into_array(),
                const_neg.into_array(),
                const_null.into_array(),
                const_nullable_i32.into_array(),
                const_long_utf8.into_array(),
                const_binary.into_array(),
                const_decimal.into_array(),
                const_timestamp.into_array(),
            ],
            N,
            Validity::NonNullable,
        )?;
        Ok(arr.into_array())
    }
}
