// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::dtype::FieldNames;
use vortex_array::validity::Validity;
use vortex_array::vtable::ArrayId;
use vortex_buffer::buffer;
use vortex_error::VortexResult;

use crate::fixtures::FlatLayoutFixture;

pub struct PrimitivesFixture;

impl FlatLayoutFixture for PrimitivesFixture {
    fn name(&self) -> &str {
        "primitives.vortex"
    }

    fn description(&self) -> &str {
        "All primitive types (u8-u64, i32, i64, f32, f64) including a nullable i32 column"
    }

    fn expected_encodings(&self) -> Vec<ArrayId> {
        vec![Primitive::ID]
    }

    fn build(&self) -> VortexResult<ArrayRef> {
        let arr = StructArray::try_new(
            FieldNames::from([
                "u8",
                "u16",
                "u32",
                "u64",
                "i32",
                "i64",
                "f32",
                "f64",
                "nullable_i32",
                "f32_special",
                "f64_special",
            ]),
            vec![
                PrimitiveArray::new(buffer![0u8, 128, 255], Validity::NonNullable).into_array(),
                PrimitiveArray::new(buffer![0u16, 32768, 65535], Validity::NonNullable)
                    .into_array(),
                PrimitiveArray::new(
                    buffer![0u32, 2_147_483_648, 4_294_967_295],
                    Validity::NonNullable,
                )
                .into_array(),
                PrimitiveArray::new(
                    buffer![0u64, 9_223_372_036_854_775_808, u64::MAX],
                    Validity::NonNullable,
                )
                .into_array(),
                PrimitiveArray::new(buffer![i32::MIN, 0i32, i32::MAX], Validity::NonNullable)
                    .into_array(),
                PrimitiveArray::new(buffer![i64::MIN, 0i64, i64::MAX], Validity::NonNullable)
                    .into_array(),
                PrimitiveArray::new(buffer![f32::MIN, 0.0f32, f32::MAX], Validity::NonNullable)
                    .into_array(),
                PrimitiveArray::new(buffer![f64::MIN, 0.0f64, f64::MAX], Validity::NonNullable)
                    .into_array(),
                PrimitiveArray::from_option_iter([Some(1i32), None, Some(42)]).into_array(),
                // Special float values: NaN, infinities, negative zero, subnormal
                PrimitiveArray::new(
                    buffer![f32::NAN, f32::INFINITY, f32::NEG_INFINITY],
                    Validity::NonNullable,
                )
                .into_array(),
                PrimitiveArray::new(
                    buffer![f64::NAN, -0.0f64, f64::MIN_POSITIVE / 2.0],
                    Validity::NonNullable,
                )
                .into_array(),
            ],
            3,
            Validity::NonNullable,
        )?;
        Ok(arr.into_array())
    }
}
