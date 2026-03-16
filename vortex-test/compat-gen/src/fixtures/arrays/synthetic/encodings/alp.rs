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
use vortex::encodings::alp::ALP;
use vortex::encodings::alp::alp_encode;
use vortex::error::VortexResult;

use super::N;
use crate::fixtures::ArrayFixture;

pub struct AlpFixture;

impl ArrayFixture for AlpFixture {
    fn name(&self) -> &str {
        "alp.vortex"
    }

    fn description(&self) -> &str {
        "Near-integer floats and decimal-like prices for ALP encoding (f32 + f64)"
    }

    fn expected_encodings(&self) -> Vec<ArrayId> {
        vec![ALP::ID]
    }

    fn build(&self) -> VortexResult<ArrayRef> {
        let f64_prices: Vec<f64> = (0..N).map(|i| 100.0 + (i as f64) * 0.25).collect();
        let f32_near_int: Vec<f32> = (0..N).map(|i| i as f32).collect();
        let f64_currency: Vec<f64> = (0..N).map(|i| ((i % 10000) as f64) / 100.0).collect();
        let f64_nullable = PrimitiveArray::from_option_iter(
            (0..N as i64).map(|i| (i % 10 != 0).then_some(50.0 + (i as f64) * 0.125)),
        );
        let f64_patched: Vec<f64> = (0..N)
            .map(|i| {
                if i % 100 == 0 {
                    std::f64::consts::PI * (i as f64 + 1.0)
                } else {
                    (i as f64) * 0.01
                }
            })
            .collect();

        let arr = StructArray::try_new(
            FieldNames::from([
                "f64_prices",
                "f32_near_int",
                "f64_currency",
                "f64_nullable",
                "f64_patched",
            ]),
            vec![
                alp_encode(
                    &PrimitiveArray::new(Buffer::from(f64_prices), Validity::NonNullable),
                    None,
                )?
                .into_array(),
                alp_encode(
                    &PrimitiveArray::new(Buffer::from(f32_near_int), Validity::NonNullable),
                    None,
                )?
                .into_array(),
                alp_encode(
                    &PrimitiveArray::new(Buffer::from(f64_currency), Validity::NonNullable),
                    None,
                )?
                .into_array(),
                alp_encode(&f64_nullable, None)?.into_array(),
                alp_encode(
                    &PrimitiveArray::new(Buffer::from(f64_patched), Validity::NonNullable),
                    None,
                )?
                .into_array(),
            ],
            N,
            Validity::NonNullable,
        )?;
        Ok(arr.into_array())
    }
}
