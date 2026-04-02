// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex::array::ArrayId;
use vortex::array::ArrayRef;
use vortex::array::IntoArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::StructArray;
use vortex::array::dtype::FieldNames;
use vortex::array::validity::Validity;
use vortex::encodings::pco::Pco;
use vortex::error::VortexResult;

use super::N;
use crate::fixtures::FlatLayoutFixture;

pub struct PcoFixture;

impl FlatLayoutFixture for PcoFixture {
    fn name(&self) -> &str {
        "pco.vortex"
    }

    fn description(&self) -> &str {
        "Various numeric patterns for Pco (patas compression) encoding"
    }

    fn expected_encodings(&self) -> Vec<ArrayId> {
        vec![Pco::ID]
    }

    fn build(&self) -> VortexResult<ArrayRef> {
        let irregular_i64: PrimitiveArray =
            (0..N as i64).map(|i| i * i + (i % 17) * 1000).collect();
        let smooth_f64: PrimitiveArray = (0..N)
            .map(|i| {
                let t = i as f64 / N as f64;
                t * t * (3.0 - 2.0 * t) * 100.0
            })
            .collect();
        let pattern_u32: PrimitiveArray = (0..N as u32)
            .map(|i| i.wrapping_mul(2_654_435_761) % 65536)
            .collect();
        let nullable_f32 = PrimitiveArray::from_option_iter(
            (0..N).map(|i| (i % 9 != 0).then_some((i as f32) * 0.1 + ((i * 3 % 7) as f32) * 0.01)),
        );
        let negative_i32: PrimitiveArray = (0..N as i32).map(|i| -10_000 + (i % 257)).collect();
        let constant_u16: PrimitiveArray = std::iter::repeat_n(17u16, N).collect();
        let spike_outliers: PrimitiveArray = (0..N)
            .map(|i| {
                if i % 257 == 0 {
                    1_000_000.0 + i as f64
                } else {
                    std::f64::consts::PI
                }
            })
            .collect();
        let narrow_i16: PrimitiveArray = (0..N as i16).map(|i| (i % 17) - 8).collect();

        let arr = StructArray::try_new(
            FieldNames::from([
                "irregular_i64",
                "smooth_f64",
                "pattern_u32",
                "nullable_f32",
                "negative_i32",
                "constant_u16",
                "spike_outliers",
                "narrow_i16",
            ]),
            vec![
                Pco::from_primitive(&irregular_i64, 8, 0)?.into_array(),
                Pco::from_primitive(&smooth_f64, 8, 0)?.into_array(),
                Pco::from_primitive(&pattern_u32, 8, 0)?.into_array(),
                Pco::from_primitive(&nullable_f32, 8, 0)?.into_array(),
                Pco::from_primitive(&negative_i32, 8, 0)?.into_array(),
                Pco::from_primitive(&constant_u16, 8, 0)?.into_array(),
                Pco::from_primitive(&spike_outliers, 8, 0)?.into_array(),
                Pco::from_primitive(&narrow_i16, 8, 0)?.into_array(),
            ],
            N,
            Validity::NonNullable,
        )?;
        Ok(arr.into_array())
    }
}
