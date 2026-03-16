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
use vortex::encodings::pco::Pco;
use vortex::encodings::pco::PcoArray;
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
        let irregular_i64: Vec<i64> = (0..N as i64).map(|i| i * i + (i % 17) * 1000).collect();
        let smooth_f64: Vec<f64> = (0..N)
            .map(|i| {
                let t = i as f64 / N as f64;
                t * t * (3.0 - 2.0 * t) * 100.0
            })
            .collect();
        let pattern_u32: Vec<u32> = (0..N as u32)
            .map(|i| i.wrapping_mul(2_654_435_761) % 65536)
            .collect();
        let nullable_f32 = PrimitiveArray::from_option_iter(
            (0..N).map(|i| (i % 9 != 0).then_some((i as f32) * 0.1 + ((i * 3 % 7) as f32) * 0.01)),
        );
        let negative_i32: Vec<i32> = (0..N as i32).map(|i| -10_000 + (i % 257)).collect();
        let constant_u16: Vec<u16> = vec![17; N];
        let spike_outliers: Vec<f64> = (0..N)
            .map(|i| {
                if i % 257 == 0 {
                    1_000_000.0 + i as f64
                } else {
                    std::f64::consts::PI
                }
            })
            .collect();
        let narrow_i16: Vec<i16> = (0..N as i16).map(|i| (i % 17) - 8).collect();

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
                PcoArray::from_primitive(
                    &PrimitiveArray::new(Buffer::from(irregular_i64), Validity::NonNullable),
                    8,
                    0,
                )?
                .into_array(),
                PcoArray::from_primitive(
                    &PrimitiveArray::new(Buffer::from(smooth_f64), Validity::NonNullable),
                    8,
                    0,
                )?
                .into_array(),
                PcoArray::from_primitive(
                    &PrimitiveArray::new(Buffer::from(pattern_u32), Validity::NonNullable),
                    8,
                    0,
                )?
                .into_array(),
                PcoArray::from_primitive(&nullable_f32, 8, 0)?.into_array(),
                PcoArray::from_primitive(
                    &PrimitiveArray::new(Buffer::from(negative_i32), Validity::NonNullable),
                    8,
                    0,
                )?
                .into_array(),
                PcoArray::from_primitive(
                    &PrimitiveArray::new(Buffer::from(constant_u16), Validity::NonNullable),
                    8,
                    0,
                )?
                .into_array(),
                PcoArray::from_primitive(
                    &PrimitiveArray::new(Buffer::from(spike_outliers), Validity::NonNullable),
                    8,
                    0,
                )?
                .into_array(),
                PcoArray::from_primitive(
                    &PrimitiveArray::new(Buffer::from(narrow_i16), Validity::NonNullable),
                    8,
                    0,
                )?
                .into_array(),
            ],
            N,
            Validity::NonNullable,
        )?;
        Ok(arr.into_array())
    }
}
