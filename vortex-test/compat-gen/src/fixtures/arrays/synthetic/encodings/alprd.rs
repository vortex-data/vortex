// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex::array::ArrayId;
use vortex::array::ArrayRef;
use vortex::array::ArrayVTable;
use vortex::array::IntoArray;
use vortex::array::LEGACY_SESSION;
use vortex::array::VortexSessionExecute;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::StructArray;
use vortex::array::dtype::FieldNames;
use vortex::array::validity::Validity;
use vortex::encodings::alp::ALPRD;
use vortex::encodings::alp::RDEncoder;
use vortex::error::VortexResult;

use super::N;
use crate::fixtures::FlatLayoutFixture;

pub struct AlprdFixture;

fn special_f64(i: usize) -> f64 {
    match i % 9 {
        0 => 0.0,
        1 => -0.0,
        2 => f64::from_bits(0x7ff8_0000_0000_0001),
        3 => f64::from_bits(0x7ff8_0000_0000_1234),
        4 => f64::from_bits(0xfff8_0000_0000_5678),
        5 => f64::INFINITY,
        6 => f64::NEG_INFINITY,
        7 => f64::MIN_POSITIVE,
        _ => -f64::from_bits(1),
    }
}

impl FlatLayoutFixture for AlprdFixture {
    fn name(&self) -> &str {
        "alprd.vortex"
    }

    fn description(&self) -> &str {
        "Real-valued doubles with small deltas for ALPRD encoding"
    }

    fn expected_encodings(&self) -> Vec<ArrayId> {
        vec![ALPRD.id()]
    }

    fn build(&self) -> VortexResult<ArrayRef> {
        // NOTE: `FlatLayoutFixture::build` has a fixed trait signature without `ExecutionCtx`, so
        // we construct a legacy ctx locally at this trait boundary.
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let sensor: PrimitiveArray = (0..N)
            .map(|i| {
                let noise = ((i * 7 + 13) % 100) as f64 / 1000.0;
                98.6 + noise
            })
            .collect();

        let drift: PrimitiveArray = (0..N)
            .map(|i| 1000.0 + (i as f64) * 0.001 + ((i * 3) % 7) as f64 * 0.0001)
            .collect();
        let constant_series: PrimitiveArray = std::iter::repeat_n(12.125f64, N).collect();
        let decreasing: PrimitiveArray = (0..N)
            .map(|i| 512.0 - (i as f64) * 0.000_5 - ((i * 5 % 13) as f64) * 0.000_01)
            .collect();
        let oscillating: PrimitiveArray = (0..N)
            .map(|i| {
                let phase = ((i % 9) as i32 - 4) as f64;
                -0.25 + phase * 0.000_1 + (i as f64) * 0.000_001
            })
            .collect();
        let periodic_resets: PrimitiveArray = (0..N)
            .map(|i| {
                let block = i / 64;
                let offset = i % 64;
                block as f64 * 10.0 + (offset as f64) * 0.000_2
            })
            .collect();

        let sensor_nullable_vals: Vec<f64> = (0..N)
            .map(|i| {
                let noise = ((i * 11 + 3) % 100) as f64 / 1000.0;
                37.0 + noise
            })
            .collect();
        let sensor_nullable = PrimitiveArray::from_option_iter((0..N).map(|i| {
            if i % 13 == 0 {
                None
            } else {
                let noise = ((i * 11 + 3) % 100) as f64 / 1000.0;
                Some(37.0 + noise)
            }
        }));
        let special_values: PrimitiveArray = (0..N)
            .map(|i| {
                if i % 16 == 0 {
                    special_f64(i)
                } else {
                    42.125 + ((i * 5 % 17) as f64) * 0.000_01
                }
            })
            .collect();
        let boundary_specials: PrimitiveArray = (0..N)
            .map(|i| match i {
                0 => f64::from_bits(0x7ff8_0000_0000_0001),
                1 => -0.0,
                511 => f64::INFINITY,
                512 => f64::NEG_INFINITY,
                513 => f64::from_bits(0xfff8_0000_0000_5678),
                1023 => f64::from_bits(1),
                _ => 9.875 + ((i * 3 % 11) as f64) * 0.000_1,
            })
            .collect();
        let nullable_special_vals: Vec<f64> = (0..N)
            .map(|i| {
                if i % 32 == 7 {
                    special_f64(i)
                } else {
                    11.5 + ((i * 13 % 19) as f64) * 0.000_01
                }
            })
            .collect();
        let nullable_specials = PrimitiveArray::from_option_iter((0..N).map(|i| {
            if i % 29 == 0 || i == 0 || i == N - 1 {
                None
            } else {
                Some(nullable_special_vals[i])
            }
        }));

        let sensor_enc = RDEncoder::new::<f64>(sensor.as_slice::<f64>());
        let drift_enc = RDEncoder::new::<f64>(drift.as_slice::<f64>());
        let constant_enc = RDEncoder::new::<f64>(constant_series.as_slice::<f64>());
        let decreasing_enc = RDEncoder::new::<f64>(decreasing.as_slice::<f64>());
        let oscillating_enc = RDEncoder::new::<f64>(oscillating.as_slice::<f64>());
        let periodic_resets_enc = RDEncoder::new::<f64>(periodic_resets.as_slice::<f64>());
        let nullable_enc = RDEncoder::new::<f64>(&sensor_nullable_vals);
        let special_enc = RDEncoder::new::<f64>(special_values.as_slice::<f64>());
        let boundary_enc = RDEncoder::new::<f64>(boundary_specials.as_slice::<f64>());
        let nullable_special_enc = RDEncoder::new::<f64>(&nullable_special_vals);

        let arr = StructArray::try_new(
            FieldNames::from([
                "sensor",
                "drift",
                "constant_series",
                "decreasing",
                "oscillating",
                "periodic_resets",
                "sensor_nullable",
                "special_values",
                "boundary_specials",
                "nullable_specials",
            ]),
            vec![
                sensor_enc.encode(sensor.as_view(), &mut ctx).into_array(),
                drift_enc.encode(drift.as_view(), &mut ctx).into_array(),
                constant_enc
                    .encode(constant_series.as_view(), &mut ctx)
                    .into_array(),
                decreasing_enc
                    .encode(decreasing.as_view(), &mut ctx)
                    .into_array(),
                oscillating_enc
                    .encode(oscillating.as_view(), &mut ctx)
                    .into_array(),
                periodic_resets_enc
                    .encode(periodic_resets.as_view(), &mut ctx)
                    .into_array(),
                nullable_enc
                    .encode(sensor_nullable.as_view(), &mut ctx)
                    .into_array(),
                special_enc
                    .encode(special_values.as_view(), &mut ctx)
                    .into_array(),
                boundary_enc
                    .encode(boundary_specials.as_view(), &mut ctx)
                    .into_array(),
                nullable_special_enc
                    .encode(nullable_specials.as_view(), &mut ctx)
                    .into_array(),
            ],
            N,
            Validity::NonNullable,
        )?;
        Ok(arr.into_array())
    }
}
