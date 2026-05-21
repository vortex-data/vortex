// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::f64;

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
use vortex::encodings::alp::ALP;
use vortex::encodings::alp::alp_encode;
use vortex::error::VortexResult;

use super::N;
use crate::fixtures::FlatLayoutFixture;

pub struct AlpFixture;

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

fn special_f32(i: usize) -> f32 {
    match i % 9 {
        0 => 0.0,
        1 => -0.0,
        2 => f32::from_bits(0x7fc0_0001),
        3 => f32::from_bits(0x7fc0_1234),
        4 => f32::from_bits(0xffc0_5678),
        5 => f32::INFINITY,
        6 => f32::NEG_INFINITY,
        7 => f32::MIN_POSITIVE,
        _ => -f32::from_bits(1),
    }
}

impl FlatLayoutFixture for AlpFixture {
    fn name(&self) -> &str {
        "alp.vortex"
    }

    fn description(&self) -> &str {
        "Near-integer floats and decimal-like prices for ALP encoding (f32 + f64)"
    }

    fn expected_encodings(&self) -> Vec<ArrayId> {
        vec![ALP.id()]
    }

    fn build(&self) -> VortexResult<ArrayRef> {
        let f64_prices: PrimitiveArray = (0..N).map(|i| 100.0 + (i as f64) * 0.25).collect();
        let f32_near_int: PrimitiveArray = (0..N).map(|i| i as f32).collect();
        let f64_negative_near_int: PrimitiveArray = (0..N)
            .map(|i| -(i as f64) - ((i % 7) as f64) * 0.000_1)
            .collect();
        let f64_currency: PrimitiveArray = (0..N).map(|i| ((i % 10000) as f64) / 100.0).collect();
        let f64_nullable = PrimitiveArray::from_option_iter(
            (0..N as i64).map(|i| (i % 10 != 0).then_some(50.0 + (i as f64) * 0.125)),
        );
        let f64_patched: PrimitiveArray = (0..N)
            .map(|i| {
                if i % 100 == 0 {
                    f64::consts::PI * (i as f64 + 1.0)
                } else {
                    (i as f64) * 0.01
                }
            })
            .collect();
        let f64_patch_heavy: PrimitiveArray = (0..N)
            .map(|i| {
                if i % 7 == 0 || i % 11 == 0 {
                    10_000.0 + (i as f64).powi(2)
                } else {
                    250.0 + ((i % 37) as f64) * 0.01
                }
            })
            .collect();
        let f64_special_values: PrimitiveArray = (0..N).map(special_f64).collect();
        let f32_special_values: PrimitiveArray = (0..N).map(special_f32).collect();
        let f64_extremes: PrimitiveArray = (0..N)
            .map(|i| match i % 10 {
                0 => f64::MAX,
                1 => f64::MIN,
                2 => f64::EPSILON,
                3 => -f64::EPSILON,
                4 => f64::MIN_POSITIVE,
                5 => -f64::MIN_POSITIVE,
                6 => f64::consts::PI,
                7 => -f64::consts::E,
                8 => f64::from_bits(1),
                _ => -f64::from_bits(2),
            })
            .collect();
        let f64_boundary_specials: PrimitiveArray = (0..N)
            .map(|i| match i {
                0 => f64::from_bits(0x7ff8_0000_0000_0001),
                1 => -0.0,
                511 => f64::INFINITY,
                512 => f64::NEG_INFINITY,
                513 => f64::from_bits(0xfff8_0000_0000_5678),
                1023 => f64::from_bits(1),
                _ => 12.5 + (i as f64) * 0.000_001,
            })
            .collect();

        let arr = StructArray::try_new(
            FieldNames::from([
                "f64_prices",
                "f32_near_int",
                "f64_negative_near_int",
                "f64_currency",
                "f64_nullable",
                "f64_patched",
                "f64_patch_heavy",
                "f64_special_values",
                "f32_special_values",
                "f64_extremes",
                "f64_boundary_specials",
            ]),
            vec![
                alp_encode(
                    f64_prices.as_view(),
                    None,
                    &mut LEGACY_SESSION.create_execution_ctx(),
                )?
                .into_array(),
                alp_encode(
                    f32_near_int.as_view(),
                    None,
                    &mut LEGACY_SESSION.create_execution_ctx(),
                )?
                .into_array(),
                alp_encode(
                    f64_negative_near_int.as_view(),
                    None,
                    &mut LEGACY_SESSION.create_execution_ctx(),
                )?
                .into_array(),
                alp_encode(
                    f64_currency.as_view(),
                    None,
                    &mut LEGACY_SESSION.create_execution_ctx(),
                )?
                .into_array(),
                alp_encode(
                    f64_nullable.as_view(),
                    None,
                    &mut LEGACY_SESSION.create_execution_ctx(),
                )?
                .into_array(),
                alp_encode(
                    f64_patched.as_view(),
                    None,
                    &mut LEGACY_SESSION.create_execution_ctx(),
                )?
                .into_array(),
                alp_encode(
                    f64_patch_heavy.as_view(),
                    None,
                    &mut LEGACY_SESSION.create_execution_ctx(),
                )?
                .into_array(),
                alp_encode(
                    f64_special_values.as_view(),
                    None,
                    &mut LEGACY_SESSION.create_execution_ctx(),
                )?
                .into_array(),
                alp_encode(
                    f32_special_values.as_view(),
                    None,
                    &mut LEGACY_SESSION.create_execution_ctx(),
                )?
                .into_array(),
                alp_encode(
                    f64_extremes.as_view(),
                    None,
                    &mut LEGACY_SESSION.create_execution_ctx(),
                )?
                .into_array(),
                alp_encode(
                    f64_boundary_specials.as_view(),
                    None,
                    &mut LEGACY_SESSION.create_execution_ctx(),
                )?
                .into_array(),
            ],
            N,
            Validity::NonNullable,
        )?;
        Ok(arr.into_array())
    }
}
