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
use vortex::array::arrays::VarBinViewArray;
use vortex::array::dtype::FieldNames;
use vortex::array::validity::Validity;
use vortex::encodings::zstd::Zstd;
use vortex::error::VortexResult;

use super::N;
use crate::fixtures::FlatLayoutFixture;

pub struct ZstdFixture;

impl FlatLayoutFixture for ZstdFixture {
    fn name(&self) -> &str {
        "zstd.vortex"
    }

    fn description(&self) -> &str {
        "Primitive and string arrays compressed with Zstd encoding"
    }

    fn expected_encodings(&self) -> Vec<ArrayId> {
        vec![Zstd.id()]
    }

    fn build(&self) -> VortexResult<ArrayRef> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let ints: PrimitiveArray = (0..N as i32).map(|i| i / 8).collect();
        let floats: PrimitiveArray = (0..N)
            .map(|i| {
                if i % 257 == 0 {
                    100_000.0 + i as f64
                } else {
                    f64::consts::PI
                }
            })
            .collect();
        let nullable_i64 = PrimitiveArray::from_option_iter((0..N as i64).map(|i| {
            if i % 9 == 0 {
                None
            } else {
                Some(10_000 + i * 3)
            }
        }));
        let utf8 = VarBinViewArray::from_iter_str((0..N).map(|i| {
            if i % 3 == 0 {
                "https://example.com/path"
            } else if i % 3 == 1 {
                "hello-world"
            } else {
                "compression-fixture"
            }
        }));
        let nullable_utf8 = VarBinViewArray::from_iter_nullable_str((0..N).map(|i| {
            if i % 11 == 0 {
                None
            } else if i % 2 == 0 {
                Some("nullable-zstd-string")
            } else {
                Some("another-string")
            }
        }));
        // Highly compressible: all zeros
        let all_zeros: PrimitiveArray = std::iter::repeat_n(0i64, N).collect();
        // All-null column
        let all_null_i32 = PrimitiveArray::from_option_iter((0..N).map(|_| None::<i32>));
        // Pseudo-random data (low compressibility)
        let pseudo_random: PrimitiveArray =
            (0..N as u32).map(|i| i.wrapping_mul(2654435761)).collect();

        let arr = StructArray::try_new(
            FieldNames::from([
                "ints",
                "floats",
                "nullable_i64",
                "utf8",
                "nullable_utf8",
                "all_zeros",
                "all_null_i32",
                "pseudo_random",
            ]),
            vec![
                Zstd::from_primitive(&ints, 3, 128, &mut ctx)?.into_array(),
                Zstd::from_primitive(&floats, 3, 128, &mut ctx)?.into_array(),
                Zstd::from_primitive(&nullable_i64, 3, 128, &mut ctx)?.into_array(),
                Zstd::from_var_bin_view(&utf8, 3, 128, &mut ctx)?.into_array(),
                Zstd::from_var_bin_view(&nullable_utf8, 3, 128, &mut ctx)?.into_array(),
                Zstd::from_primitive(&all_zeros, 3, 128, &mut ctx)?.into_array(),
                Zstd::from_primitive(&all_null_i32, 3, 128, &mut ctx)?.into_array(),
                Zstd::from_primitive(&pseudo_random, 3, 128, &mut ctx)?.into_array(),
            ],
            N,
            Validity::NonNullable,
        )?;
        Ok(arr.into_array())
    }
}
