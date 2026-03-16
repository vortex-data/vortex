// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex::array::ArrayRef;
use vortex::array::IntoArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::StructArray;
use vortex::array::arrays::VarBinViewArray;
use vortex::array::dtype::FieldNames;
use vortex::array::validity::Validity;
use vortex::array::vtable::ArrayId;
use vortex::buffer::Buffer;
use vortex::encodings::zstd::Zstd;
use vortex::encodings::zstd::ZstdArray;
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
        vec![Zstd::ID]
    }

    fn build(&self) -> VortexResult<ArrayRef> {
        let ints: Vec<i32> = (0..N as i32).map(|i| i / 8).collect();
        let floats: Vec<f64> = (0..N)
            .map(|i| {
                if i % 257 == 0 {
                    100_000.0 + i as f64
                } else {
                    3.14159
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

        let arr = StructArray::try_new(
            FieldNames::from(["ints", "floats", "nullable_i64", "utf8", "nullable_utf8"]),
            vec![
                ZstdArray::from_primitive(
                    &PrimitiveArray::new(Buffer::from(ints), Validity::NonNullable),
                    3,
                    128,
                )?
                .into_array(),
                ZstdArray::from_primitive(
                    &PrimitiveArray::new(Buffer::from(floats), Validity::NonNullable),
                    3,
                    128,
                )?
                .into_array(),
                ZstdArray::from_primitive(&nullable_i64, 3, 128)?.into_array(),
                ZstdArray::from_var_bin_view(&utf8, 3, 128)?.into_array(),
                ZstdArray::from_var_bin_view(&nullable_utf8, 3, 128)?.into_array(),
            ],
            N,
            Validity::NonNullable,
        )?;
        Ok(arr.into_array())
    }
}
