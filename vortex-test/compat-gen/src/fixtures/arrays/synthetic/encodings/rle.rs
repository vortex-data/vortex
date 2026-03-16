// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex::array::ArrayRef;
use vortex::array::IntoArray;
use vortex::array::arrays::BoolArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::StructArray;
use vortex::array::arrays::VarBinArray;
use vortex::array::dtype::FieldNames;
use vortex::array::validity::Validity;
use vortex::array::vtable::ArrayId;
use vortex::buffer::Buffer;
use vortex::encodings::fastlanes::RLE;
use vortex::encodings::fastlanes::RLEArray;
use vortex::error::VortexResult;

use super::N;
use crate::fixtures::ArrayFixture;

pub struct RleFixture;

impl ArrayFixture for RleFixture {
    fn name(&self) -> &str {
        "rle.vortex"
    }

    fn description(&self) -> &str {
        "Data with long runs of repeated values for RLE encoding"
    }

    fn expected_encodings(&self) -> Vec<ArrayId> {
        vec![RLE::ID]
    }

    fn build(&self) -> VortexResult<ArrayRef> {
        let runs_i32: Vec<i32> = (0..N as i32).map(|i| i / 64).collect();
        let labels = ["active", "inactive", "pending"];
        let runs_str: Vec<&str> = (0..N).map(|i| labels[i / 341 % labels.len()]).collect();
        let str_col = VarBinArray::from(runs_str);
        let runs_bool = BoolArray::from_iter((0..N).map(|i| (i / 128) % 2 == 0));
        let single_run: Vec<u64> = vec![42u64; N];
        let nullable_runs = PrimitiveArray::from_option_iter(
            (0..N as i32).map(|i| (i / 64 % 3 != 0).then_some(i / 64 * 10)),
        );

        let arr = StructArray::try_new(
            FieldNames::from([
                "runs_i32",
                "runs_str",
                "runs_bool",
                "single_run",
                "nullable_runs",
            ]),
            vec![
                RLEArray::encode(&PrimitiveArray::new(
                    Buffer::from(runs_i32),
                    Validity::NonNullable,
                ))?
                .into_array(),
                str_col.into_array(),
                runs_bool.into_array(),
                RLEArray::encode(&PrimitiveArray::new(
                    Buffer::from(single_run),
                    Validity::NonNullable,
                ))?
                .into_array(),
                RLEArray::encode(&nullable_runs)?.into_array(),
            ],
            N,
            Validity::NonNullable,
        )?;
        Ok(arr.into_array())
    }
}
