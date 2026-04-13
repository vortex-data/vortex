// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex::array::ArrayId;
use vortex::array::ArrayRef;
use vortex::array::IntoArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::StructArray;
use vortex::array::dtype::FieldNames;
use vortex::array::validity::Validity;
use vortex::encodings::fastlanes::RLE;
use vortex::error::VortexResult;

use super::N;
use crate::fixtures::FlatLayoutFixture;

pub struct RleFixture;

impl FlatLayoutFixture for RleFixture {
    fn name(&self) -> &str {
        "rle.vortex"
    }

    fn description(&self) -> &str {
        "Primitive data with long runs of repeated values for RLE encoding"
    }

    fn expected_encodings(&self) -> Vec<ArrayId> {
        vec![RLE::ID]
    }

    fn build(&self) -> VortexResult<ArrayRef> {
        let runs_i32: PrimitiveArray = (0..N as i32).map(|i| i / 64).collect();
        let single_run: PrimitiveArray = std::iter::repeat_n(42u64, N).collect();
        let nullable_runs = PrimitiveArray::from_option_iter(
            (0..N as i32).map(|i| (i / 64 % 3 != 0).then_some(i / 64 * 10)),
        );
        let alternating_singletons: PrimitiveArray = (0..N as u16).collect();
        let boundary_run_lengths = [31usize, 32, 63, 64, 127, 128];
        let mut exact_boundary_runs_vec = Vec::with_capacity(N);
        let mut boundary_value = 0u16;
        while exact_boundary_runs_vec.len() < N {
            for run_len in boundary_run_lengths {
                let take = run_len.min(N - exact_boundary_runs_vec.len());
                exact_boundary_runs_vec.extend(std::iter::repeat_n(boundary_value, take));
                boundary_value = boundary_value.wrapping_add(1);
                if exact_boundary_runs_vec.len() == N {
                    break;
                }
            }
        }
        let exact_boundary_runs: PrimitiveArray = exact_boundary_runs_vec.into_iter().collect();
        let giant_final_run: PrimitiveArray = (0..N as u32)
            .map(|i| if i < 32 { i } else { 999 })
            .collect();
        let all_null_i32 = PrimitiveArray::from_option_iter((0..N).map(|_| None::<i32>));
        let short_runs_u8: PrimitiveArray = (0..N).map(|i| (i / 16) as u8).collect();

        let arr = StructArray::try_new(
            FieldNames::from([
                "runs_i32",
                "single_run",
                "nullable_runs",
                "alternating_singletons",
                "exact_boundary_runs",
                "giant_final_run",
                "all_null_i32",
                "short_runs_u8",
            ]),
            vec![
                RLE::encode(runs_i32.as_view())?.into_array(),
                RLE::encode(single_run.as_view())?.into_array(),
                RLE::encode(nullable_runs.as_view())?.into_array(),
                RLE::encode(alternating_singletons.as_view())?.into_array(),
                RLE::encode(exact_boundary_runs.as_view())?.into_array(),
                RLE::encode(giant_final_run.as_view())?.into_array(),
                RLE::encode(all_null_i32.as_view())?.into_array(),
                RLE::encode(short_runs_u8.as_view())?.into_array(),
            ],
            N,
            Validity::NonNullable,
        )?;
        Ok(arr.into_array())
    }
}
