// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex::array::ArrayId;
use vortex::array::ArrayRef;
use vortex::array::ArrayVTable;
use vortex::array::IntoArray;
use vortex::array::arrays::BoolArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::StructArray;
use vortex::array::arrays::VarBinArray;
use vortex::array::dtype::FieldNames;
use vortex::array::validity::Validity;
use vortex::encodings::runend::RunEnd;
use vortex::encodings::runend::compress::runend_encode;
use vortex::error::VortexResult;

use super::N;
use crate::fixtures::FlatLayoutFixture;

pub struct RunEndFixture;

impl FlatLayoutFixture for RunEndFixture {
    fn name(&self) -> &str {
        "runend.vortex"
    }

    fn description(&self) -> &str {
        "Data with variable-length runs for RunEnd encoding"
    }

    fn expected_encodings(&self) -> Vec<ArrayId> {
        vec![RunEnd.id()]
    }

    fn build(&self) -> VortexResult<ArrayRef> {
        let run_lengths = [1usize, 5, 10, 50, 100];
        let mut values = Vec::with_capacity(N);
        let mut run_idx = 0i64;
        let mut rl_idx = 0;
        while values.len() < N {
            let run_len = run_lengths[rl_idx % run_lengths.len()].min(N - values.len());
            for _ in 0..run_len {
                values.push(run_idx);
            }
            run_idx += 1;
            rl_idx += 1;
        }
        let run_prim: PrimitiveArray = values.into_iter().collect();
        let (run_ends, run_values) = runend_encode(run_prim.as_view());
        let run_col = RunEnd::try_new(run_ends.into_array(), run_values)?;

        let statuses = ["open", "closed", "pending", "cancelled"];
        let mut status_values = Vec::new();
        let mut status_ends = Vec::new();
        let mut s_idx = 0;
        let mut remaining = N;
        let mut status_end = 0u16;
        while remaining > 0 {
            let run_len = (32 + s_idx * 7 % 64).min(remaining);
            status_values.push(statuses[s_idx % statuses.len()]);
            status_end += run_len as u16;
            status_ends.push(status_end);
            s_idx += 1;
            remaining -= run_len;
        }
        let status_ends_prim: PrimitiveArray = status_ends.into_iter().collect();
        let status_col = RunEnd::try_new(
            status_ends_prim.into_array(),
            VarBinArray::from_strs(status_values).into_array(),
        )?;

        let uniform_prim: PrimitiveArray = (0..N as i32).map(|i| i / 64).collect();
        let (uniform_ends, uniform_values) = runend_encode(uniform_prim.as_view());
        let uniform_col = RunEnd::try_new(uniform_ends.into_array(), uniform_values)?;

        let bool_ends: PrimitiveArray = (1..=N / 32).map(|i| (i * 32) as u16).collect();
        let bool_values =
            BoolArray::from_iter((0..bool_ends.len()).map(|i| i % 2 == 0)).into_array();
        let bool_runs = RunEnd::try_new(bool_ends.into_array(), bool_values)?;
        let nullable_run_values = PrimitiveArray::from_option_iter([
            Some(10i32),
            None,
            Some(-5),
            Some(77),
            None,
            Some(0),
        ]);
        let nullable_runs = RunEnd::try_new(
            PrimitiveArray::from_iter([16u16, 64, 128, 256, 512, N as u16]).into_array(),
            nullable_run_values.into_array(),
        )?;
        let single_run = RunEnd::try_new(
            PrimitiveArray::from_iter([N as u64]).into_array(),
            PrimitiveArray::from_iter([1234i64]).into_array(),
        )?;
        let singleton_values: PrimitiveArray = (0..N as i16).map(|i| i - 512).collect();
        let singleton_ends: PrimitiveArray = (1..=N as u16).collect();
        let alternating_singletons =
            RunEnd::try_new(singleton_ends.into_array(), singleton_values.into_array())?;

        let arr = StructArray::try_new(
            FieldNames::from([
                "run_values",
                "statuses",
                "uniform_runs",
                "bool_runs",
                "nullable_runs",
                "single_run",
                "alternating_singletons",
            ]),
            vec![
                run_col.into_array(),
                status_col.into_array(),
                uniform_col.into_array(),
                bool_runs.into_array(),
                nullable_runs.into_array(),
                single_run.into_array(),
                alternating_singletons.into_array(),
            ],
            N,
            Validity::NonNullable,
        )?;
        Ok(arr.into_array())
    }
}
