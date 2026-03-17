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
use vortex::encodings::runend::RunEnd;
use vortex::encodings::runend::RunEndArray;
use vortex::encodings::runend::compress::runend_encode;
use vortex::error::VortexResult;

use super::N;
use crate::fixtures::ArrayFixture;

pub struct RunEndFixture;

impl ArrayFixture for RunEndFixture {
    fn name(&self) -> &str {
        "runend.vortex"
    }

    fn description(&self) -> &str {
        "Data with variable-length runs for RunEnd encoding"
    }

    fn expected_encodings(&self) -> Vec<ArrayId> {
        vec![RunEnd::ID]
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
        let run_prim = PrimitiveArray::new(Buffer::from(values), Validity::NonNullable);
        let (run_ends, run_values) = runend_encode(&run_prim);
        let run_col = RunEndArray::try_new(run_ends.into_array(), run_values)?;

        let statuses = ["open", "closed", "pending", "cancelled"];
        let mut status_values = Vec::with_capacity(N);
        let mut s_idx = 0;
        let mut remaining = N;
        while remaining > 0 {
            let run_len = (32 + s_idx * 7 % 64).min(remaining);
            for _ in 0..run_len {
                status_values.push(statuses[s_idx % statuses.len()]);
            }
            s_idx += 1;
            remaining -= run_len;
        }
        let status_col = VarBinArray::from(status_values);

        let uniform_runs: Vec<i32> = (0..N as i32).map(|i| i / 64).collect();
        let uniform_prim = PrimitiveArray::new(Buffer::from(uniform_runs), Validity::NonNullable);
        let (uniform_ends, uniform_values) = runend_encode(&uniform_prim);
        let uniform_col = RunEndArray::try_new(uniform_ends.into_array(), uniform_values)?;

        let bool_runs = BoolArray::from_iter((0..N).map(|i| (i / 32) % 2 == 0));

        let arr = StructArray::try_new(
            FieldNames::from(["run_values", "statuses", "uniform_runs", "bool_runs"]),
            vec![
                run_col.into_array(),
                status_col.into_array(),
                uniform_col.into_array(),
                bool_runs.into_array(),
            ],
            N,
            Validity::NonNullable,
        )?;
        Ok(arr.into_array())
    }
}
