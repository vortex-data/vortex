// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Reassembling primitive decimal parts across storage widths and lengths.

mod common;

use divan::Bencher;
use divan::black_box;
use vortex_array::VortexSessionExecute;
use vortex_array::array_session;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::DecimalType;
use vortex_array::validity::Validity;
use vortex_decimal_byte_parts::_benchmarking::assemble_decimal;
use vortex_decimal_byte_parts::split_decimal;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::common::cases;
use crate::common::decimal_array;

fn main() {
    divan::main();
}

#[divan::bench(args = cases())]
fn dbp_assemble(bencher: Bencher, (values_type, len): (DecimalType, usize)) {
    let decimal = decimal_array(values_type, len, Validity::NonNullable);
    let mut ctx = array_session().create_execution_ctx();
    let parts = split_decimal(&decimal, &mut ctx).vortex_expect("split benchmark input");
    let msp = parts
        .msp
        .execute::<PrimitiveArray>(&mut ctx)
        .vortex_expect("execute benchmark MSP");
    let lower_parts = parts
        .lower_parts
        .into_iter()
        .map(|part| part.execute::<PrimitiveArray>(&mut ctx))
        .collect::<VortexResult<Vec<_>>>()
        .vortex_expect("execute benchmark lower parts");
    let decimal_dtype = decimal.decimal_dtype();

    bencher.bench(|| {
        assemble_decimal(black_box(&msp), black_box(&lower_parts), decimal_dtype)
            .vortex_expect("assemble decimal byte parts")
    });
}
