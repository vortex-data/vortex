// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![no_main]

use libfuzzer_sys::Corpus;
use libfuzzer_sys::fuzz_target;
use vortex_error::vortex_panic;
use vortex_fuzz::FuzzRowOrder;
use vortex_fuzz::run_row_order_fuzz;

fuzz_target!(|fuzz: FuzzRowOrder| -> Corpus {
    match run_row_order_fuzz(fuzz) {
        Ok(true) => Corpus::Keep,
        Ok(false) => Corpus::Reject,
        Err(e) => vortex_panic!("{e}"),
    }
});
