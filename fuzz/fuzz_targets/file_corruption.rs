// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![no_main]

use libfuzzer_sys::Corpus;
use libfuzzer_sys::fuzz_target;
use vortex_error::vortex_panic;
use vortex_fuzz::FuzzFileCorruption;
use vortex_fuzz::run_file_corruption;

fuzz_target!(|input: FuzzFileCorruption| -> Corpus {
    match run_file_corruption(input) {
        Ok(true) => Corpus::Keep,
        Ok(false) => Corpus::Reject,
        Err(e) => vortex_panic!("{e}"),
    }
});
