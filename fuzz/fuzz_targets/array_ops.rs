// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![no_main]
#![allow(clippy::unwrap_used, clippy::result_large_err)]

use libfuzzer_sys::Corpus;
use libfuzzer_sys::fuzz_target;
use vortex_error::vortex_panic;
use vortex_fuzz::FuzzArrayAction;
use vortex_fuzz::run_fuzz_action;

fuzz_target!(
    init: {
        tracing_subscriber::fmt::init();
    },
    |fuzz_action: FuzzArrayAction| -> Corpus {
    match run_fuzz_action(fuzz_action) {
        Ok(true) => Corpus::Keep,
        Ok(false) => Corpus::Reject,
        Err(e) => {
            vortex_panic!("{e}");
        }
    }
});
