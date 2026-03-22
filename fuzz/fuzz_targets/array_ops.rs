// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![no_main]
#![allow(clippy::unwrap_used, clippy::result_large_err)]

use std::str::FromStr;

use libfuzzer_sys::Corpus;
use libfuzzer_sys::fuzz_target;
use tracing::level_filters::LevelFilter;
use vortex_error::vortex_panic;
use vortex_fuzz::FuzzArrayAction;
use vortex_fuzz::run_fuzz_action;

fuzz_target!(
    init: {
        let fmt = tracing_subscriber::fmt::format()
            .with_ansi(false) // Colour output is messed up in raw logs
            .without_time() // We run fuzzer in CI which prepends timestamps
            .compact();
        let level = std::env::var("RUST_LOG").map(
            |v| LevelFilter::from_str(v.as_str()).unwrap()).unwrap_or(LevelFilter::INFO);
        tracing_subscriber::fmt()
            .event_format(fmt)
            .with_max_level(level)
            .init();
    },
    |fuzz_action: FuzzArrayAction| -> Corpus {
    match run_fuzz_action(fuzz_action) {
        Ok(true) => Corpus::Keep,
        Ok(false) => Corpus::Reject,
        Err(e) => vortex_panic!("{e}"),
    }
});
