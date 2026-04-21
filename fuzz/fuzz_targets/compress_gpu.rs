// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![no_main]

use libfuzzer_sys::Corpus;
use libfuzzer_sys::fuzz_target;
use vortex::io::runtime::BlockingRuntime;
use vortex_error::vortex_panic;
use vortex_fuzz::FuzzCompressGpu;
use vortex_fuzz::RUNTIME;
use vortex_fuzz::run_compress_gpu;

fuzz_target!(|fuzz: FuzzCompressGpu| -> Corpus {
    match RUNTIME.block_on(run_compress_gpu(fuzz)) {
        Ok(true) => Corpus::Keep,
        Ok(false) => Corpus::Reject,
        Err(e) => {
            vortex_panic!("{e}");
        }
    }
});
