// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![no_main]
#![expect(clippy::unwrap_used)]

use libfuzzer_sys::Corpus;
use libfuzzer_sys::fuzz_target;
use vortex_error::vortex_panic;
use vortex_fuzz::FuzzCompressGpu;
use vortex_fuzz::RUNTIME;
use vortex_fuzz::run_compress_gpu;
use vortex_io::runtime::blocking::BlockingRuntime;

fuzz_target!(|fuzz: FuzzCompressGpu| -> Corpus {
    match RUNTIME.block_on(run_compress_gpu(fuzz)) {
        Ok(true) => Corpus::Keep,
        Ok(false) => Corpus::Reject,
        Err(e) => {
            vortex_panic!("{e}");
        }
    }
});
