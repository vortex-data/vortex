// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![no_main]
#![expect(clippy::unwrap_used)]

use libfuzzer_sys::Corpus;
use libfuzzer_sys::fuzz_target;
use vortex_error::vortex_panic;
use vortex_fuzz::FuzzCompressGpu;
use vortex_fuzz::run_compress_gpu;

fuzz_target!(|fuzz: FuzzCompressGpu| -> Corpus {
    // Use tokio runtime to run async GPU fuzzer
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    match rt.block_on(run_compress_gpu(fuzz)) {
        Ok(true) => Corpus::Keep,
        Ok(false) => Corpus::Reject,
        Err(e) => {
            vortex_panic!("{e}");
        }
    }
});
