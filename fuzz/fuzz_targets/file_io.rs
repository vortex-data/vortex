// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![no_main]

use libfuzzer_sys::Corpus;
use libfuzzer_sys::fuzz_target;
use vortex_fuzz::FileScan;
use vortex_fuzz::FuzzFileAction;
use vortex_fuzz::run_file_io;

fuzz_target!(|fuzz: FuzzFileAction| -> Corpus {
    if run_file_io(fuzz, FileScan::V1) {
        Corpus::Keep
    } else {
        Corpus::Reject
    }
});
