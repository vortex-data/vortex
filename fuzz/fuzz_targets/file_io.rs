// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![no_main]

use libfuzzer_sys::Corpus;
use libfuzzer_sys::fuzz_target;
use vortex_fuzz::FuzzFileAction;
use vortex_fuzz::run_file_io;

fuzz_target!(|fuzz: FuzzFileAction| -> Corpus {
    if run_file_io(fuzz) {
        Corpus::Keep
    } else {
        Corpus::Reject
    }
});
