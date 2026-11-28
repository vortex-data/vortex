// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![no_main]

use libfuzzer_sys::Corpus;
use libfuzzer_sys::fuzz_target;
use vortex_fuzz::file::FileRoundTrip;

fuzz_target!(|fuzz: FileRoundTrip| -> Corpus {
    if fuzz.check() {
        Corpus::Keep
    } else {
        Corpus::Reject
    }
});
