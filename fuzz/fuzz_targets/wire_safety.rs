// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![no_main]

use libfuzzer_sys::Corpus;
use libfuzzer_sys::fuzz_target;
use vortex_fuzz::FuzzWireSafety;
use vortex_fuzz::run_wire_safety;

fuzz_target!(|input: FuzzWireSafety| -> Corpus {
    run_wire_safety(input);
    Corpus::Keep
});
