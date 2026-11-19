// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![no_main]

use libfuzzer_sys::fuzz_target;
use vortex_fuzz::arrow_roundtrip::ArrowRoundTrip;

fuzz_target!(|case: ArrowRoundTrip| case.check());
