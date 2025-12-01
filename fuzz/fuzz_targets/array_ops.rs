// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![no_main]
#![allow(clippy::unwrap_used, clippy::result_large_err)]

use libfuzzer_sys::Corpus;
use libfuzzer_sys::fuzz_target;
use vortex_array::Array;
use vortex_array::ArrayRef;
use vortex_btrblocks::BtrBlocksCompressor;
use vortex_error::VortexUnwrap;
use vortex_error::vortex_panic;
use vortex_fuzz::FuzzArrayAction;
use vortex_fuzz::FuzzCompressor;
use vortex_fuzz::run_fuzz_action;
use vortex_layout::layouts::compact::CompactCompressor;

/// Native compressor that supports both BtrBlocks and Compact strategies.
struct NativeCompressor;

impl FuzzCompressor for NativeCompressor {
    fn compress_default(&self, array: &dyn Array) -> ArrayRef {
        BtrBlocksCompressor::default()
            .compress(array)
            .vortex_unwrap()
    }

    fn compress_compact(&self, array: &dyn Array) -> ArrayRef {
        CompactCompressor::default().compress(array).vortex_unwrap()
    }
}

fuzz_target!(|fuzz_action: FuzzArrayAction| -> Corpus {
    match run_fuzz_action(fuzz_action, &NativeCompressor) {
        Ok(true) => Corpus::Keep,
        Ok(false) => Corpus::Reject,
        Err(e) => {
            vortex_panic!("{e}");
        }
    }
});
