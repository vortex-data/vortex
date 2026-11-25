// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex::VortexSessionDefault;
use vortex::array::IntoArray;
use vortex::array::VortexSessionExecute;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::validity::Validity;
use vortex::buffer::buffer;
use vortex::compressor::BtrBlocksCompressor;
use vortex::session::VortexSession;

// use wasm_bindgen::prelude::*;

pub fn main() {
    // Extremely simple test of compression/decompression and a few compute functions.
    let array = PrimitiveArray::new(buffer![1i32; 1024], Validity::AllValid).into_array();

    let session = VortexSession::default();
    let mut ctx = session.create_execution_ctx();
    let compressed = BtrBlocksCompressor::from_session(&session)
        .compress(&array, &mut ctx)
        .unwrap();
    println!("Compressed size: {}", compressed.len());
    println!("Tree view: {}", compressed.display_tree());

    let decompressed = compressed.execute::<PrimitiveArray>(&mut ctx).unwrap();
    assert_eq!(decompressed.dtype(), array.dtype());
    assert_eq!(decompressed.as_slice::<i32>(), &[1i32; 1024]);
}
