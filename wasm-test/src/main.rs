// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex::array::IntoArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::validity::Validity;
use vortex::buffer::buffer;
use vortex::compressor::BtrBlocksCompressor;

//use wasm_bindgen::prelude::*;

pub fn main() {
    // Extremely simple test of compression/decompression and a few compute functions.
    let array = PrimitiveArray::new(buffer![1i32; 1024], Validity::AllValid).into_array();

    let compressed = BtrBlocksCompressor::default().compress(&array).unwrap();
    println!("Compressed size: {}", compressed.len());
    println!("Tree view: {}", compressed.display_tree());
}
