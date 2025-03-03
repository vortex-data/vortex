use vortex::Array;
use vortex::arrays::PrimitiveArray;
use vortex::buffer::buffer;
use vortex::compressor::BtrBlocksCompressor;
use vortex::validity::Validity;

//use wasm_bindgen::prelude::*;

pub fn main() {
    // Extremely simple test of compression/decompression and a few compute functions.
    let array = PrimitiveArray::new(buffer![1i32; 1024], Validity::AllValid).to_array();

    let compressed = BtrBlocksCompressor.compress(&array).unwrap();
    println!("Compressed size: {}", compressed.len());
    println!("Tree view: {}", compressed.tree_display());
}
