// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Bytes of the row-dictionary codes under the compact preset and under zstd at several levels.

#![allow(clippy::unwrap_used, clippy::cast_possible_truncation)]

use std::sync::LazyLock;
use std::time::Instant;

use mimalloc::MiMalloc;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::Dict;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::dict::DictArraySlotsExt;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_shredded_map::ShredOptions;
use vortex_zstd::Zstd;

#[path = "../benches/common/mod.rs"]
mod common;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    let data = LazyLock::force(&common::DATA);
    let mut ctx = common::SESSION.create_execution_ctx();
    let opts = ShredOptions { max_distinct_rows: 1.0, ..ShredOptions::default() };
    let encoded = vortex_shredded_map::encode(&data.map, &opts, &mut ctx).unwrap();
    let dict = encoded.as_opt::<Dict>().unwrap();
    let codes = dict.codes().clone().execute::<PrimitiveArray>(&mut ctx).unwrap();
    let values: Vec<u32> = codes.as_slice::<u32>().to_vec();
    let distinct = dict.values().len();
    let narrow = if distinct <= 256 {
        PrimitiveArray::new(values.iter().map(|&v| v as u8).collect::<Buffer<u8>>(), Validity::NonNullable)
    } else if distinct <= 65536 {
        PrimitiveArray::new(values.iter().map(|&v| v as u16).collect::<Buffer<u16>>(), Validity::NonNullable)
    } else {
        codes.clone()
    };
    println!("rows {} distinct rows {distinct} ptype {}", values.len(), narrow.ptype());
    let compact = common::compress_with(&narrow.clone().into_array(), true);
    println!("  compact preset ({}): {} B", compact.encoding_id(), compact.nbytes());
    for level in [3, 9, 19, 22] {
        for frame in [0usize, 1 << 16] {
            let t = Instant::now();
            let z = Zstd::from_primitive(&narrow, level, frame, &mut ctx).unwrap();
            let bytes: usize = z.into_array().nbytes() as usize;
            println!("  zstd level {level:>2} frame {:>6}: {bytes} B ({:.2}s)", if frame == 0 { "whole".into() } else { frame.to_string() }, t.elapsed().as_secs_f64());
        }
    }
}
