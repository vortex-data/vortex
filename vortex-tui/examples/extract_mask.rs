// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
// One-off: extract the nested `closure_local_decode_mask_le: list(bool)` column
// from the mp4-index Vortex file and dump each boolean mask as one row of
// tab-separated 0/1 elements, for the list-OnPair experiment.
#![allow(deprecated)]

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use arrow_array::cast::AsArray;
use arrow_array::{Array, BooleanArray};
use vortex::array::arrow::IntoArrowArray;
use vortex::array::stream::ArrayStreamExt;
use vortex::error::VortexResult;
use vortex::file::OpenOptionsSessionExt;
use vortex::io::session::RuntimeSessionExt;
use vortex::session::VortexSession;
use vortex::VortexSessionDefault;

/// Resolve a list array's (offsets, values) regardless of i32/i64 offset width.
fn list_parts(a: &dyn Array) -> (Vec<usize>, std::sync::Arc<dyn Array>) {
    if let Some(l) = a.as_any().downcast_ref::<arrow_array::ListArray>() {
        (l.offsets().iter().map(|&o| o as usize).collect(), l.values().clone())
    } else if let Some(l) = a.as_any().downcast_ref::<arrow_array::LargeListArray>() {
        (l.offsets().iter().map(|&o| o as usize).collect(), l.values().clone())
    } else {
        panic!("not a list array: {:?}", a.data_type());
    }
}

#[tokio::main]
async fn main() -> VortexResult<()> {
    let file = PathBuf::from(std::env::args().nth(1).expect("usage: extract_mask <file> <out>"));
    let out = PathBuf::from(std::env::args().nth(2).expect("usage: extract_mask <file> <out>"));
    let session = VortexSession::default().with_tokio().allow_unknown();

    let array = session
        .open_options()
        .open_path(&file)
        .await?
        .scan()?
        .into_array_stream()?
        .read_all()
        .await?;

    let arrow = array.into_arrow_preferred()?;
    let root = arrow.as_struct();

    // col_v0_tracks: list<struct> -> frames_by_video: list<struct>
    //   -> closure_local_decode_mask_le: list<bool>
    let (_, tracks_vals) = list_parts(root.column_by_name("col_v0_tracks").unwrap());
    let track_struct = tracks_vals.as_struct();
    let (_, fbv_vals) = list_parts(track_struct.column_by_name("frames_by_video").unwrap());
    let frame_struct = fbv_vals.as_struct();
    let mask_col = frame_struct.column_by_name("closure_local_decode_mask_le").unwrap();
    let (offsets, bits) = list_parts(mask_col);
    let bools = bits.as_any().downcast_ref::<BooleanArray>().expect("bool values");

    let mut w = BufWriter::new(File::create(&out)?);
    let n = offsets.len() - 1;
    let mut total = 0usize;
    for i in 0..n {
        let (s, e) = (offsets[i], offsets[i + 1]);
        for j in s..e {
            if j > s {
                w.write_all(b"\t").unwrap();
            }
            w.write_all(if bools.value(j) { b"1" } else { b"0" }).unwrap();
        }
        w.write_all(b"\n").unwrap();
        total += e - s;
    }
    w.flush().unwrap();
    eprintln!("masks={n}  total_bools={total}  avg_len={:.1}", total as f64 / n as f64);
    Ok(())
}
