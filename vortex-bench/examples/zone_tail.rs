// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Temporary probe: where do zone map segments sit relative to EOF?
#![allow(clippy::unwrap_used, clippy::cast_precision_loss, clippy::use_debug)]

use vortex::file::OpenOptionsSessionExt;
use vortex::layout::LayoutRef;
use vortex_bench::SESSION;

fn walk(layout: &LayoutRef, path: String, in_zones: bool, out: &mut Vec<(String, bool, u32)>) {
    for id in layout.segment_ids() {
        out.push((path.clone(), in_zones, *id));
    }
    let is_zoned = layout.encoding_id().as_ref() == "vortex.zoned";
    for slot in 0..layout.nslots() {
        if let Some(child) = layout.slot(slot).unwrap() {
            let name = layout.slot_type(slot).unwrap().name();
            walk(
                &child,
                format!("{path}/{name}"),
                in_zones || (is_zoned && slot == 1),
                out,
            );
        }
    }
}

#[tokio::main]
async fn main() {
    let mut tails = vec![];
    for path in std::env::args().skip(1) {
        let vxf = SESSION.open_options().open_path(&path).await.unwrap();
        let file_size = std::fs::metadata(&path).unwrap().len();
        let footer = vxf.footer();
        let specs = footer.segment_map();
        let data_end = specs.iter().map(|s| s.offset + s.length as u64).max().unwrap_or(0);
        let mut segs = vec![];
        walk(footer.layout(), String::new(), false, &mut segs);
        let zones: Vec<_> = segs
            .iter()
            .filter(|(_, z, _)| *z)
            .map(|(p, _, id)| (p.clone(), specs[*id as usize]))
            .collect();
        let zone_bytes: u64 = zones.iter().map(|(_, s)| s.length as u64).sum();
        let need = zones.iter().map(|(_, s)| file_size - s.offset).max().unwrap_or(0);
        let in_64k = zones.iter().filter(|(_, s)| file_size - s.offset <= 65_535 + 8).count();
        // Bytes of non-zone data interleaved inside the tail that would cover all zone maps.
        let min_zone_off = zones.iter().map(|(_, s)| s.offset).min().unwrap_or(file_size);
        let non_zone_in_tail: u64 = segs
            .iter()
            .filter(|(_, z, _)| !*z)
            .map(|(_, _, id)| specs[*id as usize])
            .filter(|s| s.offset >= min_zone_off)
            .map(|s| s.length as u64)
            .sum();
        println!(
            "{path}\n  file={file_size} footer_tail={} segments={} zone_segs={} zone_bytes={zone_bytes} \
             zone_segs_within_64KiB={in_64k} tail_to_cover_all_zones={need} non_zone_bytes_in_that_tail={non_zone_in_tail}",
            file_size - data_end,
            specs.len(),
            zones.len(),
        );
        if std::env::var("VERBOSE").is_ok() {
            for (p, s) in &zones {
                println!("    {p:<40} len={:>8} from_eof={:>12}", s.length, file_size - s.offset);
            }
        }
        tails.push(need);
    }
}
