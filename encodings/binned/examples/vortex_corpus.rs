// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compresses a corpus with Vortex's compact BtrBlocks preset under three setups (Pco only,
//! Pco and Binned, Binned only) and reports total size, decode throughput and scalar latency.
//!
//! Usage: `cargo run --release -p vortex-binned --example vortex_corpus -- <corpus dir>`

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::use_debug)]
#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::collections::BTreeMap;
use std::fs;
use std::hint::black_box;
use std::path::Path;
use std::time::Duration;
use std::time::Instant;

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::session::ArraySessionExt;
use vortex_btrblocks::BtrBlocksCompressorBuilder;
use vortex_btrblocks::SchemeExt;
use vortex_btrblocks::schemes::float;
use vortex_btrblocks::schemes::integer;
use vortex_edition::EDITION_DECLARATIONS;
use vortex_edition::EDITION_FAMILIES;
use vortex_edition::EditionSession;
use vortex_edition::EditionSessionExt;
use vortex_edition::declarations::core::CORE_2026_08_3;
use vortex_session::VortexSession;

fn session() -> VortexSession {
    let session = vortex_array::array_session().with::<EditionSession>();
    vortex_alp::initialize(&session);
    vortex_fastlanes::initialize(&session);
    vortex_runend::initialize(&session);
    vortex_sequence::initialize(&session);
    vortex_sparse::initialize(&session);
    vortex_zigzag::initialize(&session);
    vortex_datetime_parts::initialize(&session);
    vortex_decimal_byte_parts::initialize(&session);
    session.arrays().register(vortex_pco::Pco);
    for family in EDITION_FAMILIES {
        session.editions().declare_family(family).unwrap();
    }
    for declaration in EDITION_DECLARATIONS {
        session.register_edition(declaration).unwrap();
    }
    session.enable_edition(CORE_2026_08_3).unwrap();
    vortex_binned::initialize(&session);
    session
        .enable_edition(vortex_binned::editions::BINNED_2026_10)
        .unwrap();
    session
}

fn load(dir: &Path, id: &str, dtype: &str) -> Option<ArrayRef> {
    let bytes = fs::read(dir.join("data").join(format!("{id}.bin"))).ok()?;
    macro_rules! parse {
        ($t:ty, $n:literal) => {
            PrimitiveArray::from_iter(
                bytes
                    .chunks_exact($n)
                    .map(|c| <$t>::from_le_bytes(c.try_into().unwrap())),
            )
            .into_array()
        };
    }
    Some(match dtype {
        "u8" => parse!(u8, 1),
        "i8" => parse!(i8, 1),
        "u16" => parse!(u16, 2),
        "i16" => parse!(i16, 2),
        "u32" => parse!(u32, 4),
        "i32" => parse!(i32, 4),
        "u64" => parse!(u64, 8),
        "i64" => parse!(i64, 8),
        "f32" => parse!(f32, 4),
        "f64" => parse!(f64, 8),
        _ => return None,
    })
}

#[derive(Default)]
struct Totals {
    bytes: usize,
    decode: Duration,
    probe_ns: Vec<f64>,
    encodings: BTreeMap<String, usize>,
}

fn top_encoding(array: &ArrayRef) -> String {
    array.encoding_id().to_string()
}

fn main() {
    let dir = std::env::args().nth(1).expect("usage: vortex_corpus <dir>");
    let dir = Path::new(&dir);
    let session = session();
    let pco = [integer::PcoScheme.id(), float::PcoScheme.id()];
    let binned = [integer::BinnedScheme.id(), float::BinnedScheme.id()];
    let setups = [
        ("pco only", Some(binned.to_vec())),
        ("pco + binned", None),
        ("binned only", Some(pco.to_vec())),
    ];
    let compressors: Vec<_> = setups
        .iter()
        .map(|(name, exclude)| {
            let builder = BtrBlocksCompressorBuilder::from_session(&session).with_compact();
            let builder = match exclude {
                Some(ids) => builder.exclude_schemes(ids.iter().copied()),
                None => builder,
            };
            (*name, builder.build())
        })
        .collect();

    let mut totals: Vec<Totals> = setups.iter().map(|_| Totals::default()).collect();
    let mut raw = 0usize;
    let manifest = fs::read_to_string(dir.join("manifest.jsonl")).expect("manifest");
    for line in manifest.lines().filter(|l| !l.trim().is_empty()) {
        let entry: serde_json::Value = serde_json::from_str(line).unwrap();
        let (Some(id), Some(dtype)) = (entry["id"].as_str(), entry["dtype"].as_str()) else {
            continue;
        };
        let Some(input) = load(dir, id, dtype) else {
            continue;
        };
        raw += input.nbytes() as usize;
        let mut ctx = session.create_execution_ctx();
        let mut row = format!("{id}");
        for ((_, compressor), t) in compressors.iter().zip(&mut totals) {
            let compressed = compressor.compress(&input, &mut ctx).unwrap();
            let bytes = compressed.nbytes() as usize;
            let start = Instant::now();
            let mut runs = 0;
            while runs < 3 || (start.elapsed() < Duration::from_millis(100) && runs < 20) {
                black_box(compressed.clone().execute::<PrimitiveArray>(&mut ctx).unwrap());
                runs += 1;
            }
            t.decode += start.elapsed() / runs;
            let n = input.len();
            let start = Instant::now();
            let mut x = 0x9E37_79B9_7F4A_7C15u64;
            for _ in 0..200 {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                black_box(compressed.execute_scalar((x % n as u64) as usize, &mut ctx).unwrap());
            }
            t.probe_ns
                .push(start.elapsed().as_nanos() as f64 / 200.0);
            t.bytes += bytes;
            *t.encodings.entry(top_encoding(&compressed)).or_default() += 1;
            row.push_str(&format!(",{bytes},{}", top_encoding(&compressed)));
        }
        println!("{row}");
    }

    eprintln!("\nraw bytes: {raw}");
    for ((name, _), t) in compressors.iter().zip(&totals) {
        let geo = (t.probe_ns.iter().map(|x| x.ln()).sum::<f64>() / t.probe_ns.len() as f64).exp();
        eprintln!(
            "{name:>14}: ratio {:.3}  decode {:.2} GB/s  scalar geomean {:.0} ns  top encodings {:?}",
            raw as f64 / t.bytes as f64,
            raw as f64 / t.decode.as_secs_f64() / 1e9,
            geo,
            t.encodings
        );
    }
}
