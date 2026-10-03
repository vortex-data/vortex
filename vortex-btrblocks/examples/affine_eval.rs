// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compare the Affine scheme against the default cascade on real integer columns.
//!
//! Each column is a raw little-endian `i64` file listed in a manifest TSV whose first two fields
//! are the file stem and the row count:
//!
//! ```text
//! cargo run --release -p vortex-btrblocks --features pco --example affine_eval -- \
//!     <dir with manifest.tsv and *.i64> [column filter...]
//! ```
//!
//! For every column and compressor configuration it prints a TSV row with the compressed bits per
//! value, the encoding tree, compression time, full decode time and `scalar_at` time, after
//! checking that the compressed array decodes back to the input.

#![expect(clippy::unwrap_used, clippy::expect_used)]

use std::fs;
use std::hint::black_box;
use std::path::PathBuf;
use std::sync::LazyLock;
use std::time::Duration;
use std::time::Instant;

use mimalloc::MiMalloc;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::validity::Validity;
use vortex_btrblocks::BtrBlocksCompressor;
use vortex_btrblocks::BtrBlocksCompressorBuilder;
use vortex_btrblocks::SchemeExt;
use vortex_btrblocks::schemes::integer::AffineScheme;
use vortex_btrblocks::schemes::integer::DeltaScheme;
use vortex_buffer::Buffer;
use vortex_fastlanes::Affine;
use vortex_fastlanes::AffineArraySlotsExt;
use vortex_fastlanes::AffineOptions;
use vortex_session::VortexSession;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = vortex_array::array_session();
    vortex_fastlanes::initialize(&session);
    session
});

static AFFINE_AUTO: AffineScheme = AffineScheme::auto();
static AFFINE_SCALE: AffineScheme = AffineScheme::fixed(AffineOptions::SCALE);
static AFFINE_SLOPE: AffineScheme = AffineScheme::fixed(AffineOptions::SLOPE);
static AFFINE_ALL: AffineScheme = AffineScheme::fixed(AffineOptions::ALL);
static AFFINE_LECO: AffineScheme = AffineScheme::fixed(AffineOptions::LECO);

fn configs() -> Vec<(&'static str, BtrBlocksCompressor)> {
    let base = || BtrBlocksCompressorBuilder::from_session(&SESSION).unrestricted();
    vec![
        ("default", base().build()),
        (
            "no-delta",
            base()
                .exclude_schemes([DeltaScheme::default().id()])
                .build(),
        ),
        ("affine-auto", base().with_new_scheme(&AFFINE_AUTO).build()),
        ("affine-scale", base().with_new_scheme(&AFFINE_SCALE).build()),
        ("affine-slope", base().with_new_scheme(&AFFINE_SLOPE).build()),
        ("affine-all", base().with_new_scheme(&AFFINE_ALL).build()),
        ("leco-vortex", base().with_new_scheme(&AFFINE_LECO).build()),
        ("compact", base().with_compact().build()),
        (
            "compact+affine",
            base().with_compact().with_new_scheme(&AFFINE_AUTO).build(),
        ),
    ]
}

/// The encoding tree to `depth` levels, with Affine's mode spelled out.
fn describe(array: &ArrayRef, depth: usize) -> String {
    let id = array.encoding_id().to_string();
    let mut name = id.rsplit('.').next().unwrap_or(&id).to_string();
    if let Some(affine) = array.as_opt::<Affine>() {
        let scale = affine.scales().as_constant().is_none() || !is_one(affine.scales());
        let slope = affine.slopes().as_constant().is_none() || !is_zero(affine.slopes());
        name = format!(
            "affine[{}{}]",
            if scale { "gcd" } else { "" },
            if slope { "+slope" } else { "" }
        );
    }
    if depth == 0 || array.nchildren() == 0 {
        return name;
    }
    let children: Vec<String> = array
        .children()
        .iter()
        .filter(|c| c.as_constant().is_none())
        .map(|c| describe(c, depth - 1))
        .collect();
    if children.is_empty() {
        name
    } else {
        format!("{name}({})", children.join(","))
    }
}

fn is_one(array: &ArrayRef) -> bool {
    array
        .as_constant()
        .and_then(|s| s.as_primitive().as_::<i64>())
        .is_some_and(|v| v == 1)
}

fn is_zero(array: &ArrayRef) -> bool {
    array
        .as_constant()
        .and_then(|s| s.as_primitive().as_::<i64>())
        .is_some_and(|v| v == 0)
}

fn time<R>(mut f: impl FnMut() -> R) -> (Duration, R) {
    let mut best = Duration::MAX;
    let mut out = None;
    for _ in 0..3 {
        let start = Instant::now();
        let r = f();
        best = best.min(start.elapsed());
        out = Some(r);
    }
    (best, out.unwrap())
}

fn main() {
    let mut args = std::env::args().skip(1);
    let dir = PathBuf::from(args.next().expect("usage: affine_eval <dir> [filters...]"));
    let filters: Vec<String> = args.collect();
    let manifest = fs::read_to_string(dir.join("manifest.tsv")).unwrap();
    // `AFFINE_EVAL_CONFIGS=default,affine-auto` restricts the run to the named configurations.
    let only = std::env::var("AFFINE_EVAL_CONFIGS").ok();
    let configs: Vec<_> = configs()
        .into_iter()
        .filter(|(name, _)| only.as_ref().is_none_or(|o| o.split(',').any(|c| c == *name)))
        .collect();

    println!("column\trows\tconfig\tbits_per_value\tencoding\tcompress_ms\tdecode_ns_per_value\tscalar_at_ns");
    for line in manifest.lines() {
        let name = line.split('\t').next().unwrap();
        if !filters.is_empty() && !filters.iter().any(|f| name.contains(f.as_str())) {
            continue;
        }
        let bytes = fs::read(dir.join(format!("{name}.i64"))).unwrap();
        let values: Buffer<i64> = bytes
            .chunks_exact(8)
            .map(|b| i64::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let n = values.len();
        if n == 0 {
            continue;
        }
        let input = PrimitiveArray::new(values.clone(), Validity::NonNullable).into_array();
        let probes: Vec<usize> = (0..1000u64)
            .map(|i| (i.wrapping_mul(0x9E37_79B9_7F4A_7C15) % n as u64) as usize)
            .collect();

        for (config, compressor) in &configs {
            let mut ctx = SESSION.create_execution_ctx();
            let start = Instant::now();
            let compressed = compressor.compress(&input, &mut ctx).unwrap();
            let compress_ms = start.elapsed().as_secs_f64() * 1e3;

            let (mut decode, decoded) = time(|| {
                compressed
                    .clone()
                    .execute::<PrimitiveArray>(&mut ctx)
                    .unwrap()
            });
            // `AFFINE_EVAL_SLICE=65536` times decoding cache-sized slices instead, which keeps
            // allocation and page faults of the full output out of the kernel comparison.
            if let Some(slice) = std::env::var("AFFINE_EVAL_SLICE")
                .ok()
                .and_then(|s| s.parse::<usize>().ok())
            {
                decode = (0..5)
                    .map(|_| {
                        let start = Instant::now();
                        for begin in (0..n).step_by(slice) {
                            let part = compressed.slice(begin..(begin + slice).min(n)).unwrap();
                            black_box(part.execute::<PrimitiveArray>(&mut ctx).unwrap());
                        }
                        start.elapsed()
                    })
                    .min()
                    .unwrap();
            }
            assert_eq!(
                decoded.as_slice::<i64>(),
                values.as_slice(),
                "{name} {config} does not round-trip"
            );
            let (scalar_at, _) = time(|| {
                for &i in &probes {
                    black_box(compressed.execute_scalar(i, &mut ctx).unwrap());
                }
            });

            println!(
                "{name}\t{n}\t{config}\t{:.3}\t{}\t{:.1}\t{:.3}\t{:.0}",
                compressed.nbytes() as f64 * 8.0 / n as f64,
                describe(&compressed, 3),
                compress_ms,
                decode.as_secs_f64() * 1e9 / n as f64,
                scalar_at.as_secs_f64() * 1e9 / probes.len() as f64,
            );
        }
    }
}
