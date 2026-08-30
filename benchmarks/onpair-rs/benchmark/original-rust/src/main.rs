use std::fs;
use std::hint::black_box;
use std::path::Path;
use std::time::{Duration, Instant};

use onpair_rs::OnPair16;

const MAGIC: &[u8; 8] = b"ONPAIR01";

struct Corpus {
    bytes: Vec<u8>,
    offsets: Vec<usize>,
}

fn load(path: &Path) -> Result<Corpus, String> {
    let input = fs::read(path).map_err(|error| error.to_string())?;
    if input.len() < 24 || &input[..8] != MAGIC {
        return Err("invalid ONPAIR01 corpus".into());
    }
    let payload = u64::from_le_bytes(input[8..16].try_into().map_err(|_| "payload")?) as usize;
    let rows = u64::from_le_bytes(input[16..24].try_into().map_err(|_| "rows")?) as usize;
    let mut bytes = Vec::with_capacity(payload);
    let mut offsets = Vec::with_capacity(rows + 1);
    offsets.push(0);
    let mut cursor = 24;
    for _ in 0..rows {
        let length_end = cursor + 4;
        let length = u32::from_le_bytes(
            input
                .get(cursor..length_end)
                .ok_or("truncated length")?
                .try_into()
                .map_err(|_| "length")?,
        ) as usize;
        cursor = length_end;
        let end = cursor + length;
        bytes.extend_from_slice(input.get(cursor..end).ok_or("truncated row")?);
        cursor = end;
        offsets.push(bytes.len());
    }
    if cursor != input.len() || bytes.len() != payload {
        return Err("corpus framing mismatch".into());
    }
    Ok(Corpus { bytes, offsets })
}

fn measure<T>(
    warmups: usize,
    iterations: usize,
    mut operation: impl FnMut() -> T,
) -> (Duration, T) {
    for _ in 0..warmups {
        black_box(operation());
    }
    let mut samples = Vec::with_capacity(iterations);
    let mut last = None;
    for _ in 0..iterations {
        let start = Instant::now();
        let value = operation();
        samples.push(start.elapsed());
        last = Some(black_box(value));
    }
    samples.sort_unstable();
    (
        samples[samples.len() / 2],
        last.expect("iterations must be positive"),
    )
}

fn main() -> Result<(), String> {
    let path = std::env::args_os()
        .nth(1)
        .ok_or("usage: onpair-original-bench CORPUS.onpair")?;
    let warmups: usize = std::env::var("ONPAIR_WARMUPS")
        .unwrap_or_else(|_| "2".into())
        .parse()
        .map_err(|_| "warmups")?;
    let iterations: usize = std::env::var("ONPAIR_ITERATIONS")
        .unwrap_or_else(|_| "5".into())
        .parse()
        .map_err(|_| "iterations")?;
    let threshold: u16 = std::env::var("ONPAIR_ORIGINAL_THRESHOLD")
        .unwrap_or_else(|_| "5".into())
        .parse()
        .map_err(|_| "threshold")?;
    if iterations == 0 {
        return Err("iterations must be positive".into());
    }

    let corpus = load(Path::new(&path))?;
    let (full, compressed) = measure(warmups, iterations, || {
        let mut compressor =
            OnPair16::with_capacity(threshold, corpus.offsets.len() - 1, corpus.bytes.len());
        compressor.compress_bytes(black_box(&corpus.bytes), black_box(&corpus.offsets));
        compressor
    });

    let mut decoded = vec![0; corpus.bytes.len() + 16];
    let written = compressed.decompress_all(&mut decoded);
    let correct = written == corpus.bytes.len() && decoded[..written] == corpus.bytes;
    let reported_bytes = compressed.space_used();
    let boundary_bytes = corpus.offsets.len() * size_of::<usize>();
    let actual_logical_bytes = reported_bytes + boundary_bytes;
    println!(
        "rust_original,bits=16,threshold={threshold},rows={},payload_bytes={},warmups={warmups},iterations={iterations},full_fair_ms={:.6},reported_bytes={reported_bytes},boundary_bytes={boundary_bytes},actual_logical_bytes={actual_logical_bytes},actual_ratio={:.9},correct={correct}",
        corpus.offsets.len() - 1,
        corpus.bytes.len(),
        full.as_secs_f64() * 1e3,
        actual_logical_bytes as f64 / corpus.bytes.len() as f64,
    );
    if !correct {
        return Err("roundtrip mismatch".into());
    }
    Ok(())
}
