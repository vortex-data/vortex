// SPDX-License-Identifier: Apache-2.0

//! Reproducible in-memory encoding benchmark used for Rust/C++ comparisons.

#![allow(clippy::panic_in_result_fn, clippy::unwrap_in_result)]

use std::fs;
use std::hint::black_box;
use std::mem::MaybeUninit;
use std::path::Path;
use std::time::{Duration, Instant};

use onpair::{Config, DECODE_PADDING, MaxDictBits, Parser, Threshold};

const MAGIC: &[u8; 8] = b"ONPAIR01";

struct Corpus {
    bytes: Vec<u8>,
    offsets: Vec<u32>,
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
        let end = cursor + 4;
        let length = u32::from_le_bytes(
            input
                .get(cursor..end)
                .ok_or("truncated length")?
                .try_into()
                .map_err(|_| "length")?,
        ) as usize;
        cursor = end;
        let end = cursor + length;
        bytes.extend_from_slice(input.get(cursor..end).ok_or("truncated row")?);
        cursor = end;
        offsets.push(u32::try_from(bytes.len()).map_err(|_| "payload exceeds u32")?);
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

/// Match the C++ `BitWriter`: LSB-first fixed-width codes and a zero sentinel.
fn pack_codes(codes: &[u16], bits: u8) -> Vec<u64> {
    let mut packed = Vec::with_capacity((codes.len() * bits as usize).div_ceil(64) + 1);
    let mask = (1u64 << bits) - 1;
    let mut buffer = 0u64;
    let mut shift = 0u8;
    for &code in codes {
        let code = u64::from(code) & mask;
        buffer |= code << shift;
        shift += bits;
        if shift >= 64 {
            packed.push(buffer);
            shift -= 64;
            buffer = code >> (bits - shift);
        }
    }
    if shift > 0 {
        packed.push(buffer);
    }
    if !codes.is_empty() {
        packed.push(0);
    }
    packed
}

fn packed_codes_match(codes: &[u16], packed: &[u64], bits: u8) -> bool {
    let mask = (1u64 << bits) - 1;
    codes.iter().enumerate().all(|(index, &expected)| {
        let bit = index * bits as usize;
        let word = bit / 64;
        let shift = bit % 64;
        let mut value = packed[word] >> shift;
        if shift + bits as usize > 64 {
            value |= packed[word + 1] << (64 - shift);
        }
        value & mask == u64::from(expected)
    })
}

fn main() -> Result<(), String> {
    let path = std::env::args_os()
        .nth(1)
        .ok_or("usage: encode_bench CORPUS.onpair")?;
    let bits: u8 = std::env::var("ONPAIR_BITS")
        .unwrap_or_else(|_| "12".into())
        .parse()
        .map_err(|_| "bits")?;
    let warmups: usize = std::env::var("ONPAIR_WARMUPS")
        .unwrap_or_else(|_| "2".into())
        .parse()
        .map_err(|_| "warmups")?;
    let iterations: usize = std::env::var("ONPAIR_ITERATIONS")
        .unwrap_or_else(|_| "5".into())
        .parse()
        .map_err(|_| "iterations")?;
    if iterations == 0 {
        return Err("iterations must be positive".into());
    }

    let corpus = load(Path::new(&path))?;
    let config = Config {
        max_dict_bits: MaxDictBits::new(bits).map_err(|error| error.to_string())?,
        threshold: Threshold::new(0.15).map_err(|error| error.to_string())?,
        seed: Some(42),
    };
    let parser =
        Parser::train(&corpus.bytes, &corpus.offsets, config).map_err(|error| error.to_string())?;

    let (train, _) = measure(warmups, iterations, || {
        Parser::train(black_box(&corpus.bytes), black_box(&corpus.offsets), config).unwrap()
    });
    let (parse_native, column) = measure(warmups, iterations, || {
        parser
            .parse(black_box(&corpus.bytes), black_box(&corpus.offsets))
            .unwrap()
    });
    let (parse_packed, packed_column) = measure(warmups, iterations, || {
        let column = parser
            .parse(black_box(&corpus.bytes), black_box(&corpus.offsets))
            .unwrap();
        let packed = pack_codes(&column.codes, bits);
        (packed, column)
    });
    let (full_native, _) = measure(warmups, iterations, || {
        onpair::compress(black_box(&corpus.bytes), black_box(&corpus.offsets), config).unwrap()
    });
    let (full_packed, _) = measure(warmups, iterations, || {
        let column =
            onpair::compress(black_box(&corpus.bytes), black_box(&corpus.offsets), config).unwrap();
        let packed = pack_codes(&column.codes, bits);
        (packed, column.row_offsets)
    });

    let view = column.view();
    let mut decoded = vec![MaybeUninit::uninit(); corpus.bytes.len() + DECODE_PADDING];
    // SAFETY: the trusted encoder produced `column`, and the output includes
    // the exact decoded size plus the documented read padding.
    let written = unsafe { view.decompress_into(&mut decoded) };
    // SAFETY: `decompress_into` initialized the first `written` bytes.
    let decoded = unsafe { std::slice::from_raw_parts(decoded.as_ptr().cast::<u8>(), written) };
    let roundtrip_correct = decoded == corpus.bytes;
    let packed_correct = packed_codes_match(&packed_column.1.codes, &packed_column.0, bits);

    // A native `u16` stream is already fixed-width output at 16 bits. Narrower
    // modes must include packing to compare with the C++ Store.
    let parse_fair = if bits == 16 {
        parse_native
    } else {
        parse_packed
    };
    let full_fair = if bits == 16 { full_native } else { full_packed };
    println!(
        "rust,bits={bits},rows={},payload_bytes={},warmups={warmups},iterations={iterations},dict_tokens={},codes={},train_ms={:.6},parse_native_ms={:.6},parse_packed_ms={:.6},parse_fair_ms={:.6},full_native_ms={:.6},full_packed_ms={:.6},full_fair_ms={:.6},roundtrip_correct={roundtrip_correct},packed_correct={packed_correct}",
        corpus.offsets.len() - 1,
        corpus.bytes.len(),
        column.dict.num_tokens(),
        column.codes.len(),
        train.as_secs_f64() * 1e3,
        parse_native.as_secs_f64() * 1e3,
        parse_packed.as_secs_f64() * 1e3,
        parse_fair.as_secs_f64() * 1e3,
        full_native.as_secs_f64() * 1e3,
        full_packed.as_secs_f64() * 1e3,
        full_fair.as_secs_f64() * 1e3,
    );
    if !roundtrip_correct || !packed_correct {
        return Err("verification failed".into());
    }
    Ok(())
}
