// SPDX-License-Identifier: Apache-2.0

//! Unified, sequential end-to-end OnPair compression benchmark.

#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::fs;
use std::hint::black_box;
use std::mem::MaybeUninit;
use std::path::Path;
use std::time::Instant;

use onpair_original::OnPair16;
use onpair_snapshot::Column as SnapshotColumn;
use onpair_snapshot::Config as SnapshotConfig;
use onpair_snapshot::DECODE_PADDING;
use onpair_snapshot::MaxDictBits;
use onpair_snapshot::Threshold;
use serde::Serialize;
use sha2::Digest;
use sha2::Sha256;
use vortex_onpair_sys::Column as CppColumn;
use vortex_onpair_sys::OnPairTrainingConfig as CppConfig;

const MAGIC: &[u8; 8] = b"ONPAIR01";
const SAMPLE_FRACTION: f64 = 0.15;
const SEED: u64 = 42;

#[derive(Clone)]
struct Corpus {
    bytes: Vec<u8>,
    offsets: Vec<u32>,
    sha256: String,
}

#[derive(Clone)]
struct Block {
    bytes: Vec<u8>,
    offsets_u32: Vec<u32>,
    offsets_u64: Vec<u64>,
    offsets_usize: Vec<usize>,
}

impl Block {
    fn rows(&self) -> usize {
        self.offsets_u32.len() - 1
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum Algorithm {
    SnapshotGreedy,
    CppBoost,
    PaperRust16,
}

impl Algorithm {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "snapshot-greedy" => Ok(Self::SnapshotGreedy),
            "cpp-boost" => Ok(Self::CppBoost),
            "paper-rust16" => Ok(Self::PaperRust16),
            _ => Err(format!("unknown algorithm {value}")),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::SnapshotGreedy => "snapshot-greedy",
            Self::CppBoost => "cpp-boost",
            Self::PaperRust16 => "paper-rust16",
        }
    }
}

enum Encoded {
    Snapshot {
        column: SnapshotColumn<u32>,
        packed: Option<Vec<u64>>,
    },
    Cpp(CppColumn),
    Original(OnPair16),
}

#[derive(Default, Serialize)]
struct SizeMetrics {
    dict_token_bytes: usize,
    dict_offsets_bytes: usize,
    packed_code_bytes: usize,
    row_offsets_bytes: usize,
    logical_encoded_bytes: usize,
    native_physical_bytes: usize,
    code_count: usize,
    component_breakdown_complete: bool,
}

impl SizeMetrics {
    fn add(&mut self, other: &Self) {
        self.dict_token_bytes += other.dict_token_bytes;
        self.dict_offsets_bytes += other.dict_offsets_bytes;
        self.packed_code_bytes += other.packed_code_bytes;
        self.row_offsets_bytes += other.row_offsets_bytes;
        self.logical_encoded_bytes += other.logical_encoded_bytes;
        self.native_physical_bytes += other.native_physical_bytes;
        self.code_count += other.code_count;
        self.component_breakdown_complete &= other.component_breakdown_complete;
    }
}

#[derive(Serialize)]
struct Report {
    schema_version: u8,
    corpus: String,
    corpus_sha256: String,
    algorithm: String,
    effective_path: String,
    bits: u8,
    sample_fraction: Option<f64>,
    seed: Option<u64>,
    block_target_bytes: usize,
    blocks: usize,
    rows: usize,
    payload_bytes: usize,
    warmups: usize,
    iterations: usize,
    samples_ms: Vec<f64>,
    min_ms: f64,
    median_ms: f64,
    max_ms: f64,
    throughput_mib_s: f64,
    payload_ratio: f64,
    correct: bool,
    sizes: SizeMetrics,
}

fn load(path: &Path) -> Result<Corpus, String> {
    let input = fs::read(path).map_err(|error| format!("{}: {error}", path.display()))?;
    let sha256 = Sha256::digest(&input)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    if input.len() < 24 || &input[..8] != MAGIC {
        return Err("invalid ONPAIR01 corpus".to_string());
    }
    let payload = read_u64(&input[8..16])? as usize;
    let rows = read_u64(&input[16..24])? as usize;
    let mut bytes = Vec::with_capacity(payload);
    let mut offsets = Vec::with_capacity(rows + 1);
    offsets.push(0);
    let mut cursor = 24usize;
    for _ in 0..rows {
        let length_end = cursor.checked_add(4).ok_or("length overflow")?;
        let length = read_u32(input.get(cursor..length_end).ok_or("truncated length")?)? as usize;
        cursor = length_end;
        let row_end = cursor.checked_add(length).ok_or("row overflow")?;
        bytes.extend_from_slice(input.get(cursor..row_end).ok_or("truncated row")?);
        cursor = row_end;
        offsets.push(u32::try_from(bytes.len()).map_err(|_| "payload exceeds u32")?);
    }
    if cursor != input.len() || bytes.len() != payload {
        return Err("corpus framing mismatch".to_string());
    }
    Ok(Corpus {
        bytes,
        offsets,
        sha256,
    })
}

fn read_u64(bytes: &[u8]) -> Result<u64, String> {
    Ok(u64::from_le_bytes(
        bytes.try_into().map_err(|_| "truncated u64")?,
    ))
}

fn read_u32(bytes: &[u8]) -> Result<u32, String> {
    Ok(u32::from_le_bytes(
        bytes.try_into().map_err(|_| "truncated u32")?,
    ))
}

fn partition(corpus: &Corpus, target: usize) -> Vec<Block> {
    let mut blocks = Vec::new();
    let mut row = 0usize;
    while row + 1 < corpus.offsets.len() {
        let first = row;
        let start = corpus.offsets[first] as usize;
        let mut end = start;
        while row + 1 < corpus.offsets.len() {
            let candidate = corpus.offsets[row + 1] as usize;
            if row != first && candidate - start > target {
                break;
            }
            end = candidate;
            row += 1;
            if end - start >= target {
                break;
            }
        }
        let offsets_u32: Vec<u32> = corpus.offsets[first..=row]
            .iter()
            .map(|offset| offset - corpus.offsets[first])
            .collect();
        let offsets_u64 = offsets_u32
            .iter()
            .map(|&offset| u64::from(offset))
            .collect();
        let offsets_usize = offsets_u32.iter().map(|&offset| offset as usize).collect();
        blocks.push(Block {
            bytes: corpus.bytes[start..end].to_vec(),
            offsets_u32,
            offsets_u64,
            offsets_usize,
        });
    }
    blocks
}

fn snapshot_config(bits: u8) -> Result<SnapshotConfig, String> {
    Ok(SnapshotConfig {
        max_dict_bits: MaxDictBits::new(bits).map_err(|error| error.to_string())?,
        threshold: Threshold::new(SAMPLE_FRACTION).map_err(|error| error.to_string())?,
        seed: Some(SEED),
    })
}

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
    packed
}

fn unpack_codes(packed: &[u64], code_count: usize, bits: u8) -> Vec<u16> {
    let mask = (1u64 << bits) - 1;
    (0..code_count)
        .map(|index| {
            let bit_position = index * bits as usize;
            let word = bit_position / 64;
            let offset = bit_position % 64;
            let low = packed[word] >> offset;
            let value = if offset + bits as usize <= 64 {
                low
            } else {
                low | (packed[word + 1] << (64 - offset))
            };
            (value & mask) as u16
        })
        .collect()
}

fn compress_block(block: &Block, bits: u8, algorithm: Algorithm) -> Result<Encoded, String> {
    match algorithm {
        Algorithm::SnapshotGreedy => {
            let column =
                onpair_snapshot::compress(&block.bytes, &block.offsets_u32, snapshot_config(bits)?)
                    .map_err(|error| error.to_string())?;
            let packed = (bits < 16).then(|| pack_codes(&column.codes, bits));
            Ok(Encoded::Snapshot { column, packed })
        }
        Algorithm::CppBoost => {
            let column = CppColumn::compress(
                &block.bytes,
                &block.offsets_u64,
                CppConfig {
                    bits: u32::from(bits),
                    threshold: SAMPLE_FRACTION,
                    seed: SEED,
                },
            )
            .map_err(|error| error.to_string())?;
            Ok(Encoded::Cpp(column))
        }
        Algorithm::PaperRust16 => {
            if bits != 16 {
                return Err("paper-rust16 only supports 16 bits".to_string());
            }
            let mut column = OnPair16::with_capacity(5, block.rows(), block.bytes.len());
            column.compress_bytes(&block.bytes, &block.offsets_usize);
            Ok(Encoded::Original(column))
        }
    }
}

fn compress_all(blocks: &[Block], bits: u8, algorithm: Algorithm) -> Result<Vec<Encoded>, String> {
    blocks
        .iter()
        .map(|block| compress_block(block, bits, algorithm))
        .collect()
}

fn size_metrics(encoded: &Encoded, block: &Block, bits: u8) -> Result<SizeMetrics, String> {
    let mut metrics = SizeMetrics {
        component_breakdown_complete: true,
        ..Default::default()
    };
    match encoded {
        Encoded::Snapshot { column, packed } => {
            metrics.dict_token_bytes = column.dict.logical_len();
            metrics.dict_offsets_bytes = size_of_val(column.dict.offsets());
            metrics.code_count = column.codes.len();
            metrics.packed_code_bytes = (metrics.code_count * bits as usize).div_ceil(8);
            metrics.row_offsets_bytes = column.row_offsets.len() * size_of::<u32>();
            metrics.native_physical_bytes = column.dict.bytes().len()
                + metrics.dict_offsets_bytes
                + column.codes.len() * size_of::<u16>()
                + metrics.row_offsets_bytes
                + packed
                    .as_ref()
                    .map_or(0, |values| values.len() * size_of::<u64>());
        }
        Encoded::Cpp(column) => {
            let parts = column.parts().map_err(|error| error.to_string())?;
            metrics.dict_token_bytes = parts.dict_bytes.len();
            metrics.dict_offsets_bytes = size_of_val(parts.dict_offsets);
            metrics.code_count = parts.codes_boundaries.last().copied().unwrap_or(0) as usize;
            metrics.packed_code_bytes = (metrics.code_count * bits as usize).div_ceil(8);
            metrics.row_offsets_bytes = size_of_val(parts.codes_boundaries);
            metrics.native_physical_bytes = parts.dict_bytes.len()
                + metrics.dict_offsets_bytes
                + size_of_val(parts.codes_packed)
                + metrics.row_offsets_bytes;
        }
        Encoded::Original(column) => {
            metrics.component_breakdown_complete = false;
            metrics.row_offsets_bytes = block.offsets_u32.len() * size_of::<u32>();
            metrics.logical_encoded_bytes = column.space_used() + metrics.row_offsets_bytes;
            metrics.native_physical_bytes =
                column.space_used() + block.offsets_usize.len() * size_of::<usize>();
            return Ok(metrics);
        }
    }
    metrics.logical_encoded_bytes = metrics.dict_token_bytes
        + metrics.dict_offsets_bytes
        + metrics.packed_code_bytes
        + metrics.row_offsets_bytes;
    Ok(metrics)
}

fn verify(encoded: &mut Encoded, block: &Block, bits: u8) -> Result<(), String> {
    match encoded {
        Encoded::Snapshot { column, packed } => {
            if let Some(packed) = packed {
                let unpacked = unpack_codes(packed, column.codes.len(), bits);
                if unpacked != column.codes {
                    return Err("snapshot packed codes mismatch".to_string());
                }
            }
            let view = column.view();
            let max_row = block
                .offsets_u32
                .windows(2)
                .map(|pair| (pair[1] - pair[0]) as usize)
                .max()
                .unwrap_or(0);
            let mut output = vec![MaybeUninit::uninit(); max_row + DECODE_PADDING];
            for row in 0..block.rows() {
                let written = unsafe { view.decompress_row_into(row, &mut output) };
                let decoded =
                    unsafe { std::slice::from_raw_parts(output.as_ptr().cast::<u8>(), written) };
                let start = block.offsets_u32[row] as usize;
                let end = block.offsets_u32[row + 1] as usize;
                if decoded != &block.bytes[start..end] {
                    return Err(format!("snapshot roundtrip mismatch at row {row}"));
                }
            }
        }
        Encoded::Cpp(column) => {
            let mut decoded = Vec::new();
            for row in 0..block.rows() {
                column
                    .decompress_row(row, &mut decoded)
                    .map_err(|error| error.to_string())?;
                let start = block.offsets_u32[row] as usize;
                let end = block.offsets_u32[row + 1] as usize;
                if decoded != block.bytes[start..end] {
                    return Err(format!("C++ roundtrip mismatch at row {row}"));
                }
            }
        }
        Encoded::Original(column) => {
            let max_row = block
                .offsets_u32
                .windows(2)
                .map(|pair| (pair[1] - pair[0]) as usize)
                .max()
                .unwrap_or(0);
            let mut decoded = vec![0; max_row + 16];
            for row in 0..block.rows() {
                let written = column.decompress_string(row, &mut decoded);
                let start = block.offsets_u32[row] as usize;
                let end = block.offsets_u32[row + 1] as usize;
                if decoded[..written] != block.bytes[start..end] {
                    return Err(format!("paper Rust roundtrip mismatch at row {row}"));
                }
            }
        }
    }
    Ok(())
}

fn effective_path(_algorithm: Algorithm) -> String {
    "scalar".to_string()
}

fn main() -> Result<(), String> {
    let mut args = std::env::args_os().skip(1);
    let algorithm = Algorithm::parse(
        &args
            .next()
            .ok_or("usage: onpair-unified-bench ALGORITHM CORPUS BLOCK_MIB BITS [WARMUPS] [ITERATIONS]")?
            .to_string_lossy(),
    )?;
    let path = args.next().ok_or("missing corpus")?;
    let block_mib: usize = args
        .next()
        .ok_or("missing block MiB")?
        .to_string_lossy()
        .parse()
        .map_err(|_| "invalid block MiB")?;
    let bits: u8 = args
        .next()
        .ok_or("missing bits")?
        .to_string_lossy()
        .parse()
        .map_err(|_| "invalid bits")?;
    let warmups: usize = args
        .next()
        .map(|value| value.to_string_lossy().parse())
        .transpose()
        .map_err(|_| "invalid warmups")?
        .unwrap_or(1);
    let iterations: usize = args
        .next()
        .map(|value| value.to_string_lossy().parse())
        .transpose()
        .map_err(|_| "invalid iterations")?
        .unwrap_or(3);
    if block_mib == 0 || iterations == 0 || !(9..=16).contains(&bits) {
        return Err("block MiB and iterations must be positive; bits must be 9..=16".to_string());
    }
    let path = Path::new(&path);
    let corpus = load(path)?;
    let block_target_bytes = block_mib
        .checked_mul(1024 * 1024)
        .ok_or("block MiB overflow")?;
    let blocks = partition(&corpus, block_target_bytes);
    if blocks.is_empty() {
        return Err("corpus has no rows".to_string());
    }

    for _ in 0..warmups {
        black_box(compress_all(&blocks, bits, algorithm)?);
    }
    let mut samples_ms = Vec::with_capacity(iterations);
    let mut last = None;
    for _ in 0..iterations {
        let start = Instant::now();
        let output = compress_all(&blocks, bits, algorithm)?;
        let elapsed = start.elapsed().as_secs_f64() * 1e3;
        black_box(&output);
        samples_ms.push(elapsed);
        last = Some(output);
    }
    let mut output = last.expect("iterations checked positive");
    let mut sizes = SizeMetrics {
        component_breakdown_complete: true,
        ..Default::default()
    };
    for ((encoded, block), block_index) in output.iter_mut().zip(&blocks).zip(0usize..) {
        verify(encoded, block, bits).map_err(|error| format!("block {block_index}: {error}"))?;
        sizes.add(&size_metrics(encoded, block, bits)?);
    }

    let mut sorted_samples = samples_ms.clone();
    sorted_samples.sort_by(f64::total_cmp);
    let min_ms = sorted_samples[0];
    let midpoint = sorted_samples.len() / 2;
    let median_ms = if sorted_samples.len().is_multiple_of(2) {
        (sorted_samples[midpoint - 1] + sorted_samples[midpoint]) / 2.0
    } else {
        sorted_samples[midpoint]
    };
    let max_ms = sorted_samples[sorted_samples.len() - 1];
    let payload_bytes: usize = blocks.iter().map(|block| block.bytes.len()).sum();
    let rows: usize = blocks.iter().map(Block::rows).sum();
    let report = Report {
        schema_version: 1,
        corpus: path
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned(),
        corpus_sha256: corpus.sha256,
        algorithm: algorithm.name().to_string(),
        effective_path: effective_path(algorithm),
        bits,
        sample_fraction: (algorithm != Algorithm::PaperRust16).then_some(SAMPLE_FRACTION),
        seed: (algorithm != Algorithm::PaperRust16).then_some(SEED),
        block_target_bytes,
        blocks: blocks.len(),
        rows,
        payload_bytes,
        warmups,
        iterations,
        samples_ms,
        min_ms,
        median_ms,
        max_ms,
        throughput_mib_s: payload_bytes as f64 / (1024.0 * 1024.0) / (median_ms / 1e3),
        payload_ratio: sizes.logical_encoded_bytes as f64 / payload_bytes as f64,
        correct: true,
        sizes,
    };
    println!(
        "{}",
        serde_json::to_string(&report).map_err(|error| error.to_string())?
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partition_preserves_whole_rows_and_payload() {
        let corpus = Corpus {
            bytes: b"abcdefghij".to_vec(),
            offsets: vec![0, 3, 7, 10],
            sha256: String::new(),
        };
        let blocks = partition(&corpus, 5);

        assert_eq!(blocks.len(), 3);
        assert_eq!(blocks[0].bytes, b"abc");
        assert_eq!(blocks[1].bytes, b"defg");
        assert_eq!(blocks[2].bytes, b"hij");
        assert!(blocks.iter().all(|block| block.offsets_u32[0] == 0));
        assert_eq!(blocks.iter().map(Block::rows).sum::<usize>(), 3);
        assert_eq!(
            blocks.iter().map(|block| block.bytes.len()).sum::<usize>(),
            10
        );
    }
}
