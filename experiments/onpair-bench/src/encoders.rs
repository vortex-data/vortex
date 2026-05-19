// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use anyhow::Result;
use fsst::Compressor;
use onpair_lib::Column;
use onpair_lib::OnPairTrainingConfig;
use onpair_lib::bits::unpack_codes_to_u16;

use crate::frontcode::front_coded_bytes_size;
use crate::frontcode::front_coded_size;

/// Raw bytes + 4-byte per-row offsets.
pub fn raw_size(rows: &[Vec<u8>]) -> usize {
    rows.iter().map(|r| r.len()).sum::<usize>() + rows.len() * 4
}

/// One zstd compression over the concatenated bytes, plus per-row offsets.
/// Note: this loses random access (need to decompress everything to read row N).
pub fn zstd_monolithic(rows: &[Vec<u8>], level: i32) -> Result<usize> {
    let mut concat: Vec<u8> = Vec::new();
    for r in rows {
        concat.extend_from_slice(r);
    }
    let compressed = zstd::stream::encode_all(&concat[..], level)?;
    Ok(compressed.len() + rows.len() * 4)
}

/// Per-block zstd: compress each block of `block_size` rows independently.
/// Per-block overhead = block-start offset (4 bytes). Per-row offsets within
/// the decompressed block are reconstructed at read time from the row order.
pub fn zstd_block(rows: &[Vec<u8>], block_size: usize, level: i32) -> Result<usize> {
    let mut total = 0;
    let mut i = 0;
    let n = rows.len();
    while i < n {
        let end = (i + block_size).min(n);
        let mut concat = Vec::new();
        for r in &rows[i..end] {
            concat.extend_from_slice(r);
        }
        let compressed = zstd::stream::encode_all(&concat[..], level)?;
        // per-block: compressed payload + block offset (4) + per-row length-in-block (we need
        // row offsets to seek within decompressed block). Count 4 per row.
        total += compressed.len() + 4 + (end - i) * 4;
        i = end;
    }
    Ok(total)
}

/// FSST symbol table (256 entries × 8 bytes) + sum of compressed row sizes
/// + per-row offsets.
pub fn fsst_size(rows: &[Vec<u8>]) -> usize {
    let lines: Vec<&[u8]> = rows.iter().map(|r| r.as_slice()).collect();
    let compressor = Compressor::train(&lines);
    let mut payload = 0usize;
    for r in &lines {
        let compressed = compressor.compress(r);
        payload += compressed.len();
    }
    // Symbol table: 256 symbols, each up to 8 bytes (u64) — count fixed 256*8.
    let symbol_table = 256 * 8;
    symbol_table + payload + rows.len() * 4
}

/// Result of running OnPair on a column: the compressed column + the
/// per-row decoded tokens (for downstream front-coding).
pub struct OnPairOut {
    pub col: Column,
    /// `tokens[i]` is the u16 token sequence for row i.
    pub tokens: Vec<Vec<u16>>,
}

pub fn onpair_compress(rows: &[Vec<u8>], bits: u32) -> Result<OnPairOut> {
    let mut bytes = Vec::with_capacity(rows.iter().map(|r| r.len()).sum());
    let mut offsets: Vec<u64> = Vec::with_capacity(rows.len() + 1);
    offsets.push(0);
    for r in rows {
        bytes.extend_from_slice(r);
        offsets.push(bytes.len() as u64);
    }
    let cfg = OnPairTrainingConfig {
        bits,
        threshold: 0.5,
        seed: 7,
    };
    let col = Column::compress(&bytes, &offsets, cfg)
        .map_err(|e| anyhow::anyhow!("onpair compress: {e:?}"))?;

    // Extract per-row token sequences from the packed code stream.
    let parts = col
        .parts()
        .map_err(|e| anyhow::anyhow!("onpair parts: {e:?}"))?;
    let total_tokens = *parts.codes_boundaries.last().unwrap_or(&0) as usize;
    let all_codes = unpack_codes_to_u16(parts.codes_packed, total_tokens, parts.bits);
    let mut tokens = Vec::with_capacity(rows.len());
    for w in parts.codes_boundaries.windows(2) {
        let (a, b) = (w[0] as usize, w[1] as usize);
        tokens.push(all_codes[a..b].to_vec());
    }
    let _ = parts;
    Ok(OnPairOut { col, tokens })
}

/// Total OnPair on-disk size: dict bytes + dict offsets + packed codes + row
/// boundaries.
pub fn onpair_size(out: &OnPairOut) -> Result<usize> {
    let p = out
        .col
        .parts()
        .map_err(|e| anyhow::anyhow!("onpair parts: {e:?}"))?;
    Ok(p.dict_bytes.len()
        + p.dict_offsets.len() * 4
        + p.codes_packed.len() * 8
        + p.codes_boundaries.len() * 4)
}

/// Public helper to introspect dict size for documentation.
pub fn onpair_dict_size_components(out: &OnPairOut) -> Result<(usize, usize, usize)> {
    let p = out
        .col
        .parts()
        .map_err(|e| anyhow::anyhow!("onpair parts: {e:?}"))?;
    Ok((
        p.dict_bytes.len(),
        p.dict_offsets.len(),
        p.dict_offsets.len() - 1, // num tokens
    ))
}

/// OnPair dict (overhead only — same as `onpair_size` minus the per-row payload).
pub fn onpair_dict_overhead(out: &OnPairOut) -> Result<usize> {
    let p = out
        .col
        .parts()
        .map_err(|e| anyhow::anyhow!("onpair parts: {e:?}"))?;
    Ok(p.dict_bytes.len() + p.dict_offsets.len() * 4)
}

/// OnPair codes + token-space block front-coding. The codes themselves are
/// replaced by the front-coded payload; the dict still ships once.
pub fn onpair_front_coded(out: &OnPairOut, block: usize) -> Result<usize> {
    let dict_overhead = onpair_dict_overhead(out)?;
    let refs: Vec<&[u16]> = out.tokens.iter().map(|v| v.as_slice()).collect();
    let bits = out.col.bits();
    Ok(dict_overhead + front_coded_size(&refs, block, bits))
}

/// Byte-level front-coding (DELTA_BYTE_ARRAY style), no dict.
pub fn bytes_front_coded(rows: &[Vec<u8>], block: usize) -> usize {
    let refs: Vec<&[u8]> = rows.iter().map(|r| r.as_slice()).collect();
    front_coded_bytes_size(&refs, block)
}
