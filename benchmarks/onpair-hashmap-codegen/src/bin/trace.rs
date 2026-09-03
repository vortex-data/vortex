use std::collections::BTreeMap;
use std::fs::File;
use std::fs::{self};
use std::io::BufWriter;
use std::io::Write;
use std::path::Path;

use onpair::Config;
use onpair::Dictionary;
use onpair::DictionaryView;
use onpair::MaxDictBits;
use onpair::Parser;
use onpair::Threshold;

const CORPUS_MAGIC: &[u8; 8] = b"ONPAIR01";
const TRACE_MAGIC: &[u8; 8] = b"OPHASH01";

struct Corpus {
    bytes: Vec<u8>,
    offsets: Vec<u32>,
}

#[repr(C, packed)]
#[derive(Clone, Copy)]
struct ShortKey {
    bytes: u64,
    length: u8,
}

fn load_corpus(path: &Path) -> Result<Corpus, String> {
    let input = fs::read(path).map_err(|error| error.to_string())?;
    if input.len() < 24 || &input[..8] != CORPUS_MAGIC {
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

fn load_le(bytes: &[u8], length: usize) -> u64 {
    let mut word = [0u8; 8];
    word[..length].copy_from_slice(&bytes[..length]);
    u64::from_le_bytes(word)
}

fn mask(length: usize) -> u64 {
    if length == 8 {
        u64::MAX
    } else {
        (1u64 << (length * 8)) - 1
    }
}

fn write_u64(writer: &mut impl Write, value: u64) -> std::io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

fn interleave_rows<T: Copy>(values: &[T], ranges: &[(usize, usize)], width: usize) -> Vec<T> {
    let mut interleaved = Vec::with_capacity(values.len());
    for rows in ranges.chunks(width) {
        let mut positions: Vec<_> = rows.iter().map(|&(start, _)| start).collect();
        loop {
            let mut advanced = false;
            for (lane, &(_, end)) in rows.iter().enumerate() {
                if positions[lane] < end {
                    interleaved.push(values[positions[lane]]);
                    positions[lane] += 1;
                    advanced = true;
                }
            }
            if !advanced {
                break;
            }
        }
    }
    interleaved
}

fn main() -> Result<(), String> {
    let corpus_path = std::env::args_os()
        .nth(1)
        .ok_or("usage: trace CORPUS OUTPUT")?;
    let output_path = std::env::args_os()
        .nth(2)
        .ok_or("usage: trace CORPUS OUTPUT")?;
    let bits: u8 = std::env::var("ONPAIR_BITS")
        .unwrap_or_else(|_| "16".into())
        .parse()
        .map_err(|_| "bits")?;
    let mut corpus = load_corpus(Path::new(&corpus_path))?;
    if let Some(limit) = std::env::var_os("TRACE_PAYLOAD_LIMIT") {
        let limit: usize = limit
            .to_string_lossy()
            .parse()
            .map_err(|_| "TRACE_PAYLOAD_LIMIT")?;
        let offset_count = corpus
            .offsets
            .partition_point(|offset| *offset as usize <= limit)
            .max(2);
        corpus.offsets.truncate(offset_count);
        corpus
            .bytes
            .truncate(*corpus.offsets.last().ok_or("empty offsets")? as usize);
    }
    let config = Config {
        max_dict_bits: MaxDictBits::new(bits).map_err(|error| error.to_string())?,
        threshold: Threshold::new(0.15).map_err(|error| error.to_string())?,
        seed: Some(42),
    };
    let parser =
        Parser::train(&corpus.bytes, &corpus.offsets, config).map_err(|error| error.to_string())?;
    let column = parser
        .parse(&corpus.bytes, &corpus.offsets)
        .map_err(|error| error.to_string())?;
    let dict = parser.dict.as_view();

    let mut short_entries = Vec::new();
    let mut long_entries = BTreeMap::new();
    for token in 0..dict.num_tokens() {
        let id = token as u16;
        let value = dict.token(id);
        if value.len() <= 8 {
            short_entries.push((
                ShortKey {
                    bytes: load_le(value, value.len()),
                    length: value.len() as u8,
                },
                id,
            ));
        } else {
            long_entries.entry(load_le(value, 8)).or_insert(id);
        }
    }

    let mut short_probes = Vec::new();
    let mut long_probes = Vec::new();
    let mut short_ranges = Vec::with_capacity(corpus.offsets.len() - 1);
    let mut long_ranges = Vec::with_capacity(corpus.offsets.len() - 1);
    for row in 0..corpus.offsets.len() - 1 {
        let short_start = short_probes.len();
        let long_start = long_probes.len();
        let mut position = corpus.offsets[row] as usize;
        let end = corpus.offsets[row + 1] as usize;
        let code_start = column.row_offsets[row] as usize;
        let code_end = column.row_offsets[row + 1] as usize;
        for &token in &column.codes[code_start..code_end] {
            let token_length = dict.token_len(token);
            let remaining = end - position;
            let low64 = load_le(&corpus.bytes[position..], remaining.min(8));
            if remaining > 8 {
                long_probes.push(low64);
            }
            if token_length <= 8 {
                for length in (token_length..=remaining.min(8)).rev() {
                    short_probes.push(ShortKey {
                        bytes: low64 & mask(length),
                        length: length as u8,
                    });
                }
            }
            position += token_length;
        }
        if position != end {
            return Err("token stream did not consume row".into());
        }
        short_ranges.push((short_start, short_probes.len()));
        long_ranges.push((long_start, long_probes.len()));
    }
    if let Some(width) = std::env::var_os("TRACE_INTERLEAVE_WIDTH") {
        let width: usize = width
            .to_string_lossy()
            .parse()
            .map_err(|_| "TRACE_INTERLEAVE_WIDTH")?;
        if width == 0 {
            return Err("TRACE_INTERLEAVE_WIDTH must be positive".into());
        }
        short_probes = interleave_rows(&short_probes, &short_ranges, width);
        long_probes = interleave_rows(&long_probes, &long_ranges, width);
    }

    let file = File::create(output_path).map_err(|error| error.to_string())?;
    let mut writer = BufWriter::new(file);
    writer
        .write_all(TRACE_MAGIC)
        .map_err(|error| error.to_string())?;
    for count in [
        short_entries.len(),
        long_entries.len(),
        short_probes.len(),
        long_probes.len(),
    ] {
        write_u64(&mut writer, count as u64).map_err(|error| error.to_string())?;
    }
    for (key, value) in &short_entries {
        write_u64(&mut writer, key.bytes).map_err(|error| error.to_string())?;
        writer
            .write_all(&[key.length])
            .and_then(|()| writer.write_all(&value.to_le_bytes()))
            .map_err(|error| error.to_string())?;
    }
    for (&key, &value) in &long_entries {
        write_u64(&mut writer, key).map_err(|error| error.to_string())?;
        writer
            .write_all(&value.to_le_bytes())
            .map_err(|error| error.to_string())?;
    }
    for key in &short_probes {
        write_u64(&mut writer, key.bytes).map_err(|error| error.to_string())?;
        writer
            .write_all(&[key.length])
            .map_err(|error| error.to_string())?;
    }
    for &key in &long_probes {
        write_u64(&mut writer, key).map_err(|error| error.to_string())?;
    }
    writer.flush().map_err(|error| error.to_string())?;

    println!(
        "trace,bits={bits},dict_tokens={},short_entries={},long_entries={},short_probes={},long_probes={},total_probes={}",
        dict.num_tokens(),
        short_entries.len(),
        long_entries.len(),
        short_probes.len(),
        long_probes.len(),
        short_probes.len() + long_probes.len(),
    );
    Ok(())
}
