// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Serialization: protobuf metadata plus two buffers (index and data) per stream.
//!
//! The data buffer is used as-is, so a file buffer is never copied. The index buffer holds each
//! [`Packed`] column's residual bits, byte aligned, in a fixed order; their lengths follow from
//! the stream length.

use prost::Message;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;

use crate::ans::N_LANES;
use crate::latent::Latent;
use crate::modes::Encoded;
use crate::modes::Mode;
use crate::modes::Numeric;
use crate::packed::Packed;
use crate::stream::Bin;
use crate::stream::Stream;

#[derive(Clone, PartialEq, Eq, Message)]
pub struct PackedMeta {
    #[prost(uint64, tag = "1")]
    pub base: u64,
    #[prost(sint64, tag = "2")]
    pub slope: i64,
    #[prost(uint32, tag = "3")]
    pub width: u32,
}

#[derive(Clone, PartialEq, Eq, Message)]
pub struct StreamMeta {
    #[prost(uint32, tag = "1")]
    pub delta_order: u32,
    #[prost(uint32, tag = "2")]
    pub ans_size_log: u32,
    /// Bin lower bounds as differences from the previous bin's (the first from 0).
    #[prost(uint64, repeated, tag = "3")]
    pub bin_lower_deltas: Vec<u64>,
    #[prost(bytes = "vec", tag = "4")]
    pub bin_offset_bits: Vec<u8>,
    #[prost(uint32, repeated, tag = "5")]
    pub bin_weights: Vec<u32>,
    #[prost(uint64, tag = "6")]
    pub dominant: u64,
    #[prost(message, optional, tag = "7")]
    pub starts: Option<PackedMeta>,
    #[prost(message, optional, tag = "8")]
    pub ans_bits: Option<PackedMeta>,
    #[prost(message, optional, tag = "9")]
    pub coded_before: Option<PackedMeta>,
    #[prost(message, repeated, tag = "10")]
    pub inits: Vec<PackedMeta>,
    #[prost(message, optional, tag = "11")]
    pub states: Option<PackedMeta>,
}

/// Metadata for an [`Encoded`] array.
#[derive(Clone, PartialEq, Eq, Message)]
pub struct BinnedMetadata {
    /// 0 classic, 1 int mult, 2 float mult, 3 float quant.
    #[prost(uint32, tag = "1")]
    pub mode: u32,
    /// The int mult base, float mult base bits, or float quant shift.
    #[prost(uint64, tag = "2")]
    pub mode_param: u64,
    #[prost(message, optional, tag = "3")]
    pub primary: Option<StreamMeta>,
    #[prost(message, optional, tag = "4")]
    pub secondary: Option<StreamMeta>,
    #[prost(message, optional, tag = "5")]
    pub lookbacks: Option<StreamMeta>,
}

fn packed_meta(p: &Packed) -> PackedMeta {
    PackedMeta {
        base: p.base(),
        slope: p.slope().cast_signed(),
        width: u32::from(p.width()),
    }
}

fn stream_parts<L: Latent>(s: &Stream<L>) -> (StreamMeta, [ByteBuffer; 2]) {
    let mut prev = 0u64;
    let bin_lower_deltas = s
        .bins
        .iter()
        .map(|b| {
            let delta = b.lower.to_u64().wrapping_sub(prev);
            prev = b.lower.to_u64();
            delta
        })
        .collect();
    let meta = StreamMeta {
        delta_order: u32::from(s.delta_order),
        ans_size_log: u32::from(s.ans_size_log),
        bin_lower_deltas,
        bin_offset_bits: s.bins.iter().map(|b| b.offset_bits).collect(),
        bin_weights: s.bins.iter().map(|b| u32::from(b.weight)).collect(),
        dominant: s.dominant.to_u64(),
        starts: Some(packed_meta(&s.starts)),
        ans_bits: Some(packed_meta(&s.ans_bits)),
        coded_before: Some(packed_meta(&s.coded_before)),
        inits: s.inits.iter().map(packed_meta).collect(),
        states: Some(packed_meta(&s.states)),
    };
    let mut index = Vec::new();
    for p in [&s.starts, &s.ans_bits, &s.coded_before]
        .into_iter()
        .chain(&s.inits)
        .chain([&s.states])
    {
        index.extend_from_slice(p.residual_bytes());
    }
    (meta, [ByteBuffer::from(index), s.data.clone()])
}

fn latent_from<L: Latent>(v: u64) -> VortexResult<L> {
    let l = L::from_u64(v);
    vortex_ensure!(l.to_u64() == v, "value {v} does not fit a {}-bit latent", L::BITS);
    Ok(l)
}

fn take_packed(
    len: usize,
    meta: Option<&PackedMeta>,
    index: &[u8],
    pos: &mut usize,
) -> VortexResult<Packed> {
    let meta = meta.ok_or_else(|| vortex_err!("missing packed index column"))?;
    let width = u8::try_from(meta.width)?;
    let n_bytes = Packed::residual_len(len, width);
    let end = *pos + n_bytes;
    vortex_ensure!(end <= index.len(), "index buffer too short");
    let packed = Packed::from_parts(
        len,
        meta.base,
        meta.slope.cast_unsigned(),
        width,
        &index[*pos..end],
    )?;
    *pos = end;
    Ok(packed)
}

fn stream_from_parts<L: Latent>(
    n: usize,
    meta: &StreamMeta,
    index: &[u8],
    data: ByteBuffer,
) -> VortexResult<Stream<L>> {
    let n_bins = meta.bin_lower_deltas.len();
    vortex_ensure!(
        meta.bin_offset_bits.len() == n_bins && meta.bin_weights.len() == n_bins,
        "bin columns disagree in length"
    );
    let mut lower = 0u64;
    let bins = (0..n_bins)
        .map(|i| {
            lower = lower.wrapping_add(meta.bin_lower_deltas[i]);
            Ok(Bin {
                lower: latent_from::<L>(lower)?,
                offset_bits: meta.bin_offset_bits[i],
                weight: u16::try_from(meta.bin_weights[i])?,
            })
        })
        .collect::<VortexResult<Vec<_>>>()?;

    let n_blocks = n.div_ceil(crate::stream::BLOCK_SIZE);
    let mut pos = 0;
    let starts = take_packed(n_blocks, meta.starts.as_ref(), index, &mut pos)?;
    let ans_bits = take_packed(n_blocks, meta.ans_bits.as_ref(), index, &mut pos)?;
    let coded_before = take_packed(n_blocks + 1, meta.coded_before.as_ref(), index, &mut pos)?;
    let inits = meta
        .inits
        .iter()
        .map(|m| take_packed(n_blocks, Some(m), index, &mut pos))
        .collect::<VortexResult<Vec<_>>>()?;
    let n_coded = if coded_before.is_empty() {
        0
    } else {
        usize::try_from(coded_before.get(n_blocks))?
    };
    let n_states = if n_bins > 1 { n_coded * N_LANES } else { 0 };
    let states = take_packed(n_states, meta.states.as_ref(), index, &mut pos)?;
    vortex_ensure!(pos == index.len(), "index buffer has trailing bytes");

    Ok(Stream {
        n,
        delta_order: u8::try_from(meta.delta_order)?,
        ans_size_log: u8::try_from(meta.ans_size_log)?,
        bins,
        starts,
        ans_bits,
        states,
        dominant: latent_from::<L>(meta.dominant)?,
        coded_before,
        inits,
        data,
    })
}

/// Builds the stream described by `meta` from the next index and data buffer pair.
fn next_stream<L: Latent>(
    pairs: &mut std::slice::ChunksExact<'_, ByteBuffer>,
    n: usize,
    meta: &StreamMeta,
) -> VortexResult<Stream<L>> {
    let pair = pairs
        .next()
        .ok_or_else(|| vortex_err!("missing stream buffers"))?;
    stream_from_parts(n, meta, pair[0].as_slice(), pair[1].clone())
}

impl<T: Numeric> Encoded<T> {
    /// Splits into metadata and buffers: an index and a data buffer per stream, in the order
    /// primary, secondary, lookbacks.
    pub fn to_parts(&self) -> (BinnedMetadata, Vec<ByteBuffer>) {
        let (mode, mode_param) = match self.mode {
            Mode::Classic => (0, 0),
            Mode::IntMult(base) => (1, base),
            Mode::FloatMult(base) => (2, base.to_bits()),
            Mode::FloatQuant(k) => (3, u64::from(k)),
        };
        let mut buffers = Vec::new();
        let mut part = |s: Option<(StreamMeta, [ByteBuffer; 2])>| {
            s.map(|(meta, bufs)| {
                buffers.extend(bufs);
                meta
            })
        };
        let primary = part(Some(stream_parts(&self.primary)));
        let secondary = part(self.secondary.as_ref().map(stream_parts));
        let lookbacks = part(self.lookbacks.as_ref().map(stream_parts));
        (
            BinnedMetadata {
                mode,
                mode_param,
                primary,
                secondary,
                lookbacks,
            },
            buffers,
        )
    }

    pub fn from_parts(n: usize, meta: &BinnedMetadata, buffers: &[ByteBuffer]) -> VortexResult<Self> {
        let mode = match meta.mode {
            0 => Mode::Classic,
            1 => Mode::IntMult(meta.mode_param),
            2 => Mode::FloatMult(f64::from_bits(meta.mode_param)),
            3 => Mode::FloatQuant(u8::try_from(meta.mode_param)?),
            other => vortex_bail!("unknown binned mode {other}"),
        };
        let expected = 2 * (1
            + usize::from(meta.secondary.is_some())
            + usize::from(meta.lookbacks.is_some()));
        vortex_ensure!(
            buffers.len() == expected,
            "expected {expected} buffers, got {}",
            buffers.len()
        );
        // The buffer count was checked above, so each present stream has its pair.
        let mut pairs = buffers.chunks_exact(2);
        let primary_meta = meta
            .primary
            .as_ref()
            .ok_or_else(|| vortex_err!("missing primary stream"))?;
        let primary = next_stream(&mut pairs, n, primary_meta)?;
        let secondary = meta
            .secondary
            .as_ref()
            .map(|m| next_stream(&mut pairs, n, m))
            .transpose()?;
        let lookbacks = meta
            .lookbacks
            .as_ref()
            .map(|m| next_stream(&mut pairs, n, m))
            .transpose()?;
        Ok(Self {
            n,
            mode,
            primary,
            secondary,
            lookbacks,
        })
    }

    /// Exact serialized size: metadata plus buffers.
    pub fn serialized_nbytes(&self) -> usize {
        let (meta, buffers) = self.to_parts();
        meta.encoded_len() + buffers.iter().map(ByteBuffer::len).sum::<usize>()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::modes::Decoder;
    use crate::modes::compress;
    use crate::stream::Config;

    #[test]
    fn roundtrip_through_parts() -> VortexResult<()> {
        let values: Vec<f64> = (0..20_000)
            .map(|i| f64::from((i * 7919) % 1000) / 100.0 + f64::from(i % 3))
            .collect();
        let enc = compress(&values, &Config::default())?;
        let (meta, buffers) = enc.to_parts();
        let meta = BinnedMetadata::decode(meta.encode_to_vec().as_slice())?;
        let back = Encoded::<f64>::from_parts(values.len(), &meta, &buffers)?;
        let dec = Decoder::new(Arc::new(back))?;
        assert_eq!(dec.decode(), values);
        Ok(())
    }
}
