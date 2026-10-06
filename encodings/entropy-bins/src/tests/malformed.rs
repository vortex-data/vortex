// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Corrupt arrays must fail or decode to garbage, never read out of bounds, panic or hang.

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::array_session;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;

use crate::EntropyBins;
use crate::EntropyBinsOptions;
use crate::array::EntropyBinsData;
use crate::coder::FLAG_UNIFORM;
use crate::coder::TAIL_PADDING;
use crate::pack::pack;
use crate::pack::unpack;
use crate::tests::skewed;

/// 3 blocks of 1024 rows: a constant (uniform) block, then two skewed coded blocks.
fn encoded() -> VortexResult<EntropyBinsData> {
    let mut values = vec![7i64; 1024];
    values.extend(skewed(2048, 3));
    let array = PrimitiveArray::new(Buffer::from(values), Validity::NonNullable);
    let encoded =
        EntropyBins::from_primitive(array.as_view(), 8, EntropyBinsOptions::new(0, 1024))?;
    Ok(encoded.data().clone())
}

/// Rebuild the array from corrupted data, then decode it. Validation may reject the data, and
/// decoding may fail or return garbage: either is fine as long as nothing panics.
fn decode(data: EntropyBinsData) -> VortexResult<ArrayRef> {
    let array = EntropyBins::try_new(
        DType::Primitive(PType::I64, Nullability::NonNullable),
        data,
        Validity::NonNullable,
    )?
    .into_array();
    let mut ctx = array_session().create_execution_ctx();
    array
        .execute::<PrimitiveArray>(&mut ctx)
        .map(IntoArray::into_array)
}

fn with_bytes(data: &EntropyBinsData, f: impl FnOnce(&mut Vec<u8>)) -> EntropyBinsData {
    let mut bytes = data.data.to_vec();
    f(&mut bytes);
    // No spare capacity, so a sanitizer sees any read past the tail padding.
    bytes.shrink_to_fit();
    let mut corrupt = data.clone();
    corrupt.data = ByteBuffer::from(bytes);
    corrupt
}

#[test]
fn uniform_block_id_out_of_range() -> VortexResult<()> {
    let data = encoded()?;
    let start = data.block_start(0);
    assert_ne!(
        data.data[start] & FLAG_UNIFORM,
        0,
        "block 0 should be uniform"
    );
    let corrupt = with_bytes(&data, |b| b[start + 1] = 200);
    assert!(decode(corrupt).is_err());
    Ok(())
}

#[test]
fn truncated_last_block() -> VortexResult<()> {
    let data = encoded()?;
    let start = data.block_start(2);
    assert_eq!(
        data.data[start] & FLAG_UNIFORM,
        0,
        "block 2 should be coded"
    );
    // Keep the last block's header and lane states, claim no words, and drop everything after
    // them but the tail padding: an unbounded id kernel would read far past the buffer.
    let kept = 2 + 14;
    let mut bytes = data.data[..start + kept].to_vec();
    bytes[start] &= 3;
    bytes[start + 1] = 0;
    bytes.resize(bytes.len() + TAIL_PADDING, 0);
    // No spare capacity, so a sanitizer sees any read past the padding.
    bytes.shrink_to_fit();
    let mut lengths = unpack(&data.block_lengths, 3);
    lengths[2] = kept as u64;
    let mut corrupt = data;
    corrupt.data = ByteBuffer::from(bytes);
    corrupt.block_lengths = ByteBuffer::from(pack(&lengths));
    drop(decode(corrupt));
    Ok(())
}

#[test]
fn bins_wider_than_their_offsets() -> VortexResult<()> {
    let mut data = with_bytes(&encoded()?, |_| {});
    for chunk in &mut data.metadata.chunks {
        chunk.widths.fill(62);
    }
    drop(decode(data));
    Ok(())
}

#[test]
fn table_log_out_of_range() -> VortexResult<()> {
    let mut data = encoded()?;
    data.metadata.chunks[0].ans_log = 70;
    assert!(decode(data).is_err());
    Ok(())
}

#[test]
fn huge_weights_requantize_quickly() -> VortexResult<()> {
    let mut data = encoded()?;
    let chunk = &mut data.metadata.chunks[0];
    chunk.ans_log = 0;
    chunk.weights.fill(u32::MAX);
    drop(decode(data));
    Ok(())
}
