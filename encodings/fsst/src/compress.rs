// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// Compress a set of values into an Array.

use fsst::Compressor;
use fsst::Symbol;
use vortex_array::IntoArray;
use vortex_array::accessor::ArrayAccessor;
use vortex_array::arrays::varbin::builder::VarBinBuilder;
use vortex_array::dtype::DType;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;

/// Compress a string array using FSST.
use crate::FSSTArray;
use crate::FSSTData;
pub fn fsst_compress<A: ArrayAccessor<[u8]>>(
    strings: A,
    len: usize,
    dtype: &DType,
    compressor: &Compressor,
) -> FSSTArray {
    strings.with_iterator(|iter| fsst_compress_iter(iter, len, dtype.clone(), compressor))
}

/// Train a compressor from an array.
///
/// # Panics
///
/// If the provided array is not FSST compressible.
pub fn fsst_train_compressor<A: ArrayAccessor<[u8]>>(array: &A) -> Compressor {
    array.with_iterator(|iter| fsst_train_compressor_iter(iter))
}

/// Train a [compressor][Compressor] from an iterator of bytestrings.
fn fsst_train_compressor_iter<'a, I>(iter: I) -> Compressor
where
    I: Iterator<Item = Option<&'a [u8]>>,
{
    let mut lines = Vec::with_capacity(8_192);

    for string in iter {
        match string {
            None => {}
            Some(b) => lines.push(b),
        }
    }

    Compressor::train(&lines)
}

/// Most strings are small in practice. If we encounter a larger string, we reallocate
/// the buffer to hold enough capacity for the worst-case compressed value.
const DEFAULT_BUFFER_LEN: usize = 1024 * 1024;

/// Compress from an iterator of bytestrings using FSST.
pub fn fsst_compress_iter<'a, I>(
    iter: I,
    len: usize,
    dtype: DType,
    compressor: &Compressor,
) -> FSSTArray
where
    I: Iterator<Item = Option<&'a [u8]>>,
{
    let mut buffer = Vec::with_capacity(DEFAULT_BUFFER_LEN);
    let mut builder = VarBinBuilder::<i32>::with_capacity(len);
    let mut uncompressed_lengths: BufferMut<i32> = BufferMut::with_capacity(len);
    for string in iter {
        match string {
            None => {
                builder.append_null();
                uncompressed_lengths.push(0);
            }
            Some(s) => {
                uncompressed_lengths.push(
                    s.len()
                        .try_into()
                        .vortex_expect("string length must fit in i32"),
                );

                // make sure the buffer is 2x+7 larger than the input
                let target_size = 2 * s.len() + 7;
                if target_size > buffer.len() {
                    let additional_capacity = target_size - buffer.len();
                    buffer.reserve(additional_capacity);
                }

                // SAFETY: buffer is always sized to be large enough
                unsafe { compressor.compress_into(s, &mut buffer) };

                builder.append_value(&buffer);
            }
        }
    }

    let codes = builder.finish(DType::Binary(dtype.nullability()));
    let symbols: Buffer<Symbol> = Buffer::copy_from(compressor.symbol_table());
    let symbol_lengths: Buffer<u8> = Buffer::<u8>::copy_from(compressor.symbol_lengths());

    let uncompressed_lengths = uncompressed_lengths.into_array();

    FSSTArray::try_from_data(
        FSSTData::try_new(dtype, symbols, symbol_lengths, codes, uncompressed_lengths)
            .vortex_expect("building FSSTArray from parts"),
    )
    .vortex_expect("FSSTData is always valid")
}

#[cfg(test)]
mod tests {
    use fsst::CompressorBuilder;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::scalar::Scalar;

    use crate::compress::DEFAULT_BUFFER_LEN;
    use crate::fsst_compress_iter;

    #[test]
    fn test_large_string() {
        let big_string: String = "abc"
            .chars()
            .cycle()
            .take(10 * DEFAULT_BUFFER_LEN)
            .collect();

        let compressor = CompressorBuilder::default().build();

        let compressed = fsst_compress_iter(
            [Some(big_string.as_bytes())].into_iter(),
            1,
            DType::Utf8(Nullability::NonNullable),
            &compressor,
        );

        let decoded = compressed.scalar_at(0).unwrap();

        let expected = Scalar::utf8(big_string, Nullability::NonNullable);

        assert_eq!(decoded, expected);
    }
}
