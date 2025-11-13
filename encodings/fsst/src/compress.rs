// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// Compress a set of values into an Array.

use fsst::{Compressor, Symbol};
use vortex_array::accessor::ArrayAccessor;
use vortex_array::arrays::builder::VarBinBuilder;
use vortex_array::{Array, IntoArray};
use vortex_buffer::{Buffer, BufferMut};
use vortex_dtype::DType;
use vortex_error::{VortexExpect, VortexUnwrap};

use crate::FSSTArray;

/// Compress a string array using FSST.
pub fn fsst_compress<A: ArrayAccessor<[u8]> + AsRef<dyn Array>>(
    strings: A,
    compressor: &Compressor,
) -> FSSTArray {
    let len = strings.as_ref().len();
    let dtype = strings.as_ref().dtype().clone();
    strings.with_iterator(|iter| fsst_compress_iter(iter, len, dtype, compressor))
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
    // TODO(aduffy): this might be too small.
    let mut buffer = Vec::with_capacity(16 * 1024 * 1024);
    let mut builder = VarBinBuilder::<i32>::with_capacity(len);
    let mut uncompressed_lengths: BufferMut<i32> = BufferMut::with_capacity(len);
    for string in iter {
        match string {
            None => {
                builder.append_null();
                uncompressed_lengths.push(0);
            }
            Some(s) => {
                uncompressed_lengths.push(s.len().try_into().vortex_unwrap());

                // SAFETY: buffer is large enough
                unsafe { compressor.compress_into(s, &mut buffer) };

                builder.append_value(&buffer);
            }
        }
    }

    let codes = builder.finish(DType::Binary(dtype.nullability()));
    let symbols: Buffer<Symbol> = Buffer::copy_from(compressor.symbol_table());
    let symbol_lengths: Buffer<u8> = Buffer::<u8>::copy_from(compressor.symbol_lengths());

    let uncompressed_lengths = uncompressed_lengths.into_array();

    FSSTArray::try_new(dtype, symbols, symbol_lengths, codes, uncompressed_lengths)
        .vortex_expect("building FSSTArray from parts")
}
