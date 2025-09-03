// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// Compress a set of values into an Array.

use fsst::{Compressor, Symbol};
use vortex_array::arrays::builder::VarBinBuilder;
use vortex_array::arrays::{VarBinVTable, VarBinViewVTable};
use vortex_array::{Array, IntoArray};
use vortex_buffer::{Buffer, BufferMut, ByteBuffer};
use vortex_dtype::DType;
use vortex_error::{VortexExpect, VortexResult, VortexUnwrap, vortex_bail};

use crate::FSSTArray;

/// Compress an array using FSST.
///
/// # Panics
///
/// If the `strings` array is not encoded as either [`vortex_array::arrays::VarBinArray`] or
/// [`vortex_array::arrays::VarBinViewArray`].
pub fn fsst_compress(strings: &dyn Array, compressor: &Compressor) -> VortexResult<FSSTArray> {
    let len = strings.len();
    let dtype = strings.dtype().clone();

    // Compress VarBinArray
    if let Some(varbin) = strings.as_opt::<VarBinVTable>() {
        return Ok(fsst_compress_iter(varbin.iter(), len, dtype, compressor));
    }

    // Compress VarBinViewArray
    if let Some(varbin_view) = strings.as_opt::<VarBinViewVTable>() {
        return Ok(fsst_compress_iter(
            varbin_view.iter(),
            len,
            dtype,
            compressor,
        ));
    }

    vortex_bail!(
        "cannot fsst_compress array with unsupported encoding {:?}",
        strings.encoding_id()
    )
}

/// Train a compressor from an array.
///
/// # Panics
///
/// If the provided array is not FSST compressible.
pub fn fsst_train_compressor(array: &dyn Array) -> VortexResult<Compressor> {
    if let Some(varbin) = array.as_opt::<VarBinVTable>() {
        Ok(fsst_train_compressor_iter(varbin.iter()))
    } else if let Some(varbin_view) = array.as_opt::<VarBinViewVTable>() {
        Ok(fsst_train_compressor_iter(varbin_view.iter()))
    } else {
        vortex_bail!(
            "cannot fsst_compress array with unsupported encoding {:?}",
            array.encoding_id()
        )
    }
}

/// Train a [compressor][Compressor] from an iterator of bytestrings.
fn fsst_train_compressor_iter<I>(iter: I) -> Compressor
where
    I: Iterator<Item = Option<ByteBuffer>>,
{
    let mut lines = Vec::with_capacity(8_192);

    for string in iter {
        match string {
            None => {}
            Some(b) => lines.push(b),
        }
    }

    // TODO(aduffy): make Compressor::train take an AsRef<[u8]> instead of forcing &[u8]
    let lines2 = lines.iter().map(|b| b.as_slice()).collect();

    Compressor::train(&lines2)
}

/// Compress from an iterator of bytestrings using FSST.
pub fn fsst_compress_iter<I>(
    iter: I,
    len: usize,
    dtype: DType,
    compressor: &Compressor,
) -> FSSTArray
where
    I: Iterator<Item = Option<ByteBuffer>>,
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
                unsafe { compressor.compress_into(s.as_slice(), &mut buffer) };

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
