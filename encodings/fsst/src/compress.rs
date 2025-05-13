// Compress a set of values into an Array.

use fsst::{Compressor, Symbol};
use vortex_array::accessor::ArrayAccessor;
use vortex_array::arrays::builder::VarBinBuilder;
use vortex_array::arrays::{VarBinVTable, VarBinViewVTable};
use vortex_array::{Array, ArrayExt, IntoArray};
use vortex_buffer::{Buffer, BufferMut};
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
        return varbin
            .with_iterator(|iter| fsst_compress_iter(iter, len, dtype, compressor))
            .map_err(|err| err.with_context("Failed to compress VarBinArray with FSST"));
    }

    // Compress VarBinViewArray
    if let Some(varbin_view) = strings.as_opt::<VarBinViewVTable>() {
        return varbin_view
            .with_iterator(|iter| fsst_compress_iter(iter, len, dtype, compressor))
            .map_err(|err| err.with_context("Failed to compress VarBinViewArray with FSST"));
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
        varbin
            .with_iterator(|iter| fsst_train_compressor_iter(iter))
            .map_err(|err| err.with_context("Failed to train FSST Compressor from VarBinArray"))
    } else if let Some(varbin_view) = array.as_opt::<VarBinViewVTable>() {
        varbin_view
            .with_iterator(|iter| fsst_train_compressor_iter(iter))
            .map_err(|err| err.with_context("Failed to train FSST Compressor from VarBinViewArray"))
    } else {
        vortex_bail!(
            "cannot fsst_compress array with unsupported encoding {:?}",
            array.encoding_id()
        )
    }
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
