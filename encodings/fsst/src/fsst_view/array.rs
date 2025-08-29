// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use core::fmt;
use std::fmt::Formatter;
use std::sync::{Arc, LazyLock};

use fsst::{Compressor, Symbol};
use num_traits::AsPrimitive;
use vortex_array::compute::is_sorted;
use vortex_array::stats::{ArrayStats, StatsSetRef};
use vortex_array::validity::Validity;
use vortex_array::vtable::{ArrayVTable, NotSupported, VTable, ValidityVTableFromValidityHelper};
use vortex_array::{Array, ArrayRef, EncodingId, EncodingRef, vtable};
use vortex_buffer::{Alignment, Buffer, ByteBuffer};
use vortex_dtype::{DType, match_each_unsigned_integer_ptype};
use vortex_error::{VortexExpect, VortexResult, vortex_ensure};

use crate::fsst_view::View;
#[derive(Debug, Copy, Clone)]
pub struct FSSTViewEncoding;

vtable!(FSSTView);

impl VTable for FSSTViewVTable {
    type Array = FSSTViewArray;
    type Encoding = FSSTViewEncoding;
    type ArrayVTable = Self;
    type CanonicalVTable = Self;
    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromValidityHelper;
    type VisitorVTable = Self;
    type ComputeVTable = NotSupported;
    type EncodeVTable = Self;
    type SerdeVTable = Self;
    type PipelineVTable = NotSupported;

    fn id(_: &Self::Encoding) -> EncodingId {
        EncodingId::new_ref("vortex.fsstview")
    }

    fn encoding(_: &Self::Array) -> EncodingRef {
        EncodingRef::new_ref(FSSTViewEncoding.as_ref())
    }
}

impl ArrayVTable<FSSTViewVTable> for FSSTViewVTable {
    fn len(array: &FSSTViewArray) -> usize {
        array.views.len()
    }

    fn dtype(array: &FSSTViewArray) -> &DType {
        &array.dtype
    }

    fn stats(array: &FSSTViewArray) -> StatsSetRef<'_> {
        array.stats_set.to_ref(array.as_ref())
    }
}

#[derive(Clone)]
pub struct FSSTViewArray {
    /// A list of 16-byte views into the FSST buffer
    pub(crate) views: Buffer<View>,
    pub(crate) dtype: DType,
    /// A packed buffer containing FSST-encoded string data without any internal padding
    pub(crate) fsst_buffer: ByteBuffer,
    /// `compressed_offsets[i]` is the offset into `fsst_buffer` where the `i`-th compressed
    /// string starts.
    pub(crate) compressed_offsets: ArrayRef,
    /// Offsets of all the uncompressed strings, in the original order based on the buffer
    /// type instead.
    pub(crate) uncompressed_offsets: ArrayRef,

    /// A cached compressor, to amortize the cost of building over and over again
    pub(crate) compressor: Arc<LazyLock<Compressor, Box<dyn Fn() -> Compressor + Send>>>,
    /// FSST compressor used to encode/decode the strings in the fsst_buffer
    pub(crate) symbols: Buffer<Symbol>,
    pub(crate) symbol_lengths: ByteBuffer,
    /// Validity information, dictating presence of nulls
    pub(crate) validity: Validity,
    pub(crate) stats_set: ArrayStats,
}

impl fmt::Debug for FSSTViewArray {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FSSTViewArray")
            .field("views_length", &self.views.len())
            .field("fsst_buffer_size", &self.fsst_buffer.len())
            .field("validity", &self.validity)
            .finish()
    }
}

impl FSSTViewArray {
    /// Create a new `FSSTViewArray`, panicking if the components cannot construct a valid array.
    ///
    /// See [`FSSTViewArray::try_new`] for the validation that is performed.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        views: Buffer<View>,
        buffer: ByteBuffer,
        symbols: Buffer<Symbol>,
        symbol_lengths: ByteBuffer,
        compressed_offsets: ArrayRef,
        uncompressed_offsets: ArrayRef,
        dtype: DType,
        validity: Validity,
    ) -> Self {
        Self::try_new(
            views,
            buffer,
            symbols,
            symbol_lengths,
            compressed_offsets,
            uncompressed_offsets,
            dtype,
            validity,
        )
        .vortex_expect("FSSTViewArray new")
    }

    /// Create a new FSSTView array from components, performing validation.
    ///
    /// # Validation
    ///
    /// The following preconditions must be met for the components, or an error is returned:
    ///
    /// * Any non-inlined `views` must point to valid indices in the buffer
    /// * The `compressed_offsets` and `uncompressed_offsets` must be valid, sorted offset arrays
    ///   with DType `U64`. They cannot be nullable, and must share the same length
    /// * The last `compressed_offsets` element cannot point past the end of the `buffer` of encoded
    ///   strings
    /// * The provided symbol table must be a valid FSST symbol table, i.e. it must be properly
    ///   aligned to a `u64` and must have length <= 255
    /// * The `dtype` must be `Utf8` or `Binary`
    /// * If a validity array is provided, it must have a length that matches the length of `views`
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(
        views: Buffer<View>,
        buffer: ByteBuffer,
        symbols: Buffer<Symbol>,
        symbol_lengths: ByteBuffer,
        compressed_offsets: ArrayRef,
        uncompressed_offsets: ArrayRef,
        dtype: DType,
        validity: Validity,
    ) -> VortexResult<Self> {
        vortex_ensure!(
            dtype.is_utf8() || dtype.is_binary(),
            "FSSTViewArray: unexpected DType {dtype}"
        );

        vortex_ensure!(
            symbols.len() <= 255,
            "Symbol table cannot exceed length 255, was {}",
            symbols.len()
        );

        vortex_ensure!(
            symbols.len() == symbol_lengths.len(),
            "symbol table of size {} has invalid symbol_lengths size {}",
            symbols.len(),
            symbol_lengths.len()
        );

        vortex_ensure!(
            compressed_offsets.dtype().is_primitive()
                && !compressed_offsets.dtype().is_nullable()
                && compressed_offsets.dtype().as_ptype().is_unsigned_int(),
            "expected unsigned int DType for compressed offsets, was {}",
            compressed_offsets.dtype()
        );
        vortex_ensure!(
            uncompressed_offsets.dtype().is_primitive()
                && !uncompressed_offsets.dtype().is_nullable()
                && uncompressed_offsets.dtype().as_ptype().is_unsigned_int(),
            "expected unsigned int DType for uncompressed offsets, was {}",
            uncompressed_offsets.dtype()
        );

        vortex_ensure!(
            compressed_offsets.len() == uncompressed_offsets.len(),
            "compressed offsets and uncompressed offset must be same size"
        );

        vortex_ensure!(
            !compressed_offsets.is_empty() && !uncompressed_offsets.is_empty(),
            "offsets cannot be empty"
        );

        vortex_ensure!(
            is_sorted(&compressed_offsets)?,
            "compressed offsets must be sorted",
        );
        vortex_ensure!(
            is_sorted(&uncompressed_offsets)?,
            "uncompressed offsets must be sorted"
        );

        let final_offset = compressed_offsets
            .scalar_at(compressed_offsets.len() - 1)
            .as_primitive()
            .as_::<u32>()
            .vortex_expect("compressed offsets cannot contain nulls");
        vortex_ensure!(
            final_offset as usize <= buffer.len(),
            "compressed offsets point beyond end of the buffer"
        );

        let max_index = compressed_offsets.len() - 1;
        // Validate all the view pointers.
        for view in views.iter() {
            if !view.is_inlined() {
                let outlined = unsafe { view.outline };
                vortex_ensure!(
                    (outlined.index as usize) <= max_index,
                    "view index {} out of bounds for FSSTViewArray with {} compressed strings",
                    outlined.index,
                    max_index
                );
            }
        }

        // Verify the validity length equals the correct length
        if let Some(validity_array) = validity.as_array() {
            vortex_ensure!(
                validity_array.len() == views.len(),
                "FSSTViewArray: provided validity array of length {}, expected {}",
                validity_array.len(),
                views.len()
            );
        }

        let symbols_clone = symbols.clone();
        let symbol_lengths_clone = symbol_lengths.clone();

        Ok(Self {
            fsst_buffer: buffer,
            views,
            compressed_offsets,
            uncompressed_offsets,
            validity,
            symbols,
            symbol_lengths,
            compressor: Arc::new(LazyLock::new(Box::new(move || {
                Compressor::rebuild_from(symbols_clone.as_slice(), symbol_lengths_clone.as_slice())
            }))),
            dtype,
            stats_set: Default::default(),
        })
    }

    /// Create a new `FSSTViewArray` without checking the validation. It is
    /// thus the caller's responsibility to ensure that the provided components adhere
    /// to the validation contract.
    ///
    /// # Safety
    ///
    /// This is only safe if the caller previously performs the validation, OR
    /// if building from the components of an already validated FSSTViewArray and you
    /// have verified that whatever transformation performed on the components has
    /// not altered the preconditions.
    ///
    /// See [`FSSTViewArray::try_new`] for guidance on validation that must be performed.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn new_unchecked(
        views: Buffer<View>,
        buffer: ByteBuffer,
        symbols: Buffer<Symbol>,
        symbol_lengths: ByteBuffer,
        compressed_offsets: ArrayRef,
        uncompressed_offsets: ArrayRef,
        dtype: DType,
        validity: Validity,
    ) -> Self {
        let symbols_clone = symbols.clone();
        let symbol_lengths_clone = symbol_lengths.clone();

        Self {
            fsst_buffer: buffer,
            views,
            compressed_offsets,
            uncompressed_offsets,
            validity,
            symbols,
            symbol_lengths,
            compressor: Arc::new(LazyLock::new(Box::new(move || {
                Compressor::rebuild_from(symbols_clone.as_slice(), symbol_lengths_clone.as_slice())
            }))),
            dtype,
            stats_set: Default::default(),
        }
    }
}

impl FSSTViewArray {
    pub fn views(&self) -> &Buffer<View> {
        &self.views
    }

    pub fn buffer(&self) -> &ByteBuffer {
        &self.fsst_buffer
    }
    pub fn uncompressed_offsets(&self) -> &ArrayRef {
        &self.uncompressed_offsets
    }

    pub fn compressed_offsets(&self) -> &ArrayRef {
        &self.compressed_offsets
    }

    pub fn symbols(&self) -> &Buffer<Symbol> {
        &self.symbols
    }

    pub fn symbol_lengths(&self) -> &ByteBuffer {
        &self.symbol_lengths
    }

    pub fn bytes_at(&self, index: usize) -> ByteBuffer {
        let view = self.views[index];
        // If view is a pointer to the slice, ignore it
        if view.is_inlined() {
            let inlined = unsafe { view.inline };
            let len = inlined.len as usize;

            let start = index * size_of::<View>() + 4;
            let end = start + len;
            // Return a handle to bytes pointing into the `views` buffer
            self.views
                .clone()
                .into_byte_buffer()
                .slice_with_alignment(start..end, Alignment::of::<u8>())
        } else {
            // Return a ByteBuffer wrapping the vector
            let outline = unsafe { view.outline };
            let buf_index = outline.index as usize;

            let (start, end) = match_each_unsigned_integer_ptype!(
                self.compressed_offsets.dtype().as_ptype(),
                |P| {
                    let start: usize = self
                        .compressed_offsets
                        .scalar_at(buf_index)
                        .as_primitive()
                        .as_::<P>()
                        .unwrap_or_default()
                        .as_();

                    let end = self
                        .compressed_offsets
                        .scalar_at(buf_index + 1)
                        .as_primitive()
                        .as_::<P>()
                        .unwrap_or_default()
                        .as_();

                    (start, end)
                }
            );

            let encoded = self.fsst_buffer.slice(start..end);

            let result = self
                .compressor
                .decompressor()
                .decompress(encoded.as_slice());

            ByteBuffer::from(result)
        }
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::IntoArray;
    use vortex_array::validity::Validity;
    use vortex_buffer::{Buffer, ByteBuffer, buffer};
    use vortex_dtype::{DType, Nullability};

    use crate::fsst_view::FSSTViewArray;
    use crate::view_outlined;

    #[test]
    #[should_panic(expected = "view index 10 out of bounds")]
    fn test_validate_views() {
        let _ = FSSTViewArray::new(
            buffer![view_outlined!(14, 10, b"01234567891234")],
            ByteBuffer::empty(),
            Buffer::empty(),
            ByteBuffer::empty(),
            buffer![0u32].into_array(),
            buffer![0u32].into_array(),
            DType::Utf8(Nullability::NonNullable),
            Validity::NonNullable,
        );
    }

    #[test]
    #[should_panic(expected = "compressed offsets point beyond end of the buffer")]
    fn test_validate_offsets() {
        let _ = FSSTViewArray::new(
            Buffer::empty(),                   // views
            ByteBuffer::empty(),               // fsst_buffer
            Buffer::empty(),                   // symbols
            ByteBuffer::empty(),               // symbol lengths
            buffer![0u32, 5u32].into_array(),  // compressed offsets
            buffer![0u32, 10u32].into_array(), // uncompressed offsets
            DType::Utf8(Nullability::NonNullable),
            Validity::NonNullable,
        );
    }

    #[test]
    #[should_panic(expected = "expected unsigned int DType for compressed offsets, was i64")]
    fn test_validate_compressed_ptype() {
        let _ = FSSTViewArray::new(
            Buffer::empty(),                   // views
            ByteBuffer::empty(),               // fsst_buffer
            Buffer::empty(),                   // symbols
            ByteBuffer::empty(),               // symbol lengths
            buffer![0i64, 5i64].into_array(),  // compressed offsets
            buffer![0u32, 10u32].into_array(), // uncompressed offsets
            DType::Utf8(Nullability::NonNullable),
            Validity::NonNullable,
        );
    }

    #[test]
    #[should_panic(expected = "expected unsigned int DType for uncompressed offsets, was i64")]
    fn test_validate_uncompressed_ptype() {
        let _ = FSSTViewArray::new(
            Buffer::empty(),                   // views
            ByteBuffer::empty(),               // fsst_buffer
            Buffer::empty(),                   // symbols
            ByteBuffer::empty(),               // symbol lengths
            buffer![0u32, 5u32].into_array(),  // compressed offsets
            buffer![0i64, 10i64].into_array(), // uncompressed offsets
            DType::Utf8(Nullability::NonNullable),
            Validity::NonNullable,
        );
    }
}
