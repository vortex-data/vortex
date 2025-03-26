use fsst::{Decompressor, Symbol};
use vortex_array::arrays::VarBinEncoding;
use vortex_array::stats::{ArrayStats, StatsSetRef};
use vortex_array::variants::{BinaryArrayTrait, Utf8ArrayTrait};
use vortex_array::vtable::{EncodingVTable, VTableRef};
use vortex_array::{
    Array, ArrayImpl, ArrayRef, ArrayStatisticsImpl, ArrayValidityImpl, ArrayVariantsImpl,
    Encoding, SerdeMetadata,
};
use vortex_buffer::Buffer;
use vortex_dtype::DType;
use vortex_error::{VortexResult, vortex_bail};
use vortex_mask::Mask;

use crate::serde::FSSTMetadata;

#[derive(Clone, Debug)]
pub struct FSSTArray {
    dtype: DType,
    symbols: Buffer<Symbol>,
    symbol_lengths: Buffer<u8>,
    codes: ArrayRef,
    uncompressed_lengths: ArrayRef,
    stats_set: ArrayStats,
}

pub struct FSSTEncoding;
impl Encoding for FSSTEncoding {
    type Array = FSSTArray;
    type Metadata = SerdeMetadata<FSSTMetadata>;
}

impl FSSTArray {
    /// Build an FSST array from a set of `symbols` and `codes`.
    ///
    /// Symbols are 8-bytes and can represent short strings, each of which is assigned
    /// a code.
    ///
    /// The `codes` array is a Binary array where each binary datum is a sequence of 8-bit codes.
    /// Each code corresponds either to a symbol, or to the "escape code",
    /// which tells the decoder to emit the following byte without doing a table lookup.
    pub fn try_new(
        dtype: DType,
        symbols: Buffer<Symbol>,
        symbol_lengths: Buffer<u8>,
        codes: ArrayRef,
        uncompressed_lengths: ArrayRef,
    ) -> VortexResult<Self> {
        // Check: symbols must not have length > MAX_CODE
        if symbols.len() > 255 {
            vortex_bail!(InvalidArgument: "symbols array must have length <= 255");
        }
        if symbols.len() != symbol_lengths.len() {
            vortex_bail!(InvalidArgument: "symbols and symbol_lengths arrays must have same length");
        }

        if uncompressed_lengths.len() != codes.len() {
            vortex_bail!(InvalidArgument: "uncompressed_lengths must be same len as codes");
        }

        if !uncompressed_lengths.dtype().is_int() || uncompressed_lengths.dtype().is_nullable() {
            vortex_bail!(InvalidArgument: "uncompressed_lengths must have integer type and cannot be nullable, found {}", uncompressed_lengths.dtype());
        }

        if codes.encoding() != VarBinEncoding.id() {
            vortex_bail!(
                InvalidArgument: "codes must have varbin encoding, was {}",
                codes.encoding()
            );
        }

        // Check: strings must be a Binary array.
        if !matches!(codes.dtype(), DType::Binary(_)) {
            vortex_bail!(InvalidArgument: "codes array must be DType::Binary type");
        }

        Ok(Self {
            dtype,
            symbols,
            symbol_lengths,
            codes,
            uncompressed_lengths,
            stats_set: Default::default(),
        })
    }

    /// Access the symbol table array
    pub fn symbols(&self) -> &Buffer<Symbol> {
        &self.symbols
    }

    /// Access the symbol table array
    pub fn symbol_lengths(&self) -> &Buffer<u8> {
        &self.symbol_lengths
    }

    /// Access the codes array
    pub fn codes(&self) -> &ArrayRef {
        &self.codes
    }

    /// Get the DType of the codes array
    #[inline]
    pub fn codes_dtype(&self) -> &DType {
        self.codes.dtype()
    }

    /// Get the uncompressed length for each element in the array.
    pub fn uncompressed_lengths(&self) -> &ArrayRef {
        &self.uncompressed_lengths
    }

    /// Get the DType of the uncompressed lengths array
    #[inline]
    pub fn uncompressed_lengths_dtype(&self) -> &DType {
        self.uncompressed_lengths.dtype()
    }

    /// Build a [`Decompressor`][fsst::Decompressor] that can be used to decompress values from
    /// this array.
    ///
    /// This is private to the crate to avoid leaking `fsst-rs` types as part of the public API.
    pub(crate) fn decompressor(&self) -> Decompressor {
        Decompressor::new(self.symbols().as_slice(), self.symbol_lengths().as_slice())
    }
}

impl ArrayImpl for FSSTArray {
    type Encoding = FSSTEncoding;

    fn _len(&self) -> usize {
        self.codes.len()
    }

    fn _dtype(&self) -> &DType {
        &self.dtype
    }

    fn _vtable(&self) -> VTableRef {
        VTableRef::new_ref(&FSSTEncoding)
    }

    fn _with_children(&self, children: &[ArrayRef]) -> VortexResult<Self> {
        let codes = children[0].clone();
        let uncompressed_lengths = children[1].clone();

        Self::try_new(
            self.dtype().clone(),
            self.symbols().clone(),
            self.symbol_lengths().clone(),
            codes,
            uncompressed_lengths,
        )
    }
}

impl ArrayStatisticsImpl for FSSTArray {
    fn _stats_ref(&self) -> StatsSetRef<'_> {
        self.stats_set.to_ref(self)
    }
}

impl ArrayValidityImpl for FSSTArray {
    fn _is_valid(&self, index: usize) -> VortexResult<bool> {
        self.codes().is_valid(index)
    }

    fn _all_valid(&self) -> VortexResult<bool> {
        self.codes().all_valid()
    }

    fn _all_invalid(&self) -> VortexResult<bool> {
        self.codes().all_invalid()
    }

    fn _validity_mask(&self) -> VortexResult<Mask> {
        self.codes().validity_mask()
    }
}

impl ArrayVariantsImpl for FSSTArray {
    fn _as_utf8_typed(&self) -> Option<&dyn Utf8ArrayTrait> {
        Some(self)
    }

    fn _as_binary_typed(&self) -> Option<&dyn BinaryArrayTrait> {
        Some(self)
    }
}

impl Utf8ArrayTrait for FSSTArray {}

impl BinaryArrayTrait for FSSTArray {}
