// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::fmt::Formatter;
use std::hash::Hash;
use std::ops::Range;
use std::sync::Arc;
use std::sync::LazyLock;

use fsst::Compressor;
use fsst::Decompressor;
use fsst::Symbol;
use vortex_array::Array;
use vortex_array::ArrayBufferVisitor;
use vortex_array::ArrayChildVisitor;
use vortex_array::ArrayEq;
use vortex_array::ArrayHash;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::DeserializeMetadata;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::Precision;
use vortex_array::ProstMetadata;
use vortex_array::SerializeMetadata;
use vortex_array::arrays::VarBinArray;
use vortex_array::arrays::VarBinVTable;
use vortex_array::buffer::BufferHandle;
use vortex_array::serde::ArrayChildren;
use vortex_array::stats::ArrayStats;
use vortex_array::stats::StatsSetRef;
use vortex_array::vtable;
use vortex_array::vtable::ArrayId;
use vortex_array::vtable::ArrayVTable;
use vortex_array::vtable::ArrayVTableExt;
use vortex_array::vtable::BaseArrayVTable;
use vortex_array::vtable::EncodeVTable;
use vortex_array::vtable::NotSupported;
use vortex_array::vtable::VTable;
use vortex_array::vtable::ValidityChild;
use vortex_array::vtable::ValidityVTableFromChild;
use vortex_array::vtable::VisitorVTable;
use vortex_buffer::Buffer;
use vortex_dtype::DType;
use vortex_dtype::Nullability;
use vortex_dtype::PType;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;

use crate::fsst_compress;
use crate::fsst_train_compressor;
use crate::kernel::PARENT_KERNELS;

vtable!(FSST);

#[derive(Clone, prost::Message)]
pub struct FSSTMetadata {
    #[prost(enumeration = "PType", tag = "1")]
    uncompressed_lengths_ptype: i32,
}

impl FSSTMetadata {
    pub fn get_uncompressed_lengths_ptype(&self) -> VortexResult<PType> {
        PType::try_from(self.uncompressed_lengths_ptype)
            .map_err(|_| vortex_err!("Invalid PType {}", self.uncompressed_lengths_ptype))
    }
}

impl VTable for FSSTVTable {
    type Array = FSSTArray;

    type Metadata = ProstMetadata<FSSTMetadata>;

    type ArrayVTable = Self;
    type CanonicalVTable = Self;
    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromChild;
    type VisitorVTable = Self;
    type ComputeVTable = NotSupported;
    type EncodeVTable = Self;

    fn id(&self) -> ArrayId {
        ArrayId::new_ref("vortex.fsst")
    }

    fn encoding(_array: &Self::Array) -> ArrayVTable {
        FSSTVTable.as_vtable()
    }

    fn metadata(array: &FSSTArray) -> VortexResult<Self::Metadata> {
        Ok(ProstMetadata(FSSTMetadata {
            uncompressed_lengths_ptype: PType::try_from(array.uncompressed_lengths().dtype())?
                as i32,
        }))
    }

    fn serialize(metadata: Self::Metadata) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(metadata.serialize()))
    }

    fn deserialize(buffer: &[u8]) -> VortexResult<Self::Metadata> {
        Ok(ProstMetadata(
            <ProstMetadata<FSSTMetadata> as DeserializeMetadata>::deserialize(buffer)?,
        ))
    }

    fn build(
        &self,
        dtype: &DType,
        len: usize,
        metadata: &Self::Metadata,
        buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
    ) -> VortexResult<FSSTArray> {
        if buffers.len() != 2 {
            vortex_bail!(InvalidArgument: "Expected 2 buffers, got {}", buffers.len());
        }
        let symbols = Buffer::<Symbol>::from_byte_buffer(buffers[0].clone().try_to_host()?);
        let symbol_lengths = Buffer::<u8>::from_byte_buffer(buffers[1].clone().try_to_host()?);

        if children.len() != 2 {
            vortex_bail!(InvalidArgument: "Expected 2 children, got {}", children.len());
        }
        let codes = children.get(0, &DType::Binary(dtype.nullability()), len)?;
        let codes = codes
            .as_opt::<VarBinVTable>()
            .ok_or_else(|| {
                vortex_err!(
                    "Expected VarBinArray for codes, got {}",
                    codes.encoding_id()
                )
            })?
            .clone();
        let uncompressed_lengths = children.get(
            1,
            &DType::Primitive(
                metadata.0.get_uncompressed_lengths_ptype()?,
                Nullability::NonNullable,
            ),
            len,
        )?;

        FSSTArray::try_new(
            dtype.clone(),
            symbols,
            symbol_lengths,
            codes,
            uncompressed_lengths,
        )
    }

    fn with_children(array: &mut Self::Array, children: Vec<ArrayRef>) -> VortexResult<()> {
        vortex_ensure!(
            children.len() == 2,
            "FSSTArray expects 2 children, got {}",
            children.len()
        );

        let mut children_iter = children.into_iter();
        let codes = children_iter
            .next()
            .ok_or_else(|| vortex_err!("FSSTArray with_children missing codes"))?;

        let codes = codes
            .as_opt::<VarBinVTable>()
            .ok_or_else(|| {
                vortex_err!(
                    "Expected VarBinArray for codes, got {}",
                    codes.encoding_id()
                )
            })?
            .clone();
        let uncompressed_lengths = children_iter
            .next()
            .ok_or_else(|| vortex_err!("FSSTArray with_children missing uncompressed_lengths"))?;

        array.codes = codes;
        array.uncompressed_lengths = uncompressed_lengths;

        Ok(())
    }

    fn execute_parent(
        array: &Self::Array,
        parent: &ArrayRef,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Canonical>> {
        PARENT_KERNELS.execute(array, parent, child_idx, ctx)
    }

    fn slice(array: &Self::Array, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        // SAFETY: slicing the `codes` leaves the symbol table intact
        Ok(Some(
            unsafe {
                FSSTArray::new_unchecked(
                    array.dtype().clone(),
                    array.symbols().clone(),
                    array.symbol_lengths().clone(),
                    VarBinVTable::slice(array.codes().as_::<VarBinVTable>(), range.clone())?
                        .vortex_expect("varbin slice cannot fail")
                        .as_::<VarBinVTable>()
                        .clone(),
                    array.uncompressed_lengths().slice(range),
                )
            }
            .into_array(),
        ))
    }
}

#[derive(Clone)]
pub struct FSSTArray {
    dtype: DType,
    symbols: Buffer<Symbol>,
    symbol_lengths: Buffer<u8>,
    codes: VarBinArray,
    /// NOTE(ngates): this === codes, but is stored as an ArrayRef so we can return &ArrayRef!
    codes_array: ArrayRef,
    /// Lengths of the original values before compression, can be compressed.
    uncompressed_lengths: ArrayRef,
    stats_set: ArrayStats,

    /// Memoized compressor used for push-down of compute by compressing the RHS.
    compressor: Arc<LazyLock<Compressor, Box<dyn Fn() -> Compressor + Send>>>,
}

impl Debug for FSSTArray {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FSSTArray")
            .field("dtype", &self.dtype)
            .field("symbols", &self.symbols)
            .field("symbol_lengths", &self.symbol_lengths)
            .field("codes", &self.codes)
            .field("uncompressed_lengths", &self.uncompressed_lengths)
            .finish()
    }
}

#[derive(Debug)]
pub struct FSSTVTable;

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
        codes: VarBinArray,
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

        // Check: strings must be a Binary array.
        if !matches!(codes.dtype(), DType::Binary(_)) {
            vortex_bail!(InvalidArgument: "codes array must be DType::Binary type");
        }

        // SAFETY: all components validated above
        unsafe {
            Ok(Self::new_unchecked(
                dtype,
                symbols,
                symbol_lengths,
                codes,
                uncompressed_lengths,
            ))
        }
    }

    pub(crate) unsafe fn new_unchecked(
        dtype: DType,
        symbols: Buffer<Symbol>,
        symbol_lengths: Buffer<u8>,
        codes: VarBinArray,
        uncompressed_lengths: ArrayRef,
    ) -> Self {
        let symbols2 = symbols.clone();
        let symbol_lengths2 = symbol_lengths.clone();
        let compressor = Arc::new(LazyLock::new(Box::new(move || {
            Compressor::rebuild_from(symbols2.as_slice(), symbol_lengths2.as_slice())
        })
            as Box<dyn Fn() -> Compressor + Send>));
        let codes_array = codes.to_array();

        Self {
            dtype,
            symbols,
            symbol_lengths,
            codes,
            codes_array,
            uncompressed_lengths,
            stats_set: Default::default(),
            compressor,
        }
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
    pub fn codes(&self) -> &VarBinArray {
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
    pub fn decompressor(&self) -> Decompressor<'_> {
        Decompressor::new(self.symbols().as_slice(), self.symbol_lengths().as_slice())
    }

    /// Retrieves the FSST compressor.
    pub fn compressor(&self) -> &Compressor {
        self.compressor.as_ref()
    }
}

impl BaseArrayVTable<FSSTVTable> for FSSTVTable {
    fn len(array: &FSSTArray) -> usize {
        array.codes().len()
    }

    fn dtype(array: &FSSTArray) -> &DType {
        &array.dtype
    }

    fn stats(array: &FSSTArray) -> StatsSetRef<'_> {
        array.stats_set.to_ref(array.as_ref())
    }

    fn array_hash<H: std::hash::Hasher>(array: &FSSTArray, state: &mut H, precision: Precision) {
        array.dtype.hash(state);
        array.symbols.array_hash(state, precision);
        array.symbol_lengths.array_hash(state, precision);
        array.codes.as_ref().array_hash(state, precision);
        array.uncompressed_lengths.array_hash(state, precision);
    }

    fn array_eq(array: &FSSTArray, other: &FSSTArray, precision: Precision) -> bool {
        array.dtype == other.dtype
            && array.symbols.array_eq(&other.symbols, precision)
            && array
                .symbol_lengths
                .array_eq(&other.symbol_lengths, precision)
            && array
                .codes
                .as_ref()
                .array_eq(other.codes.as_ref(), precision)
            && array
                .uncompressed_lengths
                .array_eq(&other.uncompressed_lengths, precision)
    }
}

impl ValidityChild<FSSTVTable> for FSSTVTable {
    fn validity_child(array: &FSSTArray) -> &ArrayRef {
        &array.codes_array
    }
}

impl EncodeVTable<FSSTVTable> for FSSTVTable {
    fn encode(
        _vtable: &FSSTVTable,
        canonical: &Canonical,
        like: Option<&FSSTArray>,
    ) -> VortexResult<Option<FSSTArray>> {
        let array = canonical.clone().into_varbinview();

        let compressor = match like {
            Some(like) => Compressor::rebuild_from(like.symbols(), like.symbol_lengths()),
            None => fsst_train_compressor(&array),
        };

        Ok(Some(fsst_compress(array, &compressor)))
    }
}

impl VisitorVTable<FSSTVTable> for FSSTVTable {
    fn visit_buffers(array: &FSSTArray, visitor: &mut dyn ArrayBufferVisitor) {
        visitor.visit_buffer(&array.symbols().clone().into_byte_buffer());
        visitor.visit_buffer(&array.symbol_lengths().clone().into_byte_buffer());
    }

    fn visit_children(array: &FSSTArray, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_child("codes", &array.codes().to_array());
        visitor.visit_child("uncompressed_lengths", array.uncompressed_lengths());
    }
}

#[cfg(test)]
mod test {
    use vortex_array::ProstMetadata;
    use vortex_array::test_harness::check_metadata;
    use vortex_dtype::PType;

    use crate::array::FSSTMetadata;

    #[cfg_attr(miri, ignore)]
    #[test]
    fn test_fsst_metadata() {
        check_metadata(
            "fsst.metadata",
            ProstMetadata(FSSTMetadata {
                uncompressed_lengths_ptype: PType::U64 as i32,
            }),
        );
    }
}
