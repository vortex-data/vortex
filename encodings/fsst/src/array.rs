// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::fmt::Formatter;
use std::hash::Hasher;
use std::sync::Arc;
use std::sync::LazyLock;

use fsst::Compressor;
use fsst::Decompressor;
use fsst::Symbol;
use prost::Message as _;
use vortex_array::Array;
use vortex_array::ArrayEq;
use vortex_array::ArrayHash;
use vortex_array::ArrayId;
use vortex_array::ArrayParts;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::ExecutionResult;
use vortex_array::IntoArray;
use vortex_array::Precision;
use vortex_array::TypedArrayRef;
use vortex_array::arrays::VarBin;
use vortex_array::arrays::VarBinArray;
use vortex_array::arrays::varbin::VarBinArrayExt;
use vortex_array::buffer::BufferHandle;
use vortex_array::builders::ArrayBuilder;
use vortex_array::builders::VarBinViewBuilder;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::serde::ArrayChildren;
use vortex_array::validity::Validity;
use vortex_array::vtable;
use vortex_array::vtable::VTable;
use vortex_array::vtable::ValidityChild;
use vortex_array::vtable::ValidityVTableFromChild;
use vortex_array::vtable::validity_to_child;
use vortex_buffer::Buffer;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;

use crate::canonical::canonicalize_fsst;
use crate::canonical::fsst_decode_views;
use crate::kernel::PARENT_KERNELS;
use crate::rules::RULES;

vtable!(FSST, FSST, FSSTData);

#[derive(Clone, prost::Message)]
pub struct FSSTMetadata {
    #[prost(enumeration = "PType", tag = "1")]
    uncompressed_lengths_ptype: i32,

    #[prost(enumeration = "PType", tag = "2")]
    codes_offsets_ptype: i32,
}

impl FSSTMetadata {
    pub fn get_uncompressed_lengths_ptype(&self) -> VortexResult<PType> {
        PType::try_from(self.uncompressed_lengths_ptype)
            .map_err(|_| vortex_err!("Invalid PType {}", self.uncompressed_lengths_ptype))
    }
}

impl ArrayHash for FSSTData {
    fn array_hash<H: Hasher>(&self, state: &mut H, precision: Precision) {
        self.symbols.array_hash(state, precision);
        self.symbol_lengths.array_hash(state, precision);
        self.codes.clone().into_array().array_hash(state, precision);
    }
}

impl ArrayEq for FSSTData {
    fn array_eq(&self, other: &Self, precision: Precision) -> bool {
        self.symbols.array_eq(&other.symbols, precision)
            && self
                .symbol_lengths
                .array_eq(&other.symbol_lengths, precision)
            && self
                .codes
                .clone()
                .into_array()
                .array_eq(&other.codes.clone().into_array(), precision)
    }
}

impl VTable for FSST {
    type ArrayData = FSSTData;
    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromChild;

    fn id(&self) -> ArrayId {
        Self::ID
    }

    fn validate(
        &self,
        data: &Self::ArrayData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        data.validate(dtype, len, slots)
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        3
    }

    fn buffer(array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        match idx {
            0 => BufferHandle::new_host(array.symbols().clone().into_byte_buffer()),
            1 => BufferHandle::new_host(array.symbol_lengths().clone().into_byte_buffer()),
            2 => array.codes.bytes_handle().clone(),
            _ => vortex_panic!("FSSTArray buffer index {idx} out of bounds"),
        }
    }

    fn buffer_name(_array: ArrayView<'_, Self>, idx: usize) -> Option<String> {
        match idx {
            0 => Some("symbols".to_string()),
            1 => Some("symbol_lengths".to_string()),
            2 => Some("compressed_codes".to_string()),
            _ => vortex_panic!("FSSTArray buffer_name index {idx} out of bounds"),
        }
    }

    fn serialize(array: ArrayView<'_, Self>) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(
            FSSTMetadata {
                uncompressed_lengths_ptype: array.uncompressed_lengths().dtype().as_ptype().into(),
                codes_offsets_ptype: array.codes.offsets().dtype().as_ptype().into(),
            }
            .encode_to_vec(),
        ))
    }

    fn deserialize(
        &self,
        dtype: &DType,
        len: usize,
        metadata: &[u8],
        buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
        _session: &VortexSession,
    ) -> VortexResult<ArrayParts<Self>> {
        let metadata = FSSTMetadata::decode(metadata)?;
        let symbols = Buffer::<Symbol>::from_byte_buffer(buffers[0].clone().try_to_host_sync()?);
        let symbol_lengths = Buffer::<u8>::from_byte_buffer(buffers[1].clone().try_to_host_sync()?);

        // Check for the legacy deserialization path.
        if buffers.len() == 2 {
            if children.len() != 2 {
                vortex_bail!(InvalidArgument: "Expected 2 children, got {}", children.len());
            }
            let codes = children.get(0, &DType::Binary(dtype.nullability()), len)?;
            let codes: VarBinArray = codes
                .as_opt::<VarBin>()
                .ok_or_else(|| {
                    vortex_err!(
                        "Expected VarBinArray for codes, got {}",
                        codes.encoding_id()
                    )
                })?
                .into_owned();
            let uncompressed_lengths = children.get(
                1,
                &DType::Primitive(
                    metadata.get_uncompressed_lengths_ptype()?,
                    Nullability::NonNullable,
                ),
                len,
            )?;

            FSSTData::validate_parts(
                &symbols,
                &symbol_lengths,
                &codes,
                &uncompressed_lengths,
                dtype,
                len,
            )?;
            let slots = FSSTData::make_slots(&codes, &uncompressed_lengths);
            let data = FSSTData::try_new(symbols, symbol_lengths, codes, dtype)?;
            return Ok(ArrayParts::new(self.clone(), dtype.clone(), len, data).with_slots(slots));
        }

        // Check for the current deserialization path.
        if buffers.len() == 3 {
            let uncompressed_lengths = children.get(
                0,
                &DType::Primitive(
                    metadata.get_uncompressed_lengths_ptype()?,
                    Nullability::NonNullable,
                ),
                len,
            )?;

            let codes_buffer = ByteBuffer::from_byte_buffer(buffers[2].clone().try_to_host_sync()?);
            let codes_offsets = children.get(
                1,
                &DType::Primitive(
                    PType::try_from(metadata.codes_offsets_ptype)?,
                    Nullability::NonNullable,
                ),
                // VarBin offsets are len + 1
                len + 1,
            )?;

            let codes_validity = if children.len() == 2 {
                Validity::from(dtype.nullability())
            } else if children.len() == 3 {
                let validity = children.get(2, &Validity::DTYPE, len)?;
                Validity::Array(validity)
            } else {
                vortex_bail!("Expected 0 or 1 child, got {}", children.len());
            };

            let codes = VarBinArray::try_new(
                codes_offsets,
                codes_buffer,
                DType::Binary(dtype.nullability()),
                codes_validity,
            )?;

            FSSTData::validate_parts(
                &symbols,
                &symbol_lengths,
                &codes,
                &uncompressed_lengths,
                dtype,
                len,
            )?;
            let slots = FSSTData::make_slots(&codes, &uncompressed_lengths);
            let data = FSSTData::try_new(symbols, symbol_lengths, codes, dtype)?;
            return Ok(ArrayParts::new(self.clone(), dtype.clone(), len, data).with_slots(slots));
        }

        vortex_bail!(
            "InvalidArgument: Expected 2 or 3 buffers, got {}",
            buffers.len()
        );
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        SLOT_NAMES[idx].to_string()
    }

    fn execute(array: Array<Self>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        canonicalize_fsst(array.as_view(), ctx).map(ExecutionResult::done)
    }

    fn append_to_builder(
        array: ArrayView<'_, Self>,
        builder: &mut dyn ArrayBuilder,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        let Some(builder) = builder.as_any_mut().downcast_mut::<VarBinViewBuilder>() else {
            builder.extend_from_array(
                &array
                    .array()
                    .clone()
                    .execute::<Canonical>(ctx)?
                    .into_array(),
            );
            return Ok(());
        };

        // Decompress the whole block of data into a new buffer, and create some views
        // from it instead.
        let (buffers, views) = fsst_decode_views(array, builder.completed_block_count(), ctx)?;

        builder.push_buffer_and_adjusted_views(&buffers, &views, array.array().validity_mask()?);
        Ok(())
    }

    fn execute_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_KERNELS.execute(array, parent, child_idx, ctx)
    }

    fn reduce_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        RULES.evaluate(array, parent, child_idx)
    }
}

/// Lengths of the original values before compression, can be compressed.
pub(crate) const UNCOMPRESSED_LENGTHS_SLOT: usize = 0;
/// The offsets array for the FSST-compressed codes.
#[allow(dead_code, reason = "reserved for back-compat slot numbering")]
pub(crate) const CODES_OFFSETS_SLOT: usize = 1;
/// The validity bitmap for the compressed codes.
#[allow(dead_code, reason = "reserved for back-compat slot numbering")]
pub(crate) const CODES_VALIDITY_SLOT: usize = 2;
pub(crate) const NUM_SLOTS: usize = 3;
pub(crate) const SLOT_NAMES: [&str; NUM_SLOTS] =
    ["uncompressed_lengths", "codes_offsets", "codes_validity"];

#[derive(Clone)]
pub struct FSSTData {
    symbols: Buffer<Symbol>,
    symbol_lengths: Buffer<u8>,
    codes: VarBinArray,
    /// NOTE(ngates): this === codes, but is stored as an ArrayRef so we can return &ArrayRef!
    codes_array: ArrayRef,

    /// Memoized compressor used for push-down of compute by compressing the RHS.
    compressor: Arc<LazyLock<Compressor, Box<dyn Fn() -> Compressor + Send>>>,
}

impl Debug for FSSTData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FSSTArray")
            .field("symbols", &self.symbols)
            .field("symbol_lengths", &self.symbol_lengths)
            .field("codes", &self.codes)
            .field("uncompressed_lengths", &"<outer slot>")
            .finish()
    }
}

#[derive(Clone, Debug)]
pub struct FSST;

impl FSST {
    pub const ID: ArrayId = ArrayId::new_ref("vortex.fsst");

    /// Build an FSST array from a set of `symbols` and `codes`.
    pub fn try_new(
        dtype: DType,
        symbols: Buffer<Symbol>,
        symbol_lengths: Buffer<u8>,
        codes: VarBinArray,
        uncompressed_lengths: ArrayRef,
    ) -> VortexResult<FSSTArray> {
        let len = codes.len();
        FSSTData::validate_parts(
            &symbols,
            &symbol_lengths,
            &codes,
            &uncompressed_lengths,
            &dtype,
            len,
        )?;
        let slots = vec![
            Some(uncompressed_lengths),
            Some(codes.offsets().clone()),
            validity_to_child(
                &codes
                    .validity()
                    .vortex_expect("FSST codes validity should be derivable"),
                codes.len(),
            ),
        ];
        let data = FSSTData::try_new(symbols, symbol_lengths, codes, &dtype)?;
        Ok(unsafe {
            Array::from_parts_unchecked(ArrayParts::new(FSST, dtype, len, data).with_slots(slots))
        })
    }

    pub(crate) unsafe fn new_unchecked(
        dtype: DType,
        symbols: Buffer<Symbol>,
        symbol_lengths: Buffer<u8>,
        codes: VarBinArray,
        uncompressed_lengths: ArrayRef,
    ) -> FSSTArray {
        let len = codes.len();
        let slots = vec![
            Some(uncompressed_lengths),
            Some(codes.offsets().clone()),
            validity_to_child(
                &codes
                    .validity()
                    .vortex_expect("FSST codes validity should be derivable"),
                codes.len(),
            ),
        ];
        let data = unsafe { FSSTData::new_unchecked(symbols, symbol_lengths, codes) };
        unsafe {
            Array::from_parts_unchecked(ArrayParts::new(FSST, dtype, len, data).with_slots(slots))
        }
    }
}

impl FSSTData {
    fn make_slots(codes: &VarBinArray, uncompressed_lengths: &ArrayRef) -> Vec<Option<ArrayRef>> {
        vec![
            Some(uncompressed_lengths.clone()),
            Some(codes.offsets().clone()),
            validity_to_child(
                &codes
                    .validity()
                    .vortex_expect("FSST codes validity should be derivable"),
                codes.len(),
            ),
        ]
    }

    /// Build an FSST array from a set of `symbols` and `codes`.
    ///
    /// Symbols are 8-bytes and can represent short strings, each of which is assigned
    /// a code.
    ///
    /// The `codes` array is a Binary array where each binary datum is a sequence of 8-bit codes.
    /// Each code corresponds either to a symbol, or to the "escape code",
    /// which tells the decoder to emit the following byte without doing a table lookup.
    pub fn try_new(
        symbols: Buffer<Symbol>,
        symbol_lengths: Buffer<u8>,
        codes: VarBinArray,
        _dtype: &DType,
    ) -> VortexResult<Self> {
        // SAFETY: all components validated above
        unsafe { Ok(Self::new_unchecked(symbols, symbol_lengths, codes)) }
    }

    pub fn validate(
        &self,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        Self::validate_parts(
            &self.symbols,
            &self.symbol_lengths,
            &self.codes,
            uncompressed_lengths_from_slots(slots),
            dtype,
            len,
        )
    }

    fn validate_parts(
        symbols: &Buffer<Symbol>,
        symbol_lengths: &Buffer<u8>,
        codes: &VarBinArray,
        uncompressed_lengths: &ArrayRef,
        dtype: &DType,
        len: usize,
    ) -> VortexResult<()> {
        vortex_ensure!(
            matches!(dtype, DType::Binary(_) | DType::Utf8(_)),
            "FSST arrays must be Binary or Utf8, found {dtype}"
        );

        // Check: symbols must not have length > MAX_CODE
        if symbols.len() > 255 {
            vortex_bail!(InvalidArgument: "symbols array must have length <= 255");
        }
        if symbols.len() != symbol_lengths.len() {
            vortex_bail!(InvalidArgument: "symbols and symbol_lengths arrays must have same length");
        }

        if codes.len() != len {
            vortex_bail!(InvalidArgument: "codes must have same len as outer array");
        }

        if uncompressed_lengths.len() != len {
            vortex_bail!(InvalidArgument: "uncompressed_lengths must be same len as codes");
        }

        if !uncompressed_lengths.dtype().is_int() || uncompressed_lengths.dtype().is_nullable() {
            vortex_bail!(InvalidArgument: "uncompressed_lengths must have integer type and cannot be nullable, found {}", uncompressed_lengths.dtype());
        }

        // Check: strings must be a Binary array.
        if !matches!(codes.dtype(), DType::Binary(_)) {
            vortex_bail!(InvalidArgument: "codes array must be DType::Binary type");
        }
        if codes.dtype().nullability() != dtype.nullability() {
            vortex_bail!(InvalidArgument: "codes nullability must match outer dtype nullability");
        }

        Ok(())
    }

    pub(crate) unsafe fn new_unchecked(
        symbols: Buffer<Symbol>,
        symbol_lengths: Buffer<u8>,
        codes: VarBinArray,
    ) -> Self {
        let symbols2 = symbols.clone();
        let symbol_lengths2 = symbol_lengths.clone();
        let compressor = Arc::new(LazyLock::new(Box::new(move || {
            Compressor::rebuild_from(symbols2.as_slice(), symbol_lengths2.as_slice())
        })
            as Box<dyn Fn() -> Compressor + Send>));
        let codes_array = codes.clone().into_array();

        Self {
            symbols,
            symbol_lengths,
            codes,
            codes_array,
            compressor,
        }
    }

    /// Returns the number of elements in the array.
    pub fn len(&self) -> usize {
        self.codes.len()
    }

    /// Returns `true` if the array contains no elements.
    pub fn is_empty(&self) -> bool {
        self.codes.len() == 0
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

fn uncompressed_lengths_from_slots(slots: &[Option<ArrayRef>]) -> &ArrayRef {
    slots[UNCOMPRESSED_LENGTHS_SLOT]
        .as_ref()
        .vortex_expect("FSSTArray uncompressed_lengths slot")
}

pub trait FSSTArrayExt: TypedArrayRef<FSST> {
    fn uncompressed_lengths(&self) -> &ArrayRef {
        uncompressed_lengths_from_slots(self.as_ref().slots())
    }

    fn uncompressed_lengths_dtype(&self) -> &DType {
        self.uncompressed_lengths().dtype()
    }
}

impl<T: TypedArrayRef<FSST>> FSSTArrayExt for T {}

impl ValidityChild<FSST> for FSST {
    fn validity_child(array: ArrayView<'_, FSST>) -> ArrayRef {
        array.codes_array.clone()
    }
}

#[cfg(test)]
mod test {
    use fsst::Compressor;
    use fsst::Symbol;
    use prost::Message;
    use vortex_array::ArrayPlugin;
    use vortex_array::IntoArray;
    use vortex_array::LEGACY_SESSION;
    use vortex_array::VortexSessionExecute;
    use vortex_array::accessor::ArrayAccessor;
    use vortex_array::arrays::VarBinViewArray;
    use vortex_array::buffer::BufferHandle;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::test_harness::check_metadata;
    use vortex_buffer::Buffer;
    use vortex_error::VortexError;

    use crate::FSST;
    use crate::array::FSSTArrayExt;
    use crate::array::FSSTMetadata;
    use crate::fsst_compress_iter;

    #[cfg_attr(miri, ignore)]
    #[test]
    fn test_fsst_metadata() {
        check_metadata(
            "fsst.metadata",
            &FSSTMetadata {
                uncompressed_lengths_ptype: PType::U64 as i32,
                codes_offsets_ptype: PType::I32 as i32,
            }
            .encode_to_vec(),
        );
    }

    /// The original FSST array stored codes as a VarBinArray child and required that the child
    /// have this encoding. Vortex forbids this kind of introspection, therefore we had to fix
    /// the array to store the compressed offsets and compressed data buffer separately, and only
    /// use VarBinArray to delegate behavior.
    ///
    /// This test manually constructs an old-style FSST array and ensures that it can still be
    /// deserialized.
    #[test]
    fn test_back_compat() {
        let symbols = Buffer::<Symbol>::copy_from([
            Symbol::from_slice(b"abc00000"),
            Symbol::from_slice(b"defghijk"),
        ]);
        let symbol_lengths = Buffer::<u8>::copy_from([3, 8]);

        let compressor = Compressor::rebuild_from(symbols.as_slice(), symbol_lengths.as_slice());
        let fsst_array = fsst_compress_iter(
            [Some(b"abcabcab".as_ref()), Some(b"defghijk".as_ref())].into_iter(),
            2,
            DType::Utf8(Nullability::NonNullable),
            &compressor,
        );

        let compressed_codes = fsst_array.codes().clone();

        // There were two buffers:
        // 1. The 8 byte symbols
        // 2. The symbol lengths as u8.
        let buffers = [
            BufferHandle::new_host(symbols.into_byte_buffer()),
            BufferHandle::new_host(symbol_lengths.into_byte_buffer()),
        ];

        // There were 2 children:
        // 1. The compressed codes, stored as a VarBinArray.
        // 2. The uncompressed lengths, stored as a Primitive array.
        let children = vec![
            compressed_codes.into_array(),
            fsst_array.uncompressed_lengths().clone(),
        ];

        let fsst = ArrayPlugin::deserialize(
            &FSST,
            &DType::Utf8(Nullability::NonNullable),
            2,
            &FSSTMetadata {
                uncompressed_lengths_ptype: fsst_array
                    .uncompressed_lengths()
                    .dtype()
                    .as_ptype()
                    .into(),
                // Legacy array did not store this field, use Protobuf default of 0.
                codes_offsets_ptype: 0,
            }
            .encode_to_vec(),
            &buffers,
            &children.as_slice(),
            &LEGACY_SESSION,
        )
        .unwrap();

        let decompressed = fsst
            .execute::<VarBinViewArray>(&mut LEGACY_SESSION.create_execution_ctx())
            .unwrap();
        decompressed
            .with_iterator(|it| {
                assert_eq!(it.next().unwrap(), Some(b"abcabcab".as_ref()));
                assert_eq!(it.next().unwrap(), Some(b"defghijk".as_ref()));
                Ok::<_, VortexError>(())
            })
            .unwrap()
    }
}
