// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hasher;
use std::sync::Arc;
use std::sync::OnceLock;

use fsst::Compressor;
use fsst::Decompressor;
use fsst::Symbol;
use num_traits::AsPrimitive;
use prost::Message as _;
use vortex_array::Array;
use vortex_array::ArrayEq;
use vortex_array::ArrayHash;
use vortex_array::ArrayId;
use vortex_array::ArrayParts;
use vortex_array::ArrayRef;
use vortex_array::ArraySlots;
use vortex_array::ArrayView;
use vortex_array::Canonical;
use vortex_array::EqMode;
use vortex_array::ExecutionCtx;
use vortex_array::ExecutionResult;
use vortex_array::IntoArray;
use vortex_array::TypedArrayRef;
use vortex_array::VortexSessionExecute;
use vortex_array::array_slots;
use vortex_array::arrays::VarBin;
use vortex_array::arrays::VarBinArray;
use vortex_array::arrays::varbin::VarBinArraySlotsExt;
use vortex_array::buffer::BufferHandle;
use vortex_array::builders::ArrayBuilder;
use vortex_array::builders::VarBinBuilder;
use vortex_array::builders::VarBinViewBuilder;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::OffsetBuilderPType;
use vortex_array::dtype::PType;
use vortex_array::legacy_session;
use vortex_array::match_each_integer_ptype;
use vortex_array::match_each_varbin_builder;
use vortex_array::serde::ArrayChildren;
use vortex_array::validity::Validity;
use vortex_array::vtable::VTable;
use vortex_array::vtable::ValidityVTable;
use vortex_array::vtable::child_to_validity;
use vortex_array::vtable::validity_to_child;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::canonical::canonicalize_fsst;
use crate::canonical::fsst_decode_bytes;
use crate::canonical::fsst_decode_views;
use crate::rules::RULES;

/// A [`FSST`]-encoded Vortex array.
pub type FSSTArray = Array<FSST>;

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

/// The number of entries in a fully-populated FSST symbol table.
///
/// Code 255 is reserved as the escape code, leaving 255 usable codes. [`Decompressor`] borrows
/// the symbol table as fixed-size arrays of exactly this length, so Vortex pads the symbols and
/// symbol lengths buffers out to it on construction.
pub const FSST_SYMBOL_TABLE_LEN: usize = 255;

impl ArrayHash for FSSTData {
    fn array_hash<H: Hasher>(&self, state: &mut H, precision: EqMode) {
        self.padded_symbols().array_hash(state, precision);
        self.padded_symbol_lengths().array_hash(state, precision);
        self.codes_bytes.as_host().array_hash(state, precision);
    }
}

impl ArrayEq for FSSTData {
    fn array_eq(&self, other: &Self, precision: EqMode) -> bool {
        self.padded_symbols()
            .array_eq(other.padded_symbols(), precision)
            && self
                .padded_symbol_lengths()
                .array_eq(other.padded_symbol_lengths(), precision)
            && self
                .codes_bytes
                .as_host()
                .array_eq(other.codes_bytes.as_host(), precision)
    }
}

impl VTable for FSST {
    type TypedArrayData = FSSTData;
    type OperationsVTable = Self;
    type ValidityVTable = Self;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.fsst");
        *ID
    }

    #[allow(clippy::disallowed_methods)]
    fn validate(
        &self,
        data: &Self::TypedArrayData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        // TODO(ctx): trait fixes - VTable::validate has a fixed signature.
        let mut ctx = legacy_session().create_execution_ctx();
        data.validate(dtype, len, slots, &mut ctx)
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        3
    }

    fn buffer(array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        match idx {
            0 => BufferHandle::new_host(array.symbols().into_byte_buffer()),
            1 => BufferHandle::new_host(array.symbol_lengths().into_byte_buffer()),
            2 => array.codes_bytes_handle().clone(),
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

    fn with_buffers(
        &self,
        array: ArrayView<'_, Self>,
        buffers: &[BufferHandle],
    ) -> VortexResult<ArrayParts<Self>> {
        vortex_ensure!(
            buffers.len() == 3,
            "Expected 3 buffers, got {}",
            buffers.len()
        );
        let symbols = Buffer::<Symbol>::from_byte_buffer(buffers[0].clone().try_to_host_sync()?);
        let symbol_lengths = Buffer::<u8>::from_byte_buffer(buffers[1].clone().try_to_host_sync()?);
        let data = FSSTData::try_new(symbols, symbol_lengths, buffers[2].clone(), array.len())?;
        Ok(
            ArrayParts::new(self.clone(), array.dtype().clone(), array.len(), data)
                .with_slots(array.slots().iter().cloned().collect()),
        )
    }

    fn serialize(
        array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        let codes_offsets = array.codes_offsets();
        Ok(Some(
            FSSTMetadata {
                uncompressed_lengths_ptype: array.uncompressed_lengths().dtype().as_ptype().into(),
                codes_offsets_ptype: codes_offsets.dtype().as_ptype().into(),
            }
            .encode_to_vec(),
        ))
    }

    /// Deserializes an FSST array from its serialized components.
    ///
    /// Supports two serialization formats:
    ///
    /// ## Legacy format (2 buffers, 2 children)
    ///
    /// The original FSST layout stored the compressed codes as a full `VarBinArray` child.
    /// - **Buffers**: `[symbols, symbol_lengths]`
    /// - **Children**: `[codes (VarBinArray), uncompressed_lengths (Primitive)]`
    ///
    /// The codes VarBinArray child is decomposed: its bytes become the `codes_bytes` buffer,
    /// and its offsets/validity are extracted into slots.
    /// See `FSST::deserialize_legacy`.
    ///
    /// ## Current format (3 buffers, 2-3 children)
    ///
    /// The current layout stores the compressed bytes as a raw buffer alongside the symbol
    /// table, with offsets and validity as separate children.
    /// - **Buffers**: `[symbols, symbol_lengths, compressed_codes_bytes]`
    /// - **Children**: `[uncompressed_lengths, codes_offsets, (optional) codes_validity]`
    ///
    /// The `codes_bytes` buffer is stored directly in `FSSTData`. A `VarBinArray` for the
    /// codes can be reconstructed on demand via [`FSSTArrayExt::codes()`] using the bytes
    /// from `FSSTData` combined with offsets and validity from the array's slots.
    fn deserialize(
        &self,
        dtype: &DType,
        len: usize,
        metadata: &[u8],
        buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
        session: &VortexSession,
    ) -> VortexResult<ArrayParts<Self>> {
        let metadata = FSSTMetadata::decode(metadata)?;
        let symbols = Buffer::<Symbol>::from_byte_buffer(buffers[0].clone().try_to_host_sync()?);
        let symbol_lengths = Buffer::<u8>::from_byte_buffer(buffers[1].clone().try_to_host_sync()?);

        let mut ctx = session.create_execution_ctx();
        if buffers.len() == 2 {
            return Self::deserialize_legacy(
                self,
                dtype,
                len,
                &metadata,
                &symbols,
                &symbol_lengths,
                children,
                &mut ctx,
            );
        }

        if buffers.len() == 3 {
            let uncompressed_lengths = children.get(
                0,
                &DType::Primitive(
                    metadata.get_uncompressed_lengths_ptype()?,
                    Nullability::NonNullable,
                ),
                len,
            )?;

            let codes_bytes = buffers[2].clone();
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
                vortex_bail!("Expected 2 or 3 children, got {}", children.len());
            };

            FSSTData::validate_parts(
                symbols.as_slice(),
                symbol_lengths.as_slice(),
                &codes_bytes,
                &codes_offsets,
                dtype.nullability(),
                &uncompressed_lengths,
                dtype,
                len,
                &mut ctx,
            )?;
            let slots = FSSTSlots {
                uncompressed_lengths,
                codes_offsets,
                codes_validity: validity_to_child(&codes_validity, len),
            }
            .into_slots();
            let data = FSSTData::try_new(symbols, symbol_lengths, codes_bytes, len)?;
            return Ok(ArrayParts::new(self.clone(), dtype.clone(), len, data).with_slots(slots));
        }

        vortex_bail!(
            "InvalidArgument: Expected 2 or 3 buffers, got {}",
            buffers.len()
        );
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        FSSTSlots::NAMES[idx].to_string()
    }

    fn execute(array: Array<Self>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        canonicalize_fsst(array.as_view(), ctx).map(ExecutionResult::done)
    }

    fn append_to_builder(
        array: ArrayView<'_, Self>,
        builder: &mut dyn ArrayBuilder,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        if let Some(result) =
            match_each_varbin_builder!(builder, |builder| append_to_varbin(array, builder, ctx))
        {
            return result;
        }

        let Some(builder) = builder.as_any_mut().downcast_mut::<VarBinViewBuilder>() else {
            return array
                .array()
                .clone()
                .execute::<Canonical>(ctx)?
                .into_array()
                .append_to_builder(builder, ctx);
        };

        // Decompress the whole block of data into a new buffer, and create some views
        // from it instead. The new buffer lands after any pending in-progress
        // buffer that push_buffer_and_adjusted_views will flush first.
        let next_buffer_index = builder.completed_block_count() + u32::from(builder.in_progress());
        let (buffers, views) = fsst_decode_views(array, next_buffer_index, ctx)?;

        builder.push_buffer_and_adjusted_views(
            &buffers,
            &views,
            array
                .array()
                .validity()?
                .execute_mask(array.array().len(), ctx)?,
        );
        Ok(())
    }

    fn reduce_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        RULES.evaluate(array, parent, child_idx)
    }
}

/// Decompresses the values and appends them to `builder`.
fn append_to_varbin<O: OffsetBuilderPType>(
    array: ArrayView<'_, FSST>,
    builder: &mut VarBinBuilder<O>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()>
where
    usize: AsPrimitive<O>,
{
    let (bytes, lengths) = fsst_decode_bytes(array, ctx)?;
    let validity = array
        .array()
        .validity()?
        .execute_mask(array.array().len(), ctx)?;
    match_each_integer_ptype!(lengths.ptype(), |P| {
        builder.append_values(
            bytes.as_slice(),
            lengths.as_slice::<P>().iter().scan(0usize, |end, length| {
                *end += AsPrimitive::<usize>::as_(*length);
                Some(*end)
            }),
            &validity,
        )
    })
}

#[array_slots(FSST)]
pub struct FSSTSlots {
    /// Lengths of the original values before compression, can be compressed.
    #[slot(0)]
    pub uncompressed_lengths: ArrayRef,
    /// The offsets array for the FSST-compressed codes.
    #[slot(1)]
    pub codes_offsets: ArrayRef,
    /// The validity bitmap for the compressed codes.
    #[slot(2)]
    pub codes_validity: Option<ArrayRef>,
}

/// The inner data for an FSST-compressed array.
///
/// Holds the FSST symbol table (`symbols` + `symbol_lengths`) and the raw compressed
/// codes bytes buffer. The codes offsets and validity live in the outer array's slots
/// (slots 1 and 2 respectively).
///
/// A full [`VarBinArray`] representing the codes can be reconstructed on demand via
/// [`FSSTArrayExt::codes()`], combining this buffer with the offsets/validity from slots.
#[derive(Clone)]
pub struct FSSTData {
    symbol_table: Arc<FSSTSymbolTable>,
    /// The raw compressed codes bytes, equivalent to `VarBinData::bytes`.
    codes_bytes: BufferHandle,
    /// Cached length (number of elements).
    len: usize,
}

impl Display for FSSTData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "len: {}, nsymbols: {}",
            self.len, self.symbol_table.n_symbols
        )
    }
}

impl Debug for FSSTData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FSSTArray")
            .field("symbols", &self.symbols())
            .field("symbol_lengths", &self.symbol_lengths())
            .field("codes_bytes_len", &self.codes_bytes.len())
            .field("len", &self.len)
            .field("uncompressed_lengths", &"<outer slot>")
            .field("codes_offsets", &"<outer slot>")
            .field("codes_validity", &"<outer slot>")
            .finish()
    }
}

pub struct FSSTSymbolTable {
    /// Symbols padded out to [`FSST_SYMBOL_TABLE_LEN`] entries, zero-filled past `n_symbols`,
    /// so that a [`Decompressor`] can borrow them without copying.
    padded_symbols: Buffer<Symbol>,
    /// Symbol lengths padded out to [`FSST_SYMBOL_TABLE_LEN`] entries, zero-filled past
    /// `n_symbols`.
    padded_symbol_lengths: Buffer<u8>,
    /// The number of populated symbols. Entries at or past this index are padding.
    n_symbols: usize,
    /// Memoized compressor used for push-down of compute by compressing the RHS.
    compressor: OnceLock<Compressor>,
}

impl FSSTSymbolTable {
    /// Builds a symbol table, padding `symbols` and `symbol_lengths` out to
    /// [`FSST_SYMBOL_TABLE_LEN`] entries if they are shorter.
    ///
    /// `n_symbols` is the number of populated entries; everything past it is padding. Buffers
    /// longer than [`FSST_SYMBOL_TABLE_LEN`] are truncated; callers are expected to have rejected
    /// them already (see [`FSSTData::try_new`]).
    pub fn new(symbols: Buffer<Symbol>, symbol_lengths: Buffer<u8>, n_symbols: usize) -> Self {
        Self {
            padded_symbols: pad_symbol_table(symbols, Symbol::ZERO),
            padded_symbol_lengths: pad_symbol_table(symbol_lengths, 0),
            n_symbols: n_symbols.min(FSST_SYMBOL_TABLE_LEN),
            compressor: OnceLock::new(),
        }
    }

    /// Builds a symbol table, padding `symbols` and `symbol_lengths` out to
    /// [`FSST_SYMBOL_TABLE_LEN`] entries if they are shorter.
    ///
    /// `n_symbols` is the number of populated entries; everything past it is padding. Buffers
    /// longer than [`FSST_SYMBOL_TABLE_LEN`] are truncated; callers are expected to have rejected
    /// them already (see [`FSSTData::try_new`]).
    pub fn new_padded(
        padded_symbols: Buffer<Symbol>,
        padded_symbol_lengths: Buffer<u8>,
        n_symbols: usize,
    ) -> VortexResult<Self> {
        vortex_ensure!(
            padded_symbols.len() == FSST_SYMBOL_TABLE_LEN
                && padded_symbol_lengths.len() == FSST_SYMBOL_TABLE_LEN,
            InvalidArgument: "padded symbol table must have exactly {FSST_SYMBOL_TABLE_LEN} entries, found {} symbols and {} symbol lengths",
            padded_symbols.len(),
            padded_symbol_lengths.len()
        );
        vortex_ensure!(
            n_symbols <= FSST_SYMBOL_TABLE_LEN,
            InvalidArgument: "n_symbols must be <= {FSST_SYMBOL_TABLE_LEN}, found {n_symbols}"
        );
        Ok(Self {
            padded_symbols,
            padded_symbol_lengths,
            n_symbols,
            compressor: OnceLock::new(),
        })
    }

    /// The populated symbols, excluding the padding added by [`Self::new`].
    fn symbols(&self) -> &[Symbol] {
        &self.padded_symbols.as_slice()[..self.n_symbols]
    }

    /// The populated symbol lengths, excluding the padding added by [`Self::new`].
    fn symbol_lengths(&self) -> &[u8] {
        &self.padded_symbol_lengths.as_slice()[..self.n_symbols]
    }

    /// The symbols buffer, padded to exactly [`FSST_SYMBOL_TABLE_LEN`] entries with
    /// [`Symbol::ZERO`].
    fn padded_symbols(&self) -> &Buffer<Symbol> {
        &self.padded_symbols
    }

    /// The symbol lengths buffer, padded to exactly [`FSST_SYMBOL_TABLE_LEN`] entries with zeros.
    fn padded_symbol_lengths(&self) -> &Buffer<u8> {
        &self.padded_symbol_lengths
    }

    /// Borrow the padded symbol table as the fixed-size arrays expected by [`Decompressor`].
    ///
    /// Both buffers are padded to [`FSST_SYMBOL_TABLE_LEN`] on construction, so this is a
    /// length check and a pointer cast rather than a copy.
    fn decompressor(&self) -> Decompressor<'_> {
        const PADDED: &str = "FSST symbol table is padded to FSST_SYMBOL_TABLE_LEN entries";
        let symbols = self
            .padded_symbols
            .as_slice()
            .first_chunk::<FSST_SYMBOL_TABLE_LEN>()
            .vortex_expect(PADDED);
        let symbol_lengths = self
            .padded_symbol_lengths
            .as_slice()
            .first_chunk::<FSST_SYMBOL_TABLE_LEN>()
            .vortex_expect(PADDED);
        Decompressor::new(symbols, symbol_lengths)
    }

    fn compressor(&self) -> &Compressor {
        self.compressor
            .get_or_init(|| Compressor::rebuild_from(self.symbols(), self.symbol_lengths()))
    }
}

/// Returns `buffer` resized to exactly [`FSST_SYMBOL_TABLE_LEN`] entries, filling any tail with
/// `pad`. Buffers that are already the right length are returned untouched.
fn pad_symbol_table<T: Copy>(buffer: Buffer<T>, pad: T) -> Buffer<T> {
    if buffer.len() == FSST_SYMBOL_TABLE_LEN {
        return buffer;
    }
    padded_symbol_table(buffer.as_slice(), pad)
}

/// Copies `values` into a buffer of exactly [`FSST_SYMBOL_TABLE_LEN`] entries, filling the tail
/// with `pad`.
///
/// FSST symbol tables are stored padded so that [`FSSTData::decompressor`] can borrow them as the
/// fixed-size arrays [`Decompressor`] requires, without copying.
pub(crate) fn padded_symbol_table<T: Copy>(values: &[T], pad: T) -> Buffer<T> {
    let populated = values.len().min(FSST_SYMBOL_TABLE_LEN);
    let mut padded = BufferMut::with_capacity(FSST_SYMBOL_TABLE_LEN);
    padded.extend_from_slice(&values[..populated]);
    padded.push_n(pad, FSST_SYMBOL_TABLE_LEN - populated);
    padded.freeze()
}

#[derive(Clone, Debug)]
pub struct FSST;

impl FSST {
    /// Build an FSST array from a set of `symbols` and `codes`.
    ///
    /// The `codes` VarBinArray is decomposed: its bytes are stored in [`FSSTData`], while
    /// its offsets and validity become array slots. The codes VarBinArray can be
    /// reconstructed on demand via [`FSSTArrayExt::codes()`].
    pub fn try_new(
        dtype: DType,
        symbols: Buffer<Symbol>,
        symbol_lengths: Buffer<u8>,
        codes: VarBinArray,
        uncompressed_lengths: ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<FSSTArray> {
        let len = codes.len();
        FSSTData::validate_parts_from_codes(
            symbols.as_slice(),
            symbol_lengths.as_slice(),
            &codes,
            &uncompressed_lengths,
            &dtype,
            len,
            ctx,
        )?;
        let slots = FSSTData::make_slots(&codes, &uncompressed_lengths);
        let codes_bytes = codes.bytes_handle().clone();
        let data = FSSTData::try_new(symbols, symbol_lengths, codes_bytes, len)?;
        Ok(unsafe {
            Array::from_parts_unchecked(ArrayParts::new(FSST, dtype, len, data).with_slots(slots))
        })
    }

    pub fn try_new_with_symbol_table(
        dtype: DType,
        symbol_table: Arc<FSSTSymbolTable>,
        codes: VarBinArray,
        uncompressed_lengths: ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<FSSTArray> {
        let len = codes.len();
        FSSTData::validate_parts_from_codes(
            symbol_table.symbols(),
            symbol_table.symbol_lengths(),
            &codes,
            &uncompressed_lengths,
            &dtype,
            len,
            ctx,
        )?;
        let slots = FSSTData::make_slots(&codes, &uncompressed_lengths);
        let codes_bytes = codes.bytes_handle().clone();
        let data =
            unsafe { FSSTData::new_unchecked_with_symbol_table(symbol_table, codes_bytes, len) };
        Ok(unsafe {
            Array::from_parts_unchecked(ArrayParts::new(FSST, dtype, len, data).with_slots(slots))
        })
    }

    /// Legacy deserialization path (2 buffers): the codes were stored as a full
    /// `VarBinArray` child. We decompose the VarBinArray into its bytes (stored in
    /// FSSTData) and offsets/validity (stored in slots).
    #[allow(clippy::too_many_arguments)]
    fn deserialize_legacy(
        &self,
        dtype: &DType,
        len: usize,
        metadata: &FSSTMetadata,
        symbols: &Buffer<Symbol>,
        symbol_lengths: &Buffer<u8>,
        children: &dyn ArrayChildren,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayParts<Self>> {
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

        FSSTData::validate_parts_from_codes(
            symbols.as_slice(),
            symbol_lengths.as_slice(),
            &codes,
            &uncompressed_lengths,
            dtype,
            len,
            ctx,
        )?;
        let slots = FSSTData::make_slots(&codes, &uncompressed_lengths);
        let codes_bytes = codes.bytes_handle().clone();
        let data = FSSTData::try_new(symbols.clone(), symbol_lengths.clone(), codes_bytes, len)?;
        Ok(ArrayParts::new(self.clone(), dtype.clone(), len, data).with_slots(slots))
    }

    pub(crate) unsafe fn new_unchecked_with_symbol_table(
        dtype: DType,
        symbol_table: Arc<FSSTSymbolTable>,
        codes: VarBinArray,
        uncompressed_lengths: ArrayRef,
    ) -> FSSTArray {
        let len = codes.len();
        let slots = FSSTData::make_slots(&codes, &uncompressed_lengths);
        let codes_bytes = codes.bytes_handle().clone();
        let data =
            unsafe { FSSTData::new_unchecked_with_symbol_table(symbol_table, codes_bytes, len) };
        unsafe {
            Array::from_parts_unchecked(ArrayParts::new(FSST, dtype, len, data).with_slots(slots))
        }
    }
}

impl FSSTData {
    fn make_slots(codes: &VarBinArray, uncompressed_lengths: &ArrayRef) -> ArraySlots {
        FSSTSlots {
            uncompressed_lengths: uncompressed_lengths.clone(),
            codes_offsets: codes.offsets().clone(),
            codes_validity: validity_to_child(
                &codes
                    .validity()
                    .vortex_expect("FSST codes validity should be derivable"),
                codes.len(),
            ),
        }
        .into_slots()
    }

    /// Build FSST data from a set of `symbols`, `symbol_lengths`, and compressed codes bytes.
    ///
    /// Symbols are 8-bytes and can represent short strings, each of which is assigned
    /// a code.
    ///
    /// The `codes_bytes` buffer contains the concatenated compressed bytecodes for all elements.
    /// Each element's compressed bytecodes are a sequence of 8-bit codes, where each code
    /// corresponds either to a symbol or to the "escape code" (which tells the decoder to
    /// emit the following byte without doing a table lookup).
    ///
    /// `symbols` and `symbol_lengths` hold only the populated entries; they are padded out to
    /// [`FSST_SYMBOL_TABLE_LEN`] entries so that [`Self::decompressor`] can borrow them without
    /// copying.
    ///
    /// The offsets and validity for the codes are stored in the array's slots, not here.
    /// Use [`FSSTArrayExt::codes()`] to reconstruct a full `VarBinArray`.
    pub fn try_new(
        symbols: Buffer<Symbol>,
        symbol_lengths: Buffer<u8>,
        codes_bytes: BufferHandle,
        len: usize,
    ) -> VortexResult<Self> {
        vortex_ensure!(
            symbols.len() == symbol_lengths.len(),
            InvalidArgument: "symbols and symbol_lengths arrays must have same length, found {} and {}",
            symbols.len(),
            symbol_lengths.len()
        );
        vortex_ensure!(
            symbols.len() <= FSST_SYMBOL_TABLE_LEN,
            InvalidArgument: "symbols array must have length <= {FSST_SYMBOL_TABLE_LEN}, found {}",
            symbols.len()
        );
        let n_symbols = symbols.len();
        // SAFETY: the symbol table shape is validated above.
        let symbol_table = Arc::new(FSSTSymbolTable::new(symbols, symbol_lengths, n_symbols));
        unsafe {
            Ok(Self::new_unchecked_with_symbol_table(
                symbol_table,
                codes_bytes,
                len,
            ))
        }
    }

    pub fn validate(
        &self,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        let fsst_slots = FSSTSlotsView::from_slots(slots);
        Self::validate_parts(
            self.symbol_table.symbols(),
            self.symbol_table.symbol_lengths(),
            &self.codes_bytes,
            fsst_slots.codes_offsets,
            dtype.nullability(),
            fsst_slots.uncompressed_lengths,
            dtype,
            len,
            ctx,
        )
    }

    /// Validate using the decomposed components (codes bytes + offsets + nullability).
    #[expect(clippy::too_many_arguments)]
    fn validate_parts(
        symbols: &[Symbol],
        symbol_lengths: &[u8],
        codes_bytes: &BufferHandle,
        codes_offsets: &ArrayRef,
        codes_nullability: Nullability,
        uncompressed_lengths: &ArrayRef,
        dtype: &DType,
        len: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        vortex_ensure!(
            matches!(dtype, DType::Binary(_) | DType::Utf8(_)),
            "FSST arrays must be Binary or Utf8, found {dtype}"
        );

        if symbols.len() > FSST_SYMBOL_TABLE_LEN {
            vortex_bail!(InvalidArgument: "symbols array must have length <= {FSST_SYMBOL_TABLE_LEN}");
        }

        if symbols.len() != symbol_lengths.len() {
            vortex_bail!(InvalidArgument: "symbols and symbol_lengths arrays must have same length");
        }

        Self::validate_symbol_lengths(symbol_lengths)?;

        // codes_offsets.len() - 1 == number of elements
        let codes_len = codes_offsets.len().saturating_sub(1);
        if codes_len != len {
            vortex_bail!(InvalidArgument: "codes must have same len as outer array");
        }

        if uncompressed_lengths.len() != len {
            vortex_bail!(InvalidArgument: "uncompressed_lengths must be same len as codes");
        }

        if !uncompressed_lengths.dtype().is_int() || uncompressed_lengths.dtype().is_nullable() {
            vortex_bail!(InvalidArgument: "uncompressed_lengths must have integer type and cannot be nullable, found {}", uncompressed_lengths.dtype());
        }

        // Offsets must be non-nullable integer.
        if !codes_offsets.dtype().is_int() || codes_offsets.dtype().is_nullable() {
            vortex_bail!(InvalidArgument: "codes offsets must be non-nullable integer type, found {}", codes_offsets.dtype());
        }

        if codes_nullability != dtype.nullability() {
            vortex_bail!(InvalidArgument: "codes nullability must match outer dtype nullability");
        }

        // Validate that last offset doesn't exceed bytes length (when host-resident).
        if codes_bytes.is_on_host() && codes_offsets.is_host() && !codes_offsets.is_empty() {
            let last_offset: usize = (&codes_offsets
                .execute_scalar(codes_offsets.len() - 1, ctx)
                .vortex_expect("offsets must support scalar_at"))
                .try_into()
                .vortex_expect("Failed to convert offset to usize");
            vortex_ensure!(
                last_offset <= codes_bytes.len(),
                InvalidArgument: "Last codes offset {} exceeds codes bytes length {}",
                last_offset,
                codes_bytes.len()
            );
        }

        Ok(())
    }

    fn validate_symbol_lengths(symbol_lengths: &[u8]) -> VortexResult<()> {
        let mut expected = 2;
        for (idx, &len) in symbol_lengths.iter().enumerate() {
            if len > 8 || len == 0 {
                vortex_bail!(InvalidArgument: "symbol length at index {idx} must be between 1 and 8, found {len}");
            }

            if expected == 1 {
                if len != 1 {
                    vortex_bail!(InvalidArgument: "symbol length at index {idx} must be 1 after one-byte symbols begin, found {len}");
                }
            } else {
                if len == 1 {
                    expected = 1;
                }

                if len < expected {
                    vortex_bail!(InvalidArgument: "symbol length at index {idx} violates FSST symbol table ordering");
                }
                expected = len;
            }
        }

        Ok(())
    }

    /// Validate using a VarBinArray for the codes (convenience for construction paths).
    fn validate_parts_from_codes(
        symbols: &[Symbol],
        symbol_lengths: &[u8],
        codes: &VarBinArray,
        uncompressed_lengths: &ArrayRef,
        dtype: &DType,
        len: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        Self::validate_parts(
            symbols,
            symbol_lengths,
            codes.bytes_handle(),
            codes.offsets(),
            codes.dtype().nullability(),
            uncompressed_lengths,
            dtype,
            len,
            ctx,
        )
    }

    pub(crate) unsafe fn new_unchecked_with_symbol_table(
        symbol_table: Arc<FSSTSymbolTable>,
        codes_bytes: BufferHandle,
        len: usize,
    ) -> Self {
        Self {
            symbol_table,
            codes_bytes,
            len,
        }
    }

    /// Returns the number of elements in the array.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the array contains no elements.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Access the symbol table array.
    ///
    /// This is the populated prefix of the symbol table; the padding added on construction to
    /// reach [`FSST_SYMBOL_TABLE_LEN`] is not included. Use [`Self::padded_symbols`] to get the
    /// whole buffer.
    pub fn symbols(&self) -> Buffer<Symbol> {
        self.symbol_table
            .padded_symbols()
            .slice(0..self.symbol_table.n_symbols)
    }

    /// Access the symbol lengths array.
    ///
    /// As with [`Self::symbols`], this excludes the padding added on construction.
    pub fn symbol_lengths(&self) -> Buffer<u8> {
        self.symbol_table
            .padded_symbol_lengths()
            .slice(0..self.symbol_table.n_symbols)
    }

    /// The whole symbols buffer, padded to exactly [`FSST_SYMBOL_TABLE_LEN`] entries with
    /// [`Symbol::ZERO`]. Entries at or past [`Self::n_symbols`] are padding.
    pub fn padded_symbols(&self) -> &Buffer<Symbol> {
        self.symbol_table.padded_symbols()
    }

    /// The whole symbol lengths buffer, padded to exactly [`FSST_SYMBOL_TABLE_LEN`] entries with
    /// zeros. Entries at or past [`Self::n_symbols`] are padding.
    pub fn padded_symbol_lengths(&self) -> &Buffer<u8> {
        self.symbol_table.padded_symbol_lengths()
    }

    /// The number of populated entries in the symbol table.
    pub fn n_symbols(&self) -> usize {
        self.symbol_table.n_symbols
    }

    pub(crate) fn symbol_table(&self) -> Arc<FSSTSymbolTable> {
        Arc::clone(&self.symbol_table)
    }

    /// Access the compressed codes bytes buffer handle (may be on host or device).
    pub fn codes_bytes_handle(&self) -> &BufferHandle {
        &self.codes_bytes
    }

    /// Access the compressed codes bytes on the host.
    pub fn codes_bytes(&self) -> &ByteBuffer {
        self.codes_bytes.as_host()
    }

    /// Build a [`Decompressor`] that can be used to decompress values from
    /// this array.
    pub fn decompressor(&self) -> Decompressor<'_> {
        self.symbol_table.decompressor()
    }

    /// Retrieves the FSST compressor.
    pub fn compressor(&self) -> &Compressor {
        self.symbol_table.compressor()
    }
}

pub trait FSSTArrayExt: FSSTArraySlotsExt {
    fn uncompressed_lengths_dtype(&self) -> &DType {
        self.uncompressed_lengths().dtype()
    }

    /// Reconstruct a [`VarBinArray`] for the compressed codes by combining the bytes
    /// from [`FSSTData`] with the offsets and validity stored in the array's slots.
    fn codes(&self) -> VarBinArray {
        let offsets = self.codes_offsets().clone();
        let validity =
            child_to_validity(self.codes_validity(), self.as_ref().dtype().nullability());
        let codes_bytes = self.codes_bytes_handle().clone();
        // SAFETY: components were validated at construction time.
        unsafe {
            VarBinArray::new_unchecked_from_handle(
                offsets,
                codes_bytes,
                DType::Binary(self.as_ref().dtype().nullability()),
                validity,
            )
        }
    }

    /// Get the DType of the codes array.
    fn codes_dtype(&self) -> DType {
        DType::Binary(self.as_ref().dtype().nullability())
    }
}

impl<T: TypedArrayRef<FSST>> FSSTArrayExt for T {}

impl ValidityVTable<FSST> for FSST {
    fn validity(array: ArrayView<'_, FSST>) -> VortexResult<Validity> {
        Ok(child_to_validity(
            array.codes_validity(),
            array.dtype().nullability(),
        ))
    }
}

#[cfg(test)]
mod test {
    use fsst::Compressor;
    use fsst::Symbol;
    use prost::Message;
    use vortex_array::ArrayPlugin;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::array_session;
    use vortex_array::arrays::VarBinViewArray;
    use vortex_array::buffer::BufferHandle;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::test_harness::check_metadata;
    use vortex_array::vtable::VTable as _;
    use vortex_buffer::Buffer;
    use vortex_error::VortexResult;
    use vortex_error::vortex_err;

    use crate::FSST;
    use crate::array::FSST_SYMBOL_TABLE_LEN;
    use crate::array::FSSTArrayExt;
    use crate::array::FSSTArraySlotsExt;
    use crate::array::FSSTData;
    use crate::array::FSSTMetadata;
    use crate::array::FSSTSymbolTable;
    use crate::array::padded_symbol_table;
    use crate::fsst_compress;
    use crate::fsst_train_compressor;

    #[test]
    fn slice_reuses_initialized_compressor() -> VortexResult<()> {
        let symbols = Buffer::<Symbol>::copy_from([
            Symbol::from_slice(b"abc00000"),
            Symbol::from_slice(b"defghijk"),
        ]);
        let symbol_lengths = Buffer::<u8>::copy_from([3, 8]);

        let compressor = Compressor::rebuild_from(symbols.as_slice(), symbol_lengths.as_slice());
        let mut ctx = array_session().create_execution_ctx();
        let strings = VarBinViewArray::from_iter_str(["abcabcab", "defghijk", "abcxyz"]);
        let fsst_array = fsst_compress(&strings.into_array(), &compressor, &mut ctx)?;

        let compressor_ptr = fsst_array.compressor() as *const Compressor;
        let sliced = fsst_array
            .slice(1..3)?
            .try_downcast::<FSST>()
            .map_err(|_| vortex_err!("slice must return an FSST array"))?;
        let sliced_compressor_ptr = sliced.compressor() as *const Compressor;

        assert_eq!(compressor_ptr, sliced_compressor_ptr);
        Ok(())
    }

    /// The symbol table is padded out to [`FSST_SYMBOL_TABLE_LEN`] so that `Decompressor` can
    /// borrow it directly, but the logical accessors and the serialized buffers must still only
    /// expose the populated prefix.
    #[test]
    fn symbol_table_padded_on_creation() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let strings = VarBinViewArray::from_iter_str(["abcabcab", "defghijk", "abcxyz"]);
        let compressor = Compressor::rebuild_from(
            [
                Symbol::from_slice(b"abc00000"),
                Symbol::from_slice(b"defghijk"),
            ],
            [3u8, 8],
        );
        let fsst_array = fsst_compress(&strings.into_array(), &compressor, &mut ctx)?;

        assert_eq!(fsst_array.padded_symbols().len(), FSST_SYMBOL_TABLE_LEN);
        assert_eq!(
            fsst_array.padded_symbol_lengths().len(),
            FSST_SYMBOL_TABLE_LEN
        );
        assert_eq!(fsst_array.padded_symbol_lengths().as_slice()[2..], [0; 253]);

        // Accessors and serialized buffers only see the two populated symbols.
        assert_eq!(fsst_array.n_symbols(), 2);
        assert_eq!(fsst_array.symbols().len(), 2);
        assert_eq!(fsst_array.symbol_lengths().as_slice(), &[3, 8]);
        assert_eq!(
            FSST::buffer(fsst_array.as_view(), 0).len(),
            2 * size_of::<Symbol>()
        );
        assert_eq!(FSST::buffer(fsst_array.as_view(), 1).len(), 2);

        let decompressed = fsst_array
            .into_array()
            .execute::<VarBinViewArray>(&mut ctx)?;
        assert_eq!(decompressed.bytes_at(0).as_slice(), b"abcabcab".as_ref());
        assert_eq!(decompressed.bytes_at(1).as_slice(), b"defghijk".as_ref());
        assert_eq!(decompressed.bytes_at(2).as_slice(), b"abcxyz".as_ref());
        Ok(())
    }

    /// Buffers arriving from a file hold the unpadded symbol table, so deserialization must pad
    /// them before a `Decompressor` can borrow them.
    #[test]
    fn symbol_table_padded_on_deserialize() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let input = VarBinViewArray::from_iter_str(["abcabcab", "defghijk"]).into_array();
        let compressor = fsst_train_compressor(&input, &mut ctx)?;
        let fsst_array = fsst_compress(&input, &compressor, &mut ctx)?;

        let buffers = [
            BufferHandle::new_host(fsst_array.symbols().into_byte_buffer()),
            BufferHandle::new_host(fsst_array.symbol_lengths().into_byte_buffer()),
            fsst_array.codes_bytes_handle().clone(),
        ];
        assert!(buffers[1].len() < FSST_SYMBOL_TABLE_LEN);

        let children = vec![
            fsst_array.uncompressed_lengths().clone(),
            fsst_array.codes_offsets().clone(),
        ];

        let deserialized = ArrayPlugin::deserialize(
            &FSST,
            &DType::Utf8(Nullability::NonNullable),
            2,
            &FSSTMetadata {
                uncompressed_lengths_ptype: fsst_array
                    .uncompressed_lengths()
                    .dtype()
                    .as_ptype()
                    .into(),
                codes_offsets_ptype: fsst_array.codes_offsets().dtype().as_ptype().into(),
            }
            .encode_to_vec(),
            &buffers,
            &children.as_slice(),
            &array_session(),
        )?;

        let padded = deserialized
            .clone()
            .try_downcast::<FSST>()
            .map_err(|_| vortex_err!("deserialize must return an FSST array"))?;
        assert_eq!(padded.padded_symbols().len(), FSST_SYMBOL_TABLE_LEN);
        assert_eq!(padded.n_symbols(), fsst_array.symbols().len());

        let decompressed = deserialized.execute::<VarBinViewArray>(&mut ctx)?;
        assert_eq!(decompressed.bytes_at(0).as_slice(), b"abcabcab".as_ref());
        assert_eq!(decompressed.bytes_at(1).as_slice(), b"defghijk".as_ref());
        Ok(())
    }

    /// An already-padded table must be stored as-is rather than copied again.
    #[test]
    fn padded_constructor_does_not_repad() -> VortexResult<()> {
        let symbols = padded_symbol_table(&[Symbol::from_slice(b"ab000000")], Symbol::ZERO);
        let symbol_lengths = padded_symbol_table(&[2u8], 0);
        let symbols_ptr = symbols.as_slice().as_ptr();
        let symbol_lengths_ptr = symbol_lengths.as_slice().as_ptr();

        let data = FSSTSymbolTable::new_padded(symbols, symbol_lengths, 1)?;

        assert_eq!(data.padded_symbols().as_slice().as_ptr(), symbols_ptr);
        assert_eq!(
            data.padded_symbol_lengths().as_slice().as_ptr(),
            symbol_lengths_ptr
        );
        assert_eq!(data.symbols().len(), 1);
        Ok(())
    }

    /// [`FSSTData::try_new_padded`] stores the buffers as given, so it must reject tables that are
    /// not already padded.
    #[test]
    fn rejects_unpadded_input_to_padded_constructor() {
        assert!(
            FSSTSymbolTable::new_padded(
                Buffer::<Symbol>::copy_from([Symbol::from_slice(b"ab000000")]),
                Buffer::<u8>::copy_from([2]),
                1,
            )
            .is_err()
        );
        assert!(
            FSSTSymbolTable::new_padded(
                Buffer::<Symbol>::full(Symbol::ZERO, FSST_SYMBOL_TABLE_LEN),
                Buffer::<u8>::full(0, FSST_SYMBOL_TABLE_LEN),
                FSST_SYMBOL_TABLE_LEN + 1,
            )
            .is_err()
        );
    }

    #[test]
    fn rejects_malformed_symbol_table() {
        let codes_bytes = BufferHandle::new_host(Buffer::<u8>::empty());
        assert!(
            FSSTData::try_new(
                Buffer::<Symbol>::copy_from([Symbol::from_slice(b"ab000000")]),
                Buffer::<u8>::copy_from([2, 2]),
                codes_bytes.clone(),
                0,
            )
            .is_err()
        );
        assert!(
            FSSTData::try_new(
                Buffer::<Symbol>::full(Symbol::from_slice(b"ab000000"), FSST_SYMBOL_TABLE_LEN + 1,),
                Buffer::<u8>::full(2, FSST_SYMBOL_TABLE_LEN + 1),
                codes_bytes,
                0,
            )
            .is_err()
        );
    }

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
    fn test_back_compat() -> VortexResult<()> {
        let symbols = Buffer::<Symbol>::copy_from([
            Symbol::from_slice(b"abc00000"),
            Symbol::from_slice(b"defghijk"),
        ]);
        let symbol_lengths = Buffer::<u8>::copy_from([3, 8]);

        let compressor = Compressor::rebuild_from(symbols.as_slice(), symbol_lengths.as_slice());
        let mut ctx = array_session().create_execution_ctx();
        let input = VarBinViewArray::from_iter_str(["abcabcab", "defghijk"]);
        let fsst_array = fsst_compress(&input.into_array(), &compressor, &mut ctx)?;

        let compressed_codes = fsst_array.codes();

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
            &array_session(),
        )?;

        let decompressed =
            fsst.execute::<VarBinViewArray>(&mut array_session().create_execution_ctx())?;
        let mask = decompressed
            .validity()?
            .execute_mask(decompressed.len(), &mut ctx)?;
        assert!(mask.value(0));
        assert_eq!(decompressed.bytes_at(0).as_slice(), b"abcabcab".as_ref());
        assert!(mask.value(1));
        assert_eq!(decompressed.bytes_at(1).as_slice(), b"defghijk".as_ref());
        Ok(())
    }
}
