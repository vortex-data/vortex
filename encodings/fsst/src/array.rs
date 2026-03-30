// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::fmt::Formatter;
use std::hash::Hash;
use std::sync::Arc;
use std::sync::LazyLock;

use fsst::Compressor;
use fsst::Decompressor;
use fsst::Symbol;
use vortex_array::ArrayEq;
use vortex_array::ArrayHash;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::DeserializeMetadata;
use vortex_array::DynArray;
use vortex_array::ExecutionCtx;
use vortex_array::ExecutionResult;
use vortex_array::IntoArray;
use vortex_array::Precision;
use vortex_array::ProstMetadata;
use vortex_array::SerializeMetadata;
use vortex_array::arrays::VarBin;
use vortex_array::arrays::VarBinArray;
use vortex_array::buffer::BufferHandle;
use vortex_array::builders::ArrayBuilder;
use vortex_array::builders::VarBinViewBuilder;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::serde::ArrayChildren;
use vortex_array::stats::ArrayStats;
use vortex_array::stats::StatsSetRef;
use vortex_array::validity::Validity;
use vortex_array::vtable;
use vortex_array::vtable::Array;
use vortex_array::vtable::ArrayId;
use vortex_array::vtable::VTable;
use vortex_array::vtable::ValidityChild;
use vortex_array::vtable::ValidityHelper;
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

vtable!(FSST);

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

impl VTable for FSST {
    type Array = FSSTArray;

    type Metadata = ProstMetadata<FSSTMetadata>;
    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromChild;

    fn vtable(_array: &Self::Array) -> &Self {
        &FSST
    }

    fn id(&self) -> ArrayId {
        Self::ID
    }

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
        array.uncompressed_lengths().array_hash(state, precision);
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
                .uncompressed_lengths()
                .array_eq(other.uncompressed_lengths(), precision)
    }

    fn nbuffers(_array: &FSSTArray) -> usize {
        3
    }

    fn buffer(array: &FSSTArray, idx: usize) -> BufferHandle {
        match idx {
            0 => BufferHandle::new_host(array.symbols().clone().into_byte_buffer()),
            1 => BufferHandle::new_host(array.symbol_lengths().clone().into_byte_buffer()),
            2 => array.codes.bytes_handle().clone(),
            _ => vortex_panic!("FSSTArray buffer index {idx} out of bounds"),
        }
    }

    fn buffer_name(_array: &FSSTArray, idx: usize) -> Option<String> {
        match idx {
            0 => Some("symbols".to_string()),
            1 => Some("symbol_lengths".to_string()),
            2 => Some("compressed_codes".to_string()),
            _ => vortex_panic!("FSSTArray buffer_name index {idx} out of bounds"),
        }
    }

    fn metadata(array: &FSSTArray) -> VortexResult<Self::Metadata> {
        Ok(ProstMetadata(FSSTMetadata {
            uncompressed_lengths_ptype: array.uncompressed_lengths().dtype().as_ptype().into(),
            codes_offsets_ptype: array.codes.offsets().dtype().as_ptype().into(),
        }))
    }

    fn serialize(metadata: Self::Metadata) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(metadata.serialize()))
    }

    fn deserialize(
        bytes: &[u8],
        _dtype: &DType,
        _len: usize,
        _buffers: &[BufferHandle],
        _session: &VortexSession,
    ) -> VortexResult<Self::Metadata> {
        Ok(ProstMetadata(
            <ProstMetadata<FSSTMetadata> as DeserializeMetadata>::deserialize(bytes)?,
        ))
    }

    fn append_to_builder(
        array: &FSSTArray,
        builder: &mut dyn ArrayBuilder,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        let Some(builder) = builder.as_any_mut().downcast_mut::<VarBinViewBuilder>() else {
            builder.extend_from_array(
                &array
                    .clone()
                    .into_array()
                    .execute::<Canonical>(ctx)?
                    .into_array(),
            );
            return Ok(());
        };

        // Decompress the whole block of data into a new buffer, and create some views
        // from it instead.
        let (buffers, views) = fsst_decode_views(array, builder.completed_block_count(), ctx)?;

        builder.push_buffer_and_adjusted_views(&buffers, &views, array.validity_mask()?);
        Ok(())
    }

    fn build(
        dtype: &DType,
        len: usize,
        metadata: &Self::Metadata,
        buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
    ) -> VortexResult<FSSTArray> {
        let symbols = Buffer::<Symbol>::from_byte_buffer(buffers[0].clone().try_to_host_sync()?);
        let symbol_lengths = Buffer::<u8>::from_byte_buffer(buffers[1].clone().try_to_host_sync()?);

        // Check for the legacy deserialization path.
        if buffers.len() == 2 {
            if children.len() != 2 {
                vortex_bail!(InvalidArgument: "Expected 2 children, got {}", children.len());
            }
            let codes = children.get(0, &DType::Binary(dtype.nullability()), len)?;
            let codes = codes
                .as_opt::<VarBin>()
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

            return FSSTArray::try_new(
                dtype.clone(),
                symbols,
                symbol_lengths,
                codes,
                uncompressed_lengths,
            );
        }

        // Check for the current deserialization path.
        if buffers.len() == 3 {
            let uncompressed_lengths = children.get(
                0,
                &DType::Primitive(
                    metadata.0.get_uncompressed_lengths_ptype()?,
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

            return FSSTArray::try_new(
                dtype.clone(),
                symbols,
                symbol_lengths,
                codes,
                uncompressed_lengths,
            );
        }

        vortex_bail!(
            "InvalidArgument: Expected 2 or 3 buffers, got {}",
            buffers.len()
        );
    }

    fn slots(array: &FSSTArray) -> &[Option<ArrayRef>] {
        &array.slots
    }

    fn slot_name(_array: &FSSTArray, idx: usize) -> String {
        SLOT_NAMES[idx].to_string()
    }

    fn with_slots(array: &mut FSSTArray, slots: Vec<Option<ArrayRef>>) -> VortexResult<()> {
        vortex_ensure!(
            slots.len() == NUM_SLOTS,
            "FSSTArray expects {} slots, got {}",
            NUM_SLOTS,
            slots.len()
        );

        // Reconstruct codes VarBinArray from new offsets + existing bytes + new validity
        let codes_offsets = slots[CODES_OFFSETS_SLOT]
            .clone()
            .vortex_expect("FSSTArray requires codes_offsets slot");
        let codes_validity = match &slots[CODES_VALIDITY_SLOT] {
            Some(arr) => Validity::Array(arr.clone()),
            None => Validity::from(array.codes.dtype().nullability()),
        };
        array.codes = VarBinArray::try_new_from_handle(
            codes_offsets,
            array.codes.bytes_handle().clone(),
            array.codes.dtype().clone(),
            codes_validity,
        )?;
        array.codes_array = array.codes.clone().into_array();
        array.slots = slots;
        Ok(())
    }

    fn execute(array: Arc<Array<Self>>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        canonicalize_fsst(&array, ctx).map(ExecutionResult::done)
    }

    fn execute_parent(
        array: &Array<Self>,
        parent: &ArrayRef,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_KERNELS.execute(array, parent, child_idx, ctx)
    }

    fn reduce_parent(
        array: &Array<Self>,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        RULES.evaluate(array, parent, child_idx)
    }
}

pub(crate) const UNCOMPRESSED_LENGTHS_SLOT: usize = 0;
pub(crate) const CODES_OFFSETS_SLOT: usize = 1;
pub(crate) const CODES_VALIDITY_SLOT: usize = 2;
pub(crate) const NUM_SLOTS: usize = 3;
pub(crate) const SLOT_NAMES: [&str; NUM_SLOTS] =
    ["uncompressed_lengths", "codes_offsets", "codes_validity"];

#[derive(Clone)]
pub struct FSSTArray {
    dtype: DType,
    symbols: Buffer<Symbol>,
    symbol_lengths: Buffer<u8>,
    codes: VarBinArray,
    /// NOTE(ngates): this === codes, but is stored as an ArrayRef so we can return &ArrayRef!
    codes_array: ArrayRef,
    /// Lengths of the original values before compression, can be compressed.
    slots: Vec<Option<ArrayRef>>,
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
            .field("uncompressed_lengths", self.uncompressed_lengths())
            .finish()
    }
}

#[derive(Clone, Debug)]
pub struct FSST;

impl FSST {
    pub const ID: ArrayId = ArrayId::new_ref("vortex.fsst");
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
        let codes_array = codes.clone().into_array();
        let codes_offsets_slot = Some(codes.offsets().clone());
        let codes_validity_slot = validity_to_child(codes.validity(), codes.len());

        Self {
            dtype,
            symbols,
            symbol_lengths,
            codes,
            codes_array,
            slots: vec![
                Some(uncompressed_lengths),
                codes_offsets_slot,
                codes_validity_slot,
            ],
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
        self.slots[UNCOMPRESSED_LENGTHS_SLOT]
            .as_ref()
            .vortex_expect("FSSTArray uncompressed_lengths slot")
    }

    /// Get the DType of the uncompressed lengths array
    #[inline]
    pub fn uncompressed_lengths_dtype(&self) -> &DType {
        self.uncompressed_lengths().dtype()
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

impl ValidityChild<FSST> for FSST {
    fn validity_child(array: &FSSTArray) -> &ArrayRef {
        &array.codes_array
    }
}

#[cfg(test)]
mod test {
    use fsst::Compressor;
    use fsst::Symbol;
    use vortex_array::DynArray;
    use vortex_array::IntoArray;
    use vortex_array::LEGACY_SESSION;
    use vortex_array::ProstMetadata;
    use vortex_array::VortexSessionExecute;
    use vortex_array::accessor::ArrayAccessor;
    use vortex_array::arrays::VarBinViewArray;
    use vortex_array::buffer::BufferHandle;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::test_harness::check_metadata;
    use vortex_array::vtable::VTable;
    use vortex_buffer::Buffer;
    use vortex_error::VortexError;

    use crate::FSST;
    use crate::array::FSSTMetadata;
    use crate::fsst_compress_iter;

    #[cfg_attr(miri, ignore)]
    #[test]
    fn test_fsst_metadata() {
        check_metadata(
            "fsst.metadata",
            ProstMetadata(FSSTMetadata {
                uncompressed_lengths_ptype: PType::U64 as i32,
                codes_offsets_ptype: PType::I32 as i32,
            }),
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

        let fsst = FSST::build(
            &DType::Utf8(Nullability::NonNullable),
            2,
            &ProstMetadata(FSSTMetadata {
                uncompressed_lengths_ptype: fsst_array
                    .uncompressed_lengths()
                    .dtype()
                    .as_ptype()
                    .into(),
                // Legacy array did not store this field, use Protobuf default of 0.
                codes_offsets_ptype: 0,
            }),
            &buffers,
            &children.as_slice(),
        )
        .unwrap();

        let decompressed = fsst
            .into_array()
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
