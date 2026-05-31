// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use fsst::Symbol;
use prost::Message as _;
use vortex_array::Array;
use vortex_array::ArrayId;
use vortex_array::ArrayParts;
use vortex_array::ArrayRef;
use vortex_array::ArraySlots;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::ExecutionResult;
use vortex_array::IntoArray;
use vortex_array::TypedArrayRef;
use vortex_array::array_slots;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::varbin::VarBinArrayExt;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::serde::ArrayChildren;
use vortex_array::smallvec::smallvec;
use vortex_array::validity::Validity;
use vortex_array::vtable::VTable;
use vortex_array::vtable::ValidityVTable;
use vortex_array::vtable::child_to_validity;
use vortex_array::vtable::validity_to_child;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::FSSTArray;
use crate::FSSTArrayExt;
// `FSSTView` reuses the exact same inner data representation as `FSST`: the symbol table plus
// the raw compressed byte heap. Only the *addressing* of that heap differs (offsets + sizes
// instead of monotonic offsets), and that addressing lives entirely in the array's slots.
use crate::array::FSSTData;
use crate::fsstview::canonical::canonicalize_fsstview;
use crate::fsstview::kernel::PARENT_KERNELS;
use crate::fsstview::rules::RULES;

/// An [`FSSTView`]-encoded Vortex array.
pub type FSSTViewArray = Array<FSSTView>;

/// The [`FSSTView`] encoding: a ListView-style FSST array.
#[derive(Clone, Debug)]
pub struct FSSTView;

/// The child slots of an [`FSSTView`] array.
///
/// Declared with the [`array_slots`] proc macro, which generates the slot-index constants
/// (`FSSTViewSlots::CODES_OFFSETS`, ...), the borrowed [`FSSTViewSlotsView`] struct, and the
/// typed accessor trait [`FSSTViewArraySlotsExt`] (`.uncompressed_lengths()`,
/// `.codes_offsets()`, `.codes_ends()`, `.codes_validity()`).
#[array_slots(FSSTView)]
pub struct FSSTViewSlots {
    /// Length of each original (uncompressed) value. Non-nullable integer.
    pub uncompressed_lengths: ArrayRef,
    /// Start offset of each element's compressed bytecodes within the code heap. Non-nullable
    /// integer. Unlike `FSST`, these are **not** required to be monotonic or contiguous.
    pub codes_offsets: ArrayRef,
    /// End offset of each element's compressed bytecodes within the code heap, i.e.
    /// `offset + size`. Non-nullable integer. Element `i`'s bytecodes are
    /// `codes_bytes[codes_offsets[i] .. codes_ends[i]]`.
    ///
    /// Storing the end offset (rather than the size) keeps the [`FSSTArray`] → [`FSSTView`]
    /// conversion allocation-free: for a freshly converted array the heap is contiguous, so
    /// `codes_ends` is a zero-copy slice of the monotonic offsets (`offsets[1..len + 1]`), exactly
    /// as `codes_offsets` is `offsets[0..len]`. The per-element size is derived as
    /// `codes_ends[i] - codes_offsets[i]` only where it is needed (canonicalize / `scalar_at`),
    /// never materialized for rows a selective `filter`/`take` discards.
    pub codes_ends: ArrayRef,
    /// Optional validity bitmap for the codes. Absent when the array is non-nullable.
    pub codes_validity: Option<ArrayRef>,
}

#[derive(Clone, prost::Message)]
pub struct FSSTViewMetadata {
    #[prost(enumeration = "PType", tag = "1")]
    uncompressed_lengths_ptype: i32,
    #[prost(enumeration = "PType", tag = "2")]
    codes_offsets_ptype: i32,
    #[prost(enumeration = "PType", tag = "3")]
    codes_ends_ptype: i32,
}

impl FSSTViewMetadata {
    fn get_uncompressed_lengths_ptype(&self) -> VortexResult<PType> {
        PType::try_from(self.uncompressed_lengths_ptype)
            .map_err(|_| vortex_err!("Invalid PType {}", self.uncompressed_lengths_ptype))
    }

    fn get_codes_offsets_ptype(&self) -> VortexResult<PType> {
        PType::try_from(self.codes_offsets_ptype)
            .map_err(|_| vortex_err!("Invalid PType {}", self.codes_offsets_ptype))
    }

    fn get_codes_ends_ptype(&self) -> VortexResult<PType> {
        PType::try_from(self.codes_ends_ptype)
            .map_err(|_| vortex_err!("Invalid PType {}", self.codes_ends_ptype))
    }
}

impl FSSTView {
    /// Build an [`FSSTViewArray`] from its decomposed components.
    ///
    /// `codes_offsets[i]` and `codes_ends[i]` address element `i`'s compressed bytecodes inside
    /// `codes_bytes` as the range `codes_offsets[i]..codes_ends[i]`. The offsets do not need to be
    /// sorted, contiguous, or non-overlapping.
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(
        dtype: DType,
        symbols: Buffer<Symbol>,
        symbol_lengths: Buffer<u8>,
        codes_bytes: BufferHandle,
        codes_offsets: ArrayRef,
        codes_ends: ArrayRef,
        uncompressed_lengths: ArrayRef,
        validity: Validity,
    ) -> VortexResult<FSSTViewArray> {
        let len = codes_offsets.len();
        validate_fsstview(
            &symbols,
            &symbol_lengths,
            &codes_offsets,
            &codes_ends,
            &uncompressed_lengths,
            &validity,
            &dtype,
            len,
        )?;
        let data = FSSTData::try_new(symbols, symbol_lengths, codes_bytes, len)?;
        let slots = make_slots(
            uncompressed_lengths,
            codes_offsets,
            codes_ends,
            &validity,
            len,
        );
        Ok(unsafe {
            Array::from_parts_unchecked(
                ArrayParts::new(FSSTView, dtype, len, data).with_slots(slots),
            )
        })
    }

    /// Build an [`FSSTViewArray`] without validation.
    ///
    /// # Safety
    ///
    /// The caller must uphold the same invariants validated by [`FSSTView::try_new`].
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn new_unchecked(
        dtype: DType,
        symbols: Buffer<Symbol>,
        symbol_lengths: Buffer<u8>,
        codes_bytes: BufferHandle,
        codes_offsets: ArrayRef,
        codes_ends: ArrayRef,
        uncompressed_lengths: ArrayRef,
        validity: Validity,
    ) -> FSSTViewArray {
        let len = codes_offsets.len();
        let data = unsafe { FSSTData::new_unchecked(symbols, symbol_lengths, codes_bytes, len) };
        let slots = make_slots(
            uncompressed_lengths,
            codes_offsets,
            codes_ends,
            &validity,
            len,
        );
        unsafe {
            Array::from_parts_unchecked(
                ArrayParts::new(FSSTView, dtype, len, data).with_slots(slots),
            )
        }
    }
}

/// Convert a plain [`FSSTArray`] into an [`FSSTViewArray`], sharing the symbol table and the
/// compressed byte heap (zero-copy) and addressing the codes with the FSST's existing monotonic
/// offsets.
///
/// A freshly converted view's heap is contiguous, so element `i` occupies `offsets[i]..offsets[i +
/// 1]`. Both addressing arrays are therefore **zero-copy slices of the same `offsets` buffer**:
/// `codes_offsets = offsets[0..len]` and `codes_ends = offsets[1..len + 1]`. Nothing is allocated
/// or copied — in particular the per-element size (`codes_ends[i] - codes_offsets[i]`) is never
/// materialized here, so a subsequent selective `filter`/`take` does not pay to derive sizes for
/// the rows it discards. This removes the conversion floor a very selective predicate used to hit.
pub fn fsstview_from_fsst(fsst: &FSSTArray, ctx: &mut ExecutionCtx) -> VortexResult<FSSTViewArray> {
    let codes = fsst.codes();
    let validity = codes.validity()?;
    let offsets = codes.offsets().clone().execute::<PrimitiveArray>(ctx)?;
    let len = offsets.len().saturating_sub(1);

    // Both addressing arrays are zero-copy slices of the `len + 1` monotonic offsets: element `i`'s
    // codes are `offsets[i]..offsets[i + 1]`, so `codes_ends` is simply the offsets shifted by one.
    let offsets = offsets.into_array();
    let codes_offsets = offsets.slice(0..len)?;
    let codes_ends = offsets.slice(1..len + 1)?;

    FSSTView::try_new(
        fsst.dtype().clone(),
        fsst.symbols().clone(),
        fsst.symbol_lengths().clone(),
        fsst.codes_bytes_handle().clone(),
        codes_offsets,
        codes_ends,
        fsst.uncompressed_lengths().clone(),
        validity,
    )
}

fn make_slots(
    uncompressed_lengths: ArrayRef,
    codes_offsets: ArrayRef,
    codes_ends: ArrayRef,
    validity: &Validity,
    len: usize,
) -> ArraySlots {
    smallvec![
        Some(uncompressed_lengths),
        Some(codes_offsets),
        Some(codes_ends),
        validity_to_child(validity, len),
    ]
}

#[allow(clippy::too_many_arguments)]
fn validate_fsstview(
    symbols: &Buffer<Symbol>,
    symbol_lengths: &Buffer<u8>,
    codes_offsets: &ArrayRef,
    codes_ends: &ArrayRef,
    uncompressed_lengths: &ArrayRef,
    validity: &Validity,
    dtype: &DType,
    len: usize,
) -> VortexResult<()> {
    vortex_ensure!(
        matches!(dtype, DType::Binary(_) | DType::Utf8(_)),
        "FSSTView arrays must be Binary or Utf8, found {dtype}"
    );
    if symbols.len() > 255 {
        vortex_bail!(InvalidArgument: "symbols array must have length <= 255");
    }
    if symbols.len() != symbol_lengths.len() {
        vortex_bail!(InvalidArgument: "symbols and symbol_lengths arrays must have same length");
    }
    if codes_offsets.len() != len {
        vortex_bail!(InvalidArgument: "codes_offsets must have same len as outer array");
    }
    if codes_ends.len() != len {
        vortex_bail!(InvalidArgument: "codes_ends must have same len as outer array");
    }
    if uncompressed_lengths.len() != len {
        vortex_bail!(InvalidArgument: "uncompressed_lengths must have same len as outer array");
    }
    if !codes_offsets.dtype().is_int() || codes_offsets.dtype().is_nullable() {
        vortex_bail!(InvalidArgument: "codes_offsets must be non-nullable integer, found {}", codes_offsets.dtype());
    }
    if !codes_ends.dtype().is_int() || codes_ends.dtype().is_nullable() {
        vortex_bail!(InvalidArgument: "codes_ends must be non-nullable integer, found {}", codes_ends.dtype());
    }
    if !uncompressed_lengths.dtype().is_int() || uncompressed_lengths.dtype().is_nullable() {
        vortex_bail!(InvalidArgument: "uncompressed_lengths must be non-nullable integer, found {}", uncompressed_lengths.dtype());
    }
    if validity.nullability() != dtype.nullability() {
        vortex_bail!(InvalidArgument: "validity nullability must match outer dtype nullability");
    }
    Ok(())
}

/// Typed accessors for [`FSSTViewArray`] that aren't covered by the [`array_slots`] macro.
pub trait FSSTViewArrayExt: TypedArrayRef<FSSTView> {
    /// The validity of the array, derived from the `codes_validity` slot.
    fn fsstview_validity(&self) -> Validity {
        child_to_validity(
            self.as_ref().slots()[FSSTViewSlots::CODES_VALIDITY].as_ref(),
            self.as_ref().dtype().nullability(),
        )
    }
}

impl<T: TypedArrayRef<FSSTView>> FSSTViewArrayExt for T {}

impl VTable for FSSTView {
    type TypedArrayData = FSSTData;
    type OperationsVTable = Self;
    type ValidityVTable = Self;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.fsstview");
        *ID
    }

    fn validate(
        &self,
        data: &Self::TypedArrayData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        let view = FSSTViewSlotsView::from_slots(slots);
        let validity = child_to_validity(view.codes_validity, dtype.nullability());
        validate_fsstview(
            data.symbols(),
            data.symbol_lengths(),
            view.codes_offsets,
            view.codes_ends,
            view.uncompressed_lengths,
            &validity,
            dtype,
            len,
        )
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        3
    }

    fn buffer(array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        match idx {
            0 => BufferHandle::new_host(array.symbols().clone().into_byte_buffer()),
            1 => BufferHandle::new_host(array.symbol_lengths().clone().into_byte_buffer()),
            2 => array.codes_bytes_handle().clone(),
            _ => vortex_panic!("FSSTViewArray buffer index {idx} out of bounds"),
        }
    }

    fn buffer_name(_array: ArrayView<'_, Self>, idx: usize) -> Option<String> {
        match idx {
            0 => Some("symbols".to_string()),
            1 => Some("symbol_lengths".to_string()),
            2 => Some("compressed_codes".to_string()),
            _ => vortex_panic!("FSSTViewArray buffer_name index {idx} out of bounds"),
        }
    }

    fn serialize(
        array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(
            FSSTViewMetadata {
                uncompressed_lengths_ptype: PType::try_from(array.uncompressed_lengths().dtype())?
                    as i32,
                codes_offsets_ptype: PType::try_from(array.codes_offsets().dtype())? as i32,
                codes_ends_ptype: PType::try_from(array.codes_ends().dtype())? as i32,
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
        let metadata = FSSTViewMetadata::decode(metadata)?;
        if buffers.len() != 3 {
            vortex_bail!(
                InvalidArgument: "Expected 3 buffers for fsstview, got {}",
                buffers.len()
            );
        }
        let symbols = Buffer::<Symbol>::from_byte_buffer(buffers[0].clone().try_to_host_sync()?);
        let symbol_lengths = Buffer::<u8>::from_byte_buffer(buffers[1].clone().try_to_host_sync()?);
        let codes_bytes = buffers[2].clone();

        let uncompressed_lengths = children.get(
            0,
            &DType::Primitive(
                metadata.get_uncompressed_lengths_ptype()?,
                Nullability::NonNullable,
            ),
            len,
        )?;
        let codes_offsets = children.get(
            1,
            &DType::Primitive(
                metadata.get_codes_offsets_ptype()?,
                Nullability::NonNullable,
            ),
            len,
        )?;
        let codes_ends = children.get(
            2,
            &DType::Primitive(metadata.get_codes_ends_ptype()?, Nullability::NonNullable),
            len,
        )?;

        let validity = if children.len() == 3 {
            Validity::from(dtype.nullability())
        } else if children.len() == 4 {
            Validity::Array(children.get(3, &Validity::DTYPE, len)?)
        } else {
            vortex_bail!("Expected 3 or 4 children, got {}", children.len());
        };

        validate_fsstview(
            &symbols,
            &symbol_lengths,
            &codes_offsets,
            &codes_ends,
            &uncompressed_lengths,
            &validity,
            dtype,
            len,
        )?;

        let data = FSSTData::try_new(symbols, symbol_lengths, codes_bytes, len)?;
        let slots = make_slots(
            uncompressed_lengths,
            codes_offsets,
            codes_ends,
            &validity,
            len,
        );
        Ok(ArrayParts::new(self.clone(), dtype.clone(), len, data).with_slots(slots))
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        FSSTViewSlots::NAMES[idx].to_string()
    }

    fn execute(array: Array<Self>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        canonicalize_fsstview(array.as_view(), ctx).map(ExecutionResult::done)
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

impl ValidityVTable<FSSTView> for FSSTView {
    fn validity(array: ArrayView<'_, FSSTView>) -> VortexResult<Validity> {
        Ok(child_to_validity(
            array.slots()[FSSTViewSlots::CODES_VALIDITY].as_ref(),
            array.dtype().nullability(),
        ))
    }
}
