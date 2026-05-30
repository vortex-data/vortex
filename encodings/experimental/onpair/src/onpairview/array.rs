// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use prost::Message as _;
use vortex_array::Array;
use vortex_array::ArrayParts;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::ExecutionResult;
use vortex_array::IntoArray;
use vortex_array::array_slots;
use vortex_array::buffer::BufferHandle;
use vortex_array::builders::ArrayBuilder;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::serde::ArrayChildren;
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

use crate::OnPairArray;
use crate::OnPairArrayExt;
use crate::OnPairArraySlotsExt;
use crate::OnPairData;
use crate::decode::collect_widened;
use crate::onpairview::canonical::canonicalize_onpairview;
use crate::onpairview::kernel::PARENT_KERNELS;
use crate::onpairview::rules::RULES;

/// An [`OnPairView`]-encoded Vortex array.
///
/// `OnPairView` is the [`ListView`](vortex_array::arrays::ListViewArray)-shaped
/// sibling of [`OnPair`](crate::OnPair). Where [`OnPair`](crate::OnPair) maps
/// each row to a *contiguous, monotonically increasing* run of the flat `codes`
/// token stream via a single `codes_offsets` child (exactly like
/// [`ListArray`](vortex_array::arrays::ListArray) does for elements),
/// `OnPairView` stores a pair of per-row children — `codes_offsets` **and**
/// `codes_sizes` — so a row may point at *any* `[offset, offset + size)` window
/// of the shared `codes` buffer, including out-of-order or overlapping windows.
///
/// Everything else is identical to [`OnPair`](crate::OnPair): the same
/// dictionary blob, `dict_offsets`, `codes`, `uncompressed_lengths`, validity
/// child and `bits`, and the same [`onpair::decompress_into`] decode loop.
///
/// The payoff mirrors [`ListView`](vortex_array::arrays::ListViewArray):
/// `filter`, `take` and `slice` become *metadata-only* — they rewrite only the
/// tiny per-row `codes_offsets`/`codes_sizes` children and **share the
/// (large) `codes` buffer and dictionary unchanged**, instead of rebuilding the
/// `codes` token stream like the [`List`](vortex_array::arrays::ListArray)-shaped
/// [`OnPair`](crate::OnPair) filter must. The cost is deferred to decode time,
/// where the per-row windows are gathered into a contiguous stream.
pub type OnPairViewArray = Array<OnPairView>;

/// Wire-format metadata persisted alongside the OnPairView buffer + slot children.
///
/// Identical to [`OnPairMetadata`](crate::OnPairMetadata) except for the extra
/// `codes_sizes_ptype` recording the width of the per-row `codes_sizes` child.
#[derive(Clone, prost::Message)]
pub struct OnPairViewMetadata {
    /// Width of the per-row primitive `uncompressed_lengths` child.
    #[prost(enumeration = "PType", tag = "1")]
    pub uncompressed_lengths_ptype: i32,
    /// Bits-per-token the column was compressed with (9..=16).
    #[prost(uint32, tag = "2")]
    pub bits: u32,
    /// Number of dictionary tokens. `dict_offsets` has length `dict_size + 1`.
    #[prost(uint32, tag = "3")]
    pub dict_size: u32,
    /// Total number of tokens in the shared `codes` child.
    #[prost(uint64, tag = "4")]
    pub total_tokens: u64,
    /// PType of the `dict_offsets` slot child.
    #[prost(enumeration = "PType", tag = "5")]
    pub dict_offsets_ptype: i32,
    /// PType of the `codes` slot child.
    #[prost(enumeration = "PType", tag = "6")]
    pub codes_ptype: i32,
    /// PType of the `codes_offsets` slot child.
    #[prost(enumeration = "PType", tag = "7")]
    pub codes_offsets_ptype: i32,
    /// PType of the per-row `codes_sizes` slot child.
    #[prost(enumeration = "PType", tag = "8")]
    pub codes_sizes_ptype: i32,
}

#[array_slots(OnPairView)]
pub struct OnPairViewSlots {
    /// `PrimitiveArray<u32>`, length `dict_size + 1`.
    pub dict_offsets: ArrayRef,
    /// The shared flat token stream. Never rebuilt by `filter`/`take`/`slice`.
    pub codes: ArrayRef,
    /// `PrimitiveArray`, length `num_rows`. `codes_offsets[i]` is the *start*
    /// index of row `i`'s token window in `codes`. Unlike [`OnPair`](crate::OnPair),
    /// these need not be sorted and may overlap.
    pub codes_offsets: ArrayRef,
    /// `PrimitiveArray`, length `num_rows`. `codes_sizes[i]` is the number of
    /// tokens in row `i`'s window: `codes[codes_offsets[i]..][..codes_sizes[i]]`.
    pub codes_sizes: ArrayRef,
    /// Integer `PrimitiveArray`, length `num_rows`. Used to size the canonical
    /// output buffer.
    pub uncompressed_lengths: ArrayRef,
    /// Optional validity child for the outer string column.
    pub validity: Option<ArrayRef>,
}

/// Zero-sized VTable marker for the OnPairView encoding.
#[derive(Clone, Debug)]
pub struct OnPairView;

impl OnPairView {
    /// Build an [`OnPairViewArray`] from already-materialised parts.
    #[expect(clippy::too_many_arguments, reason = "every child is a real input")]
    pub fn try_new(
        dtype: DType,
        dict_bytes: BufferHandle,
        dict_offsets: ArrayRef,
        codes: ArrayRef,
        codes_offsets: ArrayRef,
        codes_sizes: ArrayRef,
        uncompressed_lengths: ArrayRef,
        validity: Validity,
        bits: u32,
    ) -> VortexResult<OnPairViewArray> {
        validate_parts(
            &dtype,
            &dict_offsets,
            &codes,
            &codes_offsets,
            &codes_sizes,
            &uncompressed_lengths,
            bits,
        )?;
        let len = uncompressed_lengths.len();
        let data = OnPairData::new(dict_bytes, bits, len);
        let slots = OnPairViewSlots {
            dict_offsets,
            codes,
            codes_offsets,
            codes_sizes,
            uncompressed_lengths,
            validity: validity_to_child(&validity, len),
        }
        .into_slots();
        Ok(unsafe {
            Array::from_parts_unchecked(
                ArrayParts::new(OnPairView, dtype, len, data).with_slots(slots),
            )
        })
    }

    /// Build an [`OnPairViewArray`] without validation.
    ///
    /// # Safety
    ///
    /// The caller must uphold the invariants checked by [`validate_parts`]: the
    /// per-row children all share length `num_rows`, the integer children are
    /// non-nullable, and every `[codes_offsets[i], codes_offsets[i] + codes_sizes[i])`
    /// window lies within `codes`.
    #[expect(clippy::too_many_arguments, reason = "every child is a real input")]
    pub(crate) unsafe fn new_unchecked(
        dtype: DType,
        dict_bytes: BufferHandle,
        dict_offsets: ArrayRef,
        codes: ArrayRef,
        codes_offsets: ArrayRef,
        codes_sizes: ArrayRef,
        uncompressed_lengths: ArrayRef,
        validity: Validity,
        bits: u32,
    ) -> OnPairViewArray {
        let len = uncompressed_lengths.len();
        let data = OnPairData::new(dict_bytes, bits, len);
        let slots = OnPairViewSlots {
            dict_offsets,
            codes,
            codes_offsets,
            codes_sizes,
            uncompressed_lengths,
            validity: validity_to_child(&validity, len),
        }
        .into_slots();
        unsafe {
            Array::from_parts_unchecked(
                ArrayParts::new(OnPairView, dtype, len, data).with_slots(slots),
            )
        }
    }

    /// Reshape an [`OnPairArray`] into the equivalent [`OnPairViewArray`].
    ///
    /// This is the cheap, lossless bridge from the [`List`](vortex_array::arrays::ListArray)-shaped
    /// encoding to the [`ListView`](vortex_array::arrays::ListViewArray)-shaped one: it shares
    /// the dictionary blob, `dict_offsets`, `codes`, `uncompressed_lengths` and
    /// validity verbatim, and derives `codes_offsets[i] = onpair.codes_offsets[i]`
    /// and `codes_sizes[i] = onpair.codes_offsets[i + 1] - onpair.codes_offsets[i]`.
    pub fn from_onpair(
        onpair: &OnPairArray,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<OnPairViewArray> {
        let len = onpair.len();
        // OnPair's `codes_offsets` has length `len + 1`; widen to a host buffer so
        // we can split it into a per-row offset and a per-row size.
        let cumulative = collect_widened::<u32>(onpair.codes_offsets(), ctx)?;
        vortex_ensure!(
            cumulative.len() == len + 1,
            "OnPair codes_offsets length {} != rows + 1 ({})",
            cumulative.len(),
            len + 1
        );
        let offsets = Buffer::<u32>::from_iter(cumulative.as_slice()[..len].iter().copied());
        let sizes = Buffer::<u32>::from_iter((0..len).map(|i| cumulative[i + 1] - cumulative[i]));

        OnPairView::try_new(
            onpair.dtype().clone(),
            onpair.dict_bytes_handle().clone(),
            onpair.dict_offsets().clone(),
            onpair.codes().clone(),
            offsets.into_array(),
            sizes.into_array(),
            onpair.uncompressed_lengths().clone(),
            onpair.array_validity(),
            onpair.bits(),
        )
    }
}

fn validate_parts(
    dtype: &DType,
    dict_offsets: &ArrayRef,
    codes: &ArrayRef,
    codes_offsets: &ArrayRef,
    codes_sizes: &ArrayRef,
    uncompressed_lengths: &ArrayRef,
    bits: u32,
) -> VortexResult<()> {
    vortex_ensure!(
        matches!(dtype, DType::Binary(_) | DType::Utf8(_)),
        "OnPairView arrays must be Binary or Utf8, found {dtype}"
    );
    vortex_ensure!((9..=16).contains(&bits), "bits {bits} out of range [9, 16]");

    for (name, child) in [
        ("dict_offsets", dict_offsets),
        ("codes", codes),
        ("codes_offsets", codes_offsets),
        ("codes_sizes", codes_sizes),
        ("uncompressed_lengths", uncompressed_lengths),
    ] {
        if !child.dtype().is_int() || child.dtype().is_nullable() {
            vortex_bail!(InvalidArgument: "{name} must be a non-nullable integer");
        }
    }

    if codes_offsets.len() != codes_sizes.len() {
        vortex_bail!(InvalidArgument:
            "codes_offsets.len ({}) != codes_sizes.len ({})",
            codes_offsets.len(),
            codes_sizes.len()
        );
    }
    if codes_offsets.len() != uncompressed_lengths.len() {
        vortex_bail!(InvalidArgument:
            "codes_offsets.len ({}) != uncompressed_lengths.len ({})",
            codes_offsets.len(),
            uncompressed_lengths.len()
        );
    }
    Ok(())
}

impl VTable for OnPairView {
    type TypedArrayData = OnPairData;
    type OperationsVTable = Self;
    type ValidityVTable = Self;

    fn id(&self) -> vortex_array::ArrayId {
        static ID: CachedId = CachedId::new("vortex.onpairview");
        *ID
    }

    fn validate(
        &self,
        data: &Self::TypedArrayData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        let s = OnPairViewSlotsView::from_slots(slots);
        validate_parts(
            dtype,
            s.dict_offsets,
            s.codes,
            s.codes_offsets,
            s.codes_sizes,
            s.uncompressed_lengths,
            data.bits(),
        )?;
        if s.uncompressed_lengths.len() != len {
            vortex_bail!(InvalidArgument: "uncompressed_lengths must have same len as outer array");
        }
        if data.len() != len {
            vortex_bail!(InvalidArgument: "OnPairData len {} != outer len {}", data.len(), len);
        }
        Ok(())
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        1
    }

    fn buffer(array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        match idx {
            0 => array.dict_bytes_handle().clone(),
            _ => vortex_panic!("OnPairViewArray buffer index {idx} out of bounds"),
        }
    }

    fn buffer_name(_array: ArrayView<'_, Self>, idx: usize) -> Option<String> {
        match idx {
            0 => Some("dict_bytes".to_string()),
            _ => vortex_panic!("OnPairViewArray buffer_name index {idx} out of bounds"),
        }
    }

    fn serialize(
        array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        let dict_size = u32::try_from(array.dict_offsets().len().saturating_sub(1))
            .map_err(|_| vortex_err!("OnPairView dict_size exceeds u32"))?;
        let total_tokens = array.codes().len() as u64;
        Ok(Some(
            OnPairViewMetadata {
                uncompressed_lengths_ptype: array.uncompressed_lengths().dtype().as_ptype().into(),
                bits: array.bits(),
                dict_size,
                total_tokens,
                dict_offsets_ptype: array.dict_offsets().dtype().as_ptype().into(),
                codes_ptype: array.codes().dtype().as_ptype().into(),
                codes_offsets_ptype: array.codes_offsets().dtype().as_ptype().into(),
                codes_sizes_ptype: array.codes_sizes().dtype().as_ptype().into(),
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
        if buffers.len() != 1 {
            vortex_bail!(InvalidArgument: "Expected 1 buffer, got {}", buffers.len());
        }
        let metadata = OnPairViewMetadata::decode(metadata)?;
        let ptype = |raw: i32, what: &str| {
            PType::try_from(raw).map_err(|_| vortex_err!("invalid {what} ptype {raw}"))
        };
        let uncompressed_ptype =
            ptype(metadata.uncompressed_lengths_ptype, "uncompressed_lengths")?;
        let dict_offsets_ptype = ptype(metadata.dict_offsets_ptype, "dict_offsets")?;
        let codes_ptype = ptype(metadata.codes_ptype, "codes")?;
        let codes_offsets_ptype = ptype(metadata.codes_offsets_ptype, "codes_offsets")?;
        let codes_sizes_ptype = ptype(metadata.codes_sizes_ptype, "codes_sizes")?;

        let dict_offsets_len = metadata.dict_size as usize + 1;
        let total_tokens = usize::try_from(metadata.total_tokens)
            .map_err(|_| vortex_err!("total_tokens {} overflows usize", metadata.total_tokens))?;

        let prim = |ptype: PType| DType::Primitive(ptype, Nullability::NonNullable);
        let dict_offsets = children.get(0, &prim(dict_offsets_ptype), dict_offsets_len)?;
        let codes = children.get(1, &prim(codes_ptype), total_tokens)?;
        let codes_offsets = children.get(2, &prim(codes_offsets_ptype), len)?;
        let codes_sizes = children.get(3, &prim(codes_sizes_ptype), len)?;
        let uncompressed_lengths = children.get(4, &prim(uncompressed_ptype), len)?;
        let validity = match children.len() {
            5 => Validity::from(dtype.nullability()),
            6 => Validity::Array(children.get(5, &Validity::DTYPE, len)?),
            other => vortex_bail!(InvalidArgument: "Expected 5 or 6 children, got {other}"),
        };

        let data = OnPairData::new(buffers[0].clone(), metadata.bits, len);
        let slots = OnPairViewSlots {
            dict_offsets,
            codes,
            codes_offsets,
            codes_sizes,
            uncompressed_lengths,
            validity: validity_to_child(&validity, len),
        }
        .into_slots();
        Ok(ArrayParts::new(self.clone(), dtype.clone(), len, data).with_slots(slots))
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        OnPairViewSlots::NAMES[idx].to_string()
    }

    fn execute(array: Array<Self>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        canonicalize_onpairview(array.as_view(), ctx).map(ExecutionResult::done)
    }

    fn append_to_builder(
        array: ArrayView<'_, Self>,
        builder: &mut dyn ArrayBuilder,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        builder.extend_from_array(
            &array
                .array()
                .clone()
                .execute::<Canonical>(ctx)?
                .into_array(),
        );
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

impl ValidityVTable<OnPairView> for OnPairView {
    fn validity(array: ArrayView<'_, OnPairView>) -> VortexResult<Validity> {
        Ok(child_to_validity(
            array.slots()[OnPairViewSlots::VALIDITY].as_ref(),
            array.dtype().nullability(),
        ))
    }
}

/// Convenience methods on top of the macro-generated [`OnPairViewArraySlotsExt`].
pub trait OnPairViewArrayExt: OnPairViewArraySlotsExt {
    /// The outer string column validity.
    fn array_validity(&self) -> Validity {
        child_to_validity(
            self.as_ref().slots()[OnPairViewSlots::VALIDITY].as_ref(),
            self.as_ref().dtype().nullability(),
        )
    }

    /// Materialise the whole token stream widened to `u16` (the decoder width).
    fn collect_codes(&self, ctx: &mut ExecutionCtx) -> VortexResult<Buffer<u16>> {
        collect_widened::<u16>(self.codes(), ctx)
    }

    /// Materialise the per-row `codes_offsets` as a host `u32` buffer.
    fn collect_offsets(&self, ctx: &mut ExecutionCtx) -> VortexResult<Buffer<u32>> {
        collect_widened::<u32>(self.codes_offsets(), ctx)
    }

    /// Materialise the per-row `codes_sizes` as a host `u32` buffer.
    fn collect_sizes(&self, ctx: &mut ExecutionCtx) -> VortexResult<Buffer<u32>> {
        collect_widened::<u32>(self.codes_sizes(), ctx)
    }
}

impl<T: OnPairViewArraySlotsExt> OnPairViewArrayExt for T {}
