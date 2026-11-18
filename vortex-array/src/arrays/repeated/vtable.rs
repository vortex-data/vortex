// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::iter;

use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_session::registry::CachedId;

use crate::AnyCanonical;
use crate::ArrayRef;
use crate::Canonical;
use crate::EmptyArrayData;
use crate::ExecutionCtx;
use crate::ExecutionResult;
use crate::IntoArray;
use crate::array::Array;
use crate::array::ArrayId;
use crate::array::ArrayParts;
use crate::array::ArrayView;
use crate::array::OperationsVTable;
use crate::array::ProbeState;
use crate::array::VTable;
use crate::array::ValidityVTable;
use crate::array::with_empty_buffers;
use crate::arrays::ChunkedArray;
use crate::arrays::Constant;
use crate::arrays::ConstantArray;
use crate::arrays::ExtensionArray;
use crate::arrays::FixedSizeListArray;
use crate::arrays::ListViewArray;
use crate::arrays::StructArray;
use crate::arrays::VariantArray;
use crate::arrays::extension::ExtensionArrayExt;
use crate::arrays::fixed_size_list::FixedSizeListArrayExt;
use crate::arrays::fixed_size_list::FixedSizeListArraySlotsExt;
use crate::arrays::listview::ListViewArrayExt;
use crate::arrays::listview::ListViewArraySlotsExt;
use crate::arrays::repeated::RepeatedArraySlotsExt;
use crate::arrays::repeated::RepeatedSlots;
use crate::arrays::struct_::StructArrayExt;
use crate::arrays::variant::VariantArraySlotsExt;
use crate::buffer::BufferHandle;
use crate::builders::ArrayBuilder;
use crate::dtype::DType;
use crate::require_child;
use crate::scalar::Scalar;
use crate::validity::Validity;

/// The encoding of a [`RepeatedArray`].
#[derive(Clone, Debug)]
pub struct Repeated;

/// A [`Repeated`]-encoded array, which repeats its child end to end.
pub type RepeatedArray = Array<Repeated>;

impl VTable for Repeated {
    type TypedArrayData = EmptyArrayData;

    type OperationsVTable = Self;
    type ValidityVTable = Self;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.repeated");
        *ID
    }

    fn validate(
        &self,
        _data: &EmptyArrayData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        vortex_ensure_eq!(slots.len(), RepeatedSlots::COUNT);
        let child = slots[RepeatedSlots::CHILD]
            .as_ref()
            .ok_or_else(|| vortex_err!("Repeated requires a child"))?;
        vortex_ensure_eq!(
            child.dtype(),
            dtype,
            "Repeated child dtype does not match outer dtype"
        );
        vortex_ensure!(
            len == 0 || !child.is_empty(),
            "Repeated cannot repeat an empty child to {len} rows"
        );
        Ok(())
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        0
    }

    fn buffer(_array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        vortex_panic!("Repeated buffer index {idx} out of bounds")
    }

    fn buffer_name(_array: ArrayView<'_, Self>, _idx: usize) -> Option<String> {
        None
    }

    fn with_buffers(
        &self,
        array: ArrayView<'_, Self>,
        buffers: &[BufferHandle],
    ) -> VortexResult<ArrayParts<Self>> {
        with_empty_buffers(self, array, buffers)
    }

    fn append_to_builder(
        array: ArrayView<'_, Self>,
        builder: &mut dyn ArrayBuilder,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        if array.is_empty() {
            return Ok(());
        }
        let child = array.child().clone().execute::<Canonical>(ctx)?;
        // A nested child appends its children repeated, which a builder keeps as one chunk each.
        if let Some(repeated) = repeat_nested(child.clone(), array.len())? {
            return repeated.append_to_builder(builder, ctx);
        }
        let child = child.into_array();
        for copy in copies(&child, array.len())? {
            copy.append_to_builder(builder, ctx)?;
        }
        Ok(())
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        RepeatedSlots::NAMES[idx].to_string()
    }

    fn execute(array: Array<Self>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        if array.is_empty() {
            return Ok(ExecutionResult::done(
                Canonical::empty(array.dtype()).into_array(),
            ));
        }
        if array.child().len() == 1 {
            let scalar = array.child().execute_scalar(0, ctx)?;
            return Ok(ExecutionResult::done(
                ConstantArray::new(scalar, array.len()).into_array(),
            ));
        }

        let array = require_child!(array, array.child(), RepeatedSlots::CHILD => AnyCanonical);

        let child = Canonical::from(array.child().as_::<AnyCanonical>());
        if let Some(repeated) = repeat_nested(child, array.len())? {
            return Ok(ExecutionResult::done(repeated));
        }

        // Decode the child once, then let `Chunked` append the decoded child once per copy.
        let chunked =
            ChunkedArray::try_new(copies(array.child(), array.len())?, array.dtype().clone())?;
        Ok(ExecutionResult::done(chunked.into_array()))
    }

    fn reduce(array: ArrayView<'_, Self>) -> VortexResult<Option<ArrayRef>> {
        if array.is_empty() {
            return Ok(Some(Canonical::empty(array.dtype()).into_array()));
        }
        let child = array.child();
        if array.len() <= child.len() {
            return Ok(Some(child.slice(0..array.len())?));
        }
        if let Some(constant) = child.as_opt::<Constant>() {
            return Ok(Some(
                ConstantArray::new(constant.scalar().clone(), array.len()).into_array(),
            ));
        }
        // An inner repeat that ends on a whole copy of its own child repeats that child exactly.
        if let Some(inner) = child.as_opt::<Repeated>()
            && inner.len().is_multiple_of(inner.child().len())
        {
            return Ok(Some(
                RepeatedArray::try_new(inner.child().clone(), array.len())?.into_array(),
            ));
        }
        Ok(None)
    }
}

/// Repeats a nested canonical `child` to `len` rows by repeating its children, without copying
/// any values.
///
/// A canonical array is canonical at the top level only, so the result keeps its children as
/// `Repeated` arrays. Returns `None` for a child whose values are its own buffers, which must be
/// copied.
fn repeat_nested(child: Canonical, len: usize) -> VortexResult<Option<ArrayRef>> {
    let repeat = |array: &ArrayRef| -> VortexResult<ArrayRef> {
        Ok(RepeatedArray::try_new(array.clone(), len)?.into_array())
    };
    Ok(Some(match child {
        Canonical::Struct(array) => StructArray::try_new(
            array.names().clone(),
            array
                .iter_unmasked_fields()
                .map(repeat)
                .collect::<VortexResult<Vec<_>>>()?,
            len,
            repeat_validity(array.struct_validity(), len)?,
        )?
        .into_array(),
        Canonical::FixedSizeList(array) => {
            let list_size = array.list_size();
            let elements_len = len
                .checked_mul(list_size as usize)
                .ok_or_else(|| vortex_err!("{len} lists of {list_size} elements overflow usize"))?;
            FixedSizeListArray::try_new(
                RepeatedArray::try_new(array.elements().clone(), elements_len)?.into_array(),
                list_size,
                repeat_validity(array.fixed_size_list_validity(), len)?,
                len,
            )?
            .into_array()
        }
        Canonical::List(array) => {
            // SAFETY: every view of the result is a view of `array` over the same elements, so the
            // offsets and sizes keep their types and stay within the elements.
            unsafe {
                ListViewArray::new_unchecked(
                    array.elements().clone(),
                    repeat(array.offsets())?,
                    repeat(array.sizes())?,
                    repeat_validity(array.listview_validity(), len)?,
                )
            }
            .into_array()
        }
        Canonical::Extension(array) => {
            let storage = repeat(array.storage_array())?;

            // SAFETY: repeating storage from array preserves its dtype and copies its values.
            let result =
                unsafe { ExtensionArray::new_unchecked(array.ext_dtype().clone(), storage) };
            result.into_array()
        }
        Canonical::Variant(array) => VariantArray::try_new(
            repeat(array.core_storage())?,
            array.shredded().map(repeat).transpose()?,
        )?
        .into_array(),
        _ => return Ok(None),
    }))
}

/// Repeats `validity` to `len` rows.
fn repeat_validity(validity: Validity, len: usize) -> VortexResult<Validity> {
    Ok(match validity {
        Validity::Array(validity) => {
            Validity::Array(RepeatedArray::try_new(validity, len)?.into_array())
        }
        validity => validity,
    })
}

/// Returns the copies of the non-empty `child` that make up `len` rows: whole copies of `child`,
/// then a prefix of it for the last, partial copy.
fn copies(child: &ArrayRef, len: usize) -> VortexResult<impl Iterator<Item = ArrayRef>> {
    let whole = len / child.len();
    let rest = len % child.len();
    let partial = (rest > 0).then(|| child.slice(0..rest)).transpose()?;
    Ok(iter::repeat_n(child.clone(), whole).chain(partial))
}

impl OperationsVTable<Repeated> for Repeated {
    type ProbeState = ();

    fn probe_scalar(
        state: &mut ProbeState<'_, Repeated>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let child_index = index % state.array().child().len();
        state
            .slot(RepeatedSlots::CHILD)?
            .ok_or_else(|| vortex_err!("Repeated child slot is missing"))?
            .execute_scalar(child_index, ctx)
    }

    fn scalar_at(
        array: ArrayView<'_, Repeated>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        Self::probe_scalar(&mut ProbeState::once(array), index, ctx)
    }
}

impl ValidityVTable<Repeated> for Repeated {
    fn validity(array: ArrayView<'_, Repeated>) -> VortexResult<Validity> {
        repeat_validity(array.child().validity()?, array.len())
    }
}
