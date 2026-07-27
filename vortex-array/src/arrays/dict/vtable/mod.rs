// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hasher;

use num_traits::AsPrimitive;
use prost::Message;
use smallvec::smallvec;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_mask::AllOr;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use super::DictData;
use super::DictMetadata;
use super::DictOwnedExt;
use super::DictParts;
use super::array::DictSlots;
use super::array::DictSlotsView;
use crate::AnyCanonical;
use crate::ArrayEq;
use crate::ArrayHash;
use crate::ArrayRef;
use crate::Canonical;
use crate::EqMode;
use crate::IntoArray;
use crate::array::Array;
use crate::array::ArrayId;
use crate::array::ArrayParts;
use crate::array::ArrayView;
use crate::array::VTable;
use crate::array::with_empty_buffers;
use crate::arrays::ConstantArray;
use crate::arrays::Primitive;
use crate::arrays::PrimitiveArray;
use crate::arrays::VarBinViewArray;
use crate::arrays::dict::DictArrayExt;
use crate::arrays::dict::DictArraySlotsExt;
use crate::arrays::dict::compute::rules::PARENT_RULES;
use crate::arrays::dict::execute::take_canonical;
use crate::buffer::BufferHandle;
use crate::builders::ArrayBuilder;
use crate::builders::DynVarBinBuilder;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::executor::ExecutionCtx;
use crate::executor::ExecutionResult;
use crate::match_each_integer_ptype;
use crate::require_child;
use crate::scalar::Scalar;
use crate::serde::ArrayChildren;

mod kernel;
mod operations;
mod validity;

/// A [`Dict`]-encoded Vortex array.
pub type DictArray = Array<Dict>;

pub(crate) fn initialize(session: &VortexSession) {
    kernel::initialize(session);
}

#[derive(Clone, Debug)]
pub struct Dict;

impl ArrayHash for DictData {
    fn array_hash<H: Hasher>(&self, _state: &mut H, _accuracy: EqMode) {}
}

impl ArrayEq for DictData {
    fn array_eq(&self, _other: &Self, _accuracy: EqMode) -> bool {
        true
    }
}

impl VTable for Dict {
    type TypedArrayData = DictData;

    type OperationsVTable = Self;
    type ValidityVTable = Self;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.dict");
        *ID
    }

    fn validate(
        &self,
        _data: &DictData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        let view = DictSlotsView::from_slots(slots);
        let codes = view.codes;
        let values = view.values;
        vortex_ensure!(codes.len() == len, "DictArray codes length mismatch");
        vortex_ensure!(
            values
                .dtype()
                .union_nullability(codes.dtype().nullability())
                == *dtype,
            "DictArray dtype does not match codes/values dtype"
        );
        Ok(())
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        0
    }

    fn buffer(_array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        vortex_panic!("DictArray buffer index {idx} out of bounds")
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

    fn serialize(
        array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(
            DictMetadata {
                codes_ptype: PType::try_from(array.codes().dtype())? as i32,
                values_len: u32::try_from(array.values().len()).map_err(|_| {
                    vortex_err!(
                        "Dictionary values size {} overflowed u32",
                        array.values().len()
                    )
                })?,
                is_nullable_codes: Some(array.codes().dtype().is_nullable()),
                all_values_referenced: Some(array.has_all_values_referenced()),
            }
            .encode_to_vec(),
        ))
    }

    fn deserialize(
        &self,
        dtype: &DType,
        len: usize,
        metadata: &[u8],
        _buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
        _session: &VortexSession,
    ) -> VortexResult<ArrayParts<Self>> {
        let metadata = DictMetadata::decode(metadata)?;
        if children.len() != 2 {
            vortex_bail!(
                "Expected 2 children for dict encoding, found {}",
                children.len()
            )
        }
        let codes_nullable = metadata
            .is_nullable_codes
            .map(Nullability::from)
            // If no `is_nullable_codes` metadata use the nullability of the values
            // (and whole array) as before.
            .unwrap_or_else(|| dtype.nullability());
        let codes_dtype = DType::Primitive(metadata.codes_ptype(), codes_nullable);
        let codes = children.get(0, &codes_dtype, len)?;
        let values = children.get(1, dtype, metadata.values_len as usize)?;
        let all_values_referenced = metadata.all_values_referenced.unwrap_or(false);

        Ok(ArrayParts::new(self.clone(), dtype.clone(), len, unsafe {
            DictData::new_unchecked().set_all_values_referenced(all_values_referenced)
        })
        .with_slots(smallvec![Some(codes), Some(values)]))
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        DictSlots::NAMES[idx].to_string()
    }

    fn execute(array: Array<Self>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        if array.is_empty() {
            let result_dtype = array
                .dtype()
                .union_nullability(array.codes().dtype().nullability());
            return Ok(ExecutionResult::done(Canonical::empty(&result_dtype)));
        }

        let array = require_child!(array, array.codes(), DictSlots::CODES => Primitive);

        if array.codes().validity()?.definitely_all_null() {
            return Ok(ExecutionResult::done(ConstantArray::new(
                Scalar::null(array.dtype().as_nullable()),
                array.codes().len(),
            )));
        }

        let array = require_child!(array, array.values(), DictSlots::VALUES => AnyCanonical);

        let DictParts { values, codes, .. } = array.into_parts();

        Ok(ExecutionResult::done(take_canonical(
            values.as_::<AnyCanonical>(),
            &codes.downcast::<Primitive>(),
            ctx,
        )?))
    }

    fn append_to_builder(
        array: ArrayView<'_, Self>,
        builder: &mut dyn ArrayBuilder,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        // The generic path below takes the values to full logical length, allocating an
        // intermediate that is then copied into the builder again. Gather by code instead.
        if matches!(array.dtype(), DType::Utf8(_) | DType::Binary(_))
            && !array.is_empty()
            && let Some(codes) = array.codes().as_opt::<Primitive>()
            && !codes.validity()?.definitely_all_null()
            && builder.as_any().is::<DynVarBinBuilder>()
        {
            let codes = codes.into_owned();
            return append_dict_bytes(array, &codes, builder, ctx);
        }

        if !array.is_empty()
            && let (Some(codes), Some(values)) = (
                array.codes().as_opt::<Primitive>(),
                array.values().as_opt::<AnyCanonical>(),
            )
            && !codes.validity()?.definitely_all_null()
        {
            let codes = codes.into_owned();
            let canonical = take_canonical(values, &codes, ctx)?.into_array();
            canonical.append_to_builder(builder, ctx)?;
            return Ok(());
        }

        let canonical = array
            .array()
            .clone()
            .execute::<Canonical>(ctx)?
            .into_array();
        canonical.append_to_builder(builder, ctx)?;
        Ok(())
    }

    fn reduce_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_RULES.evaluate(array, parent, child_idx)
    }
}

/// Gathers UTF-8 or binary dictionary values into a [`DynVarBinBuilder`] by code.
///
/// `builder` must be a [`DynVarBinBuilder`]; the caller checks this before dispatching here.
fn append_dict_bytes(
    array: ArrayView<'_, Dict>,
    codes: &PrimitiveArray,
    builder: &mut dyn ArrayBuilder,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    let len = array.len();
    let values = array.values().clone().execute::<VarBinViewArray>(ctx)?;
    let values_mask = values.validity()?.execute_mask(values.len(), ctx)?;
    let codes_mask = codes.as_ref().validity()?.execute_mask(len, ctx)?;

    let views = values.views();
    let buffers: Vec<&[u8]> = (0..values.data_buffers().len())
        .map(|idx| values.buffer(idx).as_slice())
        .collect();

    let view_bytes = |index: usize| -> &[u8] {
        let view = &views[index];
        if view.is_inlined() {
            view.as_inlined().value()
        } else {
            let reference = view.as_view();
            &buffers[reference.buffer_index as usize][reference.as_range()]
        }
    };

    let builder = builder
        .as_any_mut()
        .downcast_mut::<DynVarBinBuilder>()
        .vortex_expect("caller checked that the builder is a DynVarBinBuilder");

    match_each_integer_ptype!(codes.ptype(), |P| {
        let codes = codes.as_slice::<P>();
        let append_code = |builder: &mut DynVarBinBuilder, row: usize| {
            let code: usize = codes[row].as_();
            if values_mask.value(code) {
                builder.append_n_values(view_bytes(code), 1);
            } else {
                // The code may point at a null dictionary entry.
                builder.append_nulls(1);
            }
        };

        match codes_mask.bit_buffer() {
            AllOr::All => {
                for row in 0..len {
                    append_code(&mut *builder, row);
                }
            }
            AllOr::None => builder.append_nulls(len),
            AllOr::Some(valid) => {
                let mut row = 0;
                valid.for_each_set_index(|index| {
                    builder.append_nulls(index - row);
                    append_code(&mut *builder, index);
                    row = index + 1;
                });
                builder.append_nulls(len - row);
            }
        }
    });

    Ok(())
}
