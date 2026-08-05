// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hasher;

use num_traits::AsPrimitive;
use prost::Message;
use smallvec::smallvec;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_mask::AllOr;
use vortex_mask::Mask;
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
use crate::CanonicalView;
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
use crate::arrays::VarBinView;
use crate::arrays::dict::DictArrayExt;
use crate::arrays::dict::DictArraySlotsExt;
use crate::arrays::dict::compute::rules::PARENT_RULES;
use crate::arrays::dict::execute::take_canonical;
use crate::buffer::BufferHandle;
use crate::builders::ArrayBuilder;
use crate::builders::VarBinBuilder;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::OffsetBuilderPType;
use crate::dtype::PType;
use crate::executor::ExecutionCtx;
use crate::executor::ExecutionResult;
use crate::match_each_integer_ptype;
use crate::match_each_varbin_builder;
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
            codes.as_::<Primitive>(),
            ctx,
        )?))
    }

    fn append_to_builder(
        array: ArrayView<'_, Self>,
        builder: &mut dyn ArrayBuilder,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        if !array.is_empty()
            && let (Some(codes), Some(values)) = (
                array.codes().as_opt::<Primitive>(),
                array.values().as_opt::<AnyCanonical>(),
            )
            && !codes.validity()?.definitely_all_null()
        {
            if let CanonicalView::VarBinView(values) = values
                && let Some(result) = match_each_varbin_builder!(builder, |builder| {
                    let validity = array.validity()?.execute_mask(array.len(), ctx)?;
                    append_dict_to_varbin(codes, values, validity, builder)
                })
            {
                return result;
            }
            let canonical = take_canonical(values, codes, ctx)?.into_array();
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

/// Gathers the dictionary values straight into `builder`.
///
/// The canonical route first takes the values to full logical length, which allocates and then
/// re-reads a views buffer proportional to the row count. The dictionary is usually far smaller
/// than the column, so resolving each code against it in place skips that intermediate entirely
/// and leaves one `memcpy` per row as the only work.
fn append_dict_to_varbin<O: OffsetBuilderPType>(
    codes: ArrayView<'_, Primitive>,
    values: ArrayView<'_, VarBinView>,
    validity: Mask,
    builder: &mut VarBinBuilder<O>,
) -> VortexResult<()>
where
    usize: AsPrimitive<O>,
{
    let len = codes.as_ref().len();

    // Resolve the dictionary's storage once so that looking up a code is an O(1) read.
    let views = values.views();
    let buffers = values
        .data_buffers()
        .iter()
        .map(|buffer| buffer.as_host().as_slice())
        .collect::<Vec<_>>();

    match_each_integer_ptype!(codes.ptype(), |C| {
        let codes = codes.as_slice::<C>();
        let view = |row: usize| &views[AsPrimitive::<usize>::as_(codes[row])];

        // Both passes below resolve a row through its code, so the byte total comes from the same
        // walk over the valid rows that the copy will make.
        let num_bytes = match validity.bit_buffer() {
            AllOr::All => (0..len).map(|row| view(row).len() as usize).sum(),
            AllOr::None => {
                builder.push_nulls(len);
                return Ok(());
            }
            AllOr::Some(bits) => {
                let mut total = 0;
                bits.for_each_set_index(|row| total += view(row).len() as usize);
                total
            }
        };

        builder.append_valid_slices(num_bytes, &validity, |row| view(row).bytes(&buffers))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::VortexSessionExecute;
    use crate::array_session;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::VarBinViewArray;
    use crate::arrays::dict::DictArray;
    use crate::assert_arrays_eq;
    use crate::dtype::Nullability::Nullable;

    const LONG: &str = "a string that is far too long to be inlined in a view";

    #[test]
    fn append_to_builder_gathers_through_the_dictionary() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let dict = DictArray::try_new(
            PrimitiveArray::from_option_iter([Some(0u32), Some(2), None, Some(1), Some(0)])
                .into_array(),
            VarBinViewArray::from_iter([Some(LONG), None, Some("short")], DType::Utf8(Nullable))
                .into_array(),
        )?;

        let mut builder = VarBinBuilder::<i32>::new(DType::Utf8(Nullable));
        dict.append_to_builder(&mut builder, &mut ctx)?;

        let expected = VarBinViewArray::from_iter(
            [Some(LONG), Some("short"), None, None, Some(LONG)],
            DType::Utf8(Nullable),
        );
        assert_arrays_eq!(builder.finish_into_varbin(), expected, &mut ctx);
        Ok(())
    }
}
