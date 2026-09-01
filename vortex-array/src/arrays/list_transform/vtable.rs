// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;
use vortex_mask::Mask;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use super::array::output_dtype;
use super::rules::PARENT_RULES;
use crate::ArrayParts;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::ExecutionResult;
use crate::IntoArray;
use crate::array::Array;
use crate::array::ArrayId;
use crate::array::ArrayView;
use crate::array::EmptyArrayData;
use crate::array::VTable;
use crate::array::ValidityVTable;
use crate::array::with_empty_buffers;
use crate::arrays::ConstantArray;
use crate::arrays::FixedSizeList;
use crate::arrays::FixedSizeListArray;
use crate::arrays::InterleaveArray;
use crate::arrays::List;
use crate::arrays::ListArray;
use crate::arrays::ListTransformArrayExt;
use crate::arrays::ListView;
use crate::arrays::ListViewArray;
use crate::arrays::PiecewiseSequenceArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::fixed_size_list::FixedSizeListArrayExt;
use crate::arrays::fixed_size_list::FixedSizeListArraySlotsExt;
use crate::arrays::list::ListArrayExt;
use crate::arrays::list::ListArraySlotsExt;
use crate::arrays::listview::ListViewArrayExt;
use crate::arrays::listview::ListViewArraySlotsExt;
use crate::arrays::listview::ListViewRebuildMode;
use crate::arrays::scalar_fn::ScalarFnArrayExt;
use crate::arrays::template::TemplateInputArrayExt;
use crate::arrays::template::instantiate;
use crate::arrays::template::template_scope;
use crate::buffer::BufferHandle;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::matcher::Matcher;
use crate::scalar::Scalar;
use crate::scalar_fn::fns::operators::Operator;
use crate::serde::ArrayChildren;
use crate::validity::Validity;
use crate::vtable::NotSupported;

/// A lazy structural list transformation.
pub type ListTransformArray = Array<ListTransform>;

#[derive(Clone, Debug)]
pub struct ListTransform;

impl VTable for ListTransform {
    type TypedArrayData = EmptyArrayData;
    type OperationsVTable = NotSupported;
    type ValidityVTable = Self;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.list-transform");
        *ID
    }

    fn validate(
        &self,
        _data: &Self::TypedArrayData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        vortex_ensure!(
            slots.len() >= 2,
            "ListTransformArray requires list and body slots, got {}",
            slots.len()
        );
        let list = slots[0]
            .as_ref()
            .ok_or_else(|| vortex_error::vortex_err!("ListTransformArray list slot is missing"))?;
        let body = slots[1]
            .as_ref()
            .ok_or_else(|| vortex_error::vortex_err!("ListTransformArray body slot is missing"))?;
        vortex_ensure!(
            list.len() == len,
            "ListTransformArray list length does not match outer length"
        );
        vortex_ensure!(
            body.is_empty(),
            "ListTransformArray body must have length zero"
        );
        vortex_ensure!(
            output_dtype(list.dtype(), body.dtype())? == *dtype,
            "ListTransformArray dtype does not match its list and body children"
        );
        vortex_ensure!(
            slots[2..]
                .iter()
                .all(|capture| capture.as_ref().is_some_and(|capture| capture.len() == len)),
            "ListTransformArray captures must be present and match the outer length"
        );
        Ok(())
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        0
    }

    fn buffer(_array: ArrayView<'_, Self>, _idx: usize) -> BufferHandle {
        vortex_panic!("ListTransformArray has no buffers")
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

    fn slot_name(array: ArrayView<'_, Self>, idx: usize) -> String {
        match idx {
            0 => "list".to_string(),
            1 => "body".to_string(),
            index if index < array.slots().len() => format!("capture[{}]", index - 2),
            _ => vortex_panic!("ListTransformArray slot index {idx} out of bounds"),
        }
    }

    fn serialize(
        _array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        // Template scopes are process-local.  Expression serialization retains source lambdas,
        // whereas persistent lazy-array serialization is deliberately deferred.
        Ok(None)
    }

    fn deserialize(
        &self,
        _dtype: &DType,
        _len: usize,
        _metadata: &[u8],
        _buffers: &[BufferHandle],
        _children: &dyn ArrayChildren,
        _session: &VortexSession,
    ) -> VortexResult<ArrayParts<Self>> {
        vortex_bail!("ListTransformArray is not serializable")
    }

    fn execute(array: Array<Self>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        let list = array.list().clone().execute_until::<AnyList>(ctx)?;
        let captures = array.captures().cloned().collect::<Vec<_>>();
        let body = array.body().clone();
        let result = if let Some(list) = list.as_opt::<List>() {
            execute_list(list.into_owned(), body, captures, ctx)?
        } else if let Some(list) = list.as_opt::<FixedSizeList>() {
            execute_fixed_size_list(list.into_owned(), body, captures, ctx)?
        } else if let Some(list) = list.as_opt::<ListView>() {
            execute_list_view(list.into_owned(), body, captures, ctx)?
        } else {
            unreachable!("AnyList only matches list encodings")
        };
        Ok(ExecutionResult::done(result))
    }

    fn reduce_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_RULES.evaluate(array, parent, child_idx)
    }
}

impl ValidityVTable<ListTransform> for ListTransform {
    fn validity(array: ArrayView<'_, ListTransform>) -> VortexResult<Validity> {
        array.list().validity()
    }
}

fn execute_list(
    list: ListArray,
    body: ArrayRef,
    captures: Vec<ArrayRef>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let list = list.reset_offsets(false, ctx)?;
    let offsets = list.offsets();
    let sizes = offsets
        .slice(1..offsets.len())?
        .binary(offsets.slice(0..list.len())?, Operator::Sub)?;
    let transformed = transform_elements(
        body,
        list.elements().clone(),
        sizes,
        list.list_validity(),
        captures,
        ctx,
    )?;
    ListArray::try_new(transformed, list.offsets().clone(), list.list_validity())
        .map(IntoArray::into_array)
}

fn execute_fixed_size_list(
    list: FixedSizeListArray,
    body: ArrayRef,
    captures: Vec<ArrayRef>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let size = list.list_size();
    let sizes = ConstantArray::new(u64::from(size), list.len()).into_array();
    let transformed = transform_elements(
        body,
        list.elements().clone(),
        sizes,
        list.fixed_size_list_validity(),
        captures,
        ctx,
    )?;
    FixedSizeListArray::try_new(
        transformed,
        size,
        list.fixed_size_list_validity(),
        list.len(),
    )
    .map(IntoArray::into_array)
}

fn execute_list_view(
    list: ListViewArray,
    body: ArrayRef,
    captures: Vec<ArrayRef>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    // Rebuild to a logical element domain.  This duplicates overlaps and omits elements hidden by
    // null containers, giving every invocation exactly one outer-row parent.
    let list = list.rebuild(ListViewRebuildMode::MakeZeroCopyToList, ctx)?;
    let transformed = transform_elements(
        body,
        list.elements().clone(),
        list.sizes().clone(),
        list.listview_validity(),
        captures,
        ctx,
    )?;
    // SAFETY: rebuild produced sequential non-overlapping views and the transformed elements have
    // exactly the same invocation domain.
    Ok(unsafe {
        ListViewArray::new_unchecked(
            transformed,
            list.offsets().clone(),
            list.sizes().clone(),
            list.listview_validity(),
        )
        .with_zero_copy_to_list(true)
    }
    .into_array())
}

fn transform_elements(
    body: ArrayRef,
    elements: ArrayRef,
    sizes: ArrayRef,
    validity: Validity,
    captures: Vec<ArrayRef>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let invocation_count = elements.len();
    let parents = parent_indices(sizes.clone(), invocation_count)?;
    let invocation_mask = if matches!(validity, Validity::NonNullable | Validity::AllValid) {
        None
    } else {
        let validity = validity.take(&parents)?;
        let mask = validity.execute_mask(invocation_count, ctx)?;
        (mask.true_count() != invocation_count).then_some(mask)
    };

    let elements = match &invocation_mask {
        Some(mask) => elements.filter(mask.clone())?,
        None => elements,
    };
    let parents = match &invocation_mask {
        Some(mask) => parents.filter(mask.clone())?,
        None => parents,
    };
    let mut inputs = vec![elements];
    if template_uses_slot(&body, 1)? {
        let local = local_indices(sizes, invocation_count)?;
        inputs.push(match &invocation_mask {
            Some(mask) => local.filter(mask.clone())?,
            None => local,
        });
    } else {
        // Capture slots begin at 2 even for a one-parameter lambda. This unused placeholder
        // preserves that structural numbering without creating a local-index sequence.
        inputs.push(ConstantArray::new(0_u64, inputs[0].len()).into_array());
    }
    inputs.extend(
        captures
            .into_iter()
            .map(|capture| capture.take(parents.clone()))
            .collect::<VortexResult<Vec<_>>>()?,
    );

    let transformed = match template_scope(&body)? {
        Some(scope) => instantiate(&body, scope, &inputs)?,
        None => instantiate_constant_body(&body, inputs.first().map_or(0, ArrayRef::len))?,
    };
    match invocation_mask {
        Some(mask) => scatter_valid_invocations(transformed, &mask, body.dtype()),
        None => Ok(transformed),
    }
}

fn template_uses_slot(body: &ArrayRef, slot: usize) -> VortexResult<bool> {
    fn visit(body: &ArrayRef, slot: usize, found: &mut bool) -> VortexResult<()> {
        if let Some(input) = body.as_opt::<crate::arrays::TemplateInput>() {
            *found |= input.slot() == slot;
            return Ok(());
        }
        if let Some(scalar) = body.as_opt::<crate::arrays::ScalarFn>() {
            for child in scalar.iter_children() {
                visit(child, slot, found)?;
            }
        } else if let Some(transform) = body.as_opt::<ListTransform>() {
            visit(transform.list(), slot, found)?;
            for capture in transform.captures() {
                visit(capture, slot, found)?;
            }
        }
        Ok(())
    }
    let mut found = false;
    visit(body, slot, &mut found)?;
    Ok(found)
}

fn instantiate_constant_body(body: &ArrayRef, len: usize) -> VortexResult<ArrayRef> {
    if let Some(constant) = body.as_opt::<crate::arrays::Constant>() {
        return Ok(ConstantArray::new(constant.scalar().clone(), len).into_array());
    }
    // A constant-only scalar tree still needs the invocation length propagated through every
    // scalar-function node. The dummy input supplies that length; it cannot be observed because
    // this body has no template inputs.
    let invocation = ConstantArray::new(0_u8, len).into_array();
    instantiate(
        body,
        crate::arrays::TemplateScope::fresh(),
        std::slice::from_ref(&invocation),
    )
}

fn parent_indices(sizes: ArrayRef, element_count: usize) -> VortexResult<ArrayRef> {
    let parents = sizes.len();
    let dtype = DType::Primitive(
        sizes.dtype().as_ptype().to_unsigned(),
        Nullability::NonNullable,
    );
    let sizes = sizes.cast(dtype)?;
    let starts = PrimitiveArray::from_iter((0..parents).map(|index| index as u64)).into_array();
    let multipliers = ConstantArray::new(0_u64, parents).into_array();
    PiecewiseSequenceArray::try_new(starts, sizes, multipliers, element_count)
        .map(IntoArray::into_array)
}

fn local_indices(sizes: ArrayRef, element_count: usize) -> VortexResult<ArrayRef> {
    let parents = sizes.len();
    let dtype = DType::Primitive(
        sizes.dtype().as_ptype().to_unsigned(),
        Nullability::NonNullable,
    );
    let sizes = sizes.cast(dtype)?;
    let starts = ConstantArray::new(0_u64, parents).into_array();
    let multipliers = ConstantArray::new(1_u64, parents).into_array();
    PiecewiseSequenceArray::try_new(starts, sizes, multipliers, element_count)
        .map(IntoArray::into_array)
}

fn scatter_valid_invocations(
    transformed: ArrayRef,
    mask: &Mask,
    dtype: &DType,
) -> VortexResult<ArrayRef> {
    vortex_ensure!(
        transformed.len() == mask.true_count(),
        "list_transform() produced {} visible invocations, expected {}",
        transformed.len(),
        mask.true_count()
    );
    let mut arrays = BufferMut::<u8>::with_capacity(mask.len());
    let mut rows = BufferMut::<u64>::with_capacity(mask.len());
    let mut visible = 0_u64;
    for valid in mask.iter() {
        arrays.push(if valid { 0 } else { 1 });
        rows.push(if valid {
            let row = visible;
            visible += 1;
            row
        } else {
            0
        });
    }
    let placeholder = if dtype.is_nullable() {
        Scalar::null(dtype.clone())
    } else {
        Scalar::zero_value(dtype)
    };
    InterleaveArray::try_new(
        vec![transformed, ConstantArray::new(placeholder, 1).into_array()],
        arrays.into_array(),
        rows.into_array(),
    )
    .map(IntoArray::into_array)
}

struct AnyList;
impl Matcher for AnyList {
    type Match<'a> = ();

    fn try_match(array: &ArrayRef) -> Option<Self::Match<'_>> {
        (array.is::<List>() || array.is::<ListView>() || array.is::<FixedSizeList>()).then_some(())
    }
}
