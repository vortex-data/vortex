// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use super::merge_typed_scalar_as_variant;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::Array;
use crate::array::ArrayView;
use crate::arrays::ChunkedArray;
use crate::arrays::ConstantArray;
use crate::arrays::Struct;
use crate::arrays::Variant;
use crate::arrays::VariantArray;
use crate::arrays::scalar_fn::ExactScalarFn;
use crate::arrays::scalar_fn::ScalarFnArrayView;
use crate::arrays::scalar_fn::ScalarFnFactoryExt;
use crate::arrays::struct_::StructArrayExt;
use crate::arrays::variant::VariantArrayExt;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::kernel::ExecuteParentKernel;
use crate::kernel::ParentKernelSet;
use crate::scalar_fn::fns::variant_get::VariantGet;
use crate::scalar_fn::fns::variant_get::VariantGetOptions;
use crate::scalar_fn::fns::variant_get::VariantPath;
use crate::scalar_fn::fns::variant_get::VariantPathElement;

pub(super) const PARENT_KERNELS: ParentKernelSet<Variant> =
    ParentKernelSet::new(&[ParentKernelSet::lift(&VariantGetKernel)]);

#[derive(Default, Debug)]
struct VariantGetKernel;

impl ExecuteParentKernel<Variant> for VariantGetKernel {
    type Parent = ExactScalarFn<VariantGet>;

    fn execute_parent(
        &self,
        array: ArrayView<'_, Variant>,
        parent: ScalarFnArrayView<'_, VariantGet>,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        // This kernel only handles VariantGet over the input child. If the canonical
        // core storage is itself a VariantArray, let normal execution unwrap that layer.
        if child_idx != 0 || array.core_storage().is::<Variant>() {
            return Ok(None);
        }

        // Raw core storage is the authoritative fallback for paths that are not
        // perfectly represented by the canonical shredded tree.
        let core_validity = array.core_storage().validity()?;
        let make_fallback = |ctx: &mut ExecutionCtx| {
            execute_fallback_variant_get(
                array.len(),
                parent.options.clone(),
                array.core_storage().clone(),
                ctx,
            )
        };

        // Canonical shredded storage is a logical typed tree. We can only walk object
        // fields here; list indexes and missing fields must fall back to core storage.
        let typed = array
            .shredded()
            .map(|shredded| {
                typed_shredded_path(shredded, parent.options.path().elements(), ctx)?
                    .map(|typed| {
                        let len = typed.len();
                        typed.mask(core_validity.to_array(len))
                    })
                    .transpose()
            })
            .transpose()?
            .flatten();

        let Some(typed) = typed else {
            return make_fallback(ctx).map(Some);
        };
        // A shredded Variant child still needs VariantGet at the root to produce a
        // concrete requested dtype.
        if typed.dtype().is_variant()
            && parent
                .options
                .dtype()
                .is_some_and(|dtype| !dtype.is_variant())
        {
            return execute_fallback_variant_get(
                array.len(),
                VariantGetOptions::new(VariantPath::root(), parent.options.dtype().cloned()),
                typed,
                ctx,
            )
            .map(Some);
        }
        // Untyped VariantGet must return variant scalars, so typed shredded values
        // are wrapped as variants and merged with raw object fallback where needed.
        if parent.options.dtype().is_none_or(DType::is_variant) {
            let fallback = match typed.dtype() {
                DType::Struct(..) => Some(make_fallback(ctx)?),
                DType::List(..) | DType::FixedSizeList(..) => {
                    return make_fallback(ctx).map(Some);
                }
                _ => {
                    let typed_mask = typed.validity()?.execute_mask(typed.len(), ctx)?;
                    (!typed_mask.all_true())
                        .then(|| make_fallback(ctx))
                        .transpose()?
                }
            };
            return merge_typed_as_variant(typed, fallback, ctx).map(Some);
        }

        // For concrete output dtypes, trust the shredded child only when its logical
        // dtype matches the request; otherwise the raw fallback owns cast semantics.
        let requested_dtype = parent
            .options
            .dtype()
            .vortex_expect("variant dtype handled above");
        if typed.dtype().as_nullable() != requested_dtype.as_nullable() {
            return make_fallback(ctx).map(Some);
        }

        let typed = typed.cast(parent.dtype().clone())?;
        let typed_mask = typed.validity()?.execute_mask(typed.len(), ctx)?;
        if typed_mask.all_true() {
            return Ok(Some(typed));
        }

        // Null typed rows are not necessarily missing from the logical variant value;
        // fill those rows from core storage and keep valid typed rows unchanged.
        // TODO: we are computing the entire fallback array here but only take indices where
        // typed_mask is false, so we can narrow the core_storage to only the false
        // indices and compute from there then zip
        let fallback = make_fallback(ctx)?;
        typed_mask
            .into_array()
            .zip(typed, fallback)?
            .execute::<ArrayRef>(ctx)
            .map(Some)
    }
}

fn typed_shredded_path(
    shredded: &ArrayRef,
    path: &[VariantPathElement],
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ArrayRef>> {
    let mut current = shredded.clone();
    for element in path {
        let VariantPathElement::Field(name) = element else {
            return Ok(None);
        };
        let DType::Struct(..) = current.dtype() else {
            return Ok(None);
        };
        let current_struct = current.execute::<Array<Struct>>(ctx)?;
        let Some(field) = current_struct.unmasked_field_by_name_opt(name.as_ref()) else {
            return Ok(None);
        };
        let len = current_struct.len();
        let current_validity = current_struct.validity()?.to_array(len);

        current = field.clone().mask(current_validity.clone())?;
    }

    Ok(Some(current))
}

fn merge_typed_as_variant(
    typed: ArrayRef,
    fallback: Option<ArrayRef>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let dtype = DType::Variant(Nullability::Nullable);
    // TODO(variant): replace this with a Variant builder once one exists.
    // Chunked<Variant> canonicalizes to VariantArray, so this row-wise fallback is safe.
    let mut chunks = Vec::with_capacity(typed.len());

    for idx in 0..typed.len() {
        let typed_scalar = typed.execute_scalar(idx, ctx)?;
        let fallback_scalar = fallback
            .as_ref()
            .map(|fallback| fallback.execute_scalar(idx, ctx))
            .transpose()?;
        let scalar = merge_typed_scalar_as_variant(typed_scalar, fallback_scalar, &dtype)?;

        chunks.push(ConstantArray::new(scalar, 1).into_array());
    }

    let core_storage = ChunkedArray::try_new(chunks, dtype)?.into_array();
    VariantArray::try_new(core_storage, None).map(|array| array.into_array())
}

fn execute_fallback_variant_get(
    len: usize,
    options: VariantGetOptions,
    core_storage: ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    VariantGet
        .try_new_array(len, options, [core_storage])?
        .execute::<ArrayRef>(ctx)
}
