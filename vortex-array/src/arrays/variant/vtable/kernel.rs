// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::Buffer;
use vortex_error::VortexError;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use super::merge_typed_scalar_as_variant;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::Array;
use crate::array::ArrayView;
use crate::arrays::Struct;
use crate::arrays::Variant;
use crate::arrays::scalar_fn::ExactScalarFn;
use crate::arrays::scalar_fn::ScalarFnArrayView;
use crate::arrays::struct_::StructArrayExt;
use crate::arrays::variant::VariantArraySlotsExt;
use crate::builders::VariantBuilder;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::kernel::ExecuteParentKernel;
use crate::optimizer::ArrayOptimizer;
use crate::optimizer::kernels::ArrayKernelsExt;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::fns::variant_get::VariantGet;
use crate::scalar_fn::fns::variant_get::VariantGetOptions;
use crate::scalar_fn::fns::variant_get::VariantPath;
use crate::scalar_fn::fns::variant_get::VariantPathElement;
use crate::validity::Validity;

pub(crate) fn initialize(session: &VortexSession) {
    let kernels = session.kernels();
    kernels.register_execute_parent_kernel(VariantGet.id(), Variant, VariantGetKernel);
}

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
            execute_fallback_variant_get(parent.options.clone(), array.core_storage().clone(), ctx)
        };

        // Canonical shredded storage is a logical typed tree. We can only walk object
        // fields here; list indexes and missing fields must fall back to core storage.
        let typed = array
            .shredded()
            .map(|shredded| {
                typed_shredded_path(shredded, parent.options.path().elements(), ctx)?
                    .map(|(typed, remaining)| {
                        let typed = mask_with_validity(typed, &core_validity)?.optimize_ctx(ctx)?;
                        Ok::<_, VortexError>((typed, remaining))
                    })
                    .transpose()
            })
            .transpose()?
            .flatten();

        let Some((typed, remaining)) = typed else {
            return make_fallback(ctx).map(Some);
        };
        // A shredded Variant child holds the rest of the path, in its own shredded tree or in
        // its residual value. It also still needs VariantGet at its root to produce a concrete
        // requested dtype.
        if typed.dtype().is_variant()
            && (!remaining.is_empty()
                || parent
                    .options
                    .dtype()
                    .is_some_and(|dtype| !dtype.is_variant()))
        {
            return execute_fallback_variant_get(
                VariantGetOptions::new(
                    VariantPath::new(remaining.iter().cloned()),
                    parent.options.dtype().cloned(),
                ),
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
        if typed_mask.all_false() {
            return make_fallback(ctx).map(Some);
        }
        // Usually the null typed rows are missing from core storage as well, as for a field
        // that only some objects have. Extract from just those rows first: when none of them
        // holds a value, the typed child is the whole answer and the rest of core storage is
        // never read.
        let null_rows = !&typed_mask;
        let null_rows_fallback = execute_fallback_variant_get(
            parent.options.clone(),
            array.core_storage().filter(null_rows)?,
            ctx,
        )?;
        if null_rows_fallback
            .validity()?
            .execute_mask(null_rows_fallback.len(), ctx)?
            .all_false()
        {
            return Ok(Some(typed));
        }

        // Scatter the extracted rows back to their positions: row `i` reads the extracted row
        // that it is the n-th null typed row for. Valid typed rows read an arbitrary extracted
        // row, which the zip discards.
        let mut extracted = 0u64;
        let positions = typed_mask
            .to_bit_buffer()
            .iter()
            .map(|valid| {
                let position = extracted;
                extracted += u64::from(!valid);
                position.min(null_rows_fallback.len() as u64 - 1)
            })
            .collect::<Buffer<u64>>();
        let fallback = null_rows_fallback.take(positions.into_array())?;
        typed_mask
            .into_array()
            .zip(typed, fallback)?
            .execute::<ArrayRef>(ctx)
            .map(Some)
    }
}

/// Nulls out the rows of `array` that `validity` marks invalid.
///
/// A validity without nulls leaves `array` as it is: wrapping it in a mask would hide its
/// encoding from the kernels that extract from it.
fn mask_with_validity(array: ArrayRef, validity: &Validity) -> VortexResult<ArrayRef> {
    match validity {
        Validity::NonNullable | Validity::AllValid => Ok(array),
        validity => {
            let len = array.len();
            array.mask(validity.to_array(len))
        }
    }
}

/// Walks `path` through the shredded tree, returning the array it reaches and the elements of
/// `path` that are left.
///
/// The walk stops early at a shredded Variant child, which holds the rest of the path itself.
/// Returns `None` if the shredded tree does not represent the path.
fn typed_shredded_path<'a>(
    shredded: &ArrayRef,
    path: &'a [VariantPathElement],
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<(ArrayRef, &'a [VariantPathElement])>> {
    let mut current = shredded.clone();
    for (idx, element) in path.iter().enumerate() {
        if current.dtype().is_variant() {
            return Ok(Some((current, &path[idx..])));
        }
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
        current = mask_with_validity(field.clone(), &current_struct.validity()?)?;
    }

    Ok(Some((current, &[])))
}

fn merge_typed_as_variant(
    typed: ArrayRef,
    fallback: Option<ArrayRef>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let dtype = DType::Variant(Nullability::Nullable);
    let mut builder = VariantBuilder::new(Nullability::Nullable);
    for idx in 0..typed.len() {
        let typed_scalar = typed.execute_scalar(idx, ctx)?;
        let fallback_scalar = fallback
            .as_ref()
            .map(|fallback| fallback.execute_scalar(idx, ctx))
            .transpose()?;
        let scalar = merge_typed_scalar_as_variant(typed_scalar, fallback_scalar, &dtype)?;
        builder.append_n_scalars(&scalar, 1)?;
    }
    builder.finish_into_variant().map(IntoArray::into_array)
}

fn execute_fallback_variant_get(
    options: VariantGetOptions,
    core_storage: ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    VariantGet::try_new(core_storage, options)?
        .into_array()
        .execute::<ArrayRef>(ctx)
}
