// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use super::merge_typed_scalar_as_variant;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::Array;
use crate::array::ArrayView;
use crate::arrays::ChunkedArray;
use crate::arrays::ConstantArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::Struct;
use crate::arrays::Variant;
use crate::arrays::VariantArray;
use crate::arrays::scalar_fn::ExactScalarFn;
use crate::arrays::scalar_fn::ScalarFnArrayView;
use crate::arrays::struct_::StructArrayExt;
use crate::arrays::variant::VariantArraySlotsExt;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::kernel::ExecuteParentKernel;
use crate::optimizer::kernels::ArrayKernelsExt;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::fns::variant_get::VariantGet;
use crate::scalar_fn::fns::variant_get::VariantGetOptions;
use crate::scalar_fn::fns::variant_get::VariantPath;
use crate::scalar_fn::fns::variant_get::VariantPathElement;

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
        let resolved = array
            .shredded()
            .map(|shredded| typed_shredded_path(shredded, parent.options.path().elements(), ctx))
            .transpose()?
            .flatten();

        let Some(ShreddedLookup {
            node,
            rest,
            mut validities,
        }) = resolved
        else {
            return make_fallback(ctx).map(Some);
        };
        if !core_validity.definitely_no_nulls() {
            validities.push(core_validity.to_array(node.len()));
        }

        let requests_concrete = parent
            .options
            .dtype()
            .is_some_and(|dtype| !dtype.is_variant());
        let typed = if !rest.is_empty() {
            // The walk stopped at a partially shredded (nested Variant) field. That field carries
            // its own core storage and shredded tree, so it resolves the rest of the path itself.
            execute_fallback_variant_get(
                VariantGetOptions::new(
                    VariantPath::new(rest.iter().cloned()),
                    parent.options.dtype().cloned(),
                ),
                node,
                ctx,
            )?
        } else if node.dtype().is_variant() && requests_concrete {
            // A shredded Variant child still needs VariantGet at the root to produce a
            // concrete requested dtype.
            execute_fallback_variant_get(
                VariantGetOptions::new(VariantPath::root(), parent.options.dtype().cloned()),
                node,
                ctx,
            )?
        } else {
            node
        };
        // Null ancestors make the path null. Masking only after the nested lookups keeps those
        // lookups on the unwrapped nested Variant, where their kernels apply.
        let typed = validities
            .into_iter()
            .try_fold(typed, |typed, validity| typed.mask(validity))?;

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

        // Only the null typed rows need the fallback, so evaluate it over just those rows of
        // core storage. When none of them hold the path, the typed nulls are already correct.
        let null_rows = !typed_mask.clone();
        // Execute the filter before VariantGet so the core storage encoding's own VariantGet
        // kernel applies, rather than row-by-row evaluation through the lazy filter.
        let null_core = array
            .core_storage()
            .filter(null_rows.clone())?
            .execute::<ArrayRef>(ctx)?;
        let fallback = execute_fallback_variant_get(parent.options.clone(), null_core, ctx)?;
        if fallback.all_invalid(ctx)? {
            return Ok(Some(typed));
        }

        // Scatter the fallback rows back into place: null typed row `i` takes the fallback value
        // at its position among the null rows, valid typed rows take a null index.
        let mut next = 0u64;
        let scatter =
            PrimitiveArray::from_option_iter(null_rows.to_bit_buffer().iter().map(|is_null_row| {
                is_null_row.then(|| {
                    let index = next;
                    next += 1;
                    index
                })
            }));
        let fallback = fallback.take(scatter.into_array())?;
        typed_mask
            .into_array()
            .zip(typed, fallback)?
            .execute::<ArrayRef>(ctx)
            .map(Some)
    }
}

/// The shredded node a path walk reached. See [`typed_shredded_path`].
struct ShreddedLookup<'a> {
    /// The node the walk reached, not yet masked by the validity of its ancestors.
    node: ArrayRef,
    /// The part of the path `node` still has to resolve.
    rest: &'a [VariantPathElement],
    /// The validity of every ancestor struct that can hold nulls.
    validities: Vec<ArrayRef>,
}

/// Walks `path` through the canonical shredded tree.
///
/// The walk stops early at a partially shredded field, which is itself a Variant whose own storage
/// resolves the rest of the path. Returns `None` when the path is not represented in shredded
/// storage.
fn typed_shredded_path<'a>(
    shredded: &ArrayRef,
    path: &'a [VariantPathElement],
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ShreddedLookup<'a>>> {
    let mut current = shredded.clone();
    let mut validities = Vec::new();
    for (idx, element) in path.iter().enumerate() {
        let VariantPathElement::Field(name) = element else {
            return Ok(None);
        };
        match current.dtype() {
            DType::Struct(..) => {}
            DType::Variant(_) => {
                return Ok(Some(ShreddedLookup {
                    node: current,
                    rest: &path[idx..],
                    validities,
                }));
            }
            _ => return Ok(None),
        }
        let current_struct = current.execute::<Array<Struct>>(ctx)?;
        let Some(field) = current_struct.unmasked_field_by_name_opt(name.as_ref()) else {
            return Ok(None);
        };
        let validity = current_struct.validity()?;
        if !validity.definitely_no_nulls() {
            validities.push(validity.to_array(current_struct.len()));
        }
        current = field.clone();
    }

    Ok(Some(ShreddedLookup {
        node: current,
        rest: &[],
        validities,
    }))
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
    options: VariantGetOptions,
    core_storage: ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    VariantGet::try_new(core_storage, options)?
        .into_array()
        .execute::<ArrayRef>(ctx)
}
