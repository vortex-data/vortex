// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::Buffer;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use super::merge_typed_scalar_as_variant;
use crate::ArrayRef;
use crate::Canonical;
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
use crate::builders::builder_with_capacity_in;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::kernel::ExecuteParentKernel;
use crate::optimizer::ArrayOptimizer;
use crate::optimizer::kernels::ArrayKernelsExt;
use crate::scalar::Scalar;
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

        // Null typed rows are not necessarily missing from the logical variant value;
        // fill those rows from core storage and keep valid typed rows unchanged.
        let typed_mask = typed.validity()?.execute_mask(typed.len(), ctx)?;
        if typed_mask.all_false() {
            return make_fallback(ctx).map(Some);
        }

        // The core storage holds no value of a row that the typed child holds, so a typed value
        // of another dtype than the requested one converts from the typed child.
        let typed = convert_typed(typed, parent.dtype(), ctx)?;
        if typed_mask.all_true() {
            return Ok(Some(typed));
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

/// Converts the typed values `typed` to `dtype`, as a single value converts: a value that does
/// not convert is a null.
fn convert_typed(typed: ArrayRef, dtype: &DType, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
    // Only the nullability differs, which keeps the encoding of the typed values.
    if typed.dtype().eq_ignore_nullability(dtype) {
        return typed.cast(dtype.clone());
    }

    // Most columns convert as a whole. A cast fails on the first value that does not convert,
    // and only then is each value converted on its own.
    if let Ok(converted) = typed
        .cast(dtype.clone())
        .and_then(|converted| converted.execute::<Canonical>(ctx))
    {
        return Ok(converted.into_array());
    }

    let mut builder = builder_with_capacity_in(dtype, typed.len(), ctx.allocator());
    for idx in 0..typed.len() {
        let scalar = typed.execute_scalar(idx, ctx)?;
        let converted = scalar
            .cast(dtype)
            .unwrap_or_else(|_| Scalar::null(dtype.clone()));
        builder.append_scalar(&converted)?;
    }
    Ok(builder.finish())
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

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::ArrayRef;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::arrays::ConstantArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::StructArray;
    use crate::arrays::VariantArray;
    use crate::assert_arrays_eq;
    use crate::dtype::DType;
    use crate::dtype::FieldNames;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::expr::root;
    use crate::expr::variant_get;
    use crate::scalar::Scalar;
    use crate::scalar_fn::fns::variant_get::VariantPath;
    use crate::validity::Validity;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    /// Variant objects whose field `a` is shredded as `a`, over core storage that holds no value
    /// of `a`, as shredding leaves it.
    fn shredded_a(a: ArrayRef) -> VortexResult<ArrayRef> {
        let len = a.len();
        let core = ConstantArray::new(Scalar::variant(Scalar::null(DType::Null)), len).into_array();
        let shredded = StructArray::try_new(
            FieldNames::from_iter(["a"]),
            vec![a],
            len,
            Validity::NonNullable,
        )?
        .into_array();
        Ok(VariantArray::try_new(core, Some(shredded))?.into_array())
    }

    /// A typed value of another dtype than the requested one converts from the typed child, as
    /// a single value converts. A value that does not convert is a null.
    #[rstest]
    #[case::narrower(
        PrimitiveArray::from_option_iter([Some(1i64), Some(-300), None]),
        PType::I32,
        PrimitiveArray::from_option_iter([Some(1i32), Some(-300), None]),
    )]
    #[case::wider(
        PrimitiveArray::from_option_iter([Some(1i32), Some(-300), None]),
        PType::I64,
        PrimitiveArray::from_option_iter([Some(1i64), Some(-300), None]),
    )]
    #[case::out_of_range(
        PrimitiveArray::from_option_iter([Some(1i64), Some(-300), None]),
        PType::I8,
        PrimitiveArray::from_option_iter([Some(1i8), None, None]),
    )]
    #[case::to_float(
        PrimitiveArray::from_option_iter([Some(1i64), Some(-300), None]),
        PType::F64,
        PrimitiveArray::from_option_iter([Some(1f64), Some(-300.0), None]),
    )]
    fn converts_typed_values(
        #[case] typed: PrimitiveArray,
        #[case] requested: PType,
        #[case] expected: PrimitiveArray,
    ) -> VortexResult<()> {
        let array = shredded_a(typed.into_array())?;
        let expr = variant_get(
            root(),
            VariantPath::field("a"),
            Some(DType::Primitive(requested, Nullability::Nullable)),
        );

        let mut ctx = SESSION.create_execution_ctx();
        let actual = array.apply(&expr)?.execute::<ArrayRef>(&mut ctx)?;

        assert_arrays_eq!(actual, expected.into_array(), &mut ctx);
        Ok(())
    }
}
