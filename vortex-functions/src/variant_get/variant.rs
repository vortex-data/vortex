// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::Array;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::FixedSizeList;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::ListView;
use vortex_array::arrays::ListViewArray;
use vortex_array::arrays::Struct;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::Variant;
use vortex_array::arrays::fixed_size_list::FixedSizeListArrayExt;
use vortex_array::arrays::fixed_size_list::FixedSizeListArraySlotsExt;
use vortex_array::arrays::listview::ListViewArrayExt;
use vortex_array::arrays::listview::ListViewArraySlotsExt;
use vortex_array::arrays::scalar_fn::ExactScalarFn;
use vortex_array::arrays::scalar_fn::ScalarFnArrayView;
use vortex_array::arrays::struct_::StructArrayExt;
use vortex_array::arrays::variant::VariantArraySlotsExt;
use vortex_array::arrays::variant::merge_typed_scalar_as_variant;
use vortex_array::builders::VariantBuilder;
use vortex_array::builders::builder_with_capacity_in;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::kernel::ExecuteParentKernel;
use vortex_array::optimizer::ArrayOptimizer;
use vortex_array::optimizer::kernels::ArrayKernelsExt;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use crate::builtins::ArrayFunctions;
use crate::variant_get::VariantGet;
use crate::variant_get::VariantGetOptions;
use crate::variant_get::VariantPath;
use crate::variant_get::VariantPathElement;
use crate::variant_get::convert_to_bool;
use crate::variant_get::convert_to_decimal;

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
            let typed_mask = typed.validity()?.execute_mask(typed.len(), ctx)?;
            if typed_mask.all_false() {
                return make_fallback(ctx).map(Some);
            }

            let nested = execute_fallback_variant_get(
                VariantGetOptions::new(
                    VariantPath::new(remaining.iter().cloned()),
                    parent.options.dtype().cloned(),
                ),
                typed,
                ctx,
            )?;

            // A row without the nested Variant can still hold the path in core storage, as the
            // rows of an unshredded chunk do once canonicalization pads the shredded child.
            return fill_absent_from_core(
                nested,
                typed_mask,
                array.core_storage(),
                parent.options,
                ctx,
            )
            .map(Some);
        }
        // A remaining path through a shredded list cannot be walked column-wise; defer to
        // scalar extraction so shredded list elements and residual core storage are both read.
        if !remaining.is_empty() {
            return Ok(None);
        }
        // Untyped VariantGet must return variant scalars, so typed shredded values
        // are wrapped as variants and merged with raw object fallback where needed.
        if parent.options.dtype().is_none_or(DType::is_variant) {
            let fallback = match typed.dtype() {
                DType::Struct(..) => Some(make_fallback(ctx)?),
                DType::List(..) | DType::FixedSizeList(..) => {
                    return Ok(None);
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

        // A shredded struct or list can hold only some of an object's fields or nested
        // partially-shredded values, with the rest in core storage. A row is read from the
        // shredded child only when every leaf of it holds a value there. Scalar extraction
        // merges shredded and core storage for the other rows before it casts them.
        if matches!(
            typed.dtype(),
            DType::Struct(..) | DType::List(..) | DType::FixedSizeList(..)
        ) {
            let Some((typed, complete)) = complete_rows(&typed, ctx)? else {
                return Ok(None);
            };

            if typed.dtype().as_nullable() != parent.dtype().as_nullable() {
                return Ok(None);
            }

            // A null Variant row is null in the result, so the shredded child gives it as well.
            let core_mask = core_validity.execute_mask(typed.len(), ctx)?;
            let complete = &complete | &!&core_mask;
            if complete.all_false() {
                return Ok(None);
            }

            let typed = typed.cast(parent.dtype().clone())?;
            if complete.all_true() {
                return Ok(Some(typed));
            }

            let incomplete = execute_fallback_variant_get(
                parent.options.clone(),
                array.array().filter(!&complete)?,
                ctx,
            )?;
            return scatter_absent(typed, complete, incomplete, ctx).map(Some);
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
        fill_absent_from_core(typed, typed_mask, array.core_storage(), parent.options, ctx)
            .map(Some)
    }
}

/// Fills the rows of `extracted` that `present` marks absent from the shredded child, with the
/// extraction of the whole path from `core`.
///
/// Usually the absent rows are missing from core storage as well, as for a field that only some
/// objects have. These rows are extracted on their own first: when none of them holds a value,
/// `extracted` is the whole answer and the rest of core storage is never read.
fn fill_absent_from_core(
    extracted: ArrayRef,
    present: Mask,
    core: &ArrayRef,
    options: &VariantGetOptions,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    if present.all_true() {
        return Ok(extracted);
    }

    let absent = execute_fallback_variant_get(options.clone(), core.filter(!&present)?, ctx)?;
    if absent
        .validity()?
        .execute_mask(absent.len(), ctx)?
        .all_false()
    {
        return Ok(extracted);
    }

    scatter_absent(extracted, present, absent, ctx)
}

/// Puts the rows of `absent` in the positions that `present` marks absent, and keeps the rows of
/// `extracted` in the other positions.
fn scatter_absent(
    extracted: ArrayRef,
    present: Mask,
    absent: ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    // Scatter the extracted rows back to their positions: row `i` reads the extracted row that
    // it is the n-th absent row for. Present rows read an arbitrary extracted row, which the zip
    // discards.
    let last = absent.len() as u64 - 1;
    let mut taken = 0u64;
    let positions = present
        .to_bit_buffer()
        .iter()
        .map(|is_present| {
            let position = taken;
            taken += u64::from(!is_present);
            position.min(last)
        })
        .collect::<Buffer<u64>>();
    let fallback = absent.take(positions.into_array())?;

    present
        .into_array()
        .zip(extracted, fallback)?
        .execute::<ArrayRef>(ctx)
}

/// Returns the shredded struct or list `typed` with each nested Variant replaced by its shredded
/// child, and the rows of it whose every leaf holds a value.
///
/// A null leaf can mean that the value of the leaf is in core storage, so such a row must merge
/// the two. Returns `None` if a row cannot be read from `typed` alone, such as a row with a
/// nested Variant that can keep a value in core storage, or a list with a null element.
fn complete_rows(
    typed: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<(ArrayRef, Mask)>> {
    let len = typed.len();
    match typed.dtype() {
        DType::Variant(_) => {
            let variant = typed.clone().execute::<Array<Variant>>(ctx)?;

            // The core storage holds no value of a row that holds a shredded value other than
            // an object, so such a row is complete if its shredded value is. An object can keep
            // some of its fields in core storage.
            let Some(shredded) = variant
                .shredded()
                .filter(|shredded| !shredded.dtype().is_struct())
            else {
                return Ok(None);
            };

            let shredded = mask_with_validity(shredded.clone(), &variant.validity()?)?;
            complete_rows(&shredded, ctx)
        }
        DType::Struct(..) => {
            let typed = typed.clone().execute::<Array<Struct>>(ctx)?;
            let validity = typed.struct_validity();
            let mut complete = validity.execute_mask(len, ctx)?;
            let mut fields = Vec::with_capacity(typed.names().len());
            for field in typed.iter_unmasked_fields() {
                let Some((field, field_complete)) = complete_rows(field, ctx)? else {
                    return Ok(None);
                };
                complete = &complete & &field_complete;
                fields.push(field);
            }

            let typed = StructArray::try_new(typed.names().clone(), fields, len, validity)?;
            Ok(Some((typed.into_array(), complete)))
        }
        DType::List(..) => {
            let typed = typed.clone().execute::<Array<ListView>>(ctx)?;
            let Some(elements) = complete_elements(typed.elements(), ctx)? else {
                return Ok(None);
            };

            let validity = typed.listview_validity();
            let complete = validity.execute_mask(len, ctx)?;
            let typed = ListViewArray::try_new(
                elements,
                typed.offsets().clone(),
                typed.sizes().clone(),
                validity,
                ctx,
            )?;
            Ok(Some((typed.into_array(), complete)))
        }
        DType::FixedSizeList(..) => {
            let typed = typed.clone().execute::<Array<FixedSizeList>>(ctx)?;
            let Some(elements) = complete_elements(typed.elements(), ctx)? else {
                return Ok(None);
            };

            let validity = typed.fixed_size_list_validity();
            let complete = validity.execute_mask(len, ctx)?;
            let typed = FixedSizeListArray::try_new(elements, typed.list_size(), validity, len)?;
            Ok(Some((typed.into_array(), complete)))
        }
        _ => {
            let complete = typed.validity()?.execute_mask(len, ctx)?;
            Ok(Some((typed.clone(), complete)))
        }
    }
}

/// Returns the elements of a list as [`complete_rows`] does, if all of them are complete.
///
/// The check covers all elements, also elements that no list refers to. This can find a list
/// incomplete that is complete, which only makes the extraction slower.
fn complete_elements(
    elements: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ArrayRef>> {
    let Some((elements, complete)) = complete_rows(elements, ctx)? else {
        return Ok(None);
    };
    Ok(complete.all_true().then_some(elements))
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
            .ok()
            .or_else(|| convert_to_bool(&scalar, dtype))
            .or_else(|| convert_to_decimal(&scalar, dtype))
            .unwrap_or_else(|| Scalar::null(dtype.clone()));
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
            return Ok(
                matches!(current.dtype(), DType::List(..) | DType::FixedSizeList(..))
                    .then_some((current, &path[idx..])),
            );
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
    use std::sync::Arc;
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::arrays::ListArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::StructArray;
    use vortex_array::arrays::VarBinViewArray;
    use vortex_array::arrays::VariantArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::builders::VariantBuilder;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::scalar::Scalar;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_expr::ArrayExprExt;
    use vortex_expr::root;
    use vortex_mask::Mask;
    use vortex_session::VortexSession;

    use crate::expr::variant_get;
    use crate::variant_get::VariantPath;
    use crate::variant_get::VariantPathElement;
    use crate::variant_get::tests::variant_object;
    use crate::variant_get::tests::variant_rows;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty();

        vortex_array::initialize(&session);

        crate::initialize(&session);

        session
    });

    fn int(value: i32) -> Scalar {
        Scalar::primitive(value, Nullability::NonNullable)
    }

    /// A Variant whose objects have the field `name` shredded as `typed`, over `core`.
    fn shredded(name: &str, typed: ArrayRef, core: ArrayRef) -> VortexResult<ArrayRef> {
        let shredded = StructArray::try_from_iter([(name, typed)])?.into_array();

        Ok(VariantArray::try_new(core, Some(shredded))?.into_array())
    }

    /// Variant objects whose field `a` is shredded as `a`, over core storage that holds no value
    /// of `a`, as shredding leaves it.
    fn shredded_a(a: ArrayRef) -> VortexResult<ArrayRef> {
        let len = a.len();
        let core = ConstantArray::new(Scalar::variant(Scalar::null(DType::Null)), len).into_array();

        shredded("a", a, core)
    }

    fn execute(array: ArrayRef, path: VariantPath, dtype: Option<DType>) -> VortexResult<ArrayRef> {
        let mut ctx = SESSION.create_execution_ctx();

        array
            .apply(&variant_get(root(), path, dtype))?
            .execute::<ArrayRef>(&mut ctx)
    }

    fn nullable_i32() -> DType {
        DType::Primitive(PType::I32, Nullability::Nullable)
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
        let actual = execute(
            array,
            VariantPath::field("a"),
            Some(DType::Primitive(requested, Nullability::Nullable)),
        )?;

        let mut ctx = SESSION.create_execution_ctx();
        assert_arrays_eq!(actual, expected.into_array(), &mut ctx);
        Ok(())
    }

    /// Numbers in a Variant, as a partially shredded field holds them.
    fn numbers_in_variants() -> VortexResult<ArrayRef> {
        let mut builder = VariantBuilder::new(Nullability::Nullable);
        for value in [0i64, -3] {
            let value = Scalar::primitive(value, Nullability::NonNullable);
            builder.append_n_scalars(&Scalar::variant(value), 1)?;
        }
        builder.append_n_scalars(&Scalar::null(DType::Variant(Nullability::Nullable)), 1)?;

        Ok(builder.finish_into_variant()?.into_array())
    }

    /// Numbers and strings read as Booleans as Arrow's Variant conversion reads them: zero is
    /// false, and a string must have one of the spellings that Arrow accepts. A shredded value
    /// gives what the unshredded value gives.
    #[rstest]
    #[case::numbers(
        Ok(PrimitiveArray::from_option_iter([Some(0i64), Some(-3), None]).into_array()),
        BoolArray::from_iter([Some(false), Some(true), None]),
    )]
    #[case::strings(
        Ok(VarBinViewArray::from_iter_nullable_str([
            Some("true"),
            Some(" No "),
            Some("1"),
            Some("maybe"),
            None,
        ])
        .into_array()),
        BoolArray::from_iter([Some(true), Some(false), Some(true), None, None]),
    )]
    #[case::numbers_in_variants(
        numbers_in_variants(),
        BoolArray::from_iter([Some(false), Some(true), None]),
    )]
    fn converts_typed_values_to_booleans(
        #[case] typed: VortexResult<ArrayRef>,
        #[case] expected: BoolArray,
    ) -> VortexResult<()> {
        let array = shredded_a(typed?)?;
        let actual = execute(
            array,
            VariantPath::field("a"),
            Some(DType::Bool(Nullability::Nullable)),
        )?;

        let mut ctx = SESSION.create_execution_ctx();
        assert_arrays_eq!(actual, expected.into_array(), &mut ctx);
        Ok(())
    }

    /// A row that the shredded child does not hold reads its value from core storage, and the
    /// other rows keep their typed values.
    #[test]
    fn fills_absent_rows_from_core() -> VortexResult<()> {
        let typed = PrimitiveArray::from_option_iter([Some(10i32), None, None]).into_array();
        let core = variant_rows([
            Scalar::variant(variant_object([])),
            Scalar::variant(variant_object([("a", int(5))])),
            Scalar::variant(variant_object([("b", int(6))])),
        ])?;
        let array = shredded("a", typed, core)?;

        let actual = execute(array, VariantPath::field("a"), Some(nullable_i32()))?;

        let mut ctx = SESSION.create_execution_ctx();
        assert_arrays_eq!(
            actual,
            PrimitiveArray::from_option_iter([Some(10i32), Some(5), None]).into_array(),
            &mut ctx
        );
        Ok(())
    }

    /// The input shapes of [`reads_sliced_and_filtered_shredded_rows`].
    #[derive(Clone, Copy, Debug)]
    enum Shape {
        Sliced,
        Filtered,
    }

    /// A slice or a filter of a shredded Variant keeps the typed rows and reads the other rows
    /// from core storage.
    #[rstest]
    #[case::sliced(Shape::Sliced, [Some(5i32), Some(30), Some(7)])]
    #[case::filtered(Shape::Filtered, [Some(10i32), Some(5), Some(7)])]
    fn reads_sliced_and_filtered_shredded_rows(
        #[case] shape: Shape,
        #[case] expected: [Option<i32>; 3],
    ) -> VortexResult<()> {
        let typed =
            PrimitiveArray::from_option_iter([Some(10i32), None, Some(30), None]).into_array();
        let core = variant_rows([
            Scalar::variant(variant_object([])),
            Scalar::variant(variant_object([("a", int(5))])),
            Scalar::variant(variant_object([])),
            Scalar::variant(variant_object([("a", int(7))])),
        ])?;
        let array = shredded("a", typed, core)?;
        let array = match shape {
            Shape::Sliced => array.slice(1..4)?,
            Shape::Filtered => array.filter(Mask::from_iter([true, true, false, true]))?,
        };

        let actual = execute(array, VariantPath::field("a"), Some(nullable_i32()))?;

        let mut ctx = SESSION.create_execution_ctx();
        assert_arrays_eq!(
            actual,
            PrimitiveArray::from_option_iter(expected).into_array(),
            &mut ctx
        );
        Ok(())
    }

    /// A path continues through a shredded Variant child. A row reads the leaf from the typed
    /// child of the nested Variant, from the core storage of the nested Variant, or from the top
    /// core storage when the nested Variant does not hold the row.
    #[test]
    fn reads_path_through_shredded_variant() -> VortexResult<()> {
        let nested_typed = PrimitiveArray::from_option_iter([Some(1i32), None, None]).into_array();
        let nested_core = variant_rows([
            Scalar::variant(variant_object([])),
            Scalar::variant(variant_object([("c", int(2))])),
            Scalar::null(DType::Variant(Nullability::Nullable)),
        ])?;
        let nested = shredded("c", nested_typed, nested_core)?;
        let core = variant_rows([
            Scalar::variant(variant_object([])),
            Scalar::variant(variant_object([])),
            Scalar::variant(variant_object([("b", variant_object([("c", int(3))]))])),
        ])?;
        let array = shredded("b", nested, core)?;

        let path = VariantPath::new([
            VariantPathElement::field("b"),
            VariantPathElement::field("c"),
        ]);
        let actual = execute(array, path, Some(nullable_i32()))?;

        let mut ctx = SESSION.create_execution_ctx();
        assert_arrays_eq!(
            actual,
            PrimitiveArray::from_option_iter([Some(1i32), Some(2), Some(3)]).into_array(),
            &mut ctx
        );
        Ok(())
    }

    /// Variant objects whose field `items` is shredded as a list of integers in the first row,
    /// and is only in core storage in the second row.
    fn shredded_items() -> VortexResult<ArrayRef> {
        let mut ctx = SESSION.create_execution_ctx();
        let items = ListArray::try_new(
            buffer![10i32, 20].into_array(),
            buffer![0u32, 2, 2].into_array(),
            Validity::from_iter([true, false]),
            &mut ctx,
        )?
        .into_array();
        let list = Scalar::list(
            DType::Variant(Nullability::NonNullable),
            vec![Scalar::variant(int(30)), Scalar::variant(int(40))],
            Nullability::NonNullable,
        );
        let core = variant_rows([
            Scalar::variant(variant_object([])),
            Scalar::variant(variant_object([("items", list)])),
        ])?;

        shredded("items", items, core)
    }

    /// An index into a shredded list reads the shredded elements and the elements in core
    /// storage, with and without a requested dtype.
    #[rstest]
    #[case::typed(Some(nullable_i32()))]
    #[case::untyped(None)]
    fn reads_index_into_shredded_list(#[case] dtype: Option<DType>) -> VortexResult<()> {
        let path = VariantPath::new([
            VariantPathElement::field("items"),
            VariantPathElement::index(1),
        ]);
        let untyped = dtype.is_none();

        let mut actual = execute(shredded_items()?, path, dtype)?;
        if untyped {
            actual = execute(actual, VariantPath::root(), Some(nullable_i32()))?;
        }

        let mut ctx = SESSION.create_execution_ctx();
        assert_arrays_eq!(
            actual,
            PrimitiveArray::from_option_iter([Some(20i32), Some(40)]).into_array(),
            &mut ctx
        );
        Ok(())
    }

    /// A shredded list of the requested dtype gives its rows in bulk, and a row that only core
    /// storage holds reads from there.
    #[test]
    fn reads_shredded_list() -> VortexResult<()> {
        let dtype = DType::List(
            Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
            Nullability::Nullable,
        );

        let actual = execute(shredded_items()?, VariantPath::field("items"), Some(dtype))?;

        let mut ctx = SESSION.create_execution_ctx();
        let expected = ListArray::try_new(
            buffer![10i32, 20, 30, 40].into_array(),
            buffer![0u32, 2, 4].into_array(),
            Validity::AllValid,
            &mut ctx,
        )?;
        assert_arrays_eq!(actual, expected.into_array(), &mut ctx);
        Ok(())
    }

    /// A shredded struct of the requested dtype gives the rows whose every field it holds. A row
    /// with a null field merges the shredded child with core storage, and a row whose whole
    /// struct is only in core storage reads from there.
    #[test]
    fn reads_shredded_struct() -> VortexResult<()> {
        let s = StructArray::try_from_iter_with_validity(
            [
                (
                    "x",
                    PrimitiveArray::from_option_iter([Some(1i32), Some(3), None, Some(7)]),
                ),
                (
                    "y",
                    PrimitiveArray::from_option_iter([Some(2i32), None, None, Some(8)]),
                ),
            ],
            Validity::from_iter([true, true, false, true]),
        )?
        .into_array();
        let dtype = s.dtype().clone();
        let core = variant_rows([
            Scalar::variant(variant_object([])),
            Scalar::variant(variant_object([])),
            Scalar::variant(variant_object([(
                "s",
                variant_object([("x", int(5)), ("y", int(6))]),
            )])),
            Scalar::variant(variant_object([])),
        ])?;
        let array = shredded("s", s, core)?;

        let actual = execute(array, VariantPath::field("s"), Some(dtype))?;

        let mut ctx = SESSION.create_execution_ctx();
        let expected = StructArray::try_from_iter_with_validity(
            [
                (
                    "x",
                    PrimitiveArray::from_option_iter([Some(1i32), Some(3), Some(5), Some(7)]),
                ),
                (
                    "y",
                    PrimitiveArray::from_option_iter([Some(2i32), None, Some(6), Some(8)]),
                ),
            ],
            Validity::AllValid,
        )?;
        assert_arrays_eq!(actual, expected.into_array(), &mut ctx);
        Ok(())
    }
}
