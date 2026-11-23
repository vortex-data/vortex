// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::Array;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::Struct;
use vortex_array::arrays::UnionArray;
use vortex_array::arrays::Variant;
use vortex_array::arrays::scalar_fn::ExactScalarFn;
use vortex_array::arrays::scalar_fn::ScalarFnArrayView;
use vortex_array::arrays::struct_::StructArrayExt;
use vortex_array::arrays::union::UnionArrayExt;
use vortex_array::arrays::union::UnionArraySlotsExt;
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
use vortex_buffer::BitBufferMut;
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
                        let typed =
                            mask_with_validity(typed, &core_validity, ctx)?.optimize_ctx(ctx)?;
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
        // partially-shredded values, with the rest in core storage. Defer to scalar
        // extraction, which merges shredded and core storage per row before casting.
        if matches!(
            typed.dtype(),
            DType::Struct(..) | DType::List(..) | DType::FixedSizeList(..)
        ) {
            return Ok(None);
        }

        // Null typed rows are not necessarily missing from the logical variant value;
        // fill those rows from core storage and keep valid typed rows unchanged.
        let typed_mask = typed.validity()?.execute_mask(typed.len(), ctx)?;
        if typed_mask.all_false() {
            return make_fallback(ctx).map(Some);
        }

        // The core storage holds no value of a row that the typed child holds, so a typed value
        // of another dtype than the requested one converts from the typed child.
        let typed = if typed.dtype().is_union() {
            convert_union(typed, parent.dtype(), ctx)?
        } else {
            convert_typed(typed, parent.dtype(), ctx)?
        };
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
            .unwrap_or_else(|| Scalar::null(dtype.clone()));
        builder.append_scalar(&converted)?;
    }
    Ok(builder.finish())
}

/// The rows of each member of the union `union`, and the member, in the order of its variants.
fn union_members(union: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Vec<(Mask, ArrayRef)>> {
    let len = union.len();
    let union = union.execute::<UnionArray>(ctx)?;
    let valid = union.type_ids().validity()?.execute_mask(len, ctx)?;
    let type_ids = union.type_ids().clone().execute::<PrimitiveArray>(ctx)?;
    let variants = union.variants();

    let mut rows = vec![BitBufferMut::new_unset(len); variants.len()];
    for (idx, type_id) in type_ids.as_slice::<u8>().iter().enumerate() {
        if valid.value(idx)
            && let Some(member) = variants.tag_to_child_index(*type_id)
        {
            rows[member].set(idx);
        }
    }

    Ok(rows
        .into_iter()
        .zip(union.iter_children().cloned())
        .map(|(rows, member)| (Mask::from(rows.freeze()), member))
        .collect())
}

/// Merges `members`, each with the rows it holds, into one array of `dtype`. A row that no member
/// holds is null.
fn merge_members(
    members: Vec<(Mask, ArrayRef)>,
    dtype: &DType,
    len: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let dtype = dtype.as_nullable();
    let mut merged = ConstantArray::new(Scalar::null(dtype.clone()), len).into_array();
    for (rows, member) in members {
        merged = rows.into_array().zip(member.cast(dtype.clone())?, merged)?;
    }
    merged.execute::<ArrayRef>(ctx)
}

/// The field `name` of the struct members of the union `union`.
///
/// A row of another member has no such field, and is null. If the members shred the field as
/// different dtypes, the field is the Variant of each value. Returns `None` if no struct member
/// shreds the field.
fn union_field(
    union: ArrayRef,
    name: &str,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ArrayRef>> {
    let len = union.len();
    let mut fields = Vec::new();
    for (rows, member) in union_members(union, ctx)? {
        if !member.dtype().is_struct() {
            continue;
        }

        let member = member.execute::<Array<Struct>>(ctx)?;
        if let Some(field) = member.unmasked_field_by_name_opt(name) {
            fields.push((
                rows,
                mask_with_validity(field.clone(), &member.validity()?, ctx)?,
            ));
        }
    }

    let Some(dtype) = fields.first().map(|(_, field)| field.dtype().as_nullable()) else {
        return Ok(None);
    };
    if fields
        .iter()
        .any(|(_, field)| !field.dtype().eq_ignore_nullability(&dtype))
    {
        return merge_members_as_variant(&fields, len, ctx).map(Some);
    }

    merge_members(fields, &dtype, len, ctx).map(Some)
}

/// Merges `members`, each with the rows it holds, into one array of the Variants of their
/// values. A row that no member holds is null.
fn merge_members_as_variant(
    members: &[(Mask, ArrayRef)],
    len: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let dtype = DType::Variant(Nullability::Nullable);
    let mut builder = VariantBuilder::new(Nullability::Nullable);
    for idx in 0..len {
        let value = match members.iter().find(|(rows, _)| rows.value(idx)) {
            Some((_, member)) => member.execute_scalar(idx, ctx)?,
            None => Scalar::null(dtype.clone()),
        };
        let value = merge_typed_scalar_as_variant(value, None, &dtype)?;
        builder.append_n_scalars(&value, 1)?;
    }
    builder.finish_into_variant().map(IntoArray::into_array)
}

/// Converts the values of the union `union` to `dtype`, each member as [`convert_typed`]
/// converts it.
fn convert_union(union: ArrayRef, dtype: &DType, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
    let len = union.len();
    let members = union_members(union, ctx)?
        .into_iter()
        .map(|(rows, member)| Ok((rows, convert_typed(member, dtype, ctx)?)))
        .collect::<VortexResult<Vec<_>>>()?;
    merge_members(members, dtype, len, ctx)
}

/// Nulls out the rows of `array` that `validity` marks invalid.
///
/// A validity without nulls leaves `array` as it is: wrapping it in a mask would hide its
/// encoding from the kernels that extract from it.
fn mask_with_validity(
    array: ArrayRef,
    validity: &Validity,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    match validity {
        Validity::NonNullable | Validity::AllValid => Ok(array),
        validity if array.dtype().is_union() => {
            // The validity of a union is the validity of its type ids.
            let len = array.len();
            let union = array.execute::<UnionArray>(ctx)?;
            let type_ids = union.type_ids().clone().mask(validity.to_array(len))?;
            let union = UnionArray::try_new(
                type_ids,
                union.variants().clone(),
                union.iter_children().cloned(),
            )?;
            Ok(union.into_array())
        }
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

        if current.dtype().is_union() {
            let Some(field) = union_field(current, name.as_ref(), ctx)? else {
                return Ok(None);
            };
            current = field;
            continue;
        }

        let DType::Struct(..) = current.dtype() else {
            return Ok(None);
        };
        let current_struct = current.execute::<Array<Struct>>(ctx)?;
        let Some(field) = current_struct.unmasked_field_by_name_opt(name.as_ref()) else {
            return Ok(None);
        };
        current = mask_with_validity(field.clone(), &current_struct.validity()?, ctx)?;
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
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::StructArray;
    use vortex_array::arrays::UnionArray;
    use vortex_array::arrays::VarBinViewArray;
    use vortex_array::arrays::VariantArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::builders::VariantBuilder;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::FieldNames;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::dtype::UnionVariants;
    use vortex_array::expr::root;
    use vortex_array::scalar::Scalar;
    use vortex_array::validity::Validity;
    use vortex_error::VortexResult;
    use vortex_error::vortex_err;
    use vortex_session::VortexSession;

    use crate::expr::variant_get;
    use crate::variant_get::VariantPath;
    use crate::variant_get::VariantPathElement;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty();
        crate::initialize(&session);
        session
    });

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

    /// A number reads as a Boolean as Arrow's Variant conversion reads it: zero is false.
    #[test]
    fn converts_typed_numbers_to_booleans() -> VortexResult<()> {
        let typed = PrimitiveArray::from_option_iter([Some(0i64), Some(-3), None]);
        let array = shredded_a(typed.into_array())?;
        let expr = variant_get(
            root(),
            VariantPath::field("a"),
            Some(DType::Bool(Nullability::Nullable)),
        );

        let mut ctx = SESSION.create_execution_ctx();
        let actual = array.apply(&expr)?.execute::<ArrayRef>(&mut ctx)?;

        assert_arrays_eq!(
            actual,
            BoolArray::from_iter([Some(false), Some(true), None]).into_array(),
            &mut ctx
        );
        Ok(())
    }

    /// The core storage of `len` rows that each hold `value`, or no value.
    fn core(value: Option<Scalar>, len: usize) -> ArrayRef {
        let value = value.unwrap_or_else(|| Scalar::null(DType::Null));
        ConstantArray::new(Scalar::variant(value), len).into_array()
    }

    /// The Variants of `core`, whose field `a` is shredded as `a`.
    fn shredded_over(core: ArrayRef, a: ArrayRef) -> VortexResult<ArrayRef> {
        let len = a.len();
        let shredded = StructArray::try_new(
            FieldNames::from_iter(["a"]),
            vec![a],
            len,
            Validity::NonNullable,
        )?
        .into_array();
        Ok(VariantArray::try_new(core, Some(shredded))?.into_array())
    }

    /// A union of an integer `int` and a string `str`, with the rows `1`, `"x"` and null.
    fn int_or_str() -> VortexResult<ArrayRef> {
        let variants = UnionVariants::new(
            ["int", "str"].into(),
            vec![
                DType::Primitive(PType::I64, Nullability::NonNullable),
                DType::Utf8(Nullability::NonNullable),
            ],
        )?;
        let type_ids = PrimitiveArray::from_option_iter([Some(0u8), Some(1), None]).into_array();
        let ints = PrimitiveArray::from_iter([1i64, 0, 0]).into_array();
        let strs = VarBinViewArray::from_iter_str(["", "x", ""]).into_array();
        Ok(UnionArray::try_new(type_ids, variants, [ints, strs])?.into_array())
    }

    #[rstest]
    #[case::member(
        DType::Primitive(PType::I64, Nullability::Nullable),
        PrimitiveArray::from_option_iter([Some(1i64), None, None]).into_array(),
    )]
    #[case::other_member(
        DType::Utf8(Nullability::Nullable),
        VarBinViewArray::from_iter_nullable_str([None, Some("x"), None]).into_array(),
    )]
    #[case::converted(
        DType::Primitive(PType::F64, Nullability::Nullable),
        PrimitiveArray::from_option_iter([Some(1f64), None, None]).into_array(),
    )]
    fn reads_a_typed_value_from_a_union(
        #[case] requested: DType,
        #[case] expected: ArrayRef,
    ) -> VortexResult<()> {
        let array = shredded_over(core(None, 3), int_or_str()?)?;
        let expr = variant_get(root(), VariantPath::field("a"), Some(requested));

        let mut ctx = SESSION.create_execution_ctx();
        let actual = array.apply(&expr)?.execute::<ArrayRef>(&mut ctx)?;

        assert_arrays_eq!(actual, expected, &mut ctx);
        Ok(())
    }

    /// A row that the union does not hold reads the core storage.
    #[test]
    fn reads_a_variant_from_a_union() -> VortexResult<()> {
        let fallback = Scalar::struct_(
            DType::struct_(
                [("a", DType::Primitive(PType::F64, Nullability::NonNullable))],
                Nullability::NonNullable,
            ),
            [Scalar::primitive(3.5f64, Nullability::NonNullable)],
        );
        let array = shredded_over(core(Some(fallback), 3), int_or_str()?)?;
        let expr = variant_get(root(), VariantPath::field("a"), None);

        let mut ctx = SESSION.create_execution_ctx();
        let actual = array.apply(&expr)?.execute::<ArrayRef>(&mut ctx)?;

        let expected = [
            Scalar::primitive(1i64, Nullability::NonNullable),
            Scalar::utf8("x", Nullability::NonNullable),
            Scalar::primitive(3.5f64, Nullability::NonNullable),
        ];
        for (idx, value) in expected.into_iter().enumerate() {
            let actual = actual.execute_scalar(idx, &mut ctx)?;
            assert_eq!(actual.as_variant().value(), Some(&value), "row {idx}");
        }
        Ok(())
    }

    /// A struct that holds a union reads as a Variant whose field holds the value of the member.
    #[test]
    fn reads_a_struct_with_a_union_as_a_variant() -> VortexResult<()> {
        let array = shredded_over(core(None, 3), int_or_str()?)?;
        let expr = variant_get(root(), VariantPath::root(), None);

        let mut ctx = SESSION.create_execution_ctx();
        let actual = array.apply(&expr)?.execute::<ArrayRef>(&mut ctx)?;

        let expected = [
            Some(Scalar::primitive(1i64, Nullability::NonNullable)),
            Some(Scalar::utf8("x", Nullability::NonNullable)),
            None,
        ];
        for (idx, value) in expected.into_iter().enumerate() {
            let object = actual.execute_scalar(idx, &mut ctx)?;
            let object = object
                .as_variant()
                .value()
                .ok_or_else(|| vortex_err!("row {idx} is not an object"))?
                .as_struct();
            let field = object
                .field("a")
                .and_then(|field| field.as_variant().value().cloned());
            assert_eq!(field, value, "row {idx}");
        }
        Ok(())
    }

    /// A field that two struct members shred as different dtypes reads the value of each.
    #[rstest]
    #[case::int(
        DType::Primitive(PType::I64, Nullability::Nullable),
        PrimitiveArray::from_option_iter([Some(5i64), None]).into_array(),
    )]
    #[case::string(
        DType::Utf8(Nullability::Nullable),
        VarBinViewArray::from_iter_nullable_str([None, Some("y")]).into_array(),
    )]
    fn reads_a_field_of_members_of_different_dtypes(
        #[case] requested: DType,
        #[case] expected: ArrayRef,
    ) -> VortexResult<()> {
        let object = |dtype: DType| DType::struct_([("x", dtype)], Nullability::NonNullable);
        let variants = UnionVariants::new(
            ["int", "str"].into(),
            vec![
                object(DType::Primitive(PType::I64, Nullability::NonNullable)),
                object(DType::Utf8(Nullability::NonNullable)),
            ],
        )?;
        let type_ids = PrimitiveArray::from_option_iter([Some(0u8), Some(1)]).into_array();
        let ints = StructArray::try_new(
            FieldNames::from_iter(["x"]),
            vec![PrimitiveArray::from_iter([5i64, 0]).into_array()],
            2,
            Validity::NonNullable,
        )?
        .into_array();
        let strs = StructArray::try_new(
            FieldNames::from_iter(["x"]),
            vec![VarBinViewArray::from_iter_str(["", "y"]).into_array()],
            2,
            Validity::NonNullable,
        )?
        .into_array();
        let union = UnionArray::try_new(type_ids, variants, [ints, strs])?.into_array();

        let array = shredded_over(core(None, 2), union)?;
        let expr = variant_get(
            root(),
            VariantPath::new([
                VariantPathElement::Field("a".into()),
                VariantPathElement::Field("x".into()),
            ]),
            Some(requested),
        );

        let mut ctx = SESSION.create_execution_ctx();
        let actual = array.apply(&expr)?.execute::<ArrayRef>(&mut ctx)?;

        assert_arrays_eq!(actual, expected, &mut ctx);
        Ok(())
    }

    /// A field of a struct member reads from that member. The rows of another member have no
    /// such field.
    #[test]
    fn reads_a_field_of_a_union_member() -> VortexResult<()> {
        let object = DType::struct_(
            [("x", DType::Primitive(PType::I64, Nullability::NonNullable))],
            Nullability::NonNullable,
        );
        let variants = UnionVariants::new(
            ["object", "str"].into(),
            vec![object, DType::Utf8(Nullability::NonNullable)],
        )?;
        let type_ids = PrimitiveArray::from_option_iter([Some(0u8), Some(1), Some(0)]).into_array();
        let objects = StructArray::try_new(
            FieldNames::from_iter(["x"]),
            vec![PrimitiveArray::from_iter([5i64, 0, 7]).into_array()],
            3,
            Validity::NonNullable,
        )?
        .into_array();
        let strs = VarBinViewArray::from_iter_str(["", "y", ""]).into_array();
        let union = UnionArray::try_new(type_ids, variants, [objects, strs])?.into_array();

        let array = shredded_over(core(None, 3), union)?;
        let expr = variant_get(
            root(),
            VariantPath::new([
                VariantPathElement::Field("a".into()),
                VariantPathElement::Field("x".into()),
            ]),
            Some(DType::Primitive(PType::I64, Nullability::Nullable)),
        );

        let mut ctx = SESSION.create_execution_ctx();
        let actual = array.apply(&expr)?.execute::<ArrayRef>(&mut ctx)?;

        let expected = PrimitiveArray::from_option_iter([Some(5i64), None, Some(7)]);
        assert_arrays_eq!(actual, expected.into_array(), &mut ctx);
        Ok(())
    }

    /// A string reads as a Boolean as Arrow's Variant conversion reads it, so a shredded string
    /// field gives what the unshredded value gives.
    #[test]
    fn converts_typed_strings_to_booleans() -> VortexResult<()> {
        let typed = VarBinViewArray::from_iter_nullable_str([
            Some("true"),
            Some(" No "),
            Some("1"),
            Some("maybe"),
            None,
        ]);
        let array = shredded_a(typed.into_array())?;
        let expr = variant_get(
            root(),
            VariantPath::field("a"),
            Some(DType::Bool(Nullability::Nullable)),
        );

        let mut ctx = SESSION.create_execution_ctx();
        let actual = array.apply(&expr)?.execute::<ArrayRef>(&mut ctx)?;

        assert_arrays_eq!(
            actual,
            BoolArray::from_iter([Some(true), Some(false), Some(true), None, None]).into_array(),
            &mut ctx
        );
        Ok(())
    }

    /// A number in a nested Variant, as a partially shredded field holds it, reads as a Boolean
    /// too.
    #[test]
    fn converts_numbers_in_variants_to_booleans() -> VortexResult<()> {
        let mut builder = VariantBuilder::new(Nullability::Nullable);
        for value in [0i64, -3] {
            let value = Scalar::primitive(value, Nullability::NonNullable);
            builder.append_n_scalars(&Scalar::variant(value), 1)?;
        }
        builder.append_n_scalars(&Scalar::null(DType::Variant(Nullability::Nullable)), 1)?;
        let array = shredded_a(builder.finish_into_variant()?.into_array())?;
        let expr = variant_get(
            root(),
            VariantPath::field("a"),
            Some(DType::Bool(Nullability::Nullable)),
        );

        let mut ctx = SESSION.create_execution_ctx();
        let actual = array.apply(&expr)?.execute::<ArrayRef>(&mut ctx)?;

        assert_arrays_eq!(
            actual,
            BoolArray::from_iter([Some(false), Some(true), None]).into_array(),
            &mut ctx
        );
        Ok(())
    }
}
