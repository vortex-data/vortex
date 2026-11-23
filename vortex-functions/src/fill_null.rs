// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VTable;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::FixedSizeList;
use vortex_array::arrays::ListView;
use vortex_array::arrays::Map;
use vortex_array::arrays::ScalarFn;
use vortex_array::arrays::Struct;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::scalar_fn::ExactScalarFn;
use vortex_array::arrays::scalar_fn::ScalarFnArrayExt;
use vortex_array::arrays::scalar_fn::ScalarFnArrayView;
use vortex_array::arrays::struct_::StructArrayExt;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::kernel::ExecuteParentKernel;
use vortex_array::optimizer::kernels::ArrayKernelsExt;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_array::scalar_fn::builtin::fill_null::FillNull;
use vortex_array::scalar_fn::builtin::fill_null::short_circuit;
use vortex_array::validity::Validity;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

use crate::builtins::ArrayFunctions;
use crate::zip::Zip;

/// Registers `Zip`-backed [`FillNull`] kernels for nested canonical encodings.
pub(crate) fn initialize(session: &VortexSession) {
    let kernels = session.kernels();
    kernels.register_execute_parent_kernel(FillNull.id(), Struct, StructFillNullKernel);
    kernels.register_execute_parent_kernel(FillNull.id(), ListView, ZipFillNullKernel);
    kernels.register_execute_parent_kernel(FillNull.id(), FixedSizeList, ZipFillNullKernel);
    kernels.register_execute_parent_kernel(FillNull.id(), Map, ZipFillNullKernel);
}

/// Fills null struct rows with the fields of a struct scalar.
///
/// Each field is zipped over the struct validity, so the fields of valid rows stay as they are and
/// null rows take the fields of the fill value. The result has the nullability of the fill value.
#[derive(Default, Debug)]
struct StructFillNullKernel;

impl ExecuteParentKernel<Struct> for StructFillNullKernel {
    type Parent = ExactScalarFn<FillNull>;

    fn execute_parent(
        &self,
        array: ArrayView<'_, Struct>,
        parent: ScalarFnArrayView<'_, FillNull>,
        child_idx: usize,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        if child_idx != 0 {
            return Ok(None);
        }
        let scalar_fn_array = parent
            .as_opt::<ScalarFn>()
            .vortex_expect("ExactScalarFn matcher confirmed ScalarFnArray");
        let Some(fill_value) = scalar_fn_array.get_child(1).as_constant() else {
            return Ok(None);
        };
        let arr = array.array().clone();
        if let Some(result) = short_circuit(&arr, fill_value)? {
            return Ok(Some(result));
        }
        if !FillNull::supports_dtype(arr.dtype()) {
            vortex_bail!("fill_null is not supported for {} arrays", arr.dtype());
        }

        let nullability = fill_value.dtype().nullability();
        let fill_value = fill_value.cast(&array.dtype().with_nullability(nullability))?;
        let fill_struct = fill_value.as_struct();
        let fill_fields = fill_struct
            .fields_iter()
            .ok_or_else(|| vortex_err!("fill_null requires a non-null fill value"))?;

        let len = array.len();
        let is_valid = array.validity()?.to_array(len);

        let fields = array
            .iter_unmasked_fields()
            .zip(fill_fields)
            .map(|(field, fill_field)| {
                let fill = ConstantArray::new(fill_field, len).into_array();
                is_valid.zip(field.clone(), fill)
            })
            .collect::<VortexResult<Vec<_>>>()?;

        Ok(Some(
            StructArray::try_new(
                array.names().clone(),
                fields,
                len,
                Validity::from(nullability),
            )?
            .into_array(),
        ))
    }
}

/// Fills nulls by zipping the input with a constant fill over the input validity.
///
/// Reuses `zip` instead of rebuilding offsets here. The zip result is nullable like the input, so
/// a cast gives it the nullability of the fill value. The result is lazy.
#[derive(Default, Debug)]
struct ZipFillNullKernel;

impl<V: VTable> ExecuteParentKernel<V> for ZipFillNullKernel {
    type Parent = ExactScalarFn<FillNull>;

    fn execute_parent(
        &self,
        array: ArrayView<'_, V>,
        parent: ScalarFnArrayView<'_, FillNull>,
        child_idx: usize,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        if child_idx != 0 {
            return Ok(None);
        }
        let scalar_fn_array = parent
            .as_opt::<ScalarFn>()
            .vortex_expect("ExactScalarFn matcher confirmed ScalarFnArray");
        let Some(fill_value) = scalar_fn_array.get_child(1).as_constant() else {
            return Ok(None);
        };
        let arr = array.array().clone();
        if let Some(result) = short_circuit(&arr, fill_value)? {
            return Ok(Some(result));
        }
        if !FillNull::supports_dtype(arr.dtype()) {
            vortex_bail!("fill_null is not supported for {} arrays", arr.dtype());
        }

        let len = arr.len();
        let result_dtype = arr
            .dtype()
            .with_nullability(fill_value.dtype().nullability());
        let fill = ConstantArray::new(fill_value.cast(&result_dtype)?, len).into_array();
        let is_valid = arr.validity()?.to_array(len);

        Zip::try_new(arr, fill, is_valid)?
            .into_array()
            .cast(result_dtype)
            .map(Some)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::ArrayRef;
    use vortex_array::Canonical;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::arrays::ListViewArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::Struct;
    use vortex_array::arrays::StructArray;
    use vortex_array::arrays::UnionArray;
    use vortex_array::arrays::VarBinViewArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::builders::builder_with_capacity_in;
    use vortex_array::builtins::ArrayBuiltins;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::FieldNames;
    use vortex_array::dtype::MapDType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::dtype::StructFields;
    use vortex_array::dtype::UnionVariants;
    use vortex_array::memory::BufferAllocatorRef;
    use vortex_array::scalar::Scalar;
    use vortex_array::scalar_fn::builtin::fill_null::FillNull;
    use vortex_array::validity::Validity;
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty();
        vortex_array::initialize(&session);
        crate::initialize(&session);
        session
    });

    fn i32_dtype() -> DType {
        DType::Primitive(PType::I32, Nullability::NonNullable)
    }

    fn map_dtype() -> MapDType {
        MapDType::try_new(i32_dtype(), i32_dtype(), false).vortex_expect("valid map dtype")
    }

    fn names() -> FieldNames {
        FieldNames::from_iter(["id", "inner"])
    }

    fn inner_dtype() -> DType {
        DType::struct_(
            [("name", DType::Utf8(Nullability::Nullable))],
            Nullability::Nullable,
        )
    }

    /// Null rows take every field of the fill value, including a nested struct, while the
    /// fields of valid rows, and the nulls inside them, stay as they are.
    #[rstest]
    #[case::non_nullable_fill(Nullability::NonNullable)]
    #[case::nullable_fill(Nullability::Nullable)]
    fn fill_null_struct(#[case] nullability: Nullability) -> VortexResult<()> {
        let inner = |values: &[Option<&str>], validity: Validity| -> VortexResult<ArrayRef> {
            Ok(StructArray::try_new(
                FieldNames::from_iter(["name"]),
                vec![VarBinViewArray::from_iter_nullable_str(values.iter().copied()).into_array()],
                values.len(),
                validity,
            )?
            .into_array())
        };

        // Rows 1 and 3 are null. Row 2 is valid with a null nested struct.
        let input = StructArray::try_new(
            names(),
            vec![
                PrimitiveArray::from_option_iter([Some(1i32), Some(2), None, Some(4)]).into_array(),
                inner(
                    &[Some("a"), Some("b"), None, Some("d")],
                    Validity::from_iter([true, true, false, true]),
                )?,
            ],
            4,
            Validity::from_iter([true, false, true, false]),
        )?
        .into_array();

        let fill_dtype = DType::Struct(
            StructFields::new(
                names(),
                vec![
                    DType::Primitive(PType::I32, Nullability::Nullable),
                    inner_dtype(),
                ],
            ),
            nullability,
        );
        let fill = Scalar::struct_(
            fill_dtype,
            [
                Scalar::primitive(9i32, Nullability::Nullable),
                Scalar::struct_(inner_dtype(), [Scalar::utf8("fill", Nullability::Nullable)]),
            ],
        );

        let mut ctx = SESSION.create_execution_ctx();
        let result = input.fill_null(fill)?.execute::<ArrayRef>(&mut ctx)?;
        assert!(result.is::<Struct>());

        let expected = StructArray::try_new(
            names(),
            vec![
                PrimitiveArray::from_option_iter([Some(1i32), Some(9), None, Some(9)]).into_array(),
                inner(
                    &[Some("a"), Some("fill"), None, Some("fill")],
                    Validity::from_iter([true, true, false, true]),
                )?,
            ],
            4,
            Validity::from(nullability),
        )?;
        assert_arrays_eq!(result, expected, &mut ctx);
        Ok(())
    }

    /// Inputs with both valid and null rows execute for every zip-backed nested dtype that
    /// `supports_dtype` accepts, with a nullable or a non-nullable fill value.
    #[rstest]
    #[case::list(DType::List(Arc::new(i32_dtype()), Nullability::Nullable))]
    #[case::fixed_size_list(DType::FixedSizeList(Arc::new(i32_dtype()), 2, Nullability::Nullable))]
    #[case::struct_(DType::struct_([("value", i32_dtype())], Nullability::Nullable))]
    #[case::map(DType::Map(map_dtype(), Nullability::Nullable))]
    fn supported_zip_dtypes_fill_mixed_validity(
        #[case] dtype: DType,
        #[values(Nullability::NonNullable, Nullability::Nullable)] fill_nullability: Nullability,
    ) -> VortexResult<()> {
        assert!(FillNull::supports_dtype(&dtype), "{dtype}");

        let mut ctx = SESSION.create_execution_ctx();
        let fill = Scalar::default_value(&dtype.as_nonnullable())
            .cast(&dtype.with_nullability(fill_nullability))?;
        let mut builder = builder_with_capacity_in(&dtype, 2, ctx.allocator());
        builder.append_scalar(&fill.cast(&dtype)?)?;
        builder.append_null();
        let filled = builder
            .finish()
            .fill_null(fill.clone())?
            .execute::<Canonical>(&mut ctx)?
            .into_array();

        assert_eq!(filled.dtype(), fill.dtype(), "{dtype}");
        assert_eq!(filled.execute_scalar(1, &mut ctx)?, fill, "{dtype}");
        assert_arrays_eq!(filled, ConstantArray::new(fill, 2), &mut ctx);
        Ok(())
    }

    /// A list of unions with mixed validity fails with an error instead of panicking while
    /// materializing the fill constant.
    #[test]
    fn list_of_union_with_mixed_validity_errors() -> VortexResult<()> {
        let variants = UnionVariants::try_new(
            ["num"].into(),
            vec![DType::Primitive(PType::I32, Nullability::NonNullable)],
            vec![0],
        )?;
        let elements = UnionArray::try_new(
            PrimitiveArray::from_iter([0u8]).into_array(),
            variants.clone(),
            vec![PrimitiveArray::from_iter([1i32]).into_array()],
        )?
        .into_array();
        let mut ctx = SESSION.create_execution_ctx();
        let input = ListViewArray::try_new(
            elements,
            PrimitiveArray::from_iter([0u32, 1]).into_array(),
            PrimitiveArray::from_iter([1u32, 0]).into_array(),
            Validity::from_iter([true, false]),
            &mut ctx,
        )?
        .into_array();
        let fill = Scalar::list(
            Arc::new(DType::Union(variants.clone(), Nullability::NonNullable)),
            vec![Scalar::union(
                variants,
                0,
                Scalar::from(2i32),
                Nullability::NonNullable,
            )?],
            Nullability::NonNullable,
        );

        let error = input
            .fill_null(fill)?
            .execute::<Canonical>(&mut ctx)
            .unwrap_err();
        assert!(
            error.to_string().contains("fill_null is not supported"),
            "{error}"
        );
        Ok(())
    }

    fn array_from_scalars(dtype: &DType, scalars: &[Scalar]) -> VortexResult<ArrayRef> {
        let mut builder =
            builder_with_capacity_in(dtype, scalars.len(), BufferAllocatorRef::static_ref());
        for scalar in scalars {
            builder.append_scalar(scalar)?;
        }
        Ok(builder.finish())
    }

    fn nullable_i32(value: Option<i32>) -> Scalar {
        match value {
            Some(value) => Scalar::primitive(value, Nullability::Nullable),
            None => Scalar::null(DType::Primitive(PType::I32, Nullability::Nullable)),
        }
    }

    fn nullable_i32_list(values: &[Option<i32>], nullability: Nullability) -> Scalar {
        Scalar::list(
            Arc::new(DType::Primitive(PType::I32, Nullability::Nullable)),
            values.iter().copied().map(nullable_i32).collect(),
            nullability,
        )
    }

    fn point_dtype() -> DType {
        DType::struct_([("x", i32_dtype())], Nullability::NonNullable)
    }

    fn points(xs: &[i32], nullability: Nullability) -> Scalar {
        let points = xs
            .iter()
            .map(|&x| Scalar::struct_(point_dtype(), [Scalar::from(x)]))
            .collect();
        Scalar::fixed_size_list(Arc::new(point_dtype()), points, nullability)
    }

    fn map_utf8_dtype() -> MapDType {
        MapDType::try_new(i32_dtype(), DType::Utf8(Nullability::Nullable), false)
            .vortex_expect("valid map dtype")
    }

    fn map(entries: &[(i32, Option<&str>)], nullability: Nullability) -> Scalar {
        Scalar::map(
            DType::Map(map_utf8_dtype(), nullability),
            entries.iter().map(|&(key, value)| {
                let value = match value {
                    Some(value) => Scalar::utf8(value, Nullability::Nullable),
                    None => Scalar::null(DType::Utf8(Nullability::Nullable)),
                };
                (Scalar::from(key), value)
            }),
        )
    }

    fn items_dtype() -> DType {
        DType::List(Arc::new(i32_dtype()), Nullability::NonNullable)
    }

    fn items(values: &[i32], nullability: Nullability) -> Scalar {
        let dtype = DType::struct_([("items", items_dtype())], nullability);
        let items = Scalar::list(
            Arc::new(i32_dtype()),
            values.iter().copied().map(Scalar::from).collect(),
            Nullability::NonNullable,
        );
        Scalar::struct_(dtype, [items])
    }

    /// Null rows of nested dtypes take the whole fill value, while valid rows, including empty
    /// lists and null elements inside them, stay as they are. `None` input rows are null.
    #[rstest]
    #[case::list(
        DType::List(
            Arc::new(DType::Primitive(PType::I32, Nullability::Nullable)),
            Nullability::Nullable,
        ),
        |n| nullable_i32_list(&[Some(7), Some(8)], n),
        vec![
            Some(nullable_i32_list(&[Some(1), None], Nullability::Nullable)),
            None,
            Some(nullable_i32_list(&[], Nullability::Nullable)),
            None,
        ],
        |n| vec![
            nullable_i32_list(&[Some(1), None], n),
            nullable_i32_list(&[Some(7), Some(8)], n),
            nullable_i32_list(&[], n),
            nullable_i32_list(&[Some(7), Some(8)], n),
        ],
    )]
    #[case::fixed_size_list_of_structs(
        DType::FixedSizeList(Arc::new(point_dtype()), 2, Nullability::Nullable),
        |n| points(&[7, 8], n),
        vec![
            Some(points(&[1, 2], Nullability::Nullable)),
            None,
            Some(points(&[3, 4], Nullability::Nullable)),
        ],
        |n| vec![points(&[1, 2], n), points(&[7, 8], n), points(&[3, 4], n)],
    )]
    #[case::map(
        DType::Map(map_utf8_dtype(), Nullability::Nullable),
        |n| map(&[(9, Some("z")), (10, None)], n),
        vec![
            None,
            Some(map(&[(1, Some("a"))], Nullability::Nullable)),
            None,
            Some(map(&[], Nullability::Nullable)),
        ],
        |n| vec![
            map(&[(9, Some("z")), (10, None)], n),
            map(&[(1, Some("a"))], n),
            map(&[(9, Some("z")), (10, None)], n),
            map(&[], n),
        ],
    )]
    #[case::struct_of_lists(
        DType::struct_([("items", items_dtype())], Nullability::Nullable),
        |n| items(&[5, 6], n),
        vec![
            Some(items(&[1], Nullability::Nullable)),
            None,
            Some(items(&[], Nullability::Nullable)),
        ],
        |n| vec![items(&[1], n), items(&[5, 6], n), items(&[], n)],
    )]
    fn fill_null_nested(
        #[case] dtype: DType,
        #[case] fill: fn(Nullability) -> Scalar,
        #[case] input: Vec<Option<Scalar>>,
        #[case] expected: fn(Nullability) -> Vec<Scalar>,
        #[values(Nullability::NonNullable, Nullability::Nullable)] fill_nullability: Nullability,
    ) -> VortexResult<()> {
        let input = input
            .into_iter()
            .map(|row| row.unwrap_or_else(|| Scalar::null(dtype.clone())))
            .collect::<Vec<_>>();
        let input = array_from_scalars(&dtype, &input)?;
        let fill = fill(fill_nullability);
        let expected = array_from_scalars(fill.dtype(), &expected(fill_nullability))?;

        let mut ctx = SESSION.create_execution_ctx();
        let result = input.fill_null(fill)?;
        assert_arrays_eq!(result, expected, &mut ctx);
        Ok(())
    }
}
