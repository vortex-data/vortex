// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::ArrayView;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::arrays::Struct;
use crate::arrays::StructArray;
use crate::arrays::scalar_fn::ExactScalarFn;
use crate::arrays::scalar_fn::ScalarFnArrayView;
use crate::arrays::struct_::StructArrayExt;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::StructFields;
use crate::kernel::ExecuteParentKernel;
use crate::optimizer::kernels::ArrayKernelsExt;
use crate::scalar::Scalar;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::fns::cast::Cast;

pub(crate) fn initialize(session: &VortexSession) {
    session
        .kernels()
        .register_execute_parent_kernel(Cast.id(), Struct, StructCastKernel);
}

#[derive(Debug)]
struct StructCastKernel;

impl ExecuteParentKernel<Struct> for StructCastKernel {
    type Parent = ExactScalarFn<Cast>;

    fn execute_parent(
        &self,
        array: ArrayView<'_, Struct>,
        parent: ScalarFnArrayView<'_, Cast>,
        _child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let dtype = parent.options;
        if array.dtype() == parent.options {
            return Ok(Some(array.array().clone()));
        }

        struct_cast(array, dtype, ctx)
    }
}

pub(crate) fn struct_cast(
    array: ArrayView<Struct>,
    dtype: &DType,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ArrayRef>> {
    let Some(target_sdtype) = dtype.as_struct_fields_opt() else {
        return Ok(None);
    };

    let cast_fields = struct_cast_fields(array, target_sdtype)?;

    let validity = array
        .validity()?
        .cast_nullability(dtype.nullability(), array.len(), ctx)?;

    Ok(Some(
        unsafe {
            StructArray::new_unchecked(cast_fields, target_sdtype.clone(), array.len(), validity)
        }
        .into_array(),
    ))
}

pub(crate) fn struct_cast_fields(
    array: ArrayView<Struct>,
    target_type: &StructFields,
) -> VortexResult<Vec<ArrayRef>> {
    let source_sdtype = array.struct_fields();

    let fields_match_order = target_type.nfields() == source_sdtype.nfields()
        && target_type
            .names()
            .iter()
            .zip(source_sdtype.names().iter())
            .all(|(f1, f2)| f1 == f2);

    let mut cast_fields = Vec::with_capacity(target_type.nfields());
    // Re-order, handle fields by value instead.
    if fields_match_order {
        for (field, target_type) in array.iter_unmasked_fields().zip_eq(target_type.fields()) {
            let cast_field = field.cast(target_type)?;
            cast_fields.push(cast_field);
        }
    } else {
        for (target_name, target_type) in target_type.names().iter().zip_eq(target_type.fields()) {
            match source_sdtype.find(target_name) {
                None => {
                    // No source field with this name => evolve the schema compatibly.
                    // If the field is nullable, we add a new ConstantArray field with the type.
                    vortex_ensure!(
                        target_type.is_nullable(),
                        "CAST for struct only supports added nullable fields"
                    );

                    cast_fields.push(
                        ConstantArray::new(Scalar::null(target_type), array.len()).into_array(),
                    );
                }
                Some(src_field_idx) => {
                    // Field exists in source field. Cast it to the target type.
                    let cast_field = array.unmasked_field(src_field_idx).cast(target_type)?;
                    cast_fields.push(cast_field);
                }
            }
        }
    }

    Ok(cast_fields)
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::ArrayRef;
    use crate::ExecutionCtx;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::arrays::ConstantArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::StructArray;
    use crate::arrays::VarBinArray;
    use crate::arrays::scalar_fn::ScalarFnFactoryExt;
    use crate::arrays::struct_::StructArrayExt;
    use crate::assert_arrays_eq;
    use crate::builtins::ArrayBuiltins;
    use crate::compute::conformance::cast::test_cast_conformance;
    use crate::dtype::DType;
    use crate::dtype::DecimalDType;
    use crate::dtype::FieldNames;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::dtype::StructFields;
    use crate::optimizer::kernels::ArrayKernelsExt;
    use crate::optimizer::kernels::ExecuteParentFn;
    use crate::optimizer::kernels::KernelSession;
    use crate::scalar::Scalar;
    use crate::scalar_fn::fns::cast::Cast;
    use crate::validity::Validity;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(crate::array_session);

    fn null_struct_cast_execute_parent(
        child: &ArrayRef,
        parent: &ArrayRef,
        _child_idx: usize,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let Some(target_fields) = parent.dtype().as_struct_fields_opt() else {
            return Ok(None);
        };
        let fields: Vec<ArrayRef> = target_fields
            .fields()
            .map(|dtype| ConstantArray::new(Scalar::null(dtype), child.len()).into_array())
            .collect();

        StructArray::try_new(
            target_fields.names().clone(),
            fields,
            child.len(),
            Validity::from(parent.dtype().nullability()),
        )
        .map(|array| Some(array.into_array()))
    }

    #[rstest]
    #[case(create_test_struct(false))]
    #[case(create_test_struct(true))]
    #[case(create_nested_struct())]
    #[case(create_simple_struct())]
    fn test_cast_struct_conformance(#[case] array: StructArray) {
        test_cast_conformance(&array.into_array());
    }

    #[test]
    fn struct_cast_execute_parent_uses_session_plugin() {
        let source = StructArray::try_new(
            FieldNames::from(["a"]),
            vec![VarBinArray::from_vec(vec!["A"], DType::Utf8(Nullability::Nullable)).into_array()],
            1,
            Validity::NonNullable,
        )
        .unwrap()
        .into_array();
        let child_id = source.encoding_id();

        let utf8_null = DType::Utf8(Nullability::Nullable);
        let target = DType::Struct(
            StructFields::new(FieldNames::from(["b"]), vec![utf8_null.clone()]),
            Nullability::NonNullable,
        );

        let cast = Cast
            .try_new_array(source.len(), target.clone(), [source])
            .unwrap();
        let parent_id = cast.encoding_id();
        let session = VortexSession::empty().with_some(KernelSession::empty());
        session.kernels().register_execute_parent(
            parent_id,
            child_id,
            &[null_struct_cast_execute_parent as ExecuteParentFn],
        );
        let mut ctx = session.create_execution_ctx();

        let result = cast.execute::<StructArray>(&mut ctx).unwrap();

        assert_eq!(result.dtype(), &target);
        assert_arrays_eq!(
            result.unmasked_field_by_name("b").unwrap(),
            ConstantArray::new(Scalar::null(utf8_null), 1)
        );
    }

    fn create_test_struct(nullable: bool) -> StructArray {
        let names = FieldNames::from(["a", "b"]);

        let a = buffer![1i32, 2, 3].into_array();
        let b = VarBinArray::from_iter(
            vec![Some("x"), None, Some("z")],
            DType::Utf8(Nullability::Nullable),
        )
        .into_array();

        StructArray::try_new(
            names,
            vec![a, b],
            3,
            if nullable {
                Validity::AllValid
            } else {
                Validity::NonNullable
            },
        )
        .unwrap()
    }

    fn create_nested_struct() -> StructArray {
        // Create inner struct
        let inner_names = FieldNames::from(["x", "y"]);

        let x = buffer![1.0f32, 2.0, 3.0].into_array();
        let y = buffer![4.0f32, 5.0, 6.0].into_array();
        let inner_struct = StructArray::try_new(inner_names, vec![x, y], 3, Validity::NonNullable)
            .unwrap()
            .into_array();

        // Create outer struct with inner struct as a field
        let outer_names: FieldNames = ["id", "point"].into();
        // Outer struct would have fields: id (I64) and point (inner struct)

        let ids = buffer![100i64, 200, 300].into_array();

        StructArray::try_new(
            outer_names,
            vec![ids, inner_struct],
            3,
            Validity::NonNullable,
        )
        .unwrap()
    }

    fn create_simple_struct() -> StructArray {
        let names = FieldNames::from(["value"]);
        // Simple struct with a single U8 field

        let values = buffer![42u8].into_array();

        StructArray::try_new(names, vec![values], 1, Validity::NonNullable).unwrap()
    }

    #[test]
    fn cast_nullable_all_invalid() {
        let empty_struct = StructArray::try_new(
            FieldNames::from(["a"]),
            vec![PrimitiveArray::new::<i32>(buffer![], Validity::AllInvalid).into_array()],
            0,
            Validity::AllInvalid,
        )
        .unwrap()
        .into_array();

        let target_dtype = DType::struct_(
            [("a", DType::Primitive(PType::I32, Nullability::NonNullable))],
            Nullability::NonNullable,
        );

        let result = empty_struct.cast(target_dtype.clone()).unwrap();
        assert_eq!(result.dtype(), &target_dtype);
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn cast_duplicate_field_names_to_nullable() {
        let names = FieldNames::from(["a", "a"]);
        let field1 = buffer![1i32, 2, 3].into_array();
        let field2 = buffer![10i64, 20, 30].into_array();

        let struct_array =
            StructArray::try_new(names, vec![field1, field2], 3, Validity::NonNullable).unwrap();

        let target_dtype = struct_array.dtype().as_nullable();

        let cast = struct_array
            .into_array()
            .cast(target_dtype.clone())
            .unwrap();
        assert_eq!(cast.dtype(), &target_dtype);
        assert_eq!(cast.len(), 3);
        let nfields = cast
            .execute::<StructArray>(&mut SESSION.create_execution_ctx())
            .unwrap()
            .struct_fields()
            .nfields();
        assert_eq!(nfields, 2);
    }

    #[test]
    fn cast_add_fields() {
        let names = FieldNames::from(["a", "b"]);
        let field1 = buffer![1i32, 2, 3].into_array();
        let field2 = buffer![10i64, 20, 30].into_array();
        let target_dtype = DType::struct_(
            [
                ("a", field1.dtype().clone()),
                ("b", field2.dtype().clone()),
                (
                    "c",
                    DType::Decimal(DecimalDType::new(38, 10), Nullability::Nullable),
                ),
            ],
            Nullability::NonNullable,
        );

        let struct_array =
            StructArray::try_new(names, vec![field1, field2], 3, Validity::NonNullable).unwrap();

        let result = struct_array
            .into_array()
            .cast(target_dtype.clone())
            .unwrap();
        assert_eq!(result.dtype(), &target_dtype);
        assert_eq!(result.len(), 3);
        let nfields = result
            .execute::<StructArray>(&mut SESSION.create_execution_ctx())
            .unwrap()
            .struct_fields()
            .nfields();
        assert_eq!(nfields, 3);
    }
}
