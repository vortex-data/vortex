// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;
use vortex_dtype::DType;
use vortex_error::{VortexExpect, VortexResult, vortex_bail};

use crate::arrays::{StructArray, StructVTable};
use crate::compute::{CastKernel, CastKernelAdapter, cast};
use crate::vtable::ValidityHelper;
use crate::{ArrayRef, IntoArray, register_kernel};

impl CastKernel for StructVTable {
    fn cast(&self, array: &StructArray, dtype: &DType) -> VortexResult<Option<ArrayRef>> {
        let Some(target_sdtype) = dtype.as_struct_fields_opt() else {
            return Ok(None);
        };

        let source_sdtype = array
            .dtype()
            .as_struct_fields_opt()
            .vortex_expect("struct array must have struct dtype");

        if target_sdtype.names() != source_sdtype.names() {
            vortex_bail!("cannot cast {} to {}", array.dtype(), dtype);
        }

        let validity = array
            .validity()
            .clone()
            .cast_nullability(dtype.nullability(), array.len())?;

        StructArray::try_new(
            target_sdtype.names().clone(),
            array
                .fields()
                .iter()
                .zip_eq(target_sdtype.fields())
                .map(|(field, dtype)| cast(field, &dtype))
                .try_collect()?,
            array.len(),
            validity,
        )
        .map(|a| Some(a.into_array()))
    }
}

register_kernel!(CastKernelAdapter(StructVTable).lift());

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_buffer::buffer;
    use vortex_dtype::{DType, FieldNames, Nullability, PType};

    use crate::IntoArray;
    use crate::arrays::{PrimitiveArray, StructArray, VarBinArray};
    use crate::compute::conformance::cast::test_cast_conformance;
    use crate::validity::Validity;

    #[rstest]
    #[case(create_test_struct(false))]
    #[case(create_test_struct(true))]
    #[case(create_nested_struct())]
    #[case(create_simple_struct())]
    fn test_cast_struct_conformance(#[case] array: StructArray) {
        test_cast_conformance(array.as_ref());
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
            vec![PrimitiveArray::new::<i32>(buffer![], Validity::AllInvalid).to_array()],
            0,
            Validity::AllInvalid,
        )
        .unwrap()
        .to_array();

        let target_dtype = DType::struct_(
            [("a", DType::Primitive(PType::I32, Nullability::NonNullable))],
            Nullability::NonNullable,
        );

        let result = crate::compute::cast(&empty_struct, &target_dtype).unwrap();
        assert_eq!(result.dtype(), &target_dtype);
        assert_eq!(result.len(), 0);
    }
}
