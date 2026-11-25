// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::StructFields;
use vortex_array::scalar::Scalar;

use crate::lit;
use crate::test_harness;

#[test]
fn dtype() {
    let dtype = test_harness::struct_dtype();

    assert_eq!(
        lit(10).return_dtype(&dtype).unwrap(),
        DType::Primitive(PType::I32, Nullability::NonNullable)
    );
    assert_eq!(
        lit(i64::MAX).return_dtype(&dtype).unwrap(),
        DType::Primitive(PType::I64, Nullability::NonNullable)
    );
    assert_eq!(
        lit(true).return_dtype(&dtype).unwrap(),
        DType::Bool(Nullability::NonNullable)
    );
    assert_eq!(
        lit(Scalar::null(DType::Bool(Nullability::Nullable)))
            .return_dtype(&dtype)
            .unwrap(),
        DType::Bool(Nullability::Nullable)
    );

    let sdtype = DType::Struct(
        StructFields::new(
            ["dog", "cat"].into(),
            vec![
                DType::Primitive(PType::U32, Nullability::NonNullable),
                DType::Utf8(Nullability::NonNullable),
            ],
        ),
        Nullability::NonNullable,
    );
    assert_eq!(
        lit(Scalar::struct_(
            sdtype.clone(),
            vec![Scalar::from(32_u32), Scalar::from("rufus".to_string())]
        ))
        .return_dtype(&dtype)
        .unwrap(),
        sdtype
    );
}
