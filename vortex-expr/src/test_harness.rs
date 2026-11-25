// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Fixtures shared by expression tests across crates.

use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::StructFields;

pub fn struct_dtype() -> DType {
    DType::Struct(
        StructFields::new(
            ["a", "col1", "col2", "bool1", "bool2"].into(),
            vec![
                DType::Primitive(PType::I32, Nullability::NonNullable),
                DType::Primitive(PType::U16, Nullability::Nullable),
                DType::Primitive(PType::U16, Nullability::Nullable),
                DType::Bool(Nullability::NonNullable),
                DType::Bool(Nullability::NonNullable),
            ],
        ),
        Nullability::NonNullable,
    )
}
