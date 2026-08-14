// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_array::ArrayRef;
use vortex_array::EmptyMetadata;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::extension::ExtensionArrayExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::extension::ExtDType;
use vortex_array::scalar::Scalar;
use vortex_array::validity::Validity;
use vortex_buffer::buffer;
use vortex_error::VortexResult;

use crate::tests::SESSION;
use crate::types::unit_vector::AnyUnitVector;
use crate::types::unit_vector::UnitVector;
use crate::types::vector::AnyVector;
use crate::utils::test_helpers::vector_array;
use crate::utils::unit_norm_tolerance;

fn unit_vector(dimensions: u32, values: &[f64]) -> VortexResult<ArrayRef> {
    let mut ctx = SESSION.create_execution_ctx();
    let vector = vector_array(dimensions, values)?;
    let vector: ExtensionArray = vector.execute(&mut ctx)?;

    UnitVector::try_new_unit_vector_array(vector.storage_array().clone(), &mut ctx)
}

fn storage_dtype(ptype: PType, dimensions: u32) -> DType {
    DType::FixedSizeList(
        Arc::new(DType::Primitive(ptype, Nullability::NonNullable)),
        dimensions,
        Nullability::NonNullable,
    )
}

fn unit_dtype(ptype: PType, dimensions: u32) -> VortexResult<DType> {
    let dtype = ExtDType::<UnitVector>::try_new(EmptyMetadata, storage_dtype(ptype, dimensions))?;

    Ok(DType::Extension(dtype.erased()))
}

#[test]
fn checked_constructor_accepts_unit_and_zero_rows() -> VortexResult<()> {
    let array = unit_vector(2, &[0.6, 0.8, 0.0, 0.0])?;

    assert!(array.dtype().as_extension().is::<AnyUnitVector>());
    assert!(array.dtype().as_extension().is::<AnyVector>());
    Ok(())
}

#[test]
fn checked_constructor_rejects_non_unit_row() {
    assert!(unit_vector(2, &[3.0, 4.0]).is_err());
}

#[test]
fn checked_constructor_rejects_nonzero_row_with_underflowed_norm() {
    assert!(unit_vector(2, &[f64::from_bits(1), 0.0]).is_err());
}

#[test]
fn scalar_constructor_rejects_non_unit_value() -> VortexResult<()> {
    let element_dtype = DType::Primitive(PType::F64, Nullability::NonNullable);
    let storage = Scalar::fixed_size_list(
        element_dtype,
        vec![
            Scalar::primitive(3.0f64, Nullability::NonNullable),
            Scalar::primitive(4.0f64, Nullability::NonNullable),
        ],
        Nullability::NonNullable,
    );

    assert!(Scalar::try_new(unit_dtype(PType::F64, 2)?, storage.into_value()).is_err());
    Ok(())
}

#[test]
fn checked_constructor_ignores_null_row_payloads() -> VortexResult<()> {
    let elements = buffer![3.0f64, 4.0, 0.6, 0.8].into_array();
    let validity = Validity::Array(BoolArray::from_iter([false, true]).into_array());
    let storage = FixedSizeListArray::try_new(elements, 2, validity, 2)?.into_array();
    let mut ctx = SESSION.create_execution_ctx();

    UnitVector::try_new_unit_vector_array(storage, &mut ctx)?;
    Ok(())
}

#[test]
fn f16_tolerance_is_capped() {
    assert_eq!(unit_norm_tolerance(PType::F16, 768), 1e-2);
    assert!(unit_norm_tolerance(PType::F32, 768) < 1e-2);
}
