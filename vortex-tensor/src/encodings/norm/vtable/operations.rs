// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex::array::IntoArray;
use vortex::array::arrays::ConstantArray;
use vortex::array::arrays::FixedSizeList;
use vortex::array::builtins::ArrayBuiltins;
use vortex::array::vtable::OperationsVTable;
use vortex::dtype::Nullability;
use vortex::error::VortexResult;
use vortex::error::vortex_err;
use vortex::scalar::Scalar;
use vortex::scalar_fn::fns::operators::Operator;

use crate::encodings::norm::array::NormVectorArray;
use crate::encodings::norm::vtable::NormVector;
use crate::utils::extension_list_size;
use crate::utils::extension_storage;

impl OperationsVTable<NormVector> for NormVector {
    fn scalar_at(array: &NormVectorArray, index: usize) -> VortexResult<Scalar> {
        let ext = array
            .vector_array()
            .dtype()
            .as_extension_opt()
            .ok_or_else(|| {
                vortex_err!(
                    "expected Vector extension dtype, got {}",
                    array.vector_array().dtype()
                )
            })?;
        let list_size = extension_list_size(ext)?;

        // Get the storage (FixedSizeList) and slice out the elements for this row.
        let storage = extension_storage(array.vector_array())?;
        let fsl = storage
            .as_opt::<FixedSizeList>()
            .ok_or_else(|| vortex_err!("expected FixedSizeList storage"))?;
        let row_elements = fsl.fixed_size_list_elements_at(index)?;

        // Multiply all elements by the norm using a ConstantArray broadcast.
        let norm_scalar = array.norms().scalar_at(index)?;
        let norm_broadcast = ConstantArray::new(norm_scalar, list_size).into_array();
        let scaled = row_elements.binary(norm_broadcast, Operator::Mul)?;

        // Rebuild the FSL scalar, then wrap in the extension type.
        let element_dtype = ext
            .storage_dtype()
            .as_fixed_size_list_element_opt()
            .ok_or_else(|| {
                vortex_err!(
                    "expected FixedSizeList storage dtype, got {}",
                    ext.storage_dtype()
                )
            })?;

        let children: Vec<Scalar> = (0..list_size)
            .map(|i| scaled.scalar_at(i))
            .collect::<VortexResult<_>>()?;

        let fsl_scalar =
            Scalar::fixed_size_list(element_dtype.clone(), children, Nullability::NonNullable);

        Ok(Scalar::extension_ref(ext.clone(), fsl_scalar))
    }
}
