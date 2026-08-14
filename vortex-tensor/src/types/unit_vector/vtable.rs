// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::EmptyMetadata;
use vortex_array::dtype::DType;
use vortex_array::dtype::extension::ExtDType;
use vortex_array::dtype::extension::ExtId;
use vortex_array::dtype::extension::ExtVTable;
use vortex_array::scalar::PValue;
use vortex_array::scalar::ScalarValue;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_session::registry::CachedId;

use crate::types::unit_vector::UnitVector;
use crate::types::vector::validate_vector_storage_dtype;
use crate::utils::unit_norm_tolerance;

impl ExtVTable for UnitVector {
    type Metadata = EmptyMetadata;
    type NativeValue<'a> = &'a ScalarValue;

    fn id(&self) -> ExtId {
        static ID: CachedId = CachedId::new("vortex.tensor.unit_vector");
        *ID
    }

    fn serialize_metadata(&self, _metadata: &Self::Metadata) -> VortexResult<Vec<u8>> {
        Ok(Vec::new())
    }

    fn deserialize_metadata(&self, _metadata: &[u8]) -> VortexResult<Self::Metadata> {
        Ok(EmptyMetadata)
    }

    fn validate_dtype(ext_dtype: &ExtDType<Self>) -> VortexResult<()> {
        validate_vector_storage_dtype(ext_dtype.storage_dtype())
    }

    fn unpack_native<'a>(
        ext_dtype: &'a ExtDType<Self>,
        storage_value: &'a ScalarValue,
    ) -> VortexResult<Self::NativeValue<'a>> {
        let elements = storage_value.as_list();
        let DType::FixedSizeList(element_dtype, ..) = ext_dtype.storage_dtype() else {
            unreachable!("UnitVector dtype validation established fixed-size-list storage")
        };
        let element_ptype = element_dtype.as_ptype();
        let tolerance = unit_norm_tolerance(element_ptype, elements.len());

        let (norm_squared, is_zero) = elements.iter().try_fold(
            (0.0, true),
            |(sum_squared, is_zero), element| -> VortexResult<_> {
                let value = element
                    .as_ref()
                    .ok_or_else(|| vortex_err!("UnitVector scalar elements must be non-null"))?
                    .as_primitive();
                let value = match value {
                    PValue::F16(value) => value.to_f64(),
                    PValue::F32(value) => *value as f64,
                    PValue::F64(value) => *value,
                    _ => unreachable!("UnitVector dtype validation established float elements"),
                };

                Ok((sum_squared + value * value, is_zero && value == 0.0))
            },
        )?;
        let norm = norm_squared.sqrt();

        vortex_ensure!(
            is_zero || (norm - 1.0).abs() <= tolerance,
            "UnitVector scalar must have L2 norm 1.0 or be exactly zero, got {norm:.6}",
        );

        Ok(storage_value)
    }
}
