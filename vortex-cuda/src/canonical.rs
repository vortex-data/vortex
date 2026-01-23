// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use async_trait::async_trait;
use vortex_array::Canonical;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::BoolArrayParts;
use vortex_array::arrays::DecimalArray;
use vortex_array::arrays::DecimalArrayParts;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::PrimitiveArrayParts;
use vortex_array::buffer::BufferHandle;
use vortex_error::VortexResult;

/// Move all canonical data from to_host from device.
#[async_trait]
pub trait CanonicalCudaExt {
    async fn to_host(self) -> VortexResult<Self>
    where
        Self: Sized;
}

#[async_trait]
impl CanonicalCudaExt for Canonical {
    async fn to_host(self) -> VortexResult<Self> {
        match self {
            n @ Canonical::Null(_) => Ok(n),
            Canonical::Bool(bool) => {
                // NOTE: update to copy to host when adding buffer handle.
                // Also update other method to copy validity to host.
                let BoolArrayParts { bits, validity, .. } = bool.into_parts();
                Ok(Canonical::Bool(BoolArray::from_bit_buffer(bits, validity)))
            }
            Canonical::Primitive(prim) => {
                let PrimitiveArrayParts {
                    ptype,
                    buffer,
                    validity,
                    ..
                } = prim.into_parts();
                Ok(Canonical::Primitive(PrimitiveArray::from_byte_buffer(
                    buffer.try_into_host()?.await?,
                    ptype,
                    validity,
                )))
            }
            Canonical::Decimal(decimal) => {
                let DecimalArrayParts {
                    decimal_dtype,
                    values,
                    values_type,
                    validity,
                    ..
                } = decimal.into_parts();
                Ok(Canonical::Decimal(unsafe {
                    DecimalArray::new_unchecked_handle(
                        BufferHandle::new_host(values.try_into_host()?.await?),
                        values_type,
                        decimal_dtype,
                        validity,
                    )
                }))
            }
            _ => todo!(),
        }
    }
}
