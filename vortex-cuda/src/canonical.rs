// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use async_trait::async_trait;
use futures::future::try_join_all;
use vortex::array::ArrayRef;
use vortex::array::Canonical;
use vortex::array::IntoArray;
use vortex::array::VortexSessionExecute;
use vortex::array::arrays::BoolArray;
use vortex::array::arrays::DecimalArray;
use vortex::array::arrays::ExtensionArray;
use vortex::array::arrays::ListViewArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::StructArray;
use vortex::array::arrays::VarBinViewArray;
use vortex::array::arrays::bool::BoolDataParts;
use vortex::array::arrays::decimal::DecimalDataParts;
use vortex::array::arrays::extension::ExtensionArrayExt;
use vortex::array::arrays::listview::ListViewDataParts;
use vortex::array::arrays::primitive::PrimitiveDataParts;
use vortex::array::arrays::struct_::StructDataParts;
use vortex::array::arrays::varbinview::BinaryView;
use vortex::array::arrays::varbinview::VarBinViewDataParts;
use vortex::array::buffer::BufferHandle;
use vortex::array::legacy_session;
use vortex::array::validity::Validity;
use vortex::buffer::BitBuffer;
use vortex::buffer::Buffer;
use vortex::buffer::ByteBuffer;
use vortex::error::VortexResult;

/// Copy a canonical child array to the host.
async fn child_into_host(child: ArrayRef) -> VortexResult<ArrayRef> {
    #[expect(
        clippy::disallowed_methods,
        reason = "CanonicalCudaExt threads no session through"
    )]
    Ok(child
        .execute::<Canonical>(&mut legacy_session().create_execution_ctx())?
        .into_host()
        .await?
        .into_array())
}

/// Move all canonical data from to_host from device.
#[async_trait]
pub trait CanonicalCudaExt {
    async fn into_host(self) -> VortexResult<Self>
    where
        Self: Sized;
}

/// Copies an array-backed validity mask back to the host.
///
/// Only [`Validity::Array`] owns a buffer; the other variants are metadata and pass through.
/// Migrating the values of a nullable array without its validity leaves the mask on the
/// device, and the first host read of it — canonicalising to Arrow, say — panics in
/// `BufferHandle::unwrap_host`.
async fn validity_into_host(validity: Validity) -> VortexResult<Validity> {
    let Validity::Array(array) = validity else {
        return Ok(validity);
    };
    #[expect(
        clippy::disallowed_methods,
        reason = "CanonicalCudaExt threads no session through"
    )]
    let mut ctx = legacy_session().create_execution_ctx();
    Ok(Validity::Array(
        array
            .execute::<Canonical>(&mut ctx)?
            .into_host()
            .await?
            .into_array(),
    ))
}

#[async_trait]
impl CanonicalCudaExt for Canonical {
    async fn into_host(self) -> VortexResult<Self> {
        match self {
            Canonical::Struct(struct_array) => {
                // Children should all be canonical now
                let len = struct_array.len();
                let StructDataParts {
                    fields,
                    struct_fields,
                    validity,
                    ..
                } = struct_array.into_data_parts();

                let mut host_fields = vec![];
                for field in fields.iter() {
                    #[expect(
                        clippy::disallowed_methods,
                        reason = "CanonicalCudaExt threads no session through"
                    )]
                    let mut ctx = legacy_session().create_execution_ctx();
                    host_fields.push(
                        field
                            .clone()
                            .execute::<Canonical>(&mut ctx)?
                            .into_host()
                            .await?
                            .into_array(),
                    );
                }

                Ok(Canonical::Struct(StructArray::new(
                    struct_fields.names().clone(),
                    host_fields,
                    len,
                    validity_into_host(validity).await?,
                )))
            }
            n @ Canonical::Null(_) => Ok(n),
            Canonical::Bool(bool) => {
                let len = bool.len();
                let validity = validity_into_host(bool.validity()?).await?;
                let BoolDataParts { bits, meta } = bool.into_data().into_parts(len);

                let bits = BitBuffer::new_with_offset(
                    bits.try_into_host()?.await?,
                    meta.len(),
                    meta.offset(),
                );
                Ok(Canonical::Bool(BoolArray::new(bits, validity)))
            }
            Canonical::Primitive(prim) => {
                let PrimitiveDataParts {
                    ptype,
                    buffer,
                    validity,
                    ..
                } = prim.into_data_parts();
                Ok(Canonical::Primitive(PrimitiveArray::from_byte_buffer(
                    buffer.try_into_host()?.await?,
                    ptype,
                    validity_into_host(validity).await?,
                )))
            }
            Canonical::Decimal(decimal) => {
                let DecimalDataParts {
                    decimal_dtype,
                    values,
                    values_type,
                    validity,
                    ..
                } = decimal.into_data_parts();
                let validity = validity_into_host(validity).await?;
                Ok(Canonical::Decimal(unsafe {
                    DecimalArray::new_unchecked_handle(
                        BufferHandle::new_host(values.try_into_host()?.await?),
                        values_type,
                        decimal_dtype,
                        validity,
                    )
                }))
            }
            Canonical::VarBinView(varbinview) => {
                let VarBinViewDataParts {
                    views,
                    buffers,
                    validity,
                    dtype,
                } = varbinview.into_data_parts();
                let validity = validity_into_host(validity).await?;

                // Copy all device views to host
                let host_views = views.try_into_host()?.await?;
                let host_views = Buffer::<BinaryView>::from_byte_buffer(host_views);

                // Copy any string data buffers back over to the host
                let host_buffers = buffers
                    .iter()
                    .cloned()
                    .map(|b| b.try_into_host())
                    .collect::<VortexResult<Vec<_>>>()?;
                let host_buffers = try_join_all(host_buffers).await?;
                let host_buffers: Arc<[ByteBuffer]> = Arc::from(host_buffers);

                Ok(Canonical::VarBinView(unsafe {
                    VarBinViewArray::new_unchecked(host_views, host_buffers, dtype, validity)
                }))
            }
            Canonical::List(list) => {
                let ListViewDataParts {
                    elements,
                    offsets,
                    sizes,
                    validity,
                    ..
                } = list.into_data_parts();
                let validity = validity_into_host(validity).await?;

                Ok(Canonical::List(ListViewArray::try_new(
                    child_into_host(elements).await?,
                    child_into_host(offsets).await?,
                    child_into_host(sizes).await?,
                    validity,
                )?))
            }
            Canonical::Extension(ext) => {
                // Copy the storage array to host and rewrap in ExtensionArray.
                #[expect(
                    clippy::disallowed_methods,
                    reason = "CanonicalCudaExt threads no session through"
                )]
                let mut ctx = legacy_session().create_execution_ctx();
                let host_storage = ext
                    .storage_array()
                    .clone()
                    .execute::<Canonical>(&mut ctx)?
                    .into_host()
                    .await?
                    .into_array();
                Ok(Canonical::Extension(ExtensionArray::new(
                    ext.ext_dtype().clone(),
                    host_storage,
                )))
            }
            c => todo!("{} not implemented", c.dtype()),
        }
    }
}
