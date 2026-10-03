// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Device payload fixtures shared by native UTF-8 input tests.
//!
//! These fixtures distinguish metadata reads from host payload acquisition.

use std::any::Any;
use std::ops::Range;
use std::sync::Arc;

use futures::future::BoxFuture;
use vortex_buffer::Alignment;
use vortex_buffer::Buffer;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::ArrayParts;
use crate::ArrayRef;
use crate::IntoArray as _;
use crate::arrays::ConstantArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::VarBin;
use crate::arrays::VarBinArray;
use crate::arrays::varbin::VarBinData;
use crate::arrays::varbin::VarBinSlots;
use crate::buffer::BufferHandle;
use crate::buffer::DeviceBuffer;
use crate::dtype::DType;
use crate::validity::Validity;

/// A device payload that rejects every attempt to acquire host bytes.
#[derive(Debug, PartialEq, Eq, Hash)]
pub(super) struct UnreadablePayload {
    /// The payload length reported by metadata without reading any bytes.
    pub(super) len: usize,
}

impl DeviceBuffer for UnreadablePayload {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn len(&self) -> usize {
        self.len
    }

    fn alignment(&self) -> Alignment {
        Alignment::of::<u8>()
    }

    fn copy_to_host_sync(&self, _alignment: Alignment) -> VortexResult<ByteBuffer> {
        vortex_bail!("payload host access rejected")
    }

    fn copy_to_host(
        &self,
        _alignment: Alignment,
    ) -> VortexResult<BoxFuture<'static, VortexResult<ByteBuffer>>> {
        vortex_bail!("payload host access rejected")
    }

    fn slice(&self, range: Range<usize>) -> Arc<dyn DeviceBuffer> {
        Arc::new(Self { len: range.len() })
    }

    fn aligned(self: Arc<Self>, _alignment: Alignment) -> VortexResult<Arc<dyn DeviceBuffer>> {
        Ok(self)
    }
}

/// Build one 13-byte offset row whose payload cannot be read on the host.
pub(super) fn unreadable_offset_array(validity: Validity) -> VortexResult<ArrayRef> {
    let offsets =
        PrimitiveArray::new(Buffer::from(vec![0u32, 13]), Validity::NonNullable).into_array();
    let bytes = BufferHandle::new_device(Arc::new(UnreadablePayload { len: 13 }));
    let dtype = DType::Utf8(validity.nullability());
    let data =
        VarBinData::try_build_from_handle(offsets.clone(), bytes, dtype.clone(), validity.clone())?;
    let slots = VarBinSlots {
        offsets,
        validity: matches!(validity, Validity::AllInvalid)
            .then(|| ConstantArray::new(false, 1).into_array()),
    }
    .into_slots();

    Ok(
        VarBinArray::try_from_parts(ArrayParts::new(VarBin, dtype, 1, data).with_slots(slots))?
            .into_array(),
    )
}
