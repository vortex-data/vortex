// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Device reads preserve narrow and equal-width MSPs and reject wider legacy storage.

use std::any::Any;
use std::future::Future;
use std::ops::Range;
use std::pin::Pin;
use std::sync::Arc;

use prost::Message;
use rstest::rstest;
use vortex_array::ArrayDeserialization;
use vortex_array::ArrayPlugin;
use vortex_array::IntoArray;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::Widen;
use vortex_array::arrays::widen::WidenArraySlotsExt;
use vortex_array::buffer::BufferHandle;
use vortex_array::buffer::DeviceBuffer;
use vortex_array::dtype::DType;
use vortex_array::dtype::DecimalDType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::validity::Validity;
use vortex_buffer::Alignment;
use vortex_buffer::ByteBuffer;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use super::super::DecimalBytePartsPlugin;
use super::super::DecimalBytePartsV2Metadata;
use super::super::decimal_byte_parts_v1_id;
use super::super::decimal_byte_parts_v2_id;
use super::super::v1::DecimalBytePartsMetadata;
use super::session;
use crate::DecimalByteParts;
use crate::DecimalBytePartsArraySlotsExt;

/// Fails every host copy so the test detects reads that try to move device data.
#[derive(Debug, PartialEq, Eq, Hash)]
struct TestDeviceBuffer(ByteBuffer);

impl DeviceBuffer for TestDeviceBuffer {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn len(&self) -> usize {
        self.0.len()
    }

    fn alignment(&self) -> Alignment {
        self.0.alignment()
    }

    fn copy_to_host_sync(&self, _alignment: Alignment) -> VortexResult<ByteBuffer> {
        vortex_bail!("the MSP reader must not copy device memory to the host")
    }

    fn copy_to_host(
        &self,
        _alignment: Alignment,
    ) -> VortexResult<Pin<Box<dyn Future<Output = VortexResult<ByteBuffer>> + Send>>> {
        vortex_bail!("the MSP reader must not copy device memory to the host")
    }

    fn slice(&self, range: Range<usize>) -> Arc<dyn DeviceBuffer> {
        Arc::new(Self(self.0.slice(range)))
    }

    fn aligned(self: Arc<Self>, alignment: Alignment) -> VortexResult<Arc<dyn DeviceBuffer>> {
        assert!(self.alignment().is_aligned_to(alignment));
        Ok(self)
    }
}

#[rstest]
#[case::narrow(18, true)]
#[case::equal(9, true)]
#[case::wider_legacy(4, false)]
fn read_device_msp(
    #[case] precision: u8,
    #[case] accepted: bool,
    #[values(false, true)] v2: bool,
) -> VortexResult<()> {
    let dtype = DType::Decimal(DecimalDType::new(precision, 0), Nullability::NonNullable);
    let children = vec![
        PrimitiveArray::from_buffer_handle(
            BufferHandle::new_device(Arc::new(TestDeviceBuffer(
                buffer![99i32; 3].into_byte_buffer(),
            ))),
            PType::I32,
            Validity::NonNullable,
        )
        .into_array(),
    ];
    let (id, metadata) = if v2 {
        (
            decimal_byte_parts_v2_id(),
            DecimalBytePartsV2Metadata {
                msp_ptype: PType::I32 as i32,
                lower_part_ptypes: vec![],
            }
            .encode_to_vec(),
        )
    } else {
        (
            decimal_byte_parts_v1_id(),
            DecimalBytePartsMetadata {
                zeroth_child_ptype: PType::I32 as i32,
                lower_part_count: 0,
            }
            .encode_to_vec(),
        )
    };
    let result = DecimalBytePartsPlugin.deserialize(
        ArrayDeserialization::new(id, &dtype, 3, &metadata, &[], &children),
        &session(),
    );
    if accepted {
        let array = result?;
        let array = array.as_opt::<DecimalByteParts>().unwrap();
        let msp = array.msp();
        let stored = if let Some(widen) = msp.as_opt::<Widen>() {
            widen.values().clone()
        } else {
            msp.clone()
        };
        assert!(stored.as_::<Primitive>().buffer_handle().is_on_device());
    } else {
        assert!(result.is_err_and(|error| error.to_string().contains("must be in host memory")));
    }

    Ok(())
}
