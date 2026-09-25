// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Bool;
use crate::arrays::BoolArray;
use crate::scalar_fn::fns::mask::MaskReduce;
use crate::validity::Validity;

impl MaskReduce for Bool {
    const VALIDITY_IS_METADATA_ONLY: bool = true;

    fn mask(array: ArrayView<'_, Bool>, mask: &ArrayRef) -> VortexResult<Option<ArrayRef>> {
        Ok(Some(
            BoolArray::try_new_from_handle(
                array.bits.clone(),
                array.meta.offset(),
                array.len(),
                array.validity()?.and(Validity::Array(mask.clone()))?,
            )?
            .into_array(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::any::Any;
    use std::ops::Range;
    use std::sync::Arc;

    use futures::future::BoxFuture;
    use rstest::rstest;
    use vortex_buffer::Alignment;
    use vortex_buffer::ByteBuffer;
    use vortex_error::VortexResult;
    use vortex_error::vortex_bail;

    use crate::ArrayEq;
    use crate::ArrayRef;
    use crate::EqMode;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::array_session;
    use crate::arrays::Bool;
    use crate::arrays::BoolArray;
    use crate::buffer::BufferHandle;
    use crate::buffer::DeviceBuffer;
    use crate::compute::conformance::mask::test_mask_conformance;
    use crate::scalar_fn::fns::mask::MaskReduce;
    use crate::validity::Validity;

    // Any access beyond buffer metadata would invalidate the metadata-only reduction.
    #[derive(Debug, PartialEq, Eq, Hash)]
    struct MetadataOnlyDeviceBuffer(usize);

    impl DeviceBuffer for MetadataOnlyDeviceBuffer {
        fn as_any(&self) -> &dyn Any {
            self
        }

        fn len(&self) -> usize {
            self.0
        }

        fn alignment(&self) -> Alignment {
            Alignment::of::<u8>()
        }

        fn copy_to_host_sync(&self, _alignment: Alignment) -> VortexResult<ByteBuffer> {
            vortex_bail!("mask reduction must not copy device values")
        }

        fn copy_to_host(
            &self,
            _alignment: Alignment,
        ) -> VortexResult<BoxFuture<'static, VortexResult<ByteBuffer>>> {
            vortex_bail!("mask reduction must not copy device values")
        }

        fn slice(&self, _range: Range<usize>) -> Arc<dyn DeviceBuffer> {
            panic!("mask reduction must preserve the original buffer");
        }

        fn aligned(self: Arc<Self>, alignment: Alignment) -> VortexResult<Arc<dyn DeviceBuffer>> {
            assert_eq!(self.alignment(), alignment);
            Ok(self)
        }
    }

    #[test]
    fn mask_preserves_device_buffer_and_bit_offset() -> VortexResult<()> {
        let handle = BufferHandle::new_device(Arc::new(MetadataOnlyDeviceBuffer(1)));
        let input = BoolArray::try_new_from_handle(handle, 3, 5, Validity::AllValid)?;
        let mask = BoolArray::from_iter([true, false, true, false, true]).into_array();
        let masked = <Bool as MaskReduce>::mask(input.as_view(), &mask)?
            .expect("Boolean masking must reduce");
        let output = masked.as_::<Bool>();

        assert!(output.bits.array_eq(&input.bits, EqMode::Ptr));
        assert_eq!(output.meta.offset(), input.meta.offset());
        assert_eq!(output.len(), input.len());
        let Validity::Array(validity) = masked.validity()? else {
            panic!("mask must remain array-backed");
        };
        assert!(ArrayRef::ptr_eq(&validity, &mask));

        Ok(())
    }

    #[rstest]
    #[case(BoolArray::from_iter([true, false, true, true, false]))]
    #[case(BoolArray::from_iter([Some(true), None, Some(false), Some(true), None]))]
    #[case(BoolArray::from_iter([true]))]
    #[case(BoolArray::from_iter([false, false]))]
    #[case(BoolArray::from_iter((0..100).map(|i| i % 2 == 0)))]
    fn test_mask_bool_conformance(#[case] array: BoolArray) {
        test_mask_conformance(
            &array.into_array(),
            &mut array_session().create_execution_ctx(),
        );
    }
}
