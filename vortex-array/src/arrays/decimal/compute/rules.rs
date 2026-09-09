// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Decimal;
use crate::arrays::DecimalArray;
use crate::arrays::Masked;
use crate::arrays::slice::SliceReduce;
use crate::arrays::slice::SliceReduceAdaptor;
use crate::match_each_decimal_value_type;
use crate::optimizer::rules::ArrayParentReduceRule;
use crate::optimizer::rules::ParentRuleSet;
use crate::scalar_fn::fns::cast::CastReduceAdaptor;
use crate::scalar_fn::fns::mask::MaskReduceAdaptor;

pub(crate) static RULES: ParentRuleSet<Decimal> = ParentRuleSet::new(&[
    ParentRuleSet::lift(&DecimalMaskedValidityRule),
    ParentRuleSet::lift(&CastReduceAdaptor(Decimal)),
    ParentRuleSet::lift(&MaskReduceAdaptor(Decimal)),
    ParentRuleSet::lift(&SliceReduceAdaptor(Decimal)),
]);

/// Rule to push down validity masking from MaskedArray parent into DecimalArray child.
///
/// When a DecimalArray is wrapped by a MaskedArray, this rule merges the mask's validity
/// with the DecimalArray's existing validity, eliminating the need for the MaskedArray wrapper.
#[derive(Default, Debug)]
pub struct DecimalMaskedValidityRule;

impl ArrayParentReduceRule<Decimal> for DecimalMaskedValidityRule {
    type Parent = Masked;

    fn reduce_parent(
        &self,
        array: ArrayView<'_, Decimal>,
        parent: ArrayView<'_, Masked>,
        _child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        // Merge the parent's validity mask into the child's validity
        // TODO(joe): make this lazy
        let masked_array = match_each_decimal_value_type!(array.values_type(), |D| {
            // SAFETY: Since we are only flipping some bits in the validity, all invariants that
            // were upheld are still upheld.
            unsafe {
                DecimalArray::new_unchecked(
                    array.buffer::<D>(),
                    array.decimal_dtype(),
                    array.validity()?.and(parent.validity()?)?,
                )
            }
            .into_array()
        });

        Ok(Some(masked_array))
    }
}

impl SliceReduce for Decimal {
    fn slice(array: ArrayView<'_, Self>, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        let byte_width = array.values_type().byte_width();
        let byte_range = range.start * byte_width..range.end * byte_width;
        let values = array.buffer_handle().slice(byte_range);
        let validity = array.validity()?.slice(range)?;

        // SAFETY: Slicing on element boundaries preserves the buffer alignment, values type,
        // decimal precision and scale, and validity length invariants.
        let result = unsafe {
            DecimalArray::new_unchecked_handle(
                values,
                array.values_type(),
                array.decimal_dtype(),
                validity,
            )
            .into_array()
        };
        Ok(Some(result))
    }
}

#[cfg(test)]
mod tests {
    use std::any::Any;
    use std::ops::Range;
    use std::sync::Arc;

    use futures::future::BoxFuture;
    use num_traits::AsPrimitive;
    use rstest::rstest;
    use vortex_buffer::Alignment;
    use vortex_buffer::Buffer;
    use vortex_buffer::ByteBuffer;
    use vortex_error::VortexResult;
    use vortex_error::vortex_bail;
    use vortex_error::vortex_err;

    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::array_session;
    use crate::arrays::Decimal;
    use crate::arrays::DecimalArray;
    use crate::assert_arrays_eq;
    use crate::buffer::BufferHandle;
    use crate::buffer::DeviceBuffer;
    use crate::dtype::DecimalDType;
    use crate::dtype::DecimalType;
    use crate::match_each_decimal_value_type;
    use crate::validity::Validity;

    // Host-backed storage exposes device slicing without allowing implicit host copies.
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
            vortex_bail!("decimal slicing must not copy device values to the host")
        }

        fn copy_to_host(
            &self,
            _alignment: Alignment,
        ) -> VortexResult<BoxFuture<'static, VortexResult<ByteBuffer>>> {
            vortex_bail!("decimal slicing must not copy device values to the host")
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
    fn test_slice_buffer_handle(
        #[values(
            DecimalType::I8,
            DecimalType::I16,
            DecimalType::I32,
            DecimalType::I64,
            DecimalType::I128,
            DecimalType::I256
        )]
        values_type: DecimalType,
        #[values(
            Validity::NonNullable,
            Validity::AllValid,
            Validity::AllInvalid,
            Validity::from_iter([true, false, true, false, true, true])
        )]
        validity: Validity,
        #[values(false, true)] on_device: bool,
    ) -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let bytes = match_each_decimal_value_type!(values_type, |D| {
            Buffer::<D>::from_iter([-12i8, 34, -56, 78, 90, 12].map(|value| value.as_()))
                .into_byte_buffer()
        });
        let handle = if on_device {
            BufferHandle::new_device(Arc::new(TestDeviceBuffer(bytes.clone())))
        } else {
            BufferHandle::new_host(bytes.clone())
        };
        let array = DecimalArray::try_new_handle(
            handle,
            values_type,
            DecimalDType::new(3, 1),
            validity.clone(),
        )?
        .into_array();
        let sliced = array.slice(1..5)?;

        for range in [0..4, 0..2, 1..3, 3..4] {
            let nested = sliced.slice(range.clone())?;
            let decimal = nested.as_::<Decimal>();
            let byte_width = values_type.byte_width();
            let original_range = range.start + 1..range.end + 1;
            let expected_bytes =
                bytes.slice(original_range.start * byte_width..original_range.end * byte_width);
            let handle = decimal.buffer_handle();

            assert_eq!(nested.len(), range.len());
            assert_eq!(nested.dtype(), array.dtype());
            assert_eq!(decimal.values_type(), values_type);
            assert_eq!(handle.len(), range.len() * byte_width);
            assert_eq!(handle.is_on_device(), on_device);
            assert!(handle.is_aligned_to(bytes.alignment()));

            let actual_bytes = if on_device {
                &handle
                    .as_device()
                    .as_any()
                    .downcast_ref::<TestDeviceBuffer>()
                    .ok_or_else(|| vortex_err!("expected TestDeviceBuffer"))?
                    .0
            } else {
                handle.as_host()
            };
            assert_eq!(actual_bytes, &expected_bytes);
            assert_eq!(actual_bytes.as_ptr(), expected_bytes.as_ptr());
            assert_arrays_eq!(
                decimal.validity()?.to_array(nested.len()),
                validity.to_array(array.len()).slice(original_range)?,
                &mut ctx
            );
        }

        Ok(())
    }
}
