// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Decimal;
use crate::arrays::DecimalArray;
use crate::arrays::Masked;
use crate::arrays::decimal::DecimalArrayExt;
use crate::arrays::decimal::DecimalArraySlotsExt;
use crate::arrays::dict::TakeExecute;
use crate::arrays::dict::TakeReduce;
use crate::arrays::dict::TakeReduceAdaptor;
use crate::arrays::filter::FilterReduce;
use crate::arrays::filter::FilterReduceAdaptor;
use crate::arrays::slice::SliceReduce;
use crate::arrays::slice::SliceReduceAdaptor;
use crate::builtins::ArrayBuiltins;
use crate::integer;
use crate::optimizer::rules::ArrayParentReduceRule;
use crate::optimizer::rules::ParentRuleSet;
use crate::scalar::Scalar;
use crate::scalar_fn::fns::cast::CastReduceAdaptor;
use crate::scalar_fn::fns::fill_null::FillNullReduce;
use crate::scalar_fn::fns::fill_null::FillNullReduceAdaptor;
use crate::scalar_fn::fns::mask::MaskReduceAdaptor;

pub(crate) static RULES: ParentRuleSet<Decimal> = ParentRuleSet::new(&[
    ParentRuleSet::lift(&DecimalMaskedValidityRule),
    ParentRuleSet::lift(&CastReduceAdaptor(Decimal)),
    ParentRuleSet::lift(&FillNullReduceAdaptor(Decimal)),
    ParentRuleSet::lift(&FilterReduceAdaptor(Decimal)),
    ParentRuleSet::lift(&MaskReduceAdaptor(Decimal)),
    ParentRuleSet::lift(&SliceReduceAdaptor(Decimal)),
    ParentRuleSet::lift(&TakeReduceAdaptor(Decimal)),
]);

fn rewrap(array: ArrayView<'_, Decimal>, values: ArrayRef) -> VortexResult<Option<ArrayRef>> {
    Ok(Some(
        DecimalArray::try_new_values(values, array.decimal_dtype())?.into_array(),
    ))
}

/// Pushes a Masked parent's validity into the decimal's integer child.
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
        rewrap(
            array,
            array.values().clone().mask(parent.validity()?.to_array(array.len()))?,
        )
    }
}

impl SliceReduce for Decimal {
    fn slice(array: ArrayView<'_, Self>, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        rewrap(array, array.values().slice(range)?)
    }
}

impl FilterReduce for Decimal {
    fn filter(array: ArrayView<'_, Self>, mask: &Mask) -> VortexResult<Option<ArrayRef>> {
        rewrap(array, array.values().filter(mask.clone())?)
    }
}

impl TakeReduce for Decimal {
    fn take(array: ArrayView<'_, Self>, indices: &ArrayRef) -> VortexResult<Option<ArrayRef>> {
        rewrap(array, array.values().take(indices.clone())?)
    }
}

impl TakeExecute for Decimal {
    fn take(
        array: ArrayView<'_, Self>,
        indices: &ArrayRef,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        rewrap(array, array.values().take(indices.clone())?)
    }
}

impl FillNullReduce for Decimal {
    fn fill_null(array: ArrayView<'_, Self>, fill_value: &Scalar) -> VortexResult<Option<ArrayRef>> {
        let value = fill_value
            .as_decimal()
            .decimal_value()
            .vortex_expect("FillNull requires a non-null fill value");
        let dtype = array
            .values_dtype()
            .with_nullability(fill_value.dtype().nullability());
        let value = integer::scalar_from_integer(value, &dtype)?;
        rewrap(array, array.values().fill_null(value)?)
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
    use crate::arrays::decimal::DecimalArrayExt;
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

    // Exercise composed byte offsets at every storage width without allowing a host copy.
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
    ) -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let bytes = match_each_decimal_value_type!(values_type, |D| {
            Buffer::<D>::from_iter([-12i8, 34, -56, 78, 90, 12].map(|value| value.as_()))
                .into_byte_buffer()
        });
        let handle = BufferHandle::new_device(Arc::new(TestDeviceBuffer(bytes.clone())));
        let validity = Validity::from_iter([true, false, true, false, true, true]);
        let array = DecimalArray::try_new_handle(
            handle,
            values_type,
            DecimalDType::new(
                match values_type {
                    DecimalType::I8 => 2,
                    DecimalType::I16 => 4,
                    DecimalType::I32 => 9,
                    DecimalType::I64 => 18,
                    DecimalType::I128 => 38,
                    DecimalType::I256 => 76,
                },
                1,
            ),
            validity.clone(),
        )?
        .into_array();
        let sliced = array.slice(1..5)?;

        let range = 1..3;
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
        assert!(handle.is_on_device());
        assert!(handle.is_aligned_to(bytes.alignment()));

        let actual_bytes = &handle
            .as_device()
            .as_any()
            .downcast_ref::<TestDeviceBuffer>()
            .ok_or_else(|| vortex_err!("expected TestDeviceBuffer"))?
            .0;
        assert_eq!(actual_bytes, &expected_bytes);
        assert_eq!(actual_bytes.as_ptr(), expected_bytes.as_ptr());
        assert_arrays_eq!(
            decimal.validity()?.to_array(nested.len()),
            validity.to_array(array.len()).slice(original_range)?,
            &mut ctx
        );

        Ok(())
    }
}
