// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::compute::{TakeKernel, TakeKernelAdapter, take};
use vortex_array::{Array, ArrayRef, register_kernel};
use vortex_error::VortexResult;

use crate::{ALPArray, ALPVTable};

impl TakeKernel for ALPVTable {
    fn take(&self, array: &ALPArray, indices: &dyn Array) -> VortexResult<ArrayRef> {
        let taken_encoded = take(array.encoded(), indices)?;
        let taken_patches = array
            .patches()
            .map(|p| p.take(indices))
            .transpose()?
            .flatten()
            .map(|patches| {
                patches.cast_values(
                    &array
                        .dtype()
                        .with_nullability(taken_encoded.dtype().nullability()),
                )
            })
            .transpose()?;
        Ok(ALPArray::try_new(taken_encoded, array.exponents(), taken_patches)?.to_array())
    }
}

register_kernel!(TakeKernelAdapter(ALPVTable).lift());

#[cfg(test)]
mod test {
    use vortex_array::IntoArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::compute::conformance::take::test_take_conformance;
    use vortex_buffer::buffer;

    use crate::ALPEncoding;

    #[test]
    fn test_take_alp_array() {
        // Test with f32 values
        let values = buffer![1.23f32, 4.56, 7.89, 10.11, 12.13];
        let array = values.into_array();
        let alp = ALPEncoding
            .encode(&array.to_canonical().unwrap(), None)
            .unwrap()
            .unwrap();
        test_take_conformance(alp.as_ref());

        // Test with f64 values
        let values = buffer![100.1f64, 200.2, 300.3, 400.4, 500.5];
        let array = values.into_array();
        let alp = ALPEncoding
            .encode(&array.to_canonical().unwrap(), None)
            .unwrap()
            .unwrap();
        test_take_conformance(alp.as_ref());

        // Test with nullable values
        let array =
            PrimitiveArray::from_option_iter([Some(1.1f32), None, Some(2.2), Some(3.3), None]);
        let alp = ALPEncoding
            .encode(&array.to_canonical().unwrap(), None)
            .unwrap()
            .unwrap();
        test_take_conformance(alp.as_ref());

        // Test with single element
        let values = buffer![42.42f64];
        let array = values.into_array();
        let alp = ALPEncoding
            .encode(&array.to_canonical().unwrap(), None)
            .unwrap()
            .unwrap();
        test_take_conformance(alp.as_ref());
    }
}
