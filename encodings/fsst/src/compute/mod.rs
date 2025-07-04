// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod compare;
mod filter;

use vortex_array::arrays::VarBinVTable;
use vortex_array::compute::{TakeKernel, TakeKernelAdapter, fill_null, take};
use vortex_array::{Array, ArrayRef, IntoArray, register_kernel};
use vortex_error::VortexResult;
use vortex_scalar::{Scalar, ScalarValue};

use crate::{FSSTArray, FSSTVTable};

impl TakeKernel for FSSTVTable {
    // Take on an FSSTArray is a simple take on the codes array.
    fn take(&self, array: &FSSTArray, indices: &dyn Array) -> VortexResult<ArrayRef> {
        Ok(FSSTArray::try_new(
            array
                .dtype()
                .clone()
                .union_nullability(indices.dtype().nullability()),
            array.symbols().clone(),
            array.symbol_lengths().clone(),
            take(array.codes().as_ref(), indices)?
                .as_::<VarBinVTable>()
                .clone(),
            fill_null(
                &take(array.uncompressed_lengths(), indices)?,
                &Scalar::new(
                    array.uncompressed_lengths_dtype().clone(),
                    ScalarValue::from(0),
                ),
            )?,
        )?
        .into_array())
    }
}

register_kernel!(TakeKernelAdapter(FSSTVTable).lift());

#[cfg(test)]
mod tests {
    use vortex_array::arrays::{PrimitiveArray, VarBinArray};
    use vortex_array::compute::take;
    use vortex_dtype::{DType, Nullability};

    use crate::{fsst_compress, fsst_train_compressor};

    #[test]
    fn test_take_null() {
        let arr = VarBinArray::from_iter([Some("h")], DType::Utf8(Nullability::NonNullable));
        let compr = fsst_train_compressor(arr.as_ref()).unwrap();
        let fsst = fsst_compress(arr.as_ref(), &compr).unwrap();

        let idx1: PrimitiveArray = (0..1).collect();

        assert_eq!(
            take(fsst.as_ref(), idx1.as_ref()).unwrap().dtype(),
            &DType::Utf8(Nullability::NonNullable)
        );

        let idx2: PrimitiveArray = PrimitiveArray::from_option_iter(vec![Some(0)]);

        assert_eq!(
            take(fsst.as_ref(), idx2.as_ref()).unwrap().dtype(),
            &DType::Utf8(Nullability::Nullable)
        );
    }
}
