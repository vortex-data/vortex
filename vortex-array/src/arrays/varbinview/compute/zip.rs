// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::arrays::{VarBinViewArray, VarBinViewVTable};
use crate::builders::VarBinViewBuilder;
use crate::compute::{ZipKernel, ZipKernelAdapter, zip_impl_with_builder, zip_return_dtype};
use crate::{Array, ArrayRef, register_kernel};

impl ZipKernel for VarBinViewVTable {
    fn zip(
        &self,
        if_true: &VarBinViewArray,
        if_false: &dyn Array,
        mask: &vortex_mask::Mask,
    ) -> VortexResult<Option<ArrayRef>> {
        let Some(if_false) = if_false.as_opt::<VarBinViewVTable>() else {
            return Ok(None);
        };
        Ok(Some(zip_impl_with_builder(
            if_true.as_ref(),
            if_false.as_ref(),
            mask,
            Box::new(VarBinViewBuilder::with_buffer_deduplication(
                zip_return_dtype(if_true.as_ref(), if_false.as_ref()),
                if_true.len(),
            )),
        )?))
    }
}

register_kernel!(ZipKernelAdapter(VarBinViewVTable).lift());

#[cfg(test)]
mod tests {
    use arrow_array::cast::AsArray;
    use arrow_select::zip::zip as arrow_zip;
    use vortex_dtype::{DType, Nullability};
    use vortex_mask::Mask;

    use crate::IntoArray;
    use crate::arrays::VarBinViewVTable;
    use crate::arrow::IntoArrowArray;
    use crate::builders::{ArrayBuilder as _, BufferGrowthStrategy, VarBinViewBuilder};
    use crate::compute::zip;

    #[test]
    fn test_varbinview_zip() {
        let if_true = {
            let mut builder = VarBinViewBuilder::new(
                DType::Utf8(Nullability::NonNullable),
                10,
                Default::default(),
                BufferGrowthStrategy::fixed(64 * 1024),
            );
            for _ in 0..100 {
                builder.append_value("Hello");
                builder.append_value("Hello this is a long string that won't be inlined.");
            }
            builder.finish()
        };

        let if_false = {
            let mut builder = VarBinViewBuilder::new(
                DType::Utf8(Nullability::NonNullable),
                10,
                Default::default(),
                BufferGrowthStrategy::fixed(64 * 1024),
            );
            for _ in 0..100 {
                builder.append_value("Hello2");
                builder.append_value("Hello2 this is a long string that won't be inlined.");
            }
            builder.finish()
        };

        // [1,2,4,5,7,8,..]
        let mask = Mask::from_indices(200, (0..100).filter(|i| i % 3 != 0).collect());

        let zipped = zip(&if_true, &if_false, &mask).unwrap();
        let zipped = zipped.as_opt::<VarBinViewVTable>().unwrap();
        assert_eq!(zipped.nbuffers(), 2);

        // assert the result is the same as arrow
        let expected = arrow_zip(
            mask.into_array()
                .into_arrow_preferred()
                .unwrap()
                .as_boolean(),
            &if_true.into_arrow_preferred().unwrap(),
            &if_false.into_arrow_preferred().unwrap(),
        )
        .unwrap();

        let actual = zipped.clone().into_array().into_arrow_preferred().unwrap();
        assert_eq!(actual.as_ref(), expected.as_ref());
    }
}
