// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::VarBin;
use crate::arrays::VarBinArray;
use crate::scalar_fn::fns::mask::MaskReduce;
use crate::validity::Validity;

impl MaskReduce for VarBin {
    fn mask(array: ArrayView<'_, VarBin>, mask: &ArrayRef) -> VortexResult<Option<ArrayRef>> {
        Ok(Some(
            VarBinArray::try_new(
                array.offsets().clone(),
                array.bytes().clone(),
                array.dtype().as_nullable(),
                array.validity().and(Validity::Array(mask.clone()))?,
            )?
            .into_array(),
        ))
    }
}

#[cfg(test)]
mod test {
    use crate::IntoArray;
    use crate::arrays::VarBinArray;
    use crate::compute::conformance::mask::test_mask_conformance;
    use crate::dtype::DType;
    use crate::dtype::Nullability;

    #[test]
    fn test_mask_var_bin_array() {
        let array = VarBinArray::from_vec(
            vec!["hello", "world", "filter", "good", "bye"],
            DType::Utf8(Nullability::NonNullable),
        );
        test_mask_conformance(&array.into_array());

        let array = VarBinArray::from_iter(
            vec![Some("hello"), None, Some("filter"), Some("good"), None],
            DType::Utf8(Nullability::Nullable),
        );
        test_mask_conformance(&array.into_array());
    }
}
