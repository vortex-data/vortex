// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::compute::{CastKernel, CastKernelAdapter, cast};
use vortex_array::{ArrayRef, register_kernel};
use vortex_dtype::DType;
use vortex_error::VortexResult;

use crate::{FSSTArray, FSSTVTable};

impl CastKernel for FSSTVTable {
    fn cast(&self, array: &FSSTArray, dtype: &DType) -> VortexResult<Option<ArrayRef>> {
        // FSST is a string compression encoding. 
        // For nullability changes, we can cast the codes and symbols arrays
        if array.dtype().eq_ignore_nullability(dtype) {
            // Cast codes array to handle nullability
            let new_codes = cast(array.codes(), &array.codes().dtype().with_nullability(dtype.nullability()))?;
            
            Ok(Some(FSSTArray::try_new(
                dtype.clone(),
                array.symbols().clone(),
                new_codes,
                array.uncompressed_lengths().clone(),
            )?
            .into_array()))
        } else {
            // For type changes (e.g., Utf8 to Binary), we need to decode
            // because FSST compression is specific to the string representation
            let decoded = array.to_canonical()?.into_array();
            cast(&decoded, dtype).map(Some)
        }
    }
}

register_kernel!(CastKernelAdapter(FSSTVTable).lift());

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::arrays::VarBinArray;
    use vortex_array::compute::cast;
    use vortex_array::compute::conformance::cast::test_cast_conformance;
    use vortex_array::{IntoArray, ToCanonical};
    use vortex_dtype::{DType, Nullability};

    use crate::{FSSTArray, FSSTEncoding};
    use vortex_array::encoding::ArrayEncoding;

    #[test]
    fn test_cast_fsst_nullability() {
        let strings = VarBinArray::from_iter(
            vec![Some("hello"), Some("world"), Some("hello world")],
            DType::Utf8(Nullability::NonNullable),
        );
        
        let fsst = FSSTEncoding.encode(&strings.into_array(), None).unwrap().unwrap();
        
        // Cast to nullable
        let casted = cast(fsst.as_ref(), &DType::Utf8(Nullability::Nullable)).unwrap();
        assert_eq!(casted.dtype(), &DType::Utf8(Nullability::Nullable));
    }

    #[test]
    fn test_cast_fsst_to_binary() {
        let strings = VarBinArray::from_iter(
            vec![Some("test"), Some("data")],
            DType::Utf8(Nullability::NonNullable),
        );
        
        let fsst = FSSTEncoding.encode(&strings.into_array(), None).unwrap().unwrap();
        
        // Cast UTF8 to Binary
        let casted = cast(fsst.as_ref(), &DType::Binary(Nullability::NonNullable)).unwrap();
        assert_eq!(casted.dtype(), &DType::Binary(Nullability::NonNullable));
        
        // Verify content
        let decoded = casted.to_canonical().unwrap();
        let varbin = decoded.as_varbin().unwrap();
        assert_eq!(varbin.as_slice::<&[u8]>()[0], b"test");
        assert_eq!(varbin.as_slice::<&[u8]>()[1], b"data");
    }

    #[rstest]
    #[case(VarBinArray::from_iter(
        vec![Some("hello"), Some("world"), Some("hello world")],
        DType::Utf8(Nullability::NonNullable)
    ))]
    #[case(VarBinArray::from_iter(
        vec![Some("foo"), None, Some("bar"), Some("foobar")],
        DType::Utf8(Nullability::Nullable)
    ))]
    #[case(VarBinArray::from_iter(
        vec![Some("test")],
        DType::Utf8(Nullability::NonNullable)
    ))]
    fn test_cast_fsst_conformance(#[case] array: VarBinArray) {
        let fsst = FSSTEncoding.encode(&array.into_array(), None).unwrap().unwrap();
        test_cast_conformance(fsst.as_ref());
    }
}