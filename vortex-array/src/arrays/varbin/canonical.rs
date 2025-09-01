// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::{BinaryViewArray, StringViewArray};
use arrow_schema::DataType;
use vortex_dtype::DType;
use vortex_error::VortexExpect;

use crate::arrays::VarBinVTable;
use crate::arrays::varbin::VarBinArray;
use crate::arrow::{FromArrowArray, IntoArrowArray};
use crate::vtable::CanonicalVTable;
use crate::{ArrayRef, Canonical, ToCanonical};

impl CanonicalVTable<VarBinVTable> for VarBinVTable {
    fn canonicalize(array: &VarBinArray) -> Canonical {
        let dtype = array.dtype().clone();
        let nullable = dtype.is_nullable();

        let array_ref = array
            .to_array()
            .into_arrow_preferred()
            .vortex_expect("VarBinArray must be convertible to arrow array");

        let array = match (&dtype, array_ref.data_type()) {
            (DType::Utf8(_), DataType::Utf8) => {
                Arc::new(StringViewArray::from(array_ref.as_string::<i32>()))
                    as Arc<dyn arrow_array::Array>
            }
            (DType::Utf8(_), DataType::LargeUtf8) => {
                Arc::new(StringViewArray::from(array_ref.as_string::<i64>()))
                    as Arc<dyn arrow_array::Array>
            }

            (DType::Binary(_), DataType::Binary) => {
                Arc::new(BinaryViewArray::from(array_ref.as_binary::<i32>()))
            }
            (DType::Binary(_), DataType::LargeBinary) => {
                Arc::new(BinaryViewArray::from(array_ref.as_binary::<i64>()))
            }
            // If its already a view, no need to do anything
            (DType::Binary(_), DataType::BinaryView) | (DType::Utf8(_), DataType::Utf8View) => {
                array_ref
            }
            _ => unreachable!("VarBinArray must have Utf8 or Binary dtype, instead got: {dtype}",),
        };
        Canonical::VarBinView(ArrayRef::from_arrow(array.as_ref(), nullable).to_varbinview())
    }
}

#[cfg(test)]
mod test {
    use rstest::rstest;
    use vortex_dtype::{DType, Nullability};

    use crate::arrays::varbin::builder::VarBinBuilder;
    use crate::canonical::ToCanonical;

    #[rstest]
    #[case(DType::Utf8(Nullability::Nullable))]
    #[case(DType::Binary(Nullability::Nullable))]
    fn test_canonical_varbin(#[case] dtype: DType) {
        let mut varbin = VarBinBuilder::<i32>::with_capacity(10);
        varbin.append_null();
        varbin.append_null();
        // inlined value
        varbin.append_value("123456789012".as_bytes());
        // non-inlinable value
        varbin.append_value("1234567890123".as_bytes());
        let varbin = varbin.finish(dtype.clone());

        let canonical = varbin.to_varbinview();
        assert_eq!(canonical.dtype(), &dtype);

        assert!(!canonical.is_valid(0));
        assert!(!canonical.is_valid(1));

        // First value is inlined (12 bytes)
        assert!(canonical.views()[2].is_inlined());
        assert_eq!(canonical.bytes_at(2).as_slice(), "123456789012".as_bytes());

        // Second value is not inlined (13 bytes)
        assert!(!canonical.views()[3].is_inlined());
        assert_eq!(canonical.bytes_at(3).as_slice(), "1234567890123".as_bytes());
    }
}
