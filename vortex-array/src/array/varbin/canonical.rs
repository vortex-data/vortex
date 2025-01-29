use arrow_array::ArrayRef;
use arrow_schema::DataType;
use vortex_dtype::{DType, PType};
use vortex_error::VortexResult;

use crate::array::varbin::VarBinArray;
use crate::array::{VarBinEncoding, VarBinViewArray};
use crate::arrow::{infer_data_type, FromArrowArray, IntoArrowArray};
use crate::compute::{preferred_arrow_data_type, to_arrow};
use crate::encoding::ArrayEncodingRef;
use crate::vtable::CanonicalVTable;
use crate::{ArrayDType, ArrayData, Canonical, IntoArrayData, IntoCanonical};

impl CanonicalVTable<VarBinArray> for VarBinEncoding {
    fn into_canonical(&self, array: VarBinArray) -> VortexResult<Canonical> {
        let dtype = array.dtype().clone();
        let nullable = dtype.is_nullable();

        let array_ref = array.into_array().into_arrow_preferred()?;
        let array = match dtype {
            DType::Utf8(_) => arrow_cast::cast(array_ref.as_ref(), &DataType::Utf8View)?,
            DType::Binary(_) => arrow_cast::cast(array_ref.as_ref(), &DataType::BinaryView)?,

            _ => unreachable!("VarBinArray must have Utf8 or Binary dtype"),
        };
        VarBinViewArray::try_from(ArrayData::from_arrow(array, nullable)).map(Canonical::VarBinView)
    }
}

#[cfg(test)]
mod test {
    use rstest::rstest;
    use vortex_dtype::{DType, Nullability};

    use crate::array::varbin::builder::VarBinBuilder;
    use crate::validity::ArrayValidity;
    use crate::{ArrayDType, IntoCanonical};

    #[rstest]
    #[case(DType::Utf8(Nullability::Nullable))]
    #[case(DType::Binary(Nullability::Nullable))]
    fn test_canonical_varbin(#[case] dtype: DType) {
        let mut varbin = VarBinBuilder::<i32>::with_capacity(10);
        varbin.push_null();
        varbin.push_null();
        // inlined value
        varbin.push_value("123456789012".as_bytes());
        // non-inlinable value
        varbin.push_value("1234567890123".as_bytes());
        let varbin = varbin.finish(dtype.clone());

        let canonical = varbin.into_canonical().unwrap().into_varbinview().unwrap();
        assert_eq!(canonical.dtype(), &dtype);

        assert!(!canonical.is_valid(0).unwrap());
        assert!(!canonical.is_valid(1).unwrap());

        // First value is inlined (12 bytes)
        assert!(canonical.views()[2].is_inlined());
        assert_eq!(canonical.bytes_at(2).as_slice(), "123456789012".as_bytes());

        // Second value is not inlined (13 bytes)
        assert!(!canonical.views()[3].is_inlined());
        assert_eq!(canonical.bytes_at(3).as_slice(), "1234567890123".as_bytes());
    }
}
