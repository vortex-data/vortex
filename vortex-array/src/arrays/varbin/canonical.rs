use arrow_schema::DataType;
use vortex_dtype::DType;
use vortex_error::VortexResult;

use crate::arrays::varbin::VarBinArray;
use crate::arrays::VarBinViewArray;
use crate::arrow::{FromArrowArray, IntoArrowArray};
use crate::{Array, ArrayCanonicalImpl, ArrayRef, Canonical, TryFromArrayRef};

impl ArrayCanonicalImpl for VarBinArray {
    fn _to_canonical(&self) -> VortexResult<Canonical> {
        let dtype = self.dtype().clone();
        let nullable = dtype.is_nullable();

        let array_ref = self.to_array().into_arrow_preferred()?;
        let array = match dtype {
            DType::Utf8(_) => arrow_cast::cast(array_ref.as_ref(), &DataType::Utf8View)?,
            DType::Binary(_) => arrow_cast::cast(array_ref.as_ref(), &DataType::BinaryView)?,

            _ => unreachable!("VarBinArray must have Utf8 or Binary dtype"),
        };
        VarBinViewArray::try_from_array(ArrayRef::from_arrow(array, nullable))
            .map(Canonical::VarBinView)
    }
}

#[cfg(test)]
mod test {
    use rstest::rstest;
    use vortex_dtype::{DType, Nullability};

    use crate::array::Array;
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

        let canonical = varbin.to_varbinview().unwrap();
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
