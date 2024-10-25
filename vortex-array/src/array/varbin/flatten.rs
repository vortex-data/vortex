use arrow_schema::DataType;
use vortex_error::VortexResult;

use crate::array::varbin::arrow::varbin_to_arrow;
use crate::array::varbin::VarBinArray;
use crate::array::VarBinViewArray;
use crate::arrow::FromArrowArray;
use crate::{Array, ArrayDType, Canonical, IntoCanonical};

impl IntoCanonical for VarBinArray {
    fn into_canonical(self) -> VortexResult<Canonical> {
        let nullable = self.dtype().is_nullable();
        let array_ref = varbin_to_arrow(&self)?;
        let casted = arrow_cast::cast(array_ref.as_ref(), &DataType::Utf8View)?;

        VarBinViewArray::try_from(Array::from_arrow(casted, nullable)).map(Canonical::VarBinView)
    }
}

#[cfg(test)]
mod test {
    use vortex_dtype::{DType, Nullability};

    use crate::array::varbin::builder::VarBinBuilder;
    use crate::validity::ArrayValidity;
    use crate::IntoCanonical;

    #[test]
    fn test_canonical_varbin() {
        let mut varbin = VarBinBuilder::<i32>::with_capacity(10);
        varbin.push_null();
        varbin.push_null();
        // inlined value
        varbin.push_value("123456789012".as_bytes());
        // non-inlinable value
        varbin.push_value("1234567890123".as_bytes());
        let varbin = varbin.finish(DType::Utf8(Nullability::Nullable));

        let canonical = varbin.into_canonical().unwrap().into_varbinview().unwrap();

        assert!(!canonical.is_valid(0));
        assert!(!canonical.is_valid(1));

        // First value is inlined (12 bytes)
        assert!(canonical.view_at(2).unwrap().is_inlined());
        assert_eq!(
            canonical.bytes_at(2).unwrap().as_slice(),
            "123456789012".as_bytes()
        );

        // Second value is not inlined (13 bytes)
        assert!(!canonical.view_at(3).unwrap().is_inlined());
        assert_eq!(
            canonical.bytes_at(3).unwrap().as_slice(),
            "1234567890123".as_bytes()
        );
    }
}
