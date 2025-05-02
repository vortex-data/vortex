use vortex_array::compute::{ScalarAtFn, scalar_at};
use vortex_array::vtable::ComputeVTable;
use vortex_array::{Array, ToCanonical};
use vortex_error::VortexResult;
use vortex_scalar::Scalar;

use crate::{DeltaArray, DeltaEncoding};

impl ComputeVTable for DeltaEncoding {
    fn scalar_at_fn(&self) -> Option<&dyn ScalarAtFn<&dyn Array>> {
        Some(self)
    }
}

impl ScalarAtFn<&DeltaArray> for DeltaEncoding {
    fn scalar_at(&self, array: &DeltaArray, index: usize) -> VortexResult<Scalar> {
        let decompressed = array.slice(index, index + 1)?.to_primitive()?;
        scalar_at(&decompressed, 0)
    }
}

#[cfg(test)]
mod test {
    use vortex_array::compute::scalar_at;
    use vortex_error::VortexError;

    use super::*;

    #[test]
    fn test_scalar_at_non_jagged_array() {
        let delta = DeltaArray::try_from_vec((0u32..2048).collect())
            .unwrap()
            .into_array();

        assert_eq!(scalar_at(&delta, 0).unwrap(), 0_u32.into());
        assert_eq!(scalar_at(&delta, 1).unwrap(), 1_u32.into());
        assert_eq!(scalar_at(&delta, 10).unwrap(), 10_u32.into());
        assert_eq!(scalar_at(&delta, 1023).unwrap(), 1023_u32.into());
        assert_eq!(scalar_at(&delta, 1024).unwrap(), 1024_u32.into());
        assert_eq!(scalar_at(&delta, 1025).unwrap(), 1025_u32.into());
        assert_eq!(scalar_at(&delta, 2047).unwrap(), 2047_u32.into());

        assert!(matches!(
            scalar_at(&delta, 2048),
            Err(VortexError::OutOfBounds(2048, 0, 2048, _))
        ));

        assert!(matches!(
            scalar_at(&delta, 2049),
            Err(VortexError::OutOfBounds(2049, 0, 2048, _))
        ));
    }

    #[test]
    fn test_scalar_at_jagged_array() {
        let delta = DeltaArray::try_from_vec((0u32..2000).collect())
            .unwrap()
            .into_array();

        assert_eq!(scalar_at(&delta, 0).unwrap(), 0_u32.into());
        assert_eq!(scalar_at(&delta, 1).unwrap(), 1_u32.into());
        assert_eq!(scalar_at(&delta, 10).unwrap(), 10_u32.into());
        assert_eq!(scalar_at(&delta, 1023).unwrap(), 1023_u32.into());
        assert_eq!(scalar_at(&delta, 1024).unwrap(), 1024_u32.into());
        assert_eq!(scalar_at(&delta, 1025).unwrap(), 1025_u32.into());
        assert_eq!(scalar_at(&delta, 1999).unwrap(), 1999_u32.into());

        assert!(matches!(
            scalar_at(&delta, 2000),
            Err(VortexError::OutOfBounds(2000, 0, 2000, _))
        ));

        assert!(matches!(
            scalar_at(&delta, 2001),
            Err(VortexError::OutOfBounds(2001, 0, 2000, _))
        ));
    }
}
