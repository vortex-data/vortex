use vortex_error::{VortexError, VortexExpect, VortexResult};

use crate::encoding::Encoding;
use crate::ArrayData;

/// A trait implemented by encodings to verify an opaque [`ArrayData`].
///
/// The caller will already have verified that the encoding ID matches, but the implementor must
/// verify the following:
/// * The metadata is valid (for example, flatbuffer validation to allow unchecked access later).
/// * The number of buffers is correct.
/// * The buffers are correctly aligned.
/// * The number of children is correct.
///
/// Do not validate that children can be accessed as this will trigger eager recursive validation.
pub trait ValidateVTable<Array> {
    fn validate(&self, _array: &Array) -> VortexResult<()> {
        // TODO(ngates): remove this default implementation once we migrate Arrays to
        //  [u8] metadata.
        Ok(())
    }
}

impl<E: Encoding> ValidateVTable<ArrayData> for E
where
    E: ValidateVTable<E::Array>,
    for<'a> &'a E::Array: TryFrom<&'a ArrayData, Error = VortexError>,
{
    fn validate(&self, array: &ArrayData) -> VortexResult<()> {
        let (array_ref, encoding) = array
            .try_downcast_ref::<E>()
            .vortex_expect("Failed to downcast encoding");
        encoding.validate(array_ref)
    }
}
