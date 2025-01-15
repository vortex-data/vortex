use vortex_error::VortexResult;

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
pub trait ValidateVTable {
    fn validate(&self, array: &ArrayData) -> VortexResult<()>;
}
