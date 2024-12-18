use vortex_dtype::DType;
use vortex_error::{vortex_bail, vortex_err, VortexError, VortexResult};

use crate::arrow::{Datum, FromArrowArray};
use crate::encoding::Encoding;
use crate::{ArrayDType, ArrayData};

pub trait LikeFn<Array> {
    fn like(
        &self,
        array: &Array,
        pattern: &ArrayData,
        options: LikeOptions,
    ) -> VortexResult<ArrayData>;
}

impl<E: Encoding> LikeFn<ArrayData> for E
where
    E: LikeFn<E::Array>,
    for<'a> &'a E::Array: TryFrom<&'a ArrayData, Error = VortexError>,
{
    fn like(
        &self,
        array: &ArrayData,
        pattern: &ArrayData,
        options: LikeOptions,
    ) -> VortexResult<ArrayData> {
        let array_ref = <&E::Array>::try_from(array)?;
        let encoding = array
            .encoding()
            .as_any()
            .downcast_ref::<E>()
            .ok_or_else(|| vortex_err!("Mismatched encoding"))?;
        LikeFn::like(encoding, array_ref, pattern, options)
    }
}

/// Options for SQL LIKE function
#[derive(Default, Debug, Clone, Copy)]
pub struct LikeOptions {
    pub negated: bool,
    pub case_insensitive: bool,
}

/// Perform SQL left LIKE right
///
/// There are two wildcards supported with the LIKE operator:
/// - %: matches zero or more characters
/// - _: matches exactly one character
pub fn like(
    array: &ArrayData,
    pattern: &ArrayData,
    options: LikeOptions,
) -> VortexResult<ArrayData> {
    if !matches!(array.dtype(), DType::Utf8(..)) {
        vortex_bail!("Expected utf8 array, got {}", array.dtype());
    }
    if !matches!(pattern.dtype(), DType::Utf8(..)) {
        vortex_bail!("Expected utf8 pattern, got {}", array.dtype());
    }

    if let Some(f) = array.encoding().like_fn() {
        let result = f.like(array, pattern, options)?;

        debug_assert_eq!(
            result.len(),
            array.len(),
            "Like length mismatch {}",
            array.encoding().id()
        );
        debug_assert_eq!(
            result.dtype(),
            &DType::Bool((array.dtype().is_nullable() || pattern.dtype().is_nullable()).into()),
            "Like dtype mismatch {}",
            array.encoding().id()
        );

        return Ok(result);
    }

    // Otherwise, we canonicalize into a UTF8 array.
    log::debug!(
        "No like implementation found for encoding {}",
        array.encoding().id(),
    );
    arrow_like(array, pattern, options)
}

/// Implementation of `LikeFn` using the Arrow crate.
pub(crate) fn arrow_like(
    child: &ArrayData,
    pattern: &ArrayData,
    options: LikeOptions,
) -> VortexResult<ArrayData> {
    let nullable = child.dtype().is_nullable();
    let child = Datum::try_from(child.clone())?;
    let pattern = Datum::try_from(pattern.clone())?;

    let array = match (options.negated, options.case_insensitive) {
        (false, false) => arrow_string::like::like(&child, &pattern)?,
        (true, false) => arrow_string::like::nlike(&child, &pattern)?,
        (false, true) => arrow_string::like::ilike(&child, &pattern)?,
        (true, true) => arrow_string::like::nilike(&child, &pattern)?,
    };

    Ok(ArrayData::from_arrow(&array, nullable))
}
