use vortex_dtype::DType;
use vortex_error::{vortex_bail, vortex_err, VortexError, VortexResult};

use crate::arrow::{from_arrow_array_with_len, Datum};
use crate::encoding::Encoding;
use crate::Array;

pub trait LikeFn<A> {
    fn like(&self, array: A, pattern: &Array, options: LikeOptions) -> VortexResult<Option<Array>>;
}

impl<E: Encoding> LikeFn<Array> for E
where
    E: LikeFn<E::Array>,
    E::Array: TryFrom<Array, Error = VortexError>,
{
    fn like(
        &self,
        array: Array,
        pattern: &Array,
        options: LikeOptions,
    ) -> VortexResult<Option<Array>> {
        let encoding = array.vtable().clone();
        LikeFn::like(
            encoding
                .as_any()
                .downcast_ref::<E>()
                .ok_or_else(|| vortex_err!("Mismatched encoding"))?,
            <E::Array as TryFrom<Array>>::try_from(array)?,
            pattern,
            options,
        )
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
pub fn like(array: Array, pattern: &Array, options: LikeOptions) -> VortexResult<Array> {
    if !matches!(array.dtype(), DType::Utf8(..)) {
        vortex_bail!("Expected utf8 array, got {}", array.dtype());
    }
    if !matches!(pattern.dtype(), DType::Utf8(..)) {
        vortex_bail!("Expected utf8 pattern, got {}", array.dtype());
    }
    if array.len() != pattern.len() {
        vortex_bail!(
            "Length mismatch lhs len {} ({}) != rhs len {} ({})",
            array.len(),
            array.encoding(),
            pattern.len(),
            pattern.encoding()
        );
    }

    let expected_dtype =
        DType::Bool((array.dtype().is_nullable() || pattern.dtype().is_nullable()).into());
    let array_encoding = array.encoding();

    let result = array
        .vtable()
        .like_fn()
        .and_then(|f| f.like(array.clone(), pattern, options).transpose())
        .unwrap_or_else(|| {
            // Otherwise, we canonicalize into a UTF8 array.
            log::debug!(
                "No like implementation found for encoding {}",
                array.encoding(),
            );
            arrow_like(array, pattern, options)
        })?;

    debug_assert_eq!(
        result.len(),
        pattern.len(),
        "Like length mismatch {}",
        array_encoding
    );
    debug_assert_eq!(
        result.dtype(),
        &expected_dtype,
        "Like dtype mismatch {}",
        array_encoding
    );

    Ok(result)
}

/// Implementation of `LikeFn` using the Arrow crate.
pub(crate) fn arrow_like(
    array: Array,
    pattern: &Array,
    options: LikeOptions,
) -> VortexResult<Array> {
    let nullable = array.dtype().is_nullable();
    let len = array.len();
    debug_assert_eq!(
        array.len(),
        pattern.len(),
        "Arrow Like: length mismatch for {}",
        array.encoding()
    );
    let lhs = Datum::try_new(array)?;
    let rhs = Datum::try_new(pattern.clone())?;

    let result = match (options.negated, options.case_insensitive) {
        (false, false) => arrow_string::like::like(&lhs, &rhs)?,
        (true, false) => arrow_string::like::nlike(&lhs, &rhs)?,
        (false, true) => arrow_string::like::ilike(&lhs, &rhs)?,
        (true, true) => arrow_string::like::nilike(&lhs, &rhs)?,
    };

    from_arrow_array_with_len(&result, len, nullable)
}
