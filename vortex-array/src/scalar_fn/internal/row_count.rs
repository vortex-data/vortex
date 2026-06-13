// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::LazyLock;

use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_session::registry::CachedId;

use crate::ArrayRef;
use crate::arrays::ScalarFn;
use crate::arrays::scalar_fn::ExactScalarFn;
use crate::arrays::scalar_fn::ScalarFnArrayExt;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::expr::BoundExpr;
use crate::expr::placeholder::Placeholder;
use crate::expr::placeholder::PlaceholderId;
use crate::expr::placeholder::PlaceholderRef;
use crate::scalar_fn::internal::placeholder::PlaceholderFn;

static ROW_COUNT_DTYPE: LazyLock<DType> =
    LazyLock::new(|| DType::Primitive(PType::U64, Nullability::NonNullable));

/// Placeholder for the row count of the current evaluation scope.
#[derive(Clone, Debug)]
pub struct RowCount;

impl Placeholder for RowCount {
    type Payload = ();

    fn id(&self) -> PlaceholderId {
        static ID: CachedId = CachedId::new("vortex.row_count");
        *ID
    }

    fn dtype(&self) -> &DType {
        &ROW_COUNT_DTYPE
    }

    fn display_name(&self) -> &str {
        "row_count"
    }

    fn payload(&self) -> &Self::Payload {
        &()
    }
}

/// Returns the row-count placeholder reference.
pub fn row_count_ref() -> PlaceholderRef {
    PlaceholderRef::new(RowCount)
}

/// Returns the row-count placeholder expression.
pub fn row_count() -> BoundExpr {
    BoundExpr::Placeholder(row_count_ref())
}

fn is_row_count(placeholder: &PlaceholderRef) -> bool {
    placeholder.id() == RowCount.id() && placeholder.dtype() == RowCount.dtype()
}

/// Returns whether `array` contains a row-count placeholder.
pub fn contains_row_count(array: &ArrayRef) -> bool {
    if let Some(view) = array.as_opt::<ExactScalarFn<PlaceholderFn>>() {
        return is_row_count(view.options);
    }
    match array.as_opt::<ScalarFn>() {
        Some(view) => view.iter_children().any(contains_row_count),
        None => false,
    }
}

/// Replaces every row-count placeholder with `replacement`.
pub fn substitute_row_count(array: ArrayRef, replacement: &ArrayRef) -> VortexResult<ArrayRef> {
    substitute_placeholders(array, &|placeholder| {
        is_row_count(placeholder).then(|| replacement.clone())
    })
}

/// Replaces placeholders resolved by `resolve`.
///
/// Returning `None` from `resolve` leaves the placeholder unresolved, so execution will fail with
/// the placeholder's own error if no later pass resolves it.
pub fn substitute_placeholders(
    array: ArrayRef,
    resolve: &dyn Fn(&PlaceholderRef) -> Option<ArrayRef>,
) -> VortexResult<ArrayRef> {
    if let Some(view) = array.as_opt::<ExactScalarFn<PlaceholderFn>>()
        && let Some(replacement) = resolve(view.options)
    {
        vortex_ensure!(
            replacement.len() == array.len(),
            "Placeholder replacement length {} does not match scope length {}",
            replacement.len(),
            array.len(),
        );
        vortex_ensure!(
            replacement.dtype() == array.dtype(),
            "Placeholder replacement dtype {} does not match scope dtype {}",
            replacement.dtype(),
            array.dtype(),
        );
        return Ok(replacement);
    }

    if !array.is::<ScalarFn>() {
        return Ok(array);
    }

    let nchildren = array.nchildren();
    let mut array = array;
    for slot_idx in 0..nchildren {
        // SAFETY: `substitute_placeholders` always returns an array with the same dtype and
        // length as its input. Placeholders are replaced with checked arrays, and ScalarFn
        // recursion preserves both by operating on each slot in place.
        let (taken, child) = unsafe { array.take_slot_unchecked(slot_idx)? };
        let new_child = substitute_placeholders(child, resolve)?;
        array = unsafe { taken.put_slot_unchecked(slot_idx, new_child)? };
    }
    Ok(array)
}

#[cfg(test)]
mod tests {
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::scalar_fn::internal::row_count::row_count;

    #[test]
    fn row_count_helper_dtype() {
        let expr = row_count();
        assert_eq!(
            expr.dtype(),
            &DType::Primitive(PType::U64, Nullability::NonNullable),
        );
    }
}
