// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;

use crate::ArrayRef;
use crate::EmptyArrayData;
use crate::array::ArrayParts;
use crate::array_slots;
use crate::arrays::Widen;
use crate::arrays::WidenArray;
use crate::dtype::DType;

/// The child slots of a [`WidenArray`].
#[array_slots(Widen)]
pub struct WidenSlots {
    /// The stored values, in an integer type strictly narrower than the logical dtype.
    #[slot(0)]
    pub values: ArrayRef,
}

impl WidenArray {
    /// Creates a `Widen` that presents `values` with the wider logical `dtype`, without reading
    /// any values.
    ///
    /// `values` and `dtype` must both be integer dtypes with the same signedness and nullability,
    /// and `values` must be strictly narrower. When `values` is itself a `Widen`, the new array
    /// stores its values instead, so that `Widen` arrays do not nest.
    pub fn try_new(mut values: ArrayRef, dtype: DType) -> VortexResult<Self> {
        // NB: Check `values` before the flatten. The values of a `Widen` child are narrower than
        // the child, so the check in `try_from_parts` alone would accept a child as wide as `dtype`
        // or wider.
        validate_dtypes(values.dtype(), &dtype)?;

        while let Some(widen) = values.as_opt::<Widen>() {
            values = widen.values().clone();
        }

        let len = values.len();
        Self::try_from_parts(ArrayParts::new(
            Widen,
            dtype,
            len,
            EmptyArrayData,
            WidenSlots { values }.into_slots(),
        ))
    }
}

/// Checks that the dtypes of `values` and `logical` meet the requirements of
/// [`WidenArray::try_new`].
pub(super) fn validate_dtypes(values: &DType, logical: &DType) -> VortexResult<()> {
    let (values_ptype, logical_ptype) = match (values, logical) {
        (DType::Primitive(values_ptype, _), DType::Primitive(logical_ptype, _))
            if values_ptype.is_int() && logical_ptype.is_int() =>
        {
            (*values_ptype, *logical_ptype)
        }
        _ => vortex_bail!("Widen requires integer dtypes, got {values} values for {logical}"),
    };

    vortex_ensure!(
        values_ptype.is_signed_int() == logical_ptype.is_signed_int(),
        "Widen requires values with the signedness of {logical}, got {values}"
    );
    vortex_ensure!(
        values_ptype.byte_width() < logical_ptype.byte_width(),
        "Widen requires values narrower than {logical}, got {values}"
    );
    vortex_ensure!(
        values.nullability() == logical.nullability(),
        "Widen requires values with the nullability of {logical}, got {values}"
    );

    Ok(())
}
