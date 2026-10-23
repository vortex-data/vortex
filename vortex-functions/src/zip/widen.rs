// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::Widen;
use vortex_array::arrays::widen::WidenArraySlotsExt;
use vortex_array::arrays::widen::compute::cast_keeping_nullability;
use vortex_array::arrays::widen::compute::ptype_holding_both_values;
use vortex_array::arrays::widen::compute::ptype_holding_values_and_scalar;
use vortex_array::arrays::widen::compute::wrap_values;
use vortex_error::VortexResult;

use crate::builtins::ArrayFunctions;
use crate::zip::ZipKernel;

/// Zips the values of a [`Widen`] with another `Widen` or a constant at the type that holds both.
/// The adaptor normalizes either branch to `array` and adjusts the mask accordingly.
impl ZipKernel for Widen {
    fn zip(
        array: ArrayView<'_, Self>,
        if_false: &ArrayRef,
        mask: &ArrayRef,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let (if_false_values, ptype) = if let Some(other) = if_false.as_opt::<Widen>() {
            let Some(ptype) = ptype_holding_both_values(array, other) else {
                return Ok(None);
            };
            (other.values().clone(), ptype)
        } else if let Some(constant) = if_false.as_constant() {
            let Some(ptype) = ptype_holding_values_and_scalar(array, constant) else {
                return Ok(None);
            };
            (if_false.clone(), ptype)
        } else {
            return Ok(None);
        };

        let if_true_values = cast_keeping_nullability(array.values(), ptype)?;
        let if_false_values = cast_keeping_nullability(&if_false_values, ptype)?;

        let zipped = mask.zip(if_true_values, if_false_values)?;
        let dtype = array.dtype().with_nullability(zipped.dtype().nullability());

        wrap_values(zipped, dtype).map(Some)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::ArrayRef;
    use vortex_array::Canonical;
    use vortex_array::ExecutionCtx;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::arrays::DecimalArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::Widen;
    use vortex_array::arrays::WidenArray;
    use vortex_array::arrays::widen::WidenArraySlotsExt;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::DecimalDType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::builtins::ArrayFunctions;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty();
        vortex_array::initialize(&session);
        crate::initialize(&session);
        session
    });

    /// Returns `array` executed to its canonical form.
    fn canonical(array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
        Ok(array.clone().execute::<Canonical>(ctx)?.into_array())
    }

    /// Returns a `Widen` of the `logical` type over `values`, with the nullability of `values`.
    fn widen(values: impl IntoArray, logical: PType) -> WidenArray {
        let values = values.into_array();
        let dtype = DType::Primitive(logical, values.dtype().nullability());
        WidenArray::try_new(values, dtype).vortex_expect("fixture values are narrower integers")
    }

    /// Returns the ptype that stores the values of `array`.
    fn stored_ptype(array: &ArrayRef) -> PType {
        match array.as_opt::<Widen>() {
            Some(widen) => widen.values().dtype().as_ptype(),
            None => array.dtype().as_ptype(),
        }
    }

    /// Nullable `i64` values, all valid, at both ends of the `i8` range.
    fn signed() -> WidenArray {
        let values = [Some(i8::MIN), Some(-1), Some(0), Some(1), Some(i8::MAX)];
        widen(PrimitiveArray::from_option_iter(values), PType::I64)
    }

    /// `i64` values in the `i16` range, with null rows between them.
    fn partially_valid() -> WidenArray {
        let values = [Some(-300i16), None, Some(0), None, Some(300)];
        widen(PrimitiveArray::from_option_iter(values), PType::I64)
    }

    /// A decimal `Widen` over values that fit a narrower decimal.
    fn decimal_widen(validity: Validity) -> VortexResult<ArrayRef> {
        let values = DecimalArray::new(
            buffer![-125i16, 50, 50, 250, 900],
            DecimalDType::new(4, 2),
            validity,
        );
        let dtype = DType::Decimal(DecimalDType::new(18, 2), values.dtype().nullability());
        Ok(WidenArray::try_new(values.into_array(), dtype)?.into_array())
    }

    /// The zip runs on the values at the type that holds both branches. The null mask row takes the
    /// `if_false` branch.
    #[rstest]
    #[case::constant_if_false(
        signed().into_array(),
        ConstantArray::new(1_000i64, 5).into_array(),
    )]
    #[case::constant_if_true(
        ConstantArray::new(1_000i64, 5).into_array(),
        signed().into_array(),
    )]
    #[case::nullable_if_false(
        ConstantArray::new(1_000i64, 5).into_array(),
        partially_valid().into_array(),
    )]
    #[case::widen_both(
        signed().into_array(),
        widen(buffer![1_000i16, 2, 300, -4, 5], PType::I64).into_array(),
    )]
    fn test_zip(#[case] if_true: ArrayRef, #[case] if_false: ArrayRef) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let mask = BoolArray::from_iter([Some(true), None, Some(false), Some(true), Some(false)]);
        let mask = mask.into_array();

        let zipped = mask
            .zip(if_true.clone(), if_false.clone())?
            .execute::<ArrayRef>(&mut ctx)?;

        assert_eq!(stored_ptype(&zipped), PType::I16);
        let expected = mask.zip(
            canonical(&if_true, &mut ctx)?,
            canonical(&if_false, &mut ctx)?,
        )?;
        assert_arrays_eq!(zipped, expected, &mut ctx);

        Ok(())
    }

    #[rstest]
    #[case::true_branch(true, false)]
    #[case::false_branch(false, true)]
    #[case::both_branches(true, true)]
    fn test_decimal_zip_falls_back_to_canonical(
        #[case] widen_if_true: bool,
        #[case] widen_if_false: bool,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let if_true = decimal_widen(Validity::from_iter([true, false, true, true, false]))?;
        let if_false = WidenArray::try_new(
            DecimalArray::from_iter([500i16, -600, 700, -800, 900], DecimalDType::new(4, 2))
                .into_array(),
            DType::Decimal(DecimalDType::new(18, 2), Nullability::NonNullable),
        )?
        .into_array();
        let canonical_true = canonical(&if_true, &mut ctx)?;
        let canonical_false = canonical(&if_false, &mut ctx)?;
        let mask = BoolArray::from_iter([Some(true), Some(true), None, Some(false), Some(true)])
            .into_array();
        let expected = mask
            .zip(canonical_true.clone(), canonical_false.clone())?
            .execute::<Canonical>(&mut ctx)?;
        let if_true = if widen_if_true {
            if_true
        } else {
            canonical_true
        };
        let if_false = if widen_if_false {
            if_false
        } else {
            canonical_false
        };

        let actual = mask
            .zip(if_true, if_false)?
            .execute::<Canonical>(&mut ctx)?;

        assert_arrays_eq!(actual, expected, &mut ctx);

        Ok(())
    }
}
