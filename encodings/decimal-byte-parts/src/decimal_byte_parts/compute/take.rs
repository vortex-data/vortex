// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::IntoArray;
use vortex_array::arrays::Widen;
use vortex_array::arrays::WidenArray;
use vortex_array::arrays::dict::TakeReduce;
use vortex_array::arrays::widen::WidenArraySlotsExt;
use vortex_error::VortexResult;

use crate::DecimalByteParts;
use crate::decimal_byte_parts::DecimalBytePartsArrayExt;
use crate::decimal_byte_parts::DecimalBytePartsArraySlotsExt;

impl TakeReduce for DecimalByteParts {
    fn take(array: ArrayView<'_, Self>, indices: &ArrayRef) -> VortexResult<Option<ArrayRef>> {
        // Taking with nullable indices makes every taken part nullable, but lower parts must
        // stay non-nullable — validity belongs to the MSP alone. Fall back to the
        // canonical path rather than rebuilding parts we would have to strip nullability from.
        //
        // TODO(mk): Support lower parts using fill_null for nullable indices.
        if indices.dtype().is_nullable() && !array.lower_parts().is_empty() {
            return Ok(None);
        }

        let msp = take_msp(array.msp(), indices)?;
        let lower_parts = array
            .lower_parts()
            .iter()
            .map(|part| part.take(indices.clone()))
            .collect::<VortexResult<Vec<_>>>()?;

        DecimalByteParts::try_new_with_lower_parts(msp, lower_parts, array.decimal_dtype())
            .map(|array| Some(array.into_array()))
    }
}

/// Takes the rows of `msp` at `indices`, and keeps a [`Widen`] MSP at the top of the slot.
fn take_msp(msp: &ArrayRef, indices: &ArrayRef) -> VortexResult<ArrayRef> {
    let Some(widen) = msp.as_opt::<Widen>() else {
        return msp.take(indices.clone());
    };

    // NB: Take under the `Widen`, so that the slot holds `Widen(Dict)` and not `Dict(Widen)`.
    // `stored_msp` removes a `Widen` only at the top of the MSP slot, because an integer `Widen`
    // has no wire form until #10733. A take on a `Widen` array runs its gather instead of leaving a
    // lazy `Dict` under the `Widen`, because the DuckDB exporter and the dict layout writer look
    // for a `Dict` at the root of an array (see the `vortex_array::arrays::widen` module docs). The
    // MSP slot is never a root, so a lazy `Dict` is fine here.
    let values = widen.values().take(indices.clone())?;
    let dtype = msp.dtype().with_nullability(values.dtype().nullability());

    Ok(WidenArray::try_new(values, dtype)?.into_array())
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::DecimalArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::DecimalDType;
    use vortex_array::dtype::i256;
    use vortex_array::optimizer::ArrayOptimizer;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::DecimalByteParts;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty();
        crate::initialize(&session);
        session
    });

    /// Taking pushes down into the parts during optimization, with no execution context in
    /// play: `ArrayRef::take` wraps the array in a `Dict`, and optimizing it must rewrite that
    /// into a `DecimalByteParts` of taken parts.
    #[test]
    fn take_pushes_down_without_executing() -> VortexResult<()> {
        let decimal = DecimalArray::new(
            Buffer::from(vec![1i128 << 70, 2, 3]),
            DecimalDType::new(38, 2),
            Validity::NonNullable,
        );
        let indices = buffer![0u64, 2].into_array();
        let taken = DecimalByteParts::encode(&decimal, &mut SESSION.create_execution_ctx())?
            .into_array()
            .take(indices)?
            .optimize_ctx(&SESSION)?;

        assert!(
            taken.is::<DecimalByteParts>(),
            "expected the take to reduce into the encoding, got {}",
            taken.encoding_id()
        );
        Ok(())
    }

    /// Taking with nullable indices must still round-trip the wide values, including the
    /// null row, on arrays that carry lower parts.
    #[rstest]
    #[case::one_lower_part(DecimalArray::new(
        Buffer::from(vec![1i128 << 70, 2, 3]),
        DecimalDType::new(38, 2),
        Validity::NonNullable,
    ))]
    #[case::three_lower_parts(DecimalArray::new(
        Buffer::from(vec![i256::from_parts(1 << 70, 1), i256::from_parts(2, 0), i256::from_parts(3, 0)]),
        DecimalDType::new(76, 2),
        Validity::NonNullable,
    ))]
    fn take_with_nullable_indices(#[case] decimal: DecimalArray) -> VortexResult<()> {
        let session = VortexSession::empty();
        crate::initialize(&session);
        let mut ctx = session.create_execution_ctx();

        let indices = PrimitiveArray::from_option_iter([Some(0u64), None, Some(2u64)]).into_array();
        let expected = decimal.clone().into_array().take(indices.clone())?;

        let taken = DecimalByteParts::encode(&decimal, &mut ctx)?
            .into_array()
            .take(indices)?;

        assert_arrays_eq!(taken, expected, &mut ctx);
        Ok(())
    }
}
