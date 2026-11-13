// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::array::ArrayView;
use crate::arrays::ListView;
use crate::arrays::map::Map;
use crate::arrays::map::MapArrayExt;
use crate::arrays::map::MapArraySlotsExt;
use crate::arrays::map::compute::rebuild_map_from_array;
use crate::scalar_fn::fns::zip::ZipKernel;

/// Zips the list-view entries of two maps and wraps the result in the map type.
impl ZipKernel for Map {
    fn zip(
        if_true: ArrayView<'_, Map>,
        if_false: &ArrayRef,
        mask: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let Some(if_false) = if_false.as_opt::<Map>() else {
            return Ok(None);
        };

        // The zipped entries keep the key and value dtypes of `if_true`, so maps whose nested
        // nullability differs take the generic path.
        if if_true.map_dtype() != if_false.map_dtype() {
            return Ok(None);
        }

        let Some(entries) = <ListView as ZipKernel>::zip(
            if_true.entries().as_::<ListView>(),
            if_false.entries(),
            mask,
            ctx,
        )?
        else {
            return Ok(None);
        };

        rebuild_map_from_array(if_true.map_dtype().clone(), entries).map(Some)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_error::VortexResult;
    use vortex_mask::Mask;
    use vortex_session::VortexSession;

    use crate::ArrayRef;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::arrays::Map;
    use crate::assert_arrays_eq;
    use crate::builders::builder_with_capacity_in;
    use crate::builtins::ArrayBuiltins;
    use crate::dtype::DType;
    use crate::dtype::MapDType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::memory::BufferAllocatorRef;
    use crate::scalar::Scalar;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    fn map_dtype() -> VortexResult<MapDType> {
        MapDType::try_new(
            DType::Primitive(PType::I32, Nullability::NonNullable),
            DType::Utf8(Nullability::Nullable),
            false,
        )
    }

    /// Map `i` holds `i % 3` entries `{offset + j: "v{offset + j}"}`, or is null for every fourth
    /// row when `nullability` is nullable.
    fn map(offset: i32, i: i32, nullability: Nullability) -> VortexResult<Scalar> {
        let dtype = DType::Map(map_dtype()?, nullability);
        if nullability.is_nullable() && i % 4 == 0 {
            return Ok(Scalar::null(dtype));
        }

        Scalar::try_map(
            dtype,
            (0..i % 3).map(|j| {
                let key = offset + j;
                (
                    Scalar::from(key),
                    Scalar::utf8(format!("v{key}"), Nullability::Nullable),
                )
            }),
        )
    }

    fn maps(
        nullability: Nullability,
        scalars: impl IntoIterator<Item = VortexResult<Scalar>>,
    ) -> VortexResult<ArrayRef> {
        let mut builder = builder_with_capacity_in(
            &DType::Map(map_dtype()?, nullability),
            0,
            BufferAllocatorRef::static_ref(),
        );
        for scalar in scalars {
            builder.append_scalar(&scalar?)?;
        }
        Ok(builder.finish())
    }

    /// A fragmented mask selects whole maps, including null maps, from the chosen side.
    #[rstest]
    #[case::non_nullable(Nullability::NonNullable, Nullability::NonNullable)]
    #[case::nullable_if_true(Nullability::Nullable, Nullability::NonNullable)]
    #[case::nullable_if_false(Nullability::NonNullable, Nullability::Nullable)]
    fn zip_maps(
        #[case] true_nullability: Nullability,
        #[case] false_nullability: Nullability,
    ) -> VortexResult<()> {
        let len = 20;
        let selected = |i: i32| i % 2 == 1 || i == 4;
        let nullability = true_nullability | false_nullability;

        let if_true = maps(
            true_nullability,
            (0..len).map(|i| map(0, i, true_nullability)),
        )?;
        let if_false = maps(
            false_nullability,
            (0..len).map(|i| map(100, i, false_nullability)),
        )?;
        let mask = Mask::from_iter((0..len).map(selected));

        let mut ctx = SESSION.create_execution_ctx();
        let result = mask
            .into_array()
            .zip(if_true, if_false)?
            .execute::<ArrayRef>(&mut ctx)?;
        assert!(result.is::<Map>());

        let expected = maps(
            nullability,
            (0..len).map(|i| {
                let selected_map = if selected(i) {
                    map(0, i, true_nullability)?
                } else {
                    map(100, i, false_nullability)?
                };
                selected_map.cast(&DType::Map(map_dtype()?, nullability))
            }),
        )?;
        assert_arrays_eq!(result, expected, &mut ctx);
        Ok(())
    }
}
