// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::ArrayRef;
use crate::array::ArrayView;
use crate::arrays::ListView;
use crate::arrays::map::Map;
use crate::arrays::map::MapArrayExt;
use crate::arrays::map::compute::rebuild_map_from_array;
use crate::dtype::DType;
use crate::dtype::MapDType;
use crate::executor::ExecutionCtx;
use crate::scalar_fn::fns::cast::CastKernel;
use crate::scalar_fn::fns::cast::CastReduce;

fn prepare_map_cast_target(
    array: ArrayView<'_, Map>,
    dtype: &DType,
) -> VortexResult<Option<(MapDType, DType)>> {
    let Some(target_map_dtype) = dtype.as_map_opt() else {
        return Ok(None);
    };

    if target_map_dtype.keys_sorted() && !array.keys_sorted() {
        vortex_bail!(
            "Cannot cast {} to {dtype}: source does not assert sorted map keys",
            array.dtype()
        );
    }

    let target_entries_dtype = DType::List(
        Arc::new(target_map_dtype.entries_dtype()),
        dtype.nullability(),
    );
    Ok(Some((target_map_dtype.clone(), target_entries_dtype)))
}

impl CastReduce for Map {
    fn cast(array: ArrayView<'_, Self>, dtype: &DType) -> VortexResult<Option<ArrayRef>> {
        let Some((target_map_dtype, target_entries_dtype)) = prepare_map_cast_target(array, dtype)?
        else {
            return Ok(None);
        };

        let Some(entries) = <ListView as CastReduce>::cast(array.entries(), &target_entries_dtype)?
        else {
            return Ok(None);
        };

        rebuild_map_from_array(target_map_dtype, entries).map(Some)
    }
}

impl CastKernel for Map {
    fn cast(
        array: ArrayView<'_, Self>,
        dtype: &DType,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let Some((target_map_dtype, target_entries_dtype)) = prepare_map_cast_target(array, dtype)?
        else {
            return Ok(None);
        };

        let Some(entries) =
            <ListView as CastKernel>::cast(array.entries(), &target_entries_dtype, ctx)?
        else {
            return Ok(None);
        };

        rebuild_map_from_array(target_map_dtype, entries).map(Some)
    }
}
