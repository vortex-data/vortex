// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
use vortex::array::ArrayRef;
use vortex::array::Canonical;
use vortex::array::ExecutionCtx;
use vortex::array::IntoArray;
use vortex::array::arrays::VarBinViewArray;
use vortex::array::builtins::ArrayBuiltins;
use vortex::dtype::DType;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;

use crate::exporter::ColumnExporter;
use crate::exporter::ConversionCache;
use crate::exporter::all_invalid;
use crate::exporter::bool;
use crate::exporter::decimal;
use crate::exporter::extension;
use crate::exporter::fixed_size_list;
use crate::exporter::list_view;
use crate::exporter::primitive;
use crate::exporter::struct_;
use crate::exporter::varbinview;

pub(crate) fn new_exporter(
    array: ArrayRef,
    cache: &ConversionCache,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Box<dyn ColumnExporter>> {
    match array.execute::<Canonical>(ctx)? {
        Canonical::Null(_) => Ok(all_invalid::new_exporter()),
        Canonical::Bool(array) => bool::new_exporter(array, ctx),
        Canonical::Primitive(array) => primitive::new_exporter(array, ctx),
        Canonical::Decimal(array) => decimal::new_exporter(array, ctx),
        Canonical::VarBinView(array) => varbinview::new_exporter(array, ctx),
        // DuckDB has no fixed-size binary type, so the values export as BLOBs.
        Canonical::FixedSizeBinary(array) => {
            let binary = DType::Binary(array.dtype().nullability());
            let array = array
                .into_array()
                .cast(binary)?
                .execute::<VarBinViewArray>(ctx)?;
            varbinview::new_exporter(array, ctx)
        }
        Canonical::List(array) => list_view::new_exporter(array, cache, ctx),
        Canonical::Map(_) => vortex_bail!("Map arrays can't be exported to DuckDB"),
        Canonical::FixedSizeList(array) => fixed_size_list::new_exporter(array, cache, ctx),
        Canonical::Struct(array) => struct_::new_exporter(array, cache, ctx),
        Canonical::Union(_) => {
            vortex_bail!("TODO(connor)[Union]: implement DuckDB export for Union arrays")
        }
        Canonical::Extension(ext) => extension::new_exporter(ext, ctx),
        Canonical::Variant(_) => {
            vortex_bail!("Variant arrays can't be exported to DuckDB")
        }
    }
}
