// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_schema::DataType;
use vortex_compute::arrow::IntoArrow;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::arrays::ListVTable;
use crate::arrays::VarBinVTable;

/// Trait for executing a Vortex array to produce an Arrow array.
pub trait ArrowArrayExecutor {
    /// Execute the array to produce an Arrow array.
    ///
    /// If a [`DataType`] is given, the array will be converted to the desired Arrow type.
    fn execute_arrow(
        &self,
        // TODO(ngates): do we even want optional data type? Or do we make it required and tell
        //  users to call `DType::into_arrow` to get a default logical type? I'm inclined to think
        //  the latter is preferable. Although there's a world where the user may want the minimal
        //  conversion to Arrow without knowing what that conversion is. In which case, should
        //  DictionaryArray and non-logical types be supported? Should the user provide a list of
        //  supported Arrow arrays? I dunno...
        data_type: Option<&DataType>,
        session: &VortexSession,
    ) -> VortexResult<ArrowArrayRef>;
}

impl ArrowArrayExecutor for crate::ArrayRef {
    fn execute_arrow(
        &self,
        data_type: &Option<DataType>,
        session: &VortexSession,
    ) -> VortexResult<ArrowArrayRef> {
        match data_type {
            None => {
                // Special-case the Arrow-shaped encodings that are not part of our vector API
                // to avoid unnecessary conversions.
                if let Some(_varbin) = self.as_opt::<VarBinVTable>() {
                    // Convert directly to preferred Arrow VarBin array.
                }
                if let Some(_list) = self.as_opt::<ListVTable>() {
                    // Convert directly to preferred Arrow List array.
                }

                let vector = self.execute(session)?;
                vector.into_arrow()
            }
            // Once we know the target Arrow DataType, how do we get there? Should we allow crates
            // to register Arrow conversion kernels? Should we wrap up the Vortex array in a
            // cast expression and re-run the optimizer? Should we just execute to a vector and
            // then convert?
            Some(dt) => match dt {
                DataType::Null => {}
                DataType::Boolean => {}
                DataType::Int8 => {}
                DataType::Int16 => {}
                DataType::Int32 => {}
                DataType::Int64 => {}
                DataType::UInt8 => {}
                DataType::UInt16 => {}
                DataType::UInt32 => {}
                DataType::UInt64 => {}
                DataType::Float16 => {}
                DataType::Float32 => {}
                DataType::Float64 => {}
                DataType::Timestamp(..) => {}
                DataType::Date32 => {}
                DataType::Date64 => {}
                DataType::Time32(_) => {}
                DataType::Time64(_) => {}
                DataType::Duration(_) => {}
                DataType::Interval(_) => {}
                DataType::Binary => {}
                DataType::FixedSizeBinary(_) => {}
                DataType::LargeBinary => {}
                DataType::BinaryView => {}
                DataType::Utf8 => {}
                DataType::LargeUtf8 => {}
                DataType::Utf8View => {}
                DataType::List(_) => {}
                DataType::ListView(_) => {}
                DataType::FixedSizeList(..) => {}
                DataType::LargeList(_) => {}
                DataType::LargeListView(_) => {}
                DataType::Struct(_) => {}
                DataType::Union(..) => {}
                DataType::Dictionary(..) => {}
                DataType::Decimal32(..) => {}
                DataType::Decimal64(..) => {}
                DataType::Decimal128(..) => {}
                DataType::Decimal256(..) => {}
                DataType::Map(..) => {}
                DataType::RunEndEncoded(..) => {}
            },
        }
    }
}
