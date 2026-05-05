// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`VarBinArrowEncoder`] — short-circuits VarBin → Arrow byte arrays for offset-based targets.

use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::types::BinaryType;
use arrow_array::types::LargeBinaryType;
use arrow_array::types::LargeUtf8Type;
use arrow_array::types::Utf8Type;
use arrow_schema::DataType;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::array::ArrayId;
use crate::array::ArrayPlugin;
use crate::arrays::VarBin;
use crate::arrays::varbin::VarBinArrayExt;
use crate::arrow::ArrowEncoder;
use crate::arrow::ArrowSession;
use crate::arrow::executor::byte::to_arrow_byte_array;
use crate::dtype::DType;
use crate::dtype::PType;

/// Forward [`ArrowEncoder`] keyed by the [`crate::arrays::VarBin`] [`ArrayId`].
///
/// Handles the four offset-based byte targets ([`DataType::Utf8`], [`DataType::LargeUtf8`],
/// [`DataType::Binary`], [`DataType::LargeBinary`]) without going through
/// [`crate::arrays::VarBinView`]. Returns [`None`] for any other target so the dispatcher
/// falls back to the canonical encoder.
#[derive(Debug, Default)]
pub struct VarBinArrowEncoder;

impl VarBinArrowEncoder {
    /// The encoding [`ArrayId`] this encoder is registered against.
    pub fn array_id() -> ArrayId {
        VarBin.id()
    }
}

impl ArrowEncoder for VarBinArrowEncoder {
    fn preferred_arrow_type(
        &self,
        array: &ArrayRef,
        _session: &ArrowSession,
    ) -> VortexResult<Option<DataType>> {
        let Some(varbin) = array.as_opt::<VarBin>() else {
            return Ok(None);
        };
        let offsets_ptype = PType::try_from(varbin.offsets().dtype())?;
        let use_large = matches!(offsets_ptype, PType::I64 | PType::U64);
        Ok(Some(match (varbin.dtype(), use_large) {
            (DType::Utf8(_), false) => DataType::Utf8,
            (DType::Utf8(_), true) => DataType::LargeUtf8,
            (DType::Binary(_), false) => DataType::Binary,
            (DType::Binary(_), true) => DataType::LargeBinary,
            _ => unreachable!("VarBinArray must have Utf8 or Binary dtype"),
        }))
    }

    fn to_arrow_array(
        &self,
        array: ArrayRef,
        target: &DataType,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrowArrayRef>> {
        match target {
            DataType::Utf8 => to_arrow_byte_array::<Utf8Type>(array, ctx).map(Some),
            DataType::LargeUtf8 => to_arrow_byte_array::<LargeUtf8Type>(array, ctx).map(Some),
            DataType::Binary => to_arrow_byte_array::<BinaryType>(array, ctx).map(Some),
            DataType::LargeBinary => to_arrow_byte_array::<LargeBinaryType>(array, ctx).map(Some),
            _ => Ok(None),
        }
    }
}
