// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Operate on native wide integer buffers without converting them to byte lists.
//!
//! Selection preserves element boundaries and validity. Casts and fills use checked signed
//! integer conversions when the required width changes.

use std::ops::Range;

use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use super::WideIntegerArray;
use super::WideIntegerArrayExt;
use super::WideIntegerEncoding;
use crate::ArrayRef;
use crate::ArrayVTable;
use crate::ArrayView;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::Dict;
use crate::arrays::dict::TakeExecuteAdaptor;
use crate::arrays::fixed_width::FixedWidthArray;
use crate::arrays::slice::SliceReduce;
use crate::arrays::slice::SliceReduceAdaptor;
use crate::buffer::BufferHandle;
use crate::dtype::DType;
use crate::dtype::integer::signed_integer_type;
use crate::integer;
use crate::optimizer::kernels::ArrayKernelsExt;
use crate::optimizer::rules::ParentRuleSet;
use crate::scalar::Scalar;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::fns::cast::CastReduce;
use crate::scalar_fn::fns::cast::CastReduceAdaptor;
use crate::scalar_fn::fns::fill_null::FillNull;
use crate::scalar_fn::fns::fill_null::FillNullExecuteAdaptor;
use crate::scalar_fn::fns::fill_null::FillNullKernel;
use crate::scalar_fn::fns::mask::MaskReduce;
use crate::scalar_fn::fns::mask::MaskReduceAdaptor;
use crate::validity::Validity;

pub(super) const RULES: ParentRuleSet<WideIntegerEncoding> = ParentRuleSet::new(&[
    ParentRuleSet::lift(&SliceReduceAdaptor(WideIntegerEncoding)),
    ParentRuleSet::lift(&MaskReduceAdaptor(WideIntegerEncoding)),
    ParentRuleSet::lift(&CastReduceAdaptor(WideIntegerEncoding)),
]);

pub(crate) fn initialize(session: &VortexSession) {
    let kernels = session.kernels();
    kernels.register_execute_parent_kernel(
        Dict.id(),
        WideIntegerEncoding,
        TakeExecuteAdaptor(WideIntegerEncoding),
    );
    kernels.register_execute_parent_kernel(
        FillNull.id(),
        WideIntegerEncoding,
        FillNullExecuteAdaptor(WideIntegerEncoding),
    );
}

impl FixedWidthArray for WideIntegerEncoding {
    fn byte_width(array: ArrayView<'_, Self>) -> usize {
        array.values_type().byte_width()
    }

    fn values(array: ArrayView<'_, Self>) -> ByteBuffer {
        array.buffer_handle().to_host_sync()
    }

    fn with_values(
        array: ArrayView<'_, Self>,
        values: ByteBuffer,
        _len: usize,
        validity: Validity,
    ) -> VortexResult<WideIntegerArray> {
        WideIntegerArray::try_new_handle(
            BufferHandle::new_host(values),
            array.values_type(),
            validity,
        )
    }
}

impl SliceReduce for WideIntegerEncoding {
    fn slice(array: ArrayView<'_, Self>, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        let width = array.values_type().byte_width();
        Ok(Some(
            WideIntegerArray::try_new_handle(
                array
                    .buffer_handle()
                    .slice(range.start * width..range.end * width),
                array.values_type(),
                array.integer_validity().slice(range)?,
            )?
            .into_array(),
        ))
    }
}

impl MaskReduce for WideIntegerEncoding {
    const VALIDITY_IS_METADATA_ONLY: bool = true;

    fn mask(array: ArrayView<'_, Self>, mask: &ArrayRef) -> VortexResult<Option<ArrayRef>> {
        Ok(Some(
            WideIntegerArray::try_new_handle(
                array.buffer_handle().clone(),
                array.values_type(),
                array
                    .integer_validity()
                    .and(Validity::Array(mask.clone()))?,
            )?
            .into_array(),
        ))
    }
}

impl CastReduce for WideIntegerEncoding {
    fn cast(array: ArrayView<'_, Self>, dtype: &DType) -> VortexResult<Option<ArrayRef>> {
        if signed_integer_type(dtype) != Some(array.values_type()) {
            return Ok(None);
        }

        let Some(validity) = array
            .integer_validity()
            .trivially_cast_nullability(dtype.nullability(), array.len())?
        else {
            return Ok(None);
        };

        Ok(Some(
            WideIntegerArray::try_new_handle(
                array.buffer_handle().clone(),
                array.values_type(),
                validity,
            )?
            .into_array(),
        ))
    }
}

impl FillNullKernel for WideIntegerEncoding {
    fn fill_null(
        array: ArrayView<'_, Self>,
        fill_value: &Scalar,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        integer::fill_null(array.array(), fill_value, ctx).map(Some)
    }
}
