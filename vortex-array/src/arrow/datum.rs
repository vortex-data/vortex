// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use arrow_array::Array as ArrowArray;
use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::Datum as ArrowDatum;
use arrow_schema::DataType;
use arrow_schema::Field;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_panic;

use crate::ArrayRef;
use crate::IntoArray;
use crate::LEGACY_SESSION;
use crate::VortexSessionExecute;
use crate::arrays::Constant;
use crate::arrays::ConstantArray;
use crate::arrow::ArrowSessionExt;
use crate::arrow::FromArrowArray;
use crate::executor::ExecutionCtx;

/// A wrapper around a generic Arrow array that can be used as a Datum in Arrow compute.
#[derive(Debug)]
pub struct Datum {
    array: ArrowArrayRef,
    is_scalar: bool,
}

impl Datum {
    /// Create a new [`Datum`] from an [`ArrayRef`], which can then be passed to Arrow compute.
    pub fn try_new(array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        let session = ctx.session().clone();
        if array.is::<Constant>() {
            Ok(Self {
                array: session
                    .arrow()
                    .execute_arrow(array.slice(0..1)?, None, ctx)?,
                is_scalar: true,
            })
        } else {
            Ok(Self {
                array: session.arrow().execute_arrow(array.clone(), None, ctx)?,
                is_scalar: false,
            })
        }
    }

    /// Create a new [`Datum`] from an `DynArrayData`, which can then be passed to Arrow compute.
    /// This not try and convert the array to a scalar if it is constant.
    pub fn try_new_array(array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        let session = ctx.session().clone();
        Ok(Self {
            array: session.arrow().execute_arrow(array.clone(), None, ctx)?,
            is_scalar: false,
        })
    }

    pub fn try_new_with_target_datatype(
        array: &ArrayRef,
        target_datatype: &DataType,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Self> {
        let session = ctx.session().clone();
        let target_field = Field::new(
            String::new(),
            target_datatype.clone(),
            array.dtype().is_nullable(),
        );

        if array.is::<Constant>() {
            Ok(Self {
                array: session.arrow().execute_arrow(
                    array.slice(0..1)?,
                    Some(&target_field),
                    ctx,
                )?,
                is_scalar: true,
            })
        } else {
            Ok(Self {
                array: session
                    .arrow()
                    .execute_arrow(array.clone(), Some(&target_field), ctx)?,
                is_scalar: false,
            })
        }
    }

    pub fn data_type(&self) -> &DataType {
        self.array.data_type()
    }
}

impl ArrowDatum for Datum {
    fn get(&self) -> (&dyn ArrowArray, bool) {
        (&self.array, self.is_scalar)
    }
}

/// Convert an Arrow array to an Array with a specific length.
/// This is useful for compute functions that delegate to Arrow using [Datum],
/// which will return a scalar (length 1 Arrow array) if the input array is constant.
///
/// # Error
///
/// The provided array must have length
#[deprecated(
    note = "Relies on the hidden global `LEGACY_SESSION`; use `from_arrow_columnar` with an explicit `ExecutionCtx` instead"
)]
pub fn from_arrow_array_with_len<A>(array: A, len: usize, nullable: bool) -> VortexResult<ArrayRef>
where
    ArrayRef: FromArrowArray<A>,
{
    let array = ArrayRef::from_arrow(array, nullable)?;
    if array.len() == len {
        return Ok(array);
    }

    if array.len() != 1 {
        vortex_panic!(
            "Array length mismatch, expected {} got {} for encoding {}",
            len,
            array.len(),
            array.encoding_id()
        );
    }

    Ok(ConstantArray::new(
        array
            .execute_scalar(0, &mut LEGACY_SESSION.create_execution_ctx())
            .vortex_expect("array of length 1 must support execute_scalar(0)"),
        len,
    )
    .into_array())
}

/// Convert an Arrow array to an Array with a specific length, using the provided
/// [`ExecutionCtx`].
///
/// This is useful for compute functions that delegate to Arrow using [Datum],
/// which will return a scalar (length 1 Arrow array) if the input array is constant.
///
/// # Error
///
/// The provided array must have length `len` or `1`.
pub fn from_arrow_columnar<A>(
    array: A,
    len: usize,
    nullable: bool,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef>
where
    ArrayRef: FromArrowArray<A>,
{
    let array = ArrayRef::from_arrow(array, nullable)?;
    if array.len() == len {
        return Ok(array);
    }

    if array.len() != 1 {
        vortex_panic!(
            "Array length mismatch, expected {} got {} for encoding {}",
            len,
            array.len(),
            array.encoding_id()
        );
    }

    Ok(ConstantArray::new(array.execute_scalar(0, ctx)?, len).into_array())
}
