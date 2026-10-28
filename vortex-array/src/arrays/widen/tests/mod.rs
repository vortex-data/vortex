// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Tests of the `Widen` encoding over integer and decimal values.
//!
//! The named integer fixtures cover signed and unsigned values, each kind of validity and a single
//! row, and the `compute` tests run their operations over them. The `decimal` modules build their
//! own decimal fixtures, and compare each result with the result on the canonical array.

mod aggregates;
mod compute;
mod decimal;
mod decimal_aggregates;
mod decimal_compare;
mod decimal_fallback;
mod encoding;
mod numeric;
mod serde;

use std::sync::LazyLock;

use vortex_buffer::buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::Canonical;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::Widen;
use crate::arrays::WidenArray;
use crate::arrays::widen::WidenArraySlotsExt;
use crate::dtype::DType;
use crate::dtype::DecimalDType;
use crate::dtype::PType;
use crate::validity::Validity;

static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

/// Returns a `Widen` of the `logical` type over `values`, with the nullability of `values`.
fn widen(values: impl IntoArray, logical: PType) -> WidenArray {
    let values = values.into_array();
    let dtype = DType::Primitive(logical, values.dtype().nullability());
    WidenArray::try_new(values, dtype).vortex_expect("fixture values are narrower integers")
}

/// Returns the ptype that stores the values of `array`: the ptype of the values of a `Widen`, and
/// the ptype of `array` otherwise.
fn stored_ptype(array: &ArrayRef) -> PType {
    match array.as_opt::<Widen>() {
        Some(widen) => widen.values().dtype().as_ptype(),
        None => array.dtype().as_ptype(),
    }
}

/// Returns the decimal dtype of the values child, or of `array` when it is not a `Widen`.
fn stored_decimal_dtype(array: &ArrayRef) -> DecimalDType {
    let dtype = match array.as_opt::<Widen>() {
        Some(widen) => widen.values().dtype().as_decimal_opt().copied(),
        None => array.dtype().as_decimal_opt().copied(),
    };

    dtype.vortex_expect("the fixture has a decimal dtype")
}

/// Returns `array` executed to its canonical form.
fn canonical(array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
    Ok(array.clone().execute::<Canonical>(ctx)?.into_array())
}

/// Nullable `i64` values, all valid, at both ends of the `i8` range.
fn signed() -> WidenArray {
    let values = [Some(i8::MIN), Some(-1), Some(0), Some(1), Some(i8::MAX)];
    widen(PrimitiveArray::from_option_iter(values), PType::I64)
}

/// Nullable `u32` values, all valid, at both ends of the `u16` range.
fn unsigned() -> WidenArray {
    let values = [Some(0u16), Some(1), Some(255), Some(256), Some(u16::MAX)];
    widen(PrimitiveArray::from_option_iter(values), PType::U32)
}

/// Non-nullable `i32` values at both ends of the `i16` range.
fn non_nullable() -> WidenArray {
    widen(buffer![i16::MIN, -1, 0, 1, i16::MAX], PType::I32)
}

/// `i64` values in the `i16` range, with null rows between them.
fn partially_valid() -> WidenArray {
    let values = [Some(-300i16), None, Some(0), None, Some(300)];
    widen(PrimitiveArray::from_option_iter(values), PType::I64)
}

/// `i64` rows that are all null, over payloads at both ends of the `i8` range.
fn all_invalid() -> WidenArray {
    let values = PrimitiveArray::new(buffer![i8::MIN, 0, i8::MAX], Validity::AllInvalid);
    widen(values, PType::I64)
}

/// One non-nullable `u64` row at the top of the `u32` range.
fn one_row() -> WidenArray {
    widen(buffer![u32::MAX], PType::U64)
}
