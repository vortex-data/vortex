// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Canonical [`ArrowDecoder`] / [`ArrowDTypeReader`] — the default fallback for the reverse
//! (Arrow → Vortex) direction.
//!
//! These plugins should always be the **last** entries in their respective chains: they accept
//! every Arrow type the legacy [`FromArrowArray`] / [`FromArrowType`] implementations support,
//! so anything earlier in the chain that wants to override their behavior must run first.

use arrow_array::Array as ArrowArray;
use arrow_schema::Field;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::arrow::ArrowDTypeReader;
use crate::arrow::ArrowDecoder;
use crate::arrow::FromArrowArray;
use crate::dtype::DType;
use crate::dtype::arrow::FromArrowType;

/// Default [`ArrowDecoder`] that delegates to the legacy
/// [`FromArrowArray`](crate::arrow::FromArrowArray) implementation.
#[derive(Debug, Default)]
pub struct CanonicalArrowDecoder;

impl ArrowDecoder for CanonicalArrowDecoder {
    fn try_decode(
        &self,
        array: &dyn ArrowArray,
        field: &Field,
        _session: &VortexSession,
    ) -> VortexResult<Option<ArrayRef>> {
        Ok(Some(ArrayRef::from_arrow(array, field.is_nullable())?))
    }
}

/// Default [`ArrowDTypeReader`] that delegates to the legacy
/// [`FromArrowType`](crate::dtype::arrow::FromArrowType) implementation. Matches every
/// Arrow [`Field`].
#[derive(Debug, Default)]
pub struct CanonicalArrowDTypeReader;

impl ArrowDTypeReader for CanonicalArrowDTypeReader {
    fn try_read_dtype(&self, field: &Field) -> VortexResult<Option<DType>> {
        Ok(Some(DType::from_arrow(field)))
    }
}
