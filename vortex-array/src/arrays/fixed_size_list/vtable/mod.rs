// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::{DType, Nullability};
use vortex_scalar::Scalar;

use crate::arrays::FixedSizeListArray;
use crate::stats::StatsSetRef;
use crate::vtable::{
    ArrayVTable, CanonicalVTable, NotSupported, OperationsVTable, VTable, ValidityHelper,
    ValidityVTableFromValidityHelper,
};
use crate::{Array, ArrayRef, Canonical, EncodingId, EncodingRef, IntoArray, vtable};

mod array;
mod canonical;
mod operations;
mod serde;
mod validity;
mod visitor;

vtable!(FixedSizeList);

#[derive(Clone, Debug)]
pub struct FixedSizeListEncoding;

impl VTable for FixedSizeListVTable {
    type Array = FixedSizeListArray;
    type Encoding = FixedSizeListEncoding;

    type ArrayVTable = Self;
    type CanonicalVTable = Self;
    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromValidityHelper;
    type VisitorVTable = Self;
    type ComputeVTable = NotSupported;
    type EncodeVTable = NotSupported;
    type PipelineVTable = NotSupported;
    type SerdeVTable = Self;

    fn id(_encoding: &Self::Encoding) -> EncodingId {
        EncodingId::new_ref("vortex.list")
    }

    fn encoding(_array: &Self::Array) -> EncodingRef {
        EncodingRef::new_ref(FixedSizeListEncoding.as_ref())
    }
}
