// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::arrays::ConstantArray;
use crate::vtable::{NotSupported, VTable};
use crate::{EncodingId, EncodingRef, vtable};

mod array;
mod canonical;
mod encode;
mod operations;
mod operator;
mod serde;
mod validity;
mod visitor;

vtable!(Constant);

#[derive(Clone, Debug)]
pub struct ConstantEncoding;

impl VTable for ConstantVTable {
    type Array = ConstantArray;
    type Encoding = ConstantEncoding;

    type ArrayVTable = Self;
    type CanonicalVTable = Self;
    type OperationsVTable = Self;
    type ValidityVTable = Self;
    type VisitorVTable = Self;
    // TODO(ngates): implement a compute kernel for elementwise operations
    type ComputeVTable = NotSupported;
    type EncodeVTable = Self;
    type OperatorVTable = Self;
    type SerdeVTable = Self;

    fn id(_encoding: &Self::Encoding) -> EncodingId {
        EncodingId::new_ref("vortex.constant")
    }

    fn encoding(_array: &Self::Array) -> EncodingRef {
        EncodingRef::new_ref(ConstantEncoding.as_ref())
    }
}
