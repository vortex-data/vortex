// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Helper macros for working with the different variants of [`PrimitiveVector`] and
//! [`PrimitiveVectorMut`].
//!
//! All macros are exported at the crate level with `#[macro_use]`.
//!
//! [`PrimitiveVector`]: crate::PrimitiveVector
//! [`PrimitiveVectorMut`]: crate::PrimitiveVectorMut

/// TODO(connor): Write docs.
#[macro_export]
macro_rules! match_each_pvector {
    ($self:expr, | $vec:ident | $body:block) => {{
        match $self {
            PrimitiveVector::U8(v) => {
                let $vec = v;
                $body
            }
            PrimitiveVector::U16(v) => {
                let $vec = v;
                $body
            }
            PrimitiveVector::U32(v) => {
                let $vec = v;
                $body
            }
            PrimitiveVector::U64(v) => {
                let $vec = v;
                $body
            }
            PrimitiveVector::I8(v) => {
                let $vec = v;
                $body
            }
            PrimitiveVector::I16(v) => {
                let $vec = v;
                $body
            }
            PrimitiveVector::I32(v) => {
                let $vec = v;
                $body
            }
            PrimitiveVector::I64(v) => {
                let $vec = v;
                $body
            }
            PrimitiveVector::F16(v) => {
                let $vec = v;
                $body
            }
            PrimitiveVector::F32(v) => {
                let $vec = v;
                $body
            }
            PrimitiveVector::F64(v) => {
                let $vec = v;
                $body
            }
        }
    }};
}

/// TODO(connor): Write docs.
#[macro_export]
macro_rules! match_each_pvector_mut {
    ($self:expr, | $vec:ident | $body:block) => {{
        match $self {
            PrimitiveVectorMut::U8(v) => {
                let $vec = v;
                $body
            }
            PrimitiveVectorMut::U16(v) => {
                let $vec = v;
                $body
            }
            PrimitiveVectorMut::U32(v) => {
                let $vec = v;
                $body
            }
            PrimitiveVectorMut::U64(v) => {
                let $vec = v;
                $body
            }
            PrimitiveVectorMut::I8(v) => {
                let $vec = v;
                $body
            }
            PrimitiveVectorMut::I16(v) => {
                let $vec = v;
                $body
            }
            PrimitiveVectorMut::I32(v) => {
                let $vec = v;
                $body
            }
            PrimitiveVectorMut::I64(v) => {
                let $vec = v;
                $body
            }
            PrimitiveVectorMut::F16(v) => {
                let $vec = v;
                $body
            }
            PrimitiveVectorMut::F32(v) => {
                let $vec = v;
                $body
            }
            PrimitiveVectorMut::F64(v) => {
                let $vec = v;
                $body
            }
        }
    }};
}
