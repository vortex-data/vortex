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

/// TODO(connor): Write docs.
#[macro_export]
macro_rules! match_each_pvector_mut_pair {
    ($self:expr, $other:expr, | $vec:ident, $vec_other:ident | $body:block) => {{
        match ($self, $other) {
            (PrimitiveVectorMut::U8(a), PrimitiveVectorMut::U8(b)) => {
                let $vec = a;
                let $vec_other = b;
                $body
            }
            (PrimitiveVectorMut::U16(a), PrimitiveVectorMut::U16(b)) => {
                let $vec = a;
                let $vec_other = b;
                $body
            }
            (PrimitiveVectorMut::U32(a), PrimitiveVectorMut::U32(b)) => {
                let $vec = a;
                let $vec_other = b;
                $body
            }
            (PrimitiveVectorMut::U64(a), PrimitiveVectorMut::U64(b)) => {
                let $vec = a;
                let $vec_other = b;
                $body
            }
            (PrimitiveVectorMut::I8(a), PrimitiveVectorMut::I8(b)) => {
                let $vec = a;
                let $vec_other = b;
                $body
            }
            (PrimitiveVectorMut::I16(a), PrimitiveVectorMut::I16(b)) => {
                let $vec = a;
                let $vec_other = b;
                $body
            }
            (PrimitiveVectorMut::I32(a), PrimitiveVectorMut::I32(b)) => {
                let $vec = a;
                let $vec_other = b;
                $body
            }
            (PrimitiveVectorMut::I64(a), PrimitiveVectorMut::I64(b)) => {
                let $vec = a;
                let $vec_other = b;
                $body
            }
            (PrimitiveVectorMut::F16(a), PrimitiveVectorMut::F16(b)) => {
                let $vec = a;
                let $vec_other = b;
                $body
            }
            (PrimitiveVectorMut::F32(a), PrimitiveVectorMut::F32(b)) => {
                let $vec = a;
                let $vec_other = b;
                $body
            }
            (PrimitiveVectorMut::F64(a), PrimitiveVectorMut::F64(b)) => {
                let $vec = a;
                let $vec_other = b;
                $body
            }
            _ => ::vortex_error::vortex_panic!("Mismatched primitive vector types"),
        }
    }};
}

/// TODO(connor): Write docs.
#[macro_export]
macro_rules! match_each_pvector_mut_immut_pair {
    ($self:expr, $other:expr, | $vec:ident, $vec_other:ident | $body:block) => {{
        match ($self, $other) {
            (PrimitiveVectorMut::U8(a), PrimitiveVector::U8(b)) => {
                let $vec = a;
                let $vec_other = b;
                $body
            }
            (PrimitiveVectorMut::U16(a), PrimitiveVector::U16(b)) => {
                let $vec = a;
                let $vec_other = b;
                $body
            }
            (PrimitiveVectorMut::U32(a), PrimitiveVector::U32(b)) => {
                let $vec = a;
                let $vec_other = b;
                $body
            }
            (PrimitiveVectorMut::U64(a), PrimitiveVector::U64(b)) => {
                let $vec = a;
                let $vec_other = b;
                $body
            }
            (PrimitiveVectorMut::I8(a), PrimitiveVector::I8(b)) => {
                let $vec = a;
                let $vec_other = b;
                $body
            }
            (PrimitiveVectorMut::I16(a), PrimitiveVector::I16(b)) => {
                let $vec = a;
                let $vec_other = b;
                $body
            }
            (PrimitiveVectorMut::I32(a), PrimitiveVector::I32(b)) => {
                let $vec = a;
                let $vec_other = b;
                $body
            }
            (PrimitiveVectorMut::I64(a), PrimitiveVector::I64(b)) => {
                let $vec = a;
                let $vec_other = b;
                $body
            }
            (PrimitiveVectorMut::F16(a), PrimitiveVector::F16(b)) => {
                let $vec = a;
                let $vec_other = b;
                $body
            }
            (PrimitiveVectorMut::F32(a), PrimitiveVector::F32(b)) => {
                let $vec = a;
                let $vec_other = b;
                $body
            }
            (PrimitiveVectorMut::F64(a), PrimitiveVector::F64(b)) => {
                let $vec = a;
                let $vec_other = b;
                $body
            }
            _ => ::vortex_error::vortex_panic!("Mismatched primitive vector types"),
        }
    }};
}
