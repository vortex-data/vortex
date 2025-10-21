// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Helper macros for working with the different variants of [`Vector`](crate::Vector) and
//! [`VectorMut`](crate::VectorMut).
//!
//! All macros are exported at the crate level with `#[macro_use]`.

// TODO(connor): Finish implementing the rest of the macros.

/// TODO(connor): Write docs.
#[macro_export]
macro_rules! match_each_vector {
    ($self:expr, | $vec:ident | $body:block) => {{
        match $self {
            Vector::Null(v) => {
                let $vec = v;
                $body
            }
            Vector::Bool(v) => {
                let $vec = v;
                $body
            }
            Vector::Primitive(v) => {
                let $vec = v;
                $body
            }
        }
    }};
}

/// TODO(connor): Write docs.
#[macro_export]
macro_rules! match_each_vector_mut {
    ($self:expr, | $vec:ident | $body:block) => {{
        match $self {
            VectorMut::Null(v) => {
                let $vec = v;
                $body
            }
            VectorMut::Bool(v) => {
                let $vec = v;
                $body
            }
            VectorMut::Primitive(v) => {
                let $vec = v;
                $body
            }
        }
    }};
}
