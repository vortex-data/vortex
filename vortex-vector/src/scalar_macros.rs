// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

/// Macro to match each variant of a `Scalar` enum.
#[macro_export]
macro_rules! match_each_scalar {
    ($self:expr, | $scalar:ident | $body:block) => {{
        match $self {
            $crate::Scalar::Null($scalar) => $body,
            $crate::Scalar::Bool($scalar) => $body,
            $crate::Scalar::Decimal($scalar) => $body,
            $crate::Scalar::Primitive($scalar) => $body,
            $crate::Scalar::String($scalar) => $body,
            $crate::Scalar::Binary($scalar) => $body,
            $crate::Scalar::List($scalar) => $body,
            $crate::Scalar::FixedSizeList($scalar) => $body,
            $crate::Scalar::Struct($scalar) => $body,
        }
    }};
}
