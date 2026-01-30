// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::BufferString;
use vortex_buffer::ByteBuffer;
use vortex_dtype::DType;

use crate::DecimalValue;
use crate::PValue;

pub enum Scalar {
    Null(NullScalar),
    Bool(BoolScalar),
    Primitive(PrimitiveScalar),
    Decimal(DecimalScalar),
    Utf8(Utf8Scalar),
    Binary(BinaryScalar),
    List(ListScalar),
    FixedSizeList(FixedSizeListScalar),
    Struct(StructScalar),
    Extension(ExtensionScalar),
}

/// Macro to match each variant of a `Scalar` enum.
macro_rules! match_each_scalar {
    ($self:expr, | $scalar:ident | $body:block) => {{
        match $self {
            $crate::v2::Scalar::Null($scalar) => $body,
            $crate::v2::Scalar::Bool($scalar) => $body,
            $crate::v2::Scalar::Decimal($scalar) => $body,
            $crate::v2::Scalar::Primitive($scalar) => $body,
            $crate::v2::Scalar::Utf8($scalar) => $body,
            $crate::v2::Scalar::Binary($scalar) => $body,
            $crate::v2::Scalar::List($scalar) => $body,
            $crate::v2::Scalar::FixedSizeList($scalar) => $body,
            $crate::v2::Scalar::Struct($scalar) => $body,
            $crate::v2::Scalar::Extension($scalar) => $body,
        }
    }};
}

impl Scalar {
    pub fn dtype(&self) -> &DType {
        match_each_scalar!(self, |s| { s.dtype() })
    }
}

pub struct NullScalar;
impl NullScalar {
    pub fn dtype(&self) -> &DType {
        &DType::Null
    }
}

pub struct BoolScalar {
    dtype: DType,
    value: Option<bool>,
}
impl BoolScalar {
    pub fn dtype(&self) -> &DType {
        &self.dtype
    }
}

pub struct PrimitiveScalar {
    dtype: DType,
    value: Option<PValue>,
}
impl PrimitiveScalar {
    pub fn dtype(&self) -> &DType {
        &self.dtype
    }
}

pub struct DecimalScalar {
    dtype: DType,
    value: Option<DecimalValue>,
}
impl DecimalScalar {
    pub fn dtype(&self) -> &DType {
        &self.dtype
    }
}

pub struct Utf8Scalar {
    dtype: DType,
    value: Option<BufferString>,
}
impl Utf8Scalar {
    pub fn dtype(&self) -> &DType {
        &self.dtype
    }
}

pub struct BinaryScalar {
    dtype: DType,
    value: Option<ByteBuffer>,
}
impl BinaryScalar {
    pub fn dtype(&self) -> &DType {
        &self.dtype
    }
}

pub struct ListScalar {
    dtype: DType,
    // TODO(ngates): replace with ArrayRef when we move into vortex-array crate
    value: Option<Vec<Scalar>>,
}
impl ListScalar {
    pub fn dtype(&self) -> &DType {
        &self.dtype
    }
}

pub struct FixedSizeListScalar {
    dtype: DType,
    // TODO(ngates): replace with ArrayRef when we move into vortex-array crate
    value: Option<Vec<Scalar>>,
}
impl FixedSizeListScalar {
    pub fn dtype(&self) -> &DType {
        &self.dtype
    }
}

pub struct StructScalar {
    dtype: DType,
    // TODO(ngates): replace with StructArray when we move into vortex-array crate
    value: Option<Vec<Scalar>>,
}
impl StructScalar {
    pub fn dtype(&self) -> &DType {
        &self.dtype
    }
}

pub struct ExtensionScalar {
    dtype: DType,
    storage: Box<Scalar>,
}
impl ExtensionScalar {
    pub fn dtype(&self) -> &DType {
        &self.dtype
    }
}
