// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Signed integer dtypes wider than the native primitive dtype family.
//!
//! [`WideInteger`] gives 128-bit and 256-bit integers a logical identity independent of decimal
//! precision and scale. The extension storage is a fixed-size list of little-endian bytes, so a
//! native integer buffer can be exposed as canonical storage without copying its values.

use std::fmt;
use std::sync::Arc;

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_session::registry::CachedId;

use crate::dtype::DType;
use crate::dtype::DecimalType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::dtype::extension::ExtDType;
use crate::dtype::extension::ExtId;
use crate::dtype::extension::ExtVTable;
use crate::dtype::i256;
use crate::scalar::PValue;
use crate::scalar::ScalarValue;

/// The width of a signed integer outside the primitive dtype family.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum IntegerWidth {
    /// A signed 128-bit integer.
    I128,
    /// A signed 256-bit integer.
    I256,
}

impl IntegerWidth {
    /// Returns the number of bytes occupied by one integer.
    pub const fn byte_width(self) -> usize {
        match self {
            Self::I128 => 16,
            Self::I256 => 32,
        }
    }

    /// Returns the native signed storage type.
    pub const fn values_type(self) -> DecimalType {
        match self {
            Self::I128 => DecimalType::I128,
            Self::I256 => DecimalType::I256,
        }
    }
}

impl fmt::Display for IntegerWidth {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::I128 => f.write_str("i128"),
            Self::I256 => f.write_str("i256"),
        }
    }
}

/// A signed integer whose width is 128 or 256 bits.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct WideInteger;

impl WideInteger {
    /// Creates an integer dtype with fixed width and the requested nullability.
    pub fn new(width: IntegerWidth, nullability: Nullability) -> ExtDType<Self> {
        ExtDType::try_new(
            width,
            DType::FixedSizeList(
                Arc::new(PType::U8.into()),
                width.byte_width() as u32,
                nullability,
            ),
        )
        .vortex_expect("WideInteger uses its validated fixed-size byte storage")
    }

    /// Returns the width when `dtype` is a built-in wide signed integer.
    pub fn width(dtype: &DType) -> Option<IntegerWidth> {
        dtype.as_extension_opt()?.metadata_opt::<Self>().copied()
    }

    pub(crate) fn unpack_value(
        width: IntegerWidth,
        storage_value: &ScalarValue,
    ) -> VortexResult<i256> {
        let ScalarValue::Tuple(values) = storage_value else {
            vortex_bail!("Expected fixed-size integer bytes, got {storage_value:?}");
        };
        let width = width.byte_width();
        vortex_ensure!(
            values.len() == width,
            "Expected {width} integer bytes, got {}",
            values.len()
        );
        let mut bytes = [0u8; 32];
        for (out, value) in bytes.iter_mut().zip(values) {
            let Some(ScalarValue::Primitive(PValue::U8(value))) = value else {
                return Err(vortex_err!("Expected a non-null integer byte, got {value:?}"));
            };
            *out = *value;
        }
        if width == 16 && bytes[15] & 0x80 != 0 {
            bytes[16..].fill(255);
        }

        Ok(i256::from_le_bytes(bytes))
    }
}

impl ExtVTable for WideInteger {
    type Metadata = IntegerWidth;
    type NativeValue<'a> = i256;

    fn id(&self) -> ExtId {
        static ID: CachedId = CachedId::new("vortex.integer");
        *ID
    }

    fn serialize_metadata(&self, metadata: &IntegerWidth) -> VortexResult<Vec<u8>> {
        Ok(vec![metadata.byte_width() as u8])
    }

    fn deserialize_metadata(&self, metadata: &[u8]) -> VortexResult<IntegerWidth> {
        match metadata {
            [16] => Ok(IntegerWidth::I128),
            [32] => Ok(IntegerWidth::I256),
            _ => vortex_bail!("Expected integer width 16 or 32 bytes, got {metadata:?}"),
        }
    }

    fn validate_dtype(dtype: &ExtDType<Self>) -> VortexResult<()> {
        let expected = DType::FixedSizeList(
            Arc::new(PType::U8.into()),
            dtype.metadata().byte_width() as u32,
            dtype.storage_dtype().nullability(),
        );
        vortex_ensure!(
            dtype.storage_dtype() == &expected,
            "Expected integer storage {expected}, got {}",
            dtype.storage_dtype()
        );

        Ok(())
    }

    fn unpack_native<'a>(
        dtype: &'a ExtDType<Self>,
        storage_value: &'a ScalarValue,
    ) -> VortexResult<i256> {
        Self::unpack_value(*dtype.metadata(), storage_value)
    }
}
