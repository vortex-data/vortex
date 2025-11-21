// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Conversion logic from this "legacy" scalar crate to Vortex Vector scalars.

use vortex_dtype::{
    DType, DecimalType, PrecisionScale, match_each_decimal_value_type, match_each_native_ptype,
};
use vortex_error::VortexExpect;
use vortex_vector::binaryview::{BinaryViewScalar, StringScalar};
use vortex_vector::bool::BoolScalar;
use vortex_vector::decimal::DScalar;
use vortex_vector::null::NullScalar;
use vortex_vector::primitive::PScalar;

use crate::Scalar;

impl Scalar {
    pub fn into_vector(self) -> vortex_vector::Scalar {
        match self.dtype() {
            DType::Null => NullScalar.into(),
            DType::Bool(_) => BoolScalar::new(self.as_bool().value()).into(),
            DType::Primitive(ptype, _) => {
                match_each_native_ptype!(ptype, |T| {
                    PScalar::new(self.as_primitive().typed_value()).into()
                })
            }
            DType::Decimal(dec_dtype, _) => {
                let dscalar = self.as_decimal();
                let dec_type = DecimalType::smallest_decimal_value_type(dec_dtype);
                match_each_decimal_value_type!(dec_type, |D| {
                    let ps = PrecisionScale::<D>::new(dec_dtype.precision(), dec_dtype.scale());
                    DScalar::maybe_new(
                        ps,
                        dscalar
                            .decimal_value()
                            .map(|d| d.cast::<D>().vortex_expect("Failed to cast decimal value")),
                    )
                    .vortex_expect("Failed to create decimal scalar")
                    .into()
                })
            }
            DType::Utf8(_) => StringScalar::new(self.as_utf8().value()).into(),
            DType::Binary(_) => BinaryViewScalar::new(self.as_binary().value()).into(),
            DType::List(..) => {}
            DType::FixedSizeList(..) => {}
            DType::Struct(..) => {}
            DType::Extension(_) => {}
        }
    }
}
