use vortex_buffer::Buffer;
use vortex_dtype::{DType, PType, match_each_native_ptype};
use vortex_error::VortexResult;

use crate::ToCanonical;
use crate::arrays::PrimitiveArray;
use crate::compute::{cast, min_max};
use crate::validity::Validity;

impl PrimitiveArray {
    pub fn downscale(&self) -> VortexResult<PrimitiveArray> {
        let Some(min_max) = min_max(self.as_ref())? else {
            return Ok(match_each_native_ptype!(self.ptype(), |P| {
                PrimitiveArray::new(Buffer::<P>::zeroed(self.len()), Validity::AllInvalid)
            }));
        };

        // If we can't cast to i64, then leave the array as its original type.
        // It's too big to downcast anyway.
        let Ok(min) = i64::try_from(&min_max.min.cast(&PType::I64.into())?) else {
            return Ok(self.clone());
        };
        let Ok(max) = i64::try_from(&min_max.max.cast(&PType::I64.into())?) else {
            return Ok(self.clone());
        };

        if min < 0 || max < 0 {
            // Signed
            if min >= i8::MIN as i64 && max <= i8::MAX as i64 {
                return cast(
                    self.as_ref(),
                    &DType::Primitive(PType::I8, self.dtype().nullability()),
                )?
                .to_primitive();
            }

            if min >= i16::MIN as i64 && max <= i16::MAX as i64 {
                return cast(
                    self.as_ref(),
                    &DType::Primitive(PType::I16, self.dtype().nullability()),
                )?
                .to_primitive();
            }

            if min >= i32::MIN as i64 && max <= i32::MAX as i64 {
                return cast(
                    self.as_ref(),
                    &DType::Primitive(PType::I32, self.dtype().nullability()),
                )?
                .to_primitive();
            }
        } else {
            // Unsigned
            if max <= u8::MAX as i64 {
                return cast(
                    self.as_ref(),
                    &DType::Primitive(PType::U8, self.dtype().nullability()),
                )?
                .to_primitive();
            }

            if max <= u16::MAX as i64 {
                return cast(
                    self.as_ref(),
                    &DType::Primitive(PType::U16, self.dtype().nullability()),
                )?
                .to_primitive();
            }

            if max <= u32::MAX as i64 {
                return cast(
                    self.as_ref(),
                    &DType::Primitive(PType::U32, self.dtype().nullability()),
                )?
                .to_primitive();
            }
        }

        Ok(self.clone())
    }
}
