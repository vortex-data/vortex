use flatbuffers::{FlatBufferBuilder, WIPOffset};
use serde::{Deserialize, Serialize};
use vortex_error::{VortexError, VortexExpect as _};
use vortex_flatbuffers::{scalar as fb, WriteFlatBuffer};

use crate::{Scalar, ScalarValue};

impl TryFrom<fb::ScalarValue<'_>> for ScalarValue {
    type Error = VortexError;

    fn try_from(value: fb::ScalarValue<'_>) -> Result<Self, Self::Error> {
        let reader = flexbuffers::Reader::get_root(value.flex().bytes())?;
        Ok(Self::deserialize(reader)?)
    }
}

impl WriteFlatBuffer for Scalar {
    type Target<'a> = fb::Scalar<'a>;

    fn write_flatbuffer<'fb>(
        &self,
        fbb: &mut FlatBufferBuilder<'fb>,
    ) -> WIPOffset<Self::Target<'fb>> {
        let dtype = Some(self.dtype.write_flatbuffer(fbb));
        let value = Some(self.value.write_flatbuffer(fbb));
        fb::Scalar::create(fbb, &fb::ScalarArgs { dtype, value })
    }
}

impl WriteFlatBuffer for ScalarValue {
    type Target<'a> = fb::ScalarValue<'a>;

    fn write_flatbuffer<'fb>(
        &self,
        fbb: &mut FlatBufferBuilder<'fb>,
    ) -> WIPOffset<Self::Target<'fb>> {
        let mut value_se = flexbuffers::FlexbufferSerializer::new();
        self.serialize(&mut value_se)
            .map_err(VortexError::FlexBuffersSerError)
            .vortex_expect("Failed to serialize ScalarValue");
        let flex = Some(fbb.create_vector(value_se.view()));
        fb::ScalarValue::create(fbb, &fb::ScalarValueArgs { flex })
    }
}
