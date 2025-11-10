// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use vortex_buffer::BitBuffer;

use crate::MaskValues;

impl Serialize for MaskValues {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.buffer.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for MaskValues {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let buffer = BitBuffer::deserialize(deserializer)?;
        Ok(MaskValues::from_buffer(buffer))
    }
}
