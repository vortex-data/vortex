// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::marker::PhantomData;

use serde::de::Visitor;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::{Alignment, Buffer, BufferMut, ByteBuffer};

impl<T> Serialize for Buffer<T>
where
    T: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_bytes(self.inner().as_ref())
    }
}

struct BufferVisitor<T> {
    _marker: PhantomData<T>,
}

impl<T> Default for BufferVisitor<T> {
    fn default() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<'de, T> Visitor<'de> for BufferVisitor<T>
where
    T: Deserialize<'de>,
{
    type Value = Buffer<T>;

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(formatter, "byte buffer")
    }

    fn visit_byte_buf<E>(self, v: Vec<u8>) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        let bytes = ByteBuffer::copy_from_aligned(v, Alignment::of::<T>());
        Ok(Buffer::from_byte_buffer(bytes))
    }

    fn visit_borrowed_bytes<E>(self, v: &'de [u8]) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        let bytes = ByteBuffer::copy_from_aligned(v, Alignment::of::<T>());
        Ok(Buffer::from_byte_buffer(bytes))
    }

    fn visit_bytes<E>(self, v: &[u8]) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        let bytes = ByteBuffer::copy_from_aligned(v, Alignment::of::<T>());
        Ok(Buffer::from_byte_buffer(bytes))
    }

    fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
    where
        A: serde::de::SeqAccess<'de>,
    {
        let mut buffer = seq
            .size_hint()
            .map(|hint| BufferMut::<T>::with_capacity(hint))
            .unwrap_or_default();

        while let Some(e) = seq.next_element()? {
            buffer.push(e);
        }

        Ok(buffer.freeze())
    }
}

impl<'de, T> Deserialize<'de> for Buffer<T>
where
    T: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_byte_buf(BufferVisitor::<T>::default())
    }
}
