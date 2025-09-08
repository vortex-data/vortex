// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use prost::Message;
use vortex_buffer::{ByteBuffer, ByteBufferMut};
use vortex_error::{VortexExpect, VortexResult, vortex_ensure, vortex_err};

use crate::layouts::dict::bloom::BloomFilter;
use crate::layouts::dict::bloom::sbbf::Sbbf;

#[derive(prost::Message)]
pub struct Header {
    #[prost(oneof = "Filter", tags = "1")]
    pub filter: Option<Filter>,
}

// TODO(aduffy): how to shim in extensible tokenizers? Maybe make it serialize into some format.
#[derive(prost::Oneof)]
#[repr(u8)]
pub enum Filter {
    #[prost(message, tag = "1")]
    Sbbf(SbbfHeader),
}

#[derive(prost::Message)]
pub struct SbbfHeader {
    #[prost(bytes, tag = "1")]
    blocks: prost::bytes::Bytes,
}

/// The selected tokenizer.
#[derive(Debug, prost::Enumeration)]
pub enum SbbfTokenizer {
    None = 0,
    Word = 1,
}

impl BloomFilter {
    /// Serialize a single bloom filter out as a single readable message.
    pub fn serialize(&self) -> ByteBuffer {
        match self {
            BloomFilter::SplitBlockWord(filter) => {
                let header = Header {
                    filter: Some(Filter::Sbbf(SbbfHeader {
                        blocks: filter.serialize().into_inner(),
                    })),
                };

                let header = header.encode_to_vec();
                let mut result = ByteBufferMut::with_capacity(2048);
                // Push an 8-byte length prefix for the protobuf header
                let len_32: u32 = header
                    .len()
                    .try_into()
                    .vortex_expect("header message cannot exceed 4GiB");
                let len_prefix = len_32.to_le_bytes();
                result.extend_from_slice(&len_prefix);
                result.extend_from_slice(&header);

                result.freeze()
            }
        }
    }

    /// Try and deserialize an instance of the BloomFilter from a provided serialized representation.
    ///
    /// Returns the rest of the stream so you can parse out another Bloom Filter, if it is packed
    /// immediately after this one.
    pub fn try_deserialize(bytes: &[u8]) -> VortexResult<(Self, &[u8])> {
        vortex_ensure!(bytes.len() >= 4, "len bytes must be present");

        let (len_bytes, rest) = bytes.split_at(4);
        let header_len =
            u32::from_le_bytes([len_bytes[0], len_bytes[1], len_bytes[2], len_bytes[3]]) as usize;

        vortex_ensure!(
            rest.len() >= header_len,
            "buffer not large enough to read entire message header"
        );

        let (header_proto, rest) = rest.split_at(header_len);
        let header = Header::decode(header_proto)
            .map_err(|err| vortex_err!("Invalid header proto: {err}"))?;

        let filter = header
            .filter
            .ok_or_else(|| vortex_err!("BloomFilter cannot be decoded, Header::filter not set"))?;

        let Filter::Sbbf(sbbf_header) = filter;

        // Build the bloom filter and tokenizer from the serialized copy.
        let sbbf = Sbbf::try_deserialize(sbbf_header.blocks)?;

        let filter = BloomFilter::SplitBlockWord(sbbf);

        Ok((filter, rest))
    }
}
#[cfg(test)]
mod tests {
    use crate::layouts::dict::bloom::BloomFilter;

    #[test]
    fn test_roundtrip() {
        let mut filter = BloomFilter::new_sbbf(32);

        for id in 0..1000 {
            filter.insert(format!("identifier {id}").as_str());
        }

        // serialize the filter out.
        let serialized = filter.serialize();

        // Read the filter back from serialized copy.
        let (deserialized, _) = BloomFilter::try_deserialize(serialized.as_slice()).unwrap();

        // Make sure the deserialized filter contains all the items we inserted!
        for id in 0..1000 {
            assert!(deserialized.check(format!("identifier {id}").as_str()));
        }

        // Prefix hits should work with our default word tokenizer
        assert!(deserialized.check_prefix("identifier"));

        // Make sure it *doesn't* contain a few items we didn't insert
        assert!(!deserialized.check("this was not inserted"));
        assert!(!deserialized.check("nor was this"));
    }
}
