// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex::array::Canonical;
use vortex::array::ExecutionCtx;
use vortex::array::arrays::ExtensionArray;
use vortex::array::arrays::extension::ExtensionArrayExt;
use vortex::buffer::Buffer;
use vortex::error::VortexResult;
use vortex::error::vortex_ensure;
use vortex::mask::Mask;

use crate::duckdb::VectorRef;
use crate::exporter::ColumnExporter;
use crate::exporter::all_invalid;
use crate::exporter::validity;

const UUID_BYTE_LEN: usize = 16;

struct UuidExporter {
    /// UUID_BYTE_LEN big-engian bytes
    bytes: Buffer<u8>,
}

pub(crate) fn new_exporter(
    ext: ExtensionArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Box<dyn ColumnExporter>> {
    let len = ext.len();
    let storage = ext
        .storage_array()
        .clone()
        .execute::<Canonical>(ctx)?
        .into_fixed_size_list();
    let parts = storage.into_data_parts();

    if parts.validity.definitely_all_null() {
        return Ok(all_invalid::new_exporter());
    }
    let mask = parts.validity.to_array(len).execute::<Mask>(ctx)?;

    let bytes = parts
        .elements
        .execute::<Canonical>(ctx)?
        .into_primitive()
        .to_buffer::<u8>();
    vortex_ensure!(
        bytes.len() == len * UUID_BYTE_LEN,
        "UUID storage has {} bytes, expected {}",
        bytes.len(),
        len * UUID_BYTE_LEN
    );

    Ok(validity::new_exporter(
        mask,
        Box::new(UuidExporter { bytes }),
    ))
}

impl ColumnExporter for UuidExporter {
    fn export(
        &self,
        offset: usize,
        len: usize,
        vector: &mut VectorRef,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        let src = &self.bytes[offset * UUID_BYTE_LEN..(offset + len) * UUID_BYTE_LEN];
        let dest = unsafe { vector.as_slice_mut::<i128>(len) };

        for (chunk, out) in src.as_chunks::<UUID_BYTE_LEN>().0.iter().zip(dest) {
            let mut be_bytes = [0u8; UUID_BYTE_LEN];
            be_bytes.copy_from_slice(chunk);
            let be = u128::from_be_bytes(be_bytes);
            *out = (be ^ (1u128 << 127)) as i128;
        }

        Ok(())
    }
}

#[cfg(test)]
#[expect(clippy::cast_possible_truncation, reason = "test-only hex parsing")]
mod tests {
    use vortex::array::IntoArray as _;
    use vortex::array::VortexSessionExecute as _;
    use vortex::array::arrays::ExtensionArray;
    use vortex::array::arrays::FixedSizeListArray;
    use vortex::array::arrays::PrimitiveArray;
    use vortex::array::validity::Validity;
    use vortex::dtype::extension::ExtDType;
    use vortex::encodings::uuid::Uuid;
    use vortex::encodings::uuid::UuidMetadata;

    use super::*;
    use crate::SESSION;
    use crate::cpp;
    use crate::duckdb::DataChunk;
    use crate::duckdb::LogicalType;

    fn parse_uuid(uuid: &str) -> [u8; UUID_BYTE_LEN] {
        let hex: Vec<u8> = uuid
            .chars()
            .filter(|c| *c != '-')
            .map(|c| c.to_digit(16).expect("hex digit") as u8)
            .collect();
        assert_eq!(hex.len(), UUID_BYTE_LEN * 2);
        let mut bytes = [0u8; UUID_BYTE_LEN];
        for (i, byte) in bytes.iter_mut().enumerate() {
            *byte = (hex[2 * i] << 4) | hex[2 * i + 1];
        }
        bytes
    }

    fn uuid_array(uuids: &[Option<&str>]) -> ExtensionArray {
        let mut bytes = Vec::with_capacity(uuids.len() * UUID_BYTE_LEN);
        for uuid in uuids {
            bytes.extend_from_slice(&parse_uuid(
                uuid.unwrap_or("00000000-0000-0000-0000-000000000000"),
            ));
        }
        let validity = Validity::from_iter(uuids.iter().map(Option::is_some));
        let storage = FixedSizeListArray::new(
            PrimitiveArray::from_iter(bytes).into_array(),
            UUID_BYTE_LEN as u32,
            validity,
            uuids.len(),
        )
        .into_array();
        let ext_dtype =
            ExtDType::try_with_vtable(Uuid, UuidMetadata::default(), storage.dtype().clone())
                .expect("valid uuid storage")
                .erased();
        ExtensionArray::new(ext_dtype, storage)
    }

    #[test]
    fn test_uuid_exporter() {
        let arr = uuid_array(&[
            Some("550e8400-e29b-41d4-a716-446655440000"),
            Some("00000000-0000-0000-0000-000000000000"),
            Some("ffffffff-ffff-ffff-ffff-ffffffffffff"),
        ]);

        let mut chunk = DataChunk::new([LogicalType::new(cpp::duckdb_type::DUCKDB_TYPE_UUID)]);
        let mut ctx = SESSION.create_execution_ctx();

        new_exporter(arr, &mut ctx)
            .unwrap()
            .export(0, 3, chunk.get_vector_mut(0), &mut ctx)
            .unwrap();
        chunk.set_len(3);

        assert_eq!(
            String::try_from(&*chunk).unwrap(),
            r#"Chunk - [1 Columns]
- FLAT UUID: 3 = [ 550e8400-e29b-41d4-a716-446655440000, 00000000-0000-0000-0000-000000000000, ffffffff-ffff-ffff-ffff-ffffffffffff]
"#
        );
    }

    #[test]
    fn test_uuid_exporter_with_nulls() {
        let arr = uuid_array(&[
            Some("550e8400-e29b-41d4-a716-446655440000"),
            None,
            Some("ffffffff-ffff-ffff-ffff-ffffffffffff"),
        ]);

        let mut chunk = DataChunk::new([LogicalType::new(cpp::duckdb_type::DUCKDB_TYPE_UUID)]);
        let mut ctx = SESSION.create_execution_ctx();

        new_exporter(arr, &mut ctx)
            .unwrap()
            .export(0, 3, chunk.get_vector_mut(0), &mut ctx)
            .unwrap();
        chunk.set_len(3);

        assert_eq!(
            String::try_from(&*chunk).unwrap(),
            r#"Chunk - [1 Columns]
- FLAT UUID: 3 = [ 550e8400-e29b-41d4-a716-446655440000, NULL, ffffffff-ffff-ffff-ffff-ffffffffffff]
"#
        );
    }
}
