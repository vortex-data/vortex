// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fs;
use std::path::Path;

const MAGIC: &[u8; 8] = b"ONPAIR01";

pub struct Corpus {
    pub source: String,
    pub bytes: Vec<u8>,
    pub offsets_u32: Vec<u32>,
}

impl Corpus {
    pub fn load(path: &Path) -> Result<Self, String> {
        let input = fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        if input.len() < 24 || &input[..8] != MAGIC {
            return Err(format!("{} is not an ONPAIR01 corpus", path.display()));
        }
        let payload_bytes = read_u64(&input[8..16])?;
        let rows = read_u64(&input[16..24])?;
        let rows = usize::try_from(rows).map_err(|_| "row count does not fit usize")?;

        let mut bytes = Vec::with_capacity(
            usize::try_from(payload_bytes).map_err(|_| "payload does not fit usize")?,
        );
        let mut offsets_u32 = Vec::with_capacity(rows + 1);
        offsets_u32.push(0);
        let mut cursor = 24usize;
        for _ in 0..rows {
            let end = cursor.checked_add(4).ok_or("corpus offset overflow")?;
            let len = read_u32(input.get(cursor..end).ok_or("truncated row length")?)? as usize;
            cursor = end;
            let end = cursor.checked_add(len).ok_or("corpus offset overflow")?;
            bytes.extend_from_slice(input.get(cursor..end).ok_or("truncated row payload")?);
            cursor = end;
            offsets_u32.push(u32::try_from(bytes.len()).map_err(|_| "payload exceeds 4 GiB")?);
        }
        if cursor != input.len() {
            return Err("trailing bytes after final corpus row".to_string());
        }
        if bytes.len() as u64 != payload_bytes {
            return Err(format!(
                "header declares {payload_bytes} payload bytes, found {}",
                bytes.len()
            ));
        }

        Ok(Self {
            source: path.display().to_string(),
            bytes,
            offsets_u32,
        })
    }

    pub fn rows(&self) -> usize {
        self.offsets_u32.len() - 1
    }
}

fn read_u64(bytes: &[u8]) -> Result<u64, String> {
    Ok(u64::from_le_bytes(
        bytes.try_into().map_err(|_| "truncated u64")?,
    ))
}

fn read_u32(bytes: &[u8]) -> Result<u32, String> {
    Ok(u32::from_le_bytes(
        bytes.try_into().map_err(|_| "truncated u32")?,
    ))
}
