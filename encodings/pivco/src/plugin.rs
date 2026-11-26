// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`ArrayPlugin`] serialization and deserialization for `vortex.pivco`.

use std::sync::Arc;

use prost::Message as _;
use vortex_array::ArrayDeserialization;
use vortex_array::ArrayId;
use vortex_array::ArrayPlugin;
use vortex_array::ArrayRef;
use vortex_array::ArraySerialization;
use vortex_array::IntoArray;
use vortex_array::VTable;
use vortex_array::dtype::Nullability;
use vortex_array::validity::Validity;
use vortex_array::vtable::child_to_validity;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

use crate::array::Pivco;
use crate::array::PivcoData;
use crate::array::PivcoMetadata;
use crate::array::PivcoSlots;
use crate::array::pivco_id;
use crate::codec::DEFAULT_BLOCK_SHIFT;
use crate::codec::MAX_BLOCK_SHIFT;
use crate::codec::MIN_BLOCK_SHIFT;
use crate::tree::PivcoTree;

/// [`ArrayPlugin`] implementation for [`Pivco`].
#[derive(Clone, Debug)]
pub struct PivcoPlugin;

impl ArrayPlugin for PivcoPlugin {
    fn id(&self) -> ArrayId {
        VTable::id(&Pivco)
    }

    fn serialize(
        &self,
        array: &ArrayRef,
        _session: &VortexSession,
    ) -> VortexResult<Option<ArraySerialization>> {
        let view = array
            .as_opt::<Pivco>()
            .ok_or_else(|| vortex_err!("PivcoPlugin cannot serialize {}", array.encoding_id()))?;

        let metadata = view.data().to_metadata();
        let children: Vec<ArrayRef> = view.slots()[PivcoSlots::VALIDITY_CHILD]
            .clone()
            .into_iter()
            .collect();

        Ok(Some(ArraySerialization::new(
            pivco_id(),
            metadata.encode_to_vec(),
            array.buffers(),
            children,
        )))
    }

    fn deserialize(
        &self,
        parts: ArrayDeserialization<'_>,
        _session: &VortexSession,
    ) -> VortexResult<ArrayRef> {
        vortex_ensure_eq!(
            parts.serialized_id,
            pivco_id(),
            "PivcoPlugin does not recognize serialized ID {}",
            parts.serialized_id
        );
        let ArrayDeserialization {
            dtype,
            len,
            metadata,
            buffers,
            children,
            ..
        } = parts;

        vortex_ensure!(
            dtype.is_int(),
            "PivcoArray requires an integer dtype, got {dtype}"
        );
        vortex_ensure_eq!(
            buffers.len(),
            3,
            "PivcoArray expects 3 buffers (block_headers, tree_payload, raw_bits), got {}",
            buffers.len()
        );

        let metadata = PivcoMetadata::decode(metadata)?;
        let block_shift = if metadata.block_shift == 0 {
            DEFAULT_BLOCK_SHIFT
        } else {
            u8::try_from(metadata.block_shift)?
        };
        vortex_ensure!(
            (MIN_BLOCK_SHIFT..=MAX_BLOCK_SHIFT).contains(&block_shift),
            "PivcoMetadata block_shift {block_shift} must be in {MIN_BLOCK_SHIFT}..={MAX_BLOCK_SHIFT}"
        );
        let block_values = 1usize << block_shift;
        vortex_ensure!(
            usize::try_from(metadata.offset).is_ok_and(|off| off < block_values),
            "PivcoMetadata offset {} must be < {block_values}",
            metadata.offset
        );
        let offset = u16::try_from(metadata.offset)?;
        vortex_ensure!(
            usize::try_from(metadata.last_block_len).is_ok_and(|n| n <= block_values),
            "PivcoMetadata last_block_len {} must be <= {block_values}",
            metadata.last_block_len
        );
        let last_block_len = u16::try_from(metadata.last_block_len)?;
        if len == 0 {
            vortex_ensure_eq!(offset, 0, "Empty PivcoArray must have offset 0");
            vortex_ensure_eq!(
                last_block_len,
                0,
                "Empty PivcoArray must have last_block_len 0"
            );
        }

        let tree = if len == 0 && metadata.tree_bytes.is_empty() {
            PivcoTree::single_symbol(0)
        } else {
            PivcoTree::from_bytes(&metadata.tree_bytes)?
        };

        let validity = match children.len() {
            0 => Validity::from(dtype.nullability()),
            1 => {
                vortex_ensure!(
                    dtype.is_nullable(),
                    "Non-nullable PivcoArray cannot have a validity child"
                );
                let validity_arr = children.get(0, &Validity::DTYPE, len)?;
                child_to_validity(Some(&validity_arr), Nullability::Nullable)
            }
            other => vortex_bail!("PivcoArray expects 0 or 1 children, got {other}"),
        };

        let data = PivcoData::try_new(
            block_shift,
            offset,
            last_block_len,
            metadata.tree_base_offset,
            metadata.raw_base_word_offset,
            Arc::new(tree),
            buffers[0].clone(),
            buffers[1].clone(),
            buffers[2].clone(),
        )?;
        Ok(Pivco::try_new(dtype.clone(), len, data, validity)?.into_array())
    }
}
