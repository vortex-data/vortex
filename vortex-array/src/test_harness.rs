use std::io::Write;

use goldenfile::Mint;
use goldenfile::differs::binary_diff;
use itertools::Itertools;
use vortex_error::VortexResult;

use crate::arrays::BoolArray;
use crate::{DeserializeMetadata, SerializeMetadata};

/// Check that a named metadata matches its previous versioning.
///
/// Goldenfile takes care of checking for equality against a checked-in file.
#[allow(clippy::unwrap_used)]
pub fn check_metadata<T>(name: &str, metadata: T)
where
    T: SerializeMetadata,
    T: DeserializeMetadata,
{
    let mut mint = Mint::new("goldenfiles/");
    let meta = metadata.serialize();
    let mut f = mint
        .new_goldenfile_with_differ(name, Box::new(binary_diff))
        .unwrap();
    f.write_all(&meta).unwrap();
}

/// Outputs the indices of the true values in a BoolArray
pub fn to_int_indices(indices_bits: BoolArray) -> VortexResult<Vec<u64>> {
    let buffer = indices_bits.boolean_buffer();
    let mask = indices_bits.validity_mask()?;
    Ok(buffer
        .iter()
        .enumerate()
        .filter_map(|(idx, v)| (v && mask.value(idx)).then_some(idx as u64))
        .collect_vec())
}
