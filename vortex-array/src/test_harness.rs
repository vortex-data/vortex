use std::io::Write;

use goldenfile::differs::binary_diff;
use goldenfile::Mint;
use vortex_error::VortexExpect;

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
    if let Some(meta) = metadata
        .serialize()
        .vortex_expect("Failed to serialize metadata")
    {
        let mut f = mint
            .new_goldenfile_with_differ(name, Box::new(binary_diff))
            .unwrap();
        f.write_all(&meta).unwrap();
    }
}
