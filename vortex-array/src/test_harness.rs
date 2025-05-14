use std::io::Write;

use goldenfile::Mint;
use goldenfile::differs::binary_diff;

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
