use std::io::Write;

use goldenfile::differs::binary_diff;
use goldenfile::Mint;

use crate::ArrayMetadata;

/// Check that a named metadata matches its previous versioning.
///
/// Goldenfile takes care of checking for equality against a checked-in file.
#[allow(clippy::unwrap_used)]
pub fn check_metadata<T: ArrayMetadata>(name: &str, metadata: T) {
    let mut mint = Mint::new("goldenfiles/");
    let meta = metadata.try_serialize_metadata().unwrap().to_vec();

    let mut f = mint
        .new_goldenfile_with_differ(name, Box::new(binary_diff))
        .unwrap();
    f.write_all(&meta).unwrap();
}
