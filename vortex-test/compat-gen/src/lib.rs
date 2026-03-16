// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

pub mod adapter;
pub mod fixtures;
pub mod manifest;
pub mod validate;

#[cfg(test)]
mod tests {
    use vortex_buffer::ByteBuffer;

    use crate::adapter;
    use crate::fixtures::all_fixtures;

    #[test]
    fn roundtrip_all_fixtures() {
        let tmp = tempfile::tempdir().unwrap();
        let fixtures = all_fixtures();

        for fixture in &fixtures {
            eprintln!("--- writing {} ---", fixture.name());
            let entries = fixture.write(tmp.path()).unwrap();
            for entry in &entries {
                let path = tmp.path().join(&entry.name);
                let bytes = std::fs::read(&path).unwrap();
                eprintln!("  reading back {}...", entry.name);
                let _array = adapter::read_file(ByteBuffer::from(bytes)).unwrap();
                eprintln!("  OK: {}", entry.name);
            }
        }
    }

    #[test]
    fn roundtrip_dataset_regular() {
        let tmp = tempfile::tempdir().unwrap();
        for dataset in all_fixtures()
            .into_iter()
            .filter(|f| f.name().contains(".regular."))
        {
            eprintln!("--- {} ---", dataset.name());
            let entries = dataset.write(tmp.path()).unwrap();
            for entry in &entries {
                let path = tmp.path().join(&entry.name);
                let bytes = std::fs::read(&path).unwrap();
                eprintln!("  reading back...");
                let _array = adapter::read_file(ByteBuffer::from(bytes)).unwrap();
                eprintln!("  OK");
            }
        }
    }

    #[test]
    fn roundtrip_dataset_compact() {
        let tmp = tempfile::tempdir().unwrap();
        for dataset in all_fixtures()
            .into_iter()
            .filter(|f| f.name().contains(".compact."))
        {
            eprintln!("--- {} ---", dataset.name());
            let entries = dataset.write(tmp.path()).unwrap();
            for entry in &entries {
                let path = tmp.path().join(&entry.name);
                let bytes = std::fs::read(&path).unwrap();
                eprintln!("  reading back...");
                let _array = adapter::read_file(ByteBuffer::from(bytes)).unwrap();
                eprintln!("  OK");
            }
        }
    }
}
