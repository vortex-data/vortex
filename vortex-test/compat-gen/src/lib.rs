// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

pub mod adapter;
pub mod fixtures;
pub mod manifest;
pub mod validate;

#[cfg(test)]
mod tests {
    use vortex::file::WriteStrategyBuilder;
    use vortex_array::IntoArray;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::StructArray;
    use vortex_array::arrays::VarBinViewArray;
    use vortex_array::dtype::FieldNames;
    use vortex_array::validity::Validity;
    use vortex_buffer::ByteBuffer;

    use crate::adapter;
    use crate::fixtures::all_fixtures;
    use crate::fixtures::check_expected_encodings;
    use crate::fixtures::dataset_fixtures;
    use crate::fixtures::synthetic_fixtures;

    fn is_clickbench_fixture(name: &str) -> bool {
        name.contains("clickbench")
    }

    fn boundary_length_array(len: usize) -> vortex_error::VortexResult<vortex_array::ArrayRef> {
        let ints = PrimitiveArray::from_iter((0..len as i32).map(|i| i - 17));
        let nullable_ints = PrimitiveArray::from_option_iter(
            (0..len as i64).map(|i| if i % 5 == 0 { None } else { Some(i * 3 - 7) }),
        );
        let bools = BoolArray::from_iter((0..len).map(|i| i % 3 == 0));
        let strings = VarBinViewArray::from_iter_nullable_str((0..len).map(|i| match i % 5 {
            0 => None,
            1 => Some(""),
            2 => Some("edge"),
            3 => Some("boundary-length-string"),
            _ => Some("zz"),
        }));

        Ok(StructArray::try_new(
            FieldNames::from(["ints", "nullable_ints", "bools", "strings"]),
            vec![
                ints.into_array(),
                nullable_ints.into_array(),
                bools.into_array(),
                strings.into_array(),
            ],
            len,
            Validity::NonNullable,
        )?
        .into_array())
    }

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

    #[test]
    fn roundtrip_to_bytes_synthetic_fixtures() {
        for fixture in synthetic_fixtures() {
            eprintln!("--- writing {} to bytes ---", fixture.name());
            let array = fixture.build().unwrap();
            check_expected_encodings(&array, fixture.as_ref()).unwrap();
            let bytes = adapter::write_file_to_bytes(array).unwrap();
            let _array = adapter::read_file(bytes).unwrap();
            eprintln!("  OK: {}", fixture.name());
        }
    }

    #[test]
    fn roundtrip_to_bytes_non_clickbench_datasets() {
        for dataset in dataset_fixtures()
            .into_iter()
            .filter(|f| !is_clickbench_fixture(f.name()))
        {
            eprintln!("--- writing {} regular to bytes ---", dataset.name());
            let array = dataset.build().unwrap();
            let regular_bytes = adapter::write_compressed_to_bytes(
                array.clone(),
                WriteStrategyBuilder::default().build(),
            )
            .unwrap();
            let _regular = adapter::read_file(regular_bytes).unwrap();

            eprintln!("--- writing {} compact to bytes ---", dataset.name());
            let compact_bytes = adapter::write_compressed_to_bytes(
                array,
                WriteStrategyBuilder::default()
                    .with_compact_encodings()
                    .build(),
            )
            .unwrap();
            let _compact = adapter::read_file(compact_bytes).unwrap();
        }
    }

    #[test]
    fn roundtrip_synthetic_boundary_lengths_to_bytes() {
        const BOUNDARY_LENGTHS: [usize; 15] = [
            0, 1, 2, 31, 32, 63, 64, 127, 128, 255, 256, 511, 512, 1023, 1025,
        ];

        for len in BOUNDARY_LENGTHS {
            eprintln!(
                "--- writing shared boundary fixture length {} to bytes ---",
                len
            );
            let boundary_array = boundary_length_array(len).unwrap();
            if len == 0 {
                assert!(adapter::write_file_to_bytes(boundary_array).is_err());
                continue;
            }
            let bytes = adapter::write_file_to_bytes(boundary_array).unwrap();
            let _array = adapter::read_file(bytes).unwrap();
        }
    }
}
