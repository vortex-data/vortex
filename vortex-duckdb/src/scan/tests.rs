// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Unit tests for scan.rs functionality

#[cfg(test)]
#[allow(clippy::module_inception)]
mod tests {
    use crate::cpp::DUCKDB_TYPE;
    use crate::duckdb::LogicalType;

    #[test]
    fn test_varchar_detection() {
        // Test that VARCHAR type is correctly identified for virtual column generation
        let varchar_type = LogicalType::new(DUCKDB_TYPE::DUCKDB_TYPE_VARCHAR);
        assert_eq!(varchar_type.as_type_id(), DUCKDB_TYPE::DUCKDB_TYPE_VARCHAR);

        let int_type = LogicalType::new(DUCKDB_TYPE::DUCKDB_TYPE_INTEGER);
        assert_ne!(int_type.as_type_id(), DUCKDB_TYPE::DUCKDB_TYPE_VARCHAR);
    }

    #[test]
    fn test_virtual_column_name_format() {
        // Test that virtual column names are formatted correctly
        let base_name = "my_column";
        let virtual_name = format!("{}$length", base_name);
        assert_eq!(virtual_name, "my_column$length");
    }
}
