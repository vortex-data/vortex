// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#[cfg(test)]
mod tests {
    use std::fmt::Write as _;
    use std::io::Write as _;
    use std::mem::size_of;
    use std::process::Command;
    use std::process::Stdio;

    use vortex_velox::*;

    fn compile_stdin(
        compiler: &str,
        arguments: &[&str],
        source: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut child = Command::new(compiler)
            .args(arguments)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;
        child
            .stdin
            .take()
            .ok_or_else(|| std::io::Error::other("compiler stdin is unavailable"))?
            .write_all(source.as_bytes())?;
        let output = child.wait_with_output()?;
        assert!(
            output.status.success(),
            "{compiler} rejected the adapter header:\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
        Ok(())
    }

    #[test]
    fn c_header_matches_rust_layout() -> Result<(), Box<dyn std::error::Error>> {
        let mut source =
            String::from("#include <stddef.h>\n#include <stdint.h>\n#include \"vortex_velox.h\"\n");
        source.push_str(
            "_Static_assert(sizeof(vx_velox_ptype) == sizeof(uint32_t), \"ptype width\");\n",
        );
        source.push_str(
        "_Static_assert(sizeof(vx_velox_binary_operator) == sizeof(uint32_t), \"operator width\");\n",
    );
        source.push_str(
        "_Static_assert(sizeof(vx_velox_scan_selection_include) == sizeof(uint32_t), \"selection width\");\n",
    );
        source.push_str(
        "_Static_assert(sizeof(vx_velox_primitive_type) == sizeof(uint32_t), \"primitive width\");\n",
    );
        source.push_str(
        "_Static_assert(sizeof(vx_velox_validity_kind) == sizeof(uint32_t), \"validity width\");\n",
    );
        source.push_str(
            "_Static_assert(VX_VELOX_PTYPE_F64 == 10, \"ptype value\");\n\
         _Static_assert(VX_VELOX_OPERATOR_KLEENE_OR == 7, \"operator value\");\n\
         _Static_assert(VX_VELOX_SELECTION_EXCLUDE == 2, \"selection value\");\n\
         _Static_assert(VX_VELOX_PRIMITIVE_F64 == 10, \"primitive value\");\n\
         _Static_assert(VX_VELOX_VALIDITY_BITMAP == 3, \"validity value\");\n",
        );

        macro_rules! check_layout {
        ($type:ty, [$($field:ident),+ $(,)?]) => {{
            writeln!(
                source,
                "_Static_assert(sizeof({0}) == {1}, \"{0} size\");",
                stringify!($type),
                size_of::<$type>()
            )?;
            $(
                writeln!(
                    source,
                    "_Static_assert(offsetof({0}, {1}) == {2}, \"{0}.{1} offset\");",
                    stringify!($type),
                    stringify!($field),
                    std::mem::offset_of!($type, $field)
                )?;
            )+
        }};
    }

        check_layout!(vx_velox_scan_selection, [indices, length, include]);
        check_layout!(
            vx_velox_scan_options,
            [
                struct_size,
                abi_version,
                projection,
                filter,
                row_range_begin,
                row_range_end,
                selection,
                limit,
                ordered,
            ]
        );
        check_layout!(
            vx_velox_read_request,
            [struct_size, offset, length, alignment]
        );
        check_layout!(vx_velox_buffer, [struct_size, data, length, owner, release]);
        check_layout!(
            vx_velox_read_at_callbacks,
            [
                struct_size,
                abi_version,
                context,
                size,
                read_ranges,
                last_error,
                release_context,
                is_cancelled,
                concurrency,
            ]
        );
        check_layout!(vx_velox_natural_split, [struct_size, row_begin, row_end]);
        check_layout!(
            vx_velox_buffer_owner,
            [struct_size, owner, retain, release, retained_bytes]
        );
        check_layout!(
            vx_velox_primitive_view,
            [
                struct_size,
                primitive_type,
                length,
                values,
                values_length,
                validity_kind,
                validity,
                validity_length,
                validity_bit_offset,
                buffers,
                values_alignment,
                validity_alignment,
            ]
        );
        check_layout!(vx_velox_visit_request, [struct_size, rows, row_count]);
        check_layout!(
            vx_velox_visitor,
            [
                struct_size,
                abi_version,
                context,
                visit_primitive,
                last_error,
            ]
        );
        check_layout!(
            vx_velox_arrow_memory_callbacks,
            [
                struct_size,
                abi_version,
                context,
                retain_context,
                release_context,
                report_allocation,
                report_free,
                last_error,
            ]
        );

        let manifest = env!("CARGO_MANIFEST_DIR");
        let include = format!("-I{manifest}/cinclude");
        let base_include = format!("-I{manifest}/../vortex-ffi/cinclude");
        let compiler = std::env::var("CC").unwrap_or_else(|_| "cc".to_owned());
        compile_stdin(
            &compiler,
            &[
                "-std=c11",
                "-fsyntax-only",
                "-x",
                "c",
                &include,
                &base_include,
                "-",
            ],
            &source,
        )
    }

    #[test]
    fn header_compiles_with_host_arrow_declarations() -> Result<(), Box<dyn std::error::Error>> {
        let manifest = env!("CARGO_MANIFEST_DIR");
        let include = format!("-I{manifest}/cinclude");
        let base_include = format!("-I{manifest}/../vortex-ffi/cinclude");
        let compiler = std::env::var("CXX").unwrap_or_else(|_| "c++".to_owned());
        let source =
            std::fs::read_to_string(format!("{manifest}/tests/velox_include_contract.cpp"))?;
        compile_stdin(
            &compiler,
            &[
                "-std=c++20",
                "-fsyntax-only",
                "-x",
                "c++",
                &include,
                &base_include,
                "-",
            ],
            &source,
        )
    }
}
