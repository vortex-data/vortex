// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::boxed_local)]
mod dtype;
mod expr;
mod read;
mod scalar;
mod session;
mod write;

use std::sync::LazyLock;

use dtype::*;
use expr::*;
use read::*;
use scalar::*;
use session::*;
use vortex::VortexSessionDefault;
use vortex::io::runtime::BlockingRuntime;
use vortex::io::runtime::current::CurrentThreadRuntime;
use vortex::io::session::RuntimeSessionExt;
use vortex::session::VortexSession as RustVortexSession;
use write::*;

/// By default, the C++ API uses a current-thread runtime, providing control of the threading
/// model to the C++ side.
///
// TODO(ngates): in the future, we could expose an API for C++ to spawn threads that can drive
//  this runtime.
pub(crate) static RUNTIME: LazyLock<CurrentThreadRuntime> =
    LazyLock::new(CurrentThreadRuntime::new);
pub(crate) static SESSION: LazyLock<RustVortexSession> =
    LazyLock::new(|| RustVortexSession::default().with_handle(RUNTIME.handle()));

#[cxx::bridge(namespace = "vortex::ffi")]
#[allow(let_underscore_drop)]
mod ffi {
    extern "Rust" {
        type DType;
        // Factory functions for creating DType
        fn dtype_null() -> Box<DType>;
        fn dtype_bool(nullable: bool) -> Box<DType>;
        fn dtype_primitive(ptype: PType, nullable: bool) -> Box<DType>;
        fn dtype_decimal(precision: u8, scale: i8, nullable: bool) -> Box<DType>;
        fn dtype_utf8(nullable: bool) -> Box<DType>;
        fn dtype_binary(nullable: bool) -> Box<DType>;
        unsafe fn from_arrow(ffi_schema: *mut u8, non_nullable: bool) -> Result<Box<DType>>;
        // Methods for DType
        fn to_string(self: &DType) -> String;

        type Scalar;
        fn bool_scalar_new(value: bool) -> Box<Scalar>;
        fn i8_scalar_new(value: i8) -> Box<Scalar>;
        fn i16_scalar_new(value: i16) -> Box<Scalar>;
        fn i32_scalar_new(value: i32) -> Box<Scalar>;
        fn i64_scalar_new(value: i64) -> Box<Scalar>;
        fn u8_scalar_new(value: u8) -> Box<Scalar>;
        fn u16_scalar_new(value: u16) -> Box<Scalar>;
        fn u32_scalar_new(value: u32) -> Box<Scalar>;
        fn u64_scalar_new(value: u64) -> Box<Scalar>;
        fn f32_scalar_new(value: f32) -> Box<Scalar>;
        fn f64_scalar_new(value: f64) -> Box<Scalar>;
        fn string_scalar_new(value: &str) -> Box<Scalar>;
        fn binary_scalar_new(value: &[u8]) -> Box<Scalar>;
        fn cast_scalar(self: &Scalar, dtype: &DType) -> Result<Box<Scalar>>;

        type Expr;
        fn literal(scalar: Box<Scalar>) -> Box<Expr>;
        fn root() -> Box<Expr>;
        fn column(name: String) -> Box<Expr>;
        fn get_item(field: String, child: Box<Expr>) -> Box<Expr>;
        fn not_(child: Box<Expr>) -> Box<Expr>;
        fn is_null(child: Box<Expr>) -> Box<Expr>;
        // binary op
        fn eq(lhs: Box<Expr>, rhs: Box<Expr>) -> Box<Expr>;
        fn not_eq_(lhs: Box<Expr>, rhs: Box<Expr>) -> Box<Expr>;
        fn gt(lhs: Box<Expr>, rhs: Box<Expr>) -> Box<Expr>;
        fn gt_eq(lhs: Box<Expr>, rhs: Box<Expr>) -> Box<Expr>;
        fn lt(lhs: Box<Expr>, rhs: Box<Expr>) -> Box<Expr>;
        fn lt_eq(lhs: Box<Expr>, rhs: Box<Expr>) -> Box<Expr>;
        fn and_(lhs: Box<Expr>, rhs: Box<Expr>) -> Box<Expr>;
        fn or_(lhs: Box<Expr>, rhs: Box<Expr>) -> Box<Expr>;
        fn checked_add(lhs: Box<Expr>, rhs: Box<Expr>) -> Box<Expr>;
        fn select(fields: Vec<String>, child: Box<Expr>) -> Box<Expr>;

        type VortexFile;
        fn row_count(self: &VortexFile) -> u64;
        fn has_file_stats(self: &VortexFile) -> bool;
        fn scan_builder(self: &VortexFile) -> Result<Box<VortexScanBuilder>>;
        fn open_file(path: &str) -> Result<Box<VortexFile>>;
        fn open_file_from_buffer(data: &[u8]) -> Result<Box<VortexFile>>;

        type VortexScanBuilder;
        fn with_filter(self: &mut VortexScanBuilder, filter: Box<Expr>);
        fn with_filter_ref(self: &mut VortexScanBuilder, filter: &Expr);
        fn with_projection(self: &mut VortexScanBuilder, projection: Box<Expr>);
        fn with_projection_ref(self: &mut VortexScanBuilder, projection: &Expr);
        fn with_row_range(self: &mut VortexScanBuilder, row_range_start: u64, row_range_end: u64);
        fn with_include_by_index(self: &mut VortexScanBuilder, include_by_index: &[u64]);
        fn with_limit(self: &mut VortexScanBuilder, limit: usize);
        unsafe fn with_output_schema(
            self: &mut VortexScanBuilder,
            output_schema: *mut u8,
        ) -> Result<()>;
        unsafe fn scan_builder_into_stream(
            builder: Box<VortexScanBuilder>,
            out_stream: *mut u8,
        ) -> Result<()>;
        fn scan_builder_into_threadsafe_cloneable_reader(
            builder: Box<VortexScanBuilder>,
        ) -> Result<Box<ThreadsafeCloneableReader>>;

        type ThreadsafeCloneableReader;
        unsafe fn clone_a_stream(self: &ThreadsafeCloneableReader, out_stream: *mut u8);

        type VortexWriteStrategyBuilder;
        fn write_strategy_builder_new() -> Box<VortexWriteStrategyBuilder>;
        fn write_strategy_builder_with_compact_encodings(
            builder: &mut VortexWriteStrategyBuilder,
        ) -> Result<()>;
        fn write_strategy_builder_with_row_block_size(
            builder: &mut VortexWriteStrategyBuilder,
            row_block_size: usize,
        ) -> Result<()>;
        fn write_strategy_builder_build(
            builder: Box<VortexWriteStrategyBuilder>,
        ) -> Box<VortexWriteStrategy>;

        type VortexWriteStrategy;

        type VortexWriteOptions;
        fn write_options_new() -> Box<VortexWriteOptions>;
        fn write_options_new_with_session(session: &VortexSession) -> Box<VortexWriteOptions>;
        fn write_options_exclude_dtype(options: &mut VortexWriteOptions);
        fn write_options_with_strategy(
            options: &mut VortexWriteOptions,
            strategy: &VortexWriteStrategy,
        );
        fn write_options_with_file_statistics(
            options: &mut VortexWriteOptions,
            statistics: &[FileStat],
        ) -> Result<()>;
        fn write_options_without_file_statistics(options: &mut VortexWriteOptions);
        fn write_options_into_writer(
            options: Box<VortexWriteOptions>,
            path: &str,
        ) -> Box<VortexWriter>;
        unsafe fn write_array_stream(
            options: Box<VortexWriteOptions>,
            input_stream: *mut u8,
            path: &str,
        ) -> Result<()>;

        type VortexWriter;
        unsafe fn writer_push_array_stream(
            writer: &mut VortexWriter,
            input_stream: *mut u8,
        ) -> Result<()>;
        fn writer_bytes_written(writer: &VortexWriter) -> u64;
        fn writer_buffered_bytes(writer: &VortexWriter) -> u64;
        fn writer_finish(writer: Box<VortexWriter>) -> Result<()>;

        type VortexSession;
        fn session_new() -> Box<VortexSession>;
    }

    #[repr(u8)]
    #[derive(Debug, Clone, Copy)]
    enum PType {
        U8,
        U16,
        U32,
        U64,
        I8,
        I16,
        I32,
        I64,
        F16,
        F32,
        F64,
    }

    #[repr(u8)]
    #[derive(Debug, Clone, Copy)]
    enum FileStat {
        IsConstant = 0,
        IsSorted = 1,
        IsStrictSorted = 2,
        Max = 3,
        Min = 4,
        Sum = 5,
        NullCount = 6,
        UncompressedSizeInBytes = 7,
        NaNCount = 8,
    }
}
