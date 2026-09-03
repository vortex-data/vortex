// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;
use std::task::Poll;

use vortex::file::OpenOptionsSessionExt;
use vortex::file::VortexFile;
use vortex::layout::scan::multi::MultiLayoutDataSource;
use vortex::mask::Mask;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_io::VortexReadAt;
use vortex_io::runtime::BlockingRuntime;

use crate::ffi::ffi_runtime;
use crate::ffi::try_or;
use crate::ffi::vx_data_source_new_with;
use crate::ffi::vx_expression_ref;
use crate::ffi::vx_session_ref;
use crate::ffi::vx_velox_data_source;
use crate::ffi::vx_velox_error;
use crate::ffi::vx_velox_expression;
use crate::ffi::vx_velox_session;
use crate::read_at::vx_velox_read_at;

/// A stable natural row range reported by a Vortex file.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct vx_velox_natural_split {
    /// Set this field to `sizeof(vx_velox_natural_split)`.
    pub struct_size: usize,
    /// The first row in the split.
    pub row_begin: u64,
    /// One past the final row in the split.
    pub row_end: u64,
}

/// An opened Vortex file that uses Velox callbacks for all reads.
pub struct vx_velox_source {
    file: VortexFile,
    file_size: u64,
    natural_splits: Vec<Range<u64>>,
}

impl vx_velox_source {
    pub(crate) fn file(&self) -> &VortexFile {
        &self.file
    }
}

/// Open a Vortex source through a callback reader.
///
/// The source retains the session and reader state. The caller can free both input handles after
/// this function returns.
///
/// # Safety
///
/// `session` and `reader` must point to live handles. `error_out` must be null or valid for one
/// error pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_velox_source_new(
    session: *const vx_velox_session,
    reader: *const vx_velox_read_at,
    error_out: *mut *mut vx_velox_error,
) -> *mut vx_velox_source {
    try_or(error_out, std::ptr::null_mut(), || {
        let session = unsafe { vx_session_ref(session)? }.clone();
        let reader = unsafe {
            reader
                .as_ref()
                .ok_or_else(|| vortex_err!("Vortex Velox reader must not be null"))?
        }
        .reader();
        let file_size = ffi_runtime().block_on(reader.size())?;
        let file = ffi_runtime().block_on(
            session
                .open_options()
                .with_file_size(file_size)
                .with_layout_reader_cache()
                .open_read(reader),
        )?;
        let natural_splits = file.splits()?;
        Ok(Box::into_raw(Box::new(vx_velox_source {
            file,
            file_size,
            natural_splits,
        })))
    })
}

/// Free a callback-backed Vortex source and release its callback-owned input buffers.
///
/// # Safety
///
/// `source` must be null or a pointer returned by [`vx_velox_source_new`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vx_velox_source_free(source: *mut vx_velox_source) {
    if !source.is_null() {
        // SAFETY: The caller transfers the unique pointer returned by the constructor.
        drop(unsafe { Box::from_raw(source) });
        // The runtime defers some stream drops to its next turn. Run one turn so the C ABI
        // releases every callback-owned buffer before this function returns.
        let mut yielded = false;
        ffi_runtime().block_on(futures::future::poll_fn(|context| {
            if yielded {
                Poll::Ready(())
            } else {
                yielded = true;
                context.waker().wake_by_ref();
                Poll::Pending
            }
        }));
    }
}

/// Return the file row count.
///
/// # Safety
///
/// `source` must point to a live source.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vx_velox_source_row_count(source: *const vx_velox_source) -> u64 {
    // SAFETY: The caller provides a live source.
    unsafe { &*source }.file.row_count()
}

/// Return the file size in bytes.
///
/// # Safety
///
/// `source` must point to a live source.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vx_velox_source_file_size(source: *const vx_velox_source) -> u64 {
    // SAFETY: The caller provides a live source.
    unsafe { &*source }.file_size
}

/// Return the number of natural row splits.
///
/// # Safety
///
/// `source` must point to a live source.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vx_velox_source_natural_split_count(
    source: *const vx_velox_source,
) -> usize {
    // SAFETY: The caller provides a live source.
    unsafe { &*source }.natural_splits.len()
}

/// Write one natural row split.
///
/// # Safety
///
/// `source` must point to a live source. `split_out` must point to a structure with a valid size.
/// `error_out` must be null or valid for one error pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_velox_source_natural_split_at(
    source: *const vx_velox_source,
    index: usize,
    split_out: *mut vx_velox_natural_split,
    error_out: *mut *mut vx_velox_error,
) -> i32 {
    try_or(error_out, 1, || {
        let source = unsafe {
            source
                .as_ref()
                .ok_or_else(|| vortex_err!("Vortex Velox source must not be null"))?
        };
        let split_out = unsafe {
            split_out
                .as_mut()
                .ok_or_else(|| vortex_err!("Natural split output must not be null"))?
        };
        if split_out.struct_size < size_of::<vx_velox_natural_split>() {
            vortex_bail!(
                "Natural split structure is too small: expected at least {}, got {}",
                size_of::<vx_velox_natural_split>(),
                split_out.struct_size
            );
        }
        let split = source
            .natural_splits
            .get(index)
            .ok_or_else(|| vortex_err!("Natural split index out of bounds: {}", index))?;
        split_out.row_begin = split.start;
        split_out.row_end = split.end;
        Ok(0)
    })
}

/// Evaluate whether natural splits cannot match an expression.
///
/// Each output byte is one when the matching split cannot produce a true expression result. Zero
/// means that the split can match or that available statistics cannot prove exclusion.
///
/// # Safety
///
/// `source` and `expression` must point to live handles. `pruned_out` must identify `split_count`
/// writable bytes unless `split_count` is zero. `error_out` must be null or valid for one error
/// pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_velox_source_prune_natural_splits(
    source: *const vx_velox_source,
    expression: *const vx_velox_expression,
    first_split: usize,
    split_count: usize,
    pruned_out: *mut u8,
    error_out: *mut *mut vx_velox_error,
) -> i32 {
    try_or(error_out, 1, || {
        let source = unsafe {
            source
                .as_ref()
                .ok_or_else(|| vortex_err!("Vortex Velox source must not be null"))?
        };
        let split_end = first_split
            .checked_add(split_count)
            .ok_or_else(|| vortex_err!("Natural split range overflow"))?;
        if split_end > source.natural_splits.len() {
            vortex_bail!(
                "Natural split range out of bounds: {}..{} for {} splits",
                first_split,
                split_end,
                source.natural_splits.len()
            );
        }
        if split_count == 0 {
            return Ok(0);
        }
        if pruned_out.is_null() {
            vortex_bail!("Natural split pruning output must not be null");
        }
        let expression = unsafe { vx_expression_ref(expression)? };
        let bound = expression.bind(source.file.dtype())?;
        let reader = source.file.layout_reader()?;
        let output = unsafe { std::slice::from_raw_parts_mut(pruned_out, split_count) };
        for (decision, row_range) in output.iter_mut().zip(
            source.natural_splits[first_split..split_end]
                .iter()
                .cloned(),
        ) {
            let row_count = usize::try_from(row_range.end - row_range.start)?;
            let mask = reader.pruning_evaluation(&row_range, &bound, Mask::new_true(row_count))?;
            *decision = u8::from(ffi_runtime().block_on(mask)?.all_false());
        }
        Ok(0)
    })
}

/// Create a standard Vortex data source for this file.
///
/// The caller owns the returned handle and must free it through `vx_data_source_free`.
///
/// # Safety
///
/// `source` must point to a live source. `error_out` must be null or valid for one error pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_velox_source_data_source(
    source: *const vx_velox_source,
    error_out: *mut *mut vx_velox_error,
) -> *const vx_velox_data_source {
    try_or(error_out, std::ptr::null(), || {
        let source = unsafe {
            source
                .as_ref()
                .ok_or_else(|| vortex_err!("Vortex Velox source must not be null"))?
        };
        let data_source = MultiLayoutDataSource::new_with_first(
            source.file.layout_reader()?,
            Vec::new(),
            vec![Some(source.file_size)],
            source.file.session(),
        );
        Ok(vx_data_source_new_with(data_source))
    })
}
