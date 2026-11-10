// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Batch-sized run selections over a cached DuckDB values dictionary.
//!
//! Codes remain local to each request because the generic dictionary exporter executes all codes.

use std::marker::PhantomData;

use vortex::array::ArrayRef;
use vortex::array::Canonical;
use vortex::array::ExecutionCtx;
use vortex::array::IntoArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::match_each_integer_ptype;
use vortex::dtype::IntegerPType;
use vortex::encodings::runend::RunEndArray;
use vortex::encodings::runend::RunEndArrayExt;
use vortex::encodings::runend::RunEndArraySlotsExt;
use vortex::error::VortexExpect;
use vortex::error::VortexResult;

use crate::convert::ToDuckDBScalar;
use crate::duckdb::ReusableDict;
use crate::duckdb::SelectionVector;
use crate::duckdb::VectorRef;
use crate::exporter::ColumnExporter;
use crate::exporter::DictionaryParts;
use crate::exporter::cache::ConversionCache;
use crate::exporter::cached_values_dict;
use crate::exporter::canonical;

/// Keeps the values dictionary cached while generating selections for each requested batch.
struct RunEndExporter<E: IntegerPType> {
    ends: PrimitiveArray,
    ends_type: PhantomData<E>,
    values: ArrayRef,
    values_dict: ReusableDict,
    run_end_offset: usize,
}

pub(crate) fn new_exporter_with_flatten(
    array: RunEndArray,
    cache: &ConversionCache,
    ctx: &mut ExecutionCtx,
    flatten: bool,
) -> VortexResult<Box<dyn ColumnExporter>> {
    if array.is_empty() {
        return canonical::new_exporter(Canonical::empty(array.dtype()).into_array(), cache, ctx);
    }

    // Our canonicalization is faster than creating a dictionary vector and
    // letting duckdb flatten it for us.
    if flatten {
        return canonical::new_exporter(array.into_array(), cache, ctx);
    }

    let offset = array.offset();
    let ends = array.ends().clone();
    let values = array.values().clone();
    let ends = ends.execute::<PrimitiveArray>(ctx)?;
    let values_dict = cached_values_dict(values.clone(), cache, ctx)?;

    match_each_integer_ptype!(ends.ptype(), |E| {
        Ok(Box::new(RunEndExporter {
            ends,
            ends_type: PhantomData::<E>,
            values,
            values_dict,
            run_end_offset: offset,
        }))
    })
}

impl<E: IntegerPType> RunEndExporter<E> {
    fn first_run(&self, offset: usize) -> usize {
        let offset =
            E::from_usize(offset).vortex_expect("the request offset fits the run-end type");
        self.ends
            .as_slice::<E>()
            .partition_point(|&end| end <= offset)
    }

    fn write_run_codes(&self, offset: usize, first_run: usize, codes: &mut [u32]) {
        let mut position =
            E::from_usize(offset).vortex_expect("the request offset fits the run-end type");
        let end_offset = position
            + E::from_usize(codes.len()).vortex_expect("the request length fits the run-end type");
        let mut remaining = codes;

        for (index, &end) in self.ends.as_slice::<E>()[first_run..].iter().enumerate() {
            let next_end = end.min(end_offset);
            let run_len = (next_end - position)
                .to_usize()
                .vortex_expect("run length fits usize");
            let code = u32::try_from(first_run + index).vortex_expect("run index exceeds u32");
            remaining[..run_len].fill(code);
            remaining = &mut remaining[run_len..];
            if remaining.is_empty() {
                return;
            }
            position = next_end;
        }

        assert!(
            remaining.is_empty(),
            "run ends must cover the requested range"
        );
    }
}

impl<E: IntegerPType> DictionaryParts for RunEndExporter<E> {
    fn values(&self) -> &ReusableDict {
        &self.values_dict
    }

    fn values_key(&self) -> usize {
        self.values.addr()
    }

    fn write_codes(&self, offset: usize, codes: &mut [u32]) {
        if codes.is_empty() {
            return;
        }

        let offset = self.run_end_offset + offset;
        self.write_run_codes(offset, self.first_run(offset), codes);
    }
}

impl<E: IntegerPType> ColumnExporter for RunEndExporter<E> {
    fn as_dictionary(&self) -> Option<&dyn DictionaryParts> {
        Some(self)
    }

    fn export(
        &self,
        offset: usize,
        len: usize,
        vector: &mut VectorRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        if len == 0 {
            return Ok(());
        }

        let offset = self.run_end_offset + offset;
        let first_run = self.first_run(offset);
        let end =
            E::from_usize(offset + len).vortex_expect("the request end fits the run-end type");
        if end <= self.ends.as_slice::<E>()[first_run] {
            let constant = self.values.execute_scalar(first_run, ctx)?;
            let value = constant.try_to_duckdb_scalar()?;
            vector.reference_value(&value);
            return Ok(());
        }

        let mut selection = SelectionVector::with_capacity(len);
        // SAFETY: the selection has capacity for `len` entries. The validated run ends cover the
        // requested range, and write_run_codes fills every entry before DuckDB reads the selection.
        let codes = unsafe { selection.as_slice_mut(len) };
        self.write_run_codes(offset, first_run, codes);
        vector.reuse_dictionary(&self.values_dict, &selection, len);

        Ok(())
    }
}

#[cfg(test)]
mod tests;
