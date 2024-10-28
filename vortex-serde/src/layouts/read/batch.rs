use std::mem;
use std::sync::Arc;

use vortex_array::array::StructArray;
use vortex_array::{Array, IntoArray};
use vortex_error::{vortex_err, VortexResult};

use crate::layouts::read::{LayoutReader, ReadResult};

#[derive(Debug)]
pub struct BatchReader {
    names: Arc<[Arc<str>]>,
    children: Vec<Box<dyn LayoutReader>>,
    arrays: Vec<Option<Array>>,
}

impl BatchReader {
    pub fn new(names: Arc<[Arc<str>]>, children: Vec<Box<dyn LayoutReader>>) -> Self {
        let arrays = vec![None; children.len()];
        Self {
            names,
            children,
            arrays,
        }
    }

    pub(crate) fn read(&mut self) -> VortexResult<Option<ReadResult>> {
        let mut messages = Vec::new();
        for (i, child_array) in self
            .arrays
            .iter_mut()
            .enumerate()
            .filter(|(_, a)| a.is_none())
        {
            match self.children[i].read_next()? {
                Some(rr) => match rr {
                    ReadResult::ReadMore(message) => {
                        messages.extend(message);
                    }
                    ReadResult::Batch(a) => *child_array = Some(a),
                },
                None => {
                    debug_assert!(
                        self.arrays.iter().all(Option::is_none),
                        "Expected layout to produce an array but it was empty"
                    );
                    return Ok(None);
                }
            }
        }

        if messages.is_empty() {
            let child_arrays = mem::replace(&mut self.arrays, vec![None; self.children.len()])
                .into_iter()
                .enumerate()
                .map(|(i, a)| a.ok_or_else(|| vortex_err!("Missing child array at index {}", i)))
                .collect::<VortexResult<Vec<_>>>()?;
            Ok(Some(ReadResult::Batch(
                StructArray::from_fields(&self.names.iter().zip(child_arrays).collect::<Vec<_>>())?
                    .into_array(),
            )))
        } else {
            Ok(Some(ReadResult::ReadMore(messages)))
        }
    }
}
