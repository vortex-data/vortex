// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Operation-owned probes bind indexed or legacy addressing on first use.

use std::collections::BTreeMap;
use std::mem::replace;
use std::sync::Arc;

use pco::data_types::Number;
use pco::data_types::NumberType;
use pco::match_number_enum;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::half;
use vortex_array::scalar::PValue;
use vortex_array::scalar::Scalar;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::Pco;
use crate::access::AccessState;
use crate::array::number_type_from_ptype;
use crate::decoder::DecodedPages;
use crate::page_index::LogicalPageIndex;
use crate::page_index::PageIndex;

/// Construction is empty; first use binds this state to one array owner and a typed decoder.
/// Touched decoded pages, validity masks, and rank tables are retained across probes.
#[derive(Default)]
pub(crate) struct IndexedProbeState {
    bound: Option<Box<dyn BoundProbe>>,
}

trait BoundProbe: Send + Sync {
    fn scalar(
        &mut self,
        array: ArrayView<'_, Pco>,
        row: usize,
        ranked: bool,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar>;
}

struct TypedProbe<T: Number> {
    owner: Arc<LogicalPageIndex>,
    access: AccessState,
    prepared: BTreeMap<usize, AccessState>,
    decoded: Option<DecodedPages<T>>,
}

impl<T> BoundProbe for TypedProbe<T>
where
    T: Number + NativePType + Into<PValue>,
{
    fn scalar(
        &mut self,
        array: ArrayView<'_, Pco>,
        row: usize,
        ranked: bool,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        if ranked && let Some(position) = self.access.cached_position(row) {
            match position {
                None => return Ok(Scalar::null(array.dtype().clone())),
                Some((page, value)) => {
                    if let Some(values) = self
                        .decoded
                        .as_ref()
                        .and_then(|decoded| decoded.last_values(page))
                    {
                        return Ok(Scalar::primitive(
                            values.as_slice::<T>()[value],
                            array.dtype().nullability(),
                        ));
                    }
                }
            }
        }
        let view = self.owner.as_ref();
        let page = view.find_row(row);
        if self.access.page_id() != Some(page) {
            let next = self.prepared.remove(&page).unwrap_or_default();
            let previous = replace(&mut self.access, next);
            if let Some(previous_page) = previous.page_id() {
                self.prepared.insert(previous_page, previous);
            }
        }
        let position = if ranked {
            self.access.resolve::<true>(view, array, row, ctx)?
        } else {
            self.access.resolve::<false>(view, array, row, ctx)?
        };
        let Some((page, value)) = position else {
            return Ok(Scalar::null(array.dtype().clone()));
        };
        if self.decoded.is_none() {
            self.decoded = Some(DecodedPages::new(array)?);
        }
        let decoded = self
            .decoded
            .as_mut()
            .vortex_expect("decoder prepared above");
        let values = decoded.values(view, array, page, ctx)?;
        Ok(Scalar::primitive(
            values.as_slice::<T>()[value],
            array.dtype().nullability(),
        ))
    }
}

fn bind<T: Number + NativePType + Into<PValue>>(owner: Arc<LogicalPageIndex>) -> Box<dyn BoundProbe> {
    Box::new(TypedProbe::<T> {
        owner,
        access: AccessState::default(),
        prepared: BTreeMap::new(),
        decoded: None,
    })
}

pub(crate) fn scalar_at(
    array: ArrayView<'_, Pco>,
    index: usize,
    state: Option<&mut IndexedProbeState>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Scalar> {
    // The public probe dispatcher already checks visible bounds. A zero-page array
    // stores no defined values, so no validity/index/decoder preparation is needed.
    if array.pages.is_empty() {
        return Ok(Scalar::null(array.dtype().clone()));
    }
    let mut once = IndexedProbeState::default();
    let ranked = state.is_some();
    let state = state.unwrap_or(&mut once);
    if state.bound.is_none() {
        let PageIndex::IndexedFast(owner) = array.page_index()? else {
            unreachable!("indexed probes are dispatched only for a present prefix")
        };
        state.bound = Some(
            match_number_enum!(number_type_from_ptype(array.dtype().as_ptype()), NumberType<T> => {
                bind::<T>(Arc::clone(owner))
            }),
        );
    }
    state.bound.as_mut().vortex_expect("bound above").scalar(
        array,
        array.slice_start() + index,
        ranked,
        ctx,
    )
}
