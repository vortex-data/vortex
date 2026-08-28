// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Optional indexes attached to an OnPair array.

use std::hash::Hash;
use std::hash::Hasher;
use std::sync::Arc;
use std::sync::OnceLock;

use onpair::search::index::TokenFrequencyIndex;
use onpair::search::index::TokenFrequencyIndexStorage;
use onpair::search::index::build_token_frequency_index as build_onpair_token_frequency_index;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;

use crate::OnPair;
use crate::OnPairArray;
use crate::OnPairArrayExt;
use crate::OnPairArraySlotsExt;
use crate::OnPairSlotsView;
use crate::array::num_tokens_from_offsets;
use crate::decode::collect_widened;

const TOKEN_FREQUENCY_INDEX_BIT: u64 = 1 << 0;
const SUPPORTED_INDEX_BITS: u64 = TOKEN_FREQUENCY_INDEX_BIT;

/// Optional advisory indexes attached to an OnPair encoding.
///
/// Each index has a stable bit position and a fixed, append-only child layout.
#[derive(Copy, Clone, Debug, Default, Eq, Hash, PartialEq)]
pub struct OnPairIndexSet(u64);

impl OnPairIndexSet {
    /// An empty set containing no auxiliary indexes.
    pub const fn empty() -> Self {
        Self(0)
    }

    /// Include the token-frequency index used by substring-search planning.
    pub const fn with_token_frequency(mut self) -> Self {
        self.0 |= TOKEN_FREQUENCY_INDEX_BIT;
        self
    }

    /// Whether the token-frequency index is present.
    pub const fn has_token_frequency(self) -> bool {
        self.0 & TOKEN_FREQUENCY_INDEX_BIT != 0
    }

    pub(crate) const fn bits(self) -> u64 {
        self.0
    }

    pub(crate) fn from_bits(bits: u64) -> VortexResult<Self> {
        let unsupported = bits & !SUPPORTED_INDEX_BITS;
        vortex_ensure!(
            unsupported == 0,
            "Unsupported OnPair index flags: {unsupported:#x}"
        );
        Ok(Self(bits))
    }

    pub(crate) const fn child_count(self) -> usize {
        if self.has_token_frequency() { 1 } else { 0 }
    }
}

/// Typed index children in their fixed OnPair slot order.
#[derive(Clone, Debug, Default)]
pub struct OnPairIndexChildren {
    token_frequency: Option<ArrayRef>,
}

impl OnPairIndexChildren {
    /// Attach the cumulative token-frequency child.
    pub fn with_token_frequency(mut self, child: ArrayRef) -> Self {
        self.token_frequency = Some(child);
        self
    }

    /// Return the cumulative token-frequency child, if present.
    pub fn token_frequency(&self) -> Option<&ArrayRef> {
        self.token_frequency.as_ref()
    }

    pub(crate) fn as_view(&self) -> OnPairIndexChildrenView<'_> {
        OnPairIndexChildrenView {
            token_frequency: self.token_frequency(),
        }
    }

    pub(crate) fn into_token_frequency(self) -> Option<ArrayRef> {
        self.token_frequency
    }

    fn from_array(array: ArrayView<'_, OnPair>) -> Self {
        Self {
            token_frequency: array.token_frequency_index_child().cloned(),
        }
    }
}

/// Borrowed index children used while validating an OnPair array.
#[derive(Copy, Clone, Debug)]
pub(crate) struct OnPairIndexChildrenView<'a> {
    token_frequency: Option<&'a ArrayRef>,
}

impl<'a> OnPairIndexChildrenView<'a> {
    pub(crate) fn from_slots(slots: &OnPairSlotsView<'a>) -> Self {
        Self {
            token_frequency: slots.token_frequency_index_child,
        }
    }
}

/// Vortex-owned storage retained by OnPair's token-frequency index.
#[derive(Clone, Debug)]
pub(crate) struct VortexTokenFrequencyIndexStorage {
    cumulative: Buffer<u32>,
}

impl VortexTokenFrequencyIndexStorage {
    fn new(cumulative: Buffer<u32>) -> Self {
        Self { cumulative }
    }
}

impl TokenFrequencyIndexStorage for VortexTokenFrequencyIndexStorage {
    fn cumulative(&self) -> &[u32] {
        self.cumulative.as_slice()
    }
}

pub(crate) type VortexTokenFrequencyIndex = TokenFrequencyIndex<VortexTokenFrequencyIndexStorage>;

/// Runtime state for the recognized indexes attached to an OnPair array.
#[derive(Clone, Debug, Default)]
pub(crate) struct OnPairIndexesData {
    token_frequency: Option<Arc<OnceLock<VortexTokenFrequencyIndex>>>,
}

impl OnPairIndexesData {
    pub(crate) fn from_set(indexes: OnPairIndexSet) -> Self {
        Self {
            token_frequency: indexes
                .has_token_frequency()
                .then(|| Arc::new(OnceLock::new())),
        }
    }

    pub(crate) fn set(&self) -> OnPairIndexSet {
        if self.token_frequency.is_some() {
            OnPairIndexSet::default().with_token_frequency()
        } else {
            OnPairIndexSet::default()
        }
    }

    pub(crate) fn array_hash<H: Hasher>(&self, state: &mut H) {
        self.set().hash(state);
    }

    pub(crate) fn same_indexes(&self, other: &Self) -> bool {
        self.set() == other.set()
    }

    pub(crate) fn validate_children(
        &self,
        children: OnPairIndexChildrenView<'_>,
        num_tokens: usize,
    ) -> VortexResult<()> {
        vortex_ensure!(
            self.token_frequency.is_some() == children.token_frequency.is_some(),
            "OnPair token-frequency child presence does not match index flags"
        );
        if let Some(child) = children.token_frequency {
            validate_token_frequency_child(child, num_tokens)?;
        }
        Ok(())
    }

    fn with_token_frequency(mut self, index: VortexTokenFrequencyIndex) -> Self {
        self.token_frequency = Some(Arc::new(OnceLock::from(index)));
        self
    }

    fn token_frequency(&self) -> Option<&Arc<OnceLock<VortexTokenFrequencyIndex>>> {
        self.token_frequency.as_ref()
    }
}

/// Build and attach a token-frequency index to an OnPair array.
///
/// If the array is already indexed, it is returned unchanged. In particular,
/// an index reconstructed from storage remains lazily safety-validated until
/// an operation accesses it.
pub fn build_token_frequency_index(
    array: OnPairArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<OnPairArray> {
    if array.token_frequency_index_child().is_some() {
        return Ok(array);
    }

    let codes = collect_widened::<u16>(array.codes(), ctx)?;
    let num_tokens = num_tokens_from_offsets(array.dict_offsets())?;
    let index = build_onpair_token_frequency_index(codes.as_slice(), num_tokens)
        .map_err(|error| vortex_err!("OnPair token-frequency index failed: {error}"))?;
    let cumulative = Buffer::from(index.into_storage().into_raw());
    let index = TokenFrequencyIndex::validate_safety(
        VortexTokenFrequencyIndexStorage::new(cumulative.clone()),
        num_tokens,
        codes.len(),
    )
    .map_err(
        |error| vortex_err!(InvalidArgument: "Unsafe OnPair token-frequency index: {error}"),
    )?;

    let indexes = array.data().indexes().clone().with_token_frequency(index);
    let data = array.data().clone().with_indexes(indexes);
    let index_children = OnPairIndexChildren::from_array(array.as_view())
        .with_token_frequency(cumulative.into_array());

    OnPair::try_new_with_data(
        array.dtype().clone(),
        data,
        array.dict_offsets().clone(),
        array.codes().clone(),
        array.codes_offsets().clone(),
        array.uncompressed_lengths().clone(),
        array.array_validity(),
        index_children,
    )
}

fn validate_token_frequency_child(child: &ArrayRef, num_tokens: usize) -> VortexResult<()> {
    let expected_len = num_tokens
        .checked_add(1)
        .ok_or_else(|| vortex_err!(InvalidArgument: "OnPair token-frequency length overflow"))?;
    vortex_ensure!(
        child.dtype() == &DType::Primitive(PType::U32, Nullability::NonNullable),
        "OnPair token-frequency index child must be non-nullable u32"
    );
    vortex_ensure!(
        child.len() == expected_len,
        "OnPair token-frequency index child length {} does not match num_tokens + 1 ({})",
        child.len(),
        expected_len
    );
    Ok(())
}

/// Lazily materialize and safety-validate an array's token-frequency index.
pub(crate) fn token_frequency_index<'a>(
    array: ArrayView<'a, OnPair>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<&'a VortexTokenFrequencyIndex>> {
    let Some(cache) = array.data().indexes().token_frequency() else {
        return Ok(None);
    };

    let index = match cache.get() {
        Some(index) => index,
        None => {
            let child = array
                .token_frequency_index_child()
                .ok_or_else(|| vortex_err!("Missing OnPair token-frequency index child"))?;
            let cumulative = collect_widened::<u32>(child, ctx)?;
            let index = TokenFrequencyIndex::validate_safety(
                VortexTokenFrequencyIndexStorage::new(cumulative),
                num_tokens_from_offsets(array.dict_offsets())?,
                array.codes().len(),
            )
            .map_err(|error| {
                vortex_err!(InvalidArgument: "Unsafe OnPair token-frequency index: {error}")
            })?;
            cache.get_or_init(|| index)
        }
    };
    Ok(Some(index))
}

#[cfg(test)]
mod tests;
