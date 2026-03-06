// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::fmt::Debug;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;
use std::ops::Range;
use std::sync::Arc;

use vortex_buffer::ByteBuffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_mask::Mask;

use crate::ArrayEq;
use crate::ArrayHash;
use crate::ArrayRef;
use crate::Canonical;
use crate::DynArray;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::LEGACY_SESSION;
use crate::VortexSessionExecute;
use crate::array::ArrayVisitor;
use crate::array::visitor::ArrayVisitorExt;
use crate::arrays::ConstantVTable;
use crate::arrays::DictArray;
use crate::arrays::FilterArray;
use crate::arrays::ScalarFnVTable;
use crate::arrays::SliceArray;
use crate::buffer::BufferHandle;
use crate::builders::ArrayBuilder;
use crate::compute;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::expr::stats::Precision;
use crate::expr::stats::Stat;
use crate::expr::stats::StatsProviderExt;
use crate::hash;
use crate::optimizer::ArrayOptimizer;
use crate::scalar::Scalar;
use crate::scalar_fn::ReduceNode;
use crate::scalar_fn::ReduceNodeRef;
use crate::scalar_fn::ScalarFnRef;
use crate::stats::ArrayStats;
use crate::stats::HasArrayStats;
use crate::stats::StatsSetRef;
use crate::validity::Validity;
use crate::vtable::ArrayId;
use crate::vtable::ArrayVTableExt;
use crate::vtable::DynVTable;
use crate::vtable::OperationsVTable;
use crate::vtable::VTable;
use crate::vtable::ValidityVTable;

/// The typed representation of a Vortex array, pairing a vtable with its array data and other
/// common fields.
#[derive(Debug, Clone)]
pub struct Array<V: VTable> {
    vtable: V,
    len: usize,
    dtype: DType,
    pub(crate) data: V::Array,
    child_slots: Vec<Option<ArrayRef>>,
    stats: ArrayStats,
}

impl<V: VTable> Array<V> {
    /// Constructs a new `Array<V>` by extracting common fields from the encoding struct.
    pub fn new(vtable: V, data: V::Array) -> Self {
        let len = V::len(&data);
        let dtype = V::dtype(&data).clone();
        let stats = data.array_stats().clone();
        let child_slots = (0..V::nchildren(&data))
            .map(|i| Some(V::child(&data, i)))
            .collect();
        Self {
            vtable,
            len,
            dtype,
            data,
            child_slots,
            stats,
        }
    }

    /// Access the encoding-specific data.
    pub fn data(&self) -> &V::Array {
        &self.data
    }

    /// Returns the stats of the array.
    #[inline(always)]
    pub fn array_stats(&self) -> &ArrayStats {
        &self.stats
    }
}

impl<V: VTable> ReduceNode for Array<V> {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn node_dtype(&self) -> VortexResult<DType> {
        Ok(self.dtype.clone())
    }

    fn scalar_fn(&self) -> Option<&ScalarFnRef> {
        self.data.as_opt::<ScalarFnVTable>().map(|a| a.scalar_fn())
    }

    fn child(&self, idx: usize) -> ReduceNodeRef {
        self.nth_child(idx)
            .unwrap_or_else(|| vortex_panic!("Child index out of bounds: {}", idx))
    }

    fn child_count(&self) -> usize {
        self.nchildren()
    }
}

impl<V: VTable> DynArray for Array<V> {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_arc(self: Arc<Self>) -> Arc<dyn Any + Send + Sync> {
        self
    }

    fn to_array(&self) -> ArrayRef {
        Arc::new(Self {
            vtable: self.vtable.clone(),
            len: self.len,
            dtype: self.dtype.clone(),
            data: self.data.clone(),
            child_slots: self.child_slots.clone(),
            stats: self.stats.clone(),
        })
    }

    fn len(&self) -> usize {
        self.len
    }

    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn vtable(&self) -> &dyn DynVTable {
        V::vtable()
    }

    fn encoding_id(&self) -> ArrayId {
        V::id(&self.data)
    }

    fn slice(&self, range: Range<usize>) -> VortexResult<ArrayRef> {
        let start = range.start;
        let stop = range.end;

        if start == 0 && stop == self.len() {
            return Ok(self.to_array());
        }

        vortex_ensure!(
            start <= self.len(),
            "OutOfBounds: start {start} > length {}",
            self.len()
        );
        vortex_ensure!(
            stop <= self.len(),
            "OutOfBounds: stop {stop} > length {}",
            self.len()
        );

        vortex_ensure!(start <= stop, "start ({start}) must be <= stop ({stop})");

        if start == stop {
            return Ok(Canonical::empty(self.dtype()).into_array());
        }

        let sliced = SliceArray::try_new(self.to_array(), range)?
            .into_array()
            .optimize()?;

        // Propagate some stats from the original array to the sliced array.
        if !sliced.is::<ConstantVTable>() {
            self.statistics().with_iter(|iter| {
                sliced.statistics().inherit(iter.filter(|(stat, value)| {
                    matches!(
                        stat,
                        Stat::IsConstant | Stat::IsSorted | Stat::IsStrictSorted
                    ) && value.as_ref().as_exact().is_some_and(|v| {
                        Scalar::try_new(DType::Bool(Nullability::NonNullable), Some(v.clone()))
                            .vortex_expect("A stat that was expected to be a boolean stat was not")
                            .as_bool()
                            .value()
                            .unwrap_or_default()
                    })
                }));
            });
        }

        Ok(sliced)
    }

    fn filter(&self, mask: Mask) -> VortexResult<ArrayRef> {
        FilterArray::try_new(self.to_array(), mask)?
            .into_array()
            .optimize()
    }

    fn take(&self, indices: ArrayRef) -> VortexResult<ArrayRef> {
        DictArray::try_new(indices, self.to_array())?
            .into_array()
            .optimize()
    }

    fn scalar_at(&self, index: usize) -> VortexResult<Scalar> {
        vortex_ensure!(index < self.len(), OutOfBounds: index, 0, self.len());
        if self.is_invalid(index)? {
            return Ok(Scalar::null(self.dtype().clone()));
        }
        let scalar = <V::OperationsVTable as OperationsVTable<V>>::scalar_at(&self.data, index)?;
        vortex_ensure!(self.dtype() == scalar.dtype(), "Scalar dtype mismatch");
        Ok(scalar)
    }

    fn is_valid(&self, index: usize) -> VortexResult<bool> {
        vortex_ensure!(index < self.len(), OutOfBounds: index, 0, self.len());
        match self.validity()? {
            Validity::NonNullable | Validity::AllValid => Ok(true),
            Validity::AllInvalid => Ok(false),
            Validity::Array(a) => a
                .scalar_at(index)?
                .as_bool()
                .value()
                .ok_or_else(|| vortex_err!("validity value at index {} is null", index)),
        }
    }

    fn is_invalid(&self, index: usize) -> VortexResult<bool> {
        Ok(!self.is_valid(index)?)
    }

    fn all_valid(&self) -> VortexResult<bool> {
        match self.validity()? {
            Validity::NonNullable | Validity::AllValid => Ok(true),
            Validity::AllInvalid => Ok(false),
            Validity::Array(a) => Ok(a.statistics().compute_min::<bool>().unwrap_or(false)),
        }
    }

    fn all_invalid(&self) -> VortexResult<bool> {
        match self.validity()? {
            Validity::NonNullable | Validity::AllValid => Ok(false),
            Validity::AllInvalid => Ok(true),
            Validity::Array(a) => Ok(!a.statistics().compute_max::<bool>().unwrap_or(true)),
        }
    }

    fn valid_count(&self) -> VortexResult<usize> {
        if let Some(Precision::Exact(invalid_count)) =
            self.statistics().get_as::<usize>(Stat::NullCount)
        {
            return Ok(self.len() - invalid_count);
        }

        let count = match self.validity()? {
            Validity::NonNullable | Validity::AllValid => self.len(),
            Validity::AllInvalid => 0,
            Validity::Array(a) => {
                let sum = compute::sum(&a)?;
                sum.as_primitive()
                    .as_::<usize>()
                    .ok_or_else(|| vortex_err!("sum of validity array is null"))?
            }
        };
        vortex_ensure!(count <= self.len(), "Valid count exceeds array length");

        self.statistics()
            .set(Stat::NullCount, Precision::exact(self.len() - count));

        Ok(count)
    }

    fn invalid_count(&self) -> VortexResult<usize> {
        Ok(self.len() - self.valid_count()?)
    }

    fn validity(&self) -> VortexResult<Validity> {
        if self.dtype().is_nullable() {
            let validity = <V::ValidityVTable as ValidityVTable<V>>::validity(&self.data)?;
            if let Validity::Array(array) = &validity {
                vortex_ensure!(array.len() == self.len(), "Validity array length mismatch");
                vortex_ensure!(
                    matches!(array.dtype(), DType::Bool(Nullability::NonNullable)),
                    "Validity array is not non-nullable boolean: {}",
                    self.encoding_id(),
                );
            }
            Ok(validity)
        } else {
            Ok(Validity::NonNullable)
        }
    }

    fn validity_mask(&self) -> VortexResult<Mask> {
        match self.validity()? {
            Validity::NonNullable | Validity::AllValid => Ok(Mask::new_true(self.len())),
            Validity::AllInvalid => Ok(Mask::new_false(self.len())),
            Validity::Array(a) => {
                a.try_to_mask_fill_null_false(&mut LEGACY_SESSION.create_execution_ctx())
            }
        }
    }

    fn to_canonical(&self) -> VortexResult<Canonical> {
        self.to_array()
            .execute(&mut LEGACY_SESSION.create_execution_ctx())
    }

    fn append_to_builder(
        &self,
        builder: &mut dyn ArrayBuilder,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        if builder.dtype() != self.dtype() {
            vortex_panic!(
                "Builder dtype mismatch: expected {}, got {}",
                self.dtype(),
                builder.dtype(),
            );
        }
        let len = builder.len();

        V::append_to_builder(&self.data, builder, ctx)?;

        assert_eq!(
            len + self.len(),
            builder.len(),
            "Builder length mismatch after writing array for encoding {}",
            self.encoding_id(),
        );
        Ok(())
    }

    fn statistics(&self) -> StatsSetRef<'_> {
        self.stats.to_ref(self)
    }

    fn with_children(&self, children: Vec<ArrayRef>) -> VortexResult<ArrayRef> {
        let mut data = self.data.clone();
        V::with_children(&mut data, children)?;
        Ok(data.into_array())
    }
}

impl<V: VTable> ArrayHash for Array<V> {
    fn array_hash<H: Hasher>(&self, state: &mut H, precision: hash::Precision) {
        self.data.encoding_id().hash(state);
        V::array_hash(&self.data, state, precision);
    }
}

impl<V: VTable> ArrayEq for Array<V> {
    fn array_eq(&self, other: &Self, precision: hash::Precision) -> bool {
        V::array_eq(&self.data, &other.data, precision)
    }
}

impl<V: VTable> ArrayVisitor for Array<V> {
    fn children(&self) -> Vec<ArrayRef> {
        self.child_slots.iter().filter_map(|s| s.clone()).collect()
    }

    fn nchildren(&self) -> usize {
        self.child_slots.iter().filter(|s| s.is_some()).count()
    }

    fn nth_child(&self, idx: usize) -> Option<ArrayRef> {
        self.child_slots.get(idx).and_then(|s| s.clone())
    }

    fn children_names(&self) -> Vec<String> {
        (0..self.child_slots.len())
            .filter(|i| self.child_slots[*i].is_some())
            .map(|i| V::child_name(&self.data, i))
            .collect()
    }

    fn named_children(&self) -> Vec<(String, ArrayRef)> {
        (0..self.child_slots.len())
            .filter_map(|i| {
                self.child_slots[i]
                    .clone()
                    .map(|child| (V::child_name(&self.data, i), child))
            })
            .collect()
    }

    fn buffers(&self) -> Vec<ByteBuffer> {
        (0..V::nbuffers(&self.data))
            .map(|i| V::buffer(&self.data, i).to_host_sync())
            .collect()
    }

    fn buffer_handles(&self) -> Vec<BufferHandle> {
        (0..V::nbuffers(&self.data))
            .map(|i| V::buffer(&self.data, i))
            .collect()
    }

    fn buffer_names(&self) -> Vec<String> {
        (0..V::nbuffers(&self.data))
            .filter_map(|i| V::buffer_name(&self.data, i))
            .collect()
    }

    fn named_buffers(&self) -> Vec<(String, BufferHandle)> {
        (0..V::nbuffers(&self.data))
            .filter_map(|i| {
                V::buffer_name(&self.data, i).map(|name| (name, V::buffer(&self.data, i)))
            })
            .collect()
    }

    fn nbuffers(&self) -> usize {
        V::nbuffers(&self.data)
    }

    fn metadata(&self) -> VortexResult<Option<Vec<u8>>> {
        V::serialize(V::metadata(&self.data)?)
    }

    fn metadata_fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match V::metadata(&self.data) {
            Err(e) => write!(f, "<serde error: {e}>"),
            Ok(metadata) => Debug::fmt(&metadata, f),
        }
    }

    fn is_host(&self) -> bool {
        for array in self.depth_first_traversal() {
            if !array.buffer_handles().iter().all(BufferHandle::is_on_host) {
                return false;
            }
        }

        true
    }
}

impl<V: VTable> IntoArray for Array<V> {
    fn into_array(self) -> ArrayRef {
        Arc::new(self)
    }
}
