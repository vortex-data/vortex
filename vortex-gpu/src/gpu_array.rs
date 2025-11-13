// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use cudarc::driver::{CudaSlice, CudaView, CudaViewMut};
use vortex_array::arrays::{ChunkedArray, PrimitiveArray, StructArray};
use vortex_array::validity::Validity;
use vortex_array::{Canonical, IntoArray};
use vortex_buffer::BufferMut;
use vortex_dtype::{FieldNames, NativePType, PType, match_each_native_ptype};
use vortex_error::{VortexExpect, VortexResult, vortex_err, vortex_panic};

#[derive(Clone)]
pub enum GpuVector {
    Primitive(GpuPrimitiveVector),
    Struct(GpuStructVector),
}

impl GpuVector {
    pub fn into_host_array(self) -> VortexResult<Canonical> {
        match self {
            GpuVector::Primitive(p) => p.into_host_array().map(Canonical::Primitive),
            GpuVector::Struct(s) => s.into_host_array().map(Canonical::Struct),
        }
    }

    pub fn into_primitive(self) -> GpuPrimitiveVector {
        match self {
            GpuVector::Primitive(p) => p,
            _ => vortex_panic!("Not a primitive gpu array"),
        }
    }

    pub fn into_struct(self) -> GpuStructVector {
        match self {
            GpuVector::Struct(s) => s,
            _ => vortex_panic!("Not a struct gpu array"),
        }
    }
}

#[derive(Clone)]
pub struct GpuPrimitiveVector {
    values: CudaSlice<u8>,
    len: usize,
    ptype: PType,
}

impl GpuPrimitiveVector {
    pub fn from_casted_array<T: NativePType>(values: CudaSlice<T>, ptype: PType) -> Self {
        let len = values.len();
        Self::new(values, len, ptype)
    }

    pub fn from_slice_with_len<T: NativePType>(values: CudaSlice<T>, len: usize) -> Self {
        Self::new(values, len, T::PTYPE)
    }

    fn new<T: NativePType>(values: CudaSlice<T>, len: usize, ptype: PType) -> Self {
        assert_eq!(
            T::PTYPE,
            ptype,
            "Target ptype {} didn't match the underlying ptype {}",
            T::PTYPE,
            ptype
        );
        assert!(
            len <= values.len(),
            "Given length {len} must be shorter than provided slice length {}",
            values.len()
        );
        let stream = values.stream().clone();
        let values_ptr = values.leak();
        Self {
            values: unsafe { stream.upgrade_device_ptr(values_ptr, size_of::<T>() * len) },
            len,
            ptype,
        }
    }

    pub fn as_slice<T: NativePType>(&self) -> CudaView<'_, T> {
        assert_eq!(
            T::PTYPE,
            self.ptype,
            "Target ptype {} didn't match the underlying ptype {}",
            T::PTYPE,
            self.ptype
        );
        unsafe {
            self.values
                .as_view()
                .transmute::<T>(self.len)
                .vortex_expect("asserted before")
                .slice(0..self.len)
        }
    }

    pub fn as_mut_slice<T: NativePType>(&mut self) -> CudaViewMut<'_, T> {
        assert_eq!(
            T::PTYPE,
            self.ptype,
            "Target ptype {} didn't match the underlying ptype {}",
            T::PTYPE,
            self.ptype
        );
        unsafe {
            self.values
                .as_view_mut()
                .transmute_mut::<T>(self.len)
                .vortex_expect("asserted before")
                .slice_mut(0..self.len)
        }
    }

    pub fn into_host_array(self) -> VortexResult<PrimitiveArray> {
        let stream = self.values.stream();
        match_each_native_ptype!(self.ptype, |P| {
            let mut buffer = BufferMut::<P>::with_capacity(self.len);
            unsafe { buffer.set_len(self.len) }
            stream
                .memcpy_dtoh(&self.as_slice::<P>(), &mut buffer[..])
                .map_err(|e| vortex_err!("Failed to copy to device: {e}"))?;
            stream
                .synchronize()
                .map_err(|e| vortex_err!("Failed to synchronize: {e}"))?;
            Ok(PrimitiveArray::new(buffer.freeze(), Validity::NonNullable)
                .reinterpret_cast(self.ptype))
        })
    }
}

#[derive(Clone)]
pub struct GpuStructVector {
    names: FieldNames,
    children: Box<[Vec<GpuVector>]>,
    len: usize,
}

impl GpuStructVector {
    pub fn new(names: FieldNames, children: Box<[Vec<GpuVector>]>, len: usize) -> Self {
        Self {
            names,
            children,
            len,
        }
    }

    pub fn child(&self, idx: usize) -> &[GpuVector] {
        &self.children[idx]
    }

    pub fn into_host_array(self) -> VortexResult<StructArray> {
        let children = self
            .children
            .into_iter()
            .map(|c| {
                let children = c
                    .into_iter()
                    .map(|cc| cc.into_host_array())
                    .collect::<VortexResult<Vec<_>>>()?;
                match children.len() {
                    0 => vortex_panic!("Can't have 0 length children"),
                    1 => Ok(children
                        .into_iter()
                        .next()
                        .vortex_expect("there's one element")
                        .into_array()),
                    _ => Ok(children
                        .into_iter()
                        .map(|c| c.into_array())
                        .collect::<ChunkedArray>()
                        .into_array()),
                }
            })
            .collect::<VortexResult<Vec<_>>>()?;
        StructArray::try_new(self.names, children, self.len, Validity::NonNullable)
    }
}
