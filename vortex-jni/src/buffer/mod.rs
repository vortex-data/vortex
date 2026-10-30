// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Replace the C Data Interface batch owner with independent Java allocation owners.

use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::ptr::NonNull;
use std::sync::Arc;

use arrow_array::ffi::FFI_ArrowArray;
use arrow_buffer::BooleanBuffer;
use arrow_buffer::Buffer;
use arrow_buffer::NullBuffer;
use arrow_data::ArrayData;
use jni::Env;
use jni::JavaVM;
use jni::objects::JObject;
use jni::objects::JObjectArray;
use jni::refs::Global;
use jni::refs::IntoAuto;
use vortex::error::vortex_err;

use crate::errors::JNIError;
use crate::io::with_jvm;

struct JavaBuffer {
    vm: JavaVM,
    object: Global<JObject<'static>>,
    capacity: usize,
}

impl Drop for JavaBuffer {
    fn drop(&mut self) {
        if let Err(error) = with_jvm(&self.vm, |env| {
            env.call_method(
                &self.object,
                jni::jni_str!("release"),
                jni::jni_sig!("()V"),
                &[],
            )?;
            Ok(())
        }) {
            tracing::error!(%error, "failed to release Java Arrow buffer");
        }
    }
}

/// Snapshot producer addresses before Arrow can replace unaligned buffers with owned copies.
pub(crate) fn exported_buffer_addresses(array: &FFI_ArrowArray) -> BTreeSet<usize> {
    fn collect(array: &FFI_ArrowArray, addresses: &mut BTreeSet<usize>) {
        addresses.extend((0..array.num_buffers()).map(|index| array.buffer(index) as usize));
        for index in 0..array.num_children() {
            collect(array.child(index), addresses);
        }
        if let Some(dictionary) = array.dictionary() {
            collect(dictionary, addresses);
        }
    }
    let mut addresses = BTreeSet::new();
    collect(array, &mut addresses);
    addresses
}

/// The caller must keep the exported root alive until every independent retain is acquired.
pub(crate) fn import_buffers(
    env: &mut Env,
    data: ArrayData,
    exported_addresses: &BTreeSet<usize>,
    buffers: JObjectArray,
) -> Result<ArrayData, JNIError> {
    let mut owners = BTreeMap::new();
    for index in 0..buffers.len(env)? {
        // Wide schemas can contain thousands of buffers; do not accumulate JNI local references.
        let object = buffers.get_element(env, index)?.auto();
        let address = env
            .call_method(&object, jni::jni_str!("address"), jni::jni_sig!("()J"), &[])?
            .j()?;
        let address =
            usize::try_from(address).map_err(|_| vortex_err!("invalid Java buffer address"))?;
        let capacity = env
            .call_method(
                &object,
                jni::jni_str!("capacity"),
                jni::jni_sig!("()J"),
                &[],
            )?
            .j()?;
        let capacity =
            usize::try_from(capacity).map_err(|_| vortex_err!("invalid Java buffer capacity"))?;
        let global = env.new_global_ref(&object)?;
        let vm = env.get_java_vm()?;
        env.call_method(&object, jni::jni_str!("retain"), jni::jni_sig!("()V"), &[])?;
        let owner = Arc::new(JavaBuffer {
            vm,
            object: global,
            capacity,
        });
        // Multiple slices can start at the same address. Keep the largest available extent.
        let entry = owners.entry(address).or_insert_with(|| Arc::clone(&owner));
        if entry.capacity < capacity {
            *entry = owner;
        }
    }
    replace_owners(data, &mut |buffer| {
        if buffer.is_empty() {
            return Ok(Buffer::from_slice_ref([] as [u8; 0]));
        }
        let address = buffer.as_ptr() as usize;
        // Arrow's alignment copies already own their allocations. Keep only buffers whose
        // addresses differ from the producer's, so a missing Java owner still fails import.
        if !exported_addresses.contains(&address) {
            return Ok(buffer.clone());
        }
        let (_, owner) = owners
            .range(..=address)
            .rev()
            .find(|(start, owner)| {
                let offset = address - **start;
                offset <= owner.capacity && buffer.len() <= owner.capacity - offset
            })
            .ok_or_else(|| vortex_err!("imported buffer has no Java owner covering its range"))?;
        // SAFETY: the Java reference manager was retained above and owns this exact memory
        // range. The Arc releases that reference only after all Rust slices are dropped.
        Ok(unsafe {
            Buffer::from_custom_allocation(
                NonNull::new_unchecked(buffer.as_ptr().cast_mut()),
                buffer.len(),
                Arc::<JavaBuffer>::clone(owner),
            )
        })
    })
}

fn replace_owners(
    data: ArrayData,
    replace: &mut impl FnMut(&Buffer) -> Result<Buffer, JNIError>,
) -> Result<ArrayData, JNIError> {
    let buffers = data
        .buffers()
        .iter()
        .map(&mut *replace)
        .collect::<Result<Vec<_>, _>>()?;
    let nulls = data
        .nulls()
        .map(|nulls| {
            let buffer = BooleanBuffer::new(replace(nulls.buffer())?, nulls.offset(), nulls.len());
            // SAFETY: only the owner changed; the bitmap and its null count are unchanged.
            Ok::<_, JNIError>(unsafe { NullBuffer::new_unchecked(buffer, nulls.null_count()) })
        })
        .transpose()?;
    let children = data
        .child_data()
        .iter()
        .cloned()
        .map(|child| replace_owners(child, replace))
        .collect::<Result<Vec<_>, _>>()?;
    // SAFETY: all buffers still describe exactly the imported data, with unchanged addresses,
    // lengths, offsets and child layouts. Only their allocation owners have changed.
    Ok(unsafe {
        data.into_builder()
            .buffers(buffers)
            .nulls(nulls)
            .child_data(children)
            .build_unchecked()
    })
}

#[cfg(test)]
mod tests;
