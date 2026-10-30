// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_array::Array;
use arrow_array::Int32Array;
use arrow_array::StructArray;
use arrow_array::ffi::FFI_ArrowArray;
use arrow_array::ffi::FFI_ArrowSchema;
use arrow_array::ffi::from_ffi;
use arrow_schema::DataType;
use arrow_schema::Field;
use vortex::error::VortexResult;

use super::*;
use crate::io::tests::assert_weak_ref_clears;
use crate::io::tests::test_vm;

#[test]
fn test_replacing_owners_releases_unrelated_columns() -> VortexResult<()> {
    let first = Int32Array::from(vec![Some(1), None, Some(3)]);
    let second = Int32Array::from(vec![4, 5, 6]);
    let first_data = first.to_data();
    let first_values = first_data.buffers()[0].clone();
    let first_nulls = first_data.nulls().unwrap().buffer().clone();
    let second_values = second.to_data().buffers()[0].clone();
    let root = StructArray::from(vec![
        (
            Arc::new(Field::new("a", DataType::Int32, true)),
            Arc::new(first) as _,
        ),
        (
            Arc::new(Field::new("b", DataType::Int32, false)),
            Arc::new(second) as _,
        ),
    ]);
    let data = root.to_data();
    let exported = FFI_ArrowArray::new(&data);
    let schema = FFI_ArrowSchema::try_from(data.data_type())?;
    // SAFETY: the array and schema were exported above and are consumed exactly once.
    let imported = unsafe { from_ffi(exported, &schema) }?;
    drop(data);
    drop(root);
    drop(first_data);
    let original = [first_values, first_nulls, second_values];
    let baseline = original
        .iter()
        .map(Buffer::strong_count)
        .collect::<Vec<_>>();
    let rebound = replace_owners(imported, &mut |buffer| {
        let source = original
            .iter()
            .find(|b| b.as_ptr() == buffer.as_ptr())
            .unwrap();
        assert_eq!(source.len(), buffer.len());
        Ok(source.clone())
    })
    .map_err(|error| vortex_err!("{error}"))?;
    let kept = rebound.child_data()[0].buffers()[0].clone();
    assert_eq!(kept.as_ptr(), original[0].as_ptr());
    assert_eq!(rebound.child_data()[0].nulls().unwrap().null_count(), 1);
    drop(rebound);
    assert_eq!(original[0].strong_count(), 2);
    assert_eq!(original[1].strong_count(), 1);
    assert_eq!(original[2].strong_count(), 1);
    assert!(baseline[2] > original[2].strong_count());
    drop(kept);
    assert_eq!(original[0].strong_count(), 1);
    Ok(())
}

#[test]
fn test_sliced_validity_preserves_offsets() -> VortexResult<()> {
    let array = Int32Array::from(vec![Some(1), None, Some(3), Some(4)]);
    let data = array.to_data().slice(1, 2);
    let rebound = replace_owners(data.clone(), &mut |buffer| Ok(buffer.clone()))
        .map_err(|error| vortex_err!("{error}"))?;
    assert_eq!(rebound.offset(), data.offset());
    assert_eq!(rebound.nulls(), data.nulls());
    assert_eq!(rebound.buffers()[0].as_ptr(), data.buffers()[0].as_ptr());
    assert_eq!(Int32Array::from(rebound), array.slice(1, 2));
    Ok(())
}

#[test]
fn test_release_from_native_thread() -> VortexResult<()> {
    let Some(vm) = test_vm() else {
        return Ok(());
    };
    // Semaphore.release() has the same JNI signature as NativeBuffer.release(), and
    // its permit count lets us observe exactly how often the drop callback ran.
    let (owner, semaphore, weak) = with_jvm(vm, |env| {
        let object = env.new_object(
            jni::jni_str!("java/util/concurrent/Semaphore"),
            jni::jni_sig!("(I)V"),
            &[jni::JValue::Int(0)],
        )?;
        let weak = env.new_object(
            jni::jni_str!("java/lang/ref/WeakReference"),
            jni::jni_sig!("(Ljava/lang/Object;)V"),
            &[jni::JValue::Object(&object)],
        )?;
        Ok((
            JavaBuffer {
                vm: vm.clone(),
                object: env.new_global_ref(&object)?,
                capacity: 0,
            },
            env.new_global_ref(&object)?,
            env.new_global_ref(&weak)?,
        ))
    })?;
    std::thread::spawn(move || drop(owner))
        .join()
        .map_err(|_| vortex_err!("release thread panicked"))?;
    let permits = with_jvm(vm, |env| {
        Ok(env
            .call_method(
                &semaphore,
                jni::jni_str!("availablePermits"),
                jni::jni_sig!("()I"),
                &[],
            )?
            .i()?)
    })?;
    assert_eq!(permits, 1);
    drop(semaphore);
    assert_weak_ref_clears(vm, &weak)
}
