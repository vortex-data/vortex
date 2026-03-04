// Test for prefer_into_array lint.

use std::sync::Arc;

// Minimal reproduction of the Vortex array types.
type ArrayRef = Arc<dyn DynArray>;

trait DynArray {
    fn to_array(&self) -> ArrayRef;
}

trait IntoArray {
    fn into_array(self) -> ArrayRef;
}

struct MyArray;

impl DynArray for MyArray {
    fn to_array(&self) -> ArrayRef {
        Arc::new(MyArray)
    }
}

impl IntoArray for MyArray {
    fn into_array(self) -> ArrayRef {
        Arc::new(self)
    }
}

// Make owned MyArray auto-deref to &dyn DynArray, mirroring the Vortex Deref impl.
impl std::ops::Deref for MyArray {
    type Target = dyn DynArray;
    fn deref(&self) -> &Self::Target {
        self
    }
}

fn from_ref(arr: &dyn DynArray) -> ArrayRef {
    // OK: calling to_array() on a reference is fine.
    arr.to_array()
}

fn from_owned() -> ArrayRef {
    let owned = MyArray;
    // BAD: calling to_array() on an owned value clones unnecessarily.
    owned.to_array()
}

fn from_owned_inline() -> ArrayRef {
    // BAD: calling to_array() on an owned value clones unnecessarily.
    MyArray.to_array()
}

fn from_owned_good() -> ArrayRef {
    // OK: using into_array() on an owned value is preferred.
    MyArray.into_array()
}

fn main() {
    let _ = from_ref(&MyArray);
    let _ = from_owned();
    let _ = from_owned_inline();
    let _ = from_owned_good();
}
