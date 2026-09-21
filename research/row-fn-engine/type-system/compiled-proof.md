<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# An isolated Rust proof

One shared function binder and typed row loop compiled against two unrelated host descriptors.
Each adapter compiled in its own crate. Both adapters returned borrowed views, and unsupported
semantic types were rejected.

This experiment establishes that those Rust boundaries are feasible. It does not extract RowFn or
implement a real Vortex, Arrow, DataFusion, or DuckDB adapter. It contains no timing measurements.

[Back to the type-system overview](README.md).

## What the experiment establishes

| Question | Evidence |
| --- | --- |
| Can generic dispatch avoid a Vortex type? | `bind_add<H>` uses the host's descriptor through a small semantic interface. |
| Can one row loop use different representations? | `Add<H>::execute` reads dense integers in one host and integer fields in another. |
| Can independent adapter crates implement the input trait for `i64`? | Each implements the foreign `Input<LocalHost>` trait for the foreign scalar. Both compiled. |
| Can adapters return borrowed views? | Both return slices whose pointers match the original storage. |
| Does the borrow remain tied to the column? | Returning a local column's view as `'static` fails with `E0515`. |
| Does equal physical width imply semantic support? | A money extension over integer storage is rejected at binding and batch access. |
| Are ordinary kernel errors preserved? | Overflow and unequal lengths return their declared errors. |

The adapters use safe indexed access. This experiment does not prove an unsafe input contract or
sink initialization. Output is an owned `Vec<i64>`. Nullability, demand, constants, decoding,
serialization, dynamic plugins, metadata-bearing output, and code generation remain outside its
scope. The public design sketches remain uncompiled.

## Environment and commands

The experiment ran on 2026-09-21 in `/private/tmp/row-fn-type-proof.DnuYNL`. All source files below
were created in that temporary directory. Only this Markdown record is retained in the repository.

`rustc --version --verbose` returned:

```text
rustc 1.98.0 (88d9e12ae 2026-08-18)
binary: rustc
commit-hash: 88d9e12ae178fab0fb5cc050a94da85685d449ea
commit-date: 2026-08-18
host: aarch64-apple-darwin
release: 1.98.0
LLVM version: 22.1.8
```

These exact compilation commands each returned exit code 0 with no output. The two adapter
compilations were independent, after the core compilation completed.

```sh
rustc --edition=2024 --crate-name row_core --crate-type rlib row_core.rs -o librow_core.rlib -D warnings
rustc --edition=2024 --crate-name flat_host --crate-type rlib flat_host.rs --extern row_core=librow_core.rlib -o libflat_host.rlib -D warnings
rustc --edition=2024 --crate-name cell_host --crate-type rlib cell_host.rs --extern row_core=librow_core.rlib -o libcell_host.rlib -D warnings
rustc --edition=2024 main.rs --extern row_core=librow_core.rlib --extern flat_host=libflat_host.rlib --extern cell_host=libcell_host.rlib -L dependency=. -o type-proof -D warnings
./type-proof
```

The executable returned exit code 0 and printed:

```text
flat host: [11, 22, 33]
cell host: [11, 22, 33]
borrowed views alias their original storage
opaque semantics rejected at binding and batch access
overflow and length mismatch rejected
```

The intentionally invalid borrow example used this command:

```sh
rustc --edition=2024 --crate-type rlib borrow_escape.rs --extern row_core=librow_core.rlib --extern flat_host=libflat_host.rlib -L dependency=. -D warnings
```

It returned exit code 1 with this expected diagnostic:

```text
error[E0515]: cannot return value referencing local variable `local`
 --> borrow_escape.rs:9:5
  |
9 |     <i64 as Input<FlatHost>>::bind(&local).unwrap()
  |     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^------^^^^^^^^^^
  |     |                              |
  |     |                              `local` is borrowed here
  |     returns a value referencing data owned by the current function

error: aborting due to 1 previous error

For more information about this error, try `rustc --explain E0515`.
```

## Exact source

The source is intentionally small and includes no external dependencies. It is an experiment,
not the proposed library API. Each block names the file required by the commands above.

<details>
<summary>row_core.rs</summary>

```rust
// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::marker::PhantomData;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SemanticType {
    I64,
    Opaque,
}

pub trait Host: 'static {
    type NativeType;
    type Column;

    fn describe(dtype: &Self::NativeType) -> SemanticType;
}

pub trait Input<H: Host>: Sized {
    type View<'a>;

    fn bind(column: &H::Column) -> Result<Self::View<'_>, &'static str>;
    fn len(view: &Self::View<'_>) -> usize;
    fn get(view: &Self::View<'_>, index: usize) -> Self;
}

pub struct Add<H> {
    host: PhantomData<H>,
}

pub fn bind_add<H: Host>(
    lhs: &H::NativeType,
    rhs: &H::NativeType,
) -> Result<Add<H>, &'static str>
where
    i64: Input<H>,
{
    if H::describe(lhs) != SemanticType::I64 || H::describe(rhs) != SemanticType::I64 {
        return Err("addition requires signed 64-bit integers");
    }

    Ok(Add { host: PhantomData })
}

impl<H: Host> Add<H>
where
    i64: Input<H>,
{
    pub fn execute(
        &self,
        lhs: &H::Column,
        rhs: &H::Column,
    ) -> Result<Vec<i64>, &'static str> {
        let lhs = <i64 as Input<H>>::bind(lhs)?;
        let rhs = <i64 as Input<H>>::bind(rhs)?;
        let len = <i64 as Input<H>>::len(&lhs);
        if len != <i64 as Input<H>>::len(&rhs) {
            return Err("inputs must have equal lengths");
        }

        let mut output = Vec::with_capacity(len);
        for index in 0..len {
            let lhs = <i64 as Input<H>>::get(&lhs, index);
            let rhs = <i64 as Input<H>>::get(&rhs, index);
            output.push(lhs.checked_add(rhs).ok_or("integer overflow")?);
        }

        Ok(output)
    }
}
```

</details>

<details>
<summary>flat_host.rs</summary>

```rust
// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use row_core::{Host, Input, SemanticType};

pub struct FlatHost;

pub enum NativeType {
    Signed64,
    Extension(&'static str),
}

pub struct Column {
    pub dtype: NativeType,
    pub values: Vec<i64>,
}

impl Host for FlatHost {
    type NativeType = NativeType;
    type Column = Column;

    fn describe(dtype: &NativeType) -> SemanticType {
        match dtype {
            NativeType::Signed64 => SemanticType::I64,
            NativeType::Extension(_) => SemanticType::Opaque,
        }
    }
}

impl Input<FlatHost> for i64 {
    type View<'a> = &'a [i64];

    fn bind(column: &Column) -> Result<Self::View<'_>, &'static str> {
        if FlatHost::describe(&column.dtype) != SemanticType::I64 {
            return Err("flat column does not contain semantic i64 values");
        }

        Ok(&column.values)
    }

    fn len(view: &Self::View<'_>) -> usize {
        view.len()
    }

    fn get(view: &Self::View<'_>, index: usize) -> i64 {
        view[index]
    }
}
```

</details>

<details>
<summary>cell_host.rs</summary>

```rust
// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use row_core::{Host, Input, SemanticType};

pub struct CellHost;

pub struct NativeType {
    pub family: &'static str,
    pub bits: u16,
}

pub struct Cell {
    pub tag: u8,
    pub value: i64,
}

pub struct Column {
    pub dtype: NativeType,
    pub cells: Vec<Cell>,
}

impl Host for CellHost {
    type NativeType = NativeType;
    type Column = Column;

    fn describe(dtype: &NativeType) -> SemanticType {
        match (dtype.family, dtype.bits) {
            ("signed", 64) => SemanticType::I64,
            _ => SemanticType::Opaque,
        }
    }
}

impl Input<CellHost> for i64 {
    type View<'a> = &'a [Cell];

    fn bind(column: &Column) -> Result<Self::View<'_>, &'static str> {
        if CellHost::describe(&column.dtype) != SemanticType::I64 {
            return Err("cell column does not contain semantic i64 values");
        }

        Ok(&column.cells)
    }

    fn len(view: &Self::View<'_>) -> usize {
        view.len()
    }

    fn get(view: &Self::View<'_>, index: usize) -> i64 {
        view[index].value
    }
}
```

</details>

<details>
<summary>main.rs</summary>

```rust
// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use cell_host::{Cell, CellHost};
use flat_host::FlatHost;
use row_core::{Input, bind_add};

fn cells(values: &[i64]) -> cell_host::Column {
    cell_host::Column {
        dtype: cell_host::NativeType { family: "signed", bits: 64 },
        cells: values.iter().map(|&value| Cell { tag: 7, value }).collect(),
    }
}

fn main() -> Result<(), &'static str> {
    let lhs = flat_host::Column {
        dtype: flat_host::NativeType::Signed64,
        values: vec![1, 2, 3],
    };
    let rhs = flat_host::Column {
        dtype: flat_host::NativeType::Signed64,
        values: vec![10, 20, 30],
    };
    let add_flat = bind_add::<FlatHost>(&lhs.dtype, &rhs.dtype)?;
    let flat = add_flat.execute(&lhs, &rhs)?;
    assert_eq!(flat, vec![11, 22, 33]);
    let flat_view = <i64 as Input<FlatHost>>::bind(&lhs)?;
    assert_eq!(flat_view.as_ptr(), lhs.values.as_ptr());

    let cell_lhs = cells(&[1, 2, 3]);
    let cell_rhs = cells(&[10, 20, 30]);
    let add_cells = bind_add::<CellHost>(&cell_lhs.dtype, &cell_rhs.dtype)?;
    let cell = add_cells.execute(&cell_lhs, &cell_rhs)?;
    assert_eq!(cell, flat);
    let cell_view = <i64 as Input<CellHost>>::bind(&cell_lhs)?;
    assert_eq!(cell_view.as_ptr(), cell_lhs.cells.as_ptr());

    let opaque_flat = flat_host::NativeType::Extension("foreign.money");
    assert!(bind_add::<FlatHost>(&opaque_flat, &rhs.dtype).is_err());
    let opaque_cell = cell_host::NativeType { family: "money", bits: 64 };
    assert!(bind_add::<CellHost>(&opaque_cell, &cell_rhs.dtype).is_err());
    let unsupported = flat_host::Column { dtype: opaque_flat, values: vec![1, 2, 3] };
    assert!(add_flat.execute(&unsupported, &rhs).is_err());

    assert_eq!(add_cells.execute(&cells(&[i64::MAX]), &cells(&[1])), Err("integer overflow"));
    assert_eq!(add_cells.execute(&cells(&[1]), &cells(&[])), Err("inputs must have equal lengths"));

    println!("flat host: {flat:?}");
    println!("cell host: {cell:?}");
    println!("borrowed views alias their original storage");
    println!("opaque semantics rejected at binding and batch access");
    println!("overflow and length mismatch rejected");

    Ok(())
}
```

</details>

<details>
<summary>borrow_escape.rs</summary>

```rust
// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use flat_host::{Column, FlatHost, NativeType};
use row_core::Input;

pub fn escape() -> <i64 as Input<FlatHost>>::View<'static> {
    let local = Column { dtype: NativeType::Signed64, values: vec![1] };
    <i64 as Input<FlatHost>>::bind(&local).unwrap()
}
```

</details>
