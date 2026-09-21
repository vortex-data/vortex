<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# DType usage inventory for the RowFn framework

[Type overview](README.md). This source inventory uses Vortex `f5b3b26`. The row module is unchanged
at the main research baseline `96bd521`.

This file lists every place where the RowFn framework (`vortex-array/src/scalar_fn/unstable/row/`)
and its three users (`vortex-array/src/scalar_fn/fns/binary/numeric/row.rs`,
`vortex-spatial/src/scalar_fn/*.rs`, `vortex-tensor/src/scalar_fns/*.rs`) touch Vortex's type
system: `DType`, `PType`, `Nullability`, `ExtDType`/`ExtDTypeRef`, `Scalar`, and `ScalarValue`. Each
touchpoint is tagged with the abstract type-system operation it needs. The result is short: the
framework core needs about ten operations, and only three functions ever look inside a `DType`
(`validate_output_label`, `result_dtype`, and `validate_output`). Everything else is either an
opaque pass-through, a predicate that an element type implements for itself, or an error message.

## Method

- Search: `rg -n '\b(DType|PType|Nullability|ExtDType|ExtDTypeRef|ScalarValue|Scalar)\b'` over the row module and the three user directories, plus `dtype()`, `storage_dtype`, `is_nullable`, `nullability(`, `with_nullability`, `eq_ignore_nullability`, `as_nonnullable`, `as_nullable`, and the user helper functions (`is_native_geometry`, `polygon_storage_dtype`, `validate_tensor_float_input`, `tensor_element_ptype`).
- Every non-test hit is listed below. `use` lines are not listed as separate rows; they carry no operation.
- Test code and the bench are summarized per file at the end. They only exercise the operations that production code already needs.
- Line numbers were taken from the current tree on 2026-09-21.

### Operation codes

| Code | Operation | Meaning |
|---|---|---|
| PASS | Opaque pass-through | Store, borrow, or forward a type value or a slice of them. Needs only `Clone`. |
| EQ | Equality | Compare two types for exact equality (includes nullability in Vortex). |
| EQN | Equality with relaxed nullability | Covers outer normalization in output validation and recursive `eq_ignore_nullability` in tensor dispatch. These are different contracts. |
| ISN | Nullability read | `is_nullable()` or `nullability()`. |
| WN | Nullability write | `with_nullability(..)`, `as_nonnullable()`, or building a type with `Nullability::NonNullable`. |
| NOR | Nullability union | OR over the input nullabilities. |
| PK | Primitive kind | Match `DType::Primitive(ptype, _)`, compare a `PType`, ask `is_float()`, or map a `PType` to a Rust type and back. |
| SK | Simple kind match | `matches!(dtype, DType::Bool(_))` or `DType::Utf8(_)`, ignoring nullability. |
| PAR | Parametric construction or read | Build `FixedSizeList(elem, size, ..)` or nested `List<List<Struct>>`, or read a list size. |
| XM | Extension match | `let DType::Extension(ext) = ..` or `as_extension_opt()`. |
| XW | Extension wrap | Build `DType::Extension(label)` as a type, or `ExtensionArray::try_new(label, storage)` as an array. |
| XS | Extension storage unwrap | `ext.storage_dtype()`, or unwrap an extension array or scalar to its storage. |
| XI | Extension identity and metadata | `ext.is::<T>()`, `ext.metadata::<T>()`, `metadata_opt::<AnyTensor>()`, `ExtDType::<T>::try_new(metadata, storage)`. |
| DISP | Display | Format a type into an error message. |
| VAL | Value system | Build a typed null `Scalar`, test `is_null()`, or extract a `ScalarValue::{Primitive,Bool,Utf8}` from a batch constant. |
| CAST | Cast | `array.cast(dtype)` to change outer nullability only. |
| ADT | Array type read | `array.dtype()` to obtain the type of a runtime column. |

## Framework production code

| Location | What it does | Ops |
|---|---|---|
| `vortex-array/src/scalar_fn/unstable/row/row_fn.rs:88` | `dispatch(.., args: &[DType], ..)`: the function receives the argument types. | PASS |
| `vortex-array/src/scalar_fn/unstable/row/visitor/row_visitor.rs:78` | `with_output_dtype(self, dtype: DType)`: declares the output label. The doc example at `:71` builds `DType::Extension(ext.with_nullability(NonNullable))`. | PASS (XW, WN in docs) |
| `vortex-array/src/scalar_fn/unstable/row/visitor/plan.rs:40` | `BatchPlanner.dtypes: &[DType]`. | PASS |
| `visitor/plan.rs:43` | `BatchPlanner.output_dtype: Option<DType>`. | PASS |
| `visitor/plan.rs:50` | `BatchPlanner::new(dtypes)`. | PASS |
| `visitor/plan.rs:64` | `with_output_dtype` stores the declared dtype. | PASS |
| `visitor/plan.rs:137` | `BatchPlan.storage_dtype: DType`. | PASS |
| `visitor/plan.rs:141` | `BatchPlan.output_label: Option<ExtDTypeRef>`: the label is an extension dtype. | PASS |
| `visitor/plan.rs:153` to `:158` | `BatchPlan::new` calls `validate_output_label`. | PASS |
| `visitor/plan.rs:170` | `storage_dtype()` getter. | PASS |
| `visitor/plan.rs:175` to `:177` | `output_dtype()`: `DType::Extension(output_label.clone())` when a label exists. | XW |
| `visitor/plan.rs:188` to `:193` | `result_dtype(args)`: `output_dtype.nullability() \| Nullability::from(args.iter().any(DType::is_nullable))`, then `with_nullability`. | ISN, NOR, WN |
| `visitor/plan.rs:201` to `:208` | `relabel_output(values)`: `output_label.with_nullability(values.dtype().nullability())`, then `ExtensionArray::try_new(label, values)`. | ADT, ISN, WN, XW |
| `visitor/plan.rs:220` to `:225` | `ensure_reproduced_by`: `actual.storage_dtype == self.storage_dtype`, both displayed on failure. | EQ, DISP |
| `visitor/plan.rs:227` to `:232` | `actual.output_label == self.output_label` (`ExtDTypeRef` equality), displayed via `output_dtype()`. | EQ, XW, DISP |
| `visitor/plan.rs:253` to `:256` | `validate_output_label`: `!output_dtype.is_nullable()`, displayed. | ISN, DISP |
| `visitor/plan.rs:258` | `output_dtype == *storage_dtype` returns "no label". | EQ |
| `visitor/plan.rs:262` to `:267` | `let DType::Extension(output_label) = output_dtype else bail`, displaying both dtypes. | XM, DISP |
| `visitor/plan.rs:268` to `:273` | `*output_label.storage_dtype() == *storage_dtype`, displayed. | XS, EQ, DISP |
| `vortex-array/src/scalar_fn/unstable/row/visitor/check.rs:80` to `:82` | `validate_owned_visit(dtypes)` calls `Args::validate(dtypes)`. | PASS |
| `visitor/check.rs:84` to `:88` | `Out::element_dtype()` must satisfy `!dtype.is_nullable()`, displayed. | ISN, DISP |
| `visitor/check.rs:94` to `:101` | `validate_sink_visit(dtypes, params)` calls `Args::validate(dtypes)`. | PASS |
| `visitor/check.rs:103` to `:107` | `Sink::storage_dtype(params)` must satisfy `!dtype.is_nullable()`, displayed. | ISN, DISP |
| `vortex-array/src/scalar_fn/unstable/row/visitor/execute.rs:54`, `:60`, `:72`, `:92` | `ExecuteRows`: `dtypes` field, `output_dtype` field, constructor, `with_output_dtype`. | PASS |
| `visitor/execute.rs:224`, `:230`, `:245`, `:267` | `ExecuteValidRows`: same four. | PASS |
| `visitor/execute.rs:368`, `:374`, `:389`, `:411` | `ExecuteFilteredRows`: same four. | PASS |
| `vortex-array/src/scalar_fn/unstable/row/visitor/retry.rs:49`, `:77` | `ExecuteDenseWithRetry`: `output_dtype` field and `with_output_dtype`. | PASS |
| `vortex-array/src/scalar_fn/unstable/row/vtable.rs:60` | `ScalarFnVTable::return_dtype(.., args: &[DType]) -> DType` delegates to `row_fn_return_dtype`. | PASS |
| `vtable.rs:96` to `:102` | `row_fn_return_dtype`: plans with `BatchPlanner::new(args)` and returns `plan.result_dtype(args)`. | PASS |
| `vtable.rs:142` to `:148` | `execute_nullary_rows`: plans with `&[]`, `result_dtype(&[])`, and `relabel_output`. | PASS |
| `vortex-array/src/scalar_fn/unstable/row/batch/mod.rs:46` | `RowFnExecutionArgs.arg_dtypes: SmallVec<[DType; 4]>`. | PASS |
| `batch/mod.rs:54` | `RowFnExecutionArgs.result_dtype: DType`. | PASS |
| `vortex-array/src/scalar_fn/unstable/row/batch/args.rs:31`, `:42`, `:54` | `BorrowedRowFnArgs.dtypes` field, constructor, getter. | PASS |
| `vortex-array/src/scalar_fn/unstable/row/batch/planning.rs:25` | `plan: impl FnOnce(&[DType]) -> VortexResult<BatchPlan>`. | PASS |
| `batch/planning.rs:41` to `:42` | `arg_dtypes = inputs.iter().map(\|input\| input.dtype().clone())`. | ADT |
| `batch/planning.rs:43` to `:44` | `plan(&arg_dtypes)` and `plan.result_dtype(&arg_dtypes)`. | PASS |
| `vortex-array/src/scalar_fn/unstable/row/batch/execute/output.rs:20` | `all_null`: `Scalar::null(self.result_dtype.clone())` into a `ConstantArray`. | VAL |
| `batch/execute/output.rs:62` to `:64` | `finalize_kernel_output(.., result_dtype: &DType, ..)`. | PASS |
| `batch/execute/output.rs:79` to `:81` | `validate_output(.., result_dtype: &DType, ..)`. | PASS |
| `batch/execute/output.rs:91` to `:96` | `values.dtype().with_nullability(result_dtype.nullability()) == *result_dtype`, displayed. This is "equal ignoring outer nullability" spelled out. | ADT, ISN, WN, EQN, DISP |
| `batch/execute/output.rs:106` to `:110` | `cast_output_nullability`: `values.dtype() == result_dtype`, else `values.cast(result_dtype.clone())`. | ADT, EQ, CAST |
| `vortex-array/src/scalar_fn/unstable/row/batch/execute/mod.rs:62` | Null-constant short circuit: `constant.scalar().is_null()`. | VAL |
| `batch/execute/mod.rs:73` | All-constant fast path uses `batch_const`, which sees through `Extension` wrappers (see `element_tuple.rs:127`). | XS |
| `vortex-array/src/scalar_fn/unstable/row/types/element/input.rs:61` | `InputElement::validate(dtype: &DType)`: the per-element predicate. | PASS (the impls decide) |
| `vortex-array/src/scalar_fn/unstable/row/types/element/output.rs:26` | `OutputElement::element_dtype() -> DType`: a type that is a property of the Rust type. | PASS (the impls decide) |
| `vortex-array/src/scalar_fn/unstable/row/types/sink/mod.rs:129` | `OutputSink::storage_dtype(params) -> DType`. | PASS (the impls decide) |
| `vortex-array/src/scalar_fn/unstable/row/types/element/primitive.rs:33` to `:43` | `impl<T: NativePType> InputElement for T`: `let DType::Primitive(ptype, _) = dtype`, `vortex_ensure_eq!(*ptype, T::PTYPE)`, displaying `T::PTYPE` and the dtype. Nullability is ignored. | PK, DISP |
| `types/element/primitive.rs:50` to `:65` | `decode_constant`: `ScalarValue::Primitive(value)`, then `value.cast::<T>()`. | VAL |
| `types/element/primitive.rs:101` to `:102` | `element_dtype()`: `DType::Primitive(T::PTYPE, Nullability::NonNullable)`. | PK, WN |
| `vortex-array/src/scalar_fn/unstable/row/types/element/bool.rs:33` to `:38` | `matches!(dtype, DType::Bool(_))`, displayed. | SK, DISP |
| `types/element/bool.rs:45` to `:55` | `decode_constant`: `ScalarValue::Bool(value)`. | VAL |
| `types/element/bool.rs:93` to `:94` | `element_dtype()`: `DType::Bool(Nullability::NonNullable)`. | SK, WN |
| `vortex-array/src/scalar_fn/unstable/row/types/element/utf8.rs:158` | `decode_utf8` passes `array.dtype().clone()` to `VarBinViewArray::try_new` for re-validation. | ADT |
| `types/element/utf8.rs:168` to `:180` | `decode_constant_utf8`: `ScalarValue::Utf8(value)`. | VAL |
| `types/element/utf8.rs:224` to `:229` | `matches!(dtype, DType::Utf8(_))`, displayed. | SK, DISP |
| `vortex-array/src/scalar_fn/unstable/row/types/element/tuple/element_tuple.rs:127` to `:128` | `batch_const`: `array.as_opt::<Extension>()` whose `storage_array()` is a `Constant` counts as a batch constant (kept wrapped to preserve the extension dtype). | XS |
| `types/element/tuple/element_tuple.rs:169` | `ElementTuple::validate(dtypes: &[DType])`. | PASS |
| `types/element/tuple/element_tuple.rs:249` to `:256` | Arity 0: `dtypes.len() == 0`. | PASS |
| `types/element/tuple/element_tuple.rs:313` to `:323` | Arity 1..12: `dtypes.len() == N`, then `T_i::validate(&dtypes[i])` per argument. | PASS |
| `vortex-array/src/scalar_fn/unstable/row/types/sink/uninit_element.rs:99` to `:100` | `storage_dtype` returns `T::element_dtype()`. | PASS |
| `vortex-array/src/scalar_fn/unstable/row/types/sink/fixed_size_list.rs:104` to `:109` | `DType::FixedSizeList(Arc::new(T::element_dtype()), size_as_u32, Nullability::NonNullable)`. | PAR, WN |
| `vortex-array/src/scalar_fn/unstable/row/types/sink/utf8.rs:101` to `:102` | `storage_dtype`: `DType::Utf8(Nullability::NonNullable)`. | SK, WN |
| `types/sink/utf8.rs:143` | `finish` passes `DType::Utf8(NonNullable)` to `VarBinViewArray::new_handle`. | SK, WN |

## User production code

| Location | What it does | Ops |
|---|---|---|
| `vortex-array/src/scalar_fn/fns/binary/numeric/mod.rs:64` | `Binary::execute` routes `DType::Primitive(..)` inputs to the row kernel; decimals stay columnar (`:47`, `:77` also match `Primitive`/`Decimal` and call `with_nullability`). | PK, WN |
| `vortex-array/src/scalar_fn/fns/binary/numeric/row.rs:69` | `dispatch(.., args: &[DType], ..)`. | PASS |
| `numeric/row.rs:72` to `:75` | `PType::try_from(args.first()?)`: extract the primitive kind of the first argument; errors on non-primitive. | PK |
| `numeric/row.rs:77` | `match_each_native_ptype!(ptype, \|T\| ..)`: kind to Rust type. | PK |
| `numeric/row.rs:111` | `T::PTYPE.is_float()`: kind class query to choose deferred vs immediate errors. | PK |
| `vortex-spatial/src/scalar_fn/row.rs:82` to `:86` | `GeometryRow::validate`: `is_native_geometry(dtype)`, displayed. | XM, XI, DISP |
| `vortex-spatial/src/extension/mod.rs:68` to `:77` | `is_native_geometry`: `as_extension_opt()` then `ext.is::<Point>() \|\| .. \|\| ext.is::<Rect>()` (seven identities). | XM, XI |
| `vortex-spatial/src/scalar_fn/row.rs:155` to `:156` | `PolygonSink::storage_dtype`: `polygon_storage_dtype(Dimension::Xy, Nullability::NonNullable)`. | PAR, WN |
| `vortex-spatial/src/extension/polygon.rs:95` to `:99` | `polygon_storage_dtype`: `List(List(Struct{x,y}, NonNullable), nullability)`. | PAR, WN |
| `vortex-spatial/src/scalar_fn/convex_hull.rs:32` to `:52` | `convex_hull_dtype`: arity check, `as_extension_opt()`, `is::<MultiPoint>()`, `metadata::<MultiPoint>().clone()`, `ExtDType::<Polygon>::try_new(metadata, polygon_storage_dtype(..)).erased()`, two displays. | XM, XI, PAR, DISP |
| `convex_hull.rs:98` | `visitor.with_output_dtype(DType::Extension(convex_hull_dtype(args)?))`. | XW |
| `vortex-spatial/src/scalar_fn/area.rs:60`, `contains.rs:60`, `distance.rs:64`, `intersects.rs:59` | `_args: &[DType]` unused: these dispatch on `GeometryRow` alone. | PASS |
| `vortex-tensor/src/scalar_fns/row.rs:39` to `:51` | `tensor_element_ptype(args)`: `first.eq_ignore_nullability(argument)` for every other argument, two displays, then `validate_tensor_float_input(first)?.element_ptype()`. | EQN, XM, XI, PK, DISP |
| `vortex-tensor/src/utils.rs:31` to `:46` | `validate_tensor_float_input`: `as_extension_opt()`, `metadata_opt::<AnyTensor>()`, `ptype.is_float()`. | XM, XI, PK |
| `vortex-tensor/src/scalar_fns/row.rs:123` to `:132` | `TensorRow<T>::validate`: `validate_tensor_float_input(dtype)`, then `element_ptype() == T::PTYPE`, displayed. | XM, XI, PK, DISP |
| `tensor/row.rs:143` | `decode`: `validate_tensor_float_input(array.dtype())?.list_size()` (extension metadata gives the row width). | ADT, XI, PAR |
| `tensor/row.rs:148` to `:151` | `decode_constant`: same `list_size()` read, then `constant.scalar().as_extension().to_storage_scalar()`. | ADT, XI, PAR, XS, VAL |
| `vortex-tensor/src/scalar_fns/cosine_similarity.rs:86`, `:89`; `inner_product.rs:86`, `:89`; `l2_norm.rs:90`, `:93` | `args: &[DType]` then `match_each_float_ptype!(tensor_element_ptype(args)?, \|T\| ..)`. | PASS, PK |

## Tests and benches (summary)

| File | Touchpoints | What they exercise |
|---|---|---|
| `vortex-array/src/scalar_fn/unstable/row/batch/tests.rs` (about 60 hits, for example `:108`, `:233`, `:256`, `:644` to `:652`, `:1006` to `:1012`, `:1088`, `:1120`, `:1225` to `:1235`, `:1280`) | `validate` delegations, `element_dtype`/`storage_dtype` returning `DType::from(i64::PTYPE)`, `Timestamp::new(..).erased()` labels, `DType::Extension(..)`, `Nullability::{Nullable,NonNullable}`, `Scalar::null(..)`, `Scalar::list_empty(..)`, and `_args: &[DType]` in every test `dispatch`. | PASS, PK, WN, XW, VAL, EQ |
| `vortex-array/src/scalar_fn/unstable/row/types/element/tuple/tests.rs:24` to `:25`, `:49`, `:162` | `validate` delegation; a `Timestamp` extension over a constant to test `batch_const`. | PASS, XW, XS |
| `vortex-array/src/scalar_fn/unstable/row/types/element/utf8.rs:288` to `:352` (tests) | `DType::Utf8(Nullable/NonNullable)` array construction; `Scalar::from(&str)`. | SK, WN, VAL |
| `vortex-array/src/scalar_fn/unstable/row/execute/sink.rs:350`, `:370` to `:371` (tests) | A test sink returning `DType::from(i64::PTYPE)`. | PK |
| `vortex-array/src/scalar_fn/unstable/row/vtable.rs:248`, `:288`, `:310`, `:333` (tests) | `_args: &[DType]` in test `dispatch` methods. | PASS |
| `vortex-array/benches/row_fn_output.rs:22`, `:140`, `:203`, `:224`, `:254`, `:284`, `:305` | A bench element delegating `validate` to `i64`; `_args: &[DType]` in bench `dispatch` methods. | PASS |
| `vortex-spatial/src/scalar_fn/{area,contains,distance,intersects,convex_hull,collect,envelope,length,make_line}.rs` tests | `DType::Primitive(PType::I32/F64, NonNullable)` to assert rejection, `.as_nullable()`, `is_nullable()`, `Scalar::null(..)`. | PK, WN, ISN, VAL |
| `vortex-tensor/src/scalar_fns/{l2_norm,l2_normalize,cosine_similarity,inner_product}.rs` tests | `DType::FixedSizeList(..)` storage, `ExtDType::<Vector>::try_new(..)`, `Scalar::null(DType::Extension(..))`, struct field nullability checks. | PAR, XI, XW, VAL, ISN |

Files `vortex-spatial/src/scalar_fn/{collect,envelope,length,make_line}.rs` implement
`ScalarFnVTable` directly, not `RowFn`. They are not row functions and are out of scope here, but
they show what a richer function needs: `DType::List` matching, `as_nonnullable()`,
`ExtDType::try_new` with derived metadata, and `coordinate_dimension(storage_dtype)`
(`make_line.rs:82` to `:90`).

## The minimal operation set

Grouping the rows above, the framework and its users need these operations from a type system.
"Core" means the framework itself needs it. "Element" means an `InputElement`, `OutputElement`, or
`OutputSink` implementation needs it. "User" means a `RowFn::dispatch` needs it.

| # | Operation | Core | Element | User | Where |
|---|---|---|---|---|---|
| 1 | Clone, store, and forward a type (PASS) | yes | yes | yes | everywhere |
| 2 | Exact equality (EQ) | yes | no | no | `plan.rs:221`, `:228`, `:258`; `output.rs:107` |
| 3 | Equality with relaxed nullability (EQN) | yes | no | yes | `output.rs:92`; `tensor/row.rs:46` |
| 4 | Read outer nullability (ISN) | yes | no | no | `plan.rs:191`, `:206`, `:254`; `check.rs:86`, `:105`; `output.rs:92` |
| 5 | Set outer nullability, or build a non-nullable type (WN) | yes | yes | yes | `plan.rs:193`, `:206`; `output.rs:92`; every `element_dtype`/`storage_dtype` |
| 6 | OR of input nullabilities (NOR) | yes | no | no | `plan.rs:191` |
| 7 | Primitive kind: match, compare, class query, kind to Rust type (PK) | no | yes | yes | `primitive.rs:35`, `:38`, `:102`; `numeric/row.rs:72`, `:77`, `:111`; `tensor/row.rs:51`, `:127` |
| 8 | Simple kind match for Bool and Utf8 (SK) | no | yes | no | `bool.rs:35`, `:94`; `utf8.rs:226`; `sink/utf8.rs:102` |
| 9 | Parametric construction and parameter read (PAR) | no | yes | yes | `fixed_size_list.rs:105`; `spatial/row.rs:156`; `tensor/row.rs:143` |
| 10 | Extension label: match, storage unwrap, equality, wrap (XM, XS, XW) | yes | no | yes | `plan.rs:177`, `:208`, `:262`, `:269`; `convex_hull.rs:98`; `element_tuple.rs:127` |
| 11 | Extension identity and metadata (XI) | no | yes | yes | spatial `is_native_geometry`; tensor `AnyTensor` |
| 12 | Display for errors (DISP) | yes | yes | yes | 26 formatted types across 15 rows |
| 13 | Typed null scalar, null test, constant value extraction (VAL) | yes | yes | no | `output.rs:20`; `mod.rs:62`; `primitive.rs:58`; `bool.rs:53`; `utf8.rs:176` |
| 14 | Cast to change outer nullability (CAST) | yes | no | no | `output.rs:110` |
| 15 | Read the type of a runtime column (ADT) | yes | yes | no | `planning.rs:42`; `output.rs:92`, `:107`; `plan.rs:206`; `utf8.rs:158`; `tensor/row.rs:143` |

## Implications for the type boundary

Outer nullability work concentrates in planning and finalization. A separate type-use envelope can
move that work to the adapter. Recursive child constraints still need explicit comparisons.
`eq_ignore_nullability` ignores nested nullability too. Operation EQN records both forms of relaxed
comparison and does not mean they have identical contracts.

The core inspects extension labels while it validates and applies output metadata. Primitive-kind
selection and domain-specific metadata belong to element implementations and function dispatch.
Scalar extraction belongs to the value and decoding boundary.

The [associated-type option](associated-types.md) turns this inventory into a proposed interface.
The [semantic binding contract](portable-contract.md) adds the laws required for portable functions.

## Counts

Production code only (tests excluded). A row that carries several codes counts once per code, except
`DISP` (see below).

| Code | Framework | Users | Total |
|---|---|---|---|
| PASS | 45 | 8 | 53 |
| EQ | 4 | 0 | 4 |
| EQN | 1 | 1 | 2 |
| ISN | 6 | 0 | 6 |
| WN | 8 | 3 | 11 |
| NOR | 1 | 0 | 1 |
| PK | 2 | 8 | 10 |
| SK | 5 | 0 | 5 |
| PAR | 1 | 5 | 6 |
| XM | 1 | 6 | 7 |
| XW | 3 | 1 | 4 |
| XS | 3 | 1 | 4 |
| XI | 0 | 8 | 8 |
| DISP | 20 | 6 | 26 |
| VAL | 5 | 1 | 6 |
| CAST | 1 | 0 | 1 |
| ADT | 5 | 2 | 7 |

`DISP` counts individual formatted types (a message that prints two types counts twice). Every other
code counts table rows.

Reading the table: 45 of the framework's touchpoints are pass-throughs that need nothing from the
type beyond `Clone`. The next largest group is `Display` for error messages. The genuinely
structural operations in the core (EQ, EQN, ISN, WN, NOR, XM, XW, XS, CAST) total 28 sites, and 19
of them are in two files: `visitor/plan.rs` and `batch/execute/output.rs`. The other nine are the
two non-nullable checks in `visitor/check.rs`, five non-nullable constructors in element and sink
impls, and two extension unwraps for constant detection.
