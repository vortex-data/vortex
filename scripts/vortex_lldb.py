# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Vortex summaries and array children for LLDB (including rust-lldb and CodeLLDB).

From the repository root, with a debug build of compress-bench:

    rust-lldb -o 'command script import scripts/vortex_lldb.py' -- \
        target/debug/compress-bench \
        --datasets '^wide table cols=100 chunks=1 rows=1000$' \
        --formats vortex --ops compress --iterations 1 --threads 1

Then:

    (lldb) breakpoint set --file canonical.rs --line 509
    (lldb) run
    (lldb) frame variable array
    (lldb) frame variable array.encoding_id array.dtype array.len
    (lldb) frame variable array.data
    (lldb) frame variable --raw-output array

The array's synthetic children expose ArrayInner fields, with the erased data
cast to its concrete type when vtable symbols and debug types are available.
Use `type category disable vortex` to restore the standard Rust presentation.
Scheme trait objects show dyn Scheme<concrete_type> @ address; Arc<dyn Scheme>
also shows strong and weak reference counts.

Formatting only reads memory and debug information; it never calls the inferior.
ID resolution depends on the current lasso/DashMap/hashbrown layouts. Unsupported
layouts or unavailable memory fall back to numeric IDs, never guessed names.
"""

from __future__ import annotations

import json
import re

import lldb

_ARRAY_REF = "vortex_array::array::erased::ArrayRef"
_TYPED_ARRAY = r"^vortex_array::array::typed::Array<.+>$"
_ID = "vortex_session::registry::Id"
_INTERNER = "vortex_session::registry::INTERNER"
_CATEGORY = "vortex"
_MAX_BUCKETS = 1 << 20

# Cache only one process stop: IDs and vtable addresses must not survive a restart.
_cache_key: tuple[int, int] | None = None
_id_names: dict[int, str] = {}
_concrete_types: dict[int, str | None] = {}


def _raw(value: lldb.SBValue) -> lldb.SBValue:
    return value.GetNonSyntheticValue()


def _field(value: lldb.SBValue, *path: str) -> lldb.SBValue:
    for name in path:
        value = _raw(value).GetChildMemberWithName(name)
    return _raw(value)


def _uint(value: lldb.SBValue) -> int:
    error = lldb.SBError()
    number = _raw(value).GetValueAsUnsigned(error)
    if not value.IsValid() or error.Fail():
        raise ValueError("unavailable integer")
    return number


def _scalar(value: lldb.SBValue) -> int:
    """Unwrap Id/Spur/NonZero and atomic integer wrappers without invoking code."""
    for _ in range(12):
        value = _raw(value)
        if value.GetNumChildren() != 1:
            return _uint(value)
        value = value.GetChildAtIndex(0)
    raise ValueError("unknown integer wrapper")


def _unwrap_to(value: lldb.SBValue, field: str) -> lldb.SBValue:
    for _ in range(12):
        value = _raw(value)
        if value.GetChildMemberWithName(field).IsValid():
            return value
        if value.GetNumChildren() != 1:
            break
        value = value.GetChildAtIndex(0)
    raise ValueError(f"cannot find {field}")


def _memory(process: lldb.SBProcess, address: int, size: int) -> bytes:
    error = lldb.SBError()
    data = process.ReadMemory(address, size, error)
    if error.Fail() or len(data) != size:
        raise ValueError("unreadable memory")
    return data


def _string(value: lldb.SBValue) -> str:
    value = _unwrap_to(value, "data_ptr")
    size = _uint(_field(value, "length"))
    if size > 4096:
        raise ValueError("invalid ID string length")
    if not size:
        return ""
    address = _uint(_field(value, "data_ptr"))
    return _memory(value.GetProcess(), address, size).decode("utf-8")


def _at(owner: lldb.SBValue, name: str, address: int, ty: lldb.SBType) -> lldb.SBValue:
    if not ty.IsValid() or not address:
        raise ValueError("unavailable type or address")
    value = owner.CreateValueFromAddress(name, address, ty)
    if not value.IsValid() or value.GetError().Fail():
        raise ValueError("unavailable value")
    return value


def _read_ids(target: lldb.SBTarget) -> dict[int, str]:
    interner = target.FindFirstGlobalVariable(_INTERNER)
    # LazyLock -> UnsafeCell<Data<T>> -> union's value -> ManuallyDrop<T>.
    rodeo = _unwrap_to(_field(interner, "data", "value", "value"), "strings")
    shards = _unwrap_to(_field(rodeo, "strings", "shards"), "data_ptr")
    count = _uint(_field(shards, "length"))
    pointer = _field(shards, "data_ptr")
    base = _uint(pointer)
    shard_type = pointer.GetType().GetPointeeType()
    stride = shard_type.GetByteSize()
    if not base or not stride or not 0 < count <= 4096:
        return {}

    names = {}
    remaining = _MAX_BUCKETS
    for shard_index in range(count):
        shard = _at(shards, "shard", base + shard_index * stride, shard_type)
        lock = _unwrap_to(shard, "raw")
        # Do not inspect a shard while a writer is changing its buckets.
        if _scalar(_field(lock, "raw", "state")):
            continue
        table = _unwrap_to(_field(lock, "data", "value"), "table")
        header = _field(table, "table")
        items = _uint(_field(header, "items"))
        if not items:
            continue
        capacity = _uint(_field(header, "bucket_mask")) + 1
        if not 0 < items <= capacity <= remaining or capacity & (capacity - 1):
            raise ValueError("invalid hash table")
        remaining -= capacity
        ctrl = _uint(_field(header, "ctrl", "pointer"))
        pair_type = table.GetType().GetTemplateArgumentType(0)
        pair_size = pair_type.GetByteSize()
        if not pair_size:
            raise ValueError("unavailable hash bucket type")
        controls = _memory(shards.GetProcess(), ctrl, capacity)
        occupied = [i for i, tag in enumerate(controls) if tag & 0x80 == 0]
        if len(occupied) != items:
            continue
        for index in occupied:
            # hashbrown stores buckets backwards immediately before the control bytes.
            pair = _at(shards, "entry", ctrl - (index + 1) * pair_size, pair_type)
            key = _scalar(_raw(pair).GetChildAtIndex(0))
            names[key] = _string(_raw(pair).GetChildAtIndex(1))
    return names


def _refresh(value: lldb.SBValue):
    global _cache_key, _id_names
    process = value.GetProcess()
    key = (process.GetUniqueID(), process.GetStopID())
    if key != _cache_key:
        _cache_key = key
        _concrete_types.clear()
        try:
            _id_names = _read_ids(value.GetTarget())
        except (ValueError, UnicodeError, RuntimeError):
            _id_names = {}


def _encoding(value: lldb.SBValue) -> str:
    _refresh(value)
    key = _scalar(value)
    return _id_names.get(key, f"Id({key})")


def id_summary(value: lldb.SBValue, _internal_dict: dict[str, object]) -> str:
    try:
        _refresh(value)
        key = _scalar(value)
        name = _id_names.get(key)
        return f"Id({json.dumps(name)})" if name is not None else f"Id({key})"
    except (ValueError, RuntimeError):
        return "Id(?)"


def _arc(value: lldb.SBValue) -> tuple[lldb.SBValue, lldb.SBValue, int]:
    pointer = _field(value, "ptr", "pointer")
    vtable = _field(pointer, "vtable")
    if vtable.IsValid():
        pointer = _field(pointer, "pointer")
    allocation = _raw(pointer.Dereference())
    if not allocation.IsValid() or allocation.GetError().Fail():
        raise ValueError("unavailable Arc allocation")
    return allocation, vtable, _uint(pointer)


def _array(value: lldb.SBValue) -> tuple[lldb.SBValue, lldb.SBValue, lldb.SBValue, int]:
    value = _raw(value)
    if value.GetTypeName() != _ARRAY_REF:
        value = _field(value, "inner")
    allocation, vtable, address = _arc(_raw(value).GetChildAtIndex(0))
    return _field(allocation, "data"), allocation, vtable, address


def _counts(allocation: lldb.SBValue) -> str:
    strong = _scalar(_field(allocation, "strong"))
    weak = _scalar(_field(allocation, "weak"))
    # Arc includes an implicit weak reference while strong references exist.
    return f"strong={strong}, weak={max(0, weak - (strong > 0))}"


def _dtype(value: lldb.SBValue) -> str:
    # Reuse rust-lldb's active-enum-variant decoding; raw DWARF contains all variants.
    value = value.GetSyntheticValue()
    summary = value.GetSummary() or value.GetValue() or "?"
    variant = summary.split("(", 1)[0].split("::")[-1]
    children = list(value)
    nullable = any(child.GetValue() == "Nullable" for child in children)
    suffix = "?" if nullable else ""
    if variant == "Primitive" and children:
        return (children[0].GetValue() or "?").split("::")[-1].lower() + suffix
    names = {
        "Null": "null",
        "Bool": "bool",
        "Utf8": "utf8",
        "Binary": "binary",
        "Struct": "{...}",
        "List": "list(...)",
        "FixedSizeList": "fixed_size_list(...)",
        "Decimal": "decimal(...)",
        "Extension": "ext",
        "Union": "union()",
        "Variant": "variant",
    }
    return names.get(variant, variant) + suffix


def _concrete_type(value: lldb.SBValue, vtable: lldb.SBValue) -> str | None:
    _refresh(value)
    address = _uint(vtable)
    if address not in _concrete_types:
        name = None
        process = value.GetProcess()
        target = value.GetTarget()
        # The first three vtable entries are drop/size/alignment, followed by methods.
        for index in range(3, 7):
            error = lldb.SBError()
            method = process.ReadPointerFromMemory(address + index * target.GetAddressByteSize(), error)
            if error.Fail():
                break
            resolved = target.ResolveLoadAddress(method)
            symbol = resolved.GetFunction().GetName() or resolved.GetSymbol().GetName() or ""
            match = re.match(
                r"^<(.+) as (?:core::fmt::Debug|vortex_array::array::(?:DynArrayData|plugin::ArrayPlugin)"
                r"|vortex_compressor::scheme::(?:Scheme|AsDynScheme))>::",
                symbol,
            )
            if match:
                name = match.group(1)
                break
        _concrete_types[address] = name
    return _concrete_types[address]


def array_summary(value: lldb.SBValue, _internal_dict: dict[str, object]) -> str:
    try:
        inner, allocation, _, address = _array(value)
        encoding = _encoding(_field(inner, "encoding_id"))
        dtype = _dtype(inner.GetChildMemberWithName("dtype"))
        length = _uint(_field(inner, "len"))
        return f"{encoding}({dtype}, len={length}) @ {address:#x} [{_counts(allocation)}]"
    except (ValueError, RuntimeError):
        return "Array(?)"


class ArrayChildren:
    """Expose metadata and concrete encoding data without consuming the ArrayRef."""

    def __init__(self, value: lldb.SBValue, _internal_dict: dict[str, object]):
        self.value = value
        self.children = []
        self.update()

    def update(self) -> bool:
        self.children = []
        try:
            inner, _, vtable, _ = _array(self.value)
            self.children = [
                inner.GetChildMemberWithName(name) for name in ("encoding_id", "dtype", "len", "slots", "stats", "data")
            ]
            name = _concrete_type(self.value, vtable)
            if name:
                ty = self.value.GetTarget().FindFirstType(name)
                if ty.IsValid():
                    data = self.children[-1]
                    if ty.GetByteSize() == 0:
                        # Apple LLDB cannot CreateValueFromAddress for zero-sized types.
                        concrete = data.Cast(ty)
                    else:
                        # DWARF describes the erased tail with alignment 1. Its actual
                        # address must be rounded up to the concrete vtable alignment.
                        process = self.value.GetProcess()
                        error = lldb.SBError()
                        align = process.ReadPointerFromMemory(
                            _uint(vtable) + 2 * self.value.GetTarget().GetAddressByteSize(), error
                        )
                        if error.Fail() or not 0 < align <= 4096 or align & (align - 1):
                            raise ValueError("invalid vtable alignment")
                        address = (data.GetLoadAddress() + align - 1) & -align
                        concrete = _at(data, "data", address, ty)
                    if concrete.IsValid():
                        self.children[-1] = concrete
        except (ValueError, RuntimeError):
            pass
        return False

    def num_children(self) -> int:
        return len(self.children)

    def get_child_index(self, name: str) -> int:
        return next((i for i, child in enumerate(self.children) if child.GetName() == name), -1)

    def get_child_at_index(self, index: int) -> lldb.SBValue | None:
        return self.children[index] if 0 <= index < len(self.children) else None


def _trait_summary(value: lldb.SBValue, trait_name: str) -> str:
    try:
        value = _raw(value)
        if value.GetTypeName().startswith("alloc::sync::Arc<"):
            allocation, vtable, address = _arc(value)
            name = _concrete_type(value, vtable) or "?"
            return f"Arc<dyn {trait_name}<{name}>> @ {address:#x} [{_counts(allocation)}]"
        vtable = _field(value, "vtable")
        pointer = _field(value, "pointer")
        name = _concrete_type(value, vtable) or "?"
        return f"dyn {trait_name}<{name}> @ {_uint(pointer):#x}"
    except (ValueError, RuntimeError):
        return f"dyn {trait_name}<?>"


def plugin_summary(value: lldb.SBValue, _internal_dict: dict[str, object]) -> str:
    return _trait_summary(value, "ArrayPlugin")


def scheme_summary(value: lldb.SBValue, _internal_dict: dict[str, object]) -> str:
    return _trait_summary(value, "Scheme")


def __lldb_init_module(debugger: lldb.SBDebugger, _internal_dict: dict[str, object]):
    global _cache_key
    _cache_key = None
    debugger.DeleteCategory(_CATEGORY)
    category = debugger.CreateCategory(_CATEGORY)
    for name, regex, callback in (
        (_ARRAY_REF, False, "array_summary"),
        (_TYPED_ARRAY, True, "array_summary"),
        (_ID, False, "id_summary"),
        (r"^alloc::sync::Arc<dyn vortex_array::array::plugin::ArrayPlugin.*>$", True, "plugin_summary"),
        (r"^(&|&mut |\*const |\*mut )?dyn vortex_array::array::plugin::ArrayPlugin$", True, "plugin_summary"),
        (r"^alloc::sync::Arc<dyn vortex_compressor::scheme::Scheme(\s*\+[^>]*)?>$", True, "scheme_summary"),
        (
            r"^(&|&mut |\*const |\*mut )?dyn vortex_compressor::scheme::Scheme(\s*\+.*)?$",
            True,
            "scheme_summary",
        ),
    ):
        spec = lldb.SBTypeNameSpecifier(name, regex)
        summary = lldb.SBTypeSummary.CreateWithFunctionName(f"{__name__}.{callback}")
        summary.SetOptions(lldb.eTypeOptionHideChildren)
        category.AddTypeSummary(spec, summary)
        if callback == "array_summary":
            synthetic = lldb.SBTypeSynthetic.CreateWithClassName(f"{__name__}.ArrayChildren")
            category.AddTypeSynthetic(spec, synthetic)
    debugger.HandleCommand(f"type category enable {_CATEGORY}")
    print("vortex_lldb: registered formatters (memory reads only)")
