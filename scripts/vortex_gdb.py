# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""
GDB pretty-printers. Usage:

(gdb) source scripts/vortex_gdb.py
...
(gdb) print values_idx_offsets
$1 = vortex.primitive(u32, len=465) @ 0x20005ab0d20 [strong=2, weak=0]
(gdb) print values
$2 = fastlanes.bitpacked(i16, len=75133) @ 0x20005ab09a0 [strong=2, weak=0]
(gdb) print encoding_id
$3 = Id("vortex.primitive")
(gdb) print plugin
$4 = dyn ArrayPlugin<vortex_array::arrays::primitive::vtable::Primitive> @ 0xaaaab6deb168
(gdb) print plugin_arc
$5 = Arc<dyn ArrayPlugin<vortex_array::arrays::primitive::vtable::Primitive> @ 0x20000ab0310> [strong=3, weak=0]
"""

from __future__ import annotations

import re
from collections.abc import Callable

import gdb

_ARRAY_PLUGIN_FQN = "vortex_array::array::plugin::ArrayPlugin"
_DYN_ARRAY_PLUGIN = "dyn " + _ARRAY_PLUGIN_FQN
_AS_STR_NAME = "vortex_session::registry::Id::as_str"
_ARRAY_REF_FQN = "vortex_array::array::erased::ArrayRef"
_ARRAY_TYPED_PREFIX = "vortex_array::array::typed::Array<"
_ID_FQN = "vortex_session::registry::Id"
_ARRAY_INNER_TY_NAME = "vortex_array::array::typed::ArrayInner<dyn vortex_array::array::DynArrayData>"
# "data: T" lives after [strong, weak] inside ArcInner. both are AtomicUsize.
_ARC_INNER_DATA_OFFSET = 16

# DType variant -> Display template. "{n}" is nullability suffix.
_DTYPE_TEMPLATES = {
    "Null": "null",
    "Bool": "bool{n}",
    "Utf8": "utf8{n}",
    "Binary": "binary{n}",
    "Union": "union(){n}",
    "Variant": "variant{n}",
    "Struct": "{{...}}{n}",
    "List": "list(...){n}",
    "FixedSizeList": "fixed_size_list(...){n}",
    "Decimal": "decimal(...){n}",
    "Extension": "ext",
}
_DTYPE_VARIANT_RE = re.compile(r"(?:^|::)DType::(\w+)")
_PTYPE_VARIANT_RE = re.compile(r"(?:^|::)PType::(\w+)")
_NULLABILITY_RE = re.compile(r"(?:^|::)Nullability::(\w+)")


def _usize_size() -> int:
    return gdb.lookup_type("usize").sizeof


def _read_usize(addr: int) -> int:
    return int(gdb.Value(addr).cast(gdb.lookup_type("usize").pointer()).dereference())


def _concrete_type(val: gdb.Value, deref_path: str) -> str | None:
    try:
        gdb.set_convenience_variable("vortex_tmp", val)
        return gdb.parse_and_eval(f"*$vortex_tmp{deref_path}").type.name
    except (gdb.error, AttributeError):
        return None


def _trait_fat_pointer(val: gdb.Value) -> tuple[int, int] | None:
    # field names vary by rustc
    for p in ("pointer", "data_ptr", "__0"):
        for v in ("vtable", "__1"):
            try:
                return int(val[p]), int(val[v])
            except (gdb.error, RuntimeError, KeyError):
                continue
    try:
        fields = [f for f in val.type.fields() if f.name and not f.artificial]
    except (gdb.error, RuntimeError):
        return None
    if len(fields) == 2:
        try:
            return int(val[fields[0].name]), int(val[fields[1].name])
        except (gdb.error, RuntimeError):
            return None
    return None


def _spur_from_id(val: gdb.Value) -> int | None:
    # Id(Spur { key: NonZero<u32>(T) }) -> T
    cur = val
    for _ in range(8):
        try:
            t = cur.type.strip_typedefs()
        except gdb.error:
            break
        if t.code != gdb.TYPE_CODE_STRUCT:
            break
        try:
            fields = [f for f in t.fields() if f.name and not f.artificial and not f.is_base_class]
        except (gdb.error, RuntimeError):
            break
        if len(fields) != 1:
            break
        cur = cur[fields[0].name]
    try:
        return int(cur)
    except (gdb.error, ValueError, TypeError):
        return None


def _str_to_python(s: gdb.Value) -> str | None:
    for ptr_n, len_n in (("data_ptr", "length"), ("ptr", "len"), ("__0", "__1")):
        try:
            data_ptr, length = int(s[ptr_n]), int(s[len_n])
            break
        except (gdb.error, RuntimeError):
            continue
    else:
        return None
    if length == 0:
        return ""
    if data_ptr == 0 or length > 4096:
        return None
    try:
        return gdb.selected_inferior().read_memory(data_ptr, length).tobytes().decode("utf-8")
    except (gdb.error, UnicodeDecodeError):
        return None


def _id_to_string(id_addr: int) -> str | None:
    sym = gdb.lookup_global_symbol(_AS_STR_NAME) or gdb.lookup_static_symbol(_AS_STR_NAME)
    if sym is None:
        return None
    try:
        id_ptr = gdb.Value(id_addr).cast(gdb.lookup_type(_ID_FQN).pointer())
        return _str_to_python(sym.value()(id_ptr))
    except gdb.error:
        return None


def _format_dtype(dtype_val: gdb.Value) -> str:
    s = str(dtype_val).strip()
    m = _DTYPE_VARIANT_RE.search(s)
    if m is None:
        return s
    variant = m.group(1)
    nm = _NULLABILITY_RE.search(s)
    n = "?" if nm is not None and nm.group(1) == "Nullable" else ""
    if variant == "Primitive":
        pm = _PTYPE_VARIANT_RE.search(s)
        return f"{pm.group(1).lower() if pm else '?'}{n}"
    tmpl = _DTYPE_TEMPLATES.get(variant)
    return tmpl.format(n=n) if tmpl is not None else f"{variant}{n}"


def _array_inner_at(addr: int) -> gdb.Value | None:
    try:
        return gdb.Value(addr).cast(gdb.lookup_type(_ARRAY_INNER_TY_NAME).pointer()).dereference()
    except gdb.error:
        return None


def _format_id(val: gdb.Value) -> str:
    addr = val.address
    if addr is not None:
        text = _id_to_string(int(addr))
        if text is not None:
            return f'Id("{text}")'
    spur = _spur_from_id(val)
    return f"Id({spur})" if spur is not None else "Id(?)"


def _format_dyn_plugin(val: gdb.Value) -> str:
    ty = _concrete_type(val, "") or "?"
    fp = _trait_fat_pointer(val)
    addr = f"{fp[0]:#x}" if fp else "?"
    return f"dyn ArrayPlugin<{ty}> @ {addr}"


def _format_arc_dyn_plugin(val: gdb.Value) -> str:
    ty = _concrete_type(val, ".ptr.pointer") or "?"
    try:
        fp = _trait_fat_pointer(val["ptr"]["pointer"])
    except gdb.error:
        fp = None
    if fp is None:
        return f"Arc<dyn ArrayPlugin<{ty}>>"
    arc_inner_base, vtable_ptr = fp
    usize = _usize_size()
    try:
        strong = _read_usize(arc_inner_base)
        # ArcInner.weak = Arc::weak_count() + 1
        weak = _read_usize(arc_inner_base + usize) - 1
        align = _read_usize(vtable_ptr + 2 * usize) or usize
    except gdb.error:
        return f"Arc<dyn ArrayPlugin<{ty}>>"
    data_offset = (2 * usize + align - 1) & ~(align - 1)
    return f"Arc<dyn ArrayPlugin<{ty}> @ {arc_inner_base + data_offset:#x}> [strong={strong}, weak={weak}]"


def _format_array_ref(val: gdb.Value) -> str:
    try:
        arc = val[val.type.fields()[0].name]
        fat_ptr = arc["ptr"]["pointer"]
    except (gdb.error, RuntimeError, IndexError):
        return "ArrayRef(?)"

    fp = _trait_fat_pointer(fat_ptr) or (int(fat_ptr), None)
    arc_inner_base = fp[0]
    inner = _array_inner_at(arc_inner_base + _ARC_INNER_DATA_OFFSET)
    if inner is None:
        return f"ArrayRef(?) @ {arc_inner_base:#x}"

    def _try(fn, default):
        try:
            return fn()
        except (gdb.error, RuntimeError):
            return default

    len_ = _try(lambda: int(inner["len"]), None)
    enc = inner["encoding_id"]
    encoding = (_id_to_string(int(enc.address)) if enc.address else None) or "?"
    dtype = _try(lambda: _format_dtype(inner["dtype"]), "?")

    head = f"{encoding}({dtype}, len={'?' if len_ is None else len_})"
    usize = _usize_size()
    try:
        strong = _read_usize(arc_inner_base)
        weak = _read_usize(arc_inner_base + usize) - 1
        return f"{head} @ {arc_inner_base:#x} [strong={strong}, weak={weak}]"
    except gdb.error:
        return f"{head} @ {arc_inner_base:#x}"


def _format_typed_array(val: gdb.Value) -> str:
    try:
        return _format_array_ref(val["inner"])
    except (gdb.error, RuntimeError):
        return "Array(?)"


class _Printer:
    def __init__(self, val: gdb.Value, fmt: Callable[[gdb.Value], str]):
        self._val, self._fmt = val, fmt

    def to_string(self) -> str:
        return self._fmt(self._val)


def _matches_dyn_plugin(name: str) -> bool:
    return (
        name in (_ARRAY_PLUGIN_FQN, _DYN_ARRAY_PLUGIN)
        or name.startswith(_DYN_ARRAY_PLUGIN + " +")
        or name.startswith(_DYN_ARRAY_PLUGIN + "+")
    )


def _lookup_printer(val: gdb.Value) -> _Printer | None:
    try:
        t = val.type.strip_typedefs()
    except gdb.error:
        return None
    name = (t.name or "").strip()

    if name == _ARRAY_REF_FQN:
        return _Printer(val, _format_array_ref)
    if name.startswith(_ARRAY_TYPED_PREFIX):
        return _Printer(val, _format_typed_array)
    if name == _ID_FQN:
        return _Printer(val, _format_id)
    if name.startswith("alloc::sync::Arc<") and _DYN_ARRAY_PLUGIN in name:
        return _Printer(val, _format_arc_dyn_plugin)
    if _matches_dyn_plugin(name):
        return _Printer(val, _format_dyn_plugin)
    if t.code in (gdb.TYPE_CODE_PTR, gdb.TYPE_CODE_REF):
        target = t.target()
        if target is not None and _matches_dyn_plugin((target.name or "").strip()):
            return _Printer(val, _format_dyn_plugin)
    return None


def _install_at_front(container) -> None:
    if _lookup_printer in container:
        container.remove(_lookup_printer)
    container.insert(0, _lookup_printer)


def register() -> None:
    # Install before rust-gdb's StdRcProvider, otherwise Arc<dyn T> from StdRc wins
    _install_at_front(gdb.pretty_printers)
    try:
        _install_at_front(gdb.current_progspace().pretty_printers)
    except (AttributeError, gdb.error):
        pass
    for of in gdb.objfiles():
        _install_at_front(of.pretty_printers)
    gdb.events.new_objfile.connect(lambda e: _install_at_front(e.new_objfile.pretty_printers))


register()
print("vortex_gdb: registered pretty-printers")
