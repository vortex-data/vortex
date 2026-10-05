# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Materialize forced-kernel controls from exactly the PR source, without changing its loops."""

import pathlib
import subprocess

root = pathlib.Path("encodings/runend/benches/ci_support")
root.mkdir(parents=True, exist_ok=True)
source = subprocess.check_output(
    ["git", "show", "cafcf0b351bcf89cee860ed1f9c690d9978e15f9:encodings/runend/src/fill.rs"],
    text=True,
)
(root / "pr.rs").write_text(source)
condition = "if rows < max_avg_run * count {"
assert source.count(condition) == 1
(root / "forced_head.rs").write_text(source.replace(condition, "if true {"))
exact = source.replace(condition, "if false {")
(root / "forced_exact.rs").write_text(exact)
start = exact.index("unsafe fn fill_exact<T: Copy>")
end = exact.index("/// Runs longer than this", start)
direct = exact[:start] + """unsafe fn fill_exact<T: Copy>(dst: *mut T, value: T, n: usize) {
    // SAFETY: the caller guarantees space for n elements.
    unsafe {
        let mut ptr = dst;
        let end = dst.add(n);
        while ptr < end {
            ptr.write(value);
            ptr = ptr.add(1);
        }
    }
}

""" + exact[end:]
(root / "staged_direct.rs").write_text(direct)
