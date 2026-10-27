# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors
"""Build-only diagnostic copy of Arrow; never modifies the registry or production dependency."""
import os
from pathlib import Path
import shutil

cargo_home = Path(os.environ.get('CARGO_HOME', Path.home() / '.cargo'))
sources = list((cargo_home / 'registry/src').glob('*/arrow-select-59.3.2'))
assert len(sources) == 1, sources
target = Path('target/scan-diagnostic-arrow-select')
shutil.copytree(sources[0], target, dirs_exist_ok=True)
path = target / 'src/coalesce/byte_view.rs'
source = path.read_text()
needle = '''                (need_gc, ideal_buffer_size)'''
assert source.count(needle) == 1
source = source.replace(needle, '''                if std::env::var_os("VORTEX_SCAN_BYTE_VIEW_DIAGNOSTICS").is_some() {
                    let buffers: Vec<_> = s.data_buffers().iter()
                        .map(|buffer| (buffer.as_ptr() as usize, buffer.len(), buffer.capacity()))
                        .collect();
                    let first_len = s.views().first().map(|view| *view as u32).unwrap_or(0);
                    eprintln!("SCAN_BYTE_VIEW rows={} capacity={} used={} gc={} first_len={} buffers={:?}",
                        s.len(), actual_buffer_size, ideal_buffer_size, need_gc, first_len, buffers);
                    static TRACED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
                    if need_gc && !TRACED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                        eprintln!("SCAN_BYTE_VIEW_FIRST_GC {}", std::backtrace::Backtrace::force_capture());
                    }
                }
                (need_gc, ideal_buffer_size)''')
path.write_text(source)
print('Prepared diagnostic Arrow copy:', target)
