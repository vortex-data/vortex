# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors
import os, random, sys, time
f = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
size = os.path.getsize(f)
random.seed(7)
offs = [random.randrange(0, size - 4096) & ~16383 for _ in range(n)]
fd = os.open(f, os.O_RDONLY)
t = time.perf_counter()
for o in offs:
    os.pread(fd, 4096, o)
dt = time.perf_counter() - t
os.close(fd)
print(f"{n} random 4KiB preads: {dt*1e6/n:.1f} us/read")
