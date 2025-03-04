import vortex as vx

class VortexFile:
    @property
    def dtype(self) -> vx.DType: ...

def open(path: str) -> vx.VortexFile: ...
