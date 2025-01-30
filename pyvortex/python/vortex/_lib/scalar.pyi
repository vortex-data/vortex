class Buffer:
    def into_python(self, *, recursive=False) -> bytes:
        """Convert this Buffer to a Python bytes."""
        pass

class BufferString:
    def into_python(self, *, recursive=False) -> str:
        """Convert this BufferString to a Python str."""
        pass

class VortexList:
    def into_python(self, *, recursive=False) -> list:
        """Convert this VortexList to a Python list."""
        pass

class VortexStruct:
    def into_python(self, *, recursive=False) -> dict:
        """Convert this VortexStruct to a Python dict."""
        pass
