from typing import final

import vortex as vx

@final
class Registry:
    def register(self, cls: type[vx.Array]): ...
    def array_ctx(self, encodings: list[type[vx.Array] | str]) -> vx.ArrayContext: ...
