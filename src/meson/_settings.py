import sys

class MesonConfig:
    def __init__(
        self,
        *,
        some_arg = None
    ):
        self.some_arg = some_arg

settings = MesonConfig()