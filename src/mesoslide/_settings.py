# to implement later for managing global setting
# see scanpy: https://github.com/scverse/scanpy/blob/main/src/scanpy/_settings.py
class MesoslideConfig:
    def __init__(
        self,
        *,
        some_arg = None
    ):
        self.some_arg = some_arg

settings = MesoslideConfig()