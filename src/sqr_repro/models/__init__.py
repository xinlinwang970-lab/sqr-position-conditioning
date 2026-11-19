__all__ = ["Separator"]


def __getattr__(name):
    if name == "Separator":
        from .separator import Separator

        return Separator
    raise AttributeError(name)

