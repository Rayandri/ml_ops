__all__ = ["transpile"]


def transpile(*args, **kwargs):
    from .transpiler import run

    return run(*args, **kwargs)
