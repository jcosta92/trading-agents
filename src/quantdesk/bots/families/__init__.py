"""Strategy families: signal and weight functions, registered by name in quantdesk.bots.generic.
Every module in this package is imported (registration), so a new family file needs no edit here."""
import importlib as _il
import pkgutil as _pk

for _m in sorted(m.name for m in _pk.iter_modules(__path__)):
    _il.import_module(f"{__name__}.{_m}")
