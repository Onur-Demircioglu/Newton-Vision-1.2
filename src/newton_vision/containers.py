"""newton_vision.containers - Model Konteynırları (Sequential vb.)."""

from __future__ import annotations
from typing import Iterable
import newton_vision as nv
from newton_vision.module import Module
from newton_vision.tensor import Tensor


class Sequential(Module):
    """Katmanları sıralı olarak birbirine bağlayan konteynır modülü.

    Örnek:
        model = nv.Sequential(
            nv.Linear(784, 128),
            nv.ReLU(),
            nv.Linear(128, 10)
        )
        logits = model(x)
    """
    def __init__(self, *args: Module | Iterable[Module]):
        super().__init__()
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            modules_list = list(args[0])
        else:
            modules_list = list(args)

        for idx, module in enumerate(modules_list):
            if not isinstance(module, Module):
                raise TypeError(f"Sequential elemanı Module örneği olmalıdır, alınan: {type(module)}")
            self.add_module(str(idx), module)

    def append(self, module: Module) -> Sequential:
        if not isinstance(module, Module):
            raise TypeError(f"Eklenen eleman Module örneği olmalıdır, alınan: {type(module)}")
        self.add_module(str(len(self._modules)), module)
        return self

    def __getitem__(self, idx: int | slice) -> Module | Sequential:
        layers = list(self._modules.values())
        if isinstance(idx, slice):
            return Sequential(layers[idx])
        return layers[idx]

    def __len__(self) -> int:
        return len(self._modules)

    def __iter__(self):
        return iter(self._modules.values())

    def forward(self, x: Tensor) -> Tensor:
        out = x
        for module in self._modules.values():
            out = module(out)
        return out


class ModuleList(Module):
    """Alt modülleri bir Python listesi gibi saklayan ve yöneten konteynır.

    Örnek:
        self.linears = nv.ModuleList([nv.Linear(10, 10) for _ in range(5)])
        for layer in self.linears:
            x = layer(x)
    """
    def __init__(self, modules: Iterable[Module] | None = None):
        super().__init__()
        if modules is not None:
            self.extend(modules)

    def __getitem__(self, idx: int | slice) -> Module | ModuleList:
        layers = list(self._modules.values())
        if isinstance(idx, slice):
            return ModuleList(layers[idx])
        return layers[idx]

    def __setitem__(self, idx: int, module: Module):
        if not isinstance(module, Module):
            raise TypeError(f"Atanan eleman Module örneği olmalıdır, alınan: {type(module)}")
        layers = list(self._modules.values())
        layers[idx] = module
        self._modules.clear()
        for i, m in enumerate(layers):
            self.add_module(str(i), m)

    def __len__(self) -> int:
        return len(self._modules)

    def __iter__(self):
        return iter(self._modules.values())

    def append(self, module: Module) -> ModuleList:
        if not isinstance(module, Module):
            raise TypeError(f"Eklenen eleman Module örneği olmalıdır, alınan: {type(module)}")
        self.add_module(str(len(self._modules)), module)
        return self

    def extend(self, modules: Iterable[Module]) -> ModuleList:
        for module in modules:
            self.append(module)
        return self
