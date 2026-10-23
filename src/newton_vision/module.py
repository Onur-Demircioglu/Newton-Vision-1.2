"""newton_vision.module - Evrensel Model ve Katman Taban Sınıfı (Module & Parameter).

PyTorch `nn.Module` standardında parametre kaydı, alt modül yönetimi,
özyineli (recursive) yaşam döngüsü ve cihaz transferi sağlar.
"""

from __future__ import annotations
from typing import Iterator, OrderedDict as OrderedDictType
from collections import OrderedDict
import numpy as np

from newton_vision.tensor import Tensor


class Parameter(Tensor):
    """Model parametresi olarak kaydedilen eğitilebilir Tensör."""
    def __init__(self, data, requires_grad: bool = True, device: str = "cpu"):
        super().__init__(data, requires_grad=requires_grad, device=device)


class Module:
    """Tüm sinir ağı katmanları ve modelleri için temel sınıf."""
    def __init__(self):
        # Attribute atamaları __setattr__ ile yönetileceğinden __dict__ üzerinden ilklendirilir
        object.__setattr__(self, "_parameters", OrderedDict())
        object.__setattr__(self, "_modules", OrderedDict())
        object.__setattr__(self, "_buffers", OrderedDict())
        object.__setattr__(self, "training", True)

    def __setattr__(self, name: str, value):
        # İki alt çizgi ile başlayan veya özel nitelikler doğrudan atanır
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return

        params = self.__dict__.get("_parameters")
        modules = self.__dict__.get("_modules")
        buffers = self.__dict__.get("_buffers")

        if buffers is not None and name in buffers:
            buffers[name] = value
        elif isinstance(value, Parameter):
            if params is not None:
                params[name] = value
            if modules is not None and name in modules:
                del modules[name]
        elif isinstance(value, Module):
            if modules is not None:
                modules[name] = value
            if params is not None and name in params:
                del params[name]
        elif isinstance(value, (list, tuple)):
            if modules is not None:
                for idx, item in enumerate(value):
                    if isinstance(item, Module):
                        modules[f"{name}_{idx}"] = item
            object.__setattr__(self, name, value)
            return
        else:
            if params is not None and name in params:
                del params[name]
            if modules is not None and name in modules:
                del modules[name]

        object.__setattr__(self, name, value)

    def __getattr__(self, name: str):
        if "_parameters" in self.__dict__:
            _parameters = self.__dict__["_parameters"]
            if name in _parameters:
                return _parameters[name]
        if "_buffers" in self.__dict__:
            _buffers = self.__dict__["_buffers"]
            if name in _buffers:
                return _buffers[name]
        if "_modules" in self.__dict__:
            _modules = self.__dict__["_modules"]
            if name in _modules:
                return _modules[name]
        raise AttributeError(f"'{type(self).__name__}' nesnesinin '{name}' niteliği bulunamadı")

    def add_module(self, name: str, module: Module | None):
        """Alt modülü modele kaydeder."""
        if not isinstance(module, Module) and module is not None:
            raise TypeError(f"Alt modül bir Module örneği olmalıdır, alınan: {type(module)}")
        if hasattr(self, name) and name not in self._modules:
            existing = self.__dict__.get(name)
            if existing is not None and not isinstance(existing, Module):
                raise KeyError(f"'{name}' niteliği zaten mevcut")
        if module is None:
            if name in self._modules:
                del self._modules[name]
        else:
            self._modules[name] = module
        object.__setattr__(self, name, module)

    def register_parameter(self, name: str, param: Parameter | None):
        """Eğitilebilir parametreyi modele kaydeder."""
        if not isinstance(param, Parameter) and param is not None:
            raise TypeError(f"Parametre bir Parameter örneği olmalıdır, alınan: {type(param)}")
        if hasattr(self, name) and name not in self._parameters:
            existing = self.__dict__.get(name)
            if existing is not None and not isinstance(existing, (Parameter, Tensor)):
                raise KeyError(f"'{name}' niteliği zaten mevcut")
        if param is None:
            if name in self._parameters:
                del self._parameters[name]
        else:
            self._parameters[name] = param
        object.__setattr__(self, name, param)

    def register_buffer(self, name: str, tensor: np.ndarray | Tensor):
        """Model durumunda saklanan ancak gradyan gerektirmeyen tampon kaydeder (ör: running_mean)."""
        self._buffers[name] = tensor
        object.__setattr__(self, name, tensor)

    def parameters(self, recurse: bool = True) -> list[Tensor]:
        """Modeldeki tüm eğitilebilir parametreleri döndürür."""
        params: list[Tensor] = list(self._parameters.values())
        if recurse:
            for module in self._modules.values():
                params.extend(module.parameters(recurse=True))
        return params

    def modules(self) -> Iterator[Module]:
        """Tüm alt modülleri özyineli olarak gezer."""
        yield self
        for module in self._modules.values():
            yield from module.modules()

    def train(self, mode: bool = True):
        """Modeli eğitim moduna alır (BatchNorm vb. istatistik günceller)."""
        self.training = mode
        for module in self._modules.values():
            module.train(mode)
        return self

    def eval(self):
        """Modeli çıkarım (evaluation) moduna alır."""
        return self.train(False)

    def zero_grad(self):
        """Modelin tüm parametrelerindeki gradyanları sıfırlar."""
        for p in self.parameters():
            p.zero_grad()

    def to(self, device: str):
        """Tüm parametreleri belirtilen cihaza taşır ('cpu' veya 'cuda')."""
        dev = device.lower()
        for name, param in self._parameters.items():
            self._parameters[name] = Parameter(param.data, requires_grad=param.requires_grad, device=dev)
            object.__setattr__(self, name, self._parameters[name])

        for module in self._modules.values():
            module.to(dev)
        return self

    def cuda(self):
        return self.to("cuda")

    def cpu(self):
        return self.to("cpu")

    def state_dict(self, prefix: str = "") -> dict[str, np.ndarray]:
        """Modelin tüm parametre ve tamponlarını sözlük olarak çıkarır."""
        state = OrderedDict()
        for name, param in self._parameters.items():
            state[f"{prefix}{name}"] = param.data.copy()
        for name, buf in self._buffers.items():
            data = buf.data if isinstance(buf, Tensor) else np.asarray(buf, dtype=np.float32)
            state[f"{prefix}{name}"] = data.copy()
        for name, module in self._modules.items():
            state.update(module.state_dict(prefix=f"{prefix}{name}."))
        return state

    def load_state_dict(self, state_dict: dict[str, np.ndarray], strict: bool = False):
        """Ağırlık sözlüğünü modele aktarır (bkz. `newton_vision.weights.load_state_dict`)."""
        from newton_vision.weights import load_state_dict
        return load_state_dict(self, state_dict, strict=strict)

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Tüm Module alt sınıfları 'forward' metodunu tanımlamalıdır.")

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
