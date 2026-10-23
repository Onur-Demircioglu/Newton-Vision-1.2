"""C++ çekirdek ve C++ çıkarım API'si derleme testleri (tests/test_cpp_build.py).

tests/test_cpp_core.cpp ve examples/cpp_inference_demo.cpp dosyalarını g++ ile derleyip çalıştırır.
g++ bulunamazsa testler açıkça SKIP edilir (sessizce geçmez).
"""

from __future__ import annotations
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Windows: PATH'teki eski MinGW (gcc 6.3) yerine MSYS2 GCC tercih edilir
MSYS2_BIN = Path(r"C:\msys64\mingw64\bin")


def _find_gxx() -> str | None:
    msys_gxx = MSYS2_BIN / "g++.exe"
    if msys_gxx.exists():
        return str(msys_gxx)
    return shutil.which("g++")


class TestCppBuild(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.gxx = _find_gxx()
        if cls.gxx is None:
            raise unittest.SkipTest("g++ bulunamadı; C++ derleme testleri atlandı")
        cls.env = dict(os.environ)
        # Çalışma zamanı DLL'leri (libgomp vb.) derleyicinin klasöründedir
        cls.env["PATH"] = str(Path(cls.gxx).parent) + os.pathsep + cls.env.get("PATH", "")
        cls.tmp = tempfile.TemporaryDirectory()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _build_and_run(self, sources: list[str], include_dirs: list[str], exe_name: str) -> str:
        exe = Path(self.tmp.name) / exe_name
        cmd = [self.gxx, "-std=c++17", "-O2", "-fopenmp"]
        cmd += [f"-I{PROJECT_ROOT / d}" for d in include_dirs]
        cmd += [str(PROJECT_ROOT / s) for s in sources] + ["-o", str(exe)]
        build = subprocess.run(cmd, capture_output=True, text=True, env=self.env)
        self.assertEqual(build.returncode, 0, f"Derleme başarısız:\n{build.stderr}")
        run = subprocess.run([str(exe)], capture_output=True, text=True, encoding="utf-8",
                             errors="replace", env=self.env, cwd=self.tmp.name)
        self.assertEqual(run.returncode, 0, f"Çalıştırma başarısız:\n{run.stdout}\n{run.stderr}")
        return run.stdout

    def test_cpp_core_self_test(self):
        out = self._build_and_run(["tests/test_cpp_core.cpp", "cpp/layers.cpp"], ["cpp"], "test_cpp_core.exe")
        self.assertIn("ALL PASS", out)

    def test_cpp_inference_demo(self):
        out = self._build_and_run(["examples/cpp_inference_demo.cpp", "cpp/layers.cpp"], ["include", "cpp"], "cpp_demo.exe")
        self.assertIn("Predicted Top Class", out)


if __name__ == "__main__":
    unittest.main()
