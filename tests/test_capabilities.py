from vocalforge.capabilities import evaluate_capabilities


def test_cpu_fallback_when_cuda_libs_missing(monkeypatch):
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    report = evaluate_capabilities(
        run=lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("nvidia-smi")),
        find_library=lambda _name: None,
        cuda_device_count=lambda: 0,
        prefer_cuda=True,
    )
    assert report.selected_device == "cpu"
    assert report.fallback_reason


def test_cuda_selected_when_all_probes_pass(monkeypatch):
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    report = evaluate_capabilities(
        run=lambda *a, **k: None,
        find_library=lambda name: f"lib{name}.so.12",
        cuda_device_count=lambda: 1,
        prefer_cuda=True,
    )
    assert report.selected_device == "cuda"
    assert report.fallback_reason is None
    assert report.cuda_runtime_libs_ok


def test_cuda_libs_discovered_via_ld_library_path(tmp_path, monkeypatch):
    monkeypatch.setenv("LD_LIBRARY_PATH", str(tmp_path))
    monkeypatch.delenv("CUDA_PATH", raising=False)
    monkeypatch.setenv("PATH", "")
    (tmp_path / "libcublas.so.12").write_bytes(b"")
    (tmp_path / "libcudart.so.12").write_bytes(b"")
    report = evaluate_capabilities(
        run=lambda *a, **k: None,
        find_library=lambda _name: None,
        cuda_device_count=lambda: 1,
        prefer_cuda=True,
    )
    assert report.cuda_runtime_libs_ok
    assert report.selected_device == "cuda"


def test_cuda_libs_discovered_via_cuda_path_windows_dlls(tmp_path, monkeypatch):
    cuda_root = tmp_path / "CUDA" / "v12.8"
    cuda_bin = cuda_root / "bin"
    cuda_bin.mkdir(parents=True)
    (cuda_bin / "cublas64_12.dll").write_bytes(b"")
    (cuda_bin / "cudart64_12.dll").write_bytes(b"")
    monkeypatch.setenv("CUDA_PATH", str(cuda_root))
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    monkeypatch.setenv("PATH", "")
    report = evaluate_capabilities(
        run=lambda *a, **k: None,
        find_library=lambda _name: None,
        cuda_device_count=lambda: 1,
        prefer_cuda=True,
    )
    assert report.cuda_runtime_libs_ok
    assert report.selected_device == "cuda"
