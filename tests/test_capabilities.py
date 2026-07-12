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
