def pytest_configure(config) -> None:
    config.addinivalue_line("markers", "gpu: test requires a working CUDA GPU")
