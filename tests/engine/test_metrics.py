from __future__ import annotations

from deepinfer.engine.metrics import EngineMetrics


def test_metrics_render_prometheus_text() -> None:
    metrics = EngineMetrics()
    metrics.inc("requests_total", 2)
    metrics.inc("generated_tokens_total", 17)
    metrics.observe("decode_step_seconds", 0.25)
    metrics.set_gauge("running_requests", 1)

    rendered = metrics.render()
    assert "deepinfer_requests_total 2" in rendered
    assert "deepinfer_running_requests 1" in rendered
    snapshot = metrics.snapshot()
    assert snapshot["generated_tokens"] == 17
    assert snapshot["summaries"]["decode_step_seconds"]["avg"] == 0.25
