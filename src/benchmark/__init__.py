"""Benchmark package with lazy imports for lightweight statistical tooling."""

__all__ = [
    "VideoEvaluator", "EvaluationResult", "CorrelationAnalysis",
    "MetricCorrelation", "BenchmarkReport",
]


def __getattr__(name):
    if name in {"VideoEvaluator", "EvaluationResult"}:
        from .evaluator import EvaluationResult, VideoEvaluator
        return {"VideoEvaluator": VideoEvaluator, "EvaluationResult": EvaluationResult}[name]
    if name in {"CorrelationAnalysis", "MetricCorrelation"}:
        from .correlation import CorrelationAnalysis, MetricCorrelation
        return {"CorrelationAnalysis": CorrelationAnalysis, "MetricCorrelation": MetricCorrelation}[name]
    if name == "BenchmarkReport":
        from .report import BenchmarkReport
        return BenchmarkReport
    raise AttributeError(name)
