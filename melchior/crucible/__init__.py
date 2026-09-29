"""Crucible — Execution-Grounded Self-Play & RPM Dataset Generator."""

__all__ = ["CrucibleRunner", "CanaryRunner"]


def __getattr__(name):
    # Decision formatting is shared with inference/training environments that
    # intentionally do not install the sklearn execution sandbox.
    if name == 'CrucibleRunner':
        from melchior.crucible.runner import CrucibleRunner
        return CrucibleRunner
    if name == 'CanaryRunner':
        from melchior.crucible.canary import CanaryRunner
        return CanaryRunner
    raise AttributeError(name)
