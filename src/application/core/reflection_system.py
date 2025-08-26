"""Reflection System - Modular facade for self-evaluation and automatic swap triggers."""

# Backwards compatibility facade that delegates to specialized reflection components
from .reflection import (
    ReflectionSystem,
    ReflectionCheckpoint,
    ReflectionResult,
    PerformanceMetrics,
    ConfidenceLevel,
    ReflectionTrigger,
    ReflectionAnalytics
)

# Export all classes for backwards compatibility
__all__ = [
    "ReflectionSystem", 
    "ReflectionCheckpoint", 
    "ReflectionResult",
    "PerformanceMetrics", 
    "ConfidenceLevel", 
    "ReflectionTrigger",
    "ReflectionAnalytics"
]
