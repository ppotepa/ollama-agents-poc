"""Types and enums for the reflection system."""

from enum import Enum


class ConfidenceLevel(Enum):
    """Confidence levels for self-assessment."""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class ReflectionTrigger(Enum):
    """Triggers for reflection checkpoints."""
    STEP_COMPLETION = "step_completion"
    ERROR_ENCOUNTERED = "error_encountered"
    LOW_CONFIDENCE = "low_confidence"
    TIMEOUT = "timeout"
    USER_REQUEST = "user_request"
    AUTOMATIC = "automatic"
