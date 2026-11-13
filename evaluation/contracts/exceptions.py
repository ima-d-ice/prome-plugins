class EvaluationError(Exception):
    """Base evaluation error."""
    pass


class MissingReferenceError(EvaluationError):
    """Raised when a required reference is missing."""
    pass


class InvalidSampleError(EvaluationError):
    """Raised when a sample is invalid."""
    pass


class MetricNotApplicableError(EvaluationError):
    """Raised when a metric is not applicable for a given sample."""
    pass


class DatasetValidationError(EvaluationError):
    """Raised when dataset validation fails."""
    pass


class JudgeConfigurationError(EvaluationError):
    """Raised when judge configuration is invalid."""
    pass


class ExperimentConfigurationError(EvaluationError):
    """Raised when experiment configuration is invalid."""
    pass


class WorktreeIsolationError(EvaluationError):
    """Raised when worktree isolation is violated."""
    pass


class RagasIntegrationError(EvaluationError):
    """Raised when Ragas integration fails."""
    pass