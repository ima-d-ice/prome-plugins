"""Prome Code Extraction Module."""

import math
from typing import List

class Calculator:
    """Performs numerical operations."""

    def __init__(self, precision: int = 2) -> None:
        self.precision = precision

    def add(self, a: float, b: float) -> float:
        """Add two numbers."""
        return round(a + b, self.precision)

def compute_hypotenuse(a: float, b: float) -> float:
    """Compute hypotenuse of right triangle."""
    return math.sqrt(a ** 2 + b ** 2)