import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Degradation:
    delay_ms: float
    status_code: int | None
    degradation_type: str


class DegradationPolicy:
    def __init__(self, seed: int = 42, max_delay_ms: int = 2000):
        if max_delay_ms < 0:
            raise ValueError("max_delay_ms deve ser não negativo")
        self.max_delay_ms = max_delay_ms
        self.random = random.Random(seed)

    def decide(self, level: int) -> Degradation:
        if type(level) is not int or not 0 <= level <= 100:
            raise ValueError("stress deve ser inteiro entre 0 e 100")
        delay_ms = self.max_delay_ms * max(0, level - 25) / 75
        probability_500 = (level - 50) / 100 if level >= 75 else 0
        probability_503 = (level - 90) / 20 if level > 90 else 0
        draw = self.random.random()
        if draw < probability_503:
            return Degradation(delay_ms, 503, "error_503")
        if draw < probability_503 + probability_500:
            return Degradation(delay_ms, 500, "error_500")
        return Degradation(delay_ms, None, "delay" if delay_ms > 0 else "none")
