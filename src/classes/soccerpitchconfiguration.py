from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class SoccerPitchConfiguration:
    width: int = 4600  # [cm]
    length: int = 3000  # [cm]

    @property
    def vertices(self) -> List[Tuple[int, int]]:
        return [
            (0, self.width),
            (0, 0),
            (self.length / 2 - 500, 0),
            (self.length / 2 + 500, 0),
            (self.length, 0),
            (self.length, self.width),
            (self.length / 2 - 500, self.width),
            (self.length / 2 + 500, self.width)
        ]

    edges: List[Tuple[int, int]] = field(default_factory=lambda: [
        (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 0), (0, 1)
    ])

    labels: List[str] = field(default_factory=lambda: [
        "01", "02", "03", "04", "05", "06", "07", "08"
    ])

    colors: List[str] = field(default_factory=lambda: [
        "#FF1493", "#FF1493", "#FF1493", "#FF1493",
        "#00BFFF", "#00BFFF", "#00BFFF", "#00BFFF"
    ])