from typing import NamedTuple

class BoundingBox(NamedTuple):
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    
    @property
    def left(self) -> float:
        return self.xmin

    @property
    def right(self) -> float:
        return self.xmax

    @property
    def bottom(self) -> float:
        return self.ymin

    @property
    def top(self) -> float:
        return self.ymax

    @property
    def xlim(self) -> tuple[float, float]:
        return (self.xmin, self.xmax)

    @property
    def ylim(self) -> tuple[float, float]:
        return (self.ymin, self.ymax)

    @property
    def xy_min(self) -> tuple[float, float]:
        return (self.xmin, self.ymin)

    @property
    def xy_max(self) -> tuple[float, float]:
        return (self.xmax, self.ymax)

    @property
    def width(self) -> float:
        return self.xmax - self.xmin

    @property
    def height(self) -> float:
        return self.ymax - self.ymin

    @property
    def size(self) -> tuple[float, float]:
        return (self.width, self.height)
