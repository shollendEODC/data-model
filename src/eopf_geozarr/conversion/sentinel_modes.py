from __future__ import annotations
from enum import Enum

class S2Type(Enum):
    L1C = "L1C"
    L2A = "L2A"

    @classmethod
    def from_filename(cls, filename: str | None) -> S2Type | None:
        if not filename:
            return None
        for member in cls:
            if member.value in filename:
                return member
        return None

class S1Type(Enum):
    GRDH = "GRDH"

    @classmethod
    def from_filename(cls, filename: str | None) -> S1Type | None:
        if not filename:
            return None
        for member in cls:
            if member.value in filename:
                return member
        return None


class S1Mode(Enum):
    IW = "IW"
    EW = "EW"

    @classmethod
    def from_filename(cls, filename: str) -> S1Mode | None:
        for member in cls:
            if member.value in filename:
                return member
        return None
        