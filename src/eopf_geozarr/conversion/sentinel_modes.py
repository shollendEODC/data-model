from __future__ import annotations

from enum import StrEnum


class S2Type(StrEnum):
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


class S1Type(StrEnum):
    GRDH = "GRDH"

    @classmethod
    def from_filename(cls, filename: str | None) -> S1Type | None:
        if not filename:
            return None
        for member in cls:
            if member.value in filename:
                return member
        return None


class S1Mode(StrEnum):
    IW = "IW"
    EW = "EW"
    SLC = "SLC"

    @classmethod
    def from_filename(cls, filename: str) -> S1Mode | None:
        for member in cls:
            if member.value in filename:
                return member
        return None
