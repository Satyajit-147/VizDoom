"""PWAD binary writer and directory reader (PLAN: WAD container format).

Layout: 12-byte header (b"PWAD", int32 lump count, int32 directory offset), lump data,
then one 16-byte directory entry per lump (int32 offset, int32 size, 8-byte NUL-padded
name). All integers little-endian. No third-party dependencies.
"""

from __future__ import annotations

import struct
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

_HEADER = struct.Struct("<4sii")
_ENTRY = struct.Struct("<ii8s")
MAX_LUMP_NAME = 8


@dataclass(frozen=True)
class Lump:
    name: str
    data: bytes


@dataclass(frozen=True)
class LumpEntry:
    name: str
    offset: int
    size: int


def _encode_name(name: str) -> bytes:
    if not name or len(name) > MAX_LUMP_NAME or not name.isascii() or name != name.upper():
        raise ValueError(f"invalid lump name {name!r}: need 1-8 uppercase ASCII characters")
    return name.encode("ascii").ljust(MAX_LUMP_NAME, b"\0")


def map_lumps(map_name: str, textmap: str) -> list[Lump]:
    """The lump run for one UDMF map: marker, TEXTMAP, ENDMAP (PLAN Appendix A)."""
    return [
        Lump(map_name, b""),
        Lump("TEXTMAP", textmap.encode("ascii")),
        Lump("ENDMAP", b""),
    ]


def encode_wad(lumps: Sequence[Lump]) -> bytes:
    names = [_encode_name(lump.name) for lump in lumps]
    data = bytearray()
    entries = []
    offset = _HEADER.size
    for name, lump in zip(names, lumps):
        entries.append(_ENTRY.pack(offset, len(lump.data), name))
        data += lump.data
        offset += len(lump.data)
    header = _HEADER.pack(b"PWAD", len(lumps), offset)
    return header + bytes(data) + b"".join(entries)


def write_wad(path: Path, lumps: Sequence[Lump]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encode_wad(lumps))


def read_wad_directory(raw: bytes) -> list[LumpEntry]:
    magic, count, dir_offset = _HEADER.unpack_from(raw, 0)
    if magic not in (b"PWAD", b"IWAD"):
        raise ValueError(f"not a WAD: magic {magic!r}")
    if dir_offset + count * _ENTRY.size > len(raw):
        raise ValueError("directory runs past end of file")
    entries = []
    for i in range(count):
        offset, size, name = _ENTRY.unpack_from(raw, dir_offset + i * _ENTRY.size)
        if offset < 0 or size < 0 or offset + size > len(raw):
            raise ValueError(f"lump {i} out of bounds: offset={offset} size={size}")
        entries.append(LumpEntry(name.rstrip(b"\0").decode("ascii"), offset, size))
    return entries


def read_wad(path: Path) -> list[Lump]:
    raw = path.read_bytes()
    return [
        Lump(e.name, raw[e.offset : e.offset + e.size]) for e in read_wad_directory(raw)
    ]
