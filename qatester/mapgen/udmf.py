"""UDMF TEXTMAP emitter (PLAN: Why UDMF; Appendix A).

Plain data classes for the five UDMF block types and a deterministic text emitter.
Only fields that differ from their defaults are written, so output stays close to
the hand-written minimal TEXTMAP in Appendix A.
"""

from __future__ import annotations

from dataclasses import dataclass, field

NAMESPACE = "zdoom"

# Boolean linedef flags the compiler may set (PLAN Appendix A, "Flags worth knowing").
LINEDEF_FLAGS = frozenset(
    {
        "blocking", "blockmonsters", "twosided", "dontpegtop", "dontpegbottom", "secret",
        "blocksound", "playeruse", "playercross", "monstercross", "repeatspecial",
    }
)  # fmt: skip

# Doom editor numbers for MapSpec thing types (PLAN Appendix A, "Thing types used").
THING_EDNUMS: dict[str, int] = {
    "player_start": 1,
    "key_red": 13,
    "key_blue": 5,
    "key_yellow": 6,
    "health": 2012,
    "armour": 2018,
    "ammo": 2007,
    "monster_imp": 3001,
    "teleport_dest": 14,
    "marker": 9001,
}


@dataclass(frozen=True)
class Vertex:
    x: float
    y: float


@dataclass(frozen=True)
class Linedef:
    v1: int
    v2: int
    sidefront: int
    sideback: int | None = None
    flags: frozenset[str] = frozenset()
    special: int = 0
    args: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        unknown = self.flags - LINEDEF_FLAGS
        if unknown:
            raise ValueError(f"unknown linedef flags: {sorted(unknown)}")
        if len(self.args) > 5:
            raise ValueError("a linedef special takes at most 5 args")
        if self.v1 == self.v2:
            raise ValueError("zero-length linedef")


@dataclass(frozen=True)
class Sidedef:
    sector: int
    texturemiddle: str | None = None
    texturetop: str | None = None
    texturebottom: str | None = None


@dataclass(frozen=True)
class Sector:
    heightfloor: int
    heightceiling: int
    texturefloor: str = "FLOOR4_8"
    textureceiling: str = "CEIL3_5"
    lightlevel: int = 160
    id: int = 0  # sector tag
    special: int = 0


@dataclass(frozen=True)
class Thing:
    x: float
    y: float
    type: int
    angle: int = 0
    id: int = 0  # TID
    # Set on every thing, or it silently vanishes at the training skill (PLAN Appendix A).
    flags: frozenset[str] = frozenset(
        {"skill1", "skill2", "skill3", "skill4", "skill5", "single"}
    )


@dataclass
class TextMap:
    vertices: list[Vertex] = field(default_factory=list)
    linedefs: list[Linedef] = field(default_factory=list)
    sidedefs: list[Sidedef] = field(default_factory=list)
    sectors: list[Sector] = field(default_factory=list)
    things: list[Thing] = field(default_factory=list)

    def to_text(self) -> str:
        parts = [f'namespace = "{NAMESPACE}";', ""]
        parts += [_vertex(v) for v in self.vertices]
        parts.append("")
        parts += [_linedef(ld) for ld in self.linedefs]
        parts.append("")
        parts += [_sidedef(sd) for sd in self.sidedefs]
        parts.append("")
        parts += [_sector(s) for s in self.sectors]
        parts.append("")
        parts += [_thing(t) for t in self.things]
        return "\n".join(parts) + "\n"


def _str(value: str) -> str:
    if not value.isascii() or '"' in value or "\\" in value:
        raise ValueError(f"unsupported UDMF string {value!r}")
    return f'"{value}"'


def _float(value: float) -> str:
    return repr(float(value))


def _block(kind: str, fields: list[tuple[str, str]]) -> str:
    return f"{kind} {{ " + " ".join(f"{k} = {v};" for k, v in fields) + " }"


def _vertex(v: Vertex) -> str:
    return _block("vertex", [("x", _float(v.x)), ("y", _float(v.y))])


def _linedef(ld: Linedef) -> str:
    fields = [("v1", str(ld.v1)), ("v2", str(ld.v2)), ("sidefront", str(ld.sidefront))]
    if ld.sideback is not None:
        fields.append(("sideback", str(ld.sideback)))
    fields += [(flag, "true") for flag in sorted(ld.flags)]
    if ld.special:
        fields.append(("special", str(ld.special)))
        fields += [(f"arg{i}", str(a)) for i, a in enumerate(ld.args) if a]
    return _block("linedef", fields)


def _sidedef(sd: Sidedef) -> str:
    fields = [("sector", str(sd.sector))]
    for name in ("texturetop", "texturebottom", "texturemiddle"):
        value = getattr(sd, name)
        if value is not None:
            fields.append((name, _str(value)))
    return _block("sidedef", fields)


def _sector(s: Sector) -> str:
    fields = [
        ("heightfloor", str(s.heightfloor)),
        ("heightceiling", str(s.heightceiling)),
        ("texturefloor", _str(s.texturefloor)),
        ("textureceiling", _str(s.textureceiling)),
        ("lightlevel", str(s.lightlevel)),
    ]
    if s.id:
        fields.append(("id", str(s.id)))
    if s.special:
        fields.append(("special", str(s.special)))
    return _block("sector", fields)


def _thing(t: Thing) -> str:
    fields = [("x", _float(t.x)), ("y", _float(t.y)), ("type", str(t.type))]
    fields.append(("angle", str(t.angle)))
    if t.id:
        fields.append(("id", str(t.id)))
    fields += [(flag, "true") for flag in sorted(t.flags)]
    return _block("thing", fields)
