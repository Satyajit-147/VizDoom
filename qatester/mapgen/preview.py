"""Top-down SVG render of a compiled map (PLAN: Derived outputs).

Regions are painted in priority order, which reproduces the compiler's ownership rule:
rooms, then void strips, then door/lift/pad sectors. Linedefs, things and labels are
drawn on top. Injected defects are red: a cross at the recorded position and a dashed
outline around the region they occupy or cut off.
"""

from __future__ import annotations

from collections.abc import Sequence
from html import escape

from qatester.mapgen.geometry import (
    DOOR_LOCKED_RAISE,
    DOOR_OPEN,
    DOOR_RAISE,
    EXIT_NORMAL,
    PLAT_DOWN_WAIT_UP_STAY,
    TELEPORT,
    CompiledMap,
)

SIZE = 900
MARGIN = 40
FILL = {"room": "#e8e4d8", "door": "#d9822b", "lift": "#8e6bbf", "pad": "#2fa4b8",
        "pocket": "#f4b6b6"}
VOID = "#2b2b2b"
SPECIAL_COLOUR = {
    DOOR_RAISE: "#d9822b",
    DOOR_LOCKED_RAISE: "#c0392b",
    DOOR_OPEN: "#2e86de",
    PLAT_DOWN_WAIT_UP_STAY: "#8e6bbf",
    TELEPORT: "#2fa4b8",
    EXIT_NORMAL: "#27ae60",
}
THING_COLOUR = {
    "player_start": "#27ae60", "key_red": "#c0392b", "key_blue": "#2e86de",
    "key_yellow": "#f1c40f", "health": "#e84393", "armour": "#16a085", "ammo": "#7f8c8d",
    "monster_imp": "#6d4c41", "marker": "#95a5a6",
}  # fmt: skip


Marker = tuple[float, float, str, str]  # x, y, label, colour


def render_svg(cm: CompiledMap, markers: Sequence[Marker] = ()) -> str:
    """`markers` are drawn on top, e.g. a bug report's findings on the intended map."""
    spec = cm.spec
    x0 = min(r.x0 for r in spec.rooms)
    y0 = min(r.y0 for r in spec.rooms)
    x1 = max(r.x1 for r in spec.rooms)
    y1 = max(r.y1 for r in spec.rooms)
    scale = (SIZE - 2 * MARGIN) / max(x1 - x0, y1 - y0)
    width = (x1 - x0) * scale + 2 * MARGIN
    height = (y1 - y0) * scale + 2 * MARGIN + 30

    def px(x: float) -> float:
        return round(MARGIN + (x - x0) * scale, 1)

    def py(y: float) -> float:  # map y grows north; SVG y grows down
        return round(MARGIN + (y1 - y) * scale, 1)

    out = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" '
            f'height="{height:.0f}" viewBox="0 0 {width:.0f} {height:.0f}" '
            'font-family="monospace" font-size="11">'
        ),
        f'<rect width="100%" height="100%" fill="{VOID}"/>',
        (
            f'<text x="{MARGIN}" y="24" fill="#fff" font-size="15">{escape(spec.map_lump)} '
            f"{escape(spec.name)} — {len(spec.rooms)} rooms, {len(cm.lines)} linedefs, "
            f"{len(cm.sectors)} sectors</text>"
        ),
    ]
    for r in sorted(cm.regions, key=lambda rg: rg.priority):
        kind = cm.sectors[r.sector].kind if r.sector is not None else None
        fill = FILL[kind] if kind is not None else VOID
        out.append(
            f'<rect x="{px(r.rect.x0)}" y="{py(r.rect.y1)}" '
            f'width="{round((r.rect.x1 - r.rect.x0) * scale, 1)}" '
            f'height="{round((r.rect.y1 - r.rect.y0) * scale, 1)}" fill="{fill}"/>'
        )
    for line in cm.lines:
        (ax, ay), (bx, by) = line.v1, line.v2
        if line.special is not None:
            colour, w = SPECIAL_COLOUR.get(line.special.special, "#e67e22"), 3
        elif line.back is None:
            colour, w = "#111", 2
        else:
            colour, w = "#9a9a9a", 1
        out.append(
            f'<line x1="{px(ax)}" y1="{py(ay)}" x2="{px(bx)}" y2="{py(by)}" '
            f'stroke="{colour}" stroke-width="{w}"/>'
        )
    for room in spec.rooms:
        cx, cy = px((room.x0 + room.x1) / 2), py((room.y0 + room.y1) / 2)
        out.append(
            f'<text x="{cx}" y="{cy}" text-anchor="middle" fill="#555">{escape(room.id)} '
            f"f{room.floor}/c{room.ceiling}</text>"
        )
    for t in spec.things:
        if t.type in ("exit_switch", "switch"):
            continue
        x, y = px(t.pos[0]), py(t.pos[1])
        colour = THING_COLOUR.get(t.type, "#000")
        out.append(f'<circle cx="{x}" cy="{y}" r="6" fill="{colour}" stroke="#000"/>')
        out.append(f'<text x="{x + 9}" y="{y + 4}" fill="#000">{escape(t.id)}</text>')
    for m in cm.marks:
        x = (m.at if m.axis == "x" else (m.lo + m.hi) / 2)
        y = (m.at if m.axis == "y" else (m.lo + m.hi) / 2)
        out.append(
            f'<text x="{px(x) + 6}" y="{py(y) - 6}" fill="#000" font-weight="bold">'
            f"{escape(m.source)}</text>"
        )
    for rec in cm.defects:
        for area in rec.region:
            out.append(
                f'<rect x="{px(area.x0)}" y="{py(area.y1)}" '
                f'width="{round((area.x1 - area.x0) * scale, 1)}" '
                f'height="{round((area.y1 - area.y0) * scale, 1)}" fill="none" stroke="#e00" '
                'stroke-width="2" stroke-dasharray="6 4"/>'
            )
        x, y = px(rec.position[0]), py(rec.position[1])
        out.append(
            f'<path d="M{x - 7} {y - 7} L{x + 7} {y + 7} M{x - 7} {y + 7} L{x + 7} {y - 7}" '
            'stroke="#e00" stroke-width="3"/>'
        )
        out.append(
            f'<text x="{x + 9}" y="{y - 9}" fill="#e00" font-weight="bold">'
            f"{escape(rec.id)} ({escape(rec.expected_oracle)})</text>"
        )
    for mx, my, label, colour in markers:
        x, y = px(mx), py(my)
        out.append(f'<circle cx="{x}" cy="{y}" r="9" fill="{colour}" fill-opacity="0.85" '
                   'stroke="#fff" stroke-width="2"/>')
        out.append(f'<text x="{x + 12}" y="{y + 4}" fill="#fff" font-weight="bold" '
                   f'stroke="#000" stroke-width="3" paint-order="stroke">{escape(label)}</text>')
    legend = [
        ("door", FILL["door"]), ("locked", SPECIAL_COLOUR[DOOR_LOCKED_RAISE]),
        ("lift", FILL["lift"]), ("teleport", FILL["pad"]),
        ("switch", SPECIAL_COLOUR[DOOR_OPEN]), ("exit", SPECIAL_COLOUR[EXIT_NORMAL]),
    ]  # fmt: skip
    lx = MARGIN
    for label, colour in legend:
        out.append(f'<rect x="{lx}" y="{height - 22}" width="12" height="12" fill="{colour}"/>')
        out.append(f'<text x="{lx + 16}" y="{height - 12}" fill="#fff">{label}</text>')
        lx += 100
    out.append("</svg>")
    return "\n".join(out) + "\n"
