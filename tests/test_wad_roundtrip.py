"""M1: write a WAD, re-read its directory, check every lump's name, offset and size."""

import struct
from pathlib import Path

import pytest

from qatester.mapgen import udmf
from qatester.mapgen.wad import Lump, encode_wad, map_lumps, read_wad, read_wad_directory


def test_roundtrip_names_offsets_sizes(tmp_path: Path) -> None:
    lumps = [
        *map_lumps("MAP01", 'namespace = "zdoom";\n'),
        Lump("EXTRA", bytes(range(7))),
        *map_lumps("MAP02", 'namespace = "zdoom";\nvertex { x = 0.0; y = 0.0; }\n'),
    ]
    path = tmp_path / "t.wad"
    path.write_bytes(encode_wad(lumps))
    raw = path.read_bytes()

    magic, count, dir_offset = struct.unpack_from("<4sii", raw, 0)
    assert magic == b"PWAD"
    assert count == len(lumps)
    assert dir_offset == 12 + sum(len(lump.data) for lump in lumps)
    assert len(raw) == dir_offset + 16 * count

    expected_offset = 12
    for entry, lump in zip(read_wad_directory(raw), lumps, strict=True):
        assert entry.name == lump.name
        assert entry.size == len(lump.data)
        assert entry.offset == expected_offset
        assert raw[entry.offset : entry.offset + entry.size] == lump.data
        expected_offset += entry.size

    assert read_wad(path) == lumps


def test_map_lump_order() -> None:
    names = [lump.name for lump in map_lumps("MAP03", "x")]
    assert names == ["MAP03", "TEXTMAP", "ENDMAP"]
    assert map_lumps("MAP03", "x")[0].data == b""
    assert map_lumps("MAP03", "x")[2].data == b""


@pytest.mark.parametrize("bad", ["", "TOOLONGNAME", "map01", "MÄP"])
def test_rejects_bad_lump_names(bad: str) -> None:
    with pytest.raises(ValueError):
        encode_wad([Lump(bad, b"")])


def test_rejects_truncated_directory() -> None:
    raw = encode_wad(map_lumps("MAP01", "x"))
    with pytest.raises(ValueError):
        read_wad_directory(raw[:-1])


def test_textmap_matches_appendix_a_shape() -> None:
    tm = udmf.TextMap(
        vertices=[udmf.Vertex(-320, -320), udmf.Vertex(0, -320)],
        linedefs=[udmf.Linedef(0, 1, 0, flags=frozenset({"blocking"}))],
        sidedefs=[udmf.Sidedef(0, texturemiddle="BRICK7")],
        sectors=[udmf.Sector(0, 128)],
        things=[udmf.Thing(-160, -160, 1)],
    )
    text = tm.to_text()
    assert text.startswith('namespace = "zdoom";\n')
    assert "vertex { x = -320.0; y = -320.0; }" in text
    assert "linedef { v1 = 0; v2 = 1; sidefront = 0; blocking = true; }" in text
    assert 'sidedef { sector = 0; texturemiddle = "BRICK7"; }' in text
    assert "heightfloor = 0; heightceiling = 128;" in text
    for flag in ("skill1", "skill2", "skill3", "skill4", "skill5", "single"):
        assert f"{flag} = true;" in text


def test_linedef_rejects_unknown_flag() -> None:
    with pytest.raises(ValueError):
        udmf.Linedef(0, 1, 0, flags=frozenset({"blokcing"}))
