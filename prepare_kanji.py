from svgelements import SVG, Group
import xml.etree.ElementTree as ET
import pyvips
import glob
import json


TOP_RADICALS = ("土", "亠", "人", "八", "十", "木", "日", "一", "丿", "口")
ALL_KANJI_PATHS = glob.glob("kanji/*.svg")


def get_rects_from_svg(svg, radicals: set[str]) -> dict[str, tuple[float]]:
    l = {}
    for group in svg.elements():
        if isinstance(group, Group):
            data = group.values["attributes"]
            if r"{http://kanjivg.tagaini.net}element" in data:
                radical: str = data[r"{http://kanjivg.tagaini.net}element"]
                if radical in radicals:
                    l[radical] = tuple(map(int, group.bbox()))
    return l


def get_primitive_labels() -> list[list[int]]:
    Y: list[list[int]] = []
    with open("data/kanji_data.json", "r") as file:
        data: dict = json.loads(file.read())

        for kanji_id, kanji_radicals in sorted(data.items(), key=lambda k: k[0]):
            kanji_entry: list[int] = []
            for radical in TOP_RADICALS:
                kanji_entry.append(1 if radical in kanji_radicals else 0)
            Y.append(kanji_entry)
    return Y


if __name__ == "__main__":
    kanji_svgs: dict[str, str] = {}
    data: dict[str, dict[str, tuple[float]]] = {}

    # Load pure kanji SVGs
    tree = ET.parse("data/kanjivg-20240807.xml").getroot()
    for kanji in tree.findall("./kanji"):
        svg = '<svg xmlns="http://www.w3.org/2000/svg" width="109" height="109" viewBox="0 0 109 109"><rect width="100%" height="100%" fill="white" /><g style="fill:none;stroke:#000000;stroke-width:3;stroke-linecap:round;stroke-linejoin:round;">'
        for g in kanji:
            svg += ET.tostring(g).decode()
        svg += "</g></svg>"
        kanji_svgs[kanji.attrib["id"][10:]] = svg

    # Get kanji radicals data
    for path in ALL_KANJI_PATHS:
        with open(path, "r") as file:
            svg = SVG.parse(file)
            if bbox := get_rects_from_svg(svg, TOP_RADICALS):
                kanji_id = path[6:-4]

                data[kanji_id] = bbox

                # Convert svg to .png file (without stroke numbers)
                img = pyvips.Image.svgload_buffer(
                    kanji_svgs[kanji_id].encode(), dpi=70)
                img.write_to_file(f"data/kanji/{kanji_id}.png")

    with open("data/kanji_data.json", "w") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
