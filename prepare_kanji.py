from dataclasses import dataclass
from svgelements import SVG, Group
import xml.etree.ElementTree as ET
import pyvips
import glob
import json

TOP_RADICALS = ("田", "大", "氵", "月", "艹", "冖", "八", "目",
                "十", "土", "亻", "亠", "人", "木", "日", )[::-1]
ALL_KANJI_PATHS = glob.glob("kanji/*.svg")


@dataclass
class Kanji:
    kanji_id: str
    kanji_elements: list

    @property
    def svg(self) -> SVG:
        with open(f"./kanji/{self.kanji_id}.svg") as file:
            temp_svg = SVG.parse(file)
        return temp_svg

    @property
    def bboxes(self) -> dict[str, tuple[int]]:
        bbox_data = {}
        for group in self.svg.elements():
            if isinstance(group, Group):
                data = group.values["attributes"]
                if r"{http://kanjivg.tagaini.net}element" in data:
                    if (radical := data[r"{http://kanjivg.tagaini.net}element"]) in TOP_RADICALS:
                        bbox_data[radical] = tuple(map(int, group.bbox()))
        return bbox_data

    def save_png(self, dpi: int = 70) -> None:
        svg_string = '<svg xmlns="http://www.w3.org/2000/svg" width="109" height="109" viewBox="0 0 109 109"><rect width="100%" height="100%" fill="white" /><g style="fill:none;stroke:#000000;stroke-width:3;stroke-linecap:round;stroke-linejoin:round;">'
        for element in self.kanji_elements:
            svg_string += ET.tostring(element).decode()
        svg_string += "</g></svg>"

        img = pyvips.Image.svgload_buffer(svg_string.encode(), dpi=dpi)
        img.write_to_file(f"data/kanji/{self.kanji_id}.png")


if __name__ == "__main__":
    data: dict[str, dict[str, tuple[float]]] = {}

    tree = ET.parse("data/kanjivg-20240807.xml").getroot()
    for raw_kanji in tree.findall("./kanji"):
        kanji = Kanji(raw_kanji.attrib["id"][10:], raw_kanji)

        kanji.save_png()
        if len(bboxes := kanji.bboxes) > 0:
            data[kanji.kanji_id] = bboxes

    with open("data/kanji_data.json", "w") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
