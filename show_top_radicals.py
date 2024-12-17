from svgelements import SVG, Group, Path, Point
import glob


def get_elements_from_svg(svg) -> list[str]:
    l = []
    for element in list(svg.elements()):
        if isinstance(element, Group):
            path_count = len([p for p in element if isinstance(p, Path)])
                
            data = element.values["attributes"]
            if r"{http://kanjivg.tagaini.net}element" in data and path_count > 1:
                element = data[r"{http://kanjivg.tagaini.net}element"]
                l.append(element)
    return l


all_elements = {}
all_paths = glob.glob("kanji/*.svg")
for path in all_paths:
    with open(path, "r") as file:
        svg = SVG.parse(file)
        for e in get_elements_from_svg(svg):
            if e in all_elements:
                all_elements[e] += 1
            else:
                all_elements[e] = 1

total = len(all_paths)
for key, val in sorted(list(all_elements.items()), key=lambda k: k[1]):
    print(f"{key}: {val/total*100:.2f}%")
print(f"Total number of kanji: {total}")
