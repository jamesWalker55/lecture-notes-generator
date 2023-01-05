import json

from jinja2 import Environment, FileSystemLoader, select_autoescape

from .paths import TEMPLATES_DIR
from .scenes import Scene, group

env = Environment(
    loader=FileSystemLoader(TEMPLATES_DIR),
    autoescape=select_autoescape(),
)

with open(R"D:\Programming\lecture-transcriber\c\text.json", "r", encoding="utf8") as f:
    scenes = json.load(f)

scenes = [Scene(*x) for x in scenes]
scenes = group(scenes)
template = env.get_template("main.html")
html = template.render(scenes=scenes)

with open("c/index.html", "w", encoding="utf8") as f:
    f.write(html)
