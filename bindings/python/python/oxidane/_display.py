"""SVG notebook display with no mandatory IPython dependency."""
import operator
import textwrap
import xml.etree.ElementTree as ET

from ._oxidane import PyMolecule, PyMolecule3D

_NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", _NS)


def _dimensions(width, height):
    values = tuple(operator.index(value) for value in (width, height))
    if any(value < 0 or value >= 2**32 for value in values):
        raise OverflowError("dimensions must fit an unsigned 32-bit integer")
    if 0 in values:
        raise ValueError("width and height must be positive")
    return values


def _element(parent, tag, text=None, **attributes):
    element = ET.SubElement(parent, f"{{{_NS}}}{tag}", {
        key.replace("_", "-"): str(value) for key, value in attributes.items()
    })
    element.text = text
    return element


def _root(width, height, title):
    root = ET.Element(f"{{{_NS}}}svg", {
        "width": str(width), "height": str(height),
        "viewBox": f"0 0 {width} {height}", "role": "img", "aria-label": title,
    })
    _element(root, "title", title)
    _element(root, "rect", width="100%", height="100%", fill="white")
    return root


def resize_svg(raw, width, height):
    width, height = _dimensions(width, height)
    root = ET.fromstring(raw)
    if "viewBox" not in root.attrib:
        root.set("viewBox", f"0 0 {root.get('width', '400')} {root.get('height', '400')}")
    root.set("width", str(width))
    root.set("height", str(height))
    # A white background keeps chemical labels legible in dark notebooks too.
    root.insert(0, ET.Element(f"{{{_NS}}}rect", {
        "width": "100%", "height": "100%", "fill": "white",
    }))
    return ET.tostring(root, encoding="unicode")


def plot_svg(svg):
    try:
        from IPython.display import SVG, display
    except ModuleNotFoundError as error:
        if error.name and error.name.split('.')[0] == 'IPython':
            raise ImportError(
                "plot() requires IPython. Install `oxidane[notebook]` with uv, "
                "or use to_svg() without notebook dependencies."
            ) from error
        raise
    display(SVG(data=svg))


def _molecule(value):
    if isinstance(value, PyMolecule):
        return value
    if isinstance(value, PyMolecule3D):
        return value.to_graph()
    if isinstance(value, str):
        return PyMolecule(value)
    raise TypeError("reaction participants must be molecules or SMILES strings")


class Reaction:
    """A concrete reaction scheme, without asserting chemical validity or balance.

    Participants may be molecules or SMILES strings. Repeated participants are
    retained, and empty sides represent sources or sinks.
    """

    def __init__(self, reactants, products, *, name="Reaction"):
        if isinstance(reactants, str) or isinstance(products, str):
            raise TypeError("reactants and products must be sequences, e.g. ['CCO']")
        if not isinstance(name, str):
            raise TypeError("name must be a string")
        self.reactants = tuple(_molecule(value) for value in reactants)
        self.products = tuple(_molecule(value) for value in products)
        self.name = name

    def to_svg(self, *, width=800, height=260):
        """Return an SVG scheme without displaying it or importing IPython."""
        width, height = _dimensions(width, height)
        root = _root(width, height, self.name)
        root.set("viewBox", "0 0 800 260")
        _element(root, "text", self.name, x=12, y=22, font_family="sans-serif", font_size=15)
        _element(root, "path", d="M 365 145 H 435 M 425 137 L 435 145 L 425 153",
                 fill="none", stroke="#243247", stroke_width=2)
        for molecules, left in [(self.reactants, 0), (self.products, 450)]:
            slot = 350 / max(1, len(molecules))
            if not molecules:
                _element(root, "text", "∅", x=left + 175, y=145,
                         text_anchor="middle", font_size=24)
            for index, molecule in enumerate(molecules):
                panel = ET.fromstring(molecule.to_svg())
                panel.set("x", str(left + index * slot + 10))
                panel.set("y", "35")
                panel.set("width", str(max(1, slot - 20)))
                panel.set("height", "215")
                root.append(panel)
                if index:
                    _element(root, "text", "+", x=left + index * slot, y=145,
                             text_anchor="middle", font_size=20)
        return ET.tostring(root, encoding="unicode")

    def plot(self, *, width=800, height=260):
        """Display once, even inside loops. Returns None."""
        plot_svg(self.to_svg(width=width, height=height))

    def _repr_svg_(self):
        return self.to_svg()


def rule_svg(rule, reactants, width, height):
    width, height = _dimensions(width, height)
    if reactants is not None:
        outcomes = rule.apply_to(reactants)
        root = _root(width, height * max(1, len(outcomes)), "Reaction SMARTS outcomes")
        if not outcomes:
            _element(root, "text", "No matching reaction outcomes", x=12, y=24,
                     font_family="sans-serif", font_size=14)
        for index, product in enumerate(outcomes):
            row = ET.fromstring(Reaction(reactants, [product], name=f"Outcome {index + 1}")
                                .to_svg(width=width, height=height))
            row.set("x", "0")
            row.set("y", str(index * height))
            root.append(row)
        return ET.tostring(root, encoding="unicode")

    left, right = rule.smarts.split(">>", 1)
    sections = [(label, textwrap.wrap(query, 88, break_on_hyphens=False) or [""])
                for label, query in [("Reactant SMARTS", left), ("Product SMARTS", right)]]
    root = _root(width, height, "Reaction SMARTS query")
    logical_height = 80 + sum(24 + 18 * len(lines) for _, lines in sections)
    root.set("viewBox", f"0 0 800 {logical_height}")
    y = 30
    for label, lines in sections:
        _element(root, "text", label, x=12, y=y, font_family="sans-serif", font_size=14)
        y += 22
        for line in lines:
            _element(root, "text", line, x=12, y=y, font_family="monospace", font_size=13)
            y += 18
        y += 12
    return ET.tostring(root, encoding="unicode")
