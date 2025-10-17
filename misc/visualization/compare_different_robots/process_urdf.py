import os
import itertools
from os import replace
import xml.etree.ElementTree as ET


def replace_macros(macro_dict, value):
    out = ""
    items = value.split(" ")

    for idx in range(len(items)):
        item = items[idx]
        if "{" in item:
            item = item.replace("{", "")
            item = item.replace("}", "")

            while "@@" in item:
                for key, value in macro_dict.items():
                    item = item.replace(key, value)

            items[idx] = str(eval(item))

        else:
            while "@@" in item:
                for key, value in macro_dict.items():
                    item = item.replace(key, value)

            items[idx] = item

    out = items[0]
    for idx in range(1, len(items)):
        out = out + " " + items[idx]

    return out


if __name__ == "__main__":
    ratios = [1.0 + (i + 1) * 0.075 for i in range(10)]

    leg_body_ratio = [1 / ratio for ratio in reversed(ratios)] + [1.0] + [ratio for ratio in ratios]
    arm_body_ratio = [1 / ratio for ratio in reversed(ratios)] + [1.0] + [ratio for ratio in ratios]
    base_length = 0.5

    replace_dicts = {
        f"{abr_idx:02d}_{lbr_idx:02d}": {
            "@@UpperArm": str(base_length * abr / 2.0),
            "@@LowerArm": str(base_length * abr / 2.0),
            "@@Thigh": str(base_length * lbr / 2.0),
            "@@Shin": str(base_length * lbr / 2.0),
        }
        for lbr, abr in itertools.product(leg_body_ratio, arm_body_ratio)
        for lbr_idx, lbr in enumerate(leg_body_ratio) for abr_idx, abr in enumerate(arm_body_ratio)
    }

    os.makedirs("robots/parsed_humanoid", exist_ok=True)

    for name, replace_dict in replace_dicts.items():
        print(replace_dict)
        tree = ET.parse('robots/humanoid.urdf')

        root = tree.getroot()

        for elem in tree.iter():
            for k, v in elem.attrib.items():
                elem.attrib[k] = replace_macros(replace_dict, v)

        tree.write(f"robots/parsed_humanoid/{name}.urdf")
