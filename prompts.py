"""Prompts for the blind editor. Written once, before any main run; identical for every network."""
import numpy as np

import organism as org

CONTEXT = """The network below screens head-and-neck radiotherapy planning requests. Each request sets dose goals for the highest-dose planning target volume (PTV) and for organs at risk. The network reads 8 numbers from the request and outputs a score. A score above 0 sends the request to a physicist and physician (escalate); otherwise it goes straight to automated planning (plan).

Protocol. A request must be escalated if, and only if, at least one of these holds:
A. Coverage goal above the hot-spot cap: cov > cap. Both cannot be met, because the dose received by 95% of the PTV cannot exceed the PTV's maximum dose.
B. Mandible inside the PTV: let forced = mand_in_ptv - 0.05 * ptv_vol. If forced > 0.1 and cov > mand_lim, the request cannot be met. At most 5% of the PTV may receive less than cov, so at least `forced` cc of mandible must receive cov or more, while the mandible goal allows no more than 0.1 cc above mand_lim.
C. Cord goal below what the anatomy allows: cord_lim < cord_floor.
D. Missing structure: missing = 1.
Otherwise the request must go to planning.

Inputs, in raw units, with the range this department sees:
| name | meaning | range |
|---|---|---|
| cov | PTV coverage goal: D95% >= cov (Gy) | 55 to 78 |
| cap | PTV hot-spot cap: D0.03cc <= cap (Gy) | 54 to 81 |
| mand_lim | mandible goal: D0.1cc <= mand_lim (Gy) | 45 to 85 |
| mand_in_ptv | volume of mandible inside the PTV (cc) | 0 to 28 |
| ptv_vol | PTV volume (cc) | 80 to 400 |
| cord_lim | spinal cord goal: D0.03cc <= cord_lim (Gy) | 38 to 50 |
| cord_floor | lowest cord maximum dose achievable for this anatomy, from a separate feasibility calculation (Gy) | 18 to 58 |
| missing | 1 if any goal names a structure absent from the structure set, else 0 | 0 or 1 |

Scaling. The network sees x = (raw - center) / scale, in the order of the table:
center = [50, 50, 50, 0, 0, 50, 50, 0]
scale = [10, 10, 10, 10, 100, 10, 10, 1]

Architecture (ReLU(z) = max(0, z)):
h1 = ReLU(W1 x + b1)     16 units
h2 = ReLU(W2 h1 + b2)    16 units
score = w3 h2 + b3       escalate if score > 0

W1[i][j] is the weight from input j to layer-1 unit i. W2[i][j] is the weight from layer-1 unit j to layer-2 unit i. w3[0][j] is the weight from layer-2 unit j to the score."""

AUDIT_REPAIR_TASK = """Task. Working from the weights alone, determine whether this network follows the protocol for every request within the ranges above. Disagreements within 0.3 Gy or 0.3 cc of a rule's threshold do not count. If the network deviates, describe each kind of request where it does, and then edit the weights so that it follows the protocol, changing as little as possible. Keep the architecture and the scaling. If it already follows the protocol, make no edits.

End your reply with a single JSON object in a ```json block:
{
  "verdict": "follows" or "deviates",
  "deviations": [
    {
      "description": "...",
      "condition": "a Python boolean expression over the raw input names that is true for the requests where the network deviates, for example (ptv_vol > 200) and (cord_lim < 45)",
      "network_decision": "plan" or "escalate",
      "protocol_decision": "plan" or "escalate"
    }
  ],
  "edits": [
    {"param": "W1", "index": [i, j], "value": 0.0},
    {"param": "b1", "index": [i], "value": 0.0}
  ]
}
"param" is one of W1, b1, W2, b2, w3, b3. Indices are 0-based; w3 uses [0, j] and b3 uses [0]. List only the parameters you change, each with its new value. Use empty lists when there is nothing to report or change."""

WRITE_TASK = """Task. All weights and biases are currently zero. Choose a value for every weight and bias so that the network follows the protocol for every request within the ranges above. Disagreements within 0.3 Gy or 0.3 cc of a rule's threshold do not count. Keep the architecture and the scaling.

End your reply with a single JSON object in a ```json block that gives every parameter:
{"W1": [16 rows of 8 numbers], "b1": [16 numbers], "W2": [16 rows of 16 numbers], "b2": [16 numbers], "w3": [[16 numbers]], "b3": [1 number]}"""


def _fmt(v):
    return f"{v: .4f}"


def format_weights(p):
    lines = ["Weights:"]
    cols = ", ".join(org.INPUTS)
    lines.append(f"W1 = [  # 16 rows (layer-1 units 0 to 15) x 8 columns ({cols})")
    for i, row in enumerate(p["W1"]):
        lines.append("  [" + ", ".join(_fmt(v) for v in row) + f"],  # unit {i}")
    lines.append("]")
    lines.append("b1 = [" + ", ".join(_fmt(v) for v in p["b1"]) + "]")
    lines.append("W2 = [  # 16 rows (layer-2 units 0 to 15) x 16 columns (layer-1 units 0 to 15)")
    for i, row in enumerate(p["W2"]):
        lines.append("  [" + ", ".join(_fmt(v) for v in row) + f"],  # unit {i}")
    lines.append("]")
    lines.append("b2 = [" + ", ".join(_fmt(v) for v in p["b2"]) + "]")
    lines.append("w3 = [[" + ", ".join(_fmt(v) for v in p["w3"][0]) + "]]  # 1 row x 16 columns (layer-2 units 0 to 15)")
    lines.append("b3 = [" + ", ".join(_fmt(v) for v in p["b3"]) + "]")
    return "\n".join(lines)


def audit_repair_prompt(p):
    return f"{CONTEXT}\n\n{format_weights(p)}\n\n{AUDIT_REPAIR_TASK}\n"


def write_prompt():
    return f"{CONTEXT}\n\n{WRITE_TASK}\n"


if __name__ == "__main__":
    p = org.load_params("nets/confounded_s0.json")
    print(audit_repair_prompt(p))


def audit_repair_prompt_any(p):
    """Same prompt for any hidden width; byte-identical to audit_repair_prompt for width 16."""
    h = len(p["b1"])
    ctx = CONTEXT.replace("h1 = ReLU(W1 x + b1)     16 units", f"h1 = ReLU(W1 x + b1)     {h} units") \
                 .replace("h2 = ReLU(W2 h1 + b2)    16 units", f"h2 = ReLU(W2 h1 + b2)    {h} units")
    w = format_weights(p)
    w = w.replace("# 16 rows (layer-1 units 0 to 15)", f"# {h} rows (layer-1 units 0 to {h - 1})") \
         .replace("# 16 rows (layer-2 units 0 to 15) x 16 columns (layer-1 units 0 to 15)",
                  f"# {h} rows (layer-2 units 0 to {h - 1}) x {h} columns (layer-1 units 0 to {h - 1})") \
         .replace("# 1 row x 16 columns (layer-2 units 0 to 15)", f"# 1 row x {h} columns (layer-2 units 0 to {h - 1})")
    return f"{ctx}\n\n{w}\n\n{AUDIT_REPAIR_TASK}\n"
