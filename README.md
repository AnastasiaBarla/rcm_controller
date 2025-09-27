# RCM-Constrained MIS Demos (Franka Panda + MuJoCo)

This repository contains three small Python demos for my thesis on **Remote Center of Motion (RCM)** control for minimally invasive surgery (MIS) using a Franka Panda model in **MuJoCo**.  
The goal is to keep a fixed trocar point while the instrument tip tracks a desired trajectory inside the patient.

---

## Methods implemented

| Script | Control level | Brief description | Model XML |
|---|---|---|---|
| `src/paper1.py` | **Kinematic (velocity)** | Task-space resolved-rate control with an RCM constraint as in Aghakhani et al. (ICRA 2013). | `models/panda.xml` |
| `src/paper2.py` | **Dynamic (torque)** | Redundant **torque-controlled** formulation with an RCM constraint as in Sandoval Arévalo et al. (IROS 2017). | `models/panda1.xml` |
| `src/paper2_contact.py` | **Dynamic (torque) + contact** | Same as paper2, but adds **physical contact** between the instrument and the trocar. | `models/panda2.xml` |

Each script loads its model with a **relative path** (no hard-coded desktop paths), so it should work on any machine after cloning.

---

## Repository Structure
- `src/` – Python scripts
- `models/` – `panda.xml`, `panda1.xml`, `panda2.xml` and the `assets/` they need
- `models/LICENSE` – License for the Panda model
- `requirements.txt` – Python dependencies
- `.gitignore` – Files Git should ignore

## Scripts and Models
- `src/paper1.py` → uses `models/panda.xml` (Paper 1 baseline)
- `src/paper2.py` → uses `models/panda1.xml` (Paper 2 baseline)
- `src/paper2_contact.py` → uses `models/panda2.xml` (Paper 2 with extra/contact features)

> Note: Each script loads its model with a **relative path** so it runs on any computer.

## Prerequisites
- Python 3.10+
- MuJoCo installed on your machine (per MuJoCo’s install instructions)

## Setup
```bash
pip install -r requirements.txt
```

## How to run

From the repository root:

```bash
# Kinematic RCM (Aghakhani13)
python src/paper1.py

# Torque-level RCM (Sandoval17)
python src/paper2.py

# Torque-level RCM + trocar contact
python src/paper2_contact.py
```

Each demo opens the MuJoCo viewer and drives the instrument tip along a preset trajectory while enforcing the RCM at the trocar.

If the viewer window fails to open, install system OpenGL/GLFW libs (e.g. Ubuntu:``` sudo apt-get install -y libglfw3 libgl1```).

## Configuration notes

The desired tip trajectory and controller gains are defined near the top of each script.

XML variants differ only where needed (e.g., contact sites, tool attachments, actuator type and limits). Keep models/assets/ next to the XMLs.

Scripts load models with a relative path (portable):
```bash
from pathlib import Path
HERE = Path(__file__).resolve().parent
MODEL_XML = HERE.parent / "models" / "<panda.xml | panda1.xml | panda2.xml>"
```
## References

N. Aghakhani, M. Geravand, N. Shahriari, M. Vendittelli, and G. Oriolo,
“Task control with remote center of motion constraint for minimally invasive robotic surgery,”
Proc. 2013 IEEE Int. Conf. on Robotics and Automation (ICRA), Karlsruhe, Germany, May 2013, pp. 5807–5812. doi: 10.1109/ICRA.2013.6631364.

J. S. Sandoval Arévalo, G. Poisson, and P. Wieyers,
“A new kinematic formulation of the RCM constraint for redundant torque-controlled robots,”
Proc. 2017 IEEE/RSJ Int. Conf. on Intelligent Robots and Systems (IROS), Vancouver, Canada, Sept. 2017, pp. 1–6. doi: 10.1109/IROS.2017.8206265.

## License

Model assets: see ```models/LICENSE ```(original license from the Panda model source).
