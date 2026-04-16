# Seismic Visualization

![Volve seismic section](volve10r12-full-twt-sub3d.gif)

Interactive visualization and animation of a 3D seismic volume with horizon overlay, using Equinor's [Volve dataset](https://data.equinor.com/dataset/Volve).

## Getting Started

Install dependencies:

```bash
pip install -r requirements.txt
```

Open the tutorial notebook:

```bash
jupyter notebook seis_viz_tutorial.ipynb
```

## Structure

- **`seis_viz.py`** — Core module: seismic loading (segyio), horizon mapping, plotting, animation export
- **`seis_viz_tutorial.ipynb`** — Interactive walkthrough with inline slider and animation export
- **`seis_viz_3d.ipynb`** — Optional 3D volume rendering with PyVista
- **`data/`** — Volve seismic cube (SEG-Y) and Top Hugin horizon

## 3D Visualization (optional)

The 3D notebook requires PyVista and a GPU-capable environment:

```bash
pip install pyvista
```

## Acknowledgements

The author thanks Equinor AS, the former Volve license partners ExxonMobil Exploration and Production Norway AS and Bayerngas (now Spirit Energy) for permission to use the Volve dataset, and to the many persons who have contributed to the work here. Please visit data.equinor.com for more information about the Volve dataset and [license terms of use](https://datavillagesa.blob.core.windows.net/disclaimers/HRS%20and%20Terms%20and%20conditions%20for%20license%20to%20data%20-%20Volve.pdf).
