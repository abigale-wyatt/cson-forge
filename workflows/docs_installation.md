# Get Started with CSON Forge

## Prerequisites

- Python 3.8 or higher
- Conda or Mamba package manager
- Git

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/cson-forge.git
cd cson-forge
```

2. Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate cson-forge
```

## Verify Installation

To verify that everything is installed correctly, run:

```python
import cson_forge
import config
print(f"System detected: {config.system}")
```

## Register for data access

CSON Forge facilitates access to a collection of open datasets required to forcs regional oceanographic models. 
These data are documented in ROMS Tools [here](https://roms-tools.readthedocs.io/en/latest/datasets.html).

Access to most of the data is facilitated automatically. 
- [Sign up for access](https://help.marine.copernicus.eu/en/articles/4220332-how-to-sign-up-for-copernicus-marine-service) to the Copernicus Marine Service 
- [Sign up for access](https://www.tpxo.net/global) to TPXO data