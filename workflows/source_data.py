from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional
from datetime import datetime, timedelta

import shutil
import tempfile
from urllib.request import urlopen

import copernicusmarine
import gdown
import roms_tools as rt

import config


# -----------------------------------------
# Dataset registry (name -> handler + metadata)
# -----------------------------------------


class DatasetHandler:
    """Container for a dataset handler and its required SourceData attributes."""

    def __init__(self, func: Callable["SourceData", Path], requires: List[str]):
        self.func = func
        self.requires = requires


DATASET_REGISTRY: Dict[str, DatasetHandler] = {}


def register_dataset(name: str, requires: Optional[List[str]] = None) -> Callable:
    """
    Decorator to register a dataset handler.

    Parameters
    ----------
    name : str
        Dataset name (e.g. "GLORYS_REGIONAL", "UNIFIED_BGC", "SRTM15").
        Stored in upper case.
    requires : list of str, optional
        Names of SourceData attributes that must be non-None for this
        dataset to be prepared (e.g. ["grid", "grid_name", "start_time", "end_time"]).

    Usage
    -----
        @register_dataset("GLORYS_REGIONAL", requires=["grid", "grid_name", "start_time", "end_time"])
        def _prepare_glorys_regional(self): ...
    """
    if requires is None:
        requires = []

    def decorator(func: Callable["SourceData", Path]) -> Callable:
        DATASET_REGISTRY[name.upper()] = DatasetHandler(func=func, requires=requires)
        return func

    return decorator


# -----------------------------------------
# Constants (SRTM15 versioning)
# -----------------------------------------

SRTM15_VERSION = "V2.7"
SRTM15_URL = f"https://topex.ucsd.edu/pub/srtm15_plus/SRTM15_{SRTM15_VERSION}.nc"


# -----------------------------------------
# constants: GLORYS
# -----------------------------------------

# GLORYS dataset key: set based on system type
# - "mac" (local-client) → "GLORYS_REGIONAL" (regional subset)
# - "hpc" systems (anvil, perlmutter, etc.) → "GLORYS" (full dataset)
if config.system == "mac":
    glorys_dataset_key: str = "GLORYS_GLOBAL"
else:
    glorys_dataset_key: str = "GLORYS_GLOBAL"

glorys_dataset_id: str = "cmems_mod_glo_phy_my_0.083deg_P1D-m"

# -----------------------------------------
# Logical source-name → dataset key mapping
# -----------------------------------------


SOURCE_ALIAS: Dict[str, str] = {
    # GLORYS
    "GLORYS": glorys_dataset_key,
    "GLORYS_GLOBAL": "GLORYS_GLOBAL",
    "GLORYS_REGIONAL": "GLORYS_REGIONAL",
    # UNIFIED biogeochemistry
    "UNIFIED": "UNIFIED_BGC",
    "UNIFIED_BGC": "UNIFIED_BGC",
    # SRTM15 bathymetry
    "SRTM15": f"SRTM15_{SRTM15_VERSION}".upper(),
    # Rivers, etc. (placeholder – add real dataset handlers as needed)
    "DAI": "DAI",  # expected to correspond to a DAI dataset if/when added
}


def map_source_to_dataset_key(name: str) -> str:
    """
    Map a logical source name (e.g. 'GLORYS', 'UNIFIED') to a dataset key
    used in DATASET_REGISTRY / SourceData.paths.

    If no alias is defined, the uppercased name is returned as-is.
    """
    return SOURCE_ALIAS.get(name.upper(), name.upper())


# -----------------------------------------
# SourceData
# -----------------------------------------


@dataclass
class SourceData:
    """
    Handles creation and caching of source data files
    (GLORYS_REGIONAL, UNIFIED_BGC, SRTM15, etc.) for ROMS preprocessing.

    Parameters
    ----------
    datasets : list of str
        Names of datasets to prepare, e.g. ["GLORYS_REGIONAL", "UNIFIED_BGC", "SRTM15"].
    clobber : bool, optional
        If True, re-download/rebuild datasets even if files exist.
    grid, grid_name, start_time, end_time : optional
        Only required for datasets whose handlers declare them via
        `requires=[...]` in the @register_dataset decorator.
        For example, GLORYS_REGIONAL needs all four.
    """

    datasets: List[str]
    clobber: bool = False

    # Optional attributes — only required if a dataset handler declares them
    grid: Optional[object] = None
    grid_name: Optional[str] = None
    start_time: Optional[object] = None
    end_time: Optional[object] = None

    def __post_init__(self):
        # Normalize dataset names
        self.datasets = [SOURCE_ALIAS[ds.upper()] for ds in self.datasets]
        
        # Validate requested datasets
        known = set(DATASET_REGISTRY.keys())
        unknown = set(self.datasets) - known
        if unknown:
            raise ValueError(
                f"Unknown dataset(s) requested: {', '.join(sorted(unknown))}. "
                f"Known datasets: {', '.join(sorted(known))}"
            )

        # Per-dataset paths (generic) + convenience attrs
        self.paths: Dict[str, Path] = {}
        self.srtm15_path: Optional[Path] = None

    # -----------------------------------------
    # Public API
    # -----------------------------------------

    def prepare_all(self):
        """Prepare all requested source datasets and populate `self.paths`."""
        for name in self.datasets:
            handler = DATASET_REGISTRY[name]

            # Make sure required attributes are provided
            missing = [attr for attr in handler.requires if getattr(self, attr) is None]
            if missing:
                raise ValueError(
                    f"Dataset '{name}' requires attributes {missing}, "
                    "but they were not provided to SourceData()."
                )

            path = handler.func(self)  # call handler with this instance
            self.paths[name] = path  # store generically

        return self

    # -----------------------------------------
    # Helpers for model.py (logical source → path)
    # -----------------------------------------

    def dataset_key_for_source(self, logical_name: str) -> str:
        """
        Given a logical source name (e.g. "GLORYS", "UNIFIED"), return the
        dataset key (e.g. "GLORYS_REGIONAL", "UNIFIED_BGC") used in
        DATASET_REGISTRY and `self.paths`.
        """
        return map_source_to_dataset_key(logical_name)

    def path_for_source(self, logical_name: str) -> Path:
        """
        Return the prepared file path associated with a logical source name.

        Parameters
        ----------
        logical_name : str
            Logical source name, e.g. "GLORYS", "UNIFIED", "DAI".

        Returns
        -------
        Path
            Path to the corresponding dataset file.

        Raises
        ------
        KeyError
            If the mapped dataset key was not among `self.datasets` or has
            not been prepared (i.e., `prepare_all()` has not been called
            or the dataset was omitted).
        """
        key = self.dataset_key_for_source(logical_name)
        try:
            return self.paths[key]
        except KeyError:
            raise KeyError(
                f"Source '{logical_name}' maps to dataset '{key}', "
                f"but that dataset was not prepared. Available datasets: "
                f"{', '.join(sorted(self.paths.keys()))}"
            )

    # -----------------------------------------
    # Internals / helpers
    # -----------------------------------------

    def _construct_glorys_path(self, date: datetime, is_regional: bool) -> Path:
        """Construct filename for a single day of GLORYS data."""
        date_str = date.strftime('%Y%m%d')
        dataset_name = "GLORYS_REGIONAL" if is_regional else "GLORYS_GLOBAL"
        if is_regional:
            fn = f"{glorys_dataset_id}_REGIONAL_{self.grid_name}_{date_str}.nc"
        else:
            fn = f"{glorys_dataset_id}_GLOBAL_{date_str}.nc"
        dataset_dir = config.paths.source_data / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        return dataset_dir / fn

    def _prepare_glorys_daily(self, is_regional: bool, bounds: Dict[str, Optional[float]]) -> List[Path]:
        """
        Download or reuse daily GLORYS subsets.
        
        Parameters
        ----------
        is_regional : bool
            True for regional (grid-based), False for global.
        bounds : dict
            Dictionary with keys minimum_longitude, maximum_longitude,
            minimum_latitude, maximum_latitude. For global, all values are None.
        """
        paths = []
        
        # Iterate over each day from start_time to end_time
        current_date = datetime(self.start_time.year, self.start_time.month, self.start_time.day)
        end_date = datetime(self.end_time.year, self.end_time.month, self.end_time.day)
        
        while current_date <= end_date:
            # Construct path for this day
            path = self._construct_glorys_path(current_date, is_regional)
            paths.append(path)
            
            needs_download = self.clobber or (not path.exists())
            
            if needs_download:
                if path.exists():
                    dataset_type = "GLORYS_REGIONAL" if is_regional else "GLORYS_GLOBAL"
                    print(f"⚠️  Clobber=True: removing existing {dataset_type} file {path.name}")
                    path.unlink()
                
                dataset_type = "GLORYS_REGIONAL" if is_regional else "GLORYS_GLOBAL"
                date_str = current_date.strftime('%Y-%m-%d')
                print(f"⬇️  Downloading {dataset_type} for {date_str} → {path.name}")

                copernicusmarine.subset(
                    dataset_id=glorys_dataset_id,
                    variables=["thetao", "so", "uo", "vo", "zos"],
                    coordinates_selection_method="outside",
                    start_datetime=current_date,
                    end_datetime=current_date,
                    output_filename=path.name,
                    output_directory=path.parent,
                    **bounds,
                )
            else:
                dataset_type = "GLORYS_REGIONAL" if is_regional else "GLORYS_GLOBAL"
                date_str = current_date.strftime('%Y-%m-%d')
                print(f"✔️  Using existing {dataset_type} file for {date_str}: {path.name}")
            
            # Move to next day
            current_date += timedelta(days=1)
        
        # Store paths - use first path or a list
        self.paths["GLORYS"] = paths[0] if len(paths) == 1 else paths
        return paths


# ---------------------------
# GLORYS_REGIONAL handler
# ---------------------------


@register_dataset(
    "GLORYS_REGIONAL",
    requires=["grid", "grid_name", "start_time", "end_time"],
)
def _prepare_glorys_regional(self: SourceData) -> List[Path]:
    """Download or reuse daily regional GLORYS subsets for this grid and time range."""
    is_regional = True
    bounds = rt.get_glorys_bounds(self.grid)
    return self._prepare_glorys_daily(is_regional, bounds)


# ---------------------------
# GLORYS_GLOBAL handler
# ---------------------------


@register_dataset(
    "GLORYS_GLOBAL",
    requires=["start_time", "end_time"],
)
def _prepare_glorys_global(self: SourceData) -> List[Path]:
    """Download or reuse daily global GLORYS subsets for this time range."""
    is_regional = False
    bounds = {
        "minimum_longitude": None,
        "maximum_longitude": None,
        "minimum_latitude": None,
        "maximum_latitude": None,
    }
    return self._prepare_glorys_daily(is_regional, bounds)

# ---------------------------
# UNIFIED BGC handler
# ---------------------------


@register_dataset("UNIFIED_BGC")
def _prepare_unified_bgc_dataset(self: SourceData) -> Path:
    """Ensure the UNIFIED_BGC dataset exists locally."""
    url_bgc_forcing = (
        "https://drive.google.com/uc?id=1wUNwVeJsd6yM7o-5kCx-vM3wGwlnGSiq"
    )
    dataset_dir = config.paths.source_data / "UNIFIED_BGC"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    path = dataset_dir / "BGCdataset.nc"
    needs_download = self.clobber or (not path.exists())

    if needs_download:
        if path.exists():
            print(f"⚠️  Clobber=True: removing existing BGC file {path.name}")
            path.unlink()

        print(f"⬇️  Downloading BGC dataset → {path}")
        gdown.download(url_bgc_forcing, str(path), quiet=False)
    else:
        print(f"✔️  Using existing BGC dataset: {path}")

    self.bgc_forcing_path = path
    self.paths["UNIFIED_BGC"] = path
    return path


# ---------------------------
# SRTM15+ handler
# ---------------------------


@register_dataset("SRTM15")
def _prepare_srtm15(self: SourceData) -> Path:
    """
    Ensure the SRTM15 bathymetry dataset exists locally.

    Download if:
      - the file does not exist, or
      - clobber=True.

    The file is stored under config.paths.source_data / "SRTM15" / "SRTM15_{SRTM15_VERSION}.nc".
    """
    dataset_dir = config.paths.source_data / "SRTM15"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    path = dataset_dir / f"SRTM15_{SRTM15_VERSION}.nc"

    needs_download = self.clobber or (not path.exists())

    if needs_download:
        if path.exists():
            print(f"⚠️  Clobber=True: removing existing SRTM15 file {path.name}")
            path.unlink()

        print(f"⬇️  Downloading SRTM15+ {SRTM15_VERSION} bathymetry → {path}")

        with tempfile.NamedTemporaryFile(delete=False, dir=str(dataset_dir)) as tmpfile:
            with urlopen(SRTM15_URL) as r:
                shutil.copyfileobj(r, tmpfile)
            tmp_path = Path(tmpfile.name)

        tmp_path.replace(path)
        print(f"✔️  SRTM15+ download complete: {path}")
    else:
        print(f"✔️  Using existing SRTM15+ dataset: {path}")

    self.srtm15_path = path
    return path
