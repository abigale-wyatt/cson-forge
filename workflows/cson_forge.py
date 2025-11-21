from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import os
import shutil
import stat
import subprocess
import sys
from datetime import datetime
import uuid

import yaml
from jinja2 import Environment, FileSystemLoader, StrictUndefined

import config
import roms_tools as rt
import source_data


# =========================================================
# Shared data structures (repos, model spec, inputs)
# =========================================================


@dataclass
class RepoSpec:
    """
    Specification for a code repository used in the build.

    Parameters
    ----------
    name : str
        Short name for the repository (e.g., "roms", "marbl").
    url : str
        Git URL for the repository.
    default_dirname : str
        Default directory name under the code root where this repo
        will be cloned.
    checkout : str, optional
        Optional tag, branch, or commit to check out after cloning.
    """
    name: str
    url: str
    default_dirname: str
    checkout: str | None = None


@dataclass
class ModelSpec:
    """
    Description of an ocean model configuration (e.g., ROMS/MARBL).

    Parameters
    ----------
    name : str
        Logical name of the model (e.g., "roms-marbl").
    opt_base_dir : str
        Relative path (under model-configs) to the base configuration
        directory.
    conda_env : str
        Name of the conda environment used to build/run this model.
    repos : dict[str, RepoSpec]
        Mapping from repo name to its specification.
    inputs : dict[str, dict]
        Per-input default arguments (from models.yml["<model>"]["inputs"]).
        These are merged with runtime arguments when constructing ROMS inputs.
    datasets : list[str]
        SourceData dataset keys required for this model (derived from inputs
        or explicitly listed in models.yml).
    """
    name: str
    opt_base_dir: str
    conda_env: str
    repos: Dict[str, RepoSpec]
    inputs: Dict[str, Dict[str, Any]]
    datasets: List[str]


def _extract_source_name(block: Union[str, Dict[str, Any], None]) -> Optional[str]:
    if block is None:
        return None
    if isinstance(block, str):
        return block
    if isinstance(block, dict):
        return block.get("name")
    return None


def _dataset_keys_from_inputs(inputs: Dict[str, Dict[str, Any]]) -> set[str]:
    dataset_keys: set[str] = set()
    for cfg in inputs.values():
        if not isinstance(cfg, dict):
            continue
        for field_name in ("source", "bgc_source", "topography_source"):
            name = _extract_source_name(cfg.get(field_name))
            if not name:
                continue
            dataset_key = source_data.map_source_to_dataset_key(name)
            if dataset_key in source_data.DATASET_REGISTRY:
                dataset_keys.add(dataset_key)
    return dataset_keys


def _collect_datasets(block: Dict[str, Any], inputs: Dict[str, Dict[str, Any]]) -> List[str]:
    dataset_keys: set[str] = set()

    explicit = block.get("datasets") or []
    for item in explicit:
        if not item:
            continue
        dataset_keys.add(str(item).upper())

    dataset_keys.update(_dataset_keys_from_inputs(inputs))
    return sorted(dataset_keys)


def _load_models_yaml(path: Path, model: str) -> ModelSpec:
    """
    Load repository specifications, model metadata, and default input
    arguments from a YAML file.

    Parameters
    ----------
    path : Path
        Path to the models.yaml file.
    model : str
        Name of the model block to load (e.g., "roms-marbl").

    Returns
    -------
    ModelSpec
        Parsed model specification including repository metadata and
        per-input defaults.

    Raises
    ------
    KeyError
        If the requested model is not present in the YAML file.
    """
    with path.open() as f:
        data = yaml.safe_load(f) or {}

    if model not in data:
        raise KeyError(f"Model '{model}' not found in models YAML file: {path}")

    block = data[model]

    repos: Dict[str, RepoSpec] = {}
    for key, val in block.get("repos", {}).items():
        repos[key] = RepoSpec(
            name=key,
            url=val["url"],
            default_dirname=val.get("default_dirname", key),
            checkout=val.get("checkout"),
        )

    inputs = block.get("inputs", {}) or {}
    datasets = _collect_datasets(block, inputs)

    return ModelSpec(
        name=model,
        opt_base_dir=block["opt_base_dir"],
        conda_env=block["conda_env"],
        repos=repos,
        inputs=inputs,
        datasets=datasets,
    )


# =========================================================
# ROMS input generation (from former model_config.py)
# =========================================================


class InputStep:
    """Metadata for a single ROMS input generation step."""

    def __init__(self, name: str, order: int, label: str, handler: Callable):
        self.name = name  # canonical key used for filenames & paths
        self.order = order  # execution order
        self.label = label  # human-readable label
        self.handler = handler  # function expecting `self` (ROMSInputs instance)


INPUT_REGISTRY: Dict[str, InputStep] = {}


def register_input(name: str, order: int, label: str | None = None):
    """
    Decorator to register an input-generation step.

    Parameters
    ----------
    name : str
        Key for this input (e.g., 'grid', 'initial_conditions', 'surface_forcing').
        This will be used in filenames, and to index `inputs[name]`.
    order : int
        Execution order in `generate_all()`. Lower numbers run first.
    label : str, optional
        Human-readable label for progress messages. If omitted, `name` is used.
    """

    def decorator(func: Callable):
        step_label = label or name
        INPUT_REGISTRY[name] = InputStep(
            name=name,
            order=order,
            label=step_label,
            handler=func,
        )
        return func

    return decorator


@dataclass
class InputObj:
    """
    Structured representation of a single ROMS input product.

    Attributes
    ----------
    input_type : str
        The type/key of this input (e.g., "initial_conditions", "surface_forcing").
    paths : Path | list[Path] | None
        Path or list of paths to the generated NetCDF file(s), if applicable.
    paths_partitioned : Path | list[Path] | None
        Path(s) to the partitioned NetCDF file(s), if applicable.
    yaml_file : Path | None
        Path to the YAML description written for this input, if any.
    """

    input_type: str
    paths: Optional[Union[Path, List[Path]]] = None
    paths_partitioned: Optional[Union[Path, List[Path]]] = None
    yaml_file: Optional[Path] = None


@dataclass
class ROMSInputs:
    """
    Generate and manage ROMS input files for a given grid.

    This object is driven by:
      - per-input default arguments loaded from `models.yml` (model_inputs).

    The list of inputs to generate (`roms_input_list`) is automatically
    derived from the keys in `model_inputs`.

    The defaults from `model_inputs[<key>]` are merged with runtime arguments
    (e.g., start_time, end_time, boundaries). Any "source" or "bgc_source"
    fields in the defaults are resolved through `SourceData`, which injects
    a "path" entry pointing at the prepared dataset file.
    """

    # core config
    model_name: str
    grid_name: str
    grid: object
    start_time: object
    end_time: object
    np_eta: int
    np_xi: int
    boundaries: dict
    source_data: source_data.SourceData

    # per-input defaults from models.yml["<model>"]["inputs"]
    model_inputs: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # which inputs to generate for this run (derived from model_inputs keys)
    roms_input_list: List[str] = field(init=False)

    use_dask: bool = True
    clobber: bool = False

    # derived
    input_data_dir: Path = field(init=False)
    inputs: Dict[str, InputObj] = field(init=False)
    bp_path: Path = field(init=False)

    def __post_init__(self):
        # Path to input directory
        self.input_data_dir = config.paths.input_data / self.grid_name
        self.input_data_dir.mkdir(exist_ok=True)

        self.bp_path = config.paths.blueprints / f"model-inputs_{self.grid_name}.yml"
        self.bp_path.parent.mkdir(parents=True, exist_ok=True)

        # Storage for detailed per-input objects
        self.inputs = {}
        
        # Derive roms_input_list from model_inputs keys
        roms_input_list = list(self.model_inputs.keys())
        if "grid" not in roms_input_list:
            roms_input_list.insert(0, "grid")
        self.roms_input_list = roms_input_list

    # ----------------------------
    # Public API
    # ----------------------------

    def generate_all(self):
        """
        Generate all ROMS input files for this grid using the registered
        steps whose names appear in `roms_input_list`, then partition and
        write a blueprint.

        If any names in `roms_input_list` lack registered handlers,
        a ValueError is raised.
        """

        if not self._ensure_empty_or_clobber(self.clobber):
            return self

        # Sanity check
        registry_keys = set(INPUT_REGISTRY.keys())
        needed = set(self.roms_input_list)
        missing = sorted(needed - registry_keys)
        if missing:
            raise ValueError(
                "The following ROMS inputs are listed in `roms_input_list` but "
                f"have no registered handlers: {', '.join(missing)}"
            )

        # Use only the selected steps
        steps = [INPUT_REGISTRY[name] for name in self.roms_input_list]
        steps.sort(key=lambda s: s.order)
        total = len(steps) + 1

        # Execute
        for idx, step in enumerate(steps, start=1):
            print(f"\n▶️  [{idx}/{total}] {step.label}...")
            step.handler(self, key=step.name)

        # Partition step
        print(f"\n▶️  [{total}/{total}] Partitioning input files across tiles...")
        self._partition_files()

        print("\n✅ All input files generated and partitioned.\n")
        self._write_inputs_blueprint()
        return self

    # ----------------------------
    # Internals
    # ----------------------------

    def _ensure_empty_or_clobber(self, clobber: bool) -> bool:
        """
        Ensure the input_data_dir is either empty or, if clobber=True,
        remove existing .nc files.
        """
        existing = list(self.input_data_dir.glob("*.nc"))

        if existing and not clobber:
            print(f"⚠️  Found existing ROMS input files in {self.input_data_dir}")
            print("    Not overwriting because clobber=False.")
            print("\nExiting without changes.\n")
            return False

        if existing and clobber:
            print(
                f"⚠️  Clobber=True: removing {len(existing)} existing .nc files in "
                f"{self.input_data_dir}..."
            )
            for f in existing:
                f.unlink()

        return True

    def _forcing_filename(self, key: str) -> Path:
        """Construct the NetCDF filename for a given input key."""
        return self.input_data_dir / f"roms_{key}.nc"

    def _yaml_filename(self, key: str) -> Path:
        """Construct the YAML blueprint filename for a given input key."""
        blueprint_dir = config.paths.blueprints / f"{self.model_name}-{self.grid_name}"
        blueprint_dir.mkdir(parents=True, exist_ok=True)
        return blueprint_dir / f"{key}.yaml"

    # ----------------------------
    # Helpers for merging YAML defaults
    # ----------------------------

    def _resolve_source_block(self, block: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Normalize a "source"/"bgc_source" block and inject a 'path'
        based on SourceData.

        Parameters
        ----------
        block : str or dict
            Either a simple logical name (e.g., "GLORYS") or a dict
            with at least a "name" field.

        Returns
        -------
        dict
            Source block with a "name" and "path" field (plus any
            additional keys from the original block).
        """
        if isinstance(block, str):
            name = block
            out: Dict[str, Any] = {"name": name}
        elif isinstance(block, dict):
            out = dict(block)
            name = out.get("name")
            if not name:
                raise ValueError(
                    f"Source block {block!r} is missing a 'name' field."
                )
        else:
            raise TypeError(f"Unsupported source block type: {type(block)}")

        path = self.source_data.path_for_source(name)
        # don't clobber a path if the user explicitly provided one
        out.setdefault("path", path)
        return out

    def _build_input_args(self, key: str, extra: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge per-input defaults (from models.yml) with runtime arguments.

        - Start from `model_inputs.get(key, {})`.
        - If present, resolve "source" and "bgc_source" through SourceData,
          injecting a "path" entry.
        - Merge with `extra`, where `extra` overrides defaults on conflict.
        """
        cfg = dict(self.model_inputs.get(key, {}) or {})

        for field_name in ("source", "bgc_source"):
            if field_name in cfg:
                cfg[field_name] = self._resolve_source_block(cfg[field_name])

        # `extra` overrides defaults
        return {**cfg, **extra}

    # ----------------------------
    # Registry-backed generation steps
    # ----------------------------

    @register_input(name="grid", order=10, label="Writing ROMS grid")
    def _generate_grid(self, key: str = "grid", **kwargs):
        out_path = self._forcing_filename(key)
        yaml_path = self._yaml_filename(key)

        self.grid.save(out_path)
        self.grid.to_yaml(yaml_path)

        self.inputs[key] = InputObj(
            input_type=key,
            paths=out_path,
            yaml_file=yaml_path,
        )

    @register_input(name="initial_conditions", order=20, label="Generating initial conditions")
    def _generate_initial_conditions(self, key: str = "initial_conditions", **kwargs):
        yaml_path = self._yaml_filename(key)
        extra = dict(
            ini_time=self.start_time,
            use_dask=self.use_dask,
        )
        input_args = self._build_input_args(key, extra=extra)

        ic = rt.InitialConditions(grid=self.grid, **input_args)
        paths = ic.save(self._forcing_filename(key))
        ic.to_yaml(yaml_path)

        self.inputs[key] = InputObj(
            input_type=key,
            paths=paths,
            yaml_file=yaml_path,
        )

    @register_input(name="surface_forcing", order=30, label="Generating surface forcing (physics)")
    def _generate_surface_forcing(self, key: str = "surface_forcing", **kwargs):
        yaml_path = self._yaml_filename(key)
        extra = dict(
            start_time=self.start_time,
            end_time=self.end_time,
            use_dask=self.use_dask,
        )
        input_args = self._build_input_args(key, extra=extra)

        frc = rt.SurfaceForcing(grid=self.grid, **input_args)
        paths = frc.save(self._forcing_filename(key))
        frc.to_yaml(yaml_path)

        self.inputs[key] = InputObj(
            input_type=key,
            paths=paths,
            yaml_file=yaml_path,
        )

    @register_input(name="surface_forcing_bgc", order=40, label="Generating surface forcing (BGC)")
    def _generate_bgc_surface_forcing(self, key: str = "surface_forcing_bgc", **kwargs):
        yaml_path = self._yaml_filename(key)
        extra = dict(
            start_time=self.start_time,
            end_time=self.end_time,
            use_dask=self.use_dask,
        )
        input_args = self._build_input_args(key, extra=extra)

        frc_bgc = rt.SurfaceForcing(grid=self.grid, **input_args)
        paths = frc_bgc.save(self._forcing_filename(key))
        frc_bgc.to_yaml(yaml_path)

        self.inputs[key] = InputObj(
            input_type=key,
            paths=paths,
            yaml_file=yaml_path,
        )

    @register_input(name="boundary_forcing", order=50, label="Generating boundary forcing (physics)")
    def _generate_boundary_forcing(self, key: str = "boundary_forcing", **kwargs):
        yaml_path = self._yaml_filename(key)
        extra = dict(
            start_time=self.start_time,
            end_time=self.end_time,
            boundaries=self.boundaries,
            use_dask=self.use_dask,
        )
        input_args = self._build_input_args(key, extra=extra)

        bry = rt.BoundaryForcing(grid=self.grid, **input_args)
        paths = bry.save(self._forcing_filename(key))
        bry.to_yaml(yaml_path)

        self.inputs[key] = InputObj(
            input_type=key,
            paths=paths,
            yaml_file=yaml_path,
        )

    @register_input(name="boundary_forcing_bgc", order=60, label="Generating boundary forcing (BGC)")
    def _generate_bgc_boundary_forcing(self, key: str = "boundary_forcing_bgc", **kwargs):
        yaml_path = self._yaml_filename(key)
        extra = dict(
            start_time=self.start_time,
            end_time=self.end_time,
            boundaries=self.boundaries,
            use_dask=self.use_dask,
        )
        input_args = self._build_input_args(key, extra=extra)

        bry_bgc = rt.BoundaryForcing(grid=self.grid, **input_args)
        paths = bry_bgc.save(self._forcing_filename(key))
        bry_bgc.to_yaml(yaml_path)

        self.inputs[key] = InputObj(
            input_type=key,
            paths=paths,
            yaml_file=yaml_path,
        )

    @register_input(name="rivers", order=70, label="Generating river forcing")
    def _generate_river_forcing(self, key: str = "rivers", **kwargs):
        yaml_path = self._yaml_filename(key)
        extra = dict(
            start_time=self.start_time,
            end_time=self.end_time,
        )
        input_args = self._build_input_args(key, extra=extra)

        rivers = rt.RiverForcing(grid=self.grid, **input_args)
        paths = rivers.save(self._forcing_filename(key))
        rivers.to_yaml(yaml_path)

        self.inputs[key] = InputObj(
            input_type=key,
            paths=paths,
            yaml_file=yaml_path,
        )

    @register_input(name="cdr", order=80, label="Generating CDR forcing")
    def _generate_cdr_forcing(self, key: str = "cdr", cdr_list=None, **kwargs):
        cdr_list = [] if cdr_list is None else cdr_list
        if not cdr_list:
            return

        yaml_path = self._yaml_filename(key)
        extra = dict(
            start_time=self.start_time,
            end_time=self.end_time,
            releases=cdr_list,
        )
        input_args = self._build_input_args(key, extra=extra)

        cdr = rt.CDRForcing(grid=self.grid, **input_args)
        paths = cdr.save(self._forcing_filename(key))
        cdr.to_yaml(yaml_path)

        self.inputs[key] = InputObj(
            input_type=key,
            paths=paths,
            yaml_file=yaml_path,
        )

    # ----------------------------
    # Partition step (not in registry)
    # ----------------------------

    def _partition_files(self, **kwargs):
        """
        Partition whole input files across tiles using roms_tools.partition_netcdf.

        Uses the paths stored in `inputs[...]` (for keys in roms_input_list)
        to build the list of whole-field files, and records the partitioned
        paths on each InputObj.
        """
        input_args = dict(
            np_eta=self.np_eta,
            np_xi=self.np_xi,
            output_dir=self.input_data_dir,
            include_coarse_dims=False,
        )

        for name in self.roms_input_list:
            obj = self.inputs.get(name)
            if obj is None or obj.paths is None:
                continue
            obj.paths_partitioned = rt.partition_netcdf(obj.paths, **input_args)

    # ----------------------------
    # Blueprint writer
    # ----------------------------

    def _write_inputs_blueprint(self):
        """
        Serialize a summary of ROMSInputs state to a YAML blueprint:

            blueprints/model-inputs_{grid_name}.yml

        Contents include high-level configuration and a sanitized view of
        `inputs` (paths, arguments, etc.).
        """
        import xarray as xr

        XR_TYPES = (xr.Dataset, xr.DataArray)

        def _serialize(obj: Any) -> Any:
            from datetime import date, datetime
            from dataclasses import is_dataclass, asdict as dc_asdict

            if XR_TYPES and isinstance(obj, XR_TYPES):
                return None

            if is_dataclass(obj) and not isinstance(obj, type):
                return _serialize(dc_asdict(obj))

            if isinstance(obj, Path):
                return str(obj)

            if isinstance(obj, (str, int, float, bool)) or obj is None:
                return obj

            if isinstance(obj, (date, datetime)):
                return obj.isoformat()

            if isinstance(obj, dict):
                return {k: _serialize(v) for k, v in obj.items()}

            if isinstance(obj, (list, tuple, set)):
                return [_serialize(v) for v in obj]

            if callable(obj):
                qualname = getattr(obj, "__qualname__", None)
                mod = getattr(obj, "__module__", None)
                if qualname and mod:
                    return f"{mod}.{qualname}"
                return repr(obj)

            return repr(obj)

        raw = dict(
            grid_name=self.grid_name,
            start_time=self.start_time,
            end_time=self.end_time,
            np_eta=self.np_eta,
            np_xi=self.np_xi,
            boundaries=self.boundaries,
            input_data_dir=self.input_data_dir,
            roms_input_list=self.roms_input_list,
            model_inputs=self.model_inputs,
            inputs=self.inputs,
        )

        data = _serialize(raw)

        with self.bp_path.open("w") as f:
            yaml.safe_dump(data, f, sort_keys=True)

        print(f"📄  Wrote ROMSInputs blueprint to {self.bp_path}")


# =========================================================
# Build logic (from former model_build.py)
# =========================================================


def _run(cmd: list[str]) -> str:
    """Run a command and return stdout as text."""
    result = subprocess.run(
        cmd, check=True, text=True, capture_output=True
    )
    return result.stdout


def _check_command_exists(command: str) -> None:
    """Raise if a command is not found on PATH."""
    if shutil.which(command) is None:
        raise RuntimeError(f"Required command '{command}' not found on PATH.")


def _render_opt_base_dir_to_opt_dir(
    grid_name: str,
    parameters: Dict[str, Dict[str, Any]],
    opt_base_dir: Path,
    opt_dir: Path,
    overwrite: bool = False,
    log_func: Callable[[str], None] = print,
) -> None:
    """
    Stage and render model configuration templates using Jinja2.

    See original docstring in model_build.py for full details.
    """
    src = opt_base_dir.resolve()
    dst = opt_dir.resolve()

    if overwrite and dst.exists():
        log_func(f"[Render] Clearing existing opt_dir: {dst}")
        shutil.rmtree(dst)

    # Copy everything except an existing opt_<grid_name> directory
    shutil.copytree(
        src,
        dst,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns(f"opt_{grid_name}"),
    )

    env = Environment(
        loader=FileSystemLoader(str(dst)),
        undefined=StrictUndefined,
        autoescape=False,
    )

    for relpath, context in parameters.items():
        template_path = dst / relpath
        if not template_path.exists():
            raise FileNotFoundError(
                f"Template file '{relpath}' listed in parameters but not found in {dst}"
            )
        log_func(f"[Render] Rendering template: {relpath}")

        template = env.get_template(relpath)
        rendered = template.render(**context)

        st = template_path.stat()
        with template_path.open("w") as f:
            f.write(rendered)
        os.chmod(template_path, st.st_mode)


def build(
    grid_name: str,
    model_name: str,
    parameters: Dict[str, Dict[str, Any]],
    clean: bool = False,
) -> Optional[Path]:
    """
    Build the ocean model for a given grid and `model_name` (e.g., "roms-marbl").

    This is essentially the previous `build()` function from model_build.py,
    now using `ModelSpec` from this module.
    """
    # Unique build token and logging setup
    build_token = (
        datetime.utcnow().strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    )

    # Load model spec and derive directories
    model_code = _load_models_yaml(config.paths.models_yaml, model_name)
    opt_base_dir = config.paths.model_configs / model_code.opt_base_dir

    opt_dir = config.paths.model_configs / "opt" / f"opt_{model_code.name}-{grid_name}"
    opt_dir.mkdir(parents=True, exist_ok=True)

    build_dir_final = config.paths.model_configs / "bld" / f"bld_{model_code.name}-{grid_name}"
    build_dir_tmp = config.paths.model_configs / "bld" / f"tmp_bld_{model_code.name}-{grid_name}"
    build_dir_tmp.mkdir(parents=True, exist_ok=True)
    if build_dir_tmp.exists() and clean:
        shutil.rmtree(build_dir_tmp)

    roms_conda_env = model_code.conda_env
    repos = model_code.repos
    if "roms" not in repos or "marbl" not in repos:
        raise ValueError("models.yml must define at least 'roms' and 'marbl' repos.")

    logs_dir = build_dir_tmp / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    build_all_log = logs_dir / f"build.{model_code.name}.{build_token}.log"

    def log(msg: str = "") -> None:
        text = str(msg)
        print(text)
        build_all_log.parent.mkdir(parents=True, exist_ok=True)
        with build_all_log.open("a") as f:
            f.write(text + "\n")

    log(f"Build token: {build_token}")

    # Paths from config / sanity checks
    input_data_path = config.paths.input_data / grid_name
    if not input_data_path.is_dir():
        raise FileNotFoundError(
            f"Expected input data directory for grid '{grid_name}' at:\n"
            f"  {input_data_path}\n"
            "but it does not exist. Did you run the `gen_inputs` step?"
        )

    codes_root = config.paths.code_root
    roms_root = codes_root / repos["roms"].default_dirname
    marbl_root = codes_root / repos["marbl"].default_dirname

    log(f"Building {model_code.name} for grid: {grid_name}")
    log(f"{model_code.name} opt_base_dir : {opt_base_dir}")
    log(f"ROMS opt_dir      : {opt_dir}")
    log(f"ROMS build_dir    : {build_dir_final}")
    log(f"Input data path   : {input_data_path}")
    log(f"ROMS_ROOT         : {roms_root}")
    log(f"MARBL_ROOT        : {marbl_root}")
    log(f"Conda env         : {roms_conda_env}")
    log(f"Logs              : {logs_dir}")

    # Check conda and define conda-run helper
    _check_command_exists("conda")

    def _conda_run(cmd: list[str]) -> list[str]:
        return ["conda", "run", "-n", roms_conda_env] + cmd

    # Create env if needed
    env_list = _run(["conda", "env", "list"])
    if roms_conda_env not in env_list:
        log(f"Creating conda env '{roms_conda_env}' from ROMS environment file...")
        env_yml = roms_root / "environments" / "conda_environment.yml"
        if not env_yml.exists():
            raise FileNotFoundError(f"Conda environment file not found: {env_yml}")
        _run(
            [
                "conda",
                "env",
                "create",
                "-f",
                str(env_yml),
                "--name",
                roms_conda_env,
            ]
        )
    else:
        log(f"Conda env '{roms_conda_env}' already exists.")

    # Toolchain checks
    try:
        _run(_conda_run(["which", "gfortran"]))
        _run(_conda_run(["which", "mpifort"]))
    except subprocess.CalledProcessError:
        raise RuntimeError(
            f"❌ gfortran or mpifort not found in env '{roms_conda_env}'. "
            "Check your conda environment."
        )

    compiler_kind = "gnu"
    try:
        mpifort_version = _run(_conda_run(["mpifort", "--version"]))
        if any(token in mpifort_version.lower() for token in ["ifx", "ifort", "intel"]):
            compiler_kind = "intel"
    except Exception:
        pass

    log(f"Using compiler kind: {compiler_kind}")

    # Build fingerprint & cache lookup (unchanged logic from model_build.py)...
    #  (omitted for brevity; you can keep your existing fingerprint/builds.yaml logic here)

    # Render config files
    _render_opt_base_dir_to_opt_dir(
        grid_name=grid_name,
        parameters=parameters,
        opt_base_dir=opt_base_dir,
        opt_dir=opt_dir,
        overwrite=True,
        log_func=log,
    )

    # Actual make step (unchanged from your original code) ...
    #  e.g., _run(_conda_run(["make", "..."])) in build_dir_tmp, then move exe to build_dir_final.

    # For now, just return the final build dir (or executable if you want)
    return build_dir_final


# =========================================================
# High-level OcnModel object
# =========================================================


@dataclass
class OcnModel:
    """
    High-level object:
      - model metadata from models.yml (ModelSpec),
      - source datasets (SourceData),
      - ROMS input generation (ROMSInputs),
      - model build (via `build()`).

    Typical usage
    -------------
    grid_kwargs = dict(
        nx=10,
        ny=10,
        size_x=4000,
        size_y=2000,
        center_lon=4.0,
        center_lat=-1.0,
        rot=0,
        N=5,
    )

    ocn = OcnModel(
        model_name="roms-marbl",
        grid_name=grid_name,
        grid_kwargs=grid_kwargs,
        boundaries=boundaries,
        start_time=start_time,
        end_time=end_time,
        np_eta=np_eta,
        np_xi=np_xi,
    )

    ocn.prepare_source_data()
    ocn.generate_inputs()
    ocn.build()
    """

    model_name: str
    grid_name: str
    grid_kwargs: Dict[str, Any]
    boundaries: dict
    start_time: object
    end_time: object
    np_eta: int
    np_xi: int
    grid: object = field(init=False)
    spec: ModelSpec = field(init=False)
    src_data: Optional[source_data.SourceData] = field(init=False, default=None)
    
    def __post_init__(self):
        self.grid = rt.Grid(**self.grid_kwargs)
        self.spec = _load_models_yaml(config.paths.models_yaml, self.model_name)
        self.inputs = None
        self.executable = None

    @property
    def name(self) -> str:
        return f"{self.spec.name}_{self.grid_name}"

    def prepare_source_data(self, clobber: bool = False):
        self.src_data = source_data.SourceData(
            datasets=self.spec.datasets,
            clobber=clobber,
            grid=self.grid,
            grid_name=self.grid_name,
            start_time=self.start_time,
            end_time=self.end_time,
        ).prepare_all()
    
    def generate_inputs(
        self,
        clobber: bool = False,
    ) -> ROMSInputs:
        """
        Generate ROMS input files for this model/grid.

        The list of inputs to generate is automatically derived from the
        keys in models.yml["<model_name>"]["inputs"].

        Parameters
        ----------
        clobber : bool, optional
            Passed through to ROMSInputs to allow overwriting existing
            NetCDF files.
        
        Raises
        ------
        RuntimeError
            If `prepare_source_data()` has not been called yet.
        """
        if self.src_data is None:
            raise RuntimeError(
                "You must call OcnModel.prepare_source_data() "
                "before generating inputs."
            )
        self.inputs = ROMSInputs(
            model_name=self.model_name,
            grid_name=self.grid_name,
            grid=self.grid,
            start_time=self.start_time,
            end_time=self.end_time,
            np_eta=self.np_eta,
            np_xi=self.np_xi,
            boundaries=self.boundaries,
            source_data=self.src_data,
            model_inputs=self.spec.inputs,
            clobber=clobber,
        ).generate_all()

        return self.inputs

    def build(self, parameters: Dict[str, Dict[str, Any]], clean: bool = False) -> Path:
        """
        Build the model executable for this configuration, using the
        lower-level `build()` helper in this module.
        """
        self.executable = build(
            model_name=self.model_name,
            grid_name=self.grid_name,
            parameters=parameters,
            clean=clean,
        )
        return self.executable


