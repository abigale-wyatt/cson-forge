from pathlib import Path
from glob import glob
import shutil

import stat
from jinja2 import Environment, FileSystemLoader, StrictUndefined

import config


from pathlib import Path
from glob import glob


from dataclasses import dataclass, field

import roms_tools as rt
import config


from pathlib import Path
from glob import glob  # still here in case you use it elsewhere
import roms_tools as rt
import config


@dataclass
class ROMSInputs:
    """Generate and manage ROMS input files for a given grid."""

    # class-level constants
    roms_input_list = [
        "grd",
        "ic",
        "frc",
        "frc_bgc",
        "bry",
        "bry_bgc",
        "rivers",
        "cdr",
    ]

    # dataclass fields (constructor args)
    grid_name: str
    grid: object
    start_time: object
    end_time: object
    np_eta: int
    np_xi: int
    boundaries: dict
    source_data: object
    use_dask: bool = True
    clobber: bool = False

    # derived / internal fields (not passed by user)
    input_data_dir: Path = field(init=False)
    grid_yaml: Path = field(init=False)
    glorys_path: Path = field(init=False)
    bgc_forcing_path: Path = field(init=False)
    paths: dict = field(init=False)
    files_whole: list | None = field(init=False, default=None)
    _files_partitioned: object | None = field(init=False, default=None)

    def __post_init__(self):
        # paths to input directory & blueprint
        self.input_data_dir = Path(config.paths.input_data) / self.grid_name
        self.input_data_dir.mkdir(exist_ok=True)

        self.grid_yaml = Path(config.paths.blueprints) / f"{self.grid_name}.yaml"

        # source data paths from SourceData
        self.glorys_path = self.source_data.paths["GLORYS"]
        self.bgc_forcing_path = self.source_data.paths["UNIFIED_BGC"]

        # initialize paths dict for all ROMS inputs
        self.paths = {k: None for k in self.roms_input_list}

        # ensure directory state
        # TODO: push this into each input type so we can be more selective about
        #       what to keep and what to replace
        self._ensure_empty_or_clobber(self.clobber)

    # ----------------------------
    # Public API
    # ----------------------------

    def generate_all(self):
        """Generate all ROMS input files for this grid."""

        # 1. Grid
        print("\n▶️  [1/9] Writing ROMS grid...")
        self._generate_grid()

        # 2. Initial conditions
        print("▶️  [2/9] Generating initial conditions...")
        self._generate_initial_conditions()

        # 3. Surface forcing (physics)
        print("▶️  [3/9] Generating surface forcing (physics)...")
        self._generate_surface_forcing()

        # 4. Surface forcing (BGC)
        print("▶️  [4/9] Generating surface forcing (BGC)...")
        self._generate_bgc_surface_forcing()

        # 5. Boundary forcing (physics)
        print("▶️  [5/9] Generating boundary forcing (physics)...")
        self._generate_boundary_forcing()

        # 6. Boundary forcing (BGC)
        print("▶️  [6/9] Generating boundary forcing (BGC)...")
        self._generate_bgc_boundary_forcing()

        # 7. River forcing
        print("▶️  [7/9] Generating river forcing...")
        self._generate_river_forcing()

        # 8. CDR forcing
        print("▶️  [8/9] Generating CDR forcing...")
        self._generate_cdr_forcing()

        # 9. Partition
        print("\n📦  [9/9] Partitioning input files across tiles...")
        self._partition_files(
            np_eta=self.np_eta,
            np_xi=self.np_xi,
            output_dir=self.input_data_dir,
            include_coarse_dims=False,
        )

        print("✅ All input files generated and partitioned.\n")

    # ----------------------------
    # Internals
    # ----------------------------

    def _ensure_empty_or_clobber(self, clobber: bool):
        # TODO: this should not be all or nothing — push check/rm into each dataset
        existing = list(self.input_data_dir.glob("*.nc"))
        if existing and not clobber:
            file_list = ", ".join(f.name for f in existing)
            raise RuntimeError(
                f"Input directory '{self.input_data_dir}' is not empty. "
                f"Existing files: {file_list}. Use clobber=True to overwrite."
            )
        if existing and clobber:
            print(f"⚠️  Clobber=True: removing {len(existing)} existing .nc files...")
            for f in existing:
                f.unlink()

    def _forcing_filename(self, key):
        return self.input_data_dir / f"roms_{key}"

    def _yaml_filename(self, key):
        return config.paths.blueprints / (self.grid_name + f".{key}.yaml")

    def _generate_grid(self, **kwargs):
        out_path = self._forcing_filename("grd")
        self.grid.save(out_path)
        self.paths["grd"] = out_path
        self.grid.to_yaml(self._yaml_filename("grd"))

    def _generate_initial_conditions(self, **kwargs):
        self.ic = rt.InitialConditions(
            grid=self.grid,
            ini_time=self.start_time,
            source={
                "name": "GLORYS",
                "path": self.glorys_path,
            },
            bgc_source={
                "name": "UNIFIED",
                "path": self.bgc_forcing_path,
                "climatology": True,
            },
            use_dask=self.use_dask,
        )
        self.paths["ic"] = self.ic.save(self._forcing_filename("ic"))
        self.ic.to_yaml(self._yaml_filename("ic"))

    def _generate_surface_forcing(self, **kwargs):
        self.frc = rt.SurfaceForcing(
            grid=self.grid,
            start_time=self.start_time,
            end_time=self.end_time,
            source={"name": "ERA5"},
            type="physics",
            use_dask=self.use_dask,
        )
        self.paths["frc"] = self.frc.save(self._forcing_filename("frc"))
        self.frc.to_yaml(self._yaml_filename("frc"))

    def _generate_bgc_surface_forcing(self, **kwargs):
        self.frc_bgc = rt.SurfaceForcing(
            grid=self.grid,
            start_time=self.start_time,
            end_time=self.end_time,
            source={
                "name": "UNIFIED",
                "path": self.bgc_forcing_path,
                "climatology": True,
            },
            type="bgc",
            use_dask=self.use_dask,
        )
        self.paths["frc_bgc"] = self.frc_bgc.save(self._forcing_filename("frc_bgc"))
        self.frc_bgc.to_yaml(
            self._yaml_filename("frc_bgc")
        )

    def _generate_boundary_forcing(self, **kwargs):
        self.bry = rt.BoundaryForcing(
            grid=self.grid,
            start_time=self.start_time,
            end_time=self.end_time,
            boundaries=self.boundaries,
            source={
                "name": "GLORYS",
                "path": self.glorys_path,
            },
            type="physics",
            use_dask=self.use_dask,
        )
        self.paths["bry"] = self.bry.save(self._forcing_filename("bry"))
        self.bry.to_yaml(self._yaml_filename("bry"))

    def _generate_bgc_boundary_forcing(self, **kwargs):
        self.bry_bgc = rt.BoundaryForcing(
            grid=self.grid,
            start_time=self.start_time,
            end_time=self.end_time,
            boundaries=self.boundaries,
            source={
                "name": "UNIFIED",
                "path": self.bgc_forcing_path,
                "climatology": True,
            },
            type="bgc",
            use_dask=self.use_dask,
        )
        self.paths["bry_bgc"] = self.bry_bgc.save(self._forcing_filename("bry_bgc"))
        self.bry_bgc.to_yaml(
            self._yaml_filename("bry_bgc")
        )

    def _generate_river_forcing(self, **kwargs):
        self.rivers = rt.RiverForcing(
            grid=self.grid,
            start_time=self.start_time,
            end_time=self.end_time,
            include_bgc=True,
        )
        self.paths["rivers"] = self.rivers.save(self._forcing_filename("rivers"))
        self.rivers.to_yaml(self._yaml_filename("rivers"))

    def _generate_cdr_forcing(self, cdr_list=None):
        if cdr_list is None:
            cdr_list = []
        # TODO: cdr_list is a list of releases that are passed in
        if cdr_list:
            self.cdr = rt.CDRForcing(
                grid=self.grid,
                start_time=self.start_time,
                end_time=self.end_time,
                releases=cdr_list,
            )
            self.paths["cdr"] = self.cdr.save(self._forcing_filename("cdr"))
            self.cdr.to_yaml(self._yaml_filename("cdr"))

    def _partition_files(self, **kwargs):
        # Build files_whole from the paths of generated ROMS inputs
        self.files_whole = []

        for k, v in self.paths.items():
            if v is None:
                continue
            if isinstance(v, list):
                self.files_whole.extend([p for p in v if p is not None])
            else:
                self.files_whole.append(v)
        # partition files
        self._files_partitioned = rt.partition_netcdf(
            self.files_whole,
            **kwargs,
        )


# Optional convenience function that mirrors your original API
def gen_inputs(
    grid_name,
    grid,
    start_time,
    end_time,
    np_eta,
    np_xi,
    boundaries,
    source_data,
    clobber: bool = False,
):
    roms_inputs = ROMSInputs(
        grid_name=grid_name,
        grid=grid,
        start_time=start_time,
        end_time=end_time,
        np_eta=np_eta,
        np_xi=np_xi,
        boundaries=boundaries,
        source_data=source_data,
        clobber=clobber,
    )
    roms_inputs.generate_all()
    return roms_inputs


def render_source(grid_name, parameters):
    """
    Stage and render model configuration templates using Jinja2.

    This function creates a working copy of the model configuration directory,
    renders selected files in place using the provided parameter dictionary,
    and preserves original file permissions. It is typically used to generate
    model-specific ROMS/MARBL configuration files with substituted variables
    before compilation or execution.

    Workflow:
        1. Copies the contents of `config.paths.model_config` into a sibling
           `../opt_$(GRID_NAME)/` directory (creating it if necessary), excluding any
           pre-existing `../opt_$(GRID_NAME)` directory.
        2. Initializes a Jinja2 environment to process templates in the
           ../opt_$(GRID_NAME) directory.
        3. For each file listed in the `parameters` dictionary, loads it as
           a template and replaces template tokens with the provided values.
        4. Writes rendered content back to the same path, preserving file
           permissions.
        5. Prints a list of rendered files to stdout.

    Args:
        parameters (dict[str, dict[str, Any]]):
            A mapping of relative filenames (e.g. "param.opt") to dictionaries
            of template variables and their values. For example:
                {
                    "param.opt": {"NP_XI": 5, "NP_ETA": 2},
                    "river_frc.opt": {"nriv": 190},
                }

    Raises:
        FileNotFoundError: If a template file listed in `parameters` is missing
            from the staged configuration directory.
        jinja2.exceptions.UndefinedError: If a template references a variable
            not defined in its corresponding context dictionary.

    Returns:
        None
            Prints a list of successfully rendered files to the console.

    Example:
        >>> parameters = {
        ...     "param.opt": {"NP_XI": 5, "NP_ETA": 2},
        ...     "river_frc.opt": {"nriv": 190},
        ... }
        >>> render_source(parameters)
        Rendered files:
          - /path/to/model-configs/opt_$(GRID_NAME)/param.opt
          - /path/to/model-configs/opt_$(GRID_NAME)/river_frc.opt
    """

    # --- 1) stage a working copy into tmp/ ---
    src = config.paths.model_config.resolve()
    dst = (src / f"../opt_{grid_name}").resolve()

    # copy everything except an existing tmp/
    shutil.copytree(
        src, dst, dirs_exist_ok=True, ignore=shutil.ignore_patterns(f"opt_{grid_name}")
    )

    # --- 2) set up Jinja to load from tmp/ and render files in-place ---
    env = Environment(
        loader=FileSystemLoader(str(dst)),
        undefined=StrictUndefined,  # error on missing variables
        autoescape=False,  # plain text files
        keep_trailing_newline=True,
        trim_blocks=False,
        lstrip_blocks=False,
    )

    rendered = []

    for relname, context in parameters.items():
        relpath = Path(relname)
        target = dst / relpath
        if not target.exists():
            raise FileNotFoundError(f"Template not found in tmp/: {target}")

        # load by path relative to dst
        template = env.get_template(str(relpath.as_posix()))
        rendered_text = template.render(**context)

        # preserve original permissions when writing back
        try:
            orig_mode = target.stat().st_mode
        except FileNotFoundError:
            orig_mode = None

        target.write_text(rendered_text)

        if orig_mode is not None:
            target.chmod(stat.S_IMODE(orig_mode))

        rendered.append(str(target))

    print("Rendered files:")
    for f in rendered:
        print("  -", f)
