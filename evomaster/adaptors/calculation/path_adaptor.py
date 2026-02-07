"""Path adaptor: Bohrium HTTPS storage and path resolution for calculation MCP tools.

Align with _tmp/MatMaster: storage is type "https" with Bohrium plugin (access_key, project_id)
so calculation servers use Bohrium-backed HTTP storage with auth. Executor is None.

1. **storage**: {"type": "https", "plugin": {"type": "bohrium", "access_key": BOHRIUM_ACCESS_KEY, "project_id": BOHRIUM_PROJECT_ID, "app_key": "agent"}} from env.
2. **executor**: None.
3. Path args: upload local workspace files to OSS, pass https URL.
4. Map /workspace/<name> to workspace_root/<name>.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urlparse

from .oss_upload import upload_file_to_oss

logger = logging.getLogger(__name__)


def _bohrium_https_storage() -> Dict[str, Any]:
    """Build storage config for calculation MCP: type https + Bohrium plugin (same as _tmp/MatMaster)."""
    access_key = os.getenv("BOHRIUM_ACCESS_KEY", "").strip()
    try:
        project_id = int(os.getenv("BOHRIUM_PROJECT_ID", "-1"))
    except (TypeError, ValueError):
        project_id = -1
    return {
        "type": "https",
        "plugin": {
            "type": "bohrium",
            "access_key": access_key,
            "project_id": project_id,
            "app_key": "agent",
        },
    }

# Remote tool name -> list of argument names that are input file paths (upload to OSS, pass URL).
CALCULATION_PATH_ARGS: Dict[str, List[str]] = {
    "get_structure_info": ["structure_path"],
    "get_molecule_info": ["molecule_path"],
    "build_bulk_structure_by_template": [],
    "build_bulk_structure_by_wyckoff": [],
    "make_supercell_structure": ["structure_path"],
    "apply_structure_transformation": ["structure_path"],
    "build_molecule_structures_from_smiles": [],
    "add_cell_for_molecules": ["molecule_path"],
    "build_surface_slab": ["material_path"],
    "build_surface_adsorbate": ["surface_path", "adsorbate_path"],
    "build_surface_interface": ["material1_path", "material2_path"],
    "make_defect_structure": ["structure_path"],
    "make_doped_structure": ["structure_path"],
    "make_amorphous_structure": ["molecule_paths"],
    "add_hydrogens": ["structure_path"],
    "generate_ordered_replicas": ["structure_path"],
    "remove_solvents": ["structure_path"],
    "optimize_structure": ["input_structure"],
    "calculate_phonon": ["input_structure"],
    "run_molecular_dynamics": ["initial_structure"],
    "calculate_elastic_constants": ["input_structure"],
    "run_neb": ["initial_structure", "final_structure"],
    "extract_material_data_from_pdf": ["pdf_path"],
    "extract_info_from_webpage": [],
}


def _path_keys_from_schema(input_schema: Optional[Dict[str, Any]]) -> List[str]:
    """From MCP tool input_schema (JSON Schema), return param names that look path/file-related."""
    if not input_schema or not isinstance(input_schema, dict):
        return []
    props = input_schema.get("properties") or {}
    path_keywords = ("path", "file", "url", "structure", "pdf", "cif", "input_structure", "material_path", "surface_path", "adsorbate_path", "molecule_path")
    out = []
    for key, spec in props.items():
        if not isinstance(spec, dict):
            continue
        key_lower = key.lower()
        desc = (spec.get("description") or spec.get("title") or "").lower()
        if any(kw in key_lower for kw in ("path", "structure", "file", "pdf")):
            out.append(key)
        elif any(kw in desc for kw in ("path", "file", "url", "structure", "cif", "input")):
            out.append(key)
    return out


# Schema keys that are NOT input file paths (do not upload / replace with OSS URL).
_NON_PATH_SCHEMA_KEYS = frozenset({"crystal_structure", "output_file"})


def _path_arg_names_from_schema(schema: Optional[Dict[str, Any]]) -> Set[str]:
    """From MCP tool input_schema, collect property names that are input file paths (upload to OSS, pass URL)."""
    out: Set[str] = set()
    if not schema or not isinstance(schema, dict):
        return out
    props = schema.get("properties") or {}
    key_hints = ("structure_path", "molecule_path", "material_path", "surface_path", "adsorbate_path", "input_structure", "initial_structure", "final_structure", "pdf_path")
    for key, prop in props.items():
        key_lower = key.lower()
        if key_lower in _NON_PATH_SCHEMA_KEYS:
            continue
        if any(h in key_lower for h in key_hints):
            out.add(key)
            continue
        if key_lower.endswith("_path") or key_lower == "pdf_path":
            out.add(key)
        elif isinstance(prop, dict):
            desc = (prop.get("description") or prop.get("title") or "").lower()
            if "input" in desc and ("path" in desc or "file" in desc or "url" in desc):
                out.add(key)
    return out


def _is_local_path(value: Any) -> bool:
    if not value or not isinstance(value, str):
        return False
    value = value.strip()
    if not value:
        return False
    parsed = urlparse(value)
    if parsed.scheme in ("http", "https"):
        return False
    if value.lower().startswith("local://"):
        return False
    return True


def _workspace_path_to_local(value: str, workspace_root: Path) -> Path:
    """Map /workspace/... or relative path to actual local Path under workspace_root."""
    value = value.strip().replace("\\", "/")
    if value.startswith("/workspace/"):
        rel = value[len("/workspace/"):].lstrip("/")
        return (workspace_root / rel).resolve()
    if value.startswith("/workspace"):
        rel = value[len("/workspace"):].lstrip("/")
        return (workspace_root / (rel or ".")).resolve()
    path = Path(value)
    if not path.is_absolute():
        return (workspace_root / path).resolve()
    return path


def _resolve_one(value: str, workspace_root: Path) -> str:
    """If value is a local path, upload to OSS and return the OSS URL. Path args must be OSS links for remote MCP."""
    if not _is_local_path(value):
        return value
    path = _workspace_path_to_local(value, workspace_root)
    if not path.exists():
        raise FileNotFoundError(
            f"Path argument file not found: {path}. For calculation MCP tools, input files must exist in workspace so they can be uploaded to OSS and passed as URL."
        )
    if not path.is_file():
        raise ValueError(f"Path argument is not a file: {path}. Only files can be uploaded to OSS.")
    try:
        return upload_file_to_oss(path, workspace_root)
    except Exception as e:
        raise RuntimeError(
            f"Cannot pass local file to calculation MCP: OSS upload required but failed for {path}. "
            "Install oss2 (pip install oss2) and set OSS_ENDPOINT, OSS_BUCKET_NAME and OSS credentials (e.g. OSS_ACCESS_KEY_ID, OSS_ACCESS_KEY_SECRET)."
        ) from e


class CalculationPathAdaptor:
    """Bohrium HTTPS storage (executor=None) and path→OSS URL for calculation MCP tools. Matches _tmp/MatMaster."""

    def resolve_args(
        self,
        workspace_path: str,
        args: Dict[str, Any],
        tool_name: str,
        server_name: str,
        input_schema: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Inject executor=None, storage=https+Bohrium plugin (from BOHRIUM_ACCESS_KEY, BOHRIUM_PROJECT_ID); path args → OSS URL."""
        out = dict(args)
        out["executor"] = None
        out["storage"] = _bohrium_https_storage()

        remote_name = tool_name
        if server_name and tool_name.startswith(server_name + "_"):
            remote_name = tool_name[len(server_name) + 1 :]
        path_arg_names = set(CALCULATION_PATH_ARGS.get(remote_name, [])) | _path_arg_names_from_schema(input_schema)
        if not path_arg_names or not workspace_path:
            return out

        workspace_root = Path(workspace_path).resolve()
        for key in sorted(path_arg_names):
            if key not in out:
                continue
            val = out[key]
            if isinstance(val, list):
                out[key] = [_resolve_one(str(v), workspace_root) for v in val]
            else:
                out[key] = _resolve_one(str(val), workspace_root)
        return out


def get_calculation_path_adaptor() -> CalculationPathAdaptor:
    """Return a shared calculation path adaptor instance."""
    return CalculationPathAdaptor()
