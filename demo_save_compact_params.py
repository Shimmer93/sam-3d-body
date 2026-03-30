"""
Compact-parameter export demo for SAM 3D Body.

This script mirrors the inference flow of `demo.py`, but instead of rendering and
saving visualization images, it saves a compact `.npz` package for each input
image. The goal is to keep only the most useful per-person parameters needed for
later downstream processing while avoiding large mesh and keypoint outputs.

Why these fields are saved
--------------------------
For each image, this script saves a compact set of outputs that are useful for
later reconstruction, rendering, or reprojection:

- `mhr_model_params`:
    Final refined MHR model parameter vector for each detected person. This is
    the compact parameterization consumed by the MHR body model.

- `shape_params`:
    Predicted body shape latent parameters for each detected person. These are
    required in addition to `mhr_model_params` to reconstruct the final body.

- `pred_cam_t`:
    Predicted camera translation `[tx, ty, tz]` for each detected person. This is
    useful for later rendering or reprojection.

- `cam_int`:
    Full camera intrinsics matrix for the image. This is a better design than
    saving only a scalar focal length because it preserves both focal length(s)
    and principal point. If you need the old scalar focal length, it can be
    recovered as `cam_int[0, 0]`.

- `image_size`:
    Original image size stored as `[height, width]`. This lets downstream code
    interpret the camera parameters in the correct image coordinate system.

What is intentionally NOT saved
-------------------------------
- `pred_pose_raw` is intentionally not saved. In this repository's inference path,
  the model performs a final refinement / re-run step and the raw pose token-space
  representation is explicitly no longer treated as the final valid refined
  representation. The compact refined representation to keep is
  `mhr_model_params`, not `pred_pose_raw`.

Saved file format
-----------------
One compressed `.npz` file is written per input image. The filename matches the
input image basename, e.g. `example.jpg` -> `example.npz`.

Each `.npz` contains exactly these arrays:

1. `mhr_model_params`
   - Meaning: final refined MHR parameter vector per detected person
   - Type: `np.ndarray`
   - Shape: `(N, 204)`
   - Dtype: `float32`

2. `shape_params`
   - Meaning: body shape latent parameters per detected person
   - Type: `np.ndarray`
   - Shape: `(N, 45)`
   - Dtype: `float32`

3. `pred_cam_t`
   - Meaning: predicted camera translation per detected person
   - Type: `np.ndarray`
   - Shape: `(N, 3)`
   - Dtype: `float32`

4. `cam_int`
   - Meaning: full camera intrinsics matrix used during inference
   - Type: `np.ndarray`
   - Shape: `(3, 3)`
   - Dtype: `float32`

5. `image_size`
   - Meaning: original input image size as `[height, width]`
   - Type: `np.ndarray`
   - Shape: `(2,)`
   - Dtype: `int32`

Here, `N` is the number of detected people in the image.

No-detection behavior
---------------------
If no people are detected, the script still writes a valid `.npz` file with:
- `mhr_model_params`: shape `(0, 204)`
- `shape_params`: shape `(0, 45)`
- `pred_cam_t`: shape `(0, 3)`
- `cam_int`: shape `(3, 3)`
- `image_size`: shape `(2,)`

Camera-intrinsics behavior
--------------------------
This script computes and saves the exact `cam_int` used during inference:

- If a FOV estimator is enabled, `cam_int` is predicted from the input image.
- Otherwise, it falls back to the repository's default camera intrinsics:
    fx = fy = sqrt(height^2 + width^2)
    cx = width / 2
    cy = height / 2

The same `cam_int` is then passed into `process_one_image(...)`, ensuring the
saved camera intrinsics match the camera intrinsics actually used by the model.
"""

import argparse
import os
from pathlib import Path
import sys
from glob import glob
from typing import Any

import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml", ".sl"],
    pythonpath=True,
    dotenv=True,
)

import cv2
import numpy as np
import torch
from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body
from tqdm import tqdm
import trimesh


def load_image_rgb_and_size(image_path: str):
    """
    Load an image with OpenCV, convert it to RGB, and return the original image
    size as `[height, width]`.
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    height, width = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    image_size = np.asarray([height, width], dtype=np.int32)
    return img_rgb, image_size


def build_default_cam_int(image_size: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    Build the same default camera intrinsics matrix used by `prepare_batch(...)`
    when no explicit camera intrinsics or FOV estimator output are provided.
    """
    height = float(image_size[0])
    width = float(image_size[1])
    focal = float((height**2 + width**2) ** 0.5)

    cam_int = torch.tensor(
        [
            [
                [focal, 0.0, width / 2.0],
                [0.0, focal, height / 2.0],
                [0.0, 0.0, 1.0],
            ]
        ],
        dtype=torch.float32,
        device=device,
    )
    return cam_int


def resolve_cam_intrinsics(
    image_path: str,
    image_rgb: np.ndarray,
    image_size: np.ndarray,
    fov_estimator,
    device: torch.device,
) -> torch.Tensor:
    """
    Resolve the camera intrinsics that will be used for inference.

    If a FOV estimator is available, use it to predict camera intrinsics from the
    RGB image. Otherwise, reproduce the repository's default intrinsics from the
    image size.
    """
    if fov_estimator is not None:
        cam_int = fov_estimator.get_cam_intrinsics(image_rgb)
        if not torch.is_tensor(cam_int):
            cam_int = torch.as_tensor(cam_int, dtype=torch.float32, device=device)
        else:
            cam_int = cam_int.to(device=device, dtype=torch.float32)
    else:
        cam_int = build_default_cam_int(image_size, device)

    if cam_int.ndim == 2:
        cam_int = cam_int.unsqueeze(0)

    if tuple(cam_int.shape) != (1, 3, 3):
        raise ValueError(
            f"Unexpected camera intrinsics shape for image '{image_path}': "
            f"got {tuple(cam_int.shape)}, expected (1, 3, 3)"
        )

    return cam_int


def validate_conversion_args(args) -> None:
    """Validate CLI arguments for optional SMPL/SMPL-X export."""
    if args.export_smpl_params:
        if args.smpl_model_path == "":
            raise ValueError(
                "--smpl_model_path is required when --export_smpl_params is set"
            )
        if not os.path.exists(args.smpl_model_path):
            raise FileNotFoundError(
                f"SMPL model path does not exist: {args.smpl_model_path}"
            )

    if args.export_smplx_params:
        if args.smplx_model_path == "":
            raise ValueError(
                "--smplx_model_path is required when --export_smplx_params is set"
            )
        if not os.path.exists(args.smplx_model_path):
            raise FileNotFoundError(
                f"SMPL-X model path does not exist: {args.smplx_model_path}"
            )


def find_local_mhr_repo() -> tuple[Path | None, Path | None]:
    """Locate a local MHR checkout plus its SMPL conversion tool directory."""
    repo_root = Path(root)
    candidate_roots = []

    env_override = os.environ.get("MHR_REPO_PATH", "").strip()
    if env_override:
        candidate_roots.append(Path(env_override).expanduser())

    # `pyrootutils` resolves to the nearest project root, which for this script is
    # usually `external/sam-3d-body` rather than the top-level repo. Search both
    # that subtree and a few parent-level layouts.
    search_roots = [repo_root, *list(repo_root.parents[:3])]
    for search_root in search_roots:
        candidate_roots.extend(
            [
                search_root / "external" / "MHR",
                search_root / "external" / "mhr",
            ]
        )

    seen_roots = set()
    deduped_candidate_roots = []
    for candidate_root in candidate_roots:
        resolved = candidate_root.expanduser().resolve(strict=False)
        if resolved in seen_roots:
            continue
        seen_roots.add(resolved)
        deduped_candidate_roots.append(candidate_root)

    for candidate_root in deduped_candidate_roots:
        tool_dir = candidate_root / "tools" / "mhr_smpl_conversion"
        if (candidate_root / "mhr").is_dir() and tool_dir.is_dir():
            return candidate_root, tool_dir

    return None, None


def register_local_mhr_conversion_paths() -> Path | None:
    """
    Expose the official MHR repo and its conversion tool on `sys.path` when present.

    The conversion code in `tools/mhr_smpl_conversion` is not packaged as a normal
    installable module, so register both the repo root and the tool directory.
    """
    mhr_repo_dir, conversion_tool_dir = find_local_mhr_repo()
    if conversion_tool_dir is None:
        return None

    for path in (mhr_repo_dir, conversion_tool_dir):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

    return conversion_tool_dir


def import_smpl_conversion_dependencies():
    """
    Import the optional MHR->SMPL conversion stack only when it is requested.

    The implementation follows the official MHR conversion tool documented at:
    https://github.com/facebookresearch/MHR/blob/main/tools/mhr_smpl_conversion/README.md
    """
    conversion_tool_dir = register_local_mhr_conversion_paths()
    missing_packages = []

    try:
        import smplx
    except ImportError:
        smplx = None
        missing_packages.append("smplx")

    try:
        from mhr.mhr import MHR
    except ImportError:
        MHR = None
        missing_packages.append("mhr")

    Conversion = None
    try:
        from smpl_mhr import Conversion
    except ImportError:
        try:
            from conversion import Conversion
        except ImportError:
            try:
                from tools.mhr_smpl_conversion.conversion import Conversion
            except ImportError:
                Conversion = None

    if missing_packages or Conversion is None:
        details = []
        if missing_packages:
            details.append(f"missing packages: {', '.join(sorted(missing_packages))}")
        if Conversion is None:
            details.append("missing conversion module: smpl_mhr")
        if conversion_tool_dir is None:
            details.append(
                "MHR conversion repo not found; initialize `external/MHR` or set `MHR_REPO_PATH`"
            )
        raise ImportError(
            "Optional SMPL/SMPL-X export requires the MHR conversion stack from "
            "https://github.com/facebookresearch/MHR/blob/main/tools/mhr_smpl_conversion/README.md; "
            + "; ".join(details)
        )

    return smplx, MHR, Conversion


def import_direct_smpl_export_dependencies():
    """
    Import the minimal dependencies for direct SAM3D->SMPL export.

    Unlike the full MHR conversion stack, this path avoids `MHR.from_files(...)`
    and fits SMPL directly from SAM3D-predicted MHR-space vertices.
    """
    register_local_mhr_conversion_paths()

    try:
        import smplx
    except ImportError as exc:
        raise ImportError(
            "Optional SMPL export requires `smplx` to be installed."
        ) from exc

    try:
        from pytorch_fitting import PyTorchSMPLFitting
    except ImportError as exc:
        raise ImportError(
            "Optional SMPL export requires the local `external/MHR/tools/mhr_smpl_conversion` "
            "tooling to be importable."
        ) from exc

    return smplx, PyTorchSMPLFitting


def load_mhr_to_smpl_surface_mapping() -> tuple[np.ndarray, np.ndarray]:
    """Load the precomputed MHR->SMPL barycentric surface mapping."""
    mapping_path = (
        Path(root).resolve() / ".." / "MHR" / "tools" / "mhr_smpl_conversion" / "assets" / "mhr2smpl_mapping.npz"
    ).resolve()
    if not mapping_path.exists():
        raise FileNotFoundError(f"MHR->SMPL mapping file not found: {mapping_path}")

    mapping = np.load(mapping_path)
    return mapping["triangle_ids"], mapping["baryc_coords"]


def build_direct_smpl_exporter(
    args, smplx_module, pytorch_smpl_fitting_cls, device: torch.device
) -> dict[str, Any]:
    """Build a direct SAM3D vertex -> SMPL exporter without `MHR.from_files(...)`."""
    scripted_mhr_model = torch.jit.load(args.mhr_path, map_location="cpu")
    mhr_faces = scripted_mhr_model.character_torch.mesh.faces.detach().cpu().numpy()
    del scripted_mhr_model

    mapped_face_id, baryc_coords = load_mhr_to_smpl_surface_mapping()

    try:
        smpl_model = smplx_module.SMPL(
            model_path=args.smpl_model_path,
            gender=args.smpl_gender,
        )
    except ModuleNotFoundError as exc:
        if exc.name == "chumpy":
            raise ModuleNotFoundError(
                "Loading the provided SMPL `.pkl` file requires the `chumpy` package. "
                "Install `chumpy`, or switch `--smpl_model_path` to an official SMPL `.npz` "
                "model file."
            ) from exc
        raise
    smpl_model = smpl_model.to(str(device))
    smpl_template_mesh = trimesh.Trimesh(
        smpl_model.v_template.detach().cpu().numpy(),
        smpl_model.faces,
        process=False,
    )
    smpl_edges = torch.from_numpy(smpl_template_mesh.edges_unique.copy()).long()

    return {
        "kind": "smpl",
        "prefix": "smpl",
        "mode": "direct_vertices_to_smpl",
        "smpl_model": smpl_model,
        "smpl_edges": smpl_edges,
        "smpl_model_type": "smpl",
        "hand_pose_dim": 0,
        "mhr_faces": mhr_faces,
        "mapped_face_id": mapped_face_id,
        "baryc_coords": baryc_coords,
        "solver_cls": pytorch_smpl_fitting_cls,
        "device": str(device),
        "batch_size": args.smpl_conversion_batch_size,
        "empty_parameters": build_empty_smpl_parameter_template("smpl", smpl_model),
    }


def find_mhr_assets_dir(args) -> Path | None:
    """Locate the full MHR asset bundle required by `MHR.from_files(...)`."""
    if getattr(args, "mhr_path", ""):
        mhr_path = Path(args.mhr_path).expanduser()
        # `--mhr_path` points to `.../assets/mhr_model.pt` in the common setup.
        if mhr_path.suffix:
            candidate_dirs = [mhr_path.parent, mhr_path.parent / "assets"]
        else:
            candidate_dirs = [mhr_path, mhr_path / "assets"]
    else:
        candidate_dirs = []

        env_override = os.environ.get("MHR_ASSET_PATH", "").strip()
        if env_override:
            candidate_dirs.append(Path(env_override).expanduser())

        repo_root = Path(root)
        search_roots = [repo_root, *list(repo_root.parents[:3])]
        checkpoint_path = getattr(args, "checkpoint_path", "")
        if checkpoint_path:
            checkpoint_dir = Path(checkpoint_path).expanduser().parent
            search_roots = [checkpoint_dir, checkpoint_dir.parent, *search_roots]

        for search_root in search_roots:
            candidate_dirs.extend(
                [
                    search_root / "external" / "MHR" / "assets",
                    search_root / "external" / "mhr" / "assets",
                    search_root / "assets",
                ]
            )

    seen_dirs = set()
    required_files = (
        "lod1.fbx",
        "compact_v6_1.model",
        "corrective_activation.npz",
        "corrective_blendshapes_lod1.npz",
    )
    for candidate_dir in candidate_dirs:
        resolved = candidate_dir.expanduser().resolve(strict=False)
        if resolved in seen_dirs:
            continue
        seen_dirs.add(resolved)
        if all((candidate_dir / filename).exists() for filename in required_files):
            return candidate_dir

    return None


def build_empty_smpl_parameter_template(
    model_kind: str, smpl_model: Any
) -> dict[str, np.ndarray]:
    """Build empty parameter arrays for no-detection outputs."""
    template = {
        "betas": np.empty((0, int(smpl_model.num_betas)), dtype=np.float32),
        "body_pose": np.empty(
            (0, 69 if model_kind == "smpl" else 63), dtype=np.float32
        ),
        "global_orient": np.empty((0, 3), dtype=np.float32),
        "transl": np.empty((0, 3), dtype=np.float32),
    }

    if model_kind == "smplx":
        hand_pose_dim = 6 if bool(getattr(smpl_model, "use_pca", False)) else 45
        expression_dim = int(getattr(smpl_model, "num_expression_coeffs", 10))
        template.update(
            {
                "left_hand_pose": np.empty((0, hand_pose_dim), dtype=np.float32),
                "right_hand_pose": np.empty((0, hand_pose_dim), dtype=np.float32),
                "expression": np.empty((0, expression_dim), dtype=np.float32),
                "jaw_pose": np.empty((0, 3), dtype=np.float32),
                "leye_pose": np.empty((0, 3), dtype=np.float32),
                "reye_pose": np.empty((0, 3), dtype=np.float32),
            }
        )

    return template


def build_optional_smpl_exporters(args, device: torch.device) -> list[dict[str, Any]]:
    """Create the optional SMPL and/or SMPL-X converters requested by the user."""
    validate_conversion_args(args)

    if not args.export_smpl_params and not args.export_smplx_params:
        return []

    exporters = []

    if args.export_smpl_params:
        smplx_module, pytorch_smpl_fitting_cls = import_direct_smpl_export_dependencies()
        exporters.append(
            build_direct_smpl_exporter(
                args=args,
                smplx_module=smplx_module,
                pytorch_smpl_fitting_cls=pytorch_smpl_fitting_cls,
                device=device,
            )
        )

    if args.export_smplx_params:
        smplx, MHR, Conversion = import_smpl_conversion_dependencies()
        mhr_assets_dir = find_mhr_assets_dir(args)
        if mhr_assets_dir is None:
            if getattr(args, "mhr_path", ""):
                mhr_path = Path(args.mhr_path).expanduser()
                expected_dir = mhr_path.parent if mhr_path.suffix else mhr_path
                raise FileNotFoundError(
                    "Optional SMPL-X export requires the full MHR asset bundle in "
                    f"`{expected_dir}`. Expected files such as `lod1.fbx`, "
                    "`compact_v6_1.model`, `corrective_activation.npz`, and "
                    "`corrective_blendshapes_lod1.npz` next to the provided "
                    "`--mhr_path`."
                )
            raise FileNotFoundError(
                "Optional SMPL-X export requires the full MHR asset bundle. "
                "Expected files such as `lod1.fbx`, `compact_v6_1.model`, "
                "`corrective_activation.npz`, and `corrective_blendshapes_lod1.npz`."
            )

        try:
            mhr_model = MHR.from_files(folder=mhr_assets_dir, device=device, lod=1)
        except Exception as exc:
            raise RuntimeError(
                "Failed to initialize `mhr.MHR.from_files()` for optional SMPL-X export"
            ) from exc

        smplx_model = smplx.SMPLX(
            model_path=args.smplx_model_path,
            gender=args.smplx_gender,
        )
        exporters.append(
            {
                "kind": "smplx",
                "prefix": "smplx",
                "converter": Conversion(
                    mhr_model=mhr_model,
                    smpl_model=smplx_model,
                    method=args.smpl_conversion_method,
                    batch_size=args.smpl_conversion_batch_size,
                ),
                "batch_size": args.smpl_conversion_batch_size,
                "empty_parameters": build_empty_smpl_parameter_template(
                    "smplx", smplx_model
                ),
            }
        )

    return exporters


def to_numpy_float32(value: Any) -> np.ndarray:
    """Convert tensors / arrays to `np.float32` arrays."""
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    else:
        value = np.asarray(value)
    return value.astype(np.float32, copy=False)


def complete_optional_parameter_dict(
    parameters: dict[str, Any], empty_template: dict[str, np.ndarray], num_people: int
) -> dict[str, np.ndarray]:
    """
    Normalize converter outputs to a stable numpy schema.

    The current MHR conversion tool returns the core fitted parameters. For SMPL-X,
    we also zero-fill optional jaw/eye pose entries so that all exported `.npz`
    files keep the same schema, including the no-detection case.
    """
    normalized = {
        key: to_numpy_float32(value)
        for key, value in parameters.items()
        if value is not None
    }

    for key, empty_value in empty_template.items():
        if key not in normalized:
            normalized[key] = np.zeros(
                (num_people, *empty_value.shape[1:]), dtype=np.float32
            )

    return normalized


def prefix_parameter_dict(prefix: str, parameters: dict[str, np.ndarray]) -> dict:
    """Prefix parameter keys so SMPL and SMPL-X can coexist in one `.npz`."""
    return {f"{prefix}_{key}": value for key, value in parameters.items()}


def run_sam3d_to_smpl_conversion(
    outputs, exporter: dict[str, Any]
) -> dict[str, np.ndarray]:
    """Run one optional MHR->SMPL(X) conversion pass and package its arrays."""
    if len(outputs) == 0:
        return prefix_parameter_dict(
            exporter["prefix"], exporter["empty_parameters"].copy()
        )

    if exporter.get("mode") == "direct_vertices_to_smpl":
        results_parameters = run_direct_vertices_to_smpl_conversion(outputs, exporter)
        completed_parameters = complete_optional_parameter_dict(
            parameters=results_parameters,
            empty_template=exporter["empty_parameters"],
            num_people=len(outputs),
        )
        return prefix_parameter_dict(exporter["prefix"], completed_parameters)

    convert_fn = getattr(exporter["converter"], "convert_sam3d_output_to_smpl", None)
    if convert_fn is None and exporter["kind"] == "smplx":
        convert_fn = getattr(exporter["converter"], "convert_sam3d_output_to_smplx", None)

    if convert_fn is None:
        raise AttributeError(
            "The installed MHR conversion package does not expose "
            "`convert_sam3d_output_to_smpl(...)`"
        )

    results = convert_fn(
        sam3d_outputs=outputs,
        return_smpl_meshes=False,
        return_smpl_parameters=True,
        return_smpl_vertices=False,
        return_fitting_errors=False,
        batch_size=exporter["batch_size"],
    )

    if results.result_parameters is None:
        raise RuntimeError("The SMPL conversion finished without returning parameters")

    completed_parameters = complete_optional_parameter_dict(
        parameters=results.result_parameters,
        empty_template=exporter["empty_parameters"],
        num_people=len(outputs),
    )
    return prefix_parameter_dict(exporter["prefix"], completed_parameters)


def run_direct_vertices_to_smpl_conversion(
    outputs, exporter: dict[str, Any]
) -> dict[str, Any]:
    """Fit SMPL directly from SAM3D-predicted MHR-space vertices."""
    pred_vertices = np.stack(
        [np.asarray(person_output["pred_vertices"], dtype=np.float32) for person_output in outputs],
        axis=0,
    )
    pred_cam_t = np.stack(
        [np.asarray(person_output["pred_cam_t"], dtype=np.float32) for person_output in outputs],
        axis=0,
    )

    device = torch.device(exporter["device"])
    # SAM3D `pred_vertices` are in meters and camera-relative. Convert to the MHR
    # world-space convention used by the official conversion code.
    mhr_vertices = torch.from_numpy(
        100.0 * pred_vertices + 100.0 * pred_cam_t[:, None, :]
    ).to(device=device, dtype=torch.float32)

    source_vertices = mhr_vertices * 0.01  # centimeters -> meters
    mapped_face_id = torch.from_numpy(exporter["mapped_face_id"]).long().to(device)
    baryc_coords = (
        torch.from_numpy(exporter["baryc_coords"]).to(device=device, dtype=torch.float32)[
            None, :, :, None
        ]
    )
    source_faces = torch.from_numpy(exporter["mhr_faces"]).long().to(device)

    triangles = source_vertices[:, source_faces[mapped_face_id], :]
    target_vertices = (triangles * baryc_coords).sum(dim=2)

    solver = exporter["solver_cls"](
        smpl_model=exporter["smpl_model"],
        smpl_edges=exporter["smpl_edges"],
        smpl_model_type=exporter["smpl_model_type"],
        hand_pose_dim=exporter["hand_pose_dim"],
        device=exporter["device"],
        batch_size=exporter["batch_size"],
    )
    return solver.fit(
        target_vertices=target_vertices,
        single_identity=False,
        is_tracking=False,
    )


def package_optional_smpl_outputs(outputs, exporters) -> dict:
    """Collect all optional SMPL / SMPL-X exports requested for this image."""
    packaged = {}
    for exporter in exporters:
        packaged.update(run_sam3d_to_smpl_conversion(outputs, exporter))
    return packaged


def empty_compact_result(image_size: np.ndarray, cam_int: torch.Tensor) -> dict:
    """Create an empty compact result package for images with no detections."""
    cam_int_np = cam_int.detach().cpu().numpy().astype(np.float32)
    if cam_int_np.shape[0] == 1:
        cam_int_np = cam_int_np[0]

    return {
        "mhr_model_params": np.empty((0, 204), dtype=np.float32),
        "shape_params": np.empty((0, 45), dtype=np.float32),
        "pred_cam_t": np.empty((0, 3), dtype=np.float32),
        "cam_int": cam_int_np,
        "image_size": image_size.astype(np.int32, copy=False),
    }


def stack_person_field(outputs, key: str, expected_dim: int) -> np.ndarray:
    """
    Stack one per-person field from the estimator outputs.

    Args:
        outputs: list of per-person dictionaries returned by `process_one_image`
        key: output dictionary key to collect
        expected_dim: expected trailing dimension for validation

    Returns:
        Array of shape `(N, expected_dim)` and dtype `float32`.
    """
    stacked = np.stack(
        [np.asarray(person_output[key], dtype=np.float32) for person_output in outputs],
        axis=0,
    )

    if stacked.ndim != 2 or stacked.shape[1] != expected_dim:
        raise ValueError(
            f"Unexpected shape for '{key}': got {stacked.shape}, "
            f"expected (N, {expected_dim})"
        )

    return stacked


def package_compact_outputs(
    outputs, image_size: np.ndarray, cam_int: torch.Tensor
) -> dict:
    """
    Convert estimator outputs into the requested compact `.npz` payload.

    Returns a dictionary containing exactly the arrays that will be written to disk.
    """
    cam_int_np = cam_int.detach().cpu().numpy().astype(np.float32)
    if cam_int_np.shape[0] == 1:
        cam_int_np = cam_int_np[0]

    if cam_int_np.shape != (3, 3):
        raise ValueError(
            f"Unexpected shape for 'cam_int': got {cam_int_np.shape}, expected (3, 3)"
        )

    if len(outputs) == 0:
        return empty_compact_result(image_size, cam_int)

    mhr_model_params = stack_person_field(outputs, "mhr_model_params", 204)
    shape_params = stack_person_field(outputs, "shape_params", 45)
    pred_cam_t = stack_person_field(outputs, "pred_cam_t", 3)

    return {
        "mhr_model_params": mhr_model_params,
        "shape_params": shape_params,
        "pred_cam_t": pred_cam_t,
        "cam_int": cam_int_np,
        "image_size": image_size.astype(np.int32, copy=False),
    }


def save_compact_npz(save_path: str, packaged_outputs: dict) -> None:
    """Save the compact result package to a compressed `.npz` file."""
    np.savez_compressed(save_path, **packaged_outputs)


def main(args):
    if args.output_folder == "":
        image_folder_name = os.path.basename(os.path.normpath(args.image_folder))
        output_folder = os.path.join("./output_params", image_folder_name)
    else:
        output_folder = args.output_folder

    os.makedirs(output_folder, exist_ok=True)

    # Use command-line args or environment variables
    mhr_path = args.mhr_path or os.environ.get("SAM3D_MHR_PATH", "")
    detector_path = args.detector_path or os.environ.get("SAM3D_DETECTOR_PATH", "")
    segmentor_path = args.segmentor_path or os.environ.get("SAM3D_SEGMENTOR_PATH", "")
    fov_path = args.fov_path or os.environ.get("SAM3D_FOV_PATH", "")

    # Initialize sam-3d-body model and optional modules
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model, model_cfg = load_sam_3d_body(
        args.checkpoint_path, device=device, mhr_path=mhr_path
    )

    human_detector, human_segmentor, fov_estimator = None, None, None
    if args.detector_name:
        from tools.build_detector import HumanDetector

        human_detector = HumanDetector(
            name=args.detector_name, device=device, path=detector_path
        )

    if (
        args.segmentor_name == "sam2" and len(segmentor_path)
    ) or args.segmentor_name != "sam2":
        from tools.build_sam import HumanSegmentor

        human_segmentor = HumanSegmentor(
            name=args.segmentor_name, device=device, path=segmentor_path
        )

    if args.fov_name:
        from tools.build_fov_estimator import FOVEstimator

        fov_estimator = FOVEstimator(name=args.fov_name, device=device, path=fov_path)

    estimator = SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=human_detector,
        human_segmentor=human_segmentor,
        fov_estimator=fov_estimator,
    )
    smpl_exporters = build_optional_smpl_exporters(args, device)

    image_extensions = [
        "*.jpg",
        "*.jpeg",
        "*.png",
        "*.gif",
        "*.bmp",
        "*.tiff",
        "*.webp",
    ]
    images_list = sorted(
        [
            image
            for ext in image_extensions
            for image in glob(os.path.join(args.image_folder, ext))
        ]
    )

    for image_path in tqdm(images_list):
        image_rgb, image_size = load_image_rgb_and_size(image_path)
        cam_int = resolve_cam_intrinsics(
            image_path=image_path,
            image_rgb=image_rgb,
            image_size=image_size,
            fov_estimator=fov_estimator,
            device=device,
        )

        outputs = estimator.process_one_image(
            image_path,
            cam_int=cam_int,
            bbox_thr=args.bbox_thresh,
            use_mask=args.use_mask,
        )

        packaged_outputs = package_compact_outputs(outputs, image_size, cam_int)
        packaged_outputs.update(package_optional_smpl_outputs(outputs, smpl_exporters))

        image_name = os.path.splitext(os.path.basename(image_path))[0]
        save_path = os.path.join(output_folder, f"{image_name}.npz")
        save_compact_npz(save_path, packaged_outputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SAM 3D Body Demo - Save compact per-image parameter packages",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
                Examples:
                python demo_save_compact_params.py --image_folder ./images --checkpoint_path ./checkpoints/model.ckpt

                Output:
                - One compressed `.npz` file per input image.
                - Each file contains:
                    * mhr_model_params (N, 204)
                    * shape_params (N, 45)
                    * pred_cam_t (N, 3)
                    * cam_int (3, 3)
                    * image_size (2,)
                    * optionally, `smpl_*` arrays when `--export_smpl_params` is used
                    * optionally, `smplx_*` arrays when `--export_smplx_params` is used

                Notes:
                - `cam_int` replaces the previous scalar focal-length export.
                - If you need a scalar focal length later, you can recover it as
                  `cam_int[0, 0]`.
                - Optional SMPL/SMPL-X export follows the MHR conversion tool:
                  https://github.com/facebookresearch/MHR/blob/main/tools/mhr_smpl_conversion/README.md

                Environment Variables:
                SAM3D_MHR_PATH: Path to MHR asset
                SAM3D_DETECTOR_PATH: Path to human detection model folder
                SAM3D_SEGMENTOR_PATH: Path to human segmentation model folder
                SAM3D_FOV_PATH: Path to fov estimation model folder
                """,
    )
    parser.add_argument(
        "--image_folder",
        required=True,
        type=str,
        help="Path to folder containing input images",
    )
    parser.add_argument(
        "--output_folder",
        default="",
        type=str,
        help="Path to output folder (default: ./output_params/<image_folder_name>)",
    )
    parser.add_argument(
        "--checkpoint_path",
        required=True,
        type=str,
        help="Path to SAM 3D Body model checkpoint",
    )
    parser.add_argument(
        "--detector_name",
        default="vitdet",
        type=str,
        help="Human detection model for demo (Default `vitdet`, add your favorite detector if needed).",
    )
    parser.add_argument(
        "--segmentor_name",
        default="sam2",
        type=str,
        help="Human segmentation model for demo (Default `sam2`, add your favorite segmentor if needed).",
    )
    parser.add_argument(
        "--fov_name",
        default="moge2",
        type=str,
        help="FOV estimation model for demo (Default `moge2`, add your favorite fov estimator if needed).",
    )
    parser.add_argument(
        "--detector_path",
        default="",
        type=str,
        help="Path to human detection model folder (or set SAM3D_DETECTOR_PATH)",
    )
    parser.add_argument(
        "--segmentor_path",
        default="",
        type=str,
        help="Path to human segmentation model folder (or set SAM3D_SEGMENTOR_PATH)",
    )
    parser.add_argument(
        "--fov_path",
        default="",
        type=str,
        help="Path to fov estimation model folder (or set SAM3D_FOV_PATH)",
    )
    parser.add_argument(
        "--mhr_path",
        default="",
        type=str,
        help="Path to MoHR/assets folder (or set SAM3D_MHR_PATH)",
    )
    parser.add_argument(
        "--bbox_thresh",
        default=0.8,
        type=float,
        help="Bounding box detection threshold",
    )
    parser.add_argument(
        "--use_mask",
        action="store_true",
        default=False,
        help="Use mask-conditioned prediction (segmentation mask is automatically generated from bbox)",
    )
    parser.add_argument(
        "--export_smpl_params",
        action="store_true",
        default=False,
        help="Also export fitted SMPL parameters using the optional MHR conversion tool",
    )
    parser.add_argument(
        "--smpl_model_path",
        default="",
        type=str,
        help="Path to the official SMPL model file or folder used for optional SMPL export",
    )
    parser.add_argument(
        "--smpl_gender",
        default="neutral",
        type=str,
        help="Gender passed to the optional SMPL model loader (default: neutral)",
    )
    parser.add_argument(
        "--export_smplx_params",
        action="store_true",
        default=False,
        help="Also export fitted SMPL-X parameters using the optional MHR conversion tool",
    )
    parser.add_argument(
        "--smplx_model_path",
        default="",
        type=str,
        help="Path to the official SMPL-X model file or folder used for optional SMPL-X export",
    )
    parser.add_argument(
        "--smplx_gender",
        default="neutral",
        type=str,
        help="Gender passed to the optional SMPL-X model loader (default: neutral)",
    )
    parser.add_argument(
        "--smpl_conversion_method",
        default="pytorch",
        type=str,
        choices=["pytorch", "pymomentum"],
        help="Optimization backend for optional SMPL/SMPL-X export",
    )
    parser.add_argument(
        "--smpl_conversion_batch_size",
        default=256,
        type=int,
        help="Batch size passed to the optional MHR SMPL conversion tool",
    )
    args = parser.parse_args()

    main(args)
