"""Check project imports and hardware, then save the actual runtime details."""

import importlib
import json
import os
import platform
import shutil
import sys
from datetime import datetime, timezone

import config


LIBRARIES = ("torch", "numpy", "nibabel", "matplotlib", "monai", "scipy", "psutil")


def collect_environment():
    """Collect diagnostics without assuming that a GPU or every library exists."""
    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": config.PROJECT_RUNTIME,
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "project_root": str(config.PROJECT_ROOT),
        "libraries": {},
        "memory": None,
        "disks": {},
        "cpu_tensor_check": "not_run",
        "gpu": {"cuda_available": False, "devices": [], "tensor_check": "not_run"},
        "errors": [],
    }
    modules = {}
    for name in LIBRARIES:
        try:
            module = importlib.import_module(name)
        except (ImportError, OSError, RuntimeError) as error:
            report["libraries"][name] = {"status": "error", "error": str(error)}
            report["errors"].append(f"{name}: {type(error).__name__}: {error}")
        else:
            modules[name] = module
            report["libraries"][name] = {
                "status": "ok", "version": str(module.__version__)
            }

    if "psutil" in modules:
        try:
            memory = modules["psutil"].virtual_memory()
            report["memory"] = {
                "total_gib": round(memory.total / 1024 ** 3, 2),
                "available_gib": round(memory.available / 1024 ** 3, 2),
            }
        except OSError as error:
            report["errors"].append(f"Memory check: {error}")

    for name, folder in (("project", config.PROJECT_ROOT), ("runtime", config.RUNTIME_ROOT)):
        disk_path = folder if folder.exists() else folder.parent
        try:
            disk = shutil.disk_usage(disk_path)
            report["disks"][name] = {
                "path": str(disk_path),
                "total_gib": round(disk.total / 1024 ** 3, 2),
                "free_gib": round(disk.free / 1024 ** 3, 2),
            }
        except OSError as error:
            report["errors"].append(f"Disk check for {folder}: {error}")

    if "torch" in modules:
        torch = modules["torch"]
        try:
            value = torch.tensor([1.0, 2.0]).sum().item()
            if value != 3.0:
                raise RuntimeError(f"Unexpected CPU tensor result: {value}")
            report["cpu_tensor_check"] = "passed"
            report["gpu"]["cuda_available"] = torch.cuda.is_available()
            report["gpu"]["torch_cuda_build"] = torch.version.cuda
            if torch.cuda.is_available():
                for index in range(torch.cuda.device_count()):
                    device = torch.cuda.get_device_properties(index)
                    report["gpu"]["devices"].append({
                        "name": device.name,
                        "total_gib": round(device.total_memory / 1024 ** 3, 2),
                    })
                value = torch.tensor([1.0, 2.0], device="cuda").sum().item()
                if value != 3.0:
                    raise RuntimeError(f"Unexpected CUDA tensor result: {value}")
                report["gpu"]["tensor_check"] = "passed"
            else:
                report["gpu"]["tensor_check"] = "not_available"
        except (OSError, RuntimeError) as error:
            report["errors"].append(f"PyTorch tensor/device check: {error}")

    report["checks_passed"] = (
        not report["errors"] and report["cpu_tensor_check"] == "passed"
    )
    return report


def run_checks():
    """Print useful results and save an environment snapshot for this runtime."""
    config.prepare_folders()
    report = collect_environment()
    output_path = config.RESULTS_DIR / f"environment_{report['runtime']}.json"
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"Runtime: {report['runtime']} | Python: {report['python']}")
    print(f"Project: {config.PROJECT_ROOT}")
    for name, details in report["libraries"].items():
        print(f"{name}: {details.get('version', details.get('error'))}")
    if report["memory"]:
        print(f"Memory (GiB): {report['memory']}")
    for name, details in report["disks"].items():
        print(f"{name} disk free: {details['free_gib']} GiB")
    print(f"CPU tensor check: {report['cpu_tensor_check']}")
    print(f"CUDA available: {report['gpu']['cuda_available']}")
    for device in report["gpu"]["devices"]:
        print(f"GPU: {device['name']} ({device['total_gib']} GiB)")
    if not report["gpu"]["cuda_available"]:
        print("CPU setup can be checked here. GPU execution still needs a GPU runtime.")
    for error in report["errors"]:
        print(f"ERROR: {error}")
    print(f"Environment checks passed: {report['checks_passed']}")
    print(f"Saved: {output_path}")
    return report


if __name__ == "__main__":
    result = run_checks()
    raise SystemExit(0 if result["checks_passed"] else 1)
