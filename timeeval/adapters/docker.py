import json
import multiprocessing
import subprocess
import sys
from dataclasses import dataclass, asdict, field
from enum import Enum
from pathlib import Path, WindowsPath, PosixPath
from typing import Optional, Any, Callable, Final, Tuple

import docker
import numpy as np
import psutil
import requests
from docker.models.containers import Container
from durations import Duration

from .base import Adapter, AlgorithmParameter

DATASET_TARGET_PATH = "/data"
RESULTS_TARGET_PATH = "/results"
SCORES_FILE_NAME = "docker-algorithm-scores.csv"
MODEL_FILE_NAME = "model.pkl"

DEFAULT_TIMEOUT = Duration("8 hours")

GB = 1024**3


class DockerJSONEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, ExecutionType):
            return o.name.lower()
        elif isinstance(o, (PosixPath, WindowsPath)):
            return str(o)
        return super().default(o)


class ExecutionType(Enum):
    TRAIN = 0
    EXECUTE = 1


class DockerTimeoutError(Exception):
    pass


class DockerAlgorithmFailedError(Exception):
    pass


@dataclass
class AlgorithmInterface:
    dataInput: Path
    dataOutput: Path
    modelInput: Path
    modelOutput: Path
    customParameters: dict = field(default_factory=dict)
    executionType: ExecutionType = ExecutionType.EXECUTE

    def to_json_string(self) -> str:
        dictionary = asdict(self)
        return json.dumps(dictionary, cls=DockerJSONEncoder)


class DockerAdapter(Adapter):
    def __init__(self, image_name: str, tag: str = "latest", group_privileges="akita", skip_pull=False,
                 timeout=DEFAULT_TIMEOUT):
        self.image_name = image_name
        self.tag = tag
        self.group = group_privileges
        self.skip_pull = skip_pull
        self.timeout = timeout

    @staticmethod
    def _get_gid(group: str) -> str:
        CMD = "getent group %s | cut -d ':' -f 3"
        return subprocess.run(CMD % group, capture_output=True, text=True, shell=True).stdout.strip()

    @staticmethod
    def _get_uid() -> str:
        return subprocess.run(["id", "-u"], capture_output=True, text=True).stdout.strip()

    @staticmethod
    def _get_resource_constraints(args: dict) -> Tuple[int, float]:
        """
        Per default: 1 task per node using all available cores and RAM (except small margin for OS).

        When not specified via "task_memory_limit" or "task_cpu_limit", the resources are equally shared between all
        concurrent tasks. This means that CPU limit is set based on node CPU count divided by the number of tasks and
        memory limit is set based on total memory of node minus 1 GB (for OS) divided by the number of tasks.

        Swap is always disabled.
        """
        tasks_per_node = args.get("tasks_per_node", 1)
        try:
            memory_limit = args["task_memory_limit"]
        except KeyError:
            memory = psutil.virtual_memory().total - 1 * GB
            memory_limit = memory // tasks_per_node
        try:
            cpu_limit = args["task_cpu_limit"]
        except KeyError:
            cpus = multiprocessing.cpu_count()
            cpu_limit = cpus / tasks_per_node
        return memory_limit, cpu_limit

    def _run_container(self, dataset_path: Path, args: dict) -> Container:
        client = docker.from_env()

        algorithm_interface = AlgorithmInterface(
            dataInput=(Path(DATASET_TARGET_PATH) / dataset_path.name).absolute(),
            dataOutput=(Path(RESULTS_TARGET_PATH) / SCORES_FILE_NAME).absolute(),
            modelInput=(Path(RESULTS_TARGET_PATH) / MODEL_FILE_NAME).absolute(),
            modelOutput=(Path(RESULTS_TARGET_PATH) / MODEL_FILE_NAME).absolute(),
            customParameters=args.get("hyper_params", {})
        )

        gid = DockerAdapter._get_gid(self.group)
        uid = DockerAdapter._get_uid()
        print(f"Running container with uid={uid} and gid={gid} privileges")

        memory_limit, cpu_limit = DockerAdapter._get_resource_constraints(args)
        cpu_shares = int(cpu_limit * 1e9)
        print(f"Restricting container to {cpu_limit} CPUs and {memory_limit / GB:.3f} GB RAM")

        return client.containers.run(
            f"{self.image_name}:{self.tag}",
            f"execute-algorithm '{algorithm_interface.to_json_string()}'",
            volumes={
                str(dataset_path.parent.absolute()): {'bind': DATASET_TARGET_PATH, 'mode': 'ro'},
                str(args.get("results_path", Path("./results")).absolute()): {'bind': RESULTS_TARGET_PATH, 'mode': 'rw'}
            },
            environment={
                "LOCAL_GID": gid,
                "LOCAL_UID": uid
            },
            mem_swappiness=0,
            mem_limit=memory_limit,
            memswap_limit=memory_limit,
            nano_cpus=cpu_shares,
            detach=True
        )

    def _run_until_timeout(self, container: Container, args: dict):
        try:
            result = container.wait(timeout=self.timeout.to_seconds())
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
            if "timed out" in str(e):
                container.stop()
                raise DockerTimeoutError(f"{self.image_name} timed out after {self.timeout}") from e
            else:
                raise e
        finally:
            print("\n#### Docker container logs ####")
            print(container.logs().decode("utf-8"))
            print("###############################\n")

        if result["StatusCode"] != 0:
            result_path = args.get("results_path", Path("./results")).absolute()
            raise DockerAlgorithmFailedError(f"Please consider log files in {result_path}!")

    def _read_results(self, args: dict) -> np.ndarray:
        return np.genfromtxt(args.get("results_path", Path("./results")) / SCORES_FILE_NAME, delimiter=",")

    # Adapter overwrites

    def _call(self, dataset: AlgorithmParameter, args: Optional[dict] = None) -> AlgorithmParameter:
        assert isinstance(dataset, (WindowsPath, PosixPath)), \
            "Docker adapters cannot handle NumPy arrays! Please put in the path to the dataset."
        args = args or {}
        container = self._run_container(dataset, args)
        self._run_until_timeout(container, args)

        return self._read_results(args)

    def get_prepare_fn(self) -> Optional[Callable[[], None]]:
        if not self.skip_pull:
            # capture variables for the function closure
            image: Final[str] = self.image_name
            tag: Final[str] = self.tag

            def prepare():
                client = docker.from_env()
                client.images.pull(image, tag=tag)
            return prepare
        else:
            return None

    def get_finalize_fn(self) -> Optional[Callable[[], None]]:
        def finalize():
            client = docker.from_env()
            client.containers.prune()
        return finalize

