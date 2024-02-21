"""Profiler CLI for NOS.

Note:
    If you have multiple GPUs and want to profile on a specific one, while
    keeping the correct PCI BUS id, you can use the following command:

    $ CUDA_DEVICE_ORDER=PCI_BUS_ID nos profile -m <model-id> -d 0 --verbose
"""

from typing import Iterator, Tuple

import humanize
import numpy as np
import typer
from PIL import Image
from rich import print

import sky
from sky.backends.cloud_vm_ray_backend import CloudVmRayResourceHandle
import requests
from nos.client import Client
import time

from nos import hub
from nos.common.profiler import ModelProfiler, ModelProfileRequest, Profiler
from nos.common.spec import ModelSpec
from nos.common.tasks import TaskType
from nos.logging import logger
from nos.server._runtime import InferenceServiceRuntime
from nos.test.utils import NOS_TEST_IMAGE


profile_cli = typer.Typer(name="profile", help="NOS Profiler CLI.", no_args_is_help=True)


def _model_inputs(task: TaskType, shape: Tuple[int, int] = None, batch_size: int = 1):
    if task == TaskType.IMAGE_EMBEDDING:
        im = Image.open(NOS_TEST_IMAGE)
        return {"images": [np.asarray(im.resize(shape)) for _ in range(batch_size)]}
    elif task == TaskType.TEXT_EMBEDDING:
        return {"texts": ["A photo of a cat."] * batch_size}
    elif task == TaskType.OBJECT_DETECTION_2D:
        im = Image.open(NOS_TEST_IMAGE)
        return {"images": [np.asarray(im.resize(shape)) for _ in range(batch_size)]}
    elif task == TaskType.IMAGE_GENERATION:
        assert batch_size == 1, "Image generation only supports batch_size=1 currently."
        return {"prompts": ["A photo of a cat."], "num_images": 1, "num_inference_steps": 10}
    elif task == TaskType.AUDIO_TRANSCRIPTION:
        from nos.test.utils import NOS_TEST_AUDIO

        assert batch_size == 1, "Audio transcription only supports batch_size=1 currently."
        return {"path": NOS_TEST_AUDIO, "chunk_length_s": 30, "return_timestamps": True}
    elif task == TaskType.DEPTH_ESTIMATION_2D:
        im = Image.open(NOS_TEST_IMAGE)
        return {"images": [np.asarray(im.resize(shape)) for _ in range(batch_size)]}
    elif task == TaskType.IMAGE_SEGMENTATION_2D:
        im = Image.open(NOS_TEST_IMAGE)
        return {"images": [np.asarray(im.resize(shape)) for _ in range(batch_size)]}
    elif task == TaskType.IMAGE_SUPER_RESOLUTION:
        im = Image.open(NOS_TEST_IMAGE)
        return {"images": [np.asarray(im.resize(shape)) for _ in range(batch_size)]}
    else:
        raise ValueError(f"Unsupported task: {task}")


def _model_methods(model_id: str = None) -> Iterator[Tuple[str, str, ModelSpec]]:
    models = hub.list()
    for _model_id in models:
        if model_id is not None and model_id != _model_id:
            continue
        spec: ModelSpec = hub.load_spec(_model_id)
        for method in spec.signature:
            yield _model_id, method, spec


def launch_profiling_cluster_with_resource(resource_name: str, cluster_name: str = None) -> CloudVmRayResourceHandle:
    # Launch a cluster with the specified resource
    if cluster_name is None:
        cluster_name = f"nos-profile-{resource_name}"

    # Check if the cluster already exists
    status = sky.status(cluster_name)
    if len(status) > 0:
        print(f"Cluster {cluster_name} already exists")
        status = sky.status(cluster_name)
        return status[0]['handle']

    # HACK: for now we've just added gpu and server resources to 
    # the http gateway so we can run the full profiling flow, but 
    # ideally this should be serialized out of the server.
    # Below will only work with the profiling rest endpoint exposed.
    # Only supports single model profiling for now. 
    # TODO: are we using the right nos version for this?
    nos_task = sky.Task(setup='pip install torch-nos',
            run='nos serve up --http')
            # workdir='~/.nosd/remote') 
    
    nos_task.set_resources(sky.Resources(accelerators=resource_name, ports=['50051']))
    
    try:
        print('launching cluster...')
        id, handle = sky.launch( 
            task=nos_task,
            cluster_name=cluster_name,
            retry_until_up=True,
            idle_minutes_to_autostop=30, # Kill the cluster after 30 mins of inactivity
            down=False, # Keep it running until then
            dryrun=False, # Turn this off when it works
            stream_logs=False,
            # detach_run=True,
        )
    except Exception as e:
        print(f"Failed to launch cluster {cluster_name} with resource {resource_name}")
        print(e)
        raise e

    print(f"Launched cluster {cluster_name} with resource {resource_name}" )

    return handle


def profile_cluster_model_id(resource_handle: CloudVmRayResourceHandle, model_id: str):
    active_ports = resource_handle.launched_resources.ports
    # We always use 50051 for consistency, uncler if this causes 
    # issues for multiple nos instances on one cluster?
    assert '50051' in active_ports
    client = Client(f"{resource_handle.head_ip}:50051")
    print(f"Connecting to {resource_handle.head_ip}:50051...")
    try:
        client.WaitForServer()
    except:
        print(f"Failed to connect to {resource_handle.head_ip}:50051")
        return
    
    if not client.IsHealthy():
        print(f"Failed to connect to {resource_handle.head_ip}:50051")
        return

    assert model_id is not None
    models: List[str] = client.ListModels()
    assert model_id in models

    """
        $ curl -X POST \
            'http://localhost:8000/v1/profile \
            -H 'accept: application/json' \
            -H 'Content-Type: multipart/form-data' \
            -F 'model_id=yolox/small' \
            -F '
    """

    print(f"Profiling {model_id}...")

    res = requests.post(
        f"http://{resource_handle.head_ip}:8000/v1/profile",
        headers={"accept": "application/json", "Content-Type": "multipart/form-data"},
        data={"model_id": model_id},
    )

    if res.status_code != 200:
        print(f"Failed to profile {model_id}")
        return
    
    catalog_path = res.json()["catalog_path"]
    print("Catalog path: ", catalog_path)


    # Old flow: just pings /infer
    """
    model = client.Module(model_id)
    assert model is not None
    assert model.GetModelInfo() is not None
    print(f"Test [model={model_id}]")

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    # exclude model loading time
    response = model(prompts=['astronaut on the moon'])

    total_runs = 3
    runs = []

    for i in range(total_runs):
        start_time = time.time()
        response = model(prompts=['astronaut on the moon'])
        end_time = time.time()
        assert isinstance(response, dict)
        print(f"Time taken: {end_time - start_time} seconds")
        runs.append(end_time - start_time)

    print(f"Average time taken: {sum(runs) / total_runs} seconds") 
    """


def profile_remote(
    model_id: str = None, resource: str = None
) -> Profiler:
    """Entrypoint for profiling remote models."""

    # Launch a profiling service on the appropriate resource if it doesn't already exist
    try:
        resource_handle = launch_profiling_cluster_with_resource(resource)
    except Exception as e:
        print(f"Failed to launch cluster with resource {resource}")
        print(e)
        return

    print("got resource handle: ", resource_handle)
    print('profiling: ', model_id)
    profile_cluster_model_id(resource_handle, model_id)
    return


def profile_models(
    model_id: str = None, device_id: int = 0, save: bool = False, verbose: bool = False, remote: str = None, catalog_path: str = None
):
    """Main entrypoint for profiling all models."""

    if remote is not None:
        # Requested remote profiling on a specific resource
        resource_handle = launch_profiling_cluster_with_resource(remote)
    import torch

    # TODO (spillai): Pytorch cuda.devices are reported in the descending order of memory,
    # so we need to force them to match the `nvidia-smi` order. Setting CUDA_DEVICE_ORDER=PCI_BUS_ID
    # allows us to keep the order consistent with `nvidia-smi`.
    assert torch.cuda.device_count() > 0, "No CUDA devices found, profiling is only supported on NVIDIA currently."
    # assert os.getenv("CUDA_DEVICE_ORDER", "") == "PCI_BUS_ID", "CUDA_DEVICE_ORDER must be PCI_BUS_ID."

    runtime = InferenceServiceRuntime.detect()
    logger.info(f"Detected runtime: {runtime}")

    # Get the device information (nvidia-gpu model type from torch)
    device: str = torch.cuda.get_device_properties(device_id).name.lower().replace(" ", "-")
    logger.info(f"Using device: {device}")

    # Profile all models
    profiler = ModelProfiler(mode="full", runtime=runtime, device_id=device_id)
    for model_id, method, spec in _model_methods(model_id):  # noqa: B020
        logger.debug(f"Profiling model: {model_id} (method: {method})")
        if model_id is None and model_id != model_id:
            logger.debug(f"Skipping model: {model_id} (not requested).")
            continue

        task: TaskType = spec.task(method)
        logger.debug(f"Task: {task}")
        if task == TaskType.IMAGE_EMBEDDING:
            profiler.add(
                ModelProfileRequest(
                    model_id=model_id,
                    method=method,
                    get_inputs=lambda: _model_inputs(task=TaskType.IMAGE_EMBEDDING, shape=(224, 224), batch_size=1),
                    batch_size=1,
                    shape=(224, 224),
                    device_name=profiler.device_name,
                ),
            )
        elif task == TaskType.TEXT_EMBEDDING:
            profiler.add(
                ModelProfileRequest(
                    model_id=model_id,
                    method=method,
                    get_inputs=lambda: _model_inputs(task=TaskType.TEXT_EMBEDDING, batch_size=1),
                    batch_size=1,
                    shape=None,
                    device_name=profiler.device_name,
                ),
            )
        elif task == TaskType.OBJECT_DETECTION_2D:
            profiler.add(
                ModelProfileRequest(
                    model_id=model_id,
                    method=method,
                    get_inputs=lambda: _model_inputs(
                        task=TaskType.OBJECT_DETECTION_2D, shape=(640, 480), batch_size=1
                    ),
                    batch_size=1,
                    shape=(640, 480),
                    device_name=profiler.device_name,
                ),
            )
        elif task == TaskType.IMAGE_GENERATION:
            profiler.add(
                ModelProfileRequest(
                    model_id=model_id,
                    method=method,
                    get_inputs=lambda: _model_inputs(task=TaskType.IMAGE_GENERATION, batch_size=1),
                    batch_size=1,
                    shape=None,
                    device_name=profiler.device_name,
                ),
            )
        elif task == TaskType.AUDIO_TRANSCRIPTION:
            profiler.add(
                ModelProfileRequest(
                    model_id=model_id,
                    method=method,
                    get_inputs=lambda: _model_inputs(task=TaskType.AUDIO_TRANSCRIPTION, batch_size=1),
                    batch_size=1,
                    shape=None,
                    device_name=profiler.device_name,
                ),
            )
        elif task == TaskType.DEPTH_ESTIMATION_2D:
            profiler.add(
                ModelProfileRequest(
                    model_id=model_id,
                    method=method,
                    get_inputs=lambda: _model_inputs(
                        task=TaskType.DEPTH_ESTIMATION_2D, shape=(640, 480), batch_size=1
                    ),
                    batch_size=1,
                    shape=(640, 480),
                    device_name=profiler.device_name,
                ),
            )
        elif task == TaskType.IMAGE_SEGMENTATION_2D:
            profiler.add(
                ModelProfileRequest(
                    model_id=model_id,
                    method=method,
                    get_inputs=lambda: _model_inputs(
                        task=TaskType.IMAGE_SEGMENTATION_2D, shape=(640, 480), batch_size=1
                    ),
                    batch_size=1,
                    shape=(640, 480),
                    device_name=profiler.device_name,
                ),
            )
        elif task == TaskType.IMAGE_SUPER_RESOLUTION:
            profiler.add(
                ModelProfileRequest(
                    model_id=model_id,
                    method=method,
                    get_inputs=lambda: _model_inputs(
                        task=TaskType.IMAGE_SUPER_RESOLUTION, shape=(160, 120), batch_size=1
                    ),
                    batch_size=1,
                    shape=(160, 120),
                    device_name=profiler.device_name,
                ),
            )
        else:
            logger.warning(f"Unsupported task: {task}, skipping.")
            continue

    # Run the profiler, and optionally save the catalog
    profiler.run()
    if save:
        profiler.save(catalog_path=catalog_path)
    return profiler, catalog_path


def profile_models_with_method(
    method_name: str, device_id: int = 0, save: bool = False, verbose: bool = False, catalog_path: str = None
) -> Profiler:
    for model_id, method, _spec in _model_methods(None):
        if method == method_name:
            profile_models(model_id, device_id, save, verbose, catalog_path)


@profile_cli.command(name="remote")
def _profile_remote(
    model_id: str = typer.Option(..., "-m", "--model-id", help="Model identifier."),
    resource: str = typer.Option(False, "--resource", "-r", help="Remote profiling on the given resource."),
):
    """Profile model on remote cluster."""
    profile_remote(model_id=model_id, resource=resource)


@profile_cli.command(name="model")
def _profile_model(
    model_id: str = typer.Option(..., "-m", "--model-id", help="Model identifier."),
    device_id: int = typer.Option(0, "--device-id", "-d", help="Device ID to use."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose profiling."),
    remote: str = typer.Option(False, "--remote", "-r", help="Remote profiling on the given resource."),
    catalog_path: str = typer.Option(None, "--catalog-path", "-e", help="Export path for the catalog json."),
):
    """Profile a specific model by its identifier."""
    profile_models(model_id, device_id=device_id, save=True, verbose=verbose, remote=remote, catalog_path=catalog_path)


@profile_cli.command(name="method")
def _profile_method(
    method_name: str = typer.Option(..., "-m", "--method-name", help="Method name."),
    device_id: int = typer.Option(0, "--device-id", "-d", help="Device ID to use."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose profiling."),
    catalog_path: str = typer.Option(None, "--catalog-path", "-e", help="Export path for the catalog json."),
):
    """Profile a specific model by its identifier."""
    profile_models_with_method(
        method_name=method_name, device_id=device_id, save=True, verbose=verbose, catalog_path=catalog_path
    )


@profile_cli.command(name="all")
def _profile_all_models(
    device_id: int = typer.Option(0, "--device-id", "-d", help="Device ID to use."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose profiling."),
    catalog_path: str = typer.Option(None, "--catalog-path", "-e", help="Export path for the catalog json."),
):
    """Profile all models."""
    profile_models(device_id=device_id, verbose=verbose, catalog_path=catalog_path)


@profile_cli.command(name="rebuild-catalog")
def _profile_rebuild_catalog(
    device_id: int = typer.Option(0, "--device-id", "-d", help="Device ID to use."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose profiling."),
    catalog_path: str = typer.Option(None, "--catalog-path", "-e", help="Export path for the catalog json."),
):
    """Profile all models and save the catalog."""
    profile_models(device_id=device_id, save=True, verbose=verbose, catalog_path=catalog_path)


@profile_cli.command(name="list")
def _profile_list(
    catalog_path: str = typer.Option(None, "--catalog-path", "-e", help="Load a preexisiting catalog."),
):
    """List all models and their methods."""
    from rich.table import Table

    from nos import hub

    table = Table(title="[green]  Models [/green]")
    table.add_column("model_id", max_width=30)
    table.add_column("method")
    table.add_column("task")
    table.add_column("runtime")
    table.add_column("device_type")
    table.add_column("device_name")
    table.add_column("it/s")
    table.add_column("cpu_memory")
    table.add_column("cpu_util")
    table.add_column("gpu_memory")
    table.add_column("gpu_util")

    for model in hub.list(private=False):
        spec: ModelSpec = hub.load_spec(model)
        for method in spec.signature:
            metadata = spec.metadata(method)
            profile = metadata.profile
            try:
                if hasattr(metadata, "resources") and metadata.resources is not None:
                    runtime = metadata.resources.runtime
                    device_type = "-".join(metadata.resources.device.split("-")[-2:])
                    cpu_memory = metadata.resources.memory
                    if type(cpu_memory) != str:
                        cpu_memory = f"{humanize.naturalsize(metadata.resources.memory, binary=True)}"
                    gpu_memory = metadata.resources.device_memory
                    if type(gpu_memory) != str:
                        gpu_memory = f"{humanize.naturalsize(metadata.resources.device_memory, binary=True)}"
                it_s = f'{profile["profiling_data"]["forward::execution"]["num_iterations"] * 1e3 / profile["profiling_data"]["forward::execution"]["total_ms"]:.1f}'
                cpu_util = f'{profile["profiling_data"]["forward::execution"]["cpu_utilization"]:0.2f}'
                gpu_util = f'{profile["profiling_data"]["forward::execution"]["gpu_utilization"]:0.2f}'
                device_name = f'{profile["device_name"]}'
            except Exception as e:
                logger.debug("Failed to load metadata: ", e)
                it_s = "-"
                cpu_util = "-"
                gpu_util = "-"
                cpu_memory = "-"
                gpu_memory = "-"
                runtime, device_type, device_name = None, None, None
            table.add_row(
                f"[green]{model}[/green]",
                method,
                spec.task(method),
                runtime,
                device_type,
                device_name,
                it_s,
                cpu_memory,
                cpu_util,
                gpu_memory,
                gpu_util,
            )
    print(table)
