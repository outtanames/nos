import time
import traceback
from functools import lru_cache
from typing import Any, Dict

import grpc
import rich.console
import rich.status
from google.protobuf import empty_pb2

from nos import hub
from nos.common import ModelSpec, TaskType, dumps
from nos.constants import DEFAULT_GRPC_PORT, NOS_PROFILING_ENABLED  # noqa F401
from nos.common import FunctionSignature, ModelSpec, TaskType, dumps, loads
from nos.common.shm import NOS_SHM_ENABLED, SharedMemoryDataDict, SharedMemoryTransportManager
from nos.constants import DEFAULT_GRPC_PORT  # noqa F401
from nos.exceptions import ModelNotFoundError
from nos.executors.ray import RayExecutor
from nos.logging import logger
from nos.managers import ModelHandle, ModelManager
from nos.protoc import import_module
from nos.version import __version__


nos_service_pb2 = import_module("nos_service_pb2")
nos_service_pb2_grpc = import_module("nos_service_pb2_grpc")


@lru_cache(maxsize=32)
def load_spec(model_name: str, task: TaskType) -> ModelSpec:
    """Get the model spec cache."""
    model_spec: ModelSpec = hub.load_spec(model_name, task=task)
    logger.info(f"Loaded model spec: {model_spec}")
    return model_spec


class InferenceService:
    """Ray-executor based inference service.

    Parameters:
        model_manager (ModelManager): Model manager.
        executor (RayExecutor): Ray executor.
        shm_manager (SharedMemoryTransportManager): Shared memory transport manager.
            Used to create shared memory buffers for inputs/outputs,
            and to copy data to/from shared memory.

    Note: To be used with the `InferenceServiceImpl` gRPC service.
    """

    def __init__(self):
        self.model_manager = ModelManager()
        self.executor = RayExecutor.get()
        try:
            self.executor.init()
        except Exception as e:
            logger.info(f"Failed to initialize executor: {e}")
            raise RuntimeError(f"Failed to initialize executor: {e}")
        if NOS_SHM_ENABLED:
            self.shm_manager = SharedMemoryTransportManager()
        else:
            self.shm_manager = None

    def execute(self, model_name: str, task: TaskType = None, inputs: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute the model.

        Args:
            model_name (str): Model identifier (e.g. `openai/clip-vit-base-patch32`).
            task (TaskType): Task type (e.g. `TaskType.OBJECT_DETECTION_2D`).
            inputs (Dict[str, Any]): Model inputs.
        Returns:
            Dict[str, Any]: Model outputs.
        """
        # Load the model spec
        try:
            model_spec: ModelSpec = load_spec(model_name, task=task)
        except Exception as e:
            raise ModelNotFoundError(f"Failed to load model spec: {model_name}, {e}")

        # TODO (spillai): Validate/Decode the inputs
        mid = time.perf_counter()
        model_inputs = FunctionSignature.validate(inputs, model_spec.signature.inputs)
        model_inputs = SharedMemoryDataDict.decode(model_inputs)
        # model_inputs = model_spec.signature._decode_inputs(inputs)
        model_inputs_types = [
            f"{k}: List[type={type(v[0])}, len={len(v)}]" if isinstance(v, list) else str(type(v))
            for k, v in model_inputs.items()
        ]
        logger.debug(
            f"Decoded inputs [inputs=({', '.join(model_inputs_types)}), elapsed={(time.perf_counter() - mid) * 1e3:.1f}ms]"
        )

        # Initialize the model (if not already initialized)
        # This call should also evict models and garbage collect if
        # too many models are loaded are loaded simultaneously.
        model_handle: ModelHandle = self.model_manager.get(model_spec)

        # Get the model handle and call it remotely (with model spec, actor handle)
        mid = time.perf_counter()
        response: Dict[str, Any] = model_handle.remote(**model_inputs)
        logger.debug(f"Executed model [name={model_spec.name}, elapsed={(time.perf_counter() - mid) * 1e3:.1f}ms]")

        # If the response is a single value, wrap it in a dict with the appropriate key
        if len(model_spec.signature.outputs) == 1:
            response = {k: response for k in model_spec.signature.outputs}

        # Encode the response
        response = SharedMemoryDataDict.encode(response)

        return response


class InferenceServiceImpl(nos_service_pb2_grpc.InferenceServiceServicer, InferenceService):
    """
    Experimental gRPC-based inference service.

    This service is used to serve models over gRPC.

    Refer to the bring-your-own-schema section:
    https://docs.ray.io/en/master/serve/direct-ingress.html?highlight=grpc#bring-your-own-schema
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def Ping(self, request: empty_pb2.Empty, context: grpc.ServicerContext) -> nos_service_pb2.PingResponse:
        """Health check."""
        return nos_service_pb2.PingResponse(status="ok")

    def GetServiceInfo(
        self, request: empty_pb2.Empty, context: grpc.ServicerContext
    ) -> nos_service_pb2.ServiceInfoResponse:
        """Get information on the service."""
        return nos_service_pb2.ServiceInfoResponse(version=__version__)

    def ListModels(self, request: empty_pb2.Empty, context: grpc.ServicerContext) -> nos_service_pb2.ModelListResponse:
        """List all models."""
        response = nos_service_pb2.ModelListResponse()
        for spec in hub.list():
            response.models.append(nos_service_pb2.ModelInfo(name=spec.name, task=spec.task.value))
        return response

    def GetModelInfo(
        self, request: nos_service_pb2.ModelInfoRequest, context: grpc.ServicerContext
    ) -> nos_service_pb2.ModelInfoResponse:
        """Get model information."""
        try:
            model_info = request.request
            spec: ModelSpec = hub.load_spec(model_info.name, task=TaskType(model_info.task))
            logger.debug(f"GetModelInfo(): {spec}")
        except KeyError as e:
            logger.error(f"Failed to load spec: [request={request.request}, e={e}]")
            context.abort(grpc.StatusCode.NOT_FOUND, str(e))
        return spec._to_proto(public=True)

    def RegisterSystemSharedMemory(
        self, request: nos_service_pb2.GenericRequest, context: grpc.ServicerContext
    ) -> nos_service_pb2.GenericResponse:
        """Register system shared memory with a dry-run inference request."""
        if not NOS_SHM_ENABLED:
            context.abort(grpc.StatusCode.UNIMPLEMENTED, "Shared memory not enabled.")

        metadata = dict(context.invocation_metadata())
        client_id = metadata.get("client_id", None)
        spec_id = metadata.get("spec_id", None)
        logger.debug(f"Registering system shared memory for client_id={client_id}, spec_id={spec_id}")
        try:
            shm_map = self.shm_manager.create(loads(request.request_bytes), namespace=f"{client_id}/{spec_id}")
            return nos_service_pb2.GenericResponse(response_bytes=dumps(shm_map))
        except Exception as e:
            logger.error(f"Failed to register system shared memory: {e}")
            context.abort(grpc.StatusCode.INTERNAL, str(e))

    def UnregisterSystemSharedMemory(
        self, request: nos_service_pb2.GenericRequest, context: grpc.ServicerContext
    ) -> nos_service_pb2.GenericResponse:
        """Unregister system shared memory."""
        if not NOS_SHM_ENABLED:
            context.abort(context, grpc.StatusCode.UNIMPLEMENTED, "Shared memory not enabled.")

        metadata = dict(context.invocation_metadata())
        client_id = metadata.get("client_id", None)
        spec_id = metadata.get("spec_id", None)
        logger.debug(f"Unregistering system shared memory for client_id={client_id}, spec_id={spec_id}")
        try:
            self.shm_manager.cleanup(namespace=f"{client_id}/{spec_id}")
        except Exception as e:
            logger.error(f"Failed to unregister system shared memory: {e}")
            context.abort(grpc.StatusCode.INTERNAL, str(e))
        return nos_service_pb2.GenericResponse()

    def Run(
        self, request: nos_service_pb2.InferenceRequest, context: grpc.ServicerContext
    ) -> nos_service_pb2.InferenceResponse:
        """Main model prediction interface."""
        model_request = request.model
        logger.debug(f"Received request: {model_request.task}, {model_request.name}")
        if model_request.task not in (
            TaskType.IMAGE_GENERATION.value,
            TaskType.IMAGE_EMBEDDING.value,
            TaskType.TEXT_EMBEDDING.value,
            TaskType.OBJECT_DETECTION_2D.value,
            TaskType.IMAGE_SEGMENTATION_2D.value,
            TaskType.CUSTOM.value,
        ):
            context.abort(grpc.StatusCode.NOT_FOUND, f"Invalid task {model_request.task}")

        try:
            st = time.perf_counter()
            logger.info(f"Executing request [task={model_request.task}, name={model_request.name}]")
            response = self.execute(model_request.name, task=TaskType(model_request.task), inputs=request.inputs)
            logger.info(
                f"Executed request [task={model_request.task}, model={model_request.name}, elapsed={(time.perf_counter() - st) * 1e3:.1f}ms]"
            )
            return nos_service_pb2.InferenceResponse(response_bytes=dumps(response))
        except (grpc.RpcError, Exception) as e:
            msg = f"Failed to execute request: [task={model_request.task}, model={model_request.name}]"
            msg += f"{traceback.format_exc()}"
            logger.error(f"{msg}, e={e}")
            context.abort(grpc.StatusCode.INTERNAL, "Internal Server Error")


def serve(address: str = f"[::]:{DEFAULT_GRPC_PORT}", max_workers: int = 1) -> None:
    """Start the gRPC server."""
    from concurrent import futures

    options = [
        ("grpc.max_message_length", 512 * 1024 * 1024),
        ("grpc.max_send_message_length", 512 * 1024 * 1024),
        ("grpc.max_receive_message_length", 512 * 1024 * 1024),
    ]
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers), options=options)
    nos_service_pb2_grpc.add_InferenceServiceServicer_to_server(InferenceServiceImpl(), server)
    server.add_insecure_port(address)

    console = rich.console.Console()
    with console.status(f"[bold green] Starting server on {address}[/bold green]") as status:
        server.start()
        console.print(
            f"[bold green] ✓ InferenceService :: Deployment complete [/bold green]",  # noqa
        )
        status.stop()
        server.wait_for_termination()
        console.print("Server stopped")


def main():
    serve()


if __name__ == "__main__":
    main()