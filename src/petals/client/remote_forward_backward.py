"""
Utility functions that call RPC forward or backward on a single remote server
"""
import asyncio
from typing import Iterable, List, Optional, Sequence, Tuple
import time

import torch
from hivemind import nested_compare, nested_flatten, nested_pack, serialize_torch_tensor
from hivemind.compression.serialization import deserialize_tensor_stream, deserialize_torch_tensor
from hivemind.p2p import StubBase
from hivemind.p2p.p2p_daemon_bindings.control import DEFAULT_MAX_MSG_SIZE, MAX_UNARY_PAYLOAD_SIZE
from hivemind.proto import runtime_pb2
from hivemind.utils.asyncio import aiter_with_timeout, iter_as_aiter
from hivemind.utils.streaming import split_for_streaming
from hivemind.utils.tensor_descr import BatchTensorDescriptor

from petals.client.config import ClientConfig
from petals.data_structures import ModuleUID, RPCInfo


async def _forward_unary(
    uid: str, serialized_tensors: Iterable[runtime_pb2.Tensor], stub, config: ClientConfig, **kwargs
) -> List[torch.Tensor]:
    outputs: runtime_pb2.ExpertResponse = await stub.rpc_forward(
        runtime_pb2.ExpertRequest(uid=uid, tensors=list(serialized_tensors), **kwargs),
        timeout=config.request_timeout,
    )
    return [deserialize_torch_tensor(t) for t in outputs.tensors]


async def _backward_unary(
    uid: str, serialized_tensors: Iterable[runtime_pb2.Tensor], stub, config: ClientConfig, **kwargs
) -> List[torch.Tensor]:
    grad_inputs: runtime_pb2.ExpertResponse = await stub.rpc_backward(
        runtime_pb2.ExpertRequest(uid=uid, tensors=list(serialized_tensors), **kwargs),
        timeout=config.request_timeout,
    )
    return [deserialize_torch_tensor(t) for t in grad_inputs.tensors]


async def _forward_stream(
    uid: str, serialized_tensors: Iterable[runtime_pb2.Tensor], stub, config: ClientConfig, **kwargs
) -> List[torch.Tensor]:
    parts = (
        runtime_pb2.ExpertRequest(uid=uid, tensors=[part], **kwargs)
        for tensor in serialized_tensors
        for part in split_for_streaming(tensor, DEFAULT_MAX_MSG_SIZE)
    )
    outputs = await asyncio.wait_for(stub.rpc_forward_stream(iter_as_aiter(parts)), config.connect_timeout)
    outputs = aiter_with_timeout(outputs, config.request_timeout)
    return await deserialize_tensor_stream(msg.tensors async for msg in outputs)


async def _backward_stream(
    uid: str, serialized_tensors: Iterable[runtime_pb2.Tensor], stub, config: ClientConfig, **kwargs
) -> List[torch.Tensor]:
    parts = (
        runtime_pb2.ExpertRequest(uid=uid, tensors=[part], **kwargs)
        for tensor in serialized_tensors
        for part in split_for_streaming(tensor, DEFAULT_MAX_MSG_SIZE)
    )
    grad_inputs = await asyncio.wait_for(stub.rpc_backward_stream(iter_as_aiter(parts)), config.connect_timeout)
    grad_inputs = aiter_with_timeout(grad_inputs, config.request_timeout)
    return await deserialize_tensor_stream(msg.tensors async for msg in grad_inputs)


async def run_remote_forward(
    uid: ModuleUID,
    stub: StubBase,
    rpc_info: RPCInfo,
    *inputs: torch.Tensor,
    config: ClientConfig,
    metadata: Optional[bytes] = None,
    **kwargs,
) -> Tuple[torch.Tensor, ...]:
    """
    Serializes input tensors and calls "rpc_forward" on a remote server.
    Mostly adapted from https://github.com/learning-at-home/hivemind/blob/7a7c93aefffc9494c39e7b170c07cb06d8c09c4c/hivemind/moe/client/expert.py#L198
    but without RemoteExpertWorker.run_coroutine() call that leads to deadlock here.
    """

    # Note: *inputs are flattened input tensors that follow the expert's info['input_schema']
    # detach to avoid pickling the computation graph
    assert len(kwargs) == len(rpc_info["keyword_names"]), f"Keyword args should be {rpc_info['keyword_names']}"
    kwargs = {key: kwargs[key] for key in rpc_info["keyword_names"]}

    # Note: we put keyword arguments in the same order as on a server to prevent f(a=1, b=2) != f(b=2, a=1) errors
    forward_inputs = tuple(nested_flatten((inputs, kwargs)))
    args_schema, kwargs_schema = rpc_info["forward_schema"]
    compression = args_schema[0].compression
    forward_schema = tuple(BatchTensorDescriptor.from_tensor(arg, compression) for arg in forward_inputs)
    inputs = tuple(tensor.cpu().detach() for tensor in forward_inputs)
    # TODO: create more explicit way to check servers schema and client's structure
    assert len(inputs) >= len(args_schema) + 1, "Inputs and prompt tensors are necessary for a forward step"

    # Asynchronous serialization
    loop = asyncio.get_running_loop()
    serialized_tensors = await asyncio.gather(
        *(
            loop.run_in_executor(None, serialize_torch_tensor, tensor.to(proto.dtype), proto.compression)
            for tensor, proto in zip(inputs, forward_schema)
        )
    )

    # Profiling: Log input tensor details
    input_size_bytes = sum(tensor.element_size() * tensor.nelement() for tensor in inputs)
    logger = config.logger if hasattr(config, 'logger') else __import__('logging').getLogger(__name__)
    step_id = None
    if metadata is not None:
        try:
            import hivemind
            meta_dict = hivemind.MSGPackSerializer.loads(metadata)
            step_id = meta_dict.get('step_id', None)
        except Exception:
            pass
    logger.info(
        f"[PROFILING] Remote Forward: "
        f"UID: {uid}, "
        f"Step: {step_id}, "
        f"Input Size: {input_size_bytes / (1024*1024):.2f} MB"
    )
    for i, tensor in enumerate(inputs):
        logger.info(
            f"[PROFILING] Remote Forward Input Tensor {i}: "
            f"shape={getattr(tensor, 'shape', None)}, "
            f"dtype={getattr(tensor, 'dtype', None)}, "
            f"size={tensor.element_size() * tensor.nelement() / (1024*1024):.2f} MB"
        )
    # call RPC on remote server
    start_time = time.time()
    size = sum(t.element_size() * t.nelement() for t in inputs)
    forward_fn = _forward_stream if size > MAX_UNARY_PAYLOAD_SIZE // 2 else _forward_unary
    deserialized_outputs = await forward_fn(uid, serialized_tensors, stub, config, metadata=metadata, **kwargs)
    latency = time.time() - start_time
    output_size_bytes = sum(t.element_size() * t.nelement() for t in deserialized_outputs if hasattr(t, 'element_size'))
    logger.info(
        f"[PROFILING] Remote Forward Done: "
        f"UID: {uid}, "
        f"Step: {step_id}, "
        f"Output Size: {output_size_bytes / (1024*1024):.2f} MB, "
        f"Latency: {latency*1000:.2f} ms, "
        f"Throughput: {output_size_bytes / (1024*1024) / latency:.2f} MB/s" if output_size_bytes > 0 else
        f"Latency: {latency*1000:.2f} ms"
    )
    for i, tensor in enumerate(deserialized_outputs):
        logger.info(
            f"[PROFILING] Remote Forward Output Tensor {i}: "
            f"shape={getattr(tensor, 'shape', None)}, "
            f"dtype={getattr(tensor, 'dtype', None)}, "
            f"size={tensor.element_size() * tensor.nelement() / (1024*1024):.2f} MB"
        )
    return nested_pack(deserialized_outputs, structure=rpc_info["outputs_schema"])


async def run_remote_backward(
    uid: ModuleUID,
    stub: StubBase,
    rpc_info: RPCInfo,
    *inputs_and_grad_outputs: torch.Tensor,
    config: ClientConfig,
    metadata: Optional[bytes] = None,
    **kwargs,
) -> Sequence[torch.Tensor]:
    """
    Serializes grad outputs and calls "rpc_backward" on a remote server.
    Mostly adapted from https://github.com/learning-at-home/hivemind/blob/7a7c93aefffc9494c39e7b170c07cb06d8c09c4c/hivemind/moe/client/expert.py#L221
    but without RemoteExpertWorker.run_coroutine() call that leads to deadlock here.
    """
    args_schema, kwargs_schema = rpc_info["forward_schema"]
    outputs_schema = rpc_info["outputs_schema"]
    compression = args_schema[0].compression
    backward_schema = tuple(BatchTensorDescriptor.from_tensor(arg, compression) for arg in inputs_and_grad_outputs)
    # TODO: create more explicit way to check servers schema and client's structure
    assert (
        len(inputs_and_grad_outputs) >= len(args_schema) + len(outputs_schema) + 1
    ), "Inputs, grad_outputs and prompt tensors are necessary for a backward step"

    # Asynchronous serialization
    loop = asyncio.get_running_loop()
    serialized_tensors = await asyncio.gather(
        *(
            loop.run_in_executor(None, serialize_torch_tensor, tensor.to(proto.dtype), proto.compression)
            for tensor, proto in zip(inputs_and_grad_outputs, backward_schema)
        )
    )

    # Profiling: Log input tensor details
    input_size_bytes = sum(tensor.element_size() * tensor.nelement() for tensor in inputs_and_grad_outputs)
    logger = config.logger if hasattr(config, 'logger') else __import__('logging').getLogger(__name__)
    step_id = None
    if metadata is not None:
        try:
            import hivemind
            meta_dict = hivemind.MSGPackSerializer.loads(metadata)
            step_id = meta_dict.get('step_id', None)
        except Exception:
            pass
    logger.info(
        f"[PROFILING] Remote Backward: "
        f"UID: {uid}, "
        f"Step: {step_id}, "
        f"Input Size: {input_size_bytes / (1024*1024):.2f} MB"
    )
    for i, tensor in enumerate(inputs_and_grad_outputs):
        logger.info(
            f"[PROFILING] Remote Backward Input Tensor {i}: "
            f"shape={getattr(tensor, 'shape', None)}, "
            f"dtype={getattr(tensor, 'dtype', None)}, "
            f"size={tensor.element_size() * tensor.nelement() / (1024*1024):.2f} MB"
        )
    start_time = time.time()
    size = sum(t.element_size() * t.nelement() for t in inputs_and_grad_outputs)
    backward_fn = _backward_stream if size > MAX_UNARY_PAYLOAD_SIZE // 2 else _backward_unary
    deserialized_grad_inputs = await backward_fn(uid, serialized_tensors, stub, config, metadata=metadata, **kwargs)
    latency = time.time() - start_time
    output_size_bytes = sum(t.element_size() * t.nelement() for t in deserialized_grad_inputs if hasattr(t, 'element_size'))
    logger.info(
        f"[PROFILING] Remote Backward Done: "
        f"UID: {uid}, "
        f"Step: {step_id}, "
        f"Output Size: {output_size_bytes / (1024*1024):.2f} MB, "
        f"Latency: {latency*1000:.2f} ms, "
        f"Throughput: {output_size_bytes / (1024*1024) / latency:.2f} MB/s" if output_size_bytes > 0 else
        f"Latency: {latency*1000:.2f} ms"
    )
    for i, tensor in enumerate(deserialized_grad_inputs):
        logger.info(
            f"[PROFILING] Remote Backward Output Tensor {i}: "
            f"shape={getattr(tensor, 'shape', None)}, "
            f"dtype={getattr(tensor, 'dtype', None)}, "
            f"size={tensor.element_size() * tensor.nelement() / (1024*1024):.2f} MB"
        )
    return deserialized_grad_inputs
