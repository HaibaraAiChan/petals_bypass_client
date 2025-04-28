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
    """
    assert len(kwargs) == len(rpc_info["keyword_names"]), f"Keyword args should be {rpc_info['keyword_names']}"
    kwargs = {key: kwargs[key] for key in rpc_info["keyword_names"]}

    # Record start time for the entire operation
    total_start_time = time.time()

    forward_inputs = tuple(nested_flatten((inputs, kwargs)))
    args_schema, kwargs_schema = rpc_info["forward_schema"]
    compression = args_schema[0].compression
    forward_schema = tuple(BatchTensorDescriptor.from_tensor(arg, compression) for arg in forward_inputs)
    inputs = tuple(tensor.cpu().detach() for tensor in forward_inputs)
    assert len(inputs) >= len(args_schema) + 1, "Inputs and prompt tensors are necessary for a forward step"

    # Record serialization start time
    serial_start_time = time.time()
    
    # Asynchronous serialization
    loop = asyncio.get_running_loop()
    serialized_tensors = await asyncio.gather(
        *(
            loop.run_in_executor(None, serialize_torch_tensor, tensor.to(proto.dtype), proto.compression)
            for tensor, proto in zip(inputs, forward_schema)
        )
    )
    
    serial_time = time.time() - serial_start_time
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

    # Log initial stats including serialization time
    logger.info(
        f"[PROFILING] Remote Forward: "
        f"UID: {uid}, "
        f"Step: {step_id}, "
        f"Input Size: {input_size_bytes / (1024*1024):.2f} MB, "
        f"Serialization Time: {serial_time*1000:.2f} ms"
    )

    # Define input tensor names based on their role
    input_names = ["hidden_states", "attention_mask", "position_ids", "prompt"]
    for i, tensor in enumerate(inputs):
        tensor_name = input_names[i] if i < len(input_names) else f"extra_input_{i}"
        logger.info(
            f"[PROFILING] Remote Forward Input {tensor_name}: "
            f"shape={getattr(tensor, 'shape', None)}, "
            f"dtype={getattr(tensor, 'dtype', None)}, "
            f"size={tensor.element_size() * tensor.nelement() / (1024*1024):.2f} MB, "
            f"Serialization Time: {serial_time*1000:.2f} ms"
        )

    # Record network transfer start time
    transfer_start_time = time.time()
    size = sum(t.element_size() * t.nelement() for t in inputs)
    forward_fn = _forward_stream if size > MAX_UNARY_PAYLOAD_SIZE // 2 else _forward_unary
    deserialized_outputs = await forward_fn(uid, serialized_tensors, stub, config, metadata=metadata, **kwargs)
    transfer_time = time.time() - transfer_start_time
    
    # Calculate total time and throughput
    total_time = time.time() - total_start_time
    output_size_bytes = sum(t.element_size() * t.nelement() for t in deserialized_outputs if hasattr(t, 'element_size'))
    
    # Log completion with detailed timing information
    logger.info(
        f"[PROFILING] Remote Forward Done: "
        f"UID: {uid}, "
        f"Step: {step_id}, "
        f"Output Size: {output_size_bytes / (1024*1024):.2f} MB, "
        f"Total Time: {total_time*1000:.2f} ms, "
        f"Serialization Time: {serial_time*1000:.2f} ms, "
        f"Network Time: {transfer_time*1000:.2f} ms, "
        f"Compute Time: {(total_time - serial_time - transfer_time)*1000:.2f} ms, "
        f"Throughput: {(input_size_bytes + output_size_bytes) / (1024*1024) / total_time:.2f} MB/s"
    )

    # Define output tensor names with timing information
    output_names = ["layer_output", "attention_weights", "hidden_states"]
    for i, tensor in enumerate(deserialized_outputs):
        output_name = output_names[i] if i < len(output_names) else f"extra_output_{i}"
        logger.info(
            f"[PROFILING] Remote Forward Output {output_name}: "
            f"shape={getattr(tensor, 'shape', None)}, "
            f"dtype={getattr(tensor, 'dtype', None)}, "
            f"size={tensor.element_size() * tensor.nelement() / (1024*1024):.2f} MB, "
            f"Network Time: {transfer_time*1000:.2f} ms"
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
    """Run backward pass for a remote block"""
    # Record start time for the entire operation
    total_start_time = time.time()
    step_id = kwargs.get("step_id", "N/A")
    
    # Get compression from schema like in run_remote_forward
    args_schema, kwargs_schema = rpc_info["forward_schema"]
    compression = args_schema[0].compression
    
    # Record serialization start time
    serial_start_time = time.time()
    
    # Prepare inputs and serialize with correct compression
    inputs_and_grad_outputs = tuple(tensor.cpu().detach() for tensor in inputs_and_grad_outputs)
    serialized_tensors = [
        serialize_torch_tensor(tensor, compression, allow_inplace=True) for tensor in inputs_and_grad_outputs
    ]
    
    serial_time = time.time() - serial_start_time
    input_size_bytes = sum(t.element_size() * t.nelement() for t in inputs_and_grad_outputs)
    logger = config.logger if hasattr(config, 'logger') else __import__('logging').getLogger(__name__)
    
    # Log initial stats including serialization time
    logger.info(
        f"[PROFILING] Remote Backward: "
        f"UID: {uid}, "
        f"Step: {step_id}, "
        f"Input Size: {input_size_bytes / (1024*1024):.2f} MB, "
        f"Serialization Time: {serial_time*1000:.2f} ms"
    )
    
    # Define tensor names based on their role in backward pass
    tensor_names = ["inputs", "grad_outputs", "prompts"]
    for i, tensor in enumerate(inputs_and_grad_outputs):
        tensor_name = tensor_names[i] if i < len(tensor_names) else f"extra_tensor_{i}"
        logger.info(
            f"[PROFILING] Remote Backward Input {tensor_name}: "
            f"shape={getattr(tensor, 'shape', None)}, "
            f"dtype={getattr(tensor, 'dtype', None)}, "
            f"size={tensor.element_size() * tensor.nelement() / (1024*1024):.2f} MB, "
            f"Serialization Time: {serial_time*1000:.2f} ms"
        )
    
    # Record network transfer start time
    transfer_start_time = time.time()
    size = sum(t.element_size() * t.nelement() for t in inputs_and_grad_outputs)
    backward_fn = _backward_stream if size > MAX_UNARY_PAYLOAD_SIZE // 2 else _backward_unary
    deserialized_grad_inputs = await backward_fn(uid, serialized_tensors, stub, config, metadata=metadata, **kwargs)
    transfer_time = time.time() - transfer_start_time
    
    # Calculate total time and throughput
    total_time = time.time() - total_start_time
    output_size_bytes = sum(t.element_size() * t.nelement() for t in deserialized_grad_inputs if hasattr(t, 'element_size'))
    
    # Log completion with detailed timing information
    logger.info(
        f"[PROFILING] Remote Backward Done: "
        f"UID: {uid}, "
        f"Step: {step_id}, "
        f"Output Size: {output_size_bytes / (1024*1024):.2f} MB, "
        f"Total Time: {total_time*1000:.2f} ms, "
        f"Serialization Time: {serial_time*1000:.2f} ms, "
        f"Network Time: {transfer_time*1000:.2f} ms, "
        f"Compute Time: {(total_time - serial_time - transfer_time)*1000:.2f} ms, "
        f"Throughput: {(input_size_bytes + output_size_bytes) / (1024*1024) / total_time:.2f} MB/s"
    )
    
    # Define output tensor names with timing information
    output_names = ["grad_inputs", "grad_prompts"]
    for i, tensor in enumerate(deserialized_grad_inputs):
        output_name = output_names[i] if i < len(output_names) else f"grad_extra_{i}"
        logger.info(
            f"[PROFILING] Remote Backward Output {output_name}: "
            f"shape={getattr(tensor, 'shape', None)}, "
            f"dtype={getattr(tensor, 'dtype', None)}, "
            f"size={tensor.element_size() * tensor.nelement() / (1024*1024):.2f} MB, "
            f"Network Time: {transfer_time*1000:.2f} ms"
        )
    return deserialized_grad_inputs
