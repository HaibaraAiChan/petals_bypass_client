from __future__ import annotations

import asyncio
import itertools
import time
import uuid
from typing import AsyncIterator, List, Optional, Tuple

import torch
from hivemind import MSGPackSerializer, anext, deserialize_torch_tensor, get_logger, serialize_torch_tensor
from hivemind.moe.client.remote_expert_worker import RemoteExpertWorker
from hivemind.p2p import P2P
from hivemind.proto import runtime_pb2
from hivemind.utils.tensor_descr import BatchTensorDescriptor

from petals.client.config import ClientConfig
from petals.client.routing import RemoteSequenceManager, maybe_log_traceback
from petals.data_structures import CHAIN_DELIMITER, ModuleUID, RemoteSpanInfo, RPCInfo
from petals.server.handler import TransformerConnectionHandler
from petals.utils.misc import DUMMY, DUMMY_INT64, is_dummy
from petals.utils.packaging import pack_args_kwargs

logger = get_logger(__name__)


class _ServerInferenceSession:
    """
    An interface to a single multi-step *inference* session for a a set of blocks on a specific server.

    :note: This class is *not* fault-tolerant out of the box.
    """

    def __init__(
        self,
        config: ClientConfig,
        span: RemoteSpanInfo,
        uid: ModuleUID,
        rpc_info: RPCInfo,
        inputs_queue: asyncio.Queue,
        outputs_aiter: AsyncIterator,
        *,
        max_length: int,
        **metadata,
    ):
        self.config = config
        self.span, self.uid, self.rpc_info = span, uid, rpc_info
        self.num_blocks = uid.count(CHAIN_DELIMITER) + 1
        self._inputs_queue: asyncio.Queue[runtime_pb2.ExpertRequest] = inputs_queue
        self._outputs_stream: AsyncIterator[runtime_pb2.ExpertResponse] = outputs_aiter
        self.session_id = str(uuid.uuid4())
        self.session_metadata = dict(max_length=max_length, **metadata)
        self.stepped = False
        self.closed = False

        self._position = 0
        self.history = None  # Used in case of server failures to regenerate attention caches on new servers
        self.next_session = None

    @classmethod
    async def create(
        cls,
        config: ClientConfig,
        p2p: P2P,
        span: RemoteSpanInfo,
        uid: ModuleUID,
        rpc_info: RPCInfo,
        **metadata,
    ) -> _ServerInferenceSession:
        """Create a new session for a given remote module. This code is meant to be run inside RemoteExpertWorker"""
        stub = TransformerConnectionHandler.get_stub(p2p, span.peer_id)
        inputs_queue = asyncio.Queue()
        outputs_stream = await asyncio.wait_for(
            stub.rpc_inference(cls._read_inputs_from_queue(inputs_queue)),
            config.connect_timeout,
        )
        return cls(config, span, uid, rpc_info, inputs_queue, outputs_stream, **metadata)

    @staticmethod
    async def _read_inputs_from_queue(queue: asyncio.Queue, input_timeout: Optional[float] = None) -> AsyncIterator:
        while True:
            next_input_message = await asyncio.wait_for(queue.get(), input_timeout)
            yield next_input_message
            if not next_input_message.uid and not next_input_message.tensors:
                break  # this message means "done sending"

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, start_from_position: int):
        assert start_from_position <= self._position
        self._position = start_from_position
        if self.history is not None and self.history.shape[1] >= start_from_position:
            self.history = self.history[:, :start_from_position, :] if start_from_position > 0 else None

    def step(
        self,
        inputs: torch.Tensor,
        prompts: torch.Tensor,
        hypo_ids: torch.LongTensor,
        *,
        step_id: str,
    ) -> torch.Tensor:
        """
        Inference step: send a chunk of input tensors and receive a chunk of outputs
        :prompts: optional DEEP prompts, added to a prefix of each layer's outputs,
          if specified, deep prompts should have shape [num_layers, batch_size, prefix_len, hid_size]
        """
        if self.closed:
            raise Exception("Session is closed, cannot perform step")

        n_input_tokens = inputs.shape[1]
        if self.history is None:
            self.history = inputs
        elif self.history.shape[1] == self._position:
            self.history = torch.cat([self.history, inputs[:, -n_input_tokens:]], dim=1)
        assert self.history.shape[1] == self._position + n_input_tokens, (
            f"Broken input cache: span={self.span} shape={self.history.shape} "
            f"position={self._position} n_input_tokens={n_input_tokens}"
        )

        if not self.stepped:
            inputs = self.history  # Pass full inputs including prefix
        else:
            inputs = inputs[:, -n_input_tokens:]  # No need to pass prefix further

        # serialize inputs and put them into the queue
        input_tensors, args_structure = pack_args_kwargs(inputs, prompts, hypo_ids)

        # Profiling: Log input tensor details
        hidden_states_size_bytes = inputs.element_size() * inputs.nelement()
        logger.info(
            f"[PROFILING] Client-to-Server Transmission: "
            f"To server {self.span.peer_id.to_base58()}, "
            f"Session: {self.session_id}, "
            f"Step: {step_id}, "
            f"Hidden States Shape: {inputs.shape}, "
            f"Size: {hidden_states_size_bytes / (1024*1024):.2f} MB"
        )

        request_metadata = dict(session_id=self.session_id, step_id=step_id)
        if not self.stepped:
            request_metadata.update(self.session_metadata)
        if self._position is not None:
            request_metadata["start_from_position"] = self._position
        if self.config.use_server_to_server:
            next_servers = self._collect_next_servers()
            if next_servers:
                request_metadata["next_servers"] = next_servers
                logger.info(
                    f"[PROFILING] Server-to-Server Enabled: "
                    f"Next servers: {[server[0] for server in next_servers]}"
                )
            else:
                logger.info("[PROFILING] Server-to-Server Disabled: No next servers available")
        else:
            logger.info("[PROFILING] Server-to-Server Disabled: Configuration setting")

        request_metadata["args_structure"] = args_structure

        # TODO: make possible to use different compression method for different tensors
        server_side_inference_schema, kwargs_schema = self.rpc_info["inference_schema"]
        compression = server_side_inference_schema[0].compression
        inference_schema = tuple(BatchTensorDescriptor.from_tensor(arg, compression) for arg in input_tensors)

        # TODO: create more explicit way to check servers schema and client's structure
        assert len(input_tensors) >= len(
            server_side_inference_schema
        ), "Hidden_state, prompts and hypo_ids tensors are necessary for an inference step"

        outputs_serialized = RemoteExpertWorker.run_coroutine(
            self._step(
                runtime_pb2.ExpertRequest(
                    uid=self.uid,
                    tensors=[
                        serialize_torch_tensor(tensor.to(proto.dtype), proto.compression)
                        for tensor, proto in zip(input_tensors, inference_schema)
                    ],
                    metadata=MSGPackSerializer.dumps(request_metadata),
                )
            )
        )
        outputs = list(map(deserialize_torch_tensor, outputs_serialized.tensors))
        assert (
            outputs[0].shape == inputs.shape
        ), f"output activation shape is different from input shape: {outputs[0].shape} != {inputs.shape}"

        self._position += n_input_tokens

        return outputs[0]

    def _collect_next_servers(self) -> List[Tuple[str, str, int, int]]:
        next_servers = []
        session = self.next_session
        while session is not None:
            next_servers.append(
                (session.span.peer_id.to_base58(), session.session_id, session.span.start, session.span.end)
            )
            session = session.next_session
        return next_servers

    async def _step(self, inputs_serialized: runtime_pb2.ExpertRequest) -> runtime_pb2.ExpertResponse:
        """Inference step on serialized data. This code is meant to be run inside RemoteExpertWorker"""
        await self._inputs_queue.put(inputs_serialized)
        self.stepped = True
        
        # Profiling: Record start time
        start_time = time.time()
        
        response = await asyncio.wait_for(anext(self._outputs_stream), self.config.request_timeout)
        
        # Profiling: Calculate processing time
        processing_time = time.time() - start_time
        
        # 安全地获取 step_id
        step_id = 'unknown'
        try:
            # 尝试反序列化 metadata
            if hasattr(inputs_serialized, 'metadata') and inputs_serialized.metadata:
                metadata_dict = MSGPackSerializer.loads(inputs_serialized.metadata)
                if isinstance(metadata_dict, dict) and 'step_id' in metadata_dict:
                    step_id = metadata_dict['step_id']
        except Exception as e:
            logger.debug(f"Failed to parse metadata: {e}")
        
        # Profiling: Log output tensor details if available
        if response.tensors and len(response.tensors) > 0:
            output_tensor = response.tensors[0]
            output_size_bytes = 0
            
            # 尝试多种方式获取数据大小
            if hasattr(output_tensor, 'data'):
                output_size_bytes = len(output_tensor.data)
            elif hasattr(output_tensor, 'serialized_data'):
                output_size_bytes = len(output_tensor.serialized_data)
            elif hasattr(output_tensor, 'buffer'):
                output_size_bytes = len(output_tensor.buffer)
                
            logger.info(
                f"[PROFILING] Server-to-Client Transmission: "
                f"From server {self.span.peer_id.to_base58()}, "
                f"Session: {self.session_id}, "
                f"Step: {step_id}, "
                f"Size: {output_size_bytes / (1024*1024):.2f} MB, "
                f"Processing Time: {processing_time*1000:.2f} ms, "
                f"Throughput: {output_size_bytes / (1024*1024) / processing_time:.2f} MB/s" if output_size_bytes > 0 else
                f"[PROFILING] Server-to-Client Transmission: "
                f"From server {self.span.peer_id.to_base58()}, "
                f"Session: {self.session_id}, "
                f"Step: {step_id}, "
                f"Processing Time: {processing_time*1000:.2f} ms"
            )
        
        return response

    def close(self):
        """Finish a given inference session, close the underlying connection"""
        if self._outputs_stream is None:
            return  # already closed
        RemoteExpertWorker.run_coroutine(self._aclose_stream())
        self._outputs_stream = self._inputs_queue = None
        self.closed = True

    async def _aclose_stream(self):
        """Close the inference session. This code is meant to be run inside RemoteExpertWorker"""
        if self._outputs_stream is None:
            return  # already closed
        if self.stepped:
            await self._inputs_queue.put(runtime_pb2.ExpertRequest())  # empty request will trigger end of session
            try:
                await anext(self._outputs_stream)
            except StopAsyncIteration:
                pass

    def __del__(self):
        self.close()

    def __enter__(self):
        assert not self.closed
        return self

    def __exit__(self, *exc_details):
        self.close()


class InferenceSession:
    """
    An interface to a multi-step *inference* session for a sequence of remote transformer blocks
    """

    def __init__(self, sequence_manager: RemoteSequenceManager, max_length: int):
        self._sequence_manager = sequence_manager
        self._closed = False
        self._server_sessions = []
        self._position = 0
        self._max_length = max_length
        self.output_ids = None
        self.past_key_values = None
        
        # Profiling: Initialize statistics
        self._total_tokens = 0
        self._total_e2e_time = 0
        self._total_server_time = 0
        self._total_transmission_time = 0
        self._step_count = 0
        
        # Profiling: Initialize latency tracking
        self._latency_stats = {
            "e2e": [],
            "server": [],
            "transmission": [],
            "per_token": []
        }

    @property
    def num_blocks(self) -> int:
        return len(self._sequence_manager)

    @property
    def position(self) -> int:
        return self._position

    @position.setter
    def position(self, start_from_position: int) -> None:
        self._position = start_from_position
        for session in self._server_sessions:
            assert isinstance(session, _ServerInferenceSession)
            session.position = start_from_position

    def _enter_server_sessions(self, chosen_spans: List[RemoteSpanInfo]) -> List[_ServerInferenceSession]:
        server_sessions = []
        try:
            for span in chosen_spans:
                span_uids = CHAIN_DELIMITER.join(self._sequence_manager.block_uids[span.start : span.end])
                metadata = self._sequence_manager.get_request_metadata("rpc_inference", span_uids, peer_id=span.peer_id)
                session = RemoteExpertWorker.run_coroutine(
                    _ServerInferenceSession.create(
                        self._sequence_manager.config,
                        self._sequence_manager.state.p2p,
                        span,
                        span_uids,
                        rpc_info=self._sequence_manager.rpc_info,
                        max_length=self._max_length,
                        **metadata,
                    )
                )
                server_sessions.append(session)
                session.__enter__()
            return server_sessions
        except:
            self._exit_server_sessions(server_sessions)
            raise

    def _exit_server_sessions(self, server_sessions: List[_ServerInferenceSession]) -> None:
        for session in reversed(server_sessions):
            try:
                session.__exit__(None, None, None)
            except Exception:
                logger.debug("Caught exception while closing connection to server:", exc_info=True)

    def __enter__(self) -> "InferenceSession":
        assert not self._closed and not self._server_sessions
        return self

    def step(
        self,
        inputs: torch.Tensor,
        prompts: Optional[torch.Tensor] = None,
        hypo_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert not self._closed
        if torch.is_grad_enabled():
            logger.warning("Running inference session with grad enabled. Gradients will *not* be propagated correctly.")

        # Profiling: Record start time for end-to-end latency
        e2e_start_time = time.time()

        if prompts is None or is_dummy(prompts):
            prompts = DUMMY
        else:
            assert prompts.ndim == 4, "deep prompts should have shape [num_blocks, batch_size, prefix_len, hid_size]"
            assert prompts.shape[0] == self.num_blocks
            assert prompts.shape[1] in (inputs.shape[0], 1)
            assert prompts.shape[2] <= inputs.shape[1]
            assert prompts.shape[3] == inputs.shape[2]

        if hypo_ids is None or is_dummy(hypo_ids):
            hypo_ids = DUMMY_INT64
        else:
            assert len(hypo_ids) == len(inputs)
            assert hypo_ids.dtype == torch.int64

        inputs_device = inputs.device
        inputs_dtype = inputs.dtype
        inputs = inputs.cpu()
        prompts = prompts.cpu()
        hypo_ids = hypo_ids.cpu()
        step_id = str(uuid.uuid4())

        n_input_tokens = inputs.shape[1]
        if self._position + n_input_tokens > self._max_length:
            raise ValueError(
                f"Maximum length exceeded: prefix {self._position} + current {n_input_tokens} exceeds pre-allocated maximum {self._max_length}"
            )

        server_idx = 0
        block_idx = 0
        while block_idx < self.num_blocks:
            for attempt_no in itertools.count():
                logger.debug(f"Inference: block {block_idx}, attempt {attempt_no}")
                server_session = None
                try:
                    if not self._server_sessions or attempt_no >= 1:
                        self._update_sequence(server_idx, block_idx, attempt_no)

                    server_session = self._server_sessions[server_idx]
                    assert server_session.position == self.position, f"{server_session.position} and {self.position}"
                    inputs = server_session.step(
                        inputs,
                        prompts[server_session.span.start : server_session.span.end],
                        hypo_ids,
                        step_id=step_id,
                    )

                    server_idx += 1
                    block_idx = server_session.span.end
                    self._sequence_manager.on_request_success(server_session.span.peer_id)
                    break
                except Exception as e:
                    self._sequence_manager.on_request_failure(
                        server_session.span.peer_id if server_session is not None else None
                    )
                    if attempt_no + 1 == self._sequence_manager.config.max_retries:
                        raise
                    delay = self._sequence_manager.get_retry_delay(attempt_no)
                    logger.warning(
                        f"Caught exception when running inference via {server_session.span if server_session is not None else None} "
                        f"(retry in {delay:.0f} sec): {repr(e)}"
                    )
                    maybe_log_traceback(e)
                    time.sleep(delay)

        self._position += n_input_tokens
        outputs = inputs[:, -n_input_tokens:]
        outputs = outputs.to(device=inputs_device, dtype=inputs_dtype)
        
        # Profiling: Calculate end-to-end latency
        e2e_latency = time.time() - e2e_start_time
        
        # Profiling: Update statistics
        self._total_tokens += n_input_tokens
        self._total_e2e_time += e2e_latency
        self._step_count += 1
        self._latency_stats["e2e"].append(e2e_latency)
        self._latency_stats["per_token"].append(e2e_latency / n_input_tokens)
        
        # Profiling: Log end-to-end latency
        logger.info(
            f"[PROFILING] End-to-End Latency: "
            f"Session: {self._server_sessions[0].session_id if self._server_sessions else 'unknown'}, "
            f"Step: {step_id}, "
            f"Total Time: {e2e_latency*1000:.2f} ms, "
            f"Tokens: {n_input_tokens}, "
            f"Latency per Token: {e2e_latency*1000/n_input_tokens:.2f} ms/token"
        )
        
        # Profiling: Log cumulative statistics
        if self._step_count % 10 == 0 or self._step_count == 1:  # Log every 10 steps or on first step
            avg_e2e_latency = self._total_e2e_time / self._step_count
            avg_per_token = self._total_e2e_time / self._total_tokens
            logger.info(
                f"[PROFILING] Cumulative Statistics: "
                f"Steps: {self._step_count}, "
                f"Total Tokens: {self._total_tokens}, "
                f"Avg E2E Latency: {avg_e2e_latency*1000:.2f} ms, "
                f"Avg Latency per Token: {avg_per_token*1000:.2f} ms/token"
            )
        
        return outputs

    def _update_sequence(self, server_idx: int, block_idx: int, attempt_no: int) -> int:
        # If there is a failed server session, this code closes it
        self._exit_server_sessions(self._server_sessions[server_idx : server_idx + 1])

        n_prev_spans = len(self._server_sessions)
        update_end = self._server_sessions[server_idx].span.end if server_idx < n_prev_spans else self.num_blocks
        if attempt_no >= 1:
            logger.debug(
                f"Due to a server failure, remote attention caches "
                f"from block {block_idx} to {update_end} will be regenerated"
            )

        updated_spans = self._sequence_manager.make_sequence(
            block_idx, update_end, mode="min_latency", cache_tokens_needed=self._max_length
        )
        # make_sequence() could return a longer sequence
        updated_spans[-1].end = min(updated_spans[-1].end, update_end)
        updated_sessions = self._enter_server_sessions(updated_spans)
        logger.debug(f"Found path from block {block_idx} to {update_end} via {len(updated_spans)} servers")

        # If there is a failed span, this code replaces it, otherwise it just adds new ones
        if server_idx < n_prev_spans:
            updated_sessions[0].history = self._server_sessions[server_idx].history
        self._server_sessions[server_idx : server_idx + 1] = updated_sessions

        # Update links to the next server session for direct server-to-server communication via rpc_push()
        # Ensure all server sessions are properly linked
        for i in range(len(self._server_sessions) - 1):
            self._server_sessions[i].next_session = self._server_sessions[i + 1]

    def close(self, *exc_details):
        """Finish a given inference session, close the underlying connection"""
        if not self._closed:
            # Profiling: Log final statistics
            if self._step_count > 0:
                avg_e2e_latency = self._total_e2e_time / self._step_count
                avg_per_token = self._total_e2e_time / self._total_tokens
                min_per_token = min(self._latency_stats["per_token"]) * 1000
                max_per_token = max(self._latency_stats["per_token"]) * 1000
                p95_per_token = sorted(self._latency_stats["per_token"])[int(0.95 * len(self._latency_stats["per_token"]))] * 1000
                
                logger.info(
                    f"[PROFILING] Final Statistics: "
                    f"Steps: {self._step_count}, "
                    f"Total Tokens: {self._total_tokens}, "
                    f"Avg E2E Latency: {avg_e2e_latency*1000:.2f} ms, "
                    f"Avg Latency per Token: {avg_per_token*1000:.2f} ms/token, "
                    f"Min Latency per Token: {min_per_token:.2f} ms/token, "
                    f"Max Latency per Token: {max_per_token:.2f} ms/token, "
                    f"P95 Latency per Token: {p95_per_token:.2f} ms/token"
                )
            
            self._exit_server_sessions(self._server_sessions)
            self._server_sessions.clear()
            self._closed = True

    def __exit__(self, *exc_details):
        self.close(*exc_details)

    def __del__(self):
        self.close()

    @property
    def last_token_id(self) -> Optional[torch.Tensor]:  # Backward compatibility with Petals < 2.1.0
        return self.output_ids[:, -1:] if self.output_ids is not None else None

    @last_token_id.setter
    def last_token_id(self, value: torch.Tensor):  # Backward compatibility with Petals < 2.1.0
        if self.output_ids is None:
            raise RuntimeError("Can't override `last_token_id` since the session has not stepped yet")
        self.output_ids[:, -1:] = value
