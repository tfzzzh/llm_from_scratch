import torch
from torch import nn
from typing import List, Optional, Dict, Tuple, Iterable, Callable, Type
from torch.distributed import ProcessGroup
import torch.distributed as dist
import numpy as np


# [TODO]: support multi-group of parameters
# Assumption: only one group of parameter is used
class Zero(torch.optim.Optimizer):
    """
    This is an implementation of Zero1 distributed optimization method. This method reduces
        memory requirements for optimizer states from Psi to Psi / world_size

    Sketch of the method:
        1. partition learning parameters into each workers
        2. each worker update the learning parameters using a local optimizer (e.g. AdamW)
        3. use gather and make model parameter be the same for each worker
    """
    def __init__(
        self,
        params: Iterable[nn.Parameter],
        group: Optional[ProcessGroup],
        optimizer_cls: Type[torch.optim.Optimizer],
        **kwargs,
    ):
        assert "lr" in kwargs

        # collect parameters which requires graidents
        self.params = [param for param in params if param.requires_grad]
        assert len(self.params) > 0, "no params requires gradients"

        # partitioning parameters (init)
        world_size = dist.get_world_size(group)
        rank = dist.get_rank(group)
        self.world_size = world_size
        self.rank = rank
        self.pg_group = group

        # chunk parameters and pair sliced parameter to its real version
        # chunk_ranges index (param_idx, rank)
        self.chunk_ranges: List[Dict[int, Tuple[int, int]]] = [
            get_partition_layout(x, world_size) for x in self.params
        ]
        self.params_data_chunked: List[Dict[int, torch.Tensor]] = []
        self.params_chunked_rank: List[nn.Parameter] = []
        self.params_chunked_full: List[Tuple[nn.Parameter, int, int]] = []
        self.chunk_parameters_by_rank(rank)

        # create suboptimizers
        self.child_opt = optimizer_cls(self.params_chunked_rank, **kwargs)
        assert len(self.child_opt.param_groups) == 1

        # create wrapped parameters
        # the learning rate will forward to the suboptimizer
        lr = kwargs["lr"]
        super().__init__(self.params, dict(lr=lr))
        assert len(self.param_groups) == 1

    def chunk_parameters_by_rank(self, rank):
        for i in range(len(self.params)):
            param = self.params[i]
            self.params_data_chunked.append({})
            for rk, (start, end) in self.chunk_ranges[i].items():
                data_chunked = param.data.view(-1)[start:end]
                self.params_data_chunked[i][rk] = data_chunked

                if rk == rank:
                    param_rank = nn.Parameter(data_chunked)
                    self.params_chunked_rank.append(param_rank)
                    self.params_chunked_full.append((param, start, end))

        # each optimizer shall have at least one tensor to optimize
        assert len(self.params_chunked_rank) == len(self.params_chunked_full)
        assert len(self.params_chunked_rank) > 0

    def step(self, closure: Optional[Callable] = None):
        # set learning rate
        lr = self.param_groups[0]["lr"]
        self.child_opt.param_groups[0]["lr"] = lr

        # collect gradients
        for i in range(len(self.params_chunked_rank)):
            param, start, end = self.params_chunked_full[i]
            assert param.grad is not None
            self.params_chunked_rank[i].grad = param.grad.view(-1)[start:end]

        # perform real optimizer
        loss = self.child_opt.step(closure)

        # perform gather
        self._collect_parameters()

        return loss

    def _collect_parameters(self):
        output_tensor_lists: List[List[torch.Tensor]] = []
        input_tensor_list: List[torch.Tensor] = []
        device = self.params[0].device
        dtype = self.params[0].dtype

        for i in range(len(self.params)):
            output_tensor_lists.append([])

            for rk in range(self.world_size):
                if rk in self.chunk_ranges[i]:
                    data_chunked = self.params_data_chunked[i][rk]

                else:
                    data_chunked = torch.empty(0, device=device, dtype=dtype)

                output_tensor_lists[i].append(data_chunked)
                if rk == self.rank:
                    input_tensor_list.append(data_chunked)

        all_gather_coalesced(output_tensor_lists, input_tensor_list, self.pg_group)


def get_chunk_size(length: int, num_worker: int) -> int:
    return (length - 1) // num_worker + 1


def get_partition_layout(
    x: torch.Tensor, world_size: int
) -> Dict[int, Tuple[int, int]]:
    n = x.numel()
    assert n > 0, "only non empty tensors are supported"
    chunk_size = get_chunk_size(n, world_size)
    end_points = np.zeros(world_size + 1, dtype=np.int64)
    end_points[1:] = chunk_size
    end_points = np.cumsum(end_points)

    assert end_points[-1] >= n
    layouts = {}
    for i in range(world_size):
        start = int(end_points[i])
        end = min(n, int(end_points[i + 1]))

        if start < end:
            layouts[i] = (start, end)

    return layouts


def all_gather_coalesced(
    output_tensor_lists: List[List[torch.Tensor]],
    input_tensor_list: List[torch.Tensor],
    group: Optional[ProcessGroup] = None,
    async_op: bool = False,
) -> Optional[torch._C.Future]:
    """
    This function support different length all_gather even for gloo backend

    All gathers a list of tensors to all processes in a group.

    Args:
        output_tensor_lists (list[list[Tensor]] OUT): Output tensor. index means: [param_idx][group_rank]
        input_tensor_list (list[Tensor]): List of tensors to all_gather from. index means [param_idx]
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        async_op (bool, optional): Whether this op should be an async op.

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group.

    """
    assert len(output_tensor_lists) > 0
    assert len(input_tensor_list) == len(output_tensor_lists)
    device = input_tensor_list[0].device
    dtype = input_tensor_list[0].dtype
    group_size = len(output_tensor_lists[0])

    assert (
        group_size > 1
    ), "You should probably not call `all_gather_coalesced` with a single rank, as it copies data over"
    assert group_size == dist.get_world_size(group)

    for input_tensor in input_tensor_list:
        assert device == input_tensor.device
        assert dtype == input_tensor.dtype

    for output_tensor_list in output_tensor_lists:
        assert len(output_tensor_list) == group_size
        for output_tensor in output_tensor_list:
            assert device == output_tensor.device
            assert dtype == output_tensor.dtype

    # Invert from `[param_idx][group_rank]` to `[group_rank][param_idx]`
    output_tensor_lists = [
        [output_tensor_list[group_rank] for output_tensor_list in output_tensor_lists]
        for group_rank in range(group_size)
    ]

    input_tensor_buffer = torch._utils._flatten_dense_tensors(input_tensor_list)
    output_tensor_buffer_list = [
        torch._utils._flatten_dense_tensors(output_tensor_list)
        for output_tensor_list in output_tensor_lists
    ]

    # get lens
    rank = dist.get_rank(group)
    input_buffer_old_length = len(input_tensor_buffer)
    output_buffer_old_lengthes = [len(buffer) for buffer in output_tensor_buffer_list]
    assert input_buffer_old_length == output_buffer_old_lengthes[rank]

    # get max_length and padding
    pad_length = max(output_buffer_old_lengthes)
    input_tensor_buffer, output_tensor_buffer_list = pad_tensor_buffers(
        device, dtype, input_tensor_buffer, output_tensor_buffer_list, pad_length
    )

    # msg = f"rank={rank} output_tensor_buffer_list=\n{output_tensor_buffer_list}\ninput_tensor_buffer=\n{input_tensor_buffer}\n"
    # print(msg)

    work = dist.all_gather(
        output_tensor_buffer_list, input_tensor_buffer, group=group, async_op=async_op
    )

    # slice buffers
    for i in range(len(output_tensor_buffer_list)):
        output_tensor_buffer_list[i] = output_tensor_buffer_list[i][
            : output_buffer_old_lengthes[i]
        ]

    def update_output():
        for original_buffer_list, gathered_buffer_tensor in zip(
            output_tensor_lists, output_tensor_buffer_list
        ):
            for original_buffer, gathered_buffer in zip(
                original_buffer_list,
                torch._utils._unflatten_dense_tensors(
                    gathered_buffer_tensor, original_buffer_list
                ),
            ):
                original_buffer.copy_(gathered_buffer)

    if async_op is True:
        # register update to the future (what )
        return work.get_future().then(lambda fut: update_output())
    else:
        # No need to run `work.wait()` since `dist.reduce_scatter` already waits
        update_output()


def pad_tensor_buffers(
    device,
    dtype,
    input_tensor_buffer,
    output_tensor_buffer_list,
    pad_length,
):
    input_tensor_buffer_pad = torch.concat(
        [
            input_tensor_buffer,
            torch.zeros(
                pad_length - len(input_tensor_buffer), device=device, dtype=dtype
            ),
        ],
        dim=0,
    )
    output_tensor_buffer_list_pad = [
        torch.concat(
            [buffer, torch.zeros(pad_length - len(buffer), device=device, dtype=dtype)],
            dim=0,
        )
        for buffer in output_tensor_buffer_list
    ]

    return input_tensor_buffer_pad, output_tensor_buffer_list_pad
