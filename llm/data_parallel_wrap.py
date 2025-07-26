import torch
import torch.distributed as dist
from torch import nn
from torch import Tensor
from typing import List, Optional, Dict, Tuple, Iterator


###########
# interface
###########
class DataParallel(nn.Module):
    def __init__(
        self,
        module: nn.Module,
        module_dtype,
        process_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        # wrap for old model
        super().__init__()
        self.module = module
        self.module_dtype = module_dtype

        # number of worker to perform data parallel
        self.dp_rank = dist.get_rank(process_group)
        self.dp_world_size = dist.get_world_size(process_group)
        self.dp_process_group = process_group

        # whether to synchronize gradients during backward pass. Set to False when using gradient accumulation
        self.require_backward_grad_sync = True

        # make all worker has the same init weights
        self._broadcast_weights()

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def _broadcast_weights(self):
        for param in self.parameters():
            dist.broadcast(param.data, src=0)

    def finish_gradient_synchronization(self):
        """wait until graident synchronization done
        usage:
            ...
            logits = ddp_model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            ddp_model.finish_gradient_synchronization()
            optimizer.step()
        """
        raise NotImplementedError


###########
# DataParallelReduceInterleaved
###########
class DataParallelReduceInterleaved(DataParallel):
    """use asynchronous all-reduce to hide communication cost"""

    def __init__(self, module, module_dtype, process_group):
        super().__init__(module, module_dtype, process_group)
        self.module = module

        self.reduce_handles = []
        self.register_backward_hook(self._allreduce_grads_async)

    def register_backward_hook(self, hook):
        """Registers a backward hook for all parameters of the model that require gradients."""
        # called every time a gradient with respect to the tensor is computed
        for p in self.module.parameters():
            if p.requires_grad is True:
                p.register_hook(hook)

    def _allreduce_grads_async(self, grad):
        """Performs an all-reduce operation to synchronize gradients across multiple processes."""
        # No synchronization needed during gradient accumulation, except at the final accumulation step.
        # dist.all_reduce not differentiable
        #   c10d::allreduce_: an autograd kernel was not registered to the Autograd key(s)
        if self.require_backward_grad_sync:
            grad.data /= self.dp_world_size
            handle = dist.all_reduce(
                grad.data,
                op=dist.ReduceOp.SUM,
                group=self.dp_process_group,
                async_op=True,
            )
            self.reduce_handles.append(handle)

    def finish_gradient_synchronization(self):
        for handle in self.reduce_handles:
            handle.wait()

        self.reduce_handles.clear()


###########
# DataParallelBucket
###########
class DataParallelBucket(DataParallel):
    def __init__(
        self,
        module: nn.Module,
        module_dtype=torch.float32,
        process_group: Optional[torch.distributed.ProcessGroup] = None,
        bucket_size_mb: float = 5.0,
        grad_type=torch.float32,
    ):
        """

        Args:
            module (nn.Module): the model to perform data parallel
            bucket_size (int, optional): how many number of parameters a bucket shall contain. Defaults to 5*1024*1024.
            grad_type (_type_, optional). Defaults to torch.float32.
        """
        super().__init__(module, module_dtype, process_group)

        if module_dtype == torch.float32:
            element_size = 4

        elif module_dtype == torch.float16:
            element_size = 2

        elif module_dtype == torch.bfloat16:
            element_size = 2

        else:
            raise NotImplementedError

        self.bucket_size = int(bucket_size_mb * 1024 * 1024) // element_size
        self.bucket_manager = BucketManager(
            module.parameters(), self.bucket_size, process_group, grad_type
        )
        self.register_backward_hook()

    def register_backward_hook(self):
        """
        Registers a backward hook to manually accumulate and synchronize gradients.
        This hook is executed after gradient accumulation

        The gradient accumulation functions are stored to prevent them from going out of scope.

        References:
        - https://github.com/NVIDIA/Megatron-LM/issues/690
        - https://pytorch.org/docs/stable/generated/torch.autograd.graph.Node.register_hook.html
        - https://arxiv.org/abs/2006.15704 (page 5)
        """
        self.grad_accs = []
        for param in self.module.parameters():
            if param.requires_grad:
                # Expand create a new node in computational graph s.t we can access its grad_fn.
                param_tmp = param.expand_as(param)

                # Get the gradient accumulator function.
                # next_functions is a tuple that contains the next functions to call during backpropagation
                #   content: ((function, input_index), (function, input_index), ...)
                # next function for grad_fn of the expand node is AccumulateGrad
                #
                # AccumulateGrad is triggered when PyTorch is ready to accumulate/store the final gradient for that parameter
                # after AccumulateGrad done, the gradient of a parameter is ready
                #
                # Calling flow
                # Multiple contributions accumulate: If a parameter is used multiple times, all partial gradients are computed
                # AccumulateGrad executes: This function runs once when all partial gradients for that parameter have been computed and are ready to be accumulated
                # Hook is triggered: the registered hook is called at this exact moment
                grad_acc_fn = param_tmp.grad_fn.next_functions[0][0]
                grad_acc_fn.register_hook(
                    self._make_param_hook(param, self.bucket_manager)
                )
                self.grad_accs.append(grad_acc_fn)

    def _make_param_hook(
        self, param: torch.nn.Parameter, bucket_manager: "BucketManager"
    ):
        """
        Creates the a hook for each parameter to handle gradient accumulation and synchronization.
        """

        def param_hook(*unused):
            """
            The hook called after the gradient is ready. It performs the following:
            1. Accumulates the gradient into the main gradient.
            2. Adds a post-backward callback to wait for gradient synchronization completion.
            3. Marks the parameter as ready for synchronization.
            """
            if param.requires_grad:
                assert param.grad is not None
                param.main_grad.add_(param.grad.data)  # accumulate the gradients
                param.grad = None

                # skip the gradient synchronization (gradient accumulation/PP micro batches)
                if self.require_backward_grad_sync:
                    bucket_manager.mark_param_as_ready(param)

        return param_hook

    def finish_gradient_synchronization(self):
        """
        A post-backward callback that waits for gradient synchronization to finish, then copies
        the synchronized gradients back to the parameters' grad attribute.

        This method is called after the backward pass and before the optimizer step.
        """
        if self.require_backward_grad_sync:
            # wait for all reduce complete
            self.bucket_manager.wait()
            # self._post_backward_callback_set = False
            # copy to params.grad so we can use the optimizer to update the parameters
            for p in self.module.parameters():
                if p.requires_grad:
                    p.grad = p.main_grad.to(
                        p.dtype
                    ).data.clone()  # In PyTorch, you cannot assign a gradient with one data type to a tensor of another data type.

            # now the accumulated gradient shall set to 0
            self.reset()

    def reset(self):
        self.bucket_manager.reset()


class BucketManager:
    def __init__(
        self,
        params: Iterator[torch.nn.Parameter],
        bucket_size: int,
        process_group: Optional[torch.distributed.ProcessGroup] = None,
        grad_type: torch.dtype = torch.float32,
    ):
        """
        Args:
            params (Iterator[torch.nn.Parameter]): all parameters of a model
            bucket_size (int): number of free parameters a bucket can has
            process_group (Optional[torch.distributed.ProcessGroup], optional): process group for data parallel. Defaults to None.
            grad_type (torch.dtype, optional): type for accumulative gradient. Defaults to torch.float32.
        """
        # Convert parameter generator to a list.
        self.params = list(params)
        # List of buckets.
        self.buckets = []
        self.process_group = process_group
        self.process_group_size = dist.get_world_size(group=self.process_group)
        # Actual sizes of each bucket.
        self.bucket_size = bucket_size
        self.bucket_sizes = None
        # Map each parameter to its corresponding bucket/place (start, end, bucket_idx).
        self.params_to_bucket_location: Dict[Tensor, Tuple[slice, int]] = {}
        # List of tensors to store gradients, one tensor per bucket.
        self.grad_data_list = []
        self.grad_type = grad_type
        # Divide gradients into buckets based on the provided bucket size.
        self._initialize_buckets()

    def _initialize_buckets(self) -> None:
        """Divides model parameters into buckets for gradient synchronization based on the bucket size."""
        # Assign parameters to buckets.
        cur_bucket_size = 0
        cur_bucket_idx = 0
        bucket_sizes: List[int] = []
        device = None
        for param in self.params:
            # handle the case when param is frozon for training
            if not param.requires_grad:
                continue

            # record device of a parameter
            if device is None:
                device = param.device

            # If the bucket is empty, add the parameter to the bucket.
            if cur_bucket_size == 0:
                self.params_to_bucket_location[param] = (
                    slice(0, param.numel()),
                    cur_bucket_idx,
                )  # (start_index, end_index, bucket_id)
                cur_bucket_size = param.numel()
                continue

            # If the parameter cannot fit in the current bucket, create a new bucket
            if cur_bucket_size + param.numel() > self.bucket_size:
                bucket_sizes.append(
                    cur_bucket_size
                )  # bookmark bucket size of last created bucket
                cur_bucket_idx += 1
                self.params_to_bucket_location[param] = (
                    slice(0, param.numel()),
                    cur_bucket_idx,
                )
                cur_bucket_size = param.numel()
            else:
                self.params_to_bucket_location[param] = (
                    slice(cur_bucket_size, cur_bucket_size + param.numel()),
                    cur_bucket_idx,
                )
                cur_bucket_size += param.numel()

        bucket_sizes.append(cur_bucket_size)
        self.bucket_sizes = bucket_sizes
        assert len(bucket_sizes) == cur_bucket_idx + 1

        # Gather information about the bucket sizes and the parameters in each bucket
        # bucket_sizes = [0] * (cur_bucket_idx + 1)
        buckets_to_params = [[] for _ in range(cur_bucket_idx + 1)]
        for param, (location, idx) in self.params_to_bucket_location.items():
            # bucket_location
            # bucket_sizes[idx] = max(bucket_sizes[idx], location.stop)
            buckets_to_params[idx].append(param)

        # Create tensors for storing gradients and initialize Bucket objects.
        for i in range(len(bucket_sizes)):
            self.grad_data_list.append(
                torch.zeros(bucket_sizes[i], dtype=self.grad_type, device=device)
            )
            self.buckets.append(
                Bucket(buckets_to_params[i], self.grad_data_list[i], self.process_group)
            )

        # Create gradient views for each parameter.
        for param in self.params[::-1]:
            if not param.requires_grad:
                continue
            location, bucket_id = self.params_to_bucket_location[param]
            bucket_grad = self.grad_data_list[bucket_id]
            # param.main_grad is used for gradient calculation
            param.main_grad = self._get_view_from_tensor(
                bucket_grad, param.shape, location.start, location.stop
            )

    def _get_view_from_tensor(
        self, tensor: torch.Tensor, shape: torch.Size, start: int, end: int
    ) -> torch.Tensor:
        return tensor[start:end].view(shape)

    def reset(self) -> None:
        # Reset all buckets by clearing the gradients and internal states.
        for bucket in self.buckets:
            bucket.reset()

    def wait(self) -> None:
        # Wait for all buckets to complete their gradient synchronization
        for bucket in self.buckets:
            bucket.wait()

    def mark_param_as_ready(self, param: torch.nn.Parameter) -> None:
        # Mark a parameter's gradient as ready for synchronization.
        bucket_idx = self.params_to_bucket_location[param][1]
        self.buckets[bucket_idx].mark_param_as_ready(param)


class Bucket:
    def __init__(
        self,
        params: List[torch.nn.Parameter],
        grad_data: torch.Tensor,
        process_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        # Set of parameters in this bucket.
        self.params = set(params)
        # Parameters that have their gradients ready for synchronization. launch all reduce when all parameters have gradients ready
        self.params_with_grad_ready = set()
        # Parameters that have their gradients ready for synchronization. launch all reduce when all parameters have gradients ready
        self.grad_data = grad_data  # parameter in the same bucket will have the same reduced gradient
        # Process group for gradient synchronization.
        self.process_group = process_group
        self.process_group_size = dist.get_world_size(group=self.process_group)
        # Handle for the async allreduce operation.
        self.handle = None

        self.reset()

    def sync_gradient(self) -> None:
        """Launch an asynchronous all-reduce operation to synchronize gradients across processes."""
        # previous sync aready done
        assert self.handle is None
        self.grad_data /= self.process_group_size

        # self.grad_data is both input and output of all_reduce
        # return Async work handle, if async_op is set to True
        self.handle = dist.all_reduce(
            self.grad_data, group=self.process_group, async_op=True
        )

    def reset(self) -> None:
        """Reset the bucket to its initial state. Typically called after the gradient synchronization is finished."""
        self.handle = None
        # Clear the set of parameters ready for gradient synchronization.
        self.params_with_grad_ready.clear()
        # Zero the gradient tensor.
        self.grad_data.zero_()

    def wait(self) -> None:
        """wait for the allreduce operation to finish"""
        assert (
            self.handle is not None
        ), "You should launch an allreduce operation before waiting for it to finish"
        # Block until the all-reduce operation finishes.
        self.handle.wait()

    def mark_param_as_ready(self, param: torch.nn.Parameter) -> None:
        """Mark a parameter as ready for gradient synchronization. Launches synchronization when all parameters in the bucket have their gradients ready."""
        # TODO: why grad_data not a set? -> it is a vector concated by all parameters in the bucket
        assert param in self.params and param not in self.params_with_grad_ready
        self.params_with_grad_ready.add(param)
        # When all parameters in the bucket have their gradients ready, synchronize gradients
        if len(self.params_with_grad_ready) == len(self.params):
            self.sync_gradient()
