from typing import Optional, Dict, Any
import torch
import torch.nn as nn
from torch.ao.pruning.sparsifier.base_sparsifier import BaseSparsifier

from sparsimony.distributions.base import BaseDistribution
from sparsimony.schedulers.base import BaseScheduler
from sparsimony.parametrization.fake_sparsity import FakeSparsityDenseGradBuffer
from sparsimony.utils import get_mask, get_parametrization
from sparsimony.dst.base import DSTMixin
from sparsimony.pruners.unstructured import (
    UnstructuredMagnitudePruner,
    UnstructuredGradientGrower,
    UnstructuredRandomPruner,
)


class RigL(DSTMixin, BaseSparsifier):

    def __init__(
        self,
        scheduler: BaseScheduler,
        distribution: BaseDistribution,
        optimizer: torch.optim.Optimizer,
        defaults: Optional[Dict[str, Any]] = None,
        sparsity: float = 0.5,
        grown_weights_init: float = 0.0,
        init_method: Optional[str] = "grad_flow",
        *args,
        **kwargs,
    ):
        self.scheduler = scheduler
        self.distribution = distribution
        self.sparsity = sparsity
        self.grown_weights_init = grown_weights_init
        self.init_method = init_method
        if defaults is None:
            defaults = dict(parametrization=FakeSparsityDenseGradBuffer)
        super().__init__(
            optimizer=optimizer, defaults=defaults, *args, **kwargs
        )

    def prune_mask(
        self,
        target_sparsity: float,
        mask: torch.Tensor,
        weights: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        mask.data = UnstructuredMagnitudePruner.calculate_mask(
            target_sparsity, mask, weights
        )
        return mask

    def grow_mask(
        self,
        sparsity: float,
        mask: torch.Tensor,
        original_weights: torch.Tensor,
        dense_grads: torch.Tensor,
    ) -> torch.Tensor:
        # Grow new weights
        new_mask = UnstructuredGradientGrower.calculate_mask(
            sparsity, mask, dense_grads
        )
        assert new_mask.data_ptr() != mask.data_ptr()
        # Assign newly grown weights to self.grown_weights_init in place
        original_weights.data = torch.where(
            new_mask != mask,
            torch.full_like(
                original_weights, fill_value=self.grown_weights_init
            ),
            original_weights,
        )
        # Overwrite old mask
        mask.data = new_mask.data
        return mask

    def _step(self) -> bool:
        _topo_updated = False
        self._step_count += 1
        prune_ratio = self.scheduler(self._step_count)
        if prune_ratio is not None:
            self._logger.info(f"Updating topology at step {self._step_count}")
            self._distribute_sparsity(self.sparsity)
            for config in self.groups:
                parametrization = get_parametrization(
                    config["module"], config["tensor_name"]
                )
                if (
                    hasattr(parametrization, "is_replica_")
                    and parametrization.is_replica_
                ):
                    continue
                config["prune_ratio"] = prune_ratio
                config["dense_grads"] = self._get_dense_grads(**config)
                self.update_mask(**config)
            self._broadcast_masks()
            _topo_updated = True
        if self.scheduler.next_step_update(self._step_count):
            self._accumulate_grads()
        return _topo_updated

    def _get_dense_grads(
        self, module: nn.Module, tensor_name: str, **kwargs
    ) -> torch.Tensor:
        parametrization = getattr(module.parametrizations, tensor_name)[0]
        parametrization.accumulate = False
        return parametrization.dense_grad

    def _accumulate_grads(self) -> None:
        for config in self.groups:
            parametrization = get_parametrization(
                config["module"], tensor_name=config["tensor_name"]
            )
            parametrization.accumulate = True

    def _initialize_masks(self) -> None:
        self._distribute_sparsity(self.sparsity)
        for config in self.groups:
            # Prune to target sparsity for this step
            mask = get_mask(config["module"], config["tensor_name"])
            if self.random_mask_init:
                # Randomly prune for step 1
                mask.data = UnstructuredRandomPruner.calculate_mask(
                    config["sparsity"], mask
                )
            else:
                # use pruning criterion
                weights = getattr(config["module"], config["tensor_name"])
                mask.data = self.prune_mask(config["sparsity"], mask, weights)

    def update_mask(
        self,
        module: nn.Module,
        tensor_name: str,
        sparsity: float,
        prune_ratio: float,
        dense_grads: torch.Tensor,
        **kwargs,
    ):
        mask = get_mask(module, tensor_name)
        if sparsity == 0:
            mask.data = torch.ones_like(mask)
        else:
            original_weights = getattr(
                module.parametrizations, tensor_name
            ).original
            weights = getattr(module, tensor_name)
            target_sparsity = self.get_sparsity_from_prune_ratio(
                mask, prune_ratio
            )
            self.prune_mask(target_sparsity, mask, weights)
            self.grow_mask(sparsity, mask, original_weights, dense_grads)
            self._assert_sparsity_level(mask, sparsity)
