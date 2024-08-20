from sparsimony import rigl
import torch.nn as nn


class SparsimonySparsity(object):

    def __init__(self, model, optimizer, sparsity_level):
        self.model = model
        self.optimizer = optimizer
        self.sparsity_level = sparsity_level

    def sparse_optimizer(self):
        ## T_end parameter should be adjusted to ~75% of the total
        # training steps
        sparsifier = rigl(
            self.optimizer, sparsity=self.sparsity_level, t_end=10000
        )
        sparse_config = [
            {"tensor_fqn": f"{fqn}.weight"}
            for fqn, module in self.model.named_modules()
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d)
        ]

        # print(self.model)
        print("SPARSE_CONFIG", sparse_config)
        # try:
        sparsifier.prepare(self.model, sparse_config)
        print("SPARSIFIER", sparsifier)
        print(self.model)
        # except:
        # import pdb
        # pdb.set_trace()
        # print("SPARSIFIER", sparsifier)
        return sparsifier
