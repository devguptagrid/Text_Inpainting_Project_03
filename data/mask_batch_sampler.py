from torch.utils.data import Sampler
import random

class MaskBatchSampler(Sampler):

    def __init(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

        
        # 🔥 Step 1: group indices by mask ratio
        self.groups = {}

        for idx in range(len(dataset)):

            sample = dataset[idx]
            ratio = float(sample["mask_ratio"])

            if ratio not in self.groups:
                self.groups[ratio] = []

            self.groups[ratio].append(idx)

    def __iter__(self):

        batches = []

        # 🔥 Step 2: create batches per group
        for ratio in self.groups:

            indices = self.groups[ratio]
            random.shuffle(indices)

            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                batches.append(batch)

        # 🔥 Step 3: shuffle batch order
        random.shuffle(batches)

        return iter(batches)

    def __len__(self):
        return len(self.dataset) // self.batch_size

