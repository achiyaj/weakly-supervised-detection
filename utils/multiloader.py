from random import choices


class MultiLoader:
    def __init__(self, dataset_getters, sampling_ratios):
        self.dataset_getters = dataset_getters
        self.sampling_ratios = sampling_ratios

    def __iter__(self):
        self.datasets = [iter(dataset_getter()) for dataset_getter in self.dataset_getters]
        self.dataset_lengths = [len(self.datasets[i]) for i in range(len(self.datasets))]
        self.times_to_sample_left = self.sampling_ratios.copy()
        self.num_objs_left = [len(self.datasets[i]) * self.sampling_ratios[i] for i in range(len(self.datasets))]
        return self

    @staticmethod
    def normalize_to_dist(in_list):
        list_sum = sum(in_list)
        return [x / list_sum for x in in_list]

    def __next__(self):
        if sum(self.num_objs_left) == 0:
            raise StopIteration
        dset_idx = choices(range(len(self.datasets)), self.normalize_to_dist(self.num_objs_left))[0]
        cur_batch = next(self.datasets[dset_idx])
        self.num_objs_left[dset_idx] -= 1
        if (self.num_objs_left[dset_idx] % self.dataset_lengths[dset_idx] == 0) and \
                self.times_to_sample_left[dset_idx] > 0:
            self.datasets[dset_idx] = iter(self.dataset_getters[dset_idx]())
            self.times_to_sample_left[dset_idx] -= 1

        return cur_batch

    def __len__(self):
        return sum([len(self.datasets[i]) * self.sampling_ratios[i] for i in range(len(self.datasets))])
