from random import choices


class MultiLoader:
    def __init__(self, datasets, sampling_ratios):
        self.datasets = datasets
        self.sampling_ratios = sampling_ratios

    def __iter__(self):
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
        self.num_objs_left[dset_idx] -= 1
        if self.num_objs_left[dset_idx] % len(self.datasets[dset_idx]) and self.times_to_sample_left[dset_idx] > 0:
            self.datasets[dset_idx] = iter(self.datasets[dset_idx])
            self.times_to_sample_left[dset_idx] -= 1

        return next(self.datasets[dset_idx])

    def __len__(self):
        return sum([len(self.datasets[i]) * self.sampling_ratios[i] for i in range(len(self.datasets))])