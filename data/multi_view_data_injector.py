from torchvision.transforms import transforms


class MultiViewDataInjector(object):
    def __init__(self, *args):
        self.transforms = args[0]
        self.random_flip = transforms.RandomHorizontalFlip()

    def __call__(self, sample, *with_consistent_flipping):
        output = [transform(sample) for transform in self.transforms]
        return output