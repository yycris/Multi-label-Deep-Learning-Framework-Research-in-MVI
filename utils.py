import transforms as T

# mean1, std1---AP
# mean2, std2---VP
class TrainTransforms:
    def __init__(self):
        self.train_transforms = T.Compose([
            T.RandomHorizontalFlip(prob=0.25),
            T.RandomVerticalFlip(prob=0.25),
            T.RandomRotation(10),
            T.ToTensor(),
            # T.Normalize([0.04488417, 0.04488417, 0.01361942],
            #             [0.14777726, 0.14777726, 0.11397097],
            #             [0.05089118, 0.05089118, 0.013708],
            #             [0.16724654, 0.16724654, 0.11444444]),
            T.Normalize([0.04488417, 0.04488417, 0.01361942, 0.00050994], [0.14777726, 0.14777726, 0.11397097, 0.00750913],
                        [0.05089118, 0.05089118, 0.013708, 0.00050994], [0.16724654, 0.16724654, 0.11444444, 0.00750913])
        ])
    def __call__(self, img_x, img_y):
        return self.train_transforms(img_x, img_y)

class ValidTransforms:
    def __init__(self):
        self.valid_transforms = T.Compose([
            T.ToTensor(),
            # T.Normalize([0.04488417, 0.04488417, 0.01361942],
            #             [0.14777726, 0.14777726, 0.11397097],
            #             [0.05089118, 0.05089118, 0.013708],
            #             [0.16724654, 0.16724654, 0.11444444]),
            T.Normalize([0.04488417, 0.04488417, 0.01361942, 0.00050994],
                        [0.14777726, 0.14777726, 0.11397097, 0.00750913],
                        [0.05089118, 0.05089118, 0.013708, 0.00050994],
                        [0.16724654, 0.16724654, 0.11444444, 0.00750913])
        ])

    def __call__(self, img_x, img_y):
        return self.valid_transforms(img_x, img_y)
