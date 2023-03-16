import torchvision


class AverageMeter:
    """ Computes and stores the average and current value """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0
        self.avg = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count

class Preprocessor:
    """
    Preprocesses data use with sphereface.
    """
    
    def __init__(self):        
        """
        Variable
        -------
        transform_train : Torchvision Transform
            The transformations need for sphereface dataaugmentation. This
            includes the sphereface data normalization. ---This is only for
            training data as it also has the horizontal flip augmentation.---

        """
        self.train_tran = torchvision.transforms.Compose([
            torchvision.transforms.Resize(96),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(127.5, 128),
            torchvision.transforms.RandomHorizontalFlip(p=0.5)
        ])
        
        
        """
        Variable
        -------
        transform_test : Torchvision Transform
            The transformations need for sphereface dataaugmentation. This
            includes the sphereface data normalization.
        """
        self.test_tran = torchvision.transforms.Compose([
            torchvision.transforms.Resize(96),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(127.5, 128)
        ])