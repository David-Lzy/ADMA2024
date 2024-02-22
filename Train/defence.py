from CODE.Utils.package import *


class Defence(nn.Module):

    def __init__(
        self,
        mother_model: object,
        augmentation: object,
        aug_paramater: dict = dict(),
        **kwargs
    ):
        super(Defence, self).__init__()
        self.augmentation = augmentation
        self.mother_model = mother_model(**kwargs)
        self.aug_paramater = aug_paramater
        self.__class__.__name__ = self.mother_model.__class__.__name__

    def forward(self, x):
        x = self.augmentation(x, **self.aug_paramater)
        return self.mother_model(x)
