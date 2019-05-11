from fastai.vision import models, get_transforms, cnn_learner, ImageDataBunch, imagenet_stats, ClassificationInterpretation
from fastai.metrics import accuracy

PATH = 'DATA PATH'

tfms = get_transforms(flip_vert=True, max_lighting=0.1,
                      max_zoom=1.05, max_warp=0.1)
data = ImageDataBunch.from_folder(
    PATH, ds_tfms=tfms, bs=64, size=224, num_workers=4).normalize(imagenet_stats)

model = cnn_learner(data, models.resnet34, metrics=accuracy, pretrained=True)
model.fit_one_cycle(5)

model.save('foodnotfoodv1')
model.unfreeze()
model.lr_find()
model.recorder.plot()

model.fit_one_cycle(2, max_lr=slice(1e-5, 1e-4))

interp = ClassificationInterpretation.from_learner(model)
interp.plot_confusion_matrix()

model.export()
