
from doctest import Example
import numpy as np
import torch
from torchmetrics import AveragePrecision, Accuracy, StatScores

preds_idx = np.array([1, 0, 3, 2 ,2, 0])
preds = np.zeros((preds_idx.size, preds_idx.max() + 1))
preds[np.arange(preds_idx.size), preds_idx] = 1
preds = torch.from_numpy(preds)
print(preds)

# pred example
# tensor([[0., 1., 0., 0.],
#         [1., 0., 0., 0.],
#         [0., 0., 0., 1.],
#         [0., 0., 1., 0.],
#         [0., 0., 1., 0.],
#         [1., 0., 0., 0.]], dtype=torch.float64)

target_idx = np.array([1, 0, 3, 2 ,1, 0])
target = torch.from_numpy(target_idx)
print(target)
# target example
# tensor([1, 0, 3, 2, 1, 0])



# return value is in the form of: each class is a column(PFR, FNR, ACC)
def conpute_stat(preds, target, class_num):
  stat_scores = StatScores(reduce='macro', num_classes=class_num)
  stat = stat_scores(preds, target)
  FPR_stat = torch.div(stat[:, 1], (stat[:, 1] + stat[:, 2]))
  FNR_stat = torch.div(stat[:, 3], (stat[:, 3] + stat[:, 0]))
  acc_stat = torch.div((stat[:, 0] + stat[:, 2]), (stat[:, 3] + stat[:, 1] + stat[:, 0] + stat[:, 2]))
  return FPR_stat, FNR_stat, acc_stat


