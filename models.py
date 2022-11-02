import torch.nn as nn
import efficientnet_pytorch
from se_block import SEBlock


class EfficientNet_b4(nn.Module):
    def __init__(self):
        super(EfficientNet_b4, self).__init__()
        self.model = efficientnet_pytorch.EfficientNet.from_pretrained('efficientnet-b4')
        self.seblock = SEBlock(1792)

        self.classifier_layer = nn.Sequential(

            nn.Linear(1792, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(0.6),
            nn.Linear(256, 128),
        )

    def forward(self, inputs):

        x = self.model.extract_features(inputs)
        x = self.seblock(x) * x
        # Pooling and final linear layer
        x = self.model._avg_pooling(x)
        x = x.flatten(start_dim=1)
        x = self.model._dropout(x)
        x = self.classifier_layer(x)
        return x
#
# class EfficientNet_b4_01(nn.Module):
#     def __init__(self):
#         super(EfficientNet_b4_01, self).__init__()
#         self.model = efficientnet_pytorch.EfficientNet.from_pretrained('efficientnet-b4')
#
#         self.classifier_layer = nn.Sequential(
#
#             nn.Linear(1792, 512),
#             nn.BatchNorm1d(512),
#             nn.Dropout(0.6),
#             nn.Linear(512, 128),
#         )
#
#     def forward(self, inputs):
#
#         x = self.model.extract_features(inputs)
#         # Pooling and final linear layer
#         x = self.model._avg_pooling(x)
#         x = x.flatten(start_dim=1)
#         x = self.model._dropout(x)
#         x = self.classifier_layer(x)
#         return x