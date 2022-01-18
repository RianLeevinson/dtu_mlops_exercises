import torch
from torchvision.models import resnet50

model = resnet50(pretrained=True)
script_model = torch.jit.script(model)
script_model.save('deployable_model.pt')
