Here, we're going to show you how we can wrap a custom Pytorch model with NOS and optimize it for inference. Let's start with a simple `CustomModel` that we implement with Pytorch.


```python
from typing import Union, List
from PIL import Image

import numpy as np
import torch
from torch import nn

class CustomModel(nn.Module):
	def __init__(self):
		self.device = "gpu" if torch.cuda.is_available() else "cpu"
		self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
		self.model.eval()

	def forward(self, images: Union[Image.Image, np.ndarray]):
		return self.model.visual(images)
```

### Wrapping a Custom Pytorch Model with `nos.trace`

```python
from nos.common import ModelSpec, TaskType

# Get the model spec for remote execution
spec = ModelSpec.from_cls(
    CustomModel,
    init_args=(),
    init_kwargs={"model_name": "resnet18"},
    pip=["onnx", "onnxruntime", "pydantic<2"],
)
spec
```

We use Ray to orchestrate custom model workers with configure runtime environments. For example, if you want to run a model on a custom runtime environment, you can provide custom `pip` requirements (see [example](/docs/custom-model-support.md)).