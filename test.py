# Load model directly
from transformers import AutoModel
model = AutoModel.from_pretrained("prajjwal1/bert-tiny", torch_dtype="auto")
print("成功加载")