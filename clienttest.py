from tritonclient.http import InferenceServerClient, InferInput, InferRequestedOutput
import numpy as np
import torch

# 创建客户端实例
client = InferenceServerClient(url="localhost:8000")

# 准备输入数据
input_data = np.zeros([2,3,640,640], dtype=np.float32) # 根据你的模型准备输入数据
infer_input = InferInput('images', input_data.shape, 'FP32')
infer_input.set_data_from_numpy(input_data)

# 准备输出
outputs = [InferRequestedOutput('output0')]

# 发送请求
response = client.infer(model_name='yolo',
                        inputs=[infer_input],
                        outputs=outputs)

# return {'model_name': 'yolo', 'model_version': '1', 'outputs': [{'name': 'output0', 'datatype': 'FP32', 'shape': [1, 84, 8400], 'parameters': {'binary_data_size': 2822400}}]}
print(response.as_numpy('output0').shape) 