from tritonclient.http import InferenceServerClient, InferInput, InferRequestedOutput
import numpy as np
import torch
import time

# 创建客户端实例
client = InferenceServerClient(url="localhost:8000")

# 准备输入数据
input_data = np.zeros([8,3,640,640], dtype=np.float32) # 根据你的模型准备输入数据
infer_input = InferInput('images', input_data.shape, 'FP32')
infer_input.set_data_from_numpy(input_data)

# 准备输出
outputs = [InferRequestedOutput('output0')]


# 准备输入数据1
input_data1 = np.zeros([7,3,640,640], dtype=np.float32) # 根据你的模型准备输入数据
infer_input1 = InferInput('images', input_data1.shape, 'FP32')
infer_input1.set_data_from_numpy(input_data1)

# 准备输出1
outputs1 = [InferRequestedOutput('output0')]


# 发送请求
start = time.time()
response = client.infer(model_name='yolo11',
                        inputs=[infer_input],
                        outputs=outputs)

response1 = client.infer(model_name='yolo11',
                        inputs=[infer_input1],
                        outputs=outputs1)
end  = time.time()

# return {'model_name': 'yolo', 'model_version': '1', 'outputs': [{'name': 'output0', 'datatype': 'FP32', 'shape': [1, 84, 8400], 'parameters': {'binary_data_size': 2822400}}]}
print(response1.as_numpy('output0').shape) 
print((end-start)*1000)

# input [8, 3 , 640, 640] in yolo11n
# infer 1 time using 300ms
# infer 2 times using 500ms
