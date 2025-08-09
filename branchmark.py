from tritonclient.http import InferenceServerClient, InferInput, InferRequestedOutput
import numpy as np
import torch
import time
import matplotlib.pyplot as plt

def preper_data(batch_size=1):
    inputdata = np.random.random([batch_size, 3, 640, 640], dtype=np.float32)
    input = [InferInput('images', inputdata.shape, 'FP32')]
    input[0].set_data_from_numpy(inputdata)
    output = [InferRequestedOutput('output0')]
    return input, output

def branchmark(client:InferenceServerClient, model_name:str, batch_size=1, test_times=10000):
    times = []
    all_start = time.time()
    for i in test_times:
        input, output = preper_data(batch_size)
        one_time_start = time.time()
        response = client.infer(model_name=model_name,
                                inputs=input,
                                outputs=output
                                )
        one_time_end = time.time()
        times.append(one_time_end-one_time_start)
    all_done = time.time()

    avg_lantency = sum(times)/len(times)
    qps = test_times/sum(times)
    p99_lantency = np.percentile(times, 99)
    return avg_lantency*1000, qps, p99_lantency*1000

if __name__ == '__main__':
    print('test start!')

    client = InferenceServerClient(url='localhost:8000')
    model_name = 'yolo11'

    data = {
        'avg':[],
        'onePic_avg':[],
        'qps':[],
        'p99':[]
    }
    max_batch = 8
    for i in range(max_batch):
        avg, qps, p99 = branchmark(client, model_name, i)
        avg_for_one_pic = avg/i
        data['avg'].append(avg)
        data['onePic_avg'].append(avg_for_one_pic)
        data['qps'].append(qps)
        data['p99'].append(p99)

    configs = [i+1 for i in max_batch]
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    x = np.arange(max_batch)
    width = 0.35

    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Inference Performance Comparison Across Batches', fontsize=18, fontweight='bold')

    plots_data = {
    'avg':('Avg Latency', 'Average End-to-End Latency', 'Latency (ms)', 'skyblue', 'navy'),
    'onePic_avg':('Single Pic Avg Latency', 'Single Pic Average Latency', 'Latency (ms)', 'lightgreen', 'darkgreen'),
    'qps':('QPS', 'Queries Per Second', 'QPS', 'coral', 'red'),
    'p99':('P99', 'P99 Latency', 'Latency (ms)', 'violet', 'purple')
    }

    for k, v in data:
        axs[0, 0].bar(x - width/2, v, width, label=plots_data[k][2], color=plots_data[k][-2])
        axs[0, 0].plot(x, v, 'o-', color=plots_data[k][-1], linewidth=2, label='Trend')
        axs[0, 0].set_title(plots_data[k][1], fontsize=14)
        axs[0, 0].set_xlabel('Batch Configuration')
        axs[0, 0].set_ylabel(plots_data[k][2])
        axs[0, 0].set_xticks(x)
        axs[0, 0].set_xticklabels(configs, rotation=45, ha='right')  # 旋转标签避免重叠
        axs[0, 0].legend()
        axs[0, 0].grid(axis='y', alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    plt.savefig('./brenchmark.png', dpi=300, bbox_inches='tight')




