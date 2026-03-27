
::: {.cell .markdown}

## Measure inference performance of PyTorch model on CPU 

First, we are going to measure the inference performance of an already-trained PyTorch model on CPU. After completing this section, you should understand:

* how to measure the inference latency of a PyTorch model
* how to measure the throughput of batch inference of a PyTorch model
* how to compare eager model execution vs a compiled model

You will execute this notebook *in a Jupyter container running on a compute instance*, not on the general-purpose Chameleon Jupyter environment from which you provision resources.

:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import time
import numpy as np
import clip
```
:::


::: {.cell .markdown}

First, let's load our MLP head and the CLIP ViT-L/14 model (used to compute image embeddings). Note that for now, we will use the CPU for inference, not GPU.

:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
model_path = "models/aesthetic_mlp.pth"  
device = torch.device("cpu")
model = torch.load(model_path, map_location=device, weights_only=False)
model.eval()

# Load CLIP model for computing image embeddings
clip_model, clip_preprocess = clip.load("ViT-L/14", device=device)

def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)
```
:::

::: {.cell .markdown}

and also prepare our test dataset, using CLIP's own preprocessing:

:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
data_dir = os.getenv("AESTHETIC_DATA_DIR", "aesthetic-hub")
test_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'test'), transform=clip_preprocess)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
```
:::


::: {.cell .markdown}

We will measure:

* the size of the model on disk
* the latency when doing inference on single samples
* the throughput when doing inference on batches of data
* and the test accuracy

:::


::: {.cell .markdown}

#### Model size

We'll start with model size. Our `aesthetic_mlp.pth` is a lightweight MLP head (768 → 1024 → 128 → 64 → 16 → 1) that maps CLIP ViT-L/14 embeddings to aesthetic scores, so it is very small.
:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
model_size = os.path.getsize(model_path) 
print(f"Model Size on Disk: {model_size/ (1e6) :.2f} MB")
```
:::

::: {.cell .markdown}

#### Sample predictions

Next, we'll verify the model produces reasonable aesthetic scores. The MLP head takes CLIP ViT-L/14 embeddings (768-dim, L2-normalized) and outputs aesthetic quality scores (0-10).

:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
with torch.no_grad():
    images, _ = next(iter(test_loader))
    image_features = clip_model.encode_image(images.to(device))
    embeddings = torch.from_numpy(normalized(image_features.cpu().numpy())).float().to(device)
    scores = model(embeddings).squeeze()
    mean_score = scores.mean().item()
    std_score = scores.std().item()
```
:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
print("Sample predicted aesthetic scores (0-10):")
for i in range(min(5, len(scores))):
    print(f"  Image {i+1}: {scores[i].item():.2f}")
print(f"\nBatch mean: {mean_score:.2f}, std: {std_score:.2f}")
```
:::



::: {.cell .markdown}

#### Inference latency

Now, we'll measure how long it takes the model to return a prediction for a single sample. We will run 100 trials, and then compute aggregate statistics.

:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
num_trials = 100  # Number of trials

# Pre-compute a single CLIP embedding for benchmarking MLP latency
with torch.no_grad():
    sample_image, _ = next(iter(test_loader))
    sample_features = clip_model.encode_image(sample_image[:1].to(device))
    single_embedding = torch.from_numpy(normalized(sample_features.cpu().numpy())).float().to(device)

# Warm-up run 
with torch.no_grad():
    model(single_embedding)

latencies = []
with torch.no_grad():
    for _ in range(num_trials):
        start_time = time.time()
        _ = model(single_embedding)
        latencies.append(time.time() - start_time)
```
:::



::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
print(f"Inference Latency (single sample, median): {np.percentile(latencies, 50) * 1000:.2f} ms")
print(f"Inference Latency (single sample, 95th percentile): {np.percentile(latencies, 95) * 1000:.2f} ms")
print(f"Inference Latency (single sample, 99th percentile): {np.percentile(latencies, 99) * 1000:.2f} ms")
print(f"Inference Throughput (single sample): {num_trials/np.sum(latencies):.2f} FPS")
```
:::

::: {.cell .markdown}

#### Batch throughput 

Finally, we'll measure the rate at which the model can return predictions for batches of data. 

:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
num_batches = 50  # Number of trials

# Pre-compute a batch of CLIP embeddings for benchmarking MLP throughput
with torch.no_grad():
    batch_images, _ = next(iter(test_loader))
    batch_features = clip_model.encode_image(batch_images.to(device))
    batch_embeddings = torch.from_numpy(normalized(batch_features.cpu().numpy())).float().to(device)

# Warm-up run 
with torch.no_grad():
    model(batch_embeddings)

batch_times = []
with torch.no_grad():
    for _ in range(num_batches):
        start_time = time.time()
        _ = model(batch_embeddings)
        batch_times.append(time.time() - start_time)
```
:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
batch_fps = (batch_embeddings.shape[0] * num_batches) / np.sum(batch_times) 
print(f"Batch Throughput: {batch_fps:.2f} FPS")
```
:::


::: {.cell .markdown}

#### Summary of results

:::

::: {.cell .code}
```python
# runs in jupyter container on node-serve-model
print(f"Model Size on Disk: {model_size/ (1e6) :.2f} MB")
print(f"Mean Predicted Score: {mean_score:.2f} (std: {std_score:.2f})")
print(f"Inference Latency (single sample, median): {np.percentile(latencies, 50) * 1000:.2f} ms")
print(f"Inference Latency (single sample, 95th percentile): {np.percentile(latencies, 95) * 1000:.2f} ms")
print(f"Inference Latency (single sample, 99th percentile): {np.percentile(latencies, 99) * 1000:.2f} ms")
print(f"Inference Throughput (single sample): {num_trials/np.sum(latencies):.2f} FPS")
print(f"Batch Throughput: {batch_fps:.2f} FPS")
```
:::

<!-- 

compute_gigaio 

  Model name:             AMD EPYC 7763 64-Core Processor
    CPU family:           25
    Model:                1
    Thread(s) per core:   2
    Core(s) per socket:   64

-->


<!-- summary for aesthetic_mlp model

Model Size on Disk: 9.23 MB
Accuracy: 90.59% (3032/3347 correct)
Inference Latency (single sample, median): 60.16 ms
Inference Latency (single sample, 95th percentile): 77.22 ms
Inference Latency (single sample, 99th percentile): 77.37 ms
Inference Throughput (single sample): 15.82 FPS
Batch Throughput: 83.66 FPS


Model Size on Disk: 9.23 MB
Accuracy: 90.59% (3032/3347 correct)
Inference Latency (single sample, median): 73.97 ms
Inference Latency (single sample, 95th percentile): 83.16 ms
Inference Latency (single sample, 99th percentile): 83.94 ms
Inference Throughput (single sample): 13.34 FPS
Batch Throughput: 98.80 FPS

-->


<!-- summary for aesthetic_mlp compiled model

Model Size on Disk: 9.23 MB
Accuracy: 90.59% (3032/3347 correct)
Inference Latency (single sample, median): 26.92 ms
Inference Latency (single sample, 95th percentile): 49.79 ms
Inference Latency (single sample, 99th percentile): 64.55 ms
Inference Throughput (single sample): 32.35 FPS
Batch Throughput: 249.08 FPS

Model Size on Disk: 9.23 MB
Accuracy: 90.59% (3032/3347 correct)
Inference Latency (single sample, median): 34.14 ms
Inference Latency (single sample, 95th percentile): 53.85 ms
Inference Latency (single sample, 99th percentile): 60.23 ms
Inference Throughput (single sample): 27.39 FPS
Batch Throughput: 281.65 FPS

-->

<!-- 

(Intel CPU)

Model Size on Disk: 9.23 MB
Accuracy: 90.59% (3032/3347 correct)
Inference Latency (single sample, median): 12.69 ms
Inference Latency (single sample, 95th percentile): 12.83 ms
Inference Latency (single sample, 99th percentile): 12.97 ms
Inference Throughput (single sample): 78.73 FPS
Batch Throughput: 161.27 FPS

With compiling

Model Size on Disk: 9.23 MB
Accuracy: 90.59% (3032/3347 correct)
Inference Latency (single sample, median): 8.47 ms
Inference Latency (single sample, 95th percentile): 8.58 ms
Inference Latency (single sample, 99th percentile): 8.79 ms
Inference Throughput (single sample): 117.86 FPS
Batch Throughput: 474.67 FPS



-->

::: {.cell .markdown}

When you are done, download the fully executed notebook from the Jupyter container environment for later reference. (Note: because it is an executable file, and you are downloading it from a site that is not secured with HTTPS, you may have to explicitly confirm the download in some browsers.)

:::


::: {.cell .markdown}

### Eager mode execution vs compiled model

We had just evaluated a model in eager mode. However, in some (although, not all) cases we may get better performance from compiling the model into a graph, and executing it as a graph.

Go back to the cell where the model is loaded, and add

```python
model.compile()
```

just below the call to `torch.load`. Then, run the notebook again ("Run > Run All Cells"). 

When you are done, download the fully executed notebook **again** from the Jupyter container environment for later reference.


:::
