# 11_hub_production.md

This notebook demonstrates a fully independent, production-ready pipeline for aesthetic scoring using the best settings for ViT, Global MLP, and Personalized MLP, including Triton Inference Server deployment. All setup steps are included—no prior notebooks required.

## Pipeline Overview

1. **Environment & Docker Setup**
   - Pull/build all required containers
   - Launch Triton with recommended settings (2 instances, dynamic batching)
2. **ViT Benchmark (Compiled, Batch=128)**
   - Run only the best production setting
3. **Global MLP Benchmark**
   - Use dynamic quantized ONNX model
   - Benchmark at optimal batch size
4. **Personalized MLP Benchmark**
   - Use graph-optimized ONNX model
   - Benchmark at optimal batch size
5. **Triton Serving Benchmarks**
   - Global and Personalized MLPs
   - 2 instances, dynamic batching (preferred_batch_size: [4,8,16,32,64,128])

## All steps are self-contained. No other notebooks are required.
