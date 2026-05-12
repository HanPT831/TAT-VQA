<div align="center">

<h2>Tracing the Arrow of Time: <br>
 Diagnosing Temporal
Information Flow in Video-LLMs</h2>

</div>
This is the repository of Tracing the Arrow of Time (TAT): Diagnosing Temporal Information Flow in Video-LLMs. This repository has the code for fine-tuning Video-LLM with Arrow-of-Time (AoT) supervision, our current version is  preliminary, we will refine and release the ckeckpoint soon.

```
TAT
 |-- TAT-Probe (RQ1 and layer-wise probing)
 |-- TAT-VQA (RQ2 and RQ3)
```

- Prepare the envirment:
    ```shell
    conda create -n TAT
    conda activate TAT
    pip install -r requirements.txt
    ```
### AoT VQA

#### RQ2: Projector and Layer-Wise Fine-Tuning 
- AoT-only instruction tuning:
    - Download Something-something v2 dataset
    - Construct AoT VQA dataset using aot.py
    - Set `qwen_model_path` and `train_corpus:aot_only` in TAT_vqa/scripts/videochat_qwen/config_7b_aot_only_ivs1.py
    - Set `spatial_pool_shape` and `num_temporal_pool_tokens` for RQ2: projector analysis
    - Set `return_index` for RQ2: layer-wise analysis, which `-1` represents last layer
    - Download InternVideo2 model and modify the config of vision encoder
    ```shell
    bash scripts/videochat_qwen/run_7b_aot_only_ivs1.sh
    ```
#### RQ3: AoT Supervision For Broader Temporal Reasoning
- AoT & instruction joint tuning:
    - Download instruction data from Videogpt+
    - set `data_dir` in TAT_vqa/configs/instruction_data.py
    - Set `qwen_model_path` and `train_corpus:aot_mix` in TAT_vqa/scripts/videochat_qwen/config_7b_aot_only_ivs1.py
    - Download InternVideo2 model and modify the config of vision encoder in TAT_vqa/scripts/videochat_qwen/config_7b_aot_mix_ivs1.py
    ```shell
    bash scripts/videochat_qwen/run_7b_aot_only_ivs1.sh
    ```

- Runing evaluation:
    ```shell
    # AoT-PPB 
    python demo/demo_qwen.py
    # TVBench
    python demo/demo_tvbench_qwen.py
    ```

# Acknowledgement

Our code is based on following projects: [VideoChat2](https://github.com/OpenGVLab/Ask-Anything/tree/main/video_chat2), [InternVideo2](https://github.com/OpenGVLab/InternVideo/tree/main/InternVideo2),
[VideoGPT+](https://github.com/mbzuai-oryx/videogpt-plus)
