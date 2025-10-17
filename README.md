# ORIGEN: Zero-Shot 3D Orientation Grounding in Text-to-Image Generation

<!-- Title image -->
<p align="center">
  <img src="./assets/teaser.jpg" width="800"/>
</p>

<!-- Badges -->
<p align="center">
  <a href="https://arxiv.org/abs/2503.22194">
    <img src="https://img.shields.io/badge/arXiv-2503.22194-b31b1b.svg" />
  </a>
  <a href="https://origen2025.github.io/">
    <img src="https://img.shields.io/badge/Website-origen2025.github.io-blue.svg" />
  </a>
</p>

<!-- Authors -->
<p align="center">
  <a href="https://myh4832.github.io/">Yunhong Min*</a>,
  <a href="https://choidaedae.github.io/">Daehyeon Choi*</a>,
  <a href="https://32v.github.io/">Kyeongmin Yeo</a>,
  <a href="https://jyunlee.github.io/">Jihyun Lee</a>,
  <a href="https://mhsung.github.io">Minhyuk Sung</a>
  
  <p align="center">
    KAIST
</p>


## 💡 Abstract

> We introduce ORIGEN, the first zero-shot method for 3D orientation grounding in text-to-image generation across multiple objects and diverse categories. While previous work on spatial grounding in image generation has mainly focused on 2D positioning, it lacks control over 3D orientation. To address this, we propose a reward-guided sampling approach using a pretrained discriminative model for 3D orientation estimation and a one-step text-to-image generative flow model. While gradient-ascent-based optimization is a natural choice for reward-based guidance, it struggles to maintain image realism. Instead, we adopt a sampling-based approach using Langevin dynamics, which extends gradient ascent by simply injecting random noise--requiring just a single additional line of code. Additionally, we introduce adaptive time rescaling based on the reward function to accelerate convergence. Our experiments show that ORIGEN outperforms both training-based and test-time guidance methods across quantitative metrics and user studies.

<!-- News -->
## 🔥 News
- **[2025.10]** We have released the implementation of *ORIGEN: Zero-Shot 3D Orientation Grounding in Text-to-Image Generation*.
- **[2025.09]** ORIGEN is accepted to NeurIPS 2025!

<!-- Code -->
## 🚀 Code

### Setup

Create a Conda environment:

```
conda create -n origen python=3.10
conda activate origen
```

Clone this repository:
```
git clone https://github.com/KAIST-Visual-AI-Group/ORIGEN.git
cd ORIGEN
```

Install PyTorch and requirements:

```
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

### Inference

You can run ORIGEN using the following commands. We provide example data files for both single- and multi-object orientation grounding in `data/single.json` and `data/multi.json`. You may modify these files to define your own orientation conditions. Additionally, you can visualize the estimated orientation and corresponding reward values on the generated images by adding the `--save_reward` option.

**Single-Object**

```
CUDA_VISIBLE_DEVICES={$DEVICE} python3 main.py --save_reward --config config/orientation_grounding.yaml --data_path data/single.json
```

**Multi-Object**

```
CUDA_VISIBLE_DEVICES={$DEVICE} python3 main.py --save_reward --config config/orientation_grounding.yaml --data_path data/multi.json
```

### Running ORIBENCH

We provide curated benchmark datas for 3D orientation grounding in `data/ORIBENCH_single.json` and `data/ORIBENCH_multi.json`. You can run this benchmark datas using the following commands.

**ORIBENCH-Single**

```
CUDA_VISIBLE_DEVICES={$DEVICE} python3 bench_main.py --config config/orientation_grounding.yaml --data_path data/ORIBENCH_single.json
```

**ORIBENCH-Multi**

```
CUDA_VISIBLE_DEVICES={$DEVICE} python3 bench_main.py --config config/orientation_grounding.yaml --data_path data/ORIBENCH_multi.json
```


## Citation
If you find our code helpful, please consider citing our work:
```
@article{min2025origen,
  title={ORIGEN: Zero-Shot 3D Orientation Grounding in Text-to-Image Generation},
  author={Min, Yunhong and Choi, Daehyeon and Yeo, Kyeongmin and Lee, Jihyun and Sung, Minhyuk},
  journal={arXiv preprint arXiv:2503.22194},
  year={2025}
}
```