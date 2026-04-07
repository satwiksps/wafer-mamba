# Wafer-Mamba

This repository contains the official implementation and experimental artifacts for:

**"Hybrid Quantum-MambaVision: A Quantum-enhanced State Space Model for Calibrated Mixed-type Wafer Defect Detection"**

> **Accepted Paper** &mdash; [EFMxDM Workshop](https://efficient-fmxdm.github.io/EFMxDM/) at [PAKDD 2026](https://pakdd2026.org/), Hong Kong

## Abstract

We propose a hybrid quantum-classical architecture that augments [MambaVision-T](https://huggingface.co/nvidia/MambaVision-T-1K) with a **Quantum Context Adapter (QCA)** and **LoRA** fine-tuning for multi-label semiconductor wafer defect classification. The QCA injects a lightweight PennyLane-based variational quantum circuit between the backbone stages, providing a quantum-enhanced channel re-calibration signal whose influence is governed by a learnable gate scalar. Combined with Focal Loss and a cosine-annealed training schedule, the model achieves **97.84% subset accuracy** and **0.9947 Micro-F1** on the MixedType Wafer Defect dataset — outperforming classical MambaVision, ResNet-50 and ViT baselines while adding only **~5.8K** quantum-adapter parameters.


## Architecture

| Component | Description |
|---|---|
| **Backbone** | MambaVision-Tiny (`nvidia/MambaVision-T-1K`) — hybrid SSM + attention |
| **Quantum Context Adapter** | 4-qubit PennyLane circuit (AngleEmbedding → StronglyEntanglingLayers), inserted after Level 2 via forward hook |
| **LoRA** | Rank-64, α=128, applied to `in_proj`, `out_proj`, `x_proj`, `dt_proj`, `fc1`, `fc2` |
| **Classification Head** | LayerNorm → Linear (640 → 8) |
| **Loss** | Focal Loss (γ=2) |

<details>
  <summary>Click to view full architecture</summary>

| ID | Layer | Type | Output | Status |
| :--- | :--- | :--- | :--- | :--- |
| 0 | base_model.model.model.patch_embed.proj | Identity | [1, 3, 384, 384] | 🔒 Frozen |
| 1 | base_model.model.model.patch_embed.conv_down.0 | Conv2d | [1, 32, 192, 192] | 🔒 Frozen |
| 2 | base_model.model.model.patch_embed.conv_down.1 | BatchNorm2d | [1, 32, 192, 192] | 🔒 Frozen |
| 3 | base_model.model.model.patch_embed.conv_down.2 | ReLU | [1, 32, 192, 192] | 🔒 Frozen |
| 4 | base_model.model.model.patch_embed.conv_down.3 | Conv2d | [1, 80, 96, 96] | 🔒 Frozen |
| 5 | base_model.model.model.patch_embed.conv_down.4 | BatchNorm2d | [1, 80, 96, 96] | 🔒 Frozen |
| 6 | base_model.model.model.patch_embed.conv_down.5 | ReLU | [1, 80, 96, 96] | 🔒 Frozen |
| 7 | base_model.model.model.levels.0.blocks.0.conv1 | Conv2d | [1, 80, 96, 96] | 🔒 Frozen |
| 8 | base_model.model.model.levels.0.blocks.0.norm1 | BatchNorm2d | [1, 80, 96, 96] | 🔒 Frozen |
| 9 | base_model.model.model.levels.0.blocks.0.act1 | GELU | [1, 80, 96, 96] | 🔒 Frozen |
| 10 | base_model.model.model.levels.0.blocks.0.conv2 | Conv2d | [1, 80, 96, 96] | 🔒 Frozen |
| 11 | base_model.model.model.levels.0.blocks.0.norm2 | BatchNorm2d | [1, 80, 96, 96] | 🔒 Frozen |
| 12 | base_model.model.model.levels.0.blocks.0.drop_path | Identity | [1, 80, 96, 96] | 🔒 Frozen |
| 13 | base_model.model.model.levels.0.blocks.0 | ConvBlock | [1, 80, 96, 96] | 🔒 Frozen |
| 14 | base_model.model.model.levels.0.downsample.reduction.0 | Conv2d | [1, 160, 48, 48] | 🔒 Frozen |
| 15 | base_model.model.model.levels.0.downsample.reduction | Sequential | [1, 160, 48, 48] | 🔒 Frozen |
| 16 | base_model.model.model.levels.0.downsample | Downsample | [1, 160, 48, 48] | 🔒 Frozen |
| 17 | base_model.model.model.levels.0 | MambaVisionLayer | [1, 160, 48, 48] (+1 aux) | 🔒 Frozen |
| 18 | base_model.model.model.levels.1.blocks.0.conv1 | Conv2d | [1, 160, 48, 48] | 🔒 Frozen |
| 19 | base_model.model.model.levels.1.blocks.0.norm1 | BatchNorm2d | [1, 160, 48, 48] | 🔒 Frozen |
| 20 | base_model.model.model.levels.1.blocks.0.act1 | GELU | [1, 160, 48, 48] | 🔒 Frozen |
| 21 | base_model.model.model.levels.1.blocks.0.conv2 | Conv2d | [1, 160, 48, 48] | 🔒 Frozen |
| 22 | base_model.model.model.levels.1.blocks.0.norm2 | BatchNorm2d | [1, 160, 48, 48] | 🔒 Frozen |
| 23 | base_model.model.model.levels.1.blocks.0.drop_path | DropPath | [1, 160, 48, 48] | 🔒 Frozen |
| 24 | base_model.model.model.levels.1.blocks.0 | ConvBlock | [1, 160, 48, 48] | 🔒 Frozen |
| 25 | base_model.model.model.levels.1.blocks.1.conv1 | Conv2d | [1, 160, 48, 48] | 🔒 Frozen |
| 26 | base_model.model.model.levels.1.blocks.1.norm1 | BatchNorm2d | [1, 160, 48, 48] | 🔒 Frozen |
| 27 | base_model.model.model.levels.1.blocks.1.act1 | GELU | [1, 160, 48, 48] | 🔒 Frozen |
| 28 | base_model.model.model.levels.1.blocks.1.conv2 | Conv2d | [1, 160, 48, 48] | 🔒 Frozen |
| 29 | base_model.model.model.levels.1.blocks.1.norm2 | BatchNorm2d | [1, 160, 48, 48] | 🔒 Frozen |
| 30 | base_model.model.model.levels.1.blocks.1.drop_path | DropPath | [1, 160, 48, 48] | 🔒 Frozen |
| 31 | base_model.model.model.levels.1.blocks.1 | ConvBlock | [1, 160, 48, 48] | 🔒 Frozen |
| 32 | base_model.model.model.levels.1.blocks.2.conv1 | Conv2d | [1, 160, 48, 48] | 🔒 Frozen |
| 33 | base_model.model.model.levels.1.blocks.2.norm1 | BatchNorm2d | [1, 160, 48, 48] | 🔒 Frozen |
| 34 | base_model.model.model.levels.1.blocks.2.act1 | GELU | [1, 160, 48, 48] | 🔒 Frozen |
| 35 | base_model.model.model.levels.1.blocks.2.conv2 | Conv2d | [1, 160, 48, 48] | 🔒 Frozen |
| 36 | base_model.model.model.levels.1.blocks.2.norm2 | BatchNorm2d | [1, 160, 48, 48] | 🔒 Frozen |
| 37 | base_model.model.model.levels.1.blocks.2.drop_path | DropPath | [1, 160, 48, 48] | 🔒 Frozen |
| 38 | base_model.model.model.levels.1.blocks.2 | ConvBlock | [1, 160, 48, 48] | 🔒 Frozen |
| 39 | base_model.model.model.levels.1.downsample.reduction.0 | Conv2d | [1, 320, 24, 24] | 🔒 Frozen |
| 40 | base_model.model.model.levels.1.downsample.reduction | Sequential | [1, 320, 24, 24] | 🔒 Frozen |
| 41 | base_model.model.model.levels.1.downsample | Downsample | [1, 320, 24, 24] | 🔒 Frozen |
| 42 | base_model.model.model.levels.1 | MambaVisionLayer | [1, 320, 24, 24] (+1 aux) | 🔒 Frozen |
| 43 | base_model.model.model.levels.2.blocks.0.norm1 | LayerNorm | [4, 196, 320] | 🔒 Frozen |
| 44 | base_model.model.model.levels.2.blocks.0.mixer.in_proj.base_layer | Linear | [4, 196, 320] | 🔒 Frozen |
| 45 | base_model.model.model.levels.2.blocks.0.mixer.in_proj.lora_dropout.default | Identity | [4, 196, 320] | 🔒 Frozen |
| 46 | base_model.model.model.levels.2.blocks.0.mixer.in_proj.lora_A.default | Linear | [4, 196, 64] | 🟢 Trainable |
| 47 | base_model.model.model.levels.2.blocks.0.mixer.in_proj.lora_B.default | Linear | [4, 196, 320] | 🟢 Trainable |
| 48 | base_model.model.model.levels.2.blocks.0.mixer.in_proj | Linear | [4, 196, 320] | 🔒 Frozen |
| 49 | base_model.model.model.levels.2.blocks.0.mixer.x_proj.base_layer | Linear | [784, 36] | 🔒 Frozen |
| 50 | base_model.model.model.levels.2.blocks.0.mixer.x_proj.lora_dropout.default | Identity | [784, 160] | 🔒 Frozen |
| 51 | base_model.model.model.levels.2.blocks.0.mixer.x_proj.lora_A.default | Linear | [784, 64] | 🟢 Trainable |
| 52 | base_model.model.model.levels.2.blocks.0.mixer.x_proj.lora_B.default | Linear | [784, 36] | 🟢 Trainable |
| 53 | base_model.model.model.levels.2.blocks.0.mixer.x_proj | Linear | [784, 36] | 🔒 Frozen |
| 54 | base_model.model.model.levels.2.blocks.0.mixer.dt_proj.base_layer | Linear | [784, 160] | 🔒 Frozen |
| 55 | base_model.model.model.levels.2.blocks.0.mixer.dt_proj.lora_dropout.default | Identity | [784, 20] | 🔒 Frozen |
| 56 | base_model.model.model.levels.2.blocks.0.mixer.dt_proj.lora_A.default | Linear | [784, 64] | 🟢 Trainable |
| 57 | base_model.model.model.levels.2.blocks.0.mixer.dt_proj.lora_B.default | Linear | [784, 160] | 🟢 Trainable |
| 58 | base_model.model.model.levels.2.blocks.0.mixer.dt_proj | Linear | [784, 160] | 🔒 Frozen |
| 59 | base_model.model.model.levels.2.blocks.0.mixer.out_proj.base_layer | Linear | [4, 196, 320] | 🔒 Frozen |
| 60 | base_model.model.model.levels.2.blocks.0.mixer.out_proj.lora_dropout.default | Identity | [4, 196, 320] | 🔒 Frozen |
| 61 | base_model.model.model.levels.2.blocks.0.mixer.out_proj.lora_A.default | Linear | [4, 196, 64] | 🟢 Trainable |
| 62 | base_model.model.model.levels.2.blocks.0.mixer.out_proj.lora_B.default | Linear | [4, 196, 320] | 🟢 Trainable |
| 63 | base_model.model.model.levels.2.blocks.0.mixer.out_proj | Linear | [4, 196, 320] | 🔒 Frozen |
| 64 | base_model.model.model.levels.2.blocks.0.mixer | MambaVisionMixer | [4, 196, 320] | 🔒 Frozen |
| 65 | base_model.model.model.levels.2.blocks.0.drop_path | DropPath | [4, 196, 320] | 🔒 Frozen |
| 66 | base_model.model.model.levels.2.blocks.0.norm2 | LayerNorm | [4, 196, 320] | 🔒 Frozen |
| 67 | base_model.model.model.levels.2.blocks.0.mlp.fc1.base_layer | Linear | [4, 196, 1280] | 🔒 Frozen |
| 68 | base_model.model.model.levels.2.blocks.0.mlp.fc1.lora_dropout.default | Identity | [4, 196, 320] | 🔒 Frozen |
| 69 | base_model.model.model.levels.2.blocks.0.mlp.fc1.lora_A.default | Linear | [4, 196, 64] | 🟢 Trainable |
| 70 | base_model.model.model.levels.2.blocks.0.mlp.fc1.lora_B.default | Linear | [4, 196, 1280] | 🟢 Trainable |
| 71 | base_model.model.model.levels.2.blocks.0.mlp.fc1 | Linear | [4, 196, 1280] | 🔒 Frozen |
| 72 | base_model.model.model.levels.2.blocks.0.mlp.act | GELU | [4, 196, 1280] | 🔒 Frozen |
| 73 | base_model.model.model.levels.2.blocks.0.mlp.drop1 | Dropout | [4, 196, 1280] | 🔒 Frozen |
| 74 | base_model.model.model.levels.2.blocks.0.mlp.norm | Identity | [4, 196, 1280] | 🔒 Frozen |
| 75 | base_model.model.model.levels.2.blocks.0.mlp.fc2.base_layer | Linear | [4, 196, 320] | 🔒 Frozen |
| 76 | base_model.model.model.levels.2.blocks.0.mlp.fc2.lora_dropout.default | Identity | [4, 196, 1280] | 🔒 Frozen |
| 77 | base_model.model.model.levels.2.blocks.0.mlp.fc2.lora_A.default | Linear | [4, 196, 64] | 🟢 Trainable |
| 78 | base_model.model.model.levels.2.blocks.0.mlp.fc2.lora_B.default | Linear | [4, 196, 320] | 🟢 Trainable |
| 79 | base_model.model.model.levels.2.blocks.0.mlp.fc2 | Linear | [4, 196, 320] | 🔒 Frozen |
| 80 | base_model.model.model.levels.2.blocks.0.mlp.drop2 | Dropout | [4, 196, 320] | 🔒 Frozen |
| 81 | base_model.model.model.levels.2.blocks.0.mlp | Mlp | [4, 196, 320] | 🔒 Frozen |
| 82 | base_model.model.model.levels.2.blocks.0.drop_path | DropPath | [4, 196, 320] | 🔒 Frozen |
| 83 | base_model.model.model.levels.2.blocks.0 | Block | [4, 196, 320] | 🔒 Frozen |
| 84 | base_model.model.model.levels.2.blocks.1.norm1 | LayerNorm | [4, 196, 320] | 🔒 Frozen |
| 85 | base_model.model.model.levels.2.blocks.1.mixer.in_proj.base_layer | Linear | [4, 196, 320] | 🔒 Frozen |
| 86 | base_model.model.model.levels.2.blocks.1.mixer.in_proj.lora_dropout.default | Identity | [4, 196, 320] | 🔒 Frozen |
| 87 | base_model.model.model.levels.2.blocks.1.mixer.in_proj.lora_A.default | Linear | [4, 196, 64] | 🟢 Trainable |
| 88 | base_model.model.model.levels.2.blocks.1.mixer.in_proj.lora_B.default | Linear | [4, 196, 320] | 🟢 Trainable |
| 89 | base_model.model.model.levels.2.blocks.1.mixer.in_proj | Linear | [4, 196, 320] | 🔒 Frozen |
| 90 | base_model.model.model.levels.2.blocks.1.mixer.x_proj.base_layer | Linear | [784, 36] | 🔒 Frozen |
| 91 | base_model.model.model.levels.2.blocks.1.mixer.x_proj.lora_dropout.default | Identity | [784, 160] | 🔒 Frozen |
| 92 | base_model.model.model.levels.2.blocks.1.mixer.x_proj.lora_A.default | Linear | [784, 64] | 🟢 Trainable |
| 93 | base_model.model.model.levels.2.blocks.1.mixer.x_proj.lora_B.default | Linear | [784, 36] | 🟢 Trainable |
| 94 | base_model.model.model.levels.2.blocks.1.mixer.x_proj | Linear | [784, 36] | 🔒 Frozen |
| 95 | base_model.model.model.levels.2.blocks.1.mixer.dt_proj.base_layer | Linear | [784, 160] | 🔒 Frozen |
| 96 | base_model.model.model.levels.2.blocks.1.mixer.dt_proj.lora_dropout.default | Identity | [784, 20] | 🔒 Frozen |
| 97 | base_model.model.model.levels.2.blocks.1.mixer.dt_proj.lora_A.default | Linear | [784, 64] | 🟢 Trainable |
| 98 | base_model.model.model.levels.2.blocks.1.mixer.dt_proj.lora_B.default | Linear | [784, 160] | 🟢 Trainable |
| 99 | base_model.model.model.levels.2.blocks.1.mixer.dt_proj | Linear | [784, 160] | 🔒 Frozen |
| 100 | base_model.model.model.levels.2.blocks.1.mixer.out_proj.base_layer | Linear | [4, 196, 320] | 🔒 Frozen |
| 101 | base_model.model.model.levels.2.blocks.1.mixer.out_proj.lora_dropout.default | Identity | [4, 196, 320] | 🔒 Frozen |
| 102 | base_model.model.model.levels.2.blocks.1.mixer.out_proj.lora_A.default | Linear | [4, 196, 64] | 🟢 Trainable |
| 103 | base_model.model.model.levels.2.blocks.1.mixer.out_proj.lora_B.default | Linear | [4, 196, 320] | 🟢 Trainable |
| 104 | base_model.model.model.levels.2.blocks.1.mixer.out_proj | Linear | [4, 196, 320] | 🔒 Frozen |
| 105 | base_model.model.model.levels.2.blocks.1.mixer | MambaVisionMixer | [4, 196, 320] | 🔒 Frozen |
| 106 | base_model.model.model.levels.2.blocks.1.drop_path | DropPath | [4, 196, 320] | 🔒 Frozen |
| 107 | base_model.model.model.levels.2.blocks.1.norm2 | LayerNorm | [4, 196, 320] | 🔒 Frozen |
| 108 | base_model.model.model.levels.2.blocks.1.mlp.fc1.base_layer | Linear | [4, 196, 1280] | 🔒 Frozen |
| 109 | base_model.model.model.levels.2.blocks.1.mlp.fc1.lora_dropout.default | Identity | [4, 196, 320] | 🔒 Frozen |
| 110 | base_model.model.model.levels.2.blocks.1.mlp.fc1.lora_A.default | Linear | [4, 196, 64] | 🟢 Trainable |
| 111 | base_model.model.model.levels.2.blocks.1.mlp.fc1.lora_B.default | Linear | [4, 196, 1280] | 🟢 Trainable |
| 112 | base_model.model.model.levels.2.blocks.1.mlp.fc1 | Linear | [4, 196, 1280] | 🔒 Frozen |
| 113 | base_model.model.model.levels.2.blocks.1.mlp.act | GELU | [4, 196, 1280] | 🔒 Frozen |
| 114 | base_model.model.model.levels.2.blocks.1.mlp.drop1 | Dropout | [4, 196, 1280] | 🔒 Frozen |
| 115 | base_model.model.model.levels.2.blocks.1.mlp.norm | Identity | [4, 196, 1280] | 🔒 Frozen |
| 116 | base_model.model.model.levels.2.blocks.1.mlp.fc2.base_layer | Linear | [4, 196, 320] | 🔒 Frozen |
| 117 | base_model.model.model.levels.2.blocks.1.mlp.fc2.lora_dropout.default | Identity | [4, 196, 1280] | 🔒 Frozen |
| 118 | base_model.model.model.levels.2.blocks.1.mlp.fc2.lora_A.default | Linear | [4, 196, 64] | 🟢 Trainable |
| 119 | base_model.model.model.levels.2.blocks.1.mlp.fc2.lora_B.default | Linear | [4, 196, 320] | 🟢 Trainable |
| 120 | base_model.model.model.levels.2.blocks.1.mlp.fc2 | Linear | [4, 196, 320] | 🔒 Frozen |
| 121 | base_model.model.model.levels.2.blocks.1.mlp.drop2 | Dropout | [4, 196, 320] | 🔒 Frozen |
| 122 | base_model.model.model.levels.2.blocks.1.mlp | Mlp | [4, 196, 320] | 🔒 Frozen |
| 123 | base_model.model.model.levels.2.blocks.1.drop_path | DropPath | [4, 196, 320] | 🔒 Frozen |
| 124 | base_model.model.model.levels.2.blocks.1 | Block | [4, 196, 320] | 🔒 Frozen |
| 125 | base_model.model.model.levels.2.blocks.2.norm1 | LayerNorm | [4, 196, 320] | 🔒 Frozen |
| 126 | base_model.model.model.levels.2.blocks.2.mixer.in_proj.base_layer | Linear | [4, 196, 320] | 🔒 Frozen |
| 127 | base_model.model.model.levels.2.blocks.2.mixer.in_proj.lora_dropout.default | Identity | [4, 196, 320] | 🔒 Frozen |
| 128 | base_model.model.model.levels.2.blocks.2.mixer.in_proj.lora_A.default | Linear | [4, 196, 64] | 🟢 Trainable |
| 129 | base_model.model.model.levels.2.blocks.2.mixer.in_proj.lora_B.default | Linear | [4, 196, 320] | 🟢 Trainable |
| 130 | base_model.model.model.levels.2.blocks.2.mixer.in_proj | Linear | [4, 196, 320] | 🔒 Frozen |
| 131 | base_model.model.model.levels.2.blocks.2.mixer.x_proj.base_layer | Linear | [784, 36] | 🔒 Frozen |
| 132 | base_model.model.model.levels.2.blocks.2.mixer.x_proj.lora_dropout.default | Identity | [784, 160] | 🔒 Frozen |
| 133 | base_model.model.model.levels.2.blocks.2.mixer.x_proj.lora_A.default | Linear | [784, 64] | 🟢 Trainable |
| 134 | base_model.model.model.levels.2.blocks.2.mixer.x_proj.lora_B.default | Linear | [784, 36] | 🟢 Trainable |
| 135 | base_model.model.model.levels.2.blocks.2.mixer.x_proj | Linear | [784, 36] | 🔒 Frozen |
| 136 | base_model.model.model.levels.2.blocks.2.mixer.dt_proj.base_layer | Linear | [784, 160] | 🔒 Frozen |
| 137 | base_model.model.model.levels.2.blocks.2.mixer.dt_proj.lora_dropout.default | Identity | [784, 20] | 🔒 Frozen |
| 138 | base_model.model.model.levels.2.blocks.2.mixer.dt_proj.lora_A.default | Linear | [784, 64] | 🟢 Trainable |
| 139 | base_model.model.model.levels.2.blocks.2.mixer.dt_proj.lora_B.default | Linear | [784, 160] | 🟢 Trainable |
| 140 | base_model.model.model.levels.2.blocks.2.mixer.dt_proj | Linear | [784, 160] | 🔒 Frozen |
| 141 | base_model.model.model.levels.2.blocks.2.mixer.out_proj.base_layer | Linear | [4, 196, 320] | 🔒 Frozen |
| 142 | base_model.model.model.levels.2.blocks.2.mixer.out_proj.lora_dropout.default | Identity | [4, 196, 320] | 🔒 Frozen |
| 143 | base_model.model.model.levels.2.blocks.2.mixer.out_proj.lora_A.default | Linear | [4, 196, 64] | 🟢 Trainable |
| 144 | base_model.model.model.levels.2.blocks.2.mixer.out_proj.lora_B.default | Linear | [4, 196, 320] | 🟢 Trainable |
| 145 | base_model.model.model.levels.2.blocks.2.mixer.out_proj | Linear | [4, 196, 320] | 🔒 Frozen |
| 146 | base_model.model.model.levels.2.blocks.2.mixer | MambaVisionMixer | [4, 196, 320] | 🔒 Frozen |
| 147 | base_model.model.model.levels.2.blocks.2.drop_path | DropPath | [4, 196, 320] | 🔒 Frozen |
| 148 | base_model.model.model.levels.2.blocks.2.norm2 | LayerNorm | [4, 196, 320] | 🔒 Frozen |
| 149 | base_model.model.model.levels.2.blocks.2.mlp.fc1.base_layer | Linear | [4, 196, 1280] | 🔒 Frozen |
| 150 | base_model.model.model.levels.2.blocks.2.mlp.fc1.lora_dropout.default | Identity | [4, 196, 320] | 🔒 Frozen |
| 151 | base_model.model.model.levels.2.blocks.2.mlp.fc1.lora_A.default | Linear | [4, 196, 64] | 🟢 Trainable |
| 152 | base_model.model.model.levels.2.blocks.2.mlp.fc1.lora_B.default | Linear | [4, 196, 1280] | 🟢 Trainable |
| 153 | base_model.model.model.levels.2.blocks.2.mlp.fc1 | Linear | [4, 196, 1280] | 🔒 Frozen |
| 154 | base_model.model.model.levels.2.blocks.2.mlp.act | GELU | [4, 196, 1280] | 🔒 Frozen |
| 155 | base_model.model.model.levels.2.blocks.2.mlp.drop1 | Dropout | [4, 196, 1280] | 🔒 Frozen |
| 156 | base_model.model.model.levels.2.blocks.2.mlp.norm | Identity | [4, 196, 1280] | 🔒 Frozen |
| 157 | base_model.model.model.levels.2.blocks.2.mlp.fc2.base_layer | Linear | [4, 196, 320] | 🔒 Frozen |
| 158 | base_model.model.model.levels.2.blocks.2.mlp.fc2.lora_dropout.default | Identity | [4, 196, 1280] | 🔒 Frozen |
| 159 | base_model.model.model.levels.2.blocks.2.mlp.fc2.lora_A.default | Linear | [4, 196, 64] | 🟢 Trainable |
| 160 | base_model.model.model.levels.2.blocks.2.mlp.fc2.lora_B.default | Linear | [4, 196, 320] | 🟢 Trainable |
| 161 | base_model.model.model.levels.2.blocks.2.mlp.fc2 | Linear | [4, 196, 320] | 🔒 Frozen |
| 162 | base_model.model.model.levels.2.blocks.2.mlp.drop2 | Dropout | [4, 196, 320] | 🔒 Frozen |
| 163 | base_model.model.model.levels.2.blocks.2.mlp | Mlp | [4, 196, 320] | 🔒 Frozen |
| 164 | base_model.model.model.levels.2.blocks.2.drop_path | DropPath | [4, 196, 320] | 🔒 Frozen |
| 165 | base_model.model.model.levels.2.blocks.2 | Block | [4, 196, 320] | 🔒 Frozen |
| 166 | base_model.model.model.levels.2.blocks.3.norm1 | LayerNorm | [4, 196, 320] | 🔒 Frozen |
| 167 | base_model.model.model.levels.2.blocks.3.mixer.in_proj.base_layer | Linear | [4, 196, 320] | 🔒 Frozen |
| 168 | base_model.model.model.levels.2.blocks.3.mixer.in_proj.lora_dropout.default | Identity | [4, 196, 320] | 🔒 Frozen |
| 169 | base_model.model.model.levels.2.blocks.3.mixer.in_proj.lora_A.default | Linear | [4, 196, 64] | 🟢 Trainable |
| 170 | base_model.model.model.levels.2.blocks.3.mixer.in_proj.lora_B.default | Linear | [4, 196, 320] | 🟢 Trainable |
| 171 | base_model.model.model.levels.2.blocks.3.mixer.in_proj | Linear | [4, 196, 320] | 🔒 Frozen |
| 172 | base_model.model.model.levels.2.blocks.3.mixer.x_proj.base_layer | Linear | [784, 36] | 🔒 Frozen |
| 173 | base_model.model.model.levels.2.blocks.3.mixer.x_proj.lora_dropout.default | Identity | [784, 160] | 🔒 Frozen |
| 174 | base_model.model.model.levels.2.blocks.3.mixer.x_proj.lora_A.default | Linear | [784, 64] | 🟢 Trainable |
| 175 | base_model.model.model.levels.2.blocks.3.mixer.x_proj.lora_B.default | Linear | [784, 36] | 🟢 Trainable |
| 176 | base_model.model.model.levels.2.blocks.3.mixer.x_proj | Linear | [784, 36] | 🔒 Frozen |
| 177 | base_model.model.model.levels.2.blocks.3.mixer.dt_proj.base_layer | Linear | [784, 160] | 🔒 Frozen |
| 178 | base_model.model.model.levels.2.blocks.3.mixer.dt_proj.lora_dropout.default | Identity | [784, 20] | 🔒 Frozen |
| 179 | base_model.model.model.levels.2.blocks.3.mixer.dt_proj.lora_A.default | Linear | [784, 64] | 🟢 Trainable |
| 180 | base_model.model.model.levels.2.blocks.3.mixer.dt_proj.lora_B.default | Linear | [784, 160] | 🟢 Trainable |
| 181 | base_model.model.model.levels.2.blocks.3.mixer.dt_proj | Linear | [784, 160] | 🔒 Frozen |
| 182 | base_model.model.model.levels.2.blocks.3.mixer.out_proj.base_layer | Linear | [4, 196, 320] | 🔒 Frozen |
| 183 | base_model.model.model.levels.2.blocks.3.mixer.out_proj.lora_dropout.default | Identity | [4, 196, 320] | 🔒 Frozen |
| 184 | base_model.model.model.levels.2.blocks.3.mixer.out_proj.lora_A.default | Linear | [4, 196, 64] | 🟢 Trainable |
| 185 | base_model.model.model.levels.2.blocks.3.mixer.out_proj.lora_B.default | Linear | [4, 196, 320] | 🟢 Trainable |
| 186 | base_model.model.model.levels.2.blocks.3.mixer.out_proj | Linear | [4, 196, 320] | 🔒 Frozen |
| 187 | base_model.model.model.levels.2.blocks.3.mixer | MambaVisionMixer | [4, 196, 320] | 🔒 Frozen |
| 188 | base_model.model.model.levels.2.blocks.3.drop_path | DropPath | [4, 196, 320] | 🔒 Frozen |
| 189 | base_model.model.model.levels.2.blocks.3.norm2 | LayerNorm | [4, 196, 320] | 🔒 Frozen |
| 190 | base_model.model.model.levels.2.blocks.3.mlp.fc1.base_layer | Linear | [4, 196, 1280] | 🔒 Frozen |
| 191 | base_model.model.model.levels.2.blocks.3.mlp.fc1.lora_dropout.default | Identity | [4, 196, 320] | 🔒 Frozen |
| 192 | base_model.model.model.levels.2.blocks.3.mlp.fc1.lora_A.default | Linear | [4, 196, 64] | 🟢 Trainable |
| 193 | base_model.model.model.levels.2.blocks.3.mlp.fc1.lora_B.default | Linear | [4, 196, 1280] | 🟢 Trainable |
| 194 | base_model.model.model.levels.2.blocks.3.mlp.fc1 | Linear | [4, 196, 1280] | 🔒 Frozen |
| 195 | base_model.model.model.levels.2.blocks.3.mlp.act | GELU | [4, 196, 1280] | 🔒 Frozen |
| 196 | base_model.model.model.levels.2.blocks.3.mlp.drop1 | Dropout | [4, 196, 1280] | 🔒 Frozen |
| 197 | base_model.model.model.levels.2.blocks.3.mlp.norm | Identity | [4, 196, 1280] | 🔒 Frozen |
| 198 | base_model.model.model.levels.2.blocks.3.mlp.fc2.base_layer | Linear | [4, 196, 320] | 🔒 Frozen |
| 199 | base_model.model.model.levels.2.blocks.3.mlp.fc2.lora_dropout.default | Identity | [4, 196, 1280] | 🔒 Frozen |
| 200 | base_model.model.model.levels.2.blocks.3.mlp.fc2.lora_A.default | Linear | [4, 196, 64] | 🟢 Trainable |
| 201 | base_model.model.model.levels.2.blocks.3.mlp.fc2.lora_B.default | Linear | [4, 196, 320] | 🟢 Trainable |
| 202 | base_model.model.model.levels.2.blocks.3.mlp.fc2 | Linear | [4, 196, 320] | 🔒 Frozen |
| 203 | base_model.model.model.levels.2.blocks.3.mlp.drop2 | Dropout | [4, 196, 320] | 🔒 Frozen |
| 204 | base_model.model.model.levels.2.blocks.3.mlp | Mlp | [4, 196, 320] | 🔒 Frozen |
| 205 | base_model.model.model.levels.2.blocks.3.drop_path | DropPath | [4, 196, 320] | 🔒 Frozen |
| 206 | base_model.model.model.levels.2.blocks.3 | Block | [4, 196, 320] | 🔒 Frozen |
| 207 | base_model.model.model.levels.2.blocks.4.norm1 | LayerNorm | [4, 196, 320] | 🔒 Frozen |
| 208 | base_model.model.model.levels.2.blocks.4.mixer.qkv | Linear | [4, 196, 960] | 🔒 Frozen |
| 209 | base_model.model.model.levels.2.blocks.4.mixer.q_norm | Identity | [4, 8, 196, 40] | 🔒 Frozen |
| 210 | base_model.model.model.levels.2.blocks.4.mixer.k_norm | Identity | [4, 8, 196, 40] | 🔒 Frozen |
| 211 | base_model.model.model.levels.2.blocks.4.mixer.proj | Linear | [4, 196, 320] | 🔒 Frozen |
| 212 | base_model.model.model.levels.2.blocks.4.mixer.proj_drop | Dropout | [4, 196, 320] | 🔒 Frozen |
| 213 | base_model.model.model.levels.2.blocks.4.mixer | Attention | [4, 196, 320] | 🔒 Frozen |
| 214 | base_model.model.model.levels.2.blocks.4.drop_path | DropPath | [4, 196, 320] | 🔒 Frozen |
| 215 | base_model.model.model.levels.2.blocks.4.norm2 | LayerNorm | [4, 196, 320] | 🔒 Frozen |
| 216 | base_model.model.model.levels.2.blocks.4.mlp.fc1.base_layer | Linear | [4, 196, 1280] | 🔒 Frozen |
| 217 | base_model.model.model.levels.2.blocks.4.mlp.fc1.lora_dropout.default | Identity | [4, 196, 320] | 🔒 Frozen |
| 218 | base_model.model.model.levels.2.blocks.4.mlp.fc1.lora_A.default | Linear | [4, 196, 64] | 🟢 Trainable |
| 219 | base_model.model.model.levels.2.blocks.4.mlp.fc1.lora_B.default | Linear | [4, 196, 1280] | 🟢 Trainable |
| 220 | base_model.model.model.levels.2.blocks.4.mlp.fc1 | Linear | [4, 196, 1280] | 🔒 Frozen |
| 221 | base_model.model.model.levels.2.blocks.4.mlp.act | GELU | [4, 196, 1280] | 🔒 Frozen |
| 222 | base_model.model.model.levels.2.blocks.4.mlp.drop1 | Dropout | [4, 196, 1280] | 🔒 Frozen |
| 223 | base_model.model.model.levels.2.blocks.4.mlp.norm | Identity | [4, 196, 1280] | 🔒 Frozen |
| 224 | base_model.model.model.levels.2.blocks.4.mlp.fc2.base_layer | Linear | [4, 196, 320] | 🔒 Frozen |
| 225 | base_model.model.model.levels.2.blocks.4.mlp.fc2.lora_dropout.default | Identity | [4, 196, 1280] | 🔒 Frozen |
| 226 | base_model.model.model.levels.2.blocks.4.mlp.fc2.lora_A.default | Linear | [4, 196, 64] | 🟢 Trainable |
| 227 | base_model.model.model.levels.2.blocks.4.mlp.fc2.lora_B.default | Linear | [4, 196, 320] | 🟢 Trainable |
| 228 | base_model.model.model.levels.2.blocks.4.mlp.fc2 | Linear | [4, 196, 320] | 🔒 Frozen |
| 229 | base_model.model.model.levels.2.blocks.4.mlp.drop2 | Dropout | [4, 196, 320] | 🔒 Frozen |
| 230 | base_model.model.model.levels.2.blocks.4.mlp | Mlp | [4, 196, 320] | 🔒 Frozen |
| 231 | base_model.model.model.levels.2.blocks.4.drop_path | DropPath | [4, 196, 320] | 🔒 Frozen |
| 232 | base_model.model.model.levels.2.blocks.4 | Block | [4, 196, 320] | 🔒 Frozen |
| 233 | base_model.model.model.levels.2.blocks.5.norm1 | LayerNorm | [4, 196, 320] | 🔒 Frozen |
| 234 | base_model.model.model.levels.2.blocks.5.mixer.qkv | Linear | [4, 196, 960] | 🔒 Frozen |
| 235 | base_model.model.model.levels.2.blocks.5.mixer.q_norm | Identity | [4, 8, 196, 40] | 🔒 Frozen |
| 236 | base_model.model.model.levels.2.blocks.5.mixer.k_norm | Identity | [4, 8, 196, 40] | 🔒 Frozen |
| 237 | base_model.model.model.levels.2.blocks.5.mixer.proj | Linear | [4, 196, 320] | 🔒 Frozen |
| 238 | base_model.model.model.levels.2.blocks.5.mixer.proj_drop | Dropout | [4, 196, 320] | 🔒 Frozen |
| 239 | base_model.model.model.levels.2.blocks.5.mixer | Attention | [4, 196, 320] | 🔒 Frozen |
| 240 | base_model.model.model.levels.2.blocks.5.drop_path | DropPath | [4, 196, 320] | 🔒 Frozen |
| 241 | base_model.model.model.levels.2.blocks.5.norm2 | LayerNorm | [4, 196, 320] | 🔒 Frozen |
| 242 | base_model.model.model.levels.2.blocks.5.mlp.fc1.base_layer | Linear | [4, 196, 1280] | 🔒 Frozen |
| 243 | base_model.model.model.levels.2.blocks.5.mlp.fc1.lora_dropout.default | Identity | [4, 196, 320] | 🔒 Frozen |
| 244 | base_model.model.model.levels.2.blocks.5.mlp.fc1.lora_A.default | Linear | [4, 196, 64] | 🟢 Trainable |
| 245 | base_model.model.model.levels.2.blocks.5.mlp.fc1.lora_B.default | Linear | [4, 196, 1280] | 🟢 Trainable |
| 246 | base_model.model.model.levels.2.blocks.5.mlp.fc1 | Linear | [4, 196, 1280] | 🔒 Frozen |
| 247 | base_model.model.model.levels.2.blocks.5.mlp.act | GELU | [4, 196, 1280] | 🔒 Frozen |
| 248 | base_model.model.model.levels.2.blocks.5.mlp.drop1 | Dropout | [4, 196, 1280] | 🔒 Frozen |
| 249 | base_model.model.model.levels.2.blocks.5.mlp.norm | Identity | [4, 196, 1280] | 🔒 Frozen |
| 250 | base_model.model.model.levels.2.blocks.5.mlp.fc2.base_layer | Linear | [4, 196, 320] | 🔒 Frozen |
| 251 | base_model.model.model.levels.2.blocks.5.mlp.fc2.lora_dropout.default | Identity | [4, 196, 1280] | 🔒 Frozen |
| 252 | base_model.model.model.levels.2.blocks.5.mlp.fc2.lora_A.default | Linear | [4, 196, 64] | 🟢 Trainable |
| 253 | base_model.model.model.levels.2.blocks.5.mlp.fc2.lora_B.default | Linear | [4, 196, 320] | 🟢 Trainable |
| 254 | base_model.model.model.levels.2.blocks.5.mlp.fc2 | Linear | [4, 196, 320] | 🔒 Frozen |
| 255 | base_model.model.model.levels.2.blocks.5.mlp.drop2 | Dropout | [4, 196, 320] | 🔒 Frozen |
| 256 | base_model.model.model.levels.2.blocks.5.mlp | Mlp | [4, 196, 320] | 🔒 Frozen |
| 257 | base_model.model.model.levels.2.blocks.5.drop_path | DropPath | [4, 196, 320] | 🔒 Frozen |
| 258 | base_model.model.model.levels.2.blocks.5 | Block | [4, 196, 320] | 🔒 Frozen |
| 259 | base_model.model.model.levels.2.blocks.6.norm1 | LayerNorm | [4, 196, 320] | 🔒 Frozen |
| 260 | base_model.model.model.levels.2.blocks.6.mixer.qkv | Linear | [4, 196, 960] | 🔒 Frozen |
| 261 | base_model.model.model.levels.2.blocks.6.mixer.q_norm | Identity | [4, 8, 196, 40] | 🔒 Frozen |
| 262 | base_model.model.model.levels.2.blocks.6.mixer.k_norm | Identity | [4, 8, 196, 40] | 🔒 Frozen |
| 263 | base_model.model.model.levels.2.blocks.6.mixer.proj | Linear | [4, 196, 320] | 🔒 Frozen |
| 264 | base_model.model.model.levels.2.blocks.6.mixer.proj_drop | Dropout | [4, 196, 320] | 🔒 Frozen |
| 265 | base_model.model.model.levels.2.blocks.6.mixer | Attention | [4, 196, 320] | 🔒 Frozen |
| 266 | base_model.model.model.levels.2.blocks.6.drop_path | DropPath | [4, 196, 320] | 🔒 Frozen |
| 267 | base_model.model.model.levels.2.blocks.6.norm2 | LayerNorm | [4, 196, 320] | 🔒 Frozen |
| 268 | base_model.model.model.levels.2.blocks.6.mlp.fc1.base_layer | Linear | [4, 196, 1280] | 🔒 Frozen |
| 269 | base_model.model.model.levels.2.blocks.6.mlp.fc1.lora_dropout.default | Identity | [4, 196, 320] | 🔒 Frozen |
| 270 | base_model.model.model.levels.2.blocks.6.mlp.fc1.lora_A.default | Linear | [4, 196, 64] | 🟢 Trainable |
| 271 | base_model.model.model.levels.2.blocks.6.mlp.fc1.lora_B.default | Linear | [4, 196, 1280] | 🟢 Trainable |
| 272 | base_model.model.model.levels.2.blocks.6.mlp.fc1 | Linear | [4, 196, 1280] | 🔒 Frozen |
| 273 | base_model.model.model.levels.2.blocks.6.mlp.act | GELU | [4, 196, 1280] | 🔒 Frozen |
| 274 | base_model.model.model.levels.2.blocks.6.mlp.drop1 | Dropout | [4, 196, 1280] | 🔒 Frozen |
| 275 | base_model.model.model.levels.2.blocks.6.mlp.norm | Identity | [4, 196, 1280] | 🔒 Frozen |
| 276 | base_model.model.model.levels.2.blocks.6.mlp.fc2.base_layer | Linear | [4, 196, 320] | 🔒 Frozen |
| 277 | base_model.model.model.levels.2.blocks.6.mlp.fc2.lora_dropout.default | Identity | [4, 196, 1280] | 🔒 Frozen |
| 278 | base_model.model.model.levels.2.blocks.6.mlp.fc2.lora_A.default | Linear | [4, 196, 64] | 🟢 Trainable |
| 279 | base_model.model.model.levels.2.blocks.6.mlp.fc2.lora_B.default | Linear | [4, 196, 320] | 🟢 Trainable |
| 280 | base_model.model.model.levels.2.blocks.6.mlp.fc2 | Linear | [4, 196, 320] | 🔒 Frozen |
| 281 | base_model.model.model.levels.2.blocks.6.mlp.drop2 | Dropout | [4, 196, 320] | 🔒 Frozen |
| 282 | base_model.model.model.levels.2.blocks.6.mlp | Mlp | [4, 196, 320] | 🔒 Frozen |
| 283 | base_model.model.model.levels.2.blocks.6.drop_path | DropPath | [4, 196, 320] | 🔒 Frozen |
| 284 | base_model.model.model.levels.2.blocks.6 | Block | [4, 196, 320] | 🔒 Frozen |
| 285 | base_model.model.model.levels.2.blocks.7.norm1 | LayerNorm | [4, 196, 320] | 🔒 Frozen |
| 286 | base_model.model.model.levels.2.blocks.7.mixer.qkv | Linear | [4, 196, 960] | 🔒 Frozen |
| 287 | base_model.model.model.levels.2.blocks.7.mixer.q_norm | Identity | [4, 8, 196, 40] | 🔒 Frozen |
| 288 | base_model.model.model.levels.2.blocks.7.mixer.k_norm | Identity | [4, 8, 196, 40] | 🔒 Frozen |
| 289 | base_model.model.model.levels.2.blocks.7.mixer.proj | Linear | [4, 196, 320] | 🔒 Frozen |
| 290 | base_model.model.model.levels.2.blocks.7.mixer.proj_drop | Dropout | [4, 196, 320] | 🔒 Frozen |
| 291 | base_model.model.model.levels.2.blocks.7.mixer | Attention | [4, 196, 320] | 🔒 Frozen |
| 292 | base_model.model.model.levels.2.blocks.7.drop_path | DropPath | [4, 196, 320] | 🔒 Frozen |
| 293 | base_model.model.model.levels.2.blocks.7.norm2 | LayerNorm | [4, 196, 320] | 🔒 Frozen |
| 294 | base_model.model.model.levels.2.blocks.7.mlp.fc1.base_layer | Linear | [4, 196, 1280] | 🔒 Frozen |
| 295 | base_model.model.model.levels.2.blocks.7.mlp.fc1.lora_dropout.default | Identity | [4, 196, 320] | 🔒 Frozen |
| 296 | base_model.model.model.levels.2.blocks.7.mlp.fc1.lora_A.default | Linear | [4, 196, 64] | 🟢 Trainable |
| 297 | base_model.model.model.levels.2.blocks.7.mlp.fc1.lora_B.default | Linear | [4, 196, 1280] | 🟢 Trainable |
| 298 | base_model.model.model.levels.2.blocks.7.mlp.fc1 | Linear | [4, 196, 1280] | 🔒 Frozen |
| 299 | base_model.model.model.levels.2.blocks.7.mlp.act | GELU | [4, 196, 1280] | 🔒 Frozen |
| 300 | base_model.model.model.levels.2.blocks.7.mlp.drop1 | Dropout | [4, 196, 1280] | 🔒 Frozen |
| 301 | base_model.model.model.levels.2.blocks.7.mlp.norm | Identity | [4, 196, 1280] | 🔒 Frozen |
| 302 | base_model.model.model.levels.2.blocks.7.mlp.fc2.base_layer | Linear | [4, 196, 320] | 🔒 Frozen |
| 303 | base_model.model.model.levels.2.blocks.7.mlp.fc2.lora_dropout.default | Identity | [4, 196, 1280] | 🔒 Frozen |
| 304 | base_model.model.model.levels.2.blocks.7.mlp.fc2.lora_A.default | Linear | [4, 196, 64] | 🟢 Trainable |
| 305 | base_model.model.model.levels.2.blocks.7.mlp.fc2.lora_B.default | Linear | [4, 196, 320] | 🟢 Trainable |
| 306 | base_model.model.model.levels.2.blocks.7.mlp.fc2 | Linear | [4, 196, 320] | 🔒 Frozen |
| 307 | base_model.model.model.levels.2.blocks.7.mlp.drop2 | Dropout | [4, 196, 320] | 🔒 Frozen |
| 308 | base_model.model.model.levels.2.blocks.7.mlp | Mlp | [4, 196, 320] | 🔒 Frozen |
| 309 | base_model.model.model.levels.2.blocks.7.drop_path | DropPath | [4, 196, 320] | 🔒 Frozen |
| 310 | base_model.model.model.levels.2.blocks.7 | Block | [4, 196, 320] | 🔒 Frozen |
| 311 | base_model.model.model.levels.2.downsample.reduction.0 | Conv2d | [1, 640, 12, 12] | 🔒 Frozen |
| 312 | base_model.model.model.levels.2.downsample.reduction | Sequential | [1, 640, 12, 12] | 🔒 Frozen |
| 313 | base_model.model.model.levels.2.downsample | Downsample | [1, 640, 12, 12] | 🔒 Frozen |
| 314 | base_model.model.quantum_adapter.reducer | Linear | [1, 4] | 🟢 Trainable |
| 315 | base_model.model.quantum_adapter.q_layer | TorchLayer | [1, 4] | 🟢 Trainable |
| 316 | base_model.model.quantum_adapter.expander | Linear | [1, 640] | 🟢 Trainable |
| 317 | base_model.model.quantum_adapter | QuantumContextAdapter | [1, 640, 12, 12] | 🟢 Trainable |
| 318 | base_model.model.model.levels.2 | MambaVisionLayer | [1, 640, 12, 12] (+1 aux) | 🔒 Frozen |
| 319 | base_model.model.model.levels.3.blocks.0.norm1 | LayerNorm | [4, 49, 640] | 🔒 Frozen |
| 320 | base_model.model.model.levels.3.blocks.0.mixer.in_proj.base_layer | Linear | [4, 49, 640] | 🔒 Frozen |
| 321 | base_model.model.model.levels.3.blocks.0.mixer.in_proj.lora_dropout.default | Identity | [4, 49, 640] | 🔒 Frozen |
| 322 | base_model.model.model.levels.3.blocks.0.mixer.in_proj.lora_A.default | Linear | [4, 49, 64] | 🟢 Trainable |
| 323 | base_model.model.model.levels.3.blocks.0.mixer.in_proj.lora_B.default | Linear | [4, 49, 640] | 🟢 Trainable |
| 324 | base_model.model.model.levels.3.blocks.0.mixer.in_proj | Linear | [4, 49, 640] | 🔒 Frozen |
| 325 | base_model.model.model.levels.3.blocks.0.mixer.x_proj.base_layer | Linear | [196, 56] | 🔒 Frozen |
| 326 | base_model.model.model.levels.3.blocks.0.mixer.x_proj.lora_dropout.default | Identity | [196, 320] | 🔒 Frozen |
| 327 | base_model.model.model.levels.3.blocks.0.mixer.x_proj.lora_A.default | Linear | [196, 64] | 🟢 Trainable |
| 328 | base_model.model.model.levels.3.blocks.0.mixer.x_proj.lora_B.default | Linear | [196, 56] | 🟢 Trainable |
| 329 | base_model.model.model.levels.3.blocks.0.mixer.x_proj | Linear | [196, 56] | 🔒 Frozen |
| 330 | base_model.model.model.levels.3.blocks.0.mixer.dt_proj.base_layer | Linear | [196, 320] | 🔒 Frozen |
| 331 | base_model.model.model.levels.3.blocks.0.mixer.dt_proj.lora_dropout.default | Identity | [196, 40] | 🔒 Frozen |
| 332 | base_model.model.model.levels.3.blocks.0.mixer.dt_proj.lora_A.default | Linear | [196, 64] | 🟢 Trainable |
| 333 | base_model.model.model.levels.3.blocks.0.mixer.dt_proj.lora_B.default | Linear | [196, 320] | 🟢 Trainable |
| 334 | base_model.model.model.levels.3.blocks.0.mixer.dt_proj | Linear | [196, 320] | 🔒 Frozen |
| 335 | base_model.model.model.levels.3.blocks.0.mixer.out_proj.base_layer | Linear | [4, 49, 640] | 🔒 Frozen |
| 336 | base_model.model.model.levels.3.blocks.0.mixer.out_proj.lora_dropout.default | Identity | [4, 49, 640] | 🔒 Frozen |
| 337 | base_model.model.model.levels.3.blocks.0.mixer.out_proj.lora_A.default | Linear | [4, 49, 64] | 🟢 Trainable |
| 338 | base_model.model.model.levels.3.blocks.0.mixer.out_proj.lora_B.default | Linear | [4, 49, 640] | 🟢 Trainable |
| 339 | base_model.model.model.levels.3.blocks.0.mixer.out_proj | Linear | [4, 49, 640] | 🔒 Frozen |
| 340 | base_model.model.model.levels.3.blocks.0.mixer | MambaVisionMixer | [4, 49, 640] | 🔒 Frozen |
| 341 | base_model.model.model.levels.3.blocks.0.drop_path | DropPath | [4, 49, 640] | 🔒 Frozen |
| 342 | base_model.model.model.levels.3.blocks.0.norm2 | LayerNorm | [4, 49, 640] | 🔒 Frozen |
| 343 | base_model.model.model.levels.3.blocks.0.mlp.fc1.base_layer | Linear | [4, 49, 2560] | 🔒 Frozen |
| 344 | base_model.model.model.levels.3.blocks.0.mlp.fc1.lora_dropout.default | Identity | [4, 49, 640] | 🔒 Frozen |
| 345 | base_model.model.model.levels.3.blocks.0.mlp.fc1.lora_A.default | Linear | [4, 49, 64] | 🟢 Trainable |
| 346 | base_model.model.model.levels.3.blocks.0.mlp.fc1.lora_B.default | Linear | [4, 49, 2560] | 🟢 Trainable |
| 347 | base_model.model.model.levels.3.blocks.0.mlp.fc1 | Linear | [4, 49, 2560] | 🔒 Frozen |
| 348 | base_model.model.model.levels.3.blocks.0.mlp.act | GELU | [4, 49, 2560] | 🔒 Frozen |
| 349 | base_model.model.model.levels.3.blocks.0.mlp.drop1 | Dropout | [4, 49, 2560] | 🔒 Frozen |
| 350 | base_model.model.model.levels.3.blocks.0.mlp.norm | Identity | [4, 49, 2560] | 🔒 Frozen |
| 351 | base_model.model.model.levels.3.blocks.0.mlp.fc2.base_layer | Linear | [4, 49, 640] | 🔒 Frozen |
| 352 | base_model.model.model.levels.3.blocks.0.mlp.fc2.lora_dropout.default | Identity | [4, 49, 2560] | 🔒 Frozen |
| 353 | base_model.model.model.levels.3.blocks.0.mlp.fc2.lora_A.default | Linear | [4, 49, 64] | 🟢 Trainable |
| 354 | base_model.model.model.levels.3.blocks.0.mlp.fc2.lora_B.default | Linear | [4, 49, 640] | 🟢 Trainable |
| 355 | base_model.model.model.levels.3.blocks.0.mlp.fc2 | Linear | [4, 49, 640] | 🔒 Frozen |
| 356 | base_model.model.model.levels.3.blocks.0.mlp.drop2 | Dropout | [4, 49, 640] | 🔒 Frozen |
| 357 | base_model.model.model.levels.3.blocks.0.mlp | Mlp | [4, 49, 640] | 🔒 Frozen |
| 358 | base_model.model.model.levels.3.blocks.0.drop_path | DropPath | [4, 49, 640] | 🔒 Frozen |
| 359 | base_model.model.model.levels.3.blocks.0 | Block | [4, 49, 640] | 🔒 Frozen |
| 360 | base_model.model.model.levels.3.blocks.1.norm1 | LayerNorm | [4, 49, 640] | 🔒 Frozen |
| 361 | base_model.model.model.levels.3.blocks.1.mixer.in_proj.base_layer | Linear | [4, 49, 640] | 🔒 Frozen |
| 362 | base_model.model.model.levels.3.blocks.1.mixer.in_proj.lora_dropout.default | Identity | [4, 49, 640] | 🔒 Frozen |
| 363 | base_model.model.model.levels.3.blocks.1.mixer.in_proj.lora_A.default | Linear | [4, 49, 64] | 🟢 Trainable |
| 364 | base_model.model.model.levels.3.blocks.1.mixer.in_proj.lora_B.default | Linear | [4, 49, 640] | 🟢 Trainable |
| 365 | base_model.model.model.levels.3.blocks.1.mixer.in_proj | Linear | [4, 49, 640] | 🔒 Frozen |
| 366 | base_model.model.model.levels.3.blocks.1.mixer.x_proj.base_layer | Linear | [196, 56] | 🔒 Frozen |
| 367 | base_model.model.model.levels.3.blocks.1.mixer.x_proj.lora_dropout.default | Identity | [196, 320] | 🔒 Frozen |
| 368 | base_model.model.model.levels.3.blocks.1.mixer.x_proj.lora_A.default | Linear | [196, 64] | 🟢 Trainable |
| 369 | base_model.model.model.levels.3.blocks.1.mixer.x_proj.lora_B.default | Linear | [196, 56] | 🟢 Trainable |
| 370 | base_model.model.model.levels.3.blocks.1.mixer.x_proj | Linear | [196, 56] | 🔒 Frozen |
| 371 | base_model.model.model.levels.3.blocks.1.mixer.dt_proj.base_layer | Linear | [196, 320] | 🔒 Frozen |
| 372 | base_model.model.model.levels.3.blocks.1.mixer.dt_proj.lora_dropout.default | Identity | [196, 40] | 🔒 Frozen |
| 373 | base_model.model.model.levels.3.blocks.1.mixer.dt_proj.lora_A.default | Linear | [196, 64] | 🟢 Trainable |
| 374 | base_model.model.model.levels.3.blocks.1.mixer.dt_proj.lora_B.default | Linear | [196, 320] | 🟢 Trainable |
| 375 | base_model.model.model.levels.3.blocks.1.mixer.dt_proj | Linear | [196, 320] | 🔒 Frozen |
| 376 | base_model.model.model.levels.3.blocks.1.mixer.out_proj.base_layer | Linear | [4, 49, 640] | 🔒 Frozen |
| 377 | base_model.model.model.levels.3.blocks.1.mixer.out_proj.lora_dropout.default | Identity | [4, 49, 640] | 🔒 Frozen |
| 378 | base_model.model.model.levels.3.blocks.1.mixer.out_proj.lora_A.default | Linear | [4, 49, 64] | 🟢 Trainable |
| 379 | base_model.model.model.levels.3.blocks.1.mixer.out_proj.lora_B.default | Linear | [4, 49, 640] | 🟢 Trainable |
| 380 | base_model.model.model.levels.3.blocks.1.mixer.out_proj | Linear | [4, 49, 640] | 🔒 Frozen |
| 381 | base_model.model.model.levels.3.blocks.1.mixer | MambaVisionMixer | [4, 49, 640] | 🔒 Frozen |
| 382 | base_model.model.model.levels.3.blocks.1.drop_path | DropPath | [4, 49, 640] | 🔒 Frozen |
| 383 | base_model.model.model.levels.3.blocks.1.norm2 | LayerNorm | [4, 49, 640] | 🔒 Frozen |
| 384 | base_model.model.model.levels.3.blocks.1.mlp.fc1.base_layer | Linear | [4, 49, 2560] | 🔒 Frozen |
| 385 | base_model.model.model.levels.3.blocks.1.mlp.fc1.lora_dropout.default | Identity | [4, 49, 640] | 🔒 Frozen |
| 386 | base_model.model.model.levels.3.blocks.1.mlp.fc1.lora_A.default | Linear | [4, 49, 64] | 🟢 Trainable |
| 387 | base_model.model.model.levels.3.blocks.1.mlp.fc1.lora_B.default | Linear | [4, 49, 2560] | 🟢 Trainable |
| 388 | base_model.model.model.levels.3.blocks.1.mlp.fc1 | Linear | [4, 49, 2560] | 🔒 Frozen |
| 389 | base_model.model.model.levels.3.blocks.1.mlp.act | GELU | [4, 49, 2560] | 🔒 Frozen |
| 390 | base_model.model.model.levels.3.blocks.1.mlp.drop1 | Dropout | [4, 49, 2560] | 🔒 Frozen |
| 391 | base_model.model.model.levels.3.blocks.1.mlp.norm | Identity | [4, 49, 2560] | 🔒 Frozen |
| 392 | base_model.model.model.levels.3.blocks.1.mlp.fc2.base_layer | Linear | [4, 49, 640] | 🔒 Frozen |
| 393 | base_model.model.model.levels.3.blocks.1.mlp.fc2.lora_dropout.default | Identity | [4, 49, 2560] | 🔒 Frozen |
| 394 | base_model.model.model.levels.3.blocks.1.mlp.fc2.lora_A.default | Linear | [4, 49, 64] | 🟢 Trainable |
| 395 | base_model.model.model.levels.3.blocks.1.mlp.fc2.lora_B.default | Linear | [4, 49, 640] | 🟢 Trainable |
| 396 | base_model.model.model.levels.3.blocks.1.mlp.fc2 | Linear | [4, 49, 640] | 🔒 Frozen |
| 397 | base_model.model.model.levels.3.blocks.1.mlp.drop2 | Dropout | [4, 49, 640] | 🔒 Frozen |
| 398 | base_model.model.model.levels.3.blocks.1.mlp | Mlp | [4, 49, 640] | 🔒 Frozen |
| 399 | base_model.model.model.levels.3.blocks.1.drop_path | DropPath | [4, 49, 640] | 🔒 Frozen |
| 400 | base_model.model.model.levels.3.blocks.1 | Block | [4, 49, 640] | 🔒 Frozen |
| 401 | base_model.model.model.levels.3.blocks.2.norm1 | LayerNorm | [4, 49, 640] | 🔒 Frozen |
| 402 | base_model.model.model.levels.3.blocks.2.mixer.qkv | Linear | [4, 49, 1920] | 🔒 Frozen |
| 403 | base_model.model.model.levels.3.blocks.2.mixer.q_norm | Identity | [4, 16, 49, 40] | 🔒 Frozen |
| 404 | base_model.model.model.levels.3.blocks.2.mixer.k_norm | Identity | [4, 16, 49, 40] | 🔒 Frozen |
| 405 | base_model.model.model.levels.3.blocks.2.mixer.proj | Linear | [4, 49, 640] | 🔒 Frozen |
| 406 | base_model.model.model.levels.3.blocks.2.mixer.proj_drop | Dropout | [4, 49, 640] | 🔒 Frozen |
| 407 | base_model.model.model.levels.3.blocks.2.mixer | Attention | [4, 49, 640] | 🔒 Frozen |
| 408 | base_model.model.model.levels.3.blocks.2.drop_path | DropPath | [4, 49, 640] | 🔒 Frozen |
| 409 | base_model.model.model.levels.3.blocks.2.norm2 | LayerNorm | [4, 49, 640] | 🔒 Frozen |
| 410 | base_model.model.model.levels.3.blocks.2.mlp.fc1.base_layer | Linear | [4, 49, 2560] | 🔒 Frozen |
| 411 | base_model.model.model.levels.3.blocks.2.mlp.fc1.lora_dropout.default | Identity | [4, 49, 640] | 🔒 Frozen |
| 412 | base_model.model.model.levels.3.blocks.2.mlp.fc1.lora_A.default | Linear | [4, 49, 64] | 🟢 Trainable |
| 413 | base_model.model.model.levels.3.blocks.2.mlp.fc1.lora_B.default | Linear | [4, 49, 2560] | 🟢 Trainable |
| 414 | base_model.model.model.levels.3.blocks.2.mlp.fc1 | Linear | [4, 49, 2560] | 🔒 Frozen |
| 415 | base_model.model.model.levels.3.blocks.2.mlp.act | GELU | [4, 49, 2560] | 🔒 Frozen |
| 416 | base_model.model.model.levels.3.blocks.2.mlp.drop1 | Dropout | [4, 49, 2560] | 🔒 Frozen |
| 417 | base_model.model.model.levels.3.blocks.2.mlp.norm | Identity | [4, 49, 2560] | 🔒 Frozen |
| 418 | base_model.model.model.levels.3.blocks.2.mlp.fc2.base_layer | Linear | [4, 49, 640] | 🔒 Frozen |
| 419 | base_model.model.model.levels.3.blocks.2.mlp.fc2.lora_dropout.default | Identity | [4, 49, 2560] | 🔒 Frozen |
| 420 | base_model.model.model.levels.3.blocks.2.mlp.fc2.lora_A.default | Linear | [4, 49, 64] | 🟢 Trainable |
| 421 | base_model.model.model.levels.3.blocks.2.mlp.fc2.lora_B.default | Linear | [4, 49, 640] | 🟢 Trainable |
| 422 | base_model.model.model.levels.3.blocks.2.mlp.fc2 | Linear | [4, 49, 640] | 🔒 Frozen |
| 423 | base_model.model.model.levels.3.blocks.2.mlp.drop2 | Dropout | [4, 49, 640] | 🔒 Frozen |
| 424 | base_model.model.model.levels.3.blocks.2.mlp | Mlp | [4, 49, 640] | 🔒 Frozen |
| 425 | base_model.model.model.levels.3.blocks.2.drop_path | DropPath | [4, 49, 640] | 🔒 Frozen |
| 426 | base_model.model.model.levels.3.blocks.2 | Block | [4, 49, 640] | 🔒 Frozen |
| 427 | base_model.model.model.levels.3.blocks.3.norm1 | LayerNorm | [4, 49, 640] | 🔒 Frozen |
| 428 | base_model.model.model.levels.3.blocks.3.mixer.qkv | Linear | [4, 49, 1920] | 🔒 Frozen |
| 429 | base_model.model.model.levels.3.blocks.3.mixer.q_norm | Identity | [4, 16, 49, 40] | 🔒 Frozen |
| 430 | base_model.model.model.levels.3.blocks.3.mixer.k_norm | Identity | [4, 16, 49, 40] | 🔒 Frozen |
| 431 | base_model.model.model.levels.3.blocks.3.mixer.proj | Linear | [4, 49, 640] | 🔒 Frozen |
| 432 | base_model.model.model.levels.3.blocks.3.mixer.proj_drop | Dropout | [4, 49, 640] | 🔒 Frozen |
| 433 | base_model.model.model.levels.3.blocks.3.mixer | Attention | [4, 49, 640] | 🔒 Frozen |
| 434 | base_model.model.model.levels.3.blocks.3.drop_path | DropPath | [4, 49, 640] | 🔒 Frozen |
| 435 | base_model.model.model.levels.3.blocks.3.norm2 | LayerNorm | [4, 49, 640] | 🔒 Frozen |
| 436 | base_model.model.model.levels.3.blocks.3.mlp.fc1.base_layer | Linear | [4, 49, 2560] | 🔒 Frozen |
| 437 | base_model.model.model.levels.3.blocks.3.mlp.fc1.lora_dropout.default | Identity | [4, 49, 640] | 🔒 Frozen |
| 438 | base_model.model.model.levels.3.blocks.3.mlp.fc1.lora_A.default | Linear | [4, 49, 64] | 🟢 Trainable |
| 439 | base_model.model.model.levels.3.blocks.3.mlp.fc1.lora_B.default | Linear | [4, 49, 2560] | 🟢 Trainable |
| 440 | base_model.model.model.levels.3.blocks.3.mlp.fc1 | Linear | [4, 49, 2560] | 🔒 Frozen |
| 441 | base_model.model.model.levels.3.blocks.3.mlp.act | GELU | [4, 49, 2560] | 🔒 Frozen |
| 442 | base_model.model.model.levels.3.blocks.3.mlp.drop1 | Dropout | [4, 49, 2560] | 🔒 Frozen |
| 443 | base_model.model.model.levels.3.blocks.3.mlp.norm | Identity | [4, 49, 2560] | 🔒 Frozen |
| 444 | base_model.model.model.levels.3.blocks.3.mlp.fc2.base_layer | Linear | [4, 49, 640] | 🔒 Frozen |
| 445 | base_model.model.model.levels.3.blocks.3.mlp.fc2.lora_dropout.default | Identity | [4, 49, 2560] | 🔒 Frozen |
| 446 | base_model.model.model.levels.3.blocks.3.mlp.fc2.lora_A.default | Linear | [4, 49, 64] | 🟢 Trainable |
| 447 | base_model.model.model.levels.3.blocks.3.mlp.fc2.lora_B.default | Linear | [4, 49, 640] | 🟢 Trainable |
| 448 | base_model.model.model.levels.3.blocks.3.mlp.fc2 | Linear | [4, 49, 640] | 🔒 Frozen |
| 449 | base_model.model.model.levels.3.blocks.3.mlp.drop2 | Dropout | [4, 49, 640] | 🔒 Frozen |
| 450 | base_model.model.model.levels.3.blocks.3.mlp | Mlp | [4, 49, 640] | 🔒 Frozen |
| 451 | base_model.model.model.levels.3.blocks.3.drop_path | DropPath | [4, 49, 640] | 🔒 Frozen |
| 452 | base_model.model.model.levels.3.blocks.3 | Block | [4, 49, 640] | 🔒 Frozen |
| 453 | base_model.model.model.levels.3 | MambaVisionLayer | [1, 640, 12, 12] (+1 aux) | 🔒 Frozen |
| 454 | base_model.model.model.norm | BatchNorm2d | [1, 640, 12, 12] | 🔒 Frozen |
| 455 | base_model.model.model.avgpool | AdaptiveAvgPool2d | [1, 640, 1, 1] | 🔒 Frozen |
| 456 | base_model.model.model.head | Linear | [1, 1000] | 🔒 Frozen |

</details>

## Repository Structure

```
wafer-mamba/
├── notebooks/
│   ├── training/                       # Kaggle training notebooks
│   │   ├── wafer-mamba-hybrid.ipynb    # Hybrid Quantum-MambaVision (proposed)
│   │   ├── wafer-mamba-classical.ipynb # Classical MambaVision + LoRA
│   │   ├── wafer-resnet.ipynb          # ResNet-50 + LoRA baseline
│   │   └── wafer-vit.ipynb             # ViT-Small baseline
│   └── analysis/
│       ├── analysis_for_paper.ipynb    # Figures & tables for the paper
│       ├── detailed_analysis.ipynb     # Extended analysis
│       └── requirements.txt            # Dependencies for analysis
├── models/                             # Trained model checkpoints (.pth)
│   ├── hybrid-mamba/
│   ├── classical-mamba/
│   ├── resnet/
│   └── vit/
├── results/                            # Per-epoch predictions & logs (.json)
│   ├── hybrid-mamba/
│   ├── classical-mamba/
│   ├── resnet/
│   └── vit/
├── figures/                            # Publication-ready figures (PDF)
│   ├── overal_architecture_diagram.pdf
│   ├── quantum_adapter_diagram.pdf
│   ├── lora_diagram.pdf
│   ├── training_dynamics_*.pdf
│   ├── defect_distribution.pdf
│   └── ... (18 figures total)
└── data/
    └── Description.pdf                 # Dataset description
```

## Dataset

**MixedType Wafer Defect Datasets** — available on [Kaggle](https://www.kaggle.com/datasets/co1d7era/mixedtype-wafer-defect-datasets).

- Format: `.npz` (wafer map images + multi-hot labels)
- 8 defect classes, multi-label
- Split: 70% train / 10% validation / 20% test (stratified)
- Input resolution: 384 × 384 (grayscale → 3-channel)

## Getting Started

### Training (Kaggle — NVIDIA T4 GPU)

All four models were trained on **Kaggle** notebooks with a T4 GPU. Each notebook's first cell installs the required dependencies:

```bash
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1
pip install causal-conv1d==1.4.0 --no-build-isolation
pip install mamba-ssm==2.2.4 --no-build-isolation
pip install mambavision timm pillow
pip install "transformers>=4.43" "accelerate>=0.33" "datasets>=2.19" "evaluate>=0.4" "peft>=0.12.0" "scikit-learn>=1.3" "bitsandbytes>=0.43.0"
pip install pennylane pennylane-lightning[gpu]
```

To reproduce training:
1. Upload the [MixedType Wafer Defect Dataset](https://www.kaggle.com/datasets/co1d7era/mixedtype-wafer-defect-datasets) as a Kaggle dataset
2. Open the desired notebook from `notebooks/training/` in a Kaggle notebook environment with GPU T4 × 2 accelerator
3. Run all cells — each notebook is self-contained

### Training Configuration

| Parameter | Value |
|---|---|
| Epochs | 15 |
| Batch size | 64 × 2 (gradient accumulation) |
| Optimizer | AdamW (fused) |
| Learning rate | 6e-4 (Mamba, ViT) / 5e-5 (ResNet) |
| Scheduler | Cosine with 10% warmup |
| Early stopping | Patience = 3 (on Macro-F1) |
| Precision | FP16 |
| Seed | 42 |

### Analysis (Local)

For running the analysis notebooks locally:

```bash
# Create virtual environment (Python 3.11)
python -m venv .venv

# Activate
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install analysis dependencies
pip install -r notebooks/analysis/requirements.txt
```

The analysis notebooks load pre-computed results from `results/` and generate all figures in `figures/`.
