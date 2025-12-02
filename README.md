# Mini-Omni
本仓库基于 Mini-Omni 推理框架，构建一个面向多模态端到端语音的最小可用推理系统。目标是实现本地语音输入到语音输出的闭环，并逐步扩展图像和其他模态。

本项目并非官方仓库的完整复刻，而是基于个人任务需求而整理的精简版本。

---

## 一、项目内容概述

本仓库包含如下核心能力：

1. 基于官方推理脚本裁剪与封装的最小推理引擎 OmniEngine  
2. 支持本地音频输入  
3. 支持完整音频输出生成（非流式）  
4. 可用于命令行脚本、Streamlit、Gradio 等多种上层应用  
5. 与官方 inference.py 完全兼容

---

## 二、代码结构

```
mini-omni/
├── inference.py
├── omni_engine.py
├── test_omni_engine.py
├── webui/
├── utils/
├── data/
└── checkpoint/ (首次运行会自动下载)
```

其中 omni_engine.py 为自定义推理引擎核心文件。

---

## 三、OmniEngine 简介

OmniEngine 对官方 OmniInference 进行包装，提供一个更清晰的接口用于开发和测试。

核心功能包括:

1. generate(audio_path):  
   输入一段音频文件，输出完整回答音频的 PCM 字节数据。

2. 支持返回 numpy 格式音频，方便进一步处理。

3. 底层依赖官方 run_AT_batch_stream，实现原生的 Snac 解码音频输出。

该类为本项目最重要的基础模块，后续所有前端、API、WebSocket 等均会基于其实现。

---

## 四、OmniEngine 源码

文件路径: omni_engine.py

```python
from omni_engine import OmniEngine
import wave
import numpy as np
```

源码位置参考仓库文件 omni_engine.py。

---

## 五、最小运行样例

文件: test_omni_engine.py

运行方式:

```sh
conda activate omni
python test_omni_engine.py
```

示例会:

1. 加载模型
2. 读取 `data/samples/output1.wav`
3. 使用 OmniEngine 生成回答音频
4. 将回答保存为 `response.wav`

---

## 六、开发任务与里程碑

以下为当前项目的阶段任务规划。

### 1. 第一阶段目标（截至 12 月 4 日）

目标准确性要求为完成 Mini-Omni 最小推理闭环。

具体任务包括：

- **编写 OmniEngine 类**  
  已完成: OmniEngine 已实现基本推理与音频合成

- **支持本地音频输入并产生完整输出音频**  
  已完成: 可复现音频输入到音频输出

- **能够生成一个可播放的 wav 文件**  
  已完成: test_omni_engine.py 已验证

- **能够从 run_AT_batch_stream 中获取音频流并合并**  
  已完成: generate 方法已封装

- **编写简单测试脚本**  
  已完成

当前进度: 第一阶段已完成 OmniEngine 的实现，并验证推理功能正常工作。

---

## 七、后续工作计划

1. 扩展 generate_multimodal 接口，加入图像特征
2. 对接前端 WebSocket 实现流式音频输出
3. 完善错误处理、异常输出与日志
4. 构建统一的 API 服务层
5. 完整撰写阶段报告文档

---

## 八、环境安装

```sh
conda create -n omni python=3.10
conda activate omni
pip install -r requirements.txt
```

模型会在首次运行 inference 或 OmniEngine 时自动下载。

---

## 九、免责声明

本项目为个人学习与研究目的构建，与官方项目无直接关联。严禁将本仓库用于任何商业用途或违反相关法律法规的场景。
