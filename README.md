# Mini-Omni
本仓库基于 Mini-Omni 推理框架，构建一个面向多模态端到端语音的最小可用推理系统。目标是实现本地语音输入到语音输出的闭环，并逐步扩展图像和其他模态。

本项目并非官方仓库的完整复刻，而是基于个人任务需求而整理的精简版本。

---

## 一、项目内容概述

本仓库包含如下核心能力：

1. 基于官方推理脚本裁剪与封装的最小推理引擎 OmniEngine  
2. 支持本地音频输入  
3. 支持完整音频输出生成（非流式）  
4. 支持流式音频输出和打断机制  
5. 支持多模态输入：音频 + 图像  
6. 可用于命令行脚本、Streamlit、Gradio 等多种上层应用  
7. 与官方 inference.py 完全兼容

---

## 二、代码结构

```
mini-omni/
├── inference.py
├── omni_engine.py
├── test_omni_engine.py
├── test_interrupt.py
├── server.py
├── realtime_vad_player.py
├── stream_test.py
├── demo_snac_token_decode.py
├── webui/
│   ├── omni_gradio.py
│   └── omni_streamlit.py
├── utils/
├── data/
└── checkpoint/ (首次运行会自动下载)
```

其中 omni_engine.py 为自定义推理引擎核心文件。server.py 提供 Flask API 服务。webui/ 包含 Gradio 和 Streamlit 界面。realtime_vad_player.py 实现实时语音交互。

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

## 四、API 服务

文件: server.py

提供 Flask API 服务，支持流式音频输出。

运行方式:

```sh
python server.py
```

API 端点:

- POST /chat: 接受音频数据，返回流式音频响应。

---

## 五、Web 界面

### Gradio 界面

文件: webui/omni_gradio.py

运行方式:

```sh
python webui/omni_gradio.py
```

支持上传音频文件并生成响应。

### Streamlit 界面

文件: webui/omni_streamlit.py

运行方式:

```sh
streamlit run webui/omni_streamlit.py
```

支持实时语音录制和播放。

---

## 六、实时语音交互

文件: realtime_vad_player.py

运行方式:

```sh
python realtime_vad_player.py
```

使用麦克风实时输入语音，自动检测语音活动并生成响应。支持打断机制（用户说话时停止AI回复）和多模态输入（音频+图像）。

### 打断机制
- 当 AI 正在播放回复时，如果检测到用户新语音，立即停止播放并清空队列。

### 视觉流采样
- 后台线程每秒捕获摄像头帧。
- 检测到说话开始时，抓取当前帧，resize 并与音频一起送给引擎。

---

## 七、打断机制测试

文件: test_interrupt.py

运行方式:

```sh
python test_interrupt.py
```

使用音频文件模拟实时输入，测试打断机制是否正常工作。

## 七、最小运行样例

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

## 八、文本到音频演示

文件: demo_snac_token_decode.py

运行方式:

```sh
python demo_snac_token_decode.py
```

演示从文本生成音频的功能，使用 SNAC 解码器输出 wav 文件。

---

## 九、开发任务与里程碑

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

### 2. 第二阶段目标（截至 12 月 12 日）

目标实现流式音频输出、API 服务和 Web 界面。

具体任务包括：

- **构建统一的 API 服务层**  
  已完成: server.py 提供 Flask API，支持流式音频输出。

- **对接前端 WebSocket 实现流式音频输出**  
  已完成: webui/omni_gradio.py 和 omni_streamlit.py 实现 Gradio 和 Streamlit 界面，支持实时交互。

- **实现实时语音交互**  
  已完成: realtime_vad_player.py 支持麦克风输入和实时播放响应。

- **添加演示脚本**  
  已完成: demo_snac_token_decode.py 演示文本到音频生成。

当前进度: 第二阶段已完成，项目支持多种交互方式，包括命令行、Web UI 和实时语音。

### 3. 第三阶段目标（截至 12 月 12 日）

目标实现打断机制和多模态视觉输入。

具体任务包括：

- **实现打断机制**  
  已完成: realtime_vad_player.py 支持在播放时检测新语音并停止当前回复。

- **添加视觉流采样与对齐**  
  已完成: 集成 OpenCV，检测说话开始时捕获图像，与音频一起送给引擎。

- **添加测试脚本**  
  已完成: test_interrupt.py 用于测试打断机制。

当前进度: 第三阶段已完成，项目支持全双工对话和多模态输入。

## 十、后续工作计划

1. 完善错误处理、异常输出与日志
2. 完整撰写阶段报告文档
3. 添加更多演示和测试用例
4. 优化视觉输入处理和模型集成

---

## 十一、环境安装

```sh
conda create -n omni python=3.10
conda activate omni
pip install -r requirements.txt
```

模型会在首次运行 inference 或 OmniEngine 时自动下载。

---

## 十二、免责声明

本项目为个人学习与研究目的构建，与官方项目无直接关联。严禁将本仓库用于任何商业用途或违反相关法律法规的场景。
