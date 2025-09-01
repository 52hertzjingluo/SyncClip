# SyncClip (影音同步剪) v2.0

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org)
[![Transformers](https://img.shields.io/badge/Transformers-4.20%2B-orange.svg)](https://huggingface.co/transformers)
[![Gradio](https://img.shields.io/badge/Gradio-UI-green.svg)](https://gradio.app)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**SyncClip (影音同步剪)** v2.0 是一个基于深度学习的工具，用于将 YouTube 解说视频与电影原片进行视觉匹配、片段提取和同步剪辑。v2.0 版本引入了基于镜头检测的非线性重组匹配架构、ChatterboxTTS 配音生成、T5 意译系统、增强的 CSV 批量处理和音频分析功能。结合 CLIP 和 DINOv2 模型的语义匹配、BLIP 的场景描述、Whisper 的音频转录、librosa 的音频特征提取，以及 PySceneDetect 的场景检测，实现更智能的剪辑和交互体验。

**作者**: 52hertzjingluo
**版本**: 2.0  

## 🆕 v2.0 更新说明（重点：Gradio 界面 `FCR_G.py`）

以下是 `FCR_G.py`（Gradio 界面）相对于旧版（基于 `Film_commentary_reorganized.py` 的 Gradio 版本）的更新之处，结合 `FCR_test_v2.0.py` 的增强功能：

### 1. **支持 CSV 批量处理**
- **新增功能**: 现支持上传 CSV 文件进行批量任务处理，用户可上传包含多组 YouTube 视频和电影路径的 CSV 文件，自动批量生成同步剪辑。
- **界面改进**:
  - 新增 CSV 文件上传控件，支持解析 `input.csv`（包含 `youtube_clip` 和 `movie_path` 字段）。
  - 显示批量任务进度条，实时更新每个任务的状态（成功/失败）。
  - 提供批量结果下载链接，包括每个任务的 `FINAL_task_xxxx.mp4` 和汇总报告 `batch_summary.json`。
- **技术支持**: 集成 `process_csv_batch` 方法，自动为每个任务创建独立目录，生成详细的 `match_report.txt` 和 `quality_report.txt`。

### 2. **集成 ChatterboxTTS 配音**
- **新增功能**: 当检测到解说视频时，支持生成基于意译文稿的 TTS 配音音轨，使用 ChatterboxTTS 模型，并按原始时间戳精确对齐。
- **界面改进**:
  - 新增 TTS 参数配置面板，允许用户调整 `exaggeration`（夸张度）和 `cfg_weight`（配置权重）。
  - 显示 TTS 生成进度和预览音频片段。
  - 提供选项切换原始音频或 TTS 音轨用于最终合成。
- **技术支持**: 调用 `stage5_generate_and_compose_tts_audio` 方法，使用 FFmpeg 进行延迟对齐和音轨混合。

### 3. **双文稿意译系统**
- **新增功能**: 集成 T5 模型（`humarin/chatgpt_paraphraser_on_T5_base`），为解说视频生成原始和意译双文稿，均带时间戳，用于 TTS 或报告。
- **界面改进**:
  - 显示原始转录文稿和意译文稿的对比视图（支持下载为 JSON）。
  - 允许用户选择是否启用意译（默认启用）。
  - 提供文稿预览功能，展示时间戳和文本内容。
- **技术支持**: 使用 `ParaphraseGenerator` 类进行意译，保存结果至 `original_transcript` 和 `paraphrased_transcript`。

### 4. **增强的音频分析**
- **新增功能**: 使用 librosa 提取 MFCC、F0、谱质心等特征，判断输入视频是否为解说音频（基于 `AudioAnalyzer` 类）。
- **界面改进**:
  - 显示音频相似度得分和解说/原片判断结果。
  - 提供音频特征可视化（例如 MFCC 热图），帮助用户理解分析过程。
- **技术支持**: 集成 `is_narration` 方法，结合加权相似度计算（MFCC: 0.5, F0: 0.2, 谱特征: 0.2, 能量: 0.1）。

### 5. **基于镜头检测的非线性匹配**
- **新增功能**: 引入 PySceneDetect 的 `ContentDetector`，实现镜头级非线性重组匹配，支持时间跳跃和闪回场景。
- **界面改进**:
  - 显示镜头分割结果和匹配时间线预览。
  - 新增 `shot_similarity_threshold` 滑块，允许用户动态调整镜头匹配阈值（默认 0.82）。
  - 提供粗定位间隔（`sparse_interval_sec`）配置，默认 2 秒。
- **技术支持**: 集成 `stage0_generate_feature_bank`、`stage1_global_coarse_localization`、`stage2_shot_matching` 和 `stage3_intra_shot_alignment` 方法。

### 6. **优化的竖屏蒙版裁剪**
- **新增功能**: 增强竖屏视频处理，自动检测黑边并应用蒙版裁剪，提升匹配精度。
- **界面改进**:
  - 显示视频方向检测结果（横屏/竖屏/方形）和裁剪区域预览。
  - 允许用户手动调整裁剪参数（如蒙版边界）。
- **技术支持**: 增强 `detect_video_orientation` 和 `preprocess_video` 方法，支持动态蒙版生成。


### 7. **性能与兼容性优化**
- **新增功能**:
  - 支持 Apple Silicon MPS 加速（除 CUDA/CPU 外）。
  - 增强缓存机制，优化特征保存/加载效率（使用 `save_features` 和 `load_features`）。
- **界面改进**:
  - 显示设备选择（CPU/GPU/MPS）和性能统计。
  - 提供缓存清理按钮，释放磁盘空间。
- **技术支持**: 集成 `get_device` 方法，自动检测最佳硬件。

## 🛠️ 环境要求

### 系统要求
- **操作系统**: Linux / macOS / Windows（推荐 Linux/macOS 以获得最佳性能）
- **Python版本**: 3.8+
- **硬件**: 推荐 NVIDIA GPU（CUDA 11+）或 Apple Silicon（MPS）；至少 8GB RAM；CPU 可作为 fallback。

### 核心依赖
```txt
torch>=1.9.0
torchvision>=0.10.0
transformers>=4.20.0
opencv-python>=4.5.0
openai-whisper
gradio>=3.0.0
numpy>=1.21.0
scipy>=1.7.0
Pillow>=8.0.0
librosa>=0.9.0
soundfile>=0.10.0
py-scenedetect>=0.6.0
torchaudio>=0.10.0
chatterbox-tts
yt-dlp (可选)
pandas>=1.3.0
```

### 外部工具
- **FFmpeg**: 用于视频裁剪、音频提取、TTS 合成和合并（必需）。
- **CUDA Toolkit** / **MPS**: 根据硬件选择。

## 📦 安装指南

### 1. 克隆项目
```bash
git clone <repository-url>
cd syncclip
```

### 2. 创建虚拟环境
```bash
# 使用conda (推荐)
conda create -n syncclip python=3.9
conda activate syncclip

# 或使用venv
python -m venv syncclip
source syncclip/bin/activate  # Linux/macOS
# syncclip\Scripts\activate  # Windows
```

### 3. 安装Python依赖
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers opencv-python numpy scipy Pillow gradio openai-whisper yt-dlp pandas librosa soundfile py-scenedetect chatterbox-tts
```

### 4. 安装FFmpeg
```bash
# Ubuntu/Debian
sudo apt update && sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Windows
# 下载FFmpeg并添加至 PATH（https://ffmpeg.org/download.html）
```

### 5. 验证安装
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, MPS: {torch.backends.mps.is_available()}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
ffmpeg -version
whisper --model base --help
python -c "from chatterbox.tts import ChatterboxTTS; print('ChatterboxTTS 加载成功')"
```

如果模型下载失败：
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

## 🚀 快速开始

### 命令行使用 (`FCR_test_v2.0.py`)
```python
from FCR_test_v2_0 import main

# 全局配置
csv_path = "/path/to/input.csv"
output_dir = "/path/to/output"
audio_similarity_threshold = 0.65
shot_similarity_threshold = 0.82
sparse_interval_sec = 2
tts_params = {"exaggeration": 0.5, "cfg_weight": 0.2}

main()
```

### Gradio 界面使用 (`FCR_G.py`)
```bash
python FCR_G.py
```
- 访问 http://localhost:7866
- 上传 YouTube 视频/CSV 文件和电影文件。
- 设置输出目录和参数（阈值、TTS 配置）。
- 点击“开始处理”，查看实时进度和结果。

## 📚 API文档

### 主类 `YouTubeMovieMatcher`
```python
class YouTubeMovieMatcher:
    def __init__(self, output_dir: str, log_level: str = "INFO"):
        # 初始化输出目录、日志、设备等
```

#### 核心方法
```python
def process_csv_batch(
    self, csv_path: str, audio_similarity_threshold: float = 0.65,
    shot_similarity_threshold: float = 0.82, sparse_interval_sec: int = 2,
    tts_params: Dict = {"exaggeration": 0.5, "cfg_weight": 0.2}
) -> List[Dict]:
    """
    CSV批量处理流程。

    参数:
        csv_path: CSV文件路径 (str)
        audio_similarity_threshold: 音频相似度阈值 (float)
        shot_similarity_threshold: 镜头匹配阈值 (float)
        sparse_interval_sec: 粗定位间隔 (int)
        tts_params: TTS参数 (Dict)

    返回:
        List[Dict]: 任务结果
    """
```

## ⚙️ 配置选项

### 全局配置
```python
audio_similarity_threshold = 0.65
shot_similarity_threshold = 0.82
sparse_interval_sec = 2
tts_params = {"exaggeration": 0.5, "cfg_weight": 0.2}
```

### 设备优化
```python
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 指定GPU
```

### 缓存管理
- 特征保存在 `output_dir/proceed/cache`。
- 清理缓存：删除缓存目录。

## 📝 使用示例

### 示例1: CSV 批量处理
```python
matcher = YouTubeMovieMatcher(output_dir="/path/to/output")
results = matcher.process_csv_batch(csv_path="/path/to/input.csv")
```

### 示例2: Gradio 界面
- 上传 `input.csv` 或单个视频。
- 设置输出目录和参数。
- 输出保存至 `output_dir/batch_run_.../task_xxxx`。

## 📊 输出格式

### 最终视频
- **路径**: `output_dir/batch_run_.../task_xxxx/FINAL_task_xxxx.mp4`
- **格式**: MP4 (H.264/AAC)，包含 TTS 或原始音频。

### 报告文件
- **匹配报告**: `match_report.txt`
- **质量报告**: `quality_report.txt`
- **批量汇总**: `batch_summary.json`, `batch_report.txt`

### 日志文件
- **路径**: `output_dir/logs/matcher_log_YYYYMMDD_HHMMSS.log`

## 🔧 故障排除

### 常见问题
1. **CUDA/MPS 错误**:
   - 减少 `sparse_interval_sec` 或使用 CPU: `export CUDA_VISIBLE_DEVICES=""`
2. **FFmpeg 未找到**:
   - 安装 FFmpeg 并添加至 PATH。
3. **模型下载失败**:
   - 设置 `HF_ENDPOINT=https://hf-mirror.com`。
4. **匹配质量低**:
   - 调整阈值或检查电影版本。
5. **TTS 失败**:
   - 检查 ChatterboxTTS 依赖或调整参数。

## 🔬 技术原理

### 整体架构
```
输入: CSV/视频
↓
预处理 & 音频分析
↓
转录 & 意译
↓
镜头检测 & 特征提取
↓
非线性匹配 & 渲染
↓
TTS合成 & 最终合并
↓
输出: 视频 + 报告
```

### 关键算法
- **音频分析**: MFCC/F0/谱特征 + 加权相似度。
- **意译**: T5 paraphrase。
- **镜头匹配**: PySceneDetect + CLIP/DINOv2。
- **TTS 合成**: ChatterboxTTS + FFmpeg。

## 🤝 贡献指南

1. Fork 项目。
2. 创建分支: `git checkout -b feature/new-feature`
3. 安装开发依赖: `pip install -r requirements-dev.txt`
4. 运行测试: `python -m pytest tests/`
5. 提交 Pull Request。

## 📄 许可证

MIT 许可证 - 查看 [LICENSE](LICENSE)。

## 致谢

- [OpenAI CLIP](https://github.com/openai/CLIP)
- [Facebook DINOv2](https://github.com/facebookresearch/dinov2)
- [Salesforce BLIP](https://github.com/salesforce/BLIP)
- [OpenAI Whisper](https://github.com/openai/whisper)
- [ChatterboxTTS](https://github.com/chatterbox-tts)
- [Librosa](https://librosa.org/)
- [PySceneDetect](https://pyscenedetect.readthedocs.io/)
- [FFmpeg](https://ffmpeg.org/)
- [Gradio](https://gradio.app/)
- [SciPy](https://scipy.org/)

## 联系方式

- **作者**: 52hertzjingluo
- **问题反馈**: GitHub Issues
- **功能建议**: 提交 Pull Request

---

⭐ 请给个 Star 支持！
