# SyncClip

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org)
[![Transformers](https://img.shields.io/badge/Transformers-4.20%2B-orange.svg)](https://huggingface.co/transformers)
[![Gradio](https://img.shields.io/badge/Gradio-UI-green.svg)](https://gradio.app)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

一个基于深度学习的工具，用于将 YouTube 解说视频与电影原片进行视觉匹配、片段提取和同步剪辑。结合 CLIP 模型的语义匹配、BLIP 模型的场景描述、Whisper 的音频转录，以及高级优化如核密度估计和时间跳跃处理，实现自动生成同步的高光剪辑视频。

**作者**: 52hertzjingluo

## ✨ 核心特性

### 多模态匹配与分析
- **CLIP 视觉特征匹配**: 使用 OpenAI CLIP 模型计算帧间语义相似度，支持高精度匹配。
- **BLIP 场景描述生成**: 通过 Salesforce BLIP 模型为关键帧生成详细文本描述，提升匹配准确性。
- **Whisper 音频转录**: 提取解说音频并转录文本，支持多种模型大小（tiny, base, small, medium, large），用于辅助匹配和报告生成。
- **核密度估计优化**: 使用 SciPy 的 gaussian_kde 进行片段匹配平滑处理，提高连续性。

### 智能视频处理
- **竖屏视频自动裁剪**: 检测视频方向（横屏/竖屏/方形），自动去除黑边并裁剪内容区域，支持自定义采样帧数。
- **时间跳跃智能处理**: 允许非线性匹配，处理电影中的闪回或跳跃场景，支持自适应阈值调整。
- **特征缓存与断点续传**: 使用 MD5 哈希和 pickle/numpy 缓存提取的特征，避免重复计算。
- **自适应时长调整**: 自动分组匹配片段，确保输出片段时长与 YouTube 视频严格同步，无速度调整。

### 用户界面与输出
- **Gradio Web 界面**: 通过 `FCR_G.py` 提供交互式 UI，支持文件上传、参数调整和实时进度显示。
- **无缝视频合并**: 使用 FFmpeg 合并剪辑片段，支持高质量 H.264 编码和过渡效果。
- **详细报告生成**: 输出 JSON 匹配结果、TXT 质量报告、场景描述和候选片段信息。
- **资源管理**: 自动清理临时文件，支持 GPU/CPU 自动切换。

### 高级优化
- **相似度阈值自适应**: 自动重试降低阈值（最多 3 次），确保匹配质量置信度 >50%。
- **批量处理支持**: 支持并行提取帧和特征，优化内存使用。
- **质量评估**: 计算匹配置信度、时间跳跃统计和覆盖率，提供改进建议。

## 🛠️ 环境要求

### 系统要求
- **操作系统**: Linux / macOS / Windows（推荐 Linux/macOS 以获得最佳性能）
- **Python版本**: 3.8+
- **硬件**: 推荐 NVIDIA GPU（CUDA 11+）以加速模型推理；至少 8GB RAM；如果无 GPU，可 fallback 到 CPU，但速度较慢。

### 核心依赖
```txt
torch>=1.9.0
torchvision>=0.10.0
transformers>=4.20.0
opencv-python>=4.5.0
whisper (OpenAI Whisper)
gradio>=3.0.0 (仅用于 Gradio 界面)
numpy>=1.21.0
scipy>=1.7.0
Pillow>=8.0.0
yt-dlp (可选，用于在线下载 YouTube 视频)
```

### 外部工具
- **FFmpeg**: 用于视频裁剪、音频提取和合并（必需）。
- **CUDA Toolkit** (可选): 如果使用 GPU，确保安装匹配的 CUDA 版本。

## 📦 安装指南

### 1. 克隆项目
```bash
git clone <repository-url>
cd youtube-movie-matcher
```

### 2. 创建虚拟环境
```bash
# 使用conda (推荐)
conda create -n matcher python=3.9
conda activate matcher

# 或使用venv
python -m venv matcher
source matcher/bin/activate  # Linux/macOS
# matcher\Scripts\activate  # Windows
```

### 3. 安装Python依赖
```bash
# 基础安装 (GPU版本)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers opencv-python numpy scipy Pillow gradio openai-whisper yt-dlp

# 或使用requirements.txt（如果有）
pip install -r requirements.txt
```

### 4. 安装FFmpeg
```bash
# Ubuntu/Debian
sudo apt update && sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Windows
# 下载FFmpeg可执行文件并添加到PATH环境变量（从 https://ffmpeg.org/download.html）
```

### 5. 验证安装
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
ffmpeg -version
whisper --model base --help  # 测试Whisper
```

如果出现模型下载失败，可设置 Hugging Face 镜像：
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

## 🚀 快速开始

### 命令行使用 (`Film_commentary_reorganized.py`)
```python
from Film_commentary_reorganized import main  # 假设main函数已定义

# 基本用法
main(
    youtube_url="https://www.youtube.com/watch?v=example",  # 或本地路径
    movie_path="path/to/movie.mp4",
    output_dir="path/to/output",
    frame_interval=1.2,  # 帧间隔（秒）
    similarity_threshold=0.80,  # 相似度阈值
    whisper_model="base",  # Whisper模型
    allow_time_jumps=True  # 允许时间跳跃
)
```

### Gradio 界面使用 (`FCR_G.py`)
运行脚本启动 Web 界面：
```bash
python FCR_G.py
```
- 访问 http://localhost:7866
- 上传 YouTube 解说视频和电影文件。
- 输入输出目录（所有文件将保存到 `<output_dir>/proceed`）。
- 调整参数并点击“开始处理”。

输出将包括最终视频、匹配报告和质量报告。

## 📚 API文档

### 主类 `YouTubeMovieMatcher`
```python
class YouTubeMovieMatcher:
    def __init__(self, output_dir: str, log_level: str = "INFO"):
        # 初始化输出目录、日志、设备等
```

#### 核心方法
```python
def process_visual_matching_enhanced(
    self, youtube_url: str, movie_path: str,
    frame_interval: float = 1.0, similarity_threshold: float = 0.85,
    whisper_model: str = "base", allow_time_jumps: bool = True,
    use_cache: bool = True
) -> Optional[Dict]:
    """
    完整处理流程：下载/加载视频、预处理、提取帧/特征、匹配、生成片段和视频。

    参数:
        youtube_url: YouTube URL 或本地路径 (str)
        movie_path: 电影文件路径 (str)
        frame_interval: 帧提取间隔 (float, 默认1.0秒)
        similarity_threshold: 相似度阈值 (float, 默认0.85)
        whisper_model: Whisper模型大小 (str, 默认"base")
        allow_time_jumps: 是否允许时间跳跃 (bool, 默认True)
        use_cache: 是否使用特征缓存 (bool, 默认True)

    返回:
        Dict: 处理结果，包括状态、输出路径、质量信息等
    """
```

#### 辅助方法（示例）
- `load_models()`: 加载 CLIP 和 BLIP 模型。
- `detect_video_orientation(video_path: str) -> Dict`: 检测视频方向和内容区域。
- `extract_frames_with_timestamps(video_path: str, interval: float) -> List[Dict]`: 提取带时间戳的帧。
- `match_frames_combined(...) -> Tuple[List[Dict], List[Dict]]`: 进行 CLIP+BLIP 匹配，返回主/备匹配。
- `group_matches_into_segments_flexible(matches: List[Dict]) -> List[Dict]`: 分组匹配为片段。
- `create_synchronized_clips_precise(...) -> List[str]`: 生成精确同步剪辑。
- `merge_clips_seamless(clips: List[str], output_video: str) -> bool`: 无缝合并剪辑。

更多方法详见代码文件。

## ⚙️ 配置选项

### 匹配参数
```python
# 在process_visual_matching_enhanced中调整
frame_interval=1.0      # 越小越精确，但计算量越大
similarity_threshold=0.85  # 0.7-0.9 范围，较低时匹配更多但质量可能下降
whisper_model="medium"  # 更大模型转录更准，但更慢
allow_time_jumps=False  # 禁用以强制线性匹配
```

### 设备优化
```python
# 自动检测，也可手动
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 指定GPU
```

### 缓存管理
- 特征保存在 `output_dir/proceed/cache`。
- 要清除缓存：删除缓存目录。

## 📝 使用示例

### 示例1: 命令行匹配 YouTube 视频
```python
matcher = YouTubeMovieMatcher(output_dir="/path/to/output")
result = matcher.process_visual_matching_enhanced(
    youtube_url="https://www.youtube.com/watch?v=O80j8oHUD10",
    movie_path="/path/to/movie.mp4",
    frame_interval=1.2,
    similarity_threshold=0.80
)
print(result["output_video"])  # /path/to/output/proceed/final_output.mp4
```

### 示例2: Gradio 界面处理本地文件
- 上传文件到界面。
- 设置输出目录为 `/path/to/output`。
- 输出保存在 `/path/to/output/proceed`（包括 `final_output.mp4`、`match_report.txt` 等）。

### 示例3: 处理竖屏解说视频
工具自动检测并裁剪黑边，确保匹配准确。

## 📊 输出格式

### 最终视频
- **路径**: `output_dir/proceed/final_output.mp4`
- **格式**: MP4 (H.264/AAC)，时长与 YouTube 视频匹配片段总和一致。

### 匹配结果 JSON
- **路径**: `output_dir/proceed/results/visual_match_results.json`
- **内容示例**:
```json
{
  "quality_info": {"confidence": 85.5},
  "primary_segments": [{"youtube_start": 0.0, "movie_start": 60.0, "avg_similarity": 0.92}],
  "time_jump_stats": {"total_segments": 10, "segments_with_jumps": 3}
}
```

### 报告文件
- **匹配报告**: `match_report.txt` - 片段详情、转录文本。
- **质量报告**: `quality_report.txt` - 置信度、覆盖率、改进建议。
- **场景描述**: `scene_descriptions.json` - 帧级描述。
- **候选片段**: `alternative_segments/alternative_segments.json` - 次优匹配。

### 日志文件
- **路径**: `output_dir/proceed/logs/visual_matcher_YYYYMMDD_HHMMSS.log`
- **内容**: 详细处理日志、性能指标、错误信息。

## 🔧 故障排除

### 常见问题

#### 1. CUDA 错误（如 out of memory）
```
RuntimeError: CUDA out of memory
```
**解决方案**:
- 减少 `frame_interval` 或使用 CPU: `export CUDA_VISIBLE_DEVICES=""`
- 清理缓存: 删除 `cache` 目录
- 升级 GPU 驱动

#### 2. FFmpeg 未找到
```
FileNotFoundError: 'ffmpeg'
```
**解决方案**:
- 安装 FFmpeg 并确保在 PATH 中: `which ffmpeg`

#### 3. 模型下载失败
```
ConnectionError: Failed to download model
```
**解决方案**:
- 设置镜像: `export HF_ENDPOINT=https://hf-mirror.com`
- 检查网络或使用 VPN

#### 4. 匹配质量低
- 检查阈值并重试。
- 确保电影版本匹配解说内容。
- 增加帧间隔以覆盖更多内容。

#### 5. Gradio 界面无法启动
- 检查端口 7866 是否占用。
- 运行 `gradio reload` 或重启脚本。

如果问题持续，请检查日志文件并在 Issues 中报告。

## 🔬 技术原理

### 整体架构
```
输入: YouTube视频 + 电影原片
↓
视频预处理 (方向检测 + 裁剪)
↓
帧提取 + 特征计算 (CLIP嵌入 + BLIP描述)
↓
相似度匹配 (余弦相似度 + 文本匹配)
↓
片段分组 (核密度估计 + 时间跳跃处理)
↓
剪辑生成 + 合并 (FFmpeg)
↓
输出: 同步视频 + 报告
```

### 关键算法
- **相似度计算**: CLIP 特征的 cosine_similarity + BLIP 描述的语义权重。
- **片段优化**: gaussian_kde 平滑匹配分布；interp1d 插值时间序列。
- **质量评估**: 平均相似度 * 覆盖率 * (1 - 跳跃惩罚)。
- **缓存机制**: MD5 哈希确保文件唯一性，支持断点续传。

## 🤝 贡献指南

### 开发环境设置
1. Fork 项目。
2. 创建分支: `git checkout -b feature/new-feature`
3. 安装开发依赖: `pip install -r requirements-dev.txt` (添加 pytest 等)。
4. 运行测试: `python -m pytest tests/`

### 代码规范
- 遵循 PEP 8。
- 添加类型注解 (typing)。
- 编写 docstring。
- 确保测试覆盖率 >80%。

### 提交流程
1. 测试通过。
2. 提交: `git commit -m "feat: add new feature"`
3. 推送: `git push origin feature/new-feature`
4. 创建 Pull Request。

欢迎贡献新功能，如添加更多模型支持或优化算法。

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 致谢

- [OpenAI CLIP](https://github.com/openai/CLIP) - 视觉语义匹配。
- [Salesforce BLIP](https://github.com/salesforce/BLIP) - 图像描述生成。
- [OpenAI Whisper](https://github.com/openai/whisper) - 音频转录。
- [FFmpeg](https://ffmpeg.org/) - 视频处理。
- [Gradio](https://gradio.app/) - Web 界面。
- [SciPy](https://scipy.org/) - 科学计算。

## 联系方式

- **作者**: 52hertzjingluo
- **问题反馈**: 请使用 GitHub Issues。
- **功能建议**: 欢迎提交 Pull Request。

---

⭐ 如果这个项目对您有帮助，请给个 Star 支持！
