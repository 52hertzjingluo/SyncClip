# 智能视频高光片段检测系统

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

一个基于深度学习的智能视频高潮片段检测和提取系统，结合视觉理解、音频分析和场景检测技术，自动识别视频中的精彩时刻。

**作者**: 黄彦喆

## ✨ 核心特性

### 多模态AI分析
- **BLIP视觉理解**: 使用Salesforce BLIP模型生成视频帧的详细描述
- **CLIP语义匹配**: 通过OpenAI CLIP模型识别视频中的戏剧性场景
- **音频能量分析**: 提取RMS能量特征识别音频高潮

### 智能场景检测
- **自动场景分割**: 基于视觉变化检测场景切换点
- **无效场景过滤**: 智能识别并过滤黑屏、白屏、模糊等无效内容
- **场景内容分类**: 自动分类动作、对话、风景等不同类型场景

### 高级功能
- **自适应权重调整**: 根据场景类型动态调整重要性权重
- **时间精确控制**: 支持指定时间段分析和输出时长限制
- **GPU加速优化**: 支持CUDA、MPS等多种硬件加速
- **详细日志记录**: 完整的处理过程记录和调试信息

## 🛠️ 环境要求

### 系统要求
- **操作系统**: Linux / macOS / Windows
- **Python版本**: 3.8+
- **硬件**: 推荐使用GPU（CUDA/MPS）以获得最佳性能

### 核心依赖
```txt
torch>=1.9.0
torchvision>=0.10.0
transformers>=4.20.0
opencv-python>=4.5.0
librosa>=0.9.0
Pillow>=8.0.0
numpy>=1.21.0
```

### 外部工具
- **FFmpeg**: 用于视频/音频处理（必需）

## 📦 安装指南

### 1. 克隆项目
```bash
git clone <repository-url>
cd video-highlight-detection
```

### 2. 创建虚拟环境
```bash
# 使用conda
conda create -n highlight python=3.9
conda activate highlight

# 或使用venv
python -m venv highlight
source highlight/bin/activate  # Linux/macOS
# highlight\Scripts\activate  # Windows
```

### 3. 安装Python依赖
```bash
# 基础安装
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers opencv-python librosa Pillow numpy

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
# 下载FFmpeg并添加到PATH环境变量
```

### 5. 验证安装
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
ffmpeg -version
```

## 🚀 快速开始

### 基础使用
```python
from highlight_detection import main

# 基本用法
main(
    video_path="input_video.mp4",
    output_path="highlights.mp4"
)
```

### 高级配置
```python
main(
    video_path="movie.mp4",
    output_path="movie_highlights.mp4",
    start_time=60,                    # 从60秒开始分析
    end_time=3600,                   # 到3600秒结束
    speed=1.2,                       # 1.2倍速输出
    max_output_duration=300,         # 最大输出5分钟
    enable_scene_analysis=True,      # 启用场景分析
    scene_analysis_output="scenes.json"  # 保存场景分析结果
)
```

## 📚 API文档

### 主函数 `main()`

```python
def main(video_path, output_path, temp_audio_path="temp_audio.wav", 
         start_time=None, end_time=None, speed=1.0, max_output_duration=None, 
         enable_scene_analysis=True, scene_analysis_output=None):
```

#### 参数说明
| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `video_path` | str | ✅ | 输入视频文件路径 |
| `output_path` | str | ✅ | 输出高潮片段视频路径 |
| `temp_audio_path` | str | ❌ | 临时音频文件路径（默认：temp_audio.wav） |
| `start_time` | float | ❌ | 分析起始时间（秒） |
| `end_time` | float | ❌ | 分析结束时间（秒） |
| `speed` | float | ❌ | 输出视频速度倍率（默认：1.0） |
| `max_output_duration` | int | ❌ | 最大输出时长（秒） |
| `enable_scene_analysis` | bool | ❌ | 是否启用场景分析（默认：True） |
| `scene_analysis_output` | str | ❌ | 场景分析结果保存路径 |

### 核心功能函数

#### 场景检测
```python
def detect_scene_changes(video_path, threshold=30.0, min_scene_len=15, 
                        filter_invalid=True, logger=None):
    """
    检测视频场景变化
    
    参数:
        threshold: 场景切换阈值（0-100）
        min_scene_len: 最小场景长度（帧数）
        filter_invalid: 是否过滤无效场景
    
    返回:
        List[Dict]: 场景信息列表
    """
```

#### 视频内容分析
```python
def analyze_video_content(key_frames, device=None, logger=None):
    """
    使用BLIP模型分析视频内容
    
    参数:
        key_frames: 关键帧列表
        device: 计算设备
    
    返回:
        Dict: 场景描述字典
    """
```

#### 高潮检测
```python
def detect_highlights(video_scores, audio_scores, total_frames, fps, 
                     segment_duration=5, video_weight=0.5, audio_weight=0.5, 
                     max_output_duration=None, logger=None):
    """
    检测高潮片段
    
    参数:
        video_scores: 视频特征分数
        audio_scores: 音频特征分数
        video_weight: 视频权重（0-1）
        audio_weight: 音频权重（0-1）
    
    返回:
        List[Tuple]: 高潮时间段列表 [(start, end), ...]
    """
```

## ⚙️ 配置选项

### 场景检测参数
```python
# 场景切换敏感度调整
detect_scene_changes(
    threshold=30.0,      # 数值越小越敏感（10-50）
    min_scene_len=15,    # 最小场景长度（帧）
    filter_invalid=True  # 过滤黑屏等无效场景
)
```

### 高潮检测权重
```python
# 调整视频/音频权重比例
detect_highlights(
    video_weight=0.7,    # 更依重视频内容
    audio_weight=0.3,    # 减少音频影响
    segment_duration=5   # 分析片段长度（秒）
)
```

### 设备优化
```python
# 系统会自动检测最佳设备，也可手动指定
device = torch.device("cuda:0")  # 使用特定GPU
device = torch.device("mps")     # Apple Silicon
device = torch.device("cpu")     # CPU模式
```

## 📝 使用示例

### 示例1: 电影高潮提取
```python
# 提取一部2小时电影的5分钟精华
main(
    video_path="movie_2h.mp4",
    output_path="movie_highlights_5min.mp4",
    max_output_duration=300,
    speed=1.0
)
```

### 示例2: 体育比赛精彩时刻
```python
# 提取足球比赛精彩片段（更注重音频）
main(
    video_path="football_match.mp4",
    output_path="football_highlights.mp4",
    start_time=600,  # 跳过前10分钟
    end_time=5400,   # 只分析90分钟
    max_output_duration=180  # 输出3分钟精华
)
```

### 示例3: 教学视频重点提取
```python
# 提取在线课程的重点内容
main(
    video_path="lecture.mp4",
    output_path="lecture_keypoints.mp4",
    speed=1.5,  # 1.5倍速播放
    enable_scene_analysis=True,
    scene_analysis_output="lecture_analysis.json"
)
```

## 📊 输出格式

### 高潮视频
- **格式**: MP4 (H.264/AAC)
- **特点**: 自动拼接的高潮片段，保持原始画质

### 场景分析JSON
```json
{
  "total_scenes": 15,
  "scenes": [
    {
      "scene_idx": 0,
      "start_time": 0.0,
      "end_time": 12.5,
      "duration": 12.5,
      "summary": "a man sitting in a car",
      "frame_descriptions": [
        {
          "timestamp": 2.1,
          "description": "a man driving a car",
          "detailed": "a photography of a man driving a car"
        }
      ]
    }
  ]
}
```

### 日志文件
- **位置**: `highlight_detection_YYYYMMDD_HHMMSS.log`
- **内容**: 详细的处理过程、性能指标、错误信息

## 🔧 故障排除

### 常见问题

#### 1. CUDA初始化错误
```
Error 804: forward compatibility was attempted on non supported HW
```
**解决方案**:
```bash
# 重启系统或重新安装NVIDIA驱动
sudo reboot

# 或强制使用CPU
export CUDA_VISIBLE_DEVICES=""
```

#### 2. FFmpeg未找到
```
FileNotFoundError: [Errno 2] No such file or directory: 'ffmpeg'
```
**解决方案**:
```bash
# 安装FFmpeg并确保在PATH中
which ffmpeg  # 验证安装
```

#### 3. 内存不足
```
RuntimeError: CUDA out of memory
```
**解决方案**:
```python
# 减少batch size或使用CPU
device = torch.device("cpu")
```

#### 4. 模型下载失败
```
ConnectionError: Failed to download model
```
**解决方案**:
```bash
# 设置代理或使用镜像源
export HF_ENDPOINT=https://hf-mirror.com
```

## 🔬 技术原理

### 多模态融合架构
```
输入视频 → 场景检测 → 关键帧提取 → BLIP描述生成
    ↓
音频提取 → RMS特征 → 能量分析 → 音频评分
    ↓
CLIP分析 → 语义匹配 → 视觉评分 → 加权融合 → 高潮检测
```

### 智能权重系统
- **动作场景**: 权重 +30%
- **对话场景**: 权重 -30%
- **风景场景**: 权重 -10%
- **无效场景**: 权重 -90%

### 自适应阈值算法
使用70%分位数作为动态阈值，确保在不同类型视频中都能提取到合适数量的高潮片段。

## 🤝 贡献指南

### 开发环境设置
1. Fork本项目
2. 创建功能分支: `git checkout -b feature-name`
3. 安装开发依赖: `pip install -r requirements-dev.txt`
4. 运行测试: `python -m pytest tests/`

### 代码规范
- 遵循PEP 8代码风格
- 添加类型注解
- 编写详细的docstring
- 确保测试覆盖率 > 80%

### 提交流程
1. 确保所有测试通过
2. 提交代码: `git commit -m "feat: add new feature"`
3. 推送分支: `git push origin feature-name`
4. 创建Pull Request

## 📄 许可证

本项目采用MIT许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 致谢

- [Salesforce BLIP](https://github.com/salesforce/BLIP) - 视觉语言理解
- [OpenAI CLIP](https://github.com/openai/CLIP) - 视觉语言表示
- [FFmpeg](https://ffmpeg.org/) - 多媒体处理
- [Librosa](https://librosa.org/) - 音频分析

## 联系方式

- **作者**: 黄彦喆
- **问题反馈**: 请使用Github Issues
- **功能建议**: 欢迎提交Pull Request

---

⭐ 如果这个项目对您有帮助，请给个Star支持！
