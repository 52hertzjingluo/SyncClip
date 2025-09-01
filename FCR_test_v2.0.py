"""
@brief: 解说视频重组 v2.0
@authors: Huang YanZhe (Author)
@version: 2.0

@features :
    - 完整集成 ChatterboxTTS 模块。当检测到解说视频时，使用意译后的文稿，并严格按照原始时间戳生成全新的TTS配音音轨。
    - “架构：基于镜头检测的非线性重组匹配”作为核心引擎，注入到原有的视觉匹配流程中。
    - 意译功能现在会同时返回“原始文稿”和“意译文稿”两份带时间戳的数据，格式统一。
    - 所有原始功能均被增强：CSV批量处理、增强的音频分析、增强的竖屏蒙版裁切。
    - main函数中提供了清晰的全局配置区，可统一设置包括TTS在内的各项关键参数。
"""
import cv2
import torch
import numpy as np
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    CLIPProcessor, CLIPModel, Dinov2Model, AutoImageProcessor as DinoImageProcessor
)
from PIL import Image
import subprocess
import os
import logging
from datetime import datetime
import json
from sklearn.metrics.pairwise import cosine_similarity
import whisper
import re
from typing import List, Dict, Tuple, Optional
import time
import traceback
import warnings
import pickle
import hashlib
import pandas as pd
import librosa
import soundfile as sf
from scipy.spatial.distance import cosine as cosine_distance
from scenedetect import detect, ContentDetector, SceneManager

# 导入TTS相关库
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

warnings.filterwarnings('ignore')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 自定义JSON编码器
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.bool_): return bool(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# ParaphraseGenerator 类
class ParaphraseGenerator:
    def __init__(self, device):
        self.device = device
        self.model = None
        self.tokenizer = None
    def load(self):
        if self.model is None:
            model_name = "humarin/chatgpt_paraphraser_on_T5_base"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
    def paraphrase_text(self, text: str, num_return_sequences: int = 1) -> List[str]:
        if not text or not text.strip(): return [""]
        if not self.model: self.load()
        input_ids = self.tokenizer(f'paraphrase: {text}', return_tensors="pt", padding="longest", max_length=128, truncation=True).input_ids.to(self.device)
        outputs = self.model.generate(input_ids, temperature=1.0, repetition_penalty=10.0, num_return_sequences=num_return_sequences, no_repeat_ngram_size=2, num_beams=5, num_beam_groups=1, max_length=128)
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

# AudioAnalyzer 类
class AudioAnalyzer:
    def __init__(self, logger):
        self.logger = logger
    def extract_audio_features(self, audio_path: str) -> Optional[Dict]:
        try:
            y, sr = librosa.load(audio_path, sr=16000)
            if len(y) == 0: return None
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
            f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            rms = librosa.feature.rms(y=y)
            return {'mfcc_mean': np.mean(mfccs, axis=1), 'mfcc_std': np.std(mfccs, axis=1),'f0_mean': np.nanmean(f0), 'f0_std': np.nanstd(f0),'spectral_centroid_mean': np.mean(spectral_centroid), 'spectral_bandwidth_mean': np.mean(spectral_bandwidth),'rms_mean': np.mean(rms)}
        except Exception as e:
            self.logger.error(f"提取音频特征失败 from {audio_path}: {e}"); return None
    def compare_audio_similarity(self, features1: Dict, features2: Dict) -> float:
        if not features1 or not features2: return 0.0
        weights = {'mfcc': 0.5, 'f0': 0.2, 'spectral': 0.2, 'energy': 0.1}
        total_similarity = 0.0
        try:
            mfcc_vec1 = np.nan_to_num(np.concatenate([features1['mfcc_mean'], features1['mfcc_std']]))
            mfcc_vec2 = np.nan_to_num(np.concatenate([features2['mfcc_mean'], features2['mfcc_std']]))
            total_similarity += (1 - cosine_distance(mfcc_vec1, mfcc_vec2)) * weights['mfcc']
            f0_mean_dist = abs(features1['f0_mean'] - features2['f0_mean']) / max(features1['f0_mean'], features2['f0_mean'], 1)
            total_similarity += (1 - np.nan_to_num(f0_mean_dist)) * weights['f0']
            spec_vec1 = np.nan_to_num([features1['spectral_centroid_mean'], features1['spectral_bandwidth_mean']])
            spec_vec2 = np.nan_to_num([features2['spectral_centroid_mean'], features2['spectral_bandwidth_mean']])
            total_similarity += (1 - cosine_distance(spec_vec1, spec_vec2)) * weights['spectral']
            rms_dist = abs(features1['rms_mean'] - features2['rms_mean']) / max(features1['rms_mean'], features2['rms_mean'], 1e-6)
            total_similarity += (1 - np.nan_to_num(rms_dist)) * weights['energy']
            return max(0.0, float(total_similarity))
        except Exception as e:
            self.logger.error(f"比较音频相似度时出错: {e}"); return 0.0
    def is_narration(self, youtube_audio: str, movie_audio: str, threshold: float = 0.7) -> Tuple[bool, float]:
        self.logger.info("开始进行音色相似度分析...")
        youtube_features = self.extract_audio_features(youtube_audio)
        movie_features = self.extract_audio_features(movie_audio)
        if not youtube_features or not movie_features:
            self.logger.warning("无法提取任一音频特征，默认判断为解说"); return True, 0.0
        similarity = self.compare_audio_similarity(youtube_features, movie_features)
        is_narration_flag = similarity < threshold
        self.logger.info(f"音频相似度: {similarity:.3f} (阈值: {threshold}). 判断为: {'解说音频' if is_narration_flag else '原片音频'}")
        return is_narration_flag, similarity

class YouTubeMovieMatcher:
    def __init__(self, output_dir: str, log_level: str = "INFO"):
        self.output_dir = output_dir
        self.logger = self.setup_logging(log_level)
        self.device = self.get_device()
        self.models = {}; self.processors = {}
        self.resources_to_cleanup = []
        self.cache_dir = self.ensure_dir(os.path.join(self.output_dir, "cache"))
        
    def setup_logging(self, log_level: str) -> logging.Logger:
        logger = logging.getLogger(self.__class__.__name__); logger.setLevel(getattr(logging, log_level.upper()))
        if logger.handlers: logger.handlers.clear()
        console_handler = logging.StreamHandler(); console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'); console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        os.makedirs(os.path.join(self.output_dir, "logs"), exist_ok=True)
        log_file = os.path.join(self.output_dir, "logs", f"matcher_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        file_handler = logging.FileHandler(log_file, encoding='utf-8'); file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s'); file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler); logger.info(f"日志系统初始化完成，日志文件：{log_file}"); return logger
    
    def get_device(self) -> torch.device:
        try:
            if torch.cuda.is_available(): device = torch.device("cuda"); self.logger.info(f"使用CUDA GPU: {torch.cuda.get_device_name(0)}")
            elif torch.backends.mps.is_available(): device = torch.device("mps"); self.logger.info("使用 Apple Metal (MPS)")
            else: device = torch.device("cpu"); self.logger.info("使用CPU")
            return device
        except Exception as e: self.logger.error(f"获取设备失败: {e}"); return torch.device("cpu")

    def cleanup_resources(self):
        self.logger.info("清理临时资源..."); import shutil
        for resource in self.resources_to_cleanup:
            try:
                if os.path.exists(resource):
                    if os.path.isfile(resource): os.remove(resource)
                    else: shutil.rmtree(resource)
                    self.logger.debug(f"已删除资源: {resource}")
            except Exception as e: self.logger.error(f"删除资源失败 {resource}: {e}")
        self.resources_to_cleanup = []
    
    def add_resource_for_cleanup(self, resource: str):
        if resource and resource not in self.resources_to_cleanup: self.resources_to_cleanup.append(resource)

    def ensure_dir(self, path: str) -> Optional[str]:
        try: os.makedirs(path, exist_ok=True); return path
        except Exception as e: self.logger.error(f"创建目录失败 {path}: {e}"); return None
    
    def get_file_hash(self, file_path: str) -> Optional[str]:
        if not os.path.exists(file_path): return None
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""): hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e: self.logger.error(f"计算文件哈希失败: {e}"); return None
    
    def save_features(self, features: np.ndarray, video_path: str, feature_type: str):
        try:
            video_hash = self.get_file_hash(video_path)
            if not video_hash: video_hash = os.path.basename(video_path).replace('.', '_')
            features_path = os.path.join(self.cache_dir, f"{feature_type}_{video_hash}.npy")
            np.save(features_path, features); self.logger.info(f"{feature_type} 特征已保存到缓存: {features_path}"); return True
        except Exception as e: self.logger.error(f"保存{feature_type}特征失败: {e}"); return False
    
    def load_features(self, video_path: str, feature_type: str) -> Optional[np.ndarray]:
        try:
            video_hash = self.get_file_hash(video_path)
            if not video_hash: video_hash = os.path.basename(video_path).replace('.', '_')
            features_path = os.path.join(self.cache_dir, f"{feature_type}_{video_hash}.npy")
            if not os.path.exists(features_path): return None
            features = np.load(features_path); self.logger.info(f"从缓存加载{feature_type}特征成功: {features_path}"); return features
        except Exception as e: self.logger.error(f"加载{feature_type}特征失败: {e}"); return None
    
    def load_model(self, model_key: str):
        if model_key in self.models: return
        self.logger.info(f"开始加载模型: {model_key}...")
        try:
            if model_key == 'clip':
                model_name = "openai/clip-vit-large-patch14"; self.processors['clip'] = CLIPProcessor.from_pretrained(model_name); self.models['clip'] = CLIPModel.from_pretrained(model_name).to(self.device).eval()
            elif model_key == 'dinov2':
                model_name = "facebook/dinov2-large"; self.processors['dinov2'] = DinoImageProcessor.from_pretrained(model_name); self.models['dinov2'] = Dinov2Model.from_pretrained(model_name).to(self.device).eval()
            elif model_key == 'whisper':
                model_name = "base"; self.models['whisper'] = whisper.load_model(model_name, device=self.device)
            elif model_key == 'tts':
                self.logger.info("加载 ChatterboxTTS 模型 (首次加载可能需要较长时间)...")
                self.models['tts'] = ChatterboxTTS.from_pretrained(device=self.device)
            else: raise ValueError(f"未知的模型键: {model_key}")
            self.logger.info(f"模型 {model_key} 加载成功")
        except Exception as e: self.logger.error(f"加载模型 {model_key} 失败: {e}"); raise

    def detect_video_orientation(self, video_path: str) -> Dict:
        self.logger.info(f"检测视频方向: {os.path.basename(video_path)}")
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened(): self.logger.error(f"无法打开视频: {video_path}"); return None
            width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps, total_frames = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            aspect_ratio = width / height if height > 0 else 0
            video_info = {'path': video_path, 'width': width, 'height': height, 'aspect_ratio': aspect_ratio,'fps': fps, 'total_frames': total_frames,'is_portrait': aspect_ratio < 1.0, 'content_region': None}
            if video_info['is_portrait']:
                self.logger.info(f"检测到竖屏视频 ({width}x{height})，分析内容区域...")
                content_region = self.detect_content_region(cap, width, height)
                video_info['content_region'] = content_region
                if content_region: self.logger.info(f"检测到内容区域: x={content_region['x']}, y={content_region['y']}, w={content_region['width']}, h={content_region['height']}")
            cap.release()
            return video_info
        except Exception as e: self.logger.error(f"检测视频方向失败: {e}"); return None

    def detect_content_region(self, cap, width: int, height: int, sample_frames: int = 20) -> Optional[Dict]:
        self.logger.debug("使用增强的形态学算法检测内容区域...")
        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_indices = np.linspace(0, total_frames - 1, sample_frames, dtype=int)
            combined_mask = np.zeros((height, width), dtype=np.uint8)
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret: continue
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
                combined_mask = cv2.bitwise_or(combined_mask, mask)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(width*0.05), int(height*0.01)))
            closed_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
            opened_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_OPEN, kernel, iterations=2)
            contours, _ = cv2.findContours(opened_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours: self.logger.warning("在蒙版中未找到任何轮廓"); return None
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            if w < width * 0.6 or h < height * 0.4: self.logger.warning(f"检测到的内容区域过小(w:{w}, h:{h})，可能不准确，放弃裁剪。"); return None
            return {'x': x, 'y': y, 'width': w, 'height': h}
        except Exception as e: self.logger.error(f"检测内容区域时出错: {e}"); return None

    def preprocess_video(self, video_path: str, video_info: Dict, output_prefix: str) -> str:
        if not video_info or not video_info.get('content_region'):
            self.logger.info("视频无需预处理。")
            return video_path
        processed_dir = self.ensure_dir(os.path.join(self.output_dir, "temp_preprocessed"))
        processed_path = os.path.join(processed_dir, f"{output_prefix}_processed.mp4")
        if os.path.exists(processed_path):
            self.logger.info(f"使用已存在的预处理视频: {processed_path}"); self.add_resource_for_cleanup(processed_dir); return processed_path
        region = video_info['content_region']
        crop_filter = f"crop={region['width']}:{region['height']}:{region['x']}:{region['y']}"
        cmd = ["ffmpeg", "-y", "-i", video_path, "-vf", crop_filter, "-c:a", "copy", processed_path]
        self.logger.info(f"正在执行裁剪预处理: {crop_filter}...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            self.logger.info("视频预处理成功。"); self.add_resource_for_cleanup(processed_dir); return processed_path
        else:
            self.logger.error(f"视频预处理失败: {result.stderr}"); return video_path

    def download_youtube_video(self, url: str, max_retries: int = 3) -> Optional[Dict]:
        self.logger.info(f"开始下载YouTube视频: {url}")
        videos_dir = self.ensure_dir(os.path.join(self.output_dir, "videos"))
        ydl_opts = {'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best', 'outtmpl': os.path.join(videos_dir, '%(title)s.%(ext)s'),}
        for attempt in range(max_retries):
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=True)
                    video_path = ydl.prepare_filename(info)
                    self.add_resource_for_cleanup(video_path)
                    return {'title': info.get('title', '未知'), 'duration': info.get('duration', 0), 'video_path': video_path, 'url': url}
            except Exception as e:
                self.logger.error(f"下载尝试 {attempt + 1} 失败: {e}")
                if attempt < max_retries - 1: time.sleep(5)
        return None
    
    # 所有报告生成函数
    def save_match_report(self, shot_matches: List[Dict], youtube_info: Dict, movie_info: Dict, youtube_transcript: List[Dict], output_path: str):
        self.logger.info(f"生成匹配报告: {output_path}")
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"YouTube解说视频与电影原片匹配报告 (架构 v6.0)\n{'='*80}\n\n")
                f.write(f"YouTube视频: {os.path.basename(youtube_info.get('path', 'N/A'))}\n")
                f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write("匹配镜头详情\n" + "-"*80 + "\n\n")
                if not shot_matches: f.write("未找到任何高置信度的匹配镜头。\n")
                for i, match in enumerate(shot_matches):
                    yt_shot, movie_shot = match['yt_shot'], match['movie_shot']
                    yt_start_time = yt_shot['start_frame'] / youtube_info['fps']
                    yt_end_time = yt_shot['end_frame'] / youtube_info['fps']
                    movie_start_time = movie_shot['start_frame'] / movie_info['fps']
                    movie_end_time = movie_shot['end_frame'] / movie_info['fps']
                    f.write(f"匹配镜头 {i+1}:\n")
                    f.write(f"  - YouTube镜头 {yt_shot['id']:>2}: [{yt_start_time:7.2f}s - {yt_end_time:7.2f}s]\n")
                    f.write(f"  - 电影镜头   {movie_shot['id']:>2}: [{movie_start_time:7.2f}s - {movie_end_time:7.2f}s]\n")
                    f.write(f"  - 镜头相似度: {match['similarity']:.4f}\n")
                    f.write(f"  - 关联解说:\n")
                    found_transcript = False
                    for transcript in youtube_transcript:
                        if max(transcript['start'], yt_start_time) < min(transcript['end'], yt_end_time):
                            f.write(f"    [{transcript['start']:.2f}s] {transcript['text']}\n"); found_transcript = True
                    if not found_transcript: f.write("    (此镜头无对应解说文稿)\n")
                    f.write("\n")
        except Exception as e: self.logger.error(f"保存匹配报告失败: {str(e)}")

    def save_quality_report(self, quality_info: Dict, output_path: str):
        self.logger.info(f"生成质量报告: {output_path}")
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("匹配质量报告 (架构 v6.0)\n" + "=" * 80 + "\n\n")
                total_shots = quality_info.get('total_yt_shots', 1); matched_shots = quality_info.get('matched_shots', 0)
                match_ratio = (matched_shots / total_shots) * 100 if total_shots > 0 else 0
                f.write(f"总解说镜头数: {total_shots}\n")
                f.write(f"成功匹配镜头数: {matched_shots} ({match_ratio:.1f}%)\n")
                f.write(f"平均镜头相似度: {quality_info.get('avg_similarity', 0):.4f}\n")
                f.write(f"整体置信度(基于匹配率): {quality_info.get('confidence', 0):.1f}%\n\n")
                f.write(f"质量评级: ")
                confidence = quality_info.get('confidence', 0)
                if confidence >= 90: f.write("★★★★★ 优秀\n")
                elif confidence >= 75: f.write("★★★★☆ 良好\n")
                elif confidence >= 60: f.write("★★★☆☆ 中等\n")
                else: f.write("★★☆☆☆ 较差\n")
                if confidence < 75: f.write("\n建议:\n  - 尝试适当降低 `shot_similarity_threshold` 配置值。\n  - 检查电影原片版本是否与解说素材一致。\n")
        except Exception as e: self.logger.error(f"保存质量报告失败: {str(e)}")

    def save_alternative_segments(self, *args, **kwargs):
        self.logger.debug("新架构下无候选片段概念，跳过保存。")
        pass

class EnhancedYouTubeMovieMatcher(YouTubeMovieMatcher):
    def __init__(self, output_dir: str, log_level: str = "INFO"):
        super().__init__(output_dir, log_level)
        self.audio_analyzer = AudioAnalyzer(self.logger)
        self.paraphrase_generator = ParaphraseGenerator(self.device)
    
    def read_csv_file(self, csv_path: str) -> List[Dict]:
        try:
            self.logger.info(f"读取CSV文件: {csv_path}")
            df = pd.read_csv(csv_path, header=None, names=['youtube_clip', 'movie_path'])
            valid_entries = []
            for idx, row in df.iterrows():
                if not os.path.exists(row['youtube_clip']): self.logger.warning(f"文件不存在: {row['youtube_clip']}"); continue
                if not os.path.exists(row['movie_path']): self.logger.warning(f"文件不存在: {row['movie_path']}"); continue
                valid_entries.append({'youtube_clip': row['youtube_clip'], 'movie_path': row['movie_path'], 'index': idx})
            self.logger.info(f"共读取 {len(valid_entries)} 个有效条目")
            return valid_entries
        except Exception as e:
            self.logger.error(f"读取CSV文件失败: {e}"); return []
            
    def extract_audio_from_clip(self, video_path: str, output_prefix: str) -> Optional[str]:
        try:
            audio_dir = self.ensure_dir(os.path.join(self.output_dir, "temp_audio"))
            audio_path = os.path.join(audio_dir, f"{output_prefix}_audio.wav")
            cmd = ["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0 and os.path.exists(audio_path):
                self.add_resource_for_cleanup(audio_path)
                return audio_path
            else:
                self.logger.error(f"音频提取失败: {result.stderr}"); return None
        except Exception as e:
            self.logger.error(f"提取音频异常: {e}"); return None

    def transcribe_and_paraphrase(self, audio_path: str, output_prefix: str) -> Dict:
        self.logger.info("开始转录并意译音频...")
        self.load_model('whisper')
        self.paraphrase_generator.load()
        transcription = self.models['whisper'].transcribe(audio_path, language="zh", word_timestamps=True)
        original_segments, paraphrased_segments = [], []
        for segment in transcription["segments"]:
            original_text = segment["text"].strip()
            original_segments.append({"start": segment["start"], "end": segment["end"], "text": original_text})
            paraphrases = self.paraphrase_generator.paraphrase_text(original_text)
            paraphrased_segments.append({"start": segment["start"], "end": segment["end"], "text": paraphrases[0] if paraphrases else original_text})
        report_dir = self.ensure_dir(os.path.join(self.output_dir, "reports"))
        report_path = os.path.join(report_dir, f"{output_prefix}_paraphrase_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            for orig, para in zip(original_segments, paraphrased_segments):
                f.write(f"[{orig['start']:.2f}s -> {orig['end']:.2f}s]\n  原文: {orig['text']}\n  意译: {para['text']}\n\n")
        return {'original_transcript': original_segments, 'paraphrased_transcript': paraphrased_segments}
        
    def process_single_clip(self, youtube_clip: str, movie_path: str, audio_threshold: float, output_prefix: str) -> Dict:
        self.logger.info(f"\n--- 步骤 2: 音频分析与意译 ---")
        result = {'is_narration': True, 'audio_similarity': 0.0}
        try:
            youtube_audio = self.extract_audio_from_clip(youtube_clip, f"{output_prefix}_yt")
            movie_audio = self.extract_audio_from_clip(movie_path, f"{output_prefix}_movie")
            if not youtube_audio or not movie_audio: raise IOError("音频提取失败")
            is_narration, similarity = self.audio_analyzer.is_narration(youtube_audio, movie_audio, audio_threshold)
            result['is_narration'], result['audio_similarity'] = is_narration, similarity
            if is_narration:
                self.logger.info("检测为解说音频，进行转录和意译...")
                paraphrase_data = self.transcribe_and_paraphrase(youtube_audio, output_prefix)
                result.update(paraphrase_data)
            else:
                self.logger.info("检测为原片音频，跳过转录和改写。")
            return result
        except Exception as e:
            self.logger.error(f"处理音频片段失败: {e}"); return result

    # 架构的核心实现
    def stage0_generate_feature_bank(self, video_path: str) -> bool:
        self.logger.info(f"===== 阶段 0: 为 {os.path.basename(video_path)} 生成特征银行 =====")
        if self.load_features(video_path, 'clip') is not None and self.load_features(video_path, 'dino') is not None:
            self.logger.info("所有模型特征缓存已存在。"); return True
        self.load_model('clip'); self.load_model('dinov2')
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened(): raise IOError(f"无法打开视频: {video_path}")
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); all_clip, all_dino = [], []
            batch_size = 16
            for start_frame in range(0, total_frames, batch_size):
                frames = []; end_frame = min(start_frame + batch_size, total_frames)
                for i in range(start_frame, end_frame):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, i); ret, frame = cap.read()
                    if ret: frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                if not frames: continue
                with torch.no_grad():
                    clip_in = self.processors['clip'](images=frames, return_tensors="pt").to(self.device); clip_feat = self.models['clip'].get_image_features(**clip_in)
                    all_clip.append((clip_feat / clip_feat.norm(dim=-1, keepdim=True)).cpu().numpy())
                    dino_in = self.processors['dinov2'](images=frames, return_tensors="pt").to(self.device); dino_out = self.models['dinov2'](**dino_in).last_hidden_state.mean(dim=1)
                    all_dino.append((dino_out / dino_out.norm(dim=-1, keepdim=True)).cpu().numpy())
                if (start_frame // batch_size) % 10 == 0: self.logger.info(f"特征银行生成进度: {end_frame}/{total_frames}")
            cap.release()
            self.save_features(np.vstack(all_clip), video_path, 'clip'); self.save_features(np.vstack(all_dino), video_path, 'dino')
            return True
        except Exception as e: self.logger.error(f"生成特征银行失败: {e}\n{traceback.format_exc()}"); return False

    def stage1_global_coarse_localization(self, yt_clip_path: str, movie_path: str, sparse_interval_sec: int) -> Optional[List[Dict]]:
        self.logger.info("===== 阶段 1: 全局粗定位 =====")
        try:
            yt_feats = self.load_features(yt_clip_path, 'clip'); movie_feats = self.load_features(movie_path, 'clip')
            if yt_feats is None or movie_feats is None: raise FileNotFoundError("特征银行不完整")
        except Exception as e: self.logger.error(f"加载CLIP特征失败: {e}"); return None
        info_yt = self.detect_video_orientation(yt_clip_path); info_movie = self.detect_video_orientation(movie_path)
        yt_fps = info_yt['fps']; movie_fps = info_movie['fps']
        yt_sfeats = yt_feats[::int(yt_fps*sparse_interval_sec)]; movie_sfeats = movie_feats[::int(movie_fps*sparse_interval_sec)]
        if len(yt_sfeats)==0 or len(movie_sfeats)<len(yt_sfeats): self.logger.error("稀疏特征不足"); return None
        sim_mat = cosine_similarity(yt_sfeats, movie_sfeats); win_size = len(yt_sfeats)
        scores = [np.diagonal(sim_mat[:, i:i+win_size]).mean() for i in range(len(movie_sfeats) - win_size + 1)]
        best_idx = np.argmax(scores); max_sim = scores[best_idx]
        start_f = best_idx * int(movie_fps*sparse_interval_sec); end_f = (best_idx+win_size)*int(movie_fps*sparse_interval_sec)
        pad_f = int(movie_fps*60); start_f = max(0, start_f-pad_f); end_f = min(len(movie_feats), end_f+pad_f)
        zone = {'start_frame': start_f, 'end_frame': end_f, 'confidence': float(max_sim)}
        self.logger.info(f"粗定位完成。候选区域: 帧[{start_f}-{end_f}], 置信度: {max_sim:.3f}"); return [zone]

    def stage2_shot_matching(self, yt_clip_path: str, movie_path: str, candidate_zones: List[Dict], shot_sim_threshold: float) -> Optional[List[Dict]]:
        self.logger.info("===== 阶段 2: 镜头解构与无序匹配 =====")
        yt_shots = self._detect_shots(yt_clip_path); movie_shots = []
        for zone in candidate_zones: movie_shots.extend(self._detect_shots(movie_path, start_frame=zone['start_frame'], end_frame=zone['end_frame']))
        if not yt_shots or not movie_shots: self.logger.error("镜头检测失败"); return None
        yt_fps = self._generate_shot_fingerprints(yt_clip_path, yt_shots); movie_fps = self._generate_shot_fingerprints(movie_path, movie_shots)
        if not yt_fps or not movie_fps: self.logger.error("指纹生成失败"); return None
        matches = []; movie_fp_mat = np.array([fp['fingerprint'] for fp in movie_fps])
        for yt_fp in yt_fps:
            sims = cosine_similarity([yt_fp['fingerprint']], movie_fp_mat)[0]; best_idx = np.argmax(sims)
            if float(sims[best_idx]) >= shot_sim_threshold:
                matches.append({'yt_shot': yt_fp['shot_info'], 'movie_shot': movie_fps[best_idx]['shot_info'], 'similarity': float(sims[best_idx])})
        self.logger.info(f"镜头匹配完成，找到 {len(matches)}/{len(yt_shots)} 个高置信度匹配。"); return matches
        
    def _detect_shots(self, video_path: str, start_frame: Optional[int] = None, end_frame: Optional[int] = None) -> List[Dict]:
        """使用PySceneDetect检测镜头"""
        try:
            from scenedetect import FrameTimecode # 确保FrameTimecode被导入
            
            scene_manager = SceneManager()
            scene_manager.add_detector(ContentDetector(threshold=27.0)) # 阈值可调
            
            # 为了创建正确的时间码对象，需要获取视频的FPS
            video_info = self.detect_video_orientation(video_path)
            if not video_info or 'fps' not in video_info:
                self.logger.error(f"无法获取视频信息以进行镜头检测: {video_path}")
                return []
            fps = video_info['fps']

            start_timecode = None
            end_timecode = None
            
            # 根据传入的帧号创建FrameTimecode对象
            if start_frame is not None:
                start_timecode = FrameTimecode(timecode=start_frame, fps=fps)
            if end_frame is not None:
                end_timecode = FrameTimecode(timecode=end_frame, fps=fps)

            detect(
                video_path, 
                scene_manager, 
                start_time=start_timecode, 
                end_time=end_timecode
            )
            
            scene_list = scene_manager.get_scene_list()
            
            shots = [{
                'id': i,
                'start_frame': scene[0].get_frames(),
                'end_frame': scene[1].get_frames() - 1 # 结束帧-1以匹配常规用法
            } for i, scene in enumerate(scene_list)]
            
            return shots
        except Exception as e:
            self.logger.error(f"镜头检测失败 for {video_path}: {e}")
            self.logger.error(traceback.format_exc()) # 打印详细错误堆栈
            return []

    def _generate_shot_fingerprints(self, video_path: str, shot_list: List[Dict]) -> Optional[List[Dict]]:
        try:
            feats = self.load_features(video_path, 'dino');
            if feats is None: raise FileNotFoundError("DINOv2特征缓存不存在")
        except Exception as e: self.logger.error(f"{e}"); return None
        return [{'shot_info': s, 'fingerprint': np.mean(feats[s['start_frame']:s['end_frame']+1], axis=0)} for s in shot_list if len(feats[s['start_frame']:s['end_frame']+1]) > 0]

    def stage3_intra_shot_alignment(self, shot_matches: List[Dict]) -> Optional[List[Dict]]:
        self.logger.info("===== 阶段 3: 镜头内精细对齐 =====")
        final_map = []
        for match in shot_matches:
            yt_shot, movie_shot = match['yt_shot'], match['movie_shot']
            yt_indices = np.arange(yt_shot['start_frame'], yt_shot['end_frame'] + 1)
            movie_indices = np.linspace(movie_shot['start_frame'], movie_shot['end_frame'], num=len(yt_indices))
            for i, yt_frame in enumerate(yt_indices):
                final_map.append({'yt_frame': int(yt_frame), 'movie_frame': int(round(movie_indices[i]))})
        final_map.sort(key=lambda x: x['yt_frame'])
        self.logger.info(f"精细对齐完成，生成了 {len(final_map)} 个帧映射点。"); return final_map

    def stage4_render_video(self, movie_path: str, timeline_map: List[Dict], output_path: str) -> bool:
        self.logger.info("===== 阶段 4: 资产重组与渲染 =====")
        if not timeline_map: self.logger.error("时间线映射为空"); return False
        segments = []
        if timeline_map:
            current_seg = {'start': timeline_map[0]['movie_frame'], 'end': timeline_map[0]['movie_frame']}
            for i in range(1, len(timeline_map)):
                if timeline_map[i]['movie_frame'] == current_seg['end'] + 1: current_seg['end'] = timeline_map[i]['movie_frame']
                else: segments.append(current_seg); current_seg = {'start': timeline_map[i]['movie_frame'], 'end': timeline_map[i]['movie_frame']}
            segments.append(current_seg)
        movie_info = self.detect_video_orientation(movie_path); movie_fps = movie_info['fps']
        clips_dir = self.ensure_dir(os.path.join(self.output_dir, "temp_clips"))
        clip_paths = []
        for i, seg in enumerate(segments):
            start_t = seg['start']/movie_fps; duration = (seg['end']-seg['start']+1)/movie_fps
            clip_p = os.path.join(clips_dir, f"part_{i:04d}.mp4")
            if duration <= 0.04: continue
            cmd = ["ffmpeg","-y","-ss",str(start_t),"-i",movie_path,"-t",str(duration),"-an","-c:v","libx264", "-preset", "fast", "-crf", "18", clip_p]
            subprocess.run(cmd, capture_output=True)
            if os.path.exists(clip_p): clip_paths.append(clip_p)
        if not clip_paths: self.logger.error("未能生成任何临时片段"); return False
        list_file = os.path.join(clips_dir, "concat_list.txt")
        with open(list_file, "w") as f:
            for p in clip_paths: f.write(f"file '{os.path.abspath(p)}'\n")
        concat_cmd = ["ffmpeg","-y","-f","concat","-safe","0","-i",list_file,"-c","copy",output_path]
        result = subprocess.run(concat_cmd, capture_output=True)
        if result.returncode!=0: self.logger.error(f"视觉部分合并失败！\n{result.stderr.decode()}"); return False
        self.logger.info(f"视觉部分渲染成功: {output_path}"); return True
        
    def combine_video_and_audio(self, video_path: str, audio_path: str, output_path: str) -> bool:
        self.logger.info("正在合并最终视频和音频...")
        cmd = ["ffmpeg", "-y", "-i", video_path, "-i", audio_path, "-c:v", "copy", "-c:a", "aac", "-shortest", output_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0: self.logger.info(f"最终成品生成成功: {output_path}"); return True
        else: self.logger.error(f"最终合并失败: {result.stderr}"); return False

    # TTS生成与合成模块
    def stage5_generate_and_compose_tts_audio(self, transcript_data: List[Dict], total_duration: float, output_prefix: str, tts_params: Dict) -> Optional[str]:
        self.logger.info("===== 阶段 5: 生成并合成TTS音轨 =====")
        self.load_model('tts')
        tts_model = self.models['tts']
        
        tts_parts_dir = self.ensure_dir(os.path.join(self.output_dir, "temp_tts_parts"))
        self.add_resource_for_cleanup(tts_parts_dir)
        
        tts_files_info = []
        for i, segment in enumerate(transcript_data):
            text = segment['text']; start_time = segment['start']
            if not text: continue
            self.logger.info(f"TTS生成: [{start_time:.2f}s] \"{text[:30]}...\"")
            try:
                wav = tts_model.generate(text, **tts_params)
                part_path = os.path.join(tts_parts_dir, f"part_{i:04d}.wav")
                ta.save(part_path, wav, tts_model.sr)
                tts_files_info.append({'path': part_path, 'start': start_time})
            except Exception as e: self.logger.error(f"TTS片段生成失败: {e}")

        if not tts_files_info: self.logger.error("未能生成任何TTS音频片段"); return None

        self.logger.info("使用FFmpeg合成完整TTS音轨...")
        final_tts_path = os.path.join(self.output_dir, f"{output_prefix}_tts_track.wav")
        
        inputs = []; filter_complex_parts = []
        for i, info in enumerate(tts_files_info):
            inputs.extend(["-i", info['path']])
            delay_ms = int(info['start'] * 1000)
            filter_complex_parts.append(f"[{i}:a]adelay={delay_ms}|{delay_ms}[a{i}]")

        filter_complex = ";".join(filter_complex_parts)
        mix_inputs = "".join([f"[a{i}]" for i in range(len(tts_files_info))])
        filter_complex += f";{mix_inputs}amix=inputs={len(tts_files_info)}:duration=first:dropout_transition=0[out]"
        
        cmd = ["ffmpeg", "-y"] + inputs + ["-filter_complex", filter_complex, "-map", "[out]", "-t", str(total_duration), "-ar", str(tts_model.sr), final_tts_path]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0 and os.path.exists(final_tts_path):
            self.logger.info(f"完整TTS音轨合成成功: {final_tts_path}"); return final_tts_path
        else:
            self.logger.error(f"TTS音轨合成失败: {result.stderr}"); return None

    def process_visual_matching_for_clip(self, youtube_clip: str, movie_path: str, shot_similarity_threshold: float, sparse_interval_sec: int) -> Dict:
        self.logger.info("\n--- 步骤 3: 视觉匹配流程---")
        result = {'visual_matching_status': 'failed'}
        try:
            if not self.stage0_generate_feature_bank(youtube_clip): raise ValueError("阶段0失败(YT)")
            if not self.stage0_generate_feature_bank(movie_path): raise ValueError("阶段0失败(Movie)")
            
            candidate_zones = self.stage1_global_coarse_localization(youtube_clip, movie_path, sparse_interval_sec)
            if not candidate_zones: raise ValueError("阶段1: 粗定位失败")
            
            shot_matches = self.stage2_shot_matching(youtube_clip, movie_path, candidate_zones, shot_similarity_threshold)
            if not shot_matches: raise ValueError("阶段2: 镜头匹配失败或无高置信度匹配")
            
            timeline_map = self.stage3_intra_shot_alignment(shot_matches)
            if not timeline_map: raise ValueError("阶段3: 精细对齐失败")
            
            visual_output_path = os.path.join(self.output_dir, f"visual_{os.path.splitext(os.path.basename(youtube_clip))[0]}.mp4")
            if not self.stage4_render_video(movie_path, timeline_map, visual_output_path):
                raise ValueError("阶段4: 视频渲染失败")
            
            total_yt_shots = len(self._detect_shots(youtube_clip))
            quality_info = {
                'total_yt_shots': total_yt_shots, 'matched_shots': len(shot_matches),
                'avg_similarity': np.mean([m['similarity'] for m in shot_matches]) if shot_matches else 0,
                'confidence': (len(shot_matches) / total_yt_shots) * 100 if total_yt_shots > 0 else 0
            }
            result.update({'visual_output_path': visual_output_path, 'visual_matching_status': 'success', 'quality_info': quality_info, 'shot_matches': shot_matches})
            return result
        except Exception as e:
            self.logger.error(f"视觉匹配流程发生严重错误: {e}\n{traceback.format_exc()}"); result['error_message'] = str(e); return result

    # 原始的批处理总控函数
    def process_csv_batch(self, csv_path: str, audio_similarity_threshold: float, shot_similarity_threshold: float, sparse_interval_sec: int, tts_params: Dict):
        self.logger.info(f"开始CSV批量处理任务: {csv_path}")
        entries = self.read_csv_file(csv_path)
        batch_dir = self.ensure_dir(os.path.join(self.output_dir, f"batch_run_{datetime.now().strftime('%Y%m%d_%H%M')}"))
        batch_results = []
        for entry in entries:
            task_name = f"task_{entry['index']:04d}"
            task_dir = self.ensure_dir(os.path.join(batch_dir, task_name))
            original_output_dir, self.output_dir = self.output_dir, task_dir
            
            self.logger.info(f"\n{'='*25} 开始处理任务: {task_name} {'='*25}")
            result = {'index': entry['index'], 'youtube_clip': entry['youtube_clip'], 'movie_path': entry['movie_path']}
            
            try:
                # 1. 视频预处理
                self.logger.info("\n--- 步骤 1: 视频预处理 ---")
                yt_info = self.detect_video_orientation(entry['youtube_clip'])
                processed_yt_path = self.preprocess_video(entry['youtube_clip'], yt_info, task_name)
                
                # 2. 音频分析与意译
                audio_result = self.process_single_clip(processed_yt_path, entry['movie_path'], audio_similarity_threshold, task_name)
                result.update(audio_result)
                
                # 3. 视觉匹配
                visual_result = self.process_visual_matching_for_clip(processed_yt_path, entry['movie_path'], shot_similarity_threshold, sparse_interval_sec)
                result.update(visual_result)
                
                # 4. 最终合成
                if result.get('visual_matching_status') == 'success':
                    self.logger.info("\n--- 步骤 4: 最终合成 ---")
                    final_output_path = os.path.join(self.output_dir, f"FINAL_{task_name}.mp4")
                    
                    final_audio_path = None
                    if result.get('is_narration') and result.get('paraphrased_transcript'):
                        yt_info_full = self.detect_video_orientation(processed_yt_path)
                        total_duration = yt_info_full['total_frames'] / yt_info_full['fps']
                        final_audio_path = self.stage5_generate_and_compose_tts_audio(result['paraphrased_transcript'], total_duration, task_name, tts_params)
                    
                    if not final_audio_path:
                        self.logger.warning("未使用TTS音轨，将使用原始YouTube音频。")
                        final_audio_path = self.extract_audio_from_clip(processed_yt_path, f"{task_name}_final_audio")
                    
                    if not final_audio_path: raise ValueError("最终音频轨道不可用")
                    if self.combine_video_and_audio(result['visual_output_path'], final_audio_path, final_output_path):
                        result.update({'status': 'success', 'final_output_path': final_output_path})
                    else: raise ValueError("最终音视频合并失败")
                else: raise ValueError(f"视觉匹配失败: {result.get('error_message', '未知错误')}")
            except Exception as e:
                self.logger.error(f"处理任务 {task_name} 失败: {e}\n{traceback.format_exc()}"); result.update({'status': 'error', 'message': str(e)})

            batch_results.append(result)
            
            if result.get('visual_matching_status') == 'success':
                yt_full_info = self.detect_video_orientation(entry['youtube_clip']); movie_full_info = self.detect_video_orientation(entry['movie_path'])
                yt_full_info['path'] = entry['youtube_clip']
                self.save_match_report(result['shot_matches'], yt_full_info, movie_full_info, result.get('original_transcript',[]), os.path.join(task_dir, "match_report.txt"))
                self.save_quality_report(result['quality_info'], os.path.join(task_dir, "quality_report.txt"))
            with open(os.path.join(task_dir, "summary.json"), 'w', encoding='utf-8') as f: json.dump(result, f, cls=NumpyEncoder, indent=2, ensure_ascii=False)
            
            self.output_dir = original_output_dir
            self.cleanup_resources()
        
        # 保存总报告
        summary_path = os.path.join(batch_dir, "batch_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            summary_data = {'total_clips': len(entries),'processed': len(batch_results),'narration_clips': sum(1 for r in batch_results if r.get('is_narration')),'original_clips': sum(1 for r in batch_results if r.get('is_narration') is False),'success_visual_match': sum(1 for r in batch_results if r.get('visual_matching_status') == 'success'),'generated_videos': sum(1 for r in batch_results if r.get('status') == 'success'),'results': batch_results}
            json.dump(summary_data, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        report_path = os.path.join(batch_dir, "batch_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"批处理报告\n{'='*80}\n\n"); f.write(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            for key, value in summary_data.items():
                if key != 'results': f.write(f"{key}: {value}\n")
            f.write("\n详细结果\n" + "-"*40 + "\n")
            for r in batch_results: f.write(f"\n片段 {r['index']}:\n  状态: {r.get('status', 'unknown')}\n  消息: {r.get('message', 'N/A')}\n")
        self.logger.info(f"CSV批量处理任务完成。汇总报告: {summary_path}")
        return batch_results

def main():
    # --- 全局配置---
    csv_path = "/root/input/input.csv"
    output_dir = "/root/input/output"
    
    # 音频分析阈值
    audio_similarity_threshold = 0.65
    
    # 视觉匹配阈值
    shot_similarity_threshold = 0.82

    # 粗定位采样间隔（秒）
    sparse_interval_sec = 2

    # TTS生成参数
    tts_params = {
        "exaggeration": 0.5,
        "cfg_weight": 0.2
    }
    
    # --- 执行 ---
    matcher = EnhancedYouTubeMovieMatcher(output_dir=output_dir, log_level="INFO")
    
    results = matcher.process_csv_batch(
        csv_path=csv_path, 
        audio_similarity_threshold=audio_similarity_threshold,
        shot_similarity_threshold=shot_similarity_threshold,
        sparse_interval_sec=sparse_interval_sec,
        tts_params=tts_params
    )
    
    # --- 结果汇总 ---
    if results:
        success_count = sum(1 for r in results if r.get('status') == 'success')
        print(f"\n{'='*30}\n✅ 批处理完成！\n{'='*30}")
        print(f"📊 总任务数: {len(results)}")
        print(f"✨ 成功生成视频: {success_count}")
        print(f"❌ 失败/错误: {len(results) - success_count}")
        print(f"📁 详细输出请查看目录: {output_dir}")

if __name__ == "__main__":
    main()
