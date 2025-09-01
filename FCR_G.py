import gradio as gr
import cv2
import torch
import numpy as np
from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration, CLIPProcessor, CLIPModel
from PIL import Image
import subprocess
import os
import logging
from datetime import datetime
from collections import defaultdict
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
from scipy.stats import gaussian_kde
from scipy.interpolate import interp1d
import shutil

warnings.filterwarnings('ignore')

# 设置环境变量以禁用分词器并行处理
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 自定义 JSON 编码器处理 NumPy 类型
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

class YouTubeMovieMatcher:
    """YouTube解说视频与电影原片匹配剪辑工具"""
    
    def __init__(self, output_dir: str, log_level: str = "INFO"):
        # Set the main output directory to a 'proceed' folder inside the provided output_dir
        self.main_output_dir = os.path.join(output_dir, "proceed")
        self.output_dir = self.main_output_dir  # Use proceed as the main output directory
        self.logger = self.setup_logging(log_level)
        self.device = self.get_device()
        self.clip_model = None
        self.clip_processor = None
        self.blip_model = None
        self.blip_processor = None
        self.resources_to_cleanup = []
        
        # 创建缓存目录和其他子目录
        self.cache_dir = self.ensure_dir(os.path.join(self.main_output_dir, "cache"))
        self.logs_dir = self.ensure_dir(os.path.join(self.main_output_dir, "logs"))
        self.audio_dir = self.ensure_dir(os.path.join(self.main_output_dir, "audio"))
        self.transcripts_dir = self.ensure_dir(os.path.join(self.main_output_dir, "transcripts"))
        self.clips_dir = self.ensure_dir(os.path.join(self.main_output_dir, "clips"))
        self.results_dir = self.ensure_dir(os.path.join(self.main_output_dir, "results"))
        self.descriptions_dir = self.ensure_dir(os.path.join(self.main_output_dir, "descriptions"))
        
    def setup_logging(self, log_level: str) -> logging.Logger:
        """配置日志记录"""
        logger = logging.getLogger('YouTubeMovieMatcher')
        logger.setLevel(getattr(logging, log_level.upper()))
        
        if logger.handlers:
            logger.handlers.clear()
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # 文件处理器
        log_file = os.path.join(
            self.logs_dir, 
            f"visual_matcher_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(funcName)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"日志系统初始化完成，日志文件：{log_file}")
        return logger
    
    def get_device(self) -> torch.device:
        """获取最佳可用设备（GPU或CPU）"""
        try:
            if torch.cuda.is_available():
                device = torch.device("cuda")
                self.logger.info(f"使用CUDA GPU: {torch.cuda.get_device_name(0)}")
                self.logger.info(f"GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            else:
                device = torch.device("cpu")
                self.logger.info("使用CPU")
            return device
        except Exception as e:
            self.logger.error(f"获取设备失败: {str(e)}")
            return torch.device("cpu")

    def cleanup_resources(self):
        """清理所有跟踪的资源"""
        self.logger.info("清理资源...")
        for resource in self.resources_to_cleanup:
            try:
                if os.path.exists(resource):
                    if os.path.isfile(resource):
                        os.remove(resource)
                    else:
                        for root, dirs, files in os.walk(resource, topdown=False):
                            for name in files:
                                os.remove(os.path.join(root, name))
                            for name in dirs:
                                os.rmdir(os.path.join(root, name))
                        os.rmdir(resource)
                    self.logger.debug(f"已删除资源: {resource}")
            except Exception as e:
                self.logger.error(f"删除资源失败 {resource}: {str(e)}")
        self.resources_to_cleanup = []
    
    def add_resource_for_cleanup(self, resource: str):
        """添加需要清理的资源路径"""
        if resource and resource not in self.resources_to_cleanup:
            self.resources_to_cleanup.append(resource)
            self.logger.debug(f"已添加清理资源: {resource}")
    
    def ensure_dir(self, path: str):
        """确保目录存在，如果不存在则创建"""
        try:
            os.makedirs(path, exist_ok=True)
            self.logger.debug(f"确保目录存在: {path}")
            return path
        except Exception as e:
            self.logger.error(f"创建目录失败 {path}: {str(e)}")
            return None
    
    def get_file_hash(self, file_path: str) -> str:
        """计算文件的MD5哈希值"""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            self.logger.error(f"计算文件哈希失败: {str(e)}")
            return None
    
    def save_features(self, features: np.ndarray, descriptions: List[str], 
                     video_path: str, feature_type: str):
        """保存提取的特征到缓存"""
        try:
            video_hash = self.get_file_hash(video_path)
            if not video_hash:
                video_hash = os.path.basename(video_path).replace('.', '_')
            
            cache_prefix = f"{feature_type}_{video_hash}"
            features_path = os.path.join(self.cache_dir, f"{cache_prefix}_features.npy")
            np.save(features_path, features)
            descriptions_path = os.path.join(self.cache_dir, f"{cache_prefix}_descriptions.pkl")
            with open(descriptions_path, 'wb') as f:
                pickle.dump(descriptions, f)
            meta_path = os.path.join(self.cache_dir, f"{cache_prefix}_meta.json")
            meta_info = {
                'video_path': video_path,
                'feature_type': feature_type,
                'features_shape': features.shape,
                'descriptions_count': len(descriptions),
                'created_time': datetime.now().isoformat()
            }
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(meta_info, f, indent=2)
            self.logger.info(f"特征已保存到缓存: {cache_prefix}")
            return True
        except Exception as e:
            self.logger.error(f"保存特征失败: {str(e)}")
            return False
    
    def load_features(self, video_path: str, feature_type: str) -> Tuple[Optional[np.ndarray], Optional[List[str]]]:
        """从缓存加载特征"""
        try:
            video_hash = self.get_file_hash(video_path)
            if not video_hash:
                video_hash = os.path.basename(video_path).replace('.', '_')
            cache_prefix = f"{feature_type}_{video_hash}"
            features_path = os.path.join(self.cache_dir, f"{cache_prefix}_features.npy")
            descriptions_path = os.path.join(self.cache_dir, f"{cache_prefix}_descriptions.pkl")
            meta_path = os.path.join(self.cache_dir, f"{cache_prefix}_meta.json")
            if not all(os.path.exists(p) for p in [features_path, descriptions_path, meta_path]):
                return None, None
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta_info = json.load(f)
            if meta_info['video_path'] != video_path:
                self.logger.warning(f"缓存的视频路径不匹配，跳过缓存")
                return None, None
            features = np.load(features_path)
            with open(descriptions_path, 'rb') as f:
                descriptions = pickle.load(f)
            self.logger.info(f"从缓存加载特征成功: {cache_prefix}")
            self.logger.info(f"特征形状: {features.shape}, 描述数量: {len(descriptions)}")
            return features, descriptions
        except Exception as e:
            self.logger.error(f"加载特征失败: {str(e)}")
            return None, None
    
    def load_models(self):
        """加载CLIP和BLIP模型"""
        if self.clip_model is None:
            try:
                self.logger.info("加载CLIP模型用于视觉特征匹配...")
                self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
                self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
                if torch.cuda.is_available():
                    self.clip_model = self.clip_model.cuda()
                else:
                    self.clip_model = self.clip_model.cpu()
                self.logger.info("CLIP模型加载成功")
            except Exception as e:
                self.logger.error(f"CLIP模型加载失败: {str(e)}")
                raise
        if self.blip_model is None:
            try:
                self.logger.info("加载BLIP模型用于场景描述...")
                self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
                self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
                if torch.cuda.is_available():
                    self.blip_model = self.blip_model.cuda()
                else:
                    self.blip_model = self.blip_model.cpu()
                self.logger.info("BLIP模型加载成功")
            except Exception as e:
                self.logger.error(f"BLIP模型加载失败: {str(e)}")
                raise

    def detect_video_orientation(self, video_path: str) -> Dict:
        """检测视频方向和实际内容区域"""
        self.logger.info(f"检测视频方向: {video_path}")
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                self.logger.error(f"无法打开视频: {video_path}")
                return None
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            aspect_ratio = width / height
            video_info = {
                'width': width,
                'height': height,
                'aspect_ratio': aspect_ratio,
                'fps': fps,
                'total_frames': total_frames,
                'is_portrait': aspect_ratio < 1.0,
                'is_landscape': aspect_ratio > 1.2,
                'is_square': 0.9 <= aspect_ratio <= 1.1,
                'content_region': None
            }
            if video_info['is_portrait'] or video_info['is_square']:
                self.logger.info(f"检测到竖屏/方形视频 ({width}x{height})，分析内容区域...")
                content_region = self.detect_content_region(cap, width, height)
                video_info['content_region'] = content_region
                if content_region:
                    self.logger.info(f"检测到内容区域: x={content_region['x']}, y={content_region['y']}, "
                                   f"w={content_region['width']}, h={content_region['height']}")
            cap.release()
            return video_info
        except Exception as e:
            self.logger.error(f"检测视频方向失败: {str(e)}")
            if 'cap' in locals() and cap.isOpened():
                cap.release()
            return None

    def detect_content_region(self, cap, width: int, height: int, sample_frames: int = 10) -> Dict:
        """检测视频中的实际内容区域（去除黑边）"""
        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_indices = np.linspace(0, total_frames - 1, sample_frames, dtype=int)
            all_masks = []
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    continue
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
                kernel = np.ones((5, 5), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                all_masks.append(mask)
            if not all_masks:
                return None
            combined_mask = np.zeros_like(all_masks[0])
            for mask in all_masks:
                combined_mask = cv2.bitwise_or(combined_mask, mask)
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return None
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            area_ratio = (w * h) / (width * height)
            if area_ratio < 0.2 or area_ratio > 0.95:
                self.logger.warning(f"检测到的内容区域比例异常: {area_ratio:.2f}")
                if width < height:
                    crop_ratio = 0.7
                    new_h = int(height * crop_ratio)
                    y = (height - new_h) // 2
                    return {
                        'x': 0,
                        'y': y,
                        'width': width,
                        'height': new_h
                    }
            padding = 10
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(width - x, w + 2 * padding)
            h = min(height - y, h + 2 * padding)
            return {
                'x': x,
                'y': y,
                'width': w,
                'height': h
            }
        except Exception as e:
            self.logger.error(f"检测内容区域失败: {str(e)}")
            return None

    def preprocess_video(self, video_path: str, video_info: Dict) -> str:
        """预处理视频（裁剪黑边等）"""
        if not video_info.get('content_region'):
            self.logger.info("视频不需要预处理")
            return video_path
        self.logger.info("开始预处理视频...")
        dir_path = os.path.dirname(video_path)
        filename = os.path.basename(video_path)
        name, ext = os.path.splitext(filename)
        processed_path = os.path.join(dir_path, f"{name}_processed{ext}")
        if os.path.exists(processed_path):
            self.logger.info(f"使用已存在的预处理视频: {processed_path}")
            return processed_path
        region = video_info['content_region']
        crop_filter = f"crop={region['width']}:{region['height']}:{region['x']}:{region['y']}"
        cmd = [
            "ffmpeg", "-i", video_path,
            "-vf", crop_filter,
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "18",
            "-c:a", "copy",
            "-y", processed_path
        ]
        try:
            self.logger.info(f"执行裁剪: {crop_filter}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0 and os.path.exists(processed_path):
                self.logger.info(f"视频预处理成功: {processed_path}")
                self.add_resource_for_cleanup(processed_path)
                return processed_path
            else:
                self.logger.error(f"视频预处理失败: {result.stderr}")
                return video_path
        except Exception as e:
            self.logger.error(f"预处理异常: {str(e)}")
            return video_path

    def download_youtube_video(self, url: str, max_retries: int = 3) -> Optional[Dict]:
        """从本地获取视频文件"""
        self.logger.info(f"从本地获取视频文件: {url}")
        if not os.path.exists(url):
            self.logger.error(f"本地视频文件不存在: {url}")
            return None
        try:
            cap = cv2.VideoCapture(url)
            if not cap.isOpened():
                self.logger.error(f"无法打开视频文件: {url}")
                return None
            duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            safe_title = re.sub(r'[<>:"/\\|?*]', '_', os.path.basename(url).rsplit('.', 1)[0])
            video_info = {
                'title': safe_title,
                'duration': duration,
                'video_path': url,
                'url': url,
            }
            self.logger.info(f"视频加载成功: {url}")
            self.logger.info(f"标题: {video_info['title']}")
            self.logger.info(f"时长: {video_info['duration']} 秒")
            self.add_resource_for_cleanup(url)
            return video_info
        except Exception as e:
            self.logger.error(f"加载本地视频失败: {str(e)}")
            return None

    def extract_frames_with_timestamps(self, video_path: str, interval: float = 1.0) -> List[Dict]:
        """从视频中按时间间隔提取帧"""
        self.logger.info(f"从视频中提取帧，间隔: {interval}秒")
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                self.logger.error(f"无法打开视频文件: {video_path}")
                return []
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps
            self.logger.info(f"视频信息: FPS={fps:.2f}, 总帧数={total_frames}, 时长={duration:.1f}秒")
            frames_data = []
            frame_interval = max(1, int(fps * interval))
            for frame_idx in range(0, total_frames, frame_interval):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    timestamp = frame_idx / fps
                    frames_data.append({
                        'frame_idx': frame_idx,
                        'timestamp': timestamp,
                        'frame': cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    })
                    if len(frames_data) % 50 == 0:
                        self.logger.info(f"已提取 {len(frames_data)} 帧...")
            cap.release()
            self.logger.info(f"共提取 {len(frames_data)} 帧")
            return frames_data
        except Exception as e:
            self.logger.error(f"提取帧失败: {str(e)}")
            if 'cap' in locals() and cap.isOpened():
                cap.release()
            return []

    def extract_combined_features(self, frames_data: List[Dict], video_path: str = None) -> Tuple[np.ndarray, List[str]]:
        """使用CLIP和BLIP提取组合特征，支持缓存"""
        if video_path:
            features, descriptions = self.load_features(video_path, "combined")
            if features is not None and descriptions is not None:
                self.logger.info("使用缓存的特征，跳过提取")
                return features, descriptions
        self.load_models()
        if not frames_data:
            self.logger.warning("没有帧可用于特征提取")
            return np.array([]), []
        self.logger.info(f"使用CLIP+BLIP提取 {len(frames_data)} 帧的组合特征...")
        clip_features = []
        blip_descriptions = []
        batch_size = 16
        total_frames = len(frames_data)
        try:
            for i in range(0, total_frames, batch_size):
                batch_frames = frames_data[i:i + batch_size]
                images = [Image.fromarray(frame['frame']) for frame in batch_frames]
                for frame in batch_frames:
                    del frame['frame']
                clip_inputs = self.clip_processor(images=images, return_tensors="pt")
                if torch.cuda.is_available():
                    clip_inputs = clip_inputs.to("cuda")
                with torch.no_grad():
                    image_features = self.clip_model.get_image_features(**clip_inputs)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    clip_features.append(image_features.cpu().numpy())
                for img in images:
                    blip_inputs = self.blip_processor(img, return_tensors="pt")
                    if torch.cuda.is_available():
                        blip_inputs = blip_inputs.to("cuda")
                    with torch.no_grad():
                        out = self.blip_model.generate(**blip_inputs, max_length=30)
                        description = self.blip_processor.decode(out[0], skip_special_tokens=True)
                        blip_descriptions.append(description)
                if (i + batch_size) % 50 == 0 or (i + batch_size) >= total_frames:
                    self.logger.info(f"处理进度: {min(i + batch_size, total_frames)}/{total_frames} ({(min(i + batch_size, total_frames)/total_frames)*100:.1f}%)")
            if clip_features:
                clip_features = np.vstack(clip_features)
            else:
                clip_features = np.array([])
            self.logger.info(f"特征提取完成，CLIP特征维度: {clip_features.shape}")
            self.logger.info(f"BLIP描述数量: {len(blip_descriptions)}")
            if video_path and clip_features.size > 0:
                self.save_features(clip_features, blip_descriptions, video_path, "combined")
            return clip_features, blip_descriptions
        except Exception as e:
            self.logger.error(f"特征提取失败: {str(e)}")
            return np.array([]), []

    def match_frames_combined(self, youtube_features: np.ndarray, movie_features: np.ndarray,
                             youtube_descriptions: List[str], movie_descriptions: List[str],
                             youtube_timestamps: List[float], movie_timestamps: List[float],
                             similarity_threshold: float = 0.85) -> Tuple[List[Dict], List[Dict]]:
        """基于CLIP特征进行匹配，BLIP描述作为辅助信息"""
        self.logger.info(f"开始组合特征匹配，相似度阈值: {similarity_threshold}")
        if youtube_features.size == 0 or movie_features.size == 0:
            self.logger.warning("没有特征数据可用于匹配")
            return [], []
        self.logger.info("计算CLIP特征相似度矩阵...")
        clip_similarities = cosine_similarity(youtube_features, movie_features)
        primary_matches = []
        alternative_matches = []
        used_movie_indices = set()
        for yt_idx in range(len(youtube_features)):
            movie_similarities = clip_similarities[yt_idx]
            candidate_indices = [idx for idx, sim in enumerate(movie_similarities) if sim > similarity_threshold]
            if not candidate_indices:
                best_match_idx = np.argmax(movie_similarities)
                best_similarity = movie_similarities[best_match_idx]
                alt_match_idx = None
                if len(movie_similarities) > 1:
                    modified_similarities = movie_similarities.copy()
                    modified_similarities[best_match_idx] = -1
                    alt_match_idx = np.argmax(modified_similarities)
                if best_match_idx not in used_movie_indices:
                    primary_match = {
                        'youtube_idx': yt_idx,
                        'movie_idx': best_match_idx,
                        'youtube_time': youtube_timestamps[yt_idx],
                        'movie_time': movie_timestamps[best_match_idx],
                        'similarity': best_similarity,
                        'youtube_description': youtube_descriptions[yt_idx] if yt_idx < len(youtube_descriptions) else "",
                        'movie_description': movie_descriptions[best_match_idx] if best_match_idx < len(movie_descriptions) else ""
                    }
                    primary_matches.append(primary_match)
                    used_movie_indices.add(best_match_idx)
                    if alt_match_idx is not None and alt_match_idx not in used_movie_indices:
                        alt_similarity = movie_similarities[alt_match_idx]
                        alt_match = {
                            'youtube_idx': yt_idx,
                            'movie_idx': alt_match_idx,
                            'youtube_time': youtube_timestamps[yt_idx],
                            'movie_time': movie_timestamps[alt_match_idx],
                            'similarity': alt_similarity,
                            'youtube_description': youtube_descriptions[yt_idx] if yt_idx < len(youtube_descriptions) else "",
                            'movie_description': movie_descriptions[alt_match_idx] if alt_match_idx < len(movie_descriptions) else ""
                        }
                        alternative_matches.append(alt_match)
                    if len(primary_matches) % 10 == 0:
                        self.logger.info(f"已找到 {len(primary_matches)} 个主匹配和 {len(alternative_matches)} 个候选匹配")
                continue
            sorted_indices = sorted(candidate_indices, key=lambda idx: movie_similarities[idx], reverse=True)
            best_match_idx = sorted_indices[0]
            best_similarity = movie_similarities[best_match_idx]
            alt_match_idx = None
            alt_similarity = 0
            if len(sorted_indices) > 1:
                alt_match_idx = sorted_indices[1]
                alt_similarity = movie_similarities[alt_match_idx]
            if best_match_idx not in used_movie_indices:
                primary_match = {
                    'youtube_idx': yt_idx,
                    'movie_idx': best_match_idx,
                    'youtube_time': youtube_timestamps[yt_idx],
                    'movie_time': movie_timestamps[best_match_idx],
                    'similarity': best_similarity,
                    'youtube_description': youtube_descriptions[yt_idx] if yt_idx < len(youtube_descriptions) else "",
                    'movie_description': movie_descriptions[best_match_idx] if best_match_idx < len(movie_descriptions) else ""
                }
                primary_matches.append(primary_match)
                used_movie_indices.add(best_match_idx)
                if alt_match_idx and alt_match_idx not in used_movie_indices:
                    alt_match = {
                        'youtube_idx': yt_idx,
                        'movie_idx': alt_match_idx,
                        'youtube_time': youtube_timestamps[yt_idx],
                        'movie_time': movie_timestamps[alt_match_idx],
                        'similarity': alt_similarity,
                        'youtube_description': youtube_descriptions[yt_idx] if yt_idx < len(youtube_descriptions) else "",
                        'movie_description': movie_descriptions[alt_match_idx] if alt_match_idx < len(movie_descriptions) else ""
                    }
                    alternative_matches.append(alt_match)
                if len(primary_matches) % 10 == 0:
                    self.logger.info(f"已找到 {len(primary_matches)} 个主匹配和 {len(alternative_matches)} 个候选匹配")
        similarities = [m['similarity'] for m in primary_matches]
        avg_similarity = np.mean(similarities) if similarities else 0
        above_threshold = len([s for s in similarities if s >= similarity_threshold])
        below_threshold = len(primary_matches) - above_threshold
        self.logger.info(f"视觉匹配完成: 主匹配 {len(primary_matches)} 个, 候选匹配 {len(alternative_matches)} 个")
        self.logger.info(f"匹配质量: 平均相似度={avg_similarity:.3f}, 达到阈值={above_threshold}, 低于阈值={below_threshold}")
        return primary_matches, alternative_matches

    def group_matches_into_segments_flexible(self, matches: List[Dict], min_segment_duration: float = 2.0,
                                            max_time_gap: float = 5.0) -> List[Dict]:
        """将匹配的帧组合为连续片段（允许电影时间跳跃，优化覆盖率）"""
        self.logger.info("将匹配帧组合为片段（灵活模式，优化覆盖率）...")
        if not matches:
            self.logger.warning("没有匹配数据可用于生成片段")
            return []
        matches.sort(key=lambda x: x['youtube_time'])
        segments = []
        current_segment = None
        last_youtube_time = matches[-1]['youtube_time'] if matches else 0
        for match in matches:
            if current_segment is None:
                current_segment = {
                    'youtube_start': match['youtube_time'],
                    'youtube_end': match['youtube_time'],
                    'movie_times': [match['movie_time']],
                    'matches': [match],
                    'similarities': [match['similarity']],
                    'descriptions': set()
                }
            else:
                youtube_time_gap = match['youtube_time'] - current_segment['youtube_end']
                if youtube_time_gap <= max_time_gap:
                    current_segment['youtube_end'] = match['youtube_time']
                    current_segment['movie_times'].append(match['movie_time'])
                    current_segment['matches'].append(match)
                    current_segment['similarities'].append(match['similarity'])
                else:
                    segment = self._finalize_segment_precise(current_segment, min_segment_duration)
                    if segment:
                        segments.append(segment)
                    current_segment = {
                        'youtube_start': match['youtube_time'],
                        'youtube_end': match['youtube_time'],
                        'movie_times': [match['movie_time']],
                        'matches': [match],
                        'similarities': [match['similarity']],
                        'descriptions': set()
                    }
        if current_segment:
            segment = self._finalize_segment_precise(current_segment, min_segment_duration)
            if segment:
                segments.append(segment)
        if segments:
            filled_segments = []
            segments.sort(key=lambda x: x['youtube_start'])
            youtube_duration = last_youtube_time
            current_time = 0.0
            for segment in segments:
                if segment['youtube_start'] > current_time + max_time_gap:
                    gap_segment = {
                        'youtube_start': current_time,
                        'youtube_end': segment['youtube_start'],
                        'movie_start': segment['movie_start'],
                        'movie_end': segment['movie_start'],
                        'matches': [],
                        'avg_similarity': 0.0,
                        'descriptions': [],
                        'time_jumps': 0,
                        'movie_time_sequence': []
                    }
                    filled_segments.append(gap_segment)
                    self.logger.debug(f"填补间隙: YouTube [{current_time:.1f}-{segment['youtube_start']:.1f}]s")
                filled_segments.append(segment)
                current_time = segment['youtube_end']
            if current_time < youtube_duration - min_segment_duration:
                gap_segment = {
                    'youtube_start': current_time,
                    'youtube_end': youtube_duration,
                    'movie_start': segments[-1]['movie_end'] if segments else 0.0,
                    'movie_end': segments[-1]['movie_end'] + (youtube_duration - current_time) if segments else youtube_duration,
                    'matches': [],
                    'avg_similarity': 0.0,
                    'descriptions': [],
                    'time_jumps': 0,
                    'movie_time_sequence': []
                }
                filled_segments.append(gap_segment)
                self.logger.debug(f"填补末尾间隙: YouTube [{current_time:.1f}-{youtube_duration:.1f}]s")
            segments = filled_segments
        self.logger.info(f"组合完成，共生成 {len(segments)} 个片段")
        total_duration = sum(seg['youtube_end'] - seg['youtube_start'] for seg in segments)
        self.logger.info(f"总覆盖时长: {total_duration:.1f}s (目标: {last_youtube_time:.1f}s)")
        for i, seg in enumerate(segments):
            youtube_duration = seg['youtube_end'] - seg['youtube_start']
            self.logger.info(f"片段 {i+1}: YouTube [{seg['youtube_start']:.1f}-{seg['youtube_end']:.1f}]s "
                            f"-> 电影 [{seg['movie_start']:.1f}-{seg['movie_end']:.1f}]s "
                            f"(YouTube时长: {youtube_duration:.1f}s, "
                            f"相似度: {seg['avg_similarity']:.3f}, 跳跃次数: {seg.get('time_jumps', 0)})")
        return segments

    def _finalize_segment_precise(self, segment_data: Dict, min_duration: float) -> Optional[Dict]:
        """精确完成片段处理，确保时长完全对应"""
        self.logger.debug("开始精确片段处理...")
        youtube_start = segment_data['youtube_start']
        youtube_end = segment_data['youtube_end']
        youtube_duration = youtube_end - youtube_start
        if youtube_duration < min_duration:
            self.logger.debug(f"片段时长 {youtube_duration:.1f}s 小于最小要求 {min_duration}s，跳过")
            return None
        movie_times = sorted(segment_data['movie_times'])
        matches = segment_data['matches']
        self.logger.debug(f"YouTube片段: [{youtube_start:.1f}-{youtube_end:.1f}]s, 时长: {youtube_duration:.1f}s")
        self.logger.debug(f"电影时间点数量: {len(movie_times)}")
        time_jumps = 0
        jump_threshold = 10.0
        for i in range(1, len(movie_times)):
            if abs(movie_times[i] - movie_times[i-1]) > jump_threshold:
                time_jumps += 1
        try:
            youtube_times = [m['youtube_time'] for m in matches]
            match_movie_times = [m['movie_time'] for m in matches]
            if len(youtube_times) == 0:
                self.logger.warning("没有匹配数据")
                return None
            elif len(youtube_times) == 1:
                movie_start = match_movie_times[0]
                movie_end = movie_start + youtube_duration
            else:
                best_start_idx = 0
                best_continuity_score = float('inf')
                for start_idx in range(len(movie_times)):
                    continuity_score = 0
                    prev_time = movie_times[start_idx]
                    for i in range(start_idx + 1, min(start_idx + 5, len(movie_times))):
                        time_diff = abs(movie_times[i] - prev_time)
                        if time_diff > jump_threshold:
                            continuity_score += time_diff * 10
                        else:
                            continuity_score += time_diff
                        prev_time = movie_times[i]
                    if continuity_score < best_continuity_score:
                        best_continuity_score = continuity_score
                        best_start_idx = start_idx
                movie_start = movie_times[best_start_idx]
                corresponding_youtube_idx = None
                for i, m in enumerate(matches):
                    if abs(m['movie_time'] - movie_start) < 0.1:
                        corresponding_youtube_idx = i
                        break
                if corresponding_youtube_idx is not None:
                    youtube_offset = matches[corresponding_youtube_idx]['youtube_time'] - youtube_start
                    movie_start = movie_start - youtube_offset
                movie_end = movie_start + youtube_duration
                self.logger.debug(f"选择电影起始点: {movie_start:.1f}s (连续性分数: {best_continuity_score:.1f})")
            if movie_start < 0:
                self.logger.debug(f"调整负数起始时间: {movie_start:.1f}s -> 0s")
                movie_start = 0
                movie_end = youtube_duration
            avg_similarity = np.mean(segment_data['similarities']) if segment_data['similarities'] else 0.0
            descriptions = set()
            for match in segment_data['matches']:
                if match.get('youtube_description'):
                    descriptions.add(match['youtube_description'])
                if match.get('movie_description'):
                    descriptions.add(match['movie_description'])
            segment = {
                'youtube_start': youtube_start,
                'youtube_end': youtube_end,
                'movie_start': movie_start,
                'movie_end': movie_end,
                'matches': segment_data['matches'],
                'avg_similarity': avg_similarity,
                'descriptions': list(descriptions)[:5],
                'time_jumps': time_jumps,
                'movie_time_sequence': movie_times
            }
            actual_movie_duration = movie_end - movie_start
            self.logger.debug(f"片段生成: YouTube [{youtube_start:.1f}-{youtube_end:.1f}]s ({youtube_duration:.1f}s), "
                            f"电影 [{movie_start:.1f}-{movie_end:.1f}]s ({actual_movie_duration:.1f}s)")
            if abs(actual_movie_duration - youtube_duration) > 0.1:
                self.logger.warning(f"时长不匹配！YouTube: {youtube_duration:.1f}s, 电影: {actual_movie_duration:.1f}s")
            return segment
        except Exception as e:
            self.logger.error(f"精确片段处理失败: {str(e)}")
            return None

    def extract_audio_and_transcribe(self, video_path: str, output_prefix: str) -> Optional[List[Dict]]:
        """提取音频并转录为文字，保存带时间戳的文本文件"""
        self.logger.info(f"提取音频并转录: {video_path}")
        audio_path = os.path.join(self.audio_dir, f"{output_prefix}_audio.wav")
        cmd = [
            "ffmpeg", "-i", video_path,
            "-vn", "-acodec", "pcm_s16le",
            "-ar", "16000", "-ac", "1",
            "-y", audio_path
        ]
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            if not os.path.exists(audio_path):
                self.logger.error("音频提取失败")
                self.logger.debug(f"FFmpeg输出: {result.stderr}")
                return None
            self.add_resource_for_cleanup(audio_path)
            self.logger.info("音频提取成功，开始转录...")
            model = whisper.load_model("base")
            result = model.transcribe(
                audio_path,
                language="zh" if any(char >= '\u4e00' and char <= '\u9fff' for char in output_prefix) else "en",
                verbose=True,
                word_timestamps=True
            )
            segments = []
            for segment in result["segments"]:
                segments.append({
                    "id": segment["id"],
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": segment["text"].strip()
                })
            json_path = os.path.join(self.transcripts_dir, f"{output_prefix}_transcript.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(segments, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
            txt_path = os.path.join(self.transcripts_dir, f"{output_prefix}_transcript_timestamps.txt")
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(f"解说文字转录 - {output_prefix}\n")
                f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 80 + "\n\n")
                for seg in segments:
                    start_time = f"{int(seg['start']//60):02d}:{seg['start']%60:06.3f}"
                    end_time = f"{int(seg['end']//60):02d}:{seg['end']%60:06.3f}"
                    f.write(f"[{start_time} --> {end_time}]\n")
                    f.write(f"{seg['text']}\n\n")
                f.write("=" * 80 + "\n")
                f.write(f"总片段数: {len(segments)}\n")
                if segments:
                    total_duration = segments[-1]['end']
                    f.write(f"总时长: {int(total_duration//60):02d}:{total_duration%60:06.3f}\n")
            self.logger.info(f"转录完成，共 {len(segments)} 个片段")
            self.logger.info(f"JSON格式: {json_path}")
            self.logger.info(f"TXT格式: {txt_path}")
            return segments
        except Exception as e:
            self.logger.error(f"音频处理失败: {str(e)}")
            return None

    def create_synchronized_clips_precise(self, movie_path: str, youtube_path: str, segments: List[Dict],
                                         output_dir: str) -> List[str]:
        """创建精确同步的视频片段，确保时长完全一致"""
        self.logger.info("创建精确同步视频片段...")
        clips_dir = self.ensure_dir(output_dir)
        if not clips_dir:
            return []
        output_clips = []
        segments_count = len(segments)
        for i, segment in enumerate(segments):
            try:
                youtube_duration = segment['youtube_end'] - segment['youtube_start']
                if youtube_duration <= 0:
                    self.logger.error(f"片段 {i+1} YouTube时长无效: {youtube_duration:.1f}s")
                    continue
                expected_movie_duration = segment['movie_end'] - segment['movie_start']
                if abs(expected_movie_duration - youtube_duration) > 0.1:
                    self.logger.warning(f"片段 {i+1} 时长不匹配！YouTube: {youtube_duration:.1f}s, "
                                       f"计算的电影时长: {expected_movie_duration:.1f}s")
                    segment['movie_end'] = segment['movie_start'] + youtube_duration
                output_clip = os.path.join(clips_dir, f"clip_{i+1:03d}.mp4")
                if self._create_precise_clip(movie_path, segment, output_clip, youtube_duration):
                    output_clips.append(output_clip)
                    self.add_resource_for_cleanup(output_clip)
                    probe_cmd = [
                        "ffprobe", "-v", "error",
                        "-show_entries", "format=duration",
                        "-of", "default=noprint_wrappers=1:nokey=1",
                        output_clip
                    ]
                    probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
                    if probe_result.returncode == 0:
                        actual_duration = float(probe_result.stdout.strip())
                        self.logger.info(f"片段 {i+1} 创建成功 - 期望时长: {youtube_duration:.1f}s, "
                                        f"实际时长: {actual_duration:.1f}s")
                        if abs(actual_duration - youtube_duration) > 0.5:
                            self.logger.warning(f"片段 {i+1} 时长偏差较大！")
            except Exception as e:
                self.logger.error(f"处理片段 {i+1} 时出错: {str(e)}")
        return output_clips

    def _create_precise_clip(self, movie_path: str, segment: Dict, output_path: str, 
                            target_duration: float) -> bool:
        """创建精确时长的片段"""
        try:
            start_time = max(0, segment['movie_start'])
            cmd = [
                "ffmpeg",
                "-ss", str(start_time),
                "-t", str(target_duration),
                "-i", movie_path,
                "-vf", "scale=1280:720,fps=fps=30",
                "-c:v", "libx264",
                "-preset", "medium",
                "-crf", "18",
                "-c:a", "aac",
                "-b:a", "192k",
                "-ar", "48000",
                "-ac", "2",
                "-avoid_negative_ts", "make_zero",
                "-fflags", "+genpts",
                "-y", output_path
            ]
            timeout = min(300, max(60, target_duration * 3))
            self.logger.debug(f"执行FFmpeg命令提取 {target_duration:.1f}s 片段...")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            if result.returncode == 0 and os.path.exists(output_path):
                return True
            else:
                self.logger.error(f"FFmpeg错误: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            self.logger.error(f"片段处理超时: {output_path}")
            return False
        except Exception as e:
            self.logger.error(f"创建片段失败: {str(e)}")
            return False

    def merge_clips_seamless(self, clips: List[str], output_path: str) -> bool:
        """无缝合并所有片段"""
        if not clips:
            self.logger.error("没有可合并的片段")
            return False
        self.logger.info(f"开始无缝合并 {len(clips)} 个片段...")
        try:
            total_duration = 0
            for clip in clips:
                probe_cmd = [
                    "ffprobe", "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    clip
                ]
                probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
                if probe_result.returncode == 0:
                    duration = float(probe_result.stdout.strip())
                    total_duration += duration
                    self.logger.debug(f"{os.path.basename(clip)}: {duration:.2f}s")
            self.logger.info(f"预计总时长: {total_duration:.2f}s")
            list_file = os.path.join(self.output_dir, "clips_list.txt")
            with open(list_file, "w") as f:
                for clip in clips:
                    f.write(f"file '{os.path.abspath(clip)}'\n")
            cmd = [
                "ffmpeg", "-f", "concat", "-safe", "0",
                "-i", list_file,
                "-c:v", "libx264",
                "-preset", "medium",
                "-crf", "18",
                "-c:a", "aac",
                "-b:a", "192k",
                "-movflags", "+faststart",
                "-y", output_path
            ]
            timeout = min(1800, max(300, len(clips) * 30))
            self.logger.info("执行视频合并...")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            if result.returncode == 0 and os.path.exists(output_path):
                probe_cmd = [
                    "ffprobe", "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    output_path
                ]
                probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
                if probe_result.returncode == 0:
                    actual_duration = float(probe_result.stdout.strip())
                    self.logger.info(f"合并成功！总时长: {actual_duration:.2f}s "
                                    f"(预计: {total_duration:.2f}s, 差异: {abs(actual_duration-total_duration):.2f}s)")
                return True
            else:
                self.logger.error(f"视频合并失败: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            self.logger.error("视频合并超时")
            return False
        except Exception as e:
            self.logger.error(f"合并失败: {str(e)}")
            return False
        finally:
            if 'list_file' in locals() and os.path.exists(list_file):
                os.remove(list_file)

    def calculate_match_quality(self, segments: List[Dict], threshold: float) -> Dict:
        """计算匹配质量指标"""
        if not segments:
            return {
                "total_segments": 0,
                "avg_similarity": 0,
                "above_threshold": 0,
                "below_threshold": 0,
                "confidence": 0.0
            }
        similarities = [seg['avg_similarity'] for seg in segments]
        avg_similarity = np.mean(similarities)
        above_threshold = len([s for s in similarities if s >= threshold])
        below_threshold = len(similarities) - above_threshold
        confidence = max(0.0, min(100.0, (avg_similarity - 0.5) * 200))
        return {
            "total_segments": len(segments),
            "avg_similarity": avg_similarity,
            "above_threshold": above_threshold,
            "below_threshold": below_threshold,
            "confidence": confidence
        }

    def save_match_report(self, segments: List[Dict], youtube_info: Dict, 
                         youtube_transcript: List[Dict], output_path: str):
        """保存详细的匹配报告"""
        self.logger.info("生成匹配报告...")
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("YouTube解说视频与电影原片匹配报告\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"YouTube视频: {youtube_info['title']}\n")
                f.write(f"视频时长: {youtube_info['duration']} 秒\n")
                f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write("匹配片段详情\n")
                f.write("-" * 80 + "\n\n")
                for i, seg in enumerate(segments):
                    youtube_duration = seg['youtube_end'] - seg['youtube_start']
                    movie_duration = seg['movie_end'] - seg['movie_start']
                    f.write(f"片段 {i+1}:\n")
                    f.write(f"  YouTube时间: {seg['youtube_start']:.1f}s - {seg['youtube_end']:.1f}s\n")
                    f.write(f"  电影时间: {seg['movie_start']:.1f}s - {seg['movie_end']:.1f}s\n")
                    f.write(f"  YouTube时长: {youtube_duration:.1f}s\n")
                    f.write(f"  电影片段时长: {movie_duration:.1f}s (应与YouTube时长一致)\n")
                    f.write(f"  平均相似度: {seg['avg_similarity']:.3f}\n")
                    if seg.get('time_jumps', 0) > 0:
                        f.write(f"  时间跳跃: {seg['time_jumps']} 次\n")
                    f.write(f"  解说内容:\n")
                    for transcript in youtube_transcript:
                        if (transcript['start'] >= seg['youtube_start'] - 1 and 
                            transcript['start'] <= seg['youtube_end'] + 1):
                            f.write(f"    [{transcript['start']:.1f}s] {transcript['text']}\n")
                    if seg.get('descriptions'):
                        f.write(f"  场景描述:\n")
                        for desc in seg['descriptions'][:3]:
                            f.write(f"    - {desc}\n")
                    f.write("\n")
                f.write("-" * 80 + "\n")
                f.write(f"总片段数: {len(segments)}\n")
            self.logger.info(f"匹配报告已保存: {output_path}")
            return True
        except Exception as e:
            self.logger.error(f"保存报告失败: {str(e)}")
            return False

    def save_quality_report(self, quality_info: Dict, output_path: str):
        """保存匹配质量报告"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("匹配质量报告\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"总片段数: {quality_info['total_segments']}\n")
                f.write(f"平均相似度: {quality_info['avg_similarity']:.4f}\n")
                f.write(f"达到阈值的片段: {quality_info['above_threshold']} ({(quality_info['above_threshold']/quality_info['total_segments'])*100:.1f}%)\n")
                f.write(f"低于阈值的片段: {quality_info['below_threshold']} ({(quality_info['below_threshold']/quality_info['total_segments'])*100:.1f}%)\n")
                f.write(f"整体置信度: {quality_info['confidence']:.1f}%\n\n")
                f.write(f"质量评级: ")
                if quality_info['confidence'] >= 90:
                    f.write("★★★★★ 优秀 (Excellent)\n")
                elif quality_info['confidence'] >= 70:
                    f.write("★★★★☆ 良好 (Good)\n")
                elif quality_info['confidence'] >= 50:
                    f.write("★★★☆☆ 中等 (Medium)\n")
                elif quality_info['confidence'] >= 30:
                    f.write("★★☆☆☆ 一般 (Fair)\n")
                else:
                    f.write("★☆☆☆☆ 较差 (Poor)\n")
                f.write("\n建议:\n")
                if quality_info['confidence'] < 60:
                    f.write("  - 尝试降低相似度阈值\n")
                    f.write("  - 使用更短的帧间隔（更细粒度的视觉匹配）\n")
                    f.write("  - 确保电影版本与解说视频匹配\n")
            self.logger.info(f"质量报告已保存: {output_path}")
            return True
        except Exception as e:
            self.logger.error(f"保存质量报告失败: {str(e)}")
            return False

    def save_alternative_segments(self, segments: List[Dict], youtube_info: Dict, 
                                 youtube_transcript: List[Dict], output_dir: str):
        """保存候选片段信息到单独文件夹"""
        self.logger.info("保存候选片段信息...")
        alt_dir = self.ensure_dir(os.path.join(output_dir, "alternative_segments"))
        if not alt_dir:
            return False
        report_path = os.path.join(alt_dir, "alternative_segments_report.txt")
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("YouTube解说视频与电影原片候选匹配片段报告\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"YouTube视频: {youtube_info['title']}\n")
                f.write(f"视频时长: {youtube_info['duration']} 秒\n")
                f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"注意: 这些是概率次高的匹配片段，仅供参考\n\n")
                f.write("候选片段详情\n")
                f.write("-" * 80 + "\n\n")
                for i, seg in enumerate(segments):
                    f.write(f"候选片段 {i+1}:\n")
                    f.write(f"  YouTube时间: {seg['youtube_start']:.1f}s - {seg['youtube_end']:.1f}s\n")
                    f.write(f"  电影时间: {seg['movie_start']:.1f}s - {seg['movie_end']:.1f}s\n")
                    f.write(f"  持续时间: {seg['youtube_end'] - seg['youtube_start']:.1f}s\n")
                    f.write(f"  平均相似度: {seg['avg_similarity']:.3f}\n")
                    if seg.get('descriptions'):
                        f.write(f"  场景描述:\n")
                        for desc in seg['descriptions'][:3]:
                            f.write(f"    - {desc}\n")
                    f.write("\n")
                f.write("-" * 80 + "\n")
                f.write(f"总候选片段数: {len(segments)}\n")
            self.logger.info(f"候选片段报告已保存: {report_path}")
            json_path = os.path.join(alt_dir, "alternative_segments.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(segments, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
            return True
        except Exception as e:
            self.logger.error(f"保存候选片段失败: {str(e)}")
            return False

    def process_visual_matching_enhanced(self, youtube_url: str, movie_path: str,
                                        frame_interval: float = 1.0,
                                        similarity_threshold: float = 0.85,
                                        whisper_model: str = "base",
                                        allow_time_jumps: bool = True,
                                        use_cache: bool = True) -> Optional[Dict]:
        """增强的主处理函数 - 精确时间匹配版本"""
        result = {
            "status": "error",
            "message": "Unknown error",
            "output_dir": self.output_dir
        }
        try:
            if not os.path.exists(movie_path):
                msg = f"电影文件不存在: {movie_path}"
                self.logger.error(msg)
                result["message"] = msg
                return result
            if not self.ensure_dir(self.output_dir):
                msg = f"无法创建输出目录: {self.output_dir}"
                self.logger.error(msg)
                result["message"] = msg
                return result
            self.logger.info("\n===== 步骤1: 获取本地视频 =====")
            youtube_info = self.download_youtube_video(youtube_url)
            if not youtube_info:
                msg = "本地视频加载失败"
                self.logger.error(msg)
                result["message"] = msg
                return result
            youtube_path = youtube_info["video_path"]
            self.logger.info("\n===== 步骤2: 提取解说音频并转录 =====")
            youtube_transcript = self.extract_audio_and_transcribe(youtube_path, "youtube_narration")
            self.logger.info("\n===== 步骤3: 视频预处理 =====")
            self.logger.info("检测YouTube视频方向...")
            youtube_video_info = self.detect_video_orientation(youtube_path)
            if youtube_video_info:
                self.logger.info(f"YouTube视频: {youtube_video_info['width']}x{youtube_video_info['height']}, "
                                f"宽高比: {youtube_video_info['aspect_ratio']:.2f}, "
                                f"竖屏: {youtube_video_info['is_portrait']}")
                if youtube_video_info['is_portrait'] or youtube_video_info['is_square']:
                    processed_youtube = self.preprocess_video(youtube_path, youtube_video_info)
                    if processed_youtube != youtube_path:
                        youtube_path = processed_youtube
                        self.logger.info("YouTube视频已预处理（裁剪黑边）")
            self.logger.info("检测电影视频方向...")
            movie_video_info = self.detect_video_orientation(movie_path)
            if movie_video_info:
                self.logger.info(f"电影视频: {movie_video_info['width']}x{movie_video_info['height']}, "
                                f"宽高比: {movie_video_info['aspect_ratio']:.2f}")
            self.logger.info("\n===== 步骤4: 提取视频帧 =====")
            self.logger.info("提取YouTube视频帧...")
            youtube_frames = self.extract_frames_with_timestamps(youtube_path, frame_interval)
            self.logger.info("提取电影帧...")
            movie_frames = self.extract_frames_with_timestamps(movie_path, frame_interval)
            if not youtube_frames or not movie_frames:
                msg = "帧提取失败"
                self.logger.error(msg)
                result["message"] = msg
                return result
            self.logger.info("\n===== 步骤5: 提取视觉特征 (CLIP+BLIP) =====")
            self.logger.info("处理YouTube视频帧...")
            youtube_features, youtube_descriptions = self.extract_combined_features(
                youtube_frames, youtube_path if use_cache else None)
            movie_features, movie_descriptions = self.extract_combined_features(
                movie_frames, movie_path if use_cache else None)
            descriptions_path = os.path.join(self.descriptions_dir, "scene_descriptions.json")
            with open(descriptions_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "youtube_descriptions": [
                        {"time": frame['timestamp'], "description": desc}
                        for frame, desc in zip(youtube_frames, youtube_descriptions)
                    ],
                    "movie_descriptions": [
                        {"time": frame['timestamp'], "description": desc}
                        for frame, desc in zip(movie_frames, movie_descriptions)
                    ]
                }, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
            result["descriptions_path"] = descriptions_path
            self.logger.info("\n===== 步骤6: 视觉特征匹配 =====")
            youtube_timestamps = [f['timestamp'] for f in youtube_frames]
            movie_timestamps = [f['timestamp'] for f in movie_frames]
            primary_matches, alternative_matches = self.match_frames_combined(
                youtube_features, movie_features,
                youtube_descriptions, movie_descriptions,
                youtube_timestamps, movie_timestamps,
                similarity_threshold
            )
            retry_count = 0
            while retry_count < 3:
                self.logger.info("\n===== 步骤7: 组合匹配片段 =====")
                primary_segments = self.group_matches_into_segments_flexible(primary_matches)
                alternative_segments = self.group_matches_into_segments_flexible(alternative_matches)
                if not primary_segments:
                    if retry_count < 2:
                        self.logger.warning("无法生成有效片段，降低阈值并重试...")
                        similarity_threshold *= 0.8
                        self.logger.info(f"降低相似度阈值至: {similarity_threshold:.3f} (重试 #{retry_count+1})")
                        primary_matches, alternative_matches = self.match_frames_combined(
                            youtube_features, movie_features,
                            youtube_descriptions, movie_descriptions,
                            youtube_timestamps, movie_timestamps,
                            similarity_threshold
                        )
                        retry_count += 1
                        continue
                    else:
                        msg = "重试多次后仍然无法生成有效片段"
                        self.logger.error(msg)
                        result["message"] = msg
                        return result
                quality_info = self.calculate_match_quality(primary_segments, similarity_threshold)
                if quality_info["confidence"] < 50 and retry_count < 2:
                    self.logger.warning(f"匹配质量较低 (置信度={quality_info['confidence']:.1f}%)，尝试降低阈值...")
                    similarity_threshold *= 0.9
                    self.logger.info(f"降低相似度阈值至: {similarity_threshold:.3f} (重试 #{retry_count+1})")
                    primary_matches, alternative_matches = self.match_frames_combined(
                        youtube_features, movie_features,
                        youtube_descriptions, movie_descriptions,
                        youtube_timestamps, movie_timestamps,
                        similarity_threshold
                    )
                    retry_count += 1
                    continue
                break
            time_jump_stats = {
                "total_segments": len(primary_segments),
                "segments_with_jumps": sum(1 for seg in primary_segments if seg.get('time_jumps', 0) > 0),
                "total_jumps": sum(seg.get('time_jumps', 0) for seg in primary_segments)
            }
            match_results_path = os.path.join(self.results_dir, "visual_match_results.json")
            with open(match_results_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "video_info": youtube_info,
                    "youtube_video_info": youtube_video_info,
                    "movie_video_info": movie_video_info,
                    "similarity_threshold": similarity_threshold,
                    "quality_info": quality_info,
                    "time_jump_stats": time_jump_stats,
                    "total_primary_matches": len(primary_matches),
                    "total_alternative_matches": len(alternative_matches),
                    "total_primary_segments": len(primary_segments),
                    "total_alternative_segments": len(alternative_segments),
                    "primary_segments": primary_segments,
                    "alternative_segments": alternative_segments
                }, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
            result["match_results_path"] = match_results_path
            quality_report_path = os.path.join(self.results_dir, "quality_report.txt")
            self.save_quality_report(quality_info, quality_report_path)
            result["quality_report_path"] = quality_report_path
            report_path = os.path.join(self.results_dir, "match_report.txt")
            if self.save_match_report(primary_segments, youtube_info, youtube_transcript or [], report_path):
                result["report_path"] = report_path
            if alternative_segments:
                self.save_alternative_segments(alternative_segments, youtube_info, 
                                              youtube_transcript or [], self.output_dir)
            self.logger.info("\n===== 步骤8: 创建精确同步视频片段 =====")
            primary_clips_dir = self.ensure_dir(os.path.join(self.clips_dir, "primary"))
            if primary_clips_dir:
                clips = self.create_synchronized_clips_precise(movie_path, youtube_path, primary_segments, primary_clips_dir)
                if clips:
                    output_video = os.path.join(self.output_dir, "final_output.mp4")
                    success = self.merge_clips_seamless(clips, output_video)
                    if success:
                        result["status"] = "success" if quality_info["confidence"] >= 60 else "warning"
                        result["message"] = "处理成功完成"
                        if quality_info["confidence"] < 60:
                            result["message"] = "处理完成但匹配质量较低 - 请检查报告"
                        result["output_video"] = output_video
                        result["clips"] = clips
                        result["primary_segments"] = primary_segments
                        result["alternative_segments"] = alternative_segments
                        result["quality_info"] = quality_info
                        result["time_jump_stats"] = time_jump_stats
                        result["video_preprocessing"] = {
                            "youtube_preprocessed": youtube_path != youtube_info["video_path"],
                            "youtube_video_info": youtube_video_info
                        }
                        self.logger.info("\n===== 处理完成！=====")
                        self.logger.info(f"输出视频: {output_video}")
                        self.logger.info(f"匹配质量: 置信度={quality_info['confidence']:.1f}%")
                        self.logger.info(f"时间跳跃: {time_jump_stats['segments_with_jumps']}/{time_jump_stats['total_segments']} 个片段")
                        self.logger.info(f"视频预处理: {'是' if result['video_preprocessing']['youtube_preprocessed'] else '否'}")
                        self.logger.info(f"详细报告: {report_path}")
                        self.logger.info("\n注意: 所有输出片段严格按照YouTube视频时长提取，确保时间完全对应")
                        return result
            result["status"] = "partial_success"
            result["message"] = "部分处理完成，但未生成最终视频"
            return result
        except Exception as e:
            self.logger.error(f"处理失败: {str(e)}")
            self.logger.error(traceback.format_exc())
            result["message"] = f"处理失败: {str(e)}"
            return result

def gradio_interface(youtube_video, movie_video, output_dir, frame_interval, similarity_threshold, whisper_model, allow_time_jumps):
    """Gradio 界面处理函数"""
    try:
        # 验证输入
        if not youtube_video or not movie_video:
            return "错误：请上传 YouTube 解说视频和电影视频文件", None, None, None
        if not output_dir:
            return "错误：请输入有效的输出目录", None, None, None
        
        # 创建主输出目录 'proceed' 和子目录
        proceed_dir = os.path.join(output_dir, "proceed")
        os.makedirs(proceed_dir, exist_ok=True)
        
        # 将上传的视频文件复制到 proceed 目录
        youtube_path = os.path.join(proceed_dir, os.path.basename(youtube_video.name))
        movie_path = os.path.join(proceed_dir, os.path.basename(movie_video.name))
        
        shutil.copy(youtube_video.name, youtube_path)
        shutil.copy(movie_video.name, movie_path)
        
        # 创建匹配器实例，输出目录设置为 proceed
        matcher = YouTubeMovieMatcher(output_dir=output_dir, log_level="INFO")
        
        # 调用处理函数
        result = matcher.process_visual_matching_enhanced(
            youtube_url=youtube_path,
            movie_path=movie_path,
            frame_interval=frame_interval,
            similarity_threshold=similarity_threshold,
            whisper_model=whisper_model,
            allow_time_jumps=allow_time_jumps,
            use_cache=True
        )
        
        # 处理结果
        output_message = []
        output_message.append(f"处理状态: {result['status']}")
        output_message.append(f"消息: {result['message']}")
        
        if result["status"] == "success" or result["status"] == "warning":
            output_message.append(f"\n✅ 处理成功完成！")
            output_message.append(f"📹 输出视频: {result.get('output_video', '未生成')}")
            output_message.append(f"📊 匹配质量: 置信度={result['quality_info']['confidence']:.1f}%")
            output_message.append(f"📈 主片段数: {len(result.get('primary_segments', []))}")
            output_message.append(f"📉 候选片段数: {len(result.get('alternative_segments', []))}")
            
            if result.get('time_jump_stats'):
                stats = result['time_jump_stats']
                output_message.append(f"⏱️ 时间跳跃: {stats['segments_with_jumps']}/{stats['total_segments']} 个片段包含跳跃")
                output_message.append(f"🔄 总跳跃次数: {stats['total_jumps']}")
            
            if result.get('video_preprocessing', {}).get('youtube_preprocessed'):
                output_message.append(f"✂️ YouTube视频已预处理（裁剪黑边）")
            
            output_message.append(f"📄 详细报告: {result.get('report_path', '未知')}")
            output_message.append(f"📄 质量报告: {result.get('quality_report_path', '未知')}")
            output_message.append(f"\n⚠️ 重要: 所有片段时长与YouTube视频严格对应，无速度调整")
            
            return "\n".join(output_message), result.get("output_video"), result.get("report_path"), result.get("quality_report_path")
        
        elif result["status"] == "partial_success":
            output_message.append(f"\n⚡ 部分处理完成:")
            output_message.append(result["message"])
            output_message.append(f"📈 主匹配数: {len(result.get('primary_segments', []))}")
            output_message.append(f"📉 候选匹配数: {len(result.get('alternative_segments', []))}")
            output_message.append(f"📄 详细报告: {result.get('report_path', '未知')}")
            return "\n".join(output_message), None, result.get("report_path"), result.get("quality_report_path")
        
        else:
            output_message.append(f"\n❌ 处理失败:")
            output_message.append(result["message"])
            return "\n".join(output_message), None, None, None
    
    except Exception as e:
        return f"处理失败: {str(e)}", None, None, None

def main():
    """主函数 - Gradio 界面"""
    with gr.Blocks(title="YouTube 解说视频与电影匹配工具") as demo:
        gr.Markdown("# YouTube 解说视频与电影匹配工具")
        gr.Markdown("上传 YouTube 解说视频和电影原片，设置参数，自动匹配并生成同步视频片段。")
        gr.Markdown("注意：所有输出文件将保存在您指定的输出目录下的 'proceed' 文件夹中。")
        
        with gr.Row():
            with gr.Column():
                youtube_video = gr.File(label="上传 YouTube 解说视频 (本地文件)")
                movie_video = gr.File(label="上传电影原片 (本地文件)")
                output_dir = gr.Textbox(label="输出目录", placeholder="请输入输出目录路径（如 /path/to/output）")
                frame_interval = gr.Slider(minimum=0.5, maximum=5.0, value=1.2, step=0.1, label="帧提取间隔（秒）")
                similarity_threshold = gr.Slider(minimum=0.5, maximum=1.0, value=0.80, step=0.01, label="相似度阈值")
                whisper_model = gr.Dropdown(choices=["tiny", "base", "small", "medium", "large"], value="base", label="Whisper 模型大小")
                allow_time_jumps = gr.Checkbox(label="允许电影时间跳跃", value=True)
                process_button = gr.Button("开始处理")
            
            with gr.Column():
                output_text = gr.Textbox(label="处理结果", lines=20)
                output_video = gr.Video(label="最终输出视频")
                match_report = gr.File(label="匹配报告")
                quality_report = gr.File(label="质量报告")
        
        process_button.click(
            fn=gradio_interface,
            inputs=[youtube_video, movie_video, output_dir, frame_interval, similarity_threshold, whisper_model, allow_time_jumps],
            outputs=[output_text, output_video, match_report, quality_report]
        )
    
        demo.launch(server_port=7866)

if __name__ == "__main__":
    main()