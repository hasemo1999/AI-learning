#!/usr/bin/env python3
"""
ステップ1: ラベル0枚で10万枚の眼科画像から特徴抽出
DINOv2のpooler_outputを使用（CLSより安定）
"""

import torch
import torch.nn as nn
from transformers import Dinov2Model, AutoImageProcessor
import numpy as np
from pathlib import Path
from PIL import Image
import pickle
import json
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
from loguru import logger
import pandas as pd

class EyeFeatureExtractor:
    """眼科画像からDINOv2特徴を抽出するクラス"""
    
    def __init__(self, model_name: str = "facebook/dinov2-base"):
        """
        Args:
            model_name: 使用するDINOv2モデル名
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"使用デバイス: {self.device}")
        
        # DINOv2モデルとプロセッサーをロード
        try:
            self.model = Dinov2Model.from_pretrained(model_name)
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()  # 推論モード
            logger.info(f"DINOv2モデル読み込み完了: {model_name}")
        except Exception as e:
            logger.error(f"モデル読み込みエラー: {e}")
            raise
    
    def extract_patient_id(self, image_path: Path) -> Optional[str]:
        """画像パスから患者IDを抽出"""
        try:
            # パス例: /path/to/patients/026/026147/image.jpg
            parts = image_path.parts
            for i, part in enumerate(parts):
                if part == "patients" and i + 2 < len(parts):
                    return parts[i + 2]  # 患者IDフォルダ名
            
            # フォルダ構造が異なる場合のフォールバック
            stem = image_path.stem
            if len(stem) >= 6 and stem.isdigit():
                return stem[:6]  # 最初の6桁を患者ID
            
            return None
        except Exception as e:
            logger.warning(f"患者ID抽出失敗: {image_path}, エラー: {e}")
            return None
    
    def load_image(self, image_path: Path) -> Optional[Image.Image]:
        """画像を安全に読み込み"""
        try:
            image = Image.open(image_path).convert('RGB')
            # 最小サイズチェック（DINOv2は224x224が標準）
            if min(image.size) < 224:
                logger.warning(f"画像サイズが小さすぎます: {image_path}, サイズ: {image.size}")
                return None
            return image
        except Exception as e:
            logger.warning(f"画像読み込みエラー: {image_path}, エラー: {e}")
            return None
    
    def extract_single_feature(self, image_path: Path) -> Optional[np.ndarray]:
        """単一画像から特徴を抽出"""
        image = self.load_image(image_path)
        if image is None:
            return None
        
        try:
            # 画像を前処理
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 特徴抽出（勾配計算なし）
            with torch.no_grad():
                outputs = self.model(**inputs)
                # pooler_outputを使用（分類タスクで安定、last_hidden_state[:,0,:]より推奨）
                feature = outputs.pooler_output.cpu().numpy()
                
            return feature.flatten()  # (1, 768) -> (768,)
            
        except Exception as e:
            logger.error(f"特徴抽出エラー: {image_path}, エラー: {e}")
            return None
    
    def extract_batch_features(
        self,
        image_paths: List[Path],
        batch_size: int = 32,
        save_path: Optional[Path] = None
    ) -> Tuple[np.ndarray, List[str], List[str]]:
        """
        バッチで特徴抽出を実行
        
        Args:
            image_paths: 画像パスのリスト
            batch_size: バッチサイズ
            save_path: 保存先パス
            
        Returns:
            features: 特徴行列 (N, 768)
            image_names: 画像名リスト
            patient_ids: 患者IDリスト
        """
        features = []
        image_names = []
        patient_ids = []
        
        logger.info(f"特徴抽出開始: {len(image_paths)}枚の画像")
        
        for i in tqdm(range(0, len(image_paths), batch_size), desc="特徴抽出中"):
            batch_paths = image_paths[i:i + batch_size]
            
            for path in batch_paths:
                feature = self.extract_single_feature(path)
                if feature is not None:
                    features.append(feature)
                    image_names.append(path.name)
                    
                    patient_id = self.extract_patient_id(path)
                    patient_ids.append(patient_id or "unknown")
        
        if not features:
            raise ValueError("有効な特徴が抽出できませんでした")
        
        feature_matrix = np.vstack(features)
        logger.info(f"特徴抽出完了: {feature_matrix.shape}")
        
        # 保存
        if save_path:
            self.save_features(feature_matrix, image_names, patient_ids, save_path)
        
        return feature_matrix, image_names, patient_ids
    
    def save_features(
        self,
        features: np.ndarray,
        image_names: List[str],
        patient_ids: List[str],
        save_path: Path
    ):
        """特徴とメタデータを保存"""
        save_path.mkdir(parents=True, exist_ok=True)
        
        # 特徴行列を保存
        feature_file = save_path / "features.npy"
        np.save(feature_file, features)
        
        # メタデータを保存
        metadata = {
            "image_names": image_names,
            "patient_ids": patient_ids,
            "feature_shape": features.shape,
            "model_name": "facebook/dinov2-base"
        }
        
        metadata_file = save_path / "metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        # CSV形式でも保存（確認用）
        df = pd.DataFrame({
            "image_name": image_names,
            "patient_id": patient_ids
        })
        csv_file = save_path / "image_metadata.csv"
        df.to_csv(csv_file, index=False, encoding='utf-8')
        
        logger.info(f"特徴データ保存完了: {save_path}")
        logger.info(f"- 特徴行列: {feature_file}")
        logger.info(f"- メタデータ: {metadata_file}")
        logger.info(f"- CSV: {csv_file}")


def main():
    """メイン実行関数"""
    # 設定
    IMAGE_DIR = Path("./すべての画像")  # 画像フォルダパス
    OUTPUT_DIR = Path("./dinov2_features")  # 出力フォルダ
    BATCH_SIZE = 16  # バッチサイズ（GPU性能に応じて調整）
    
    # 画像パスを取得
    if not IMAGE_DIR.exists():
        logger.error(f"画像フォルダが見つかりません: {IMAGE_DIR}")
        return
    
    # 対応する画像拡張子
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_paths = []
    
    for ext in image_extensions:
        image_paths.extend(list(IMAGE_DIR.rglob(f"*{ext}")))
        image_paths.extend(list(IMAGE_DIR.rglob(f"*{ext.upper()}")))
    
    if not image_paths:
        logger.error(f"画像が見つかりません: {IMAGE_DIR}")
        return
    
    logger.info(f"発見した画像: {len(image_paths)}枚")
    
    # 特徴抽出器を初期化
    extractor = EyeFeatureExtractor()
    
    try:
        # 特徴抽出を実行
        features, image_names, patient_ids = extractor.extract_batch_features(
            image_paths=image_paths,
            batch_size=BATCH_SIZE,
            save_path=OUTPUT_DIR
        )
        
        # 統計情報を表示
        logger.info("=== 抽出結果統計 ===")
        logger.info(f"総画像数: {len(image_names)}")
        logger.info(f"特徴次元: {features.shape[1]}")
        logger.info(f"ユニーク患者数: {len(set(patient_ids))}")
        
        # 患者別統計
        patient_counts = pd.Series(patient_ids).value_counts()
        logger.info(f"患者あたり平均画像数: {patient_counts.mean():.1f}")
        logger.info(f"最多画像患者: {patient_counts.max()}枚")
        
        logger.info("✅ ステップ1完了: 眼の特徴学習が完了しました")
        logger.info("次のステップ: step2_linear_probe.py で100枚ラベル学習")
        
    except Exception as e:
        logger.error(f"特徴抽出エラー: {e}")
        raise


if __name__ == "__main__":
    main()
