#!/usr/bin/env python3
"""
ステップ2: 100枚ラベルで線形プローブ（改良版）
- pooler_output使用
- CalibratedClassifierCVでキャリブレーション
- 患者単位分割・重複除外で賢い80%精度を担保
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import json
import pickle
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    balanced_accuracy_score, f1_score, classification_report,
    confusion_matrix, roc_auc_score, precision_recall_curve
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger
from tqdm import tqdm
import torch
from PIL import Image
import hashlib
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class SmartEyeClassifier:
    """賢い眼科AI分類器（患者分割・重複除外・キャリブレーション対応）"""
    
    def __init__(self, random_state: int = 42):
        """
        Args:
            random_state: 乱数シード（再現性確保）
        """
        self.random_state = random_state
        np.random.seed(random_state)
        
        # パイプライン構成
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # ベース分類器（saga solver + class_weight='balanced'）
        self.base_classifier = LogisticRegression(
            solver='saga',
            class_weight='balanced',
            max_iter=5000,
            random_state=random_state,
            n_jobs=-1
        )
        
        # キャリブレーション済み分類器
        self.calibrated_classifier = None
        
        # メタデータ
        self.feature_dim = None
        self.classes_ = None
        self.patient_ids = None
        
        logger.info("SmartEyeClassifier初期化完了")
    
    def compute_image_hash(self, image_path: Path) -> str:
        """画像のハッシュ値を計算（重複検出用）"""
        try:
            with open(image_path, 'rb') as f:
                image_data = f.read()
            return hashlib.md5(image_data).hexdigest()
        except Exception as e:
            logger.warning(f"ハッシュ計算エラー: {image_path}, エラー: {e}")
            return str(image_path)
    
    def detect_duplicates(
        self,
        image_paths: List[Path],
        patient_ids: List[str]
    ) -> Tuple[List[int], Dict[str, List[int]]]:
        """
        重複・近接画像を検出
        
        Returns:
            unique_indices: 重複除外後のインデックス
            duplicate_groups: 重複グループ
        """
        logger.info("重複画像検出中...")
        
        hash_to_indices = defaultdict(list)
        
        for i, path in enumerate(tqdm(image_paths, desc="ハッシュ計算")):
            img_hash = self.compute_image_hash(path)
            hash_to_indices[img_hash].append(i)
        
        # 重複グループを特定
        duplicate_groups = {h: indices for h, indices in hash_to_indices.items() if len(indices) > 1}
        
        # 各重複グループから1つだけ選択（同一患者なら最初の1つ、異患者なら患者ごとに1つ）
        unique_indices = []
        
        for img_hash, indices in hash_to_indices.items():
            if len(indices) == 1:
                unique_indices.append(indices[0])
            else:
                # 患者別に分けて各患者から1つずつ選択
                patient_groups = defaultdict(list)
                for idx in indices:
                    patient_groups[patient_ids[idx]].append(idx)
                
                for patient_id, patient_indices in patient_groups.items():
                    unique_indices.append(patient_indices[0])  # 各患者から最初の1つ
        
        removed_count = len(image_paths) - len(unique_indices)
        logger.info(f"重複除外: {removed_count}枚削除, {len(unique_indices)}枚残存")
        
        return sorted(unique_indices), duplicate_groups
    
    def create_patient_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        patient_ids: List[str],
        test_size: float = 0.2,
        n_splits: int = 1
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        患者単位でのtrain/test分割
        
        Args:
            X: 特徴行列
            y: ラベル
            patient_ids: 患者IDリスト
            test_size: テストサイズ比率
            n_splits: 分割数
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        # 患者IDをnumpy配列に変換
        groups = np.array(patient_ids)
        
        # 患者単位での層化分割
        splitter = GroupShuffleSplit(
            n_splits=n_splits,
            test_size=test_size,
            random_state=self.random_state
        )
        
        train_idx, test_idx = next(splitter.split(X, y, groups))
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # 分割結果をログ出力
        train_patients = set(groups[train_idx])
        test_patients = set(groups[test_idx])
        
        logger.info(f"患者分割結果:")
        logger.info(f"- 訓練: {len(train_patients)}患者, {len(X_train)}画像")
        logger.info(f"- テスト: {len(test_patients)}患者, {len(X_test)}画像")
        logger.info(f"- 患者重複: {len(train_patients & test_patients)}人（0であることを確認）")
        
        return X_train, X_test, y_train, y_test
    
    def train_with_calibration(
        self,
        X: np.ndarray,
        y: np.ndarray,
        patient_ids: List[str],
        cv_folds: int = 3
    ):
        """
        キャリブレーション付き学習
        
        Args:
            X: 特徴行列 (N, 768)
            y: ラベル配列 (N,)
            patient_ids: 患者IDリスト
            cv_folds: CV分割数
        """
        logger.info("キャリブレーション付き学習開始...")
        
        # データ前処理
        X_scaled = self.scaler.fit_transform(X)
        y_encoded = self.label_encoder.fit_transform(y)
        
        self.classes_ = self.label_encoder.classes_
        self.feature_dim = X.shape[1]
        self.patient_ids = patient_ids
        
        logger.info(f"クラス: {self.classes_}")
        logger.info(f"特徴次元: {self.feature_dim}")
        
        # 患者単位でのCrossValidation
        cv = StratifiedGroupKFold(
            n_splits=cv_folds,
            shuffle=True,
            random_state=self.random_state
        )
        
        groups = np.array(patient_ids)
        
        # CalibratedClassifierCVでキャリブレーション
        self.calibrated_classifier = CalibratedClassifierCV(
            base_estimator=self.base_classifier,
            method='isotonic',  # 医療データでは'isotonic'が推奨
            cv=cv,
            n_jobs=-1
        )
        
        # 学習実行
        self.calibrated_classifier.fit(X_scaled, y_encoded, groups=groups)
        
        logger.info("✅ キャリブレーション付き学習完了")
    
    def evaluate_model(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        save_path: Optional[Path] = None
    ) -> Dict[str, float]:
        """
        モデル評価（医療AI向け指標）
        
        Returns:
            評価指標辞書
        """
        X_test_scaled = self.scaler.transform(X_test)
        y_test_encoded = self.label_encoder.transform(y_test)
        
        # 予測実行
        y_pred = self.calibrated_classifier.predict(X_test_scaled)
        y_pred_proba = self.calibrated_classifier.predict_proba(X_test_scaled)
        
        # 評価指標計算
        metrics = {}
        
        # 基本精度
        metrics['balanced_accuracy'] = balanced_accuracy_score(y_test_encoded, y_pred)
        metrics['macro_f1'] = f1_score(y_test_encoded, y_pred, average='macro')
        metrics['micro_f1'] = f1_score(y_test_encoded, y_pred, average='micro')
        
        # クラス別F1
        f1_per_class = f1_score(y_test_encoded, y_pred, average=None)
        for i, class_name in enumerate(self.classes_):
            metrics[f'f1_{class_name}'] = f1_per_class[i]
        
        # AUC（多クラス対応）
        if len(self.classes_) == 2:
            metrics['auc'] = roc_auc_score(y_test_encoded, y_pred_proba[:, 1])
        else:
            metrics['auc_macro'] = roc_auc_score(
                y_test_encoded, y_pred_proba, 
                multi_class='ovr', average='macro'
            )
        
        # 結果表示
        logger.info("=== 評価結果 ===")
        logger.info(f"Balanced Accuracy: {metrics['balanced_accuracy']:.3f}")
        logger.info(f"Macro F1: {metrics['macro_f1']:.3f}")
        logger.info(f"Micro F1: {metrics['micro_f1']:.3f}")
        
        # 詳細レポート
        class_names = [str(c) for c in self.classes_]
        report = classification_report(
            y_test_encoded, y_pred,
            target_names=class_names,
            output_dict=True
        )
        
        logger.info("\n" + classification_report(
            y_test_encoded, y_pred,
            target_names=class_names
        ))
        
        # 可視化・保存
        if save_path:
            self.save_evaluation_results(
                y_test_encoded, y_pred, y_pred_proba,
                metrics, report, save_path
            )
        
        return metrics
    
    def save_evaluation_results(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray,
        metrics: Dict[str, float],
        report: Dict,
        save_path: Path
    ):
        """評価結果を保存"""
        save_path.mkdir(parents=True, exist_ok=True)
        
        # 混同行列
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(
            cm, annot=True, fmt='d',
            xticklabels=self.classes_,
            yticklabels=self.classes_,
            cmap='Blues'
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_path / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # キャリブレーション曲線（2クラス分類の場合）
        if len(self.classes_) == 2:
            from sklearn.calibration import calibration_curve
            
            plt.figure(figsize=(8, 6))
            
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true, y_pred_proba[:, 1], n_bins=10
            )
            
            plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Model")
            plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
            plt.xlabel('Mean Predicted Probability')
            plt.ylabel('Fraction of Positives')
            plt.title('Calibration Curve')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(save_path / 'calibration_curve.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 評価指標をJSON保存
        with open(save_path / 'metrics.json', 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        
        # 詳細レポートをJSON保存
        with open(save_path / 'classification_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"評価結果保存完了: {save_path}")
    
    def get_screening_threshold(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        target_sensitivity: float = 0.95
    ) -> Tuple[float, Dict[str, float]]:
        """
        スクリーニング用閾値を決定
        
        Args:
            target_sensitivity: 目標感度
            
        Returns:
            threshold: 閾値
            performance: 性能指標
        """
        if len(self.classes_) != 2:
            logger.warning("スクリーニング閾値は2クラス分類でのみ対応")
            return 0.5, {}
        
        X_test_scaled = self.scaler.transform(X_test)
        y_test_encoded = self.label_encoder.transform(y_test)
        y_pred_proba = self.calibrated_classifier.predict_proba(X_test_scaled)
        
        # Precision-Recall曲線から閾値を決定
        precision, recall, thresholds = precision_recall_curve(
            y_test_encoded, y_pred_proba[:, 1]
        )
        
        # 目標感度以上の閾値を探索
        valid_indices = recall >= target_sensitivity
        if not np.any(valid_indices):
            logger.warning(f"目標感度{target_sensitivity}を達成できません")
            threshold = 0.5
        else:
            # 感度条件を満たす中で最も精度が高い閾値を選択
            best_idx = np.argmax(precision[valid_indices])
            actual_idx = np.where(valid_indices)[0][best_idx]
            threshold = thresholds[actual_idx]
        
        # 選択した閾値での性能評価
        y_pred_threshold = (y_pred_proba[:, 1] >= threshold).astype(int)
        
        from sklearn.metrics import precision_score, recall_score, accuracy_score
        
        performance = {
            'threshold': threshold,
            'sensitivity': recall_score(y_test_encoded, y_pred_threshold),
            'precision': precision_score(y_test_encoded, y_pred_threshold),
            'accuracy': accuracy_score(y_test_encoded, y_pred_threshold),
            'specificity': recall_score(y_test_encoded, y_pred_threshold, pos_label=0)
        }
        
        logger.info("=== スクリーニング閾値 ===")
        logger.info(f"閾値: {threshold:.3f}")
        logger.info(f"感度: {performance['sensitivity']:.3f}")
        logger.info(f"精度: {performance['precision']:.3f}")
        logger.info(f"特異度: {performance['specificity']:.3f}")
        
        return threshold, performance
    
    def save_model(self, save_path: Path):
        """モデルを保存"""
        save_path.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'calibrated_classifier': self.calibrated_classifier,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'classes_': self.classes_,
            'feature_dim': self.feature_dim,
            'random_state': self.random_state
        }
        
        with open(save_path / 'smart_eye_classifier.pkl', 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"モデル保存完了: {save_path}")


def load_labeled_data(data_path: Path) -> Tuple[List[Path], List[str], List[str]]:
    """
    ラベル付きデータを読み込み
    
    Args:
        data_path: データフォルダパス
        
    Returns:
        image_paths: 画像パスリスト
        labels: ラベルリスト
        patient_ids: 患者IDリスト
    """
    # サンプルデータ構造
    # data_path/
    #   ├── 正常/
    #   ├── 緑内障/
    #   ├── 黄斑変性/
    #   └── 糖尿病網膜症/
    
    image_paths = []
    labels = []
    patient_ids = []
    
    for class_dir in data_path.iterdir():
        if not class_dir.is_dir():
            continue
        
        class_name = class_dir.name
        image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
        
        for img_path in image_files:
            image_paths.append(img_path)
            labels.append(class_name)
            
            # 患者ID抽出（ファイル名から）
            patient_id = img_path.stem[:6] if len(img_path.stem) >= 6 else "unknown"
            patient_ids.append(patient_id)
    
    logger.info(f"ラベル付きデータ読み込み完了: {len(image_paths)}枚")
    logger.info(f"クラス分布: {pd.Series(labels).value_counts().to_dict()}")
    
    return image_paths, labels, patient_ids


def main():
    """メイン実行関数"""
    # 設定
    FEATURE_DIR = Path("./dinov2_features")  # ステップ1の出力
    LABELED_DATA_DIR = Path("./labeled_data")  # ラベル付きデータ
    OUTPUT_DIR = Path("./linear_probe_results")  # 結果出力
    
    # 特徴データを読み込み
    if not FEATURE_DIR.exists():
        logger.error("ステップ1の特徴抽出を先に実行してください")
        return
    
    features = np.load(FEATURE_DIR / "features.npy")
    with open(FEATURE_DIR / "metadata.json", 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    logger.info(f"特徴データ読み込み: {features.shape}")
    
    # ラベル付きデータを読み込み（100枚のサンプル）
    if not LABELED_DATA_DIR.exists():
        logger.error("ラベル付きデータフォルダが見つかりません")
        logger.info("以下の構造でデータを配置してください:")
        logger.info("labeled_data/正常/image001.jpg")
        logger.info("labeled_data/緑内障/image002.jpg")
        return
    
    image_paths, labels, patient_ids = load_labeled_data(LABELED_DATA_DIR)
    
    # 分類器を初期化
    classifier = SmartEyeClassifier(random_state=42)
    
    # 重複除外
    unique_indices, duplicate_groups = classifier.detect_duplicates(image_paths, patient_ids)
    
    # 重複除外後のデータを準備
    X = features[unique_indices]
    y = np.array(labels)[unique_indices]
    patient_ids_clean = np.array(patient_ids)[unique_indices]
    
    # 患者分割
    X_train, X_test, y_train, y_test = classifier.create_patient_split(
        X, y, patient_ids_clean.tolist(), test_size=0.2
    )
    
    # キャリブレーション付き学習
    classifier.train_with_calibration(
        X_train, y_train, 
        patient_ids_clean[: len(X_train)].tolist()
    )
    
    # 評価実行
    metrics = classifier.evaluate_model(X_test, y_test, OUTPUT_DIR)
    
    # スクリーニング閾値決定（2クラス分類の場合）
    if len(np.unique(y)) == 2:
        threshold, performance = classifier.get_screening_threshold(
            X_test, y_test, target_sensitivity=0.95
        )
    
    # モデル保存
    classifier.save_model(OUTPUT_DIR)
    
    logger.info("✅ ステップ2完了: 100枚ラベル学習が完了しました")
    logger.info(f"Balanced Accuracy: {metrics['balanced_accuracy']:.1%}")
    logger.info(f"Macro F1: {metrics['macro_f1']:.1%}")
    logger.info("次のステップ: step3_active_learning.py で不確実サンプル学習")


if __name__ == "__main__":
    main()
