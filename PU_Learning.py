import os
import argparse
import itertools
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier
import time
import optuna
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
from optuna.samplers import TPESampler
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, roc_auc_score
from scipy.stats import ks_2samp


def setup_publication_quality_plotting():
    """
    为图片设置matplotlib全局参数。
    - 字体: Arial
    - 分辨率: 300 DPI
    """
    print("--- Setting up Matplotlib for publication-quality plotting ---")

    # --- 为矢量图设置字体类型，保证文本可编辑 ---
    mpl.rcParams['svg.fonttype'] = 'none'  # 保证 SVG 中的文本是文本对象，而不是路径
    mpl.rcParams['pdf.fonttype'] = 42  # 保证 PDF 中的文本是 Type 42 (TrueType)，可编辑
    mpl.rcParams['ps.fonttype'] = 42  # 保证 PS/EPS 中的文本是 Type 42 (TrueType)，可编辑
    # ------

    plt.rcParams.update({
        # --- 字体设置 (Font Settings) ---
        'font.family': 'sans-serif',  # 设置字体族为无衬线字体
        'font.sans-serif': ['Arial'],  # 首选字体为 Arial
        'font.size': 12,  # 全局默认字体大小
        'axes.labelsize': 12,  # x,y轴标签字体大小
        'axes.titlesize': 12,  # 图标题字体大小
        'xtick.labelsize': 10,  # x轴刻度标签字体大小
        'ytick.labelsize': 10,  # y轴刻度标签字体大小
        'legend.fontsize': 10,  # 图例字体大小

        # --- 分辨率与保存设置 (Resolution & Savefig) ---
        'figure.dpi': 300,  # 图像显示分辨率
        'savefig.dpi': 300,  # 图像保存分辨率
        'savefig.format': 'png',  # 默认保存格式，也可以是 'pdf', 'svg'
        'savefig.bbox': 'tight',  # 保存时自动裁剪空白边缘

        # --- 坐标轴与刻度设置 (Axes & Ticks) ---
        'axes.linewidth': 1.5,  # 坐标轴线宽
        'xtick.major.width': 1.5,  # x主刻度线宽
        'ytick.major.width': 1.5,  # y主刻度线宽
        'xtick.minor.width': 1.0,  # x次刻度线宽
        'ytick.minor.width': 1.0,  # y次刻度线宽
        'xtick.major.size': 3,  # x主刻度长度
        'ytick.major.size': 3,  # y主刻度长度
        'xtick.minor.size': 3,  # x次刻度长度
        'ytick.minor.size': 3,  # y次刻度长度
        'xtick.direction': 'in',  # x刻度线朝内
        'ytick.direction': 'in',  # y刻度线朝内

        # --- 移除图表顶部和右侧的“脊柱”（边框线），使其更简洁 ---
        # 'axes.spines.top': False,
        # 'axes.spines.right': False,

        # --- 图例设置 (Legend) ---
        'legend.frameon': False,  # 图例无边框
        'legend.loc': 'best',  # 自动选择最佳图例位置
    })


def estimate_class_prior(clf, X_pos, X_unlabeled, random_state=0, cv_splitter=None):
    """
    使用交叉验证估计类别先验和后验概率。
    这是核心的PU估计算法，现在是模型通用的。
    """
    # 合并数据和标签
    X = np.vstack([X_pos, X_unlabeled])
    s = np.hstack([np.ones(len(X_pos)), np.zeros(len(X_unlabeled))])

    # <---  为分类器设置随机状态
    if 'random_state' in clf.get_params():
        clf.set_params(random_state=random_state)

    # <--- 使用传入的 clf 对象
    probs = cross_val_predict(
        clf,
        X, s,
        cv=cv_splitter,
        method='predict_proba',
        n_jobs=-1
    )[:, 1]

    # ... rest of the function remains the same
    scores_pos = probs[:len(X_pos)]
    scores_unl = probs[len(X_pos):]
    c = np.clip(np.mean(scores_pos), 1e-6, 1 - 1e-6)
    p_unl_pos = np.clip(scores_unl / c, 0, 1)
    return c, p_unl_pos


class PUAnalyzer:
    """
    一个集成了数据处理、PU学习、Spy技术和敏感性分析的类。
    """

    MODELS_TO_TEST = {
        'RandomForest': RandomForestClassifier(n_estimators=100, n_jobs=-1),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        'SVM': SVC(probability=True),  # probability=True is essential for predict_proba
    }

    def __init__(self, args):

        self.args = args
        self.cv_splitter = StratifiedKFold(n_splits=self.args.cv_folds,
                                           shuffle=True,
                                           random_state=42)  # 固定随机种子

        self.df_pos = None
        self.df_unl = None
        self.df_spy = None
        self.feature_cols = None
        self.scaler = None
        self.p_unl_matrix = None
        self.p_spy_matrix = None
        self.c_values = None  # <--- 用于存储每次迭代的类别先验c
        self.best_hyperparams = {}

        self.best_tau = None  # 用于存储找到的最佳 tau
        self.best_eta = None  # 用于存储找到的最佳 eta (f_threshold)

        os.makedirs(self.args.output_dir, exist_ok=True)
        self._load_and_prepare_data()

    def _get_classifier_instance(self, model_name):
        """
        根据模型名称和已优化的超参数获取一个分类器实例。
        这是一个辅助函数，用于统一创建模型。
        """
        if model_name not in self.MODELS_TO_TEST:
            raise ValueError(f"Model '{model_name}' not found.")

        # 优先使用优化过的超参数
        if model_name in self.best_hyperparams:
            params = self.best_hyperparams[model_name].copy()
            print(f" (Instantiating {model_name} with optimized params for evaluation)")
        else:
            # 如果没有优化过的参数（例如跳过了预实验），则使用默认参数
            print(f" (Instantiating {model_name} with default params for evaluation)")
            return self.MODELS_TO_TEST[model_name]

        # 根据模型名称创建实例，并统一设置可复现的随机种子
        if model_name == 'RandomForest':
            # 确保 n_jobs 存在，以利用多核
            params['n_jobs'] = -1
            return RandomForestClassifier(**params)
        elif model_name == 'XGBoost':
            return XGBClassifier(**params)
        elif model_name == 'SVM':
            params['probability'] = True
            return SVC(**params)

    def _get_hyperparameter_space(self, trial, model_name):
        """根据模型名称，为optuna trial定义并返回超参数搜索空间。"""
        if model_name == 'RandomForest':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 30, log=True),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.5, 0.7, 0.9]),
                'n_jobs': -1
            }
        elif model_name == 'XGBoost':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 2, 5),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'use_label_encoder': False,
                'eval_metric': 'logloss',
                'n_jobs': -1,
                'gamma': trial.suggest_float('gamma', 0.001, 5, log=True),
                'lambda': trial.suggest_float('lambda', 0.001, 10, log=True),
                'alpha': trial.suggest_float('alpha', 0.001, 10, log=True),
            }
        elif model_name == 'SVM':
            return {
                'C': trial.suggest_float('C', 1e-2, 1e2, log=True),
                'gamma': trial.suggest_float('gamma', 1e-4, 1e1, log=True),
                'kernel': trial.suggest_categorical('kernel', ['rbf']),  # 仅使用RBF核，因为其他核可能不支持gamma
                'probability': True
            }

        else:
            raise ValueError(f"Unknown model name: {model_name}")

    def _load_and_prepare_data(self):
        """加载数据，执行归一化，并根据需要分离'Spy'样本。"""
        print("--- Loading and preparing data ---")
        df = pd.read_csv(self.args.input_csv)

        # # ---  ---
        # # 查找 X > 425000 且 标签为 0 的数据
        # initial_count = len(df)
        # indices_to_drop = df[
        #     (df['X'] > 425000) & (df['label'] == 0)
        #     ].index
        #
        # # 确定要删除的索引（最多2个）
        # # 如果找到的索引超过2个，我们只取前2个
        # if len(indices_to_drop) > 2:
        #     indices_to_drop = indices_to_drop[:2]
        #
        # # 执行删除
        # df = df.drop(indices_to_drop).reset_index(drop=True)
        # filtered_count = len(df)
        # print(
        #     f"  Filtered data: Removed {initial_count - filtered_count} rows where X > 425000 and label == 0 (max 2).")
        # # # ---  ---
        #
        self.feature_cols = [c for c in df.columns if c not in ['X', 'Y', 'label']]

        df_pos_full = df[df.label == 1].reset_index(drop=True)
        self.df_unl = df[df.label == 0].reset_index(drop=True)

        if self.args.spy_fraction > 0:
            print(f"Separating {self.args.spy_fraction:.0%} of positive samples as spies.")
            self.df_spy = df_pos_full.sample(frac=self.args.spy_fraction, random_state=42)
            self.df_pos = df_pos_full.drop(self.df_spy.index)
            print(
                f"  Original Positives: {len(df_pos_full)}, Spies: {len(self.df_spy)}, Remaining Positives: {len(self.df_pos)}")
        else:
            self.df_pos = df_pos_full
            print(f"  Positive samples: {len(self.df_pos)}, Unlabeled samples: {len(self.df_unl)}")

        # 使用所有数据进行归一化拟合，以保证一致性
        self.scaler = MinMaxScaler()
        self.scaler.fit(df[self.feature_cols].values)

    def _optimize_model_hyperparameters(self, model_name, X_pos, X_unl_with_spies):
        """
        使用Optuna对指定模型进行超参数优化。
        目标是最大化 'spy_separation' 或 'average_precision'。
        """
        if self.df_spy is None:
            raise RuntimeError("Hyperparameter optimization requires spies (spy_fraction > 0).")

        # --- 从 self.args 中获取用户选择的指标 ---
        optimization_metric = self.args.prestudy_metric

        def objective(trial):
            # 1. 定义超参数
            params = self._get_hyperparameter_space(trial, model_name)

            # ... (实例化模型的代码保持不变) ...
            temp_params = params.copy()
            if model_name in ['RandomForest', 'XGBoost', 'SVM']:
                temp_params['random_state'] = 42

            if model_name == 'RandomForest':
                model = RandomForestClassifier(**temp_params)
            elif model_name == 'XGBoost':
                model = XGBClassifier(**temp_params)
            elif model_name == 'SVM':
                model = SVC(**temp_params)

            # 3. 运行一次PU估计
            _, p_unl_with_spies = estimate_class_prior(
                model, X_pos, X_unl_with_spies,
                random_state=42,  # 使用固定的随机种子保证评估的一致性
                cv_splitter=self.cv_splitter
            )

            # 4. 计算评估分数 (目标函数值)
            p_unl = p_unl_with_spies[:len(self.df_unl)]
            p_spy = p_unl_with_spies[len(self.df_unl):]

            # 始终计算所有指标
            spy_separation_score = np.mean(p_spy) - np.mean(p_unl)

            # 为 AP 和 ROC-AUC 准备 y_true 和 y_scores
            y_true = np.hstack([np.zeros(len(p_unl)), np.ones(len(p_spy))])
            y_scores = np.hstack([p_unl, p_spy])

            ap_score = average_precision_score(y_true, y_scores)
            roc_score = roc_auc_score(y_true, y_scores)  # <-ROC-AUC 计算

            # 根据 self.args 中的选择返回目标值
            if optimization_metric == 'spy_separation':
                return spy_separation_score
            elif optimization_metric == 'average_precision':
                return ap_score
            elif optimization_metric == 'roc_auc':  # <--- 新增 elif
                return roc_score
            else:

                raise ValueError(f"Unknown prestudy_metric: {optimization_metric}")

        # Optuna study setup 保持不变
        sampler = TPESampler(seed=42)
        study = optuna.create_study(direction='maximize', sampler=sampler)
        study.optimize(objective, n_trials=self.args.n_trials_optuna)

        # --- (打印标签，使其动态化) ---
        print(f"  Best score ({optimization_metric}) for {model_name}: {study.best_value:.3f}")
        print(f"  Best params: {study.best_params}")

        return study.best_params, study.best_value

    def run_pu_estimation(self, model_name='RandomForest'):
        """
        <--- 执行多次PU估计，使用在预实验中找到的最佳超参数。
        """
        if model_name not in self.MODELS_TO_TEST:
            raise ValueError(f"Model '{model_name}' not found. Available models: {list(self.MODELS_TO_TEST.keys())}")

        # 检查是否有为此模型优化过的超参数
        selected_clf = self._get_classifier_instance(model_name)

        print(f"\n--- Running PU Estimation for {self.args.B} iterations using {model_name} ---")

        # ... (数据准备部分) ...
        X_pos_scaled = self.scaler.transform(self.df_pos[self.feature_cols].values)
        X_unl_scaled = self.scaler.transform(self.df_unl[self.feature_cols].values)
        if self.df_spy is not None:
            X_spy_scaled = self.scaler.transform(self.df_spy[self.feature_cols].values)
            X_unl_with_spies = np.vstack([X_unl_scaled, X_spy_scaled])
        else:
            X_unl_with_spies = X_unl_scaled

        p_unl_with_spies_list = []
        c_list = []

        for b in range(self.args.B):
            print(f"  [B={b + 1}/{self.args.B}] estimating prior with CV using {model_name}…", end='\r')

            # <--- 使用通用化的函数和选定的分类器
            c, p_unl_with_spies = estimate_class_prior(
                selected_clf,  # <--- 传入选择的模型实例
                X_pos_scaled, X_unl_with_spies,
                random_state=b,
                cv_splitter=self.cv_splitter
            )
            p_unl_with_spies_list.append(p_unl_with_spies)
            c_list.append(c)  # <---  将本次迭代的c值添加到列表中

        print("\nPU Estimation complete.")

        # <---  在循环结束后，对c值进行统计分析并存储 ---
        self.c_values = np.array(c_list)
        print("\n--- Class Prior 'c' Stability Analysis ---")
        print(f"  Mean: {self.c_values.mean():.3f}")
        print(f"  Standard Deviation: {self.c_values.std():.3f}")
        print(f"  Median: {np.median(self.c_values):.3f}")
        print(
            f"  95% Confidence Interval: [{np.percentile(self.c_values, 2.5):.3f}, {np.percentile(self.c_values, 97.5):.3f}]")

        # 将所有 p_unl 结果堆叠成矩阵
        p_matrix = np.stack(p_unl_with_spies_list, axis=0)

        # 分离真实unlabeled和spy的概率
        self.p_unl_matrix = p_matrix[:, :len(self.df_unl)]
        if self.df_spy is not None:
            self.p_spy_matrix = p_matrix[:, len(self.df_unl):]

        # ... run_pu_estimation 方法 ...
        # -----------------------------------------------------------------
        # 添加并保存逐样本的概率统计与不确定性 ---
        # -----------------------------------------------------------------
        print("\n--- Calculating and saving sample-wise probability statistics (Mean, Std, CI) ---")

        # 1. 沿着 axis=0 (B次迭代) 计算统计数据
        # self.p_unl_matrix 的形状是 (B, N_unlabeled)
        mean_prob = self.p_unl_matrix.mean(axis=0)
        std_prob = self.p_unl_matrix.std(axis=0)
        q_low = np.percentile(self.p_unl_matrix, 2.5, axis=0)  # 2.5% 分位数
        q_high = np.percentile(self.p_unl_matrix, 97.5, axis=0)  # 97.5% 分位数
        ci_width = q_high - q_low  # 95% 置信区间宽度

        # 2. 将这些统计数据添加为 self.df_unl 的新列
        #
        self.df_unl['pu_prob_mean'] = mean_prob
        self.df_unl['pu_prob_std'] = std_prob
        self.df_unl['pu_prob_q025'] = q_low
        self.df_unl['pu_prob_q975'] = q_high
        self.df_unl['pu_ci_width_95'] = ci_width

        # 3. 保存这个包含所有U样本及其统计数据的DataFrame
        output_path = os.path.join(
            self.args.output_dir,
            f'unlabeled_samples_with_pu_stats_B{self.args.B}.csv'
        )
        self.df_unl.to_csv(output_path, index=False)

        print(f"Sample-wise statistics (mean, std, 95% CI) saved to {output_path}")

    def plot_prior_stability(self):
        """
        绘制类别先验c在B次迭代中的分布图，以可视化其稳定性。(方案二：包含95% CI)
        """
        if self.c_values is None:
            print("\nWarning: Class prior 'c' values not calculated. Skipping stability plot.")
            return

        print("\n--- Generating Class Prior Stability Plot ---")
        plt.figure(figsize=(5, 4))

        mean_c = self.c_values.mean()
        std_c = self.c_values.std()
        # --- 计算 95% 置信区间 ---
        ci_low = np.percentile(self.c_values, 2.5)
        ci_high = np.percentile(self.c_values, 97.5)
        # ---  ---

        sns.histplot(self.c_values, kde=True, stat="density",
                     alpha=0.7, color="steelblue", zorder=1)

        plt.axvline(mean_c, color='red', linestyle='--', linewidth=2,
                    label=f'Mean = {mean_c:.3f}', zorder=2)

        # --- 为图例添加虚拟元素 ---
        plt.plot([], [], ' ', label=f'Std Dev = {std_c:.3f}')
        # --- 添加CI到图例 ---
        plt.plot([], [], ' ', label=f'95% CI = [{ci_low:.3f}, {ci_high:.3f}]')
        # --- 结束 ---

        # ... (函数的其余部分保持不变) ...
        plt.xlabel('Estimated Class Prior c = P(y=1)')
        plt.ylabel('Density')
        plt.legend(frameon=True)

        output_base = os.path.join(self.args.output_dir, 'fig_prior_c_stability')
        plt.savefig(f'{output_base}.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{output_base}.svg', bbox_inches='tight')
        plt.savefig(f'{output_base}.pdf', bbox_inches='tight')
        plt.close()
        print(f"Class prior stability plot saved to {output_base}.[png/svg/pdf]")

    def plot_spy_calibration(self):
        """如果使用了spy技术，绘制spy和真实unlabeled样本的概率分布图。"""
        if self.p_spy_matrix is None:
            print("\nSkipping spy calibration plot (no spies were used).")
            return

        print("\n--- Generating Spy Calibration Plot ---")
        mean_p_unl = self.p_unl_matrix.mean(axis=0)
        mean_p_spy = self.p_spy_matrix.mean(axis=0)

        plt.figure(figsize=(5, 4))
        sns.kdeplot(mean_p_unl, label=f'Real Unlabeled (Mean Prob: {np.mean(mean_p_unl):.3f})', fill=True)
        sns.kdeplot(mean_p_spy, label=f'Spies (Mean Prob: {np.mean(mean_p_spy):.3f})', fill=True, color='red',
                    alpha=0.6)

        plt.xlabel('Posterior Probability P(y=1|x)')
        plt.ylabel('Density')
        # plt.title('Spy Calibration: Probability Distribution')
        plt.legend()
        # plt.grid(axis='y', linestyle='--', alpha=0.7)

        output_base = os.path.join(self.args.output_dir, 'fig_spy_calibration')
        plt.savefig(f'{output_base}.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{output_base}.svg', bbox_inches='tight')
        plt.savefig(f'{output_base}.pdf', bbox_inches='tight')
        plt.close()
        print(f"Spy calibration plot saved to {output_base}.[png/svg/pdf]")

    def plot_spy_roc_pr(self):
        """
        绘制 Spy vs Real Unlabeled 的 ROC 和 PR 曲线。
        这用于定量评估PU模型在B次迭代后对 "已知正样本" 和 "未知样本" 的排序能力。
        """
        if self.p_spy_matrix is None:
            print("\nSkipping spy ROC/PR plot (no spies were used).")
            return None, None

        print("\n--- Generating Spy vs. Unlabeled ROC & PR Plots ---")

        # 1. 获取 B 次迭代的平均概率
        mean_p_unl = self.p_unl_matrix.mean(axis=0)
        mean_p_spy = self.p_spy_matrix.mean(axis=0)

        # 2. 构建真实标签和预测分数
        # 在这个诊断任务中，Spy是"正类"(1)，Unlabeled是"负类"(0)
        y_true = np.hstack([np.zeros(len(mean_p_unl)), np.ones(len(mean_p_spy))])
        y_scores = np.hstack([mean_p_unl, mean_p_spy])

        # 3. 计算ROC曲线和AUC
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        # 4. 计算PR曲线和AP (Average Precision)
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        avg_precision = average_precision_score(y_true, y_scores)

        # 5. 绘图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

        # --- 子图 1: ROC 曲线 ---
        ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC Curve (Spy vs. Unlabeled)')
        ax1.legend(loc="lower right")

        # --- 子图 2: PR 曲线 ---
        ax2.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP = {avg_precision:.3f})')
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curve (Spy vs. Unlabeled)')
        ax2.legend(loc="lower left")

        plt.tight_layout()

        # 6. 保存图像
        output_base = os.path.join(self.args.output_dir, 'fig_spy_roc_pr')
        plt.savefig(f'{output_base}.png', dpi=300)
        plt.savefig(f'{output_base}.svg')
        plt.savefig(f'{output_base}.pdf')
        plt.close()
        print(f"Spy ROC/PR plots saved to {output_base}.[png/svg/pdf]")
        # 返回 PR 曲线所需的数据 和 AP分值
        roc_data_tuple = (fpr, tpr, roc_auc)
        pr_data_tuple = (recall, precision, avg_precision)
        return roc_data_tuple, pr_data_tuple

    def plot_spy_cdf_ks(self):
        """
         绘制 Spy vs Real Unlabeled 的累积分布函数(CDF)图，
        并使用KS检验(Kolmogorov-Smirnov test)来量化两个分布的差异。
        """
        if self.p_spy_matrix is None:
            print("\nSkipping spy CDF/KS plot (no spies were used).")
            return None

        print("\n--- Generating Spy vs. Unlabeled CDF Plot with KS-Test ---")

        # 1. 获取 B 次迭代的平均概率
        mean_p_unl = self.p_unl_matrix.mean(axis=0)
        mean_p_spy = self.p_spy_matrix.mean(axis=0)

        # 2. 执行双样本KS检验
        ks_stat, p_value = ks_2samp(mean_p_unl, mean_p_spy)

        # 3. 绘图
        plt.figure(figsize=(5, 4))
        ax = plt.gca()  # 获取当前坐标轴

        sns.ecdfplot(mean_p_unl, label=f'Real Unlabeled (n={len(mean_p_unl)})', ax=ax, color='blue')
        sns.ecdfplot(mean_p_spy, label=f'Spies (n={len(mean_p_spy)})', ax=ax, color='red')

        # 4. 添加标注
        annotation_text = (f'KS Statistic: {ks_stat:.3f}\n'
                           f'p-value: {p_value:.2e}')  # 使用科学计数法显示p值

        # 将标注放在图的左上角
        plt.text(0.05, 0.95, annotation_text, transform=ax.transAxes,
                 fontsize=10, va='top', ha='left',
                 bbox=dict(boxstyle="round,pad=0.5", fc="wheat", alpha=0.5))

        plt.xlabel('Posterior Probability P(y=1|x)')
        plt.ylabel('Cumulative Probability (CDF)')
        # plt.title('CDF of Spy vs. Unlabeled Probabilities')
        plt.legend()

        # 5. 保存图像
        output_base = os.path.join(self.args.output_dir, 'fig_spy_cdf_ks')
        plt.savefig(f'{output_base}.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{output_base}.svg', bbox_inches='tight')
        plt.savefig(f'{output_base}.pdf', bbox_inches='tight')
        plt.close()
        print(f"Spy CDF/KS plot saved to {output_base}.[png/svg/pdf]")

        return mean_p_unl, mean_p_spy, ks_stat

    def plot_uncertainty_spatial(self):
        """
        绘制“平均后验概率”和“不确定性(CI宽度)”的空间分布图。
        这需要 'X' 和 'Y' 列在 self.df_unl 中。
        """
        # 检查是否已计算了统计数据
        if 'pu_prob_mean' not in self.df_unl.columns:
            print("\nWarning: Sample-wise statistics not found. Skipping spatial uncertainty plot.")
            print("         Run 'run_pu_estimation' first.")
            return

        # 检查是否有 'X' 和 'Y' 列
        if 'X' not in self.df_unl.columns or 'Y' not in self.df_unl.columns:
            print("\nWarning: 'X' or 'Y' columns not found. Skipping spatial uncertainty plot.")
            return

        print("\n--- Generating Spatial Probability vs. Uncertainty Plot ---")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

        # --- 子图 1: 平均概率 (易感性图) ---
        # 使用小尺寸的点和 viridis 色带 (或 'spectral_r'，通常用于地学)
        sc1 = ax1.scatter(self.df_unl['X'], self.df_unl['Y'], c=self.df_unl['pu_prob_mean'],
                          s=1, cmap='viridis', vmin=0, vmax=1)
        plt.colorbar(sc1, ax=ax1, label='Mean Posterior Probability $P(y=1|x)$')
        ax1.set_title('Mean Posterior Probability (Susceptibility)')
        ax1.set_xlabel('X Coordinate')
        ax1.set_ylabel('Y Coordinate')
        ax1.set_aspect('equal', 'box')

        # --- 子图 2: 不确定性 (95% CI 宽度) ---
        sc2 = ax2.scatter(self.df_unl['X'], self.df_unl['Y'], c=self.df_unl['pu_ci_width_95'],
                          s=1, cmap='magma', vmin=0)
        plt.colorbar(sc2, ax=ax2, label='Uncertainty (95% CI Width)')
        ax2.set_title('Prediction Uncertainty (95% CI Width)')
        ax2.set_xlabel('X Coordinate')
        ax2.set_ylabel('Y Coordinate')
        ax2.set_aspect('equal', 'box')

        plt.tight_layout()

        # 6. 保存图像
        output_base = os.path.join(self.args.output_dir, 'fig_spatial_probability_uncertainty')
        plt.savefig(f'{output_base}.png', dpi=300)
        plt.savefig(f'{output_base}.svg')
        plt.savefig(f'{output_base}.pdf')
        plt.close()
        print(f"Spatial uncertainty plot saved to {output_base}.[png/svg/pdf]")

    def plot_uncertainty_errorbars(self):
        """
        绘制U样本的概率均值及其95%置信区间的误差棒图。
        """
        if 'pu_prob_mean' not in self.df_unl.columns:
            print("\nWarning: Sample-wise statistics not found. Skipping error bar plot.")
            return

        print("\n--- Generating Probability Error Bar Plot ---")

        # 1. 按均值概率对样本排序
        df_sorted = self.df_unl.sort_values('pu_prob_mean').reset_index(drop=True)

        # 2. 如果样本太多，绘图会很慢且混乱。我们只抽样绘制 (例如每 100 个点)
        sample_stride = max(1, len(df_sorted) // 1000)  # 确保最多绘制 1000 个点
        df_plot = df_sorted.iloc[::sample_stride]

        # 3. 计算 y 轴的误差 (yerr)
        # 误差棒需要 (下误差, 上误差)
        lower_error = df_plot['pu_prob_mean'] - df_plot['pu_prob_q025']
        upper_error = df_plot['pu_prob_q975'] - df_plot['pu_prob_mean']

        # 纠正由于浮点精度问题导致的微小负值
        # 误差条的长度不能为负，使用 clip(min=0) 修正
        lower_error = lower_error.clip(lower=0)
        upper_error = upper_error.clip(lower=0)

        asymmetric_error = [lower_error, upper_error]

        # 4. 绘图
        plt.figure(figsize=(10, 5))
        plt.errorbar(
            x=df_plot.index,
            y=df_plot['pu_prob_mean'],
            yerr=asymmetric_error,
            fmt='o',  # 'o' 表示只绘制点
            markersize=2,  # 点的大小
            alpha=0.6,  # 透明度
            capsize=3,  # 误差棒顶端的帽子大小
            elinewidth=1,  # 误差棒线宽
            label='Mean Probability (with 95% CI)'
        )

        plt.xlabel(f'Unlabeled Samples (Sorted by Mean Prob., 1 in {sample_stride} shown)')
        plt.ylabel('Posterior Probability $P(y=1|x)$')
        plt.legend(loc='upper left')
        # plt.title('Per-Sample Probability and Uncertainty')
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # 5. 保存图像
        output_base = os.path.join(self.args.output_dir, 'fig_probability_error_bars')
        plt.savefig(f'{output_base}.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{output_base}.svg', bbox_inches='tight')
        plt.savefig(f'{output_base}.pdf', bbox_inches='tight')
        plt.close()
        print(f"Probability error bar plot saved to {output_base}.[png/svg/pdf]")

    def plot_prob_uncertainty_hist2d(self):
        """
        绘制概率-不确定性联合分布的 2D直方图。
        这用于可视化样本在 (概率, 不确定性) 空间中的密度分布。
        """
        # 检查是否已计算了统计数据
        if 'pu_prob_mean' not in self.df_unl.columns:
            print("\nWarning: Sample-wise statistics not found. Skipping joint 2D-hist plot.")
            print("         Run 'run_pu_estimation' first.")
            return

        print("\n--- Generating Probability vs. Uncertainty Joint 2D-Histogram Plot ---")

        plt.figure(figsize=(6, 5))  # 稍微调整尺寸以便容纳颜色条

        # 1. 绘制 2D 直方图
        # X: 平均概率
        # Y: 不确定性 (95% CI 宽度)
        # bins: 分箱数量 (例如 50x50 的网格)
        # norm=mpl.colors.LogNorm(): 关键！使用对数刻度来显示密度
        # cmap: 'viridis' 或 'cividis'
        # mincnt=1: 只显示至少包含1个样本的方格
        h, xedges, yedges, image = plt.hist2d(
            self.df_unl['pu_prob_mean'],
            self.df_unl['pu_ci_width_95'],
            bins=50,
            norm=mpl.colors.LogNorm(),  # <--- 使用 LogNorm 进行对数色标
            cmap='viridis',
            cmin=1
        )

        # 2. 添加颜色条
        # 将 plt.hist2d 返回的 image 对象传递给 colorbar
        cb = plt.colorbar(image, label='log(Sample Density)')
        cb.ax.tick_params(width=0.7)  # 保持颜色条刻度线粗细一致

        # 3. 设置标签和范围
        plt.xlabel('Mean Posterior Probability $P(y=1|x)$')
        plt.ylabel('Uncertainty (95% CI Width)')

        # 4. 设定坐标轴范围
        plt.xlim(0, 1)  # 概率必须在 [0, 1] 之间
        plt.ylim(bottom=0)  # 不确定性从 0 开始

        # 5. 保存图像
        output_base = os.path.join(self.args.output_dir, 'fig_prob_uncertainty_hist2d')  # <-- 文件名已更新
        plt.savefig(f'{output_base}.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{output_base}.svg', bbox_inches='tight')
        plt.savefig(f'{output_base}.pdf', bbox_inches='tight')
        plt.close()
        print(f"Joint probability-uncertainty 2D-histogram plot saved to {output_base}.[png/svg/pdf]")

    def plot_rn_uncertainty_errorbars(self, df_reliable_negs, tau, f_threshold):
        """
        专门为最终选定的“可靠负样本”(RN) 绘制概率误差棒图。
        """
        # 检查数据是否存在
        if 'pu_prob_mean' not in df_reliable_negs.columns:
            print("\nWarning: Sample-wise statistics not found in RN dataframe. Skipping RN error bar plot.")
            return

        if len(df_reliable_negs) == 0:
            print(f"\nWarning: No Reliable Negatives found for θ={tau}, η={f_threshold}. Skipping RN error bar plot.")
            return

        print("\n--- Generating Error Bar Plot for Reliable Negatives (RN) ONLY ---")

        # 1. 按均值概率对 RN 样本排序
        df_sorted = df_reliable_negs.sort_values('pu_prob_mean').reset_index(drop=True)

        # 2. 抽样 (如果RN数量仍然非常大)
        sample_stride = max(1, len(df_sorted) // 1000)  # 确保最多绘制 1000 个点
        df_plot = df_sorted.iloc[::sample_stride]

        # 3. 计算 y 轴的误差 (yerr)
        lower_error = df_plot['pu_prob_mean'] - df_plot['pu_prob_q025']
        upper_error = df_plot['pu_prob_q975'] - df_plot['pu_prob_mean']

        # 纠正由于浮点精度问题导致的微小负值
        # 误差条的长度不能为负，使用 clip(min=0) 修正
        lower_error = lower_error.clip(lower=0)
        upper_error = upper_error.clip(lower=0)

        asymmetric_error = [lower_error, upper_error]

        # 4. 绘图
        plt.figure(figsize=(10, 5))
        plt.errorbar(
            x=df_plot.index,
            y=df_plot['pu_prob_mean'],
            yerr=asymmetric_error,
            fmt='o',  # 'o' 表示只绘制点
            markersize=2,  # 点的大小
            alpha=0.6,  # 透明度
            capsize=3,  # 误差棒顶端的帽子大小
            elinewidth=1,  # 误差棒线宽
            label=f'RN Samples (n={len(df_sorted)}) with 95% CI'
        )

        # --- 关键：添加概率阈值 tau (θ) 作为参考线 ---
        plt.axhline(y=tau, color='red', linestyle='--', linewidth=2,
                    label=f'Probability Threshold θ = {tau}')

        plt.xlabel(f'Reliable Negative Samples (Sorted by Mean Prob., 1 in {sample_stride} shown)')
        plt.ylabel('Posterior Probability $P(y=1|x)$')
        plt.legend(loc='upper left')
        plt.title(f'Probability Distribution of Reliable Negatives (θ={tau}, η={f_threshold})')
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # --- 关键：自动缩放Y轴 ---
        # 找到所有 q_high (97.5%分位数) 的最大值
        max_y_error = df_plot['pu_prob_q975'].max()
        # 将Y轴上限设置为 tau 或 max_y 中较大的一个，再加一点余量
        upper_limit = max(max_y_error, tau) * 1.2 + 0.02  # 避免 max_y 和 tau 都很低
        plt.ylim(0, upper_limit)  # Y轴从0开始

        # 5. 保存图像
        output_base = os.path.join(
            self.args.output_dir,
            f'fig_probability_error_bars_RN_ONLY_tau{tau}_f{f_threshold}'
        )
        plt.savefig(f'{output_base}.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{output_base}.svg', bbox_inches='tight')
        plt.savefig(f'{output_base}.pdf', bbox_inches='tight')
        plt.close()
        print(f"Reliable Negatives (RN) error bar plot saved to {output_base}.[png/svg/pdf]")

    def _evaluate_final_classifier(self, df_reliable_negs):
        """
        一个辅助函数，用于训练和评估最终的监督分类器。
        <--- 它现在使用可配置的模型和优化过的超参数。
        """
        # 1. 准备训练数据
        df_train = pd.concat([self.df_pos, df_reliable_negs])
        X_train_scaled = self.scaler.transform(df_train[self.feature_cols].values)
        y_train = df_train['label'].values

        # 2. 获取最终分类器实例
        # 不再硬编码，而是根据 final_eval_model 参数来获取
        final_clf = self._get_classifier_instance(self.args.final_eval_model)

        # 显式地为最终评估设置一个固定的随机种子
        if 'random_state' in final_clf.get_params():
            final_clf.set_params(random_state=42)

        # 3. <--- 使用 cross_validate 同时评估多个指标
        scoring_metrics = [
            'accuracy',
            'precision_weighted',
            'recall_weighted',
            'f1_weighted',
            'roc_auc',
            'average_precision'
        ]
        # 定义一个字典，用于在发生错误时返回
        zero_scores = {metric: 0.0 for metric in scoring_metrics}

        # 3. 使用交叉验证评估性能
        try:
            # 使用 cross_validate
            scores_dict = cross_validate(
                final_clf,
                X_train_scaled,
                y_train,
                cv=self.cv_splitter,
                scoring=scoring_metrics,  # <--- 传入一个包含多个指标的列表
                n_jobs=-1,
                return_train_score=False  # 只关心测试集的分数
            )

            # 计算每个指标的平均值并返回
            mean_scores = {key.replace('test_', ''): np.mean(value) for key, value in scores_dict.items()}
            return mean_scores

        except ValueError:

            return zero_scores

    def plot_tradeoff_scatter(self, df_results):
        """
        创建一个气泡图，可视化四个维度（RN数量, 性能, θ, η）的关系。
        此版本包含健壮的图例分离和定位逻辑，并自动标注出性能最佳的点。
        """
        print("\n--- Generating Final Bubble Chart with Annotation ---")

        # --- 数据准备部分 ---
        df_plot = df_results.copy()
        if 'composite_score' not in df_plot.columns:
            #metrics_to_combine = ['f1_weighted', 'roc_auc', 'average_precision']
            metrics_to_combine = ['roc_auc', 'average_precision']
            if not all(metric in df_plot.columns for metric in metrics_to_combine):
                print(f"Warning: Not all required metrics found. Skipping scatter plot.")
                return
            scaler = MinMaxScaler()
            normalized_metrics = scaler.fit_transform(df_plot[metrics_to_combine])
            from scipy.stats import gmean
            df_plot['composite_score'] = gmean(normalized_metrics + 1e-9, axis=1)
        # --- 数据准备结束 ---

        plt.figure(figsize=(8, 5))

        # 1. 绘制散点图
        ax = sns.scatterplot(
            data=df_plot,
            x='num_reliable_negatives',
            y='composite_score',
            hue='tau',
            size='f_threshold',  # f_threshold 即 η
            palette='viridis',
            legend='full',
            sizes=(30, 250),
            alpha=0.7,
            edgecolor='k',
            linewidth=0.5
        )

        # --- 健壮的图例处理逻辑 ---
        handles, labels = ax.get_legend_handles_labels()
        ax.get_legend().remove()
        try:
            split_index = labels.index('f_threshold')
        except ValueError:
            split_index = len(df_plot['tau'].unique()) + 1
        hue_handles, hue_labels = handles[1:split_index], labels[1:split_index]
        size_handles, size_labels = handles[split_index + 1:], labels[split_index + 1:]
        legend1 = ax.legend(hue_handles, hue_labels, title='θ',
                            bbox_to_anchor=(1.02, 1.0), loc='upper left',
                            title_fontsize='11', fontsize='9', frameon=False,
                            labelspacing=0.2)
        ax.add_artist(legend1)
        legend2 = ax.legend(size_handles, size_labels, title='η',
                            bbox_to_anchor=(1.02, 0.6), loc='upper left',  # 将 0.65 降低到 0.55
                            title_fontsize='11', fontsize='9', frameon=False,
                            labelspacing=1
                            )
        # --- 图例处理结束 ---

        # +++ 自动寻找并标注最优点 +++

        # 1. 找到综合性能分数最高的那一行数据
        best_point_idx = df_plot['composite_score'].idxmax()
        best_point = df_plot.loc[best_point_idx]

        # 2. 提取该点的坐标和关键参数值
        x_coord = best_point['num_reliable_negatives']
        y_coord = best_point['composite_score']
        tau_val = best_point['tau']
        eta_val = best_point['f_threshold']
        rn_val = int(best_point['num_reliable_negatives'])

        # +++ 将找到的最优参数存储到类的属性中 +++
        self.best_tau = tau_val
        self.best_eta = eta_val
        print(f"\n==> Automatically found optimal parameters: θ = {self.best_tau}, η = {self.best_eta} <==")

        # 3. 创建要在图上显示的注释文本
        annotation_text = (
            f"Optimal Point\n"
            f"θ = {tau_val:.2f}\n"  # 格式化为两位小数
            f"η = {eta_val:.2f}\n"  # 格式化为两位小数
            f"RN Count = {rn_val}"
        )

        # 4. 使用 ax.annotate() 将文本和箭头添加到图上
        ax.annotate(
            text=annotation_text,
            xy=(x_coord, y_coord),  # 箭头指向的点
            xytext=(x_coord - 20, y_coord - 0.4),  # 文本框的位置
            arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
            fontsize=10,
            fontweight='bold',
            ha='center',  # 水平居中对齐
            bbox=dict(boxstyle="round,pad=0.5", fc="wheat", ec="black", lw=1, alpha=0.8)
        )
        # +++ 注释功能结束 +++

        # 设置坐标轴标签和网格
        plt.xlabel('Number of Reliable Negatives (RN Count)')
        plt.ylabel('Composite Performance Score (Normalized)')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)

        # 调整布局以确保所有元素（如图例）都在图像内
        plt.tight_layout(rect=[0, 0, 0.95, 1])  # 在右侧留出5%的空间给图a例

        # 保存图像
        output_base = os.path.join(self.args.output_dir, 'fig_tradeoff_scatter_plot_annotated')
        plt.savefig(f'{output_base}.png', dpi=300)  # bbox_inches='tight' 可能与手动布局冲突，可先移除
        plt.savefig(f'{output_base}.svg')
        plt.savefig(f'{output_base}.pdf')
        plt.close()

        print(f"Annotated scatter plot saved to {output_base}.[png/svg/pdf]")

    def perform_sensitivity_analysis(self):
        """
        <--- 对θ和f_threshold进行敏感性分析，并可视化结果。
        现在它会同时计算RN数量和最终模型的F1分数。
        """
        if self.p_unl_matrix is None:
            raise RuntimeError("PU estimation must be run before sensitivity analysis.")

        print("\n--- Performing End-to-End Sensitivity Analysis for θ and η ---")
        results = []
        param_grid = list(itertools.product(self.args.tau_list, self.args.f_threshold_list))

        # --- 遍历参数网格 ---
        for i, (tau, f_thresh) in enumerate(param_grid):
            print(f"  Testing grid point {i + 1}/{len(param_grid)}: (θ={tau}, η={f_thresh})", end='\r')

            # 1. 筛选可靠负样本
            freq = (self.p_unl_matrix <= tau).sum(axis=0) / self.args.B
            reliable_mask = freq >= f_thresh
            df_reliable_negs = self.df_unl[reliable_mask]
            num_reliable_negs = len(df_reliable_negs)

            # 2. <--- 评估最终分类器的性能，接收一个分数字典
            min_samples_for_eval = self.args.cv_folds * 2
            if num_reliable_negs >= min_samples_for_eval:
                # eval_scores 将会是 {'accuracy': 0.95, 'f1_weighted': 0.94, ...}
                eval_scores = self._evaluate_final_classifier(df_reliable_negs)
            else:
                # 如果样本不足，创建一个包含所有指标且值为0的字典
                scoring_metrics = [
                    'accuracy', 'precision_weighted', 'recall_weighted',
                    'f1_weighted', 'roc_auc', 'average_precision']
                eval_scores = {metric: 0.0 for metric in scoring_metrics}

            # 3. <--- 将所有分数存储到结果中
            stats = {
                'tau': tau,
                'f_threshold': f_thresh,
                'num_reliable_negatives': num_reliable_negs,
            }
            # 使用 update 方法将所有评估分数合并到 stats 字典中
            stats.update(eval_scores)
            results.append(stats)

        print("\nSensitivity analysis complete.")

        # 保存包含所有结果的详细CSV文件
        df_results = pd.DataFrame(results)
        csv_path = os.path.join(self.args.output_dir, 'sensitivity_analysis_full_results.csv')
        df_results.to_csv(csv_path, index=False)
        print(f"Full sensitivity analysis results saved to {csv_path}")

        # ------ 绘图部分 ------
        try:
            # 图1：RN数量热图
            pivot_counts = df_results.pivot(index='f_threshold', columns='tau', values='num_reliable_negatives')
            plt.figure(figsize=(8, 6))
            ax = sns.heatmap(pivot_counts, annot=True, annot_kws={'size': 8}, fmt=".0f",
                             cbar_kws={'label': 'Number of Reliable Negatives'},
                             cmap="viridis", linewidths=.5)
            ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)

            # --- 颜色条刻度线粗细 ---
            cbar_counts = ax.collections[-1].colorbar
            cbar_counts.ax.tick_params(width=0.7)  # 设置刻度线宽度为 0.7

            # plt.title('Sensitivity Analysis: Number of Reliable Negatives')
            plt.xlabel('Probability Threshold (θ)')
            plt.ylabel('Frequency Threshold (η)')
            heatmap_base_counts = os.path.join(self.args.output_dir, 'fig_sensitivity_heatmap_RN_Count')
            plt.savefig(f'{heatmap_base_counts}.png', dpi=300, bbox_inches='tight')
            plt.savefig(f'{heatmap_base_counts}.svg', bbox_inches='tight')
            plt.savefig(f'{heatmap_base_counts}.pdf', bbox_inches='tight')
            plt.close()
            print(f"RN Count heatmap saved to {heatmap_base_counts}.[png/svg/pdf]")

            # 图2：<--- 为每个性能指标生成一个热力图
            metrics_to_plot = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted',
                               'roc_auc', 'average_precision']
            color_maps = {'accuracy': 'viridis', 'precision_weighted': 'cividis', 'recall_weighted': 'magma',
                          'f1_weighted': 'plasma', 'roc_auc': 'inferno', 'average_precision': 'mako'}

            # <--- 3. 创建一个标签映射
            label_maps = {
                'accuracy': 'Accuracy',
                'precision_weighted': 'Precision (Weighted)',
                'recall_weighted': 'Recall (Weighted)',
                'f1_weighted': 'F1-Score (Weighted)',
                'roc_auc': 'ROC-AUC',
                'average_precision': 'PR-AUC'  # 将 'average_precision' 映射为 'PR-AUC'
            }

            for metric in metrics_to_plot:
                # 检查结果列中是否存在该指标
                if metric not in df_results.columns:
                    print(f"Warning: Metric '{metric}' not found in results. Skipping its heatmap.")
                    continue

                pivot_metric = df_results.pivot(index='f_threshold', columns='tau', values=metric)
                plt.figure(figsize=(8, 6))

                # 为指标选择一个色系
                # cmap = color_maps.get(metric, 'viridis')

                # 创建颜色条标签
                cbar_label = label_maps.get(metric, metric.title())

                ax_metric = sns.heatmap(pivot_metric, annot=True, fmt=".3f", annot_kws={'size': 8},
                                        cmap=color_maps.get(metric, 'viridis'),
                                        cbar_kws={'label': cbar_label},
                                        linewidths=.5)
                ax_metric.tick_params(axis='both', which='both', length=0)

                # --- 颜色条刻度线粗细 ---
                cbar_metric = ax_metric.collections[-1].colorbar
                cbar_metric.ax.tick_params(width=0.7)  # 设置刻度线宽度为 0.7

                # (f'Sensitivity Analysis: Final Model {metric.replace("_", " ").title()}')
                plt.xlabel('Probability Threshold (θ)')
                plt.ylabel('Frequency Threshold (η)')

                output_filename_base = \
                    f'fig_sensitivity_heatmap_{label_maps.get(metric, metric).upper().replace(" ", "_")}'
                heatmap_base_path = os.path.join(self.args.output_dir, output_filename_base)
                plt.savefig(f'{heatmap_base_path}.png', dpi=300, bbox_inches='tight')
                plt.savefig(f'{heatmap_base_path}.svg', bbox_inches='tight')
                plt.savefig(f'{heatmap_base_path}.pdf', bbox_inches='tight')
                plt.close()
                print(f"Model {metric} heatmap saved to {heatmap_base_path}.[png/svg/pdf]")

        except Exception as e:
            print(f"Could not generate heatmaps, possibly due to input parameter shape. Error: {e}")
        self.plot_tradeoff_scatter(df_results)

    def generate_final_set(self, tau, f_threshold):
        """根据选定的最佳参数生成最终的可靠负样本集。"""
        print(f"\n--- Generating final reliable negative set with θ={tau} and f_threshold={f_threshold} ---")

        # 1. 计算频率 (这个 freq 是一个长度为 N_unlabeled 的数组)
        freq = (self.p_unl_matrix <= tau).sum(axis=0) / self.args.B

        # 2. 创建掩码
        reliable_mask = freq >= f_threshold

        # 3. 从 self.df_unl (它已经包含了所有U样本的统计数据) 中筛选RN
        #    使用 .copy() 避免 SettingWithCopyWarning
        reliable_negs_df = self.df_unl[reliable_mask].copy()

        # 4. 保存CSV
        output_path = os.path.join(
            self.args.output_dir,
            f'reliable_negatives_final_tau{tau}_f{f_threshold}.csv'
        )
        reliable_negs_df.to_csv(output_path, index=False)
        print(f"Saved {len(reliable_negs_df)} reliable negatives to {output_path}")

        # -----------------------------------------------------------------
        # --- 调用专门的绘图函数来绘制RN的误差棒图 ---
        # -----------------------------------------------------------------
        # 检查是否真的找到了RN，避免对空DataFrame绘图
        if len(reliable_negs_df) > 0:
            # 调用新函数，并将RN的DataFrame和参数传递给它
            self.plot_rn_uncertainty_errorbars(reliable_negs_df, tau, f_threshold)
        else:
            print(f"Skipping RN error bar plot as 0 RNs were generated for θ={tau}, η={f_threshold}.")

    def run_model_comparison_prestudy(self):
        """
        <--- 运行一个预实验，通过Optuna优化来比较不同分类器的性能。

        : 此函数现在只优化在 --model_for_main_run 中指定的那一个模型。
        """
        print("\n" + "=" * 50)
        print(f"--- Running Pre-study Optuna Hyperparameter Optimization ---")

        # --- 获取选择的指标 ---
        metric_to_optimize = self.args.prestudy_metric

        # ---  ---
        # 不再循环所有模型，只获取用户在命令行中选择的那一个
        model_to_run = self.args.model_for_main_run

        if model_to_run not in self.MODELS_TO_TEST:
            raise ValueError(
                f"Model '{model_to_run}' specified in --model_for_main_run is not defined in MODELS_TO_TEST.")

        print(f"--- Optimizing for: {metric_to_optimize} ---")
        print(f"--- Target Model: {model_to_run} ---")
        print("=" * 50)

        if self.df_spy is None:
            # ... (返回逻辑不变) ...
            print("Error: Spy fraction must be > 0 for optimization.")
            return pd.DataFrame()  # 返回一个空的 DataFrame

        # ... (数据准备逻辑不变) ...
        X_pos_scaled = self.scaler.transform(self.df_pos[self.feature_cols].values)
        X_unl_scaled = self.scaler.transform(self.df_unl[self.feature_cols].values)
        X_spy_scaled = self.scaler.transform(self.df_spy[self.feature_cols].values)
        X_unl_with_spies = np.vstack([X_unl_scaled, X_spy_scaled])

        results = []


        print(f"\n  Optimizing model: {model_to_run}...")
        start_time = time.time()

        # 对指定的那一个模型运行超参数优化
        best_params, best_score = self._optimize_model_hyperparameters(
            model_to_run, X_pos_scaled, X_unl_with_spies
        )

        self.best_hyperparams[model_to_run] = best_params

        end_time = time.time()

        # 存储最佳结果
        metrics = {
            'model': model_to_run,
            'time_seconds': end_time - start_time,
            'best_params': best_params
        }
        metrics[metric_to_optimize] = best_score
        results.append(metrics)



        # 将结果转为DataFrame并展示
        df_comparison = pd.DataFrame(results).sort_values(by=metric_to_optimize, ascending=False)
        print("\n--- Optimized Model Parameters ---")
        df_display = df_comparison.copy()
        df_display['best_params'] = df_display['best_params'].astype(str)
        print(df_display.to_string())

        output_path = os.path.join(self.args.output_dir, f'model_optimization_{model_to_run}.csv')
        df_comparison.to_csv(output_path, index=False)
        print(f"\nOptimized model parameters saved to {output_path}")

        # --- 绘图部分不再需要，因为它只优化一个模型 ---
        # (或者您可以保留它，它只会画一个条形图)
        # 为了保持原样，我们保留绘图
        plt.figure(figsize=(5, 4))
        x_axis_metric = metric_to_optimize
        if metric_to_optimize == 'spy_separation':
            x_label = 'Mean Spy Prob - Mean Unlabeled Prob'
        elif metric_to_optimize == 'average_precision':
            x_label = 'Average Precision (Spy vs. Unlabeled)'
        else:  # 'roc_auc'
            x_label = 'ROC-AUC (Spy vs. Unlabeled)'

        sns.barplot(x=x_axis_metric, y='model', data=df_comparison, orient='h', width=0.6)
        plt.xlabel(x_label)
        plt.ylabel('Model')

        plot_base_path = os.path.join(self.args.output_dir, f'fig_model_optimization_{model_to_run}')
        plt.savefig(f'{plot_base_path}.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{plot_base_path}.svg', bbox_inches='tight')
        plt.savefig(f'{plot_base_path}.pdf', bbox_inches='tight')
        plt.close()
        print(f"Model optimization plot saved to {plot_base_path}.[png/svg/pdf]")

        return df_comparison


def plot_summary_pr_curves(pr_data_list, base_output_dir):
    """
    将所有PR曲线绘制到一张图上。
    pr_data_list 是一个列表，每个元素是:
    {'label': '1500m', 'recall': [...], 'precision': [...], 'ap': 0.87}
    """
    print("\n--- Generating Summary PR Curve Plot ---")
    plt.figure(figsize=(7, 6))
    ax = plt.gca()

    # 根据 AP 分数（降序）对列表排序，让图例更清晰
    pr_data_list.sort(key=lambda x: x['ap'], reverse=True)

    # 创建一个颜色映射
    colors = plt.cm.viridis(np.linspace(0, 1, len(pr_data_list)))

    for i, data in enumerate(pr_data_list):
        ax.plot(data['recall'], data['precision'], lw=2,
                label=f"{data['label']} (AP = {data['ap']:.3f})",
                color=colors[i])

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Summary: Precision-Recall Curves (Spy vs. Unlabeled)')
    # 将图例放在图表外部
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), fontsize='small')

    output_base = os.path.join(base_output_dir, '_SUMMARY_fig_all_pr_curves')
    plt.savefig(f'{output_base}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_base}.svg', bbox_inches='tight')
    plt.savefig(f'{output_base}.pdf', bbox_inches='tight')
    plt.close()
    print(f"Summary PR curve plot saved to {output_base}.[png/svg/pdf]")


def plot_summary_cdf_curves(cdf_data_list, base_output_dir):
    """
    将所有CDF曲线绘制到一张图上。
    cdf_data_list 是一个列表，每个元素是:
    {'label': '1500m', 'p_unl': [...], 'p_spy': [...], 'ks': 0.97}
    """
    print("\n--- Generating Summary CDF Curve Plot ---")
    plt.figure(figsize=(8, 7))
    ax = plt.gca()

    # 使用两个色系：蓝色系给Unlabeled，红色系给Spy
    unl_colors = plt.cm.Blues(np.linspace(0.4, 1, len(cdf_data_list)))
    spy_colors = plt.cm.Reds(np.linspace(0.4, 1, len(cdf_data_list)))

    # 根据 KS 分数（降序）排序
    cdf_data_list.sort(key=lambda x: x['ks'], reverse=True)

    for i, data in enumerate(cdf_data_list):
        label_base = data['label']

        # 绘制 Unlabeled 曲线 (使用虚线)
        sns.ecdfplot(data['p_unl'],
                     label=f"{label_base} (Unlabeled)",
                     ax=ax, color=unl_colors[i], linestyle='--')

        # 绘制 Spy 曲线 (使用实线)
        sns.ecdfplot(data['p_spy'],
                     label=f"{label_base} (Spy, KS={data['ks']:.3f})",
                     ax=ax, color=spy_colors[i], linestyle='-')

    ax.set_xlabel('Posterior Probability P(y=1|x)')
    ax.set_ylabel('Cumulative Probability (CDF)')
    ax.set_title('Summary: CDF Curves (Spy vs. Unlabeled)')
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), fontsize='small')

    output_base = os.path.join(base_output_dir, '_SUMMARY_fig_all_cdf_curves')
    plt.savefig(f'{output_base}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_base}.svg', bbox_inches='tight')
    plt.savefig(f'{output_base}.pdf', bbox_inches='tight')
    plt.close()
    print(f"Summary CDF curve plot saved to {output_base}.[png/svg/pdf]")


def plot_summary_roc_curves(roc_data_list, base_output_dir):
    """
    将所有ROC曲线绘制到一张图上。
    roc_data_list 是一个列表，每个元素是:
    {'label': '1500m', 'fpr': [...], 'tpr': [...], 'auc': 0.95}
    """
    print("\n--- Generating Summary ROC Curve Plot ---")
    plt.figure(figsize=(7, 6))
    ax = plt.gca()

    # 根据 AUC 分数（降序）对列表排序
    roc_data_list.sort(key=lambda x: x['auc'], reverse=True)

    # 创建一个颜色映射
    colors = plt.cm.viridis(np.linspace(0, 1, len(roc_data_list)))

    for i, data in enumerate(roc_data_list):
        ax.plot(data['fpr'], data['tpr'], lw=2,
                label=f"{data['label']} (AUC = {data['auc']:.3f})",
                color=colors[i])

    # 绘制 45 度对角线（随机猜测线）
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Summary: ROC Curves (Spy vs. Unlabeled)')
    # 将图例放在图表外部的右下角
    ax.legend(loc='lower right', bbox_to_anchor=(1.02, 0.0), fontsize='small')

    output_base = os.path.join(base_output_dir, '_SUMMARY_fig_all_roc_curves')
    plt.savefig(f'{output_base}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_base}.svg', bbox_inches='tight')
    plt.savefig(f'{output_base}.pdf', bbox_inches='tight')
    plt.close()
    print(f"Summary ROC curve plot saved to {output_base}.[png/svg/pdf]")


# =====================================================================
# =====================================================================

def plot_summary_combined_figure(pr_data_list, roc_data_list, cdf_data_list, df_final_comparison, base_output_dir):
    """
    将 ROC, PR, CDF 和 性能vs距离 总结图合并到一个 2x2 的面板中，
    用于A4排版和SCI论文。
    """
    print("\n--- Generating Combined 2x2 Summary Panel Figure ---")

    # --- 1. 创建 2x2 子图网格 ---
    # 尺寸 (12, 10) 意味着每个子图大致为 6x5
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    # ======================================================
    # --- 2. 面板 (a): ROC 曲线 (ax1) ---
    # ======================================================
    ax = ax1  # 目标是 ax[0, 0]

    # 根据 AUC 分数（降序）对列表排序
    roc_data_list.sort(key=lambda x: x['auc'], reverse=True)
    colors = plt.cm.viridis(np.linspace(0, 1, len(roc_data_list)))

    for i, data in enumerate(roc_data_list):
        ax.plot(data['fpr'], data['tpr'], lw=2,
                label=f"{data['label']} (AUC = {data['auc']:.3f})",
                color=colors[i])

    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves (Spy vs. Unlabeled)')

    ax.legend(loc='lower right', fontsize='small', frameon=True, framealpha=0.8)

    # 面板标签 (a)
    ax.text(0.02, 0.95, '(a)', transform=ax.transAxes, fontsize=14,
            fontweight='bold', va='top', ha='left')

    # ======================================================
    # --- 3. 面板 (b): PR 曲线 (ax2) ---
    # ======================================================
    ax = ax2  # 目标是 ax[0, 1]

    pr_data_list.sort(key=lambda x: x['ap'], reverse=True)
    colors = plt.cm.viridis(np.linspace(0, 1, len(pr_data_list)))

    for i, data in enumerate(pr_data_list):
        ax.plot(data['recall'], data['precision'], lw=2,
                label=f"{data['label']} (AP = {data['ap']:.3f})",
                color=colors[i])

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curves (Spy vs. Unlabeled)')

    ax.legend(loc='lower left', fontsize='small', frameon=True, framealpha=0.8)

    # 面板标签 (b)
    ax.text(0.02, 0.95, '(b)', transform=ax.transAxes, fontsize=14,
            fontweight='bold', va='top', ha='left')

    # ======================================================
    # --- 4. 面板 (c): CDF 曲线 (ax3) ---
    # ======================================================
    ax = ax3  # 目标是 ax[1, 0]

    unl_colors = plt.cm.Blues(np.linspace(0.4, 1, len(cdf_data_list)))
    spy_colors = plt.cm.Reds(np.linspace(0.4, 1, len(cdf_data_list)))
    cdf_data_list.sort(key=lambda x: x['ks'], reverse=True)

    for i, data in enumerate(cdf_data_list):
        label_base = data['label']
        # 绘制 Unlabeled 曲线 (使用虚线)
        sns.ecdfplot(data['p_unl'],
                     label=f"{label_base} (Unlabeled)",
                     ax=ax, color=unl_colors[i], linestyle='--')
        # 绘制 Spy 曲线 (使用实线)
        sns.ecdfplot(data['p_spy'],
                     label=f"{label_base} (Spy, KS={data['ks']:.3f})",
                     ax=ax, color=spy_colors[i], linestyle='-')

    ax.set_xlabel('Posterior Probability P(y=1|x)')
    ax.set_ylabel('Cumulative Probability (CDF)')
    ax.set_title('CDF Curves (Spy vs. Unlabeled)')

    ax.legend(loc='center right', fontsize='small', frameon=True, framealpha=0.8)

    # 面板标签 (c)
    ax.text(0.02, 0.95, '(c)', transform=ax.transAxes, fontsize=14,
            fontweight='bold', va='top', ha='left')

    # ======================================================
    # --- 5. [新] 面板 (d): 性能 vs 距离 (ax4) ---
    # ======================================================
    ax = ax4  # 目标是 ax[1, 1]

    try:
        # 1. 准备数据
        df_plot = df_final_comparison.copy()
        # 从 'dataset' 列 (e.g., 'samples...-1000m-...') 提取距离
        df_plot['distance'] = df_plot['dataset'].apply(
            lambda x: int(x.split('-')[1].replace('m', ''))
        )
        df_plot = df_plot.sort_values('distance')

        # 2. 绘图
        ax.plot(df_plot['distance'], df_plot['roc_auc'], marker='o',
                label='ROC-AUC', markersize=6)
        ax.plot(df_plot['distance'], df_plot['average_precision'], marker='s',
                label='PR-AUC (AP)', markersize=6)
        ax.plot(df_plot['distance'], df_plot['ks_statistic'], marker='^',
                label='KS-Statistic', markersize=6)

        # 3. 设置标签、标题和图例
        ax.set_xlabel('Distance (m)')
        ax.set_ylabel('Metric Value')
        ax.set_title('Model Performance vs. Distance')
        ax.legend(loc='best', fontsize='small', frameon=True, framealpha=0.8)
        ax.grid(True, linestyle='--', alpha=0.7)  # 添加网格线
        ax.set_ylim(bottom=0, top=1.05)  # 保证 Y 轴在 0-1 之间

    except Exception as e:
        print(f"Could not generate plot (d) 'Performance vs. Distance'. Error: {e}")
        # 在图上显示错误信息
        ax.text(0.5, 0.5, f'Error plotting (d):\n{e}',
                ha='center', va='center', color='red', wrap=True)

    # 面板标签 (d)
    ax.text(0.02, 0.95, '(d)', transform=ax.transAxes, fontsize=14,
            fontweight='bold', va='top', ha='left')

    # ======================================================
    # --- 6. 最终调整和保存 ---
    # ======================================================
    plt.tight_layout(pad=1.5)  # 自动调整子图间距

    output_base = os.path.join(base_output_dir, '_SUMMARY_fig_A4_Combined_Panel_2x2')
    plt.savefig(f'{output_base}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_base}.svg', bbox_inches='tight')
    plt.savefig(f'{output_base}.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"Combined 2x2 summary panel saved to {output_base}.[png/svg/pdf]")


if __name__ == '__main__':

    # ===== 设置全局随机种子以保证可复现性 =====
    import random

    SEED = 42
    os.environ['PYTHONHASHSEED'] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    # ==========================================

    # --- 定义要对比的数据集和输出 ---
    DATASETS_TO_RUN = [
        'samples_deposits15-1000m-2710.csv',
        'samples_deposits15-1500m-2710.csv',
        'samples_deposits15-2000m-2710.csv',
        'samples_deposits15-2500m-2710.csv',
        'samples_deposits15-3000m-2710.csv',
        'samples_deposits15-3500m-2710.csv',
        # 'samples_deposits15-4000m-2710.csv'
    ]

    # 定义一个基础输出目录，所有运行的结果将保存在此的子目录中
    BASE_OUTPUT_DIR = 'PU_Results_XGBoost_B500'

    # ------------------------------------

    p = argparse.ArgumentParser(
        description='Phase1: PU-learning Reliable Negative Selection with Spy Technique and Sensitivity Analysis')

    # --- 我们只保留影响所有运行的全局参数 ---
    p.add_argument('--B', type=int, default=500, help='Number of bootstrap iterations for PU estimation.')
    p.add_argument('--spy_fraction', type=float, default=0.2,
                   help='Fraction of positive samples to use as spies (e.g., 0.1 for 10%). Set to 0 to disable.')
    p.add_argument(
        '--tau_list', nargs='+', type=float, default=[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1],
        help='List of probability thresholds (θ) for sensitivity analysis.'
    )
    p.add_argument(
        '--f_threshold_list', nargs='+', type=float, default=[0.9, 0.91, 0.92, 0.93, 0.94,
                                                              0.95, 0.96, 0.97, 0.98, 0.99],
        help='List of frequency thresholds for sensitivity analysis.'
    )
    p.add_argument('--cv_folds', type=int, default=5, help='Number of folds for cross-validation.')
    p.add_argument('--n_trials_optuna', type=int, default=200,
                   help='Number of optimization trials for each model in the pre-study.')

    p.add_argument(
        '--prestudy_metric',
        type=str,
        default='average_precision',
        choices=['spy_separation', 'average_precision', 'roc_auc'],
        help='The metric to optimize in the pre-study (Spy vs. Unlabeled).'
    )


    # --- 模型参数被硬编码为'RandomForest', 'XGBoost', 'SVM' ---
    p.add_argument(
        '--model_for_main_run', type=str, default='XGBoost', choices=['RandomForest', 'XGBoost', 'SVM'],
        help='Model used for hyperparameter optimization and main PU estimation runs.'
    )
    p.add_argument(
        '--final_eval_model', type=str, default=None, choices=['RandomForest', 'XGBoost', 'SVM'],
        help='Model used for evaluating final P-RN classifier in sensitivity analysis.'
    )

    # --- 解析一次通用参数 ---
    base_args = p.parse_args()
    if base_args.final_eval_model is None:
        base_args.final_eval_model = base_args.model_for_main_run

    # ===== 在此处调用新的样式设置函数 =====
    setup_publication_quality_plotting()

    # --- 用于存储所有运行结果的列表 ---
    final_comparison_results = []
    all_pr_data = []
    all_cdf_data = []
    all_roc_data = []

    # --- 开始循环运行所有数据集 ---
    for dataset_file in DATASETS_TO_RUN:

        print("\n" + "=" * 80)
        print(f"--- STARTING ANALYSIS FOR: {dataset_file} ---")
        print("=" * 80 + "\n")

        # 1. 复制基础参数并为本次运行设置特定参数
        args = argparse.Namespace(**vars(base_args))
        args.input_csv = dataset_file

        # 从文件名创建唯一的输出目录
        output_dir_name = os.path.splitext(dataset_file)[0]
        args.output_dir = os.path.join(BASE_OUTPUT_DIR, output_dir_name)

        # 2. 初始化分析器
        analyzer = PUAnalyzer(args)

        # 3. 运行模型对比预实验 (现在是SVM专属优化)
        #    并捕获返回的 df_comparison
        df_prestudy = analyzer.run_model_comparison_prestudy()

        # 提取 average_precision 分数
        # (取第一行的值)
        ap_score = df_prestudy[args.prestudy_metric].iloc[0]

        # 4. 运行核心的PU估计算法
        analyzer.run_pu_estimation(model_name=args.model_for_main_run)

        # 5. 绘制各种分析图
        analyzer.plot_prior_stability()
        analyzer.plot_spy_calibration()

        # 捕获 ROC 和 PR 曲线数据
        roc_results, pr_results = analyzer.plot_spy_roc_pr()

        if pr_results:  # 检查 PR 结果是否为 None
            recall_data, precision_data, ap_score_from_plot = pr_results
        else:
            recall_data, precision_data, ap_score_from_plot = None, None, 0

        if roc_results:  # 检查 ROC 结果是否为 None
            fpr_data, tpr_data, roc_auc_score_from_plot = roc_results
        else:
            fpr_data, tpr_data, roc_auc_score_from_plot = None, None, 0

        # 捕获 CDF 曲线数据
        cdf_results = analyzer.plot_spy_cdf_ks()
        if cdf_results:  # 检查是否为 None
            p_unl_data, p_spy_data, ks_statistic = cdf_results
        else:
            p_unl_data, p_spy_data, ks_statistic = None, None, 0

        # analyzer.plot_spy_roc_pr()

        # 捕获返回的 ks_statistic
        # ks_statistic = analyzer.plot_spy_cdf_ks()

        analyzer.plot_uncertainty_spatial()
        analyzer.plot_uncertainty_errorbars()
        analyzer.plot_prob_uncertainty_hist2d()

        # 6. 运行敏感性分析
        analyzer.perform_sensitivity_analysis()

        # 7. 自动生成最终数据集
        if analyzer.best_tau is not None and analyzer.best_eta is not None:
            analyzer.generate_final_set(analyzer.best_tau, analyzer.best_eta)
        else:
            print(f"\n--- Could not determine best parameters automatically for {dataset_file} ---")

        # 8. 存储本次运行的指标

        # 为图例创建一个简短的标签，例如 "1500m"
        try:
            radius_label = dataset_file.split('-')[1]
        except:
            radius_label = dataset_file  # 备用标签

        # 存储PR数据 (确保有数据)
        if recall_data is not None:
            all_pr_data.append({
                'label': radius_label,
                'recall': recall_data,
                'precision': precision_data,
                'ap': ap_score_from_plot  # 注意：这是B次迭代均值的AP
            })

        # 存储ROC数据 (确保有数据)
        if fpr_data is not None:
            all_roc_data.append({
                'label': radius_label,
                'fpr': fpr_data,
                'tpr': tpr_data,
                'auc': roc_auc_score_from_plot
            })

        # 存储CDF数据 (确保有数据)
        if p_unl_data is not None:
            all_cdf_data.append({
                'label': radius_label,
                'p_unl': p_unl_data,
                'p_spy': p_spy_data,
                'ks': ks_statistic
            })

        # 这个不变，它存储的是 预研的AP 和 KS值，用于最终的 .csv 表格
        final_comparison_results.append({
            'dataset': dataset_file,
            'average_precision': ap_score_from_plot,  # <--- 改为使用 B 次迭代的均值 AP
            'roc_auc': roc_auc_score_from_plot,
            'ks_statistic': ks_statistic if ks_statistic is not None else np.nan
        })

        print(f"\n--- COMPLETED ANALYSIS FOR: {dataset_file} ---")

    # --- 所有运行结束后，打印最终的对比表格 ---
    print("\n" + "=" * 80)
    print("--- FINAL DATASET COMPARISON RESULTS ---")
    print("=" * 80 + "\n")

    df_final_comparison = pd.DataFrame(final_comparison_results)

    # --- 按您关心的指标排序 ---
    # (例如，首先按 'average_precision' 降序, 然后按 'ks_statistic' 降序)
    df_final_comparison = df_final_comparison.sort_values(
        by=['average_precision', 'roc_auc', 'ks_statistic'],
        ascending=[False, False, False]
    )

    print(df_final_comparison.to_string(index=False))

    # 保存对比结果
    final_csv_path = os.path.join(BASE_OUTPUT_DIR, '_FINAL_COMPARISON_METRICS.csv')
    df_final_comparison.to_csv(final_csv_path, index=False)

    print(f"\nFinal comparison table saved to: {final_csv_path}")
    # 调用新的总结绘图函数

    if all_pr_data:  # 确保列表不为空
        plot_summary_pr_curves(all_pr_data, BASE_OUTPUT_DIR)

    if all_cdf_data:  # 确保列表不为空
        plot_summary_cdf_curves(all_cdf_data, BASE_OUTPUT_DIR)

    if all_roc_data:  # 确保列表不为空
        plot_summary_roc_curves(all_roc_data, BASE_OUTPUT_DIR)

    # =====================================================================
    # =====================================================================
    # 确保所有数据都已收集，再调用组合绘图函数
    if all_pr_data and all_roc_data and all_cdf_data:
        plot_summary_combined_figure(
            all_pr_data,
            all_roc_data,
            all_cdf_data,
            df_final_comparison,  # <--- 传入包含总结指标的DataFrame
            BASE_OUTPUT_DIR
        )

    print("\n\nAnalysis complete. Check the output directory for all results.")