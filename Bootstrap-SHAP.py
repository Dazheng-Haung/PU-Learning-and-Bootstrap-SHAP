import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import shap
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import optuna
from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances
from pygam import LinearGAM, s
from sklearn.metrics import f1_score, average_precision_score
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import re


# ----------------------------------------------------------------------
# 定义一个把 ascii 数字 → unicode 下标的翻译表
sub_map = str.maketrans('0123456789', '₀₁₂₃₄₅₆₇₈₉')
# ----------------------------------------------------------------------


def plot_feature_interaction_and_importance_optimized_cb(
        interaction_df,
        importance_df,
        interaction_cmap=cm.Purples,  # 默认紫色系
        importance_cmap=cm.Greens,  # 默认绿色系
        background_circle_color='lightgray',  # 更柔和的浅灰色背景圆环
        #title="Feature Importance and Interactions",  # 更简洁专业的标题
        save_path=None
):
    """
    绘制 SHAP 特征重要性 (Vimp) 和交互作用 (Vint) 的环形网络图。
    该版本针对 SCI 论文发表进行美化，确保视觉质量和信息清晰度。

    Args:
        interaction_df (pd.DataFrame): (n_features x n_features) 的交互作用矩阵 (Vint)。
        importance_df (pd.Series): (n_features,) 的特征重要性 (Vimp)。
        interaction_cmap: Vint 的 Matplotlib Colormap。建议使用连续的颜色，如 `cm.Purples`, `cm.Blues`。
        importance_cmap: Vimp 的 Matplotlib Colormap。建议使用连续的颜色，如 `cm.Greens`, `cm.YlGn`。
        background_circle_color: 节点背景环的颜色。
        title (str): 图表标题。
        save_path (str, optional): 图像保存路径 (e.g., "plot.pdf")。
                                   函数将自动保存 .png, .pdf, .svg 三种格式。
    """

    # --- SCI 论文级美化：全局 Matplotlib 参数设置 ---
    # 临时更改 Matplotlib 参数，只作用于此函数内部
    with plt.rc_context({
        'font.family': 'Arial',  # SCI 常用字体
        'font.size': 10,  # 基础字号
        'axes.labelsize': 11,  # 坐标轴标签字号
        'axes.titlesize': 14,  # 标题字号
        'xtick.labelsize': 9,  # 刻度标签字号
        'ytick.labelsize': 9,
        'legend.fontsize': 9,  # 图例字号
        'lines.linewidth': 0.8,  # 默认线宽
        'axes.edgecolor': 'black',  # 坐标轴边框颜色
        'axes.linewidth': 0.8,  # 坐标轴边框粗细
        'xtick.direction': 'in',  # 刻度向内
        'ytick.direction': 'in',
        'xtick.major.width': 0.8,  # 刻度线宽
        'ytick.major.width': 0.8,
        'figure.dpi': 300,  # 保存图像的DPI
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',  # 保存时自动调整边距
        'savefig.pad_inches': 0.1,  # 边距微调
    }):

            #1. --- 数据准备 ---
        features = importance_df.index.tolist()
        n_features = len(features)

        # 确保 interaction_df 是按 features 顺序排列的
        interaction_df = (interaction_df + interaction_df.T) / 2  # 对称化
        interaction_df = interaction_df.loc[features, features]

        vimp_values = importance_df.values
        vint_values = interaction_df.values

        # 2. --- Vimp (节点) 归一化 ---
        # 归一化 Vimp 用于颜色
        vimp_norm_color = Normalize(vmin=vimp_values.min(), vmax=vimp_values.max())
        # 归一化 Vimp 用于节点大小 (例如, 200 到 2200)
        vimp_norm_size = Normalize(vmin=vimp_values.min(), vmax=vimp_values.max())
        vimp_sizes = vimp_norm_size(vimp_values) * 2000 + 200  # 调整范围，确保节点不会太小
        vimp_colors = importance_cmap(vimp_norm_color(vimp_values))

        # 3. --- Vint (连线) 归一化 ---
        vint_flat = vint_values[np.triu_indices(n_features, k=1)]

        # 确保vmin不会因全0而导致报错，取一个极小值
        # 并且只对非零值进行归一化，避免0值影响颜色和宽度范围
        non_zero_vint = vint_flat[vint_flat > 0]
        if len(non_zero_vint) == 0:
            vint_min_val = 1e-6  # 所有交互都为0的情况
            vint_max_val = 1e-6
        else:
            vint_min_val = non_zero_vint.min()
            vint_max_val = non_zero_vint.max()

        vint_norm_color = Normalize(vmin=vint_min_val, vmax=vint_max_val)
        vint_norm_width = Normalize(vmin=vint_min_val, vmax=vint_max_val)

        # 4. --- 设置 Matplotlib 极坐标图 ---
        # 调整 figsize 适应 A4 论文布局，例如 7x7 英寸，留有充足边距
        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(projection='polar'))
        ax.set_theta_zero_location('N')  # 0度角设置在顶部

        ax.set_theta_direction(-1)  # 顺时针方向

        # 计算每个特征的角度
        angles = np.linspace(0, 2 * np.pi, n_features, endpoint=False)

        # 5. --- 收集所有连线数据并排序 (关键调整) ---
        all_lines_data = []
        for i in range(n_features):
            for j in range(i + 1, n_features):  # 确保不重复绘制且不绘制对角线
                vint_val = vint_values[i, j]

                # 如果交互值为0或极小，则不绘制
                if vint_val <= 1e-7:  # 增加一个阈值判断，避免绘制几乎不可见的线
                    continue

                # 计算宽度和颜色
                # 调整线宽范围，使其更精细，且有明显粗细对比
                width = vint_norm_width(vint_val) * 4 + 0.5  # 映射到 [0.5, 4.5]
                color = interaction_cmap(vint_norm_color(vint_val))

                all_lines_data.append({
                    'i': i, 'j': j,
                    'width': width,
                    'color': color,
                    'vint_val': vint_val  # 用于排序
                })

        # 按交互强度 (宽度) 升序排序，这样最粗的线最后绘制
        all_lines_data.sort(key=lambda x: x['width'])

    # 6. --- 绘制连线 (Vint) ---
    for line_data in all_lines_data:
        i, j = line_data['i'], line_data['j']
        ax.plot([angles[i], angles[j]], [1, 1],
                linewidth=line_data['width'],
                color=line_data['color'],
                alpha=0.95,
                zorder=1)

        # 7. --- 绘制节点背景 (柔和灰色环) ---
    ax.scatter(angles, [1] * n_features, s=vimp_sizes + 150,  # 稍微大一点作为背景环
               color=background_circle_color, alpha=1.0, zorder=2, edgecolors='black', linewidth=0.5)

    # 8. --- 绘制节点 (Vimp) ---
    ax.scatter(angles, [1] * n_features,
               s=vimp_sizes,
               c=vimp_colors,
               edgecolors='black',
               linewidth=0.8,  # 节点边框略粗，更清晰
               zorder=3)

    # 9. --- 添加特征标签 ---
    for i, (angle, label) in enumerate(zip(angles, features)):
        label_translated = label.translate(sub_map)

        rotation_deg = np.rad2deg(angle)

        # 优化标签位置和旋转
        # 在极坐标中，文本的旋转需要考虑其相对于中心点的位置
        # 标签外移到 1.18 的位置，并调整旋转以避免倒置
        text_x = 1.25  # 进一步向外移动标签
        rotation = 0

        # 2. 根据标签在圆上的位置设置水平对齐 (ha)
        #    以确保文本朝向图表外部，而不是内部
        if 0 < rotation_deg < 180:
            # 整个右半圆 (不包括顶部和底部)
            ha = 'left'
        elif 180 < rotation_deg < 360:
            # 整个左半圆 (不包括顶部和底部)
            ha = 'right'
        else:
            # 正顶部 (0 度) 或 正底部 (180 度)
            ha = 'center'
        # --- END ---

        ax.text(angle, text_x, label_translated,
                ha=ha, va='center',
                rotation=rotation,  # <--- 这里现在总是 0
                # rotation_mode='anchor', # 当 rotation=0 时, 此参数不重要
                fontsize=12,
                weight='bold',
                zorder=4)

    # 10. --- 清理坐标轴 ---
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['polar'].set_visible(False)
    ax.set_ylim(0, 1.4)  # 留出更多空间给标签
    #ax.set_title(title, fontsize=14, pad=20, weight='bold')  # 标题字号和间距调整

    # 11. --- 添加 Colorbars ---
    # Vint Colorbar
    cax_vint = fig.add_axes([0.8, 0.55, 0.02, 0.3])  # [left, bottom, width, height]
    cb_vint = plt.colorbar(cm.ScalarMappable(norm=vint_norm_color, cmap=interaction_cmap),
                           cax=cax_vint, orientation='vertical')
    cb_vint.set_label('Interaction (Vint)', size=12, weight='bold', labelpad=10)  # 标签调整
    cax_vint.tick_params(labelsize=10)  # 刻度标签字号

    # Vimp Colorbar
    cax_vimp = fig.add_axes([0.8, 0.15, 0.02, 0.3])
    cb_vimp = plt.colorbar(cm.ScalarMappable(norm=vimp_norm_color, cmap=importance_cmap),
                           cax=cax_vimp, orientation='vertical')
    cb_vimp.set_label('Importance (Vimp)', size=12, weight='bold', labelpad=10)  # 标签调整
    cax_vimp.tick_params(labelsize=10)  # 刻度标签字号

    plt.subplots_adjust(right=0.82, left=0.05, top=0.9, bottom=0.05)  # 整体图的边距调整

    # 12. --- 保存图像 ---
    if save_path:
        # 确保 out_dir 存在
        output_dir = os.path.dirname(save_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        base_filename_without_ext = os.path.splitext(save_path)[0]

        # 导出为高质量矢量图和PNG
        fig.savefig(f"{base_filename_without_ext}.pdf", dpi=300, bbox_inches='tight')  # 首选PDF
        fig.savefig(f"{base_filename_without_ext}.svg", dpi=300, bbox_inches='tight')  # 其次SVG
        fig.savefig(f"{base_filename_without_ext}.png", dpi=600, bbox_inches='tight')  # 高DPI PNG
        print(f"Vimp/Vint 环形图已保存至 {base_filename_without_ext}.png (及 .pdf/.svg)")

    plt.close(fig)


# 在无显示器环境下保存图像
plt.switch_backend("Agg")

# -----------------------------------------------------------------------------
METRIC_BO = 'auc'  # Optuna 使用的指标 auc, f1, average_precision, accuracy
METRIC = 'auc'    # RFE-CV 和最终评估使用的指标

# 后续 GAM 分析要选取的特征数量
N_GAM = 16

# 优化试验次数
N_TRIALS = 100
N_BOOTSTRAP = 200
output_num = 5
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# 设置地球化学先验：一个合理的指示元素组合应包含的最小特征数
# 如果1-SE规则选出的特征数少于此值，则强制选择包含 N_MIN_FEATURES 个特征的模型
N_MIN_FEATURES = 2
# -----------------------------------------------------------------------------

# 数据读取
#df = pd.read_excel("P+N_21-202.xlsx", engine="openpyxl")
df = pd.read_csv("P+N_45+51_final.csv")
feature_cols = [c for c in df.columns if c not in ["label"]]
X = df[feature_cols]
y = df["label"]


# -----------------------------------------------------------------------------
# 为超参数优化 (Optuna) 准备一个固定的、分组的交叉验证方案
# 这个 kf_for_optuna 仅用于 Optuna，不会在后续 Bootstrap 中使用，因此是安全的
# -----------------------------------------------------------------------------
from sklearn.model_selection import StratifiedKFold
kf_for_optuna = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# -----------------------------------------------------------------------------


# 可选标准化
#scaler = StandardScaler()
#X = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols)

# 3. 贝叶斯超参数优化 (Optuna)
out_dir = "Bootstrap_rfe_RF_SHAP_ROCAUC_45+51_1111"
os.makedirs(out_dir, exist_ok=True)


if 'Au' in df.columns:
    print("\n" + "=" * 50)
    print("开始：'Au' 元素在正负样本中的描述性统计")

    # 1. 描述性统计
    positive_au_stats = df[df['label'] == 1]['Au'].describe()
    negative_au_stats = df[df['label'] == 0]['Au'].describe()

    print("\n--- Au in Positives (label=1) ---")
    print(positive_au_stats)
    print("\n--- Au in Negatives (label=0) ---")
    print(negative_au_stats)

    # -----------------------------------------------------
    # 2. 可视化：箱线图 (线性尺度)
    # -----------------------------------------------------
    print("\n正在生成 'Au' 分布箱线图 (线性尺度)...")
    plt.figure(figsize=(6, 5))  # 使用 (6, 5) 英寸
    # 使用 patch_artist=True 以便后续设置颜色
    ax = df.boxplot(column='Au', by='label', grid=False, patch_artist=True,
                    boxprops=dict(facecolor='#A6CEE3', edgecolor='black'),
                    medianprops=dict(color='black'))

    # 清理 pandas.boxplot 自动生成的标题
    plt.suptitle('')

    # 获取正负样本数量
    n_pos = int(positive_au_stats.get('count', 0))
    n_neg = int(negative_au_stats.get('count', 0))

    plt.title(f'Au Distribution (Linear Scale)\n(n_pos={n_pos}, n_neg={n_neg})', fontsize=12)
    ax.set_xticklabels(['Label 0 (Negative)', 'Label 1 (Positive)'])
    plt.xlabel('')
    plt.ylabel('Au Value (Linear Scale)')
    plt.tight_layout(pad=0.2)

    # 保存图像 (遵循脚本的保存规范)
    base_path = os.path.join(out_dir, "Au_distribution_linear_scale")
    plt.savefig(f"{base_path}.png", dpi=300)
    plt.savefig(f"{base_path}.svg")
    plt.savefig(f"{base_path}.pdf")
    plt.close()

    # -----------------------------------------------------
    # 3. 可视化：箱线图 (对数尺度) - 这对于地球化学数据更重要
    # -----------------------------------------------------
    #    处理 'Au' 中的0值或负值 (地质数据常见)，log(0) 是未定义的
    #    使用 log1p (log(1+x)) 是处理含0数据的标准方法

    # 复制一份df用于绘图，避免"SettingWithCopyWarning"
    plot_df = df[['Au', 'label']].copy()

    # 检查是否存在0或负值
    min_au = plot_df['Au'].min()
    if min_au <= 0:
        print("信息: 'Au' 包含0或负值，将使用 log1p (log(1+x)) 进行对数尺度绘图。")
        plot_df['Au_log'] = np.log1p(plot_df['Au'])
        y_label = 'log(1 + Au Value)'
    else:
        # 如果所有值都>0，可以使用log10
        plot_df['Au_log'] = np.log10(plot_df['Au'])
        y_label = 'log10(Au Value)'

    print("正在生成 'Au' 分布箱线图 (对数尺度)...")
    plt.figure(figsize=(6, 5))  # 使用 (6, 5) 英寸
    ax_log = plot_df.boxplot(column='Au_log', by='label', grid=False, patch_artist=True,
                             boxprops=dict(facecolor='#A6CEE3', edgecolor='black'),
                             medianprops=dict(color='black'))

    plt.suptitle('')
    plt.title(f'Au Distribution (Log Scale)\n(n_pos={n_pos}, n_neg={n_neg})', fontsize=12)

    ax_log.set_xticklabels(['Label 0 (Negative)', 'Label 1 (Positive)'])
    plt.xlabel('')
    plt.ylabel(y_label)
    plt.tight_layout(pad=0.2)

    # 保存图像
    base_path = os.path.join(out_dir, "Au_distribution_log_scale")
    plt.savefig(f"{base_path}.png", dpi=300)
    plt.savefig(f"{base_path}.svg")
    plt.savefig(f"{base_path}.pdf")
    plt.close()

    print("=" * 50 + "\n")
else:
    print("\n" + "=" * 50)
    print("警告：在数据中未找到 'Au' 列，跳过 'Au' 专项分析。")
    print("=" * 50 + "\n")


def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 20, 200),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 15),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2'])
    }
    rf = RandomForestClassifier(**params, class_weight='balanced', random_state=42, n_jobs=-1)
    scores = []

    # 使用固定的交叉验证方案进行超参数优化
    for train_idx, val_idx in kf_for_optuna.split(X, y):
        X_tr_raw, X_val_raw = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # +++ 在CV循环内部进行标准化 +++
        #scaler = StandardScaler()
        #X_tr = scaler.fit_transform(X_tr_raw)
        #X_val = scaler.transform(X_val_raw)

        # 检查验证集中是否至少有一个正样本，对于F1分数计算至关重要
        if np.sum(y_val) == 0:
            print("警告: Optuna的CV折叠中, y_val不包含正样本，跳过此折叠。")
            continue

        rf.fit(X_tr_raw, y_tr)

        # <<< 评估指标逻辑 >>>
        if METRIC_BO == 'auc':
            # ROC-AUC 仍然可用，但对不均衡数据不如F1或AUC-PR敏感
            pred = rf.predict_proba(X_val_raw)[:, 1]
            scores.append(roc_auc_score(y_val, pred))
        elif METRIC_BO == 'accuracy':
            pred = rf.predict(X_val_raw)  # Accuracy 需要类别预测
            scores.append(accuracy_score(y_val, pred))
        elif METRIC_BO == 'f1':
            # F1分数需要类别预测，而不是概率
            pred = rf.predict(X_val_raw)
            scores.append(f1_score(y_val, pred))
        elif METRIC_BO == 'average_precision':
            # AUC-PR 是一个非常好的选择
            pred = rf.predict_proba(X_val_raw)[:, 1]
            scores.append(average_precision_score(y_val, pred))
        else:  # 默认回到 f1
            pred = rf.predict(X_val_raw)
            scores.append(f1_score(y_val, pred))

        # 如果所有折叠都被跳过，返回一个差值
    if not scores:
        return 0.0
    return np.mean(scores)


# -----------------------------------------------------------------------------
# 2) 运行 Optuna
# -----------------------------------------------------------------------------
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=N_TRIALS)
best_params = study.best_params
print("Best hyperparameters:", best_params)

# 保存优化过程图
plt.figure(figsize=(8, 6))
plot_optimization_history(study)
plt.xlabel('Trial Number', fontsize=14)
plt.ylabel('Objective Value', fontsize=14)
plt.title('Optimization History', fontsize=16)
plt.tight_layout(pad=0.2)

base_path = os.path.join(out_dir, "optimization_history")
plt.savefig(f"{base_path}.png", dpi=300)
plt.savefig(f"{base_path}.svg")
plt.savefig(f"{base_path}.pdf")

plt.close()

# ---

# 1. 获取所有试验的得分
# 使用列表推导式获取所有已完成试验的得分值
trial_values = [t.value for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

# 2. 检查得分值是否都相同 (即方差是否为零)
# 如果集合中元素的数量大于1，说明至少有两个不同的得分值
if len(set(trial_values)) > 1:
    print("Objective values have variance, plotting parameter importances.")
    plt.figure(figsize=(8, 6))
    plot_param_importances(study)
    plt.xlabel('Importance', fontsize=14)
    plt.ylabel('Parameter', fontsize=14)
    plt.tight_layout(pad=0.2)

    base_path = os.path.join(out_dir, "param_importance")
    plt.savefig(f"{base_path}.png", dpi=300)
    plt.savefig(f"{base_path}.svg")
    plt.savefig(f"{base_path}.pdf")

    plt.close()

else:
    # 如果所有值都相同，则打印一条信息并跳过绘图
    print("Skipping parameter importance plot because all objective values are the same.")


# -----------------------------------------------------------------------------
# 12. Bootstrap 不确定性量化
# -----------------------------------------------------------------------------
# 为 GAM 分析准备一个数据存储列表
# 它将存储每次 bootstrap 的 (训练集特征, SHAP值)
gam_data_store = []

# 记录每次 bootstrap 选出的最优特征计数
feature_selection_counts = {f: 0 for f in feature_cols}
# 记录每次 bootstrap 的全局 SHAP 平均绝对值
shap_importance_bootstrap = {f: [] for f in feature_cols}

# 记录每次 bootstrap (及内部 CV 折叠) 的全局 SHAP 交互矩阵
# 警告：这将占用大量内存和计算时间
all_vint_matrices = []


import matplotlib as mpl
# 使用提供的新参数进行更新
mpl.rcParams.update({
    # 字体
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'svg.fonttype': 'none',
    'font.family': 'sans-serif',
    'font.serif': ['Arial'],
    'font.size': 11,

    # 颜色
    'text.color': 'black',
    'axes.labelcolor': 'black',
    'xtick.color': 'black',
    'ytick.color': 'black',
    'legend.edgecolor': 'black',
    'legend.facecolor': 'white',
    # 背景
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    # 网格和坐标轴
    'axes.grid': True,
    'grid.color': '#dddddd',
    'grid.linestyle': '--',
    'grid.alpha': 0.6,
    'axes.edgecolor': 'black',
    'axes.linewidth': 0.8,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    # 分辨率
    'figure.dpi': 300,
})



# -----------------------------------------------------------------------------
# ... Bootstrap 循环开始 ...
for b in range(1, N_BOOTSTRAP + 1):
    # --------------------------------------------------------------------------
    # 步骤 1: 执行标准的 Bootstrap 抽样
    # 从整个数据集中有放回地抽取一个和原始数据集同样大小的样本
    # --------------------------------------------------------------------------
    # 使用 random_state=42+b 确保每次迭代的抽样不同但整体可复现
    df_b = df.sample(n=len(df), replace=True, random_state=42 + b)

    # 准备本次迭代的 X 和 y
    Xb = df_b[feature_cols]
    yb = df_b["label"]


    # ----------------------------------------------------------------------
    # 步骤 1.5: 为本次平衡后的 Bootstrap 样本，生成一套5折交叉验证方案
    # 使用 StratifiedKFold 确保每个折内的正负样本比例依然是 1:1
    # ----------------------------------------------------------------------
    skf_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_folds_b = list(skf_inner.split(Xb, yb))

    # --------------------------------------------------------------------------
    # 步骤 2: 在 Bootstrap 样本内部执行 RFE-CV
    # --------------------------------------------------------------------------
    feature_sets = []
    mean_scores = []
    std_scores = []
    current_features = feature_cols.copy()

    while len(current_features) >= 1:
        fold_scores = []
        fold_shap_imps = []

        # 直接使用在循环开始时生成的、干净的交叉验证折叠
        for train_idx_b, val_idx_b in cv_folds_b:

            X_tr_raw = Xb.iloc[train_idx_b][current_features]
            y_tr = yb.iloc[train_idx_b]
            X_val_raw = Xb.iloc[val_idx_b][current_features]
            y_val = yb.iloc[val_idx_b]

            # +++ START: 在CV循环内部进行标准化 +++
            #scaler = StandardScaler()
            #X_tr = pd.DataFrame(scaler.fit_transform(X_tr_raw), columns=current_features, index=X_tr_raw.index)
            #X_val = pd.DataFrame(scaler.transform(X_val_raw), columns=current_features, index=X_val_raw.index)

            # ------------------- “安全网” -------------------
            # 在计算 AUC 之前，检查 y_val 是否真的包含两个类别
            if len(np.unique(y_val)) < 2:
                print(f"警告：在第 {b} 次 Bootstrap 的 CV 折叠中，y_val 仍只包含一个类别。跳过此折叠。")
                continue  # 跳过这个有问题的折叠
            # -----------------------------------------------

            rf = RandomForestClassifier(
                **study.best_params,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
            rf.fit(X_tr_raw, y_tr)

            # <<< RFE-CV的评估指标逻辑 >>>
            if METRIC == 'auc':
                y_pred_proba = rf.predict_proba(X_val_raw)[:, 1]
                score = roc_auc_score(y_val, y_pred_proba)
            elif METRIC == 'accuracy':
                y_pred = rf.predict(X_val_raw)  # Accuracy 需要类别预测
                score = accuracy_score(y_val, y_pred)
            elif METRIC == 'f1':
                y_pred = rf.predict(X_val_raw)
                score = f1_score(y_val, y_pred)
            elif METRIC == 'average_precision':
                y_pred_proba = rf.predict_proba(X_val_raw)[:, 1]
                score = average_precision_score(y_val, y_pred_proba)
            else:  # 默认回到 f1
                y_pred = rf.predict(X_val_raw)
                score = f1_score(y_val, y_pred)
            fold_scores.append(score)

            # ... 后续 SHAP 计算代码 ...

            explainer = shap.TreeExplainer(rf)
            sv = explainer.shap_values(X_tr_raw)
            if isinstance(sv, list):
                arr = sv[1]
            elif isinstance(sv, np.ndarray):
                arr = sv
            elif hasattr(sv, 'values'):
                arr = sv.values
            else:
                raise ValueError(f"Unknown shap_values type: {type(sv)}")
            if arr.ndim == 3:
                shap_arr = arr[:, :, 1]
            elif arr.ndim == 2:
                shap_arr = arr
            else:
                raise ValueError(f"Unexpected SHAP shape: {arr.shape}")

            fold_shap_imps.append(np.abs(shap_arr).mean(axis=0))

        feature_sets.append(current_features.copy())
        mean_scores.append(np.mean(fold_scores))
        std_scores.append(np.std(fold_scores))

        mean_imp = np.mean(np.vstack(fold_shap_imps), axis=0)
        remove_idx = np.argmin(mean_imp)
        current_features.pop(remove_idx)

    # ------------------ 1-SE 规则 (One-Standard-Error Rule) ------------------
    # 1. 按特征数量对结果进行升序排序
    num_feats = np.array([len(fs) for fs in feature_sets])
    order = np.argsort(num_feats)  # 默认升序，从特征最少到最多

    num_feats_sorted = num_feats[order]
    mean_sorted = np.array(mean_scores)[order]
    std_sorted = np.array(std_scores)[order]  # <--- 别忘了把 std_scores 也排序
    feat_sets_sorted = [feature_sets[i] for i in order]

    # 引入一个可调系数 k
    k = 1.0  # 可以尝试 0.5, 1.5, 2.0, 2.5

    # 2. 找到性能最佳的点
    best_idx_in_sorted = np.argmax(mean_sorted)
    best_score = mean_sorted[best_idx_in_sorted]
    best_std = std_sorted[best_idx_in_sorted]

    # 3. 计算性能阈值 (乘以系数 k)
    threshold = best_score - k * best_std

    # 4. 从最简模型（特征最少）开始，寻找第一个性能高于阈值的模型
    #    因为 num_feats_sorted 是升序的，所以我们找到的第一个就是最简洁的
    se_1_idx = -1  # 初始化
    for i in range(len(num_feats_sorted)):
        if mean_sorted[i] >= threshold:
            se_1_idx = i
            break  # 找到后立即停止

    # 安全检查：如果因极端情况没找到，则退回到性能最佳的模型
    if se_1_idx == -1:
        se_1_idx = best_idx_in_sorted

    # 5. 选出本次 bootstrap 迭代的最优特征集
    #best_feats_b = feat_sets_sorted[se_1_idx]
    # -------------------------------------------------------------------------
    # 应用地球化学先验，选出本次迭代的最优特征集
    # -------------------------------------------------------------------------
    # 获取1-SE规则选择的特征数量
    n_features_1se = num_feats_sorted[se_1_idx]

    # 检查该数量是否小于我们设定的最小阈值
    if n_features_1se < N_MIN_FEATURES:
        # 如果是，则忽略1-SE的结果，寻找包含 N_MIN_FEATURES 个特征的模型
        # 我们寻找第一个特征数 >= N_MIN_FEATURES 的模型，以防 N_MIN_FEATURES 本身不存在
        possible_indices = np.where(num_feats_sorted >= N_MIN_FEATURES)[0]

        if len(possible_indices) > 0:
            # 选择最接近 N_MIN_FEATURES 的那个（即最简洁的那个）
            final_idx = possible_indices[0]
            print(f"Bootstrap {b}: 1-SE选择了 {n_features_1se} 个特征, 已强制修正为 {num_feats_sorted[final_idx]} 个。")
        else:
            # 极端情况：如果 N_MIN_FEATURES 比所有模型都大，则退回1-SE的选择
            final_idx = se_1_idx
    else:
        # 如果否，则1-SE规则的选择是可接受的，保持不变
        final_idx = se_1_idx

    # 根据最终确定的索引，选出最优特征集
    best_feats_b = feat_sets_sorted[final_idx]
    # -------------------------------------------------------------------------

    # 统计
    for f in best_feats_b:
        feature_selection_counts[f] += 1

    # --------------------------------------------------------------------------
    # 步骤 3: 全局 SHAP 计算与数据存储 (为 GAM 分析做准备)
    # --------------------------------------------------------------------------
    all_X_tr_all_list = []
    all_shap_vals_all = []

    # 直接共用之前生成的 cv_folds_b
    for train_idx, _ in cv_folds_b:
        # 1. 获取原始的、未标准化的训练数据
        X_tr_raw_all = Xb.iloc[train_idx][feature_cols]
        y_tr_all = yb.iloc[train_idx]

        # 2. 对其进行标准化
        #scaler_all = StandardScaler()
        #X_tr_scaled_all = scaler_all.fit_transform(X_tr_raw_all)

        # 3. 在标准化后的数据上训练模型
        rf_all = RandomForestClassifier(**study.best_params, class_weight='balanced', random_state=42, n_jobs=-1)
        rf_all.fit(X_tr_raw_all, y_tr_all)

        # 4. 在标准化后的数据上计算SHAP值
        explainer_all = shap.TreeExplainer(rf_all)
        sv_all = explainer_all.shap_values(X_tr_raw_all)

        if isinstance(sv_all, list):
            arr_all = sv_all[1]
        else:
            arr_all = sv_all

        shap_arr_all = arr_all[:, :, 1] if arr_all.ndim == 3 else arr_all

        # 5. 存储原始数据(X_tr_raw_all)和对应的SHAP值
        all_X_tr_all_list.append(X_tr_raw_all)  # <- 存储原始值用于GAM绘图
        all_shap_vals_all.append(shap_arr_all)

        # 警告：这是整个脚本中计算最密集的部分！
        print(f"    [Bootstrap {b}/{N_BOOTSTRAP}, Fold] 正在计算 Vint...")

        # 1. 计算交互值
        siv_all = explainer_all.shap_interaction_values(X_tr_raw_all)

        # 2. 提取 'Class 1' 的交互值 (与 Step 15 的逻辑相同)
        if isinstance(siv_all, list):
            # 标准情况: [class_0_inter, class_1_inter]
            siv_class1 = siv_all[1]
        elif isinstance(siv_all, np.ndarray) and siv_all.ndim == 4:
            # 4D 数组情况: (n_samples, n_features, n_features, n_classes)
            siv_class1 = siv_all[:, :, :, 1]
        else:
            # 兜底：(n_samples, n_features, n_features)
            siv_class1 = siv_all

        # 3. 计算交互矩阵的平均值 (跨样本)
        #    我们取绝对值的平均值，与 Vimp 和 Vint 绘图逻辑保持一致
        vint_matrix_b_fold = np.abs(siv_class1).mean(axis=0)

        # 4. 存入全局列表
        all_vint_matrices.append(vint_matrix_b_fold)


    # 将本次 bootstrap 的所有训练集特征和 SHAP 值合并
    X_tr_all_combined = pd.concat(all_X_tr_all_list)
    shap_comb_all = np.vstack(all_shap_vals_all)

    # 存储本次 bootstrap 的结果以备 GAM 分析使用
    gam_data_store.append({'X': X_tr_all_combined, 'shap_values': shap_comb_all})

    # 平均绝对 SHAP
    shap_mean_all = np.mean(np.abs(shap_comb_all), axis=0)
    for idx, f in enumerate(feature_cols):
        shap_importance_bootstrap[f].append(shap_mean_all[idx])


    # 每 n 次输出一次图表
    if b % output_num == 0:
        boot_dir = os.path.join(out_dir, f"bootstrap_{b}")
        os.makedirs(boot_dir, exist_ok=True)
        # 性能曲线
        fig, ax = plt.subplots(figsize=(18 / 2.54, 10 / 2.54))
        ax.tick_params(top=False, right=False)

        # 平均 CV 曲线
        ax.plot(num_feats_sorted, mean_sorted,
                marker='o', linestyle='-',
                color='#0072B2',
                label='Mean CV')

        # ±1SD 阴影
        ax.fill_between(num_feats_sorted,
                        mean_sorted - np.array(std_scores)[order],
                        mean_sorted + np.array(std_scores)[order],
                        color='#4e79a7',
                        alpha=0.5,
                        label='±1\u2009SD')

        # 1-SE 规则选择竖线
        ax.axvline(x=num_feats_sorted[se_1_idx],
                   color='green',
                   linestyle='--',
                   linewidth=1,
                   label='1-SE Selection')

        # 标签和标题
        ax.set_xlabel("Number of Features")
        ax.set_ylabel(f"CV {METRIC.capitalize()}")

        # 图例
        ax.legend(frameon=False, loc='lower right', ncol=1)

        # 保存
        fig.tight_layout(pad=0.1)

        base_path = os.path.join(boot_dir, "performance_curve_elbow")
        fig.savefig(f"{base_path}.png")
        fig.savefig(f"{base_path}.svg")
        fig.savefig(f"{base_path}.pdf")


        plt.close(fig)

        # summary_bar
        # 使用CV分割器构建用于绘图的DataFrame

        X_all_df = pd.DataFrame(
            np.vstack([Xb.iloc[train_idx][feature_cols].values for train_idx, _ in cv_folds_b]),
            columns=feature_cols
        )

        X_all_df.rename(columns=lambda x: x.translate(sub_map), inplace=True)
        cm2inch = lambda cm: cm / 2.54

        # 1) Bar chart
        # 1) Bar chart
        plt.clf()
        shap.summary_plot(
            shap_comb_all,
            X_all_df,
            plot_type="bar",
            max_display=len(feature_cols),
            show=False,
            plot_size=(cm2inch(8), cm2inch(22))
        )
        ax = plt.gca()
        ax.set_xlabel("Mean(|SHAP value|)", fontsize=11)
        plt.tight_layout(pad=0.2)

        base_path = os.path.join(boot_dir, "all_summary_bar")
        plt.savefig(f"{base_path}.png", dpi=300)
        plt.savefig(f"{base_path}.svg")
        plt.savefig(f"{base_path}.pdf")

        plt.close()

        # 2) Beeswarm
        plt.clf()
        shap.summary_plot(
            shap_comb_all,
            X_all_df,
            max_display=len(feature_cols),
            show=False,
            plot_size=(cm2inch(10), cm2inch(22))
        )
        plt.tight_layout(pad=0.2)

        base_path = os.path.join(boot_dir, "all_summary_beeswarm")
        plt.savefig(f"{base_path}.png", dpi=300)
        plt.savefig(f"{base_path}.svg")
        plt.savefig(f"{base_path}.pdf")

        plt.close()


# 13. 保存不确定性统计结果
inc_prob = {f: c / N_BOOTSTRAP for f, c in feature_selection_counts.items()}
shap_stats = {}
for f, vals in shap_importance_bootstrap.items():
    arr = np.array(vals)
    shap_stats[f] = {
        'mean': arr.mean(),
        'ci_lower': np.percentile(arr, 2.5),
        'ci_upper': np.percentile(arr, 97.5)
    }

df_inc = pd.DataFrame.from_dict(inc_prob, orient='index', columns=['inclusion_prob'])
df_shap = pd.DataFrame.from_dict(shap_stats, orient='index')

# <<< --- 新增代码：计算 Vint 平均矩阵 --- >>>
print(f"\n正在从 {len(all_vint_matrices)} 个 Vint 矩阵中计算平均交互矩阵...")
# 1. 将所有矩阵堆叠并沿 axis=0 (即所有迭代) 计算平均值
mean_vint_matrix_array = np.mean(np.array(all_vint_matrices), axis=0)

# 2. 将对角线（主效应）清零
np.fill_diagonal(mean_vint_matrix_array, 0)

# 3. 转换为 DataFrame (使用完整的 feature_cols)
mean_interaction_df = pd.DataFrame(
    mean_vint_matrix_array,
    index=feature_cols,
    columns=feature_cols
)

# 4. 保存这个重要的结果
mean_interaction_df.to_csv(os.path.join(out_dir, 'mean_shap_interaction_matrix.csv'))
print("平均 Vint 矩阵已保存。")

df_inc.to_csv(os.path.join(out_dir, 'feature_inclusion_probability.csv'))
df_shap.to_csv(os.path.join(out_dir, 'feature_shap_uncertainty.csv'))

# 可视化不确定性
# 包括：Inclusion Probability Bar, SHAP Importance CI
# cm→inch
cm2inch = lambda cm: cm/2.54

# 先排序
df_inc_sorted = df_inc.sort_values('inclusion_prob', ascending=False)

# 作图
fig, ax = plt.subplots(
    figsize=(cm2inch(18), cm2inch(12)),  # 宽16cm, 高10cm
    dpi=300
)
ax.tick_params(top=False, right=False)
# 柱状图
ax.bar(
    df_inc_sorted.index,
    df_inc_sorted['inclusion_prob'],
    edgecolor='black',    # 如果想要黑边
    linewidth=0.5         # 边框粗细可调
)

# 坐标轴、标签

# 索引['SiO₂','Al₂O₃','Na₂O',…]
labels = [lab.translate(sub_map) for lab in df_inc_sorted.index]

ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=90, ha='right')

ax.set_ylabel('Inclusion Probability')
#ax.set_title('Feature Inclusion Probability (Bootstrap)')

ax.grid(True)

# 紧凑布局
fig.tight_layout(pad=0.2)

# 保存
base_path = os.path.join(out_dir, 'inclusion_probability')
fig.savefig(f"{base_path}.png", dpi=300)
fig.savefig(f"{base_path}.svg")
fig.savefig(f"{base_path}.pdf")

plt.close(fig)


import matplotlib.pyplot as plt
import os

# cm → inch
cm2inch = lambda cm: cm / 2.54

# 假设 shap_importance_bootstrap 已经是 {feature: [bootstrap_val1, val2, ...], ...}
#        df_shap 是之前计算好的含 mean 列的 DataFrame

# 1) 按 mean 降序取特征名称列表
sorted_feats = df_shap['mean'].sort_values(ascending=False).index.tolist()

# 2) 构造箱线图数据：按 sorted_feats 的顺序取列表
box_data = [shap_importance_bootstrap[f] for f in sorted_feats]



# 画图
# 1. figure/ax
fig, ax = plt.subplots(
    figsize=(cm2inch(18), cm2inch(12)),  # 18cm × 10cm
    dpi=300
)

# 2. 定义各部分样式
boxprops = dict(
    facecolor='#A6CEE3',    # 浅蓝灰填充
    edgecolor='black',
    linewidth=0.8
)
medianprops = dict(
    color='black',        # 深蓝中位数线
    linewidth=1.5
)
whiskerprops = dict(
    color='black',
    linewidth=0.8
)
capprops = dict(
    color='black',
    linewidth=0.8
)
flierprops = dict(
    marker='o',
    markeredgecolor='black',
    markerfacecolor='black',
    markersize=4,
    alpha=0.6
)

# 3. 绘制箱线
box_data = [shap_importance_bootstrap[f] for f in sorted_feats]
bp = ax.boxplot(
    box_data,
    patch_artist=True,
    widths=0.6,
    boxprops=boxprops,
    medianprops=medianprops,
    whiskerprops=whiskerprops,
    capprops=capprops,
    flierprops=flierprops
)

# 4. 坐标轴与网格美化
# 隐藏右、顶两面坐标轴
ax.tick_params(top=False, right=False)
ax.yaxis.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
ax.xaxis.grid(False)

# 5. 下标翻译（如果还需要的话）
# sub_map = str.maketrans('0123456789', '₀₁₂₃₄₅₆₇₈₉')
labels = [lab.translate(sub_map) for lab in sorted_feats]
ax.set_xticks(range(1, len(labels)+1))
ax.set_xticklabels(labels, rotation=90, ha='right', fontsize=11)

# 6. 标签与标题
ax.set_ylabel('|SHAP value|', fontsize=12)
#ax.set_title('Feature SHAP Importance Distribution (Bootstrap)', fontsize=14, pad=12)

# 7. 紧凑布局 & 保存
fig.tight_layout(pad=0.3)

base_path = os.path.join(out_dir, 'shap_uncertainty_boxplot_sci')
fig.savefig(f"{base_path}.png", dpi=300)
fig.savefig(f"{base_path}.svg")
fig.savefig(f"{base_path}.pdf")

plt.close(fig)

# ==============================================================================
# 步骤 14: 结合 Bootstrap 不确定性的高级 GAM 分析
# ==============================================================================

# 1. 动态选取要分析的目标特征
target_features = df_inc.sort_values('inclusion_prob', ascending=False).index[:N_GAM].tolist()
print(f"\n将对以下 {N_GAM} 个最稳健的特征进行 GAM 分析: {target_features}")

# 2. 为每个目标特征，进行 GAM 拟合并可视化不确定性
for feat in target_features:
    print(f"正在处理特征: {feat} ...")

    # 2a. 准备用于绘图的统一 X 轴
    # 找到该特征在所有 bootstrap 迭代中的全局最大最小值
    min_val = min(data['X'][feat].min() for data in gam_data_store)
    max_val = max(data['X'][feat].max() for data in gam_data_store)
    XX = np.linspace(min_val, max_val, 200).reshape(-1, 1)

    # 2b. 存储每次 bootstrap 拟合出的 GAM 曲线
    all_gam_curves = []

    feat_idx = feature_cols.index(feat)

    for b_data in gam_data_store:
        X_boot = b_data['X'][feat].values.reshape(-1, 1)
        shap_boot = b_data['shap_values'][:, feat_idx]

        # 拟合 GAM
        gam = LinearGAM(s(0, n_splines=10)).fit(X_boot, shap_boot)

        # 在统一的 X 轴上预测，并存储曲线
        all_gam_curves.append(gam.predict(XX))

    # 2c. 整合所有曲线并计算不确定性
    curves_matrix = np.vstack(all_gam_curves)
    mean_curve = np.mean(curves_matrix, axis=0)
    ci_lower = np.percentile(curves_matrix, 2.5, axis=0)
    ci_upper = np.percentile(curves_matrix, 97.5, axis=0)

    # 2d. 绘图
    fig, ax = plt.subplots(figsize=(cm2inch(16), cm2inch(12)), dpi=300)

    # 绘制所有半透明的原始 GAM 曲线
    for curve in curves_matrix:
        ax.plot(XX.ravel(), curve, color='#A6CEE3', alpha=0.1, linewidth=0.8)

    # 绘制 95% 置信区间
    ax.fill_between(XX.ravel(), ci_lower, ci_upper, color='#4e79a7', alpha=0.4, label='95% CI (Bootstrap)')

    # 绘制平均曲线
    ax.plot(XX.ravel(), mean_curve, color='#E15759', linewidth=2, label='Mean GAM Fit')

    # 绘制 y=0 参考线
    ax.axhline(0, linestyle='--', color='gray', linewidth=1)

    # 美化图表
    ax.set_xlabel(f'Feature value: {feat.translate(sub_map)}', fontsize=12)
    ax.set_ylabel(f'SHAP value for {feat.translate(sub_map)}', fontsize=12)
    ax.set_title(f'GAM Partial Dependence with Bootstrap Uncertainty', fontsize=14, pad=12)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(frameon=False)
    fig.tight_layout(pad=0.3)

    # 保存图像
    base_path = os.path.join(out_dir, f'gam_uncertainty_{feat}')
    fig.savefig(f"{base_path}.png", dpi=300)
    fig.savefig(f"{base_path}.svg")
    fig.savefig(f"{base_path}.pdf")

    plt.close(fig)

    # 可选：保存拟合数据
    pd.DataFrame({
        feat: XX.ravel(),
        'mean_shap_hat': mean_curve,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper
    }).to_csv(os.path.join(out_dir, f'gam_uncertainty_data_{feat}.csv'), index=False)

print("\n高级 GAM 分析完成！")

# ==============================================================================
# 步骤 14.5: SHAP 值聚类 与 原始样本平均 SHAP 计算
# ==============================================================================
print("\n" + "=" * 50)
print("开始：执行 SHAP 值聚类 与 原始样本平均 SHAP 计算...")

# 导入聚类和不确定性量化所需的库
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import scipy.stats
import seaborn as sns
import matplotlib as mpl
import re  # 确保 re 被导入

# --- (确保辅助函数可用) ---
try:
    cm2inch
except NameError:
    cm2inch = lambda cm: cm / 2.54

try:
    sub_map
except NameError:
    sub_map = str.maketrans('0123456789', '₀₁₂₃₄₅₆₇₈₉')
# --- (结束 辅助函数) ---


# ----------------------------------------------------------------------
# 1. 聚合所有 Bootstrap 的 SHAP 值和原始 X 值 (保留索引)
# ----------------------------------------------------------------------
print("  (1/6) 正在聚合所有 Bootstrap 迭代的 SHAP 矩阵 (保留索引)...")
all_shap_matrices_list = []
all_X_df_list = []

for i, data in enumerate(gam_data_store):
    S_matrix = data['shap_values']
    X_df = data['X']  # 这个 X_df 的 .index 是 [0-95] 的重复

    # 将 SHAP 数组 (numpy) 转换回带有原始索引的 DataFrame
    S_df_with_index = pd.DataFrame(
        S_matrix,
        columns=feature_cols,
        index=X_df.index  # <- 关键：使用 X_df 保留的原始索引
    )

    all_shap_matrices_list.append(S_df_with_index)
    all_X_df_list.append(X_df)  # X_df 也保留索引

# S_all_bootstraps_df 是一个 (76800, 38) 的 DataFrame，其 .index 是 [0-95] 的重复
S_all_bootstraps_df = pd.concat(all_shap_matrices_list)
# X_all_bootstraps_df 是一个 (76800, 38) 的 DataFrame，其 .index 是 [0-95] 的重复
X_all_bootstraps_df = pd.concat(all_X_df_list)

# ----------------------------------------------------------------------
# 2. 【新功能】计算并保存 96 行的平均 SHAP 矩阵
# ----------------------------------------------------------------------
print("  (2/6) 【新功能】正在计算 96 个原始样本的平均 SHAP 值...")
# 按原始索引（level=0）分组并计算平均值
mean_shap_per_original_sample_df = S_all_bootstraps_df.groupby(level=0).mean()

print(f"      -> 平均 SHAP 矩阵形状: {mean_shap_per_original_sample_df.shape}")

# 保存这个新结果
output_path_mean_shap = os.path.join(out_dir, "mean_shap_per_original_sample.csv")
mean_shap_per_original_sample_df.to_csv(output_path_mean_shap, index=True, index_label='original_sample_index')
print(f"      -> 已保存 96 行平均 SHAP 矩阵至: {output_path_mean_shap}")


# ----------------------------------------------------------------------
# 2.5 【新功能】合并 96 行原始数据 与 96 行平均 SHAP 值
# ----------------------------------------------------------------------
print("  (2.5/6) 【新功能】正在合并 96 行原始数据与 96 行平均 SHAP 值...")

try:
    # 1. 获取原始的 X 数据 (96 行)
    # 'X' 是在脚本最开始定义的 X = df[feature_cols]
    # 它应该已经有了正确的 0-95 索引
    original_X_data = X.copy()

    # 2. 准备 SHAP 数据 (mean_shap_per_original_sample_df)
    # 我们需要重命名它的列以添加 "SHAP_" 前缀
    avg_shap_data = mean_shap_per_original_sample_df.add_prefix('SHAP_')

    # 3. 按索引合并 (axis=1)
    # 两个 DataFrame 都有 0-95 的索引，可以完美对齐
    combined_original_and_avg_shap_df = pd.concat([original_X_data, avg_shap_data], axis=1)

    # 4. 保存
    output_path_combined = os.path.join(out_dir, "original_data_with_mean_shap.csv")
    combined_original_and_avg_shap_df.to_csv(output_path_combined, index=True, index_label='original_sample_index')

    print(f"      -> 已成功保存 (96, {combined_original_and_avg_shap_df.shape[1]}) 矩阵至: {output_path_combined}")
    print("          (包含原始特征 + 平均SHAP值)")

except Exception as e:
    print(f"      -> 警告：合并 96 行原始数据与 SHAP 时出错: {e}")
    print("          请确保 'X' 变量在当前作用域中可用。")


# ----------------------------------------------------------------------
# 3. (原始流程) 准备聚类
# ----------------------------------------------------------------------
# 我们聚类仍然使用 SHAP 值，但现在从 S_all_bootstraps_df 中获取
S_all_bootstraps = S_all_bootstraps_df.values  # 转换为 numpy 数组用于聚类

print("  (3/6) 正在对 SHAP 矩阵进行标准化 (用于聚类)...")
scaler_shap = StandardScaler()
S_all_scaled = scaler_shap.fit_transform(S_all_bootstraps)

# ----------------------------------------------------------------------
# 4. (原始流程) 确定最佳聚类数 k
# ----------------------------------------------------------------------
print("  (4/6) 正在（抽样）确定最佳聚类数 k (2到8) [使用轮廓系数]...")
n_samples_for_k = min(100000, S_all_scaled.shape[0])
sample_indices = np.random.choice(S_all_scaled.shape[0], n_samples_for_k, replace=False)
S_sample_for_k = S_all_scaled[sample_indices, :]

k_range = range(2, 9)  # 尝试 2 到 8 个簇
silhouette_scores = []

for k in k_range:
    print(f"      -> 正在测试 k={k}...")
    gmm_k = GaussianMixture(n_components=k, random_state=42, n_init=5)
    gmm_k.fit(S_sample_for_k)
    labels_k = gmm_k.predict(S_sample_for_k)

    if len(np.unique(labels_k)) < 2:
        print(f"      -> 警告: k={k} 时 GMM 拟合失败，设置轮廓系数为 -1。")
        silhouette_scores.append(-1)
    else:
        score = silhouette_score(S_sample_for_k, labels_k)
        silhouette_scores.append(score)
        print(f"      -> k={k}, Silhouette Score = {score:.4f}")

# 绘制轮廓系数曲线
plt.figure(figsize=(cm2inch(12), cm2inch(8)))
plt.plot(k_range, silhouette_scores, marker='o', color='blue')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score (Higher is better)')
plt.title('GMM Silhouette Score for Optimal k')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout(pad=0.2)
k_path = os.path.join(out_dir, "shap_cluster_silhouette_curve.pdf")
plt.savefig(k_path, dpi=300)
plt.close()
print(f"轮廓系数曲线图已保存至: {k_path}")

# 自动选择最佳 k
best_k_index = np.argmax(silhouette_scores)
N_CLUSTERS = k_range[best_k_index]
print(f"      -> 最佳聚类数 k (基于 Silhouette) = {N_CLUSTERS}")

# ----------------------------------------------------------------------
# 5. (原始流程) 执行最终聚类并量化不确定性
# ----------------------------------------------------------------------
print(f"  (5/6) 正在使用 k={N_CLUSTERS} 对完整 SHAP 矩阵执行 GMM 聚类...")
gmm = GaussianMixture(n_components=N_CLUSTERS, random_state=42, n_init=3)
gmm.fit(S_all_scaled)

cluster_labels = gmm.predict(S_all_scaled)
cluster_probabilities = gmm.predict_proba(S_all_scaled)


# 计算熵
def calculate_entropy(probs):
    return scipy.stats.entropy(probs)


cluster_entropy = np.apply_along_axis(calculate_entropy, 1, cluster_probabilities)
print("      -> 簇分配不确定性 (熵) 计算完成。")

# ----------------------------------------------------------------------
# 6. (原始流程) 分析聚类结果并保存
# ----------------------------------------------------------------------
print("  (6/6) 正在分析聚类结果并确定指示元素组合...")


# S_all_bootstraps_df 已经是带 SHAP 列名的 DataFrame，但没有 "SHAP_" 前缀
# 我们需要重命名列以匹配原始逻辑
shap_col_names = [f"SHAP_{col}" for col in feature_cols]
S_df_renamed = S_all_bootstraps_df.copy()
S_df_renamed.columns = shap_col_names


# 原始代码在这里重置了索引，我们现在也必须这样做
# 以便与 X_all_bootstraps_df 进行 concat
S_df_renamed.reset_index(drop=True, inplace=True)
X_all_bootstraps_df.reset_index(drop=True, inplace=True)

# 将聚类结果添加到 S_df_renamed
S_df_renamed['cluster'] = cluster_labels
S_df_renamed['cluster_entropy'] = cluster_entropy

# 计算簇画像
cluster_profiles = S_df_renamed.groupby('cluster')[shap_col_names].mean()

# 保存画像
profile_path = os.path.join(out_dir, 'shap_cluster_profiles.csv')
cluster_profiles.to_csv(profile_path)
print(f"      -> 指示元素组合画像已保存至: {profile_path}")

# 打印统计
cluster_stats = S_df_renamed.groupby('cluster')['cluster_entropy'].agg(['count', 'mean', 'median'])
print("\n--- 簇统计信息 (大小及平均不确定性) ---")
print(cluster_stats)

# 合并并保存（现在索引已经重置，与原始逻辑一致）
full_cluster_data = pd.concat([X_all_bootstraps_df, S_df_renamed], axis=1)
full_data_path = os.path.join(out_dir, 'shap_cluster_full_data.csv')
full_cluster_data.to_csv(full_data_path, index=False)
print(f"      -> 包含簇标签和不确定性的完整数据已保存至: {full_data_path}")

# ----------------------------------------------------------------------
# 7. (原始流程) 可视化指示元素组合 (热图)
# ----------------------------------------------------------------------
print("  (7/7) 正在生成 SCI 级美化热图...")

with plt.rc_context({
    'font.family': 'Arial', 'font.size': 10, 'axes.labelsize': 11,
    'axes.titlesize': 14, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'axes.edgecolor': 'black', 'axes.linewidth': 0.8, 'xtick.direction': 'in',
    'ytick.direction': 'in', 'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
    'figure.dpi': 300, 'savefig.dpi': 300, 'pdf.fonttype': 42,
    'ps.fonttype': 42, 'svg.fonttype': 'none'
}):
    v_abs_max = np.max(np.abs(cluster_profiles.values))
    v_abs_max = np.ceil(v_abs_max * 20) / 20

    # xtick_labels (使用 re.sub 生成 LaTeX 下标)
    # cluster_profiles.columns 是 'SHAP_Au', 'SHAP_As' ...
    # 我们需要先移除 'SHAP_'
    original_cols_for_labels = [col.replace('SHAP_', '') for col in cluster_profiles.columns]
    xtick_labels = [re.sub(r'(\d+)', r'$_{\1}$', lab) for lab in original_cols_for_labels]

    ytick_labels = cluster_profiles.index.astype(int).tolist()

    fig_height_cm = (len(ytick_labels) * 1.5) + 5
    fig_width_cm = (len(xtick_labels) * 0.7) + 7

    fig, ax = plt.subplots(figsize=(cm2inch(fig_width_cm), cm2inch(fig_height_cm)))

    ax = sns.heatmap(
        cluster_profiles,
        annot=False,
        cmap="RdBu_r",
        center=0,
        vmin=-v_abs_max,
        vmax=v_abs_max,
        linewidths=.5,
        linecolor='white',
        cbar_kws={
            "label": "Mean SHAP Value (Contribution to 'Mineral Deposit')",
            "shrink": 0.7, "aspect": 30
        }
    )

    ax.set_xticklabels(xtick_labels, rotation=90, ha='center', family='Arial')
    ax.set_yticklabels(ytick_labels, rotation=0, va='center', family='Arial')
    ax.set_yticks(np.arange(len(ytick_labels)) + 0.5)

    ax.set_xlabel("Geochemical Elements", fontsize=12, labelpad=10)
    ax.set_ylabel("SHAP Pattern (Cluster ID)", fontsize=12, labelpad=10)
    ax.set_title("SHAP-Derived Geochemical Indicator Patterns", fontsize=14, pad=15)

    cbar = ax.collections[0].colorbar
    cbar.ax.set_ylabel("Mean SHAP Value (Contribution to 'Mineral Deposit')",
                       fontsize=11, labelpad=15, rotation=270)
    cbar.ax.tick_params(labelsize=10)
    sns.despine(top=True, right=True)

    heatmap_path_base = os.path.join(out_dir, "shap_cluster_profiles_heatmap_SCI")
    plt.savefig(f"{heatmap_path_base}.pdf", bbox_inches='tight')
    plt.savefig(f"{heatmap_path_base}.svg", bbox_inches='tight')
    plt.savefig(f"{heatmap_path_base}.png", dpi=600, bbox_inches='tight')

    plt.close(fig)

print(f"SCI-ready heatmap 已保存至 {heatmap_path_base}.pdf (及 .svg/.png)")
print("=" * 50 + "\n")



# ==============================================================================
# 步骤 15: 绘制 Vimp 和 Vint 环形网络图 (基于 Bootstrap 平均值)
# ==============================================================================
print("\n" + "="*50)
print(f"开始：为 Top {N_GAM} 个特征绘制 Vimp 和 Vint (基于 Bootstrap 平均值)...")


# 我们不再训练单一的最终模型，而是使用在 Step 13 中计算的
# Bootstrap 平均 Vimp (df_shap['mean']) 和
# Bootstrap 平均 Vint (mean_interaction_df)

# 1. 选取在 Bootstrap 中最稳健的 Top N 特征
# (此变量已在 步骤 14 中定义，我们直接重用)
print(f"  (1/3) 使用 Top {N_GAM} 个最稳健的特征: {target_features}")

# 2. 从 df_shap 中提取这些特征的“平均 Vimp”
# (df_shap 包含所有特征的 mean, ci_lower, ci_upper)
mean_vimp_series = df_shap.loc[target_features]['mean']
mean_vimp_series.name = "Vimp"
print("  (2/3) 已提取平均 Vimp。")

# 3. 从 mean_interaction_df 中提取这些特征的“平均 Vint”
# (mean_interaction_df 是 (n_features x n_features) 的平均矩阵)
mean_vint_subset_df = mean_interaction_df.loc[target_features, target_features]
print("  (3/3) 已提取平均 Vint。")

# 4. 调用绘图函数
print("正在绘制 Vimp 和 Vint 环形图...")
plot_feature_interaction_and_importance_optimized_cb(
    interaction_df=mean_vint_subset_df,    # <-- 使用平均 Vint (子集)
    importance_df=mean_vimp_series,       # <-- 使用平均 Vimp (子集)
    interaction_cmap=cm.Purples,
    importance_cmap=cm.Greens,
    background_circle_color='lightgray',
    save_path=os.path.join(out_dir, "Vimp_and_Vint_Ring_Plot_Bootstrap_Mean") # < L-- 建议改个新名字
)

print("Vimp 和 Vint 环形图 (基于 Bootstrap 平均值) 绘制完成！")
print("="*50 + "\n")