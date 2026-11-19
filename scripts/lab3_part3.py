"""Assignment 3: exact transcription of lecture slide 43 and manual calculations."""

import nbformat as nb

cells = []


def md(s):
    cells.append(nb.v4.new_markdown_cell(s.strip()))


def code(s):
    cells.append(nb.v4.new_code_cell(s.strip()))


md(r"""### 3.1. Четыре способа усреднить AUC

На слайде задача **multilabel**: у объектов 0 и 1 по два правильных класса. В обычной многоклассовой задаче у каждого объекта только один класс. Это различие нужно для пообъектного усреднения. Значения score на слайде не нормированы по строкам; для независимых меток сумма вероятностей не обязана равняться единице.

- **Macro:** считаем AUC отдельно для каждого столбца «этот класс против остальных», затем берем обычное среднее. Каждый класс имеет одинаковый вес.
- **Weighted:** те же AUC столбцов, но вес класса пропорционален числу его положительных меток. Частые классы влияют сильнее.
- **Micro:** разворачиваем обе матрицы в векторы и строим один ROC по всем парам «объект–класс». Это не среднее трех AUC.
- **Samples:** считаем AUC внутри каждой строки, сравнивая score правильных и неправильных меток объекта. Потом усредняем по объектам. Если в строке только нули или только единицы, AUC строки не определен. В примере таких строк нет.

Для каждого класса строим ROC по точкам, полученным при разных порогах. AUC рассчитываем вручную как сумму площадей трапеций под полученной кривой.
""")
code("""
from fractions import Fraction

Y43 = np.array([[1, 1, 0], [0, 1, 1], [0, 1, 0], [1, 0, 0]], dtype=int)
S43 = np.array([[0.7, 0.6, 0.5], [0.3, 0.4, 0.6], [0.5, 0.9, 0.2], [0.4, 0.5, 0.1]])
class_names = ["class 1", "class 2", "class 3"]
display(pd.DataFrame(Y43, columns=class_names).rename_axis("object"))
display(pd.DataFrame(S43, columns=class_names).rename_axis("object"))


def manual_pair_auc(labels, scores):
    pos = [float(s) for y, s in zip(labels, scores) if y == 1]
    neg = [float(s) for y, s in zip(labels, scores) if y == 0]
    if not pos or not neg:
        raise ValueError("AUC requires both positive and negative observations")
    wins = sum((p > n for p in pos for n in neg))
    ties = sum((p == n for p in pos for n in neg))
    return (Fraction(2 * wins + ties, 2 * len(pos) * len(neg)), wins, ties)


def manual_roc(labels, scores):
    labels, scores = (np.asarray(labels), np.asarray(scores))
    P, N = (int(sum(labels)), int(len(labels) - sum(labels)))
    if P == 0 or N == 0:
        raise ValueError("ROC requires both classes")
    rows = []
    for threshold in [float("inf")] + sorted(set(map(float, scores)), reverse=True):
        predicted = scores >= threshold
        tp = int(np.sum(predicted & (labels == 1)))
        fp = int(np.sum(predicted & (labels == 0)))
        rows.append(
            {
                "threshold": threshold,
                "TP": tp,
                "FP": fp,
                "FN": P - tp,
                "TN": N - fp,
                "FPR": Fraction(fp, N),
                "TPR": Fraction(tp, P),
            }
        )
    area = sum(
        (
            (b["FPR"] - a["FPR"]) * (a["TPR"] + b["TPR"]) / 2
            for a, b in zip(rows, rows[1:])
        ),
        Fraction(0),
    )
    return (rows, area)


def plot_segments(ax, points, label):
    for left, right in zip(points, points[1:]):
        ax.plot(
            [float(left["FPR"]), float(right["FPR"])],
            [float(left["TPR"]), float(right["TPR"])],
            color="tab:blue",
            marker="o",
            markersize=4,
        )
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=0.8)
    ax.set(
        xlim=(-0.03, 1.03),
        ylim=(-0.03, 1.03),
        xlabel="FPR = FP/N",
        ylabel="TPR = TP/P",
        title=label,
    )
    ax.grid(alpha=0.2)


class_aucs = []
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for j, name in enumerate(class_names):
    labels, scores = (Y43[:, j], S43[:, j])
    pair_auc, wins, ties = manual_pair_auc(labels, scores)
    points, area = manual_roc(labels, scores)
    class_aucs.append(area)
    print(f"{name}: {wins} побед, {ties} равенств; AUC = {pair_auc}")
    table = pd.DataFrame(points)
    display(table)
    plot_segments(axes[j], points, f"{name}: AUC={area}")
plt.tight_layout()
plt.show()
""")
md(
    r"""Площади по оцифрованным ROC:

- class 1: (1/2)·(1/2)+(1/2)·1=3/4;
- class 2: 1·(2/3)=2/3;
- class 3: 1·1=1.

Вертикальные отрезки площади не добавляют. Все точки, включая начало (0,0) и конец (1,1), выписаны в таблицах выше.

Четыре оценки считаются так:

**Macro:** (3/4+2/3+1)/3=29/36≈0.805556, на слайде 0.806.

**Weighted:** положительных меток по классам 2, 3 и 1. Получается (2·3/4+3·2/3+1·1)/6=3/4=0.75. Значения p на слайде равны 2/4, 3/4, 1/4; это распространенности классов. Их сумма 1.5, поэтому для весов нужно разделить каждое p на 1.5. Получатся 1/3, 1/2, 1/6.

**Micro:** сравниваем все 6 положительных score со всеми 6 отрицательными. В расчете ниже видны победы и равенства, AUC=5/6≈0.833333.

**Samples:** AUC строк равны 1, 1, 1 и 1/2. В последней строке правильный class 1 имеет score 0.4: выше class 3 со score 0.1, но ниже class 2 со score 0.5. Среднее (1+1+1+1/2)/4=7/8=0.875.
"""
)
code("""
supports = Y43.sum(axis=0)
macro43 = sum(class_aucs, Fraction(0)) / 3
weighted43 = sum((int(n) * a for n, a in zip(supports, class_aucs)), Fraction(0)) / int(
    supports.sum()
)
micro43, wins43, ties43 = manual_pair_auc(Y43.ravel(), S43.ravel())
micro_points, micro_area = manual_roc(Y43.ravel(), S43.ravel())
sample_aucs = [manual_pair_auc(y, s)[0] for y, s in zip(Y43, S43)]
samples43 = sum(sample_aucs, Fraction(0)) / len(Y43)
print(f"Micro: ({wins43} + {ties43}/2) / 36 = {micro43}")
print("Samples:", [str(a) for a in sample_aucs])
averages43 = pd.DataFrame(
    {
        "method": ["macro", "weighted", "micro", "samples"],
        "exact": [str(a) for a in [macro43, weighted43, micro43, samples43]],
        "auc": [float(a) for a in [macro43, weighted43, micro43, samples43]],
        "slide_rounded": [0.806, 0.75, 0.833, 0.875],
    }
)
display(averages43)
""")
md(r"""### 3.2. Три порога, precision и recall

Для class 2 со слайда 43 рассмотрим три выбранных порога: 0,30; 0,45 и 0,65. Предсказываем принадлежность классу, если оценка модели не меньше порога. Истинно положительные объекты — {0, 1, 2}, отрицательный — {3}.

| Порог | Предсказанные объекты | TP | FP | FN | TN | Accuracy | Precision | Recall |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0,30 | {0, 1, 2, 3} | 3 | 1 | 0 | 0 | 3/4 = 0,75 | 3/4 = 0,75 | 3/3 = 1 |
| 0,45 | {0, 2, 3} | 2 | 1 | 1 | 0 | 2/4 = 0,50 | 2/3 ≈ 0,67 | 2/3 ≈ 0,67 |
| 0,65 | {2} | 1 | 0 | 2 | 1 | 2/4 = 0,50 | 1/1 = 1 | 1/3 ≈ 0,33 |

При повышении порога recall снижается. Precision сначала снижается, а затем растёт до 1. Accuracy при двух последних порогах совпадает: одна ошибка FP исчезает, но появляется одна ошибка FN. Для этого класса на данных слайда получить три разных значения accuracy при определённой precision невозможно.
""")
code("""
def manual_threshold_table(labels, scores, thresholds):
    labels, scores = (np.asarray(labels), np.asarray(scores))
    rows = []
    for t in thresholds:
        pred = scores >= t
        tp = int(np.sum(pred & (labels == 1)))
        fp = int(np.sum(pred & (labels == 0)))
        fn = int(np.sum(~pred & (labels == 1)))
        tn = int(np.sum(~pred & (labels == 0)))
        if tp + fp == 0 or tp + fn == 0:
            raise ValueError("Undefined precision or recall")
        rows.append(
            {
                "threshold": t,
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "TN": tn,
                "accuracy": (tp + tn) / len(labels),
                "precision": tp / (tp + fp),
                "recall": tp / (tp + fn),
            }
        )
    return pd.DataFrame(rows)


thresholds43 = [0.3, 0.45, 0.65]
threshold_table43 = manual_threshold_table(Y43[:, 1], S43[:, 1], thresholds43)
display(threshold_table43)

""")
code("""
from matplotlib.patches import Circle

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, t in zip(axes, thresholds43):
    positive = set(np.where(Y43[:, 1] == 1)[0])
    predicted = set(np.where(S43[:, 1] >= t)[0])
    groups = {
        "TP": positive & predicted,
        "FN": positive - predicted,
        "FP": predicted - positive,
        "TN": set(range(4)) - (positive | predicted),
    }
    ax.add_patch(Circle((0.38, 0.5), 0.32, color="tab:blue", alpha=0.2))
    ax.add_patch(Circle((0.65, 0.5), 0.27, color="tab:orange", alpha=0.2))
    for key, (x, y) in {
        "FN": (0.19, 0.5),
        "TP": (0.51, 0.5),
        "FP": (0.8, 0.5),
        "TN": (0.83, 0.9),
    }.items():
        ids = ", ".join(map(str, sorted(groups[key]))) or "∅"
        ax.text(x, y, f"{key}: {ids}", ha="center", va="center", fontsize=10)
    row = threshold_table43.loc[threshold_table43.threshold == t].iloc[0]
    ax.text(0.25, 0.12, "Синие: истинные +", ha="center", fontsize=9)
    ax.text(0.75, 0.12, "Оранжевые: предсказанные +", ha="center", fontsize=9)
    ax.set_title(
        f"t={t}: precision={int(row.TP)}/{int(row.TP + row.FP)}; recall={int(row.TP)}/{int(row.TP + row.FN)}"
    )
    ax.set(xlim=(0, 1), ylim=(0, 1), aspect="equal")
    ax.axis("off")
fig.suptitle(
    "Пересечение множеств = TP. Номера объектов со слайда; площади кругов условные."
)
plt.tight_layout()
plt.show()
fig, ax = plt.subplots(figsize=(6, 4))
ordered = threshold_table43.sort_values("recall")
ax.plot(ordered.recall, ordered.precision, "o-")
for row in ordered.itertuples():
    ax.annotate(f"t={row.threshold:g}", (row.recall, row.precision), xytext=(5, -15), textcoords="offset points")
ax.set(xlabel="Recall", ylabel="Precision", title="Class 2, слайд 43", xlim=(0, 1.05), ylim=(0, 1.05))
ax.grid(alpha=0.3)
plt.suptitle("Эскизы PR по трем порогам, не полные PR-кривые")
plt.tight_layout()
plt.show()
""")
md(
    r"""Синий круг содержит реальные положительные объекты, оранжевый предсказанные. Пересечение нужно в числителе обеих метрик. Для recall делим его размер на число объектов в синем круге, для precision на число объектов в оранжевом.

При увеличении порога recall не растет: положительных предсказаний становится меньше. Precision не обязана монотонно расти. В примере сначала она падает с 3/4 до 2/3: теряется правильный объект 1, а неправильный 3 еще остается. При следующем пороге ложных срабатываний уже нет и precision равна 1. Линии на PR-графике соединяют только три рассчитанные точки, площадь по ним не выдаем за AP.

### 3.3. Что показали четыре AUC

На слайде наибольшее значение у samples: 0,875. Оно показывает качество ранжирования меток внутри каждого объекта. Универсально лучшего способа усреднения нет: выбор зависит от того, что важно оценить.

AUC при переходе к нескольким классам не обязательно падает. В этом примере усреднённые значения ниже AUC третьего класса, равного 1, потому что в других классах есть ошибки ранжирования.

### 3.4. Где какое усреднение использовать

| Подход | Пример применения | Что проверять вместе с ним |
|---|---|---|
| Macro | Классификация редких дефектов, когда каждый тип одинаково важен | AUC и размер выборки каждого класса: у редких оценка менее устойчива |
| Weighted | Поток обращений с устойчивой частотой тем, когда важно качество на типичном составе потока | Не скрывает ли частая тема плохое распознавание редкой; веса отражают частоту, не стоимость ошибки |
| Micro | Общий список пар «документ–тег», который сортируется одним score | Сопоставимость score разных тегов; частые метки могут доминировать |
| Samples | Ранжирование подходящих тегов отдельно для каждого документа или товара | В каждой строке нужны и положительные, и отрицательные метки |


"""
)
