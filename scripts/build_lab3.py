from pathlib import Path
import nbformat as nb

R = Path(__file__).resolve().parents[1]
t = nb.read(R / "materials/Lab3.ipynb", as_version=4)
cells = []


def md(s):
    cells.append(nb.v4.new_markdown_cell(s.strip()))


def code(s):
    cells.append(nb.v4.new_code_cell(s.strip()))


cells.extend(t.cells[:2])
md(
    """Данилов Тимофей и Пивоваров Роман.

Взял [объявления о недвижимости в Португалии](https://www.kaggle.com/datasets/luvathoms/portugal-real-estate-2024) из ЛР1–2."""
)
cells.append(t.cells[2])
md(
    """### Что буду предсказывать

Разделил квартиры на два класса: от 500 тыс. € и дешевле. Порог выбрал заранее, по test его не считал. Это условная граница дорогого сегмента для этой работы.

Для прогноза взял те же восемь признаков, что в ЛР1–2: площадь, санузлы, год постройки, парковку, лифт, сертификат, округ и город. Цену оставил только в целевой переменной. Цену за м² и другие производные цены в признаки не добавлял.

Оставил квартиры с 01.07.2025, площадью 10–1000 м² и ценой от 10 тыс. до 20 млн €. Полные дубликаты удалил до разбиения. В прошлых работах удалял строки с пропусками, здесь заполняю их внутри Pipeline. Поэтому выборка больше и test новый, сравнивать его с прежним разбиением напрямую нельзя.

На test отвел 20% со стратификацией по классу. Медианы, масштабирование и кодирование категорий считаются на обучающей части каждого фолда. Test в подборе C не участвует.

Так проверяю качество на похожих объявлениях. Перенос на будущий рынок отдельно не проверял. Еще остаются возможные объявления об одной квартире: без ID объекта их нельзя надежно разделить."""
)
code("""
from pathlib import Path
import hashlib
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV,
    cross_validate,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, OneClassSVM
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("error", category=ConvergenceWarning)
ROOT = Path.cwd() if (Path.cwd() / "lab1").exists() else Path.cwd().parent
OUT = ROOT / "lab3/results"
OUT.mkdir(parents=True, exist_ok=True)
SEED, THRESHOLD = 67, 500_000
NUM = ["LivingArea", "NumberOfBathrooms", "ConstructionYear", "Parking", "Elevator"]
CAT = ["EnergyCertificate", "District", "City"]
FEATURES = NUM + CAT
plt.rcParams.update({"figure.figsize": (10, 4), "figure.dpi": 110})
pd.set_option("display.max_columns", 20)
path = ROOT / "lab1/data/portugal_listinigs.csv"
raw = pd.read_csv(path, low_memory=False)
df = raw.drop_duplicates().copy()
df["PublishDate"] = pd.to_datetime(df["PublishDate"], errors="coerce")
df["Elevator"] = df["Elevator"].map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0})
df = df.loc[
    (df["Type"] == "Apartment")
    & (df["PublishDate"] >= "2025-07-01")
    & df["LivingArea"].between(10, 1000)
    & df["Price"].between(10_000, 20_000_000)
].copy()
for col, lo, hi in [
    ("NumberOfBathrooms", 0, 10),
    ("ConstructionYear", 1900, 2026),
    ("Parking", 0, 20),
]:
    df.loc[~df[col].between(lo, hi), col] = np.nan
df["EnergyCertificate"] = df["EnergyCertificate"].replace(
    {"No Certificate": "NC", "Not available": "NC"}
)
# Приводим строковые столбцы к object и заменяем пропуски на np.nan для sklearn.
for col in CAT:
    df[col] = df[col].astype(object).where(df[col].notna(), np.nan)
y = (df.Price >= THRESHOLD).astype(int)
train_idx, test_idx = train_test_split(
    df.index, test_size=0.2, random_state=SEED, stratify=y
)
X_train, X_test = df.loc[train_idx, FEATURES], df.loc[test_idx, FEATURES]
y_train, y_test = y.loc[train_idx], y.loc[test_idx]
assert (
    set(train_idx).isdisjoint(test_idx) and "Price" not in FEATURES and len(df) >= 100
)
assert set(y_train) == set(y_test) == {0, 1}
display(
    pd.DataFrame(
        {
            "n": [len(y_train), len(y_test)],
            "expensive": [y_train.sum(), y_test.sum()],
            "expensive_share": [y_train.mean(), y_test.mean()],
        },
        index=["train", "test"],
    )
)
pd.Series(
    {
        "csv_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "random_state": SEED,
        "threshold": THRESHOLD,
        "n_train": len(y_train),
        "n_test": len(y_test),
    }
).to_json(OUT / "data_summary.json", indent=2)
pd.DataFrame(
    {
        "row_index": df.index,
        "split": np.where(df.index.isin(train_idx), "train", "test"),
        "target": y.to_numpy(),
    }
).to_csv(OUT / "split.csv", index=False)
y_train.value_counts().sort_index().plot.bar(color=["steelblue", "tomato"])
plt.xticks([0, 1], ["< 500 тыс. €", "≥ 500 тыс. €"], rotation=0)
plt.title("Баланс классов: train")
plt.ylabel("Количество")
plt.show()
""")
md(
    """### Как обучал LogisticRegression

Модель считает вероятность класса 1 через sigmoid(wᵀx+b). При обучении минимизирует log-loss с L2-регуляризацией. C задает обратную силу регуляризации: чем он меньше, тем сильнее ограничены веса.

Поставил solver=lbfgs для гладкой задачи с L2, max_iter=3000 с запасом на сходимость и tol=1e-4 для остановки. Для редкого класса использовал class_weight=balanced. Вес класса равен n/(2·n_k), поэтому ошибка на редком классе стоит дороже. В каждом фолде веса считаются заново.

Проверил C=0.01, 0.1 и 1 на трех стратифицированных фолдах train. Выбираю по balanced accuracy, чтобы оба класса влияли на оценку поровну. В Pipeline заполняю пропуски, логарифмирую площадь, масштабирую числовые признаки и кодирую категории через One-Hot. Категории с частотой меньше 30 объединяю.

На test считаю accuracy, balanced accuracy, precision, recall, F1, ROC AUC и AP. Последние две метрики считаю по score, не по готовым меткам. AP суммирует precision с весами прироста recall, это не площадь PR по трапециям. Для сравнения добавил модель, которая всегда выбирает частый класс.

У LogisticRegression оставил порог 0.5. После взвешивания классов вероятности могут быть некалиброваны, отдельно калибровку не делал."""
)
code("""
def log_area(a):
    a = np.array(a, dtype=float, copy=True)
    a[:, 0] = np.log(a[:, 0])
    return a


def preprocessor():
    return ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        (
                            "impute",
                            SimpleImputer(strategy="median", add_indicator=True),
                        ),
                        ("log", FunctionTransformer(log_area)),
                        ("scale", StandardScaler()),
                    ]
                ),
                NUM,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", min_frequency=30),
                        ),
                    ]
                ),
                CAT,
            ),
        ]
    )


folds = list(
    StratifiedKFold(3, shuffle=True, random_state=SEED).split(X_train, y_train)
)
MODELS, RESULTS = {}, {}


def fit_search(name, estimator):
    search = GridSearchCV(
        Pipeline([("prep", preprocessor()), ("model", estimator)]),
        {"model__C": [0.01, 0.1, 1.0]},
        cv=folds,
        n_jobs=2,
        error_score="raise",
        scoring={
            "balanced_accuracy": "balanced_accuracy",
            "roc_auc": "roc_auc",
            "f1": "f1",
        },
        refit="balanced_accuracy",
    )
    search.fit(X_train, y_train)
    table = pd.DataFrame(search.cv_results_)[
        [
            "param_model__C",
            "mean_test_balanced_accuracy",
            "std_test_balanced_accuracy",
            "mean_test_roc_auc",
            "mean_test_f1",
            "rank_test_balanced_accuracy",
        ]
    ]
    display(table)
    table.to_csv(OUT / (name + "_cv.csv"), index=False)
    MODELS[name] = search.best_estimator_
    print(name, search.best_params_)
    return search


lr_search = fit_search(
    "LogisticRegression",
    LogisticRegression(
        solver="lbfgs",
        class_weight="balanced",
        max_iter=3000,
        tol=1e-4,
        random_state=SEED,
    ),
)


def evaluate(name, model):
    pred = model.predict(X_test)
    score = (
        model.decision_function(X_test)
        if hasattr(model, "decision_function")
        else model.predict_proba(X_test)[:, 1]
    )
    RESULTS[name] = {
        "accuracy": accuracy_score(y_test, pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, score),
        "AP": average_precision_score(y_test, score),
    }
    pd.DataFrame(
        {
            "row_index": test_idx,
            "y_true": y_test.to_numpy(),
            "prediction": pred,
            "score": score,
        }
    ).to_csv(OUT / (name + "_test.csv"), index=False)


evaluate("Dummy", DummyClassifier(strategy="most_frequent").fit(X_train, y_train))
evaluate("LogisticRegression", lr_search.best_estimator_)
display(pd.DataFrame(RESULTS).T.round(4))
""")
code("""
r = RESULTS["LogisticRegression"]
display(
    Markdown(
        f"**Вывод.** Дорогих квартир в train {y_train.mean():.1%}. "
        f"Получил balanced accuracy {r['balanced_accuracy']:.3f} против {RESULTS['Dummy']['balanced_accuracy']:.3f} "
        f"у константной модели. ROC AUC {r['roc_auc']:.3f}. "
        f"Из квартир, которые модель назвала дорогими, {r['precision']:.1%} действительно дорогие. "
        f"При этом нашлось {r['recall']:.1%} всех дорогих квартир. "
        "По одной accuracy этого не видно: можно часто угадывать дешевый класс и пропускать дорогой."
    )
)
""")
cells.append(t.cells[3])
md(
    """### SVC с линейным ядром

Для сравнения взял SVC с kernel=linear. Он ищет широкий зазор между классами и штрафует нарушения. C определяет, насколько сильно штрафовать ошибки относительно ширины зазора.

Признаки, разбиение и фолды оставил теми же. C выбираю из 0.01, 0.1 и 1 по balanced accuracy на train. Кэшу ядра выделил 512 МБ. Вероятности не включал: для ROC AUC и AP достаточно decision_function. Класс определяется знаком score.

SVM может учитывать дисбаланс через class_weight. При balanced получается C_k=C·n/(2·n_k): за ошибку на редком классе штраф больше. Можно также передать свои веса классов, веса объектов или подобрать порог на валидации.

Дополнительно сравнил balanced и None на CV при выбранном C. Здесь интересно, сколько дорогих квартир начнем находить и сколько ложных срабатываний получим взамен. Частоты объектов от весов не меняются."""
)
code("""
svm_search = fit_search(
    "SVC_linear",
    SVC(
        kernel="linear",
        class_weight="balanced",
        cache_size=512,
        probability=False,
        tol=1e-3,
        random_state=SEED,
    ),
)
rows = []
for weights in [None, "balanced"]:
    m = clone(svm_search.best_estimator_).set_params(model__class_weight=weights)
    scores = cross_validate(
        m,
        X_train,
        y_train,
        cv=folds,
        n_jobs=2,
        error_score="raise",
        scoring={
            "balanced_accuracy": "balanced_accuracy",
            "precision": "precision",
            "recall": "recall",
        },
    )
    rows.append(
        {
            "class_weight": str(weights),
            **{k: v.mean() for k, v in scores.items() if k.startswith("test_")},
        }
    )
weight_table = pd.DataFrame(rows)
display(weight_table)
weight_table.to_csv(OUT / "svm_class_weight_cv.csv", index=False)
evaluate("SVC_linear", svm_search.best_estimator_)
metrics = pd.DataFrame(RESULTS).T
metrics.to_csv(OUT / "classification_metrics.csv")
display(metrics.round(4))
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, (name, m) in zip(axes, MODELS.items()):
    ConfusionMatrixDisplay.from_predictions(
        y_test, m.predict(X_test), ax=ax, colorbar=False
    )
    ax.set_title(name)
plt.tight_layout()
plt.savefig(OUT / "confusion_matrices.png")
plt.show()
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for name, m in MODELS.items():
    score = m.decision_function(X_test)
    RocCurveDisplay.from_predictions(y_test, score, name=name, ax=axes[0])
    PrecisionRecallDisplay.from_predictions(y_test, score, name=name, ax=axes[1])
axes[0].plot([0, 1], [0, 1], "--", color="gray")
axes[1].axhline(
    y_test.mean(), ls="--", color="gray", label="Доля положительного класса"
)
axes[1].legend()
plt.tight_layout()
plt.savefig(OUT / "roc_pr.png")
plt.show()
""")
code("""
r = RESULTS["SVC_linear"]
delta = r["balanced_accuracy"] - RESULTS["LogisticRegression"]["balanced_accuracy"]
display(
    Markdown(
        f"**Вывод.** Для SVC выбрал C={svm_search.best_params_['model__C']}. "
        f"На test получил balanced accuracy {r['balanced_accuracy']:.3f}, F1 {r['f1']:.3f} "
        f"и ROC AUC {r['roc_auc']:.3f}. Разница balanced accuracy с LogisticRegression всего {delta:+.4f}. "
        "На этом разбиении модели работают почти одинаково. Обе строят линейную границу в преобразованных признаках, "
        "хотя оптимизируют разные функции потерь. По одному test я бы не выбирал победителя. "
        "В таблице CV видно, как веса меняют precision и recall."
    )
)
""")
cells.append(t.cells[4])
from lab3_part3 import cells as part3_cells

cells.extend(part3_cells)
cells.append(t.cells[5])
md(
    """### Какие объявления считаю необычными

Для поиска аномалий взял OneClassSVM с RBF-ядром. Он строит нелинейную область типичных сочетаний признаков. Внутри predict возвращает +1, снаружи −1. Отрицательный decision_function означает кандидата в аномалии.

Здесь использую цену, площадь, санузлы, год и парковку. Цену и площадь логарифмирую. Это отдельная задача без учителя, поэтому цена здесь допустима. В классификаторы выше она не попадает.

Город и округ не включал, чтобы редкая категория сама по себе не определяла расстояние. Но у такого решения есть ограничение: местную норму цены модель не знает. Пропуски и масштабирование считаю только по train.

Проверил nu=0.01, 0.05 и 0.10. Параметр ограничивает долю ошибок на train сверху и долю опорных векторов снизу, с учетом численной точности. На test доля аномалий не обязана совпадать с nu. Gamma=scale задает масштаб RBF по размерности и дисперсии входа.

Для подробного разбора заранее выбрал nu=0.05. Остальные значения нужны, чтобы посмотреть чувствительность. На train ищу выбросы, на test проверяю необычность новых объектов. Чистой выборки нормы у меня нет, поэтому строгим novelty detection это назвать нельзя. Разметки аномалий тоже нет, precision и recall детектора посчитать не получится."""
)
code("""
ANOM = ["Price", "LivingArea", "NumberOfBathrooms", "ConstructionYear", "Parking"]


def log_price_area(a):
    a = np.array(a, dtype=float, copy=True)
    a[:, :2] = np.log(a[:, :2])
    return a


A_train, A_test = df.loc[train_idx, ANOM], df.loc[test_idx, ANOM]
anom_prep = Pipeline(
    [
        ("impute", SimpleImputer(strategy="median")),
        ("log", FunctionTransformer(log_price_area)),
        ("scale", StandardScaler()),
    ]
)
Z_train = anom_prep.fit_transform(A_train)
Z_test = anom_prep.transform(A_test)
anom_models, rows = {}, []
for nu in [0.01, 0.05, 0.10]:
    detector = OneClassSVM(kernel="rbf", gamma="scale", nu=nu, cache_size=512).fit(
        Z_train
    )
    anom_models[nu] = detector
    rows.append(
        {
            "nu": nu,
            "train_outlier_share": (detector.predict(Z_train) == -1).mean(),
            "test_outlier_share": (detector.predict(Z_test) == -1).mean(),
            "support_share": len(detector.support_) / len(Z_train),
        }
    )
sensitivity = pd.DataFrame(rows)
display(sensitivity)
sensitivity.to_csv(OUT / "oneclass_sensitivity.csv", index=False)
detector = anom_models[0.05]
anomalies = df.loc[test_idx, ANOM + ["District", "City"]].copy()
anomalies["score"] = detector.decision_function(Z_test)
anomalies["outlier"] = detector.predict(Z_test) == -1
anomalies.to_csv(OUT / "oneclass_test.csv", index_label="row_index")
display(anomalies.nsmallest(10, "score"))
display(anomalies.groupby("outlier")[ANOM].median())
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for flag, color, label in [
    (False, "steelblue", "Типичные"),
    (True, "tomato", "Кандидаты в аномалии"),
]:
    sub = anomalies[anomalies.outlier == flag]
    axes[0].scatter(sub.LivingArea, sub.Price, s=9, alpha=0.45, c=color, label=label)
axes[0].set(
    xscale="log",
    yscale="log",
    xlabel="Площадь, м²",
    ylabel="Цена, €",
    title="Проекция 5-мерного детектора",
)
axes[0].legend()
axes[1].hist(anomalies.score, bins=60)
axes[1].axvline(0, color="red", ls="--")
axes[1].set(xlabel="decision_function", ylabel="Количество", title="Test, nu=0.05")
plt.tight_layout()
plt.savefig(OUT / "oneclass.png")
plt.show()
""")
code("""
worst = anomalies.nsmallest(1, "score").iloc[0]
display(
    Markdown(
        f"**Вывод.** При nu=0.05 модель отметила {anomalies.outlier.sum()} из {len(anomalies)} "
        f"объявлений на test, это {anomalies.outlier.mean():.1%}. Самый необычный объект: "
        f"{worst.Price:,.0f} €, {worst.LivingArea:g} м², санузлов {worst.NumberOfBathrooms:g}, "
        f"год {worst.ConstructionYear:g}, парковок {worst.Parking:g}, город {worst.City}. "
        "Я бы начал проверку с этого объявления. Редкое сочетание еще не означает ошибку в данных. "
        "На графике показаны только цена и площадь, а детектор смотрит на пять признаков, "
        "поэтому часть отмеченных точек визуально выглядит обычной. "
        "По таблице с разными nu видно, насколько меняется доля кандидатов. Выбрать лучший nu без разметки не могу. "
        "Удалять эти строки из классификации не стал: дорогие квартиры как раз нужно находить. "
        "Для дальнейшей проверки нужны исходные объявления и региональные цены."
    )
)
assert all(np.isfinite(list(r.values())).all() for r in RESULTS.values())
assert np.isfinite(anomalies.score).all() and len(anomalies) == len(y_test)
print(
    "CHECKS PASSED: disjoint split, no price in classifier features, finite metrics and anomaly scores."
)
""")
md(
    """### Источники

- [Датасет Kaggle](https://www.kaggle.com/datasets/luvathoms/portugal-real-estate-2024).
- [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).
- [SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html).
- [Поиск выбросов и новизны](https://scikit-learn.org/stable/modules/outlier_detection.html)."""
)
n = nb.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        }
    },
)
nb.write(n, R / "lab3/Lab3_Danilov_Pivovarov.ipynb")
print("Built", len(cells), "cells")
