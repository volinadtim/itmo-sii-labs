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

Взяли [объявления о недвижимости в Португалии](https://www.kaggle.com/datasets/luvathoms/portugal-real-estate-2024) из ЛР1–2."""
)
cells.append(t.cells[2])
md(
    """### Что будем предсказывать

Разделили квартиры на два класса: от 500 тыс. € и дешевле. Порог выбрали заранее, по test его не считали. Это условная граница дорогого сегмента для этой работы.

Для прогноза взяли те же восемь признаков, что в ЛР1–2: площадь, санузлы, год постройки, парковку, лифт, сертификат, округ и город. Цену оставили только в целевой переменной. Цену за м² и другие производные цены в признаки не добавляли.

Оставили квартиры с 01.07.2025, площадью 10–1000 м² и ценой от 10 тыс. до 20 млн €. Полные дубликаты удалили до разбиения. В прошлых работах удаляли строки с пропусками, здесь заполняем их внутри Pipeline. Поэтому выборка больше и test новый, сравнивать его с прежним разбиением напрямую нельзя.

На test отвели 20% со стратификацией по классу. Медианы, масштабирование и кодирование категорий считаются на обучающей части каждого фолда. Test в подборе C не участвует.

Так проверяем качество на похожих объявлениях. Перенос на будущий рынок отдельно не проверяли. Еще остаются возможные объявления об одной квартире: без ID объекта их нельзя надежно разделить."""
)
code("""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV,
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
)

ROOT = Path.cwd() if (Path.cwd() / "lab1").exists() else Path.cwd().parent
SEED, THRESHOLD = (67, 500000)
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
    & df["Price"].between(10000, 20000000)
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
for col in CAT:
    df[col] = df[col].astype(object).where(df[col].notna(), np.nan)
y = (df.Price >= THRESHOLD).astype(int)
train_idx, test_idx = train_test_split(
    df.index, test_size=0.2, random_state=SEED, stratify=y
)
X_train, X_test = (df.loc[train_idx, FEATURES], df.loc[test_idx, FEATURES])
y_train, y_test = (y.loc[train_idx], y.loc[test_idx])
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
""")
md(
    """### Как обучали LogisticRegression

Модель считает вероятность класса 1 через sigmoid(wᵀx+b). При обучении минимизирует log-loss с L2-регуляризацией. C задает обратную силу регуляризации: чем он меньше, тем сильнее ограничены веса.

Поставили solver=lbfgs для гладкой задачи с L2, max_iter=3000 с запасом на сходимость и tol=1e-4 для остановки. Для редкого класса использовали class_weight=balanced. Вес класса равен n/(2·n_k), поэтому ошибка на редком классе стоит дороже. В каждом фолде веса считаются заново.

Проверили C=0.01, 0.1 и 1 на трех стратифицированных фолдах train. Выбираем по balanced accuracy, чтобы оба класса влияли на оценку поровну. В Pipeline заполняем пропуски, логарифмируем площадь, масштабируем числовые признаки и кодируем категории через One-Hot. Категории с частотой меньше 30 объединяем.

На test считаем accuracy, balanced accuracy, precision, recall, F1, ROC AUC и AP. Последние две метрики считаем по score, не по готовым меткам. AP суммирует precision с весами прироста recall, это не площадь PR по трапециям. Для сравнения добавили модель, которая всегда выбирает частый класс.

У LogisticRegression оставили порог 0.5. После взвешивания классов вероятности могут быть некалиброваны, отдельно калибровку не делали."""
)
code("""
def log_area(a):
    a = np.array(a, dtype=float, copy=True)
    a[:, 0] = np.log(a[:, 0])
    return a


def preprocessor():
    numeric = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        FunctionTransformer(log_area),
        StandardScaler(),
    )
    categorical = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore", min_frequency=30),
    )
    return ColumnTransformer([("num", numeric, NUM), ("cat", categorical, CAT)])


folds = list(
    StratifiedKFold(3, shuffle=True, random_state=SEED).split(X_train, y_train)
)
MODELS, RESULTS = ({}, {})


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
    MODELS[name] = search.best_estimator_
    print(name, search.best_params_)
    return search


lr_search = fit_search(
    "LogisticRegression",
    LogisticRegression(
        solver="lbfgs",
        class_weight="balanced",
        max_iter=3000,
        tol=0.0001,
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


evaluate("Dummy", DummyClassifier(strategy="most_frequent").fit(X_train, y_train))
evaluate("LogisticRegression", lr_search.best_estimator_)
display(pd.DataFrame(RESULTS).T.round(4))
""")
md("""
**Вывод.** Дорогих квартир в train 23.3%. Получили balanced accuracy 0.894 против 0.500 у константной модели. ROC AUC 0.965. Из квартир, которые модель назвала дорогими, 72.9% действительно дорогие. При этом нашлось 88.7% всех дорогих квартир. По одной accuracy этого не видно: можно часто угадывать дешевый класс и пропускать дорогой.
""")
cells.append(t.cells[3])
md("""
### SVC с линейным ядром

Для сравнения взяли SVC с kernel=linear. Он ищет широкий зазор между классами и штрафует нарушения. C определяет, насколько сильно штрафовать ошибки относительно ширины зазора.

Признаки, разбиение и фолды оставили теми же. C выбираем из 0.01, 0.1 и 1 по balanced accuracy на train. Кэшу ядра выделили 512 МБ. Вероятности не включали: для ROC AUC и AP достаточно decision_function. Класс определяется знаком score.

SVM может учитывать дисбаланс через class_weight. При balanced получается C_k=C·n/(2·n_k): за ошибку на редком классе штраф больше. Можно также передать свои веса классов, веса объектов или подобрать порог на валидации.

Веса увеличивают штраф за ошибку редкого класса. Это может повысить recall ценой precision; частоты объектов при этом не меняются.
""")
code("""
svm_search = fit_search(
    "SVC_linear",
    SVC(
        kernel="linear",
        class_weight="balanced",
        cache_size=512,
        probability=False,
        tol=0.001,
        random_state=SEED,
    ),
)
evaluate("SVC_linear", svm_search.best_estimator_)
metrics = pd.DataFrame(RESULTS).T
display(metrics.round(4))
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, (name, m) in zip(axes, MODELS.items()):
    ConfusionMatrixDisplay.from_predictions(
        y_test, m.predict(X_test), ax=ax, colorbar=False
    )
    ax.set_title(name)
plt.tight_layout()
plt.show()
""")
md("""
**Вывод.** Для SVC выбрали C=1.0. На test получили balanced accuracy 0.893, F1 0.798 и ROC AUC 0.965. Разница balanced accuracy с LogisticRegression всего -0.0005. На этом разбиении модели работают почти одинаково. Обе строят линейную границу в преобразованных признаках, хотя оптимизируют разные функции потерь. По одному test мы бы не выбирали победителя.
""")
cells.append(t.cells[4])
from lab3_part3 import cells as part3_cells

cells.extend(part3_cells)
cells.append(t.cells[5])
md(
    """### Какие объявления считаем необычными

Для поиска аномалий взяли OneClassSVM с RBF-ядром. Он строит нелинейную область типичных сочетаний признаков. Внутри predict возвращает +1, снаружи −1. Отрицательный decision_function означает кандидата в аномалии.

Здесь используем цену, площадь, санузлы, год и парковку. Цену и площадь логарифмируем. Это отдельная задача без учителя, поэтому цена здесь допустима. В классификаторы выше она не попадает.

Город и округ не включали, чтобы редкая категория сама по себе не определяла расстояние. Но у такого решения есть ограничение: местную норму цены модель не знает. Пропуски и масштабирование считаем только по train.

Проверили nu=0.01, 0.05 и 0.10. Параметр ограничивает долю ошибок на train сверху и долю опорных векторов снизу, с учетом численной точности. На test доля аномалий не обязана совпадать с nu. Gamma=scale задает масштаб RBF по размерности и дисперсии входа.

Для подробного разбора заранее выбрали nu=0.05. Остальные значения нужны, чтобы посмотреть чувствительность. На train ищем выбросы, на test проверяем необычность новых объектов. Чистой выборки нормы у нас нет, поэтому строгим novelty detection это назвать нельзя. Разметки аномалий тоже нет, precision и recall детектора посчитать не получится."""
)
code("""
ANOM = ["Price", "LivingArea", "NumberOfBathrooms", "ConstructionYear", "Parking"]


def log_price_area(a):
    a = np.array(a, dtype=float, copy=True)
    a[:, :2] = np.log(a[:, :2])
    return a


A_train, A_test = (df.loc[train_idx, ANOM], df.loc[test_idx, ANOM])
anom_prep = Pipeline(
    [
        ("impute", SimpleImputer(strategy="median")),
        ("log", FunctionTransformer(log_price_area)),
        ("scale", StandardScaler()),
    ]
)
Z_train = anom_prep.fit_transform(A_train)
Z_test = anom_prep.transform(A_test)
anom_models, rows = ({}, [])
for nu in [0.01, 0.05, 0.1]:
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
detector = anom_models[0.05]
anomalies = df.loc[test_idx, ANOM + ["District", "City"]].copy()
anomalies["score"] = detector.decision_function(Z_test)
anomalies["outlier"] = detector.predict(Z_test) == -1
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
plt.show()
""")
md("""
**Вывод.** При nu=0.05 модель отметила 188 из 3458 объявлений на test, это 5.4%. Самый необычный объект: 1,395,000 €, 263 м², санузлов 10, год 2016, парковок 0, город Leiria. Мы бы начали проверку с этого объявления. Редкое сочетание еще не означает ошибку в данных. На графике показаны только цена и площадь, а детектор смотрит на пять признаков, поэтому часть отмеченных точек визуально выглядит обычной. По таблице с разными nu видно, насколько меняется доля кандидатов. Выбрать лучший nu без разметки не можем. Удалять эти строки из классификации не стали: дорогие квартиры как раз нужно находить. Для дальнейшей проверки нужны исходные объявления и региональные цены.
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
