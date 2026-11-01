from pathlib import Path
import csv
import tkinter as tk
from tkinter import ttk

import numpy as np

from scipy.optimize import linear_sum_assignment

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import rand_score, mutual_info_score

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg



K = 3
MAX_ITER = 300

TEST_SIZES = (0.20, 0.30, 0.40)

SEEDS = (0, 1, 2, 3, 4)

SPLIT_SEED = 42

OUTPUT_DIR = Path(__file__).resolve().parent / "results"

COLORS = ("#2878b5", "#e07b23", "#3b9950")



def squared_distances(X, centers):
    return np.sum(
        (X[:, None, :] - centers[None, :, :]) ** 2,
        axis=2,
    )


def predict_clusters(X, centers):
    distances = squared_distances(X, centers)
    return np.argmin(distances, axis=1)


def fit_kmeans(X, k=3, seed=0, max_iter=300):
    X = np.asarray(X, dtype=float)

    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("X должен быть непустой двумерной матрицей.")

    if not np.isfinite(X).all():
        raise ValueError("В данных есть NaN или бесконечные значения.")

    if not isinstance(k, (int, np.integer)) or not 1 <= k <= len(X):
        raise ValueError("Некорректное количество кластеров.")

    if (
        not isinstance(max_iter, (int, np.integer))
        or max_iter < 1
    ):
        raise ValueError("max_iter должен быть положительным целым.")

    unique_X = np.unique(X, axis=0)

    if len(unique_X) < k:
        raise ValueError("Различных объектов меньше, чем кластеров.")

    rng = np.random.default_rng(seed)

    initial_indices = rng.choice(
        len(unique_X),
        size=k,
        replace=False,
    )

    centers = unique_X[initial_indices].copy()
    labels = predict_clusters(X, centers)

    def objective(current_centers, current_labels):
        return float(
            np.sum((X - current_centers[current_labels]) ** 2)
        )

    history = [objective(centers, labels)]
    converged = False

    for iteration in range(1, max_iter + 1):
        new_centers = np.empty_like(centers)
        nonempty_ids = []
        empty_ids = []

        for cluster_id in range(k):
            members = X[labels == cluster_id]

            if len(members) > 0:
                new_centers[cluster_id] = members.mean(axis=0)
                nonempty_ids.append(cluster_id)
            else:
                empty_ids.append(cluster_id)

        if empty_ids:
            nearest = squared_distances(
                X,
                new_centers[nonempty_ids],
            ).min(axis=1)

            for cluster_id in empty_ids:
                farthest_index = int(np.argmax(nearest))
                new_centers[cluster_id] = X[farthest_index]

                distance_to_added = np.sum(
                    (X - new_centers[cluster_id]) ** 2,
                    axis=1,
                )

                nearest = np.minimum(
                    nearest,
                    distance_to_added,
                )

        new_labels = predict_clusters(X, new_centers)

        unchanged = np.array_equal(labels, new_labels)

        centers = new_centers
        labels = new_labels

        history.append(objective(centers, labels))

        if unchanged and not empty_ids:
            converged = True
            break

    return {
        "centers": centers,
        "labels": labels,
        "inertia": history[-1],
        "history": history,
        "iterations": iteration,
        "converged": converged,
    }



def find_cluster_mapping(y_train, train_clusters, k):
    table = np.zeros((k, k), dtype=int)

    np.add.at(
        table,
        (y_train, train_clusters),
        1,
    )

    class_ids, cluster_ids = linear_sum_assignment(-table)

    mapping = np.empty(k, dtype=int)
    mapping[cluster_ids] = class_ids

    return mapping


def calculate_metrics(y_true, clusters, mapping):
    predicted_classes = mapping[clusters]

    correct = int(
        np.count_nonzero(predicted_classes == y_true)
    )

    wrong = len(y_true) - correct
    accuracy = 100.0 * correct / len(y_true)


    ri = float(rand_score(y_true, clusters))
    mi = float(mutual_info_score(y_true, clusters))

    return {
        "correct": correct,
        "wrong": wrong,
        "accuracy": accuracy,
        "ri": ri,
        "mi": mi,
        "predicted_classes": predicted_classes,
    }



def run_experiments():
    dataset = load_wine()

    X = dataset.data
    y = dataset.target

    print("=" * 75)
    print("КЛАСТЕРИЗАЦИЯ WINE")
    print(f"Объектов: {X.shape[0]}")
    print(f"Признаков: {X.shape[1]}")
    print(f"Количество кластеров: {K}")
    print("=" * 75)

    records = []
    details = []

    for test_size in TEST_SIZES:
        train_ids, test_ids = train_test_split(
            np.arange(len(X)),
            test_size=test_size,
            random_state=SPLIT_SEED,
            stratify=y,
        )


        scaler = StandardScaler()

        X_train = scaler.fit_transform(X[train_ids])
        X_test = scaler.transform(X[test_ids])

        y_train = y[train_ids]
        y_test = y[test_ids]

        pca = PCA(n_components=2, svd_solver="full")

        train_2d = pca.fit_transform(X_train)
        test_2d = pca.transform(X_test)

        train_percent = int(round(100 * (1 - test_size)))
        test_percent = int(round(100 * test_size))

        split_name = f"{train_percent}/{test_percent}"

        for seed in SEEDS:
            result = fit_kmeans(
                X_train,
                k=K,
                seed=seed,
                max_iter=MAX_ITER,
            )

            train_clusters = result["labels"]

            test_clusters = predict_clusters(
                X_test,
                result["centers"],
            )

            mapping = find_cluster_mapping(
                y_train,
                train_clusters,
                K,
            )

            train_metrics = calculate_metrics(
                y_train,
                train_clusters,
                mapping,
            )

            test_metrics = calculate_metrics(
                y_test,
                test_clusters,
                mapping,
            )

            record = {
                "split": split_name,
                "seed": seed,
                "train_n": len(y_train),
                "test_n": len(y_test),
                "correct": test_metrics["correct"],
                "wrong": test_metrics["wrong"],
                "train_accuracy_pct": train_metrics["accuracy"],
                "test_accuracy_pct": test_metrics["accuracy"],
                "train_RI": train_metrics["ri"],
                "train_MI": train_metrics["mi"],
                "test_RI": test_metrics["ri"],
                "test_MI": test_metrics["mi"],
                "iterations": result["iterations"],
                "inertia": result["inertia"],
                "converged": result["converged"],
            }

            records.append(record)

            details.append({
                "result": result,
                "train_2d": train_2d,
                "test_2d": test_2d,
                "centers_2d": pca.transform(result["centers"]),
                "train_clusters": train_clusters,
                "test_clusters": test_clusters,
                "test_ids": test_ids,
                "y_test": y_test,
                "test_predictions": test_metrics["predicted_classes"],
                "explained_variance": (
                    pca.explained_variance_ratio_.sum()
                ),
            })

            print()
            print(f"Обучение/тест: {split_name}; seed={seed}")
            print(
                f"Обучающих объектов: {len(y_train)}; "
                f"тестовых объектов: {len(y_test)}"
            )
            print(
                f"Правильно: {test_metrics['correct']}; "
                f"неправильно: {test_metrics['wrong']}"
            )
            print(
                f"Точность на тесте: "
                f"{test_metrics['accuracy']:.2f}%"
            )

            print("ОБУЧАЮЩАЯ ВЫБОРКА:")
            print(
                f"  Rand Index (RI): "
                f"{train_metrics['ri']:.6f}"
            )
            print(
                f"  Mutual Information (MI): "
                f"{train_metrics['mi']:.6f}"
            )

            print("ТЕСТОВАЯ ВЫБОРКА:")
            print(
                f"  Rand Index (RI): "
                f"{test_metrics['ri']:.6f}"
            )
            print(
                f"  Mutual Information (MI): "
                f"{test_metrics['mi']:.6f}"
            )

            print(f"Итераций: {result['iterations']}")
            print(
                "Сходимость: "
                + ("достигнута" if result["converged"] else "не достигнута")
            )
            print("-" * 75)

    return records, details



def write_csv(path, rows):
    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
            delimiter=";",
        )

        writer.writeheader()
        writer.writerows(rows)


def draw_accuracy(ax, records):
    for split in dict.fromkeys(r["split"] for r in records):
        group = [
            r for r in records
            if r["split"] == split
        ]

        ax.plot(
            [r["seed"] for r in group],
            [r["test_accuracy_pct"] for r in group],
            marker="o",
            label=f"Обучение/тест {split}",
        )

    ax.set_title("Точность на тестовой выборке")
    ax.set_xlabel("Seed начальных центров")
    ax.set_ylabel("Точность, %")
    ax.set_xticks(list(SEEDS))
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.3)
    ax.legend()


def save_results(records, details):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    write_csv(
        OUTPUT_DIR / "experiments.csv",
        records,
    )

    summary = []

    for split in dict.fromkeys(r["split"] for r in records):
        group = [
            r for r in records
            if r["split"] == split
        ]

        summary.append({
            "split": split,
            "runs": len(group),
            "mean_test_accuracy_pct": float(np.mean([
                r["test_accuracy_pct"] for r in group
            ])),
            "mean_train_RI": float(np.mean([
                r["train_RI"] for r in group
            ])),
            "mean_train_MI": float(np.mean([
                r["train_MI"] for r in group
            ])),
            "mean_test_RI": float(np.mean([
                r["test_RI"] for r in group
            ])),
            "mean_test_MI": float(np.mean([
                r["test_MI"] for r in group
            ])),
        })

    write_csv(
        OUTPUT_DIR / "summary.csv",
        summary,
    )

    prediction_rows = []
    history_rows = []

    for record, detail in zip(records, details):
        for object_id, true, cluster, predicted in zip(
            detail["test_ids"],
            detail["y_test"],
            detail["test_clusters"],
            detail["test_predictions"],
        ):
            prediction_rows.append({
                "split": record["split"],
                "seed": record["seed"],
                "object_id": int(object_id),
                "true_class": int(true),
                "cluster": int(cluster),
                "predicted_class": int(predicted),
                "correct": int(true == predicted),
            })

        for iteration, inertia in enumerate(
            detail["result"]["history"]
        ):
            history_rows.append({
                "split": record["split"],
                "seed": record["seed"],
                "iteration": iteration,
                "inertia": inertia,
            })

    write_csv(
        OUTPUT_DIR / "test_predictions.csv",
        prediction_rows,
    )

    write_csv(
        OUTPUT_DIR / "history.csv",
        history_rows,
    )

    figure = Figure(figsize=(9, 5))
    ax = figure.add_subplot(111)

    draw_accuracy(ax, records)

    figure.tight_layout()
    figure.savefig(
        OUTPUT_DIR / "accuracy.png",
        dpi=200,
    )

    print()
    print(f"Результаты сохранены в папку:\n{OUTPUT_DIR}")
    print()
    print("Файл experiments.csv содержит:")
    print("  train_RI — Rand Index на обучении")
    print("  train_MI — Mutual Information на обучении")
    print("  test_RI  — Rand Index на тесте")
    print("  test_MI  — Mutual Information на тесте")



class KMeansApp(tk.Tk):
    def __init__(self, records, details):
        super().__init__()

        self.records = records
        self.details = details

        self.title("Wine — K-means, Rand Index и Mutual Information")
        self.geometry("1280x900")

        ttk.Label(
            self,
            text=(
                "Результаты кластеризации Wine. "
                "Выберите строку для просмотра метрик."
            ),
            font=("", 12),
        ).pack(pady=8)

        table_frame = ttk.Frame(self)
        table_frame.pack(fill="x", padx=10)

        columns = (
            "split",
            "seed",
            "train",
            "test",
            "correct",
            "wrong",
            "accuracy",
            "train_ri",
            "train_mi",
            "test_ri",
            "test_mi",
        )

        headings = (
            "Обуч./тест",
            "Seed",
            "Обучение",
            "Тест",
            "Верно",
            "Ошибки",
            "Точность, %",
            "RI обуч.",
            "MI обуч.",
            "RI тест",
            "MI тест",
        )

        self.table = ttk.Treeview(
            table_frame,
            columns=columns,
            show="headings",
            height=15,
        )

        for column, heading in zip(columns, headings):
            self.table.heading(column, text=heading)
            self.table.column(
                column,
                width=100,
                anchor="center",
            )

        for index, record in enumerate(records):
            self.table.insert(
                "",
                "end",
                iid=str(index),
                values=(
                    record["split"],
                    record["seed"],
                    record["train_n"],
                    record["test_n"],
                    record["correct"],
                    record["wrong"],
                    f"{record['test_accuracy_pct']:.2f}",
                    f"{record['train_RI']:.6f}",
                    f"{record['train_MI']:.6f}",
                    f"{record['test_RI']:.6f}",
                    f"{record['test_MI']:.6f}",
                ),
            )

        scrollbar = ttk.Scrollbar(
            table_frame,
            orient="horizontal",
            command=self.table.xview,
        )

        self.table.configure(
            xscrollcommand=scrollbar.set,
        )

        self.table.pack(fill="x")
        scrollbar.pack(fill="x")

        self.table.bind(
            "<<TreeviewSelect>>",
            self.select_run,
        )

        metrics_frame = ttk.LabelFrame(
            self,
            text="Rand Index и Mutual Information выбранного запуска",
        )
        metrics_frame.pack(
            fill="x",
            padx=10,
            pady=10,
        )

        self.metrics_label = ttk.Label(
            metrics_frame,
            text="",
            font=("", 12),
            justify="left",
        )
        self.metrics_label.pack(
            anchor="w",
            padx=10,
            pady=8,
        )

        self.info_label = ttk.Label(
            self,
            text="",
            wraplength=1200,
        )
        self.info_label.pack(
            fill="x",
            padx=12,
            pady=4,
        )

        notebook = ttk.Notebook(self)
        notebook.pack(
            fill="both",
            expand=True,
            padx=10,
            pady=5,
        )

        clusters_tab = ttk.Frame(notebook)
        convergence_tab = ttk.Frame(notebook)
        accuracy_tab = ttk.Frame(notebook)

        notebook.add(
            clusters_tab,
            text="Кластеры",
        )
        notebook.add(
            convergence_tab,
            text="Сходимость",
        )
        notebook.add(
            accuracy_tab,
            text="Точность",
        )

        self.cluster_figure = Figure(figsize=(11, 4))
        self.cluster_axes = self.cluster_figure.subplots(1, 2)

        self.cluster_canvas = FigureCanvasTkAgg(
            self.cluster_figure,
            master=clusters_tab,
        )
        self.cluster_canvas.get_tk_widget().pack(
            fill="both",
            expand=True,
        )

        self.convergence_figure = Figure(figsize=(10, 4))
        self.convergence_ax = (
            self.convergence_figure.add_subplot(111)
        )

        self.convergence_canvas = FigureCanvasTkAgg(
            self.convergence_figure,
            master=convergence_tab,
        )
        self.convergence_canvas.get_tk_widget().pack(
            fill="both",
            expand=True,
        )

        accuracy_figure = Figure(figsize=(10, 4))
        accuracy_ax = accuracy_figure.add_subplot(111)

        draw_accuracy(accuracy_ax, records)
        accuracy_figure.tight_layout()

        self.accuracy_canvas = FigureCanvasTkAgg(
            accuracy_figure,
            master=accuracy_tab,
        )
        self.accuracy_canvas.get_tk_widget().pack(
            fill="both",
            expand=True,
        )
        self.accuracy_canvas.draw()

        ttk.Label(
            self,
            text=f"CSV-файлы и график сохранены в: {OUTPUT_DIR}",
            wraplength=1200,
        ).pack(
            fill="x",
            padx=12,
            pady=6,
        )

        self.table.selection_set("0")
        self.select_run()

    def select_run(self, event=None):
        selection = self.table.selection()

        if not selection:
            return

        index = int(selection[0])

        record = self.records[index]
        detail = self.details[index]

        self.metrics_label.config(
            text=(
                f"Обучающая выборка:   "
                f"Rand Index = {record['train_RI']:.6f}     "
                f"Mutual Information = {record['train_MI']:.6f}\n\n"
                f"Тестовая выборка:       "
                f"Rand Index = {record['test_RI']:.6f}     "
                f"Mutual Information = {record['test_MI']:.6f}"
            )
        )

        convergence_text = (
            "достигнута"
            if record["converged"]
            else "не достигнута за заданное число итераций"
        )

        explained = 100 * detail["explained_variance"]

        self.info_label.config(
            text=(
                f"Разбиение: {record['split']}; "
                f"seed={record['seed']}; "
                f"точность на тесте: "
                f"{record['test_accuracy_pct']:.2f}%; "
                f"итераций: {record['iterations']}; "
                f"сходимость: {convergence_text}. "
                f"PCA сохраняет {explained:.1f}% дисперсии. "
                f"Кластеризация выполнена по всем 13 признакам."
            )
        )

        self.draw_clusters(detail)
        self.draw_convergence(record["split"])

    def draw_clusters(self, detail):
        datasets = (
            (
                "Обучающая выборка",
                detail["train_2d"],
                detail["train_clusters"],
            ),
            (
                "Тестовая выборка",
                detail["test_2d"],
                detail["test_clusters"],
            ),
        )

        for ax, (title, points, clusters) in zip(
            self.cluster_axes,
            datasets,
        ):
            ax.clear()

            for cluster_id in range(K):
                mask = clusters == cluster_id

                ax.scatter(
                    points[mask, 0],
                    points[mask, 1],
                    color=COLORS[cluster_id],
                    s=30,
                    alpha=0.8,
                    label=f"Кластер {cluster_id}",
                )

            centers = detail["centers_2d"]

            ax.scatter(
                centers[:, 0],
                centers[:, 1],
                marker="X",
                color="black",
                s=150,
                label="Центры",
            )

            ax.set_title(title)
            ax.set_xlabel("Главная компонента 1")
            ax.set_ylabel("Главная компонента 2")
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8)

        self.cluster_figure.tight_layout()
        self.cluster_canvas.draw()

    def draw_convergence(self, split):
        ax = self.convergence_ax
        ax.clear()

        for record, detail in zip(
            self.records,
            self.details,
        ):
            if record["split"] != split:
                continue

            history = detail["result"]["history"]

            ax.plot(
                range(len(history)),
                history,
                marker=".",
                label=f"seed={record['seed']}",
            )

        ax.set_title(
            f"Сходимость k-means: обучение/тест {split}"
        )
        ax.set_xlabel("Итерация; 0 — начальное состояние")
        ax.set_ylabel("Сумма квадратов расстояний")
        ax.grid(alpha=0.3)
        ax.legend()

        self.convergence_figure.tight_layout()
        self.convergence_canvas.draw()



def main():
    records, details = run_experiments()

    save_results(records, details)

    app = KMeansApp(records, details)
    app.mainloop()


if __name__ == "__main__":
    main()