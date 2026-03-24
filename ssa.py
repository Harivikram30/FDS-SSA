import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def apply_single_theme():
    sns.set_theme(style="whitegrid", palette="deep", context="notebook")

def print_before_after(label, before_df, after_df):
    print(f"{label} - Before:\n{before_df}")
    print(f"{label} - After:\n{after_df}")
    if before_df.equals(after_df):
        print(f"{label} - No dataset change detected.")
    else:
        print(f"{label} - Dataset changed.")

def generate_student_data():
    data = {
        "Math": np.random.randint(35, 101, 60),
        "Science": np.random.randint(35, 101, 60),
        "English": np.random.randint(35, 101, 60)
    }
    df = pd.DataFrame(data)
    df["Total"] = df[["Math", "Science", "English"]].sum(axis=1)
    df["Average"] = df[["Math", "Science", "English"]].mean(axis=1)
    df["Pass"] = (df[["Math", "Science", "English"]] >= 40).all(axis=1).astype(int)
    print(df)
    return df

def clean_data(df):
    before_df = df.copy()
    # Keep marks within valid range and drop incomplete rows.
    subject_cols = ["Math", "Science", "English"]
    after_df = df.copy()
    for col in subject_cols:
        after_df[col] = pd.to_numeric(after_df[col], errors="coerce")
        after_df.loc[(after_df[col] < 0) | (after_df[col] > 100), col] = np.nan
    after_df = after_df.dropna(subset=subject_cols)
    after_df["Total"] = after_df[subject_cols].sum(axis=1)
    after_df["Average"] = after_df[subject_cols].mean(axis=1)
    after_df["Pass"] = (after_df[subject_cols] >= 40).all(axis=1).astype(int)
    print_before_after("Data Cleaning", before_df, after_df)
    return after_df

def numpy_operations(df):
    arr = df[["Math", "Science", "English"]].values
    print("Mean:", np.mean(arr))
    print("Sum:", np.sum(arr))
    print("Transpose:\n", arr.T)

def sampling(df):
    before_df = df.copy()
    sample = df.sample(frac=0.25)
    print_before_after("Sampling", before_df, sample)
    return sample

def correlation_analysis(df):
    corr = df[["Math", "Science", "English", "Total", "Average"]].corr()
    print(corr)
    sns.heatmap(corr, annot=True)
    plt.title("Student Score Correlation")
    plt.tight_layout()
    plt.savefig("correlation_heatmap.png")
    plt.close()

def z_test(df, population_mean=70.0, std=12.0):
    sample_mean = df["Average"].mean()
    n = len(df)
    z = (sample_mean - population_mean) / (std / np.sqrt(n))
    print("Sample Mean:", round(sample_mean, 2))
    print("Z-score:", z)

def logistic_model(df):
    X = df[["Math", "Science", "English"]]
    y = df["Pass"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model = LogisticRegression()
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, pred))

def linear_model(df):
    X = df[["Math", "Science"]]
    y = df["English"]

    model = LinearRegression()
    model.fit(X, y)

    plt.scatter(df["Math"], y, label="Actual")
    plt.scatter(df["Math"], model.predict(X), alpha=0.7, label="Predicted")
    plt.title("English Score Prediction")
    plt.xlabel("Math Score")
    plt.ylabel("English Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig("linear_regression.png")
    plt.close()

def student_analysis(df):
    subject_cols = ["Math", "Science", "English"]
    print("Average Marks:\n", df[subject_cols].mean())
    print("Highest Marks:\n", df[subject_cols].max())
    print("Lowest Marks:\n", df[subject_cols].min())
    print("Overall Pass %:", round(df["Pass"].mean() * 100, 2))


def student_distribution_plot(df):
    df[["Math", "Science", "English"]].plot(kind="hist", bins=10, alpha=0.65)
    plt.title("Student Marks Distribution")
    plt.xlabel("Marks")
    plt.tight_layout()
    plt.savefig("student_distribution.png")
    plt.close()


def student_subject_trend(df):
    df[["Math", "Science", "English"]].plot()
    plt.title("Student Subject-wise Marks Trend")
    plt.xlabel("Student Index")
    plt.ylabel("Marks")
    plt.tight_layout()
    plt.savefig("student_subject_trend.png")
    plt.close()

def run_all_analyses():
    apply_single_theme()

    print("\n--- Student Data Generation ---")
    student_df = generate_student_data()

    print("\n--- Data Cleaning ---")
    student_df = clean_data(student_df)

    print("\n--- Student Correlation ---")
    correlation_analysis(student_df)

    print("\n--- NumPy Operations on Student Marks ---")
    numpy_operations(student_df)

    print("\n--- Student Sampling ---")
    sampling(student_df)

    print("\n--- Student Z-Test ---")
    z_test(student_df)

    print("\n--- Student Pass/Fail Logistic Regression ---")
    logistic_model(student_df)

    print("\n--- Student Score Linear Regression ---")
    linear_model(student_df)

    print("\n--- Student Performance Summary ---")
    student_analysis(student_df)

    print("\n--- Student Distribution Plot ---")
    student_distribution_plot(student_df)

    print("\n--- Student Subject Trend Plot ---")
    student_subject_trend(student_df)


if __name__ == "__main__":
    run_all_analyses()