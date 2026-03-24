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
        "Math": np.random.randint(40, 100, 10),
        "Science": np.random.randint(40, 100, 10),
        "English": np.random.randint(40, 100, 10)
    }
    df = pd.DataFrame(data)
    print(df)
    return df

def generate_employee_data():
    data = {
        "ID": range(1, 21),
        "Salary": np.random.randint(20000, 80000, 20)
    }
    df = pd.DataFrame(data)
    print(df)
    return df

def generate_sales_data():
    data = np.random.randint(100, 500, (5, 3))
    df = pd.DataFrame(data, columns=["Product1", "Product2", "Product3"])
    print(df)
    return df

def clean_data(df):
    before_df = df.copy()
    after_df = df.replace("invalid", np.nan).dropna()
    print_before_after("Data Cleaning", before_df, after_df)
    return after_df

def numpy_operations(df):
    arr = df.values
    print("Mean:", np.mean(arr))
    print("Sum:", np.sum(arr))
    print("Transpose:\n", arr.T)

def sampling(df):
    before_df = df.copy()
    sample = df.sample(frac=0.25)
    print_before_after("Sampling", before_df, sample)

def correlation_analysis(df):
    corr = df.corr()
    print(corr)
    sns.heatmap(corr, annot=True)
    plt.title("Student Score Correlation")
    plt.tight_layout()
    plt.savefig("correlation_heatmap.png")
    plt.close()

def z_test(sample_mean=78.0, population_mean=75.0, std=10.0, n=30):
    z = (sample_mean - population_mean) / (std / np.sqrt(n))
    print("Z-score:", z)

def logistic_model():
    X = np.random.rand(20, 2)
    y = np.random.randint(0, 2, 20)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, pred))

def linear_model():
    X = np.arange(10).reshape(-1, 1)
    y = np.array([2*i + 3 for i in range(10)])
    
    model = LinearRegression()
    model.fit(X, y)
    
    plt.scatter(X, y)
    plt.plot(X, model.predict(X))
    plt.title("Linear Regression Fit")
    plt.tight_layout()
    plt.savefig("linear_regression.png")
    plt.close()

def financial_analysis():
    prices = np.random.randint(100, 200, 10)
    returns = np.diff(prices) / prices[:-1]
    print("Prices:", prices)
    print("Returns:", returns)

def student_analysis(df):
    print("Average Marks:\n", df.mean())
    print("Pass %:", (df >= 40).mean() * 100)

def sales_analysis(df):
    print("Total Revenue:", df.sum().sum())
    df.plot()
    plt.title("Sales Trend by Product")
    plt.tight_layout()
    plt.savefig("sales_analysis.png")
    plt.close()

def run_all_analyses():
    apply_single_theme()

    print("\n--- Student Data + Correlation ---")
    student_df = generate_student_data()
    correlation_analysis(student_df)

    print("\n--- Employee Sampling ---")
    employee_df = generate_employee_data()
    sampling(employee_df)

    print("\n--- Sales + NumPy Operations ---")
    sales_df = generate_sales_data()
    numpy_operations(sales_df)

    print("\n--- Data Cleaning ---")
    dirty_sales_df = generate_sales_data()
    clean_data(dirty_sales_df)

    print("\n--- Z-Test ---")
    z_test()

    print("\n--- Logistic Regression ---")
    logistic_model()

    print("\n--- Linear Regression ---")
    linear_model()

    print("\n--- Financial Analysis ---")
    financial_analysis()

    print("\n--- Student Performance ---")
    student_perf_df = generate_student_data()
    student_analysis(student_perf_df)

    print("\n--- Sales Analysis ---")
    sales_analysis_df = generate_sales_data()
    sales_analysis(sales_analysis_df)


if __name__ == "__main__":
    run_all_analyses()