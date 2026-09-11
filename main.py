import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def find_dataset_file(project_dir: Path = Path.cwd()) -> Path:
    """Locate the dataset CSV in the workspace, project parent folders,
    or the Downloads folder used in this environment.

    Returns the dataset path if found; otherwise raises FileNotFoundError
    with a helpful message pointing to the usual file names.
    """
    candidate_names = [
        "Student_Performance_DT - Student_Performance_DT (1).csv",
        "Student_Performance_DT.csv",
        "Student_Performance_DT - Student_Performance_DT.csv",
    ]

    # 1) Check workspace.
    for name in candidate_names:
        path = project_dir / name
        if path.exists():
            return path

    # 2) Search workspace recursively for Student_Performance CSVs.
    for csv_path in sorted(project_dir.rglob("*.csv")):
        if "Student_Performance" in csv_path.name:
            return csv_path

    # 3) Check common Downloads folder for the provided dataset file.
    downloads_dir = Path.home() / "Downloads"
    for name in candidate_names:
        path = downloads_dir / name
        if path.exists():
            return path

    # 4) Search the full parent-user profile recursively for the dataset.
    for csv_path in sorted(Path.home().rglob("*.csv")):
        if "Student_Performance" in csv_path.name:
            return csv_path

    # Fall back: if no csv is found at all, explain what happened.
    raise FileNotFoundError(
        "Could not find the dataset file. Place the CSV in the workspace "
        "folder and make sure the file name is one of: "
        + ", ".join(candidate_names)
    )


def main():
    # Load dataset
    dataset_path = find_dataset_file()
    df = pd.read_csv(dataset_path)

    print("Dataset Shape:", df.shape)
    print("\nFirst 5 rows:")
    print(df.head())

    # Encode categorical columns safely using one encoder per column.
    categorical_columns = df.select_dtypes(include="object").columns.tolist()

    for column in categorical_columns:
        encoder = LabelEncoder()
        df[column] = encoder.fit_transform(df[column].astype(str))

    # Features and target
    X = df.drop("Final_Result", axis=1)
    y = df["Final_Result"]

    # Split dataset into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Create Decision Tree model
    model = DecisionTreeClassifier(random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Test the model
    y_pred = model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)

    print("\nTraining Data Size:", len(X_train))
    print("Testing Data Size:", len(X_test))
    print("\nModel Accuracy:", accuracy)
    print("Model Accuracy Percentage:", accuracy * 100, "%")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    main()
