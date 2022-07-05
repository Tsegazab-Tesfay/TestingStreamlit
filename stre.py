
from numpy import vectorize
from requests import head
from sklearn import tree
import streamlit as st
import pandas as pd

from sklearn import metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from annotated_text import annotated_text


header = st.container()
dataset = st.container()
features = st.container()
model_training = st.container()
annotatedText = st.container()


@st.cache  # this will avoid from rerunning - in other word it will save time
def get_data(path):
    data = pd.read_csv(path)
    return data

with header:
    st.title("Project with our dataset")
    st.text("Some definition if needed")


with dataset:
    st.title("Some definition of the dataset can be included here")
    st.text("Closer explanation about the source of the dataset....")
    # Here we need to feed the path we want to process

    df = get_data("./cleaned_data.csv")
    df = df[["clean_text", "system"]]
    st.write(df.head())

    st.subheader("The label of the dataset")
    last_column = df["system"].value_counts()
    st.bar_chart(last_column)


with features:
    st.title("Features of our dataset...")
    st.markdown("* **Some explanation about the textual feature ...:**")
    st.markdown("* **Some other explanation about the system code that will be used as label:**")

with model_training:
    st.title("Decision Tree classifier")
    st.text("Adjusting the hyperparameter of the model and see how it preforms...")

    sel_col, disp_col = st.columns(2)

    max_depth = sel_col.slider("Depth can be adjusted here", min_value=10, max_value=100, value=20, step=10)
    #n_estimator = sel_col.selectbox("How many trees should be: ", options=[100, 200, 300, "No limit"], index=0)

    sel_col.text("Here is the list of features in the given data")
    sel_col.write(df.columns)

    input_features = sel_col.text_input("The name of the feature that contain the textual data", "clean_text")
    print(df.columns)

    df = df[["clean_text", "system"]]
    X = df["clean_text"]
    y = df["system"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    vectorizer = TfidfVectorizer()
    vectorizer.fit(X_train)

    X_train = vectorizer.transform(X_train)
    X_test = vectorizer.transform(X_test)

    clf = tree.DecisionTreeClassifier(max_depth=max_depth, )
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    pred = clf.predict(X_test)

    disp_col.subheader("General score")
    disp_col.write(score)

    disp_col.subheader("Precision score")
    disp_col.write(metrics.precision_score(y_test, pred, average="micro"))

    disp_col.subheader("mean squared error")
    disp_col.write(metrics.mean_squared_error(y_test, pred))

    disp_col.subheader("r2_score")
    disp_col.write(metrics.r2_score(y_test, pred))

    disp_col.subheader("F1 score")
    disp_col.write(metrics.f1_score(y_test, pred, average="micro"))

    disp_col.subheader("Recall score")
    disp_col.write(metrics.recall_score(y_test, pred, average="micro"))


# example annotated text
with annotatedText:

    st.subheader("Checking some annotated text from streamlit")
    st.markdown("* **This can be done for single pdf file**")

    annotated_text(
        "This ",
        ("is", "verb", "#8ef"),
        " some ",
        ("annotated", "adj", "#faa"),
        ("text", "noun", "#afa"),
        " for those of ",
        ("you", "pronoun", "#fea"),
        " who ",
        ("like", "verb", "#8ef"),
        " this sort of ",
        ("thing", "noun", "#afa"),
        "."
    )