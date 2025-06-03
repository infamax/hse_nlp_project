import pandas as pd
import lightgbm as lgb
import spacy


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, mean_squared_log_error


path_to_dataset = "../../dataset/auto_ru_cars_price.parquet"

data = pd.read_parquet(path_to_dataset, engine="pyarrow")

nlp = spacy.load("ru_core_news_sm")
russian_stopwords = list(nlp.Defaults.stop_words)

tfidf = TfidfVectorizer(
    max_features=100,  
    stop_words=russian_stopwords,  
    ngram_range=(1, 2) 
)


tfidf_embeddings = tfidf.fit_transform(data["description"])

description_embeddings = tfidf.fit_transform(data['description'])
description_df = pd.DataFrame(description_embeddings.toarray(), columns=tfidf.get_feature_names_out())

# Объединяем все признаки
X = pd.concat([
    data[["is_available", "mileage", "year"]],
    description_df
], axis=1)
y = data["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

params = {
    "objective": "regression",
    "metric": "mse",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
}

model = lgb.train(params, train_data, valid_sets=[test_data], num_boost_round=1000, callbacks=[
        lgb.early_stopping(stopping_rounds=50),
    ]
)

y_pred = model.predict(X_test)
print(f"MAE: {mean_absolute_error(y_test, y_pred)}")
print(f"MSE: {mean_squared_error(y_test, y_pred)}")
print(f"r2_score: {r2_score(y_test, y_pred)}")
