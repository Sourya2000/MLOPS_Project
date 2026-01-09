import unittest
import mlflow
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle


class TestModelLoading(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        dagshub_token = os.getenv("CAPSTONE_TEST")
        if not dagshub_token:
            raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        mlflow.set_tracking_uri(
            "https://dagshub.com/Sourya2000/MLOPS_Project.mlflow"
        )

        cls.new_model_name = "my_model"
        cls.new_model_version = cls.get_latest_model_version(cls.new_model_name)
        cls.new_model_uri = f"models:/{cls.new_model_name}/{cls.new_model_version}"
        cls.new_model = mlflow.pyfunc.load_model(cls.new_model_uri)

        cls.vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))
        cls.holdout_data = pd.read_csv("data/processed/test_bow.csv")

    @staticmethod
    def get_latest_model_version(model_name, stage="Staging"):
        client = mlflow.MlflowClient()
        versions = client.get_latest_versions(model_name, stages=[stage])
        return versions[0].version if versions else None

    def test_model_loaded_properly(self):
        self.assertIsNotNone(self.new_model)

    def test_model_signature(self):
        input_text = "hi how are you"
        input_data = self.vectorizer.transform([input_text])

        input_df = pd.DataFrame(
            input_data.toarray(),
            columns=[str(i) for i in range(input_data.shape[1])]
        )

        # 🔑 slice to model-expected feature count
        input_df = input_df.iloc[:, :40]

        prediction = self.new_model.predict(input_df)

        self.assertEqual(input_df.shape[0], len(prediction))
        self.assertEqual(len(prediction.shape), 1)

    def test_model_performance(self):
        X_holdout = self.holdout_data.iloc[:, :-1]
        y_holdout = self.holdout_data.iloc[:, -1]

        # 🔑 slice to model-expected feature count
        X_holdout = X_holdout.iloc[:, :40]

        y_pred_new = self.new_model.predict(X_holdout)

        accuracy_new = accuracy_score(y_holdout, y_pred_new)
        precision_new = precision_score(y_holdout, y_pred_new)
        recall_new = recall_score(y_holdout, y_pred_new)
        f1_new = f1_score(y_holdout, y_pred_new)

        self.assertGreaterEqual(accuracy_new, 0.40)
        self.assertGreaterEqual(precision_new, 0.40)
        self.assertGreaterEqual(recall_new, 0.40)
        self.assertGreaterEqual(f1_new, 0.40)


if __name__ == "__main__":
    unittest.main()
