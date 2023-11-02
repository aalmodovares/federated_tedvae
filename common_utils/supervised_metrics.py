from typing import Optional, Callable
import numpy as np


class Metric:
    def __init__(self,) -> None:
        self.name: str
        self.callable: Callable

    def __call__(self, y_true: np.ndarray, y_pred: Optional[np.ndarray]):
        return self.callable(y_true, y_pred)

class SupervisedMetric(Metric):
    def predict_output(self, model, X):
        return model.predict(X)


class MSE(SupervisedMetric):
    def __init__(self, ) -> None:
        super().__init__()
        self.callable = self.mse
        self.name = "mse"

    def mse(self, y_true, y_pred):
        se = np.square(y_pred - y_true)
        return np.mean(se)

class RMSE(SupervisedMetric):
    def __init__(self, ) -> None:
        super().__init__()
        self.callable = self.rmse
        self.name = "rmse"

    def rmse(self, y_true, y_pred):
        se = np.square(y_pred - y_true)
        mse = np.mean(se)
        return np.sqrt(mse)


class PEHE(SupervisedMetric):
    def __init__(self, ) -> None:
        super().__init__()
        self.callable = self.pehe
        self.name = "pehe"

    @staticmethod
    def ites(y1, y0):
        return y1 - y0

    def pehe(self, ite_true, ite_pred):
        return MSE()(ite_true, ite_pred)

class RPEHE(SupervisedMetric):
    def __init__(self, ) -> None:
        super().__init__()
        self.callable = self.rpehe
        self.name = "rpehe"

    def rpehe(self, ite_true, ite_pred):
        return RMSE()(ite_true, ite_pred)

