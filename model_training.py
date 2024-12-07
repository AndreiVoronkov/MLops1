# import os
# import tempfile
from typing import Dict, List

import joblib
import numpy as np
from catboost import CatBoostClassifier
from clearml import InputModel, OutputModel, Task, TaskTypes
from sklearn.ensemble import RandomForestClassifier

from exceptions import (AlreadyExistsError, 
                        ConnectionError,
                        InvalidData, 
                        NameKeyError,
                        ParamsTypeError)

MODELS_BUCKET_NAME = "models"
CLEARML_PROJECT_NAME = "nazvanie"


class ModelFactory(object):
    
    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(ModelFactory, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        self.__available_model_types = {
            "cb": CatBoostClassifier,
            "rf": RandomForestClassifier,
        }
        self.__names_fitted_models: List[str] = []
        self.__models: Dict[Model] = []

    def reload_models(self):
        self.__names_fitted_models: List[str] = []
        if len(self.__models.keys()) != 0:
            for model in self.__models.values():
                if model.fiited:
                    self.__names_fitted_models.append(model.user_model_name)

    def get_available_model_types(self, show: bool = False):
        if show:
            return [
                {"model_name": key, "model_type": str(value)}
                for (key, value) in self.__available_model_types.items()
            ]
        else:
            return self.__available_model_types

    def get_models(
        self,
        only_fitted: bool = False,
        all_params: bool = False,
        name_models: str | None = None,
    ):
        self.reload_models()
        if name_models is not None and only_fitted:
            if name_models in self.__names_fitted_models:
                name_models = [name_models]
            else:
                raise NameKeyError(
                )
        elif name_models is not None:
            if name_models in list(self.__models.keys()):
                name_models = [name_models]
            else:
                raise NameKeyError("There is no model with this name")
        else:
            if only_fitted:
                name_models = self.__names_fitted_models
            else:
                name_models = list(self.__models.keys())
        return [
            {
                "user_model_name": user_model_name,
                "type_model": self.__models[user_model_name].type_model,
                "params": self.get_params(user_model_name, all_params),
                "fitted": self.__models[user_model_name].fiited,
            }
            for user_model_name in name_models
        ]

    def init_new_model(
        self, type_model: str, user_model_name: str, params: dict = {}
    ):
        self.reload_models()
        if user_model_name in self.__models.keys():
            raise AlreadyExistsError(
                "A model with the same name already exists"
            )
        if type_model not in self.__available_model_types.keys():
            raise NameKeyError(
                "The selected model is not in the list of available ones"
            )
        self.__models[user_model_name] = Model(
            self.__available_model_types[type_model],
            type_model,
            user_model_name,
            params=params,
        )

        return {
            "user_model_name": user_model_name,
            "type_model": type_model,
            "params": params,
            "fitted": False,
        }

    def model_fit(self, X: np.array, y: np.array, user_model_name: str):
        try:
            self.reload_models()
            self.__models[user_model_name].fit(X, y)
            self.__names_fitted_models.append(user_model_name)
        except KeyError:
            raise NameKeyError("There is no model with this name")

    def model_predict(self, X: np.array, user_model_name: str):
        self.reload_models()
        if user_model_name in self.__names_fitted_models:
            preds = self.__models[user_model_name].predict(X)
            return preds
        else:
            raise NameKeyError(
                "A model with the same name was not found or was not fitted"
            )

    def get_params(self, user_model_name: str, all: bool = False) -> dict:
        self.reload_models()
        return self.__models[user_model_name].get_params(all)

    def delete_model(self, user_model_name: str):
        try:
            self.reload_models()
            del self.__models[user_model_name]
            if user_model_name in self.__names_fitted_models:
                self.__names_fitted_models.remove(user_model_name)
        except KeyError:
            raise NameKeyError("There is no model with this name")


class Model:
    getting_params_func_names = {
        CatBoostClassifier: CatBoostClassifier.get_all_params,
        RandomForestClassifier: RandomForestClassifier.get_params,
    }

    def __init__(
        self, base_model, type_model: str, user_model_name: str, params: Dict
    ) -> None:

        self.type_model: str = type_model
        self.params: Dict = params
        self.user_model_name: str = user_model_name
        self.fiited: bool = False
        try:
            self.base_model = base_model(**self.params)
        except TypeError:
            raise ParamsTypeError("Incorrect model hyperparameters passed")

    def fit(self, X: np.array, y: np.array):

        try:
            task = Task.init(
                project_name=CLEARML_PROJECT_NAME,
                task_name=self.user_model_name,
                tags=[self.type_model, "fit"],
            )
            task.set_parameters(self.params)
            model = self.base_model.__class__(**self.params)
            model.fit(X=X, y=y)
            joblib.dump(model, f"{self.user_model_name}.pkl")
            self.fiited = True
            self.base_model = model
            task.close()
        except:
            task.close()
            raise InvalidData("Incorrect training data")

    def predict(self, X: np.array):
        try:
            task = Task.init(
                project_name=CLEARML_PROJECT_NAME,
                task_name=self.user_model_name,
                tags=[self.type_model, "predict"],
                task_type=TaskTypes.inference,
            )
            input_model = InputModel(
                project=CLEARML_PROJECT_NAME,
                name=self.user_model_name + " - " + self.user_model_name,
            )
            task.connect(input_model)
            model = self.base_model
            preds = model.predict(X)
            task.set_parameters(self.params)
            self.base_model = model
            task.close()
            return preds
        except:
            task.close()
            raise InvalidData("Incorrect data for prediction")

    def get_params(self, all: bool = False) -> dict:

        if all is True:
            return self.getting_params_func_names[type(self.base_model)](
                self.base_model
            )
        elif all is False:
            return self.params
