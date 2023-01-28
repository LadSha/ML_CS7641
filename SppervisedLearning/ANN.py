import matplotlib.pyplot as plt
from sklearn import tree
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_validate, StratifiedKFold


class ANNClassifier():
    def __init__(self, model_params, feature_cols, categorical_cols=None, zero_impute_cols=None,
                 neg_impute_cols=None):
        self.feature_cols = feature_cols
        self.model_params = model_params
        self.categorical_cols = categorical_cols
        self.zero_impute_cols = zero_impute_cols
        self.neg_impute_cols = neg_impute_cols
        self.pipeline = self._create_pipeline()

    def _create_pipeline(self):
        model = self._create_model()
        preprocessing_steps = []
        if self.categorical_cols is not None:
            preprocessing_steps.append(self._create_OHE_step())
        if self.zero_impute_cols is not None:
            preprocessing_steps.append(self._create_zero_impute_step())
        if self.neg_impute_cols is not None:
            preprocessing_steps.append(self._create_neg_impute_step())
        preprocessor = self._create_column_transformer(preprocessing_steps)
        return Pipeline([("preprocessor", preprocessor), ("model", model)])

    def _create_model(self):
        return tree.DecisionTreeClassifier(
            use_label_encoder=False,
            tree_method="hist",
            **self.model_params,
        )

    def _create_OHE_step(self):
        ohe = OneHotEncoder(handle_unknown="ignore")
        return ("categorical", ohe, self.categorical_cols)

    def _create_neg_impute_step(self):
        impute = SimpleImputer(strategy="constant", fill_value=-999)
        return ("neg_impute", impute, self.neg_impute_cols)

    def _create_zero_impute_step(self):
        impute = SimpleImputer(strategy="constant", fill_value=0)
        return ("zero_impute", impute, self.zero_impute_cols)

    def _create_column_transformer(self, steps):
        return ColumnTransformer(
            transformers=steps,
            remainder="passthrough",
        )

    def predict(self, df):
        # return self.pipeline.predict_proba(df[self.feature_cols])
        pred = self.pipeline.predict(df[self.feature_cols])

        df["pred"] = pred
        return df

def _get_target_column():
    return "target"



def _get_zero_impute_columns(feature_columns):
    return [c for c in feature_columns if "mach_inv_amt" in c or "pct" in c or "cnt" in c]


def _get_neg_impute_columns(feature_columns):
    return [c for c in feature_columns if "mach_age" in c or "days_" in c]


def _split_data(df, feature_columns, target_column):
    return df[feature_columns], df[target_column]


def _get_metrics(cv_results):
    return {'mean_' + k: np.mean(v) for k, v in cv_results.items() if k not in ["fit_time", "score_time"]}


def _get_feature_importance_plot(clg_classifier, n_features=10):
    clg_classifier.pipeline["model"].get_booster().feature_names = list(
        clg_classifier.pipeline["preprocessor"].get_feature_names_out())
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    feature_importances_plot = tree.plot_importance(clg_classifier.pipeline["model"].get_booster(),
                                                   max_num_features=n_features, ax=ax).get_figure()
    feature_importances_plot.tight_layout()
    plt.show()


def train_model(df ,target_column,feature_columns,categorical_cols,zero_impute_cols,neg_impute_cols,
        model_params={"max_depth": 10, "learning_rate": .06, "scale_pos_weight": 2.8},
):

        # create custom model instance
    classifier = DTClassifier(
        model_params=model_params,
        feature_cols=feature_columns,
        categorical_cols=categorical_cols,
        zero_impute_cols=zero_impute_cols,
        neg_impute_cols=neg_impute_cols,
    )

    X, y = _split_data(df, feature_columns, target_column)

    n_folds = 5
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True)
    cv_results = cross_validate(
        estimator=classifier.pipeline,
        X=X,
        y=y,
        cv=skf,
        scoring=["accuracy", "f1", "precision", "recall", "roc_auc"],
    )
    metrics = _get_metrics(cv_results)

        # final refit on entire training set
    classifier.pipeline.fit(X, y)
        # feature importance
    _get_feature_importance_plot(classifier)
