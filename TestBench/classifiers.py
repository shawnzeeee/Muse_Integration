from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

def get_svm():
    return SVC(kernel='linear', probability=True)

def get_lda():
    return LDA()

def get_random_forest():
    return RandomForestClassifier(n_estimators=100, random_state=42)

def get_xgboost():
    return XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)

def get_gradient_boosting():
    return GradientBoostingClassifier(random_state=42)

def get_logistic_regression():
    return LogisticRegression(max_iter=1000, random_state=42)

def get_decision_tree():
    return DecisionTreeClassifier(random_state=42)

def get_knn():
    return KNeighborsClassifier()

# Exported classifiers
__all__ = [
    "get_svm",
    "get_lda",
    "get_random_forest",
    "get_xgboost",
    "get_gradient_boosting",
    "get_logistic_regression",]