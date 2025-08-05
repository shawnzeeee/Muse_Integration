from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier

def train(X, y, classifier, test_size=0.3, random_state=42):
    """
    Splits data, trains the given classifier, and returns the F1 score.
    classifier: an instantiated sklearn classifier (e.g., SVC(), LDA(), etc.)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    score = f1_score(y_test, y_pred, average='weighted')
    return score

def train_csp(X, y, classifier, test_size=0.3, random_state=42):
    """
    Splits data, trains the given classifier, and returns the F1 score.
    classifier: an instantiated sklearn classifier (e.g., SVC(), LDA(), etc.)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    score = f1_score(y_test, y_pred, average='weighted')
    return score