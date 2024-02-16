import optuna
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
# from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
# from lightgbm import LGBMClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
# from catboost import CatBoostClassifier
import sklearn
from tqdm import tqdm
from numba import jit
from sklearn.model_selection import StratifiedKFold
from optuna.integration import OptunaSearchCV
import pandas as pd
from sklearn.metrics import get_scorer
from concurrent.futures import ThreadPoolExecutor, as_completed
import concurrent.futures
import logging
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from sklearn.decomposition import IncrementalPCA
import numpy as np



def transormations_apply(img_path,img_h=224, img_w=224, norm_means=np.array([0.77148203, 0.55764165, 0.58345652]), norm_std=np.array([0.12655577, 0.14245141, 0.15189891])):
    """
    Apply a series of transformations to an image loaded from a given path and return the transformed image as a NumPy array.

    The transformations include resizing, random horizontal and vertical flips, random rotation, color jitter (for brightness, contrast, and hue adjustments), normalization, and conversion to a NumPy array.

    Parameters:
    - img_path (str): The path to the image file to be transformed.
    - img_h (int, optional): The height to which the images should be resized. Defaults to 224.
    - img_w (int, optional): The width to which the images should be resized. Defaults to 224.
    - norm_means (np.array, optional): The means for each channel (RGB) used for normalization. Defaults to np.array([0.77148203, 0.55764165, 0.58345652]).
    - norm_std (np.array, optional): The standard deviations for each channel (RGB) used for normalization. Defaults to np.array([0.12655577, 0.14245141, 0.15189891]).

    Returns:
    - numpy.ndarray: The transformed image as a NumPy array.
    """
    transform = transforms.Compose([
        transforms.Resize((img_h, img_w)), # we applied resize for memory reasons
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(norm_means, norm_std),
        transforms.Lambda(lambda x: x.numpy())  # convert the tensor to a NumPy array
    ])

    image = Image.open(img_path).convert('RGB')  

    transformed_image = transform(image)

    return transformed_image

def process_image(img_path,img_h=224, img_w=224, norm_means=np.array([0.77148203, 0.55764165, 0.58345652]), norm_std=np.array([0.12655577, 0.14245141, 0.15189891])):
    transformed_image = transormations_apply(img_path)
    # Transpose the image to H x W x C
    transformed_image = transformed_image.transpose(1, 2, 0)
    red, green, blue = transformed_image[:,:,0], transformed_image[:,:,1], transformed_image[:,:,2]
    return red, green, blue


def process_batch_pca(batch_paths=None,numof_components=None):
    """
    Processes a batch of image paths by applying transformations and Incremental PCA to each image.
    
    This function uses concurrent execution to apply image processing and dimensionality reduction
    on a batch of images specified by their file paths. It performs Incremental PCA separately on
    the red, green, and blue channels of each image and then compresses the image by flattening the
    transformed channels back into a single array.

    Parameters:
    - batch_paths (List[str], optional): List of paths to the image files to be processed. This is required
                                         when `partial_fit` is False.
    - numof_components (int, optional): The number of principal components to keep during the PCA process.
                                        If not specified, PCA is not applied.

    Returns:
    - List[numpy.ndarray]: A list of numpy arrays, where each array represents a compressed and flattened
                           version of the processed images from the batch.
    """      
    processed_images = []
        
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_image = {executor.submit(process_image, img_path): img_path for img_path in batch_paths}
        for future in as_completed(future_to_image):
            red, green, blue = future.result()
            ipca = IncrementalPCA(n_components=numof_components)
            red_transformed = ipca.fit_transform(red)
            green_transformed = ipca.fit_transform(green)
            blue_transformed = ipca.fit_transform(blue)
            img_compressed = np.dstack((red_transformed, green_transformed, blue_transformed)).flatten()
            processed_images.append(img_compressed)
        return processed_images
    
def process_batch_partial_fit(X,y, batch_size):
    """
    Processes batches of images for partial fitting by applying specified transformations.

    This function iterates over the provided image paths and labels in batches, applying
    a predefined transformation to each image. It is designed to support incremental learning
    or partial fitting scenarios by yielding batches of transformed images and their corresponding
    labels. This allows models that support partial fitting to be trained on large datasets
    without requiring all data to be loaded into memory at once.

    Parameters:
    - X (List[str]): List of paths to the images. Each path should be a string pointing to an image file.
    - y (List): List of labels corresponding to each image in X. The length of y should match the length of X.
    - batch_size (int): The number of images to process in each batch. Determines how many images and labels
                        are yielded at a time.

    Yields:
    - Tuple[numpy.ndarray, numpy.ndarray]: Each yield returns a tuple containing two numpy arrays:
      - The first array contains the transformed images from the batch. The transformations applied are
        specified by the `transormations_apply` function.
      - The second array contains the labels corresponding to each image in the batch.
    """
    n_batches = len(X) // batch_size + (1 if len(X) % batch_size else 0)

    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        batch_images = X[start_idx:end_idx]
        batch_labels = y[start_idx:end_idx]

        images = []
        labels = []

        for img_path, label in zip(batch_images, batch_labels):
            transformed_image = transormations_apply(img_path)
            images.append(transformed_image)
            labels.append(label)

        yield np.array(images), np.array(labels)


optuna_grid = {
        "RandomForestClassifier": {
            "n_estimators": optuna.distributions.IntDistribution(2, 200),
            "criterion": optuna.distributions.CategoricalDistribution(
                ["gini", "entropy"]
            ),
            "max_depth": optuna.distributions.IntDistribution(1, 50),
            "min_samples_leaf": optuna.distributions.IntDistribution(1, 10),
            "min_samples_split": optuna.distributions.IntDistribution(2, 10),
            "bootstrap": optuna.distributions.CategoricalDistribution([True, False]),
            "n_jobs": optuna.distributions.CategoricalDistribution([-1])
        },
        "KNeighborsClassifier": {
            "n_neighbors": optuna.distributions.IntDistribution(2, 15),
            "weights": optuna.distributions.CategoricalDistribution(
                ["uniform", "distance"]
            ),
            "algorithm": optuna.distributions.CategoricalDistribution(
                ["auto", "ball_tree", "kd_tree", "brute"]
            ),
            "p": optuna.distributions.IntDistribution(1, 2),
            "leaf_size": optuna.distributions.IntDistribution(5, 50),
            "n_jobs": optuna.distributions.CategoricalDistribution([-1])
        },
        "DecisionTreeClassifier": {
            "criterion": optuna.distributions.CategoricalDistribution(
                ["gini", "entropy"]
            ),
            "splitter": optuna.distributions.CategoricalDistribution(
                ["best", "random"]
            ),
            "max_depth": optuna.distributions.IntDistribution(1, 100),
            "min_samples_split": optuna.distributions.IntDistribution(2, 10),
            "min_weight_fraction_leaf": optuna.distributions.IntDistribution(0.0, 0.5),
        },
        "SVC": {
            "C": optuna.distributions.IntDistribution(1, 10),
            "kernel": optuna.distributions.CategoricalDistribution(
                ["linear", "rbf", "sigmoid", "poly"]
            ),
            "degree": optuna.distributions.IntDistribution(1, 10),
            "probability": optuna.distributions.CategoricalDistribution([True, False]),
            "shrinking": optuna.distributions.CategoricalDistribution([True, False]),
            "decision_function_shape": optuna.distributions.CategoricalDistribution(
                ["ovo", "ovr"]
            ),
        },
        "GradientBoostingClassifier": {
            "loss": optuna.distributions.CategoricalDistribution(
                ["log_loss", "exponential"]
            ),
            "learning_rate": optuna.distributions.FloatDistribution(0.001, 0.5),
            "n_estimators": optuna.distributions.IntDistribution(2, 200),
            "criterion": optuna.distributions.CategoricalDistribution(
                ["friedman_mse", "squared_error"]
            ),
            "max_depth": optuna.distributions.IntDistribution(1, 50),
            "min_samples_split": optuna.distributions.IntDistribution(2, 10),
            "min_samples_leaf": optuna.distributions.IntDistribution(1, 10),
        },
        "LinearDiscriminantAnalysis": {
            "solver": optuna.distributions.CategoricalDistribution(["lsqr", "eigen"]),
            "shrinkage": optuna.distributions.FloatDistribution(0.0, 1.0),
            "tol": optuna.distributions.CategoricalDistribution([1e-3, 1e-4, 1e-5]),
            "store_covariance": optuna.distributions.CategoricalDistribution(
                [True, False]
            ),
        },
        "LogisticRegression": {
            "penalty": optuna.distributions.CategoricalDistribution(
                [ "l2",None]
            ),
            "C": optuna.distributions.FloatDistribution(0.1, 10.0),
            "solver": optuna.distributions.CategoricalDistribution(
                ["newton-cg", "lbfgs", "sag", "saga", "newton-cholesky", "liblinear"]
            ),
            "max_iter": optuna.distributions.IntDistribution(100, 1000),
            "n_jobs": optuna.distributions.CategoricalDistribution([-1])
        },
        "GaussianNB": {
            "var_smoothing": optuna.distributions.FloatDistribution(1e-9, 1e-5)
        },
        'GaussianProcessClassifier':{
            'optimizer': optuna.distributions.CategoricalDistribution(['fmin_l_bfgs_b', None]),
            'max_iter_predict': optuna.distributions.IntDistribution(50, 200),
            'warm_start': optuna.distributions.CategoricalDistribution([True, False]),
            'n_jobs': optuna.distributions.CategoricalDistribution([-1])
        },
        "SGDClassifier": {
            "loss": optuna.distributions.CategoricalDistribution(["hinge", "modified_huber", "perceptron"]),
            "penalty": optuna.distributions.CategoricalDistribution([ "l2","l1",None, "elasticnet"]),
            # "alpha": optuna.distributions.FloatDistribution(1e-4, 1e-1),
            "tol": optuna.distributions.FloatDistribution(1e-4, 1e-1),
            "n_jobs": optuna.distributions.CategoricalDistribution([-1]),
            "verbose": optuna.distributions.CategoricalDistribution([0]),
            "learning_rate": optuna.distributions.CategoricalDistribution([ "optimal"]),
            "early_stopping": optuna.distributions.CategoricalDistribution([True]),
            "class_weight": optuna.distributions.CategoricalDistribution(["balanced",None]),
        },
        "MultinomialNB": {
            "alpha": optuna.distributions.FloatDistribution(0.0, 1.0),
            "fit_prior": optuna.distributions.CategoricalDistribution([True, False])
        },
        "Perceptron": {
            "penalty": optuna.distributions.CategoricalDistribution([ "l1", "elasticnet"]),
            "alpha": optuna.distributions.FloatDistribution(1e-4, 1e-1),
            "tol": optuna.distributions.FloatDistribution(1e-4, 1e-1),
            "n_jobs": optuna.distributions.CategoricalDistribution([-1]),
            "verbose": optuna.distributions.CategoricalDistribution([0]),
            "early_stopping": optuna.distributions.CategoricalDistribution([True]),
            "class_weight": optuna.distributions.CategoricalDistribution(["balanced",None]),
        },
        "PassiveAggressiveClassifier": {
            "C": optuna.distributions.FloatDistribution(1e-4, 2),
            "tol": optuna.distributions.FloatDistribution(1e-4, 1e-1),
            "early_stopping": optuna.distributions.CategoricalDistribution([True]),
            "n_jobs": optuna.distributions.CategoricalDistribution([-1]),
            "verbose": optuna.distributions.CategoricalDistribution([0]),
            "class_weight": optuna.distributions.CategoricalDistribution(["balanced",None]),
        },
        "MLPClassifier": {
            "hidden_layer_sizes": optuna.distributions.CategoricalDistribution([
                (64,128,256,512),(50, 100), (16,64,128,514),(32,64,128,256)]),
            "activation": optuna.distributions.CategoricalDistribution(["identity", "tanh", "relu"]),
            "solver": optuna.distributions.CategoricalDistribution([ "adam"]),
            "alpha": optuna.distributions.FloatDistribution(1e-4, 1e-1),
            "learning_rate": optuna.distributions.CategoricalDistribution(["constant", "invscaling", "adaptive"]),
            "learning_rate_init": optuna.distributions.CategoricalDistribution([1e-4, 1e-3]),
            "verbose": optuna.distributions.CategoricalDistribution([False]),
            "early_stopping": optuna.distributions.CategoricalDistribution([True])
        }
    }

# optuna.logging.set_verbosity(optuna.logging.ERROR)     

def hyperparameter_tuning(X_train, y_train, X_test, y_test, splits=5, n_trials=100, verbose=0, scoring='accuracy', estimators=[]):
    """
    Perform hyperparameter tuning on a list of estimators using OptunaSearchCV.

    This function iterates over a list of machine learning estimators, performs hyperparameter
    optimization using OptunaSearchCV, and evaluates the best model on a test dataset. It returns
    a DataFrame containing the best score and parameters for each estimator.

    Parameters:
    - X_train: numpy.ndarray or pandas.DataFrame
        The training input samples.
    - y_train: numpy.ndarray or pandas.Series
        The target values (class labels) as integers or strings.
    - X_test: numpy.ndarray or pandas.DataFrame
        The testing input samples.
    - y_test: numpy.ndarray or pandas.Series
        The target values (class labels) for the test set.
    - splits: int, optional (default=5)
        The number of folds for cross-validation.
    - n_trials: int, optional (default=100)
        The number of trials for hyperparameter optimization.
    - verbose: int, optional (default=0)
        The verbosity level of the function. Higher numbers give more detailed output.
    - scoring: str or callable, optional (default='accuracy')
        A str (see model evaluation documentation) or a scorer callable object / function
        with signature scorer(estimator, X, y).
    - estimators: list of estimator instances
        The list of estimators for which hyperparameter tuning is to be performed.

    Returns:
    - pandas.DataFrame
        A DataFrame with each row containing the name, best score, and best parameters
        of an estimator after hyperparameter tuning.
    """
    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=42)
    results = []
    
    for estimator in tqdm(estimators):
        name = estimator.__class__.__name__
        print(f'Searching for classifier: {name}')
        clf = OptunaSearchCV(estimator=estimator, scoring=scoring,
                         param_distributions=optuna_grid[name],
                         cv=cv, n_jobs=-1, 
                         verbose=verbose, n_trials=n_trials)
        clf.fit(X_train, y_train)
        scorer = get_scorer(scoring)  
        score = scorer(clf, X_test, y_test)
        print(f'Best score: {score}')
        print(f'Best parameters: {clf.best_params_}')
        # return {'name': name, 'score': score, 'best_params': clf.best_params_}
        # result = run_estimator(estimator, X_train, y_train, X_test, y_test, cv, scoring, n_trials, verbose)
        results.append({'name': name, 'score': score, 'best_params': clf.best_params_})

    results_df = pd.DataFrame(results)
    return results_df


