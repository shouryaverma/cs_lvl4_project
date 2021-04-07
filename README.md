# cs_lvl4_project
## Machine Learning Models to Detect Arrhythmia based on ECG Data - Interpretability

The analysis of electrocardiogram (ECG) signals can be time consuming as it is performedmanually by cardiologists. Therefore, automation through machine learning (ML) classification isbeing increasingly proposed which would allow ML models to learn the features of a heartbeat anddetect abnormalities. Through interpretability of these models, we would be able to understandhow a machine learning algorithm makes its decisions and what patterns are being followed forclassification. 

This project investigates global and local interpretability methods on different ML algorithms,along with building convolutional neural network and long short-term memory classifiers basedon state-of-the-art models. Partial Dependence Plots, Shapley Additive Explanations, PermutationFeature Importance, and Gradient Weighted Class Activation Maps are the four interpretabilitytechniques investigated on ML models classifying ECG rhythms. The classifiers are evaluatedusing K-Fold cross-validation and Leave Groups Out techniques, and various statistical tests areimplemented on the resulting performance metrics. The effective interpretability methods arequantitatively evaluated along with discussing how successful different interpretability methodsare at explaining the decisions of ’black-box’ classifiers.

### The src folder contains project code both in .ipynb notebook and .py format. The code files have the following description:

- data_preprocessing: Reads raw MIT-BIH data, prepares test and train datasets for both beats holdout and patient holdout method.
- pca_tsne_umap: Creates clusters on the test and train data.

- sklearn_models: Trains and tests the sklearn models, stores results and feature importance.
- cnn_model: Trains and tests the CNN model, stores results, grad-CAM and feature importance.
- lstm_model: Trains and tests the LSTM model, stores results, grad-CAM and feature importance.

- pdp_shap: Implements PDP and SHAP on NNMLP and SVC model.
- feature_imp_plots: Creates plots for feature importance of all models.

- cross_val: Performs K-Fold cross validation on all models, stores results.
- results_and_statstests: Creates plots for all results data and implements statistical tests.
