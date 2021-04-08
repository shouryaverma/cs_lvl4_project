# User manual 

The code is available in both .ipynb notebook and .py code formats. 

These files should be ran in the following order:

  - data_preprocessing: Reads raw MIT-BIH data, prepares test and train datasets for both beats holdout and patient holdout method.
  - pca_tsne_umap: Creates clusters on the test and train data.
  - sklearn_models: Trains and tests the sklearn models, stores results and feature importance.
  - cnn_model: Trains and tests the CNN model, stores results, grad-CAM and feature importance.
  - lstm_model: Trains and tests the LSTM model, stores results, grad-CAM and feature importance.
  - pdp_shap: Implements PDP and SHAP on NNMLP and SVC model.
  - feature_imp_plots: Creates plots for feature importance of all models.
  - cross_val: Performs K-Fold cross validation on all models, stores results.
  - correlation: Kendall Tau correlation for CNN and LSTM segments, and feature importanc results per model.
  - results_and_statstests: Creates plots for all results data and implements statistical tests.