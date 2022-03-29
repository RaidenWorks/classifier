import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, roc_auc_score

def pipeline_classifier(p_ls_features, p_df_features, p_df_label, p_i_algo, p_f_testsize, p_i_seed):

    # Separate features and labels
    X = p_df_features.values
    y = p_df_label.values

    # Split data into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=p_f_testsize, random_state=p_i_seed)

    # Select algorithm
    if p_i_algo == 1:
        reg = 0.1 # regularisation rate
        c_algo = LogisticRegression(C=1/reg, solver="liblinear")
        s_algo = 'Logistic Regression'
    elif p_i_algo == 2:
        c_algo = RandomForestClassifier(n_estimators=100)
        s_algo = 'Random Forest Classifier'
    elif p_i_algo == 3:
        c_algo = SVC(probability=True)
        s_algo = 'Support Vector Machine'
    elif p_i_algo == 4:
        c_algo = KNeighborsClassifier(n_neighbors=5)
        s_algo = 'K-Nearest Neighbour'

    # Generate string list for Categorical Features from p_ls_features
    ls_s_featuresCat = []
    for i in p_ls_features:
        if i == 'Gender' or i == 'Smoke' or i == 'Diabetes' or i == 'Ejection Fraction' or i == 'Favorite color':
            ls_s_featuresCat.append(i)

    # Generate string list for Numeric Features from the difference
    ls_s_featuresNum = [j for j in p_ls_features if j not in ls_s_featuresCat]

    # Generate a dynamic dictionary that updates the values when different sets of features are selected
    features_dict = {}
    val = 0
    for k in p_ls_features:
        features_dict[k] = val
        val += 1

    # Generate integer list for Numeric Features (needed for scaling pre-processing)
    ls_i_featuresNum = []
    for item in ls_s_featuresNum:
        ls_i_featuresNum.append(features_dict[item])

    # Generate integer list for Categorical Features (needed for one hot encoding pre-processing)
    ls_i_featuresCat = []
    for item in ls_s_featuresCat:
        ls_i_featuresCat.append(features_dict[item]) # for pipeline

    # Define preprocessing for numeric columns (make them on the same scale)
    numeric_features = ls_i_featuresNum
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())])

    # Define preprocessing for categorical features (encode them)
    categorical_features = ls_i_featuresCat
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])

    # Create preprocessing and training pipeline
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('logregressor', c_algo)])

    # Fit the pipeline to train a logistic regression model on the training set
    model = pipeline.fit(X_train, (y_train))

    # Get predictions from test data
    predictions = model.predict(X_test)
    y_scores = model.predict_proba(X_test)

    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_scores[:,1])

    # Format plots arrangement
    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(24, 6))

    # Text display of selected Features shown in 1st subplot
    s_features_down = '\n'.join(p_ls_features) # for displaying features as a string in plot and JPG

    # Print input parameters in 1st subplot
    fontsize = 14
    padding_left_Attrib = 0.1
    padding_left_Value = 0.40
    padding_top = 0.96
    padding_top_increment = 0.07
    ax[0].axis("off")
    ax[0].text(padding_left_Attrib, padding_top, 'Algorithm:', size=fontsize, va="top", ha="left")
    ax[0].text(padding_left_Value, padding_top, s_algo, size=fontsize, va="top", ha="left")
    ax[0].text(padding_left_Attrib, padding_top-padding_top_increment, 'Test size:', size=fontsize, va="top", ha="left")
    ax[0].text(padding_left_Value, padding_top-padding_top_increment, '{:.2f}'.format(p_f_testsize), size=fontsize, va="top", ha="left")
    ax[0].text(padding_left_Attrib, padding_top-2*padding_top_increment, 'Seed:', size=fontsize, va="top", ha="left")
    ax[0].text(padding_left_Value, padding_top-2*padding_top_increment, str(p_i_seed), size=fontsize, va="top", ha="left")
    ax[0].text(padding_left_Attrib, padding_top-3*padding_top_increment, 'Label:', size=fontsize, va='top', ha='left')
    ax[0].text(padding_left_Value, padding_top-3*padding_top_increment, 'Survive', size=fontsize, va='top', ha='left')
    ax[0].text(padding_left_Attrib, padding_top-4*padding_top_increment, 'Features:', size=fontsize, va='top', ha='left')
    ax[0].text(padding_left_Value, padding_top-4*padding_top_increment, s_features_down, size=fontsize, va='top', ha='left')

    # Plot ROC curve in 2nd subplot
    ax[1].set_aspect('equal', adjustable='box')
    ax[1].plot([0, 1], [0, 1], 'k--')
    ax[1].plot(fpr, tpr)
    ax[1].set_xlabel('False Positive Rate')
    ax[1].set_ylabel('True Positive Rate')
    ax[1].set_title('ROC Curve')

    # Plot Confusion Matrix in 3rd subplot
    cf_matrix = confusion_matrix(y_test, predictions)
    group_names = ['True non-survivor','False survivor','False non-survivor','True survivor']
    group_counts = ["{0:0.0f}".format(value) for value in
                    cf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in
                        cf_matrix.flatten()/np.sum(cf_matrix)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
            zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    hmap = sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues', ax=ax[2], square=True)
    hmap.set_title('Confusion Matrix\n')
    hmap.set_xlabel('Predicted Values')
    hmap.set_ylabel('Actual Values')
    hmap.xaxis.set_ticklabels(['non-survivor','survivor'])
    hmap.yaxis.set_ticklabels(['non-survivor','survivor'])

    # Print metrics in 4th subplot
    f_accuracy = accuracy_score(y_test, predictions)
    f_recall = recall_score(y_test, predictions, zero_division=1)
    f_precision = precision_score(y_test, predictions, zero_division=1)
    f_f1score = f1_score(y_test, predictions, zero_division=1)
    f_auc = roc_auc_score(y_test, y_scores[:,1])
    s_auc = '{:.4f}'.format(f_auc) # save the string to be used as part of the JPG name
    fontsize = 14
    padding_left_Attrib = 0.15
    padding_left_Value = 0.45
    padding_top = 0.66
    padding_top_increment = 0.07
    ax[3].axis("off")
    ax[3].text(padding_left_Attrib, padding_top, 'Accuracy:', size=fontsize, va="top", ha="left")
    ax[3].text(padding_left_Value, padding_top, '{:.4f}'.format(f_accuracy), size=fontsize, va="top", ha="left")
    ax[3].text(padding_left_Attrib, padding_top-padding_top_increment, 'Recall:', size=fontsize, va="top", ha="left")
    ax[3].text(padding_left_Value, padding_top-padding_top_increment, '{:.4f}'.format(f_recall), size=fontsize, va="top", ha="left")
    ax[3].text(padding_left_Attrib, padding_top-2*padding_top_increment, 'Precision:', size=fontsize, va="top", ha="left")
    ax[3].text(padding_left_Value, padding_top-2*padding_top_increment, '{:.4f}'.format(f_precision), size=fontsize, va="top", ha="left")
    ax[3].text(padding_left_Attrib, padding_top-3*padding_top_increment, 'F1 Score:', size=fontsize, va="top", ha="left")
    ax[3].text(padding_left_Value, padding_top-3*padding_top_increment, '{:.4f}'.format(f_f1score), size=fontsize, va="top", ha="left")
    ax[3].text(padding_left_Attrib, padding_top-4*padding_top_increment, 'AUC:', size=fontsize, va='top', ha='left')
    ax[3].text(padding_left_Value, padding_top-4*padding_top_increment, s_auc, size=fontsize, va='top', ha='left')

    return(fig, s_auc) # we return the fig as we want to use it to save a JPG (otherwise this could be a void function)