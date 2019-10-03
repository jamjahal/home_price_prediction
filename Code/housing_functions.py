# 
def check_data(dataframe):
    print(dataframe.head())
    print('')
    print('='*25)
    print('')
    msno.matrix(dataframe)
    print('')
    print('='*25)
    print('')
    print(df.isnull().sum()[df.isnull().sum() > 0].sort_values(ascending = False))
    print('')
    print('='*25)
    print('')
    print(dataframe.info())
    
# Check the correlation rank against the target
def corr_rank (df, target):
    
    """
    df = pandas dataframe or matrix
    target = target variable or column from dataframe
    """
    y= df[target]
    plt.figure(figsize=(10,50))
    sns.heatmap(df.corr()[[target]].sort_values(by = target, 
                                                    ascending=False),
               annot = True,
               cmap='RdBu_r',

               );

# Scatter Plot the variables 

def plot_vars(model,X, y, ncols=2):
    nrows = int(np.ceil((len(columns))/2))
    plt.figure(figsize=(12,9));
    fig, ax = plt.subplots(1,len(X.columns.values), nrows=nrows,ncols=ncols,sharey=True,
                           constrained_layout=True,figsize=(10, 2.5*nrows))
    model=model()
    for i,e in enumerate(X.columns):
      model.fit(X[e].values[:,np.newaxis], y.values)
      ax[i].set_title("Best fit line")
      ax[i].set_xlabel(str(e))
      ax[i].set_ylabel('Sale Price')
      ax[i].scatter(X[e].values[:,np.newaxis], y,color='g')
      ax[i].plot(X[e].values[:,np.newaxis], 
      model2.predict(X[e].values[:,np.newaxis]),color='k')
    # Generate a scatterplot of predicted values versus actual values.
    plt.scatter(preds, y, s=8, color='skyblue', alpha = 0.9)

    # Plot a line.
    plt.plot([0, np.max(y)],
             [0, np.max(y)],
             color = 'black')

    # Tweak title and axis labels.
    plt.xlabel("Predicted Values: $\hat{y}$", fontsize = 20)
    plt.ylabel("Actual Values: $y$", fontsize = 20)
    plt.title('Predicted Values vs. Actual Values', fontsize = 24);
    
# Test your Model

def test_model(model, dataframe, features, y):
    """
    model = regression model 
    name = Str. that you would like the model to be named
    dataframe = pandas dataframe as data source
    features =  series or array of columns to test in model
    y = target variable
    
    """
    X = dataframe[features]
    y = y

    # train test split
    X_train, X_val, y_train, y_val = train_test_split(X, y, 
                                                        random_state = 13)

    title = model
    title.fit(X_train, y_train)

    print(f'Train score: {title.score(X_train, y_train)}')
    print('Cross val score: {}'.format(cross_val_score(title, 
                                            X_train, y_train, cv=5).mean()))
    print(f'Test score: {title.score(X_val, y_val)}')

    val_preds = pd.DataFrame(title.predict(X_val), columns = ['val_preds'])
    val_preds.head()
    
    val_preds['val_true']= y_val.values
    val_preds['val_errors']=val_preds['val_preds']-val_preds['val_true']

    # Plot predictions vs true values
    plt.scatter(x = val_preds['val_preds'], y = val_preds['val_errors']);
    plt.axhline(0, color='red');
    
    
# Predict with model and save predictions to csv for submission to Kaggle competition

def predict_and_submit (dataframe, model, features, submission_number):
    """
    dataframe = test dataframe
    model = model
    features = matrix of variables to test
    submission_number = iteration number(str) to help in labeling files and columns
        
    """
    test_preds=model.predict(dataframe[features])
    test['preds_1'] = test_preds
    submission=test.loc[:, ['id','preds_1']]
    submission.rename(columns = {'id':'Id', 'preds_1':'SalePrice'}, inplace = True)
    # Save submission as csv
    filename='submission_'+str(submission_number)
    submission.to_csv(f'../Submissions/{filename}.csv',index=False)
    submission.head()
    
# Transform powers to regularize distribution

def power_tran (dataframe, features, target):
    X=dataframe[features]
    y=dataframe[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    pt = PowerTransformer()
    pt.fit(X_train)
    X_train = pt.transform(X_train)
    X_test = pt.transform(X_test)
    pt_y = PowerTransformer()
    
    # PowerTransformer requires a matrix/DataFrame, 
    # which is why we're using the .to_frame() method on y_train
    pt_y.fit(y_train.to_frame()) 
    y_train_pt = pt_y.transform(y_train.to_frame())
    y_test_pt = pt_y.transform(y_test.to_frame())
    

    
# This tests Ridge, LinearRegressionCV, and LassoCV models and will create a submission csv of predictions for the winning model.  Very useful and a huge time saver.

def multi_model_test(dataframe, test_dataframe, features, target, model_number): 
    """
    dataframe = dataframe to train models
    test_dataframe = dataframe to test model and submit
    features = features to test
    target = target variable column name
    model_number = iteration number to append to filename
    
    Some of this was based on the Riley's notes on power transformation.
    """
    y=dataframe[target]
    X=dataframe[features]
# Test Train split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    
# Scale variables
    pt = PowerTransformer()
    pt.fit(X_train)
    X_train = pt.transform(X_train)
    X_test = pt.transform(X_test)

# Convert back to matrix / Dataframe
    pt_y = PowerTransformer()
    pt_y.fit(y_train.to_frame()) 
    y_train_pt = pt_y.transform(y_train.to_frame())
    y_test_pt = pt_y.transform(y_test.to_frame())
    
# Test 3 models    
    lr = LinearRegression()
    lr_scores = cross_val_score(lr, X_train, y_train_pt, cv=3)
    lasso = LassoCV(n_alphas=200, cv=5)
    lasso_scores = cross_val_score(lasso, X_train, y_train_pt[:, 0], cv=3)
    ridge = RidgeCV(alphas=np.linspace(.1, 10, 100))
    ridge_scores = cross_val_score(ridge, X_train, y_train_pt, cv=3)
    print(f' The Linear Regression score is {lr_scores.mean()}.')
    print(" ")
    print(f' The Ridge score is {ridge_scores.mean()}.')    
    print(" ")
    print(f' The lasso score is {lasso_scores.mean()}.')
    print(" ")
    if ((lr_scores.mean() > ridge_scores.mean()) and (lr_scores.mean() > lasso_scores.mean())):
        lr.fit(X_train, y_train_pt)
        lr.score(X_train, y_train_pt)
        lr_test_score = lr.score(X_test, y_test_pt)
        print(f'Linear Regression was the closest model and returned a test score of {lr_test_score}.')
        print(" ")
        pred = lr.predict(X_test)
        pred_reversed = pt_y.inverse_transform(pred.reshape(-1,1))
        print(f'The R2 score is: {r2_score(y_test, pred_reversed)}.')
        model=lr
        pd.Series(lr.coef_, index=features).plot.bar(figsize=(15, 7));
    elif ((lasso_scores.mean() > ridge_scores.mean()) and 
    (lasso_scores.mean() > lr_scores.mean())):
        lasso.fit(X_train, y_train_pt)
        lasso.score(X_train, y_train_pt)
        lasso_test_score = lasso.score(X_test, y_test_pt)
        print(f'Lasso was the closest model and returned a test score of {lasso_test_score}.')
        print(" ")
        pred = lasso.predict(X_test)
        pred_reversed = pt_y.inverse_transform(pred.reshape(-1,1))
        print(f'The R2 score is: {r2_score(y_test, pred_reversed)}.')
        model=lasso
        pd.Series(lasso.coef_, index=features).plot.bar(figsize=(15, 7));
    else:
        ridge.fit(X_train, y_train_pt)
        ridge.score(X_train, y_train_pt)
        ridge_test_score = ridge.score(X_test, y_test_pt)
        print(f'Ridge was the closest model and returned a test score of {ridge_test_score}.')
        print(" ")
        pred = ridge.predict(X_test)
        pred_reversed = pt_y.inverse_transform(pred.reshape(-1,1))
        print(f'The R2 test score is: {r2_score(y_test, pred_reversed)}.')
        print(f'Ridge Coef = {ridge.coef_}')
        model=ridge
        pd.Series(data=ridge.coef_[0], index=features).plot.bar(figsize=(15, 7));
        
# # Scale variables and Prep Test data and Submission
    X_submit = test_dataframe[features]
    X_submit = pt.transform(X_submit) 
    pred = model.predict(X_submit)
    pred_reversed = pt_y.inverse_transform(pred.reshape(-1,1))
    submit = pd.DataFrame()
    submit['Id'] = test_dataframe.loc[:,'id']
    submit['SalePrice'] = pred_reversed
    filename='submission_'+str(model_number)
    submit.to_csv(f'../Submissions/{filename}.csv',index=False)