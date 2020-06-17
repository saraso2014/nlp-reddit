def eda(dataframe):
    print("MISSING VALUES:\n{}\n".format(dataframe.isnull().sum()))
    print("INDEX:\n{}\n".format(dataframe.index))
    print("DTYPES:\n{}\n".format(dataframe.dtypes))
    print("SHAPE:\n{}\n".format(dataframe.shape))
    print("DESCRIBE:\n{}\n".format(dataframe.describe().T))
    
# Chuck EDA function (extracted unique portion)
def eda_unique(dataframe):
    for item in dataframe:
        print(item)
        print(dataframe[item].nunique())

        
# Sara EDA Plot (big boi)
def eda_plt(df, sale_price, feature, kind):
    plot_kind = plt.kind
    if kind == 'scatter':
        figure = plt.kind(sale_price,feature)
        plt.xlabel('Sale Price')
        plt.ylabel('Feature')
        plt.title('Feature v Sale Price');
        return figure
    elif kind == 'hist':
        figure = plt.hist(feature)
        plt.xlabel('Distribution')
        plt.ylabel('Feature')
        plt.title('Feature Distribution')
        return figure
    elif kind == 'boxplot':
        return 'this is a boxplot'
    elif kind == 'catplot':
        return 'this is a categorical plot'
    else:
        return 'invalid seaborn plot type input'
    
def linear_model(X_train, X_test, y_train, y_test,
              transform, model, params):

    # STEP 1: select pipeline
    if (transform == 'cvec' and model == 'lr'):
        pipe = Pipeline([
            ('cvec', CountVectorizer()),
            ('lr', LogisticRegression(solver='lbfgs'))
        ])
    else:
        pipe = Pipeline([
            ('tvec', TfidfVectorizer()),
            ('lr', LogisticRegression(solver='lbfgs'))
        ])
    
    # STEP 2: run gridsearch
    gs = GridSearchCV(pipe,
                      param_grid=params,
                      n_jobs=-1,
                      cv=5)
    gs.fit(X_train,y_train)
    train_score = gs.score(X_train,y_train)
    test_score = gs.score(X_test,y_test)
    
    # STEP 3: get best params
    best_parameters = gs.best_estimator_.get_params()
    param_dict = {}
    for param_name in sorted(params.keys()):
        new_param = {
            param_name : best_parameters[param_name],
        }
        param_dict.update(new_param)
    
    # STEP 4: extract X coefficients
    coef_dict = pd.DataFrame(gs.best_estimator_.named_steps.lr.coef_, 
                      columns=gs.best_estimator_.named_steps.cvec.get_feature_names())
    
    # STEP 5: make new score dictionary
    score_values = [model,transform,
              train_score,test_score,
              param_dict,coef_dict]
    score_keys = ['Model','Transform',
            'Train Score','Test Score',
            'Best Paramenters',
            'Coefficient Dictionary']
    score_dict = dict(zip(keys,values))
    
    # STEP 6: update scores csv with new model
    #df_old_scores = pd.read_csv('all_scores.csv')
    #merge_scores = [df_old_scores, df_new_row]
    #df_all_scores = pd.concat(merge_scores)
    
  
    return score_dict