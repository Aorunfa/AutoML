
# AutoFeature自动特征筛选算法介绍
## 一.介绍
&emsp;&emsp;该自动特征筛选算法适用于结构化数据的监督学习场景，对构建的大量特征进行自动过滤，保留与标签较为相关的特征组合， 
减小特征冗余，帮助开发人员快速确定重要特征组，提高建模的效率和效果。  
该算法方案包括三种常用特征筛选的方法：
1. 过滤式：根据某个指标，判断特征与标签的相关性或分布独立性，对于小于阈值的指标进行过滤；进一步，判断特征之间的相关性或分布独立性，对大于阈值的特征组合，只保留与标签最相关的特征。  
2. 嵌套式：将特征组输出多个评分模型，获得每个模型输出的特征重要性，并对所有模型结果进行加权得到最终的特征重要评分，保留topN的特征组合。  
3. 包裹式：选择评分基模型，从一个特征开始，寻找能够使得模型预测效果最大的那一个特征，逐步扩展每阶段最优的特征组合，直到到达目标数量。**注意该方式只能保证输出的特征组合是较优的，非全局最优组合**。

## 二.使用方式
###2.1 数据准备
&emsp;&emsp;在运行算法前，需要对原始特征进行必要的数据清洗，如空值处理、异常值处理、字符数据转换等，并进行足够的特征工程。  
&emsp;&emsp;另一方面，算法能够处理回归问题的特征过滤以及二分类问题的特征过滤，对于多分类问题，在一些评估指标上尚不能实现自动化处理。以下展示sample.csv数据集里的基本数据情况。
    **数据展示：特征列，标签列，数据类型
###2.2 运行内置方法
1.根据问题指定必要的参数，必须要指定的是*fit_type*参数，若是回归问题则传入*regression*，分类问题则传入*classification*。
其余参数含义参考注释，默认值可以满足大多数使用场景。根据sample.csv数据可以进行该实例化`af = AutoFeature(fit_type='regression', fit_metric='r2')`表示问题场景为回归，评价指标选择r2
```python
def __init__(self, corrval_withlabel=0.35, corrval_withothers=0.85, p_val=0.05,
             fit_type='regression', fit_metric=None, k_cv=4, top_n=50,
             group_n=-1, params_searcher='grid', log_path='auto_events.log'):
    super(AutoFeature, self).__init__()
    # 日志模块
    self.log = Logger(path=log_path)
    self.log('--'*10 + '特征自动筛选日志' + '--'*10)
    self.fit_type = fit_type                        # 问题是回归还是分类问题
    # 过滤式方法参数
    self.corrval_withlabel = corrval_withlabel      # 与标签最低的相关性系数
    self.corrval_withothers = corrval_withothers    # 特征间最大的相关系数
    self.p_val = p_val                              # 独立性检验p值
    # 嵌套式方法参数
    self.fit_metric = fit_metric                    # 模型多折验证评价指标
    self.k_cv = k_cv                                # 进行几折交叉验证
    self.top_n = top_n                              # 保留重要性topN的特征组
    # 包裹式方法参数
    self.group_n = group_n                          # 包裹式筛选的最大特征数量
    # 超参搜索器
    self.params_searcher = params_searcher          # random or bayes
```
2.进行过滤式筛选，调用实例.filtering(**params)方法即可，
入参为数据集dataframe，数值特征列名list，分类特征列list，以及标签列名str，返回对应的过滤数据集，数值特征列和分类特征列。  
运行日志将保存在log_path中，控制台打印日志展示如下。
![过滤式筛选日志](pict/过滤控制台日志.png "过滤式筛选日志")  
```python
def filtering(self, df:pd.DataFrame, 
              feature_num:list, feature_clf:list, label_name:str):
    """
    以指标过滤式进行特征筛选
    :param df:数据集
    :param feature_num:数值特征列
    :param feature_clf:分类特征列
    :param label_name:标签列名
    :return:过滤后的数据集，数值特征列，分类特征列
    """
```
3.进行嵌套式筛选，调用实例.nesting(**params)方法，入参与过滤式的形式相似，包括数据集dataframe，特征列名list(包含数值特征或分类特征)，以及标签列名str，返回对应重要性排名在top_N的特征列list。  
控制台打印日志展示如下。
![嵌套控制台日志](pict/嵌套控制台日志.png "嵌套控制台日志")  
    
4.进行包裹式过滤方法，调用实例.wrapping(**params)方法，入参与嵌套式相同，包括数据集dataframe，特征列名list(包含数值特征或分类特征)，
以及标签列名str，同时指定评价基模型base_model，默认为cart树模型，以及要寻找的最大特征组数量group_n，默认为-1，对应为输入列名的数量，即不做限制。  
算法依次从一个特征数量开始，逐步加入新特征，每次找到能使评分模型最优的特征。最后返回在给定特征数量group_n之内，使得评分基模型效果最好的特征组列名list。  
控制台输出日志如下。
![包裹控制台日志](pict/包裹控制台日志.png "包裹控制台日志")  

## 三.附录：主要代码介绍
### 3.1过滤式评估指标
&emsp;&emsp;过滤式计算指标包括两类，第一类为皮尔森相关系数，用于度量数值变量与数值标签（回归问题）、数值变量之间的相关性，**算法保留与数值标签相关性大于阈值的特征，过滤特征间相关性大于阈值的特征保留与标签最相关的特征**；
第二类为卡方检验p值或互信息数值，见_indpendent_test()方法，用于度量分类标签（分类问题）与分类特征、数值特征的分布独立性，或用于度量数值标签（回归问题）与分类特征的分布独立性。**需要注意的是，在调用该方法时，需要对数值数据进行离散化分箱，保证结果的准确性**。
```python
def _indpendent_test(self, dist_feature, dist_target: pd.Series,
                     box_trans=False, method='ch2'):
    """
    变量独立性检验：卡方检验、互信息计算数值、KL散度
    使用于一组变量为类别变量
    :param dist_feature: 特征数组
    :param dist_target: 标签数组
    :param box_trans: 是否对特征数组进行分箱，用于数值特征离散化
    :param method: 选择独立性检验方法
    :return:
    """
    boxes = 20
    if box_trans:  # 特征离散化
        dist_feature = pd.qcut(dist_feature, q=boxes,
                               duplicates='drop', labels=False)  # 自适应最小unique
    if self.fit_type == 'regression':  # 标签离散化
        dist_target = pd.qcut(dist_target, q=boxes,
                              duplicates='drop', labels=False)
    # 转换为二维矩阵
    dist_feature = np.asarray(dist_feature).reshape(-1, 1)
    dist_target = np.asarray(dist_target).reshape(-1, 1)
    if method == 'ch2':
        # _, p = chi2(dist_feature, dist_target)
        p = chi2(dist_feature, dist_target)[1][0]  # p<0.05 拒绝原假设 变量不独立
        if p <= self.p_val:
            return p, False
        return p, True
    else:
        # 互信息值
        if box_trans:
            mi = mutual_info_regression(dist_feature, dist_target)[0]  # 越趋近0越独立
        else:
            mi = mutual_info_classif(dist_feature, dist_target)[0]
        if mi >= 0.2:
            return mi, False
        return mi, True
```

### 3.2嵌套式评分模型与参数空间
&emsp;&emsp;算法通过_set_models()内置了多种常见的评分基模型，包括cart Tree、xgboost、lightGBM、catboost，回归问题增加lasso回归，分类问题增加带l1正则化的logistic回归；
同时，基于评分基模型，通过_set_params()设置了对应模型的超参空间，使用者可以依据实际需求重写这两个方法。
```python
def _set_models(self):
    """
    初始化模型
    :return:分类问题与回归问题的模型字典
    """
    reg_keys = ['lasso', 'cart', 'xgb', 'lgb', 'cab']
    clf_keys = ['logit'] + reg_keys[1:]
    lasso = Lasso
    logit = LogisticRegression
    cart_reg = DecisionTreeRegressor
    cart_clf = DecisionTreeClassifier
    xgb_clf = XGBClassifier
    xgb_reg = XGBRegressor
    lgb_reg = LGBMRegressor
    lgb_clf = LGBMClassifier
    cab_reg = CatBoostRegressor
    cab_clf = CatBoostClassifier

    reg_model = dict(zip(reg_keys,
                         [lasso, cart_reg, xgb_reg, lgb_reg, cab_reg]))
    clf_model = dict(zip(clf_keys,
                         [logit, cart_clf, xgb_clf, lgb_clf, cab_clf]))
    if self.fit_type == 'regression':
        self.search_models = reg_model
    elif self.fit_type == 'classification':
        self.search_models = clf_model
    else:
        raise ValueError(f'错误的值{self.fit_type}, fit_type取值为regression或classification')

 def _set_params(self):
        """
        初始化各个模型超参空间字典
        """
        random_seed = 2023  # TODO设置随机种子
        cart_params = {'max_depth': [x for x in range(2, 15, 2)]}  # TODO 增加新的参数空间
        xgb_params = {'max_depth': [x for x in range(2, 11, 2)],
                      'n_estimators': [25, 50, 75, 100],
                      'learning_rate': [5e-1, 1e-1, 5e-2],
                      'importance_type': ['gain']}
        lgb_params = {'objective': ['regression'],
                      'max_depth': [x for x in range(2, 11, 2)],
                      'n_estimators': [30, 60, 100],
                      'learning_rate': [5e-1, 1e-1, 5e-2],
                      'verbosity': [-1],  # 隐藏警告信息
                      'importance_type': ['gain']}
        cab_params = {'max_depth': [x for x in range(2, 11, 2)],
                      'iterations': [25, 50, 75, 100],
                      'learning_rate': [5e-1, 1e-1, 5e-2],
                      'verbose': [False]}
        lasso_params = {'alpha': list([0.1 * x for x in range(1, 21)])}
        logit_params = {'penalty': ['l1'],
                        'solver': ['saga'],
                        'C': list([0.1 * x for x in range(1, 101, 5)])}
        self.search_params = {'lasso': lasso_params,
                              'logit': logit_params,
                              'cart': cart_params,
                              'xgb': xgb_params,
                              'lgb': lgb_params,
                              'cab': cab_params
                              }
```

### 3.3超参搜索器
&emsp;&emsp;超参搜索器主要用于搜索评分模型的最（较）优超参组合，算法内置了三种算子：网格搜索、随机搜索、贝叶斯搜索。三者各有优劣。  
&emsp;&emsp;网格搜索暴力遍历参数组合，牺牲效率换取最好的效果；随机搜索根据一定的随机性，随机抽取参数组合，迭代至最大的次数，效率最快但效果可能是最差的；贝叶斯搜索主要基于已有的参数组合效果，以一定的条件分布抽取新的参数组合，兼顾效率与效果。
```python
def _set_seacher(self, model, param_dist: dict, scoring_fun, cv=None):
    """
    自定义超参寻优方法
    :param model: 模型
    :param param_dist: 参数空间
    :param scoring_fun: 评价函数， str or function
    :param cv: k折，int or 自定义划分方法
    :return:
    """
    if cv is None:
        cv = 4
    if self.parms_search == 'grid':
        searcher = GridSearchCV(model,
                                param_grid=param_dist,
                                cv=cv,
                                scoring=scoring_fun)
    elif self.parms_search == 'random':
        searcher = RandomizedSearchCV(model,
                                      param_dist,
                                      cv=cv,
                                      scoring=scoring_fun,
                                      n_iter=50, random_state=42)
    else:
        searcher = BayesSearchCV(estimator=model,
                                 search_spaces=param_dist, n_jobs=-1, cv=cv,
                                 scoring=scoring_fun)
    return searcher
```

### 3.4内置评价指标
&emsp;&emsp;算法分别内置了回归和分类问题的模型效果评估指标，主要用于交叉验证和重要性权值计算。  
&emsp;&emsp;_set_metrics()方法用于初始化评分指标字典，_metric_fun()方法用于根据self.fit_type + self.fit_metric调用评分字典，输出评分数值。
**需要注意的是，当fit_metric为None时，回归问题默认评价指标为r2，分类问题为accuracy**。
```python
def _set_metrics(self):
    # 定义评估指标字典
    reg = {'r2': metrics.r2_score,
           'mape': metrics.mean_absolute_percentage_error,
           'mse': metrics.mean_squared_error}
    clf = {'auc': metrics.accuracy_score,
           'recall': metrics.recall_score,
           'precision': metrics.precision_score,
           'rec_pre': self.metric_rec_pre,
           'f1': metrics.f1_score,
           'roc_auc': metrics.roc_auc_score}
    self.search_metrics = {'regression': reg, 'classification': clf}

def _metric_fun(self, y_true, y_pred):
    """
    进行指标评价
    :param y_pred:
    :param y_true:
    :return:
    """
    self._set_metrics()
    if self.fit_type == 'regression':
        if self.fit_metric is None:
            self.fit_metric = 'r2'  # 回归问题默认评价指标为r2
    else:
        if self.fit_metric is None:
            self.fit_metric = 'auc'  # 分类问题默认是准确率
    val = self.search_metrics[self.fit_type][self.fit_metric](y_true, y_pred)
    return val
```


# Easyplot绘图类介绍
## 一.介绍
    
## 二.使用及实例
###2.1 初始化画布

###2.2 单个数值变量分布
    场景：
    快速画图命令：

###2.2 单个数值变量分布...


# RiskQuantify风险量化算法介绍
## 一.介绍
    开发目的、解决的问题
    简单法方案
    效果
## 二.使用方式
###2.1 数据准备

###2.2 初始化参数
    参数含义及作用：
    示例：

###2.3 方法调用