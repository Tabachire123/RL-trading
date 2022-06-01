import copy

import numpy as np
import pandas as pd

import torch

def calculate_gradient(model, interpolated_input, actions,  feature_idx, stock_idx, stock_dimension, tech_indicator_list, h = 1e-1):
    """Calculate gradient of the loss function w.r.t. the input features.

    :param model: the model to be used for the gradient calculation
    :type model: torch.nn.Module
    :param interpolated_input: the input features to be used for the gradient calculation
    :type interpolated_input: numpy.ndarray
    :param actions: the actions to be used for the gradient calculation
    :type actions: numpy.ndarray
    :param feature_idx: the index of the feature to be used for the gradient calculation
    :type feature_idx: int
    :param stock_idx: the index of the stock to be used for the gradient calculation
    :type stock_idx: int
    :param stock_dimension: the dimension of the stock to be used for the gradient calculation
    :type stock_dimension: int
    :param tech_indicator_list: the list of technical indicators to be used for the gradient calculation
    :type tech_indicator_list: list
    :param h: the step size to be used for the gradient calculation
    :type h: float
    :return: the gradient of the loss function w.r.t. the input features
    :rtype: torch.Tensor
    """
    feature_dimension = len(tech_indicator_list)
    forward_input = copy.deepcopy(interpolated_input)

    # Perturb the input features
    #forward_input[feature_idx + stock_dimension][stock_idx] += h
    forward_input[feature_idx + stock_dimension][stock_idx] = 0

    forward_Q = model.policy.evaluate_actions(torch.cuda.FloatTensor(forward_input).reshape(-1,stock_dimension*(stock_dimension + feature_dimension)),
                        torch.cuda.FloatTensor(actions).reshape(-1,stock_dimension))
    interpolated_Q = model.policy.evaluate_actions(torch.cuda.FloatTensor(interpolated_input).reshape(-1,stock_dimension*(stock_dimension + feature_dimension)), 
                         torch.cuda.FloatTensor(actions).reshape(-1,stock_dimension))
    forward_Q = forward_Q[0].detach().cpu().numpy()[0]
    interpolated_Q = interpolated_Q[0].detach().cpu().numpy()[0]
    res =  (forward_Q - interpolated_Q) / h
    return res

def calculate_meta_q(trader, model, name, df_actions, tech_indicator_list):
    """Calculate the explainability value of the model for each indicator.

    :param trader: the trader to be used for the gradient calculation
    :type trader: Trader
    :param model: the model to be used for the gradient calculation
    :type model: torch.nn.Module
    :param name: the name of the model to be used for the gradient calculation
    :type name: str
    :param df_actions: the actions to be used for the gradient calculation
    :type df_actions: pandas.DataFrame
    :param tech_indicator_list: the list of technical indicators to be used for the gradient calculation
    :type tech_indicator_list: list
    :return: the explainability value of the model for each indicator
    :rtype: pandas.DataFrame
    """
    meta_Q = {"date":[], "feature":[], "stock":[], "Saliency Map":[]}

    # Set prec_step
    if name == "A2C":
        prec_step = 1e-2
    else:
        prec_step = 1e-1

    trade = trader.get_trade()
    stock_dimension = len(trade.tic.unique())
    unique_trade_date = trade.date.unique()
    for i in range(len(unique_trade_date)-1):
        date = unique_trade_date[i]
        covs = trade[trade['date'] == date].cov_list.iloc[0]
        features = trade[trade['date'] == date][tech_indicator_list].values # N x K
        actions = df_actions.loc[date].values

        for feature_idx in range(len(tech_indicator_list)):
      
            int_grad_per_feature = 0
            for stock_idx in range(features.shape[0]):#N
        
                int_grad_per_stock = 0
                avg_interpolated_grad = 0
                for alpha in range(1, 51):
                    scale = 1/50
                    baseline_features = copy.deepcopy(features)
                    baseline_noise = np.random.normal(0, 1, stock_dimension)
                    baseline_features[:,feature_idx] = [0] * stock_dimension
                    interpolated_features = baseline_features + scale * alpha * (features - baseline_features) # N x K
                    interpolated_input = np.append(covs, interpolated_features.T, axis = 0)
                    interpolated_gradient = calculate_gradient(model, interpolated_input, actions, feature_idx, stock_idx, stock_dimension, tech_indicator_list, h = prec_step)[0]
          
                    avg_interpolated_grad += interpolated_gradient * scale
                int_grad_per_stock = (features[stock_idx][feature_idx] - 0) * avg_interpolated_grad
                int_grad_per_feature += int_grad_per_stock
      
                meta_Q['date'] += [date]
                meta_Q['feature'] += [tech_indicator_list[feature_idx]]
                #meta_Q['Saliency Map'] += [int_grad_per_feature]
                meta_Q['stock'] += [stock_idx]
                meta_Q['Saliency Map'] += [int_grad_per_stock]

    meta_Q = pd.DataFrame(meta_Q)
    return meta_Q

def calculate_meta_q2(trader, model, df_actions, tech_indicator_list):
    """Calculate the explainability value of the model for each indicator (different method we tested).

    :param trader: the trader to be used for the gradient calculation
    :type trader: Trader
    :param model: the model to be used for the gradient calculation
    :type model: torch.nn.Module
    :param df_actions: the actions to be used for the gradient calculation
    :type df_actions: pandas.DataFrame
    :param tech_indicator_list: the list of technical indicators to be used for the gradient calculation
    :type tech_indicator_list: list
    :return: the explainability value of the model for each indicator
    :rtype: pandas.DataFrame
    """
    meta_Q = {"date":[], "feature":[], "Saliency Map":[]}

    trade = trader.get_trade()
    stock_dimension = len(trade.tic.unique())
    unique_trade_date = trade.date.unique()
    for i in range(len(unique_trade_date)-1):
        date = unique_trade_date[i]
        covs = trade[trade['date'] == date].cov_list.iloc[0]
        features = trade[trade['date'] == date][tech_indicator_list].values
        input = np.append(covs, features.T, axis = 0)
        actions = df_actions.loc[date].values

        orig_Q = model.policy.evaluate_actions(torch.cuda.FloatTensor(input).reshape(-1,stock_dimension*(stock_dimension + 4)), torch.cuda.FloatTensor(actions).reshape(-1, stock_dimension))
        orig_Q = orig_Q[0].detach().cpu().numpy()[0]

        for idx in range(len(tech_indicator_list)):
            perturbed_feature = features
            perturbed_noise = np.random.normal(0, 1, stock_dimension)
            perturbed_feature[:,idx] = [0] * stock_dimension
            perturbed_input = np.append(covs, perturbed_feature.T, axis = 0)
            perturbed_Q = model.policy.evaluate_actions(torch.cuda.FloatTensor(perturbed_input).reshape(-1,stock_dimension*(stock_dimension + 4)), torch.cuda.FloatTensor(actions).reshape(-1,stock_dimension))
            perturbed_Q = perturbed_Q[0].detach().cpu().numpy()[0]
            meta_Q['date'] += [date]
            meta_Q['feature'] += [tech_indicator_list[idx]]
            meta_Q['Saliency Map'] += [orig_Q[0] - perturbed_Q[0]]

    meta_Q = pd.DataFrame(meta_Q)
    return meta_Q
