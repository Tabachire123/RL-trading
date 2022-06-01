import pandas as pd

from dow_jones import DOW_30_TICKER

def compute_actions_per_sector(actions_per_stock, sector_to_stock):
    """Compute the average action per sector.

    :param actions_per_stock: DataFrame with actions per stock
    :type actions_per_stock: DataFrame
    :param sector_to_stock: Dictionary mapping sector to stocks
    :type sector_to_stock: dict
    :return: DataFrame with average actions per sector
    :rtype: DataFrame
    """
    actions_per_sector = pd.DataFrame(index=actions_per_stock.index)

    for sector in sector_to_stock:
        actions_per_sector[sector] = actions_per_stock[sector_to_stock[sector]].sum(axis=1)

    return actions_per_sector

def compute_explicability_per_sector(meta_q, indicator, sector_to_stock, stock_to_index):
    """Compute the explicability per sector.

    :param meta_q: DataFrame with explainability values
    :type meta_q: DataFrame
    :param indicator: Indicator to explain
    :type indicator: str
    :param sector_to_stock: Dictionary mapping sector to stocks
    :type sector_to_stock: dict
    :param stock_to_index: Dictionary mapping stock to index
    :type stock_to_index: dict
    :return: DataFrame with explicability per sector
    :rtype: DataFrame
    """
    explicability_per_sector = pd.DataFrame(index=meta_q.date)
    explicability_per_sector = explicability_per_sector[~explicability_per_sector.index.duplicated(keep='first')]

    for sector in sector_to_stock:
        sector_indices = [stock_to_index[stock] for stock in sector_to_stock[sector]]
        explicability_per_sector[sector] = meta_q[(meta_q['feature'] == indicator) & (meta_q['stock'].isin(sector_indices))].groupby('date').mean()['Saliency Map']

    return explicability_per_sector


if __name__ == '__main__':
    meta_q = pd.read_csv('Data/meta_q_0_a2c.csv')
    meta_q = meta_q.loc[:, ~meta_q.columns.str.contains('^Unnamed')]
    print(meta_q)
    actions = pd.read_csv('Data/a2c_actions.csv', index_col='date')


    dow_sectors = DOW_30_TICKER
    del dow_sectors["V"]
    del dow_sectors["AXP"]
    del dow_sectors["DOW"]


    sector_to_stock = {}
    stock_to_index = {}
    for i, (stock, sector) in enumerate(dow_sectors.items()):
        stock_to_index[stock] = i
        if sector not in sector_to_stock:
            sector_to_stock[sector] = []
        sector_to_stock[sector].append(stock)


    #compute_actions_per_sector(actions, sector_to_stock).to_csv('actions_sector.csv')
    print(compute_actions_per_sector(actions, sector_to_stock))
    #compute_explicability_per_sector(meta_q, 'dx_30', sector_to_stock, stock_to_index).to_csv('explicability_sector.csv')
    print(compute_explicability_per_sector(meta_q, 'dx_5', sector_to_stock, stock_to_index))
