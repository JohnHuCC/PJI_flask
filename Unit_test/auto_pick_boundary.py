def Non_ICM_1_morethan_Q(x):
    return {
        'mu(N)_delta_Q_mu(I)': ['>=', 'delta', 'Q'],
        'mu(N)_delta_mu(I)_Q': ['>=', 'delta', 'Q'],
        'mu(N)_Q_delta_mu(I)': ['>=', 'Q', 'delta'],
        'mu(N)_Q_mu(I)_delta': ['>=', 'Q', 'delta'],
        'mu(N)_mu(I)_delta_Q': ['>=', 'delta', 'Q'],
        'mu(N)_mu(I)_Q_delta': ['>=', 'Q', 'delta'],
        'delta_mu(N)_Q_mu(I)': ['NotUsed'],
        'delta_mu(N)_mu(I)_Q': ['NotUsed'],
        'delta_Q_mu(N)_mu(I)': ['>=', 'delta', 'Q'],
        'delta_mu(I)_Q_mu(N)': ['NotUsed'],
        'delta_mu(I)_mu(N)_Q': ['NotUsed'],
        'delta_Q_mu(I)_mu(N)': ['NotUsed'],
        'Q_mu(N)_delta_mu(I)': ['>=', 'Q', 'delta'],
        'Q_mu(N)_mu(I)_delta': ['>=', 'Q', 'delta'],
        'Q_delta_mu(N)_mu(I)': ['>=', 'Q', 'delta'],
        'Q_delta_mu(I)_mu(N)': ['>=', 'delta', 'Q'],
        'Q_mu(I)_mu(N)_delta': ['NotUsed'],
        'Q_mu(I)_delta_mu(N)': ['>=', 'delta', 'Q'],
        'mu(I)_mu(N)_delta_Q': ['NotUsed'],
        'mu(I)_mu(N)_Q_delta': ['NotUsed'],
        'mu(I)_delta_mu(N)_Q': ['NotUsed'],
        'mu(I)_delta_Q_mu(N)': ['NotUsed'],
        'mu(I)_Q_mu(N)_delta': ['NotUsed'],
        'mu(I)_Q_delta_mu(N)': ['>=', 'Q', 'delta'],
    }.get(x, 'Error')


def Non_ICM_1_lessthan_Q(x):
    return {
        'mu(N)_delta_Q_mu(I)': ['NotUsed'],
        'mu(N)_delta_mu(I)_Q': ['<=', 'Q', 'delta'],
        'mu(N)_Q_delta_mu(I)': ['NotUsed'],
        'mu(N)_Q_mu(I)_delta': ['NotUsed'],
        'mu(N)_mu(I)_delta_Q': ['<=', 'Q', 'delta'],
        'mu(N)_mu(I)_Q_delta': ['NotUsed'],
        'delta_mu(N)_Q_mu(I)': ['NotUsed'],
        'delta_mu(N)_mu(I)_Q': ['NotUsed'],
        'delta_Q_mu(N)_mu(I)': ['NotUsed'],
        'delta_mu(I)_Q_mu(N)': ['<=', 'Q', 'delta'],
        'delta_mu(I)_mu(N)_Q': ['<=', 'Q', 'delta'],
        'delta_Q_mu(I)_mu(N)': ['<=', 'Q', 'delta'],
        'Q_mu(N)_delta_mu(I)': ['NotUsed'],
        'Q_mu(N)_mu(I)_delta': ['NotUsed'],
        'Q_delta_mu(N)_mu(I)': ['NotUsed'],
        'Q_delta_mu(I)_mu(N)': ['<=', 'delta', 'Q'],
        'Q_mu(I)_mu(N)_delta': ['<=', 'delta', 'Q'],
        'Q_mu(I)_delta_mu(N)': ['<=', 'delta', 'Q'],
        'mu(I)_mu(N)_delta_Q': ['<=', 'delta', 'Q'],
        'mu(I)_mu(N)_Q_delta': ['NotUsed'],
        'mu(I)_delta_mu(N)_Q': ['<=', 'Q', 'delta'],
        'mu(I)_delta_Q_mu(N)': ['<=', 'Q', 'delta'],
        'mu(I)_Q_mu(N)_delta': ['<=', 'delta', 'Q'],
        'mu(I)_Q_delta_mu(N)': ['<=', 'delta', 'Q'],
    }.get(x, 'Error')


def Non_ICM_0_morethan_Q(x):
    return {
        'mu(N)_delta_Q_mu(I)': ['NotUsed'],
        'mu(N)_delta_mu(I)_Q': ['NotUsed'],
        'mu(N)_Q_delta_mu(I)': ['>=', 'Q', 'delta'],
        'mu(N)_Q_mu(I)_delta': ['NotUsed'],
        'mu(N)_mu(I)_delta_Q': ['NotUsed'],
        'mu(N)_mu(I)_Q_delta': ['NotUsed'],
        'delta_mu(N)_Q_mu(I)': ['>=', 'delta', 'Q'],
        'delta_mu(N)_mu(I)_Q': ['NotUsed'],
        'delta_Q_mu(N)_mu(I)': ['>=', 'delta', 'Q'],
        'delta_mu(I)_Q_mu(N)': ['>=', 'Q', 'delta'],
        'delta_mu(I)_mu(N)_Q': ['>=', 'delta', 'Q'],
        'delta_Q_mu(I)_mu(N)': ['>=', 'delta', 'Q'],
        'Q_mu(N)_delta_mu(I)': ['>=', 'Q', 'delta'],
        'Q_mu(N)_mu(I)_delta': ['NotUsed'],
        'Q_delta_mu(N)_mu(I)': ['>=', 'Q', 'delta'],
        'Q_delta_mu(I)_mu(N)': ['>=', 'Q', 'delta'],
        'Q_mu(I)_mu(N)_delta': ['>=', 'Q', 'delta'],
        'Q_mu(I)_delta_mu(N)': ['>=', 'Q', 'delta'],
        'mu(I)_mu(N)_delta_Q': ['>=', 'delta', 'Q'],
        'mu(I)_mu(N)_Q_delta': ['>=', 'Q', 'delta'],
        'mu(I)_delta_mu(N)_Q': ['>=', 'delta', 'Q'],
        'mu(I)_delta_Q_mu(N)': ['>=', 'delta', 'Q'],
        'mu(I)_Q_mu(N)_delta': ['>=', 'Q', 'delta'],
        'mu(I)_Q_delta_mu(N)': ['>=', 'Q', 'delta'],
    }.get(x, 'Error')


def Non_ICM_0_lessthan_Q(x):
    return {
        'mu(N)_delta_Q_mu(I)': ['<=', 'Q', 'delta'],
        'mu(N)_delta_mu(I)_Q': ['<=', 'Q', 'delta'],
        'mu(N)_Q_delta_mu(I)': ['<=', 'delta', 'Q'],
        'mu(N)_Q_mu(I)_delta': ['<=', 'delta', 'Q'],
        'mu(N)_mu(I)_delta_Q': ['<=', 'Q', 'delta'],
        'mu(N)_mu(I)_Q_delta': ['<=', 'delta', 'Q'],
        'delta_mu(N)_Q_mu(I)': ['<=', 'Q', 'delta'],
        'delta_mu(N)_mu(I)_Q': ['<=', 'Q', 'delta'],
        'delta_Q_mu(N)_mu(I)': ['<=', 'Q', 'delta'],
        'delta_mu(I)_Q_mu(N)': ['NotUsed'],
        'delta_mu(I)_mu(N)_Q': ['NotUsed'],
        'delta_Q_mu(I)_mu(N)': ['NotUsed'],
        'Q_mu(N)_delta_mu(I)': ['<=', 'delta', 'Q'],
        'Q_mu(N)_mu(I)_delta': ['<=', 'delta', 'Q'],
        'Q_delta_mu(N)_mu(I)': ['<=', 'delta', 'Q'],
        'Q_delta_mu(I)_mu(N)': ['<=', 'Q', 'delta'],
        'Q_mu(I)_mu(N)_delta': ['NotUsed'],
        'Q_mu(I)_delta_mu(N)': ['NotUsed'],
        'mu(I)_mu(N)_delta_Q': ['<=', 'Q', 'delta'],
        'mu(I)_mu(N)_Q_delta': ['NotUsed'],
        'mu(I)_delta_mu(N)_Q': ['<=', 'Q', 'delta'],
        'mu(I)_delta_Q_mu(N)': ['<=', 'Q', 'delta'],
        'mu(I)_Q_mu(N)_delta': ['NotUsed'],
        'mu(I)_Q_delta_mu(N)': ['NotUsed'],
    }.get(x, 'Error')

# OK


def ICM_1_morethan_Q(x):
    return {
        'delta_Q_ICM': ['>=', 'delta', 'Q'],
        'delta_ICM_Q': ['NotUsed'],
        'ICM_delta_Q': ['>=', 'delta', 'Q'],
        'ICM_Q_delta': ['>=', 'Q', 'delta'],
        'Q_ICM_delta': ['>=', 'Q', 'delta'],
        'Q_delta_ICM': ['>=', 'Q', 'delta'],

    }.get(x, 'Error')


def ICM_1_lessthan_Q(x):
    return {
        'delta_Q_ICM': ['NotUsed'],
        'delta_ICM_Q': ['NotUsed'],
        'ICM_delta_Q': ['<=', 'Q', 'delta'],
        'ICM_Q_delta': ['NotUsed'],
        'Q_ICM_delta': ['NotUsed'],
        'Q_delta_ICM': ['NotUsed'],

    }.get(x, 'Error')


def ICM_0_morethan_Q(x):
    return {
        'delta_Q_ICM': ['>=', 'delta', 'Q'],
        'delta_ICM_Q': ['NotUsed'],
        'ICM_delta_Q': ['NotUsed'],
        'ICM_Q_delta': ['NotUsed'],
        'Q_ICM_delta': ['NotUsed'],
        'Q_delta_ICM': ['>=', 'Q', 'delta'],

    }.get(x, 'Error')


def ICM_0_lessthan_Q(x):
    return {
        'delta_Q_ICM': ['<=', 'Q', 'delta'],
        'delta_ICM_Q': ['<=', 'Q', 'delta'],
        'ICM_delta_Q': ['<=', 'Q', 'delta'],
        'ICM_Q_delta': ['<=', 'delta', 'Q'],
        'Q_ICM_delta': ['<=', 'delta', 'Q'],
        'Q_delta_ICM': ['<=', 'delta', 'Q'],

    }.get(x, 'Error')
