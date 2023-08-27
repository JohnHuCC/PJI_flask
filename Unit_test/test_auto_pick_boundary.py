import unittest
from auto_pick_boundary import *


def concate_ICM_string(ICM_threshold, Q, delta):
    ICM = {"ICM": ICM_threshold, "Q": Q, "delta": delta}
    ICM_Sorting = sorted(ICM, key=ICM.get)
    concate_ICM_Sorting = '_'.join(ICM_Sorting)
    return concate_ICM_Sorting


def concate_nonICM_string(mu_N, mu_I, delta, Q):
    nonICM = {"mu(N)": mu_N, "mu(I)": mu_I, "delta": delta, "Q": Q}
    nonICM_Sorting = sorted(nonICM, key=nonICM.get)
    concate_nonICM_Sorting = '_'.join(nonICM_Sorting)
    return concate_nonICM_Sorting


class test_ICM_0_morethan_Q(unittest.TestCase):
    def test_delta_Q_ICM(self):
        delta, Q, ICM_threshold = 1, 2, 3
        concate_ICM_Sorting = concate_ICM_string(ICM_threshold, Q, delta)
        result = ICM_0_morethan_Q(concate_ICM_Sorting)
        self.assertEqual(result, ['>=', 'delta', 'Q'])

    def test_delta_ICM_Q(self):
        delta, ICM_threshold, Q = 1, 2, 3
        concate_ICM_Sorting = concate_ICM_string(ICM_threshold, Q, delta)
        result = ICM_0_morethan_Q(concate_ICM_Sorting)
        self.assertEqual(result, ['NotUsed'])

    def test_ICM_delta_Q(self):
        ICM_threshold, delta, Q = 1, 2, 3
        concate_ICM_Sorting = concate_ICM_string(ICM_threshold, Q, delta)
        result = ICM_0_morethan_Q(concate_ICM_Sorting)
        self.assertEqual(result, ['NotUsed'])

    def test_ICM_Q_delta(self):
        ICM_threshold, Q, delta = 1, 2, 3
        concate_ICM_Sorting = concate_ICM_string(ICM_threshold, Q, delta)
        result = ICM_0_morethan_Q(concate_ICM_Sorting)
        self.assertEqual(result, ['NotUsed'])

    def test_Q_ICM_delta(self):
        Q, ICM_threshold, delta = 1, 2, 3
        concate_ICM_Sorting = concate_ICM_string(ICM_threshold, Q, delta)
        result = ICM_0_morethan_Q(concate_ICM_Sorting)
        self.assertEqual(result, ['NotUsed'])

    def test_Q_delta_ICM(self):
        Q, delta, ICM_threshold = 1, 2, 3
        concate_ICM_Sorting = concate_ICM_string(ICM_threshold, Q, delta)
        result = ICM_0_morethan_Q(concate_ICM_Sorting)
        self.assertEqual(result, ['>=', 'Q', 'delta'])


class test_ICM_0_lessthan_Q(unittest.TestCase):
    def test_delta_Q_ICM(self):
        delta, Q, ICM_threshold = 1, 2, 3
        concate_ICM_Sorting = concate_ICM_string(ICM_threshold, Q, delta)
        result = ICM_0_lessthan_Q(concate_ICM_Sorting)
        self.assertEqual(result, ['<=', 'Q', 'delta'])

    def test_delta_ICM_Q(self):
        delta, ICM_threshold, Q = 1, 2, 3
        concate_ICM_Sorting = concate_ICM_string(ICM_threshold, Q, delta)
        result = ICM_0_lessthan_Q(concate_ICM_Sorting)
        self.assertEqual(result, ['<=', 'Q', 'delta'])

    def test_ICM_delta_Q(self):
        ICM_threshold, delta, Q = 1, 2, 3
        concate_ICM_Sorting = concate_ICM_string(ICM_threshold, Q, delta)
        result = ICM_0_lessthan_Q(concate_ICM_Sorting)
        self.assertEqual(result, ['<=', 'Q', 'delta'])

    def test_ICM_Q_delta(self):
        ICM_threshold, Q, delta = 1, 2, 3
        concate_ICM_Sorting = concate_ICM_string(ICM_threshold, Q, delta)
        result = ICM_0_lessthan_Q(concate_ICM_Sorting)
        self.assertEqual(result, ['<=', 'delta', 'Q'])

    def test_Q_ICM_delta(self):
        Q, ICM_threshold, delta = 1, 2, 3
        concate_ICM_Sorting = concate_ICM_string(ICM_threshold, Q, delta)
        result = ICM_0_lessthan_Q(concate_ICM_Sorting)
        self.assertEqual(result, ['<=', 'delta', 'Q'])

    def test_Q_delta_ICM(self):
        Q, delta, ICM_threshold = 1, 2, 3
        concate_ICM_Sorting = concate_ICM_string(ICM_threshold, Q, delta)
        result = ICM_0_lessthan_Q(concate_ICM_Sorting)
        self.assertEqual(result, ['<=', 'delta', 'Q'])


class test_ICM_1_morethan_Q(unittest.TestCase):
    def test_delta_Q_ICM(self):
        delta, Q, ICM_threshold = 1, 2, 3
        concate_ICM_Sorting = concate_ICM_string(ICM_threshold, Q, delta)
        result = ICM_1_morethan_Q(concate_ICM_Sorting)
        self.assertEqual(result, ['>=', 'delta', 'Q'])

    def test_delta_ICM_Q(self):
        delta, ICM_threshold, Q = 1, 2, 3
        concate_ICM_Sorting = concate_ICM_string(ICM_threshold, Q, delta)
        result = ICM_1_morethan_Q(concate_ICM_Sorting)
        self.assertEqual(result, ['NotUsed'])

    def test_ICM_delta_Q(self):
        ICM_threshold, delta, Q = 1, 2, 3
        concate_ICM_Sorting = concate_ICM_string(ICM_threshold, Q, delta)
        result = ICM_1_morethan_Q(concate_ICM_Sorting)
        self.assertEqual(result, ['>=', 'delta', 'Q'])

    def test_ICM_Q_delta(self):
        ICM_threshold, Q, delta = 1, 2, 3
        concate_ICM_Sorting = concate_ICM_string(ICM_threshold, Q, delta)
        result = ICM_1_morethan_Q(concate_ICM_Sorting)
        self.assertEqual(result, ['>=', 'Q', 'delta'])

    def test_Q_ICM_delta(self):
        Q, ICM_threshold, delta = 1, 2, 3
        concate_ICM_Sorting = concate_ICM_string(ICM_threshold, Q, delta)
        result = ICM_1_morethan_Q(concate_ICM_Sorting)
        self.assertEqual(result, ['>=', 'Q', 'delta'])

    def test_Q_delta_ICM(self):
        Q, delta, ICM_threshold = 1, 2, 3
        concate_ICM_Sorting = concate_ICM_string(ICM_threshold, Q, delta)
        result = ICM_1_morethan_Q(concate_ICM_Sorting)
        self.assertEqual(result, ['>=', 'Q', 'delta'])


class test_ICM_1_lessthan_Q(unittest.TestCase):
    def test_delta_Q_ICM(self):
        delta, Q, ICM_threshold = 1, 2, 3
        concate_ICM_Sorting = concate_ICM_string(ICM_threshold, Q, delta)
        result = ICM_1_lessthan_Q(concate_ICM_Sorting)
        self.assertEqual(result, ['NotUsed'])

    def test_delta_ICM_Q(self):
        delta, ICM_threshold, Q = 1, 2, 3
        concate_ICM_Sorting = concate_ICM_string(ICM_threshold, Q, delta)
        result = ICM_1_lessthan_Q(concate_ICM_Sorting)
        self.assertEqual(result, ['NotUsed'])

    def test_ICM_delta_Q(self):
        ICM_threshold, delta, Q = 1, 2, 3
        concate_ICM_Sorting = concate_ICM_string(ICM_threshold, Q, delta)
        result = ICM_1_lessthan_Q(concate_ICM_Sorting)
        self.assertEqual(result, ['<=', 'Q', 'delta'])

    def test_ICM_Q_delta(self):
        ICM_threshold, Q, delta = 1, 2, 3
        concate_ICM_Sorting = concate_ICM_string(ICM_threshold, Q, delta)
        result = ICM_1_lessthan_Q(concate_ICM_Sorting)
        self.assertEqual(result, ['NotUsed'])

    def test_Q_ICM_delta(self):
        Q, ICM_threshold, delta = 1, 2, 3
        concate_ICM_Sorting = concate_ICM_string(ICM_threshold, Q, delta)
        result = ICM_1_lessthan_Q(concate_ICM_Sorting)
        self.assertEqual(result, ['NotUsed'])

    def test_Q_delta_ICM(self):
        Q, delta, ICM_threshold = 1, 2, 3
        concate_ICM_Sorting = concate_ICM_string(ICM_threshold, Q, delta)
        result = ICM_1_lessthan_Q(concate_ICM_Sorting)
        self.assertEqual(result, ['NotUsed'])


class test_nonICM_0_morethan_Q(unittest.TestCase):
    def test_N_delta_Q_I(self):
        mu_N, delta, Q, mu_I = 1, 2, 3, 4
        concate_nonICM_Sorting = concate_nonICM_string(mu_N, mu_I, delta, Q)
        result = Non_ICM_0_morethan_Q(concate_nonICM_Sorting)
        self.assertEqual(result, ['NotUsed'])

    def test_N_delta_I_Q(self):
        mu_N, delta, mu_I, Q = 1, 2, 3, 4
        concate_nonICM_Sorting = concate_nonICM_string(mu_N, mu_I, delta, Q)
        result = Non_ICM_0_morethan_Q(concate_nonICM_Sorting)
        self.assertEqual(result, ['NotUsed'])

    def test_N_Q_delta_I(self):
        mu_N, Q, delta, mu_I = 1, 2, 3, 4
        concate_nonICM_Sorting = concate_nonICM_string(mu_N, mu_I, delta, Q)
        result = Non_ICM_0_morethan_Q(concate_nonICM_Sorting)
        self.assertEqual(result, ['>=', 'Q', 'delta'])

    def test_N_Q_I_delta(self):
        mu_N, Q, mu_I, delta = 1, 2, 3, 4
        concate_nonICM_Sorting = concate_nonICM_string(mu_N, mu_I, delta, Q)
        result = Non_ICM_0_morethan_Q(concate_nonICM_Sorting)
        self.assertEqual(result, ['NotUsed'])

    def test_N_I_delta_Q(self):
        mu_N, mu_I, delta, Q = 1, 2, 3, 4
        concate_nonICM_Sorting = concate_nonICM_string(mu_N, mu_I, delta, Q)
        result = Non_ICM_0_morethan_Q(concate_nonICM_Sorting)
        self.assertEqual(result, ['NotUsed'])

    def test_N_I_Q_delta(self):
        mu_N, mu_I, Q, delta = 1, 2, 3, 4
        concate_nonICM_Sorting = concate_nonICM_string(mu_N, mu_I, delta, Q)
        result = Non_ICM_0_morethan_Q(concate_nonICM_Sorting)
        self.assertEqual(result, ['NotUsed'])

    def test_delta_N_Q_I(self):
        delta, mu_N, Q, mu_I = 1, 2, 3, 4
        concate_nonICM_Sorting = concate_nonICM_string(mu_N, mu_I, delta, Q)
        result = Non_ICM_0_morethan_Q(concate_nonICM_Sorting)
        self.assertEqual(result, ['>=', 'delta', 'Q'])

    def test_delta_N_I_Q(self):
        delta, mu_N, mu_I, Q = 1, 2, 3, 4
        concate_nonICM_Sorting = concate_nonICM_string(mu_N, mu_I, delta, Q)
        result = Non_ICM_0_morethan_Q(concate_nonICM_Sorting)
        self.assertEqual(result, ['NotUsed'])

    def test_delta_Q_N_I(self):
        delta, Q, mu_N, mu_I = 1, 2, 3, 4
        concate_nonICM_Sorting = concate_nonICM_string(mu_N, mu_I, delta, Q)
        result = Non_ICM_0_morethan_Q(concate_nonICM_Sorting)
        self.assertEqual(result, ['>=', 'delta', 'Q'])

    def test_delta_I_Q_N(self):
        delta, mu_I, Q, mu_N = 1, 2, 3, 4
        concate_nonICM_Sorting = concate_nonICM_string(mu_N, mu_I, delta, Q)
        result = Non_ICM_0_morethan_Q(concate_nonICM_Sorting)
        self.assertEqual(result, ['>=', 'Q', 'delta'])

    def test_delta_I_N_Q(self):
        delta, mu_I, mu_N, Q = 1, 2, 3, 4
        concate_nonICM_Sorting = concate_nonICM_string(mu_N, mu_I, delta, Q)
        result = Non_ICM_0_morethan_Q(concate_nonICM_Sorting)
        self.assertEqual(result, ['>=', 'delta', 'Q'])

    def test_delta_Q_I_N(self):
        delta, Q, mu_I, mu_N = 1, 2, 3, 4
        concate_nonICM_Sorting = concate_nonICM_string(mu_N, mu_I, delta, Q)
        result = Non_ICM_0_morethan_Q(concate_nonICM_Sorting)
        self.assertEqual(result, ['>=', 'delta', 'Q'])

    def test_Q_N_delta_I(self):
        Q, mu_N, delta, mu_I = 1, 2, 3, 4
        concate_nonICM_Sorting = concate_nonICM_string(mu_N, mu_I, delta, Q)
        result = Non_ICM_0_morethan_Q(concate_nonICM_Sorting)
        self.assertEqual(result, ['>=', 'Q', 'delta'])

    def test_Q_N_I_delta(self):
        Q, mu_N, mu_I, delta = 1, 2, 3, 4
        concate_nonICM_Sorting = concate_nonICM_string(mu_N, mu_I, delta, Q)
        result = Non_ICM_0_morethan_Q(concate_nonICM_Sorting)
        self.assertEqual(result, ['NotUsed'])

    def test_Q_delta_N_I(self):
        Q, delta, mu_N, mu_I = 1, 2, 3, 4
        concate_nonICM_Sorting = concate_nonICM_string(mu_N, mu_I, delta, Q)
        result = Non_ICM_0_morethan_Q(concate_nonICM_Sorting)
        self.assertEqual(result, ['>=', 'Q', 'delta'])

    def test_Q_delta_I_N(self):
        Q, delta, mu_I, mu_N = 1, 2, 3, 4
        concate_nonICM_Sorting = concate_nonICM_string(mu_N, mu_I, delta, Q)
        result = Non_ICM_0_morethan_Q(concate_nonICM_Sorting)
        self.assertEqual(result, ['>=', 'Q', 'delta'])

    def test_Q_I_N_delta(self):
        Q, mu_I, mu_N, delta = 1, 2, 3, 4
        concate_nonICM_Sorting = concate_nonICM_string(mu_N, mu_I, delta, Q)
        result = Non_ICM_0_morethan_Q(concate_nonICM_Sorting)
        self.assertEqual(result, ['>=', 'Q', 'delta'])

    def test_Q_I_delta_N(self):
        Q, mu_I, delta, mu_N = 1, 2, 3, 4
        concate_nonICM_Sorting = concate_nonICM_string(mu_N, mu_I, delta, Q)
        result = Non_ICM_0_morethan_Q(concate_nonICM_Sorting)
        self.assertEqual(result, ['>=', 'Q', 'delta'])

    def test_I_N_delta_Q(self):
        mu_I, mu_N, delta, Q = 1, 2, 3, 4
        concate_nonICM_Sorting = concate_nonICM_string(mu_N, mu_I, delta, Q)
        result = Non_ICM_0_morethan_Q(concate_nonICM_Sorting)
        self.assertEqual(result, ['>=', 'delta', 'Q'])

    def test_I_N_Q_delta(self):
        mu_I, mu_N, Q, delta = 1, 2, 3, 4
        concate_nonICM_Sorting = concate_nonICM_string(mu_N, mu_I, delta, Q)
        result = Non_ICM_0_morethan_Q(concate_nonICM_Sorting)
        self.assertEqual(result, ['>=', 'Q', 'delta'])

    def test_I_delta_N_Q(self):
        mu_I, delta, mu_N, Q = 1, 2, 3, 4
        concate_nonICM_Sorting = concate_nonICM_string(mu_N, mu_I, delta, Q)
        result = Non_ICM_0_morethan_Q(concate_nonICM_Sorting)
        self.assertEqual(result, ['>=', 'delta', 'Q'])

    def test_I_delta_Q_N(self):
        mu_I, delta, Q, mu_N = 1, 2, 3, 4
        concate_nonICM_Sorting = concate_nonICM_string(mu_N, mu_I, delta, Q)
        result = Non_ICM_0_morethan_Q(concate_nonICM_Sorting)
        self.assertEqual(result, ['>=', 'delta', 'Q'])

    def test_I_Q_N_delta(self):
        mu_I, Q, mu_N, delta = 1, 2, 3, 4
        concate_nonICM_Sorting = concate_nonICM_string(mu_N, mu_I, delta, Q)
        result = Non_ICM_0_morethan_Q(concate_nonICM_Sorting)
        self.assertEqual(result, ['>=', 'Q', 'delta'])

    def test_I_Q_delta_N(self):
        mu_I, Q, delta, mu_N = 1, 2, 3, 4
        concate_nonICM_Sorting = concate_nonICM_string(mu_N, mu_I, delta, Q)
        result = Non_ICM_0_morethan_Q(concate_nonICM_Sorting)
        self.assertEqual(result, ['>=', 'Q', 'delta'])


class test_nonICM_0_lessthan_Q(unittest.TestCase):
    def test_N_delta_Q_I(self):
        mu_N, delta, Q, mu_I = 1, 2, 3, 4
        concate_nonICM_Sorting = concate_nonICM_string(mu_N, mu_I, delta, Q)
        result = Non_ICM_0_lessthan_Q(concate_nonICM_Sorting)
        self.assertEqual(result, ['<=', 'Q', 'delta'])

    def test_N_delta_I_Q(self):
        mu_N, delta, mu_I, Q = 1, 2, 3, 4
        concate_nonICM_Sorting = concate_nonICM_string(mu_N, mu_I, delta, Q)
        result = Non_ICM_0_lessthan_Q(concate_nonICM_Sorting)
        self.assertEqual(result, ['<=', 'Q', 'delta'])

    def test_N_Q_delta_I(self):
        mu_N, Q, delta, mu_I = 1, 2, 3, 4
        concate_nonICM_Sorting = concate_nonICM_string(mu_N, mu_I, delta, Q)
        result = Non_ICM_0_lessthan_Q(concate_nonICM_Sorting)
        self.assertEqual(result, ['<=', 'delta', 'Q'])

    def test_N_Q_I_delta(self):
        mu_N, Q, mu_I, delta = 1, 2, 3, 4
        concate_nonICM_Sorting = concate_nonICM_string(mu_N, mu_I, delta, Q)
        result = Non_ICM_0_lessthan_Q(concate_nonICM_Sorting)
        self.assertEqual(result, ['<=', 'delta', 'Q'])

    def test_N_I_delta_Q(self):
        mu_N, mu_I, delta, Q = 1, 2, 3, 4
        concate_nonICM_Sorting = concate_nonICM_string(mu_N, mu_I, delta, Q)
        result = Non_ICM_0_lessthan_Q(concate_nonICM_Sorting)
        self.assertEqual(result, ['<=', 'Q', 'delta'])

    def test_N_I_Q_delta(self):
        mu_N, mu_I, Q, delta = 1, 2, 3, 4
        concate_nonICM_Sorting = concate_nonICM_string(mu_N, mu_I, delta, Q)
        result = Non_ICM_0_lessthan_Q(concate_nonICM_Sorting)
        self.assertEqual(result, ['<=', 'delta', 'Q'])

    def test_delta_N_Q_I(self):
        delta, mu_N, Q, mu_I = 1, 2, 3, 4
        concate_nonICM_Sorting = concate_nonICM_string(mu_N, mu_I, delta, Q)
        result = Non_ICM_0_lessthan_Q(concate_nonICM_Sorting)
        self.assertEqual(result, ['<=', 'Q', 'delta'])

    def test_delta_N_I_Q(self):
        delta, mu_N, mu_I, Q = 1, 2, 3, 4
        concate_nonICM_Sorting = concate_nonICM_string(mu_N, mu_I, delta, Q)
        result = Non_ICM_0_lessthan_Q(concate_nonICM_Sorting)
        self.assertEqual(result, ['<=', 'Q', 'delta'])

    def test_delta_Q_N_I(self):
        delta, Q, mu_N, mu_I = 1, 2, 3, 4
        concate_nonICM_Sorting = concate_nonICM_string(mu_N, mu_I, delta, Q)
        result = Non_ICM_0_lessthan_Q(concate_nonICM_Sorting)
        self.assertEqual(result, ['<=', 'Q', 'delta'])

    def test_delta_I_Q_N(self):
        delta, mu_I, Q, mu_N = 1, 2, 3, 4
        concate_nonICM_Sorting = concate_nonICM_string(mu_N, mu_I, delta, Q)
        result = Non_ICM_0_lessthan_Q(concate_nonICM_Sorting)
        self.assertEqual(result, ['NotUsed'])

    def test_delta_I_N_Q(self):
        delta, mu_I, mu_N, Q = 1, 2, 3, 4
        concate_nonICM_Sorting = concate_nonICM_string(mu_N, mu_I, delta, Q)
        result = Non_ICM_0_lessthan_Q(concate_nonICM_Sorting)
        self.assertEqual(result, ['NotUsed'])

    def test_delta_Q_I_N(self):
        delta, Q, mu_I, mu_N = 1, 2, 3, 4
        concate_nonICM_Sorting = concate_nonICM_string(mu_N, mu_I, delta, Q)
        result = Non_ICM_0_lessthan_Q(concate_nonICM_Sorting)
        self.assertEqual(result, ['NotUsed'])

    def test_Q_N_delta_I(self):
        Q, mu_N, delta, mu_I = 1, 2, 3, 4
        concate_nonICM_Sorting = concate_nonICM_string(mu_N, mu_I, delta, Q)
        result = Non_ICM_0_lessthan_Q(concate_nonICM_Sorting)
        self.assertEqual(result, ['<=', 'delta', 'Q'])

    def test_Q_N_I_delta(self):
        Q, mu_N, mu_I, delta = 1, 2, 3, 4
        concate_nonICM_Sorting = concate_nonICM_string(mu_N, mu_I, delta, Q)
        result = Non_ICM_0_lessthan_Q(concate_nonICM_Sorting)
        self.assertEqual(result, ['<=', 'delta', 'Q'])

    def test_Q_delta_N_I(self):
        Q, delta, mu_N, mu_I = 1, 2, 3, 4
        concate_nonICM_Sorting = concate_nonICM_string(mu_N, mu_I, delta, Q)
        result = Non_ICM_0_lessthan_Q(concate_nonICM_Sorting)
        self.assertEqual(result, ['<=', 'delta', 'Q'])

    def test_Q_delta_I_N(self):
        Q, delta, mu_I, mu_N = 1, 2, 3, 4
        concate_nonICM_Sorting = concate_nonICM_string(mu_N, mu_I, delta, Q)
        result = Non_ICM_0_lessthan_Q(concate_nonICM_Sorting)
        self.assertEqual(result, ['<=', 'Q', 'delta'])

    def test_Q_I_N_delta(self):
        Q, mu_I, mu_N, delta = 1, 2, 3, 4
        concate_nonICM_Sorting = concate_nonICM_string(mu_N, mu_I, delta, Q)
        result = Non_ICM_0_lessthan_Q(concate_nonICM_Sorting)
        self.assertEqual(result, ['NotUsed'])

    def test_Q_I_delta_N(self):
        Q, mu_I, delta, mu_N = 1, 2, 3, 4
        concate_nonICM_Sorting = concate_nonICM_string(mu_N, mu_I, delta, Q)
        result = Non_ICM_0_lessthan_Q(concate_nonICM_Sorting)
        self.assertEqual(result, ['NotUsed'])

    def test_I_N_delta_Q(self):
        mu_I, mu_N, delta, Q = 1, 2, 3, 4
        concate_nonICM_Sorting = concate_nonICM_string(mu_N, mu_I, delta, Q)
        result = Non_ICM_0_lessthan_Q(concate_nonICM_Sorting)
        self.assertEqual(result, ['<=', 'Q', 'delta'])

    def test_I_N_Q_delta(self):
        mu_I, mu_N, Q, delta = 1, 2, 3, 4
        concate_nonICM_Sorting = concate_nonICM_string(mu_N, mu_I, delta, Q)
        result = Non_ICM_0_lessthan_Q(concate_nonICM_Sorting)
        self.assertEqual(result, ['NotUsed'])

    def test_I_delta_N_Q(self):
        mu_I, delta, mu_N, Q = 1, 2, 3, 4
        concate_nonICM_Sorting = concate_nonICM_string(mu_N, mu_I, delta, Q)
        result = Non_ICM_0_lessthan_Q(concate_nonICM_Sorting)
        self.assertEqual(result, ['<=', 'Q', 'delta'])

    def test_I_delta_Q_N(self):
        mu_I, delta, Q, mu_N = 1, 2, 3, 4
        concate_nonICM_Sorting = concate_nonICM_string(mu_N, mu_I, delta, Q)
        result = Non_ICM_0_lessthan_Q(concate_nonICM_Sorting)
        self.assertEqual(result, ['<=', 'Q', 'delta'])

    def test_I_Q_N_delta(self):
        mu_I, Q, mu_N, delta = 1, 2, 3, 4
        concate_nonICM_Sorting = concate_nonICM_string(mu_N, mu_I, delta, Q)
        result = Non_ICM_0_lessthan_Q(concate_nonICM_Sorting)
        self.assertEqual(result, ['NotUsed'])

    def test_I_Q_delta_N(self):
        mu_I, Q, delta, mu_N = 1, 2, 3, 4
        concate_nonICM_Sorting = concate_nonICM_string(mu_N, mu_I, delta, Q)
        result = Non_ICM_0_lessthan_Q(concate_nonICM_Sorting)
        self.assertEqual(result, ['NotUsed'])


class test_nonICM_1_morethan_Q(unittest.TestCase):
    def test_N_delta_Q_I(self):
        mu_N, delta, Q, mu_I = 1, 2, 3, 4
        concate_nonICM_Sorting = concate_nonICM_string(mu_N, mu_I, delta, Q)
        result = Non_ICM_1_morethan_Q(concate_nonICM_Sorting)
        self.assertEqual(result, ['>=', 'delta', 'Q'])

    def test_N_delta_I_Q(self):
        mu_N, delta, mu_I, Q = 1, 2, 3, 4
        concate_nonICM_Sorting = concate_nonICM_string(mu_N, mu_I, delta, Q)
        result = Non_ICM_1_morethan_Q(concate_nonICM_Sorting)
        self.assertEqual(result, ['>=', 'delta', 'Q'])

    def test_N_Q_delta_I(self):
        mu_N, Q, delta, mu_I = 1, 2, 3, 4
        concate_nonICM_Sorting = concate_nonICM_string(mu_N, mu_I, delta, Q)
        result = Non_ICM_1_morethan_Q(concate_nonICM_Sorting)
        self.assertEqual(result, ['>=', 'Q', 'delta'])

    def test_N_Q_I_delta(self):
        mu_N, Q, mu_I, delta = 1, 2, 3, 4
        concate_nonICM_Sorting = concate_nonICM_string(mu_N, mu_I, delta, Q)
        result = Non_ICM_1_morethan_Q(concate_nonICM_Sorting)
        self.assertEqual(result, ['>=', 'Q', 'delta'])

    def test_N_I_delta_Q(self):
        mu_N, mu_I, delta, Q = 1, 2, 3, 4
        concate_nonICM_Sorting = concate_nonICM_string(mu_N, mu_I, delta, Q)
        result = Non_ICM_1_morethan_Q(concate_nonICM_Sorting)
        self.assertEqual(result, ['>=', 'delta', 'Q'])

    def test_N_I_Q_delta(self):
        mu_N, mu_I, Q, delta = 1, 2, 3, 4
        concate_nonICM_Sorting = concate_nonICM_string(mu_N, mu_I, delta, Q)
        result = Non_ICM_1_morethan_Q(concate_nonICM_Sorting)
        self.assertEqual(result, ['>=', 'Q', 'delta'])

    def test_delta_N_Q_I(self):
        delta, mu_N, Q, mu_I = 1, 2, 3, 4
        concate_nonICM_Sorting = concate_nonICM_string(mu_N, mu_I, delta, Q)
        result = Non_ICM_1_morethan_Q(concate_nonICM_Sorting)
        self.assertEqual(result, ['NotUsed'])

    def test_delta_N_I_Q(self):
        delta, mu_N, mu_I, Q = 1, 2, 3, 4
        concate_nonICM_Sorting = concate_nonICM_string(mu_N, mu_I, delta, Q)
        result = Non_ICM_1_morethan_Q(concate_nonICM_Sorting)
        self.assertEqual(result, ['NotUsed'])

    def test_delta_Q_N_I(self):
        delta, Q, mu_N, mu_I = 1, 2, 3, 4
        concate_nonICM_Sorting = concate_nonICM_string(mu_N, mu_I, delta, Q)
        result = Non_ICM_1_morethan_Q(concate_nonICM_Sorting)
        self.assertEqual(result, ['>=', 'delta', 'Q'])

    def test_delta_I_Q_N(self):
        delta, mu_I, Q, mu_N = 1, 2, 3, 4
        concate_nonICM_Sorting = concate_nonICM_string(mu_N, mu_I, delta, Q)
        result = Non_ICM_1_morethan_Q(concate_nonICM_Sorting)
        self.assertEqual(result, ['NotUsed'])

    def test_delta_I_N_Q(self):
        delta, mu_I, mu_N, Q = 1, 2, 3, 4
        concate_nonICM_Sorting = concate_nonICM_string(mu_N, mu_I, delta, Q)
        result = Non_ICM_1_morethan_Q(concate_nonICM_Sorting)
        self.assertEqual(result, ['NotUsed'])

    def test_delta_Q_I_N(self):
        delta, Q, mu_I, mu_N = 1, 2, 3, 4
        concate_nonICM_Sorting = concate_nonICM_string(mu_N, mu_I, delta, Q)
        result = Non_ICM_1_morethan_Q(concate_nonICM_Sorting)
        self.assertEqual(result, ['NotUsed'])

    def test_Q_N_delta_I(self):
        Q, mu_N, delta, mu_I = 1, 2, 3, 4
        concate_nonICM_Sorting = concate_nonICM_string(mu_N, mu_I, delta, Q)
        result = Non_ICM_1_morethan_Q(concate_nonICM_Sorting)
        self.assertEqual(result, ['>=', 'Q', 'delta'])

    def test_Q_N_I_delta(self):
        Q, mu_N, mu_I, delta = 1, 2, 3, 4
        concate_nonICM_Sorting = concate_nonICM_string(mu_N, mu_I, delta, Q)
        result = Non_ICM_1_morethan_Q(concate_nonICM_Sorting)
        self.assertEqual(result, ['>=', 'Q', 'delta'])

    def test_Q_delta_N_I(self):
        Q, delta, mu_N, mu_I = 1, 2, 3, 4
        concate_nonICM_Sorting = concate_nonICM_string(mu_N, mu_I, delta, Q)
        result = Non_ICM_1_morethan_Q(concate_nonICM_Sorting)
        self.assertEqual(result, ['>=', 'Q', 'delta'])

    def test_Q_delta_I_N(self):
        Q, delta, mu_I, mu_N = 1, 2, 3, 4
        concate_nonICM_Sorting = concate_nonICM_string(mu_N, mu_I, delta, Q)
        result = Non_ICM_1_morethan_Q(concate_nonICM_Sorting)
        self.assertEqual(result, ['>=', 'delta', 'Q'])

    def test_Q_I_N_delta(self):
        Q, mu_I, mu_N, delta = 1, 2, 3, 4
        concate_nonICM_Sorting = concate_nonICM_string(mu_N, mu_I, delta, Q)
        result = Non_ICM_1_morethan_Q(concate_nonICM_Sorting)
        self.assertEqual(result, ['NotUsed'])

    def test_Q_I_delta_N(self):
        Q, mu_I, delta, mu_N = 1, 2, 3, 4
        concate_nonICM_Sorting = concate_nonICM_string(mu_N, mu_I, delta, Q)
        result = Non_ICM_1_morethan_Q(concate_nonICM_Sorting)
        self.assertEqual(result, ['>=', 'delta', 'Q'])

    def test_I_N_delta_Q(self):
        mu_I, mu_N, delta, Q = 1, 2, 3, 4
        concate_nonICM_Sorting = concate_nonICM_string(mu_N, mu_I, delta, Q)
        result = Non_ICM_1_morethan_Q(concate_nonICM_Sorting)
        self.assertEqual(result, ['NotUsed'])

    def test_I_N_Q_delta(self):
        mu_I, mu_N, Q, delta = 1, 2, 3, 4
        concate_nonICM_Sorting = concate_nonICM_string(mu_N, mu_I, delta, Q)
        result = Non_ICM_1_morethan_Q(concate_nonICM_Sorting)
        self.assertEqual(result, ['NotUsed'])

    def test_I_delta_N_Q(self):
        mu_I, delta, mu_N, Q = 1, 2, 3, 4
        concate_nonICM_Sorting = concate_nonICM_string(mu_N, mu_I, delta, Q)
        result = Non_ICM_1_morethan_Q(concate_nonICM_Sorting)
        self.assertEqual(result, ['NotUsed'])

    def test_I_delta_Q_N(self):
        mu_I, delta, Q, mu_N = 1, 2, 3, 4
        concate_nonICM_Sorting = concate_nonICM_string(mu_N, mu_I, delta, Q)
        result = Non_ICM_1_morethan_Q(concate_nonICM_Sorting)
        self.assertEqual(result, ['NotUsed'])

    def test_I_Q_N_delta(self):
        mu_I, Q, mu_N, delta = 1, 2, 3, 4
        concate_nonICM_Sorting = concate_nonICM_string(mu_N, mu_I, delta, Q)
        result = Non_ICM_1_morethan_Q(concate_nonICM_Sorting)
        self.assertEqual(result, ['NotUsed'])

    def test_I_Q_delta_N(self):
        mu_I, Q, delta, mu_N = 1, 2, 3, 4
        concate_nonICM_Sorting = concate_nonICM_string(mu_N, mu_I, delta, Q)
        result = Non_ICM_1_morethan_Q(concate_nonICM_Sorting)
        self.assertEqual(result, ['>=', 'Q', 'delta'])


class test_nonICM_1_lessthan_Q(unittest.TestCase):
    def test_N_delta_Q_I(self):
        mu_N, delta, Q, mu_I = 1, 2, 3, 4
        concate_nonICM_Sorting = concate_nonICM_string(mu_N, mu_I, delta, Q)
        result = Non_ICM_1_lessthan_Q(concate_nonICM_Sorting)
        self.assertEqual(result, ['NotUsed'])

    def test_N_delta_I_Q(self):
        mu_N, delta, mu_I, Q = 1, 2, 3, 4
        concate_nonICM_Sorting = concate_nonICM_string(mu_N, mu_I, delta, Q)
        result = Non_ICM_1_lessthan_Q(concate_nonICM_Sorting)
        self.assertEqual(result, ['<=', 'Q', 'delta'])

    def test_N_Q_delta_I(self):
        mu_N, Q, delta, mu_I = 1, 2, 3, 4
        concate_nonICM_Sorting = concate_nonICM_string(mu_N, mu_I, delta, Q)
        result = Non_ICM_1_lessthan_Q(concate_nonICM_Sorting)
        self.assertEqual(result, ['NotUsed'])

    def test_N_Q_I_delta(self):
        mu_N, Q, mu_I, delta = 1, 2, 3, 4
        concate_nonICM_Sorting = concate_nonICM_string(mu_N, mu_I, delta, Q)
        result = Non_ICM_1_lessthan_Q(concate_nonICM_Sorting)
        self.assertEqual(result, ['NotUsed'])

    def test_N_I_delta_Q(self):
        mu_N, mu_I, delta, Q = 1, 2, 3, 4
        concate_nonICM_Sorting = concate_nonICM_string(mu_N, mu_I, delta, Q)
        result = Non_ICM_1_lessthan_Q(concate_nonICM_Sorting)
        self.assertEqual(result, ['<=', 'Q', 'delta'])

    def test_N_I_Q_delta(self):
        mu_N, mu_I, Q, delta = 1, 2, 3, 4
        concate_nonICM_Sorting = concate_nonICM_string(mu_N, mu_I, delta, Q)
        result = Non_ICM_1_lessthan_Q(concate_nonICM_Sorting)
        self.assertEqual(result, ['NotUsed'])

    def test_delta_N_Q_I(self):
        delta, mu_N, Q, mu_I = 1, 2, 3, 4
        concate_nonICM_Sorting = concate_nonICM_string(mu_N, mu_I, delta, Q)
        result = Non_ICM_1_lessthan_Q(concate_nonICM_Sorting)
        self.assertEqual(result, ['NotUsed'])

    def test_delta_N_I_Q(self):
        delta, mu_N, mu_I, Q = 1, 2, 3, 4
        concate_nonICM_Sorting = concate_nonICM_string(mu_N, mu_I, delta, Q)
        result = Non_ICM_1_lessthan_Q(concate_nonICM_Sorting)
        self.assertEqual(result, ['NotUsed'])

    def test_delta_Q_N_I(self):
        delta, Q, mu_N, mu_I = 1, 2, 3, 4
        concate_nonICM_Sorting = concate_nonICM_string(mu_N, mu_I, delta, Q)
        result = Non_ICM_1_lessthan_Q(concate_nonICM_Sorting)
        self.assertEqual(result, ['NotUsed'])

    def test_delta_I_Q_N(self):
        delta, mu_I, Q, mu_N = 1, 2, 3, 4
        concate_nonICM_Sorting = concate_nonICM_string(mu_N, mu_I, delta, Q)
        result = Non_ICM_1_lessthan_Q(concate_nonICM_Sorting)
        self.assertEqual(result, ['<=', 'Q', 'delta'])

    def test_delta_I_N_Q(self):
        delta, mu_I, mu_N, Q = 1, 2, 3, 4
        concate_nonICM_Sorting = concate_nonICM_string(mu_N, mu_I, delta, Q)
        result = Non_ICM_1_lessthan_Q(concate_nonICM_Sorting)
        self.assertEqual(result, ['<=', 'Q', 'delta'])

    def test_delta_Q_I_N(self):
        delta, Q, mu_I, mu_N = 1, 2, 3, 4
        concate_nonICM_Sorting = concate_nonICM_string(mu_N, mu_I, delta, Q)
        result = Non_ICM_1_lessthan_Q(concate_nonICM_Sorting)
        self.assertEqual(result, ['<=', 'Q', 'delta'])

    def test_Q_N_delta_I(self):
        Q, mu_N, delta, mu_I = 1, 2, 3, 4
        concate_nonICM_Sorting = concate_nonICM_string(mu_N, mu_I, delta, Q)
        result = Non_ICM_1_lessthan_Q(concate_nonICM_Sorting)
        self.assertEqual(result, ['NotUsed'])

    def test_Q_N_I_delta(self):
        Q, mu_N, mu_I, delta = 1, 2, 3, 4
        concate_nonICM_Sorting = concate_nonICM_string(mu_N, mu_I, delta, Q)
        result = Non_ICM_1_lessthan_Q(concate_nonICM_Sorting)
        self.assertEqual(result, ['NotUsed'])

    def test_Q_delta_N_I(self):
        Q, delta, mu_N, mu_I = 1, 2, 3, 4
        concate_nonICM_Sorting = concate_nonICM_string(mu_N, mu_I, delta, Q)
        result = Non_ICM_1_lessthan_Q(concate_nonICM_Sorting)
        self.assertEqual(result, ['NotUsed'])

    def test_Q_delta_I_N(self):
        Q, delta, mu_I, mu_N = 1, 2, 3, 4
        concate_nonICM_Sorting = concate_nonICM_string(mu_N, mu_I, delta, Q)
        result = Non_ICM_1_lessthan_Q(concate_nonICM_Sorting)
        self.assertEqual(result, ['<=', 'delta', 'Q'])

    def test_Q_I_N_delta(self):
        Q, mu_I, mu_N, delta = 1, 2, 3, 4
        concate_nonICM_Sorting = concate_nonICM_string(mu_N, mu_I, delta, Q)
        result = Non_ICM_1_lessthan_Q(concate_nonICM_Sorting)
        self.assertEqual(result, ['<=', 'delta', 'Q'])

    def test_Q_I_delta_N(self):
        Q, mu_I, delta, mu_N = 1, 2, 3, 4
        concate_nonICM_Sorting = concate_nonICM_string(mu_N, mu_I, delta, Q)
        result = Non_ICM_1_lessthan_Q(concate_nonICM_Sorting)
        self.assertEqual(result, ['<=', 'delta', 'Q'])

    def test_I_N_delta_Q(self):
        mu_I, mu_N, delta, Q = 1, 2, 3, 4
        concate_nonICM_Sorting = concate_nonICM_string(mu_N, mu_I, delta, Q)
        result = Non_ICM_1_lessthan_Q(concate_nonICM_Sorting)
        self.assertEqual(result, ['<=', 'delta', 'Q'])

    def test_I_N_Q_delta(self):
        mu_I, mu_N, Q, delta = 1, 2, 3, 4
        concate_nonICM_Sorting = concate_nonICM_string(mu_N, mu_I, delta, Q)
        result = Non_ICM_1_lessthan_Q(concate_nonICM_Sorting)
        self.assertEqual(result, ['NotUsed'])

    def test_I_delta_N_Q(self):
        mu_I, delta, mu_N, Q = 1, 2, 3, 4
        concate_nonICM_Sorting = concate_nonICM_string(mu_N, mu_I, delta, Q)
        result = Non_ICM_1_lessthan_Q(concate_nonICM_Sorting)
        self.assertEqual(result, ['<=', 'Q', 'delta'])

    def test_I_delta_Q_N(self):
        mu_I, delta, Q, mu_N = 1, 2, 3, 4
        concate_nonICM_Sorting = concate_nonICM_string(mu_N, mu_I, delta, Q)
        result = Non_ICM_1_lessthan_Q(concate_nonICM_Sorting)
        self.assertEqual(result, ['<=', 'Q', 'delta'])

    def test_I_Q_N_delta(self):
        mu_I, Q, mu_N, delta = 1, 2, 3, 4
        concate_nonICM_Sorting = concate_nonICM_string(mu_N, mu_I, delta, Q)
        result = Non_ICM_1_lessthan_Q(concate_nonICM_Sorting)
        self.assertEqual(result, ['<=', 'delta', 'Q'])

    def test_I_Q_delta_N(self):
        mu_I, Q, delta, mu_N = 1, 2, 3, 4
        concate_nonICM_Sorting = concate_nonICM_string(mu_N, mu_I, delta, Q)
        result = Non_ICM_1_lessthan_Q(concate_nonICM_Sorting)
        self.assertEqual(result, ['<=', 'delta', 'Q'])


if __name__ == '__main__':
    loader = unittest.TestLoader()
    test_ICM_0_morethan_Q = loader.loadTestsFromTestCase(test_ICM_0_morethan_Q)
    test_ICM_0_lessthan_Q = loader.loadTestsFromTestCase(test_ICM_0_lessthan_Q)
    test_ICM_1_morethan_Q = loader.loadTestsFromTestCase(test_ICM_1_morethan_Q)
    test_ICM_1_lessthan_Q = loader.loadTestsFromTestCase(test_ICM_1_lessthan_Q)
    test_nonICM_0_morethan_Q = loader.loadTestsFromTestCase(
        test_nonICM_0_morethan_Q)
    test_nonICM_0_lessthan_Q = loader.loadTestsFromTestCase(
        test_nonICM_0_lessthan_Q)
    test_nonICM_1_morethan_Q = loader.loadTestsFromTestCase(
        test_nonICM_1_morethan_Q)
    test_nonICM_1_lessthan_Q = loader.loadTestsFromTestCase(
        test_nonICM_1_lessthan_Q)

    # 建立 TestSuite 包含所有測試用例
    suite = unittest.TestSuite(
        [test_ICM_0_morethan_Q, test_ICM_0_lessthan_Q, test_ICM_1_morethan_Q, test_ICM_1_lessthan_Q, test_nonICM_0_morethan_Q, test_nonICM_0_lessthan_Q, test_nonICM_1_morethan_Q, test_nonICM_1_lessthan_Q])

    # 執行測試
    runner = unittest.TextTestRunner()
    result = runner.run(suite)
