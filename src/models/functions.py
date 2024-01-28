import pandas as pd


class ControlFunction:
    """
    $$t5=10^{a0} \cdot 10^{a2*t2} \cdot 10^{a3*t3} \cdot 10^{a6*t6} \cdot t7^{a7} 
    \cdot 10^ {a8*t8} \cdot 10^{a9_2*t9_2} \cdot 10^{a9_3*t9_3} \cdot 10^{a9_4*t9_4} 
    \cdot 10^{a9_5*t9_5} \cdot 10^{a9_6*t9_6} \cdot 10^{a10*t10} \cdot t11^{a11}
    \cdot t13^{a13}$$
    """

    def __init__(self, coefs: dict[str, float]):
        self.coefs = coefs
        self.args = {}

    def __call__(self, var: dict | pd.DataFrame) -> float | pd.Series:
        """Calculate function

        Args:
            var (dict | pd.DataFrame): input params values

        Returns:
            float | pd.Series: function results
        """
        res = (10 ** self.coefs['Intercept']
               * 10 ** (self.coefs['t2'] * var['t2'] 
                        + self.coefs['t3'] * var['t3']
                        + self.coefs['t6'] * var['t6']
                        + self.coefs['t8'] * var['t8']
                        + self.coefs['t9[T.2]'] * var['t9[T.2]']
                        + self.coefs['t9[T.3]'] * var['t9[T.3]']
                        + self.coefs['t9[T.4]'] * var['t9[T.4]']
                        + self.coefs['t9[T.5]'] * var['t9[T.5]']
                        + self.coefs['t9[T.6]'] * var['t9[T.6]']
                        + self.coefs['t10'] * var['t10'])
               * var['t7'] ** self.coefs['t7']
               * var['t11'] ** self.coefs['t11']
               * var['t13'] ** self.coefs['t13'])
        return res

    def calc(self, var: dict | pd.DataFrame) -> float | pd.Series:
        """Get function f(x) = 0

        Args:
            var (dict | pd.DataFrame): input params values

        Returns:
            float | pd.Series: function results
        """
        res = (10 ** self.coefs['Intercept']
               * 10 ** (self.coefs['t2'] * var['t2'] 
                        + self.coefs['t3'] * var['t3']
                        + self.coefs['t6'] * var['t6']
                        + self.coefs['t8'] * var['t8']
                        + self.coefs['t9[T.2]'] * var['t9[T.2]']
                        + self.coefs['t9[T.3]'] * var['t9[T.3]']
                        + self.coefs['t9[T.4]'] * var['t9[T.4]']
                        + self.coefs['t9[T.5]'] * var['t9[T.5]']
                        + self.coefs['t9[T.6]'] * var['t9[T.6]']
                        + self.coefs['t10'] * var['t10'])
               * var['t7'] ** self.coefs['t7']
               * var['t11'] ** self.coefs['t11']
               * var['t13'] ** self.coefs['t13']) - var['t5']
        return res

    def _clean_args(self) -> None:
        self.args = {}

    def define_args(self, args: dict) -> None:
        self._clean_args()
        self.args.update(args.copy())

    def solve(self, vars: list[float], varnames: list[str]) -> float:
        self.args.update({varname: var for varname, var in zip(varnames, vars)})
        return self.calc(self.args)
