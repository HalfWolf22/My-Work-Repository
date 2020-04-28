class LinearRegression:

    def __init__(self, n_iter=100, alpha=0.01):
        self.n_iter = n_iter
        self.alpha = alpha

    @staticmethod
    def get_cost(X, target, params):
        m = len(X)
        cost = 0
        for x, y in zip(X, target):
            y_hat = np.dot(params, np.array([1, x]))
            cost += (y_hat - y) ** 2

        cost = cost / (2 * m)
        return cost

    def fit(self, X, y, debug=False):
        self.X = X
        self.y = y
        self.m = len(self.y)
        self.params = np.zeros(self.X.shape[1] + 1)

        for i in range(self.n_iter):
            for x, target in zip(self.X, self.y):
                y_hat = np.dot(self.params, np.array(np.insert(x, 0, 1)))
                gradient = np.array(np.insert(x, 0, 1)) * (target - y_hat)

                self.params += self.alpha * gradient / self.m

            if debug == True:
                print(get_cost(self.X, self.y, self.params))

    def predict(self, X):
        ones = np.array([1] * X.shape[0])[:, None]
        X_temp = np.append(ones, X, axis=1)

        y_pred = np.dot(X_temp, self.params)

        return y_pred
