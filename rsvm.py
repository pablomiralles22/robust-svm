from sklearn.svm import SVC
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_results(df, y, svc=None, show_supp=False, path=""):
    plt.figure()

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.scatter(df.loc[y>0, 'x1'], df.loc[y>0, 'x2'], marker="+", c='green')
    plt.scatter(df.loc[y<0, 'x1'], df.loc[y<0, 'x2'], marker="x", c='red')

    if svc:
        # plot separation line
        intercept = -svc.intercept_[0] / svc.coef_[0][1]
        slope = - svc.coef_[0][0] / svc.coef_[0][1]
        x = np.linspace(-0.05, 1.05, 1000)
        y = slope * x + intercept
        plt.plot(x, y)

        if show_supp:
            # circle around support vectors
            sv_x = svc.support_vectors_[:,0]
            sv_y = svc.support_vectors_[:,1]
            plt.scatter(sv_x, sv_y, s=80, facecolors='none', edgecolors='#000')

    if path:
        plt.savefig(path, bbox_inches='tight')
    else:
        plt.show()

class robust_svm:
    def __init__(self, svc, eta=1.0, max_iters=10):
        self.eta = eta
        self.svc = svc
        self.max_iters = max_iters


    def fit(self, X, y):
        beta = 1. / (1 - np.exp(- self.eta))

        z = np.zeros(len(y))
        v = - np.exp(- self.eta * np.maximum(0, 1 - z))
        Cv = - beta * self.eta * v # cost implicit in svc

        for i in range(self.max_iters):
            self.svc.fit(X, y, Cv)
            #  for illustration purposes
            #  if i < 5:
            #      plot_results(X, y, self.svc, show_supp=True, path='img/robust-out-'+str(i)+'.png')

            z = self.predict(X) * y
            v = - np.exp(-self.eta * np.maximum(0, 1 - z))
            Cv = - beta * self.eta * v # cost implicit in svc


    def predict(self, X):
        return self.svc.decision_function(X)



######### create robust and standard svms
rsvm = robust_svm(SVC(kernel='linear', C=10.))
svm  = SVC(kernel='linear', C=10.)

######### generate random sample, and add outlier
N = 100
df = pd.DataFrame({
    "x1": np.random.uniform(0, 1, N),
    "x2": np.random.uniform(0, 1, N),
})

df = df.loc[np.absolute(df.x1 - df.x2) > 0.1]
y = (df.x1-df.x2 > 0) * 2 - 1

# outlier (-13.5, 13.5) -> 1
df_outliers = df.append(pd.DataFrame({"x1": [-13.5], "x2": [13.5]}, index=[N+1]))
y_outliers = np.append(y, [1])


######### generate different plots
rsvm.fit(df, y)
plot_results(df, y, rsvm.svc, show_supp=True, path='img/robust-no_out.png')

rsvm.fit(df_outliers, y_outliers)
plot_results(df, y, rsvm.svc, show_supp=True, path='img/robust-out.png')

svm.fit(df, y)
plot_results(df, y, svm, path='img/std-no_out1.png')

svm.fit(df_outliers, y_outliers)
plot_results(df, y, svm, path='img/std-out1.png')

svm.fit(df, y)
plot_results(df, y, svm, show_supp=True, path='img/std-no_out2.png')

svm.fit(df_outliers, y_outliers)
plot_results(df, y, svm, show_supp=True, path='img/std-out2.png')






