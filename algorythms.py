import numpy as np
from typing import Callable, Tuple, List

def backtracking_line_search(f, df, xk, pk, alpha = 1.0, rho = 0.5, c = 1e-4):
    fk = f(xk)
    grad_fk = df(xk)
    while f(xk + alpha * pk) > fk + c * alpha * np.dot(grad_fk, pk):
        alpha *= rho
    return alpha

def descenso_aleatorio(f, df, x0, alpha, max_iter, epsilon):
    xk = x0.copy()
    x_hist = [xk.copy()]
    f_hist = [f(x0)]
    err_hist = []

    for k in range(max_iter):
        grad = df(xk)
        d = np.random.randn(len(xk))
        d /= np.linalg.norm(d)

        while np.dot(d, grad) >= 0:
            d = np.random.randn(len(xk))
            d /= np.linalg.norm(d)
            
        alpha = backtracking_line_search(f, df, xk, d)

        xk_1 = xk + alpha * d
        x_hist.append(xk_1.copy())
        f_1 = f(xk_1)
        f_hist.append(f_1)
        
        error = np.linalg.norm(xk_1 - xk)
        err_hist.append(error)
        
        if error < epsilon:
            return xk_1, x_hist, f_hist, err_hist, k+1, True
        
        xk = xk_1

    return x_hist[-1], x_hist, f_hist, err_hist, max_iter, False

def max_naive(f, df, x0, alpha, max_iter, epsilon):
    xk = x0.copy()
    x_hist = [xk.copy()]
    f_hist = [f(x0)]
    err_hist = []

    for k in range(max_iter):
        grad = df(xk)
        
        alpha = backtracking_line_search(f, df, xk, -grad)
        
        xk_1 = xk - alpha * grad
        x_hist.append(xk_1.copy())
        f_1 = f(xk_1)
        f_hist.append(f_1)

        error = np.linalg.norm(xk_1 - xk)
        err_hist.append(error)

        if error < epsilon:
            return xk_1, x_hist, f_hist, err_hist, k + 1, True
        
        xk = xk_1

    return x_hist[-1], x_hist, f_hist, err_hist, max_iter, False

def aprox_hes(df, x, epsilon = 1e-5):
    n = x.shape[0]
    H = np.zeros((n, n))
    df_x = df(x)
    for i in range(n):
        x_eps = x.copy()
        x_eps[i] += epsilon
        df_x_eps = df(x_eps)
        H[:, i] = (df_x_eps - df_x) / epsilon
    return H

def gradiente_newton_aprox(f, df, x0, alpha, max_iter, epsilon):
    xk = x0.copy()
    x_hist = [xk.copy()]
    f_hist = [f(x0)]
    err_hist = []

    for k in range(max_iter):
        grad = df(xk)
        H_approx = aprox_hes(df, xk)
        
        try:
            delta_xk = -alpha * np.linalg.solve(H_approx, grad)
        except np.linalg.LinAlgError:
            delta_xk = -alpha * grad

        alpha = backtracking_line_search(f, df, xk, delta_xk)
        xk_1 = xk + alpha * delta_xk
        x_hist.append(xk_1.copy())
        f_1 = f(xk_1)
        f_hist.append(f_1)

        error = np.linalg.norm(xk_1 - xk)
        err_hist.append(error)
        
        if error < epsilon:
            return xk_1, x_hist, f_hist, err_hist, k + 1, True
        xk = xk_1

    return x_hist[-1], x_hist, f_hist, err_hist, max_iter, False

def gradiente_newton_exact(f, df, ddf, x0, alpha, max_iter: int, epsilon) -> Tuple[np.ndarray, List[np.ndarray], List[float], List[float], int, bool]:
    xk = x0.copy()
    x_hist = [xk.copy()]
    f_hist = [f(x0)]
    err_hist = []

    for k in range(max_iter):
        grad = df(xk)
        Hessian = ddf(xk)
        
        try:
            delta_xk = -alpha * np.linalg.solve(Hessian, grad)
        except np.linalg.LinAlgError:
            delta_xk = -alpha * grad

        alpha = backtracking_line_search(f, df, xk, delta_xk)
        xk_1 = xk + alpha * delta_xk
        x_hist.append(xk_1.copy())
        f_1 = f(xk_1)
        f_hist.append(f_1)

        error = np.linalg.norm(xk_1 - xk)
        err_hist.append(error)
        
        if error < epsilon:
            return xk_1, x_hist, f_hist, err_hist, k + 1, True
        xk = xk_1

    return x_hist[-1], x_hist, f_hist, err_hist, max_iter, False