import numpy as np
import pandas as pd
from typing import Callable, List, Tuple
from tabulate import tabulate

from algorythms import (
    descenso_aleatorio, max_naive, gradiente_newton_aprox, 
    gradiente_newton_exact
)

def find_optimal_alpha(algorithm, f, df, x0, alpha_range, max_iter, epsilon):
    best_alpha = alpha_range[0]
    best_error = float('inf')
    
    for alpha in alpha_range:
        if algorithm.__name__ == 'gradiente_newton_exact':
            _, _, _, err_hist, _, _ = algorithm(f, df, lambda x: np.eye(len(x0)), x0, alpha, max_iter, epsilon)
        else:
            _, _, _, err_hist, _, _ = algorithm(f, df, x0, alpha, max_iter, epsilon)
        
        final_error = err_hist[-1] if err_hist else float('inf')
        
        if final_error < best_error:
            best_error = final_error
            best_alpha = alpha
    
    return best_alpha

def test_optimization_algorithms(objective_functions):
    algorithms = [descenso_aleatorio, max_naive, gradiente_newton_aprox, gradiente_newton_exact]
    alpha_range = [0.001, 0.01, 0.1, 0.5, 1.0]
    max_iter = 100
    epsilon = 1e-6
    
    results = []
    
    for f, df, ddf, x0 in objective_functions:
        for algorithm in algorithms:
            optimal_alpha = find_optimal_alpha(algorithm, f, df, x0, alpha_range, max_iter, epsilon)
            
            if algorithm.__name__ == 'gradiente_newton_exact':
                x_final, x_hist, f_hist, err_hist, iterations, converged = algorithm(f, df, ddf, x0, optimal_alpha, max_iter, epsilon)
            else:
                x_final, x_hist, f_hist, err_hist, iterations, converged = algorithm(f, df, x0, optimal_alpha, max_iter, epsilon)
            
            gradients = [np.linalg.norm(df(x)) for x in x_hist]
            
            # Guardamos las primeras 3 y últimas 3 iteraciones
            iterations_to_show = list(range(min(3, iterations))) + list(range(max(0, iterations-3), iterations))
            
            for i in iterations_to_show:
                results.append({
                    'Algoritmo': algorithm.__name__,
                    'Función': f.__name__,
                    'No. Iteración': i + 1,
                    'X Final': x_hist[i],
                    'F(X) Final': f_hist[i],
                    'Converge': converged,
                    'Error Aproximado': err_hist[i] if i < len(err_hist) else None,
                    'Gradiente Normal': gradients[i],
                    'Alfa óptimo': optimal_alpha,
                })
    
    df_results = pd.DataFrame(results)
    
    # Formatear la tabla para una mejor visualización
    df_results['X Final'] = df_results['X Final'].apply(lambda x: np.array2string(x, precision=4, suppress_small=True))
    df_results['Error Aproximado'] = df_results['Error Aproximado'].apply(lambda x: f'{x:.6e}' if x is not None else None)
    df_results['Gradiente Normal'] = df_results['Gradiente Normal'].apply(lambda x: f'{x:.6e}')
    
    print(tabulate(df_results, headers='keys', tablefmt='grid', showindex=False))