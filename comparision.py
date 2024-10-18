import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
from algorythms import *

# Definir los algoritmos a utilizar
algorithms = [descenso_aleatorio, max_naive, gradiente_newton_aprox, gradiente_newton_exact]

def generate_table_and_plots(f, df, ddf, x0, alpha, max_iterations, epsilon):
    results_summary = []
    
    # Iterar sobre cada algoritmo
    for alg in algorithms:
        algorythm_name = alg.__name__
        
        if algorythm_name == "gradiente_newton_exact":
            x_optimal, path_xk, path_fxk, error_history, total_iters, has_converged = alg(f, df, ddf, x0, alpha, max_iterations, epsilon)
        else:
            x_optimal, path_xk, path_fxk, error_history, total_iters, has_converged = alg(f, df, x0, alpha, max_iterations, epsilon)
        
        # Obtener primeras y últimas tres iteraciones
        first_iters = path_xk[:3]
        last_iters = path_xk[-3:]
        
        # Cálculo de errores y normas de gradiente
        approximation_errors = [np.linalg.norm(x - x_optimal) for x in first_iters + last_iters]
        gradient_norms = [np.linalg.norm(df(x)) for x in first_iters + last_iters]
        
        # Preparar la tabla
        method_table = []
        for idx, (x_val, error_val, grad_norm_val) in enumerate(zip(first_iters + last_iters, approximation_errors, gradient_norms)):
            iter_number = idx + 1 if idx < 3 else total_iters - 2 + idx % 3
            method_table.append([algorythm_name, iter_number, x_val, error_val, grad_norm_val])
        
        results_summary.append((algorythm_name, error_history, method_table, path_xk))
    
    # Mostrar la tabla consolidada
    combined_table = [row for _, _, table, _ in results_summary for row in table]
    print(tabulate(combined_table, headers=["Algoritmo", "Iteración", "Valor de X", "Error Aproximado", "Norma del Gradiente"]))
    
    # Generar gráficos
    plt.figure(figsize=(10, 6))
    for algorythm_name, error_history, _, _ in results_summary:
        plt.semilogy(range(1, len(error_history) + 1), error_history, label=algorythm_name)
    
    plt.xlabel("Iteración")
    plt.ylabel("Error Aproximado (escala log)")
    plt.title("Comparativa del error en los algoritmos")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Graficar el recorrido de los puntos para problemas en R^2
    if len(x0) == 2:
        plt.figure(figsize=(10, 8))
        
        # Obtener el rango de valores para graficar
        all_traj_points = [point for _, _, _, traj in results_summary for point in traj]
        x_min, x_max = min(p[0] for p in all_traj_points), max(p[0] for p in all_traj_points)
        y_min, y_max = min(p[1] for p in all_traj_points), max(p[1] for p in all_traj_points)
        
        padding_x = (x_max - x_min) * 0.1
        padding_y = (y_max - y_min) * 0.1
        x_min, x_max = x_min - padding_x, x_max + padding_x
        y_min, y_max = y_min - padding_y, y_max + padding_y
        
        # Crear la malla para el gráfico de contorno
        x_values = np.linspace(x_min, x_max, 100)
        y_values = np.linspace(y_min, y_max, 100)
        X, Y = np.meshgrid(x_values, y_values)
        Z = np.array([f(np.array([xi, yi])) for xi, yi in zip(X.flatten(), Y.flatten())]).reshape(X.shape)
        
        plt.contour(X, Y, Z, levels=20, cmap='viridis')
        plt.colorbar(label="f(x)")
        
        # Graficar la secuencia de puntos de cada algoritmo
        for algorythm_name, _, _, traj in results_summary:
            plt.plot([point[0] for point in traj], [point[1] for point in traj], label=algorythm_name)
        
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Trayectoria de puntos por los algoritmos")
        plt.legend()
        plt.show()
