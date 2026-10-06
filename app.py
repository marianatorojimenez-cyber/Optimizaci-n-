import streamlit as st
import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint

st.set_page_config(page_title="Optimización Binaria MILP", layout="centered")

st.title("🧮 Optimizador Binario Interactivo (MILP)")
st.write("""
Esta aplicación resuelve problemas de optimización binaria (0-1) usando la librería `SciPy`.
Es ideal para resolver problemas tipo Mochila (Knapsack Problem) donde buscamos:

**Maximizar:**  
$$c_1 x_1 + c_2 x_2 + \dots + c_n x_n$$

**Sujeto a:**  
$$a_1 x_1 + a_2 x_2 + \dots + a_n x_n \le \text{Límite}$$

Donde cada $x_i \in \{0, 1\}$.
""")

# 1. Configuración del número de variables
n_vars = st.number_input("Número de variables (x)", min_value=2, max_value=10, value=4, step=1)

st.subheader("Parámetros del Problema")

# 2. Creación de columnas dinámicas para los coeficientes
cols = st.columns(n_vars)
c_values = []
a_values = []

for idx in range(n_vars):
    with cols[idx]:
        st.markdown(f"**Variable $x_{idx+1}$**")
        # Valor predeterminado basado en el ejercicio anterior si n_vars es 4
        default_c = [14, 5, 7, 3][idx] if idx < 4 and n_vars == 4 else 1.0
        default_a = [8, 3, 4, 2][idx] if idx < 4 and n_vars == 4 else 1.0
        
        c_val = st.number_input(f"Coeficiente (Costo/Valor) de $x_{idx+1}$", value=float(default_c), key=f"c_{idx}")
        a_val = st.number_input(f"Impacto (Peso) de $x_{idx+1}$", value=float(default_a), key=f"a_{idx}")
        c_values.append(c_val)
        a_values.append(a_val)

# 3. Límite de la restricción
limit_val = st.number_input("Límite máximo de la restricción (Capacidad)", value=9.0, step=1.0)

# Botón para resolver
if st.button("▶️ Resolver Optimización", type="primary"):
    # scipy.optimize.milp minimiza, así que multiplicamos por -1 para maximizar
    c = -np.array(c_values)
    
    # Matriz A para la restricción de desigualdad
    A = np.array([a_values])
    
    # Restricción lineal (A * x <= ub)
    constraint = LinearConstraint(A, ub=limit_val)
    
    # Límites binarios (0 <= x <= 1)
    bounds = Bounds(lb=0, ub=1)
    
    # Indicar que todas las variables son enteras (binarias)
    integrality = np.ones_like(c)
    
    # Resolver con MILP
    res = milp(c=c, constraints=constraint, bounds=bounds, integrality=integrality)
    
    st.markdown("--- ")
    if res.success:
        st.success("¡Solución Óptima Encontrada!")
        
        # Mostrar resultados de las variables
        st.subheader("Resultados de las Variables:")
        res_cols = st.columns(n_vars)
        for idx, val in enumerate(res.x):
            with res_cols[idx]:
                rounded_val = int(round(val))
                st.metric(label=f"Variable $x_{idx+1}$", value=rounded_val)
        
        # Mostrar valor de la función objetivo e impacto total consumido
        peso_total = sum(a_values[i] * res.x[i] for i in range(n_vars))
        valor_optimo = -res.fun
        
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            st.metric(label="Valor Máximo Obtenido (Z)", value=f"{valor_optimo:.2f}")
        with col_res2:
            st.metric(label="Impacto / Peso Total Usado", value=f"{peso_total:.2f} (Límite: {limit_val})")
            
        # Mostrar la ecuación final resultante
        eq_objetivo = " + ".join([f"{c_values[i]} ({int(round(res.x[i]))})" for i in range(n_vars)])
        eq_restriccion = " + ".join([f"{a_values[i]} ({int(round(res.x[i]))})" for i in range(n_vars)])
        
        st.markdown(f"**Fórmula Objetivo Aplicada:**  \n${eq_objetivo} = {valor_optimo:.2f}$")
        st.markdown(f"**Fórmula Restricción Aplicada:**  \n${eq_restriccion} = {peso_total:.2f} \le {limit_val}$")
    else:
        st.error(f"No se pudo encontrar una solución óptima. Mensaje de error: {res.message}")
