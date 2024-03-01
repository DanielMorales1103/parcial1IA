import numpy as np
import gymnasium as gym
import random
import time
from IPython.display import clear_output
import math
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

random_map = generate_random_map(size=4, p=0.80)
env = gym.make("FrozenLake-v1", is_slippery=True, desc=random_map,render_mode="human")

action_size = env.action_space.n
state_size = env.observation_space.n
q_table = np.zeros((state_size, action_size))


learning_rate = 0.1           # Tasa de aprendizaje
max_steps = 99                # Máximo número de pasos por episodio
gamma = 0.90                  # Factor de descuento

# Parámetros de exploración
epsilon = 1.0                 # Tasa de exploración inicial
max_epsilon = 1.0             # Máxima tasa de exploración
min_epsilon = 0.01            # Mínima tasa de exploración
decay_rate = 0.005            # Tasa de decaimiento exponencial para la exploración

for episode in range(250):
    state = env.reset()[0]
    step = 0
    done = False
    
    for step in range(max_steps):
        # Decidir si explorar o explotar
        tradeoff = random.uniform(0, 1)
        if tradeoff > epsilon:
            action = np.argmax(q_table[state,:])
        else:
            action = env.action_space.sample()
        
        # Realizar la acción y observar el resultado
        new_state, reward, done,_, info = env.step(action)
        # Actualizar la tabla Q usando la ecuación de Bellman
        q_table[state, action] = q_table[state, action] + learning_rate * (reward + gamma * np.max(q_table[new_state, :]) - q_table[state, action])
        
        # Nuestro nuevo estado es el estado
        state = new_state
        
        # Si caímos en un agujero o llegamos a la meta, termina el episodio
        if done == True: 
            break
    
    # Reducir epsilon (porque necesitamos menos y menos exploración)
    epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)
    
print("Entrenamiento completado. Ejecutando el agente...")   
cantEpisodes = 60 
cont = 0
for episode in range(cantEpisodes):  # Número de episodios para la ejecución posterior
    state = env.reset()[0]
    step = 0
    done = False

    for step in range(max_steps):
        action = np.argmax(q_table[state, :])

        new_state, reward, done, _, info = env.step(action)

        state = new_state

        if done:
            if reward != 0.0:
                print("Episodio {} completado en {} pasos. ¡Se alcanzó la meta!".format(episode + 1, step + 1))
                cont += 1
            break

# Cierra el entorno después de completar la ejecución
env.close()
print("Acierto de: ",cont/cantEpisodes)