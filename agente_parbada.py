import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- 1. Definición de la Clase Agente (Pájaro) ---
class Boid:
    def __init__(self, x, y, vx, vy, perception_radius=50, separation_distance=20, max_speed=5):
        self.position = np.array([x, y], dtype=float)
        self.velocity = np.array([vx, vy], dtype=float)
        self.perception_radius = perception_radius  # Radio para percibir otros boids
        self.separation_distance = separation_distance # Distancia para la regla de separación
        self.max_speed = max_speed # Velocidad máxima para limitar el movimiento

    def update_velocity(self, boids):
        # Reiniciar las fuerzas de las reglas para el cálculo
        v_separation = np.array([0.0, 0.0])
        v_alignment = np.array([0.0, 0.0])
        v_cohesion = np.array([0.0, 0.0])

        num_neighbors_separation = 0
        num_neighbors_alignment_cohesion = 0
        
        avg_velocity = np.array([0.0, 0.0])
        center_of_mass = np.array([0.0, 0.0])

        for other_boid in boids:
            if other_boid is self:
                continue

            distance = np.linalg.norm(self.position - other_boid.position)

            # Regla de Separación: Evitar chocar con boids cercanos
            if distance < self.separation_distance:
                # Moverse en la dirección opuesta al boid cercano
                v_separation += (self.position - other_boid.position) / distance
                num_neighbors_separation += 1

            # Reglas de Alineación y Cohesión: Considerar boids dentro del radio de percepción
            if distance < self.perception_radius:
                avg_velocity += other_boid.velocity
                center_of_mass += other_boid.position
                num_neighbors_alignment_cohesion += 1

        # Aplicar reglas solo si hay vecinos
        if num_neighbors_separation > 0:
            v_separation /= num_neighbors_separation
            v_separation = self._limit_speed(v_separation, 1.0) # Ajustar la fuerza de separación

        if num_neighbors_alignment_cohesion > 0:
            # Regla de Alineación: Dirigirse hacia la velocidad promedio de los vecinos
            v_alignment = avg_velocity / num_neighbors_alignment_cohesion
            v_alignment = self._limit_speed(v_alignment, 0.5) # Ajustar la fuerza de alineación

            # Regla de Cohesión: Dirigirse hacia el centro de masa de los vecinos
            v_cohesion = (center_of_mass / num_neighbors_alignment_cohesion) - self.position
            v_cohesion = self._limit_speed(v_cohesion, 0.5) # Ajustar la fuerza de cohesión

        # Combinar todas las fuerzas para la nueva velocidad
        self.velocity += v_separation + v_alignment + v_cohesion
        self.velocity = self._limit_speed(self.velocity, self.max_speed)

    def _limit_speed(self, vector, limit):
        # Limita la magnitud de un vector
        magnitude = np.linalg.norm(vector)
        if magnitude > limit:
            return vector / magnitude * limit
        return vector

    def update_position(self, x_limit, y_limit):
        self.position += self.velocity

        # Mantener los boids dentro de los límites del entorno
        if self.position[0] < 0:
            self.position[0] = x_limit
        elif self.position[0] > x_limit:
            self.position[0] = 0

        if self.position[1] < 0:
            self.position[1] = y_limit
        elif self.position[1] > y_limit:
            self.position[1] = 0

# --- 2. Configuración del Entorno y Simulación ---
def simulate_boids(num_boids=50, x_limit=800, y_limit=600, frames=200):
    boids = []
    for _ in range(num_boids):
        x = np.random.rand() * x_limit
        y = np.random.rand() * y_limit
        vx = np.random.uniform(-1, 1) * 2 # Velocidad inicial aleatoria
        vy = np.random.uniform(-1, 1) * 2
        boids.append(Boid(x, y, vx, vy))

    # Configuración de la visualización
    fig, ax = plt.subplots()
    ax.set_xlim(0, x_limit)
    ax.set_ylim(0, y_limit)
    
    # Dibujar los boids como puntos o flechas
    scat = ax.scatter([b.position[0] for b in boids], [b.position[1] for b in boids], s=10)

    def update(frame):
        for boid in boids:
            boid.update_velocity(boids) # Aplicar reglas de comportamiento [cite: 34]
            boid.update_position(x_limit, y_limit) # Actualizar posición [cite: 36, 38]

        # Actualizar los datos del scatter plot
        scat.set_offsets([[b.position[0], b.position[1]] for b in boids])
        return scat,

    ani = animation.FuncAnimation(fig, update, frames=frames, interval=50, blit=True)
    plt.show()

# --- Ejecutar la simulación ---
if __name__ == "__main__":
    simulate_boids()