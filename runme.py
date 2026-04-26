import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.ndimage import convolve

# ------ Гены ------
G = 12
(
    GEN_EAT_MINERALS,
    GEN_PHOTOSYNTHESIS,
    GEN_O2_PRODUCTION,
    GEN_O2_TOLERANCE,
    GEN_O2_RESPIRATION,
    GEN_EAT_ORGANIC,
    GEN_AGGRESSION,
    GEN_ARMOR,
    GEN_MOTILITY,
    GEN_SENSOR_FOOD,
    GEN_SENSOR_O2,
    GEN_REPRO_THRESHOLD,
) = range(G)

# ------ Параметры мира ------
W, H = 100, 60
MAX_ORG = 20000
MUTATION_RATE = 0.03
MUTATION_STRENGTH = 0.1
BASE_METABOLISM = 0.02
MOVEMENT_COST = 0.08
SENSOR_COST = 0.012
ARMOR_COST = 0.025
CROWD_PENALTY_START = 5
CROWD_PENALTY_PER_EXTRA = 0.004

MINERAL_TO_ENERGY = 18.0
ORGANIC_TO_ENERGY = 22.0
PHOTO_GAIN_BASE = 0.035
O2_RESPIRATION_BONUS = 20.0

KERNEL = np.array([[0, 1, 0],
                   [1, 1, 1],
                   [0, 1, 0]], dtype=float) / 5.0


class AlifeWorld:
    def __init__(self, seed=42):
        self.rng = np.random.default_rng(seed)

        # Среда
        self.minerals = self.rng.uniform(0.5, 1.0, (H, W))
        self.organic = np.zeros((H, W))
        self.o2 = np.full((H, W), 0.01)

        # Температура и свет (широтная зависимость)
        y_idx = np.linspace(-1, 1, H)[:, None]
        temp_1d = 0.6 + 0.3 * np.cos(y_idx * np.pi)
        light_1d = 0.4 + 0.5 * (1 - np.abs(y_idx))
        self.temp = np.broadcast_to(temp_1d, (H, W))
        self.light = np.broadcast_to(light_1d, (H, W))

        # Источники минералов (маленькие пятна, чтобы движение было выгодным)
        self.vents = self.rng.random((H, W)) < 0.015

        # Организмы
        self.N = 1000
        self.x = np.zeros(MAX_ORG, dtype=int)
        self.y = np.zeros(MAX_ORG, dtype=int)
        self.energy = np.zeros(MAX_ORG)
        self.age = np.zeros(MAX_ORG)
        self.genes = np.zeros((MAX_ORG, G))

        # Базовый геном + начальный шум
        base_genome = np.array([0.8, 0.2, 0.1, 0.05, 0.0, 0.0,
                                0.0, 0.0, 0.0, 0.0, 0.0, 0.8])
        for i in range(self.N):
            self.x[i] = self.rng.integers(0, W)
            self.y[i] = self.rng.integers(0, H)
            self.energy[i] = 0.5
            self.genes[i] = np.clip(
                base_genome + self.rng.normal(0.0, 0.03, G),
                0.0, 1.0
            )

        self.t = 0
        self.o2_history = []
        self.pop_history = []
        self.mutation_boost_ticks = 0

    # ---------- Вспомогательные методы ----------
    def _diffuse_field(self, field, rate):
        blurred = convolve(field, KERNEL, mode='reflect')
        field[:] = field * (1 - rate) + blurred * rate

    def _mutate_genes(self, genes):
        rate = MUTATION_RATE
        if self.mutation_boost_ticks > 0:
            rate *= 6.0

        child = genes.copy()
        mask = self.rng.random(G) < rate
        child += mask * self.rng.normal(0.0, MUTATION_STRENGTH, G)

        if self.rng.random() < rate * 0.15:
            idx = self.rng.integers(0, G)
            child[idx] = self.rng.random()

        return np.clip(child, 0.0, 1.0)

    def _comet_event(self):
        cx = self.rng.integers(0, W)
        cy = self.rng.integers(0, H)
        radius = self.rng.integers(5, 14)

        yy, xx = np.ogrid[:H, :W]
        mask = (xx - cx)**2 + (yy - cy)**2 <= radius**2

        self.minerals[mask] = np.clip(self.minerals[mask] + 0.4, 0.0, 1.0)
        self.organic[mask] += 0.2

        for i in range(self.N):
            dx = min(abs(self.x[i] - cx), W - abs(self.x[i] - cx))
            dy = min(abs(self.y[i] - cy), H - abs(self.y[i] - cy))
            if dx*dx + dy*dy <= radius*radius and self.rng.random() < 0.55:
                self.energy[i] = -1.0

        self.mutation_boost_ticks = 80

    def _kill_and_compact(self):
        """Удаляет мёртвых, добавляет органику, сжимает массивы"""
        old_N = self.N
        active = slice(0, old_N)

        dead = (self.energy[active] <= 0.05) | (self.age[active] > 500)

        if dead.any():
            np.add.at(
                self.organic,
                (self.y[active][dead], self.x[active][dead]),
                0.05
            )

        keep = ~dead
        new_N = int(keep.sum())

        self.x[:new_N] = self.x[active][keep]
        self.y[:new_N] = self.y[active][keep]
        self.energy[:new_N] = self.energy[active][keep]
        self.age[:new_N] = self.age[active][keep]
        self.genes[:new_N] = self.genes[active][keep]

        self.N = new_N

    # ---------- Основной шаг ----------
    def step(self):
        self.t += 1

        # Комета
        if self.t > 0 and self.t % 500 == 0:
            self._comet_event()

        # Диффузия среды
        self._diffuse_field(self.minerals, 0.01)
        self._diffuse_field(self.organic, 0.05)
        self._diffuse_field(self.o2, 0.1)

        # Восстановление минералов (источники + фон)
        self.minerals += self.vents * 0.006 * (1.0 - self.minerals)
        self.minerals += 0.0001 * (1.0 - self.minerals)

        # Клиппинг
        np.clip(self.minerals, 0.0, 1.0, out=self.minerals)
        np.clip(self.organic, 0.0, 5.0, out=self.organic)
        np.clip(self.o2, 0.0, 1.0, out=self.o2)

        # Возраст
        self.age[:self.N] += 1

        # ----- 1. Метаболизм (случайный порядок) -----
        occupancy = np.zeros((H, W), dtype=int)
        for i in range(self.N):
            occupancy[self.y[i], self.x[i]] += 1

        for i in self.rng.permutation(self.N):
            if self.energy[i] <= 0:
                continue
            x, y = self.x[i], self.y[i]
            g = self.genes[i]

            gain = 0.0

            # Минералы
            mineral_want = g[GEN_EAT_MINERALS] * 0.025
            mineral_taken = min(self.minerals[y, x], mineral_want)
            self.minerals[y, x] -= mineral_taken
            gain += mineral_taken * MINERAL_TO_ENERGY

            # Органика
            organic_want = g[GEN_EAT_ORGANIC] * 0.035
            organic_taken = min(self.organic[y, x], organic_want)
            self.organic[y, x] -= organic_taken
            gain += organic_taken * ORGANIC_TO_ENERGY

            # Фотосинтез
            photo_gain = self.light[y, x] * g[GEN_PHOTOSYNTHESIS] * PHOTO_GAIN_BASE
            gain += photo_gain

            # Кислородное дыхание (усилитель)
            total_food = mineral_taken + organic_taken
            o2_local = self.o2[y, x]
            resp_gain = total_food * o2_local * g[GEN_O2_RESPIRATION] * O2_RESPIRATION_BONUS
            gain += resp_gain

            # Затраты
            complexity = (
                g[GEN_EAT_MINERALS] * 0.004 +
                g[GEN_PHOTOSYNTHESIS] * 0.006 +
                g[GEN_O2_PRODUCTION] * 0.004 +
                g[GEN_O2_TOLERANCE] * 0.010 +
                g[GEN_O2_RESPIRATION] * 0.012 +
                g[GEN_EAT_ORGANIC] * 0.006 +
                g[GEN_AGGRESSION] * 0.010 +
                g[GEN_ARMOR] * 0.010 +
                g[GEN_MOTILITY] * 0.012 +
                g[GEN_SENSOR_FOOD] * 0.006 +
                g[GEN_SENSOR_O2] * 0.006
            )
            cost = BASE_METABOLISM + complexity
            cost += g[GEN_SENSOR_FOOD] * SENSOR_COST
            cost += g[GEN_SENSOR_O2] * SENSOR_COST
            cost += g[GEN_ARMOR] * ARMOR_COST

            crowd = occupancy[y, x]
            if crowd > CROWD_PENALTY_START:
                cost += (crowd - CROWD_PENALTY_START) * CROWD_PENALTY_PER_EXTRA

            o2_damage = o2_local * (1 - g[GEN_O2_TOLERANCE]) * 0.2
            temp_damage = max(0.0, abs(self.temp[y, x] - 0.6) - 0.2) * 0.1

            self.energy[i] += gain - cost - o2_damage - temp_damage

        # ----- 2. Смерть и компактизация -----
        self._kill_and_compact()
        if self.N == 0:
            # Всё вымерло – прекращаем (но обычно так не бывает)
            return

        # ----- 3. Движение (случайный порядок) -----
        occupancy.fill(0)
        for i in range(self.N):
            occupancy[self.y[i], self.x[i]] += 1

        for i in self.rng.permutation(self.N):
            if self.energy[i] <= 0:
                continue
            g = self.genes[i]
            if g[GEN_MOTILITY] <= self.rng.random():
                continue

            x, y = self.x[i], self.y[i]
            neigh = [(x+dx, y+dy) for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]]
            neigh = [(nx % W, ny % H) for nx, ny in neigh]
            self.rng.shuffle(neigh)   # чтобы не было системного движения вправо

            best_score = -1e9
            best_move = (x, y)
            for nx, ny in neigh:
                score = 0.0
                if g[GEN_SENSOR_FOOD] > 0:
                    food_val = self.minerals[ny, nx] + self.organic[ny, nx]
                    score += food_val * g[GEN_SENSOR_FOOD]
                if g[GEN_SENSOR_O2] > 0:
                    if g[GEN_O2_TOLERANCE] < 0.3:
                        score += (1 - self.o2[ny, nx]) * g[GEN_SENSOR_O2]
                    elif g[GEN_O2_RESPIRATION] > 0:
                        score += self.o2[ny, nx] * g[GEN_SENSOR_O2]
                # небольшой шум для разрушения симметрии
                score += self.rng.normal(0.0, 1e-6)
                if score > best_score:
                    best_score = score
                    best_move = (nx, ny)

            if best_move != (x, y):
                self.x[i], self.y[i] = best_move
                self.energy[i] -= MOVEMENT_COST

        # ----- 4. Размножение (случайный порядок, дети не участвуют в этом шаге) -----
        parent_N = self.N
        for i in self.rng.permutation(parent_N):
            if self.energy[i] <= 0:
                continue
            g = self.genes[i]
            threshold = 0.2 + g[GEN_REPRO_THRESHOLD] * 1.0
            if self.energy[i] > threshold and self.N < MAX_ORG - 10:
                self.energy[i] /= 2
                child_genes = self._mutate_genes(g)
                idx = self.N
                self.x[idx] = self.x[i]
                self.y[idx] = self.y[i]
                self.energy[idx] = self.energy[i]
                self.age[idx] = 0
                self.genes[idx] = child_genes
                self.N += 1

        # ----- 5. Выделение кислорода (связано с фотосинтезом) -----
        o2_prod = np.zeros((H, W))
        for i in range(self.N):
            if self.energy[i] <= 0:
                continue
            x, y = self.x[i], self.y[i]
            g = self.genes[i]
            # Кислород как отход фотосинтеза
            prod = g[GEN_O2_PRODUCTION] * g[GEN_PHOTOSYNTHESIS] * self.light[y, x] * 0.02
            o2_prod[y, x] += prod
        self.o2 += o2_prod * 0.1
        self.o2 *= 0.999
        np.clip(self.o2, 0.0, 1.0, out=self.o2)

        # ----- 6. Уменьшение мутационного буста (раз в тик) -----
        if self.mutation_boost_ticks > 0:
            self.mutation_boost_ticks -= 1

        # ----- 7. Статистика -----
        self.o2_history.append(self.o2.mean())
        self.pop_history.append(self.N)

        if len(self.o2_history) > 2000:
            self.o2_history.pop(0)
            self.pop_history.pop(0)

        # Печать средних генов для наблюдения
        if self.t % 100 == 0 and self.N > 0:
            m = self.genes[:self.N].mean(axis=0)
            print(
                f"t={self.t} N={self.N} O₂={self.o2.mean():.3f} "
                f"min={m[GEN_EAT_MINERALS]:.2f} "
                f"photo={m[GEN_PHOTOSYNTHESIS]:.2f} "
                f"tol={m[GEN_O2_TOLERANCE]:.2f} "
                f"resp={m[GEN_O2_RESPIRATION]:.2f} "
                f"org={m[GEN_EAT_ORGANIC]:.2f} "
                f"mot={m[GEN_MOTILITY]:.2f}"
            )

    # ---------- Визуализация ----------
    def render(self):
        img = np.zeros((H, W, 3))
        img[..., 0] = self.minerals * 0.6
        img[..., 1] = self.organic * 0.8
        img[..., 2] = self.o2 * 0.5

        for i in range(self.N):
            if self.energy[i] <= 0:
                continue
            x, y = self.x[i], self.y[i]
            g = self.genes[i]
            rgb = np.array([g[GEN_EAT_MINERALS],
                            g[GEN_PHOTOSYNTHESIS],
                            g[GEN_EAT_ORGANIC]])
            intensity = min(1.0, self.energy[i] * 1.5)
            img[y, x] = np.clip(img[y, x] + rgb * intensity * 0.5, 0, 1)
        return img


def main():
    world = AlifeWorld(seed=7)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.set_title("ALife v0.1 – Artificial Life")
    img = ax1.imshow(world.render(), interpolation='nearest')
    ax2.set_title("Population and O₂")
    line_pop, = ax2.plot([], [], 'g-', label='Population / MAX')
    line_o2, = ax2.plot([], [], 'b-', label='Mean O₂')
    ax2.set_xlim(0, 2000)
    ax2.set_ylim(0, 1.05)
    ax2.legend()
    ax2.grid(True)

    def update(frame):
        for _ in range(2):
            world.step()
        img.set_data(world.render())
        n = len(world.pop_history)
        if n > 2000:
            pop_norm = np.array(world.pop_history[-2000:]) / MAX_ORG
            o2_hist = world.o2_history[-2000:]
            x_vals = list(range(2000))
        else:
            pop_norm = np.array(world.pop_history) / MAX_ORG
            o2_hist = world.o2_history
            x_vals = list(range(n))
        line_pop.set_data(x_vals, pop_norm)
        line_o2.set_data(x_vals, o2_hist)
        ax2.relim()
        ax2.autoscale_view(scalex=False, scaley=True)
        ax1.set_title(f"t={world.t}  N={world.N}")
        return img, line_pop, line_o2

    ani = FuncAnimation(fig, update, interval=50, blit=False, cache_frame_data=False)
    plt.show()


if __name__ == "__main__":
    main()
