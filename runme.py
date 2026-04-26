import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.ndimage import convolve
from numba import njit
import sys

# ------ Константы генов ------
G = 12
GEN_EAT_MINERALS = 0
GEN_PHOTOSYNTHESIS = 1
GEN_O2_PRODUCTION = 2
GEN_O2_TOLERANCE = 3
GEN_O2_RESPIRATION = 4
GEN_EAT_ORGANIC = 5
GEN_AGGRESSION = 6
GEN_ARMOR = 7
GEN_MOTILITY = 8
GEN_SENSOR_FOOD = 9
GEN_SENSOR_O2 = 10
GEN_REPRO_THRESHOLD = 11

# ------ Параметры мира ------
W, H = 100, 60
MAX_ORG = 20000
MUTATION_RATE = 0.03
MUTATION_STRENGTH = 0.1
BASE_METABOLISM = 0.02
MOVEMENT_COST = 0.08
SENSOR_COST = 0.012
ARMOR_COST = 0.025
CROWD_PENALTY_START = 2
CROWD_PENALTY_PER_EXTRA = 0.012

MINERAL_TO_ENERGY = 18.0
ORGANIC_TO_ENERGY = 22.0
PHOTO_GAIN_BASE = 0.035
O2_RESPIRATION_BONUS = 20.0

KERNEL = np.array([[0, 1, 0],
                   [1, 1, 1],
                   [0, 1, 0]], dtype=float) / 5.0


# ---------- Numba-функции ----------
@njit(cache=True)
def build_occupancy_core(N, x, y, occupancy):
    """Заполняет карту occupancy количеством организмов в каждой клетке."""
    occupancy[:, :] = 0
    for i in range(N):
        occupancy[y[i], x[i]] += 1


@njit(cache=True)
def metabolism_core(
    N, order,
    x, y, energy, genes,
    minerals, organic, o2, temp, light,
    occupancy,
    MINERAL_TO_ENERGY, ORGANIC_TO_ENERGY,
    PHOTO_GAIN_BASE, O2_RESPIRATION_BONUS,
    BASE_METABOLISM, SENSOR_COST, ARMOR_COST,
    CROWD_PENALTY_START, CROWD_PENALTY_PER_EXTRA,
):
    for kk in range(N):
        i = order[kk]
        if energy[i] <= 0.0:
            continue

        ix = x[i]
        iy = y[i]
        g = genes[i]

        eat_minerals = g[0]
        photosynthesis = g[1]
        o2_production = g[2]
        o2_tolerance = g[3]
        o2_respiration = g[4]
        eat_organic = g[5]
        aggression = g[6]
        armor = g[7]
        motility = g[8]
        sensor_food = g[9]
        sensor_o2 = g[10]

        gain = 0.0

        # Минералы
        mineral_want = eat_minerals * 0.025
        mineral_taken = minerals[iy, ix]
        if mineral_taken > mineral_want:
            mineral_taken = mineral_want
        minerals[iy, ix] -= mineral_taken
        gain += mineral_taken * MINERAL_TO_ENERGY

        # Органика
        organic_want = eat_organic * 0.035
        organic_taken = organic[iy, ix]
        if organic_taken > organic_want:
            organic_taken = organic_want
        organic[iy, ix] -= organic_taken
        gain += organic_taken * ORGANIC_TO_ENERGY

        # Фотосинтез (с делением света)
        local_crowd = occupancy[iy, ix]
        if local_crowd < 1:
            local_crowd = 1
        light_share = light[iy, ix] / local_crowd
        photo_gain = light_share * photosynthesis * PHOTO_GAIN_BASE
        gain += photo_gain

        # Кислородное дыхание (усилитель)
        total_food = mineral_taken + organic_taken
        o2_local = o2[iy, ix]
        resp_gain = total_food * o2_local * o2_respiration * O2_RESPIRATION_BONUS
        gain += resp_gain

        # Затраты
        complexity = (
            eat_minerals * 0.004 +
            photosynthesis * 0.006 +
            o2_production * 0.004 +
            o2_tolerance * 0.010 +
            o2_respiration * 0.012 +
            eat_organic * 0.006 +
            aggression * 0.010 +
            armor * 0.010 +
            motility * 0.012 +
            sensor_food * 0.006 +
            sensor_o2 * 0.006
        )
        cost = BASE_METABOLISM + complexity
        cost += sensor_food * SENSOR_COST
        cost += sensor_o2 * SENSOR_COST
        cost += armor * ARMOR_COST

        # Штраф за толпу
        if local_crowd > CROWD_PENALTY_START:
            cost += (local_crowd - CROWD_PENALTY_START) * CROWD_PENALTY_PER_EXTRA

        # Кислородный урон (пороговый)
        o2_excess = o2_local - o2_tolerance * 0.25
        if o2_excess < 0.0:
            o2_excess = 0.0
        o2_damage = o2_excess * 0.6

        # Температурный урон
        temp_diff = temp[iy, ix] - 0.6
        if temp_diff < 0.0:
            temp_diff = -temp_diff
        temp_diff -= 0.2
        if temp_diff < 0.0:
            temp_diff = 0.0
        temp_damage = temp_diff * 0.1

        energy[i] += gain - cost - o2_damage - temp_damage


@njit(cache=True)
def o2_production_core(N, x, y, energy, genes, light, occupancy, o2):
    """Выделение кислорода (отход фотосинтеза) с разделением света."""
    for i in range(N):
        if energy[i] <= 0.0:
            continue
        ix = x[i]
        iy = y[i]
        local_crowd = occupancy[iy, ix]
        if local_crowd < 1:
            local_crowd = 1
        light_share = light[iy, ix] / local_crowd
        prod = genes[i, 2] * genes[i, 1] * light_share * 0.02
        o2[iy, ix] += prod * 0.1

    # Глобальная утечка и клиппинг
    h, w = o2.shape
    for yy in range(h):
        for xx in range(w):
            o2[yy, xx] *= 0.999
            if o2[yy, xx] < 0.0:
                o2[yy, xx] = 0.0
            elif o2[yy, xx] > 1.0:
                o2[yy, xx] = 1.0


class AlifeWorld:
    def __init__(self, seed=42):
        self.rng = np.random.default_rng(seed)

        # Среда
        self.minerals = self.rng.uniform(0.5, 1.0, (H, W))
        self.organic = np.zeros((H, W))
        self.o2 = np.full((H, W), 0.01)

        # Температура и свет
        y_idx = np.linspace(-1, 1, H)[:, None]
        temp_1d = 0.6 + 0.3 * np.cos(y_idx * np.pi)
        light_1d = 0.4 + 0.5 * (1 - np.abs(y_idx))
        self.temp = np.broadcast_to(temp_1d, (H, W))
        self.light = np.broadcast_to(light_1d, (H, W))

        # Вулканы (источники минералов)
        self.vents = self.rng.random((H, W)) < 0.015

        # Организмы
        self.N = 1000
        self.x = np.zeros(MAX_ORG, dtype=np.int32)
        self.y = np.zeros(MAX_ORG, dtype=np.int32)
        self.energy = np.zeros(MAX_ORG, dtype=np.float64)
        self.age = np.zeros(MAX_ORG, dtype=np.int32)
        self.genes = np.zeros((MAX_ORG, G), dtype=np.float64)

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
        self.occupancy = np.zeros((H, W), dtype=np.int64)

    # ---------- Вспомогательные ----------
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
        if new_N < old_N:
            self.x[:new_N] = self.x[active][keep]
            self.y[:new_N] = self.y[active][keep]
            self.energy[:new_N] = self.energy[active][keep]
            self.age[:new_N] = self.age[active][keep]
            self.genes[:new_N] = self.genes[active][keep]
        self.N = new_N

    # ---------- Шаг симуляции ----------
    def step(self):
        self.t += 1

        # Комета
        if self.t > 0 and self.t % 500 == 0:
            self._comet_event()

        # Диффузия
        self._diffuse_field(self.minerals, 0.01)
        self._diffuse_field(self.organic, 0.05)
        self._diffuse_field(self.o2, 0.1)

        # Восстановление минералов (вулканы + фон)
        self.minerals += self.vents * 0.006 * (1.0 - self.minerals)
        self.minerals += 0.0001 * (1.0 - self.minerals)

        # Клиппинг
        np.clip(self.minerals, 0.0, 1.0, out=self.minerals)
        np.clip(self.organic, 0.0, 5.0, out=self.organic)
        np.clip(self.o2, 0.0, 1.0, out=self.o2)

        # Возраст
        self.age[:self.N] += 1

        # ---- 1. Метаболизм (Numba) ----
        build_occupancy_core(self.N, self.x, self.y, self.occupancy)
        order = self.rng.permutation(self.N).astype(np.int64)

        metabolism_core(
            self.N, order,
            self.x, self.y, self.energy, self.genes,
            self.minerals, self.organic, self.o2, self.temp, self.light,
            self.occupancy,
            MINERAL_TO_ENERGY, ORGANIC_TO_ENERGY,
            PHOTO_GAIN_BASE, O2_RESPIRATION_BONUS,
            BASE_METABOLISM, SENSOR_COST, ARMOR_COST,
            CROWD_PENALTY_START, CROWD_PENALTY_PER_EXTRA,
        )

        # ---- 2. Смерть и компактизация ----
        self._kill_and_compact()
        if self.N == 0:
            return

        # ---- 3. Движение (пока без Numba – сложная логика) ----
        # Обновляем occupancy для движения
        build_occupancy_core(self.N, self.x, self.y, self.occupancy)
        for i in self.rng.permutation(self.N):
            if self.energy[i] <= 0:
                continue
            g = self.genes[i]
            if g[GEN_MOTILITY] <= self.rng.random():
                continue
            x, y = self.x[i], self.y[i]
            neigh = [(x+dx, y+dy) for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]]
            neigh = [(nx % W, ny % H) for nx, ny in neigh]
            self.rng.shuffle(neigh)
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
                score += self.rng.normal(0.0, 1e-6)
                if score > best_score:
                    best_score = score
                    best_move = (nx, ny)
            if best_move != (x, y):
                self.x[i], self.y[i] = best_move
                self.energy[i] -= MOVEMENT_COST

        # ---- 4. Размножение (с мутацией) ----
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

        # ---- 5. Выделение кислорода (Numba) ----
        build_occupancy_core(self.N, self.x, self.y, self.occupancy)
        o2_production_core(
            self.N, self.x, self.y, self.energy, self.genes,
            self.light, self.occupancy, self.o2
        )

        # ---- 6. Уменьшение мутационного буста ----
        if self.mutation_boost_ticks > 0:
            self.mutation_boost_ticks -= 1

        # ---- 7. Статистика ----
        self.o2_history.append(self.o2.mean())
        self.pop_history.append(self.N)
        if len(self.o2_history) > 2000:
            self.o2_history.pop(0)
            self.pop_history.pop(0)

        if self.t % 100 == 0 and self.N > 0:
            m = self.genes[:self.N].mean(axis=0)
            print(
                f"t={self.t} N={self.N} O₂={self.o2.mean():.3f} "
                f"min={m[0]:.2f} photo={m[1]:.2f} tol={m[3]:.2f} "
                f"resp={m[4]:.2f} org={m[5]:.2f} mot={m[8]:.2f}"
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
            rgb = np.array([g[0], g[1], g[5]])  # минералы, фотосинтез, органика
            intensity = min(1.0, self.energy[i] * 1.5)
            img[y, x] = np.clip(img[y, x] + rgb * intensity * 0.5, 0, 1)
        return img


# ---------- Параллельные батчевые прогоны ----------
def run_simulation(seed):
    world = AlifeWorld(seed=seed)
    # Прогрев JIT – несколько шагов
    for _ in range(100):
        world.step()
        if world.N == 0:
            break
    # Основной прогон
    for _ in range(10000):
        world.step()
        if world.N == 0:
            break
    if world.N > 0:
        mean_genes = world.genes[:world.N].mean(axis=0)
    else:
        mean_genes = np.zeros(G)
    return {
        "seed": seed,
        "final_N": world.N,
        "final_o2": float(world.o2.mean()),
        "mean_genes": mean_genes,
        "pop_history": world.pop_history,
        "o2_history": world.o2_history,
    }


# ---------- Точка входа ----------
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--batch":
        # Параллельные эксперименты
        from multiprocessing import Pool
        with Pool(8) as p:
            results = p.map(run_simulation, range(8))
        for r in results:
            print(f"Seed {r['seed']}: N={r['final_N']}, O₂={r['final_o2']:.3f}, "
                  f"genes={np.round(r['mean_genes'], 2)}")
    else:
        # Интерактивная анимация (с ускорением: рендерим раз в 10 шагов)
        world = AlifeWorld(seed=7)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        ax1.set_title("ALife v0.3 – Numba + редко рендерим")
        img = ax1.imshow(world.render(), interpolation='nearest')
        ax2.set_title("Population and O₂")
        line_pop, = ax2.plot([], [], 'g-', label='Population / MAX')
        line_o2, = ax2.plot([], [], 'b-', label='Mean O₂')
        ax2.set_xlim(0, 2000)
        ax2.set_ylim(0, 1.05)
        ax2.legend()
        ax2.grid(True)

        def update(frame):
            # Делаем 10 шагов симуляции, потом перерисовываем
            for _ in range(10):
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
