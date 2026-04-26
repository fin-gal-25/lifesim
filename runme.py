import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.ndimage import convolve

# Константы генов
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

# Параметры мира
W, H = 100, 60
MAX_ORG = 20000
MUTATION_RATE = 0.03          # вероятность мутации для каждого гена
MUTATION_STRENGTH = 0.1
BASE_METABOLISM = 0.02
MOVEMENT_COST = 0.08
SENSOR_COST = 0.012
ARMOR_COST = 0.025
CROWD_PENALTY_START = 5      # начиная с этого числа организмов в клетке
CROWD_PENALTY_PER_EXTRA = 0.004

# Коэффициенты преобразования ресурсов в энергию
MINERAL_TO_ENERGY = 18.0
ORGANIC_TO_ENERGY = 22.0
PHOTO_GAIN_BASE = 0.035
O2_RESPIRATION_BONUS = 20.0

# Ядро для диффузии (фон Нейман)
KERNEL = np.array([[0, 1, 0],
                   [1, 1, 1],
                   [0, 1, 0]], dtype=float) / 5.0


class AlifeWorld:
    def __init__(self, seed=42):
        self.rng = np.random.default_rng(seed)

        # Физическая среда
        self.land = np.ones((H, W), dtype=bool)
        self.minerals = self.rng.uniform(0.5, 1.0, (H, W))
        self.organic = np.zeros((H, W))
        self.o2 = np.full((H, W), 0.01)
        y_idx = np.linspace(-1, 1, H)[:, None]
        self.temp = 0.6 + 0.3 * np.cos(y_idx * np.pi)
        self.light = 0.4 + 0.5 * (1 - np.abs(y_idx))

        # Хранилище организмов
        self.N = 1000
        self.x = np.zeros(MAX_ORG, dtype=int)
        self.y = np.zeros(MAX_ORG, dtype=int)
        self.energy = np.zeros(MAX_ORG)
        self.age = np.zeros(MAX_ORG)
        self.genes = np.zeros((MAX_ORG, G))

        # Инициализация стартовых организмов
        for i in range(self.N):
            self.x[i] = self.rng.integers(0, W)
            self.y[i] = self.rng.integers(0, H)
            self.energy[i] = 0.5
            self.genes[i] = np.array([
                0.8,   # eat_minerals
                0.2,   # photosynthesis
                0.1,   # o2_production
                0.05,  # o2_tolerance
                0.0,   # o2_respiration
                0.0,   # eat_organic
                0.0,   # aggression
                0.0,   # armor
                0.0,   # motility
                0.0,   # sensor_food
                0.0,   # sensor_o2
                0.8,   # repro_threshold
            ])

        self.t = 0
        self.o2_history = []
        self.pop_history = []
        self.mutation_boost_ticks = 0

    # -----------------------------------------------------------------
    # Вспомогательные методы
    # -----------------------------------------------------------------
    def _diffuse_field(self, field, rate):
        """диффузия поля с отражающими границами"""
        blurred = convolve(field, KERNEL, mode='reflect')
        field[:] = field * (1 - rate) + blurred * rate

    def _mutate_genes(self, genes):
        """мутирует каждый ген с вероятностью MUTATION_RATE (усиленной бустом)"""
        rate = MUTATION_RATE
        if self.mutation_boost_ticks > 0:
            rate *= 6.0
            self.mutation_boost_ticks -= 1

        child = genes.copy()
        mask = self.rng.random(G) < rate
        child += mask * self.rng.normal(0.0, MUTATION_STRENGTH, G)

        # редкое крупное изменение – один ген случайно перезаписывается
        if self.rng.random() < rate * 0.15:
            idx = self.rng.integers(0, G)
            child[idx] = self.rng.random()

        return np.clip(child, 0.0, 1.0)

    def _comet_event(self):
        """комета: удар, добавление ресурсов, гибель организмов, мутационный буст"""
        cx = self.rng.integers(0, W)
        cy = self.rng.integers(0, H)
        radius = self.rng.integers(5, 14)

        yy, xx = np.ogrid[:H, :W]
        mask = (xx - cx)**2 + (yy - cy)**2 <= radius**2

        # добавляем минералы и органику
        self.minerals[mask] = np.clip(self.minerals[mask] + 0.4, 0.0, 1.0)
        self.organic[mask] += 0.2

        # убиваем часть организмов в зоне
        for i in range(self.N):
            dx = min(abs(self.x[i] - cx), W - abs(self.x[i] - cx))
            dy = min(abs(self.y[i] - cy), H - abs(self.y[i] - cy))
            if dx*dx + dy*dy <= radius*radius:
                if self.rng.random() < 0.55:
                    self.energy[i] = -1.0   # пометка на смерть

        # мутационный буст на следующие тики
        self.mutation_boost_ticks = 80

    # -----------------------------------------------------------------
    # Основной шаг симуляции
    # -----------------------------------------------------------------
    def step(self):
        self.t += 1

        # 0. Комета раз в 500 тиков
        if self.t > 0 and self.t % 500 == 0:
            self._comet_event()

        # 1. Диффузия среды
        self._diffuse_field(self.minerals, 0.01)
        self._diffuse_field(self.organic, 0.05)
        self._diffuse_field(self.o2, 0.1)

        # медленное восстановление минералов из недр
        self.minerals += 0.001 * (1 - self.minerals)

        # клиппинг
        np.clip(self.minerals, 0.0, 1.0, out=self.minerals)
        np.clip(self.organic, 0.0, 5.0, out=self.organic)
        np.clip(self.o2, 0.0, 1.0, out=self.o2)

        # 2. Подсчёт occupancy (сколько организмов в каждой клетке)
        occupancy = np.zeros((H, W), dtype=int)
        for i in range(self.N):
            if self.energy[i] > 0:
                occupancy[self.y[i], self.x[i]] += 1

        # 3. Метаболизм и движение
        # сначала все получают энергию и платят затраты, затем двигаются
        # (движение после метаболизма – чтобы сенсоры видели актуальную среду)
        for i in range(self.N):
            if self.energy[i] <= 0:
                continue

            x, y = self.x[i], self.y[i]
            g = self.genes[i]

            # ---- Получение энергии ----
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

            # Фотосинтез (бесплатно, но слабее)
            photo_gain = self.light[y, x] * g[GEN_PHOTOSYNTHESIS] * PHOTO_GAIN_BASE
            gain += photo_gain

            # Кислородное дыхание – усилитель переработки съеденной еды
            total_food = mineral_taken + organic_taken
            o2_local = self.o2[y, x]
            resp_gain = total_food * o2_local * g[GEN_O2_RESPIRATION] * O2_RESPIRATION_BONUS
            gain += resp_gain

            # ---- Затраты ----
            cost = BASE_METABOLISM

            # Стоимость сложности генома
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
            cost += complexity

            # Стоимость сенсоров (фиксированная, даже если не используются)
            cost += g[GEN_SENSOR_FOOD] * SENSOR_COST
            cost += g[GEN_SENSOR_O2] * SENSOR_COST
            # Стоимость брони
            cost += g[GEN_ARMOR] * ARMOR_COST

            # Штраф за толпу
            crowd = occupancy[y, x]
            if crowd > CROWD_PENALTY_START:
                cost += (crowd - CROWD_PENALTY_START) * CROWD_PENALTY_PER_EXTRA

            # ---- Урон ----
            # Кислородный урон
            o2_damage = o2_local * (1 - g[GEN_O2_TOLERANCE]) * 0.2
            # Температурный урон (если слишком далеко от оптимума 0.6)
            temp_damage = max(0.0, abs(self.temp[y, x] - 0.6) - 0.2) * 0.1

            self.energy[i] += gain - cost - o2_damage - temp_damage

        # 4. Смерть (голод, старость, комета уже пометила)
        for i in range(self.N):
            if self.energy[i] <= 0.05 or self.age[i] > 500:
                # труп добавляет органику
                self.organic[self.y[i], self.x[i]] += 0.05
                self.energy[i] = -1.0   # пометка на удаление

        # 5. Удаление мёртвых (сжатие)
        keep = self.energy[:self.N] > 0
        self.N = np.sum(keep)
        self.x[:self.N] = self.x[keep]
        self.y[:self.N] = self.y[keep]
        self.energy[:self.N] = self.energy[keep]
        self.age[:self.N] = self.age[keep]
        self.genes[:self.N] = self.genes[keep]

        # 6. Движение (только если есть motility и энергия)
        # сначала обновляем occupancy для движения (можно пересчитать, но упростим)
        occupancy.fill(0)
        for i in range(self.N):
            occupancy[self.y[i], self.x[i]] += 1

        for i in range(self.N):
            if self.energy[i] <= 0:
                continue
            g = self.genes[i]
            if g[GEN_MOTILITY] <= self.rng.random():
                continue   # шанс пошевелиться пропорционален motility

            x, y = self.x[i], self.y[i]
            # четырёхсвязная окрестность
            neigh = [(x+dx, y+dy) for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]]
            neigh = [(nx%W, ny%H) for nx,ny in neigh]

            best_score = -1e9
            best_move = (x, y)
            for nx, ny in neigh:
                score = 0.0
                if g[GEN_SENSOR_FOOD] > 0:
                    # привлекает минералы+органика
                    food_val = self.minerals[ny, nx] + self.organic[ny, nx]
                    score += food_val * g[GEN_SENSOR_FOOD]
                if g[GEN_SENSOR_O2] > 0:
                    if g[GEN_O2_TOLERANCE] < 0.3:
                        # кислород вреден – избегаем
                        score += (1 - self.o2[ny, nx]) * g[GEN_SENSOR_O2]
                    elif g[GEN_O2_RESPIRATION] > 0:
                        # кислород полезен – идём к нему
                        score += self.o2[ny, nx] * g[GEN_SENSOR_O2]
                if score > best_score:
                    best_score = score
                    best_move = (nx, ny)

            if best_move != (x, y):
                # перемещаем, уменьшаем энергию
                self.x[i], self.y[i] = best_move
                self.energy[i] -= MOVEMENT_COST

        # 7. Размножение
        new_children = []
        for i in range(self.N):
            if self.energy[i] <= 0:
                continue
            g = self.genes[i]
            threshold = 0.2 + g[GEN_REPRO_THRESHOLD] * 1.0
            if self.energy[i] > threshold and self.N < MAX_ORG - 10:
                self.energy[i] /= 2
                child_energy = self.energy[i]
                child_genes = self._mutate_genes(g)
                idx = self.N
                self.x[idx] = self.x[i]
                self.y[idx] = self.y[i]
                self.energy[idx] = child_energy
                self.age[idx] = 0
                self.genes[idx] = child_genes
                self.N += 1
                new_children.append(idx)

        # 8. Выделение кислорода (после метаболизма, потому что фотосинтез уже дал энергию)
        o2_prod = np.zeros((H, W))
        for i in range(self.N):
            if self.energy[i] <= 0:
                continue
            x, y = self.x[i], self.y[i]
            g = self.genes[i]
            # кислород от фотосинтеза пропорционален интенсивности фотосинтеза и свету
            prod = g[GEN_O2_PRODUCTION] * g[GEN_PHOTOSYNTHESIS] * self.light[y, x] * 0.02
            o2_prod[y, x] += prod
        self.o2 += o2_prod * 0.1
        self.o2 *= 0.999    # небольшая утечка
        np.clip(self.o2, 0.0, 1.0, out=self.o2)

        # 9. Статистика
        self.o2_history.append(self.o2.mean())
        self.pop_history.append(self.N)

        # 10. Защита от переполнения истории
        if len(self.o2_history) > 2000:
            self.o2_history.pop(0)
            self.pop_history.pop(0)

    # -----------------------------------------------------------------
    # Визуализация
    # -----------------------------------------------------------------
    def render(self):
        img = np.zeros((H, W, 3))
        # фон: минералы (красный канал), органика (зелёный), O2 (синий)
        img[..., 0] = self.minerals * 0.6
        img[..., 1] = self.organic * 0.8
        img[..., 2] = self.o2 * 0.5

        # организмы: каждый организм добавляет цвет в клетку (аккумуляция)
        # чтобы видеть переполнение, яркость ограничена 1
        for i in range(self.N):
            if self.energy[i] <= 0:
                continue
            x, y = self.x[i], self.y[i]
            g = self.genes[i]
            # цвет: красный = eat_minerals, зелёный = photosynthesis, синий = eat_organic
            rgb = np.array([g[GEN_EAT_MINERALS],
                            g[GEN_PHOTOSYNTHESIS],
                            g[GEN_EAT_ORGANIC]])
            intensity = min(1.0, self.energy[i] * 1.5)
            img[y, x] = np.clip(img[y, x] + rgb * intensity * 0.5, 0, 1)

        return img


def main():
    world = AlifeWorld(seed=7)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.set_title("Organisms and environment")
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
        # обновляем графики
        if len(world.pop_history) > 2000:
            pop_norm = np.array(world.pop_history[-2000:]) / MAX_ORG
            o2_hist = world.o2_history[-2000:]
            x_vals = list(range(len(pop_norm)))
        else:
            pop_norm = np.array(world.pop_history) / MAX_ORG
            o2_hist = world.o2_history
            x_vals = list(range(len(pop_norm)))

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
