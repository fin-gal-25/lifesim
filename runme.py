import argparse
import sys
from multiprocessing import Pool

import numpy as np
from numba import njit
from scipy.ndimage import convolve

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

KERNEL = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=float) / 5.0


@njit(cache=True)
def build_occupancy_core(N, x, y, occupancy):
    occupancy[:, :] = 0
    for i in range(N):
        occupancy[y[i], x[i]] += 1


@njit(cache=True)
def metabolism_core(
    N,
    order,
    x,
    y,
    energy,
    genes,
    minerals,
    organic,
    o2,
    temp,
    light,
    occupancy,
    mineral_to_energy,
    organic_to_energy,
    photo_gain_base,
    o2_respiration_bonus,
    base_metabolism,
    sensor_cost,
    armor_cost,
    crowd_penalty_start,
    crowd_penalty_per_extra,
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

        mineral_want = eat_minerals * 0.025
        mineral_taken = minerals[iy, ix]
        if mineral_taken > mineral_want:
            mineral_taken = mineral_want
        minerals[iy, ix] -= mineral_taken
        gain += mineral_taken * mineral_to_energy

        organic_want = eat_organic * 0.035
        organic_taken = organic[iy, ix]
        if organic_taken > organic_want:
            organic_taken = organic_want
        organic[iy, ix] -= organic_taken
        gain += organic_taken * organic_to_energy

        local_crowd = occupancy[iy, ix]
        if local_crowd < 1:
            local_crowd = 1
        light_share = light[iy, ix] / local_crowd
        gain += light_share * photosynthesis * photo_gain_base

        total_food = mineral_taken + organic_taken
        o2_local = o2[iy, ix]
        gain += total_food * o2_local * o2_respiration * o2_respiration_bonus

        complexity = (
            eat_minerals * 0.004
            + photosynthesis * 0.006
            + o2_production * 0.004
            + o2_tolerance * 0.010
            + o2_respiration * 0.012
            + eat_organic * 0.006
            + aggression * 0.010
            + armor * 0.010
            + motility * 0.012
            + sensor_food * 0.006
            + sensor_o2 * 0.006
        )
        cost = base_metabolism + complexity
        cost += sensor_food * sensor_cost
        cost += sensor_o2 * sensor_cost
        cost += armor * armor_cost

        if local_crowd > crowd_penalty_start:
            cost += (local_crowd - crowd_penalty_start) * crowd_penalty_per_extra

        o2_excess = o2_local - o2_tolerance * 0.25
        if o2_excess < 0.0:
            o2_excess = 0.0
        o2_damage = o2_excess * 0.6

        temp_diff = temp[iy, ix] - 0.6
        if temp_diff < 0.0:
            temp_diff = -temp_diff
        temp_diff -= 0.2
        if temp_diff < 0.0:
            temp_diff = 0.0
        temp_damage = temp_diff * 0.1

        energy[i] += gain - cost - o2_damage - temp_damage


@njit(cache=True)
def movement_core(
    N,
    order,
    x,
    y,
    energy,
    genes,
    minerals,
    organic,
    o2,
    movement_roll,
    direction_order,
    noise,
    movement_cost,
    width,
    height,
):
    for kk in range(N):
        i = order[kk]
        if energy[i] <= 0.0:
            continue

        g = genes[i]
        if g[GEN_MOTILITY] <= movement_roll[i]:
            continue

        x0 = x[i]
        y0 = y[i]
        best_score = -1e9
        best_x = x0
        best_y = y0

        for jj in range(4):
            d = direction_order[i, jj]
            nx = x0
            ny = y0
            if d == 0:
                nx = x0 + 1
            elif d == 1:
                nx = x0 - 1
            elif d == 2:
                ny = y0 + 1
            else:
                ny = y0 - 1

            if nx >= width:
                nx = 0
            elif nx < 0:
                nx = width - 1
            if ny >= height:
                ny = 0
            elif ny < 0:
                ny = height - 1

            score = 0.0
            if g[GEN_SENSOR_FOOD] > 0.0:
                score += (minerals[ny, nx] + organic[ny, nx]) * g[GEN_SENSOR_FOOD]
            if g[GEN_SENSOR_O2] > 0.0:
                if g[GEN_O2_TOLERANCE] < 0.3:
                    score += (1.0 - o2[ny, nx]) * g[GEN_SENSOR_O2]
                elif g[GEN_O2_RESPIRATION] > 0.0:
                    score += o2[ny, nx] * g[GEN_SENSOR_O2]

            score += noise[i, jj]
            if score > best_score:
                best_score = score
                best_x = nx
                best_y = ny

        if best_x != x0 or best_y != y0:
            x[i] = best_x
            y[i] = best_y
            energy[i] -= movement_cost


@njit(cache=True)
def o2_production_core(N, x, y, energy, genes, light, occupancy, o2):
    for i in range(N):
        if energy[i] <= 0.0:
            continue
        ix = x[i]
        iy = y[i]
        local_crowd = occupancy[iy, ix]
        if local_crowd < 1:
            local_crowd = 1
        light_share = light[iy, ix] / local_crowd
        prod = genes[i, GEN_O2_PRODUCTION] * genes[i, GEN_PHOTOSYNTHESIS] * light_share * 0.02
        o2[iy, ix] += prod * 0.1

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

        self.minerals = self.rng.uniform(0.5, 1.0, (H, W))
        self.organic = np.zeros((H, W))
        self.o2 = np.full((H, W), 0.01)

        y_idx = np.linspace(-1, 1, H)[:, None]
        temp_1d = 0.6 + 0.3 * np.cos(y_idx * np.pi)
        light_1d = 0.4 + 0.5 * (1 - np.abs(y_idx))
        self.temp = np.broadcast_to(temp_1d, (H, W))
        self.light = np.broadcast_to(light_1d, (H, W))
        self.vents = self.rng.random((H, W)) < 0.015

        self.N = 1000
        self.x = np.zeros(MAX_ORG, dtype=np.int32)
        self.y = np.zeros(MAX_ORG, dtype=np.int32)
        self.energy = np.zeros(MAX_ORG, dtype=np.float64)
        self.age = np.zeros(MAX_ORG, dtype=np.int32)
        self.genes = np.zeros((MAX_ORG, G), dtype=np.float64)

        base_genome = np.array([0.8, 0.2, 0.1, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8])
        for i in range(self.N):
            self.x[i] = self.rng.integers(0, W)
            self.y[i] = self.rng.integers(0, H)
            self.energy[i] = 0.5
            self.genes[i] = np.clip(base_genome + self.rng.normal(0.0, 0.03, G), 0.0, 1.0)

        self.t = 0
        self.o2_history = []
        self.pop_history = []
        self.mean_genes_history = []
        self.mutation_boost_ticks = 0
        self.occupancy = np.zeros((H, W), dtype=np.int64)

        self.settings = {
            "mutation_rate": MUTATION_RATE,
            "mutation_strength": MUTATION_STRENGTH,
            "base_metabolism": BASE_METABOLISM,
            "movement_cost": MOVEMENT_COST,
            "sensor_cost": SENSOR_COST,
            "armor_cost": ARMOR_COST,
            "photo_gain_base": PHOTO_GAIN_BASE,
            "o2_respiration_bonus": O2_RESPIRATION_BONUS,
        }

    def update_setting(self, key, value):
        if key not in self.settings:
            raise KeyError(f"Unknown setting key: {key}")
        if value < 0.0:
            raise ValueError(f"Negative setting value for {key}: {value}")
        self.settings[key] = float(value)

    def _diffuse_field(self, field, rate):
        blurred = convolve(field, KERNEL, mode="reflect")
        field[:] = field * (1 - rate) + blurred * rate

    def _mutate_genes(self, parent_genes):
        rate = self.settings["mutation_rate"]
        if self.mutation_boost_ticks > 0:
            rate *= 6.0

        child = parent_genes.copy()
        mask = self.rng.random(G) < rate
        child += mask * self.rng.normal(0.0, self.settings["mutation_strength"], G)

        if self.rng.random() < rate * 0.15:
            idx = self.rng.integers(0, G)
            child[idx] = self.rng.random()

        return np.clip(child, 0.0, 1.0)

    def _comet_event(self):
        cx = self.rng.integers(0, W)
        cy = self.rng.integers(0, H)
        radius = self.rng.integers(5, 14)

        yy, xx = np.ogrid[:H, :W]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2

        self.minerals[mask] = np.clip(self.minerals[mask] + 0.4, 0.0, 1.0)
        self.organic[mask] += 0.2

        for i in range(self.N):
            dx = min(abs(self.x[i] - cx), W - abs(self.x[i] - cx))
            dy = min(abs(self.y[i] - cy), H - abs(self.y[i] - cy))
            if dx * dx + dy * dy <= radius * radius and self.rng.random() < 0.55:
                self.energy[i] = -1.0

        self.mutation_boost_ticks = 80

    def _kill_and_compact(self):
        old_n = self.N
        active = slice(0, old_n)
        dead = (self.energy[active] <= 0.05) | (self.age[active] > 500)
        if dead.any():
            np.add.at(self.organic, (self.y[active][dead], self.x[active][dead]), 0.05)

        keep = ~dead
        new_n = int(keep.sum())
        if new_n < old_n:
            self.x[:new_n] = self.x[active][keep]
            self.y[:new_n] = self.y[active][keep]
            self.energy[:new_n] = self.energy[active][keep]
            self.age[:new_n] = self.age[active][keep]
            self.genes[:new_n] = self.genes[active][keep]
        self.N = new_n

    def _prepare_movement_randomness(self):
        n = self.N
        order = self.rng.permutation(n).astype(np.int64)
        movement_roll = self.rng.random(n)
        direction_priority = self.rng.random((n, 4))
        direction_order = np.argsort(direction_priority, axis=1).astype(np.int64)
        noise = self.rng.normal(0.0, 1e-6, (n, 4))
        return order, movement_roll, direction_order, noise

    def step(self):
        self.t += 1

        if self.t > 0 and self.t % 500 == 0:
            self._comet_event()

        self._diffuse_field(self.minerals, 0.01)
        self._diffuse_field(self.organic, 0.05)
        self._diffuse_field(self.o2, 0.1)

        self.minerals += self.vents * 0.006 * (1.0 - self.minerals)
        self.minerals += 0.0001 * (1.0 - self.minerals)

        np.clip(self.minerals, 0.0, 1.0, out=self.minerals)
        np.clip(self.organic, 0.0, 5.0, out=self.organic)
        np.clip(self.o2, 0.0, 1.0, out=self.o2)

        self.age[: self.N] += 1

        build_occupancy_core(self.N, self.x, self.y, self.occupancy)
        metabolism_order = self.rng.permutation(self.N).astype(np.int64)

        metabolism_core(
            self.N,
            metabolism_order,
            self.x,
            self.y,
            self.energy,
            self.genes,
            self.minerals,
            self.organic,
            self.o2,
            self.temp,
            self.light,
            self.occupancy,
            MINERAL_TO_ENERGY,
            ORGANIC_TO_ENERGY,
            self.settings["photo_gain_base"],
            self.settings["o2_respiration_bonus"],
            self.settings["base_metabolism"],
            self.settings["sensor_cost"],
            self.settings["armor_cost"],
            CROWD_PENALTY_START,
            CROWD_PENALTY_PER_EXTRA,
        )

        self._kill_and_compact()
        if self.N == 0:
            self.o2_history.append(self.o2.mean())
            self.pop_history.append(0)
            self.mean_genes_history.append(np.zeros(G))
            return

        move_order, move_roll, direction_order, noise = self._prepare_movement_randomness()
        movement_core(
            self.N,
            move_order,
            self.x,
            self.y,
            self.energy,
            self.genes,
            self.minerals,
            self.organic,
            self.o2,
            move_roll,
            direction_order,
            noise,
            self.settings["movement_cost"],
            W,
            H,
        )

        parent_n = self.N
        for i in self.rng.permutation(parent_n):
            if self.energy[i] <= 0:
                continue
            g = self.genes[i]
            threshold = 0.2 + g[GEN_REPRO_THRESHOLD] * 1.0
            if self.energy[i] > threshold and self.N < MAX_ORG - 10:
                self.energy[i] /= 2
                idx = self.N
                self.x[idx] = self.x[i]
                self.y[idx] = self.y[i]
                self.energy[idx] = self.energy[i]
                self.age[idx] = 0
                self.genes[idx] = self._mutate_genes(g)
                self.N += 1

        build_occupancy_core(self.N, self.x, self.y, self.occupancy)
        o2_production_core(self.N, self.x, self.y, self.energy, self.genes, self.light, self.occupancy, self.o2)

        if self.mutation_boost_ticks > 0:
            self.mutation_boost_ticks -= 1

        self.o2_history.append(self.o2.mean())
        self.pop_history.append(self.N)
        self.mean_genes_history.append(self.genes[: self.N].mean(axis=0) if self.N > 0 else np.zeros(G))
        if len(self.o2_history) > 2000:
            self.o2_history.pop(0)
            self.pop_history.pop(0)
            self.mean_genes_history.pop(0)

        if self.t % 100 == 0 and self.N > 0:
            m = self.mean_genes_history[-1]
            print(
                f"t={self.t} N={self.N} O₂={self.o2.mean():.3f} "
                f"min={m[0]:.2f} photo={m[1]:.2f} tol={m[3]:.2f} "
                f"resp={m[4]:.2f} org={m[5]:.2f} mot={m[8]:.2f}"
            )

    def render(self):
        img = np.zeros((H, W, 3))
        img[..., 0] = self.minerals * 0.6
        img[..., 1] = self.organic * 0.8
        img[..., 2] = self.o2 * 0.5
        for i in range(self.N):
            if self.energy[i] <= 0:
                continue
            rgb = np.array([self.genes[i, GEN_EAT_MINERALS], self.genes[i, GEN_PHOTOSYNTHESIS], self.genes[i, GEN_EAT_ORGANIC]])
            intensity = min(1.0, self.energy[i] * 1.5)
            img[self.y[i], self.x[i]] = np.clip(img[self.y[i], self.x[i]] + rgb * intensity * 0.5, 0, 1)
        return img


def run_simulation(seed, steps=10000, warmup_steps=100):
    world = AlifeWorld(seed=seed)
    for _ in range(warmup_steps):
        world.step()
        if world.N == 0:
            break

    for _ in range(steps):
        world.step()
        if world.N == 0:
            break

    mean_genes = world.genes[: world.N].mean(axis=0) if world.N > 0 else np.zeros(G)
    return {
        "seed": seed,
        "final_N": world.N,
        "final_o2": float(world.o2.mean()),
        "mean_genes": mean_genes,
        "pop_history": world.pop_history,
        "o2_history": world.o2_history,
        "mean_genes_history": world.mean_genes_history,
    }


def parse_batch_seeds(seed_string):
    chunks = [s.strip() for s in seed_string.split(",") if s.strip()]
    if not chunks:
        raise ValueError("batch seeds list is empty")
    return [int(s) for s in chunks]


def run_batch_mode(args):
    seeds = parse_batch_seeds(args.batch_seeds)
    payload = [(seed, args.batch_steps, args.warmup_steps) for seed in seeds]

    with Pool(args.batch_workers) as pool:
        results = pool.starmap(run_simulation, payload)

    for result in results:
        print(
            f"Seed {result['seed']}: N={result['final_N']}, O₂={result['final_o2']:.3f}, "
            f"genes={np.round(result['mean_genes'], 2)}"
        )

    if args.batch_plot:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
        for result in results:
            x = np.arange(len(result["pop_history"]))
            axes[0].plot(x, np.array(result["pop_history"]) / MAX_ORG, label=f"seed {result['seed']}")
            axes[1].plot(x, result["o2_history"], label=f"seed {result['seed']}")

            if result["mean_genes_history"]:
                gene_hist = np.array(result["mean_genes_history"])
                axes[2].plot(x, gene_hist[:, GEN_PHOTOSYNTHESIS], alpha=0.8, label=f"photo s{result['seed']}")

        axes[0].set_ylabel("Population / MAX")
        axes[1].set_ylabel("Mean O₂")
        axes[2].set_ylabel("Mean photo gene")
        axes[2].set_xlabel("Tick")
        for ax in axes:
            ax.grid(True)
            ax.legend(fontsize=8)
        plt.tight_layout()
        plt.show()


def run_interactive_mode(args):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.widgets import Button, Slider

    world = AlifeWorld(seed=args.seed)
    fig, (ax_world, ax_stats, ax_genes) = plt.subplots(1, 3, figsize=(17, 6))
    fig.subplots_adjust(bottom=0.34)

    img = ax_world.imshow(world.render(), interpolation="nearest")
    ax_world.set_title("ALife interactive")

    line_pop, = ax_stats.plot([], [], "g-", label="Population / MAX")
    line_o2, = ax_stats.plot([], [], "b-", label="Mean O₂")
    ax_stats.set_xlim(0, 2000)
    ax_stats.set_ylim(0, 1.05)
    ax_stats.grid(True)
    ax_stats.legend()

    line_photo, = ax_genes.plot([], [], "m-", label="mean photo")
    line_tol, = ax_genes.plot([], [], "c-", label="mean o2 tolerance")
    line_mot, = ax_genes.plot([], [], color="orange", label="mean motility")
    ax_genes.set_xlim(0, 2000)
    ax_genes.set_ylim(0, 1.05)
    ax_genes.grid(True)
    ax_genes.legend()

    sim_state = {"paused": False, "steps_per_frame": args.steps_per_frame}

    slider_axes = {
        "mutation_rate": plt.axes([0.05, 0.24, 0.24, 0.03]),
        "mutation_strength": plt.axes([0.05, 0.19, 0.24, 0.03]),
        "movement_cost": plt.axes([0.05, 0.14, 0.24, 0.03]),
        "base_metabolism": plt.axes([0.05, 0.09, 0.24, 0.03]),
        "steps_per_frame": plt.axes([0.05, 0.04, 0.24, 0.03]),
        "photo_gain_base": plt.axes([0.38, 0.24, 0.24, 0.03]),
        "o2_respiration_bonus": plt.axes([0.38, 0.19, 0.24, 0.03]),
        "sensor_cost": plt.axes([0.38, 0.14, 0.24, 0.03]),
        "armor_cost": plt.axes([0.38, 0.09, 0.24, 0.03]),
    }

    pause_button_ax = plt.axes([0.65, 0.04, 0.10, 0.03])
    reset_button_ax = plt.axes([0.77, 0.04, 0.10, 0.03])

    sliders = {
        "mutation_rate": Slider(slider_axes["mutation_rate"], "Mutation rate", 0.0, 0.2, valinit=world.settings["mutation_rate"], valstep=0.001),
        "mutation_strength": Slider(slider_axes["mutation_strength"], "Mutation str", 0.0, 0.4, valinit=world.settings["mutation_strength"], valstep=0.005),
        "movement_cost": Slider(slider_axes["movement_cost"], "Move cost", 0.0, 0.2, valinit=world.settings["movement_cost"], valstep=0.001),
        "base_metabolism": Slider(slider_axes["base_metabolism"], "Base metab", 0.0, 0.2, valinit=world.settings["base_metabolism"], valstep=0.001),
        "photo_gain_base": Slider(slider_axes["photo_gain_base"], "Photo gain", 0.0, 0.15, valinit=world.settings["photo_gain_base"], valstep=0.001),
        "o2_respiration_bonus": Slider(slider_axes["o2_respiration_bonus"], "O₂ resp bonus", 0.0, 40.0, valinit=world.settings["o2_respiration_bonus"], valstep=0.5),
        "sensor_cost": Slider(slider_axes["sensor_cost"], "Sensor cost", 0.0, 0.1, valinit=world.settings["sensor_cost"], valstep=0.001),
        "armor_cost": Slider(slider_axes["armor_cost"], "Armor cost", 0.0, 0.1, valinit=world.settings["armor_cost"], valstep=0.001),
        "steps_per_frame": Slider(slider_axes["steps_per_frame"], "Steps/frame", 1, 50, valinit=sim_state["steps_per_frame"], valstep=1),
    }

    pause_button = Button(pause_button_ax, "Pause")
    reset_button = Button(reset_button_ax, "Reset")

    def bind_setting_slider(setting_name):
        def _on_change(value):
            world.update_setting(setting_name, value)

        return _on_change

    for key in (
        "mutation_rate",
        "mutation_strength",
        "movement_cost",
        "base_metabolism",
        "photo_gain_base",
        "o2_respiration_bonus",
        "sensor_cost",
        "armor_cost",
    ):
        sliders[key].on_changed(bind_setting_slider(key))

    def on_steps_changed(value):
        sim_state["steps_per_frame"] = int(value)

    def on_pause_clicked(_):
        sim_state["paused"] = not sim_state["paused"]
        pause_button.label.set_text("Resume" if sim_state["paused"] else "Pause")

    def on_reset_clicked(_):
        for key in world.settings:
            sliders[key].reset()
        sliders["steps_per_frame"].reset()

    sliders["steps_per_frame"].on_changed(on_steps_changed)
    pause_button.on_clicked(on_pause_clicked)
    reset_button.on_clicked(on_reset_clicked)

    def update(_frame):
        if not sim_state["paused"]:
            for _ in range(sim_state["steps_per_frame"]):
                world.step()

        img.set_data(world.render())
        n = len(world.pop_history)
        if n == 0:
            return img, line_pop, line_o2, line_photo, line_tol, line_mot

        if n > 2000:
            pop = np.array(world.pop_history[-2000:]) / MAX_ORG
            o2_hist = world.o2_history[-2000:]
            genes_hist = np.array(world.mean_genes_history[-2000:])
            x_vals = np.arange(2000)
        else:
            pop = np.array(world.pop_history) / MAX_ORG
            o2_hist = world.o2_history
            genes_hist = np.array(world.mean_genes_history)
            x_vals = np.arange(n)

        line_pop.set_data(x_vals, pop)
        line_o2.set_data(x_vals, o2_hist)
        line_photo.set_data(x_vals, genes_hist[:, GEN_PHOTOSYNTHESIS])
        line_tol.set_data(x_vals, genes_hist[:, GEN_O2_TOLERANCE])
        line_mot.set_data(x_vals, genes_hist[:, GEN_MOTILITY])

        ax_stats.relim()
        ax_stats.autoscale_view(scalex=False, scaley=True)
        ax_genes.relim()
        ax_genes.autoscale_view(scalex=False, scaley=True)
        ax_world.set_title(f"t={world.t} N={world.N}")
        return img, line_pop, line_o2, line_photo, line_tol, line_mot

    FuncAnimation(fig, update, interval=args.interval_ms, blit=False, cache_frame_data=False)
    plt.show()


def run_sanity_checks():
    world = AlifeWorld(seed=123)
    for _ in range(50):
        world.step()
        assert 0 <= world.N <= MAX_ORG, f"Unexpected population: {world.N}"
        assert np.all(world.x[: world.N] >= 0) and np.all(world.x[: world.N] < W), "x out of bounds"
        assert np.all(world.y[: world.N] >= 0) and np.all(world.y[: world.N] < H), "y out of bounds"
        assert np.all(world.o2 >= 0.0) and np.all(world.o2 <= 1.0), "o2 bounds broken"

    build_occupancy_core(world.N, world.x, world.y, world.occupancy)
    if world.N > 0:
        idx = 0
        y, x = world.y[idx], world.x[idx]
        crowd = world.occupancy[y, x]
        assert crowd >= 1, "light-sharing occupancy is broken"


def build_parser():
    parser = argparse.ArgumentParser(description="ALife simulation")
    parser.add_argument("--mode", choices=["interactive", "batch"], default="interactive")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps-per-frame", type=int, default=10)
    parser.add_argument("--interval-ms", type=int, default=50)

    parser.add_argument("--batch-seeds", type=str, default="0,1,2,3,4,5,6,7")
    parser.add_argument("--batch-workers", type=int, default=8)
    parser.add_argument("--batch-steps", type=int, default=10000)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--batch-plot", action="store_true")

    parser.add_argument("--sanity-check", action="store_true")
    return parser


def main(argv):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.steps_per_frame < 1:
        raise ValueError("--steps-per-frame must be >= 1")
    if args.batch_workers < 1:
        raise ValueError("--batch-workers must be >= 1")

    if args.sanity_check:
        run_sanity_checks()

    if args.mode == "batch":
        run_batch_mode(args)
    else:
        run_interactive_mode(args)


if __name__ == "__main__":
    # backward compatibility with old flag
    if len(sys.argv) > 1 and sys.argv[1] == "--batch":
        sys.argv = [sys.argv[0], "--mode", "batch", *sys.argv[2:]]
    main(sys.argv[1:])
