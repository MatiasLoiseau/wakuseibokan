# WilliamWallacePilot.py
#
# Controlador con Q‑Learning para Wakuseibokan.
# ---------------------------------------------------------------------------

import socket
from struct import unpack
import sys
import math
import numpy as np
import os
import matplotlib.pyplot as plt

from TelemetryDictionary import telemetrydirs as td
from Command  import Command
from Fps      import Fps
import Configuration

# --------------------------- Hiper‑parámetros RL ----------------------------
alpha_start   = 0.10
gamma         = 0.99
epsilon_init  = 0.20
epsilon_min   = 0.05
epsilon_decay = 0.995

# --------------------------- Discretización de estado -----------------------
def bearing_to_enemy(my, enemy):
    dx = float(enemy[td['x']]) - float(my[td['x']])
    dz = float(enemy[td['z']]) - float(my[td['z']])
    return math.atan2(dz, dx)

def discretizar_estado(my, enemy):
    # Distancia al centro
    polardistance = math.hypot(float(my[td['x']]), float(my[td['z']]))
    dist_cat = 0 if polardistance < 1000 else 1 if polardistance < 1700 else 2

    # Distancia al enemigo
    enemydistance = math.hypot(float(enemy[td['x']]) - float(my[td['x']]),
                               float(enemy[td['z']]) - float(my[td['z']]))
    enemy_cat = 0 if enemydistance < 300 else 1 if enemydistance < 800 else 2

    # Diferencia de salud
    health_diff = float(my[td['health']]) - float(enemy[td['health']])
    health_cat = 0 if health_diff > 20 else 1 if health_diff > -20 else 2

    # Alineación del cañón (o 0 si el campo no existe)
    turret_idx  = td.get('turretbearing', None)
    turret_bear = float(my[turret_idx]) if turret_idx is not None else 0.0
    angle_diff  = abs((turret_bear - bearing_to_enemy(my, enemy) + math.pi)
                      % (2 * math.pi) - math.pi)
    ang_cat = 0 if angle_diff < math.radians(10) else \
              1 if angle_diff < math.radians(45) else 2

    return (dist_cat, enemy_cat, health_cat, ang_cat)              # 81 estados

# ------------------------------- Recompensa ---------------------------------
def calcular_recompensa(prev_enemy_hp, my, enemy):
    hp_drop = max(0, prev_enemy_hp - float(enemy[td['health']]))
    recompensa = 100.0 * hp_drop                                    # daño

    polardistance = math.hypot(float(my[td['x']]), float(my[td['z']]))
    if polardistance > 1900:  recompensa -= 20.0

    enemydistance = math.hypot(float(enemy[td['x']]) - float(my[td['x']]),
                               float(enemy[td['z']]) - float(my[td['z']]))
    if enemydistance < 800:    recompensa += 5.0

    angle_cat = discretizar_estado(my, enemy)[3]
    if angle_cat == 0:         recompensa += 2.0

    return recompensa, float(enemy[td['health']])

# ------------------------------ Apuntar y disparar --------------------------
def aim_to_enemy(my, enemy):
    bearing = bearing_to_enemy(my, enemy)
    return bearing, 0.0                                            # declinación

# -------------------------- Funciones auxiliares ----------------------------
def elegir_accion(qtable, estado, acciones, epsilon):
    return int(np.random.choice(acciones)) if np.random.rand() < epsilon \
           else int(np.argmax(qtable[estado]))

def actualizar_qtable(qtable, estado, accion, recompensa,
                      siguiente_estado, alpha):
    qtable[estado][accion] += alpha * (recompensa +
                                       gamma * np.max(qtable[siguiente_estado]) -
                                       qtable[estado][accion])

def graficar_recompensas(hist):
    plt.plot(hist)
    plt.xlabel('Episodio')
    plt.ylabel('Recompensa media acumulada')
    plt.title('Aprendizaje del Tanque - Q-Learning')
    plt.grid()
    plt.show()

# ================================ Controlador ===============================
class Controller:
    ACCIONES = {
        0: "avanza",
        1: "gira_izq",
        2: "gira_der",
        3: "dispara",
        4: "retrocede",
        5: "espera"
    }

    def __init__(self, tank_id: int, qtable_path: str):
        self.tank      = int(tank_id)
        self.qtable_fn = qtable_path

        # UDP listening socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(('0.0.0.0', 4600 + self.tank))
        self.sock.settimeout(5.0)

        # packet format
        self.pkt_len   = 80
        self.unpackfmt = '<Lififfffffffffffffff'

        # command sender
        self.cmd = Command(Configuration.ip, 4500 + self.tank)

        # Q‑table
        if os.path.isfile(self.qtable_fn):
            self.qtable = np.load(self.qtable_fn, allow_pickle=True).item()
            print(f"[INFO] Q-Table cargada con {len(self.qtable)} estados.")
        else:
            self.qtable = {}

        self.fps = Fps()    

    def read(self):
        data, _ = self.sock.recvfrom(self.pkt_len)
        return unpack(self.unpackfmt, data) if len(data) == self.pkt_len else None

    def run(self):
        print(f"Esperando telemetría de tanque {self.tank} …")
        while True:
            try:
                pkt = self.read()
                if pkt and int(pkt[td['number']]) == self.tank:
                    break
            except socket.timeout:
                print("Aún sin datos…")

        recompensa_hist = []
        episodios = 0
        epsilon   = epsilon_init
        alpha     = alpha_start

        while True:
            try:
                mypkt, enemies = None, []
                while len(enemies) < 5:
                    pkt = self.read()
                    if pkt:
                        (mypkt := pkt) if int(pkt[td['number']]) == self.tank \
                                       else enemies.append(pkt)

                if mypkt is None or not enemies:
                    continue

                my_x, my_z = float(mypkt[td['x']]), float(mypkt[td['z']])
                enemy = min(enemies,
                            key=lambda e: math.hypot(float(e[td['x']]) - my_x,
                                                     float(e[td['z']]) - my_z))

                estado = discretizar_estado(mypkt, enemy)
                self.qtable.setdefault(estado, np.zeros(len(self.ACCIONES)))

                accion = elegir_accion(self.qtable, estado,
                                       list(self.ACCIONES), epsilon)

                thrust = steering = 0.0
                turret_bear, turret_decl = 0.0, 0.0
                disparar = False

                if accion == 0:             thrust = 11.0
                elif accion == 1:           thrust, steering = 9.0,  1.5
                elif accion == 2:           thrust, steering = 9.0, -1.5
                elif accion == 3:
                    turret_bear, turret_decl = aim_to_enemy(mypkt, enemy)
                    disparar = True
                elif accion == 4:           thrust = -6.0
                # accion 5 → espera

                self.cmd.send_command(mypkt[td['timer']], self.tank,
                                      thrust, steering, turret_decl, turret_bear)
                if disparar:
                    self.cmd.fire()

                prev_hp   = getattr(self, "prev_enemy_hp",
                                    float(enemy[td['health']]))
                recompensa, hp = calcular_recompensa(prev_hp, mypkt, enemy)
                self.prev_enemy_hp = hp

                actualizar_qtable(self.qtable, estado, accion,
                                  recompensa, estado, alpha)

                recompensa_hist.append(recompensa)
                episodios += 1

                if episodios % 200 == 0:
                    epsilon = max(epsilon_min, epsilon * epsilon_decay)
                    alpha   = max(0.01, alpha * 0.995)
                    media   = np.sum(recompensa_hist[-200:])
                    print(f"[{episodios:5d}] ε={epsilon:.3f} α={alpha:.3f} "
                          f"Δrecomp_200={media:.1f}")

            except socket.timeout:
                print(f"[FIN] Episodio cerrado tras {episodios} acciones.")
                break

        if recompensa_hist:
            np.save(self.qtable_fn, self.qtable, allow_pickle=True)
            print("Q-Table guardada en", self.qtable_fn)
            graficar_recompensas(np.cumsum(recompensa_hist) /
                                 np.arange(1, len(recompensa_hist) + 1))
        else:
            print("No hubo suficiente información para actualizar la Q-Table.")

# -------------------------------- main --------------------------------------
if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Uso: python WilliamWallacePilot.py <nro_tanque> <archivo_qtable.npy>")
        sys.exit(1)

    Controller(int(sys.argv[1]), sys.argv[2]).run()
