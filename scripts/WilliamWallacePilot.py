# Controller with Q-Learning
# Designed to survive and attack enemies in the Wakuseibokan tank tournament.

import socket
from struct import *
import datetime, time
import sys
import math
import numpy as np
import os
import matplotlib.pyplot as plt

from TelemetryDictionary import telemetrydirs as td
from Command import Command
from Command import Recorder
import Configuration
from Fps import Fps

# Configuración de Q-Learning
alpha = 0.05    # Learning rate
gamma = 0.95   # Discount factor
epsilon = 0.01  # Exploration rate aumentado  # Exploration rate inicial

qtable_filename = "qtable.npy"

# Definimos discretización
def discretizar_estado(myvalues, enemyvalues):
    my_x = float(myvalues[td['x']])
    my_z = float(myvalues[td['z']])
    enemy_x = float(enemyvalues[td['x']])
    enemy_z = float(enemyvalues[td['z']])

    polardistance = math.sqrt(my_x**2 + my_z**2)
    enemydistance = math.sqrt((enemy_x - my_x)**2 + (enemy_z - my_z)**2)

    health_diff = int(myvalues[td['health']] - enemyvalues[td['health']])

    if polardistance < 1000:
        dist_cat = 0
    elif polardistance < 1700:
        dist_cat = 1
    else:
        dist_cat = 2

    if enemydistance < 300:
        enemy_cat = 0
    elif enemydistance < 800:
        enemy_cat = 1
    else:
        enemy_cat = 2

    if health_diff > 20:
        health_cat = 0
    elif health_diff > -20:
        health_cat = 1
    else:
        health_cat = 2

    return (dist_cat, enemy_cat, health_cat)

def elegir_accion(qtable, estado, acciones_disponibles):
    if np.random.rand() < epsilon:
        return np.random.choice(acciones_disponibles)
    else:
        return np.argmax(qtable[estado])

def actualizar_qtable(qtable, estado, accion, recompensa, siguiente_estado):
    mejor_q_siguiente = np.max(qtable[siguiente_estado])
    qtable[estado][accion] += alpha * (recompensa + gamma * mejor_q_siguiente - qtable[estado][accion])

def calcular_recompensa(myvalues, enemyvalues, polardistance, enemydistance):
    recompensa = 0
    if float(enemyvalues[td['health']]) < 100:
        recompensa += 50
    if polardistance > 1900:
        recompensa -= 50
    if enemydistance < 600:
        recompensa += 10
    recompensa += 1
    return recompensa

def graficar_recompensas(historial):
    plt.plot(historial)
    plt.xlabel('Episodio')
    plt.ylabel('Recompensa total')
    plt.title('Aprendizaje del Tanque - Q-Learning')
    plt.grid()
    plt.show()

class Controller:
    def __init__(self, tankparam):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        tankparam = int(tankparam)
        port = 4600 + tankparam
        self.server_address = ('0.0.0.0', port)
        print('Starting up on %s port %s' % self.server_address)

        self.sock.bind(self.server_address)
        self.sock.settimeout(5)

        self.length = 80
        self.unpackcode = '<Lififfffffffffffffff'

        self.recorder = Recorder()

        self.tank = tankparam
        self.mytimer = 0
        self.fps = Fps()
        self.fps.tic()

    def read(self):
        data, address = self.sock.recvfrom(self.length)
        if len(data) > 0 and len(data) == self.length:
            return unpack(self.unpackcode, data)
        return None

    def run(self):
        if os.path.exists(qtable_filename):
            qtable = np.load(qtable_filename, allow_pickle=True).item()
            print("Q-Table cargada.")
        else:
            qtable = {}

        acciones_disponibles = [0, 1, 2, 3, 4, 5]
        command = Command(Configuration.ip, 4500 + self.tank)

        recompensa_total = []
        episodios = 0
        print("Esperando conexión con el simulador...")

        while True:
            try:
                packet = self.read()
                if packet and int(packet[td['number']]) == self.tank:
                    print("Conexión establecida con el simulador.")
                    break
            except socket.timeout:
                print("Aún no hay datos... esperando.")

        while True:
            try:
                mypacket = None
                enemypackets = []

                while len(enemypackets) < 5:
                    packet = self.read()
                    if packet:
                        if int(packet[td['number']]) == self.tank:
                            mypacket = packet
                        else:
                            enemypackets.append(packet)

                if mypacket is None or len(enemypackets) == 0:
                    continue

                my_x = float(mypacket[td['x']])
                my_z = float(mypacket[td['z']])

                closest_enemy = None
                min_dist = float('inf')
                for enemy in enemypackets:
                    ex = float(enemy[td['x']])
                    ez = float(enemy[td['z']])
                    dist = math.sqrt((ex - my_x)**2 + (ez - my_z)**2)
                    if dist < min_dist:
                        closest_enemy = enemy
                        min_dist = dist

                estado = discretizar_estado(mypacket, closest_enemy)

                if estado not in qtable:
                    qtable[estado] = np.zeros(len(acciones_disponibles))

                accion = elegir_accion(qtable, estado, acciones_disponibles)

                thrust = 0.0
                steering = 0.0
                turretdecl = np.random.uniform(-0.4, 0.4)
                turretbearing = 0.0

                if accion == 0:
                    thrust = 11.0  # ligeramente más lento para mejorar estabilidad
                elif accion == 1:
                    thrust = 9.0  # más controlado al girar
                    steering = 1.5  # Gira más
                elif accion == 2:
                    thrust = 9.0  # más controlado al girar
                    steering = -1.5  # Gira más
                elif accion == 3:
                    command.fire()
                elif accion == 4:
                    thrust = 5.0
                elif accion == 5:
                    pass

                command.send_command(mypacket[td['timer']], self.tank,
                                     thrust, steering, turretdecl, turretbearing)

                polardistance = math.sqrt(my_x**2 + my_z**2)
                enemydistance = math.sqrt((float(closest_enemy[td['x']]) - my_x)**2 +
                                          (float(closest_enemy[td['z']]) - my_z)**2)

                recompensa = calcular_recompensa(mypacket, closest_enemy, polardistance, enemydistance)
                siguiente_estado = estado
                actualizar_qtable(qtable, estado, accion, recompensa, siguiente_estado)

                recompensa_total.append(recompensa)
                episodios += 1

            except socket.timeout:
                print(f"Episode Completed after {episodios} acciones.")
                break

        if recompensa_total:
            np.save(qtable_filename, qtable)
            print("Q-Table guardada.")
            graficar_recompensas(recompensa_total)
        else:
            print("No se recibió suficiente información. No se actualizó la Q-Table.")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python Raijin_Stormbot.py [tank_number]")
        sys.exit(1)

    controller = Controller(sys.argv[1])
    controller.run()