import socket
from struct import *
import datetime, time
import sys
import math
import numpy as np

from TelemetryDictionary import telemetrydirs as td
from Command import Command
from Command import Recorder
from Fps import Fps

class Controller:
    def __init__(self, tankparam):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        tankparam = int(tankparam)
        port = 4600 + tankparam
        self.server_address = ('0.0.0.0', port)
        print ('Starting up on %s port %s' % self.server_address)

        self.sock.bind(self.server_address)
        self.sock.settimeout(10)

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
        command = Command('192.168.122.219', 4500 + self.tank)
        ts = time.time()
        st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H-%M-%S')
        f = open(f'./data/sensor.{st}.dat', 'w')

        shouldrun = True
        while shouldrun:
            try:
                tank1values = self.read()
                if int(tank1values[td['number']]) != 1:
                    continue

                tank2values = self.read()
                if int(tank2values[td['number']]) != 2:
                    continue

                if self.tank == 1:
                    myvalues = tank1values
                    othervalues = tank2values
                else:
                    myvalues = tank2values
                    othervalues = tank1values

                self.fps.steptoc()
                print(f"Fps: {self.fps.fps}")

                if int(myvalues[td['timer']]) < self.mytimer:
                    self.recorder.newepisode()
                    print("New Episode")
                    self.mytimer = int(myvalues[td['timer']]) - 1

                self.recorder.recordvalues(myvalues, othervalues)

                f.write(','.join([str(myvalues[0]), str(myvalues[1]), str(myvalues[2]),
                                  str(myvalues[3]), str(myvalues[4]), str(myvalues[6])]) + '\n')
                f.flush()

                vec2d = (float(myvalues[td['x']]), float(myvalues[td['z']]))
                polardistance = math.sqrt(vec2d[0] ** 2 + vec2d[1] ** 2)
                print(f"Time: {myvalues[td['timer']]} Polar Distance: {polardistance}")

                thrust = 10.0 if polardistance < 1700 else 0.0
                steering = 0.0
                turretdecl = 0.0
                turretbearing = 0.0

                command.send_command(myvalues[td['timer']], self.tank, thrust,
                                     steering, turretdecl, turretbearing)

                self.mytimer += 1

            except socket.timeout:
                print("Episode Completed")
                break

        f.close()
        print('Everything successfully closed.')

if __name__ == '__main__':
    controller = Controller(sys.argv[1])
    controller.run()
