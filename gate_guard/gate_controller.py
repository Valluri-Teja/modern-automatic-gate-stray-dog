"""Gate output. SimulatedGate just prints; SerialGate talks to an Arduino/ESP32
(sends b'CLOSE\\n' / b'OPEN\\n')."""


class GateController:
    def __init__(self):
        self.is_closed = False

    def close(self):
        self.is_closed = True

    def open(self):
        self.is_closed = False

    def shutdown(self):
        pass


class SimulatedGate(GateController):
    def close(self):
        super().close()
        print("[gate] >>> CLOSE (simulated)")

    def open(self):
        super().open()
        print("[gate] >>> OPEN (simulated)")


class SerialGate(GateController):
    def __init__(self, port, baud):
        super().__init__()
        import serial  # pip install pyserial
        self._ser = serial.Serial(port, baud, timeout=1)

    def _send(self, msg):
        self._ser.write(msg)
        self._ser.flush()

    def close(self):
        super().close()
        self._send(b"CLOSE\n")
        print("[gate] >>> CLOSE (serial)")

    def open(self):
        super().open()
        self._send(b"OPEN\n")
        print("[gate] >>> OPEN (serial)")

    def shutdown(self):
        self._ser.close()


def make_gate(cfg):
    if cfg.gate == "serial":
        try:
            return SerialGate(cfg.serial_port, cfg.serial_baud)
        except Exception as e:  # missing pyserial, wrong port, ...
            print(f"[gate] serial unavailable ({e}); using simulated gate")
    return SimulatedGate()
