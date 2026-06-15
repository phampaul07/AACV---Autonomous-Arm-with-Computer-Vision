from lewansoul_servo_bus import ServoBus

servo_bus = ServoBus('/dev/ttyUSB0')

# Move servo with ID 1 to 90 degrees in 1.0 seconds
servo_bus.move_time_write(1, 90, 1.0)

# Move servo with ID 2 to 180 degrees in 2.0 seconds
servo_2 = servo_bus.get_servo(2)
servo_2.move_time_write(180, 2.0)