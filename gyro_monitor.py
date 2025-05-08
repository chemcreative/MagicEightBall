import smbus
from time import sleep
import subprocess
import os
import sys
import signal

# MPU6050 Registers and their Address
PWR_MGMT_1   = 0x6B
SMPLRT_DIV   = 0x19
CONFIG       = 0x1A
GYRO_CONFIG  = 0x1B
INT_ENABLE   = 0x38
ACCEL_XOUT_H = 0x3B
ACCEL_YOUT_H = 0x3D
ACCEL_ZOUT_H = 0x3F
GYRO_XOUT_H  = 0x43
GYRO_YOUT_H  = 0x45
GYRO_ZOUT_H  = 0x47

# Shake detection parameters
SHAKE_THRESHOLD = 1.5  # Adjust this value to change sensitivity
SAMPLES_TO_CHECK = 5   # Number of samples to check for shake
last_accel_values = []  # Store last few acceleration values

def cleanup_and_exit():
    """Clean up and exit the program"""
    print("\nCleaning up and exiting...")
    # Kill any existing main.py processes
    try:
        os.system("pkill -f 'python3 main.py'")
    except:
        pass
    sys.exit(0)

def MPU_Init():
    # write to sample rate register
    bus.write_byte_data(Device_Address, SMPLRT_DIV, 7)
    
    # Write to power management register
    bus.write_byte_data(Device_Address, PWR_MGMT_1, 1)
    
    # Write to Configuration register
    bus.write_byte_data(Device_Address, CONFIG, 0)
    
    # Write to Gyro configuration register
    bus.write_byte_data(Device_Address, GYRO_CONFIG, 24)
    
    # Write to interrupt enable register
    bus.write_byte_data(Device_Address, INT_ENABLE, 1)

def read_raw_data(addr):
    # Accelero and Gyro value are 16-bit
    high = bus.read_byte_data(Device_Address, addr)
    low = bus.read_byte_data(Device_Address, addr+1)
    
    # concatenate higher and lower value
    value = ((high << 8) | low)
    
    # to get signed value from mpu6050
    if(value > 32768):
        value = value - 65536
    return value

def detect_shake(accel_values):
    """Detect if the device has been shaken based on acceleration changes"""
    if len(accel_values) < SAMPLES_TO_CHECK:
        return False
    
    # Calculate the difference between consecutive readings
    diffs = []
    for i in range(1, len(accel_values)):
        diff = abs(accel_values[i] - accel_values[i-1])
        diffs.append(diff)
    
    # Check if any difference exceeds the threshold
    return any(diff > SHAKE_THRESHOLD for diff in diffs)

def launch_main():
    """Launch main.py in a new terminal window"""
    try:
        print("\nShake detected! Launching Magic Eight Ball...")
        # Get the directory of the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        main_path = os.path.join(script_dir, "main.py")
        
        # Launch main.py in a new terminal window
        if sys.platform == "win32":
            subprocess.Popen(['cmd', '/c', 'start', 'cmd', '/k', f'python "{main_path}"'], shell=True)
        else:  # For Linux/Raspberry Pi
            subprocess.Popen(['lxterminal', '-e', f'python3 {main_path}'])
            
        print("Magic Eight Ball activated! Continue monitoring for next shake...")
    except Exception as e:
        print(f"Error launching main.py: {e}")

def monitor_gyroscope():
    """Monitor and display gyroscope data"""
    try:
        global bus, Device_Address
        bus = smbus.SMBus(1)  # or bus = smbus.SMBus(0) for older version boards
        Device_Address = 0x68   # MPU6050 device address

        # Set up signal handler for clean exit
        signal.signal(signal.SIGINT, lambda sig, frame: cleanup_and_exit())

        MPU_Init()
        print("\nMonitoring for shakes... Press Ctrl+C to stop")
        print("Shake the device to activate the Magic Eight Ball")
        print("The gyroscope will continue monitoring for the next shake")
        
        while True:
            try:
                # Read Accelerometer raw value
                acc_x = read_raw_data(ACCEL_XOUT_H)
                acc_y = read_raw_data(ACCEL_YOUT_H)
                acc_z = read_raw_data(ACCEL_ZOUT_H)
                
                # Read Gyroscope raw value
                gyro_x = read_raw_data(GYRO_XOUT_H)
                gyro_y = read_raw_data(GYRO_YOUT_H)
                gyro_z = read_raw_data(GYRO_ZOUT_H)
                
                # Full scale range +/- 250 degree/C as per sensitivity scale factor
                Ax = acc_x/16384.0
                Ay = acc_y/16384.0
                Az = acc_z/16384.0
                
                Gx = gyro_x/131.0
                Gy = gyro_y/131.0
                Gz = gyro_z/131.0
                
                # Store acceleration values for shake detection
                current_accel = (Ax, Ay, Az)
                last_accel_values.append(current_accel)
                if len(last_accel_values) > SAMPLES_TO_CHECK:
                    last_accel_values.pop(0)
                
                # Check for shake
                if detect_shake([acc[0] for acc in last_accel_values]):  # Check X-axis
                    launch_main()  # This will exit the script after launching main.py
                
                print(f"\rGx={Gx:+.2f}°/s  Gy={Gy:+.2f}°/s  Gz={Gz:+.2f}°/s  Ax={Ax:+.2f}g  Ay={Ay:+.2f}g  Az={Az:+.2f}g", end='')
                sleep(0.1)  # Update every 100ms
                
            except Exception as e:
                print(f"\nError reading gyroscope: {e}")
                sleep(1)  # Wait a bit longer on error
                
    except KeyboardInterrupt:
        cleanup_and_exit()
    except Exception as e:
        print(f"Error initializing MPU6050: {e}")
        cleanup_and_exit()

if __name__ == "__main__":
    monitor_gyroscope() 