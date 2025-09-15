import serial
import time
import signal
import threading

exitThread = False

def handler(signum, frame):
    global exitThread
    exitThread = True

def readThread(ser):
    global exitThread
    while not exitThread:
        data = ser.readline()
        # 바이트 데이터를 문자열로 반환
        if data:
            str = data.decode()
            buf = ""
            for c in str:
                if c != '\r' or c != '\n':
                    buf = buf + c
            print("수신된 데이터 : ", buf)
    

if __name__ == "__main__":
    ser = serial.Serial(port='COM7', baudrate=115200, timeout=1)
    print("current port = ", ser.portstr)  # 연결된 포트 확인

    signal.signal(signal.SIGINT, handler)

    thread = threading.Thread(target=readThread, args=(ser,))

    thread.start()

    while True:
        data = input("STM32에 보낼 값 입력 (q 입력 시 종료): ")
        if data.lower() == 'q':
            break
        data = data + '\n'
        ser.write(data.encode())  # 문자열을 바이트로 변환 후 전송
        print(f"송신: {data}")

    exitThread = True
    thread.join()
    ser.close()