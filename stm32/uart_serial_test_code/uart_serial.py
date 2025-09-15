import serial
import time

if __name__ == "__main__":
	ser = serial.Serial(port='COM7', baudrate=115200, timeout=1)
	print("current port = ", ser.portstr) # 연결된 포트 확인
	while True:
		data = input("STM32에 보낼 값 입력 (q 입력 시 종료): ")
		if data.lower() == 'q':
		break data = data + '\n' ser.write(data.encode()) # 문자열을 바이트로 변환 후 전송
		print(f"송신: {data}")
		ser.close()