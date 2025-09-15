import serial
import time

# STM32 연결된 COM 포트와 속도
ser = serial.Serial('COM7', 115200, timeout=1)
time.sleep(2)  # 연결 안정화 대기

print("STM32에 데이터 전송 시작")

while True:
    data = input("STM32에 보낼 값 입력 (q 입력 시 종료): ")
    if data.lower() == 'q':
        break
    ser.write((data + '\n').encode())  # 문자열을 바이트로 변환 후 전송
    print(f"송신: {data}")

    # 🔹 STM32 응답 받기
    # response = ser.readline().decode().strip()
    response = ser.read_until(b"\r\n").decode().strip()

    if response:
        print(f"수신: {response}")

ser.close()
print("종료")
