from openni import openni2

# OpenNI2 라이브러리 위치 직접 명시 (환경변수 대신)
OPENNI_REDIST = r"C:\Users\SSAFY\Desktop\OpenNI_2.3.0.86\Win64-Release\tools\NiViewer"

openni2.initialize(OPENNI_REDIST)
print("OpenNI2 version:", openni2.get_version())

dev = openni2.Device.open_any()
print("Device:", dev.get_device_info())

openni2.unload()
print("OK")